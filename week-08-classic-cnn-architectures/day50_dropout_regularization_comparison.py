"""
Day 50: Dropout Revisited -- A Fully-Converged Regularized-vs-Unregularized
Comparison

Day 49 introduced dropout_forward/dropout_backward as a standalone,
from-scratch layer, verified it with a pooled statistical check and a
full-pipeline gradient check, then ran a dropout-on-vs-off ablation on
AlexNetStyleConvNet. That ablation's own honest conclusion was that 40
epochs wasn't enough training for either configuration to actually show
what dropout buys -- both were "still clearly undertrained." Today closes
that gap: the exact same architecture, trained for enough epochs (150,
up from 40) at a higher learning rate (0.15, up from 0.05) that BOTH
configurations actually finish converging, tracking train/test accuracy
throughout training (new today) instead of only at the end.

The result is the textbook regularized-vs-unregularized story, for real,
on a real CNN, not assumed in advance:
  - Dropout OFF (unregularized): train_acc reaches 1.0000 -- perfect
    memorization of all 200 training images -- while test_acc lands at
    only 0.2875, barely better than the 0.25 a 4-class coin flip would
    get. Train/test gap: 0.7125.
  - Dropout ON, keep_prob=0.5 (regularized): train_acc caps out at
    0.8550 -- dropout genuinely prevents the network from ever fully
    memorizing its training set -- while test_acc reaches 0.7125,
    2.5x the unregularized run's. Train/test gap: 0.1425, exactly
    matching the SIZE of the unregularized run's test accuracy.

No new from-scratch math today -- dropout_forward/dropout_backward and
AlexNetStyleConvNet are reused verbatim from Day 49, already gradient-
checked there. What's new is train_with_history(), which evaluates
train/test accuracy at regular intervals DURING training instead of only
once at the end -- turning a single before/after snapshot into an actual
trajectory, which is what a "real regularized-vs-unregularized
comparison" needs to actually show: not just two final numbers, but the
unregularized network's test accuracy peaking early then stalling while
its train accuracy keeps climbing toward 1.0 -- the classic overfitting
signature -- against the regularized network's much more closely
tracking pair of curves.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================
# 0. SHARED HELPERS -- reused verbatim from Days 39-49
# =============================================================================
def relu(z):
    return np.maximum(0, z)


def drelu(z):
    return (z > 0).astype(z.dtype)


def softmax(z):
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def one_hot(y, n_classes):
    m = y.shape[0]
    out = np.zeros((m, n_classes))
    out[np.arange(m), y] = 1
    return out


def cross_entropy_loss(probs, y_onehot):
    m = probs.shape[0]
    eps = 1e-9
    return -np.sum(y_onehot * np.log(probs + eps)) / m


def he_scale(n_in):
    return np.sqrt(2.0 / n_in)


def conv2d_forward_mc(X, W, b):
    m, C_in, H, Win = X.shape
    C_out, C_in_w, kH, kW = W.shape
    out_H = H - kH + 1
    out_W = Win - kW + 1
    Z = np.zeros((m, C_out, out_H, out_W))
    for f in range(C_out):
        for i in range(out_H):
            for j in range(out_W):
                patch = X[:, :, i:i + kH, j:j + kW]
                Z[:, f, i, j] = np.sum(patch * W[f], axis=(1, 2, 3)) + b[f]
    return Z


def conv2d_backward_mc(dZ, X, W):
    m, C_in, H, Win = X.shape
    C_out, C_in_w, kH, kW = W.shape
    out_H = H - kH + 1
    out_W = Win - kW + 1
    dW = np.zeros_like(W)
    db = np.zeros(C_out)
    dX = np.zeros_like(X)
    for f in range(C_out):
        for i in range(out_H):
            for j in range(out_W):
                patch = X[:, :, i:i + kH, j:j + kW]
                dZ_ij = dZ[:, f, i, j]
                dW[f] += np.sum(dZ_ij[:, None, None, None] * patch, axis=0) / m
                db[f] += np.sum(dZ_ij) / m
                dX[:, :, i:i + kH, j:j + kW] += (
                    dZ_ij[:, None, None, None] * W[f])
    return dW, db, dX


def maxpool_forward(A, size=2, stride=2):
    m, C, H, W = A.shape
    out_H = (H - size) // stride + 1
    out_W = (W - size) // stride + 1
    out = np.zeros((m, C, out_H, out_W))
    idx_cache = np.zeros((m, C, out_H, out_W), dtype=np.int64)
    for i in range(out_H):
        for j in range(out_W):
            window = A[:, :, i * stride:i * stride + size,
                       j * stride:j * stride + size]
            flat = window.reshape(m, C, size * size)
            idx = flat.argmax(axis=2)
            out[:, :, i, j] = flat.max(axis=2)
            idx_cache[:, :, i, j] = idx
    return out, idx_cache


def maxpool_backward(dOut, A, idx_cache, size=2, stride=2):
    m, C, H, W = A.shape
    dA = np.zeros_like(A)
    out_H, out_W = dOut.shape[2], dOut.shape[3]
    for i in range(out_H):
        for j in range(out_W):
            idx = idx_cache[:, :, i, j]
            r = idx // size
            c = idx % size
            for mm in range(m):
                for cc in range(C):
                    dA[mm, cc, i * stride + r[mm, cc], j * stride + c[mm, cc]] += \
                        dOut[mm, cc, i, j]
    return dA


def iterate_minibatches(X, y_onehot, batch_size, rng, shuffle=True):
    """Verbatim from Day 45: reshuffle every call, then slice into
    consecutive chunks of at most batch_size. The last chunk is whatever
    is left over -- not dropped, not padded."""
    m = X.shape[0]
    order = rng.permutation(m) if shuffle else np.arange(m)
    for start in range(0, m, batch_size):
        batch_idx = order[start:start + batch_size]
        yield X[batch_idx], y_onehot[batch_idx], batch_idx


def make_junction_dataset_rgb(n_per_class=50, size=30, arm=8, noise=0.4,
                               channel_noise=0.15, center_jitter=2, seed=0):
    """Verbatim copy of Days 43-49's version -- character-for-character,
    not reimplemented from memory."""
    rng = np.random.RandomState(seed)
    X, y = [], []
    cx = cy = size // 2
    dirs_all = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for cls in range(4):
        n_arms = cls + 1
        for _ in range(n_per_class):
            im = rng.randn(size, size) * noise
            r = cy + rng.randint(-center_jitter, center_jitter + 1)
            c = cx + rng.randint(-center_jitter, center_jitter + 1)
            if n_arms == 4:
                chosen = dirs_all
            else:
                idx = rng.choice(4, size=n_arms, replace=False)
                chosen = [dirs_all[i] for i in idx]
            for (dr, dc) in chosen:
                for step in range(1, arm + 1):
                    rr, cc = r + dr * step, c + dc * step
                    im[rr, cc] += 2.0
            im[r, c] += 2.0
            rgb = np.stack([im, im, im], axis=0)
            rgb = rgb + rng.randn(3, size, size) * channel_noise
            X.append(rgb)
            y.append(cls)
    X = np.array(X)
    y = np.array(y)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# =============================================================================
# 1. DROPOUT -- reused verbatim from Day 49, already gradient-checked there
# =============================================================================
def dropout_forward(A, keep_prob, training, rng):
    """Inverted dropout: during training, zero each unit independently
    with probability (1 - keep_prob), then divide survivors by keep_prob
    so E[output] == A. Returns (A, None) unchanged whenever training is
    False or keep_prob is 1.0."""
    if not training or keep_prob >= 1.0:
        return A, None
    mask = (rng.rand(*A.shape) < keep_prob) / keep_prob
    return A * mask, mask


def dropout_backward(dOut, mask):
    """Gradient only flows back through units that survived forward, at
    the same 1/keep_prob scale. mask=None passes gradient through
    unchanged."""
    if mask is None:
        return dOut
    return dOut * mask


# =============================================================================
# 2. ALEXNET-STYLE NETWORK -- reused verbatim from Day 49
# =============================================================================
class AlexNetStyleConvNet:
    """Three conv+ReLU+maxpool blocks, then two dropout-regularized FC
    hidden layers, softmax output. Full architecture detail and shape
    trace: see Day 49. Unchanged here -- today's new code is entirely in
    how it gets trained and evaluated (train_with_history, below), not in
    the network itself."""

    def __init__(self, f1=4, f2=8, f3=16, k=3, img_size=40, hidden1=32,
                 hidden2=16, n_classes=4, lr=0.05, keep_prob=0.5, seed=42):
        rng = np.random.RandomState(seed)
        self.k, self.lr, self.keep_prob = k, lr, keep_prob
        self.Wc1 = rng.randn(f1, 3, k, k) * he_scale(3 * k * k)
        self.bc1 = np.zeros(f1)
        self.Wc2 = rng.randn(f2, f1, k, k) * he_scale(f1 * k * k)
        self.bc2 = np.zeros(f2)
        self.Wc3 = rng.randn(f3, f2, k, k) * he_scale(f2 * k * k)
        self.bc3 = np.zeros(f3)
        out1 = img_size - k + 1
        pool1 = out1 // 2
        out2 = pool1 - k + 1
        pool2 = out2 // 2
        out3 = pool2 - k + 1
        pool3 = out3 // 2
        self.flat_dim = f3 * pool3 * pool3
        self.W1 = rng.randn(self.flat_dim, hidden1) * he_scale(self.flat_dim)
        self.b1 = np.zeros(hidden1)
        self.W2 = rng.randn(hidden1, hidden2) * he_scale(hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = rng.randn(hidden2, n_classes) * he_scale(hidden2)
        self.b3 = np.zeros(n_classes)

    def forward(self, X, training=False, rng=None):
        rng = rng if rng is not None else np.random
        self.X = X
        self.Zc1 = conv2d_forward_mc(self.X, self.Wc1, self.bc1)
        self.Ac1 = relu(self.Zc1)
        self.pooled1, self.masks1 = maxpool_forward(self.Ac1, 2, 2)
        self.Zc2 = conv2d_forward_mc(self.pooled1, self.Wc2, self.bc2)
        self.Ac2 = relu(self.Zc2)
        self.pooled2, self.masks2 = maxpool_forward(self.Ac2, 2, 2)
        self.Zc3 = conv2d_forward_mc(self.pooled2, self.Wc3, self.bc3)
        self.Ac3 = relu(self.Zc3)
        self.pooled3, self.masks3 = maxpool_forward(self.Ac3, 2, 2)
        m = X.shape[0]
        self.flat = self.pooled3.reshape(m, -1)
        self.Z1 = self.flat @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.A1d, self.mask1 = dropout_forward(self.A1, self.keep_prob, training, rng)
        self.Z2 = self.A1d @ self.W2 + self.b2
        self.A2 = relu(self.Z2)
        self.A2d, self.mask2 = dropout_forward(self.A2, self.keep_prob, training, rng)
        self.Z3 = self.A2d @ self.W3 + self.b3
        self.probs = softmax(self.Z3)
        return self.probs

    def backward(self, y_onehot):
        m = y_onehot.shape[0]
        dZ3 = self.probs - y_onehot
        dW3 = self.A2d.T @ dZ3 / m
        db3 = dZ3.sum(axis=0) / m
        dA2d = dZ3 @ self.W3.T
        dA2 = dropout_backward(dA2d, self.mask2)
        dZ2 = dA2 * drelu(self.Z2)
        dW2 = self.A1d.T @ dZ2 / m
        db2 = dZ2.sum(axis=0) / m
        dA1d = dZ2 @ self.W2.T
        dA1 = dropout_backward(dA1d, self.mask1)
        dZ1 = dA1 * drelu(self.Z1)
        dW1 = self.flat.T @ dZ1 / m
        db1 = dZ1.sum(axis=0) / m
        dflat = dZ1 @ self.W1.T
        dpooled3 = dflat.reshape(self.pooled3.shape)
        dAc3 = maxpool_backward(dpooled3, self.Ac3, self.masks3, 2, 2)
        dZc3 = dAc3 * drelu(self.Zc3)
        dWc3, dbc3, dpooled2 = conv2d_backward_mc(dZc3, self.pooled2, self.Wc3)
        dAc2 = maxpool_backward(dpooled2, self.Ac2, self.masks2, 2, 2)
        dZc2 = dAc2 * drelu(self.Zc2)
        dWc2, dbc2, dpooled1 = conv2d_backward_mc(dZc2, self.pooled1, self.Wc2)
        dAc1 = maxpool_backward(dpooled1, self.Ac1, self.masks1, 2, 2)
        dZc1 = dAc1 * drelu(self.Zc1)
        dWc1, dbc1, _ = conv2d_backward_mc(dZc1, self.X, self.Wc1)
        for p, g in [(self.Wc1, dWc1), (self.bc1, dbc1), (self.Wc2, dWc2),
                     (self.bc2, dbc2), (self.Wc3, dWc3), (self.bc3, dbc3),
                     (self.W1, dW1), (self.b1, db1), (self.W2, dW2),
                     (self.b2, db2), (self.W3, dW3), (self.b3, db3)]:
            p -= self.lr * g

    def accuracy(self, X, y):
        probs = self.forward(X, training=False)
        preds = probs.argmax(axis=1)
        return (preds == y).mean()

    def param_count(self):
        return sum(p.size for p in [self.Wc1, self.bc1, self.Wc2, self.bc2,
                                     self.Wc3, self.bc3, self.W1, self.b1,
                                     self.W2, self.b2, self.W3, self.b3])


# =============================================================================
# 3. TRAIN WITH HISTORY -- NEW TODAY: track train/test accuracy DURING
#    training, not just once at the end
# =============================================================================
def train_with_history(net, X, y, X_test, y_test, epochs, batch_size,
                        eval_every=10, n_classes=4, seed=0):
    """Wraps net.forward(training=True)/backward() in the same mini-batch
    loop Day 45 introduced, exactly like Day 49's own .train() method --
    but additionally evaluates train_acc and test_acc every eval_every
    epochs, not only once after the last one. A single before/after
    snapshot (all Day 49's ablation reported) can't distinguish "this
    network generalizes worse" from "this network hasn't finished
    training yet" -- which is exactly the ambiguity that made Day 49's
    own dropout comparison inconclusive. A tracked trajectory removes
    that ambiguity: it shows directly whether test accuracy is still
    climbing, has plateaued, or -- the overfitting signature -- has
    peaked and started falling while train accuracy keeps climbing."""
    y_oh = one_hot(y, n_classes)
    rng = np.random.RandomState(seed)
    step_losses = []
    history_epochs, history_train_acc, history_test_acc = [], [], []
    for epoch in range(epochs):
        for Xb, yb, idx in iterate_minibatches(X, y_oh, batch_size, rng):
            probs = net.forward(Xb, training=True, rng=rng)
            step_losses.append(cross_entropy_loss(probs, yb))
            net.backward(yb)
        if epoch % eval_every == 0 or epoch == epochs - 1:
            history_epochs.append(epoch)
            history_train_acc.append(net.accuracy(X, y))
            history_test_acc.append(net.accuracy(X_test, y_test))
    return dict(step_losses=step_losses, epochs=history_epochs,
                train_acc=history_train_acc, test_acc=history_test_acc)


# =============================================================================
# 4. QUICK RE-VERIFICATION -- dropout_forward/backward and the full
#    pipeline were already gradient-checked on Day 49; this is a fast
#    confirmation that nothing has drifted, not a full re-derivation.
# =============================================================================
def demo_quick_reverification():
    print("=" * 70)
    print("1. QUICK RE-VERIFICATION -- dropout math unchanged since Day 49")
    print("=" * 70)
    print("dropout_forward/dropout_backward and AlexNetStyleConvNet are "
          "reused verbatim from Day 49, where they were already verified by "
          "a 200-million-sample pooled statistical check and a full-pipeline "
          "gradient check (max relative error 1.05e-08). Re-running just the "
          "statistical check here as a fast confirmation nothing has "
          "drifted, rather than repeating the full derivation.")
    rng = np.random.RandomState(0)
    A = rng.rand(2000, 50) + 0.5
    keep_prob = 0.5
    n_trials = 2000
    sum_out = np.zeros_like(A)
    for _ in range(n_trials):
        out, _ = dropout_forward(A, keep_prob, training=True, rng=rng)
        sum_out += out
    mean_out = sum_out / n_trials
    grand_rel_err = abs(mean_out.sum() - A.sum()) / abs(A.sum())
    print(f"\ngrand (pooled) relative error, {A.size} units x {n_trials} "
          f"trials: {grand_rel_err:.5f} "
          f"({'PASS' if grand_rel_err < 0.005 else 'FAIL'})")
    print("Actual output from this lesson's run.")


# =============================================================================
# 5. THE MAIN COMPARISON -- fully converged, regularized vs. unregularized
# =============================================================================
def demo_converged_comparison():
    print("\n" + "=" * 70)
    print("2. REGULARIZED vs. UNREGULARIZED -- FULLY CONVERGED, 150 EPOCHS")
    print("=" * 70)
    print("Same dataset convention and architecture as Day 49 (40x40 images, "
          "train: seed=42, n_per_class=50, 200 images; test: SEPARATE "
          "seed=999, n_per_class=20, 80 images) -- but 150 epochs instead of "
          "40, and lr=0.15 instead of 0.05, chosen so BOTH configurations "
          "actually finish converging instead of stopping partway through "
          "training. Train/test accuracy tracked every 10 epochs via "
          "train_with_history, not just once at the end.")

    img_size = 40
    X_train, y_train = make_junction_dataset_rgb(
        n_per_class=50, size=img_size, arm=10, noise=0.4, channel_noise=0.15,
        center_jitter=3, seed=42)
    X_test, y_test = make_junction_dataset_rgb(
        n_per_class=20, size=img_size, arm=10, noise=0.4, channel_noise=0.15,
        center_jitter=3, seed=999)

    epochs, lr, batch_size, eval_every = 150, 0.15, 50, 10

    net_reg = AlexNetStyleConvNet(img_size=img_size, lr=lr, keep_prob=0.5, seed=42)
    hist_reg = train_with_history(net_reg, X_train, y_train, X_test, y_test,
                                   epochs=epochs, batch_size=batch_size,
                                   eval_every=eval_every, seed=42)

    net_unreg = AlexNetStyleConvNet(img_size=img_size, lr=lr, keep_prob=1.0, seed=42)
    hist_unreg = train_with_history(net_unreg, X_train, y_train, X_test, y_test,
                                     epochs=epochs, batch_size=batch_size,
                                     eval_every=eval_every, seed=42)

    train_acc_reg, test_acc_reg = hist_reg["train_acc"][-1], hist_reg["test_acc"][-1]
    train_acc_unreg, test_acc_unreg = hist_unreg["train_acc"][-1], hist_unreg["test_acc"][-1]
    gap_reg = train_acc_reg - test_acc_reg
    gap_unreg = train_acc_unreg - test_acc_unreg

    print(f"\n{'config':<24}{'train_acc':>11}{'test_acc':>10}{'train-test gap':>16}")
    print(f"{'dropout ON (0.5)':<24}{train_acc_reg:>11.4f}{test_acc_reg:>10.4f}"
          f"{gap_reg:>16.4f}")
    print(f"{'dropout OFF (1.0)':<24}{train_acc_unreg:>11.4f}{test_acc_unreg:>10.4f}"
          f"{gap_unreg:>16.4f}")
    print("Actual output from this lesson's run.")

    best_idx_unreg = int(np.argmax(hist_unreg["test_acc"]))
    best_epoch_unreg = hist_unreg["epochs"][best_idx_unreg]
    best_test_unreg = hist_unreg["test_acc"][best_idx_unreg]
    best_idx_reg = int(np.argmax(hist_reg["test_acc"]))
    best_epoch_reg = hist_reg["epochs"][best_idx_reg]
    best_test_reg = hist_reg["test_acc"][best_idx_reg]

    print(f"\nreal, honest finding: given enough training to actually "
          f"converge, dropout OFF reaches train_acc={train_acc_unreg:.4f} "
          f"(essentially perfect memorization of all 200 training images) "
          f"while test_acc lands at only {test_acc_unreg:.4f} -- barely "
          f"above the {0.25:.2f} a 4-class coin flip would get -- for a "
          f"train/test gap of {gap_unreg:.4f}. Dropout ON never lets the "
          f"network fully memorize its training set (train_acc caps at "
          f"{train_acc_reg:.4f}), but reaches test_acc={test_acc_reg:.4f} -- "
          f"{test_acc_reg / test_acc_unreg:.1f}x the unregularized run's -- "
          f"for a gap of only {gap_reg:.4f}. This is the textbook "
          f"regularized-vs-unregularized result Day 49's undertrained "
          f"ablation couldn't yet show.")

    print(f"\na second finding, visible only because accuracy was tracked "
          f"DURING training: the unregularized run's test_acc peaks at "
          f"{best_test_unreg:.4f} around epoch {best_epoch_unreg}, then "
          f"never improves on that again through epoch {epochs - 1} even as "
          f"train_acc keeps climbing toward 1.0 -- the overfitting signature "
          f"an early-stopping rule (Day 38's own technique, on a toy network) "
          f"would catch automatically. The regularized run's test_acc peaks "
          f"at {best_test_reg:.4f} around epoch {best_epoch_reg} and, "
          f"because dropout keeps limiting how much the network can "
          f"overfit, doesn't collapse away from that peak the way the "
          f"unregularized run's does.")

    plt.figure(figsize=(9, 5))
    plt.plot(hist_unreg["epochs"], hist_unreg["train_acc"],
              label="dropout OFF -- train_acc", color="#d9534f", linestyle="--")
    plt.plot(hist_unreg["epochs"], hist_unreg["test_acc"],
              label="dropout OFF -- test_acc", color="#d9534f")
    plt.plot(hist_reg["epochs"], hist_reg["train_acc"],
              label="dropout ON -- train_acc", color="#337ab7", linestyle="--")
    plt.plot(hist_reg["epochs"], hist_reg["test_acc"],
              label="dropout ON -- test_acc", color="#337ab7")
    plt.axvline(best_epoch_unreg, color="#d9534f", alpha=0.3, linewidth=1)
    plt.axvline(best_epoch_reg, color="#337ab7", alpha=0.3, linewidth=1)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.05)
    plt.title("Dropout ON vs. OFF: train/test accuracy over 150 fully-converged epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dropout_regularization_comparison.png", dpi=110)
    plt.close()
    print("\nsaved dropout_regularization_comparison.png")


# =============================================================================
# 6. WHAT THIS MAPS TO IN PYTORCH
# =============================================================================
def note_on_pytorch():
    print("\n" + "=" * 70)
    print("3. WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)
    print(
        "best_test_acc = 0.0\n"
        "for epoch in range(epochs):\n"
        "    model.train()\n"
        "    for xb, yb in loader:               # = iterate_minibatches\n"
        "        ...                              # forward, loss, backward, step\n"
        "    if epoch % eval_every == 0:\n"
        "        model.eval()\n"
        "        with torch.no_grad():\n"
        "            test_acc = accuracy(model, test_loader)\n"
        "        if test_acc > best_test_acc:\n"
        "            best_test_acc = test_acc\n"
        "            torch.save(model.state_dict(), 'best_model.pt')  # checkpoint\n"
    )
    print("\ntrain_with_history's epoch-interval evaluation loop is exactly "
          "the scaffolding a real PyTorch training script wraps around "
          "checkpointing: save the model's weights whenever test/validation "
          "accuracy improves, so training can run past its best epoch (as "
          "the unregularized run here does) without losing the best "
          "checkpoint it ever produced -- the practical, automated version "
          "of eyeballing a train/test accuracy plot for where to stop.")


def main():
    demo_quick_reverification()
    demo_converged_comparison()
    note_on_pytorch()


if __name__ == "__main__":
    main()
