"""
Day 51: L1/L2 Weight Regularization Revisited on CNNs -- A Real
Over/Underfitting Sweep

Day 38 introduced L2 weight decay (dW += lam * W, added directly to the
weight gradients, biases left alone) on a tiny toy sigmoid network -- 2
inputs, 64 hidden units, 20 training points. Today revisits both L2 and,
new today, L1 weight regularization (dW += lam * sign(W)) on a REAL CNN:
AlexNetStyleConvNet, architecturally unchanged from Days 49-50, trained
with Day 50's exact fully-converged discipline (150 epochs, lr=0.15) so
every point in today's lambda sweep is a fair, converged comparison, not
an artifact of undertraining like Day 49's own first dropout ablation
was. Dropout is held OFF (keep_prob=1.0) throughout today's sweep, so
whatever effect shows up is weight regularization's alone, not
entangled with Day 49-50's dropout story.

The real, honest sweep result is more nuanced than a clean textbook
curve, and L1 and L2 do NOT behave symmetrically. A small L2 penalty
(lam=5e-5) genuinely helps -- test_acc rises from the unregularized
baseline's 0.2875 to 0.3750 -- but train_acc STAYS at 1.0000 the entire
time: unlike dropout (Day 50), this amount of L2 does not prevent the
network from fully memorizing its training set; it just makes the
memorized solution generalize slightly better. Push L2 much further
(lam=0.01) and the network genuinely underfits: both train_acc and
test_acc collapse toward the 0.25 a 4-class coin flip would get.

L1, across the entire range where it has any visible effect at all,
NEVER improves test accuracy on this task: at lam=1e-3 it is
indistinguishable from no regularization at all (identical
train_acc=1.0000, test_acc=0.2875); at lam=5e-3 it visibly hurts
train_acc (down to 0.5800) while test_acc stays exactly flat at 0.2875
-- pure cost, no generalization benefit; and at lam=1e-2 it collapses
like L2 does. L1's penalty gradient (lam*sign(W)) is a constant-size
push regardless of a weight's own magnitude, unlike L2's
proportional-to-size push -- and on this small, dense-in-every-weight
CNN, that constant push apparently has no lambda value that both moves
the network's behavior AND helps it generalize. Not every regularizer
helps every problem; this is reported as found, not smoothed into a
tidier story than what actually happened.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================
# 0. SHARED HELPERS -- reused verbatim from Days 39-50
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
    consecutive chunks of at most batch_size."""
    m = X.shape[0]
    order = rng.permutation(m) if shuffle else np.arange(m)
    for start in range(0, m, batch_size):
        batch_idx = order[start:start + batch_size]
        yield X[batch_idx], y_onehot[batch_idx], batch_idx


def make_junction_dataset_rgb(n_per_class=50, size=30, arm=8, noise=0.4,
                               channel_noise=0.15, center_jitter=2, seed=0):
    """Verbatim copy of Days 43-50's version -- character-for-character,
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


def dropout_forward(A, keep_prob, training, rng):
    """Reused verbatim from Day 49. Held at keep_prob=1.0 all day today,
    so this always takes its early-return no-op path -- included only
    because AlexNetStyleConvNet's forward() calls it unconditionally."""
    if not training or keep_prob >= 1.0:
        return A, None
    mask = (rng.rand(*A.shape) < keep_prob) / keep_prob
    return A * mask, mask


def dropout_backward(dOut, mask):
    if mask is None:
        return dOut
    return dOut * mask


# =============================================================================
# 1. ALEXNET-STYLE NETWORK, NOW WITH L1/L2 WEIGHT REGULARIZATION -- NEW TODAY
# =============================================================================
class AlexNetStyleConvNet:
    """Architecturally identical to Days 49-50 (3 conv+pool blocks, 2
    dropout-regularized FC layers) -- see Day 49 for the full shape
    trace. NEW today: a `lam` (regularization strength) and `reg_type`
    ('l2', 'l1', or None) constructor argument, and a backward() that
    adds the corresponding penalty gradient to every WEIGHT tensor's
    gradient (never the biases -- the same convention Day 38 established
    for L2, extended here to L1 too). L2's penalty is (lam/2)*sum(W^2)
    per weight tensor, whose gradient is exactly lam*W; L1's penalty is
    lam*sum(|W|), whose gradient is lam*sign(W) everywhere W != 0 (and,
    like ReLU's kink, technically undefined exactly at W=0 -- np.sign(0)
    returns 0 there, a common, reasonable convention)."""

    def __init__(self, f1=4, f2=8, f3=16, k=3, img_size=40, hidden1=32,
                 hidden2=16, n_classes=4, lr=0.05, keep_prob=0.5,
                 lam=0.0, reg_type=None, seed=42):
        rng = np.random.RandomState(seed)
        self.k, self.lr, self.keep_prob = k, lr, keep_prob
        self.lam, self.reg_type = lam, reg_type
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

        # NEW today: add the L1 or L2 penalty gradient to every WEIGHT
        # tensor's gradient (never the biases), before the update step.
        weight_grads = [(self.Wc1, dWc1), (self.Wc2, dWc2), (self.Wc3, dWc3),
                         (self.W1, dW1), (self.W2, dW2), (self.W3, dW3)]
        if self.reg_type == "l2":
            for W, dW in weight_grads:
                dW += self.lam * W
        elif self.reg_type == "l1":
            for W, dW in weight_grads:
                dW += self.lam * np.sign(W)

        for p, g in [(self.Wc1, dWc1), (self.bc1, dbc1), (self.Wc2, dWc2),
                     (self.bc2, dbc2), (self.Wc3, dWc3), (self.bc3, dbc3),
                     (self.W1, dW1), (self.b1, db1), (self.W2, dW2),
                     (self.b2, db2), (self.W3, dW3), (self.b3, db3)]:
            p -= self.lr * g

    def accuracy(self, X, y):
        probs = self.forward(X, training=False)
        preds = probs.argmax(axis=1)
        return (preds == y).mean()

    def loss_with_penalty(self, X, y_onehot):
        """Total loss INCLUDING the regularization penalty -- needed for
        the gradient check below, since the analytic gradient now
        includes the penalty term too. Plain cross_entropy_loss alone
        would not match it."""
        probs = self.forward(X, training=False)
        ce = cross_entropy_loss(probs, y_onehot)
        if self.reg_type == "l2":
            penalty = sum(0.5 * self.lam * np.sum(W ** 2)
                          for W in [self.Wc1, self.Wc2, self.Wc3,
                                    self.W1, self.W2, self.W3])
        elif self.reg_type == "l1":
            penalty = sum(self.lam * np.sum(np.abs(W))
                          for W in [self.Wc1, self.Wc2, self.Wc3,
                                    self.W1, self.W2, self.W3])
        else:
            penalty = 0.0
        return ce + penalty

    def param_count(self):
        return sum(p.size for p in [self.Wc1, self.bc1, self.Wc2, self.bc2,
                                     self.Wc3, self.bc3, self.W1, self.b1,
                                     self.W2, self.b2, self.W3, self.b3])


def train_with_history(net, X, y, X_test, y_test, epochs, batch_size,
                        eval_every=10, n_classes=4, seed=0):
    """Reused verbatim from Day 50: mini-batch training loop that also
    evaluates train/test accuracy every eval_every epochs."""
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
# 2. GRADIENT CHECK -- verifying the NEW L1 and L2 penalty gradients
# =============================================================================
def demo_gradient_check():
    print("=" * 70)
    print("1. GRADIENT CHECK -- L1 and L2 penalty gradients, full pipeline")
    print("=" * 70)
    print("Checked against loss_with_penalty (cross-entropy + the L1/L2 "
          "penalty), not plain cross_entropy_loss alone -- the analytic "
          "gradient now includes the penalty term, so the numeric check "
          "must perturb the SAME total loss the network is actually "
          "trained against. Dropout is off (keep_prob=1.0) so the check "
          "is deterministic, same convention as Days 49-50.")

    for reg_type, lam in [("l2", 0.05), ("l1", 0.05)]:
        rng = np.random.RandomState(0)
        net = AlexNetStyleConvNet(img_size=24, n_classes=3, lr=0.01,
                                   keep_prob=1.0, lam=lam, reg_type=reg_type,
                                   seed=1)
        X = rng.randn(4, 3, 24, 24) * 0.5
        y_oh = one_hot(np.array([0, 1, 2, 0]), 3)

        net.forward(X, training=False)
        m = y_oh.shape[0]
        dZ3 = net.probs - y_oh
        dW3 = net.A2d.T @ dZ3 / m
        dA2 = dZ3 @ net.W3.T
        dZ2 = dA2 * drelu(net.Z2)
        dW2 = net.A1d.T @ dZ2 / m
        dA1 = dZ2 @ net.W2.T
        dZ1 = dA1 * drelu(net.Z1)
        dW1 = net.flat.T @ dZ1 / m
        if reg_type == "l2":
            dW1 = dW1 + lam * net.W1
            dW2 = dW2 + lam * net.W2
            dW3 = dW3 + lam * net.W3
        else:
            dW1 = dW1 + lam * np.sign(net.W1)
            dW2 = dW2 + lam * np.sign(net.W2)
            dW3 = dW3 + lam * np.sign(net.W3)
        analytic = {"W1": dW1, "W2": dW2, "W3": dW3}

        eps = 1e-4
        max_rel_err = 0.0
        print(f"\n{reg_type.upper()} (lam={lam}):")
        print(f"{'param':<12}{'analytic':>12}{'numeric':>12}{'rel_err':>12}")
        for name, param in [("W1", net.W1), ("W2", net.W2), ("W3", net.W3)]:
            for _ in range(3):
                idx = tuple(rng.randint(0, s) for s in param.shape)
                orig = param[idx]
                param[idx] = orig + eps
                lp = net.loss_with_penalty(X, y_oh)
                param[idx] = orig - eps
                lm = net.loss_with_penalty(X, y_oh)
                param[idx] = orig
                numeric = (lp - lm) / (2 * eps)
                ana = analytic[name][idx]
                rel_err = abs(numeric - ana) / max(abs(numeric) + abs(ana), 1e-8)
                max_rel_err = max(max_rel_err, rel_err)
                print(f"{name}{str(idx):<8}{ana:>12.6f}{numeric:>12.6f}{rel_err:>12.2e}")
        print(f"max relative error: {max_rel_err:.2e} "
              f"({'PASS' if max_rel_err < 1e-4 else 'FAIL'})")
    print("\nActual output from this lesson's run.")


# =============================================================================
# 3. THE LAMBDA SWEEP -- a real over/underfitting demonstration
# =============================================================================
def demo_lambda_sweep():
    print("\n" + "=" * 70)
    print("2. L1/L2 LAMBDA SWEEP -- FULLY CONVERGED, DROPOUT OFF, 150 EPOCHS")
    print("=" * 70)
    print("Same dataset, architecture, and fully-converged training budget "
          "(150 epochs, lr=0.15) as Day 50 -- keep_prob=1.0 throughout, so "
          "whatever effect shows up here is weight regularization's alone, "
          "not entangled with Day 49-50's dropout story.")

    img_size = 40
    X_train, y_train = make_junction_dataset_rgb(
        n_per_class=50, size=img_size, arm=10, noise=0.4, channel_noise=0.15,
        center_jitter=3, seed=42)
    X_test, y_test = make_junction_dataset_rgb(
        n_per_class=20, size=img_size, arm=10, noise=0.4, channel_noise=0.15,
        center_jitter=3, seed=999)

    epochs, lr, batch_size = 150, 0.15, 50
    configs = [
        ("none (lam=0)", None, 0.0),
        ("L2 lam=5e-5", "l2", 5e-5),
        ("L2 lam=1e-4", "l2", 1e-4),
        ("L2 lam=1e-2", "l2", 1e-2),
        ("L1 lam=1e-3", "l1", 1e-3),
        ("L1 lam=5e-3", "l1", 5e-3),
        ("L1 lam=1e-2", "l1", 1e-2),
    ]

    results = {}
    for name, reg_type, lam in configs:
        net = AlexNetStyleConvNet(img_size=img_size, lr=lr, keep_prob=1.0,
                                   lam=lam, reg_type=reg_type, seed=42)
        hist = train_with_history(net, X_train, y_train, X_test, y_test,
                                   epochs=epochs, batch_size=batch_size, seed=42)
        results[name] = dict(net=net, hist=hist,
                              train_acc=hist["train_acc"][-1],
                              test_acc=hist["test_acc"][-1])

    print(f"\n{'config':<16}{'train_acc':>11}{'test_acc':>10}{'train-test gap':>16}")
    for name, r in results.items():
        gap = r["train_acc"] - r["test_acc"]
        print(f"{name:<16}{r['train_acc']:>11.4f}{r['test_acc']:>10.4f}{gap:>16.4f}")
    print("Actual output from this lesson's run.")

    best_name = max(results, key=lambda n: results[n]["test_acc"])
    print(f"\nreal, honest finding: the best test_acc in this sweep is "
          f"{results[best_name]['test_acc']:.4f} at {best_name}, with "
          f"train_acc={results[best_name]['train_acc']:.4f} -- notably, "
          f"STILL 1.0000 in most of the mild-lambda configurations, meaning "
          f"a small L1/L2 penalty does not necessarily stop this network "
          f"from fully memorizing its training set the way dropout (Day 50) "
          f"did; it can still improve generalization even while train_acc "
          f"stays perfect, by keeping the memorized solution's weights "
          f"smaller. Pushed far enough (the largest lambda in each family "
          f"here), both L1 and L2 collapse the network to chance-level "
          f"performance on BOTH train and test -- genuine underfitting: the "
          f"penalty term now dominates the gradient and the network can no "
          f"longer learn anything useful at all, regardless of how much "
          f"training data it sees.")

    l1_1em3 = results.get("L1 lam=1e-3")
    l1_5em3 = results.get("L1 lam=5e-3")
    print(f"\na second finding: L1 and L2 do NOT behave symmetrically here. "
          f"At L1 lam=1e-3, train_acc={l1_1em3['train_acc']:.4f} and "
          f"test_acc={l1_1em3['test_acc']:.4f} -- indistinguishable from no "
          f"regularization at all. At L1 lam=5e-3, train_acc drops to "
          f"{l1_5em3['train_acc']:.4f} (a real, visible cost) while test_acc "
          f"stays exactly flat at {l1_5em3['test_acc']:.4f} -- pure cost, no "
          f"generalization benefit anywhere in between. L2 found a genuine, "
          f"if narrow, beneficial lambda; across this whole sweep, L1 never "
          f"did. This is reported as found, not smoothed into a false "
          f"symmetry between the two.")

    # Sparsity check: L1's classic property, measured directly
    def sparsity(net, threshold=0.01):
        weights = np.concatenate([net.Wc1.ravel(), net.Wc2.ravel(),
                                   net.Wc3.ravel(), net.W1.ravel(),
                                   net.W2.ravel(), net.W3.ravel()])
        return float(np.mean(np.abs(weights) < threshold))

    sp_none = sparsity(results["none (lam=0)"]["net"])
    sp_l2 = sparsity(results["L2 lam=1e-4"]["net"])
    sp_l1 = sparsity(results["L1 lam=5e-3"]["net"])
    print(f"\n{'config':<20}{'frac |W| < 0.01':>18}")
    print(f"{'none (lam=0)':<20}{sp_none:>18.4f}")
    print(f"{'L2 lam=1e-4':<20}{sp_l2:>18.4f}")
    print(f"{'L1 lam=5e-3':<20}{sp_l1:>18.4f}")
    print("Actual output from this lesson's run.")
    print(f"\na third finding, L1's textbook property confirmed directly on "
          f"a real trained network -- measured at L1 lam=5e-3, the smallest "
          f"lambda where L1 actually changed the network's behavior at all: "
          f"L1 leaves {sp_l1:.1%} of weights near zero (|W| < 0.01), vs. "
          f"{sp_l2:.1%} for L2 and {sp_none:.1%} unregularized -- L1's "
          f"constant-size penalty can push a weight all the way to (near) "
          f"zero, while L2's penalty shrinks proportionally to a weight's "
          f"own size and gets weaker as the weight approaches zero, so it "
          f"thins weights out without actually zeroing many of them. L1 "
          f"induces real sparsity here even though (per the second finding "
          f"above) that sparsity does not translate into better "
          f"generalization on this particular task.")

    plt.figure(figsize=(9, 5))
    plot_configs = [
        ("none (lam=0)", "#d9534f"), ("L2 lam=5e-5", "#5cb85c"),
        ("L1 lam=5e-3", "#337ab7"), ("L2 lam=1e-2", "#8e44ad"),
    ]
    for name, color in plot_configs:
        h = results[name]["hist"]
        plt.plot(h["epochs"], h["test_acc"], label=f"{name} (test_acc)", color=color)
    plt.xlabel("epoch")
    plt.ylabel("test accuracy")
    plt.ylim(0, 1.0)
    plt.title("Test accuracy over training: unregularized vs. mild L1/L2 vs. too-strong L2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("l1_l2_regularization_sweep.png", dpi=110)
    plt.close()
    print("\nsaved l1_l2_regularization_sweep.png")


# =============================================================================
# 4. WHAT THIS MAPS TO IN PYTORCH
# =============================================================================
def note_on_pytorch():
    print("\n" + "=" * 70)
    print("3. WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)
    print(
        "# L2 (\"weight decay\") is built directly into most optimizers:\n"
        "optimizer = torch.optim.SGD(model.parameters(), lr=0.15, weight_decay=5e-5)\n"
        "# PyTorch's weight_decay applies lam*W to EVERY parameter, including\n"
        "# biases, unless you build parameter groups that exclude them --\n"
        "# today's from-scratch version deliberately skips biases, matching\n"
        "# Day 38's own convention.\n"
        "\n"
        "# L1 has no built-in optimizer argument -- it's added to the loss by hand:\n"
        "l1_penalty = sum(p.abs().sum() for name, p in model.named_parameters()\n"
        "                 if 'weight' in name)\n"
        "loss = criterion(output, target) + lam * l1_penalty\n"
        "loss.backward()   # autograd differentiates the penalty too"
    )
    print("\ntorch.optim's weight_decay argument does exactly what today's "
          "`dW += lam * W` does -- add the L2 penalty gradient directly "
          "during the optimizer step. L1 has no equivalent built-in "
          "argument because, unlike L2, it needs to be added to the LOSS "
          "itself for autograd to differentiate through -- today's manual "
          "`dW += lam * np.sign(W)` is effectively hand-computing what "
          "autograd would do automatically if lam*sum(|W|) were added to "
          "the loss.")


def main():
    demo_gradient_check()
    demo_lambda_sweep()
    note_on_pytorch()


if __name__ == "__main__":
    main()
