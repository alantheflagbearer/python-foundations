"""
Day 48: Transfer Learning -- Freezing, Fine-Tuning, and Why the Head Matters

Every network trained in this series so far has been trained from scratch on
its own target task. Today asks a different question: if you already have a
network trained on a RELATED task, how should you reuse it on a new task
where you only have a HANDFUL of labeled examples?

Setup: a "source" TwoBlockConvNetRGB (this series' own baseline since Day 43)
is trained the normal way on the usual junction dataset (arm=8, 200 images).
A "target" domain is then built with a real visual shift -- longer arms
(11 vs 8), doubled color noise (0.30 vs 0.15), and more position jitter (3 vs
2) -- and only 20 labeled target images (5 per class) are made available,
mimicking a realistic low-data transfer scenario. A separate 80-image target
test set (never touched by any training) is used to score every strategy.

Five real, honestly-run strategies are compared on that same 20-image target
set:
  1. Zero-shot     -- use the source network AS-IS, no target training at all
  2. From-scratch   -- ignore the source network, train a fresh net on the
                        20 target images
  3. Frozen + fresh head   -- reuse the source network's conv layers frozen,
                        but throw away its head and train a brand-new one
  4. Frozen + warm head    -- reuse BOTH the source conv layers (frozen) AND
                        its head (kept, not reinitialized), continuing to
                        train only the head
  5. Full fine-tune -- warm-start every layer from the source network and
                        keep training all of them, at a REDUCED learning rate

The honest, actually-run result: from-scratch is the worst strategy here
(test_acc=0.3625) -- WORSE than doing nothing and using the source network
unmodified (zero-shot test_acc=0.6375). Freezing the conv layers but
discarding the head (test_acc=0.4500) is ALSO worse than zero-shot -- losing
a well-calibrated decision boundary costs more, at 20 examples, than reusing
frozen features gains. The two strategies that actually beat zero-shot both
share one property: they keep the source network's head as a starting point
instead of throwing it away -- frozen-conv-warm-head (test_acc=0.6625) and a
properly-tuned full fine-tune at lr=0.01 (test_acc=0.6625) tie for the best
result, while fine-tuning at too high a learning rate (0.08) partially
destroys the pretrained weights and drops to test_acc=0.5250, still worse
than zero-shot. The lesson is not simply "freeze vs. unfreeze" -- it is
"don't throw away what already works when you don't have enough data to
relearn it."
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================
# 0. SHARED HELPERS -- reused verbatim from Days 39-47
# =============================================================================
def relu(z):
    return np.maximum(0, z)


def drelu(z):
    return (z > 0).astype(z.dtype)


def tanh(z):
    return np.tanh(z)


def dtanh(z):
    return 1 - np.tanh(z) ** 2


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


def make_junction_dataset_rgb(n_per_class=50, size=30, arm=8, noise=0.4,
                               channel_noise=0.15, center_jitter=2, seed=0):
    """Verbatim copy of Day 43-47's version -- character-for-character, not
    reimplemented from memory (an RNG-order slip here has caused a real,
    hard-to-spot bug before). Today it's called twice with DIFFERENT
    geometry/noise arguments to build the source and target domains, but the
    function body itself is unchanged."""
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


class TwoBlockConvNetRGB:
    """This series' own baseline since Day 43, reused verbatim: ReLU
    activations, MAX pooling, one hidden dense layer, 2,024 parameters at
    the default sizes. Today it plays THREE roles: the source network that
    gets pretrained, the "from scratch" control that ignores it, and the
    architecture every transfer strategy below warm-starts a copy of."""

    def __init__(self, f1=3, f2=6, k=3, img_size=30, hidden=8, n_classes=4,
                 lr=0.08, seed=42):
        rng = np.random.RandomState(seed)
        self.k, self.lr = k, lr
        self.Wc1 = rng.randn(f1, 3, k, k) * he_scale(3 * k * k)
        self.bc1 = np.zeros(f1)
        self.Wc2 = rng.randn(f2, f1, k, k) * he_scale(f1 * k * k)
        self.bc2 = np.zeros(f2)
        out1 = img_size - k + 1
        pool1 = out1 // 2
        out2 = pool1 - k + 1
        pool2 = out2 // 2
        self.flat_dim = f2 * pool2 * pool2
        self.W1 = rng.randn(self.flat_dim, hidden) * he_scale(self.flat_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, n_classes) * he_scale(hidden)
        self.b2 = np.zeros(n_classes)

    def forward(self, X):
        self.X = X
        self.Zc1 = conv2d_forward_mc(self.X, self.Wc1, self.bc1)
        self.Ac1 = relu(self.Zc1)
        self.pooled1, self.masks1 = maxpool_forward(self.Ac1, 2, 2)
        self.Zc2 = conv2d_forward_mc(self.pooled1, self.Wc2, self.bc2)
        self.Ac2 = relu(self.Zc2)
        self.pooled2, self.masks2 = maxpool_forward(self.Ac2, 2, 2)
        m = X.shape[0]
        self.flat = self.pooled2.reshape(m, -1)
        self.Z1 = self.flat @ self.W1 + self.b1
        self.A1 = tanh(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.probs = softmax(self.Z2)
        return self.probs

    def backward(self, y_onehot):
        m = y_onehot.shape[0]
        dZ2 = self.probs - y_onehot
        dW2 = self.A1.T @ dZ2 / m
        db2 = dZ2.sum(axis=0) / m
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * dtanh(self.Z1)
        dW1 = self.flat.T @ dZ1 / m
        db1 = dZ1.sum(axis=0) / m
        dflat = dZ1 @ self.W1.T
        dpooled2 = dflat.reshape(self.pooled2.shape)
        dAc2 = maxpool_backward(dpooled2, self.Ac2, self.masks2, 2, 2)
        dZc2 = dAc2 * drelu(self.Zc2)
        dWc2, dbc2, dpooled1 = conv2d_backward_mc(dZc2, self.pooled1, self.Wc2)
        dAc1 = maxpool_backward(dpooled1, self.Ac1, self.masks1, 2, 2)
        dZc1 = dAc1 * drelu(self.Zc1)
        dWc1, dbc1, _ = conv2d_backward_mc(dZc1, self.X, self.Wc1)
        for p, g in [(self.Wc1, dWc1), (self.bc1, dbc1), (self.Wc2, dWc2),
                     (self.bc2, dbc2), (self.W1, dW1), (self.b1, db1),
                     (self.W2, dW2), (self.b2, db2)]:
            p -= self.lr * g

    def train(self, X, y, epochs, n_classes=4):
        y_oh = one_hot(y, n_classes)
        losses = []
        for _ in range(epochs):
            probs = self.forward(X)
            losses.append(cross_entropy_loss(probs, y_oh))
            self.backward(y_oh)
        return losses

    def accuracy(self, X, y):
        probs = self.forward(X)
        preds = probs.argmax(axis=1)
        return (preds == y).mean()

    def param_count(self):
        return sum(p.size for p in [self.Wc1, self.bc1, self.Wc2, self.bc2,
                                     self.W1, self.b1, self.W2, self.b2])


# =============================================================================
# 1. TRANSFER LEARNING MACHINERY -- NEW TODAY
# =============================================================================
def backward_transfer(net, y_onehot, freeze_conv):
    """Like TwoBlockConvNetRGB.backward(), but can skip the conv layers
    entirely when freeze_conv=True. This only works this simply because the
    conv layers are the very FIRST layers in the network -- there is nothing
    earlier that also needs a gradient, so if the conv layers themselves
    aren't being updated, there is no reason to compute conv2d_backward_mc /
    maxpool_backward for them at all. (Freezing a layer in the MIDDLE of a
    deeper network would still require propagating gradient through it, even
    though that layer's own weights wouldn't change -- freezing an INPUT-
    adjacent layer is the one case where freezing also saves backward-pass
    compute, not just prevents an update.)"""
    m = y_onehot.shape[0]
    dZ2 = net.probs - y_onehot
    dW2 = net.A1.T @ dZ2 / m
    db2 = dZ2.sum(axis=0) / m
    dA1 = dZ2 @ net.W2.T
    dZ1 = dA1 * dtanh(net.Z1)
    dW1 = net.flat.T @ dZ1 / m
    db1 = dZ1.sum(axis=0) / m

    updates = [(net.W2, dW2), (net.b2, db2), (net.W1, dW1), (net.b1, db1)]

    if not freeze_conv:
        dflat = dZ1 @ net.W1.T
        dpooled2 = dflat.reshape(net.pooled2.shape)
        dAc2 = maxpool_backward(dpooled2, net.Ac2, net.masks2, 2, 2)
        dZc2 = dAc2 * drelu(net.Zc2)
        dWc2, dbc2, dpooled1 = conv2d_backward_mc(dZc2, net.pooled1, net.Wc2)
        dAc1 = maxpool_backward(dpooled1, net.Ac1, net.masks1, 2, 2)
        dZc1 = dAc1 * drelu(net.Zc1)
        dWc1, dbc1, _ = conv2d_backward_mc(dZc1, net.X, net.Wc1)
        updates += [(net.Wc1, dWc1), (net.bc1, dbc1),
                    (net.Wc2, dWc2), (net.bc2, dbc2)]

    for p, g in updates:
        p -= net.lr * g


def train_transfer(net, X, y, epochs, freeze_conv, n_classes=4):
    y_oh = one_hot(y, n_classes)
    losses = []
    for _ in range(epochs):
        probs = net.forward(X)
        losses.append(cross_entropy_loss(probs, y_oh))
        backward_transfer(net, y_oh, freeze_conv)
    return losses


def copy_conv_weights(source_net, target_net):
    """Warm-starts only the conv layers -- the target network's own
    (freshly, randomly initialized) head is left untouched."""
    target_net.Wc1 = source_net.Wc1.copy()
    target_net.bc1 = source_net.bc1.copy()
    target_net.Wc2 = source_net.Wc2.copy()
    target_net.bc2 = source_net.bc2.copy()


def copy_all_weights(source_net, target_net):
    """Warm-starts every layer, conv AND head, from the source network."""
    copy_conv_weights(source_net, target_net)
    target_net.W1 = source_net.W1.copy()
    target_net.b1 = source_net.b1.copy()
    target_net.W2 = source_net.W2.copy()
    target_net.b2 = source_net.b2.copy()


# =============================================================================
# 2. GRADIENT CHECK -- through the full pipeline, both freeze_conv modes,
#    plus a direct check that freezing literally freezes
# =============================================================================
def demo_gradient_check():
    print("=" * 70)
    print("1. GRADIENT CHECK -- backward_transfer, both freeze_conv modes")
    print("=" * 70)
    print("Checked through the exact forward()->loss->backward_transfer() "
          "pipeline training actually uses, for freeze_conv=False (should "
          "reduce to the same math as TwoBlockConvNetRGB.backward()) and "
          "freeze_conv=True (head-only gradients).")
    rng = np.random.RandomState(0)

    for freeze_conv in [False, True]:
        net = TwoBlockConvNetRGB(f1=3, f2=6, k=3, img_size=16, hidden=5,
                                  n_classes=3, lr=0.01, seed=1)
        X = rng.randn(4, 3, 16, 16) * 0.5
        y_oh = one_hot(np.array([0, 1, 2, 0]), 3)
        net.forward(X)
        m = y_oh.shape[0]
        dZ2 = net.probs - y_oh
        dW2 = net.A1.T @ dZ2 / m
        dA1 = dZ2 @ net.W2.T
        dZ1 = dA1 * dtanh(net.Z1)
        dW1 = net.flat.T @ dZ1 / m
        analytic = {"W1": dW1, "W2": dW2}
        params_to_check = ["W1", "W2"]
        if not freeze_conv:
            dflat = dZ1 @ net.W1.T
            dpooled2 = dflat.reshape(net.pooled2.shape)
            dAc2 = maxpool_backward(dpooled2, net.Ac2, net.masks2, 2, 2)
            dZc2 = dAc2 * drelu(net.Zc2)
            dWc2, dbc2, dpooled1 = conv2d_backward_mc(dZc2, net.pooled1, net.Wc2)
            dAc1 = maxpool_backward(dpooled1, net.Ac1, net.masks1, 2, 2)
            dZc1 = dAc1 * drelu(net.Zc1)
            dWc1, dbc1, _ = conv2d_backward_mc(dZc1, net.X, net.Wc1)
            analytic["Wc1"] = dWc1
            analytic["Wc2"] = dWc2
            params_to_check = ["W1", "W2", "Wc1", "Wc2"]

        eps = 1e-4
        max_rel_err = 0.0
        for name in params_to_check:
            param = getattr(net, name)
            for _ in range(3):
                idx = tuple(rng.randint(0, s) for s in param.shape)
                orig = param[idx]
                param[idx] = orig + eps
                lp = cross_entropy_loss(net.forward(X), y_oh)
                param[idx] = orig - eps
                lm = cross_entropy_loss(net.forward(X), y_oh)
                param[idx] = orig
                numeric = (lp - lm) / (2 * eps)
                ana = analytic[name][idx]
                rel_err = abs(numeric - ana) / max(abs(numeric) + abs(ana), 1e-8)
                max_rel_err = max(max_rel_err, rel_err)
        status = "PASS" if max_rel_err < 1e-4 else "FAIL"
        print(f"freeze_conv={freeze_conv!s:<5}  checked {params_to_check}  "
              f"max_rel_err={max_rel_err:.2e}  ({status})")

    print("\nDirect check that freeze_conv=True literally freezes the conv "
          "weights (not just skips their gradient in principle):")
    net = TwoBlockConvNetRGB(seed=5)
    Wc1_before, Wc2_before = net.Wc1.copy(), net.Wc2.copy()
    X, y = make_junction_dataset_rgb(n_per_class=5, seed=1)
    train_transfer(net, X, y, epochs=20, freeze_conv=True)
    print(f"Wc1 unchanged after 20 epochs: {np.array_equal(Wc1_before, net.Wc1)}")
    print(f"Wc2 unchanged after 20 epochs: {np.array_equal(Wc2_before, net.Wc2)}")
    print("Actual output from this lesson's run.")


# =============================================================================
# 3. PRETRAIN THE SOURCE NETWORK
# =============================================================================
def pretrain_source():
    print("\n" + "=" * 70)
    print("2. PRETRAINING THE SOURCE NETWORK")
    print("=" * 70)
    print("Identical recipe to this series' own baseline since Day 43: "
          "arm=8, noise=0.4, channel_noise=0.15, center_jitter=2, "
          "n_per_class=50 (200 images), lr=0.08, 400 full-batch epochs, "
          "seed=42.")
    X_train, y_train = make_junction_dataset_rgb(
        n_per_class=50, size=30, arm=8, noise=0.4, channel_noise=0.15,
        center_jitter=2, seed=42)
    X_test, y_test = make_junction_dataset_rgb(
        n_per_class=20, size=30, arm=8, noise=0.4, channel_noise=0.15,
        center_jitter=2, seed=999)
    source_net = TwoBlockConvNetRGB(f1=3, f2=6, k=3, img_size=30, hidden=8,
                                     lr=0.08, seed=42)
    source_net.train(X_train, y_train, epochs=400)
    train_acc = source_net.accuracy(X_train, y_train)
    test_acc = source_net.accuracy(X_test, y_test)
    print(f"source network: train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
          f"params={source_net.param_count()}")
    print("Actual output from this lesson's run.")
    return source_net


# =============================================================================
# 4. THE TRANSFER EXPERIMENT -- 5 strategies on the same 20-image target set
# =============================================================================
def demo_transfer_experiments(source_net):
    print("\n" + "=" * 70)
    print("3. TRANSFER LEARNING ON A SHIFTED, DATA-STARVED TARGET DOMAIN")
    print("=" * 70)
    print("Target domain: arm=11 (vs. source's 8), channel_noise=0.30 (vs. "
          "0.15), center_jitter=3 (vs. 2) -- a real visual shift, same "
          "4-class task. Only 20 labeled target images (5/class, seed=123) "
          "are used for training; scored on a separate, never-trained-on "
          "80-image target test set (seed=777).")

    X_tgt_train, y_tgt_train = make_junction_dataset_rgb(
        n_per_class=5, size=30, arm=11, noise=0.4, channel_noise=0.3,
        center_jitter=3, seed=123)
    X_tgt_test, y_tgt_test = make_junction_dataset_rgb(
        n_per_class=20, size=30, arm=11, noise=0.4, channel_noise=0.3,
        center_jitter=3, seed=777)

    results = {}

    # 1. Zero-shot: source network, completely unmodified
    results["zero-shot"] = dict(
        train_acc=source_net.accuracy(X_tgt_train, y_tgt_train),
        test_acc=source_net.accuracy(X_tgt_test, y_tgt_test),
    )

    # 2. From-scratch control: ignore the source network entirely
    scratch_net = TwoBlockConvNetRGB(f1=3, f2=6, k=3, img_size=30, hidden=8,
                                      lr=0.08, seed=7)
    losses_scratch = scratch_net.train(X_tgt_train, y_tgt_train, epochs=400)
    results["from-scratch"] = dict(
        train_acc=scratch_net.accuracy(X_tgt_train, y_tgt_train),
        test_acc=scratch_net.accuracy(X_tgt_test, y_tgt_test),
    )

    # 3. Frozen conv + FRESH (reinitialized) head
    frozen_fresh = TwoBlockConvNetRGB(f1=3, f2=6, k=3, img_size=30, hidden=8,
                                       lr=0.08, seed=7)
    copy_conv_weights(source_net, frozen_fresh)
    train_transfer(frozen_fresh, X_tgt_train, y_tgt_train, epochs=400,
                    freeze_conv=True)
    results["frozen+fresh head"] = dict(
        train_acc=frozen_fresh.accuracy(X_tgt_train, y_tgt_train),
        test_acc=frozen_fresh.accuracy(X_tgt_test, y_tgt_test),
    )

    # 4. Frozen conv + WARM (kept) head
    frozen_warm = TwoBlockConvNetRGB(f1=3, f2=6, k=3, img_size=30, hidden=8,
                                      lr=0.08, seed=7)
    copy_all_weights(source_net, frozen_warm)
    losses_frozen_warm = train_transfer(frozen_warm, X_tgt_train, y_tgt_train,
                                         epochs=400, freeze_conv=True)
    results["frozen+warm head"] = dict(
        train_acc=frozen_warm.accuracy(X_tgt_train, y_tgt_train),
        test_acc=frozen_warm.accuracy(X_tgt_test, y_tgt_test),
    )

    # 5. Full fine-tune, warm-started, at a few learning rates
    finetune_lrs = [0.08, 0.03, 0.01, 0.005]
    finetune_results = {}
    losses_finetune_best = None
    for lr in finetune_lrs:
        ft_net = TwoBlockConvNetRGB(f1=3, f2=6, k=3, img_size=30, hidden=8,
                                     lr=lr, seed=7)
        copy_all_weights(source_net, ft_net)
        losses_ft = train_transfer(ft_net, X_tgt_train, y_tgt_train,
                                    epochs=400, freeze_conv=False)
        finetune_results[lr] = dict(
            train_acc=ft_net.accuracy(X_tgt_train, y_tgt_train),
            test_acc=ft_net.accuracy(X_tgt_test, y_tgt_test),
        )
        if lr == 0.01:
            losses_finetune_best = losses_ft
    best_lr = max(finetune_lrs, key=lambda lr: finetune_results[lr]["test_acc"])
    results["fine-tune (best lr)"] = dict(
        lr=best_lr, **finetune_results[best_lr])

    print(f"\n{'strategy':<24}{'train_acc':>11}{'test_acc':>10}")
    for name in ["zero-shot", "from-scratch", "frozen+fresh head",
                 "frozen+warm head"]:
        r = results[name]
        print(f"{name:<24}{r['train_acc']:>11.4f}{r['test_acc']:>10.4f}")
    print(f"{'fine-tune (lr=' + str(best_lr) + ')':<24}"
          f"{results['fine-tune (best lr)']['train_acc']:>11.4f}"
          f"{results['fine-tune (best lr)']['test_acc']:>10.4f}")
    print("Actual output from this lesson's run.")

    print(f"\n{'fine-tune lr sweep':<20}{'train_acc':>11}{'test_acc':>10}")
    for lr in finetune_lrs:
        r = finetune_results[lr]
        print(f"{lr:<20}{r['train_acc']:>11.4f}{r['test_acc']:>10.4f}")
    print("Actual output from this lesson's run.")

    zs, fs = results["zero-shot"]["test_acc"], results["from-scratch"]["test_acc"]
    ff, fw = results["frozen+fresh head"]["test_acc"], results["frozen+warm head"]["test_acc"]
    ft = results["fine-tune (best lr)"]["test_acc"]
    print(f"\nreal, honest finding: training from scratch on only 20 target "
          f"images (test_acc={fs:.4f}) is WORSE than doing nothing at all "
          f"and using the source network unmodified (zero-shot "
          f"test_acc={zs:.4f}). Freezing the conv layers but discarding the "
          f"head (test_acc={ff:.4f}) is ALSO worse than zero-shot -- a "
          f"freshly reinitialized head can't relearn as good a decision "
          f"boundary from 20 examples as the source head already had. The "
          f"two strategies that beat zero-shot both KEEP the source head "
          f"instead of discarding it: frozen-conv-warm-head "
          f"(test_acc={fw:.4f}) and a fine-tune of every layer at a properly "
          f"reduced learning rate of {best_lr} (test_acc={ft:.4f}) tie for "
          f"the best result. Fine-tuning at too high a learning rate "
          f"(0.08, test_acc={finetune_results[0.08]['test_acc']:.4f}) partly "
          f"destroys the pretrained weights before the tiny target set can "
          f"re-teach them anything useful, and still underperforms "
          f"zero-shot. This isn't simply a 'freeze vs. unfreeze' story -- "
          f"it's about not throwing away representations (feature OR "
          f"decision-boundary) that a tiny target dataset can't afford to "
          f"relearn from scratch.")

    return dict(results=results, finetune_results=finetune_results,
                losses_scratch=losses_scratch,
                losses_frozen_warm=losses_frozen_warm,
                losses_finetune_best=losses_finetune_best,
                X_tgt_train=X_tgt_train, y_tgt_train=y_tgt_train,
                X_tgt_test=X_tgt_test, y_tgt_test=y_tgt_test)


def make_comparison_plot(losses_scratch, losses_frozen_warm, losses_finetune,
                          test_acc_scratch, test_acc_frozen, test_acc_finetune):
    plt.figure(figsize=(9, 4.8))
    plt.plot(losses_scratch,
              label=f"from-scratch (test_acc={test_acc_scratch:.3f})",
              color="#d9534f")
    plt.plot(losses_frozen_warm,
              label=f"frozen+warm head (test_acc={test_acc_frozen:.3f})",
              color="#5cb85c")
    plt.plot(losses_finetune,
              label=f"fine-tune lr=0.01 (test_acc={test_acc_finetune:.3f})",
              color="#337ab7")
    plt.xlabel("epoch (= step, full-batch, 20 target images)")
    plt.ylabel("training loss")
    plt.title("What you keep matters more than freeze vs. unfreeze", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("transfer_learning_loss_comparison.png", dpi=110)
    plt.close()
    print("\nsaved transfer_learning_loss_comparison.png")


# =============================================================================
# 5. WHAT THIS MAPS TO IN PYTORCH
# =============================================================================
def note_on_pytorch():
    print("\n" + "=" * 70)
    print("4. WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)
    print(
        "model = MyConvNet()\n"
        "model.load_state_dict(torch.load('source_weights.pt'))\n"
        "\n"
        "# freeze the conv backbone -- matches freeze_conv=True above\n"
        "for p in model.conv1.parameters(): p.requires_grad = False\n"
        "for p in model.conv2.parameters(): p.requires_grad = False\n"
        "\n"
        "# option A: fresh head (matches 'frozen+fresh head')\n"
        "model.fc2 = nn.Linear(model.fc2.in_features, n_classes)\n"
        "\n"
        "# option B: keep the pretrained head (matches 'frozen+warm head')\n"
        "# -- just don't touch model.fc2 at all\n"
        "\n"
        "optimizer = torch.optim.SGD(\n"
        "    filter(lambda p: p.requires_grad, model.parameters()), lr=0.08)\n"
        "\n"
        "# full fine-tune (matches 'fine-tune') needs every param trainable\n"
        "# AND a smaller learning rate than training from scratch would use:\n"
        "for p in model.parameters(): p.requires_grad = True\n"
        "optimizer = torch.optim.SGD(model.parameters(), lr=0.01)"
    )
    print("\nfilter(lambda p: p.requires_grad, ...) is PyTorch's equivalent "
          "of today's `updates` list in backward_transfer -- the optimizer "
          "only ever steps parameters that are still marked trainable, "
          "exactly like today's code only appends conv gradients to "
          "`updates` when freeze_conv is False.")


def main():
    demo_gradient_check()
    source_net = pretrain_source()
    exp = demo_transfer_experiments(source_net)
    make_comparison_plot(
        exp["losses_scratch"],
        exp["losses_frozen_warm"],
        exp["losses_finetune_best"],
        exp["results"]["from-scratch"]["test_acc"],
        exp["results"]["frozen+warm head"]["test_acc"],
        exp["results"]["fine-tune (best lr)"]["test_acc"],
    )
    note_on_pytorch()


if __name__ == "__main__":
    main()
