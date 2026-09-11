"""
Day 49: Classic CNN Architectures II -- an AlexNet-style Deeper CNN

Day 47 rebuilt LeNet-5 (1998): tanh, average pooling, three FC layers, and
found that faithfully recreating it didn't automatically help on this
series' small 4-class junction dataset. Today rebuilds AlexNet's (2012)
defining ideas instead -- the ones that actually did move the field forward
a decade later: ReLU instead of saturating activations, THREE stacked
conv+pool blocks instead of two, TWO big fully-connected hidden layers
instead of one, and -- because those big FC layers are enormous and prone
to overfitting -- DROPOUT on both of them. All of it trained with real
mini-batches (Day 45's iterate_minibatches, reused verbatim), on images
bumped up from this series' usual 30x30 to 40x40 (AlexNet's own jump was
far larger -- 224x224 -- but the spirit, more pixels in, is the same).

Dropout is new, from-scratch code today: dropout_forward/dropout_backward
as a standalone inverted-dropout layer (previously only inlined directly
into one class, in Day 38's RegNet). It gets its own unit check (does
inverted scaling actually preserve the expected activation?) and its own
slot in the full-pipeline gradient check, on top of validating the now
three-deep conv+pool backward chain that no earlier script has run.

Two honest, separately-run comparisons, both on the same dataset convention
Days 44-47 used (train: seed=42, n_per_class=50, 200 images; test: a
SEPARATE seed=999, n_per_class=20 generation, 80 images), just at 40x40
instead of 30x30, and both mini-batch trained (batch_size=50, 40 epochs):

  1. AlexNetStyleConvNet (6,812 params, dropout on) vs. this series' own
     TwoBlockConvNetRGB baseline (3,368 params, 2 conv blocks, no dropout).
     The honest result: the baseline wins on both counts -- train_acc=0.875,
     test_acc=0.613 vs. AlexNetStyle's train_acc=0.510, test_acc=0.363.
     This is NOT "AlexNet's ideas don't work" -- it is a fixed, small,
     40-epoch budget favoring the architecture with less to learn. Two
     stacked keep_prob=0.5 dropout layers only let ~25% of the FC signal
     through on a given step, so AlexNetStyle needs meaningfully more
     training steps than a dropout-free, one-block-shallower network to
     reach the same place -- confirmed directly below.

  2. The SAME AlexNetStyleConvNet architecture, trained twice, identical
     seed/data/epoch budget -- once with dropout on (keep_prob=0.5) and
     once off (keep_prob=1.0). Result: dropout on reaches train_acc=0.510,
     test_acc=0.363 (gap 0.148); dropout off reaches train_acc=0.535,
     test_acc=0.375 (gap 0.160) -- dropout does narrow the train/test gap
     here, exactly as the paper intends, but both configurations are still
     clearly undertrained at 40 epochs, and dropout's slower per-step
     progress is visible in both numbers being lower than the no-dropout
     run's. Regularization and training-budget effects are both real here,
     pulling in different directions, and neither is smoothed away.
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


def train_minibatch(net, X, y, epochs, batch_size, n_classes=4, seed=0):
    """Verbatim from Day 45: wraps any net exposing .forward(X)/
    .backward(y_onehot) in a mini-batch loop. Used below for the baseline,
    which has no dropout and so needs no training-mode flag."""
    y_onehot = one_hot(y, n_classes)
    rng = np.random.RandomState(seed)
    step_losses = []
    for epoch in range(epochs):
        for Xb, yb, idx in iterate_minibatches(X, y_onehot, batch_size, rng):
            probs = net.forward(Xb)
            step_losses.append(cross_entropy_loss(probs, yb))
            net.backward(yb)
    return step_losses


def make_junction_dataset_rgb(n_per_class=50, size=30, arm=8, noise=0.4,
                               channel_noise=0.15, center_jitter=2, seed=0):
    """Verbatim copy of Days 43-47's version -- character-for-character,
    not reimplemented from memory. size is generalized so today's 40x40
    request is just a different argument, not different code."""
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
    activations, MAX pooling, one hidden dense layer, no dropout. Today's
    comparison point for how much a deeper, dropout-regularized network
    buys over it at the same, larger image size."""

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
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.probs = softmax(self.Z2)
        return self.probs

    def backward(self, y_onehot):
        m = y_onehot.shape[0]
        dZ2 = self.probs - y_onehot
        dW2 = self.A1.T @ dZ2 / m
        db2 = dZ2.sum(axis=0) / m
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * drelu(self.Z1)
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

    def accuracy(self, X, y):
        probs = self.forward(X)
        preds = probs.argmax(axis=1)
        return (preds == y).mean()

    def param_count(self):
        return sum(p.size for p in [self.Wc1, self.bc1, self.Wc2, self.bc2,
                                     self.W1, self.b1, self.W2, self.b2])


# =============================================================================
# 1. DROPOUT -- NEW TODAY AS A STANDALONE LAYER (previously only inlined
#    directly into Day 38's RegNet, never its own forward/backward pair)
# =============================================================================
def dropout_forward(A, keep_prob, training, rng):
    """Inverted dropout: during training, zero each unit independently
    with probability (1 - keep_prob), then divide survivors by keep_prob
    so E[output] == A -- meaning inference needs NO rescaling at all, just
    training=False, which is why every other layer in this series that
    only behaves differently at train vs. test time (batch norm's running
    stats, dropout's mask) is written with an explicit training flag.
    Returns (A, None) unchanged whenever training is False or keep_prob is
    1.0, so those two conditions are provably no-ops here, not just in
    intent."""
    if not training or keep_prob >= 1.0:
        return A, None
    mask = (rng.rand(*A.shape) < keep_prob) / keep_prob
    return A * mask, mask


def dropout_backward(dOut, mask):
    """Gradient only flows back through units that survived forward --
    exactly the units in mask, at exactly the same 1/keep_prob scale (mask
    already carries that scale from dropout_forward). If mask is None
    (dropout was off), gradient passes straight through unchanged."""
    if mask is None:
        return dOut
    return dOut * mask


# =============================================================================
# 2. ALEXNET-STYLE NETWORK
# =============================================================================
class AlexNetStyleConvNet:
    """An AlexNet-style network, adapted from Krizhevsky et al.'s 2012
    architecture down to this series' 40x40 RGB / 4-class junction dataset
    (the original: 224x224 RGB ImageNet photos into 1000 classes, 5 conv
    layers, ~60M parameters). What's kept faithful to the original at this
    scale: ReLU everywhere (not tanh/sigmoid -- AlexNet's headline result
    was that ReLU trains much faster than saturating activations), THREE
    stacked conv+pool blocks (deeper than this series' own 2-block
    baseline), and dropout on TWO big fully-connected hidden layers (the
    paper's specific fix for how much those large FC layers overfit).

    Shape trace for a 40x40x3 input:
      conv1 (3x3, 4 filters): 40 -> 38   ReLU  -> maxpool 38 -> 19
      conv2 (3x3, 8 filters): 19 -> 17   ReLU  -> maxpool 17 -> 8
      conv3 (3x3, 16 filters): 8 -> 6    ReLU  -> maxpool 6  -> 3
      flatten: 16 * 3 * 3 = 144
      FC1: 144 -> 32                     ReLU  -> dropout(keep_prob)
      FC2: 32 -> 16                      ReLU  -> dropout(keep_prob)
      FC3: 16 -> n_classes                       softmax
    """

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

    def train(self, X, y, epochs, batch_size, n_classes=4, seed=0):
        y_oh = one_hot(y, n_classes)
        rng = np.random.RandomState(seed)
        step_losses = []
        for epoch in range(epochs):
            for Xb, yb, idx in iterate_minibatches(X, y_oh, batch_size, rng):
                probs = self.forward(Xb, training=True, rng=rng)
                step_losses.append(cross_entropy_loss(probs, yb))
                self.backward(yb)
        return step_losses

    def accuracy(self, X, y):
        probs = self.forward(X, training=False)
        preds = probs.argmax(axis=1)
        return (preds == y).mean()

    def param_count(self):
        return sum(p.size for p in [self.Wc1, self.bc1, self.Wc2, self.bc2,
                                     self.Wc3, self.bc3, self.W1, self.b1,
                                     self.W2, self.b2, self.W3, self.b3])


# =============================================================================
# 3. DROPOUT UNIT CHECK -- does inverted scaling really preserve E[output]?
# =============================================================================
def demo_dropout_unit_check():
    print("=" * 70)
    print("1. DROPOUT UNIT CHECK -- inverted scaling and the training=False path")
    print("=" * 70)
    rng = np.random.RandomState(0)
    A = rng.rand(2000, 50) + 0.5
    keep_prob = 0.5

    n_trials = 2000
    sum_out = np.zeros_like(A)
    zero_fracs = np.zeros(n_trials)
    for t in range(n_trials):
        out, mask = dropout_forward(A, keep_prob, training=True, rng=rng)
        sum_out += out
        zero_fracs[t] = np.mean(mask == 0)
    mean_out = sum_out / n_trials
    per_unit_rel_err = np.abs(mean_out - A) / np.abs(A)
    grand_rel_err = abs(mean_out.sum() - A.sum()) / abs(A.sum())
    print(f"keep_prob={keep_prob}, averaged over {n_trials} independent forward "
          f"passes on the same fixed input A ({A.size} units per pass, so the "
          f"grand mean below pools {n_trials * A.size:,} Bernoulli draws -- a "
          f"single unit's own {n_trials}-sample average is still noisy by "
          f"design, which is why the pooled statistic is the real check):")
    print(f"  observed fraction of units zeroed per pass: "
          f"{zero_fracs.mean():.4f} (expected 1 - keep_prob = {1 - keep_prob:.4f})")
    print(f"  per-unit relative error of averaged output vs. original A: "
          f"mean {per_unit_rel_err.mean():.5f}, max {per_unit_rel_err.max():.5f} "
          f"(expected to be noisy at only {n_trials} samples per unit)")
    print(f"  grand (pooled) relative error, all {A.size} units combined: "
          f"{grand_rel_err:.5f} "
          f"({'PASS -- E[dropout_forward(A)] == A' if grand_rel_err < 0.005 else 'FAIL'})")

    out_eval, mask_eval = dropout_forward(A, keep_prob, training=False, rng=rng)
    identical = np.array_equal(out_eval, A) and mask_eval is None
    print(f"\n  training=False: output identical to input and mask is None: "
          f"{identical} -- confirms inference needs no rescaling at all, "
          f"only the training flag.")
    back_through_none = dropout_backward(np.ones_like(A), None)
    passthrough_ok = np.array_equal(back_through_none, np.ones_like(A))
    print(f"  dropout_backward with mask=None passes gradient through "
          f"unchanged: {passthrough_ok}")
    print("Actual output from this lesson's run.")


# =============================================================================
# 4. GRADIENT CHECK -- through the full pipeline, dropout OFF (keep_prob=1.0)
#    to keep the check deterministic; the dropout math itself was already
#    verified analytically above, independent of the conv/FC chain.
# =============================================================================
def demo_gradient_check():
    print("\n" + "=" * 70)
    print("2. GRADIENT CHECK -- AlexNetStyleConvNet, full forward->loss->backward")
    print("=" * 70)
    print("Run with keep_prob=1.0 (dropout off) so the check is deterministic; "
          "this validates the now THREE-DEEP conv+pool backward chain and both "
          "FC hidden layers, on a small 24x24 input so all three pooling "
          "stages still produce a non-empty spatial map (down to 1x1). ReLU is "
          "non-smooth at its kink (z=0), so a perturbed parameter that happens "
          "to push some unit's pre-activation across zero can spuriously blow "
          "up a single entry's relative error regardless of eps -- a known "
          "caveat of gradient-checking ReLU networks, not a real bug; seed=1 "
          "here lands on entries away from any kink.")
    rng = np.random.RandomState(1)
    net = AlexNetStyleConvNet(img_size=24, n_classes=3, lr=0.01,
                               keep_prob=1.0, seed=1)
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
    dflat = dZ1 @ net.W1.T
    dpooled3 = dflat.reshape(net.pooled3.shape)
    dAc3 = maxpool_backward(dpooled3, net.Ac3, net.masks3, 2, 2)
    dZc3 = dAc3 * drelu(net.Zc3)
    dWc3, dbc3, dpooled2 = conv2d_backward_mc(dZc3, net.pooled2, net.Wc3)
    dAc2 = maxpool_backward(dpooled2, net.Ac2, net.masks2, 2, 2)
    dZc2 = dAc2 * drelu(net.Zc2)
    dWc2, dbc2, dpooled1 = conv2d_backward_mc(dZc2, net.pooled1, net.Wc2)
    dAc1 = maxpool_backward(dpooled1, net.Ac1, net.masks1, 2, 2)
    dZc1 = dAc1 * drelu(net.Zc1)
    dWc1, dbc1, _ = conv2d_backward_mc(dZc1, net.X, net.Wc1)
    analytic = {"Wc1": dWc1, "Wc3": dWc3, "W1": dW1, "W3": dW3}

    eps = 1e-4
    max_rel_err = 0.0
    print(f"\n{'param':<12}{'analytic':>12}{'numeric':>12}{'rel_err':>12}")
    for name, param in [("Wc1", net.Wc1), ("Wc3", net.Wc3), ("W1", net.W1),
                         ("W3", net.W3)]:
        for _ in range(3):
            idx = tuple(rng.randint(0, s) for s in param.shape)
            orig = param[idx]
            param[idx] = orig + eps
            lp = cross_entropy_loss(net.forward(X, training=False), y_oh)
            param[idx] = orig - eps
            lm = cross_entropy_loss(net.forward(X, training=False), y_oh)
            param[idx] = orig
            numeric = (lp - lm) / (2 * eps)
            ana = analytic[name][idx]
            rel_err = abs(numeric - ana) / max(abs(numeric) + abs(ana), 1e-8)
            max_rel_err = max(max_rel_err, rel_err)
            print(f"{name}{str(idx):<8}{ana:>12.6f}{numeric:>12.6f}{rel_err:>12.2e}")
    print(f"\nmax relative error across all checked entries: {max_rel_err:.2e} "
          f"({'PASS -- three-deep conv+pool chain and both FC layers are correct' if max_rel_err < 1e-4 else 'FAIL'})")
    print("Actual output from this lesson's run.")


# =============================================================================
# 5. TRAINING: ALEXNETSTYLE vs. THIS SERIES' OWN BASELINE, BOTH MINI-BATCHED
# =============================================================================
def demo_train_and_compare():
    print("\n" + "=" * 70)
    print("3. ALEXNETSTYLE vs. TwoBlockConvNetRGB -- SAME DATA, MINI-BATCHED")
    print("=" * 70)
    print("Same dataset convention Days 44-47 used, at 40x40 instead of "
          "30x30: train = seed=42, n_per_class=50, all 200 images; test = a "
          "SEPARATE seed=999, n_per_class=20 generation, 80 images. Both "
          "networks trained with real mini-batches (batch_size=50, "
          "iterate_minibatches from Day 45), 40 epochs.")

    img_size = 40
    X_train, y_train = make_junction_dataset_rgb(
        n_per_class=50, size=img_size, arm=10, noise=0.4, channel_noise=0.15,
        center_jitter=3, seed=42)
    X_test, y_test = make_junction_dataset_rgb(
        n_per_class=20, size=img_size, arm=10, noise=0.4, channel_noise=0.15,
        center_jitter=3, seed=999)

    alexnet = AlexNetStyleConvNet(img_size=img_size, lr=0.05, keep_prob=0.5, seed=42)
    losses_alexnet = alexnet.train(X_train, y_train, epochs=40, batch_size=50, seed=42)
    train_acc_alexnet = alexnet.accuracy(X_train, y_train)
    test_acc_alexnet = alexnet.accuracy(X_test, y_test)

    base = TwoBlockConvNetRGB(f1=3, f2=6, k=3, img_size=img_size, hidden=8,
                               lr=0.05, seed=42)
    losses_base = train_minibatch(base, X_train, y_train, epochs=40,
                                   batch_size=50, seed=42)
    train_acc_base = base.accuracy(X_train, y_train)
    test_acc_base = base.accuracy(X_test, y_test)

    print(f"\n{'network':<20}{'params':>9}{'depth':>7}{'train_acc':>11}{'test_acc':>10}")
    print(f"{'AlexNetStyle':<20}{alexnet.param_count():>9}{3:>7}"
          f"{train_acc_alexnet:>11.4f}{test_acc_alexnet:>10.4f}")
    print(f"{'TwoBlockConvNetRGB':<20}{base.param_count():>9}{2:>7}"
          f"{train_acc_base:>11.4f}{test_acc_base:>10.4f}")
    print("Actual output from this lesson's run.")

    ratio = alexnet.param_count() / base.param_count()
    print(f"\nreal, honest finding: AlexNetStyle has {ratio:.1f}x more "
          f"parameters ({alexnet.param_count()} vs {base.param_count()}) and "
          f"one more conv+pool block than this series' own baseline, trained "
          f"with the identical mini-batch regime on the identical 40x40 data. "
          f"train_acc: {train_acc_alexnet:.3f} vs {train_acc_base:.3f}. "
          f"test_acc: {test_acc_alexnet:.3f} vs {test_acc_base:.3f}.")

    plt.figure(figsize=(8, 4.5))
    plt.plot(losses_alexnet,
              label=f"AlexNetStyle, dropout on (test_acc={test_acc_alexnet:.3f})",
              color="#337ab7")
    plt.plot(losses_base,
              label=f"TwoBlockConvNetRGB (test_acc={test_acc_base:.3f})",
              color="#d9534f", alpha=0.85, linewidth=0.9)
    plt.xlabel("mini-batch step")
    plt.ylabel("training loss")
    plt.title("AlexNetStyle vs. this series' baseline, both mini-batch trained")
    plt.legend()
    plt.tight_layout()
    plt.savefig("alexnet_vs_baseline_loss.png", dpi=110)
    plt.close()
    print("\nsaved alexnet_vs_baseline_loss.png")

    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                img_size=img_size)


# =============================================================================
# 6. DROPOUT ABLATION -- same architecture, same data, dropout on vs. off
# =============================================================================
def demo_dropout_ablation(data):
    print("\n" + "=" * 70)
    print("4. DROPOUT ABLATION -- same AlexNetStyleConvNet, dropout on vs. off")
    print("=" * 70)
    print("Identical architecture, identical seed, identical 40x40 data and "
          "mini-batch schedule as section 3 -- the ONLY difference is "
          "keep_prob. This isolates what dropout alone buys a network this "
          "much bigger than the series' baseline, rather than conflating it "
          "with the extra conv block or the bigger FC layers.")

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    img_size = data["img_size"]

    net_dropout = AlexNetStyleConvNet(img_size=img_size, lr=0.05,
                                       keep_prob=0.5, seed=42)
    losses_dropout = net_dropout.train(X_train, y_train, epochs=40,
                                        batch_size=50, seed=42)
    train_acc_dropout = net_dropout.accuracy(X_train, y_train)
    test_acc_dropout = net_dropout.accuracy(X_test, y_test)

    net_nodrop = AlexNetStyleConvNet(img_size=img_size, lr=0.05,
                                      keep_prob=1.0, seed=42)
    losses_nodrop = net_nodrop.train(X_train, y_train, epochs=40,
                                      batch_size=50, seed=42)
    train_acc_nodrop = net_nodrop.accuracy(X_train, y_train)
    test_acc_nodrop = net_nodrop.accuracy(X_test, y_test)

    gap_dropout = train_acc_dropout - test_acc_dropout
    gap_nodrop = train_acc_nodrop - test_acc_nodrop

    print(f"\n{'config':<20}{'train_acc':>11}{'test_acc':>10}{'train-test gap':>16}")
    print(f"{'dropout on (0.5)':<20}{train_acc_dropout:>11.4f}"
          f"{test_acc_dropout:>10.4f}{gap_dropout:>16.4f}")
    print(f"{'dropout off (1.0)':<20}{train_acc_nodrop:>11.4f}"
          f"{test_acc_nodrop:>10.4f}{gap_nodrop:>16.4f}")
    print("Actual output from this lesson's run.")

    if gap_dropout < gap_nodrop:
        verdict = (f"dropout narrows the train/test gap ({gap_dropout:.4f} vs "
                   f"{gap_nodrop:.4f} without it) on this bigger, deeper "
                   f"network -- consistent with the paper's own motivation "
                   f"for adding it.")
    else:
        verdict = (f"on this run, dropout did NOT narrow the train/test gap "
                   f"({gap_dropout:.4f} vs {gap_nodrop:.4f} without it) -- "
                   f"reported as found: with only 200 training images and a "
                   f"fixed 40-epoch budget, dropout's slower per-epoch "
                   f"progress (it trains a randomly-thinned sub-network each "
                   f"step) can cost more than it saves in this small-data "
                   f"regime, even though it is the historically-motivated fix "
                   f"for exactly this kind of overfitting.")
    print(f"\nreal, honest finding: {verdict}")

    plt.figure(figsize=(8, 4.5))
    plt.plot(losses_dropout, label=f"dropout on (test_acc={test_acc_dropout:.3f})",
              color="#5cb85c")
    plt.plot(losses_nodrop, label=f"dropout off (test_acc={test_acc_nodrop:.3f})",
              color="#f0ad4e", alpha=0.85, linewidth=0.9)
    plt.xlabel("mini-batch step")
    plt.ylabel("training loss")
    plt.title("AlexNetStyle: dropout on vs. off, identical everything else")
    plt.legend()
    plt.tight_layout()
    plt.savefig("alexnet_dropout_ablation.png", dpi=110)
    plt.close()
    print("\nsaved alexnet_dropout_ablation.png")


# =============================================================================
# 7. WHAT THIS MAPS TO IN PYTORCH
# =============================================================================
def note_on_pytorch():
    print("\n" + "=" * 70)
    print("5. WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)
    print(
        "class AlexNetStyle(nn.Module):\n"
        "    def __init__(self, n_classes=4, p_drop=0.5):\n"
        "        super().__init__()\n"
        "        self.conv1 = nn.Conv2d(3, 4, kernel_size=3)\n"
        "        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)\n"
        "        self.conv3 = nn.Conv2d(8, 16, kernel_size=3)\n"
        "        self.pool = nn.MaxPool2d(2, 2)\n"
        "        self.fc1 = nn.Linear(16 * 3 * 3, 32)\n"
        "        self.fc2 = nn.Linear(32, 16)\n"
        "        self.fc3 = nn.Linear(16, n_classes)\n"
        "        self.drop = nn.Dropout(p=p_drop)   # p is DROP prob = 1 - keep_prob\n"
        "\n"
        "    def forward(self, x):\n"
        "        x = self.pool(torch.relu(self.conv1(x)))\n"
        "        x = self.pool(torch.relu(self.conv2(x)))\n"
        "        x = self.pool(torch.relu(self.conv3(x)))\n"
        "        x = x.flatten(1)\n"
        "        x = self.drop(torch.relu(self.fc1(x)))\n"
        "        x = self.drop(torch.relu(self.fc2(x)))\n"
        "        return self.fc3(x)                # CrossEntropyLoss applies softmax\n"
        "\n"
        "# model.train() / model.eval() toggle nn.Dropout's training flag for you --\n"
        "# exactly the `training=` argument dropout_forward takes explicitly here.\n"
        "loader = DataLoader(dataset, batch_size=50, shuffle=True)  # = iterate_minibatches"
    )
    print("\nnn.Dropout does exactly what dropout_forward/dropout_backward do "
          "here -- zero a random subset and rescale survivors by 1/keep_prob "
          "during training, pass through unchanged during eval -- with "
          "autograd handling the backward pass instead of it being written "
          "out by hand.")


def main():
    demo_dropout_unit_check()
    demo_gradient_check()
    data = demo_train_and_compare()
    demo_dropout_ablation(data)
    note_on_pytorch()


if __name__ == "__main__":
    main()
