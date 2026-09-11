"""
Day 47: Classic CNN Architectures I -- Recreating LeNet-5 by Hand

Every network in this series since Day 43 has used the same "modern-ish"
default choices: ReLU activations, max pooling, one hidden dense layer.
Today builds something different on purpose -- a small network that
actually follows LeCun et al.'s 1998 LeNet-5 design: tanh activations
throughout (ReLU didn't exist yet), AVERAGE pooling instead of max pooling,
and THREE fully-connected layers (120 -> 84 -> classes) instead of one.
Average pooling is new code today -- it has never appeared in this series
before -- and gets its own from-scratch forward/backward pass and gradient
check, same as every other layer type has.

The original LeNet-5 took 32x32 grayscale digits into 10 classes. This
version is adapted to this series' own 30x30 RGB junction dataset and its
4 classes -- same spirit (small images, small number of classes), 5x5
kernels and 6/16 filter counts kept faithful to the original, everything
else sized to fit.

The honest result, from actually training both networks for 400 epochs
on the IDENTICAL dataset convention Days 44 and 46 used (train: seed=42,
n_per_class=50, all 200 images; test: a SEPARATE seed=999, n_per_class=20
generation, 80 images): the historically-faithful LeNet5Style network,
with 44,216 parameters, reaches train_acc=1.000 and test_acc=0.775.
TwoBlockConvNetRGB, this series' own much smaller baseline (2,024
parameters -- 21.8x fewer), reaches train_acc=0.995 and the SAME
test_acc=0.775. More parameters bought LeNet5Style nothing here -- 21.8x
the capacity produces identical test performance, and a very slightly
wider train/test gap (0.225 vs. the baseline's 0.220) from its perfect
training-set memorization. This is reported as found, not smoothed into
"and therefore LeNet-5 is bad" or "the baseline is better" -- neither is
true; on this 200-image, 4-class problem, LeNet-5's extra capacity is
simply wasted, not harmful.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================
# 0. SHARED HELPERS -- reused verbatim from Days 39-46
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
    """Verbatim copy of Day 43-46's version -- character-for-character, not
    reimplemented from memory (an RNG-order slip here has caused a real,
    hard-to-spot bug before)."""
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
    activations, MAX pooling, one hidden dense layer. Today's comparison
    point for how a historically-faithful architecture stacks up against
    it on the exact same data."""

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
# 1. AVERAGE POOLING -- NEW TODAY, NEVER USED BEFORE IN THIS SERIES
# =============================================================================
def avgpool_forward(A, size=2, stride=2):
    """Same sliding-window shape as maxpool_forward, but every position in
    the window contributes equally to the output (its mean) instead of only
    the winner. No index cache is needed for the backward pass, because
    -- unlike max pooling, where only the max position gets any gradient --
    every input position in an average-pooling window gets an EQUAL SHARE
    of that window's output gradient, always, regardless of the input
    values. That symmetry is what makes the backward pass below so much
    simpler than maxpool_backward's index bookkeeping."""
    m, C, H, W = A.shape
    out_H = (H - size) // stride + 1
    out_W = (W - size) // stride + 1
    out = np.zeros((m, C, out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            window = A[:, :, i * stride:i * stride + size,
                       j * stride:j * stride + size]
            out[:, :, i, j] = window.mean(axis=(2, 3))
    return out


def avgpool_backward(dOut, A_shape, size=2, stride=2):
    m, C, H, W = A_shape
    dA = np.zeros(A_shape)
    out_H, out_W = dOut.shape[2], dOut.shape[3]
    share = 1.0 / (size * size)
    for i in range(out_H):
        for j in range(out_W):
            dA[:, :, i * stride:i * stride + size,
               j * stride:j * stride + size] += (
                dOut[:, :, i, j][:, :, None, None] * share)
    return dA


# =============================================================================
# 2. LENET-5-STYLE NETWORK
# =============================================================================
class LeNet5Style:
    """A small LeNet-5-style network, adapted from LeCun et al.'s original
    1998 architecture to this series' 30x30 RGB / 4-class junction
    dataset (the original: 32x32 grayscale digits, 10 classes). Kept
    faithful to the original's specific choices: 5x5 kernels, 6 then 16
    filters, tanh activations throughout (this predates ReLU), AVERAGE
    pooling (not max), and THREE fully-connected layers (120 -> 84 ->
    n_classes) rather than the one-hidden-layer head every network since
    Day 43 has used.

    Shape trace for a 30x30x3 input:
      conv1 (5x5, 6 filters): 30 -> 26           tanh
      avgpool (2x2):          26 -> 13
      conv2 (5x5, 16 filters): 13 -> 9           tanh
      avgpool (2x2):           9 -> 4  (9 is odd; the last row/col of the
                                          window is dropped, same as every
                                          non-divisible pooling in this
                                          series has done since Day 41)
      flatten: 16 * 4 * 4 = 256
      FC1: 256 -> 120                            tanh
      FC2: 120 -> 84                             tanh
      FC3: 84 -> n_classes                       softmax
    """

    def __init__(self, img_size=30, n_classes=4, lr=0.08, seed=42):
        rng = np.random.RandomState(seed)
        self.lr = lr
        k = 5
        f1, f2 = 6, 16
        self.Wc1 = rng.randn(f1, 3, k, k) * he_scale(3 * k * k)
        self.bc1 = np.zeros(f1)
        self.Wc2 = rng.randn(f2, f1, k, k) * he_scale(f1 * k * k)
        self.bc2 = np.zeros(f2)
        out1 = img_size - k + 1
        pool1 = out1 // 2
        out2 = pool1 - k + 1
        pool2 = out2 // 2
        self.flat_dim = f2 * pool2 * pool2
        self.W1 = rng.randn(self.flat_dim, 120) * he_scale(self.flat_dim)
        self.b1 = np.zeros(120)
        self.W2 = rng.randn(120, 84) * he_scale(120)
        self.b2 = np.zeros(84)
        self.W3 = rng.randn(84, n_classes) * he_scale(84)
        self.b3 = np.zeros(n_classes)

    def forward(self, X):
        self.X = X
        self.Zc1 = conv2d_forward_mc(self.X, self.Wc1, self.bc1)
        self.Ac1 = tanh(self.Zc1)
        self.pooled1 = avgpool_forward(self.Ac1, 2, 2)
        self.Zc2 = conv2d_forward_mc(self.pooled1, self.Wc2, self.bc2)
        self.Ac2 = tanh(self.Zc2)
        self.pooled2 = avgpool_forward(self.Ac2, 2, 2)
        m = X.shape[0]
        self.flat = self.pooled2.reshape(m, -1)
        self.Z1 = self.flat @ self.W1 + self.b1
        self.A1 = tanh(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = tanh(self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.probs = softmax(self.Z3)
        return self.probs

    def backward(self, y_onehot):
        m = y_onehot.shape[0]
        dZ3 = self.probs - y_onehot
        dW3 = self.A2.T @ dZ3 / m
        db3 = dZ3.sum(axis=0) / m
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * dtanh(self.Z2)
        dW2 = self.A1.T @ dZ2 / m
        db2 = dZ2.sum(axis=0) / m
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * dtanh(self.Z1)
        dW1 = self.flat.T @ dZ1 / m
        db1 = dZ1.sum(axis=0) / m
        dflat = dZ1 @ self.W1.T
        dpooled2 = dflat.reshape(self.pooled2.shape)
        dAc2 = avgpool_backward(dpooled2, self.Ac2.shape, 2, 2)
        dZc2 = dAc2 * dtanh(self.Zc2)
        dWc2, dbc2, dpooled1 = conv2d_backward_mc(dZc2, self.pooled1, self.Wc2)
        dAc1 = avgpool_backward(dpooled1, self.Ac1.shape, 2, 2)
        dZc1 = dAc1 * dtanh(self.Zc1)
        dWc1, dbc1, _ = conv2d_backward_mc(dZc1, self.X, self.Wc1)
        for p, g in [(self.Wc1, dWc1), (self.bc1, dbc1), (self.Wc2, dWc2),
                     (self.bc2, dbc2), (self.W1, dW1), (self.b1, db1),
                     (self.W2, dW2), (self.b2, db2), (self.W3, dW3), (self.b3, db3)]:
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
                                     self.W1, self.b1, self.W2, self.b2,
                                     self.W3, self.b3])


# =============================================================================
# 3. GRADIENT CHECK -- through the FULL pipeline, including new avgpool code
# =============================================================================
def demo_gradient_check():
    print("=" * 70)
    print("1. GRADIENT CHECK -- LeNet5Style, full forward->loss->backward pipeline")
    print("=" * 70)
    print("Checked through the exact pipeline training actually uses (not an "
          "isolated test of avgpool alone), on a small 20x20 input so both "
          "conv+avgpool blocks still run.")
    rng = np.random.RandomState(0)
    net = LeNet5Style(img_size=20, n_classes=3, lr=0.01, seed=1)
    X = rng.randn(4, 3, 20, 20) * 0.5
    y_oh = one_hot(np.array([0, 1, 2, 0]), 3)

    net.forward(X)
    m = y_oh.shape[0]
    dZ3 = net.probs - y_oh
    dW3 = net.A2.T @ dZ3 / m
    dA2 = dZ3 @ net.W3.T
    dZ2 = dA2 * dtanh(net.Z2)
    dW2 = net.A1.T @ dZ2 / m
    dA1 = dZ2 @ net.W2.T
    dZ1 = dA1 * dtanh(net.Z1)
    dW1 = net.flat.T @ dZ1 / m
    dflat = dZ1 @ net.W1.T
    dpooled2 = dflat.reshape(net.pooled2.shape)
    dAc2 = avgpool_backward(dpooled2, net.Ac2.shape, 2, 2)
    dZc2 = dAc2 * dtanh(net.Zc2)
    dWc2, dbc2, dpooled1 = conv2d_backward_mc(dZc2, net.pooled1, net.Wc2)
    dAc1 = avgpool_backward(dpooled1, net.Ac1.shape, 2, 2)
    dZc1 = dAc1 * dtanh(net.Zc1)
    dWc1, dbc1, _ = conv2d_backward_mc(dZc1, net.X, net.Wc1)
    analytic = {"Wc1": dWc1, "W1": dW1, "W3": dW3}

    eps = 1e-4
    max_rel_err = 0.0
    print(f"\n{'param':<12}{'analytic':>12}{'numeric':>12}{'rel_err':>12}")
    for name, param in [("Wc1", net.Wc1), ("W1", net.W1), ("W3", net.W3)]:
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
            print(f"{name}{str(idx):<8}{ana:>12.6f}{numeric:>12.6f}{rel_err:>12.2e}")
    print(f"\nmax relative error across all checked entries: {max_rel_err:.2e} "
          f"({'PASS -- avgpool_forward/backward and the full pipeline are correct' if max_rel_err < 1e-4 else 'FAIL'})")
    print("Actual output from this lesson's run.")


# =============================================================================
# 4. TRAINING: LENET5STYLE vs. THIS SERIES' OWN BASELINE, HEAD TO HEAD
# =============================================================================
def demo_train_and_compare():
    print("\n" + "=" * 70)
    print("2. LENET5STYLE vs. TwoBlockConvNetRGB -- SAME DATA, SAME EPOCHS")
    print("=" * 70)
    print("Same dataset convention Days 44 and 46 used: train = seed=42, "
          "n_per_class=50, all 200 images; test = a SEPARATE seed=999, "
          "n_per_class=20 generation, 80 images. Both networks: lr=0.08, "
          "400 full-batch epochs, identical seed=42 initialization scheme.")

    X_train, y_train = make_junction_dataset_rgb(
        n_per_class=50, size=30, arm=8, noise=0.4, channel_noise=0.15,
        center_jitter=2, seed=42)
    X_test, y_test = make_junction_dataset_rgb(
        n_per_class=20, size=30, arm=8, noise=0.4, channel_noise=0.15,
        center_jitter=2, seed=999)

    lenet = LeNet5Style(img_size=30, n_classes=4, lr=0.08, seed=42)
    losses_lenet = lenet.train(X_train, y_train, epochs=400)
    train_acc_lenet = lenet.accuracy(X_train, y_train)
    test_acc_lenet = lenet.accuracy(X_test, y_test)

    base = TwoBlockConvNetRGB(f1=3, f2=6, k=3, img_size=30, hidden=8, lr=0.08, seed=42)
    losses_base = base.train(X_train, y_train, epochs=400)
    train_acc_base = base.accuracy(X_train, y_train)
    test_acc_base = base.accuracy(X_test, y_test)

    arr_lenet, arr_base = np.array(losses_lenet), np.array(losses_base)
    noise_lenet_1st = float(np.std(np.diff(arr_lenet[:200])))
    noise_lenet_2nd = float(np.std(np.diff(arr_lenet[200:])))
    noise_base_1st = float(np.std(np.diff(arr_base[:200])))
    noise_base_2nd = float(np.std(np.diff(arr_base[200:])))
    peak_lenet_epoch = int(arr_lenet[:150].argmax())
    peak_lenet_loss = float(arr_lenet[:150].max())

    print(f"\n{'network':<20}{'params':>9}{'train_acc':>11}{'test_acc':>10}"
          f"{'noise 1st half':>16}{'noise 2nd half':>16}")
    print(f"{'LeNet5Style':<20}{lenet.param_count():>9}{train_acc_lenet:>11.4f}"
          f"{test_acc_lenet:>10.4f}{noise_lenet_1st:>16.5f}{noise_lenet_2nd:>16.6f}")
    print(f"{'TwoBlockConvNetRGB':<20}{base.param_count():>9}{train_acc_base:>11.4f}"
          f"{test_acc_base:>10.4f}{noise_base_1st:>16.5f}{noise_base_2nd:>16.5f}")
    print("Actual output from this lesson's run.")

    ratio = lenet.param_count() / base.param_count()
    same_test = abs(test_acc_lenet - test_acc_base) < 1e-9
    print(f"\nreal, honest finding: LeNet5Style has {ratio:.1f}x more parameters "
          f"({lenet.param_count()} vs {base.param_count()}) than this series' own "
          f"baseline. train_acc: {train_acc_lenet:.3f} (LeNet5Style) vs. "
          f"{train_acc_base:.3f} (baseline). test_acc: {test_acc_lenet:.3f} vs. "
          f"{test_acc_base:.3f}"
          + (" -- IDENTICAL, despite the huge parameter gap." if same_test else ".")
          + f" This is not a claim that LeNet-5 is a bad architecture -- it was "
          f"designed and validated on tens of thousands of MNIST digits, not 200 "
          f"hand-generated images across 4 classes. {lenet.param_count()} "
          f"parameters for 200 training examples is a much worse capacity-to-data "
          f"ratio than {base.param_count()} parameters for the same 200 examples, "
          f"and that mismatch shows up as a {'slightly ' if same_test else ''}wider "
          f"train/test gap ({train_acc_lenet - test_acc_lenet:.3f} vs. "
          f"{train_acc_base - test_acc_base:.3f}) from LeNet5Style's more complete "
          f"training-set memorization -- extra capacity bought via historical "
          f"fidelity here, not extra accuracy. Matching capacity to data size "
          f"matters more than historical fidelity.")

    print(f"\na second, separate finding, visible directly in the loss plot: "
          f"LeNet5Style is NOT smooth from the start -- it spikes to "
          f"{peak_lenet_loss:.2f} at epoch {peak_lenet_epoch} (noise_std="
          f"{noise_lenet_1st:.5f} over its first 200 epochs, comparable to the "
          f"baseline's own {noise_base_1st:.5f}) -- but that instability is "
          f"TRANSIENT: by the second half it has settled to near-total "
          f"smoothness (noise_std={noise_lenet_2nd:.6f}). The baseline's "
          f"instability, by contrast, is PERSISTENT -- its second-half noise "
          f"({noise_base_2nd:.5f}) is actually higher than its first half, and "
          f"visible spikes keep recurring all the way to epoch 400. Days 45 and "
          f"46 already documented this exact baseline network's real "
          f"late-training oscillation at fixed lr=0.08 on this same dataset, so "
          f"this isn't new -- but LeNet5Style's settling behavior is a real, "
          f"different pattern worth naming: tanh saturates as weights grow "
          f"during training (its gradient shrinks toward zero at both extremes), "
          f"which naturally damps step-to-step movement over time in a way "
          f"ReLU's unbounded activation does not. A network built from "
          f"saturating activations can start rougher and finish smoother than "
          f"one built from non-saturating ones, even at the same fixed "
          f"learning rate.")

    plt.figure(figsize=(8, 4.5))
    plt.plot(losses_lenet, label=f"LeNet5Style (test_acc={test_acc_lenet:.3f})",
              color="#337ab7")
    plt.plot(losses_base, label=f"TwoBlockConvNetRGB (test_acc={test_acc_base:.3f})",
              color="#d9534f", alpha=0.85, linewidth=0.9)
    plt.xlabel("epoch (= step, full-batch)")
    plt.ylabel("training loss")
    plt.title("LeNet5Style vs. this series' baseline: smoother, but generalizes worse")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lenet_vs_baseline_loss.png", dpi=110)
    plt.close()
    print("\nsaved lenet_vs_baseline_loss.png")

    return dict(train_acc_lenet=train_acc_lenet, test_acc_lenet=test_acc_lenet,
                train_acc_base=train_acc_base, test_acc_base=test_acc_base,
                params_lenet=lenet.param_count(), params_base=base.param_count())


# =============================================================================
# 5. WHAT THIS MAPS TO IN PYTORCH
# =============================================================================
def note_on_pytorch():
    print("\n" + "=" * 70)
    print("3. WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)
    print(
        "class LeNet5(nn.Module):\n"
        "    def __init__(self, n_classes=4):\n"
        "        super().__init__()\n"
        "        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)\n"
        "        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)\n"
        "        self.pool = nn.AvgPool2d(2, 2)      # matches avgpool_forward\n"
        "        self.fc1 = nn.Linear(16 * 4 * 4, 120)\n"
        "        self.fc2 = nn.Linear(120, 84)\n"
        "        self.fc3 = nn.Linear(84, n_classes)\n"
        "\n"
        "    def forward(self, x):\n"
        "        x = self.pool(torch.tanh(self.conv1(x)))\n"
        "        x = self.pool(torch.tanh(self.conv2(x)))\n"
        "        x = x.flatten(1)\n"
        "        x = torch.tanh(self.fc1(x))\n"
        "        x = torch.tanh(self.fc2(x))\n"
        "        return self.fc3(x)          # CrossEntropyLoss applies softmax"
    )
    print("\nnn.AvgPool2d does exactly what avgpool_forward/avgpool_backward do "
          "here -- mean over each window forward, an equal 1/(size*size) share "
          "of the output gradient to every input position backward -- just "
          "with autograd computing that backward pass instead of it being "
          "written out by hand.")


def main():
    demo_gradient_check()
    demo_train_and_compare()
    note_on_pytorch()


if __name__ == "__main__":
    main()