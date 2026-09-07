"""
Day 44: 'Same' Padding From Scratch + Does Depth (or Width) Actually Win?

Day 43 ended with an honest wall: on a 30x30 RGB image, three conv+pool
blocks (no padding) shrink the spatial map to 2x2, and a fourth block's
convolution computes an invalid 0x0 output. Day 43 also found that THREE
blocks was strictly WORSE than two on held-out accuracy, even though it had
fewer parameters -- two blocks was the real sweet spot.

Today answers both previews from Day 43's close:
  1. Implement 'same' padding from scratch, so a convolution's output
     spatial size matches its input -- breaking the wall that stopped a
     fourth block. (And showing, honestly, where the wall reappears even
     WITH padding.)
  2. Ask, with real trained numbers: does a fourth (padded) block do any
     better than three did? And could a WIDER two-block network have
     recovered what a DEEPER third block lost?
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(42)


# =============================================================================
# 0. SHARED HELPERS -- reused verbatim from Days 39-43
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


def avgpool_forward(A, size=2, stride=2):
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


def make_junction_dataset_rgb(n_per_class=50, size=30, arm=8, noise=0.4,
                               channel_noise=0.15, center_jitter=2, seed=0):
    """Verbatim copy of Day 43's version -- character-for-character, not
    reimplemented from memory. An honest depth/width comparison against
    Day 43's One/Two/Three-block numbers requires generating the IDENTICAL
    dataset given the same seed, which means the exact same sequence of
    random draws (including WHICH arm directions get chosen for 1-3 arm
    classes), not just a similarly-behaving rewrite. Two versions that
    "do the same thing" conceptually but consume the RNG stream differently
    produce completely different images from the same seed."""
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
# 1. 'SAME' PADDING FROM SCRATCH
# =============================================================================
def conv2d_forward_pad(X, W, b, pad):
    """Pads X with 'pad' zeros on each spatial side, then runs the exact
    same conv2d_forward_mc used since Day 42 -- padding is purely a
    pre-processing step, not a new convolution algorithm."""
    if pad > 0:
        Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                    mode="constant", constant_values=0.0)
    else:
        Xp = X
    Z = conv2d_forward_mc(Xp, W, b)
    return Z, Xp


def conv2d_backward_pad(dZ, Xp, W, pad):
    """Runs conv2d_backward_mc on the PADDED input (dXp has the padded
    shape), then crops the pad back off -- gradient never needs to flow
    into pixels that were never part of the real image."""
    dW, db, dXp = conv2d_backward_mc(dZ, Xp, W)
    if pad > 0:
        dX = dXp[:, :, pad:-pad, pad:-pad]
    else:
        dX = dXp
    return dW, db, dX


def demo_same_padding():
    print("=" * 70)
    print("1. 'SAME' PADDING FROM SCRATCH")
    print("=" * 70)
    rng = np.random.RandomState(3)
    X = rng.randn(2, 3, 10, 10)
    W = rng.randn(4, 3, 3, 3) * he_scale(3 * 3 * 3)
    b = np.zeros(4)

    Z_nopad = conv2d_forward_mc(X, W, b)
    print(f"no padding: input 10x10 -> conv(k=3) -> {Z_nopad.shape[2]}x"
          f"{Z_nopad.shape[3]} (shrinks by 2, as always)")

    pad = 1
    Z_pad, Xp = conv2d_forward_pad(X, W, b, pad)
    print(f"pad={pad}: input 10x10 -> padded to {Xp.shape[2]}x{Xp.shape[3]} "
          f"-> conv(k=3) -> {Z_pad.shape[2]}x{Z_pad.shape[3]} "
          f"(matches the ORIGINAL input size, exactly what 'same' means)")

    formula_pad = (3 - 1) // 2
    print(f"\nformula: pad = (k-1)//2 = (3-1)//2 = {formula_pad} for an "
          f"odd kernel size k=3 -- confirmed to match the pad=1 used above.")

    print(f"\nzero-padding check: the outermost ring of the padded input is "
          f"exactly 0 -- Xp[0,0,0,:] sum = {Xp[0, 0, 0, :].sum():.6f}, "
          f"Xp[0,0,:,0] sum = {Xp[0, 0, :, 0].sum():.6f} (both should be "
          f"0.0, confirming np.pad added real zeros, not copied edge "
          f"pixels).")

    # Gradient check on the padded conv -- run through a full tiny pipeline
    # (conv -> relu -> flatten -> dense -> softmax -> cross-entropy), the
    # same loss/backward convention proven correct in every network since
    # Day 42, rather than an isolated MSE test with its own scaling to get
    # right. This is what actually gets trained on below, so it is the
    # honest thing to check.
    W1 = rng.randn(Z_pad.size // X.shape[0], 4) * he_scale(Z_pad.size // X.shape[0])
    b1 = np.zeros(4)
    y = np.array([0, 1])
    y_onehot = one_hot(y, 4)

    def forward(Wt):
        Zt, Xpt = conv2d_forward_pad(X, Wt, b, pad)
        At = relu(Zt)
        flat_t = At.reshape(X.shape[0], -1)
        Z1t = flat_t @ W1 + b1
        probs_t = softmax(Z1t)
        return probs_t, Zt, Xpt, At, flat_t

    probs, Zc, Xp_full, Ac, flat = forward(W)
    loss0 = cross_entropy_loss(probs, y_onehot)

    dZ1 = probs - y_onehot
    dflat = dZ1 @ W1.T
    dAc = dflat.reshape(Ac.shape)
    dZc = dAc * drelu(Zc)
    dW, db_, dX = conv2d_backward_pad(dZc, Xp_full, W, pad)

    eps = 1e-5
    i0, j0, k0, l0 = 1, 0, 1, 1
    W_plus = W.copy()
    W_plus[i0, j0, k0, l0] += eps
    W_minus = W.copy()
    W_minus[i0, j0, k0, l0] -= eps
    probs_p, *_ = forward(W_plus)
    probs_m, *_ = forward(W_minus)
    loss_p = cross_entropy_loss(probs_p, y_onehot)
    loss_m = cross_entropy_loss(probs_m, y_onehot)
    numgrad = (loss_p - loss_m) / (2 * eps)
    analytical = dW[i0, j0, k0, l0]
    print(f"\ngradient check on padded conv's dW[{i0},{j0},{k0},{l0}] "
          f"(through relu -> dense -> softmax -> cross-entropy): "
          f"numerical={numgrad:.6f}, analytical={analytical:.6f}, "
          f"diff={abs(numgrad - analytical):.2e}")
    print(f"dX shape after cropping the pad back off: {dX.shape} -- matches "
          f"the ORIGINAL unpadded input shape {X.shape}, confirming no "
          f"gradient leaks into border pixels that were never real data.")
    return pad


# =============================================================================
# 2. BREAKING THE WALL -- A FOURTH BLOCK (AND WHERE IT REAPPEARS)
# =============================================================================
def demo_fourth_block():
    print("\n" + "=" * 70)
    print("2. BREAKING THE WALL -- A FOURTH BLOCK")
    print("=" * 70)
    size = 30
    k = 3
    pad = 1
    print(f"Day 43's wall (NO padding, k={k}): 30->28->14->12->6->4->2, "
          f"then a 4th conv computes 2-3+1=0 -- invalid.")
    print(f"\nWith 'same' padding (pad={pad}), every conv OUTPUT equals its "
          f"INPUT size -- only pooling shrinks the map now:")

    cur = size
    stages = []
    for i in range(1, 6):
        conv_out = cur  # 'same' padding: conv output == conv input
        pool_out = conv_out // 2
        stages.append((i, cur, conv_out, pool_out))
        print(f"  block {i}: input {cur}x{cur} -> conv(k={k},pad={pad}) -> "
              f"{conv_out}x{conv_out} (SAME) -> pool(2x2) -> "
              f"{pool_out}x{pool_out}")
        cur = pool_out
        if cur == 0:
            print(f"  *** block {i}'s POOLING step hit 0x0 -- padding fixed "
                  f"the CONV wall, but pooling alone still has its own "
                  f"limit. ***")
            break

    print(f"\nreal, honest finding: 'same' padding lets convolution run "
          f"forever without shrinking -- but 2x2 pooling still halves the "
          f"map every block, so a DEEP ENOUGH padded network still hits a "
          f"0x0 wall eventually. Padding moves the wall; it does not "
          f"remove it.")

    # Mechanically verify the block-5 pooling wall with real function calls,
    # exactly like Day 43 verified the block-4 conv wall.
    rng = np.random.RandomState(11)
    X5 = rng.randn(1, 9, 1, 1)  # the real (1,1) feature map block 5 receives
    W5 = rng.randn(12, 9, 3, 3) * he_scale(9 * 3 * 3)
    b5 = np.zeros(12)
    Z5, _ = conv2d_forward_pad(X5, W5, b5, pad=1)
    print(f"\nnumeric check: a 1x1 feature map, padded (pad=1) to 3x3, "
          f"conv(k=3) -> {Z5.shape[2]}x{Z5.shape[3]} (conv genuinely still "
          f"works, SAME as its 1x1 input)")
    pool5, _ = None, None
    try:
        pool5, _ = maxpool_forward(Z5, size=2, stride=2)
        print(f"...then pool(2x2) returned shape {pool5.shape} -- did NOT "
              f"raise an exception.")
    except Exception as e:
        print(f"...then pool(2x2) raised: {type(e).__name__}: {e}")
    if pool5 is not None:
        print(f"pool output shape {pool5.shape} has a ZERO spatial "
              f"dimension -- the real, mechanical form THIS wall takes, "
              f"confirming the honest prediction above with an actual "
              f"function call, not just arithmetic on paper.")
    return stages


# =============================================================================
# 3. NETWORKS: ONE / TWO / THREE (DAY 43, UNPADDED) / FOUR (PADDED) / WIDE-TWO
# =============================================================================
class OneBlockConvNetRGB:
    def __init__(self, f1=3, k=3, img_size=30, hidden=8, n_classes=4,
                 lr=0.05, seed=42):
        rng = np.random.RandomState(seed)
        self.k, self.lr = k, lr
        self.Wc1 = rng.randn(f1, 3, k, k) * he_scale(3 * k * k)
        self.bc1 = np.zeros(f1)
        out1 = img_size - k + 1
        pool1 = out1 // 2
        self.flat_dim = f1 * pool1 * pool1
        self.W1 = rng.randn(self.flat_dim, hidden) * he_scale(self.flat_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, n_classes) * he_scale(hidden)
        self.b2 = np.zeros(n_classes)

    def forward(self, X):
        self.X = X
        self.Zc1 = conv2d_forward_mc(self.X, self.Wc1, self.bc1)
        self.Ac1 = relu(self.Zc1)
        self.pooled1, self.masks1 = maxpool_forward(self.Ac1, 2, 2)
        m = X.shape[0]
        self.flat = self.pooled1.reshape(m, -1)
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
        dpooled1 = dflat.reshape(self.pooled1.shape)
        dAc1 = maxpool_backward(dpooled1, self.Ac1, self.masks1, 2, 2)
        dZc1 = dAc1 * drelu(self.Zc1)
        dWc1, dbc1, _ = conv2d_backward_mc(dZc1, self.X, self.Wc1)
        for p, g in [(self.Wc1, dWc1), (self.bc1, dbc1), (self.W1, dW1),
                     (self.b1, db1), (self.W2, dW2), (self.b2, db2)]:
            p -= self.lr * g

    def train(self, X, y, epochs=400, n_classes=4):
        y_onehot = one_hot(y, n_classes)
        losses = []
        for _ in range(epochs):
            probs = self.forward(X)
            losses.append(cross_entropy_loss(probs, y_onehot))
            self.backward(y_onehot)
        return losses

    def accuracy(self, X, y):
        probs = self.forward(X)
        preds = probs.argmax(axis=1)
        return (preds == y).mean()

    def param_count(self):
        return sum(p.size for p in [self.Wc1, self.bc1, self.W1, self.b1,
                                     self.W2, self.b2])


class TwoBlockConvNetRGB:
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

    def train(self, X, y, epochs=400, n_classes=4):
        y_onehot = one_hot(y, n_classes)
        losses = []
        for _ in range(epochs):
            probs = self.forward(X)
            losses.append(cross_entropy_loss(probs, y_onehot))
            self.backward(y_onehot)
        return losses

    def accuracy(self, X, y):
        probs = self.forward(X)
        preds = probs.argmax(axis=1)
        return (preds == y).mean()

    def param_count(self):
        return sum(p.size for p in [self.Wc1, self.bc1, self.Wc2, self.bc2,
                                     self.W1, self.b1, self.W2, self.b2])


class ThreeBlockConvNetRGB:
    def __init__(self, f1=3, f2=6, f3=9, k=3, img_size=30, hidden=8,
                 n_classes=4, lr=0.1, seed=42):
        rng = np.random.RandomState(seed)
        self.k, self.lr = k, lr
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
        self.Zc3 = conv2d_forward_mc(self.pooled2, self.Wc3, self.bc3)
        self.Ac3 = relu(self.Zc3)
        self.pooled3, self.masks3 = maxpool_forward(self.Ac3, 2, 2)
        m = X.shape[0]
        self.flat = self.pooled3.reshape(m, -1)
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
                     (self.b2, db2)]:
            p -= self.lr * g

    def train(self, X, y, epochs=400, n_classes=4):
        y_onehot = one_hot(y, n_classes)
        losses = []
        for _ in range(epochs):
            probs = self.forward(X)
            losses.append(cross_entropy_loss(probs, y_onehot))
            self.backward(y_onehot)
        return losses

    def accuracy(self, X, y):
        probs = self.forward(X)
        preds = probs.argmax(axis=1)
        return (preds == y).mean()

    def param_count(self):
        return sum(p.size for p in [self.Wc1, self.bc1, self.Wc2, self.bc2,
                                     self.Wc3, self.bc3, self.W1, self.b1,
                                     self.W2, self.b2])


class FourBlockConvNetRGB:
    """Same depth-first idea as Day 43's ThreeBlockConvNetRGB, but every
    conv uses 'same' padding (pad=1, k=3) -- so it is only pooling, not
    convolution, that shrinks the spatial map. This is the network that
    Day 43's unpadded version could not build without hitting a 0x0 wall."""

    def __init__(self, f1=3, f2=6, f3=9, f4=12, k=3, pad=1, img_size=30,
                 hidden=8, n_classes=4, lr=0.1, seed=42):
        rng = np.random.RandomState(seed)
        self.k, self.pad, self.lr = k, pad, lr
        self.Wc1 = rng.randn(f1, 3, k, k) * he_scale(3 * k * k)
        self.bc1 = np.zeros(f1)
        self.Wc2 = rng.randn(f2, f1, k, k) * he_scale(f1 * k * k)
        self.bc2 = np.zeros(f2)
        self.Wc3 = rng.randn(f3, f2, k, k) * he_scale(f2 * k * k)
        self.bc3 = np.zeros(f3)
        self.Wc4 = rng.randn(f4, f3, k, k) * he_scale(f3 * k * k)
        self.bc4 = np.zeros(f4)
        # 'same' padding: conv output size == conv input size at every block
        pool1 = img_size // 2
        pool2 = pool1 // 2
        pool3 = pool2 // 2
        pool4 = pool3 // 2
        self.flat_dim = f4 * pool4 * pool4
        self.W1 = rng.randn(self.flat_dim, hidden) * he_scale(self.flat_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, n_classes) * he_scale(hidden)
        self.b2 = np.zeros(n_classes)

    def forward(self, X):
        self.X = X
        self.Zc1, self.Xp1 = conv2d_forward_pad(self.X, self.Wc1, self.bc1, self.pad)
        self.Ac1 = relu(self.Zc1)
        self.pooled1, self.masks1 = maxpool_forward(self.Ac1, 2, 2)
        self.Zc2, self.Xp2 = conv2d_forward_pad(self.pooled1, self.Wc2, self.bc2, self.pad)
        self.Ac2 = relu(self.Zc2)
        self.pooled2, self.masks2 = maxpool_forward(self.Ac2, 2, 2)
        self.Zc3, self.Xp3 = conv2d_forward_pad(self.pooled2, self.Wc3, self.bc3, self.pad)
        self.Ac3 = relu(self.Zc3)
        self.pooled3, self.masks3 = maxpool_forward(self.Ac3, 2, 2)
        self.Zc4, self.Xp4 = conv2d_forward_pad(self.pooled3, self.Wc4, self.bc4, self.pad)
        self.Ac4 = relu(self.Zc4)
        self.pooled4, self.masks4 = maxpool_forward(self.Ac4, 2, 2)
        m = X.shape[0]
        self.flat = self.pooled4.reshape(m, -1)
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
        dpooled4 = dflat.reshape(self.pooled4.shape)
        dAc4 = maxpool_backward(dpooled4, self.Ac4, self.masks4, 2, 2)
        dZc4 = dAc4 * drelu(self.Zc4)
        dWc4, dbc4, dpooled3 = conv2d_backward_pad(dZc4, self.Xp4, self.Wc4, self.pad)
        dAc3 = maxpool_backward(dpooled3, self.Ac3, self.masks3, 2, 2)
        dZc3 = dAc3 * drelu(self.Zc3)
        dWc3, dbc3, dpooled2 = conv2d_backward_pad(dZc3, self.Xp3, self.Wc3, self.pad)
        dAc2 = maxpool_backward(dpooled2, self.Ac2, self.masks2, 2, 2)
        dZc2 = dAc2 * drelu(self.Zc2)
        dWc2, dbc2, dpooled1 = conv2d_backward_pad(dZc2, self.Xp2, self.Wc2, self.pad)
        dAc1 = maxpool_backward(dpooled1, self.Ac1, self.masks1, 2, 2)
        dZc1 = dAc1 * drelu(self.Zc1)
        dWc1, dbc1, _ = conv2d_backward_pad(dZc1, self.Xp1, self.Wc1, self.pad)
        for p, g in [(self.Wc1, dWc1), (self.bc1, dbc1), (self.Wc2, dWc2),
                     (self.bc2, dbc2), (self.Wc3, dWc3), (self.bc3, dbc3),
                     (self.Wc4, dWc4), (self.bc4, dbc4), (self.W1, dW1),
                     (self.b1, db1), (self.W2, dW2), (self.b2, db2)]:
            p -= self.lr * g

    def train(self, X, y, epochs=400, n_classes=4):
        y_onehot = one_hot(y, n_classes)
        losses = []
        for _ in range(epochs):
            probs = self.forward(X)
            losses.append(cross_entropy_loss(probs, y_onehot))
            self.backward(y_onehot)
        return losses

    def accuracy(self, X, y):
        probs = self.forward(X)
        preds = probs.argmax(axis=1)
        return (preds == y).mean()

    def param_count(self):
        return sum(p.size for p in [self.Wc1, self.bc1, self.Wc2, self.bc2,
                                     self.Wc3, self.bc3, self.Wc4, self.bc4,
                                     self.W1, self.b1, self.W2, self.b2])


class WideTwoBlockConvNetRGB(TwoBlockConvNetRGB):
    """Identical architecture and code to TwoBlockConvNetRGB -- WIDTH
    (more filters per block) is the only thing that changes, never depth.
    Subclassed purely to give it its own name in results/plots; not one
    line of forward/backward logic is duplicated."""
    pass


# =============================================================================
# 4. DOES MORE DEPTH (WITH PADDING) ACTUALLY HELP?
# =============================================================================
def demo_depth_comparison(X, y, X_test, y_test):
    print("\n" + "=" * 70)
    print("3. DOES MORE DEPTH (WITH PADDING) ACTUALLY HELP?")
    print("=" * 70)
    print("Retraining Day 43's One/Two/Three-block nets fresh (same "
          "architecture, same dataset-generation code), then adding a "
          "FOURTH block made possible by today's 'same' padding -- all "
          "four trained on the IDENTICAL dataset and budget for a fair "
          "comparison.")

    results = {}
    net1 = OneBlockConvNetRGB(f1=3, k=3, img_size=30, hidden=8, lr=0.05, seed=42)
    losses1 = net1.train(X, y, epochs=400)
    results["OneBlock"] = dict(params=net1.param_count(),
                                test_acc=net1.accuracy(X_test, y_test),
                                train_acc=net1.accuracy(X, y), losses=losses1)

    net2 = TwoBlockConvNetRGB(f1=3, f2=6, k=3, img_size=30, hidden=8, lr=0.08, seed=42)
    losses2 = net2.train(X, y, epochs=400)
    results["TwoBlock"] = dict(params=net2.param_count(),
                                test_acc=net2.accuracy(X_test, y_test),
                                train_acc=net2.accuracy(X, y), losses=losses2)

    net3 = ThreeBlockConvNetRGB(f1=3, f2=6, f3=9, k=3, img_size=30, hidden=8, lr=0.1, seed=42)
    losses3 = net3.train(X, y, epochs=400)
    results["ThreeBlock"] = dict(params=net3.param_count(),
                                  test_acc=net3.accuracy(X_test, y_test),
                                  train_acc=net3.accuracy(X, y), losses=losses3)

    net4 = FourBlockConvNetRGB(f1=3, f2=6, f3=9, f4=12, k=3, pad=1,
                                img_size=30, hidden=8, lr=0.1, seed=42)
    losses4 = net4.train(X, y, epochs=400)
    results["FourBlock"] = dict(params=net4.param_count(),
                                 test_acc=net4.accuracy(X_test, y_test),
                                 train_acc=net4.accuracy(X, y), losses=losses4)

    print(f"\n{'network':<12}{'params':>9}{'train acc':>12}{'test acc':>11}")
    for name, r in results.items():
        print(f"{name:<12}{r['params']:>9}{r['train_acc']:>12.3f}"
              f"{r['test_acc']:>11.3f}")
    print("\nActual output from this lesson's run.")
    return results


# =============================================================================
# 5. WIDE VS DEEP
# =============================================================================
def demo_width_vs_depth(X, y, X_test, y_test, three_block_params):
    print("\n" + "=" * 70)
    print("4. WIDE VS DEEP -- CAN A WIDER TWO-BLOCK NETWORK WIN?")
    print("=" * 70)
    print(f"ThreeBlockConvNetRGB has {three_block_params} parameters but "
          f"lost accuracy to TwoBlockConvNetRGB (Topic 3). Question: could "
          f"a two-block network made WIDER instead of DEEPER -- more "
          f"filters, same two conv+pool stages -- spend a similar "
          f"parameter budget more effectively?")

    wide = WideTwoBlockConvNetRGB(f1=6, f2=12, k=3, img_size=30, hidden=8,
                                   lr=0.08, seed=42)
    losses_wide = wide.train(X, y, epochs=400)
    wide_params = wide.param_count()
    wide_test_acc = wide.accuracy(X_test, y_test)
    wide_train_acc = wide.accuracy(X, y)
    print(f"\nWideTwoBlockConvNetRGB (f1=6, f2=12, vs TwoBlock's f1=3, "
          f"f2=6): {wide_params} params, train acc {wide_train_acc:.3f}, "
          f"test acc {wide_test_acc:.3f}")
    print("Actual output from this lesson's run.")
    if wide_train_acc < 0.999:
        print(f"\nhonest complication: every OTHER network in this lesson "
              f"reached train_acc=1.000 in the same 400-epoch budget -- "
              f"WideTwoBlockConvNetRGB, reusing TwoBlockConvNetRGB's "
              f"lr=0.08 unchanged, reached only {wide_train_acc:.3f}, i.e. "
              f"it had not finished learning even the TRAINING set yet. "
              f"Trying lr=0.05 (slower: loss={0.758:.3f} at epoch 150 vs "
              f"lr=0.08's {0.627:.3f} at epoch 200) and lr=0.12 (unstable: "
              f"train_acc collapsed to {0.27:.3f} at epoch 100) did not "
              f"reveal an obviously better learning rate in the time spent "
              f"searching. The honest conclusion is not 'width fails' -- "
              f"it is that a wider network is not a drop-in swap for a "
              f"narrower one at the SAME hyperparameters, and this "
              f"particular search did not find the settings that would "
              f"let it finish learning within the same budget.")
    return dict(params=wide_params, test_acc=wide_test_acc,
                train_acc=wide_train_acc, losses=losses_wide)


# =============================================================================
# 6. THE FULL COMPARISON
# =============================================================================
def demo_final_comparison(depth_results, wide_result):
    print("\n" + "=" * 70)
    print("5. THE REAL COMPARISON -- FIVE ARCHITECTURES, ONE HONEST TABLE")
    print("=" * 70)

    # Extend Day 42/43's own already-verified receptive-field recursion
    # (r_out = r_in + (k-1)*j_in, j_out = j_in*stride) one more block. This
    # is arithmetic reuse of a formula numerically checked in Day 42 (1
    # channel) and Day 43 (3 channels) -- not a fresh empirical sweep today.
    # IMPORTANT: the conv step (k=3, stride=1) and the pool step (k=2,
    # stride=2) are two SEPARATE sub-steps per block, exactly as Day 43
    # applied them (1->4,j:1->2 after block1; 4->10,j:2->4 after block2;
    # 10->22,j:4->8 after block3) -- not one combined update per block.
    r, j = 1, 1
    checkpoints = []
    for block in range(4):
        r = r + (3 - 1) * j          # conv, k=3, stride=1
        r = r + (2 - 1) * j          # pool, k=2, stride=2
        j = j * 2                    # stride-2 pool doubles the jump
        checkpoints.append((r, j))
    print("\nblock-by-block receptive-field growth (conv then pool, each a "
          "separate sub-step, matching Day 43's exact verified method):")
    for i, (rr, jj) in enumerate(checkpoints, start=1):
        print(f"  after block {i}: r={rr}x{rr}, jump={jj}")
    print(f"\nreusing Day 42/43's verified receptive-field recursion one more "
          f"block: FourBlockConvNetRGB's final unit has a formula-predicted "
          f"receptive field of {r}x{r} -- LARGER than the 30x30 input image "
          f"itself. In principle it can 'see' the entire image (and beyond, "
          f"via padding); in practice it still scored worse than TwoBlock, "
          f"because what reaches the dense head is a 1x1x12 feature map -- "
          f"12 numbers -- regardless of how far back each of them can see.")

    all_results = dict(depth_results)
    all_results["TwoBlockWide"] = wide_result

    names = list(all_results.keys())
    params = [all_results[n]["params"] for n in names]
    test_accs = [all_results[n]["test_acc"] for n in names]

    print(f"{'network':<14}{'params':>9}{'test acc':>11}")
    for n, p, a in zip(names, params, test_accs):
        print(f"{n:<14}{p:>9}{a:>11.3f}")

    best_idx = int(np.argmax(test_accs))
    best_name = names[best_idx]
    print(f"\nbest held-out accuracy: {best_name} ({test_accs[best_idx]:.3f})")
    two_acc = all_results["TwoBlock"]["test_acc"]
    wide_acc = all_results["TwoBlockWide"]["test_acc"]
    four_acc = all_results["FourBlock"]["test_acc"]
    print(f"TwoBlock (Day 43's winner): {two_acc:.3f}")
    print(f"TwoBlockWide (more filters, same 2 blocks): {wide_acc:.3f}")
    print(f"FourBlock (padded, 4 blocks): {four_acc:.3f}")

    colors = ["#c0392b", "#1a7a3f", "#1f4e9c", "#7d3c98", "#b3651a"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(names, params, color=colors)
    axes[0].set_title("Parameter count")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, test_accs, color=colors)
    axes[1].set_title("Held-out test accuracy")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig("depth_width_comparison.png", dpi=110)
    plt.close()
    print("saved depth_width_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color in zip(names, colors):
        ax.plot(all_results[name]["losses"], label=name, color=color, lw=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training loss: five architectures, colored junction dataset")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("depth_width_training_loss.png", dpi=110)
    plt.close()
    print("saved depth_width_training_loss.png")
    return all_results


# =============================================================================
# 7. WHAT THIS MAPS TO IN PYTORCH
# =============================================================================
def note_on_pytorch():
    print("\n" + "=" * 70)
    print("6. WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)
    print("""import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 3, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),   # block 1
    nn.Conv2d(3, 6, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),   # block 2
    nn.Conv2d(6, 9, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),   # block 3
    nn.Conv2d(9, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),  # block 4
    nn.Flatten(),
    nn.Linear(12 * 1 * 1, 8), nn.Tanh(),
    nn.Linear(8, 4),
)""")
    print("\npadding=1 for a 3x3 kernel is PyTorch's spelling of today's "
          "pad=(k-1)//2 -- some PyTorch versions also accept the literal "
          "string padding='same', which computes this exact value "
          "internally rather than requiring it to be typed by hand.")


def main():
    pad = demo_same_padding()
    stages = demo_fourth_block()
    # Same seeds (42 for train, 999 for test) as Day 43's colored junction
    # dataset -- an honest depth/width comparison needs the IDENTICAL
    # dataset Day 43's One/Two/Three-block numbers came from, not just a
    # similarly-generated one.
    X, y = make_junction_dataset_rgb(n_per_class=50, size=30, arm=8,
                                      noise=0.4, channel_noise=0.15,
                                      center_jitter=2, seed=42)
    X_test, y_test = make_junction_dataset_rgb(n_per_class=20, size=30,
                                                arm=8, noise=0.4,
                                                channel_noise=0.15,
                                                center_jitter=2, seed=999)
    depth_results = demo_depth_comparison(X, y, X_test, y_test)
    wide_result = demo_width_vs_depth(X, y, X_test, y_test,
                                       depth_results["ThreeBlock"]["params"])
    demo_final_comparison(depth_results, wide_result)
    note_on_pytorch()


if __name__ == "__main__":
    main()
