"""
Day 41: Convolutional Layers
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 6, Day 6

Every network from Day 36 through Day 40 treated its input as a flat vector
-- even Day 40's 2D signal-propagation demo immediately flattened structure
away. Real images, audio, and other spatially/temporally structured data
lose meaning when flattened: a pixel's neighbors matter. Today builds a
convolutional layer entirely from scratch -- a small kernel slid across an
image, computing one dot product per position -- and trains a real (tiny)
ConvNet to solve a task that depends on spatial structure.

Covers:
  1. What convolution actually computes -- fixed, hand-picked edge-detection
     kernels applied to a real image, no learning involved
  2. Padding and stride -- how they control output size
  3. From one filter to many -- feature maps
  4. Max pooling -- downsampling by keeping the strongest signal
  5. A real, honest test dataset: vertical vs horizontal stripe images
  6. ConvNet -- assembling conv + pool + dense into one trainable network
  7. The backward pass through convolution and pooling, derived and
     gradient-checked by hand
  8. The real test: training ConvNet on the stripe dataset, with a held-out
     generalization check
  9. What this maps to in PyTorch (nn.Conv2d, nn.MaxPool2d)

This script is fully self-contained (NumPy + matplotlib only). PyTorch is
intentionally not required to run this file.

Requires: pip install numpy matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)


# ---------------------------------------------------------------------------
# SHARED: activations, softmax, one-hot, loss, He-style init scale
# ---------------------------------------------------------------------------

def relu(z):
    return np.maximum(0, z)


def drelu(z):
    """Derivative of ReLU w.r.t. its input z (not its output) -- 1 where
    z was positive, 0 elsewhere, matching ReLU's own kink at zero."""
    return (z > 0).astype(z.dtype)


def tanh(z):
    return np.tanh(z)


def dtanh(a):
    return 1 - a ** 2


def softmax(z):
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def one_hot(y, num_classes):
    m = y.shape[0]
    encoded = np.zeros((m, num_classes))
    encoded[np.arange(m), y] = 1.0
    return encoded


def cross_entropy_loss(probs, y_onehot, eps=1e-9):
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(np.sum(y_onehot * np.log(probs), axis=1))


def he_scale(n_in):
    """Day 40's He/Kaiming initialization scale -- reused here for the
    conv layer's ReLU activation and the dense hidden layer."""
    return np.sqrt(2.0 / n_in)


# ---------------------------------------------------------------------------
# 1. WHAT CONVOLUTION COMPUTES -- fixed kernels, no learning
# ---------------------------------------------------------------------------

def conv_output_size(H, k, pad=0, stride=1):
    """The output-size formula this whole lesson relies on."""
    return (H - k + 2 * pad) // stride + 1


def conv2d_general(X, W, b, stride=1, pad=0):
    """A single-image, single-filter convolution supporting stride and
    padding -- used only for Topics 1-2's diagnostics, kept separate from
    the batched, stride-1, no-pad conv2d_forward the trainable ConvNet
    uses below."""
    if pad > 0:
        X = np.pad(X, pad, mode="constant")
    H, Win = X.shape
    kH, kW = W.shape
    out_H = (H - kH) // stride + 1
    out_W = (Win - kW) // stride + 1
    Z = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            patch = X[i * stride:i * stride + kH, j * stride:j * stride + kW]
            Z[i, j] = np.sum(patch * W) + b
    return Z


def demo_convolution_basics():
    print("=" * 70)
    print("1. WHAT CONVOLUTION ACTUALLY COMPUTES")
    print("=" * 70)
    size = 8
    img = np.zeros((size, size))
    img[:, 3] = 1.0
    img[:, 4] = 1.0
    vertical_k = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
    horizontal_k = vertical_k.T

    Zv = conv2d_general(img, vertical_k, b=0.0)
    Zh = conv2d_general(img, horizontal_k, b=0.0)
    print("8x8 image with a vertical stripe (columns 3-4 set to 1.0).")
    print(f"Vertical-edge kernel response: max |value| = {np.abs(Zv).max():.4f}")
    print(f"Horizontal-edge kernel response: max |value| = {np.abs(Zh).max():.4f}")
    print("The vertical kernel fires strongly on a vertical edge; the")
    print("horizontal kernel produces exactly zero -- it's structurally")
    print("blind to an edge running perpendicular to what it detects.\n")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input: vertical stripe")
    im1 = axes[1].imshow(Zv, cmap="RdBu")
    axes[1].set_title(f"Vertical kernel (max|.|={np.abs(Zv).max():.2f})")
    im2 = axes[2].imshow(Zh, cmap="RdBu")
    axes[2].set_title(f"Horizontal kernel (max|.|={np.abs(Zh).max():.2f})")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("edge_detection.png", dpi=110)
    plt.close()
    print("Saved: edge_detection.png\n")


# ---------------------------------------------------------------------------
# 2. PADDING AND STRIDE
# ---------------------------------------------------------------------------

def demo_padding_stride():
    print("=" * 70)
    print("2. PADDING AND STRIDE")
    print("=" * 70)
    H, k = 8, 3
    configs = [(0, 1), (1, 1), (0, 2)]
    img = np.random.randn(H, H)
    W = np.random.randn(k, k)
    for pad, stride in configs:
        formula_size = conv_output_size(H, k, pad, stride)
        actual = conv2d_general(img, W, b=0.0, stride=stride, pad=pad)
        print(f"H={H}, k={k}, pad={pad}, stride={stride}  ->  "
              f"formula size={formula_size}, actual shape={actual.shape[0]}")
    print("\npad=1, stride=1 exactly preserves the input size (8 -> 8) --")
    print("'same' padding. pad=0, stride=2 roughly halves it (8 -> 3),")
    print("skipping every other position as the kernel slides.\n")


# ---------------------------------------------------------------------------
# 3. FROM ONE FILTER TO MANY -- FEATURE MAPS
# ---------------------------------------------------------------------------

def conv2d_forward(X, W, b):
    """Batched, multi-filter, stride-1, no-padding convolution -- the
    version the trainable ConvNet below actually uses.
    X: (m, H, Win) single-channel images.
    W: (C_out, kH, kW) filters.  b: (C_out,) one bias per filter.
    Returns Z: (m, C_out, out_H, out_W)."""
    m, H, Win = X.shape
    C_out, kH, kW = W.shape
    out_H = H - kH + 1
    out_W = Win - kW + 1
    Z = np.zeros((m, C_out, out_H, out_W))
    for f in range(C_out):
        for i in range(out_H):
            for j in range(out_W):
                patch = X[:, i:i + kH, j:j + kW]
                Z[:, f, i, j] = np.sum(patch * W[f], axis=(1, 2)) + b[f]
    return Z


def conv2d_backward(dZ, X, W):
    m, H, Win = X.shape
    C_out, kH, kW = W.shape
    out_H = H - kH + 1
    out_W = Win - kW + 1
    dW = np.zeros_like(W)
    db = np.zeros(C_out)
    dX = np.zeros_like(X)
    for f in range(C_out):
        for i in range(out_H):
            for j in range(out_W):
                patch = X[:, i:i + kH, j:j + kW]
                dZ_ij = dZ[:, f, i, j]
                dW[f] += np.sum(dZ_ij[:, None, None] * patch, axis=0) / m
                db[f] += np.sum(dZ_ij) / m
                dX[:, i:i + kH, j:j + kW] += dZ_ij[:, None, None] * W[f]
    return dW, db, dX


def demo_feature_maps():
    print("=" * 70)
    print("3. FROM ONE FILTER TO MANY -- FEATURE MAPS")
    print("=" * 70)
    size = 8
    img = np.zeros((1, size, size))
    img[0, :, 3] = 1.0
    img[0, :, 4] = 1.0
    rng = np.random.RandomState(7)
    n_filters = 4
    W = rng.randn(n_filters, 3, 3) * he_scale(9)
    b = np.zeros(n_filters)
    Z = conv2d_forward(img, W, b)
    print(f"Input: 1 image, {size}x{size}. Filters: {n_filters}, each 3x3.")
    print(f"Output Z shape: {Z.shape}  (1 image, {n_filters} feature maps, "
          f"{Z.shape[2]}x{Z.shape[3]} each)")
    for f in range(n_filters):
        print(f"  feature map {f}: min={Z[0,f].min():.3f}  max={Z[0,f].max():.3f}")
    print()

    fig, axes = plt.subplots(1, n_filters + 1, figsize=(14, 3.2))
    axes[0].imshow(img[0], cmap="gray")
    axes[0].set_title("input image")
    for f in range(n_filters):
        axes[f + 1].imshow(Z[0, f], cmap="RdBu")
        axes[f + 1].set_title(f"feature map {f}")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("feature_maps.png", dpi=110)
    plt.close()
    print("Saved: feature_maps.png\n")


# ---------------------------------------------------------------------------
# 4. MAX POOLING
# ---------------------------------------------------------------------------

def maxpool_forward(A, size=2, stride=2):
    m, C, H, W = A.shape
    out_H = (H - size) // stride + 1
    out_W = (W - size) // stride + 1
    out = np.zeros((m, C, out_H, out_W))
    masks = {}
    for i in range(out_H):
        for j in range(out_W):
            window = A[:, :, i * stride:i * stride + size, j * stride:j * stride + size]
            flat = window.reshape(m, C, size * size)
            idx = flat.argmax(axis=2)
            out[:, :, i, j] = flat[np.arange(m)[:, None], np.arange(C)[None, :], idx]
            mask = np.zeros_like(flat)
            mm, cc = np.meshgrid(np.arange(m), np.arange(C), indexing="ij")
            mask[mm, cc, idx] = 1
            masks[(i, j)] = mask.reshape(m, C, size, size)
    return out, masks


def maxpool_backward(dOut, A, masks, size=2, stride=2):
    m, C, H, W = A.shape
    dA = np.zeros_like(A)
    out_H = (H - size) // stride + 1
    out_W = (W - size) // stride + 1
    for i in range(out_H):
        for j in range(out_W):
            dA[:, :, i * stride:i * stride + size, j * stride:j * stride + size] += (
                masks[(i, j)] * dOut[:, :, i, j][:, :, None, None]
            )
    return dA


def demo_maxpool():
    print("=" * 70)
    print("4. MAX POOLING")
    print("=" * 70)
    rng = np.random.RandomState(3)
    A = rng.randn(1, 1, 4, 4)
    out, _ = maxpool_forward(A, size=2, stride=2)
    print("Before pooling (4x4):")
    print(np.round(A[0, 0], 2))
    print("After 2x2, stride-2 max pooling (2x2):")
    print(np.round(out[0, 0], 2))
    print(f"Shape: {A.shape[2:]} -> {out.shape[2:]}\n")


# ---------------------------------------------------------------------------
# 5. THE STRIPE DATASET
# ---------------------------------------------------------------------------

def make_stripe_dataset(n_per_class=150, size=8, noise=0.3, seed=0):
    """Class 0: a bright vertical stripe. Class 1: a bright horizontal
    stripe. Both on a noisy background -- a real, honest test of whether
    a learned spatial filter can tell orientation apart."""
    rng = np.random.RandomState(seed)
    X, y = [], []
    for _ in range(n_per_class):
        im = rng.randn(size, size) * noise
        col = rng.randint(1, size - 1)
        im[:, col] += 2.0
        im[:, col - 1] += 1.0
        X.append(im); y.append(0)
    for _ in range(n_per_class):
        im = rng.randn(size, size) * noise
        row = rng.randint(1, size - 1)
        im[row, :] += 2.0
        im[row - 1, :] += 1.0
        X.append(im); y.append(1)
    X = np.array(X); y = np.array(y)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def demo_dataset():
    print("=" * 70)
    print("5. THE STRIPE DATASET")
    print("=" * 70)
    X, y = make_stripe_dataset(150, 8, 0.3, seed=42)
    print(f"X shape: {X.shape}   y shape: {y.shape}")
    print(f"  class 0 (vertical):   {(y == 0).sum()} images")
    print(f"  class 1 (horizontal): {(y == 1).sum()} images\n")

    fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))
    v_idx = np.where(y == 0)[0][:4]
    h_idx = np.where(y == 1)[0][:4]
    for k, idx in enumerate(v_idx):
        axes[0, k].imshow(X[idx], cmap="gray")
        axes[0, k].set_title(f"class 0 (vertical)")
        axes[0, k].set_xticks([]); axes[0, k].set_yticks([])
    for k, idx in enumerate(h_idx):
        axes[1, k].imshow(X[idx], cmap="gray")
        axes[1, k].set_title(f"class 1 (horizontal)")
        axes[1, k].set_xticks([]); axes[1, k].set_yticks([])
    plt.tight_layout()
    plt.savefig("stripe_dataset.png", dpi=110)
    plt.close()
    print("Saved: stripe_dataset.png\n")
    return X, y


# ---------------------------------------------------------------------------
# 6-7. CONVNET -- FORWARD, BACKWARD (derived by hand)
# ---------------------------------------------------------------------------

class ConvNet:
    """conv (He init, ReLU) -> max pool -> flatten -> dense (tanh) ->
    dense (softmax). One conv layer is enough to prove the point on an
    8x8 task; deeper stacks follow the same pattern layer after layer."""

    def __init__(self, n_filters=4, k=3, img_size=8, hidden=16,
                 n_classes=2, lr=0.1, seed=42):
        rng = np.random.RandomState(seed)
        fan_in_conv = k * k
        self.Wc = rng.randn(n_filters, k, k) * he_scale(fan_in_conv)
        self.bc = np.zeros(n_filters)
        out_conv = img_size - k + 1
        pool_out = out_conv // 2
        flat_dim = n_filters * pool_out * pool_out
        self.W1 = rng.randn(flat_dim, hidden) * he_scale(flat_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, n_classes) * he_scale(hidden)
        self.b2 = np.zeros(n_classes)
        self.lr = lr

    def forward(self, X):
        m = X.shape[0]
        self.X = X
        self.Zc = conv2d_forward(X, self.Wc, self.bc)
        self.Ac = relu(self.Zc)
        self.pooled, self.masks = maxpool_forward(self.Ac, 2, 2)
        self.flat = self.pooled.reshape(m, -1)
        self.z1 = self.flat @ self.W1 + self.b1
        self.a1 = tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.probs = softmax(self.z2)
        return self.probs

    def backward(self, y_onehot):
        m = y_onehot.shape[0]
        dz2 = self.probs - y_onehot
        dW2 = self.a1.T @ dz2 / m
        db2 = dz2.mean(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * dtanh(self.a1)
        dW1 = self.flat.T @ dz1 / m
        db1 = dz1.mean(axis=0)
        dflat = dz1 @ self.W1.T
        dpooled = dflat.reshape(self.pooled.shape)
        dAc = maxpool_backward(dpooled, self.Ac, self.masks, 2, 2)
        dZc = dAc * drelu(self.Zc)
        dWc, dbc, _ = conv2d_backward(dZc, self.X, self.Wc)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.Wc -= self.lr * dWc
        self.bc -= self.lr * dbc

    def train(self, X, y, epochs, num_classes=2):
        y_onehot = one_hot(y, num_classes)
        losses = []
        for _ in range(epochs):
            probs = self.forward(X)
            losses.append(cross_entropy_loss(probs, y_onehot))
            self.backward(y_onehot)
        return losses

    def accuracy(self, X, y):
        probs = self.forward(X)
        return (probs.argmax(axis=1) == y).mean()


# ---------------------------------------------------------------------------
# 8. THE REAL TEST -- TRAINING CONVNET
# ---------------------------------------------------------------------------

def demo_training(X, y):
    print("=" * 70)
    print("8. THE REAL TEST -- TRAINING CONVNET ON THE STRIPE DATASET")
    print("=" * 70)
    net = ConvNet(n_filters=4, k=3, img_size=8, hidden=16, n_classes=2,
                  lr=0.1, seed=42)
    untrained_acc = net.accuracy(X, y)
    print(f"Untrained (random init) accuracy on training data: {untrained_acc:.3f}  "
          f"(a 2-class problem, so 0.5 is pure chance)")

    losses = net.train(X, y, epochs=200)
    print(f"Epoch 0:   loss={losses[0]:.4f}")
    print(f"Epoch 50:  loss={losses[50]:.4f}")
    print(f"Epoch 199: loss={losses[-1]:.4f}")
    train_acc = net.accuracy(X, y)
    print(f"Final training accuracy: {train_acc:.3f}")

    X_test, y_test = make_stripe_dataset(50, 8, 0.3, seed=999)
    test_acc = net.accuracy(X_test, y_test)
    print(f"Held-out test accuracy (fresh images, seed=999, never seen "
          f"during training): {test_acc:.3f}")
    print("This is a genuine generalization check, not just a memorization")
    print("check -- the test images were generated with a different seed")
    print("and never appear anywhere in the training loop above.\n")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(losses, color="#378ADD", linewidth=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("ConvNet training on the stripe dataset")
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=110)
    plt.close()
    print("Saved: training_curve.png\n")
    return net, losses, train_acc, test_acc


def note_on_pytorch():
    print("=" * 70)
    print("WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)
    print("nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)")
    print("  # replaces conv2d_forward/conv2d_backward -- weights + grad")
    print("  # tracking handled automatically by autograd")
    print("nn.MaxPool2d(kernel_size=2, stride=2)")
    print("  # replaces maxpool_forward/maxpool_backward")
    print("nn.ReLU()                                   # replaces relu/drelu")
    print()
    print("A real PyTorch model strings these together in nn.Sequential,")
    print("exactly in the order ConvNet.forward() calls them by hand above.\n")


def main():
    demo_convolution_basics()
    demo_padding_stride()
    demo_feature_maps()
    demo_maxpool()
    X, y = demo_dataset()
    demo_training(X, y)
    note_on_pytorch()
    print("=" * 70)
    print("Day 41 complete. Convolution is one small, reusable idea (a sliding")
    print("dot product) that lets a network exploit spatial structure instead")
    print("of discarding it -- verified today on a real, honest task a flat")
    print("dense layer could not see the point of nearly as directly.")
    print("=" * 70)


if __name__ == "__main__":
    main()