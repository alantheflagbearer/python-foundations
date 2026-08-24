"""
Day 39: Softmax, Categorical Cross-Entropy, and Mini-Batch Training
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 6, Day 4

Days 36-38 built and trained a binary classifier (two classes: 0 or 1) using
a sigmoid output and binary cross-entropy loss, and learned three ways to
keep it from overfitting. Two assumptions quietly held throughout: only two
classes ever existed, and every single gradient step used the ENTIRE
training set. Today removes both assumptions: softmax + categorical
cross-entropy generalize to any number of classes, and mini-batch training
generalizes the update rule to work on datasets too large to fit through
the network in one step.

Covers:
  1. Softmax -- turning K raw scores (logits) into a probability
     distribution over K classes
  2. Categorical cross-entropy -- the multi-class generalization of Day
     37's binary cross-entropy
  3. One-hot encoding -- representing a class label as a vector
  4. A 2-layer network with a softmax output layer, and the remarkably
     simple gradient that falls out of combining softmax with
     cross-entropy
  5. Full-batch training (every example, every step) vs mini-batch
     training (small shuffled chunks, several updates per epoch)
  6. Both trained side by side on the same real multi-class dataset
  7. What this maps to in PyTorch (nn.CrossEntropyLoss, DataLoader)

This script is fully self-contained (NumPy + scikit-learn + matplotlib
only). PyTorch is intentionally not required to run this file.

Requires: pip install numpy scikit-learn matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. SOFTMAX
# ---------------------------------------------------------------------------

def softmax(z):
    """z is (m, K) -- one row of K raw scores (logits) per example.
    Subtracting the row max before exponentiating is the standard numerical
    stability trick: it changes nothing mathematically (softmax is shift
    invariant) but prevents np.exp from overflowing on large logits."""
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def demo_softmax():
    print("=" * 70)
    print("1. SOFTMAX")
    print("=" * 70)
    logits = np.array([[2.0, 1.0, 0.1],
                        [1.0, 1.0, 1.0],
                        [5.0, 1.0, 1.0]])
    probs = softmax(logits)
    for row_logits, row_probs in zip(logits, probs):
        print(f"logits={row_logits}  ->  probs={np.round(row_probs, 4)}  "
              f"(sum={row_probs.sum():.6f})")
    print("Every row sums to 1.0 regardless of the input scale -- softmax always")
    print("produces a valid probability distribution over the K classes.\n")
    return logits, probs


# ---------------------------------------------------------------------------
# 2. CATEGORICAL CROSS-ENTROPY AND ONE-HOT ENCODING
# ---------------------------------------------------------------------------

def one_hot(y, num_classes):
    """y is a 1D array of integer class labels (0..num_classes-1).
    Returns an (m, num_classes) array where row i is all zeros except a 1
    at column y[i]."""
    m = y.shape[0]
    encoded = np.zeros((m, num_classes))
    encoded[np.arange(m), y] = 1.0
    return encoded


def cross_entropy_loss(probs, y_onehot, eps=1e-9):
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(np.sum(y_onehot * np.log(probs), axis=1))


def demo_cross_entropy():
    print("=" * 70)
    print("2. CATEGORICAL CROSS-ENTROPY AND ONE-HOT ENCODING")
    print("=" * 70)
    y = np.array([0, 2, 1])
    y_onehot = one_hot(y, num_classes=3)
    print(f"y = {y}")
    print(f"one_hot(y) =\n{y_onehot}")

    probs_good = np.array([[0.9, 0.05, 0.05],
                            [0.05, 0.05, 0.9],
                            [0.1, 0.8, 0.1]])
    probs_bad = np.array([[0.2, 0.4, 0.4],
                           [0.4, 0.4, 0.2],
                           [0.4, 0.2, 0.4]])
    ce_good = cross_entropy_loss(probs_good, y_onehot)
    ce_bad = cross_entropy_loss(probs_bad, y_onehot)
    print(f"Confident, correct predictions -> cross-entropy = {ce_good:.4f}")
    print(f"Unsure, still-technically-plausible predictions -> cross-entropy = {ce_bad:.4f}")
    print("Same shape as Day 37's binary cross-entropy, generalized to K columns --")
    print("one-hot encoding is what makes 'multiply by the true class only' work as a\n"
          "single vectorized sum instead of an if/elif per class.\n")
    return y, y_onehot


# ---------------------------------------------------------------------------
# 3. THE MULTI-CLASS DATASET
# ---------------------------------------------------------------------------

def make_multiclass_dataset():
    X, y = make_blobs(n_samples=300, centers=3, n_features=2,
                       cluster_std=1.8, random_state=42)
    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------
# 4. THE NETWORK -- softmax output layer
# ---------------------------------------------------------------------------

class SoftmaxNet:
    """2-layer network: sigmoid hidden layer, softmax output layer over K
    classes. Combining softmax with cross-entropy gives an unusually simple
    gradient at the output layer: dz2 = probs - y_onehot -- derived in the
    study guide, used directly here."""

    def __init__(self, n_in, n_hidden, n_classes, lr=0.5, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(n_in, n_hidden) * 0.5
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden, n_classes) * 0.5
        self.b2 = np.zeros(n_classes)
        self.lr = lr

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.probs = softmax(self.z2)
        return self.probs

    def backward(self, X, y_onehot):
        m = X.shape[0]
        dz2 = self.probs - y_onehot
        dW2 = self.a1.T @ dz2 / m
        db2 = dz2.mean(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.a1 * (1 - self.a1)
        dW1 = X.T @ dz1 / m
        db1 = dz1.mean(axis=0)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs, batch_size=None, seed=0):
        num_classes = self.W2.shape[1]
        y_onehot_full = one_hot(y, num_classes)
        m = X.shape[0]
        rng = np.random.RandomState(seed)
        losses = []
        updates_per_epoch = 1 if batch_size is None else int(np.ceil(m / batch_size))

        for epoch in range(epochs):
            if batch_size is None:
                self.forward(X)
                self.backward(X, y_onehot_full)
            else:
                idx = rng.permutation(m)
                for start in range(0, m, batch_size):
                    batch_idx = idx[start:start + batch_size]
                    self.forward(X[batch_idx])
                    self.backward(X[batch_idx], y_onehot_full[batch_idx])

            probs_full = self.forward(X)
            losses.append(cross_entropy_loss(probs_full, y_onehot_full))

        return losses, updates_per_epoch

    def accuracy(self, X, y):
        probs = self.forward(X)
        pred = probs.argmax(axis=1)
        return (pred == y).mean()


# ---------------------------------------------------------------------------
# demo functions
# ---------------------------------------------------------------------------

def demo_dataset():
    print("=" * 70)
    print("3. THE MULTI-CLASS DATASET")
    print("=" * 70)
    X, y = make_multiclass_dataset()
    print(f"X shape: {X.shape}   y shape: {y.shape}   classes: {sorted(set(y.tolist()))}")
    for c in sorted(set(y.tolist())):
        print(f"  class {c}: {(y == c).sum()} points")

    fig, ax = plt.subplots(figsize=(6, 5))
    colors_map = {0: "#378ADD", 1: "#D85A30", 2: "#5a2d8c"}
    for c in sorted(set(y.tolist())):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], s=18, color=colors_map[c], label=f"class {c}")
    ax.legend()
    ax.set_title("3-class dataset (make_blobs)")
    plt.tight_layout()
    plt.savefig("multiclass_dataset.png", dpi=110)
    plt.close()
    print("Saved: multiclass_dataset.png\n")
    return X, y


def demo_full_batch(X, y):
    print("=" * 70)
    print("4. FULL-BATCH TRAINING")
    print("=" * 70)
    net = SoftmaxNet(n_in=2, n_hidden=16, n_classes=3, lr=0.5, seed=42)
    losses, updates_per_epoch = net.train(X, y, epochs=300, batch_size=None)
    print(f"Every epoch = {updates_per_epoch} gradient update (the whole dataset at once)")
    print(f"Epoch 0:   loss={losses[0]:.4f}")
    print(f"Epoch 50:  loss={losses[50]:.4f}")
    print(f"Epoch 299: loss={losses[-1]:.4f}")
    print(f"Final accuracy: {net.accuracy(X, y):.3f}\n")
    return losses, updates_per_epoch


def demo_mini_batch(X, y):
    print("=" * 70)
    print("5. MINI-BATCH TRAINING")
    print("=" * 70)
    net = SoftmaxNet(n_in=2, n_hidden=16, n_classes=3, lr=0.5, seed=42)
    losses, updates_per_epoch = net.train(X, y, epochs=300, batch_size=32, seed=0)
    print(f"Every epoch = {updates_per_epoch} gradient updates (300 points / batch_size 32, "
          f"shuffled)")
    print(f"Epoch 0:   loss={losses[0]:.4f}")
    print(f"Epoch 50:  loss={losses[50]:.4f}")
    print(f"Epoch 299: loss={losses[-1]:.4f}")
    print(f"Final accuracy: {net.accuracy(X, y):.3f}\n")
    return losses, updates_per_epoch


def demo_comparison(fb_losses, fb_upe, mb_losses, mb_upe):
    print("=" * 70)
    print("6. FULL-BATCH VS MINI-BATCH, SIDE BY SIDE")
    print("=" * 70)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fb_losses, label=f"full-batch ({fb_upe} update/epoch)", color="#378ADD", linewidth=1.5)
    ax.plot(mb_losses, label=f"mini-batch ({mb_upe} updates/epoch)", color="#D85A30", linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss (full training set)")
    ax.set_title("Full-batch vs mini-batch: loss per epoch, same network/data")
    ax.legend()
    plt.tight_layout()
    plt.savefig("batch_comparison.png", dpi=110)
    plt.close()
    print("Saved: batch_comparison.png")
    print(f"Full-batch:  {fb_upe:>2} update/epoch  -> final loss {fb_losses[-1]:.4f}")
    print(f"Mini-batch:  {mb_upe:>2} updates/epoch -> final loss {mb_losses[-1]:.4f}")
    print("Same data, same architecture, same epoch budget -- mini-batch gets many more\n"
          "weight updates per epoch, which is the entire mechanical reason it can reach a\n"
          "lower loss in the same number of epochs.\n")


def note_on_pytorch():
    print("=" * 70)
    print("WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)
    print("loss_fn = nn.CrossEntropyLoss()   # takes RAW logits, not softmax output --")
    print("                                  # it applies log-softmax internally")
    print("loader = DataLoader(dataset, batch_size=32, shuffle=True)")
    print("for X_batch, y_batch in loader:   # replaces the manual idx/permutation loop")
    print("    ...")
    print()
    print("PyTorch's nn.CrossEntropyLoss expects raw, un-softmaxed logits as input --")
    print("passing already-softmaxed probabilities into it is a common bug, since it\n"
          "applies its own (numerically stabilized) log-softmax internally.\n")


def main():
    demo_softmax()
    demo_cross_entropy()
    X, y = demo_dataset()
    fb_losses, fb_upe = demo_full_batch(X, y)
    mb_losses, mb_upe = demo_mini_batch(X, y)
    demo_comparison(fb_losses, fb_upe, mb_losses, mb_upe)
    note_on_pytorch()
    print("=" * 70)
    print("Day 39 complete. Softmax + cross-entropy generalize binary classification to")
    print("any number of classes; mini-batching generalizes the update rule to datasets")
    print("too large for one full-batch step -- neither changes the underlying math.")
    print("=" * 70)


if __name__ == "__main__":
    main()