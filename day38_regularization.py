"""
Day 38: Regularization
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 6, Day 3
 
Day 36 and 37 built a network and learned it with plain gradient descent,
Momentum, and Adam. Neither day asked the question every real training run
eventually forces: what happens when the network is bigger (or the dataset
smaller) than the problem actually needs? This is where overfitting shows
up, and where regularization -- L2 weight decay, dropout, and early
stopping -- earns its place as a required tool, not an optional extra.
 
Covers:
  1. The overfitting problem, made visible on purpose -- a small, noisy
     dataset and an oversized network, trained long enough to memorize it
  2. L2 regularization (weight decay) -- penalizing large weights directly
     in the loss, derived by hand and added to the existing gradients
  3. Dropout -- randomly zeroing hidden units during training (inverted
     dropout), and why the backward pass needs the same mask
  4. Early stopping -- watching validation loss instead of training loss,
     and stopping the moment it stops improving
  5. All three side by side on the same overfitting problem
  6. What this maps to in PyTorch (nn.Dropout, weight_decay, manual
     early-stopping loops)
 
This script is fully self-contained (NumPy + scikit-learn + matplotlib
only). PyTorch is intentionally not required to run this file.
 
Requires: pip install numpy scikit-learn matplotlib
"""
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
 
np.random.seed(42)
 
 
# ---------------------------------------------------------------------------
# SHARED: sigmoid, BCE loss, and the dataset
# ---------------------------------------------------------------------------
 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
 
 
def bce_loss(pred, target, eps=1e-9):
    pred = np.clip(pred, eps, 1 - eps)
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))
 
 
def make_overfit_dataset():
    """Deliberately small and noisy -- 30 points total, split 20/10 -- so
    an oversized 64-unit network can genuinely memorize the training half."""
    X, y = make_moons(n_samples=30, noise=0.35, random_state=42)
    y = y.reshape(-1, 1)
    X_train, y_train = X[:20], y[:20]
    X_val, y_val = X[20:], y[20:]
    return X_train, y_train, X_val, y_val
 
 
# ---------------------------------------------------------------------------
# 1. THE NETWORK -- with optional L2 and optional dropout
# ---------------------------------------------------------------------------
 
class RegNet:
    """Same 2-layer sigmoid network as Day 36/37's TinyNet/Net, extended
    with two optional regularizers: L2 weight decay (lam) and inverted
    dropout (keep_prob on the hidden layer)."""
 
    def __init__(self, n_in, n_hidden, lr=0.5, lam=0.0, keep_prob=1.0, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(n_in, n_hidden) * 0.5
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden, 1) * 0.5
        self.b2 = np.zeros(1)
        self.lr = lr
        self.lam = lam
        self.keep_prob = keep_prob
 
    def forward(self, X, training=False):
        self.z1 = X @ self.W1 + self.b1
        self.h1 = sigmoid(self.z1)              # pre-dropout hidden activation
        if training and self.keep_prob < 1.0:
            self.mask = (np.random.rand(*self.h1.shape) < self.keep_prob) / self.keep_prob
            self.a1 = self.h1 * self.mask
        else:
            self.mask = None
            self.a1 = self.h1
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
 
    def backward(self, X, y_col):
        m = X.shape[0]
        dz2 = self.a2 - y_col
        dW2 = self.a1.T @ dz2 / m
        db2 = dz2.mean(axis=0)
        da1 = dz2 @ self.W2.T
        dh1 = da1 * self.mask if self.mask is not None else da1
        dz1 = dh1 * self.h1 * (1 - self.h1)
        dW1 = X.T @ dz1 / m
        db1 = dz1.mean(axis=0)
 
        # L2 weight decay: add lam * W directly to the weight gradients only
        # (biases are conventionally left unregularized).
        dW1 = dW1 + self.lam * self.W1
        dW2 = dW2 + self.lam * self.W2
 
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
 
    def train(self, X_train, y_train, X_val, y_val, epochs, patience=None):
        train_losses, val_losses = [], []
        best_val = np.inf
        best_epoch = 0
        epochs_no_improve = 0
        stopped_at = epochs
        for epoch in range(epochs):
            pred_train = self.forward(X_train, training=True)
            self.backward(X_train, y_train)
 
            train_pred_eval = self.forward(X_train, training=False)
            val_pred = self.forward(X_val, training=False)
            train_loss = bce_loss(train_pred_eval, y_train)
            val_loss = bce_loss(val_pred, y_val)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
 
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
 
            if patience is not None and epochs_no_improve >= patience:
                stopped_at = epoch + 1
                break
 
        return train_losses, val_losses, best_epoch, stopped_at
 
    def accuracy(self, X, y):
        pred = (self.forward(X, training=False) >= 0.5).astype(int)
        return (pred == y).mean()
 
 
# ---------------------------------------------------------------------------
# demo functions
# ---------------------------------------------------------------------------
 
def demo_overfitting():
    print("=" * 70)
    print("1. THE OVERFITTING PROBLEM")
    print("=" * 70)
    X_train, y_train, X_val, y_val = make_overfit_dataset()
    print(f"Train set: {X_train.shape[0]} points, Val set: {X_val.shape[0]} points")
    print("Network: 64 hidden units -- deliberately oversized for 20 training points.\n")
 
    net = RegNet(n_in=2, n_hidden=64, lr=0.5, lam=0.0, keep_prob=1.0)
    train_losses, val_losses, best_epoch, _ = net.train(
        X_train, y_train, X_val, y_val, epochs=3000
    )
    print(f"Epoch 0:    train_loss={train_losses[0]:.4f}  val_loss={val_losses[0]:.4f}")
    print(f"Epoch 500:  train_loss={train_losses[499]:.4f}  val_loss={val_losses[499]:.4f}")
    print(f"Epoch {best_epoch}: val_loss hits its BEST value = {val_losses[best_epoch]:.4f}")
    print(f"Epoch 2999: train_loss={train_losses[-1]:.4f}  val_loss={val_losses[-1]:.4f}")
    print(f"Final train accuracy: {net.accuracy(X_train, y_train):.3f}")
    print(f"Final val accuracy:   {net.accuracy(X_val, y_val):.3f}")
    print("Train loss keeps falling toward 0 while val loss bottoms out early and")
    print("climbs back up -- the network is memorizing the 20 training points instead")
    print("of learning the underlying pattern.\n")
    return train_losses, val_losses, best_epoch
 
 
def demo_l2():
    print("=" * 70)
    print("2. L2 REGULARIZATION (WEIGHT DECAY)")
    print("=" * 70)
    X_train, y_train, X_val, y_val = make_overfit_dataset()
    net = RegNet(n_in=2, n_hidden=64, lr=0.5, lam=0.02, keep_prob=1.0)
    train_losses, val_losses, best_epoch, _ = net.train(
        X_train, y_train, X_val, y_val, epochs=3000
    )
    print(f"lam=0.02 -- Epoch 2999: train_loss={train_losses[-1]:.4f}  "
          f"val_loss={val_losses[-1]:.4f}")
    print(f"Best val_loss={val_losses[best_epoch]:.4f} at epoch {best_epoch}")
    print(f"Final train accuracy: {net.accuracy(X_train, y_train):.3f}")
    print(f"Final val accuracy:   {net.accuracy(X_val, y_val):.3f}")
    print("Penalizing large weights keeps the decision boundary simpler, so val loss")
    print("no longer climbs back up the way it did with no regularization.\n")
    return train_losses, val_losses, best_epoch
 
 
def demo_dropout():
    print("=" * 70)
    print("3. DROPOUT")
    print("=" * 70)
    X_train, y_train, X_val, y_val = make_overfit_dataset()
    net = RegNet(n_in=2, n_hidden=64, lr=0.5, lam=0.0, keep_prob=0.6)
    train_losses, val_losses, best_epoch, _ = net.train(
        X_train, y_train, X_val, y_val, epochs=3000
    )
    print(f"keep_prob=0.6 -- Epoch 2999: train_loss={train_losses[-1]:.4f}  "
          f"val_loss={val_losses[-1]:.4f}")
    print(f"Best val_loss={val_losses[best_epoch]:.4f} at epoch {best_epoch}")
    print(f"Final train accuracy: {net.accuracy(X_train, y_train):.3f}")
    print(f"Final val accuracy:   {net.accuracy(X_val, y_val):.3f}")
    print("Randomly zeroing 40% of hidden units each step forces the network to not")
    print("rely on any single unit -- a different way of fighting the same problem.\n")
    return train_losses, val_losses, best_epoch
 
 
def demo_early_stopping():
    print("=" * 70)
    print("4. EARLY STOPPING")
    print("=" * 70)
    X_train, y_train, X_val, y_val = make_overfit_dataset()
    net = RegNet(n_in=2, n_hidden=64, lr=0.5, lam=0.0, keep_prob=1.0)
    train_losses, val_losses, best_epoch, stopped_at = net.train(
        X_train, y_train, X_val, y_val, epochs=3000, patience=200
    )
    print(f"Training stopped at epoch {stopped_at} (patience=200 epochs with no "
          f"val_loss improvement)")
    print(f"Best val_loss={val_losses[best_epoch]:.4f} at epoch {best_epoch}")
    print(f"train_loss at stop={train_losses[-1]:.4f}  val_loss at stop={val_losses[-1]:.4f}")
    print(f"Final train accuracy: {net.accuracy(X_train, y_train):.3f}")
    print(f"Final val accuracy:   {net.accuracy(X_val, y_val):.3f}")
    print("No change to the loss function or the network at all -- just watching val")
    print("loss and refusing to keep training once it stops paying off.\n")
    return train_losses, val_losses, best_epoch, stopped_at
 
 
def demo_comparison(results):
    print("=" * 70)
    print("5. ALL THREE, SIDE BY SIDE")
    print("=" * 70)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    titles = ["No regularization", "L2 weight decay (lam=0.02)",
              "Dropout (keep_prob=0.6)", "Early stopping (patience=200)"]
    for ax, title, (train_losses, val_losses) in zip(axes.flat, titles, results):
        ax.plot(train_losses, label="train", color="#378ADD", linewidth=1.4)
        ax.plot(val_losses, label="val", color="#D85A30", linewidth=1.4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("BCE loss")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.4)
    plt.tight_layout()
    plt.savefig("regularization_comparison.png", dpi=110)
    plt.close()
    print("Saved: regularization_comparison.png")
    print("Same network, same data, same 3000-epoch budget -- only the regularization")
    print("strategy changes, and the val-loss curve tells the whole story.\n")
 
 
def note_on_pytorch():
    print("=" * 70)
    print("WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)
    print("nn.Dropout(p=0.4)                          # keep_prob=0.6 here")
    print("optim.SGD(params, lr=0.5, weight_decay=0.02) # L2, same as lam=0.02 here")
    print("# early stopping has no built-in class -- it's the same manual")
    print("# best-val-loss-and-patience loop used in demo_early_stopping() above")
    print()
    print("PyTorch's nn.Dropout automatically does nothing at eval time (net.eval()),")
    print("the same on/off switch this script implements by hand with the")
    print("`training=True/False` flag on forward().\n")
 
 
def main():
    train_losses_0, val_losses_0, best_epoch_0 = demo_overfitting()
    train_losses_l2, val_losses_l2, best_epoch_l2 = demo_l2()
    train_losses_do, val_losses_do, best_epoch_do = demo_dropout()
    train_losses_es, val_losses_es, best_epoch_es, stopped_at = demo_early_stopping()
    demo_comparison([
        (train_losses_0, val_losses_0),
        (train_losses_l2, val_losses_l2),
        (train_losses_do, val_losses_do),
        (train_losses_es, val_losses_es),
    ])
    note_on_pytorch()
    print("=" * 70)
    print("Day 38 complete. A bigger network is not automatically a better one --")
    print("regularization is what keeps capacity from turning into memorization.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
 