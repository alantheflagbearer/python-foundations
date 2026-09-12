"""
Day 36: Neural Network Fundamentals
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 6, Day 1
 
Titanic carried the roadmap through Week 5. Starting today, the dataset
switches to small synthetic 2D problems (AND/XOR truth tables, two
interleaving moons) that are purpose-built for seeing exactly what a neural
network can and can't represent -- something a tabular dataset makes harder
to visualize directly.
 
Covers:
  1. The perceptron — a single linear unit, trained with the classic
     perceptron learning rule, solving a linearly separable problem (AND)
  2. Why one perceptron can never solve XOR — not a training failure, a
     fundamental representational limit (XOR is not linearly separable)
  3. Activation functions from scratch — sigmoid, tanh, ReLU, softmax —
     compared side by side
  4. Depth solves XOR — a hand-crafted (not learned) 2-layer network,
     showing what extra representational power depth + non-linearity buys
  5. A small network LEARNS a harder problem (two moons) via manually
     derived backpropagation — no framework, so the mechanics are visible
  6. What a framework like PyTorch automates, and why Day 37 introduces one
 
This script is fully self-contained (NumPy + scikit-learn + matplotlib only).
PyTorch is intentionally not required to run this file — see the note at
the bottom about installing it ahead of Day 37.
 
Requires: pip install numpy scikit-learn matplotlib
"""
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
 
np.random.seed(42)
 
 
# ---------------------------------------------------------------------------
# 1. THE PERCEPTRON
# ---------------------------------------------------------------------------
 
class Perceptron:
    """A single linear unit: weighted sum + step activation, trained with
    the classic perceptron learning rule (nudge weights toward correct
    predictions, one misclassified example at a time)."""
 
    def __init__(self, n_inputs, lr=0.1):
        self.w = np.zeros(n_inputs)
        self.b = 0.0
        self.lr = lr
 
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return 1 if z >= 0 else 0
 
    def train(self, X, y, epochs=20):
        history = []
        for _ in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self.forward(xi)
                error = yi - pred
                if error != 0:
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    errors += 1
            history.append(errors)
            if errors == 0:
                break
        return history
 
 
def demo_perceptron_and_gate():
    print("=" * 70)
    print("1. THE PERCEPTRON — solving AND (linearly separable)")
    print("=" * 70)
 
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
 
    p = Perceptron(n_inputs=2)
    history = p.train(X_and, y_and)
 
    print(f"Converged in {len(history)} epochs (errors per epoch: {history})")
    for xi, yi in zip(X_and, y_and):
        print(f"  {xi} -> predicted {p.forward(xi)}, actual {yi}")
    print()
    return p
 
 
# ---------------------------------------------------------------------------
# 2. WHY XOR BREAKS A SINGLE PERCEPTRON
# ---------------------------------------------------------------------------
 
def demo_perceptron_xor_failure():
    print("=" * 70)
    print("2. WHY XOR BREAKS A SINGLE PERCEPTRON")
    print("=" * 70)
 
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
 
    p = Perceptron(n_inputs=2)
    history = p.train(X_xor, y_xor, epochs=50)
 
    correct = sum(p.forward(xi) == yi for xi, yi in zip(X_xor, y_xor))
    print(f"Ran {len(history)} epochs, never converged (errors per epoch: {history[:10]}...)")
    print(f"Final accuracy: {correct}/4")
    print("No straight line can separate XOR's two classes — this is not a training")
    print("bug, it's a hard representational limit of a single linear unit.\n")
    return p
 
 
# ---------------------------------------------------------------------------
# 3. ACTIVATION FUNCTIONS
# ---------------------------------------------------------------------------
 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
 
 
def tanh(z):
    return np.tanh(z)
 
 
def relu(z):
    return np.maximum(0, z)
 
 
def softmax(z):
    ez = np.exp(z - np.max(z))  # subtract max for numerical stability
    return ez / ez.sum()
 
 
def demo_activation_functions():
    print("=" * 70)
    print("3. ACTIVATION FUNCTIONS")
    print("=" * 70)
 
    print("Sample values at z = -2, 0, 2:")
    for name, fn in [("sigmoid", sigmoid), ("tanh", tanh), ("relu", relu)]:
        vals = [round(float(fn(np.array([zv]))[0]), 4) for zv in [-2, 0, 2]]
        print(f"  {name:8s}: {vals}")
    print("  softmax([2, 1, 0.1]):", np.round(softmax(np.array([2, 1, 0.1])), 4))
 
    z_vals = np.linspace(-5, 5, 200)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (name, fn) in zip(axes, [("Sigmoid", sigmoid), ("Tanh", tanh), ("ReLU", relu)]):
        ax.plot(z_vals, fn(z_vals))
        ax.set_title(name)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("activation_functions.png", dpi=110)
    plt.close()
    print("Saved: activation_functions.png\n")
 
 
# ---------------------------------------------------------------------------
# 4. DEPTH SOLVES XOR (hand-crafted weights, not learned)
# ---------------------------------------------------------------------------
 
def demo_handcrafted_xor_network():
    print("=" * 70)
    print("4. DEPTH SOLVES XOR — a hand-crafted 2-layer network")
    print("=" * 70)
 
    # Hidden layer: one unit approximates OR, one approximates NAND.
    # Output layer: AND of those two hidden units = XOR.
    W1 = np.array([[20, -20], [20, -20]])
    b1 = np.array([-10, 30])
    W2 = np.array([[20], [20]])
    b2 = np.array([-30])
 
    def forward(x):
        h = sigmoid(x @ W1 + b1)
        out = sigmoid(h @ W2 + b2)
        return out[0]
 
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
 
    correct = 0
    for xi, yi in zip(X_xor, y_xor):
        pred = forward(xi)
        pred_class = 1 if pred >= 0.5 else 0
        correct += (pred_class == yi)
        print(f"  {xi} -> {pred:.4f} (class {pred_class}), actual {yi}")
 
    print(f"Accuracy: {correct}/4")
    print("These weights were chosen by hand, not learned — the point is that a")
    print("network WITH this shape (2 layers, non-linear activation) CAN represent")
    print("XOR at all, something no single linear unit can do regardless of training.\n")
 
 
# ---------------------------------------------------------------------------
# 5. A SMALL NETWORK LEARNS A HARDER PROBLEM (make_moons) — manual backprop
# ---------------------------------------------------------------------------
 
class TinyNet:
    """A 2-layer network (sigmoid hidden + sigmoid output) trained with
    manually derived backpropagation — no autograd, so every gradient here
    is written out explicitly."""
 
    def __init__(self, n_in, n_hidden, lr=0.5):
        self.W1 = np.random.randn(n_in, n_hidden) * 0.5
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, 1) * 0.5
        self.b2 = np.zeros(1)
        self.lr = lr
 
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
 
    def backward(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)
        dz2 = self.a2 - y                      # combined BCE + sigmoid gradient
        dW2 = self.a1.T @ dz2 / m
        db2 = dz2.mean(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.a1 * (1 - self.a1)    # sigmoid derivative, chain rule
        dW1 = X.T @ dz1 / m
        db1 = dz1.mean(axis=0)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
 
    def train(self, X, y, epochs=3000):
        losses = []
        eps = 1e-9
        for _ in range(epochs):
            pred = self.forward(X)
            # y must be reshaped to (m, 1) to match pred's shape. Without this,
            # y (m,) broadcasts against pred (m,1) as if y were a (1,m) ROW
            # vector, producing an (m,m) matrix instead of an elementwise
            # (m,1) result — a genuine bug caught while verifying this script:
            # accuracy still improved correctly (the gradients in backward()
            # already reshape y correctly), but the printed loss climbed from
            # 0.90 to 1.70 instead of dropping, because it was silently being
            # computed over the wrong (m,m) shape the whole time.
            y_col = y.reshape(-1, 1)
            loss = -np.mean(y_col * np.log(pred + eps) + (1 - y_col) * np.log(1 - pred + eps))
            losses.append(loss)
            self.backward(X, y)
        return losses
 
 
def demo_moons_comparison():
    print("=" * 70)
    print("5. A SMALL NETWORK LEARNS A HARDER PROBLEM (two moons)")
    print("=" * 70)
 
    X_moons, y_moons = make_moons(n_samples=300, noise=0.2, random_state=42)
 
    net = TinyNet(n_in=2, n_hidden=8, lr=1.0)
    losses = net.train(X_moons, y_moons, epochs=3000)
    net_pred = (net.forward(X_moons) >= 0.5).astype(int).flatten()
    net_acc = (net_pred == y_moons).mean()
    print(f"TinyNet:     loss {losses[0]:.4f} -> {losses[-1]:.4f}, final accuracy {net_acc:.3f}")
 
    perc = Perceptron(n_inputs=2, lr=0.01)
    perc.train(X_moons, y_moons, epochs=50)
    perc_pred = np.array([perc.forward(xi) for xi in X_moons])
    perc_acc = (perc_pred == y_moons).mean()
    print(f"Perceptron:  final accuracy {perc_acc:.3f}")
    print("Same data, same evaluation — the network's extra layer and non-linearity")
    print("is the only difference, and it directly explains the accuracy gap.\n")
 
    print("=" * 70)
    print("6. VISUALIZING BOTH DECISION BOUNDARIES")
    print("=" * 70)
 
    xx, yy = np.meshgrid(
        np.linspace(X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5, 200),
        np.linspace(X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5, 200),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    net_zz = net.forward(grid).reshape(xx.shape)
    perc_zz = np.array([perc.forward(pt) for pt in grid]).reshape(xx.shape)
 
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = [
        (axes[0], perc_zz, "Single perceptron (linear)", perc_acc),
        (axes[1], net_zz, "2-layer network (non-linear)", net_acc),
    ]
    for ax, zz, title, acc in panels:
        ax.contourf(xx, yy, zz, levels=[-1, 0.5, 2], colors=["#F5C4B3", "#9FE1CB"], alpha=0.6)
        ax.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap="coolwarm",
                   s=12, edgecolors="k", linewidths=0.3)
        ax.set_title(f"{title}\naccuracy={acc:.3f}")
    plt.tight_layout()
    plt.savefig("decision_boundaries.png", dpi=110)
    plt.close()
    print("Saved: decision_boundaries.png")
    print("The perceptron draws exactly one straight line. The network bends that")
    print("line into a shape that actually follows the two moons.\n")
 
 
# ---------------------------------------------------------------------------
# 6. WHAT PYTORCH AUTOMATES
# ---------------------------------------------------------------------------
 
def note_on_pytorch():
    print("=" * 70)
    print("WHAT A FRAMEWORK LIKE PYTORCH AUTOMATES")
    print("=" * 70)
    print("Every gradient in TinyNet.backward() above was derived by hand and typed")
    print("out explicitly. PyTorch's autograd computes exactly those same derivatives")
    print("automatically from the forward pass alone — the same TinyNet, expressed as")
    print("an nn.Module, needs no backward() method at all; calling .backward() on the")
    print("loss walks the computation graph and fills in every gradient itself.")
    print()
    print("PyTorch is intentionally not required to run this file. Install it ahead")
    print("of Day 37, when the training loop (loss.backward(), optimizer.step()) is")
    print("the actual topic:")
    print("  pip install torch")
    print()
 
 
def main():
    demo_perceptron_and_gate()
    demo_perceptron_xor_failure()
    demo_activation_functions()
    demo_handcrafted_xor_network()
    demo_moons_comparison()
    note_on_pytorch()
    print("=" * 70)
    print("Day 36 complete. A single perceptron can only ever draw one straight line.")
    print("Depth plus non-linearity is what lets a network bend that line into a")
    print("shape that actually fits the data — the idea every deep learning")
    print("architecture from here on builds on.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
 