"""
Day 37: Loss Functions and Optimizers
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 6, Day 2
 
Day 36 built one training loop by hand for one problem. Today generalizes
both halves of that loop: how "wrongness" gets measured (loss functions),
and how weights get updated once you know which direction reduces it
(optimizers) — comparing plain gradient descent against Momentum and Adam,
on both a simple 2D toy surface (to see the mechanism clearly) and the real
two-moons network from Day 36 (to see why it actually matters).
 
Covers:
  1. Loss functions — MSE (regression) vs binary cross-entropy
     (classification), and why the choice isn't just stylistic
  2. Plain gradient descent's zigzag problem, visualized on a toy 2D
     surface with deliberately different curvature in each direction
  3. Momentum — smoothing the zigzag by accumulating a velocity term
  4. Adam — combining momentum with a per-parameter adaptive learning rate
  5. The real test: SGD vs Momentum vs Adam training the Day 36 network on
     two moons — where Adam's advantage actually shows up
  6. What this maps to in PyTorch (nn.MSELoss, nn.BCELoss, optim.SGD,
     optim.Adam) — reference only, not required to run this file
 
This script is fully self-contained (NumPy + scikit-learn + matplotlib
only). PyTorch is intentionally not required — see the note at the bottom.
 
Requires: pip install numpy scikit-learn matplotlib
"""
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
 
np.random.seed(42)
 
 
# ---------------------------------------------------------------------------
# 1. LOSS FUNCTIONS
# ---------------------------------------------------------------------------

def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)

def bce_loss(pred, target, eps=1e-9):
    pred = np.clip(pred, eps, 1 - eps)
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))

def demo_loss_functions():
    print("=" * 70)
    print("1. LOSS FUNCTIONS — MSE vs Binary Cross-Entropy")
    print("=" * 70)

    pred_good = np.array([0.9, 0.05])
    pred_bad = np.array([0.1, 0.95])
    target = np.array([1, 0])

    print(f"Good predictions {pred_good}, target {target}")
    print(f"  MSE: {mse_loss(pred_good, target):.4f}   BCE: {bce_loss(pred_good, target):.4f}")
    print(f"Bad predictions {pred_bad}, target {target}")
    print(f"  MSE: {mse_loss(pred_bad, target):.4f}   BCE: {bce_loss(pred_bad, target):.4f}")
    print("BCE penalizes a confident wrong prediction far more harshly than MSE does —")
    print("exactly the signal you want when training a classifier.\n")
 
 
# ---------------------------------------------------------------------------
# 2-4. GRADIENT DESCENT / MOMENTUM / ADAM ON A TOY 2D SURFACE
# ---------------------------------------------------------------------------
 
def toy_surface(xy):
    x, y = xy
    return 0.1 * x ** 2 + 2 * y ** 2
 
 
def toy_gradient(xy):
    x, y = xy
    return np.array([0.2 * x, 4 * y])
 
 
def run_sgd(start, lr, steps=40):
    xy = np.array(start, dtype=float)
    path = [xy.copy()]
    for _ in range(steps):
        xy = xy - lr * toy_gradient(xy)
        path.append(xy.copy())
    return np.array(path)
 
 
def run_momentum(start, lr, beta=0.7, steps=40):
    xy = np.array(start, dtype=float)
    v = np.zeros(2)
    path = [xy.copy()]
    for _ in range(steps):
        g = toy_gradient(xy)
        v = beta * v + (1 - beta) * g
        xy = xy - lr * v
        path.append(xy.copy())
    return np.array(path)
 
 
def run_adam(start, lr, beta1=0.9, beta2=0.999, steps=40, eps=1e-8):
    xy = np.array(start, dtype=float)
    m = np.zeros(2)
    v = np.zeros(2)
    path = [xy.copy()]
    for t in range(1, steps + 1):
        g = toy_gradient(xy)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        xy = xy - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(xy.copy())
    return np.array(path)
 
 
def demo_toy_surface_optimizers():
    print("=" * 70)
    print("2-4. GRADIENT DESCENT, MOMENTUM, AND ADAM ON A TOY 2D SURFACE")
    print("=" * 70)
    print("Surface: f(x,y) = 0.1*x^2 + 2*y^2 — much steeper in y than in x,")
    print("the classic setup that makes plain gradient descent zigzag.\n")
 
    start = [-8, -4]
    path_sgd = run_sgd(start, lr=0.46)
    path_mom = run_momentum(start, lr=0.46, beta=0.7)
    path_adam = run_adam(start, lr=0.30)
 
    print(f"Plain SGD  (lr=0.46):            final loss={toy_surface(path_sgd[-1]):.5f}")
    print(f"Momentum   (lr=0.46, beta=0.7):  final loss={toy_surface(path_mom[-1]):.5f}")
    print(f"Adam       (lr=0.30):            final loss={toy_surface(path_adam[-1]):.5f}")
    print("Momentum wins here — this toy surface is smooth and low-dimensional, exactly")
    print("the regime where a well-tuned constant velocity term does great. Section 5")
    print("shows a harder problem where that ranking flips.\n")
 
    xg = np.linspace(-9, 4, 200)
    yg = np.linspace(-5, 5, 200)
    XG, YG = np.meshgrid(xg, yg)
    ZG = 0.1 * XG ** 2 + 2 * YG ** 2
 
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [
        (axes[0], path_sgd, f"Plain SGD, lr=0.46\nfinal loss={toy_surface(path_sgd[-1]):.4f}"),
        (axes[1], path_mom, f"Momentum, same lr=0.46\nfinal loss={toy_surface(path_mom[-1]):.5f}"),
        (axes[2], path_adam, f"Adam, lr=0.30\nfinal loss={toy_surface(path_adam[-1]):.4f}"),
    ]
    for ax, path, title in panels:
        ax.contour(XG, YG, ZG, levels=15, colors="#D3D1C7", linewidths=0.7)
        ax.plot(path[:, 0], path[:, 1], "-o", color="#7A4EC9", markersize=2.5, linewidth=1.1)
        ax.plot(0, 0, "*", color="#D85A30", markersize=14)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(-9, 4)
        ax.set_ylim(-5, 5)
    plt.tight_layout()
    plt.savefig("optimizer_paths.png", dpi=110)
    plt.close()
    print("Saved: optimizer_paths.png\n")
 
 
# ---------------------------------------------------------------------------
# 5. THE REAL TEST — SGD vs MOMENTUM vs ADAM ON TWO MOONS
# ---------------------------------------------------------------------------
 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
 
 
class Net:
    """Day 36's TinyNet, generalized with a pluggable optimizer."""
 
    def __init__(self, n_in, n_hidden, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(n_in, n_hidden) * 0.5
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden, 1) * 0.5
        self.b2 = np.zeros(1)
        self.m = {k: np.zeros_like(getattr(self, k)) for k in ["W1", "b1", "W2", "b2"]}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in ["W1", "b1", "W2", "b2"]}
        self.t = 0
 
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
 
    def grads(self, X, y_col):
        m = X.shape[0]
        dz2 = self.a2 - y_col
        dW2 = self.a1.T @ dz2 / m
        db2 = dz2.mean(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.a1 * (1 - self.a1)
        dW1 = X.T @ dz1 / m
        db1 = dz1.mean(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
 
    def step_sgd(self, g, lr):
        for k in g:
            setattr(self, k, getattr(self, k) - lr * g[k])
 
    def step_momentum(self, g, lr, beta=0.9):
        for k in g:
            self.m[k] = beta * self.m[k] + (1 - beta) * g[k]
            setattr(self, k, getattr(self, k) - lr * self.m[k])
 
    def step_adam(self, g, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for k in g:
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * g[k]
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (g[k] ** 2)
            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.v[k] / (1 - beta2 ** self.t)
            setattr(self, k, getattr(self, k) - lr * m_hat / (np.sqrt(v_hat) + eps))
 
 
def train(X, y, optimizer, lr, epochs=1000):
    y_col = y.reshape(-1, 1)
    net = Net(n_in=2, n_hidden=8, seed=42)
    losses = []
    for _ in range(epochs):
        pred = net.forward(X)
        losses.append(bce_loss(pred, y_col))
        g = net.grads(X, y_col)
        if optimizer == "sgd":
            net.step_sgd(g, lr)
        elif optimizer == "momentum":
            net.step_momentum(g, lr)
        elif optimizer == "adam":
            net.step_adam(g, lr)
    final_pred = (net.forward(X) >= 0.5).astype(int).flatten()
    acc = (final_pred == y).mean()
    return losses, acc
 
 
def demo_moons_optimizer_comparison():
    print("=" * 70)
    print("5. THE REAL TEST — SGD vs MOMENTUM vs ADAM ON TWO MOONS")
    print("=" * 70)
 
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
 
    configs = [("sgd", 1.0), ("momentum", 1.0), ("adam", 0.05)]
    results = {}
    for opt, lr in configs:
        losses, acc = train(X, y, opt, lr, epochs=1000)
        results[opt] = (losses, acc)
        print(f"{opt:10s} lr={lr}: final_loss={losses[-1]:.4f}  final_accuracy={acc:.3f}")
 
    print("\nSGD and Momentum both plateau around loss=0.29 and never escape it in 1000")
    print("epochs. Adam breaks through that same plateau and keeps improving to 0.06 —")
    print("this is where Adam's per-parameter adaptive scaling actually earns its keep,")
    print("unlike the toy surface above where a fixed-curvature problem favored Momentum.\n")
 
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"sgd": "#D85A30", "momentum": "#7A4EC9", "adam": "#1D9E75"}
    labels = {"sgd": "Plain SGD (lr=1.0)", "momentum": "Momentum (lr=1.0)", "adam": "Adam (lr=0.05)"}
    for opt in ["sgd", "momentum", "adam"]:
        losses, acc = results[opt]
        ax.plot(losses, label=f"{labels[opt]} — final acc {acc:.3f}", color=colors[opt], linewidth=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("binary cross-entropy loss")
    ax.legend()
    ax.set_title("Same network, same data — three optimizers, 1000 epochs")
    plt.tight_layout()
    plt.savefig("optimizer_comparison_moons.png", dpi=110)
    plt.close()
    print("Saved: optimizer_comparison_moons.png\n")
 
 
# ---------------------------------------------------------------------------
# 6. WHAT THIS MAPS TO IN PYTORCH
# ---------------------------------------------------------------------------
 
def note_on_pytorch():
    print("=" * 70)
    print("WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)
    print("loss_fn = nn.BCELoss()                         # same formula as bce_loss() above")
    print("optimizer = torch.optim.SGD(params, lr=1.0, momentum=0.9)   # step_momentum()")
    print("optimizer = torch.optim.Adam(params, lr=0.05)               # step_adam()")
    print()
    print("Every optimizer implemented by hand above has a direct PyTorch equivalent —")
    print("same update rule, same hyperparameters, computed automatically from whatever")
    print("gradients autograd produces instead of the hand-written grads() method here.")
    print("PyTorch is intentionally not required to run this file — install it if you")
    print("haven't yet (large download, ~500MB+):")
    print("  pip install torch\n")
 
 
def main():
    demo_loss_functions()
    demo_toy_surface_optimizers()
    demo_moons_optimizer_comparison()
    note_on_pytorch()
    print("=" * 70)
    print("Day 37 complete. Same gradients, three different ways of using them to update")
    print("weights — and which one wins depends on the shape of the problem, not on any")
    print("optimizer being universally 'best'.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
