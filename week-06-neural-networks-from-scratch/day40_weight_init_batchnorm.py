"""
Day 40: Weight Initialization Strategies and Batch Normalization
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 6, Day 5

Every network from Day 36 through Day 39 used one fixed rule for starting
weights: randn(...) * 0.5, picked because it happened to work on small,
shallow (2-layer) networks. Today asks what happens once a network gets
deeper, and shows that "just pick a small random number" stops being good
enough -- the SCALE of that random number, chosen systematically, is the
difference between a deep network that trains and one that never learns
anything at all.

Covers:
  1. Why initialization scale matters -- watching a signal propagate
     through a deep, UNTRAINED network under four different init scales
  2. Xavier/Glorot initialization -- scaling by the layer's fan-in
  3. He initialization -- Xavier's ReLU-oriented cousin
  4. Batch normalization -- normalizing each layer's pre-activation
     distribution directly, regardless of initialization
  5. The real test: training an actual 6-hidden-layer network on real
     data under each initialization, with and without batch norm
  6. What this maps to in PyTorch (nn.init.*, nn.BatchNorm1d)

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
# SHARED: activation, softmax, one-hot, loss
# ---------------------------------------------------------------------------

def tanh(z):
    return np.tanh(z)


def dtanh(a):
    """Derivative of tanh w.r.t. its input, expressed in terms of the
    already-computed output a = tanh(z) -- avoids recomputing tanh."""
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


def init_scale(strategy, n_in):
    """The four initialization strategies this lesson compares, as a single
    scale factor multiplying a standard-normal random draw."""
    if strategy == "naive_small":
        return 0.01
    elif strategy == "naive_large":
        return 1.0
    elif strategy == "xavier":
        return np.sqrt(1.0 / n_in)
    elif strategy == "he":
        return np.sqrt(2.0 / n_in)
    raise ValueError(f"unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# 1. SIGNAL PROPAGATION -- forward pass only, no training
# ---------------------------------------------------------------------------

def propagate_signal(n_layers, n_units, strategy, use_bn=False, n_samples=200, seed=0):
    """Pushes random input through n_layers of untrained tanh layers and
    records each layer's output std -- no training happens here at all,
    this purely observes what a given init scale does to signal as it
    flows through depth."""
    rng = np.random.RandomState(seed)
    a = rng.randn(n_samples, n_units)
    stds = []
    for _ in range(n_layers):
        scale = init_scale(strategy, n_units)
        W = rng.randn(n_units, n_units) * scale
        b = np.zeros(n_units)
        z = a @ W + b
        if use_bn:
            mu = z.mean(axis=0, keepdims=True)
            var = z.var(axis=0, keepdims=True)
            z = (z - mu) / np.sqrt(var + 1e-5)
        a = tanh(z)
        stds.append(a.std())
    return stds


def demo_init_stats():
    print("=" * 70)
    print("1. SIGNAL PROPAGATION THROUGH AN UNTRAINED DEEP NETWORK")
    print("=" * 70)
    print("10 layers, 64 units each, tanh activation, random input -- no training.\n")
    strategies = ["naive_small", "naive_large", "xavier", "he"]
    results = {}
    for strategy in strategies:
        stds = propagate_signal(n_layers=10, n_units=64, strategy=strategy)
        results[strategy] = stds
        rounded = [f"{s:.4f}" for s in stds]
        print(f"{strategy:14s} layer stds: {rounded}")
    print("\nnaive_small collapses to exactly 0.0000 by layer 4 -- the signal is dead.")
    print("naive_large stays high (~0.95) but that's SATURATION, not health -- tanh")
    print("is pinned near +-1, where its derivative is near zero.")
    print("xavier and he decay far more gradually -- real, but far survivable.\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"naive_small": "#a32020", "naive_large": "#b3651a",
              "xavier": "#378ADD", "he": "#5a2d8c"}
    for strategy in strategies:
        ax.plot(range(1, 11), results[strategy], marker="o", label=strategy,
                 color=colors[strategy], linewidth=1.6)
    ax.set_xlabel("layer")
    ax.set_ylabel("activation std at that layer")
    ax.set_title("Signal propagation: activation std by layer, 4 init strategies")
    ax.legend()
    plt.tight_layout()
    plt.savefig("init_stats.png", dpi=110)
    plt.close()
    print("Saved: init_stats.png\n")
    return results


# ---------------------------------------------------------------------------
# 2. BATCH NORMALIZATION -- forward pass only, no training
# ---------------------------------------------------------------------------

def demo_batchnorm_stats():
    print("=" * 70)
    print("2. BATCH NORMALIZATION FIXES THE SAME SIGNAL, REGARDLESS OF INIT")
    print("=" * 70)
    print("Same 10-layer, 64-unit, tanh setup -- batch norm applied before each")
    print("layer's activation this time.\n")
    all_stds = {}
    for strategy in ["naive_small", "naive_large"]:
        stds_no_bn = propagate_signal(10, 64, strategy, use_bn=False)
        stds_bn = propagate_signal(10, 64, strategy, use_bn=True)
        all_stds[(strategy, False)] = stds_no_bn
        all_stds[(strategy, True)] = stds_bn
        print(f"{strategy} WITHOUT batch norm: {[f'{s:.4f}' for s in stds_no_bn]}")
        print(f"{strategy} WITH    batch norm: {[f'{s:.4f}' for s in stds_bn]}\n")
    print("With batch norm, BOTH the collapsed init and the saturated init settle")
    print("to the same stable ~0.63 std at every layer -- batch norm makes the")
    print("signal's health independent of the initialization choice.\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        ("naive_small", False): ("#a32020", "--", "naive_small, no BN"),
        ("naive_small", True): ("#a32020", "-", "naive_small, + BN"),
        ("naive_large", False): ("#b3651a", "--", "naive_large, no BN"),
        ("naive_large", True): ("#b3651a", "-", "naive_large, + BN"),
    }
    for key, stds in all_stds.items():
        color, linestyle, label = styles[key]
        ax.plot(range(1, 11), stds, marker="o", color=color, linestyle=linestyle,
                 label=label, linewidth=1.6)
    ax.set_xlabel("layer")
    ax.set_ylabel("activation std at that layer")
    ax.set_title("Batch norm rescues both a collapsed and a saturated init")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("batchnorm_stats.png", dpi=110)
    plt.close()
    print("Saved: batchnorm_stats.png\n")


# ---------------------------------------------------------------------------
# 3. THE DEEP NETWORK -- configurable init strategy and optional batch norm
# ---------------------------------------------------------------------------

class DeepNet:
    """A genuinely deep (6 hidden layer) tanh network with a softmax output,
    generalizing Day 39's SoftmaxNet from 1 hidden layer to N, with an
    optional batch-norm layer inserted before each hidden activation."""

    def __init__(self, layer_sizes, init="xavier", use_bn=False, lr=0.5, seed=42):
        rng = np.random.RandomState(seed)
        self.Ws, self.bs = [], []
        self.n_layers = len(layer_sizes) - 1
        for i in range(self.n_layers):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            scale = init_scale(init, n_in)
            self.Ws.append(rng.randn(n_in, n_out) * scale)
            self.bs.append(np.zeros(n_out))
        self.n_hidden = self.n_layers - 1
        self.use_bn = use_bn
        if use_bn:
            self.gammas = [np.ones(layer_sizes[i + 1]) for i in range(self.n_hidden)]
            self.betas = [np.zeros(layer_sizes[i + 1]) for i in range(self.n_hidden)]
        self.lr = lr

    def forward(self, X):
        self.cache = []
        a = X
        for i in range(self.n_hidden):
            z = a @ self.Ws[i] + self.bs[i]
            if self.use_bn:
                mu = z.mean(axis=0, keepdims=True)
                var = z.var(axis=0, keepdims=True)
                xhat = (z - mu) / np.sqrt(var + 1e-5)
                z = self.gammas[i] * xhat + self.betas[i]
                a_out = tanh(z)
                self.cache.append(dict(a_in=a, z=z, mu=mu, var=var, xhat=xhat, a_out=a_out))
            else:
                a_out = tanh(z)
                self.cache.append(dict(a_in=a, a_out=a_out))
            a = a_out
        z_out = a @ self.Ws[-1] + self.bs[-1]
        probs = softmax(z_out)
        self.cache.append(dict(a_in=a, probs=probs))
        return probs

    def backward(self, y_onehot):
        m = y_onehot.shape[0]
        out = self.cache[-1]
        dz = out["probs"] - y_onehot
        dWs, dbs = [None] * self.n_layers, [None] * self.n_layers
        dgammas = [None] * self.n_hidden if self.use_bn else None
        dbetas = [None] * self.n_hidden if self.use_bn else None

        dWs[-1] = out["a_in"].T @ dz / m
        dbs[-1] = dz.mean(axis=0)
        da = dz @ self.Ws[-1].T

        for i in reversed(range(self.n_hidden)):
            c = self.cache[i]
            dz_act = da * dtanh(c["a_out"])
            if self.use_bn:
                xhat, mu, var, z = c["xhat"], c["mu"], c["var"], c["z"]
                gamma = self.gammas[i]
                dgammas[i] = np.sum(dz_act * xhat, axis=0)
                dbetas[i] = np.sum(dz_act, axis=0)
                dxhat = dz_act * gamma
                std_inv = 1.0 / np.sqrt(var + 1e-5)
                dvar = np.sum(dxhat * (z - mu) * -0.5 * std_inv ** 3, axis=0)
                dmu = np.sum(dxhat * -std_inv, axis=0) + dvar * np.mean(-2 * (z - mu), axis=0)
                dz_lin = dxhat * std_inv + dvar * 2 * (z - mu) / m + dmu / m
            else:
                dz_lin = dz_act
            dWs[i] = c["a_in"].T @ dz_lin / m
            dbs[i] = dz_lin.mean(axis=0)
            da = dz_lin @ self.Ws[i].T

        for i in range(self.n_layers):
            self.Ws[i] -= self.lr * dWs[i]
            self.bs[i] -= self.lr * dbs[i]
        if self.use_bn:
            for i in range(self.n_hidden):
                self.gammas[i] -= self.lr * dgammas[i]
                self.betas[i] -= self.lr * dbetas[i]

    def train(self, X, y, epochs, num_classes):
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
# 4. THE REAL TEST -- training a 6-hidden-layer network, 4 inits, +/- BN
# ---------------------------------------------------------------------------

def make_dataset():
    return make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=1.8, random_state=42)


def demo_training_comparison():
    print("=" * 70)
    print("3. THE REAL TEST -- TRAINING A 6-HIDDEN-LAYER NETWORK")
    print("=" * 70)
    X, y = make_dataset()
    layer_sizes = [2] + [32] * 6 + [3]
    print(f"Architecture: {layer_sizes}  (6 hidden layers of 32 units, 3-class output)\n")

    results = {}
    for init in ["naive_small", "naive_large", "xavier", "he"]:
        net = DeepNet(layer_sizes, init=init, use_bn=False, lr=0.5, seed=42)
        losses = net.train(X, y, epochs=300, num_classes=3)
        acc = net.accuracy(X, y)
        results[init] = losses
        print(f"{init:14s} (no batch norm): loss[0]={losses[0]:.4f}  "
              f"loss[299]={losses[-1]:.4f}  accuracy={acc:.3f}")

    print()
    bn_results = {}
    for init in ["naive_small", "naive_large"]:
        net = DeepNet(layer_sizes, init=init, use_bn=True, lr=0.5, seed=42)
        losses = net.train(X, y, epochs=300, num_classes=3)
        acc = net.accuracy(X, y)
        bn_results[init] = losses
        print(f"{init:14s} (+ batch norm) : loss[0]={losses[0]:.4f}  "
              f"loss[299]={losses[-1]:.4f}  accuracy={acc:.3f}")

    ns_final = bn_results["naive_small"][-1]
    nl_final = bn_results["naive_large"][-1]
    print("\nnaive_small's loss is frozen at exactly ln(3)=1.0986 for all 300 epochs --")
    print("the network outputs a near-uniform 1/3 probability for every class on every")
    print("input the whole time. Its accuracy number is NOT a sign of partial learning --")
    print("it's an artifact of argmax breaking near-exact floating-point ties on a frozen,")
    print("functionally dead network. Adding batch norm to the SAME init rescues it")
    print(f"completely: loss drops to {ns_final:.4f} and accuracy becomes genuinely real.")
    print("naive_large, by contrast, was never dead (just rough and unstable at the")
    print("start) -- adding batch norm to it here does NOT clearly help within the same")
    print(f"300-epoch budget (final loss {nl_final:.4f} vs {results['naive_large'][-1]:.4f}")
    print("without batch norm) -- an honest reminder that batch norm's clearest win is")
    print("rescuing a catastrophically bad init, not improving an already-working one.\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"naive_small": "#a32020", "naive_large": "#b3651a",
              "xavier": "#378ADD", "he": "#5a2d8c"}
    for init, losses in results.items():
        axes[0].plot(losses, label=init, color=colors[init], linewidth=1.4)
    axes[0].set_title("Without batch norm")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 2)

    for init, losses in bn_results.items():
        axes[1].plot(losses, label=f"{init} + BN", color=colors[init], linewidth=1.4)
    axes[1].set_title("With batch norm")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cross-entropy loss")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig("training_comparison.png", dpi=110)
    plt.close()
    print("Saved: training_comparison.png\n")


def note_on_pytorch():
    print("=" * 70)
    print("WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)
    print("nn.init.xavier_uniform_(layer.weight)     # Xavier/Glorot, this lesson's 'xavier'")
    print("nn.init.kaiming_normal_(layer.weight,")
    print("                        nonlinearity='relu')  # He init, this lesson's 'he'")
    print("nn.BatchNorm1d(num_features)               # drop-in layer, replaces the")
    print("                                            # manual mu/var/xhat/gamma/beta code")
    print()
    print("PyTorch's default Linear layer initialization is already a Kaiming-family")
    print("scheme (not naive_small/naive_large) -- today's 'bad' initializations were")
    print("deliberately hand-picked to be bad, not what any framework does by default.\n")


def main():
    demo_init_stats()
    demo_batchnorm_stats()
    demo_training_comparison()
    note_on_pytorch()
    print("=" * 70)
    print("Day 40 complete. Depth turns a merely suboptimal weight scale into a total")
    print("training failure -- Xavier/He choose that scale systematically, and batch")
    print("normalization removes the dependency on getting it right at all.")
    print("=" * 70)


if __name__ == "__main__":
    main()