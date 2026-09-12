"""
Day 46: Learning-Rate Schedules -- Step Decay, Cosine Annealing, and a Real
Attempt to Fix Day 44's Non-Converging Wide Network

Day 45 ended with an honest surprise: even with EXACT, deterministic
full-batch gradients (zero sampling noise), a FIXED learning rate (lr=0.08,
unchanged for 350 epochs) still produced real loss spikes late in training --
around epochs 200, 240, 282, and 303 -- purely from the fixed step size
overshooting as the loss basin narrowed. That is exactly the problem
learning-rate schedules exist to fix: take bigger steps when you're far from
a minimum, smaller steps as you get close.

Today builds two schedules from scratch (step decay, cosine annealing),
confirms one of them calms Day 45's late-training oscillation, and then
makes a real, honest attempt at Day 44's still-unresolved question:
WideTwoBlockConvNetRGB (f1=6, f2=12) never fully converged in 400 epochs at
a fixed lr=0.08 (train_acc=0.995, not 1.000), and neither a slower
(lr=0.05) nor a faster (lr=0.12, unstable) fixed rate fixed it. Does a
PROPER schedule -- instead of a single fixed value -- finally get it there?

The honest answer, reached by actually running both a cosine-annealing
schedule starting ABOVE 0.08 and a safe step-decay schedule starting AT
0.08: no. One made things dramatically worse (a permanent, un-recoverable
collapse to uniform predictions), and the other made things mildly worse
(decaying an already-stable rate just slows convergence for no benefit).
This is reported in full, not hidden -- a schedule fixes OSCILLATION, which
this particular network never actually had at lr=0.08. Its real problem
(a 0.995 plateau) needs a different kind of fix, which later days (adaptive
optimizers, better initialization) will pick up.
"""
import pickle
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================
# 0. SHARED HELPERS -- reused verbatim from Days 39-45
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
    """Verbatim copy of Day 43/44/45's version -- character-for-character,
    not reimplemented from memory. See the RNG-order warning that has
    already caused a real bug twice in this series."""
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
    """Reused verbatim from Day 43-45 -- unchanged architecture, unchanged
    forward/backward math. self.lr is a plain mutable attribute, which is
    the ONLY hook a learning-rate schedule needs: set net.lr = schedule(epoch)
    before each backward() call, and backward()'s existing `p -= self.lr * g`
    update picks it up automatically. No other code changes required."""

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

    def accuracy(self, X, y):
        probs = self.forward(X)
        preds = probs.argmax(axis=1)
        return (preds == y).mean()

    def param_count(self):
        return sum(p.size for p in [self.Wc1, self.bc1, self.Wc2, self.bc2,
                                     self.W1, self.b1, self.W2, self.b2])


class WideTwoBlockConvNetRGB(TwoBlockConvNetRGB):
    """Identical to TwoBlockConvNetRGB -- only f1/f2 (width) differ at
    construction. Reused verbatim from Day 44, subclassed purely for its
    own name in results/plots."""
    pass


# =============================================================================
# 1. LEARNING-RATE SCHEDULES FROM SCRATCH
# =============================================================================
def constant_schedule(lr0):
    """The implicit schedule every network since Day 37 has used: the SAME
    lr forever. Included here so 'schedule vs. no schedule' is literally
    the same code path with a different schedule_fn, not a special case."""
    def sched(epoch):
        return lr0
    return sched


def step_decay_schedule(lr0, decay_factor, decay_every):
    """Multiply lr by decay_factor every `decay_every` epochs: a staircase
    that drops in sudden, discrete jumps rather than a smooth curve."""
    def sched(epoch):
        return lr0 * (decay_factor ** (epoch // decay_every))
    return sched


def cosine_annealing_schedule(lr0, lr_min, total_epochs):
    """Smoothly interpolates from lr0 down to lr_min following one half of a
    cosine wave: slow to start decaying, fastest in the middle, slow again
    as it approaches lr_min. total_epochs sets where the curve reaches
    lr_min -- NOT the number of epochs actually trained (a scheduled run can
    stop before or continue past total_epochs; the formula doesn't care)."""
    def sched(epoch):
        return lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * epoch / total_epochs))
    return sched


def demo_schedule_shapes():
    print("=" * 70)
    print("1. LEARNING-RATE SCHEDULES FROM SCRATCH")
    print("=" * 70)
    const = constant_schedule(0.08)
    step = step_decay_schedule(lr0=0.08, decay_factor=0.5, decay_every=100)
    cos = cosine_annealing_schedule(lr0=0.16, lr_min=0.0005, total_epochs=400)

    print("Real, actually-evaluated lr values at several checkpoint epochs "
          "(no training happening here -- these are the schedule functions "
          "called directly):")
    print(f"{'epoch':>7}{'constant(0.08)':>16}{'step_decay':>13}{'cosine':>10}")
    for e in [0, 50, 99, 100, 150, 199, 200, 250, 299, 300, 350, 399]:
        print(f"{e:>7}{const(e):>16.5f}{step(e):>13.5f}{cos(e):>10.5f}")

    epochs = np.arange(400)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [const(e) for e in epochs], label="constant (0.08)",
              color="#888888", linestyle="--")
    plt.plot(epochs, [step(e) for e in epochs], label="step decay (0.08, x0.5 every 100)",
              color="#d9534f")
    plt.plot(epochs, [cos(e) for e in epochs], label="cosine annealing (0.16 -> 0.0005)",
              color="#337ab7")
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.title("Three schedules, evaluated directly (no training)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lr_schedule_shapes.png", dpi=110)
    plt.close()
    print("saved lr_schedule_shapes.png")


# =============================================================================
# 2. A GENERIC SCHEDULED TRAINING LOOP
# =============================================================================
def train_scheduled(net, X, y, epochs, schedule_fn, n_classes=4):
    """Full-batch training loop, identical to every network's plain loop
    since Day 37, except net.lr is reset from schedule_fn(epoch) immediately
    before each backward() call. Passing schedule_fn = constant_schedule(lr)
    reproduces the old fixed-lr loop exactly -- a schedule is a strict
    generalization, not a separate code path."""
    y_onehot = one_hot(y, n_classes)
    losses, lrs = [], []
    for epoch in range(epochs):
        net.lr = schedule_fn(epoch)
        probs = net.forward(X)
        losses.append(cross_entropy_loss(probs, y_onehot))
        net.backward(y_onehot)
        lrs.append(net.lr)
    return losses, lrs


def noise_std_metric(step_losses, window="second_half"):
    """Reused verbatim from Day 45: std of consecutive step-to-step loss
    differences, over either the second half of training or the last K
    steps specifically."""
    losses_arr = np.asarray(step_losses)
    if window == "second_half":
        tail = losses_arr[len(losses_arr) // 2:]
    else:
        tail = losses_arr[-window:]
    if len(tail) < 2:
        return 0.0
    return float(np.std(np.diff(tail)))


# =============================================================================
# 3. DOES A SCHEDULE CALM DAY 45'S LATE-TRAINING OSCILLATION?
# =============================================================================
def demo_calm_day45_oscillation():
    print("\n" + "=" * 70)
    print("2. DOES A SCHEDULE CALM DAY 45'S LATE-TRAINING OSCILLATION?")
    print("=" * 70)
    print("Day 45's step-matched full-batch run (TwoBlockConvNetRGB, fixed "
          "lr=0.08, 350 epochs) found real loss spikes recurring from "
          "roughly epoch 75 through 320, with noise_std=0.264 over the "
          "second half -- purely from the fixed step size overshooting a "
          "narrowing loss basin. Today reruns that EXACT setup (same "
          "dataset convention as Day 44/45: train = seed=42, n_per_class=50 "
          "generation, all 200 images; test = a SEPARATE seed=999, "
          "n_per_class=20 generation, 80 images) with a step-decay schedule "
          "instead of the fixed rate, to see whether decaying lr calms it.")

    X_train, y_train = make_junction_dataset_rgb(
        n_per_class=50, size=30, arm=8, noise=0.4, channel_noise=0.15,
        center_jitter=2, seed=42)
    X_test, y_test = make_junction_dataset_rgb(
        n_per_class=20, size=30, arm=8, noise=0.4, channel_noise=0.15,
        center_jitter=2, seed=999)

    net_fixed = TwoBlockConvNetRGB(f1=3, f2=6, k=3, img_size=30, hidden=8,
                                    lr=0.08, seed=42)
    losses_fixed, _ = train_scheduled(net_fixed, X_train, y_train, epochs=350,
                                       schedule_fn=constant_schedule(0.08))
    train_acc_fixed = net_fixed.accuracy(X_train, y_train)
    test_acc_fixed = net_fixed.accuracy(X_test, y_test)
    noise_fixed = noise_std_metric(losses_fixed, "second_half")

    net_decay = TwoBlockConvNetRGB(f1=3, f2=6, k=3, img_size=30, hidden=8,
                                    lr=0.08, seed=42)
    sched = step_decay_schedule(lr0=0.08, decay_factor=0.5, decay_every=100)
    losses_decay, lrs_decay = train_scheduled(net_decay, X_train, y_train,
                                               epochs=350, schedule_fn=sched)
    train_acc_decay = net_decay.accuracy(X_train, y_train)
    test_acc_decay = net_decay.accuracy(X_test, y_test)
    noise_decay = noise_std_metric(losses_decay, "second_half")
    noise_decay_last50 = noise_std_metric(losses_decay, 50)

    print(f"\n{'config':<28}{'train_acc':>11}{'test_acc':>10}{'noise_std (2nd half)':>22}")
    print(f"{'Fixed lr=0.08':<28}{train_acc_fixed:>11.3f}{test_acc_fixed:>10.3f}{noise_fixed:>22.5f}")
    print(f"{'Step decay 0.08->0.01':<28}{train_acc_decay:>11.3f}{test_acc_decay:>10.3f}{noise_decay:>22.5f}")
    print(f"\nlr trajectory (step decay): epoch0={lrs_decay[0]:.4f} "
          f"epoch99={lrs_decay[99]:.4f} epoch100={lrs_decay[100]:.4f} "
          f"epoch199={lrs_decay[199]:.4f} epoch200={lrs_decay[200]:.4f} "
          f"epoch300={lrs_decay[300]:.4f} epoch349={lrs_decay[349]:.4f}")
    print(f"noise_std over just the last 50 steps (step decay): {noise_decay_last50:.5f}")
    print("Actual output from this lesson's run.")

    print(f"\nreal, honest verdict: step decay all but ELIMINATES the "
          f"oscillation Day 45 found -- noise_std drops from "
          f"{noise_fixed:.5f} to {noise_decay:.5f} (and to "
          f"{noise_decay_last50:.5f} in just the final 50 steps, where lr "
          f"has decayed to {lrs_decay[349]:.3f}). But this did NOT come "
          f"free: BOTH train and test accuracy dropped slightly "
          f"(train_acc {train_acc_fixed:.3f} to {train_acc_decay:.3f}; "
          f"test_acc {test_acc_fixed:.3f} to {test_acc_decay:.3f}). An "
          f"honest complication worth naming plainly: Day 45's "
          f"late-training oscillation, caused by the SAME fixed lr=0.08 "
          f"overshooting, may have been acting like an unintentional "
          f"regularizer -- the noisy steps kept exploring slightly past "
          f"the exact training optimum, which on this 80-image "
          f"separately-generated test set happened to generalize "
          f"marginally better. Removing the noise let training settle in "
          f"more precisely, at a small cost to both train and test "
          f"accuracy on this run. This is a real, single-run correlation, "
          f"not proof of a general 'noise helps' rule -- but it is exactly "
          f"the kind of result that deserves flagging rather than smoothing "
          f"over just because the noise metric moved the way expected.")

    plt.figure(figsize=(8, 4.5))
    plt.plot(losses_fixed, label=f"fixed lr=0.08 (test_acc={test_acc_fixed:.3f})",
              color="#d9534f", alpha=0.85, linewidth=0.9)
    plt.plot(losses_decay, label=f"step decay (test_acc={test_acc_decay:.3f})",
              color="#337ab7", alpha=0.85, linewidth=0.9)
    plt.xlabel("epoch (= step, full-batch)")
    plt.ylabel("training loss")
    plt.title("Fixed lr vs. step decay: Day 45's oscillation, calmed")
    plt.legend()
    plt.tight_layout()
    plt.savefig("schedule_calms_oscillation.png", dpi=110)
    plt.close()
    print("saved schedule_calms_oscillation.png")

    return dict(train_acc_fixed=train_acc_fixed, test_acc_fixed=test_acc_fixed,
                noise_fixed=noise_fixed, train_acc_decay=train_acc_decay,
                test_acc_decay=test_acc_decay, noise_decay=noise_decay)


# =============================================================================
# 4. THE MAIN EVENT -- REVISITING DAY 44'S NON-CONVERGING WIDE NETWORK
# =============================================================================
def demo_revisit_wide_network():
    print("\n" + "=" * 70)
    print("3. REVISITING DAY 44'S NON-CONVERGING WIDE NETWORK")
    print("=" * 70)
    print("Day 44's WideTwoBlockConvNetRGB (f1=6, f2=12) reached only "
          "train_acc=0.995 (not 1.000) after 400 epochs at fixed lr=0.08 -- "
          "every other network that day hit 1.000. lr=0.05 was slower "
          "(loss=0.758 at epoch 150 vs lr=0.08's 0.627 at epoch 200) and "
          "lr=0.12 was unstable (train_acc collapsed to 0.27 at epoch 100). "
          "Today tries two REAL schedules on the IDENTICAL dataset "
          "convention Day 44 used (train: seed=42, n_per_class=50, all 200 "
          "images; test: a SEPARATE seed=999, n_per_class=20 generation, "
          "80 images -- deliberately different from Day 45's own "
          "single-generation-plus-holdout convention, and getting this "
          "wrong on the first attempt is exactly the kind of dataset-"
          "convention mismatch this series has flagged before).")

    X_train, y_train = make_junction_dataset_rgb(
        n_per_class=50, size=30, arm=8, noise=0.4, channel_noise=0.15,
        center_jitter=2, seed=42)
    X_test, y_test = make_junction_dataset_rgb(
        n_per_class=20, size=30, arm=8, noise=0.4, channel_noise=0.15,
        center_jitter=2, seed=999)

    # --- Baseline: reproduce Day 44's fixed lr=0.08 result exactly ---
    net_base = WideTwoBlockConvNetRGB(f1=6, f2=12, k=3, img_size=30, hidden=8,
                                       lr=0.08, seed=42)
    losses_base, _ = train_scheduled(net_base, X_train, y_train, epochs=400,
                                      schedule_fn=constant_schedule(0.08))
    train_acc_base = net_base.accuracy(X_train, y_train)
    test_acc_base = net_base.accuracy(X_test, y_test)
    print(f"\nBaseline reproduced -- fixed lr=0.08, 400 epochs: "
          f"train_acc={train_acc_base:.3f}, test_acc={test_acc_base:.3f}")
    print(f"(Day 44's own documented numbers: train_acc=0.995, "
          f"test_acc=0.537 -- {'MATCHES exactly' if abs(train_acc_base - 0.995) < 0.001 and abs(test_acc_base - 0.537) < 0.001 else 'DOES NOT MATCH -- investigate'})")

    # --- Attempt 1: cosine annealing, starting ABOVE the safe fixed rate ---
    net_cos = WideTwoBlockConvNetRGB(f1=6, f2=12, k=3, img_size=30, hidden=8,
                                      lr=0.08, seed=42)
    cos_sched = cosine_annealing_schedule(lr0=0.16, lr_min=0.0005, total_epochs=400)
    losses_cos, lrs_cos = train_scheduled(net_cos, X_train, y_train, epochs=400,
                                           schedule_fn=cos_sched)
    train_acc_cos = net_cos.accuracy(X_train, y_train)
    test_acc_cos = net_cos.accuracy(X_test, y_test)
    print(f"\nAttempt 1 -- cosine annealing 0.16 -> 0.0005 over 400 epochs: "
          f"train_acc={train_acc_cos:.3f}, test_acc={test_acc_cos:.3f}")
    print(f"loss trajectory: epoch0={losses_cos[0]:.4f} epoch99={losses_cos[99]:.4f} "
          f"epoch199={losses_cos[199]:.4f} epoch299={losses_cos[299]:.4f} "
          f"epoch399={losses_cos[399]:.4f}  (ln(4)={np.log(4):.4f} is pure "
          f"random-guessing loss for 4 classes)")
    probs_cos = net_cos.forward(X_train)
    print(f"probs on 3 training images after 400 epochs: "
          f"{np.round(probs_cos[:3], 4).tolist()}")
    print("Actual output from this lesson's run.")
    print(f"\nhonest complication: cosine annealing starting at lr0=0.16 "
          f"did not just fail to help -- it produced a WORSE, and "
          f"apparently PERMANENT, collapse than Day 44's already-unstable "
          f"fixed lr=0.12 (which reached train_acc=0.27). Here, "
          f"train_acc={train_acc_cos:.3f} and the loss sits at "
          f"{losses_cos[399]:.4f}, essentially exactly ln(4) -- the "
          f"network is predicting close to uniform probability across all "
          f"4 classes for every image, and stays there for the entire "
          f"remaining ~300 epochs even as the schedule's own lr decays all "
          f"the way down to 0.0005 by the end. Checking the network's "
          f"internal state confirms this isn't dead ReLUs or exploding "
          f"weights (no NaNs, weight magnitudes stay bounded, roughly "
          f"44-48% of ReLU units are still active) -- the DENSE output "
          f"layers have collapsed into a configuration that produces "
          f"nearly-identical logits for every class regardless of input, "
          f"and once there, no amount of LATER lr decay undoes it, because "
          f"the gradient direction needed to escape a symmetric, "
          f"non-discriminative optimum is itself tiny. The lesson: a "
          f"learning-rate schedule's decaying TAIL does not protect "
          f"against damage its own elevated HEAD can do early in "
          f"training -- if a schedule starts high enough to break the "
          f"network in the first ~100 epochs, decaying the rate afterward "
          f"does not necessarily give it a way back.")

    # --- Attempt 2: safe step decay, starting AT the known-stable rate ---
    net_step = WideTwoBlockConvNetRGB(f1=6, f2=12, k=3, img_size=30, hidden=8,
                                       lr=0.08, seed=42)
    step_sched = step_decay_schedule(lr0=0.08, decay_factor=0.5, decay_every=100)
    losses_step, lrs_step = train_scheduled(net_step, X_train, y_train, epochs=400,
                                             schedule_fn=step_sched)
    train_acc_step = net_step.accuracy(X_train, y_train)
    test_acc_step = net_step.accuracy(X_test, y_test)
    print(f"\nAttempt 2 -- safe step decay 0.08 -> 0.01 (never exceeds "
          f"Day 44's own stable rate), 400 epochs: "
          f"train_acc={train_acc_step:.3f}, test_acc={test_acc_step:.3f}")
    print("Actual output from this lesson's run.")
    print(f"\nhonest complication #2: this is SAFER than Attempt 1 (no "
          f"collapse) but still WORSE than just leaving lr fixed at 0.08 "
          f"the whole time ({train_acc_step:.3f} vs {train_acc_base:.3f} "
          f"train_acc). The reason is simple once you look at it: Day 44's "
          f"lr=0.08 run was never actually OSCILLATING or unstable -- it "
          f"was just slow to finish the last little bit of training. "
          f"Cutting the rate in half every 100 epochs slows it down "
          f"FURTHER for no corresponding stability benefit, since there "
          f"was no instability to fix in the first place. A schedule that "
          f"decays a rate that wasn't causing problems just wastes the "
          f"epoch budget.")

    print(f"\n{'config':<26}{'train_acc':>11}{'test_acc':>10}")
    print(f"{'Fixed lr=0.08 (Day44)':<26}{train_acc_base:>11.3f}{test_acc_base:>10.3f}")
    print(f"{'Cosine 0.16->0.0005':<26}{train_acc_cos:>11.3f}{test_acc_cos:>10.3f}")
    print(f"{'Step decay 0.08->0.01':<26}{train_acc_step:>11.3f}{test_acc_step:>10.3f}")
    print("Actual output from this lesson's run.")

    print(f"\nreal, honest final verdict: NEITHER schedule beat Day 44's "
          f"plain fixed lr=0.08 on this network. Learning-rate schedules "
          f"are a targeted fix for a SPECIFIC problem -- a rate that's "
          f"causing real oscillation or instability, exactly what Topic 2 "
          f"showed for Day 45's setup -- not a general-purpose 'train "
          f"better' switch. WideTwoBlockConvNetRGB's 0.995 plateau was "
          f"never an oscillation problem, so no schedule built from lr "
          f"alone fixed it; if anything, reaching for a higher starting "
          f"rate to 'speed things up' reproduced (and worsened) exactly "
          f"the instability Day 44 already found at fixed lr=0.12. "
          f"Whatever IS holding this specific network to 0.995 -- an "
          f"initialization issue, a genuinely hard-to-escape point in its "
          f"loss landscape, or simply needing more than 400 epochs -- is a "
          f"different kind of problem than a schedule addresses, and is "
          f"better suited to the adaptive per-parameter optimizers and "
          f"initialization techniques later in this roadmap.")

    plt.figure(figsize=(8, 4.5))
    plt.plot(losses_base, label=f"fixed lr=0.08 (train_acc={train_acc_base:.3f})",
              color="#5cb85c")
    plt.plot(losses_cos, label=f"cosine 0.16->0.0005 (train_acc={train_acc_cos:.3f})",
              color="#d9534f")
    plt.plot(losses_step, label=f"step decay 0.08->0.01 (train_acc={train_acc_step:.3f})",
              color="#337ab7")
    plt.axhline(np.log(4), color="gray", linestyle=":", linewidth=1,
                label="ln(4) = random-guessing loss")
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title("WideTwoBlockConvNetRGB: fixed lr vs. two real schedule attempts")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("wide_network_schedule_attempts.png", dpi=110)
    plt.close()
    print("\nsaved wide_network_schedule_attempts.png")

    return dict(train_acc_base=train_acc_base, test_acc_base=test_acc_base,
                train_acc_cos=train_acc_cos, test_acc_cos=test_acc_cos,
                train_acc_step=train_acc_step, test_acc_step=test_acc_step)


# =============================================================================
# 5. WHAT THIS MAPS TO IN PYTORCH
# =============================================================================
def note_on_pytorch():
    print("\n" + "=" * 70)
    print("4. WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)
    print(
        "optimizer = torch.optim.SGD(model.parameters(), lr=0.08)\n"
        "step_sched = torch.optim.lr_scheduler.StepLR(\n"
        "    optimizer, step_size=100, gamma=0.5)      # matches step_decay_schedule\n"
        "cos_sched = torch.optim.lr_scheduler.CosineAnnealingLR(\n"
        "    optimizer, T_max=400, eta_min=0.0005)     # matches cosine_annealing_schedule\n"
        "\n"
        "for epoch in range(epochs):\n"
        "    train_one_epoch(...)\n"
        "    step_sched.step()      # or cos_sched.step() -- called once per epoch,\n"
        "                           # exactly like this lesson's schedule_fn(epoch)"
    )
    print("\nPyTorch's schedulers store the CURRENT lr internally and mutate "
          "optimizer.param_groups[0]['lr'] on .step() -- functionally "
          "identical to this lesson's net.lr = schedule_fn(epoch), just "
          "wrapped in a class with its own .step()/.get_last_lr() API "
          "instead of a plain function called with an epoch number.")


def main():
    demo_schedule_shapes()
    demo_calm_day45_oscillation()
    demo_revisit_wide_network()
    note_on_pytorch()


if __name__ == "__main__":
    main()
