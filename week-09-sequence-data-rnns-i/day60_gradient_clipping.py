"""
Day 60: Gradient Clipping From Scratch -- An Honest Before/After on Day 59

Day 59 left two problems at T=20: gradient reaching early timesteps was
small at initialization, and the global gradient norm spiked (up to 47x
the median) as Whh's spectral radius crossed about 2. Gradient clipping
is the standard fix for the second: before each update, if the gradient's
global norm exceeds a threshold c, rescale the WHOLE gradient so its norm
is exactly c. Direction is kept; only the step length is capped.

    g_norm = sqrt(sum of squares of every gradient entry)
    if g_norm > c:   every gradient *= c / g_norm

Day 59 predicted clipping would tame the spikes but NOT rescue learning,
since clipping only shrinks gradients and cannot enlarge vanishing ones.
That prediction was wrong, and this lesson shows why:

  1. Clipping is built and checked: below the threshold it is an exact
     no-op; above it, the norm lands exactly on c and the direction is
     unchanged (cosine similarity 1.00000000). Element-wise value
     clipping, for contrast, bends the direction (cosine 0.76-0.90).
  2. Before/after on Day 59's exact T=20 run (unclipped reproduces Day
     59's 0.4950 exactly). Tracking test accuracy DURING training reveals
     what Day 59's end-of-training snapshot hid: the unclipped network
     LEARNED the task -- 0.925 test accuracy at step 820 -- then a single
     gradient of 31.4 (47x the median), applied at step 865 as a step of
     length 1.572, knocked it to 0.463, where it stayed. Clipped at 1.0
     (no step longer than 0.05), the same run finishes at 0.955.
  3. The rescue holds at every threshold tried, 0.5 through 10.
  4. Clipping also makes 4-20x larger learning rates usable at T=10:
     unclipped runs end at 0.49-0.69, clipped at 0.96-0.995.
  5. Its limit: at T=40, no threshold produces a network that learns and
     STAYS learned -- the edge of what a vanilla RNN can train reliably,
     and the motivation for the LSTM (Day 63).

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # render to files, not a screen
import matplotlib.pyplot as plt                         # the plotting API

np.random.seed(42)                                      # global seed, for any code that doesn't pass its own rng

# =============================================================================
# 0. SHARED HELPERS -- reused verbatim from Days 57-59
# =============================================================================
def softmax(z):                                         # logits -> probabilities, one row per example
    z_shifted = z - z.max(axis=1, keepdims=True)          # subtract the row max for numerical stability
    exp_z = np.exp(z_shifted)                             # exponentiate
    return exp_z / exp_z.sum(axis=1, keepdims=True)       # normalize each row to sum to 1


def one_hot(y, n_classes):                              # integer labels -> one-hot rows
    out = np.zeros((y.shape[0], n_classes))               # all-zero matrix
    out[np.arange(y.shape[0]), y] = 1                     # one 1 per row at the true class
    return out                                            # the one-hot matrix


def cross_entropy_loss(probs, y_onehot):                # mean cross-entropy over the batch
    return -np.sum(y_onehot * np.log(probs + 1e-9)) / probs.shape[0]  # -log(prob of true class), averaged


def he_scale(n_in):                                     # Gaussian init scale
    return np.sqrt(2.0 / n_in)                            # sqrt(2 / fan_in)


def rnn_forward(X, h0, Wxh, Whh, bh):                   # vanilla RNN forward pass -- verbatim from Day 57
    m, T = X.shape                                        # batch size, sequence length
    H = np.zeros((m, T, Whh.shape[0]))                    # every hidden state
    h = h0                                                 # initial state
    for t in range(T):                                     # one step at a time
        h = np.tanh(X[:, t:t + 1] @ Wxh + h @ Whh + bh)      # the recurrence, same weights every step
        H[:, t, :] = h                                       # store it
    return H                                               # all hidden states


def rnn_backward(dh_last, X, H, h0, Whh):                # BPTT -- verbatim from Day 58
    m, T = X.shape                                        # batch size, sequence length
    dWxh = np.zeros((1, Whh.shape[0]))                    # accumulators, one contribution per timestep
    dWhh = np.zeros_like(Whh)
    dbh = np.zeros(Whh.shape[0])
    dh = dh_last                                           # gradient arriving at the last state
    for t in reversed(range(T)):                           # backward through time
        h_prev = H[:, t - 1, :] if t > 0 else h0              # the state that fed step t
        da = dh * (1.0 - H[:, t, :] ** 2)                     # undo tanh
        dWxh += X[:, t:t + 1].T @ da / m                      # ACCUMULATE
        dWhh += h_prev.T @ da / m                             # ACCUMULATE
        dbh += da.sum(axis=0) / m                             # ACCUMULATE
        dh = da @ Whh.T                                       # back to h_{t-1}
    return dWxh, dWhh, dbh                                  # summed over every timestep


def make_sign_task(n, T, seed):                          # Day 59's task: is the FIRST input positive?
    rng = np.random.RandomState(seed)                     # private RNG
    X = rng.randn(n, T)                                    # every input N(0, 1)
    return X, (X[:, 0] > 0).astype(int)                    # label depends only on x_0


def spectral_radius(W):                                  # largest |eigenvalue| -- Day 59
    return np.max(np.abs(np.linalg.eigvals(W)))


def grad_ratio(net, X, y):                               # Day 59's probe, reduced to one number: ||dL/dh_0|| / ||dL/dh_{T-1}||
    net.forward(X)                                         # fills net.H and net.probs
    m, T = X.shape                                         # batch size, length
    dh = ((net.probs - one_hot(y, 2)) / m) @ net.Why.T     # true dL/dh_{T-1}
    last = np.linalg.norm(dh)                              # norm at the last step
    for t in reversed(range(T)):                           # walk back to step 0
        dh = (dh * (1.0 - net.H[:, t, :] ** 2)) @ net.Whh.T  # one step back; after t=0 this is dL/dh0's input side
        if t == 1:                                           # stop once dh holds dL/dh_0
            break
    return np.linalg.norm(dh) / last                       # how much of the gradient reaches step 0


# =============================================================================
# 1. GRADIENT CLIPPING -- the NEW piece today
# =============================================================================
def clip_by_global_norm(grads, max_norm):               # NEW today: cap the length of the whole gradient, keep its direction
    norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))  # global norm: every parameter's gradient treated as one long vector
    if max_norm is None or norm <= max_norm:               # no threshold, or already short enough
        return grads, norm, 1.0                              # return the ORIGINAL arrays untouched -- an exact no-op
    scale = max_norm / norm                                 # the single factor that makes the norm exactly max_norm
    return {k: g * scale for k, g in grads.items()}, norm, scale  # EVERY gradient scaled by the SAME factor -> direction preserved


def clip_by_value(grads, limit):                         # the ALTERNATIVE, for contrast: clip each entry to [-limit, limit] separately
    return {k: np.clip(g, -limit, limit) for k, g in grads.items()}  # large entries are cut, small ones untouched -> direction changes


def flatten(grads):                                      # concatenate every gradient into one vector, for norms and angles
    return np.concatenate([g.ravel() for g in grads.values()])


class ClippedRNNClassifier:                             # Day 58/59's classifier, with optional clipping before each update
    def __init__(self, hidden_dim=16, n_classes=2, lr=0.05, clip=None, seed=7):  # clip=None reproduces Day 59 exactly
        rng = np.random.RandomState(seed)                  # same init order as Days 58-59 -> identical starting weights
        self.hidden_dim, self.lr, self.clip = hidden_dim, lr, clip  # hidden size, learning rate, clipping threshold (or None)
        self.Wxh = rng.randn(1, hidden_dim) * he_scale(1)    # input-to-hidden weights
        self.Whh = rng.randn(hidden_dim, hidden_dim) * he_scale(hidden_dim)  # recurrent weights
        self.bh = np.zeros(hidden_dim)                       # hidden bias
        self.Why = rng.randn(hidden_dim, n_classes) * he_scale(hidden_dim)  # output weights
        self.by = np.zeros(n_classes)                        # output bias

    def forward(self, X):                                  # unchanged from Day 58
        m, T = X.shape                                       # batch size, sequence length
        self.X, self.h0 = X, np.zeros((m, self.hidden_dim))   # cache the input and the initial state for backward()
        self.H = rnn_forward(X, self.h0, self.Wxh, self.Whh, self.bh)  # every hidden state
        self.h_last = self.H[:, -1, :]                        # the final hidden state -- the classifier reads only this
        self.probs = softmax(self.h_last @ self.Why + self.by)  # class probabilities
        return self.probs                                     # hand back

    def backward(self, y_onehot):                          # Day 58's gradients, then the ONE new step: clip before updating
        m = y_onehot.shape[0]                                # batch size
        dlogits = self.probs - y_onehot                      # softmax + cross-entropy gradient
        grads = dict(Why=self.h_last.T @ dlogits / m, by=dlogits.sum(axis=0) / m)  # output-layer gradients
        grads["Wxh"], grads["Whh"], grads["bh"] = rnn_backward(dlogits @ self.Why.T, self.X, self.H, self.h0, self.Whh)  # BPTT
        grads, self.pre_clip_norm, self.scale = clip_by_global_norm(grads, self.clip)  # NEW: cap the step
        self.update_norm = self.lr * self.pre_clip_norm * self.scale  # length of the step actually taken
        for name, g in grads.items():                        # SGD update with the (possibly clipped) gradient
            getattr(self, name)[...] -= self.lr * g            # in place, so the stored arrays are updated

    def accuracy(self, X, y):                              # classification accuracy
        return (self.forward(X).argmax(axis=1) == y).mean()  # fraction correct


def train_tracked(net, X, y, X_te, y_te, epochs=150, batch_size=30, seed=7, eval_every=5):  # training + a record of everything
    rng = np.random.RandomState(seed)                      # identical shuffling to Days 58-59
    steps = dict(pre=[], update=[], clipped=[], loss=[])    # one entry per SGD step
    evals = dict(step=[], test=[], rho=[], ratio=[])        # one entry per evaluation
    step = 0                                                # running count of SGD steps
    for epoch in range(epochs):                             # full training budget
        order = rng.permutation(len(y))                       # fresh shuffle each epoch
        for start in range(0, len(order), batch_size):         # mini-batches
            idx = order[start:start + batch_size]                # this batch's indices
            y_oh = one_hot(y[idx], 2)                            # one-hot labels for this batch
            steps["loss"].append(cross_entropy_loss(net.forward(X[idx]), y_oh))  # batch loss before the update
            net.backward(y_oh)                                                    # gradients, clipping, update
            steps["pre"].append(net.pre_clip_norm)                                 # what BPTT produced
            steps["update"].append(net.update_norm)                                # what was actually applied
            steps["clipped"].append(net.scale < 1.0)                               # was this step clipped?
            step += 1                                                               # count it
        if epoch % eval_every == 0 or epoch == epochs - 1:                          # periodic evaluation
            evals["step"].append(step)                                               # when
            evals["test"].append(net.accuracy(X_te, y_te))                           # test accuracy NOW, not just at the end
            evals["rho"].append(spectral_radius(net.Whh))                            # Whh's spectral radius now
            evals["ratio"].append(grad_ratio(net, X_te, y_te))                       # how much gradient reaches step 0 now
    return {k: np.array(v) for k, v in steps.items()}, {k: np.array(v) for k, v in evals.items()}  # lists -> arrays


# =============================================================================
# 2. CHECKING THE CLIPPER
# =============================================================================
def demo_clipping_checks():
    print("=" * 70)
    print("1. CHECKING THE CLIPPER -- no-op below the threshold, exact norm and direction above it")
    print("=" * 70)
    print("Clipping introduces no new derivative, so there is nothing to gradient-check. "
          "What must be true instead: (a) below the threshold the gradient is returned "
          "unchanged; (b) above it the new global norm equals the threshold; (c) the "
          "direction is unchanged (cosine similarity 1). Element-wise value clipping is "
          "shown for contrast.")
    X, y = make_sign_task(30, 20, seed=42)                     # one real batch from Day 59's task
    net = ClippedRNNClassifier(clip=None, seed=7)              # Day 59's network
    net.forward(X)                                              # forward pass to populate the caches
    m = len(y); dl = net.probs - one_hot(y, 2)                  # batch size, and the softmax + cross-entropy gradient
    grads = dict(Why=net.h_last.T @ dl / m, by=dl.sum(axis=0) / m)  # output-layer gradients
    grads["Wxh"], grads["Whh"], grads["bh"] = rnn_backward(dl @ net.Why.T, net.X, net.H, net.h0, net.Whh)  # BPTT
    g0 = flatten(grads)                                         # the raw gradient, as one vector
    n0 = np.linalg.norm(g0)                                     # its global norm
    print(f"\nraw gradient: {g0.size} entries, global norm {n0:.4f}")

    print(f"\n{'threshold c':>12}{'norm after':>12}{'scale':>9}{'cosine':>11}{'unchanged?':>12}")
    for c in [10.0, n0 * 2, n0 * 0.5, 0.1]:                     # two thresholds above the norm, two below
        clipped, _, scale = clip_by_global_norm(grads, c)        # clip at this threshold
        g1 = flatten(clipped)                                    # clipped gradient as one vector
        cos = g0 @ g1 / (np.linalg.norm(g0) * np.linalg.norm(g1))  # cosine similarity: 1 means same direction
        same = all(clipped[k] is grads[k] for k in grads)       # identical objects -> exact no-op
        print(f"{c:>12.4f}{np.linalg.norm(g1):>12.4f}{scale:>9.4f}{cos:>11.8f}{str(same):>12}")
    print("Actual output from this lesson's run.")

    print(f"\ncontrast -- element-wise value clipping to [-limit, limit]:")
    print(f"\n{'limit':>8}{'norm after':>12}{'cosine':>11}{'entries changed':>17}")
    for limit in [0.05, 0.01, 0.002]:                           # three increasingly aggressive limits
        g1 = flatten(clip_by_value(grads, limit))                 # clip every entry separately
        cos = g0 @ g1 / (np.linalg.norm(g0) * np.linalg.norm(g1))  # direction check
        print(f"{limit:>8}{np.linalg.norm(g1):>12.4f}{cos:>11.6f}{np.sum(g0 != g1):>12} of {g0.size}")
    print("Actual output from this lesson's run.")


# =============================================================================
# 3. BEFORE/AFTER ON DAY 59'S EXACT RUN
# =============================================================================
def demo_before_after():
    print("\n" + "=" * 70)
    print("2. BEFORE/AFTER -- Day 59's exact T=20 run, unclipped vs. clipped at c=1.0")
    print("=" * 70)
    print("Identical data, initialization, learning rate (0.05), epochs (150), batch size (30) "
          "and shuffling as Day 59. The ONLY difference is the clipping threshold. Test "
          "accuracy, Whh's spectral radius, and the gradient ratio reaching step 0 are "
          "recorded every 5 epochs -- Day 59 only looked at the end.")
    T = 20                                                      # Day 59's failing length
    X_tr, y_tr = make_sign_task(600, T, seed=42)                 # Day 59's exact training data
    X_te, y_te = make_sign_task(400, T, seed=999)                # Day 59's exact test data
    runs = {}                                                    # label -> (per-step record, evaluations, final train accuracy)
    for label, c in [("unclipped", None), ("clip=1.0", 1.0)]:     # the before and the after
        net = ClippedRNNClassifier(lr=0.05, clip=c, seed=7)       # identical except for the threshold
        steps, ev = train_tracked(net, X_tr, y_tr, X_te, y_te)    # train, recording everything
        runs[label] = (steps, ev, net.accuracy(X_tr, y_tr))        # keep it

    print(f"\nconsistency check -- unclipped final test accuracy {runs['unclipped'][1]['test'][-1]:.4f} "
          f"(Day 59 reported 0.4950 for this exact run)")

    print(f"\n{'step':>6} | {'unclipped: test':>15}{'rho':>7}{'ratio':>10} | {'clip=1.0: test':>15}{'rho':>7}{'ratio':>10}")
    eu, ec = runs["unclipped"][1], runs["clip=1.0"][1]          # the two evaluation records
    for i in list(range(0, 20, 2)) + [len(eu["step"]) - 1]:       # the first 900 or so steps in detail, then the end
        print(f"{eu['step'][i]:>6} | {eu['test'][i]:>15.3f}{eu['rho'][i]:>7.2f}{eu['ratio'][i]:>10.2e} | "
              f"{ec['test'][i]:>15.3f}{ec['rho'][i]:>7.2f}{ec['ratio'][i]:>10.2e}")
    print("Actual output from this lesson's run.")

    print(f"\n{'':<24}{'unclipped':>12}{'clip=1.0':>12}")
    for name, f in [("peak test accuracy", lambda s, e, tr: f"{e['test'].max():.3f}"),
                    ("final test accuracy", lambda s, e, tr: f"{e['test'][-1]:.3f}"),
                    ("final train accuracy", lambda s, e, tr: f"{tr:.3f}"),
                    ("final spectral radius", lambda s, e, tr: f"{e['rho'][-1]:.2f}"),
                    ("largest step taken", lambda s, e, tr: f"{s['update'].max():.3f}"),
                    ("steps clipped", lambda s, e, tr: f"{s['clipped'].mean():.1%}")]:
        print(f"{name:<24}{f(*runs['unclipped']):>12}{f(*runs['clip=1.0']):>12}")
    print("Actual output from this lesson's run.")

    su = runs["unclipped"][0]                                    # the unclipped run's per-step record
    peak_i = int(np.argmax(eu["test"]))                           # evaluation with the best test accuracy
    crash = next(i for i in range(peak_i, len(eu["test"])) if eu["test"][i] < 0.6)  # first evaluation after the peak that fell below 0.6
    lo, hi = eu["step"][crash - 1], eu["step"][crash]             # the window of steps in which the collapse happened
    biggest = lo + int(np.argmax(su["pre"][lo:hi]))               # the step with the largest gradient in that window
    print(f"\nthe unclipped collapse: test accuracy {eu['test'][crash - 1]:.3f} at step {lo} -> "
          f"{eu['test'][crash]:.3f} at step {hi}; the largest gradient in that window was "
          f"{su['pre'][biggest]:.1f} at step {biggest} ({su['pre'][biggest] / np.median(su['pre']):.0f}x the run's median), "
          f"and it was applied as a step of length {su['update'][biggest]:.3f}")
    print("Actual output from this lesson's run.")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)  # top: accuracy over time; bottom: step lengths
    a1.plot(eu["step"], eu["test"], color="#d62728", marker=".", label="unclipped")  # the before
    a1.plot(ec["step"], ec["test"], color="#2ca02c", marker=".", label="clipped, c = 1.0")  # the after
    a1.axhline(0.5, color="gray", linestyle="--", linewidth=1)    # chance level
    a1.axvline(biggest, color="#d62728", alpha=0.35, linewidth=6, label=f"largest spike before the collapse (step {biggest})")  # the culprit
    a1.set_ylabel("test accuracy"); a1.set_ylim(0.4, 1.0); a1.legend(loc="center right", fontsize=8)  # labels
    a1.set_title("Day 59's T=20 run: unclipped learns, then a spike destroys it; clipped holds")  # title
    a2.plot(su["update"], color="#d62728", linewidth=0.6, label="unclipped")  # every step's length, unclipped
    a2.plot(runs["clip=1.0"][0]["update"], color="#2ca02c", linewidth=0.6, label="clipped (capped at lr x c = 0.05)")  # and clipped
    a2.set_yscale("log"); a2.set_ylabel("length of each step taken"); a2.set_xlabel("SGD step")  # log scale: lengths span decades
    a2.legend(loc="upper right", fontsize=8)                      # legend
    plt.tight_layout(); plt.savefig("clipping_before_after.png", dpi=110); plt.close()  # save and free
    print("\nsaved clipping_before_after.png")


# =============================================================================
# 4-5. HOW ROBUST, AND WHERE IT STOPS WORKING
# =============================================================================
def summarize(T, lr, clip):                              # one training run -> peak and final test accuracy, final radius
    X_tr, y_tr = make_sign_task(600, T, seed=42)          # same data seeds as every other run
    X_te, y_te = make_sign_task(400, T, seed=999)
    net = ClippedRNNClassifier(lr=lr, clip=clip, seed=7)  # same init seed
    _, ev = train_tracked(net, X_tr, y_tr, X_te, y_te)    # same 150 epochs, batch 30
    return ev["test"].max(), ev["test"][-1], ev["rho"][-1]  # peak, final, final spectral radius


def demo_robustness_and_limits():
    print("\n" + "=" * 70)
    print("3. HOW ROBUST? -- T=20, every clipping threshold")
    print("=" * 70)
    print("Same run as Section 2, sweeping the threshold c. Peak = best test accuracy at any "
          "evaluation; final = at the end of training.")
    rows = {}                                                    # (experiment, setting) -> numbers, for the summary plot
    print(f"\n{'c':>6}{'peak test':>11}{'final test':>12}{'final rho':>11}")
    for c in [None, 0.5, 1.0, 2.0, 5.0, 10.0]:                    # no clipping, then five thresholds
        pk, fn, rho = summarize(20, 0.05, c)                       # one full training run
        rows[("T20", c)] = (pk, fn)                                # keep peak and final
        print(f"{str(c):>6}{pk:>11.3f}{fn:>12.3f}{rho:>11.2f}")
    print("Actual output from this lesson's run.")

    print("\n" + "=" * 70)
    print("4. LARGER LEARNING RATES -- T=10, where Day 59's lr=0.05 run learned fine")
    print("=" * 70)
    print("A bigger learning rate means bigger steps, so exploding gradients do more damage. "
          "Each lr is trained unclipped and with c=1.0.")
    print(f"\n{'lr':>6}{'unclipped final':>17}{'clipped final':>15}{'unclipped rho':>15}{'clipped rho':>13}")
    for lr in [0.2, 0.5, 1.0]:                                     # 4x, 10x, and 20x Day 59's learning rate
        _, fu, ru = summarize(10, lr, None)                         # unclipped
        _, fc, rc = summarize(10, lr, 1.0)                          # clipped at 1.0
        rows[("T10", lr)] = (fu, fc)                                # keep both finals
        print(f"{lr:>6}{fu:>17.3f}{fc:>15.3f}{ru:>15.2f}{rc:>13.2f}")
    print("Actual output from this lesson's run.")

    print("\n" + "=" * 70)
    print("5. THE LIMIT -- T=40, every clipping threshold")
    print("=" * 70)
    print("Day 59's other failing length. Same sweep as Section 3.")
    print(f"\n{'c':>6}{'peak test':>11}{'final test':>12}{'final rho':>11}")
    for c in [None, 1.0, 2.0, 5.0, 10.0]:                         # no clipping, then four thresholds
        pk, fn, rho = summarize(40, 0.05, c)                       # one full training run
        rows[("T40", c)] = (pk, fn)                                # keep peak and final
        print(f"{str(c):>6}{pk:>11.3f}{fn:>12.3f}{rho:>11.2f}")
    print("Actual output from this lesson's run.")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)  # three panels, one per experiment
    for ax, key, cs, title in [(axes[0], "T20", [None, 0.5, 1.0, 2.0, 5.0, 10.0], "T=20: clipping rescues at every c"),
                               (axes[2], "T40", [None, 1.0, 2.0, 5.0, 10.0], "T=40: nothing holds")]:  # the two threshold sweeps
        x = np.arange(len(cs))                                          # bar positions
        ax.bar(x, [rows[(key, c)][1] for c in cs], color=["#d62728"] + ["#2ca02c"] * (len(cs) - 1), label="final")  # final accuracy
        ax.scatter(x, [rows[(key, c)][0] for c in cs], color="black", zorder=3, s=20, label="peak during training")  # peak accuracy
        ax.set_xticks(x); ax.set_xticklabels(["none" if c is None else str(c) for c in cs])  # threshold labels
        ax.set_xlabel("clip threshold c"); ax.set_title(title, fontsize=10)  # axis label and title
    lrs = [0.2, 0.5, 1.0]; x = np.arange(len(lrs))                   # the learning-rate panel
    axes[1].bar(x - 0.18, [rows[("T10", lr)][0] for lr in lrs], width=0.36, color="#d62728", label="unclipped")  # before
    axes[1].bar(x + 0.18, [rows[("T10", lr)][1] for lr in lrs], width=0.36, color="#2ca02c", label="clipped c=1.0")  # after
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"lr={lr}" for lr in lrs]); axes[1].set_title("T=10: clipping makes large lr usable", fontsize=10)
    for ax in axes:                                                    # shared styling
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1); ax.set_ylim(0, 1.05); ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("test accuracy")                                # y label once
    plt.tight_layout(); plt.savefig("clipping_limits.png", dpi=110); plt.close()  # save and free
    print("\nsaved clipping_limits.png")


def note_on_what_comes_next():
    print("\n" + "=" * 70)
    print("6. WHAT COMES NEXT")
    print("=" * 70)
    print("Clipping caps the length of a step; it cannot make a vanishing gradient larger. "
          "At T=20 that was enough, because the gradient reaching step 0 became healthy once "
          "training grew Whh's radius -- the danger was the crossing itself. At T=40 no "
          "threshold produced a network that learned and stayed learned. Fixing that needs a "
          "change to the recurrence itself, not to the optimizer: the LSTM's gated cell state "
          "(Day 63). Days 61-62 first put the vanilla RNN to work on real many-to-one and "
          "many-to-many tasks.")


def main():                                             # runs every section in order
    demo_clipping_checks()                                 # 1. the clipper behaves as specified
    demo_before_after()                                    # 2. Day 59's exact run, before and after
    demo_robustness_and_limits()                           # 3-5. threshold sweep, learning rates, the T=40 limit
    note_on_what_comes_next()                              # 6. bridge to Days 61-63


if __name__ == "__main__":                              # run only when executed directly
    main()                                                 # kick off the lesson
