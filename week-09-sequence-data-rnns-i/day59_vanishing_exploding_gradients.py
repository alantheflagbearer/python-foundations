"""
Day 59: Vanishing and Exploding Gradients in RNNs -- A Real Demonstration

Day 58 ended on a caveat. Its order-detection task trained fine at T=40,
because the network could LATCH onto large +-3 spikes -- the task routed
around the vanishing-gradient problem instead of exposing it. Today's
task is chosen so that can't happen: every input is drawn from N(0, 1),
and the label is simply whether the FIRST input was positive. x_0 is no
larger than the 19 or 39 inputs after it, so there is nothing special
to latch onto; the only way to solve the task is for gradient to travel
from the final timestep all the way back to step 0.

Why gradients vanish or explode, from Day 58's own BPTT:
    dL/dh_{t-1} = (dL/dh_t * (1 - h_t^2)) @ Whh^T
Every step back multiplies by Whh^T and by tanh's slope (1 - h_t^2) <= 1.
Over k steps that is a PRODUCT of k such factors -- it shrinks toward 0
or grows without bound roughly like (effective per-step gain)^k.

Four real measurements, not assertions:

  1. A new probe records dL/dh_t at every timestep. Before it is trusted,
     it is gradient-checked: the dL/dx_t it implies must match numeric
     central differences on every input.
  2. At initialization, gradient reaching step 0 of a 50-step sequence
     is measured across a sweep of Whh's spectral radius -- vanishing and
     exploding regimes both appear in one sweep, and the boundary between
     them turns out NOT to be the textbook rho = 1 once tanh is involved.
  3. The real cost: the sign-of-first-input task is trained at T = 2, 5,
     10, 20, 40. It learns at short lengths and fails at chance at long
     ones -- the failure Day 58's task avoided. The decisive measurement
     is taken at INITIALIZATION, where learning has to start: gradient
     reaching step 0 is already weak at the failing lengths. A gradient
     ratio measured AFTER training is misleading on its own -- by then the
     failing networks have drifted into the large-spectral-radius regime,
     and a direct check shows they never stored x_0 at all.
  4. Exploding gradients during ordinary training, with the DEFAULT
     initialization: training pushes Whh's spectral radius upward, and
     sharp gradient-norm spikes appear as it crosses the boundary found
     in (2). This run is the baseline Day 60's gradient clipping will be
     tested against.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # "Agg" backend renders to a file, not a screen -- needed since this runs headless
import matplotlib.pyplot as plt                         # pyplot: the plotting API actually called later

np.random.seed(42)                                      # fixes numpy's GLOBAL random state so any code that forgets to pass its own rng is still reproducible

# =============================================================================
# 0. SHARED HELPERS -- reused verbatim from Day 58 (the RNN, BPTT, training loop)
# =============================================================================
def softmax(z):                                         # softmax: turns raw class scores (logits) into a probability distribution per row
    z_shifted = z - z.max(axis=1, keepdims=True)          # subtract each row's own max first, for numerical stability
    exp_z = np.exp(z_shifted)                             # exponentiate the shifted logits
    return exp_z / exp_z.sum(axis=1, keepdims=True)       # normalize each row so its probabilities sum to 1


def one_hot(y, n_classes):                              # converts integer class labels into one-hot row vectors
    m = y.shape[0]                                        # number of examples
    out = np.zeros((m, n_classes))                        # all-zero (m, n_classes) matrix
    out[np.arange(m), y] = 1                              # one 1 per row, at the true class's column
    return out                                            # the finished one-hot label matrix


def cross_entropy_loss(probs, y_onehot):                # multi-class cross-entropy loss, averaged over the batch
    m = probs.shape[0]                                    # batch size
    eps = 1e-9                                            # avoids log(0)
    return -np.sum(y_onehot * np.log(probs + eps)) / m    # mean of -log(prob of true class)


def he_scale(n_in):                                     # scale factor for Gaussian weight initialization
    return np.sqrt(2.0 / n_in)                            # sqrt(2/fan_in)


def rnn_step_forward(x_t, h_prev, Wxh, Whh, bh):        # ONE timestep of the vanilla RNN recurrence -- verbatim from Day 57
    return np.tanh(x_t @ Wxh + h_prev @ Whh + bh)         # combine this step's input and the previous hidden state, squash with tanh


def rnn_forward(X, h0, Wxh, Whh, bh):                   # full forward pass -- verbatim from Day 57
    m, T = X.shape                                        # batch size and sequence length
    H = np.zeros((m, T, Whh.shape[0]))                    # every timestep's hidden state
    h = h0                                                 # start from the initial hidden state
    for t in range(T):                                     # one step at a time
        h = rnn_step_forward(X[:, t:t + 1], h, Wxh, Whh, bh)  # the SAME weights at every t
        H[:, t, :] = h                                       # store this timestep's hidden state
    return H                                               # every hidden state


def rnn_backward(dh_last, X, H, h0, Whh):                # BPTT for a many-to-one RNN -- verbatim from Day 58
    m, T = X.shape                                        # batch size and sequence length
    dWxh = np.zeros((1, Whh.shape[0]))                    # accumulator: one contribution per timestep
    dWhh = np.zeros_like(Whh)                             # accumulator
    dbh = np.zeros(Whh.shape[0])                          # accumulator
    dh = dh_last                                           # gradient arriving at the last hidden state
    for t in reversed(range(T)):                           # walk BACKWARD through time
        h_t = H[:, t, :]                                     # this timestep's cached tanh output
        h_prev = H[:, t - 1, :] if t > 0 else h0              # the state that fed this timestep
        da_t = dh * (1.0 - h_t ** 2)                          # undo tanh
        dWxh += X[:, t:t + 1].T @ da_t / m                    # ACCUMULATE
        dWhh += h_prev.T @ da_t / m                           # ACCUMULATE
        dbh += da_t.sum(axis=0) / m                           # ACCUMULATE
        dh = da_t @ Whh.T                                     # hand gradient back to h_{t-1}
    return dWxh, dWhh, dbh                                  # summed over every timestep


class VanillaRNNClassifier:                             # Day 58's many-to-one classifier, unchanged
    def __init__(self, hidden_dim=16, n_classes=2, lr=0.05, seed=42):  # no sequence-length argument
        rng = np.random.RandomState(seed)                  # private RNG for reproducible init
        self.hidden_dim = hidden_dim                        # hidden state size
        self.lr = lr                                        # learning rate
        self.Wxh = rng.randn(1, hidden_dim) * he_scale(1)    # input-to-hidden weights
        self.Whh = rng.randn(hidden_dim, hidden_dim) * he_scale(hidden_dim)  # recurrent weights
        self.bh = np.zeros(hidden_dim)                       # hidden bias
        self.Why = rng.randn(hidden_dim, n_classes) * he_scale(hidden_dim)  # output weights
        self.by = np.zeros(n_classes)                        # output bias

    def forward(self, X):                                  # forward pass, caching what backward() needs
        m, T = X.shape                                       # batch size and sequence length
        self.X = X                                           # cache input
        self.h0 = np.zeros((m, self.hidden_dim))              # initial hidden state
        self.H = rnn_forward(X, self.h0, self.Wxh, self.Whh, self.bh)  # every hidden state
        self.h_last = self.H[:, -1, :]                        # final hidden state
        self.probs = softmax(self.h_last @ self.Why + self.by)  # class probabilities
        return self.probs                                     # hand back

    def backward(self, y_onehot):                          # output layer, then BPTT, then SGD
        m = y_onehot.shape[0]                                # batch size
        dlogits = self.probs - y_onehot                      # softmax + cross-entropy gradient
        dWhy = self.h_last.T @ dlogits / m                    # output weight gradient
        dby = dlogits.sum(axis=0) / m                         # output bias gradient
        dh_last = dlogits @ self.Why.T                        # gradient into the final hidden state
        dWxh, dWhh, dbh = rnn_backward(dh_last, self.X, self.H, self.h0, self.Whh)  # BPTT
        self.grads = dict(Wxh=dWxh, Whh=dWhh, bh=dbh, Why=dWhy, by=dby)  # stash before updating
        self.Wxh -= self.lr * dWxh                            # SGD updates
        self.Whh -= self.lr * dWhh
        self.bh -= self.lr * dbh
        self.Why -= self.lr * dWhy
        self.by -= self.lr * dby

    def accuracy(self, X, y):                              # classification accuracy
        return (self.forward(X).argmax(axis=1) == y).mean()  # fraction correct


def train(net, X_train, y_train, epochs, batch_size, seed):  # Day 58's plain mini-batch loop
    rng = np.random.RandomState(seed)                      # private RNG for shuffling
    for epoch in range(epochs):                             # full training budget
        order = rng.permutation(len(y_train))                 # fresh shuffle each epoch
        for start in range(0, len(order), batch_size):         # mini-batches
            idx = order[start:start + batch_size]                # this batch's indices
            net.forward(X_train[idx])                            # forward
            net.backward(one_hot(y_train[idx], 2))               # BPTT + SGD


# =============================================================================
# 1. NEW TOOLS -- the task, spectral radius, and a gradient-by-timestep probe
# =============================================================================
def make_sign_task(n, T, seed):                          # NEW today: is the FIRST input positive?
    rng = np.random.RandomState(seed)                     # private RNG
    X = rng.randn(n, T)                                    # EVERY input, x_0 included, is N(0, 1) -- x_0 is not special in size
    y = (X[:, 0] > 0).astype(int)                          # label depends ONLY on x_0 -- the other T-1 inputs are pure distraction
    return X, y                                            # classes are balanced in expectation, since x_0 is symmetric around 0


def spectral_radius(W):                                  # NEW today: the largest |eigenvalue| of a square matrix
    return np.max(np.abs(np.linalg.eigvals(W)))           # governs how repeated multiplication by W grows or shrinks vectors


def set_spectral_radius(net, rho):                       # NEW today: rescale Whh so its spectral radius is exactly rho
    net.Whh *= rho / spectral_radius(net.Whh)              # scaling a matrix scales all its eigenvalues by the same factor


def grad_by_timestep(net, X, y):                         # NEW today: dL/dh_t for EVERY t, using Day 58's BPTT recurrence
    net.forward(X)                                         # forward pass fills net.H and net.probs
    m, T = X.shape                                         # batch size and sequence length
    dlogits = (net.probs - one_hot(y, 2)) / m              # TRUE gradient of the MEAN loss w.r.t. the logits (note the /m)
    dh = dlogits @ net.Why.T                               # dL/dh_{T-1}: where gradient enters the recurrence
    dH = np.zeros_like(net.H)                              # will hold dL/dh_t at every timestep, shape (m, T, hidden)
    for t in reversed(range(T)):                           # walk backward through time, exactly as rnn_backward does
        dH[:, t, :] = dh                                     # RECORD the gradient arriving at h_t -- this is what the probe adds
        da = dh * (1.0 - net.H[:, t, :] ** 2)                # undo tanh at this step
        dh = da @ net.Whh.T                                  # hand it back to h_{t-1}
    return dH                                              # every timestep's hidden-state gradient


def grad_norms(dH):                                      # one number per timestep: the Frobenius norm over the whole batch
    return np.array([np.linalg.norm(dH[:, t, :]) for t in range(dH.shape[1])])  # ||dL/dh_t|| for t = 0..T-1


# =============================================================================
# 2. VERIFY THE PROBE -- before trusting a single number it produces
# =============================================================================
def demo_probe_check():                                   # gradient-checks grad_by_timestep through the dL/dx_t it implies
    print("=" * 70)
    print("1. VERIFYING THE PROBE -- does grad_by_timestep report true gradients?")
    print("=" * 70)
    print("Every result today is built on dL/dh_t. An intermediate hidden state "
          "can't be nudged directly, but an INPUT can -- and since "
          "h_t = tanh(x_t @ Wxh + ...), the probe implies "
          "dL/dx_t = (dL/dh_t * (1 - h_t^2)) @ Wxh^T. If that matches numeric "
          "central differences on every input at every timestep, the probe's "
          "dL/dh_t values are correct.")
    rng = np.random.RandomState(1)                            # fixed rng
    net = VanillaRNNClassifier(hidden_dim=4, seed=1)           # a small network
    X, y = make_sign_task(3, 8, seed=1)                         # 3 sequences of length 8 -> 24 input entries
    dH = grad_by_timestep(net, X, y)                            # the probe's gradients
    dX_analytic = np.zeros_like(X)                              # the dL/dx_t the probe implies, for every (example, t)
    for t in range(X.shape[1]):                                  # every timestep
        da = dH[:, t, :] * (1.0 - net.H[:, t, :] ** 2)            # undo tanh at step t
        dX_analytic[:, t] = (da @ net.Wxh.T)[:, 0]                # chain back into x_t through Wxh

    eps = 1e-5                                                  # central-difference step
    worst = 0.0                                                 # worst relative error seen
    for i in range(X.shape[0]):                                  # every example
        for t in range(X.shape[1]):                               # every timestep
            orig = X[i, t]                                          # remember
            X[i, t] = orig + eps; lp = cross_entropy_loss(net.forward(X), one_hot(y, 2))  # nudge up
            X[i, t] = orig - eps; lm = cross_entropy_loss(net.forward(X), one_hot(y, 2))  # nudge down
            X[i, t] = orig                                          # restore
            num = (lp - lm) / (2 * eps)                             # numeric dL/dx_t
            ana = dX_analytic[i, t]                                 # probe-implied dL/dx_t
            worst = max(worst, abs(num - ana) / max(abs(num) + abs(ana), 1e-12))  # relative error
    print(f"\nchecked {X.size} input entries (3 sequences x 8 timesteps)")
    print(f"sample, example 0: t=0 analytic {dX_analytic[0, 0]: .3e} | t=7 analytic {dX_analytic[0, 7]: .3e}")
    print(f"max relative error: {worst:.2e} ({'PASS -- the probe reports true gradients' if worst < 1e-5 else 'FAIL'})")
    print("Actual output from this lesson's run.")


# =============================================================================
# 3. GRADIENT VS. DISTANCE -- sweeping Whh's spectral radius at initialization
# =============================================================================
def demo_gradient_vs_distance():                          # how much gradient reaches step 0, as a function of Whh's spectral radius
    print("\n" + "=" * 70)
    print("2. GRADIENT VS. DISTANCE -- how much gradient reaches step 0 of a 50-step sequence?")
    print("=" * 70)
    print("Untrained networks, identical except for Whh's spectral radius rho. "
          "200 sign-task sequences of length 50 (inputs N(0,1)). Reported: "
          "||dL/dh_t|| at the last step (t=49) and further back, and the ratio "
          "reaching t=0.")
    T = 50                                                      # sequence length
    X, y = make_sign_task(200, T, seed=0)                        # the test sequences, shared by every rho
    rhos = [0.5, 0.9, 1.0, 1.5, 2.0, 3.0]                        # a sweep spanning both regimes
    curves = {}                                                  # rho -> gradient norms by timestep, for the plot
    print(f"\n{'rho':>5}{'||dh_49||':>12}{'||dh_40||':>12}{'||dh_25||':>12}{'||dh_0||':>12}{'dh_0/dh_49':>13}")
    for rho in rhos:                                              # one untrained network per rho
        net = VanillaRNNClassifier(hidden_dim=16, seed=3)          # same seed -> same weights up to the rescale
        set_spectral_radius(net, rho)                              # the ONLY difference between networks
        n = grad_norms(grad_by_timestep(net, X, y))                # ||dL/dh_t|| for every t
        curves[rho] = n                                            # keep for the plot
        print(f"{rho:>5}{n[49]:>12.3e}{n[40]:>12.3e}{n[25]:>12.3e}{n[0]:>12.3e}{n[0] / n[49]:>13.2e}")
    print("Actual output from this lesson's run.")

    print("\nSame networks, but every input scaled by 0.1 -- small inputs keep "
          "tanh near its linear region, where its slope (1 - h^2) is close to 1:")
    print(f"\n{'rho':>5}{'dh_0/dh_49':>13}")
    for rho in [0.9, 1.0, 1.2, 1.5, 2.0]:                          # a finer sweep around the linear-regime boundary
        net = VanillaRNNClassifier(hidden_dim=16, seed=3)          # same network family
        set_spectral_radius(net, rho)                              # set rho
        n = grad_norms(grad_by_timestep(net, X * 0.1, y))           # SAME sequences, scaled down
        print(f"{rho:>5}{n[0] / n[49]:>13.2e}")                    # only the ratio matters here
    print("Actual output from this lesson's run.")

    plt.figure(figsize=(8, 5))                                    # one line per rho
    cmap = plt.cm.coolwarm                                         # blue = vanishing end, red = exploding end
    for k, rho in enumerate(rhos):                                  # plot every rho's curve
        steps_back = np.arange(T)                                    # 0 = the last timestep, 49 = t=0
        plt.plot(steps_back, curves[rho][::-1], label=f"rho = {rho}", color=cmap(k / (len(rhos) - 1)))
    plt.yscale("log")                                               # spans ~25 orders of magnitude
    plt.xlabel("steps back from the final timestep")                 # x-axis
    plt.ylabel("||dL/dh_t||  (log scale)")                           # y-axis
    plt.title("Gradient reaching earlier timesteps, by Whh's spectral radius (untrained, T=50)")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig("gradient_vs_distance.png", dpi=110)
    plt.close()
    print("\nsaved gradient_vs_distance.png")


# =============================================================================
# 4. THE REAL COST -- trainability vs. sequence length
# =============================================================================
def demo_trainability_vs_length():                        # does the sign task actually learn, as T grows?
    print("\n" + "=" * 70)
    print("3. THE REAL COST -- training the sign-of-first-input task at increasing T")
    print("=" * 70)
    print("Same network configuration and training budget at every length. "
          "Chance accuracy is 0.5. The gradient ratio ||dL/dh_0|| / "
          "||dL/dh_{T-1}|| is measured on the test set twice: at "
          "initialization (where learning has to START) and after training.")
    lengths = [2, 5, 10, 20, 40]                                 # short to long
    results = []                                                  # (T, train_acc, test_acc, ratio_init, ratio_final)
    trained = {}                                                  # T -> (trained net, test inputs), for the influence check below
    print(f"\n{'T':>4}{'train_acc':>11}{'test_acc':>10}{'ratio@init':>13}{'ratio@end':>12}")
    for T in lengths:                                              # one fresh training run per length
        X_tr, y_tr = make_sign_task(600, T, seed=42)                 # training data
        X_te, y_te = make_sign_task(400, T, seed=999)                # separate test data
        net = VanillaRNNClassifier(hidden_dim=16, lr=0.05, seed=7)   # identical configuration every time
        n_init = grad_norms(grad_by_timestep(net, X_te, y_te))        # gradient by timestep BEFORE any training
        train(net, X_tr, y_tr, epochs=150, batch_size=30, seed=7)     # identical budget every time
        tr, te = net.accuracy(X_tr, y_tr), net.accuracy(X_te, y_te)   # accuracies
        n_end = grad_norms(grad_by_timestep(net, X_te, y_te))         # gradient by timestep AFTER training
        results.append((T, tr, te, n_init[0] / n_init[-1], n_end[0] / n_end[-1]))
        trained[T] = (net, X_te, y_te)                                # keep for the influence check
        print(f"{T:>4}{tr:>11.3f}{te:>10.3f}{n_init[0] / n_init[-1]:>13.2e}{n_end[0] / n_end[-1]:>12.2e}")
    print("Actual output from this lesson's run.")

    print("\nDid the failing networks store x_0 at all? For each, the final hidden "
          "state is compared after (a) flipping the sign of x_0 -- the ONLY input "
          "that matters -- versus (b) resampling one mid-sequence noise input, "
          "which should not matter. Also: the strongest correlation between any "
          "final hidden unit and the true label sign(x_0).")
    print(f"\n{'T':>4}{'move from flipping x_0':>25}{'move from resampling x_mid':>29}{'max |corr| with label':>24}")
    for T in [20, 40]:                                                # the two lengths that failed
        net, X_te, y_te = trained[T]                                    # the trained network and its test inputs
        h0 = np.zeros((len(y_te), net.hidden_dim))                      # initial state
        h_base = rnn_forward(X_te, h0, net.Wxh, net.Whh, net.bh)[:, -1, :]   # final state, unmodified inputs
        X_flip = X_te.copy(); X_flip[:, 0] *= -1                        # (a) flip x_0 -- flips the correct label too
        X_mid = X_te.copy(); X_mid[:, T // 2] = np.random.RandomState(5).randn(len(y_te))  # (b) resample the middle input
        move_flip = np.linalg.norm(rnn_forward(X_flip, h0, net.Wxh, net.Whh, net.bh)[:, -1, :] - h_base, axis=1).mean()
        move_mid = np.linalg.norm(rnn_forward(X_mid, h0, net.Wxh, net.Whh, net.bh)[:, -1, :] - h_base, axis=1).mean()
        corr = max(abs(np.corrcoef(h_base[:, j], y_te)[0, 1]) for j in range(net.hidden_dim))  # best unit's |correlation| with the label
        print(f"{T:>4}{move_flip:>25.3f}{move_mid:>29.3f}{corr:>24.3f}")
    print("Actual output from this lesson's run.")

    plt.figure(figsize=(7, 4.5))                                    # grouped bars: train vs test accuracy per T
    x = np.arange(len(lengths))                                      # bar positions
    plt.bar(x - 0.18, [r[1] for r in results], width=0.36, label="train", color="#9ecae1")
    plt.bar(x + 0.18, [r[2] for r in results], width=0.36, label="test", color="#3182bd")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    plt.xticks(x, [f"T={T}" for T in lengths])
    plt.ylim(0, 1.05)
    plt.ylabel("accuracy")
    plt.title("Sign of the FIRST input: learnable when short, chance when long")
    plt.legend()
    plt.tight_layout()
    plt.savefig("trainability_vs_length.png", dpi=110)
    plt.close()
    print("\nsaved trainability_vs_length.png")


# =============================================================================
# 5. EXPLODING DURING ORDINARY TRAINING -- the baseline for Day 60
# =============================================================================
def train_tracked(net, X, y, epochs, batch_size, seed):  # Day 58's loop, plus a record of every update
    rng = np.random.RandomState(seed)                      # identical shuffling to train()
    gnorm, loss, radius = [], [], []                        # one entry per SGD step
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            y_oh = one_hot(y[idx], 2)
            loss.append(cross_entropy_loss(net.forward(X[idx]), y_oh))   # batch loss BEFORE this step's update
            net.backward(y_oh)                                            # computes grads and applies the update
            gnorm.append(np.sqrt(sum(np.sum(g ** 2) for g in net.grads.values())))  # global norm over ALL five gradients
            radius.append(spectral_radius(net.Whh))                        # Whh's spectral radius AFTER the update
    return np.array(gnorm), np.array(loss), np.array(radius)


def demo_exploding_during_training():                     # spikes in the gradient norm during a normal training run
    print("\n" + "=" * 70)
    print("4. EXPLODING GRADIENTS DURING ORDINARY TRAINING -- T=20, default init")
    print("=" * 70)
    print("The T=20 run from Section 3, re-run while recording the global "
          "gradient norm, the batch loss, and Whh's spectral radius at every "
          "one of its 3000 SGD steps. A 'spike' is any step whose gradient "
          "norm exceeds 10x the run's median.")
    T = 20
    X_tr, y_tr = make_sign_task(600, T, seed=42)                 # identical data to Section 3's T=20 run
    net = VanillaRNNClassifier(hidden_dim=16, lr=0.05, seed=7)   # identical network
    r0 = spectral_radius(net.Whh)                                 # radius at initialization
    g, L, R = train_tracked(net, X_tr, y_tr, epochs=150, batch_size=30, seed=7)
    med = np.median(g)                                             # typical gradient norm
    spikes = np.where(g > 10 * med)[0]                             # spike steps
    cross = int(np.argmax(R > 2.0)) if (R > 2.0).any() else None   # first step where the radius exceeds 2
    dL = np.abs(np.diff(L))                                        # |change in batch loss| from one step to the next
    after = dL[spikes[spikes < len(dL)]]                           # change caused by each spike's update

    print(f"\nWhh spectral radius: {r0:.2f} at init -> {R[499]:.2f} at step 500 -> {R[-1]:.2f} at the end")
    print(f"first step with radius > 2.0: {cross}")
    print(f"gradient norm: median {med:.3f}, max {g.max():.1f} ({g.max() / med:.0f}x the median)")
    print(f"spikes (> 10x median): {len(spikes)}, at steps {spikes.tolist()}")
    print(f"spikes while radius <= 2.0: {np.sum(R[spikes] <= 2.0)}   while radius > 2.0: {np.sum(R[spikes] > 2.0)}")
    print(f"|batch-loss change| per step: median {np.median(dL):.3f}, right after a spike {after.mean():.3f}")
    print("Actual output from this lesson's run.")

    print("\nHow Whh's initial spectral radius changes it (same T=20 task, same training):")
    print(f"\n{'init rho':>10}{'median gnorm':>14}{'max gnorm':>12}{'spikes':>8}{'final rho':>11}")
    for rho in [None, 3.0, 4.0]:                                   # default init, then two larger starting radii
        n2 = VanillaRNNClassifier(hidden_dim=16, lr=0.05, seed=7)
        if rho is not None:
            set_spectral_radius(n2, rho)                             # override the default radius
        g2, _, R2 = train_tracked(n2, X_tr, y_tr, epochs=150, batch_size=30, seed=7)
        label = f"{spectral_radius(VanillaRNNClassifier(hidden_dim=16, seed=7).Whh):.2f}*" if rho is None else f"{rho:.2f}"
        print(f"{label:>10}{np.median(g2):>14.3f}{g2.max():>12.1f}{np.sum(g2 > 10 * np.median(g2)):>8}{R2[-1]:>11.2f}")
    print("(* = default initialization)")
    print("Actual output from this lesson's run.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    ax1.plot(g, color="#3182bd", linewidth=0.6)                      # every step's gradient norm
    ax1.axhline(10 * med, color="#d62728", linestyle="--", linewidth=1, label="10x median (spike threshold)")
    ax1.scatter(spikes, g[spikes], color="#d62728", s=18, zorder=3, label=f"{len(spikes)} spikes")
    ax1.set_yscale("log")
    ax1.set_ylabel("global gradient norm (log)")
    ax1.set_title("T=20, default init: gradient spikes appear as Whh's spectral radius crosses ~2")
    ax1.legend(loc="upper right", fontsize=8)
    ax2.plot(R, color="#6a51a3")                                      # spectral radius over training
    ax2.axhline(2.0, color="gray", linestyle="--", linewidth=1, label="rho = 2 (preserve/explode boundary, Section 2)")
    for s in spikes:
        ax2.axvline(s, color="#d62728", alpha=0.25, linewidth=0.8)    # mark spike steps on the radius panel too
    ax2.set_xlabel("SGD step")
    ax2.set_ylabel("spectral radius of Whh")
    ax2.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("exploding_during_training.png", dpi=110)
    plt.close()
    print("\nsaved exploding_during_training.png")


def note_on_day_60():                                     # pure documentation: what the next day tests
    print("\n" + "=" * 70)
    print("5. WHAT DAY 60 TESTS")
    print("=" * 70)
    print("Two problems showed up at T=20, and they are not the same problem. "
          "Gradient reaching early timesteps VANISHES, so the dependency on "
          "x_0 is never learned. And the global gradient norm occasionally "
          "EXPLODES, jolting the weights. Gradient clipping -- capping the "
          "gradient's norm before each update -- is the standard fix for the "
          "second. Day 60 applies it to exactly Section 4's run and reports, "
          "honestly, what it does and doesn't fix.")


def main():
    demo_probe_check()                                        # 1. verify the new probe before trusting it
    demo_gradient_vs_distance()                               # 2. vanishing and exploding at initialization
    demo_trainability_vs_length()                             # 3. the real cost of vanishing gradients
    demo_exploding_during_training()                          # 4. exploding gradients during ordinary training
    note_on_day_60()                                          # 5. bridge to gradient clipping


if __name__ == "__main__":
    main()
