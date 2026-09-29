"""
Day 58: RNN Backward Pass -- Backpropagation Through Time (BPTT) From Scratch

Day 57 built the vanilla RNN's forward pass and verified it without any
gradients (a hand-unrolled correctness check, a variable-length check,
and a finite-difference sensitivity check). Today builds the backward
pass -- the first time this network can actually LEARN anything -- and
verifies it the same way every backward pass in this series has been
verified since Day 39: a real numeric gradient check.

The one idea that makes BPTT different from ordinary backprop: the SAME
Wxh, Whh, bh are reused at every timestep, so their gradients cannot be
computed once and done -- each timestep contributes its own piece, and
all T pieces must be SUMMED. Going backward from t = T-1 down to t = 0:

    da_t    = dh_t * (1 - h_t^2)            # undo tanh at this timestep
    dWxh   += x_t^T  @ da_t                  # ACCUMULATE, not overwrite
    dWhh   += h_{t-1}^T @ da_t               # ACCUMULATE, not overwrite
    dbh    += sum(da_t)                      # ACCUMULATE, not overwrite
    dh_{t-1} = da_t @ Whh^T                  # hand gradient to the PREVIOUS timestep

That last line is the "through time" part: gradient flows from h_t back
into h_{t-1} through the very same Whh the forward pass used to go the
other direction.

Real results. The gradient check passes on every one of the five
parameters (max relative error ~2e-08). Trained on Day 56's task at T=10,
the RNN goes from 0.395 test accuracy (random weights) to 1.000.

Then a prediction that turned out to be wrong. Day 57 measured the
forward-pass sensitivity of h_T to x_0 collapsing by T=40 and framed it
as a preview of vanishing gradients -- so T=40 "should" struggle. It
doesn't: identical training reaches ~0.99 test accuracy, even when the
first spike must be remembered 30-39 steps. The diagnosis, measured
rather than guessed: after training, sensitivity to a TINY nudge at x_0
DROPS ~1000x, while the effect of a full-size +3 spike at x_0 GROWS ~12x.
The network learned to ignore small noise and LATCH onto large spikes --
about a fifth of its final hidden units sit pinned near tanh's +-1
limits, and Whh's largest eigenvalue grows above 1, which tanh's bound
turns into stable saturated memory states instead of explosion. Day 57's
tiny-epsilon derivative is a local, linear measurement; latching is a
large-signal, nonlinear effect it cannot see. Vanishing gradients are
still real -- this particular task just happens to route around them,
which is exactly why Day 59 needs a task that doesn't.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # "Agg" backend renders to a file, not a screen -- needed since this runs headless
import matplotlib.pyplot as plt                         # pyplot: the plotting API actually called later (plt.plot, plt.savefig, ...)

np.random.seed(42)                                      # fixes numpy's GLOBAL random state so any code that forgets to pass its own rng is still reproducible

# =============================================================================
# 0. SHARED HELPERS -- reused verbatim from Days 56-57 (plus one_hot and
#    cross_entropy_loss, needed again now that training exists)
# =============================================================================
def softmax(z):                                         # softmax: turns raw class scores (logits) into a probability distribution per row
    z_shifted = z - z.max(axis=1, keepdims=True)          # subtract each row's own max first, for numerical stability (avoids exp() overflow)
    exp_z = np.exp(z_shifted)                             # exponentiate the shifted logits
    return exp_z / exp_z.sum(axis=1, keepdims=True)       # normalize each row so its probabilities sum to 1


def one_hot(y, n_classes):                              # converts integer class labels into one-hot row vectors
    m = y.shape[0]                                        # m = number of examples (rows) in y
    out = np.zeros((m, n_classes))                        # start with an all-zero (m, n_classes) matrix
    out[np.arange(m), y] = 1                              # set exactly one 1 per row, at the column given by that row's true class
    return out                                            # the finished one-hot label matrix


def cross_entropy_loss(probs, y_onehot):                # multi-class cross-entropy loss, averaged over the batch
    m = probs.shape[0]                                    # m = batch size (number of rows)
    eps = 1e-9                                            # tiny constant to avoid log(0) if a predicted probability is exactly 0
    return -np.sum(y_onehot * np.log(probs + eps)) / m    # -sum(true * log(pred)) per row, averaged across the m rows


def he_scale(n_in):                                     # He initialization scale factor -- used loosely here even though tanh (not ReLU) is the activation
    return np.sqrt(2.0 / n_in)                            # sqrt(2/fan_in) -- keeps initial activations from being too large or too small


def make_order_dataset(n_per_class, T, pos_lo, pos_hi, seed):  # Day 56's order-detection task, unchanged: does the high spike come before the low spike?
    rng = np.random.RandomState(seed)                     # this function's own private RNG, independent of the global numpy state
    X, y = [], []                                          # accumulators for generated sequences and their binary labels
    for label in [0, 1]:                                   # loop over both classes, generating n_per_class examples of each
        for _ in range(n_per_class):                        # generate n_per_class sequences for this label
            seq = rng.randn(T) * 0.3                         # start from a sequence of small Gaussian noise, length T
            p1, p2 = rng.choice(np.arange(pos_lo, pos_hi + 1), size=2, replace=False)  # two DISTINCT positions, drawn from [pos_lo, pos_hi]
            p_high, p_low = (p1, p2) if p1 < p2 else (p2, p1)  # sort them so p_high is temporally EARLIER, p_low is LATER, by construction
            if label == 0:                                  # for label 0, swap which spike goes early vs late
                p_high, p_low = p_low, p_high                # low spike now comes FIRST, high spike comes LATER -- the "wrong" order for label 1
            seq[p_high] += 3.0                               # place the HIGH spike (+3) at its assigned position
            seq[p_low] -= 3.0                                # place the LOW spike (-3) at its assigned position
            X.append(seq)                                    # store this generated sequence
            y.append(label)                                  # store its true label (1 = high-before-low, 0 = low-before-high)
    X = np.array(X)                                        # stack the list of sequences into one (N, T) array
    y = np.array(y)                                        # stack the list of labels into one (N,) array
    idx = rng.permutation(len(y))                           # a random shuffle order for the whole dataset
    return X[idx], y[idx]                                  # return the shuffled sequences and labels together, so pairing is preserved


def rnn_step_forward(x_t, h_prev, Wxh, Whh, bh):        # ONE timestep of the vanilla RNN recurrence -- verbatim from Day 57
    return np.tanh(x_t @ Wxh + h_prev @ Whh + bh)         # combine this step's input AND the previous hidden state, squash with tanh


def rnn_forward(X, h0, Wxh, Whh, bh):                   # runs rnn_step_forward once per timestep -- verbatim from Day 57
    m, T = X.shape                                        # batch size and sequence length
    hidden_dim = Whh.shape[0]                             # hidden state size, read off Whh's own shape
    H = np.zeros((m, T, hidden_dim))                      # will hold every timestep's hidden state, shape (batch, time, hidden)
    h = h0                                                 # start from the given initial hidden state
    for t in range(T):                                     # walk through the sequence ONE STEP AT A TIME
        x_t = X[:, t:t + 1]                                  # this timestep's input, kept 2D as (batch, 1)
        h = rnn_step_forward(x_t, h, Wxh, Whh, bh)           # the SAME weights are reused at every t
        H[:, t, :] = h                                       # store this timestep's hidden state
    return H                                               # every hidden state, from t=0 to t=T-1


# =============================================================================
# 1. BACKPROPAGATION THROUGH TIME -- the NEW piece today
# =============================================================================
def rnn_backward(dh_last, X, H, h0, Whh):                # NEW today: BPTT for a many-to-one RNN -- gradient enters ONLY at the last hidden state
    m, T = X.shape                                        # batch size and sequence length, same as the forward pass saw
    dWxh = np.zeros((1, Whh.shape[0]))                    # accumulator for Wxh's gradient -- starts at zero, receives one contribution PER timestep
    dWhh = np.zeros_like(Whh)                             # accumulator for Whh's gradient -- same idea
    dbh = np.zeros(Whh.shape[0])                          # accumulator for bh's gradient -- same idea
    dh = dh_last                                           # gradient flowing into the LAST hidden state, from the output layer
    for t in reversed(range(T)):                           # walk BACKWARD through time: T-1, T-2, ..., 0
        h_t = H[:, t, :]                                     # this timestep's hidden state (the tanh OUTPUT, cached by the forward pass)
        h_prev = H[:, t - 1, :] if t > 0 else h0              # the hidden state that FED this timestep -- h0 at the very first step
        x_t = X[:, t:t + 1]                                   # this timestep's input, same 2D shape the forward pass used
        da_t = dh * (1.0 - h_t ** 2)                          # undo tanh: d tanh(a)/da = 1 - tanh(a)^2 = 1 - h_t^2, using the CACHED output
        dWxh += x_t.T @ da_t / m                              # this timestep's contribution to Wxh's gradient, ADDED to the running total
        dWhh += h_prev.T @ da_t / m                           # this timestep's contribution to Whh's gradient, ADDED to the running total
        dbh += da_t.sum(axis=0) / m                           # this timestep's contribution to bh's gradient, ADDED to the running total
        dh = da_t @ Whh.T                                     # THE "THROUGH TIME" LINE: hand this step's gradient back to h_{t-1}, through the same Whh
    return dWxh, dWhh, dbh                                  # all three shared-parameter gradients, summed over every timestep


class VanillaRNNClassifier:                             # Day 57's many-to-one classifier -- same forward pass, now with a real backward()
    """Reads the whole sequence, classifies from the FINAL hidden state
    only. Forward pass is byte-for-byte Day 57's; backward() is new."""

    def __init__(self, hidden_dim=8, n_classes=2, lr=0.1, seed=42):  # constructor: still no sequence-length argument, same as Day 57
        rng = np.random.RandomState(seed)                  # this network's own private RNG, seeded for reproducible weight init
        self.hidden_dim = hidden_dim                        # remember the hidden state size
        self.lr = lr                                        # learning rate, used in backward() -- NEW today, since Day 57 never trained
        self.Wxh = rng.randn(1, hidden_dim) * he_scale(1)    # input-to-hidden weights: 1 scalar input per timestep -> hidden_dim
        self.Whh = rng.randn(hidden_dim, hidden_dim) * he_scale(hidden_dim)  # hidden-to-hidden weights: the RECURRENT connection
        self.bh = np.zeros(hidden_dim)                       # hidden bias
        self.Why = rng.randn(hidden_dim, n_classes) * he_scale(hidden_dim)  # output layer: final hidden state -> class logits
        self.by = np.zeros(n_classes)                        # output bias

    def forward(self, X):                                  # forward pass: identical to Day 57, plus caching what backward() needs
        m, T = X.shape                                       # batch size and sequence length -- read fresh every call
        self.X = X                                           # cache the input sequence for backward()'s dWxh
        self.h0 = np.zeros((m, self.hidden_dim))              # the initial hidden state, cached for backward()'s first-timestep dWhh
        self.H = rnn_forward(X, self.h0, self.Wxh, self.Whh, self.bh)  # every timestep's hidden state, cached for BPTT
        self.h_last = self.H[:, -1, :]                        # the FINAL hidden state -- the many-to-one classifier reads only this
        logits = self.h_last @ self.Why + self.by             # output layer pre-activation
        self.probs = softmax(logits)                           # class probabilities, cached for backward()
        return self.probs                                      # hand back to the caller

    def backward(self, y_onehot):                          # NEW today: backward pass through the output layer, then BPTT through every timestep
        m = y_onehot.shape[0]                                # batch size
        dlogits = self.probs - y_onehot                      # combined softmax+cross-entropy gradient w.r.t. logits (same shortcut as every prior day)
        dWhy = self.h_last.T @ dlogits / m                    # output layer weight gradient
        dby = dlogits.sum(axis=0) / m                         # output layer bias gradient
        dh_last = dlogits @ self.Why.T                        # gradient flowing into the FINAL hidden state -- this is BPTT's starting point
        dWxh, dWhh, dbh = rnn_backward(dh_last, self.X, self.H, self.h0, self.Whh)  # BPTT: back through every timestep, accumulating shared-param gradients
        self.grads = dict(Wxh=dWxh, Whh=dWhh, bh=dbh, Why=dWhy, by=dby)  # stash every gradient BEFORE updating, so a gradient check can inspect and undo them
        self.Wxh -= self.lr * dWxh                            # SGD update: input-to-hidden weights
        self.Whh -= self.lr * dWhh                            # SGD update: recurrent weights
        self.bh -= self.lr * dbh                              # SGD update: hidden bias
        self.Why -= self.lr * dWhy                            # SGD update: output layer weights
        self.by -= self.lr * dby                              # SGD update: output layer bias

    def accuracy(self, X, y):                              # measures classification accuracy on a given dataset
        preds = self.forward(X).argmax(axis=1)               # predicted class = whichever column has the higher probability
        return (preds == y).mean()                            # fraction of predictions that match the true label


# =============================================================================
# 2. GRADIENT CHECK -- does BPTT's accumulate-across-time math actually match?
# =============================================================================
def demo_gradient_check():                                # verifies BPTT against numeric central differences, every parameter
    print("=" * 70)                                         # section separator
    print("1. GRADIENT CHECK -- BPTT vs. numeric central differences")
    print("=" * 70)                                          # closing separator
    print("Every backward pass in this series has been checked this way "  # explains the check
          "since Day 39. BPTT is the hardest one yet to get right: three "
          "of the five parameters (Wxh, Whh, bh) receive a contribution "
          "from EVERY timestep, and a single wrong index (h_t vs. h_{t-1}, "
          "or overwriting instead of accumulating) would silently break it. "
          "Checked on a small 6-step sequence so every timestep matters.")
    rng = np.random.RandomState(1)                            # fixed rng for reproducible perturbation indices
    net = VanillaRNNClassifier(hidden_dim=4, n_classes=2, lr=0.01, seed=1)  # a small RNN just for this check
    X = rng.randn(5, 6) * 0.8                                  # a tiny batch of 5 random length-6 sequences
    y_oh = one_hot(np.array([0, 1, 0, 1, 1]), 2)               # matching one-hot labels

    net.forward(X)                                             # one forward pass to populate every cache backward() needs
    net.backward(y_oh)                                          # run the real backward() once -- NOTE: this also applies an SGD update in place
    for name in ["Wxh", "Whh", "bh", "Why", "by"]:              # undo that update for every parameter (p -= lr*g, so p += lr*g restores it)
        getattr(net, name)[...] += net.lr * net.grads[name]      # in-place restore, so perturbing below starts from the SAME weights backward() differentiated

    eps = 1e-5                                                  # perturbation size for the central-difference numeric gradient
    max_rel_err = 0.0                                           # tracks the worst relative error seen across all checked entries
    print(f"\n{'param':<10}{'index':<10}{'analytic':>14}{'numeric':>14}{'rel_err':>12}")  # header row for the printed comparison table
    for name in ["Wxh", "Whh", "bh", "Why", "by"]:              # check every single parameter of the network
        param = getattr(net, name)                              # the ACTUAL parameter array being perturbed
        for _ in range(3):                                       # 3 random entries per parameter (5 params x 3 = 15 checks)
            idx = tuple(rng.randint(0, s) for s in param.shape)   # a random valid index into this parameter
            orig = param[idx]                                     # remember the original value so it can be restored
            param[idx] = orig + eps                               # nudge this one entry up by eps
            lp = cross_entropy_loss(net.forward(X), y_oh)          # recompute the loss with that nudge applied
            param[idx] = orig - eps                               # nudge the SAME entry down by eps instead
            lm = cross_entropy_loss(net.forward(X), y_oh)          # recompute the loss with the downward nudge
            param[idx] = orig                                     # restore the original value
            numeric = (lp - lm) / (2 * eps)                        # central-difference numeric estimate of the gradient
            ana = net.grads[name][idx]                             # the analytic (BPTT) gradient at that same entry
            rel_err = abs(numeric - ana) / max(abs(numeric) + abs(ana), 1e-8)  # relative error between the two estimates
            max_rel_err = max(max_rel_err, rel_err)                # keep track of the worst one seen so far
            print(f"{name:<10}{str(idx):<10}{ana:>14.8f}{numeric:>14.8f}{rel_err:>12.2e}")  # print this entry's comparison row
    print(f"\nmax relative error across all checked entries: {max_rel_err:.2e} "  # final verdict line
          f"({'PASS -- BPTT is correct' if max_rel_err < 1e-5 else 'FAIL'})")
    print("Actual output from this lesson's run.")               # standing convention: label this as real output


# =============================================================================
# 3. DOES IT ACTUALLY LEARN? -- a real training run, short sequences
# =============================================================================
def train(net, X_train, y_train, epochs, batch_size, seed):  # a plain mini-batch training loop, recording loss and accuracy per epoch
    rng = np.random.RandomState(seed)                      # this training run's own private RNG for shuffling
    losses, accs = [], []                                   # per-epoch training loss and accuracy
    for epoch in range(epochs):                             # loop over the full training budget
        order = rng.permutation(len(y_train))                 # a fresh shuffle each epoch
        epoch_loss = 0.0                                       # running sum of this epoch's batch losses
        for start in range(0, len(order), batch_size):         # step through in mini-batches
            idx = order[start:start + batch_size]                # this batch's indices
            y_oh = one_hot(y_train[idx], 2)                      # one-hot labels for this batch
            probs = net.forward(X_train[idx])                    # forward pass
            epoch_loss += cross_entropy_loss(probs, y_oh) * len(idx)  # accumulate this batch's loss, weighted by its size
            net.backward(y_oh)                                   # BPTT + SGD update
        losses.append(epoch_loss / len(y_train))               # this epoch's average loss
        accs.append(net.accuracy(X_train, y_train))            # this epoch's training accuracy
    return losses, accs                                       # the full training history


def demo_train_short_sequence():                           # the first real training run of an RNN in this series
    print("\n" + "=" * 70)                                    # blank line then section separator
    print("2. DOES IT ACTUALLY LEARN? -- training on Day 56's task, T=10")
    print("=" * 70)                                            # closing separator
    print("A passed gradient check says the math is right. It doesn't say "  # explains why a training run follows the check
          "the network can actually solve anything. Day 56's order-"
          "detection task, at a short length (T=10) where Day 57 measured "
          "long-range sensitivity still healthy (0.66 at T=10), trained "
          "from scratch with BPTT for the first time.")

    T = 10                                                      # a SHORT sequence -- where yesterday's forward-pass sensitivity was still strong
    X_train, y_train = make_order_dataset(n_per_class=150, T=T, pos_lo=0, pos_hi=T - 1, seed=42)  # full-range spike positions, standard training set
    X_test, y_test = make_order_dataset(n_per_class=100, T=T, pos_lo=0, pos_hi=T - 1, seed=999)    # a genuinely separate test set

    net = VanillaRNNClassifier(hidden_dim=16, lr=0.1, seed=7)    # a fresh RNN -- still no T argument, same as Day 57
    acc_before = net.accuracy(X_test, y_test)                    # test accuracy BEFORE any training -- should be near chance
    losses, accs = train(net, X_train, y_train, epochs=300, batch_size=30, seed=7)  # real BPTT training
    acc_after = net.accuracy(X_test, y_test)                     # test accuracy AFTER training

    print(f"\ntest_acc BEFORE training (random weights): {acc_before:.4f}")  # report the untrained baseline
    print(f"train_acc AFTER training:                   {accs[-1]:.4f}")      # report final training accuracy
    print(f"test_acc AFTER training:                    {acc_after:.4f}")      # report final test accuracy
    print(f"training loss: {losses[0]:.4f} (epoch 0) -> {losses[-1]:.4f} (epoch {len(losses) - 1})")  # show the loss actually went down
    print("Actual output from this lesson's run.")                 # standing convention
    return losses, accs, acc_after                                  # hand results back for the plot


# =============================================================================
# 4. THE SURPRISE -- T=40 trains too. Why, when Day 57 said signal collapses?
# =============================================================================
def small_signal_sensitivity(net, T, seed):               # Day 57's measure: how much does h_T move when x_0 moves by a TINY amount?
    rng = np.random.RandomState(seed)                       # fixed rng, so before/after training compare on the identical sequence
    X = rng.randn(1, T) * 0.3                               # one noise-only sequence, same scale as the task's background noise
    h0 = np.zeros((1, net.hidden_dim))                      # initial hidden state
    eps = 1e-3                                               # a TINY perturbation -- this is a local derivative, nothing more
    Xp = X.copy(); Xp[0, 0] += eps                           # x_0 nudged up
    Xm = X.copy(); Xm[0, 0] -= eps                           # x_0 nudged down
    hp = rnn_forward(Xp, h0, net.Wxh, net.Whh, net.bh)[:, -1, :]  # final hidden state, nudged up
    hm = rnn_forward(Xm, h0, net.Wxh, net.Whh, net.bh)[:, -1, :]  # final hidden state, nudged down
    return np.linalg.norm(hp - hm) / (2 * eps)               # finite-difference estimate of ||d h_T / d x_0||


def large_signal_effect(net, T, seed, n=50):              # NEW today: how much does h_T move when x_0 gets a full-size +3 SPIKE?
    rng = np.random.RandomState(seed)                       # fixed rng, so before/after training compare on identical sequences
    X = rng.randn(n, T) * 0.3                               # n noise-only sequences
    Xs = X.copy(); Xs[:, 0] += 3.0                           # the SAME sequences, plus a task-sized +3 spike at position 0
    h0 = np.zeros((n, net.hidden_dim))                      # initial hidden state
    h_plain = rnn_forward(X, h0, net.Wxh, net.Whh, net.bh)[:, -1, :]   # final state WITHOUT the spike
    h_spike = rnn_forward(Xs, h0, net.Wxh, net.Whh, net.bh)[:, -1, :]  # final state WITH the spike
    return np.linalg.norm(h_spike - h_plain, axis=1).mean()  # average distance between the two -- a NONLINEAR, large-signal effect, not a derivative


def demo_long_sequence_surprise(short_losses):             # same training at T=40 -- and the diagnosis of why it worked
    print("\n" + "=" * 70)                                    # blank line then section separator
    print("3. THE SURPRISE -- the exact same training at T=40")
    print("=" * 70)                                            # closing separator
    print("Day 57 measured forward-pass sensitivity to x_0 collapsing to "  # frames the prediction honestly, before the result
          "0.000024 by T=40, and called it a preview of vanishing "
          "gradients. The natural prediction: an RNN should struggle to "
          "train at T=40. Tested directly -- identical network size, lr, "
          "epochs, and seed as Section 2, only the sequence length changes.")

    T = 40                                                      # a LONGER sequence -- where Day 57's forward-pass sensitivity had collapsed
    X_train, y_train = make_order_dataset(n_per_class=150, T=T, pos_lo=0, pos_hi=T - 1, seed=42)  # same generator, longer sequences
    X_test, y_test = make_order_dataset(n_per_class=100, T=T, pos_lo=0, pos_hi=T - 1, seed=999)    # separate test set

    net = VanillaRNNClassifier(hidden_dim=16, lr=0.1, seed=7)    # IDENTICAL network configuration and seed as Section 2
    small_before = small_signal_sensitivity(net, T, seed=0)       # Day 57's tiny-perturbation measure, BEFORE training
    large_before = large_signal_effect(net, T, seed=0)            # the new full-spike measure, BEFORE training
    long_losses, long_accs = train(net, X_train, y_train, epochs=300, batch_size=30, seed=7)  # IDENTICAL training budget
    acc_long = net.accuracy(X_test, y_test)                      # test accuracy after training at T=40
    small_after = small_signal_sensitivity(net, T, seed=0)        # same tiny-perturbation measure, AFTER training, same sequence
    large_after = large_signal_effect(net, T, seed=0)             # same full-spike measure, AFTER training, same sequences

    print(f"\ntrain_acc AFTER training (T=40): {long_accs[-1]:.4f}")  # report final training accuracy
    print(f"test_acc AFTER training (T=40):  {acc_long:.4f}")          # report final test accuracy
    print(f"training loss: {long_losses[0]:.4f} (epoch 0) -> {long_losses[-1]:.4f} (epoch {len(long_losses) - 1})")
    print("Actual output from this lesson's run.")                     # standing convention

    preds = net.forward(X_test).argmax(axis=1)                          # per-example test predictions
    first_spike = np.array([min(np.argmax(s), np.argmin(s)) for s in X_test])  # position of whichever spike came FIRST in each sequence
    mem_dist = (T - 1) - first_spike                                     # how many steps that first spike must be REMEMBERED until the final step
    print(f"\ntest accuracy by memory distance (steps the first spike must be carried to the end):")
    for lo, hi in [(0, 9), (10, 19), (20, 29), (30, 39)]:               # four buckets of increasing memory distance
        mask = (mem_dist >= lo) & (mem_dist <= hi)                        # test examples in this bucket
        print(f"  {lo:>2}-{hi:<2} steps: n={mask.sum():>3}   test_acc={(preds[mask] == y_test[mask]).mean():.4f}")
    print("Actual output from this lesson's run.")                     # standing convention

    net.forward(X_test)                                                  # refresh cached h_last on the test set for the saturation check
    saturated = (np.abs(net.h_last) > 0.95).mean()                       # fraction of final hidden units pinned near tanh's +-1 limits
    spectral_radius = np.max(np.abs(np.linalg.eigvals(net.Whh)))         # largest |eigenvalue| of the trained recurrent weight matrix
    print(f"\n{'measure (h_T at T=40, x_0 perturbed)':<46}{'untrained':>12}{'trained':>12}")
    print(f"{'small-signal: ||dh_T/dx_0||, eps=1e-3':<46}{small_before:>12.6f}{small_after:>12.6f}")
    print(f"{'large-signal: ||h_T(+3 spike) - h_T(none)||':<46}{large_before:>12.6f}{large_after:>12.6f}")
    print(f"\ntrained network: {saturated:.1%} of final hidden units have |h| > 0.95; "
          f"largest |eigenvalue| of Whh = {spectral_radius:.3f}")
    print("Actual output from this lesson's run.")                     # standing convention

    print(f"\nreal, honest finding: the prediction was wrong. T=40 trains "
          f"to test_acc={acc_long:.3f} -- and holds up even when the first "
          f"spike must be carried 30-39 steps. Day 57's measurement wasn't "
          f"false, it was measuring the wrong thing for THIS task: after "
          f"training, sensitivity to a TINY nudge at x_0 actually DROPS "
          f"({small_before:.6f} -> {small_after:.6f}), while the effect of a "
          f"full-size +3 spike at x_0 GROWS ({large_before:.3f} -> "
          f"{large_after:.3f}). The network learned to ignore small noise "
          f"and LATCH onto large spikes: {saturated:.0%} of its final hidden "
          f"units sit pinned near tanh's +-1 limits, and Whh's largest "
          f"eigenvalue ({spectral_radius:.2f}) is above 1 -- which, with tanh "
          f"bounding every unit, produces stable saturated memory states "
          f"rather than explosion. A tiny-epsilon derivative is a LOCAL, "
          f"linear measurement; latching is a large-signal, nonlinear "
          f"effect it fundamentally cannot see. The caveat that matters for "
          f"Day 59: this task is unusually latch-friendly (+-3 spikes over "
          f"0.3 noise, one binary fact to store). Vanishing gradients are "
          f"real -- this task just happens to route around them.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))              # two panels: what happened, and why
    ax1.plot(short_losses, label="T=10", color="#5cb85c")               # short-sequence loss curve
    ax1.plot(long_losses, label="T=40", color="#d9534f")                # long-sequence loss curve
    ax1.axhline(np.log(2), color="gray", linestyle="--", linewidth=1, label="ln(2) = chance-level loss")  # the loss of an always-50/50 model
    ax1.set_xlabel("epoch")                                              # x-axis label
    ax1.set_ylabel("training cross-entropy loss")                        # y-axis label
    ax1.set_title("Both lengths train -- T=40 was expected to struggle")  # panel title
    ax1.legend()                                                          # distinguish the curves and reference line
    labels = ["small-signal\n(tiny nudge, eps=1e-3)", "large-signal\n(full +3 spike)"]  # the two measures being contrasted
    x = np.arange(2)                                                      # bar-group positions
    ax2.bar(x - 0.18, [small_before, large_before], width=0.36, label="untrained", color="#999999")  # before training
    ax2.bar(x + 0.18, [small_after, large_after], width=0.36, label="trained", color="#337ab7")      # after training
    ax2.set_yscale("log")                                                 # log scale -- the values span ~5 orders of magnitude
    ax2.set_xticks(x)                                                     # one tick per measure
    ax2.set_xticklabels(labels)                                           # label each measure
    ax2.set_ylabel("effect of x_0 on h_T at T=40 (log scale)")            # y-axis label
    ax2.set_title("Why: training suppresses noise, latches spikes")       # panel title
    ax2.legend()                                                          # distinguish untrained vs trained
    plt.tight_layout()                                                    # avoid clipped labels/titles when saving
    plt.savefig("bptt_training_short_vs_long.png", dpi=110)              # write the figure to disk
    plt.close()                                                           # free the figure's memory now that it's saved
    print("\nsaved bptt_training_short_vs_long.png")                     # confirm the save to the console


# =============================================================================
# 5. WHAT THIS MAPS TO IN PYTORCH
# =============================================================================
def note_on_pytorch():                                     # pure documentation: prints a PyTorch equivalent, imports nothing
    print("\n" + "=" * 70)                                    # blank line then section separator
    print("4. WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)                                            # closing separator
    print(                                                       # the PyTorch equivalent of today's whole classifier
        "class RNNClassifier(nn.Module):\n"
        "    def __init__(self, hidden_dim=16, n_classes=2):\n"
        "        super().__init__()\n"
        "        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_dim,\n"
        "                          nonlinearity='tanh', batch_first=True)\n"
        "        self.out = nn.Linear(hidden_dim, n_classes)\n"
        "\n"
        "    def forward(self, x):              # x: (batch, T, 1)\n"
        "        H, h_last = self.rnn(x)        # H: every hidden state, h_last: the final one\n"
        "        return self.out(h_last[-1])    # many-to-one: classify from the last state\n"
        "\n"
        "loss.backward()   # autograd runs BPTT automatically -- the accumulate-\n"
        "                  # across-timesteps loop in rnn_backward() is exactly\n"
        "                  # what it does under the hood"
    )


def main():                                                # entry point: runs every demo in the order this document walks them
    demo_gradient_check()                                    # 1. verify BPTT against numeric gradients
    short_losses, _, _ = demo_train_short_sequence()          # 2. the first real RNN training run -- short sequences
    demo_long_sequence_surprise(short_losses)                 # 3. T=40 trains too -- and the diagnosis of why
    note_on_pytorch()                                         # 4. relate it back to a production framework


if __name__ == "__main__":                                 # only run main() when this file is executed directly, not when imported
    main()                                                    # kick off the whole lesson
