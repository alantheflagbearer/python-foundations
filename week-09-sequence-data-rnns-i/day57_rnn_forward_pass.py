"""
Day 57: Vanilla RNN Forward Pass -- From Scratch on a Real Toy Sequence

Day 56 measured two real failure modes and named what was missing: a
network that applies the SAME weights at every timestep (fixing
PlainFCNet's no-position-invariance collapse) while carrying information
forward across the WHOLE sequence, not just a fixed local window (fixing
Conv1DNet's capped receptive field). That network is a Recurrent Neural
Network. Today builds its forward pass from scratch -- backward pass and
training are Day 58's topic, so every check below is a FORWARD-pass
correctness or capability check, not an accuracy number.

The vanilla RNN recurrence is one equation, applied once per timestep:
    h_t = tanh(x_t @ Wxh + h_{t-1} @ Whh + bh)
The SAME three parameters (Wxh, Whh, bh) are reused at every single
timestep t -- this is the whole mechanism that gives an RNN both of the
properties Day 56 found missing.

Three real, measured checks, not just assertions:

  1. Correctness -- the vectorized/looped implementation is checked
     against an explicit, hand-unrolled computation of the first few
     timesteps on a tiny toy example, confirmed to match exactly.
  2. The fixed-length limitation, GONE -- the exact same untrained RNN
     (identical weights) is run forward on Day 56's order-detection
     sequences at THREE different lengths (10, 20, 30) with zero errors
     and zero architecture changes -- direct proof against Day 56's
     PlainFCNet ValueError.
  3. Long-range sensitivity -- does information from position 0 actually
     reach the final hidden state, at long range, the way Conv1DNet's
     capped receptive field structurally could not? Measured directly
     via a finite-difference perturbation of x_0, tracked across
     increasing sequence length. The real, honest result is dramatic:
     sensitivity starts at 0.70 (T=5), is still 0.44 at T=20 -- then
     collapses to 0.000024 by T=40 and is EXACTLY 0.0 by T=160 (float64
     genuinely runs out of precision to represent it). Unlike Conv1DNet,
     there is no hard structural cutoff -- but this previews Day 59's
     real topic directly: vanishing signal through repeated tanh and
     matrix-multiply steps, visible in the forward pass ALONE, before
     backpropagation ever enters the picture.

Every line of code below carries its own comment, continuing Day 52's
convention.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # "Agg" backend renders to a file, not a screen -- needed since this runs headless
import matplotlib.pyplot as plt                         # pyplot: the plotting API actually called later (plt.plot, plt.savefig, ...)

np.random.seed(42)                                      # fixes numpy's GLOBAL random state so any code that forgets to pass its own rng is still reproducible

# =============================================================================
# 0. SHARED HELPERS -- reused verbatim from Day 56 (the same toy task, so
#    today's RNN can be run against the exact same data)
# =============================================================================
def softmax(z):                                         # softmax: turns raw class scores (logits) into a probability distribution per row
    z_shifted = z - z.max(axis=1, keepdims=True)          # subtract each row's own max first, for numerical stability (avoids exp() overflow)
    exp_z = np.exp(z_shifted)                             # exponentiate the shifted logits
    return exp_z / exp_z.sum(axis=1, keepdims=True)       # normalize each row so its probabilities sum to 1


def he_scale(n_in):                                     # He initialization scale factor -- used loosely here even though tanh (not ReLU) is today's activation
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


# =============================================================================
# 1. THE VANILLA RNN CELL -- ONE equation, reused at every timestep
# =============================================================================
def rnn_step_forward(x_t, h_prev, Wxh, Whh, bh):        # NEW today: a single RNN timestep, the entire recurrence in one function
    return np.tanh(x_t @ Wxh + h_prev @ Whh + bh)         # combine this step's input AND the previous hidden state, squash with tanh


def rnn_forward(X, h0, Wxh, Whh, bh):                   # NEW today: runs rnn_step_forward once per timestep, collecting every hidden state
    m, T = X.shape                                        # batch size and sequence length -- T can be ANYTHING, unlike Day 56's PlainFCNet
    hidden_dim = Whh.shape[0]                             # hidden state size, read off Whh's own shape
    H = np.zeros((m, T, hidden_dim))                      # will hold every timestep's hidden state, shape (batch, time, hidden)
    h = h0                                                 # start from the given initial hidden state (usually all zeros)
    for t in range(T):                                     # walk through the sequence ONE STEP AT A TIME -- the defining trait of an RNN
        x_t = X[:, t:t + 1]                                  # this timestep's input, kept 2D as (batch, 1) since today's sequences are scalar-per-step
        h = rnn_step_forward(x_t, h, Wxh, Whh, bh)           # the SAME Wxh/Whh/bh are reused at every single t -- no new parameters per timestep
        H[:, t, :] = h                                       # store this timestep's hidden state
    return H                                               # every hidden state, from t=0 to t=T-1


class VanillaRNNClassifier:                             # a thin wrapper: RNN forward pass + a final softmax classifier on the LAST hidden state
    """Many-to-one: reads the whole sequence, then classifies based only
    on the FINAL hidden state h_{T-1}. Forward pass only today -- no
    backward() method exists yet (that's Day 58); weights are randomly
    initialized and never trained in this lesson."""

    def __init__(self, hidden_dim=8, n_classes=2, seed=42):  # constructor: NOTE no sequence length T argument -- unlike PlainFCNet, none is needed
        rng = np.random.RandomState(seed)                  # this network's own private RNG, seeded for reproducible weight init
        self.hidden_dim = hidden_dim                        # remember the hidden state size
        self.Wxh = rng.randn(1, hidden_dim) * he_scale(1)    # input-to-hidden weights: 1 scalar input per timestep -> hidden_dim
        self.Whh = rng.randn(hidden_dim, hidden_dim) * he_scale(hidden_dim)  # hidden-to-hidden weights: the RECURRENT connection
        self.bh = np.zeros(hidden_dim)                       # hidden bias
        self.Why = rng.randn(hidden_dim, n_classes) * he_scale(hidden_dim)  # output layer: final hidden state -> class logits
        self.by = np.zeros(n_classes)                        # output bias

    def forward(self, X):                                  # forward pass: run the recurrence over the WHOLE sequence, classify from the last state
        m, T = X.shape                                       # batch size and sequence length -- read fresh from X every call, never fixed at construction
        h0 = np.zeros((m, self.hidden_dim))                   # the initial hidden state, before any input has been seen
        self.H = rnn_forward(X, h0, self.Wxh, self.Whh, self.bh)  # every timestep's hidden state, shape (m, T, hidden_dim)
        h_last = self.H[:, -1, :]                             # the FINAL hidden state -- the many-to-one classifier reads only this
        logits = h_last @ self.Why + self.by                  # output layer pre-activation
        return softmax(logits)                                 # class probabilities


# =============================================================================
# 2. CORRECTNESS CHECK -- hand-unrolled vs. the loop, on a tiny toy example
# =============================================================================
def demo_correctness_check():                            # verifies rnn_forward's loop against an explicit, hand-written unrolling
    print("=" * 70)                                         # section separator
    print("1. CORRECTNESS CHECK -- hand-unrolled recurrence vs. the loop implementation")
    print("=" * 70)                                          # closing separator
    print("No gradients exist yet today (that's Day 58), so today's "  # explains why this check replaces a gradient check
          "verification is different: does rnn_forward's for-loop "
          "actually compute the same thing as writing out each "
          "timestep's equation by hand? Checked on a tiny 3-step example.")

    rng = np.random.RandomState(0)                            # fixed rng for reproducible toy weights/input
    hidden_dim = 2                                             # a tiny hidden size, small enough to hand-verify
    Wxh = rng.randn(1, hidden_dim) * 0.5                        # small random input-to-hidden weights
    Whh = rng.randn(hidden_dim, hidden_dim) * 0.5                # small random hidden-to-hidden weights
    bh = rng.randn(hidden_dim) * 0.1                             # small random bias
    X = rng.randn(1, 3) * 0.5                                    # ONE example, a length-3 sequence (batch size 1, for simplicity)
    h0 = np.zeros((1, hidden_dim))                               # initial hidden state, all zeros

    H_loop = rnn_forward(X, h0, Wxh, Whh, bh)                    # run the real implementation being checked

    h1_hand = np.tanh(X[:, 0:1] @ Wxh + h0 @ Whh + bh)            # HAND-COMPUTE h1 directly from the recurrence equation, using h0
    h2_hand = np.tanh(X[:, 1:2] @ Wxh + h1_hand @ Whh + bh)        # HAND-COMPUTE h2, using h1_hand (not H_loop) as the previous state
    h3_hand = np.tanh(X[:, 2:3] @ Wxh + h2_hand @ Whh + bh)        # HAND-COMPUTE h3, using h2_hand

    diffs = [np.max(np.abs(H_loop[:, 0, :] - h1_hand)),            # compare the loop's h1 against the hand-computed h1
             np.max(np.abs(H_loop[:, 1, :] - h2_hand)),             # same for h2
             np.max(np.abs(H_loop[:, 2, :] - h3_hand))]              # same for h3
    print(f"\nmax abs difference at each timestep: {[f'{d:.2e}' for d in diffs]}")  # print all three differences plainly
    print(f"overall max difference: {max(diffs):.2e} "               # the single worst difference across all 3 timesteps
          f"({'PASS -- loop and hand-unrolled recurrence agree exactly' if max(diffs) < 1e-12 else 'FAIL'})")
    print("Actual output from this lesson's run.")                   # standing convention: label this as real output


# =============================================================================
# 3. THE FIXED-LENGTH LIMITATION, GONE -- the SAME weights, three lengths
# =============================================================================
def demo_variable_length():                               # runs the SAME untrained RNN forward on 3 different sequence lengths, no errors
    print("\n" + "=" * 70)                                   # blank line then section separator
    print("2. THE FIXED-LENGTH LIMITATION, GONE -- same weights, three sequence lengths")
    print("=" * 70)                                           # closing separator
    print("Day 56's PlainFCNet raised a ValueError the moment it saw a "  # explains what's being contrasted against, citing Day 56's real result
          "sequence length it wasn't built for. Today's RNN has NO "
          "sequence-length argument in its constructor at all -- it "
          "should just work at any length, checked directly below.")

    net = VanillaRNNClassifier(hidden_dim=8, seed=42)            # ONE network, built ONCE
    for T in [10, 20, 30]:                                        # try three different sequence lengths with that SAME network
        X, y = make_order_dataset(n_per_class=5, T=T, pos_lo=0, pos_hi=T - 1, seed=42)  # a small batch of length-T sequences
        probs = net.forward(X)                                     # forward pass -- no error expected, no special-casing needed
        print(f"T={T:<4} forward pass succeeded, output shape {probs.shape}, "  # report the real, observed shape
              f"probabilities sum to 1: {np.allclose(probs.sum(axis=1), 1.0)}")  # sanity-check that softmax still behaves correctly at this length
    print("\nActual output from this lesson's run. Same Wxh, Whh, bh, Why, "  # standing convention plus the point being made
          "by -- reused unchanged across all three lengths, something "
          "PlainFCNet could never do.")


# =============================================================================
# 4. LONG-RANGE SENSITIVITY -- does information from position 0 survive to h_T?
# =============================================================================
def demo_long_range_sensitivity():                        # measures, via finite differences, how much h_T depends on x_0 as T grows
    print("\n" + "=" * 70)                                   # blank line then section separator
    print("3. LONG-RANGE SENSITIVITY -- does x_0 still reach h_T at long range?")
    print("=" * 70)                                           # closing separator
    print("Conv1DNet (Day 56) had a HARD structural cap: a kernel of "  # frames the question this section answers
          "size k literally cannot combine information more than k "
          "positions apart. An RNN has no such hard cap -- h_T depends "
          "on h_{T-1}, which depends on h_{T-2}, ... all the way back to "
          "x_0, by construction. But does that CHAIN carry a meaningful "
          "amount of signal at long range, or does it fade out? Measured "
          "directly via a finite-difference perturbation of x_0, not "
          "assumed.")

    rng = np.random.RandomState(0)                             # fixed rng for reproducible toy sequences
    hidden_dim = 8                                              # hidden size used throughout this check
    net = VanillaRNNClassifier(hidden_dim=hidden_dim, seed=7)    # ONE untrained network, reused at every sequence length below
    eps = 1e-3                                                   # perturbation size for the finite-difference sensitivity estimate
    lengths = [5, 10, 20, 40, 80, 160]                            # a range of sequence lengths, to see how sensitivity changes with distance
    sensitivities = []                                            # will collect one sensitivity value per length

    print(f"\n{'T':<6}{'|dh_T/dx_0| (finite diff)':>28}")          # table header
    for T in lengths:                                              # loop over every sequence length being tested
        X = rng.randn(1, T) * 0.5                                    # ONE random length-T sequence (batch size 1, for a clean single-example measurement)
        h0 = np.zeros((1, hidden_dim))                                # initial hidden state

        X_plus = X.copy(); X_plus[0, 0] += eps                        # a copy of X with x_0 nudged UP by eps
        X_minus = X.copy(); X_minus[0, 0] -= eps                      # a copy of X with x_0 nudged DOWN by eps

        h_T_plus = rnn_forward(X_plus, h0, net.Wxh, net.Whh, net.bh)[:, -1, :]   # final hidden state with x_0 nudged up
        h_T_minus = rnn_forward(X_minus, h0, net.Wxh, net.Whh, net.bh)[:, -1, :]  # final hidden state with x_0 nudged down

        sensitivity = np.linalg.norm(h_T_plus - h_T_minus) / (2 * eps)  # the finite-difference estimate of ||d h_T / d x_0||
        sensitivities.append(sensitivity)                              # remember this length's sensitivity
        print(f"{T:<6}{sensitivity:>28.6f}")                            # print this row of the table
    print("\nActual output from this lesson's run.")                    # standing convention

    print(f"\nreal, honest finding: sensitivity of the final hidden state "
          f"to the VERY FIRST input starts at {sensitivities[0]:.6f} at "
          f"T={lengths[0]}, is still {sensitivities[2]:.6f} at T={lengths[2]} "
          f"-- but by T={lengths[3]} it has collapsed to {sensitivities[3]:.2e}, "
          f"and by T={lengths[-1]} it is exactly {sensitivities[-1]:.1f} "
          f"(float64 has run out of precision to represent it at all). "
          f"Unlike Conv1DNet, there is no hard cutoff distance beyond "
          f"which x_0 provably cannot reach h_T -- the connection always "
          f"exists structurally. But this measurement shows that "
          f"connection collapsing in practice, not staying flat, well "
          f"before backpropagation ever enters the picture -- a direct, "
          f"forward-pass-only preview of Day 59's real topic: vanishing "
          f"(or exploding) signal through repeated tanh and "
          f"matrix-multiply steps.")

    plt.figure(figsize=(7, 5))                                          # a simple line plot of sensitivity vs. sequence length
    plt.plot(lengths, sensitivities, marker="o", color="#337ab7")        # one point per tested length
    plt.yscale("log")                                                    # log scale, since sensitivity likely spans multiple orders of magnitude
    plt.xlabel("sequence length T")                                      # x-axis label
    plt.ylabel("|d h_T / d x_0|  (finite-difference estimate)")           # y-axis label
    plt.title("Does information from x_0 survive to h_T at long range?")  # descriptive title
    plt.grid(True, which="both", alpha=0.3)                              # a light grid, helpful on a log-scale plot
    plt.tight_layout()                                                   # avoid clipped labels/titles when saving
    plt.savefig("rnn_long_range_sensitivity.png", dpi=110)               # write the figure to disk
    plt.close()                                                          # free the figure's memory now that it's saved
    print("\nsaved rnn_long_range_sensitivity.png")                      # confirm the save to the console


# =============================================================================
# 5. WHAT'S STILL MISSING
# =============================================================================
def note_on_what_comes_next():                             # pure documentation: sets up Day 58 without building anything new
    print("\n" + "=" * 70)                                    # blank line then section separator
    print("4. WHAT'S STILL MISSING")
    print("=" * 70)                                            # closing separator
    print(                                                       # a short conceptual bridge to next day's backward pass
        "Today's RNN has never been trained -- every weight is exactly "
        "what random initialization produced, and Section 3 already "
        "shows that even the FORWARD pass alone can let long-range "
        "signal fade. Turning this into something that actually learns "
        "the order-detection task requires a backward pass: propagating "
        "gradient not just through one layer, but back through EVERY "
        "timestep, reusing the SAME Wxh/Whh/bh at each one (so their "
        "gradients must be ACCUMULATED across all T steps, not computed "
        "once). That algorithm is called backpropagation through time "
        "(BPTT) -- Day 58 derives and gradient-checks it from scratch."
    )


def main():                                                # entry point: runs every demo in the order this document walks them
    demo_correctness_check()                                 # 1. verify the loop implementation against a hand-unrolled recurrence
    demo_variable_length()                                    # 2. the fixed-length limitation is gone -- same weights, three lengths
    demo_long_range_sensitivity()                             # 3. does x_0's influence survive to h_T at long range?
    note_on_what_comes_next()                                 # 4. bridge to Day 58's backward pass (BPTT)


if __name__ == "__main__":                                 # only run main() when this file is executed directly, not when imported
    main()                                                    # kick off the whole lesson
