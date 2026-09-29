"""
Day 57 Exercise -- The RNN Cell, Isolated

The main lesson's core new piece, minimal: one function implementing the
vanilla RNN recurrence, run over a short toy sequence, with a quick
correctness check against a hand-unrolled computation. No classifier
head, no sensitivity analysis, no plots -- just the cell itself. Day 58's
exercise will backprop through exactly this function.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def rnn_step_forward(x_t, h_prev, Wxh, Whh, bh):        # ONE timestep of the vanilla RNN recurrence -- the whole mechanism in one line
    return np.tanh(x_t @ Wxh + h_prev @ Whh + bh)         # combine this step's input AND the previous hidden state, squash with tanh


def rnn_forward(X, h0, Wxh, Whh, bh):                   # runs rnn_step_forward once per timestep, collecting every hidden state
    m, T = X.shape                                        # batch size and sequence length
    hidden_dim = Whh.shape[0]                             # hidden state size, read off Whh's own shape
    H = np.zeros((m, T, hidden_dim))                      # will hold every timestep's hidden state
    h = h0                                                 # start from the given initial hidden state
    for t in range(T):                                     # walk through the sequence one step at a time
        x_t = X[:, t:t + 1]                                  # this timestep's input, kept 2D as (batch, 1)
        h = rnn_step_forward(x_t, h, Wxh, Whh, bh)           # the SAME weights are reused at every t
        H[:, t, :] = h                                       # store this timestep's hidden state
    return H                                               # every hidden state, t=0 through T-1


rng = np.random.RandomState(0)                          # fixed rng for reproducible weights/input
hidden_dim = 2                                            # a tiny hidden size
Wxh = rng.randn(1, hidden_dim) * 0.5                       # small random input-to-hidden weights
Whh = rng.randn(hidden_dim, hidden_dim) * 0.5               # small random hidden-to-hidden weights
bh = rng.randn(hidden_dim) * 0.1                            # small random bias
X = rng.randn(1, 3) * 0.5                                   # one length-3 toy sequence
h0 = np.zeros((1, hidden_dim))                              # initial hidden state, all zeros

H = rnn_forward(X, h0, Wxh, Whh, bh)                     # run the implementation being checked

h1_hand = np.tanh(X[:, 0:1] @ Wxh + h0 @ Whh + bh)        # hand-compute h1 directly from the recurrence
h2_hand = np.tanh(X[:, 1:2] @ Wxh + h1_hand @ Whh + bh)    # hand-compute h2, using h1_hand
h3_hand = np.tanh(X[:, 2:3] @ Wxh + h2_hand @ Whh + bh)    # hand-compute h3, using h2_hand

print("H (loop implementation):")                        # show the loop's own output
print(H[0])                                                # one row per timestep
print("\nhand-unrolled h1, h2, h3:")                       # show the hand-computed reference
print(np.vstack([h1_hand[0], h2_hand[0], h3_hand[0]]))     # stack into the same (3, hidden_dim) shape for easy comparison
max_diff = max(np.max(np.abs(H[:, 0, :] - h1_hand)),        # the largest disagreement at any timestep
               np.max(np.abs(H[:, 1, :] - h2_hand)),
               np.max(np.abs(H[:, 2, :] - h3_hand)))
print(f"\nmax difference: {max_diff:.2e} ({'match' if max_diff < 1e-12 else 'MISMATCH'})")  # the verdict
