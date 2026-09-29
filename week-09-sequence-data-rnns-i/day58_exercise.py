"""
Day 58 Exercise -- BPTT, Isolated

Day 57's exercise built the RNN cell's forward pass. This one adds the
matching backward pass -- backpropagation through time -- and checks it
against numeric gradients. No classifier head: the "loss" here is just
the sum of the final hidden state, the simplest possible scalar to
differentiate, so the only thing being tested is BPTT itself.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def rnn_step_forward(x_t, h_prev, Wxh, Whh, bh):        # ONE timestep -- unchanged from day57_exercise.py
    return np.tanh(x_t @ Wxh + h_prev @ Whh + bh)         # combine input and previous state, squash with tanh


def rnn_forward(X, h0, Wxh, Whh, bh):                   # full forward pass -- unchanged from day57_exercise.py
    m, T = X.shape                                        # batch size and sequence length
    H = np.zeros((m, T, Whh.shape[0]))                    # every timestep's hidden state
    h = h0                                                 # start from the initial state
    for t in range(T):                                     # one step at a time
        h = rnn_step_forward(X[:, t:t + 1], h, Wxh, Whh, bh)  # same weights at every t
        H[:, t, :] = h                                       # store it
    return H                                               # all hidden states


def rnn_backward(dh_last, X, H, h0, Whh):               # NEW: backpropagation through time
    m, T = X.shape                                        # batch size and sequence length
    dWxh = np.zeros((1, Whh.shape[0]))                    # accumulators -- start at zero,
    dWhh = np.zeros_like(Whh)                             # get one contribution PER timestep
    dbh = np.zeros(Whh.shape[0])
    dh = dh_last                                           # gradient arrives at the last hidden state
    for t in reversed(range(T)):                           # walk BACKWARD through time
        h_prev = H[:, t - 1, :] if t > 0 else h0              # the state that fed timestep t
        da = dh * (1.0 - H[:, t, :] ** 2)                     # undo tanh using the cached output
        dWxh += X[:, t:t + 1].T @ da                          # ACCUMULATE -- the same Wxh was used at every t
        dWhh += h_prev.T @ da                                 # ACCUMULATE
        dbh += da.sum(axis=0)                                 # ACCUMULATE
        dh = da @ Whh.T                                       # hand gradient back to h_{t-1}: the "through time" step
    return dWxh, dWhh, dbh                                  # summed over every timestep


rng = np.random.RandomState(0)                          # fixed rng for reproducibility
hidden_dim, T = 3, 5                                      # tiny sizes, so every timestep matters
Wxh = rng.randn(1, hidden_dim) * 0.5                      # random weights
Whh = rng.randn(hidden_dim, hidden_dim) * 0.5
bh = rng.randn(hidden_dim) * 0.1
X = rng.randn(2, T)                                        # a batch of 2 length-5 sequences
h0 = np.zeros((2, hidden_dim))                             # initial state


def loss_fn():                                          # the simplest scalar: sum of the final hidden state
    return rnn_forward(X, h0, Wxh, Whh, bh)[:, -1, :].sum()


H = rnn_forward(X, h0, Wxh, Whh, bh)                     # forward pass
dh_last = np.ones((2, hidden_dim))                         # d(sum of h_last)/d h_last = all ones
dWxh, dWhh, dbh = rnn_backward(dh_last, X, H, h0, Whh)   # BPTT

eps, worst = 1e-5, 0.0                                     # perturbation size, worst relative error seen
for name, param, grad in [("Wxh", Wxh, dWxh), ("Whh", Whh, dWhh), ("bh", bh, dbh)]:
    for idx in np.ndindex(param.shape):                     # check EVERY entry -- it's small enough
        orig = param[idx]
        param[idx] = orig + eps; lp = loss_fn()              # nudge up
        param[idx] = orig - eps; lm = loss_fn()              # nudge down
        param[idx] = orig                                    # restore
        numeric = (lp - lm) / (2 * eps)                       # central difference
        rel = abs(numeric - grad[idx]) / max(abs(numeric) + abs(grad[idx]), 1e-8)
        worst = max(worst, rel)                               # track the worst
    print(f"{name}: all {param.size} entries checked")

print(f"\nmax relative error: {worst:.2e} ({'PASS' if worst < 1e-6 else 'FAIL'})")  # the verdict
