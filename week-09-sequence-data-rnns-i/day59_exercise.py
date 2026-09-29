"""
Day 59 Exercise -- Watching Gradient Vanish or Explode, Isolated

Day 58's exercise built BPTT. This one changes a single line of it --
RECORD dh at every timestep instead of only using it -- and runs it on
three recurrent weight matrices that differ only in spectral radius.
No training, no classifier: just how much gradient survives the trip
back through time.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def rnn_forward(X, h0, Wxh, Whh, bh):                   # unchanged from day58_exercise.py
    m, T = X.shape                                        # batch size and sequence length
    H = np.zeros((m, T, Whh.shape[0]))                    # every hidden state
    h = h0                                                 # start from the initial state
    for t in range(T):                                     # one step at a time
        h = np.tanh(X[:, t:t + 1] @ Wxh + h @ Whh + bh)      # the recurrence
        H[:, t, :] = h                                       # store it
    return H                                               # all hidden states


def grad_norm_by_timestep(dh_last, H, Whh):             # NEW: BPTT's hidden-state loop, recording ||dh_t|| as it goes
    T = H.shape[1]                                        # sequence length
    norms = np.zeros(T)                                   # one gradient norm per timestep
    dh = dh_last                                           # gradient arrives at the last hidden state
    for t in reversed(range(T)):                           # walk backward through time
        norms[t] = np.linalg.norm(dh)                        # RECORD -- the only new line versus Day 58
        dh = (dh * (1.0 - H[:, t, :] ** 2)) @ Whh.T          # undo tanh, then hand back to h_{t-1}
    return norms                                           # norms[0] = what reaches the very first step


rng = np.random.RandomState(0)                          # fixed rng
hidden, T = 16, 30                                        # 16 hidden units, a 30-step sequence
Wxh = rng.randn(1, hidden)                                # input weights
W = rng.randn(hidden, hidden)                             # one random recurrent matrix...
W /= np.max(np.abs(np.linalg.eigvals(W)))                 # ...normalized to spectral radius exactly 1
bh = np.zeros(hidden)                                     # no bias, to keep it simple
X = rng.randn(50, T)                                       # 50 random sequences, inputs N(0, 1)
dh_last = rng.randn(50, hidden)                            # a stand-in gradient arriving at the last step

for rho in [0.5, 1.0, 3.0]:                               # the SAME matrix, scaled to three spectral radii
    Whh = W * rho                                           # scaling a matrix scales its spectral radius
    H = rnn_forward(X, np.zeros((50, hidden)), Wxh, Whh, bh)  # forward pass
    n = grad_norm_by_timestep(dh_last, H, Whh)                 # gradient norm at every timestep
    print(f"rho={rho}:  ||dh|| at last step {n[-1]:.3e}   10 steps back {n[-11]:.3e}   "
          f"at step 0 {n[0]:.3e}   ratio {n[0] / n[-1]:.2e}")
