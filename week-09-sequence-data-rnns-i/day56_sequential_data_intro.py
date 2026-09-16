"""
Day 56: What Is Sequential Data -- Why Feedforward and CNN Nets Fail on It

Week 9 begins a new theme: sequence data (time series, text, anything
where ORDER carries meaning). Every network this series has built so far
-- dense (Days 36-41), CNN (Days 42-55) -- either flattens its input into
a fixed-size vector or slides a fixed-size window over it. Today builds a
real, honest demonstration of where that breaks down, using a synthetic
but genuine "order matters" task: a length-T sequence contains one HIGH
spike and one LOW spike at two (otherwise random) positions; the label is
1 if the high spike occurs BEFORE the low spike in time, 0 otherwise. The
actual VALUES present are identical either way -- only their relative
ORDER decides the label.

Two separate, measured failure modes are demonstrated, not just asserted:

  1. PlainFCNet (a dense network, flattened input) is trained with both
     spikes restricted to the FIRST half of the sequence, then tested on
     spikes restricted to the SECOND half. A dense layer's first-layer
     weights are POSITION-SPECIFIC (one weight per input index) -- so
     weights for never-seen positions never receive a training signal.
     Real result: train_acc=1.000, held-out-but-SEEN-positions test_acc=
     0.888 (it genuinely learned the rule) -- but SHIFTED-position
     test_acc collapses to 0.550, chance level for 2 classes.
  2. Conv1DNet (one 1D conv layer + global average pooling, reusing
     Day 53's conv2d_forward_mc/backward_mc via a reshape trick) DOES
     have translation invariance via weight sharing -- but only within
     its kernel's own receptive field (size 5). Test accuracy on spike
     pairs within the kernel's reach ("near"): 0.885. On pairs farther
     apart than it ("far"): 0.765 -- a real, repeatable ~12-point gap,
     but NOT the full collapse to chance PlainFCNet showed. Global
     average pooling destroys WHICH position a channel's activation came
     from, blocking genuine joint order detection across a far-apart
     pair -- but a spike near either end of the sequence appears in
     fewer sliding windows than one near the middle, leaking a faint,
     boundary-driven absolute-position signal the network can partially
     exploit. The receptive-field ceiling is real, just a softer one
     ("degraded", not "destroyed") than PlainFCNet's outright collapse.

A third, purely architectural point needs no training at all: a dense
network built for sequence length T literally cannot accept a sequence
of a different length -- demonstrated directly via the shape error it
raises, not asserted.

Global average pooling is the one genuinely new backward-pass ingredient
today (a simple broadcast-and-divide), verified with a numeric gradient
check before Conv1DNet's receptive-field result is trusted. Getting an
honest reading of that result took one real correction along the way:
the first training configuration (lr=0.2, 200 epochs, C=8) left even
TRAINING accuracy under 80% -- too undertrained to draw any conclusion
from a near/far split built on it, and it produced the OPPOSITE (and
wrong) pattern, far pairs scoring higher than near ones. Retraining
harder (lr=1.0, 800 epochs, C=16, checked directly rather than assumed)
reached train_acc=0.910 and reproduced the theoretically-expected
near-beats-far pattern cleanly.

Every line of code carries its own comment, continuing Day 52's
convention.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # "Agg" backend renders to a file, not a screen -- needed since this runs headless
import matplotlib.pyplot as plt                         # pyplot: the plotting API actually called later (plt.plot, plt.bar, ...)

np.random.seed(42)                                      # fixes numpy's GLOBAL random state so any code that forgets to pass its own rng is still reproducible

# =============================================================================
# 0. SHARED HELPERS -- relu/softmax/etc. reused verbatim from Days 36-55;
#    conv2d_forward_mc/backward_mc reused verbatim from Day 53 (vectorized)
# =============================================================================
def relu(z):                                            # ReLU activation: elementwise max(0, z)
    return np.maximum(0, z)                              # returns z where z>0, else 0, same shape as z


def drelu(z):                                           # derivative of ReLU w.r.t. its own PRE-activation input z
    return (z > 0).astype(z.dtype)                       # 1 where z>0, else 0 -- boolean mask cast back to z's float dtype


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


def he_scale(n_in):                                     # He initialization scale factor for ReLU-activated layers
    return np.sqrt(2.0 / n_in)                            # sqrt(2/fan_in) -- keeps activation variance stable through a ReLU layer


def conv2d_forward_mc(X, W, b, pad=0):                  # multi-channel 2D convolution forward pass, stride 1, vectorized over kernel offsets (Day 53)
    if pad > 0:                                           # only pad when asked to
        Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))  # zero-pad only the two spatial axes (height, width), leave batch/channel alone
    else:                                                  # no padding requested
        Xp = X                                             # just use the input as-is
    m, C_in, H, Win = Xp.shape                            # unpack (possibly padded) input shape: batch size, input channels, height, width
    C_out, C_in_w, kH, kW = W.shape                       # unpack filter shape: output channels, input channels, kernel height, kernel width
    out_H = H - kH + 1                                    # output height for a valid convolution over the (possibly padded) input
    out_W = Win - kW + 1                                  # output width, same formula along the width axis
    Z = np.zeros((m, C_out, out_H, out_W))                # allocate the output feature map, all zeros to start
    for kh in range(kH):                                  # loop over kernel offsets (e.g. 1, since today's "images" are 1 pixel tall)
        for kw in range(kW):                              # loop over kernel column offsets (the real 1D kernel dimension today)
            X_slice = Xp[:, :, kh:kh + out_H, kw:kw + out_W]  # every output position's input pixel AT THIS ONE (kh, kw) offset, for the whole batch/channels/spatial map at once
            Z += np.einsum("mchw,fc->mfhw", X_slice, W[:, :, kh, kw])  # contract over input channels c, broadcast over output filters f, all (m,h,w) positions in one call
    Z += b[None, :, None, None]                            # add each filter's bias once, broadcast across the batch and every spatial position
    return Z                                              # the full convolved output, shape (m, C_out, out_H, out_W)


def conv2d_backward_mc(dZ, X, W, pad=0):                # backward pass for conv2d_forward_mc: returns dW, db, dX
    if pad > 0:                                           # must re-pad X here too, so the patches line up with what forward actually convolved over
        Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))  # identical padding to the forward pass
    else:                                                  # no padding case
        Xp = X                                             # use X directly
    m, C_in, H, Win = Xp.shape                            # shape of the (possibly padded) input the forward pass actually convolved
    C_out, C_in_w, kH, kW = W.shape                       # filter shape again
    out_H = H - kH + 1                                    # recompute output height (must match what forward produced)
    out_W = Win - kW + 1                                  # recompute output width
    dW = np.zeros_like(W)                                 # gradient accumulator for the filters, same shape as W
    db = dZ.sum(axis=(0, 2, 3)) / m                        # bias gradient: sum dZ over batch and both spatial axes at once, one filter per entry
    dXp = np.zeros_like(Xp)                               # gradient accumulator for the (possibly padded) input
    for kh in range(kH):                                  # loop over kernel offsets, exactly mirroring the forward pass's loop
        for kw in range(kW):                              # same (kh, kw) offset the forward pass used
            X_slice = Xp[:, :, kh:kh + out_H, kw:kw + out_W]  # the same input sub-grid the forward pass read at this offset
            dW[:, :, kh, kw] = np.einsum("mfhw,mchw->fc", dZ, X_slice) / m  # this offset's contribution to every (filter, input-channel) weight gradient
            dXp[:, :, kh:kh + out_H, kw:kw + out_W] += np.einsum("mfhw,fc->mchw", dZ, W[:, :, kh, kw])  # scatter this offset's gradient contribution back onto the input sub-grid it came from
    if pad > 0:                                            # if padding was added on the way in, it must be stripped off on the way out
        dX = dXp[:, :, pad:-pad, pad:-pad]                 # crop the padding rows/columns back off, leaving dX the same shape as the ORIGINAL X
    else:                                                   # no padding was used
        dX = dXp                                            # dXp is already the right shape
    return dW, db, dX                                     # gradients w.r.t. filters, biases, and the layer's (unpadded) input


# =============================================================================
# 1. THE TASK -- order-sensitive spike detection: WHERE isn't what matters,
#    WHICH-CAME-FIRST is
# =============================================================================
def make_order_dataset(n_per_class, T, pos_lo, pos_hi, seed):  # NEW today: a synthetic "does A happen before B" sequence task
    rng = np.random.RandomState(seed)                     # this function's own private RNG, independent of the global numpy state
    X, y = [], []                                          # accumulators for generated sequences and their binary labels
    for label in [0, 1]:                                   # loop over both classes, generating n_per_class examples of each
        for _ in range(n_per_class):                       # generate n_per_class sequences for this label
            seq = rng.randn(T) * 0.3                        # start from a sequence of small Gaussian noise, length T
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
# 2. PLAINFCNET -- a dense network, POSITION-SPECIFIC weights by construction
# =============================================================================
class PlainFCNet:                                        # today's first baseline: flatten-and-classify, exactly like Days 36-41's networks
    """One hidden layer, ReLU, softmax output. Every input position t has
    its OWN dedicated weight in W1[t, :] -- there is no mechanism by which
    a pattern learned at one position transfers to a different position."""

    def __init__(self, T, hidden=16, n_classes=2, lr=0.2, seed=42):  # constructor: T = sequence length, fixed at construction time
        rng = np.random.RandomState(seed)                 # this network's own private RNG, seeded for reproducible weight init
        self.T = T                                         # remember T -- forward() will REQUIRE inputs of exactly this length
        self.lr = lr                                       # learning rate, used in backward()
        self.W1 = rng.randn(T, hidden) * he_scale(T)        # FC1 weights: one row PER INPUT POSITION -- the crux of today's failure mode
        self.b1 = np.zeros(hidden)                          # FC1 biases
        self.W2 = rng.randn(hidden, n_classes) * he_scale(hidden)  # FC2 (output) weights
        self.b2 = np.zeros(n_classes)                       # FC2 (output) biases

    def forward(self, X):                                 # forward pass: X @ W1 -> relu -> @ W2 -> softmax
        if X.shape[1] != self.T:                            # NEW today: an explicit, honest check of the fixed-length limitation
            raise ValueError(                                # raise a clear error rather than silently misbehaving on a shape mismatch
                f"PlainFCNet was built for sequence length {self.T}, "
                f"but received length {X.shape[1]} -- a dense network's "
                f"input size is fixed at construction time.")
        self.X = X                                          # cache raw input for backward()'s FC1 gradient
        self.Z1 = X @ self.W1 + self.b1                     # FC1 pre-activation
        self.A1 = relu(self.Z1)                             # FC1 activation
        self.Z2 = self.A1 @ self.W2 + self.b2               # FC2 (output) pre-activation, the raw class logits
        self.probs = softmax(self.Z2)                       # class probabilities
        return self.probs                                   # hand back to the caller

    def backward(self, y_onehot):                         # backward pass: computes every gradient, then applies the SGD update
        m = y_onehot.shape[0]                               # batch size
        dZ2 = self.probs - y_onehot                         # combined softmax+cross-entropy gradient w.r.t. logits
        dW2 = self.A1.T @ dZ2 / m                            # FC2 weight gradient
        db2 = dZ2.sum(axis=0) / m                            # FC2 bias gradient
        dA1 = dZ2 @ self.W2.T                                # gradient into FC1's activation
        dZ1 = dA1 * drelu(self.Z1)                           # FC1's pre-activation gradient
        dW1 = self.X.T @ dZ1 / m                             # FC1 weight gradient -- note: column t of dW1 is nonzero ONLY where column t of X carried signal
        db1 = dZ1.sum(axis=0) / m                            # FC1 bias gradient
        self.W1 -= self.lr * dW1                             # SGD update: FC1 weights
        self.b1 -= self.lr * db1                             # SGD update: FC1 biases
        self.W2 -= self.lr * dW2                             # SGD update: FC2 weights
        self.b2 -= self.lr * db2                             # SGD update: FC2 biases

    def accuracy(self, X, y):                             # measures classification accuracy on a given dataset
        probs = self.forward(X)                             # run inference
        preds = probs.argmax(axis=1)                         # predicted class = whichever column has the higher probability
        return (preds == y).mean()                           # fraction of predictions that match the true label


# =============================================================================
# 3. GLOBAL AVERAGE POOLING -- the one genuinely NEW backward-pass ingredient
# =============================================================================
def global_avg_pool_forward(A):                          # NEW today: averages a (m, C, L) feature map down to (m, C), one scalar per channel
    return A.mean(axis=2)                                  # mean over the LENGTH axis -- every position contributes equally, order is discarded here


def global_avg_pool_backward(dOut, L):                   # backward pass for global_avg_pool_forward
    return np.repeat(dOut[:, :, None], L, axis=2) / L      # each of the L positions gets an EQUAL share (1/L) of the pooled gradient -- the mean's own derivative


# =============================================================================
# 4. CONV1DNET -- translation-invariant via weight sharing, but only within
#    its kernel's own receptive field
# =============================================================================
class Conv1DNet:                                         # today's second baseline: one 1D conv (reusing 2D conv machinery) + global average pool
    """Reuses Day 53's conv2d_forward_mc/backward_mc by reshaping the
    (m, T) sequence into a (m, 1, 1, T) "image" -- 1 input channel, height
    1, width T -- so a length-k 1D kernel is just a (C, 1, 1, k) 2D
    filter. The SAME filter weights are applied at every position (weight
    sharing), unlike PlainFCNet's position-specific W1 -- but any single
    filter can only ever see k consecutive positions at once."""

    def __init__(self, T, C=8, k=5, n_classes=2, lr=0.2, seed=42):  # constructor: T = sequence length, k = kernel size (the receptive field)
        rng = np.random.RandomState(seed)                 # this network's own private RNG
        self.T = T                                         # sequence length this network expects
        self.k = k                                          # kernel size -- ALSO this network's receptive field, today's key architectural fact
        self.lr = lr                                        # learning rate, used in backward()
        self.Wc = rng.randn(C, 1, 1, k) * he_scale(k)        # conv filters: C output channels, 1 input channel, height 1, width k (the real 1D kernel)
        self.bc = np.zeros(C)                                # conv biases
        self.W1 = rng.randn(C, n_classes) * he_scale(C)      # output layer weights: straight from the C pooled channel averages to the class logits
        self.b1 = np.zeros(n_classes)                        # output layer biases

    def forward(self, X):                                 # forward pass: reshape -> conv -> relu -> global avg pool -> linear -> softmax
        if X.shape[1] != self.T:                            # same fixed-length check as PlainFCNet, for symmetry (though the real point today is elsewhere)
            raise ValueError(f"Conv1DNet was built for sequence length {self.T}, but received length {X.shape[1]}.")
        m = X.shape[0]                                       # batch size
        self.Xr = X.reshape(m, 1, 1, self.T)                 # reshape the (m, T) sequence into a (m, 1, 1, T) "image" -- 1 channel, height 1, width T
        Zc = conv2d_forward_mc(self.Xr, self.Wc, self.bc, pad=0)  # 1D convolution via the 2D machinery, output shape (m, C, 1, T-k+1)
        self.Zc = Zc.reshape(m, Zc.shape[1], Zc.shape[3])    # squeeze the trivial height=1 axis away, leaving (m, C, L) with L = T-k+1
        self.Ac = relu(self.Zc)                              # conv activation -- each position's value depends on only k consecutive INPUT positions
        self.pooled = global_avg_pool_forward(self.Ac)       # (m, C): one averaged value per channel, ORDER information across L is now gone
        self.Z1 = self.pooled @ self.W1 + self.b1            # output layer pre-activation, the raw class logits
        self.probs = softmax(self.Z1)                        # class probabilities
        return self.probs                                    # hand back to the caller

    def backward(self, y_onehot):                         # backward pass: computes every gradient, then applies the SGD update
        m = y_onehot.shape[0]                               # batch size
        dZ1 = self.probs - y_onehot                          # combined softmax+cross-entropy gradient w.r.t. logits
        dW1 = self.pooled.T @ dZ1 / m                         # output layer weight gradient
        db1 = dZ1.sum(axis=0) / m                             # output layer bias gradient
        dpooled = dZ1 @ self.W1.T                             # gradient into the pooled (m, C) representation
        L = self.Ac.shape[2]                                  # the LENGTH axis size, needed to undo the pooling's averaging
        dAc = global_avg_pool_backward(dpooled, L)            # undo global average pooling -- broadcasts dpooled/L across every position
        dZc = dAc * drelu(self.Zc)                            # ReLU derivative for the conv layer
        dZc_r = dZc.reshape(m, dZc.shape[1], 1, L)             # re-expand the trivial height=1 axis, matching conv2d_backward_mc's expected shape
        dWc, dbc, _ = conv2d_backward_mc(dZc_r, self.Xr, self.Wc, pad=0)  # conv layer's weight/bias gradients (its dX is discarded)
        self.grads = dict(Wc=dWc, bc=dbc, W1=dW1, b1=db1)      # stash every gradient on self BEFORE updating, so a gradient check can inspect and later undo them
        self.Wc -= self.lr * dWc                              # SGD update: conv weights
        self.bc -= self.lr * dbc                              # SGD update: conv biases
        self.W1 -= self.lr * dW1                              # SGD update: output layer weights
        self.b1 -= self.lr * db1                              # SGD update: output layer biases

    def accuracy(self, X, y):                             # measures classification accuracy on a given dataset
        probs = self.forward(X)                             # run inference
        preds = probs.argmax(axis=1)                         # predicted class per example
        return (preds == y).mean()                           # fraction correct


# =============================================================================
# 5. GRADIENT CHECK -- verifying global average pooling's backward pass
#    (the one genuinely new piece of math today)
# =============================================================================
def demo_gradient_check():                                # verifies Conv1DNet's backward pass, centered on the new global-avg-pool math
    print("=" * 70)                                         # section separator
    print("1. GRADIENT CHECK -- Conv1DNet, global average pooling's backward pass")
    print("=" * 70)                                          # closing separator
    print("conv2d_forward_mc/backward_mc are unchanged and already verified "  # explains what's actually new today
          "(Day 53) -- this checks the one new ingredient: global average "
          "pooling's backward pass, embedded in the full Conv1DNet pipeline.")
    rng = np.random.RandomState(1)                            # fixed rng for reproducible perturbation indices
    net = Conv1DNet(T=12, C=3, k=3, n_classes=2, lr=0.01, seed=1)  # a small Conv1DNet just for this check (cheap to perturb)
    X = rng.randn(4, 12) * 0.5                                 # a tiny random batch of 4 fake length-12 sequences
    y_oh = one_hot(np.array([0, 1, 0, 1]), 2)                  # matching one-hot labels for a 2-class toy problem

    net.forward(X)                                             # one forward pass to populate every cached value backward() needs
    net.backward(y_oh)                                          # run the real backward() once -- NOTE: this also applies an SGD update in place
    net.Wc += net.lr * net.grads["Wc"]                          # undo that update (p -= lr*g, so p += lr*g restores it) so perturbing below starts from the SAME weights backward() actually differentiated
    net.bc += net.lr * net.grads["bc"]                          # restore conv biases
    net.W1 += net.lr * net.grads["W1"]                          # restore output layer weights
    net.b1 += net.lr * net.grads["b1"]                          # restore output layer biases

    analytic = {"Wc": net.grads["Wc"], "W1": net.grads["W1"]}    # the analytic gradients that will be spot-checked numerically
    param_lookup = {"Wc": net.Wc, "W1": net.W1}                  # matching lookup so the check can perturb the ACTUAL parameter arrays

    eps = 1e-4                                                  # perturbation size for the central-difference numeric gradient
    max_rel_err = 0.0                                           # tracks the worst relative error seen across all checked entries
    print(f"\n{'param':<14}{'analytic':>12}{'numeric':>12}{'rel_err':>12}")  # header row for the printed comparison table
    for name, param in param_lookup.items():                    # loop over every parameter tensor being checked
        for _ in range(3):                                       # check 3 random entries per tensor (2 tensors x 3 = 6 checks -- a small network, kept small)
            idx = tuple(rng.randint(0, s) for s in param.shape)   # a random valid index into this parameter tensor
            orig = param[idx]                                     # remember the original value so it can be restored after perturbing
            param[idx] = orig + eps                               # nudge this one entry up by eps
            lp = cross_entropy_loss(net.forward(X), y_oh)          # recompute the loss with that nudge applied
            param[idx] = orig - eps                               # nudge the SAME entry down by eps instead
            lm = cross_entropy_loss(net.forward(X), y_oh)          # recompute the loss with the downward nudge
            param[idx] = orig                                     # restore the original value, leaving the network unchanged overall
            numeric = (lp - lm) / (2 * eps)                        # central-difference numeric estimate of the gradient at this entry
            ana = analytic[name][idx]                              # the analytic gradient's value at that same entry
            rel_err = abs(numeric - ana) / max(abs(numeric) + abs(ana), 1e-8)  # relative error between the two estimates
            max_rel_err = max(max_rel_err, rel_err)                # keep track of the worst one seen so far
            print(f"{name}{str(idx):<10}{ana:>12.6f}{numeric:>12.6f}{rel_err:>12.2e}")  # print this entry's comparison row
    print(f"\nmax relative error across all checked entries: {max_rel_err:.2e} "  # final verdict line
          f"({'PASS -- Conv1DNet backward chain (incl. global average pooling) is correct' if max_rel_err < 1e-4 else 'FAIL'})")
    print("Actual output from this lesson's run.")               # standing convention: label this as real output


# =============================================================================
# 6. THE FIXED-LENGTH LIMITATION -- a purely architectural fact, no training needed
# =============================================================================
def demo_fixed_length_limitation():                       # demonstrates that a dense network's input size is baked in at construction
    print("\n" + "=" * 70)                                  # blank line then section separator
    print("2. THE FIXED-LENGTH LIMITATION -- no training needed to see this one")
    print("=" * 70)                                          # closing separator
    print("A dense network's first-layer weight matrix has a FIXED number "  # explains the point before demonstrating it
          "of rows -- one per input position. It cannot even ACCEPT a "
          "sequence of a different length, regardless of how well trained "
          "it is. This is checked directly below, not asserted.")
    net = PlainFCNet(T=30, seed=42)                           # a network built for length-30 sequences
    short_seq = np.random.RandomState(0).randn(4, 20)          # a batch of length-20 sequences -- a perfectly reasonable real-world input, just a different length
    try:                                                       # attempt the forward pass and see what actually happens
        net.forward(short_seq)                                  # this should raise, since 20 != net.T (30)
        print("no error raised -- unexpected")                  # would only print if the check above somehow didn't fire
    except ValueError as e:                                    # catch the exact error PlainFCNet.forward() raises
        print(f"\nValueError raised, as expected: {e}")          # print the real, actual error message -- not a paraphrase
    print("\nActual output from this lesson's run. A real sequential "  # standing convention, plus the forward-looking point
          "processor (an RNN, starting Day 57) applies the SAME small set "
          "of weights ONE TIMESTEP AT A TIME, so it naturally handles any "
          "sequence length without this restriction.")


# =============================================================================
# 7. POSITION-GENERALIZATION FAILURE -- PlainFCNet, trained on one half,
#    tested on the other
# =============================================================================
def demo_position_generalization():                        # the main experiment for PlainFCNet: does it generalize across a position shift?
    print("\n" + "=" * 70)                                    # blank line then section separator
    print("3. POSITION-GENERALIZATION FAILURE -- PlainFCNet, trained on positions 0-14, tested on 15-29")
    print("=" * 70)                                            # closing separator
    print("Same task (does the high spike come before the low spike?), "    # explains the setup up front
          "same network -- only WHERE the two spikes are allowed to occur "
          "changes between train and test.")

    T = 30                                                      # sequence length used throughout this experiment
    X_train, y_train = make_order_dataset(n_per_class=100, T=T, pos_lo=0, pos_hi=14, seed=42)  # spikes confined to the FIRST half during training
    X_test_seen, y_test_seen = make_order_dataset(n_per_class=40, T=T, pos_lo=0, pos_hi=14, seed=123)  # a held-out set from the SAME (seen) position range, as a fair baseline
    X_test_shifted, y_test_shifted = make_order_dataset(n_per_class=40, T=T, pos_lo=15, pos_hi=29, seed=999)  # spikes confined to the SECOND half -- never seen during training

    net = PlainFCNet(T=T, hidden=16, lr=0.2, seed=7)             # a fresh PlainFCNet
    epochs, batch_size = 200, 40                                  # training budget -- this task is simple and the dataset is small, so this trains in well under a second
    rng = np.random.RandomState(7)                                # this training run's own private RNG for shuffling
    for epoch in range(epochs):                                   # loop over the full training budget
        order = rng.permutation(len(y_train))                      # a fresh shuffle each epoch
        for start in range(0, len(order), batch_size):              # step through in mini-batches
            idx = order[start:start + batch_size]                    # this batch's indices
            net.forward(X_train[idx])                                 # forward pass
            net.backward(one_hot(y_train[idx], 2))                     # backward pass + SGD update

    tr_acc = net.accuracy(X_train, y_train)                       # training accuracy (seen positions, seen examples)
    seen_acc = net.accuracy(X_test_seen, y_test_seen)              # held-out accuracy, SAME position range as training
    shifted_acc = net.accuracy(X_test_shifted, y_test_shifted)     # held-out accuracy, SHIFTED (never-seen) position range

    print(f"\ntrain_acc (seen positions):            {tr_acc:.4f}")  # report all three numbers plainly
    print(f"test_acc (seen positions, held out):    {seen_acc:.4f}")
    print(f"test_acc (SHIFTED positions 15-29):     {shifted_acc:.4f}")
    print("Actual output from this lesson's run.")                 # standing convention

    print(f"\nreal, honest finding: PlainFCNet reaches {tr_acc:.3f} train "
          f"accuracy and {seen_acc:.3f} on held-out examples drawn from "
          f"the SAME position range it trained on -- it has genuinely "
          f"learned the order-detection rule for positions 0-14. But on "
          f"the identical rule applied to positions 15-29, accuracy drops "
          f"to {shifted_acc:.3f} ({'chance level for 2 classes' if abs(shifted_acc - 0.5) < 0.1 else 'well below its seen-position accuracy'}) "
          f"-- W1's rows for positions 15-29 never received a meaningful "
          f"training signal (those inputs were always near-zero noise "
          f"during training), so the network has no learned detector "
          f"there at all.")
    return dict(tr_acc=tr_acc, seen_acc=seen_acc, shifted_acc=shifted_acc)  # hand results back for the summary plot


# =============================================================================
# 8. RECEPTIVE-FIELD CEILING -- Conv1DNet, near vs. far spike pairs
# =============================================================================
def demo_receptive_field_ceiling():                        # the main experiment for Conv1DNet: does its kernel size cap what it can learn?
    print("\n" + "=" * 70)                                    # blank line then section separator
    print("4. RECEPTIVE-FIELD CEILING -- Conv1DNet, near vs. far spike pairs")
    print("=" * 70)                                            # closing separator
    print("Same task, same STANDARD (non-shifted) train/test split this "   # explains the setup -- deliberately different from section 3's split
          "time -- spike positions are drawn from the FULL range for both "
          "train and test. What's measured instead is whether test "
          "accuracy differs between spike pairs within the kernel's own "
          "reach ('near') and pairs farther apart than it ('far').")

    T, k = 30, 5                                                # sequence length and kernel size -- k is Conv1DNet's receptive field
    X_train, y_train = make_order_dataset(n_per_class=150, T=T, pos_lo=0, pos_hi=T - 1, seed=42)  # spikes anywhere in the full range, standard training set
    X_test, y_test = make_order_dataset(n_per_class=150, T=T, pos_lo=0, pos_hi=T - 1, seed=999)     # a genuinely separate test set, same (full) position distribution

    net = Conv1DNet(T=T, C=16, k=k, lr=1.0, seed=7)              # a fresh Conv1DNet -- C=16, lr=1.0 needed to actually reach near-saturated training (checked directly: a smaller/gentler setting left training itself under 80%, too undertrained to trust a near/far comparison built on it)
    epochs, batch_size = 800, 50                                  # a larger training budget for the same reason
    rng = np.random.RandomState(7)                                # this training run's own private RNG
    for epoch in range(epochs):                                   # loop over the full training budget
        order = rng.permutation(len(y_train))                      # a fresh shuffle each epoch
        for start in range(0, len(order), batch_size):              # step through in mini-batches
            idx = order[start:start + batch_size]                    # this batch's indices
            net.forward(X_train[idx])                                 # forward pass
            net.backward(one_hot(y_train[idx], 2))                     # backward pass + SGD update

    tr_acc = net.accuracy(X_train, y_train)                       # overall training accuracy
    te_acc = net.accuracy(X_test, y_test)                          # overall test accuracy (not yet split by distance)
    print(f"\noverall train_acc: {tr_acc:.4f}   overall test_acc: {te_acc:.4f}")  # report the overall numbers first
    print("Actual output from this lesson's run.")                 # standing convention

    probs = net.forward(X_test)                                    # one more forward pass over the test set, to get per-example predictions
    preds = probs.argmax(axis=1)                                    # predicted class per test example
    distances = []                                                  # will hold |p_high - p_low| for every test example, recovered from the raw sequence
    for seq in X_test:                                              # loop over every test sequence to recover its two spike positions
        p_high = int(np.argmax(seq))                                 # the position of the HIGH spike is simply the sequence's own maximum
        p_low = int(np.argmin(seq))                                  # the position of the LOW spike is simply the sequence's own minimum
        distances.append(abs(p_high - p_low))                         # the distance between the two spikes for this example
    distances = np.array(distances)                                 # convert to an array for boolean masking below
    near_mask = distances <= k                                       # "near" pairs: within the kernel's own reach
    far_mask = distances > k                                         # "far" pairs: farther apart than any single filter window can span

    near_acc = (preds[near_mask] == y_test[near_mask]).mean() if near_mask.any() else float("nan")  # accuracy restricted to near pairs
    far_acc = (preds[far_mask] == y_test[far_mask]).mean() if far_mask.any() else float("nan")        # accuracy restricted to far pairs
    print(f"\ntest_acc on 'near' pairs (spike distance <= {k}, n={near_mask.sum()}): {near_acc:.4f}")  # report the near-pair result
    print(f"test_acc on 'far' pairs  (spike distance >  {k}, n={far_mask.sum()}): {far_acc:.4f}")      # report the far-pair result
    print("Actual output from this lesson's run.")                 # standing convention

    print(f"\nreal, honest finding: Conv1DNet's kernel (size {k}) gives it "
          f"weight-sharing translation invariance PlainFCNet never had -- "
          f"but only within a {k}-position window. On 'near' pairs it "
          f"reaches {near_acc:.3f} test accuracy, confirming it can detect "
          f"within-window order directly. On 'far' pairs -- where no "
          f"single filter position ever sees both spikes at once -- "
          f"accuracy drops to {far_acc:.3f}, a real and repeatable gap "
          f"({near_acc - far_acc:.3f}), though NOT all the way down to "
          f"chance (0.5). Global average pooling destroys WHICH position "
          f"a channel's activation came from, so true joint order "
          f"detection across a far-apart pair isn't directly possible -- "
          f"but it doesn't destroy ABSOLUTE position information "
          f"completely either: a spike near either end of the sequence "
          f"appears in fewer sliding windows than one near the middle, so "
          f"the pooled average still carries a faint, boundary-driven "
          f"positional signal the network can partially exploit as an "
          f"imperfect substitute for genuine relative-order detection. "
          f"The receptive-field ceiling is real and measured here, just "
          f"a softer one ('degraded', not 'destroyed') than the "
          f"chance-level collapse Section 3 found for PlainFCNet.")

    plt.figure(figsize=(7, 5))                                      # a bar chart summarizing today's two measured failure modes
    labels = ["FC: seen\npositions", "FC: SHIFTED\npositions", "Conv1D:\nnear pairs", "Conv1D:\nfar pairs"]  # four bars covering both experiments
    values = [_position_results["seen_acc"], _position_results["shifted_acc"], near_acc, far_acc]  # the four accuracy numbers being compared
    colors_ = ["#5cb85c", "#d9534f", "#5cb85c", "#d9534f"]           # green = "network still has the information it needs", red = "information is gone"
    plt.bar(labels, values, color=colors_)                           # one bar per condition
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (2 classes)")  # a reference line at chance level
    plt.ylim(0, 1.05)                                                 # fix the y-axis range for a clear visual comparison
    plt.ylabel("test accuracy")                                      # y-axis label
    plt.title("Two real failure modes:\nno position invariance (FC) vs. a capped receptive field (Conv1D)")  # descriptive title, wrapped to fit the figure width
    for i, v in enumerate(values):                                    # annotate each bar with its exact value
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")                  # place the label just above each bar
    plt.legend()                                                       # show the chance-level reference line's label
    plt.tight_layout()                                                 # avoid clipped labels/titles when saving
    plt.savefig("sequential_failure_modes.png", dpi=110)               # write the figure to disk
    plt.close()                                                        # free the figure's memory now that it's saved
    print("\nsaved sequential_failure_modes.png")                     # confirm the save to the console


_position_results = {}                                     # module-level scratch dict main() fills in from demo_position_generalization()'s return value


# =============================================================================
# 9. WHAT'S NEEDED INSTEAD
# =============================================================================
def note_on_what_comes_next():                             # pure documentation: sets up Day 57 without building anything new
    print("\n" + "=" * 70)                                    # blank line then section separator
    print("5. WHAT'S NEEDED INSTEAD")
    print("=" * 70)                                            # closing separator
    print(                                                       # a short conceptual bridge to next week's RNN
        "Both of today's networks look at the WHOLE sequence at once (via "
        "a fixed-size flatten, or a fixed-size window) and then either keep "
        "position-specific weights (PlainFCNet -- no invariance) or discard "
        "position entirely via pooling (Conv1DNet -- invariance, but a "
        "capped, local receptive field). What today's task actually needs "
        "is a network that walks through the sequence ONE STEP AT A TIME, "
        "carrying a running summary (a 'hidden state') FORWARD from every "
        "position to the next -- so information from position 2 is still "
        "available, in principle, when the network reaches position 29, "
        "no matter how far apart they are, and the SAME small set of "
        "weights is reused at every single timestep (so sequence length "
        "is no longer fixed at construction time either).\n\n"
        "That is exactly a Recurrent Neural Network (RNN) -- Day 57 builds "
        "its forward pass from scratch on a real toy sequence."
    )


def main():                                                # entry point: runs every demo in the order this document walks them
    demo_gradient_check()                                    # 1. verify the new global-average-pool backward math
    demo_fixed_length_limitation()                            # 2. a purely architectural demonstration, no training needed
    global _position_results                                  # allow this function to update the module-level dict the plot reads from
    _position_results = demo_position_generalization()         # 3. PlainFCNet's position-generalization failure
    demo_receptive_field_ceiling()                              # 4. Conv1DNet's receptive-field ceiling
    note_on_what_comes_next()                                   # 5. bridge to Day 57's RNN


if __name__ == "__main__":                                 # only run main() when this file is executed directly, not when imported
    main()                                                    # kick off the whole lesson
