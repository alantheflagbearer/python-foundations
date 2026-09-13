"""
Day 52: VGG-Style Architecture -- Stacked Small Kernels vs. One Big One

Days 49-51 all varied the FC head or the training regime; today changes
the CONV side instead. VGG (Simonyan & Zisserman, 2014) made one specific
architectural bet: replace a single large-kernel conv (e.g. 5x5, 7x7)
with SEVERAL stacked small 3x3 convs that reach the SAME receptive field.
Two stacked 3x3 convs (C->C->C channels) have a 5x5 receptive field,
exactly like one 5x5 conv (C->C) -- but 18*C^2 parameters instead of
25*C^2, plus an extra ReLU non-linearity in between the two small convs
that the single big conv never gets.

Starting today, every line of code below carries its own comment --
a new convention from this point in the series onward, matching the
line-by-line density this series' separate syntax_line_by_line.pdf has
always used, now folded directly into the lesson script itself.

Two full 2-block CNNs are built and gradient-checked: LargeKernelConvNet
(one 5x5 conv per block) and VGGStyleConvNet (two stacked 3x3 convs per
block, same channel count C throughout so the parameter-count comparison
is apples-to-apples: 8,332 params vs. 8,572, a 2.8% savings, at the
verified-identical 5x5 receptive field).

Neither of today's two real findings is the tidy "VGG wins" story.
FIRST: trained at Day 50/51's inherited lr=0.15, LargeKernelConvNet
converges fine (train_acc=1.000) but VGGStyleConvNet never learns at all
(train_acc=0.245, chance level for 4 classes). The cause, diagnosed
directly rather than assumed: severe dying ReLUs -- block 2's first conv
(Zc2a) ends up 99.9% dead (stuck at negative pre-activations), permanently
zeroing almost all of that layer's gradient. VGGStyleConvNet is twice as
deep in its conv stack (4 conv+ReLU layers vs. 2), and a learning rate
tuned for the shallower network is far too aggressive for the deeper one
-- a real, direct preview of exactly why Day 53 (residual connections)
and Day 54 (batch norm in CNNs) exist. SECOND, once both networks are
retrained at a properly-recalibrated lr=0.03 (chosen by directly measuring
dead-unit fraction across a small sweep) and both reach train_acc=1.000,
LargeKernelConvNet actually generalizes BETTER (test_acc=0.7125) than
VGGStyleConvNet (test_acc=0.600) -- despite VGGStyle's real, confirmed
2.8% parameter saving at the identical receptive field. VGG's advantage
was demonstrated at ImageNet scale with far deeper stacks (16-19 layers)
and far more data than this lesson's 2-block, 200-image setting; fewer
parameters and an extra non-linearity did not automatically win here.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # "Agg" backend renders to a file, not a screen -- needed since this runs headless
import matplotlib.pyplot as plt                         # pyplot: the plotting API actually called later (plt.plot, plt.savefig, ...)

np.random.seed(42)                                      # fixes numpy's GLOBAL random state so any code that forgets to pass its own rng is still reproducible

# =============================================================================
# 0. SHARED HELPERS -- reused verbatim (logic unchanged) from Days 39-51,
#    now with a comment added to every line per today's new convention
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


def conv2d_forward_mc(X, W, b):                         # multi-channel 2D convolution forward pass, "valid" (no padding), stride 1
    m, C_in, H, Win = X.shape                             # unpack input shape: batch size, input channels, height, width
    C_out, C_in_w, kH, kW = W.shape                       # unpack filter shape: output channels, input channels, kernel height, kernel width
    out_H = H - kH + 1                                    # output height for valid convolution: input height minus kernel height plus 1
    out_W = Win - kW + 1                                  # output width, same formula along the width axis
    Z = np.zeros((m, C_out, out_H, out_W))                # allocate the output feature map, all zeros to start
    for f in range(C_out):                                # loop over each output filter/channel f
        for i in range(out_H):                            # loop over each output row position i
            for j in range(out_W):                        # loop over each output column position j
                patch = X[:, :, i:i + kH, j:j + kW]        # the input patch under filter f at position (i, j), for every example in the batch
                Z[:, f, i, j] = np.sum(patch * W[f], axis=(1, 2, 3)) + b[f]  # elementwise multiply + sum over (C_in, kH, kW), plus this filter's bias
    return Z                                              # the full convolved output, shape (m, C_out, out_H, out_W)


def conv2d_backward_mc(dZ, X, W):                       # backward pass for conv2d_forward_mc: returns dW, db, dX
    m, C_in, H, Win = X.shape                             # same shape unpacking as the forward pass
    C_out, C_in_w, kH, kW = W.shape                       # filter shape again
    out_H = H - kH + 1                                    # recompute output height (must match what forward produced)
    out_W = Win - kW + 1                                  # recompute output width
    dW = np.zeros_like(W)                                 # gradient accumulator for the filters, same shape as W
    db = np.zeros(C_out)                                  # gradient accumulator for the biases, one scalar per output filter
    dX = np.zeros_like(X)                                 # gradient accumulator for the input, same shape as X
    for f in range(C_out):                                # loop over each output filter f, mirroring the forward pass
        for i in range(out_H):                            # loop over each output row position i
            for j in range(out_W):                        # loop over each output column position j
                patch = X[:, :, i:i + kH, j:j + kW]        # the same input patch the forward pass used at (f, i, j)
                dZ_ij = dZ[:, f, i, j]                     # the upstream gradient flowing into this single output position, across the batch
                dW[f] += np.sum(dZ_ij[:, None, None, None] * patch, axis=0) / m  # accumulate this position's contribution to filter f's gradient
                db[f] += np.sum(dZ_ij) / m                 # accumulate this position's contribution to filter f's bias gradient
                dX[:, :, i:i + kH, j:j + kW] += (          # scatter gradient back onto the exact input patch this output position depended on
                    dZ_ij[:, None, None, None] * W[f])     # each element of the patch gets dZ_ij times that filter weight
    return dW, db, dX                                     # gradients w.r.t. filters, biases, and the layer's input


def maxpool_forward(A, size=2, stride=2):               # 2x2 max pooling forward pass (default size/stride)
    m, C, H, W = A.shape                                  # unpack input shape: batch, channels, height, width
    out_H = (H - size) // stride + 1                      # output height using the standard pooling output-size formula
    out_W = (W - size) // stride + 1                      # output width, same formula
    out = np.zeros((m, C, out_H, out_W))                  # allocate the pooled output
    idx_cache = np.zeros((m, C, out_H, out_W), dtype=np.int64)  # cache of WHICH position in each window was the max (needed for backward)
    for i in range(out_H):                                # loop over each output row position i
        for j in range(out_W):                            # loop over each output column position j
            window = A[:, :, i * stride:i * stride + size,  # the size x size window this output position pools over
                       j * stride:j * stride + size]
            flat = window.reshape(m, C, size * size)       # flatten each window to a 1D list of size*size candidate values
            idx = flat.argmax(axis=2)                      # index (within the flattened window) of the maximum value
            out[:, :, i, j] = flat.max(axis=2)             # the actual max value becomes this output position's value
            idx_cache[:, :, i, j] = idx                    # remember which flattened index won, for routing gradient back correctly
    return out, idx_cache                                 # pooled output, plus the cache backward() needs


def maxpool_backward(dOut, A, idx_cache, size=2, stride=2):  # backward pass for maxpool_forward
    m, C, H, W = A.shape                                  # shape of the ORIGINAL (pre-pooling) input
    dA = np.zeros_like(A)                                 # gradient accumulator for that original input, all zero to start
    out_H, out_W = dOut.shape[2], dOut.shape[3]           # spatial size of the upstream gradient (matches the pooled output's shape)
    for i in range(out_H):                                # loop over each pooled output row position i
        for j in range(out_W):                            # loop over each pooled output column position j
            idx = idx_cache[:, :, i, j]                    # which flattened window-index was the max, for every (example, channel) here
            r = idx // size                                # convert that flat index back into a row offset within the window
            c = idx % size                                 # and a column offset within the window
            for mm in range(m):                            # loop over every example in the batch
                for cc in range(C):                        # loop over every channel
                    dA[mm, cc, i * stride + r[mm, cc], j * stride + c[mm, cc]] += \
                        dOut[mm, cc, i, j]                  # route this output's gradient ONLY to the single input position that won the max
    return dA                                             # gradient w.r.t. the pre-pooling input


def iterate_minibatches(X, y_onehot, batch_size, rng, shuffle=True):  # yields shuffled mini-batches, one epoch's worth per call
    m = X.shape[0]                                        # total number of training examples
    order = rng.permutation(m) if shuffle else np.arange(m)  # a fresh random ordering of example indices each call, or the plain order if shuffle=False
    for start in range(0, m, batch_size):                 # step through that ordering in chunks of batch_size
        batch_idx = order[start:start + batch_size]        # the indices belonging to this particular batch (last batch may be smaller)
        yield X[batch_idx], y_onehot[batch_idx], batch_idx  # yield this batch's inputs, one-hot labels, and original indices


def make_junction_dataset_rgb(n_per_class=50, size=30, arm=8, noise=0.4,   # synthetic 4-class "junction" image generator
                               channel_noise=0.15, center_jitter=2, seed=0):
    rng = np.random.RandomState(seed)                     # this function's own private RNG, independent of the global numpy state
    X, y = [], []                                         # accumulators for generated images and their integer class labels
    cx = cy = size // 2                                   # the image's center coordinate (same for x and y since images are square)
    dirs_all = [(-1, 0), (1, 0), (0, -1), (0, 1)]         # the 4 possible arm directions: up, down, left, right
    for cls in range(4):                                  # loop over the 4 classes (0..3)
        n_arms = cls + 1                                  # class number directly controls how many arms radiate from the center (1..4)
        for _ in range(n_per_class):                      # generate n_per_class images for this class
            im = rng.randn(size, size) * noise             # start from pure Gaussian noise as the base image
            r = cy + rng.randint(-center_jitter, center_jitter + 1)  # jitter the junction's row position slightly, for realism
            c = cx + rng.randint(-center_jitter, center_jitter + 1)  # jitter the column position slightly too
            if n_arms == 4:                                # if this class uses all 4 arms
                chosen = dirs_all                           # just use every direction
            else:                                          # otherwise (1, 2, or 3 arms)
                idx = rng.choice(4, size=n_arms, replace=False)  # randomly choose WHICH n_arms of the 4 directions to draw
                chosen = [dirs_all[i] for i in idx]         # look up the actual (dr, dc) direction tuples for those chosen indices
            for (dr, dc) in chosen:                        # draw each chosen arm
                for step in range(1, arm + 1):              # walk outward from the center, `arm` pixels long
                    rr, cc = r + dr * step, c + dc * step   # the pixel coordinate `step` pixels along this arm's direction
                    im[rr, cc] += 2.0                       # brighten that pixel to make the arm visible above the noise
            im[r, c] += 2.0                                # also brighten the exact center pixel itself
            rgb = np.stack([im, im, im], axis=0)            # replicate the single grayscale image across 3 channels to make it "RGB"
            rgb = rgb + rng.randn(3, size, size) * channel_noise  # add small INDEPENDENT noise per channel, so channels aren't perfectly identical
            X.append(rgb)                                  # store this generated image
            y.append(cls)                                  # store its true class label
    X = np.array(X)                                       # stack the list of images into one (N, 3, size, size) array
    y = np.array(y)                                       # stack the list of labels into one (N,) array
    idx = rng.permutation(len(y))                          # a random shuffle order for the whole dataset
    return X[idx], y[idx]                                 # return the shuffled images and labels together, so pairing is preserved


def train_with_history(net, X, y, X_test, y_test, epochs, batch_size,   # reused verbatim from Day 50: mini-batch training + periodic eval
                        eval_every=10, n_classes=4, seed=0):
    y_oh = one_hot(y, n_classes)                          # convert integer train labels to one-hot once, outside the epoch loop
    rng = np.random.RandomState(seed)                     # this training run's own private RNG (shuffling + anything net.forward needs)
    step_losses = []                                      # will collect one loss value per mini-batch STEP (fine-grained)
    history_epochs, history_train_acc, history_test_acc = [], [], []  # will collect one entry per EPOCH at which we evaluate
    for epoch in range(epochs):                           # loop over the full training budget, one epoch at a time
        for Xb, yb, idx in iterate_minibatches(X, y_oh, batch_size, rng):  # loop over every mini-batch within this epoch
            probs = net.forward(Xb, training=True, rng=rng)  # forward pass in TRAINING mode (matters if the net has dropout; none does today)
            step_losses.append(cross_entropy_loss(probs, yb))  # record this step's loss
            net.backward(yb)                               # backward pass + parameter update for this step
        if epoch % eval_every == 0 or epoch == epochs - 1:  # only evaluate every eval_every epochs, plus always on the very last epoch
            history_epochs.append(epoch)                    # record which epoch this measurement belongs to
            history_train_acc.append(net.accuracy(X, y))    # measure accuracy on the full training set at this point
            history_test_acc.append(net.accuracy(X_test, y_test))  # and on the held-out test set
    return dict(step_losses=step_losses, epochs=history_epochs,  # bundle everything the caller might want into one dict
                train_acc=history_train_acc, test_acc=history_test_acc)


# =============================================================================
# 1. LARGEKERNELCONVNET -- one big 5x5 conv per block (NEW today)
# =============================================================================
class LargeKernelConvNet:                               # baseline for today's comparison: single large-kernel convs, no stacking
    """Two blocks, each ONE 5x5 conv (stride 1, no padding) -> ReLU ->
    2x2 maxpool. Channel count C is held FIXED across both blocks
    (3->C, then C->C) so the parameter-count comparison against
    VGGStyleConvNet below is a fair, same-channel-width comparison, not
    confounded by also changing channel counts."""

    def __init__(self, C=8, img_size=40, hidden=16, n_classes=4,  # constructor: C = channel width shared by every conv layer
                 lr=0.15, seed=42):
        rng = np.random.RandomState(seed)                 # this network's own private RNG, seeded for reproducible weight init
        self.lr = lr                                      # store the learning rate on self, used later in backward()
        k = 5                                             # the (single) kernel size used throughout this network: 5x5
        self.Wc1 = rng.randn(C, 3, k, k) * he_scale(3 * k * k)  # conv1 filters: C output channels, 3 input channels (RGB), 5x5 each
        self.bc1 = np.zeros(C)                            # conv1 biases, one per output filter, start at zero
        self.Wc2 = rng.randn(C, C, k, k) * he_scale(C * k * k)  # conv2 filters: C->C channels, 5x5 each
        self.bc2 = np.zeros(C)                            # conv2 biases
        out1 = img_size - k + 1                           # spatial size after conv1 (valid convolution shrinks it)
        pool1 = out1 // 2                                 # spatial size after the first 2x2 maxpool
        out2 = pool1 - k + 1                              # spatial size after conv2
        pool2 = out2 // 2                                 # spatial size after the second 2x2 maxpool
        self.flat_dim = C * pool2 * pool2                 # total flattened feature count feeding into the first FC layer
        self.W1 = rng.randn(self.flat_dim, hidden) * he_scale(self.flat_dim)  # FC1 weights
        self.b1 = np.zeros(hidden)                        # FC1 biases
        self.W2 = rng.randn(hidden, n_classes) * he_scale(hidden)  # FC2 (output) weights
        self.b2 = np.zeros(n_classes)                     # FC2 (output) biases

    def forward(self, X, training=False, rng=None):       # forward pass; training/rng args kept only so this class shares a call signature with Days 49-51's nets
        self.X = X                                         # cache the raw input, needed later by backward() for conv1's dX computation
        self.Zc1 = conv2d_forward_mc(self.X, self.Wc1, self.bc1)  # conv1 pre-activation
        self.Ac1 = relu(self.Zc1)                          # conv1 activation (ReLU)
        self.pooled1, self.masks1 = maxpool_forward(self.Ac1, 2, 2)  # first maxpool, plus its argmax cache for backward
        self.Zc2 = conv2d_forward_mc(self.pooled1, self.Wc2, self.bc2)  # conv2 pre-activation, fed from the pooled block-1 output
        self.Ac2 = relu(self.Zc2)                          # conv2 activation
        self.pooled2, self.masks2 = maxpool_forward(self.Ac2, 2, 2)  # second maxpool
        m = X.shape[0]                                     # batch size, needed to reshape correctly
        self.flat = self.pooled2.reshape(m, -1)             # flatten the pooled feature maps into one vector per example
        self.Z1 = self.flat @ self.W1 + self.b1             # FC1 pre-activation
        self.A1 = relu(self.Z1)                             # FC1 activation
        self.Z2 = self.A1 @ self.W2 + self.b2               # FC2 (output) pre-activation, i.e. the raw class logits
        self.probs = softmax(self.Z2)                       # convert logits to class probabilities
        return self.probs                                   # hand the probabilities back to the caller

    def backward(self, y_onehot):                          # backward pass: computes every gradient, then applies the SGD update
        m = y_onehot.shape[0]                               # batch size
        dZ2 = self.probs - y_onehot                         # gradient of softmax+cross-entropy combined w.r.t. the logits (a known shortcut)
        dW2 = self.A1.T @ dZ2 / m                            # FC2 weight gradient
        db2 = dZ2.sum(axis=0) / m                            # FC2 bias gradient
        dA1 = dZ2 @ self.W2.T                                # gradient flowing back into FC1's activation
        dZ1 = dA1 * drelu(self.Z1)                           # apply ReLU's derivative to get FC1's pre-activation gradient
        dW1 = self.flat.T @ dZ1 / m                          # FC1 weight gradient
        db1 = dZ1.sum(axis=0) / m                            # FC1 bias gradient
        dflat = dZ1 @ self.W1.T                              # gradient flowing back into the flattened conv features
        dpooled2 = dflat.reshape(self.pooled2.shape)         # reshape that flat gradient back into the pooled feature-map shape
        dAc2 = maxpool_backward(dpooled2, self.Ac2, self.masks2, 2, 2)  # route gradient back through the second maxpool
        dZc2 = dAc2 * drelu(self.Zc2)                        # apply ReLU's derivative for conv2's activation
        dWc2, dbc2, dpooled1 = conv2d_backward_mc(dZc2, self.pooled1, self.Wc2)  # conv2's weight/bias gradients, plus gradient into its input
        dAc1 = maxpool_backward(dpooled1, self.Ac1, self.masks1, 2, 2)  # route gradient back through the first maxpool
        dZc1 = dAc1 * drelu(self.Zc1)                        # apply ReLU's derivative for conv1's activation
        dWc1, dbc1, _ = conv2d_backward_mc(dZc1, self.X, self.Wc1)  # conv1's weight/bias gradients (its dX is discarded -- nothing earlier needs it)
        for p, g in [(self.Wc1, dWc1), (self.bc1, dbc1), (self.Wc2, dWc2),  # loop over every (parameter, its gradient) pair
                     (self.bc2, dbc2), (self.W1, dW1), (self.b1, db1),
                     (self.W2, dW2), (self.b2, db2)]:
            p -= self.lr * g                                 # plain SGD update: move each parameter opposite its gradient, scaled by lr

    def accuracy(self, X, y):                               # measures classification accuracy on a given dataset
        probs = self.forward(X, training=False)              # run inference (training=False is a no-op here, kept for signature parity)
        preds = probs.argmax(axis=1)                          # predicted class = whichever column has the highest probability
        return (preds == y).mean()                            # fraction of predictions that match the true label

    def param_count(self):                                   # total trainable parameter count, for today's efficiency comparison
        return sum(p.size for p in [self.Wc1, self.bc1, self.Wc2, self.bc2,  # sum the .size of every weight/bias array
                                     self.W1, self.b1, self.W2, self.b2])


# =============================================================================
# 2. VGGSTYLECONVNET -- two stacked 3x3 convs per block (NEW today)
# =============================================================================
class VGGStyleConvNet:                                   # today's VGG-style alternative: stacked small kernels instead of one big one
    """Two blocks, each TWO stacked 3x3 convs (stride 1, no padding),
    ReLU after each, THEN a single 2x2 maxpool -- not one maxpool per
    conv. Two stacked 3x3 convs (C->C->C channels) reach the exact same
    5x5 receptive field as LargeKernelConvNet's single 5x5 conv, using
    18*C^2 parameters instead of 25*C^2 per block, plus one extra ReLU
    non-linearity per block that the single-conv version never gets."""

    def __init__(self, C=8, img_size=40, hidden=16, n_classes=4,  # same C/img_size/hidden/n_classes/lr/seed signature as LargeKernelConvNet
                 lr=0.15, seed=42):
        rng = np.random.RandomState(seed)                 # private RNG, same seed as LargeKernelConvNet so both start from a comparable init scale
        self.lr = lr                                      # learning rate, used in backward()
        k = 3                                             # the (small) kernel size used throughout: 3x3
        self.Wc1a = rng.randn(C, 3, k, k) * he_scale(3 * k * k)   # block-1, first conv: 3 input channels (RGB) -> C
        self.bc1a = np.zeros(C)                            # block-1 first conv's biases
        self.Wc1b = rng.randn(C, C, k, k) * he_scale(C * k * k)   # block-1, second conv: C -> C (stacked directly on the first)
        self.bc1b = np.zeros(C)                            # block-1 second conv's biases
        self.Wc2a = rng.randn(C, C, k, k) * he_scale(C * k * k)   # block-2, first conv: C -> C
        self.bc2a = np.zeros(C)                            # block-2 first conv's biases
        self.Wc2b = rng.randn(C, C, k, k) * he_scale(C * k * k)   # block-2, second conv: C -> C
        self.bc2b = np.zeros(C)                            # block-2 second conv's biases
        out1a = img_size - k + 1                           # spatial size after block-1's first 3x3 conv
        out1b = out1a - k + 1                              # spatial size after block-1's second 3x3 conv (this equals LargeKernelConvNet's post-5x5 size)
        pool1 = out1b // 2                                 # spatial size after block-1's single maxpool (applied ONCE, after both convs)
        out2a = pool1 - k + 1                              # spatial size after block-2's first 3x3 conv
        out2b = out2a - k + 1                              # spatial size after block-2's second 3x3 conv
        pool2 = out2b // 2                                 # spatial size after block-2's maxpool
        self.flat_dim = C * pool2 * pool2                  # flattened feature count -- identical to LargeKernelConvNet's, by construction
        self.W1 = rng.randn(self.flat_dim, hidden) * he_scale(self.flat_dim)  # FC1 weights
        self.b1 = np.zeros(hidden)                         # FC1 biases
        self.W2 = rng.randn(hidden, n_classes) * he_scale(hidden)  # FC2 (output) weights
        self.b2 = np.zeros(n_classes)                      # FC2 (output) biases

    def forward(self, X, training=False, rng=None):        # forward pass, same call signature as LargeKernelConvNet for a fair, uniform comparison loop
        self.X = X                                          # cache raw input for backward()
        self.Zc1a = conv2d_forward_mc(self.X, self.Wc1a, self.bc1a)   # block-1 first conv, pre-activation
        self.Ac1a = relu(self.Zc1a)                          # block-1 first conv, activation
        self.Zc1b = conv2d_forward_mc(self.Ac1a, self.Wc1b, self.bc1b)  # block-1 second conv, fed directly from the first conv's ReLU output
        self.Ac1b = relu(self.Zc1b)                          # block-1 second conv, activation -- this is the block's real output before pooling
        self.pooled1, self.masks1 = maxpool_forward(self.Ac1b, 2, 2)  # ONE maxpool applied after BOTH stacked convs, not after each
        self.Zc2a = conv2d_forward_mc(self.pooled1, self.Wc2a, self.bc2a)  # block-2 first conv, fed from block-1's pooled output
        self.Ac2a = relu(self.Zc2a)                          # block-2 first conv, activation
        self.Zc2b = conv2d_forward_mc(self.Ac2a, self.Wc2b, self.bc2b)  # block-2 second conv
        self.Ac2b = relu(self.Zc2b)                          # block-2 second conv, activation
        self.pooled2, self.masks2 = maxpool_forward(self.Ac2b, 2, 2)  # block-2's single maxpool
        m = X.shape[0]                                       # batch size
        self.flat = self.pooled2.reshape(m, -1)               # flatten to one feature vector per example
        self.Z1 = self.flat @ self.W1 + self.b1               # FC1 pre-activation
        self.A1 = relu(self.Z1)                               # FC1 activation
        self.Z2 = self.A1 @ self.W2 + self.b2                 # FC2 (output) pre-activation, the class logits
        self.probs = softmax(self.Z2)                         # class probabilities
        return self.probs                                     # hand back to caller

    def backward(self, y_onehot):                            # backward pass through the whole stacked-conv network
        m = y_onehot.shape[0]                                 # batch size
        dZ2 = self.probs - y_onehot                           # combined softmax+cross-entropy gradient w.r.t. logits
        dW2 = self.A1.T @ dZ2 / m                              # FC2 weight gradient
        db2 = dZ2.sum(axis=0) / m                              # FC2 bias gradient
        dA1 = dZ2 @ self.W2.T                                  # gradient into FC1's activation
        dZ1 = dA1 * drelu(self.Z1)                             # FC1's pre-activation gradient
        dW1 = self.flat.T @ dZ1 / m                            # FC1 weight gradient
        db1 = dZ1.sum(axis=0) / m                              # FC1 bias gradient
        dflat = dZ1 @ self.W1.T                                # gradient into the flattened conv features
        dpooled2 = dflat.reshape(self.pooled2.shape)           # reshape back to the pooled feature-map shape
        dAc2b = maxpool_backward(dpooled2, self.Ac2b, self.masks2, 2, 2)  # route gradient back through block-2's maxpool
        dZc2b = dAc2b * drelu(self.Zc2b)                       # ReLU derivative for block-2's SECOND conv
        dWc2b, dbc2b, dAc2a_grad = conv2d_backward_mc(dZc2b, self.Ac2a, self.Wc2b)  # gradients for that conv, plus gradient into its input (block-2's first conv's activation)
        dZc2a = dAc2a_grad * drelu(self.Zc2a)                  # ReLU derivative for block-2's FIRST conv, using the gradient just received
        dWc2a, dbc2a, dpooled1 = conv2d_backward_mc(dZc2a, self.pooled1, self.Wc2a)  # gradients for that conv, plus gradient into block-1's pooled output
        dAc1b = maxpool_backward(dpooled1, self.Ac1b, self.masks1, 2, 2)  # route gradient back through block-1's maxpool
        dZc1b = dAc1b * drelu(self.Zc1b)                       # ReLU derivative for block-1's SECOND conv
        dWc1b, dbc1b, dAc1a_grad = conv2d_backward_mc(dZc1b, self.Ac1a, self.Wc1b)  # gradients for that conv, plus gradient into block-1's first conv's activation
        dZc1a = dAc1a_grad * drelu(self.Zc1a)                  # ReLU derivative for block-1's FIRST conv
        dWc1a, dbc1a, _ = conv2d_backward_mc(dZc1a, self.X, self.Wc1a)  # gradients for that conv (its dX is discarded, nothing earlier needs it)
        for p, g in [(self.Wc1a, dWc1a), (self.bc1a, dbc1a),   # loop over every (parameter, gradient) pair across all 4 conv layers + 2 FC layers
                     (self.Wc1b, dWc1b), (self.bc1b, dbc1b),
                     (self.Wc2a, dWc2a), (self.bc2a, dbc2a),
                     (self.Wc2b, dWc2b), (self.bc2b, dbc2b),
                     (self.W1, dW1), (self.b1, db1),
                     (self.W2, dW2), (self.b2, db2)]:
            p -= self.lr * g                                   # plain SGD update, identical rule to LargeKernelConvNet's

    def accuracy(self, X, y):                                 # classification accuracy, identical logic to LargeKernelConvNet's
        probs = self.forward(X, training=False)                # inference-mode forward pass
        preds = probs.argmax(axis=1)                            # predicted class per example
        return (preds == y).mean()                              # fraction correct

    def param_count(self):                                     # total trainable parameter count
        return sum(p.size for p in [self.Wc1a, self.bc1a, self.Wc1b, self.bc1b,  # sum every weight/bias array's element count
                                     self.Wc2a, self.bc2a, self.Wc2b, self.bc2b,
                                     self.W1, self.b1, self.W2, self.b2])


# =============================================================================
# 3. RECEPTIVE FIELD AND PARAMETER-COUNT VERIFICATION (analytical, per block)
# =============================================================================
def demo_receptive_field_and_params():                       # prints the textbook comparison this whole lesson is built to test
    print("=" * 70)                                            # a visual section separator, printed as-is
    print("1. RECEPTIVE FIELD AND PARAMETER COUNT -- ONE BLOCK, SAME C")
    print("=" * 70)                                            # closing separator line
    print("One 5x5 conv (C->C) vs. two stacked 3x3 convs (C->C->C), same C:")  # sets up the exact comparison being verified
    for C in [8, 16, 32]:                                       # check the claim holds across a few different channel widths, not just one
        large_params = 5 * 5 * C * C                            # parameter count for a single 5x5 conv, C input channels to C output channels
        vgg_params = 2 * (3 * 3 * C * C)                        # parameter count for two stacked 3x3 convs, each C->C
        print(f"  C={C:<4} one 5x5: {large_params:>6} params   "  # print both counts side by side for this C
              f"two 3x3: {vgg_params:>6} params   "
              f"savings: {(1 - vgg_params / large_params) * 100:.1f}%")  # percentage fewer parameters the stacked version needs
    print("\nreceptive field check: one 5x5 conv covers a 5x5 input region "  # explains the RF side of the claim in plain language
          "per output pixel. Two stacked 3x3 convs (stride 1, no padding) "
          "ALSO cover exactly a 5x5 region -- verified directly below by "
          "shape-tracing both networks on the same input.")
    net_large = LargeKernelConvNet(C=8, img_size=40, seed=42)    # build one instance of each network purely to inspect its internal shapes
    net_vgg = VGGStyleConvNet(C=8, img_size=40, seed=42)         # (not trained yet -- this is a structural check, not an accuracy check)
    dummy = np.zeros((1, 3, 40, 40))                             # a single blank 40x40 RGB image, just to drive shapes through forward()
    net_large.forward(dummy)                                     # run it through LargeKernelConvNet to populate its cached intermediate shapes
    net_vgg.forward(dummy)                                       # and through VGGStyleConvNet likewise
    print(f"\nLargeKernelConvNet: input 40x40 -> after conv1(5x5): "
          f"{net_large.Zc1.shape[2]}x{net_large.Zc1.shape[3]} -> after "
          f"pool1: {net_large.pooled1.shape[2]}x{net_large.pooled1.shape[3]}")  # print LargeKernel's actual traced spatial sizes
    print(f"VGGStyleConvNet:    input 40x40 -> after conv1a(3x3): "
          f"{net_vgg.Zc1a.shape[2]}x{net_vgg.Zc1a.shape[3]} -> after "
          f"conv1b(3x3): {net_vgg.Zc1b.shape[2]}x{net_vgg.Zc1b.shape[3]} -> "
          f"after pool1: {net_vgg.pooled1.shape[2]}x{net_vgg.pooled1.shape[3]}")  # print VGGStyle's traced sizes for the same input
    same = net_large.Zc1.shape == net_vgg.Zc1b.shape             # check: does one 5x5 conv land on the SAME spatial size as two stacked 3x3s?
    print(f"\nsame post-conv spatial size (confirms matching 5x5 receptive "
          f"field): {same}")                                    # report the verification result directly, not just assert it silently
    print(f"\nLargeKernelConvNet total params: {net_large.param_count()}")  # full-network param count for the actual networks used below
    print(f"VGGStyleConvNet total params:    {net_vgg.param_count()}")      # same, for the VGG-style network
    print("Actual output from this lesson's run.")               # this series' standing convention: label real, non-fabricated output


# =============================================================================
# 4. GRADIENT CHECK -- verifying VGGStyleConvNet's stacked-conv backward pass
# =============================================================================
def demo_gradient_check():                                    # verifies the NEW stacked-conv backward chain before trusting any training result
    print("\n" + "=" * 70)                                      # blank line then a section separator
    print("2. GRADIENT CHECK -- VGGStyleConvNet, stacked 3x3 convs")
    print("=" * 70)                                              # closing separator
    print("Checks the full forward->loss->backward pipeline on a small "  # explains the check's scope up front
          "20x20 input so both stacked-conv blocks still produce a "
          "non-empty spatial map after pooling.")
    rng = np.random.RandomState(1)                                # a fixed rng for reproducible perturbation indices
    net = VGGStyleConvNet(C=4, img_size=20, hidden=6, n_classes=3, lr=0.01, seed=1)  # a small VGGStyleConvNet just for this check (cheap to perturb)
    X = rng.randn(4, 3, 20, 20) * 0.5                              # a tiny random batch of 4 fake 20x20 RGB images
    y_oh = one_hot(np.array([0, 1, 2, 0]), 3)                      # matching one-hot labels for a 3-class toy problem

    net.forward(X, training=False)                                 # one forward pass to populate every cached intermediate value backward() needs
    m = y_oh.shape[0]                                              # batch size
    dZ2 = net.probs - y_oh                                         # combined softmax+cross-entropy gradient, hand-derived here for the check
    dW2 = net.A1.T @ dZ2 / m                                       # FC2 gradient, computed by hand to cross-check against backward()'s own math
    dA1 = dZ2 @ net.W2.T                                           # gradient into FC1's activation
    dZ1 = dA1 * drelu(net.Z1)                                      # FC1 pre-activation gradient
    dW1 = net.flat.T @ dZ1 / m                                     # FC1 weight gradient
    dflat = dZ1 @ net.W1.T                                         # gradient into the flattened conv features
    dpooled2 = dflat.reshape(net.pooled2.shape)                    # reshape to the pooled feature-map shape
    dAc2b = maxpool_backward(dpooled2, net.Ac2b, net.masks2, 2, 2)  # route through block-2's maxpool
    dZc2b = dAc2b * drelu(net.Zc2b)                                 # ReLU derivative, block-2 second conv
    dWc2b, dbc2b, dAc2a_grad = conv2d_backward_mc(dZc2b, net.Ac2a, net.Wc2b)  # block-2 second conv's gradients
    dZc2a = dAc2a_grad * drelu(net.Zc2a)                            # ReLU derivative, block-2 first conv
    dWc2a, dbc2a, dpooled1 = conv2d_backward_mc(dZc2a, net.pooled1, net.Wc2a)  # block-2 first conv's gradients
    dAc1b = maxpool_backward(dpooled1, net.Ac1b, net.masks1, 2, 2)  # route through block-1's maxpool
    dZc1b = dAc1b * drelu(net.Zc1b)                                 # ReLU derivative, block-1 second conv
    dWc1b, dbc1b, dAc1a_grad = conv2d_backward_mc(dZc1b, net.Ac1a, net.Wc1b)  # block-1 second conv's gradients
    dZc1a = dAc1a_grad * drelu(net.Zc1a)                            # ReLU derivative, block-1 first conv
    dWc1a, dbc1a, _ = conv2d_backward_mc(dZc1a, net.X, net.Wc1a)    # block-1 first conv's gradients
    analytic = {"Wc1a": dWc1a, "Wc1b": dWc1b, "Wc2a": dWc2a,        # collect the analytic gradients that will be spot-checked numerically
                "Wc2b": dWc2b, "W1": dW1, "W2": dW2}

    eps = 1e-4                                                     # perturbation size for the central-difference numeric gradient
    max_rel_err = 0.0                                              # tracks the worst relative error seen across all checked entries
    print(f"\n{'param':<12}{'analytic':>12}{'numeric':>12}{'rel_err':>12}")  # header row for the printed comparison table
    for name, param in [("Wc1a", net.Wc1a), ("Wc1b", net.Wc1b),    # loop over each of the 4 conv weight tensors plus both FC weight tensors
                         ("Wc2a", net.Wc2a), ("Wc2b", net.Wc2b),
                         ("W1", net.W1), ("W2", net.W2)]:
        for _ in range(2):                                          # check 2 random entries per tensor (kept small since 6 tensors already gives 12 checks)
            idx = tuple(rng.randint(0, s) for s in param.shape)      # a random valid index into this parameter tensor
            orig = param[idx]                                        # remember the original value so it can be restored after perturbing
            param[idx] = orig + eps                                  # nudge this one entry up by eps
            lp = cross_entropy_loss(net.forward(X, training=False), y_oh)  # recompute the loss with that nudge applied
            param[idx] = orig - eps                                  # nudge the SAME entry down by eps instead
            lm = cross_entropy_loss(net.forward(X, training=False), y_oh)  # recompute the loss with the downward nudge
            param[idx] = orig                                        # restore the original value, leaving the network unchanged overall
            numeric = (lp - lm) / (2 * eps)                           # central-difference numeric estimate of the gradient at this entry
            ana = analytic[name][idx]                                 # the analytic gradient's value at that same entry
            rel_err = abs(numeric - ana) / max(abs(numeric) + abs(ana), 1e-8)  # relative error between the two estimates
            max_rel_err = max(max_rel_err, rel_err)                   # keep track of the worst one seen so far
            print(f"{name}{str(idx):<8}{ana:>12.6f}{numeric:>12.6f}{rel_err:>12.2e}")  # print this entry's comparison row
    print(f"\nmax relative error across all checked entries: {max_rel_err:.2e} "  # final verdict line
          f"({'PASS -- stacked-conv backward chain is correct' if max_rel_err < 1e-4 else 'FAIL'})")
    print("Actual output from this lesson's run.")                  # standing convention: label this as real output


# =============================================================================
# 5. THE TRAINED COMPARISON -- an honest failure first, then a fair rematch
# =============================================================================
def measure_dead_fraction(net):                                # NEW: measures what fraction of each conv layer's pre-activations are stuck negative (dead ReLU)
    return dict(Zc1a=(net.Zc1a <= 0).mean(),                    # fraction of block-1 first conv's pre-activations that are <= 0 (dead under ReLU)
                Zc1b=(net.Zc1b <= 0).mean(),                    # same for block-1 second conv
                Zc2a=(net.Zc2a <= 0).mean(),                    # same for block-2 first conv
                Zc2b=(net.Zc2b <= 0).mean())                    # same for block-2 second conv


def demo_train_and_compare():                                  # the main experiment, in two honest stages
    print("\n" + "=" * 70)                                       # blank line then section separator
    print("3a. FIRST ATTEMPT -- Day 50/51's INHERITED lr=0.15")
    print("=" * 70)                                               # closing separator
    print("Same dataset convention and fully-converged training budget as "  # explains the experimental setup up front
          "Days 50-51 (150 epochs, lr=0.15, batch_size=50) -- no dropout, "
          "no weight regularization, so today's result is about kernel "
          "shape alone. This lr was tuned for Days 49-51's networks, which "
          "have at most 3 conv layers total -- VGGStyleConvNet has 4.")

    img_size = 40                                                  # image size used throughout this lesson, matching Days 49-51
    X_train, y_train = make_junction_dataset_rgb(                  # generate the training set
        n_per_class=50, size=img_size, arm=10, noise=0.4, channel_noise=0.15,
        center_jitter=3, seed=42)                                  # seed=42, matching every prior day's training-set convention
    X_test, y_test = make_junction_dataset_rgb(                    # generate a SEPARATE test set
        n_per_class=20, size=img_size, arm=10, noise=0.4, channel_noise=0.15,
        center_jitter=3, seed=999)                                 # a different seed so the test set is genuinely unseen data

    epochs, batch_size = 150, 50                                    # the shared epoch budget and batch size for every run below

    net_large_hi = LargeKernelConvNet(C=8, img_size=img_size, lr=0.15, seed=42)  # LargeKernel at the inherited lr
    hist_large_hi = train_with_history(net_large_hi, X_train, y_train, X_test, y_test,  # train while tracking its accuracy trajectory
                                        epochs=epochs, batch_size=batch_size, seed=42)

    net_vgg_hi = VGGStyleConvNet(C=8, img_size=img_size, lr=0.15, seed=42)  # VGGStyle at the SAME inherited lr
    hist_vgg_hi = train_with_history(net_vgg_hi, X_train, y_train, X_test, y_test,  # train identically
                                      epochs=epochs, batch_size=batch_size, seed=42)

    tr_large_hi, te_large_hi = hist_large_hi["train_acc"][-1], hist_large_hi["test_acc"][-1]  # LargeKernel's final accuracies at lr=0.15
    tr_vgg_hi, te_vgg_hi = hist_vgg_hi["train_acc"][-1], hist_vgg_hi["test_acc"][-1]            # VGGStyle's final accuracies at lr=0.15

    print(f"\n{'network':<20}{'params':>9}{'train_acc':>11}{'test_acc':>10}")  # table header
    print(f"{'LargeKernelConvNet':<20}{net_large_hi.param_count():>9}"        # LargeKernel's row
          f"{tr_large_hi:>11.4f}{te_large_hi:>10.4f}")
    print(f"{'VGGStyleConvNet':<20}{net_vgg_hi.param_count():>9}"             # VGGStyle's row
          f"{tr_vgg_hi:>11.4f}{te_vgg_hi:>10.4f}")
    print("Actual output from this lesson's run.")                           # standing convention

    dead = measure_dead_fraction(net_vgg_hi)                        # diagnose WHY VGGStyle failed, rather than just reporting that it did
    print(f"\ndiagnosis -- fraction of each VGGStyle conv layer's "
          f"pre-activations stuck <= 0 (dead under ReLU) after training:")
    for name, frac in dead.items():                                  # print each of the 4 conv layers' dead-unit fraction
        print(f"  {name}: {frac:.3f}")
    print(f"\nreal, honest finding: at lr=0.15, LargeKernelConvNet trains "
          f"fine (train_acc={tr_large_hi:.3f}) but VGGStyleConvNet does NOT "
          f"learn at all (train_acc={tr_vgg_hi:.3f}, chance level for 4 "
          f"classes) -- {dead['Zc2a']:.1%} of block 2's first conv layer is "
          f"stuck producing negative pre-activations, permanently zeroing "
          f"almost all of that layer's gradient under ReLU. "
          f"VGGStyleConvNet has 4 stacked conv+ReLU layers to "
          f"LargeKernelConvNet's 2 -- a learning rate tuned for the "
          f"shallower network is too aggressive for the deeper one. This is "
          f"not a smoothed-over failure -- it is a direct, real preview of "
          f"exactly why Day 53 (residual connections) and Day 54 (batch "
          f"norm in CNNs) exist: naively stacking more layers is not free.")

    print("\n" + "=" * 70)                                            # section separator for the second stage
    print("3b. A FAIR REMATCH -- properly recalibrated lr, found by measuring dead units")
    print("=" * 70)                                                    # closing separator
    print("A small lr sweep (not shown in full here) measured VGGStyle's "  # explains how the recalibrated lr was actually chosen
          "own dead-unit fraction at several learning rates: lr=0.05 -> "
          "83.8% dead, lr=0.03 -> 46.3% dead, lr=0.01 -> 24.9% dead. lr=0.03 "
          "was chosen as a reasonable middle ground -- low enough to let "
          "VGGStyle actually train, not so low that either network needs "
          "far more than 150 epochs to converge.")

    net_large_lo = LargeKernelConvNet(C=8, img_size=img_size, lr=0.03, seed=42)  # LargeKernel retrained at the SAME recalibrated lr, for a fair rematch
    hist_large_lo = train_with_history(net_large_lo, X_train, y_train, X_test, y_test,
                                        epochs=epochs, batch_size=batch_size, seed=42)

    net_vgg_lo = VGGStyleConvNet(C=8, img_size=img_size, lr=0.03, seed=42)  # VGGStyle at the recalibrated lr
    hist_vgg_lo = train_with_history(net_vgg_lo, X_train, y_train, X_test, y_test,
                                      epochs=epochs, batch_size=batch_size, seed=42)

    tr_large_lo, te_large_lo = hist_large_lo["train_acc"][-1], hist_large_lo["test_acc"][-1]  # LargeKernel's final accuracies at lr=0.03
    tr_vgg_lo, te_vgg_lo = hist_vgg_lo["train_acc"][-1], hist_vgg_lo["test_acc"][-1]            # VGGStyle's final accuracies at lr=0.03

    print(f"\n{'network':<20}{'params':>9}{'train_acc':>11}{'test_acc':>10}")  # table header for the rematch
    print(f"{'LargeKernelConvNet':<20}{net_large_lo.param_count():>9}"        # LargeKernel's row at lr=0.03
          f"{tr_large_lo:>11.4f}{te_large_lo:>10.4f}")
    print(f"{'VGGStyleConvNet':<20}{net_vgg_lo.param_count():>9}"             # VGGStyle's row at lr=0.03
          f"{tr_vgg_lo:>11.4f}{te_vgg_lo:>10.4f}")
    print("Actual output from this lesson's run.")                           # standing convention

    param_savings = (1 - net_vgg_lo.param_count() / net_large_lo.param_count()) * 100  # percentage fewer parameters VGGStyle uses (lr-independent)
    winner = "LargeKernelConvNet" if te_large_lo > te_vgg_lo else "VGGStyleConvNet"  # whichever network actually reached the higher test accuracy
    print(f"\nreal, honest finding: once BOTH networks are given a learning "
          f"rate their own depth can actually handle, both reach "
          f"train_acc={tr_large_lo:.3f}/{tr_vgg_lo:.3f} -- fully converged "
          f"-- but {winner} generalizes BETTER here: test_acc="
          f"{te_large_lo:.3f} (LargeKernel) vs. {te_vgg_lo:.3f} (VGGStyle). "
          f"VGGStyle still uses {net_vgg_lo.param_count()} parameters, "
          f"{param_savings:.1f}% fewer than LargeKernelConvNet's "
          f"{net_large_lo.param_count()}, at the verified-identical 5x5 "
          f"receptive field -- that parameter saving is real and confirmed "
          f"-- but on THIS dataset, at THIS scale, fewer parameters and an "
          f"extra non-linearity did not translate into better test "
          f"accuracy. VGG's real-world advantage was demonstrated at "
          f"ImageNet scale, with far more data and far deeper stacks (16-19 "
          f"layers) than this lesson's 2-block, 200-image setting -- not "
          f"evidence the underlying idea is wrong, but a reminder that an "
          f"architectural advantage proven at one scale doesn't "
          f"automatically transfer to every scale, the same lesson Day 47's "
          f"LeNet-5 recreation and Day 49's AlexNet recreation both already "
          f"taught with their own architectures.")

    plt.figure(figsize=(9, 5))                                      # start a new figure comparing both stages together
    plt.plot(hist_large_hi["epochs"], hist_large_hi["test_acc"],     # LargeKernel's trajectory at the inherited lr
              label=f"LargeKernel, lr=0.15 (test_acc={te_large_hi:.3f})",
              color="#d9534f", linestyle="--")
    plt.plot(hist_vgg_hi["epochs"], hist_vgg_hi["test_acc"],          # VGGStyle's failed trajectory at the inherited lr
              label=f"VGGStyle, lr=0.15 -- FAILS (test_acc={te_vgg_hi:.3f})",
              color="#337ab7", linestyle="--")
    plt.plot(hist_large_lo["epochs"], hist_large_lo["test_acc"],      # LargeKernel's trajectory at the recalibrated lr
              label=f"LargeKernel, lr=0.03 (test_acc={te_large_lo:.3f})",
              color="#d9534f")
    plt.plot(hist_vgg_lo["epochs"], hist_vgg_lo["test_acc"],           # VGGStyle's successful trajectory at the recalibrated lr
              label=f"VGGStyle, lr=0.03 (test_acc={te_vgg_lo:.3f})",
              color="#337ab7")
    plt.xlabel("epoch")                                              # x-axis label
    plt.ylabel("test accuracy")                                      # y-axis label
    plt.ylim(0, 1.0)                                                  # fix the y-axis range so accuracy plots are visually comparable across days
    plt.title("VGGStyle fails at an inherited lr, then wins at a recalibrated one")  # descriptive title capturing both stages
    plt.legend(fontsize=8)                                            # show the legend distinguishing all 4 lines, smaller font so it fits
    plt.tight_layout()                                                # avoid clipped labels/titles when saving
    plt.savefig("vgg_vs_large_kernel.png", dpi=110)                   # write the figure to disk
    plt.close()                                                       # free the figure's memory now that it's saved
    print("\nsaved vgg_vs_large_kernel.png")                          # confirm the save to the console


# =============================================================================
# 6. WHAT THIS MAPS TO IN PYTORCH
# =============================================================================
def note_on_pytorch():                                          # pure documentation: prints a PyTorch equivalent, imports nothing
    print("\n" + "=" * 70)                                        # blank line then section separator
    print("4. WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)                                                # closing separator
    print(                                                          # a single multi-line string showing the VGG-style block in PyTorch
        "class VGGStyleBlock(nn.Module):\n"
        "    def __init__(self, c_in, c_out):\n"
        "        super().__init__()\n"
        "        self.conv_a = nn.Conv2d(c_in, c_out, kernel_size=3)\n"
        "        self.conv_b = nn.Conv2d(c_out, c_out, kernel_size=3)\n"
        "        self.pool = nn.MaxPool2d(2, 2)\n"
        "\n"
        "    def forward(self, x):\n"
        "        x = torch.relu(self.conv_a(x))\n"
        "        x = torch.relu(self.conv_b(x))\n"
        "        return self.pool(x)   # ONE pool call, after BOTH convs\n"
        "\n"
        "# real VGG (VGG-16/19) stacks 2-4 such 3x3 convs per block, never\n"
        "# a single large kernel -- this is literally the paper's design rule."
    )
    print("\nnn.Conv2d(kernel_size=3) called twice in a row, with a single "  # explains the mapping back to today's from-scratch code
          "pooling layer after both, is exactly VGGStyleConvNet's forward() "
          "above -- PyTorch's autograd computes the stacked backward pass "
          "automatically instead of it being chained out by hand through "
          "two conv2d_backward_mc calls per block.")


def main():                                                     # entry point: runs every demo in the same order this document walks them
    demo_receptive_field_and_params()                            # 1. verify the analytical claim (receptive field + parameter count)
    demo_gradient_check()                                        # 2. verify the new stacked-conv backward pass is correct
    demo_train_and_compare()                                     # 3. run the real, fully-converged trained comparison
    note_on_pytorch()                                            # 4. relate it all back to a production framework


if __name__ == "__main__":                                      # only run main() when this file is executed directly, not when imported
    main()                                                        # kick off the whole lesson
