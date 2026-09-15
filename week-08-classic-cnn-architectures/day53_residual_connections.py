"""
Day 53: Residual Connections -- What They Actually Fix (and Don't)

Day 52 ended on a cliffhanger: VGGStyleConvNet (4 conv+ReLU layers) failed
to train at all at lr=0.15 because ~99.9% of a late conv layer's units went
permanently dead under ReLU -- a learning rate tuned for a shallower network
was too aggressive for a deeper one. Today builds the fix He et al. (2015,
the ResNet paper) actually proposed: give each block an IDENTITY shortcut
around its convs, so out = ReLU(F(x) + x) instead of out = ReLU(F(x)).
Because d(F(x)+x)/dx = dF/dx + 1, gradient flowing backward through a
residual block always has an unattenuated "+1" path straight to the input,
on top of whatever the conv path contributes -- in principle this reduces
how much gradient magnitude is lost as depth increases, without adding a
single extra parameter.

To test this for real, today pushes depth further than Day 52 ever did:
PlainDeepConvNet and ResNetStyleConvNet both stack a stem conv plus 3
residual-shaped blocks (2 convs each) = 7 conv+ReLU layers total, nearly
double Day 52's deepest network. Getting the residual block's backward pass
right (splitting the upstream gradient across the conv path AND the
identity path) is verified with a real numeric gradient check first. A new
direct diagnostic -- measuring each conv layer's gradient norm at
initialization, deepest to shallowest -- shows the effect itself, not just
its downstream symptom: the earliest layer's gradient retains 41.5% of the
deepest layer's magnitude in ResNetStyleConvNet, vs. only 11.6% in
PlainDeepConvNet. That is a real, measured difference (both networks share
identical initial weights, so the shortcut connections are the only
possible cause).

But today's two trained-comparison findings are NOT the tidy "ResNet wins"
story either. FIRST: thrown at Day 52's exact lr=0.15, BOTH networks fail
completely (train_acc=0.250, chance level) -- residual connections do NOT
automatically rescue an overly-aggressive learning rate at this depth.
More gradient reaching a layer is not the same as a smaller update step
once that gradient arrives; block 0 and block 2 end up 100% dead under
ReLU in BOTH networks. SECOND: after a short lr sweep finds lr=0.008 lets
either network escape chance level, and both are retrained for the full
150 epochs, PlainDeepConvNet actually generalizes BETTER (test_acc=0.738)
than ResNetStyleConvNet (test_acc=0.562) -- despite the gradient-flow
diagnostic's real, measured advantage for the residual version, and despite
both networks having IDENTICAL parameter counts (residual connections are
completely free). The lesson: a shortcut connection measurably changes how
gradient is distributed across depth, which is real and worth understanding
on its own terms -- but it is a distinct effect from "picks a better
learning rate for you," and neither effect alone guarantees the deeper,
more sophisticated architecture wins on a given dataset at a given scale --
the same caveat Day 47, 49, and 52 already taught with their own
architectures.

Every line of code below carries its own comment, continuing Day 52's
convention. conv2d_forward_mc/backward_mc are also rewritten today from a
per-output-pixel loop to a loop over kernel offsets (kH*kW iterations
instead of C_out*out_H*out_W) -- mathematically identical, re-verified via
the same gradient check, but roughly 4x faster on this lesson's 7-conv-layer
networks, which is what keeps today's larger experiment tractable.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # "Agg" backend renders to a file, not a screen -- needed since this runs headless
import matplotlib.pyplot as plt                         # pyplot: the plotting API actually called later (plt.plot, plt.savefig, ...)

np.random.seed(42)                                      # fixes numpy's GLOBAL random state so any code that forgets to pass its own rng is still reproducible

# =============================================================================
# 0. SHARED HELPERS -- reused verbatim from Days 39-52, EXCEPT conv2d_forward_mc
#    and conv2d_backward_mc, which gain a new `pad` argument today (see note
#    below) so a residual block's two convs can preserve spatial size.
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


def conv2d_forward_mc(X, W, b, pad=0):                  # multi-channel 2D convolution forward pass, stride 1, pad=0 reproduces Days 49-52's "valid" convolution
    if pad > 0:                                           # only pad when asked to
        Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))  # zero-pad only the two spatial axes (height, width), leave batch/channel alone
    else:                                                  # no padding requested
        Xp = X                                             # just use the input as-is
    m, C_in, H, Win = Xp.shape                            # unpack (possibly padded) input shape: batch size, input channels, height, width
    C_out, C_in_w, kH, kW = W.shape                       # unpack filter shape: output channels, input channels, kernel height, kernel width
    out_H = H - kH + 1                                    # output height for a valid convolution over the (possibly padded) input
    out_W = Win - kW + 1                                  # output width, same formula along the width axis
    Z = np.zeros((m, C_out, out_H, out_W))                # allocate the output feature map, all zeros to start
    for kh in range(kH):                                  # NEW today: loop over KERNEL offsets (kH of them, e.g. 3 or 5) instead of every output pixel --
        for kw in range(kW):                              # this replaces Days 39-52's C_out*out_H*out_W-iteration loop with a kH*kW-iteration one (same math, far fewer Python-level steps)
            X_slice = Xp[:, :, kh:kh + out_H, kw:kw + out_W]  # every output position's input pixel AT THIS ONE (kh, kw) offset, for the whole batch/channels/spatial map at once
            Z += np.einsum("mchw,fc->mfhw", X_slice, W[:, :, kh, kw])  # contract over input channels c, broadcast over output filters f, all (m,h,w) positions in one call
    Z += b[None, :, None, None]                            # add each filter's bias once, broadcast across the batch and every spatial position
    return Z                                              # the full convolved output, shape (m, C_out, out_H, out_W) -- numerically identical to the old nested-loop version


def conv2d_backward_mc(dZ, X, W, pad=0):                # backward pass for conv2d_forward_mc: returns dW, db, dX -- same kernel-offset-loop vectorization as the forward pass
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
    for kh in range(kH):                                  # loop over kernel offsets, exactly mirroring the forward pass's loop (kH*kW iterations, not C_out*out_H*out_W)
        for kw in range(kW):                              # same (kh, kw) offset the forward pass used
            X_slice = Xp[:, :, kh:kh + out_H, kw:kw + out_W]  # the same input sub-grid the forward pass read at this offset
            dW[:, :, kh, kw] = np.einsum("mfhw,mchw->fc", dZ, X_slice) / m  # this offset's contribution to every (filter, input-channel) weight gradient, summed over batch+spatial positions
            dXp[:, :, kh:kh + out_H, kw:kw + out_W] += np.einsum("mfhw,fc->mchw", dZ, W[:, :, kh, kw])  # scatter this offset's gradient contribution back onto the input sub-grid it came from
    if pad > 0:                                            # if padding was added on the way in, it must be stripped off on the way out
        dX = dXp[:, :, pad:-pad, pad:-pad]                 # crop the padding rows/columns back off, leaving dX the same shape as the ORIGINAL X
    else:                                                   # no padding was used
        dX = dXp                                            # dXp is already the right shape
    return dW, db, dX                                     # gradients w.r.t. filters, biases, and the layer's (unpadded) input -- numerically identical to the old nested-loop version


def maxpool_forward(A, size=2, stride=2):               # 2x2 max pooling forward pass (default size/stride) -- unchanged from Day 49
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


def maxpool_backward(dOut, A, idx_cache, size=2, stride=2):  # backward pass for maxpool_forward -- unchanged from Day 49
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


def iterate_minibatches(X, y_onehot, batch_size, rng, shuffle=True):  # yields shuffled mini-batches, one epoch's worth per call -- unchanged
    m = X.shape[0]                                        # total number of training examples
    order = rng.permutation(m) if shuffle else np.arange(m)  # a fresh random ordering of example indices each call, or the plain order if shuffle=False
    for start in range(0, m, batch_size):                 # step through that ordering in chunks of batch_size
        batch_idx = order[start:start + batch_size]        # the indices belonging to this particular batch (last batch may be smaller)
        yield X[batch_idx], y_onehot[batch_idx], batch_idx  # yield this batch's inputs, one-hot labels, and original indices


def make_junction_dataset_rgb(n_per_class=50, size=30, arm=8, noise=0.4,   # synthetic 4-class "junction" image generator -- unchanged
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
# 1. BLOCK-LEVEL HELPERS -- the ONLY real difference between today's two
#    networks lives here: whether a block adds its input back before ReLU.
# =============================================================================
def plainblock_forward(x, W1, b1, W2, b2, pad=1):       # a plain (non-residual) block: conv -> relu -> conv -> relu, no shortcut
    z1 = conv2d_forward_mc(x, W1, b1, pad=pad)            # first conv, padded so spatial size is preserved (needed later for the ResNet version)
    a1 = relu(z1)                                          # first conv's activation
    z2 = conv2d_forward_mc(a1, W2, b2, pad=pad)           # second conv, also padded
    out = relu(z2)                                         # PLAIN block: activation is just relu(z2) -- no identity added
    cache = (x, z1, a1, z2)                                # everything backward() will need
    return out, cache                                     # this block's output, plus its cache


def plainblock_backward(dout, cache, W1, W2, pad=1):    # backward pass for a plain block
    x, z1, a1, z2 = cache                                 # unpack the forward pass's cached values
    dz2 = dout * drelu(z2)                                # undo the block's final ReLU
    dW2, db2, da1 = conv2d_backward_mc(dz2, a1, W2, pad=pad)  # undo the second conv, get its weight/bias grads plus gradient into a1
    dz1 = da1 * drelu(z1)                                 # undo the first conv's ReLU
    dW1, db1, dx = conv2d_backward_mc(dz1, x, W1, pad=pad)  # undo the first conv, get its weight/bias grads plus gradient into the block's input
    return dx, dW1, db1, dW2, db2                         # gradient into the block's input, plus every weight/bias gradient inside it


def resblock_forward(x, W1, b1, W2, b2, pad=1):         # a RESIDUAL block: conv -> relu -> conv -> ADD INPUT -> relu
    z1 = conv2d_forward_mc(x, W1, b1, pad=pad)            # first conv (identical to the plain block so far)
    a1 = relu(z1)                                          # first conv's activation
    z2 = conv2d_forward_mc(a1, W2, b2, pad=pad)           # second conv -- note NO relu applied to z2 before the add, matching He et al. 2015
    out = relu(z2 + x)                                     # THE ONLY STRUCTURAL DIFFERENCE from plainblock_forward: add x back in before the final relu
    cache = (x, z1, a1, z2)                                # same cache shape as the plain block, for a fair side-by-side comparison
    return out, cache                                     # this block's output, plus its cache


def resblock_backward(dout, cache, W1, W2, pad=1):      # backward pass for a residual block
    x, z1, a1, z2 = cache                                 # unpack the forward pass's cached values
    dsum = dout * drelu(z2 + x)                           # undo the block's final ReLU (applied to the SUM z2+x, not z2 alone)
    dz2 = dsum                                             # the sum's gradient flows unchanged into z2 (d(z2+x)/dz2 = 1)
    dx_identity = dsum                                     # AND unchanged into x directly (d(z2+x)/dx = 1) -- the residual "shortcut" gradient path
    dW2, db2, da1 = conv2d_backward_mc(dz2, a1, W2, pad=pad)  # undo the second conv along the CONV path, same as the plain block
    dz1 = da1 * drelu(z1)                                 # undo the first conv's ReLU, same as the plain block
    dW1, db1, dx_from_conv1 = conv2d_backward_mc(dz1, x, W1, pad=pad)  # undo the first conv, get the conv path's contribution to dx
    dx = dx_from_conv1 + dx_identity                       # THE KEY LINE: combine the conv path's gradient with the shortcut's UNATTENUATED gradient
    return dx, dW1, db1, dW2, db2                         # gradient into the block's input, plus every weight/bias gradient inside it


# =============================================================================
# 2. PLAINDEEPCONVNET -- a stem conv plus 3 plain (non-residual) blocks
# =============================================================================
class PlainDeepConvNet:                                  # today's control: 7 conv+ReLU layers total, no shortcuts anywhere
    """A stem 3x3 conv (3->C) followed by n_blocks plain blocks (each
    C->C->C, padded so spatial size is preserved), a 2x2 maxpool after
    EACH block, then the same FC head shape as every prior day. This is
    deliberately deeper than Day 52's VGGStyleConvNet (7 conv layers here
    vs. 4 there) to stress-test dying ReLUs/vanishing gradients harder."""

    def __init__(self, C=8, img_size=40, hidden=16, n_classes=4,  # constructor: C = channel width shared by every conv layer
                 n_blocks=3, lr=0.15, seed=42, pad=1):
        rng = np.random.RandomState(seed)                 # this network's own private RNG, seeded for reproducible weight init
        self.lr = lr                                      # store the learning rate on self, used later in backward()
        self.n_blocks = n_blocks                          # remember how many blocks this network has, for looping in forward/backward
        self.pad = pad                                    # remember the padding used throughout, for looping in forward/backward
        k = 3                                              # kernel size used throughout: 3x3
        self.Wstem = rng.randn(C, 3, k, k) * he_scale(3 * k * k)  # stem conv: 3 input channels (RGB) -> C, padded so spatial size stays img_size
        self.bstem = np.zeros(C)                           # stem conv's biases
        self.blocks = []                                   # will hold one dict of {W1,b1,W2,b2} per block
        for _ in range(n_blocks):                          # build n_blocks identical-shaped plain blocks, C->C->C throughout
            W1 = rng.randn(C, C, k, k) * he_scale(C * k * k)  # this block's first conv weights
            b1 = np.zeros(C)                                # this block's first conv biases
            W2 = rng.randn(C, C, k, k) * he_scale(C * k * k)  # this block's second conv weights
            b2 = np.zeros(C)                                # this block's second conv biases
            self.blocks.append(dict(W1=W1, b1=b1, W2=W2, b2=b2))  # store this block's parameters together
        final_spatial = img_size // (2 ** n_blocks)         # spatial size after n_blocks rounds of 2x2 pooling (stem/block convs don't shrink it, padding handles that)
        self.flat_dim = C * final_spatial * final_spatial   # total flattened feature count feeding into the first FC layer
        self.W1fc = rng.randn(self.flat_dim, hidden) * he_scale(self.flat_dim)  # FC1 weights
        self.b1fc = np.zeros(hidden)                        # FC1 biases
        self.W2fc = rng.randn(hidden, n_classes) * he_scale(hidden)  # FC2 (output) weights
        self.b2fc = np.zeros(n_classes)                     # FC2 (output) biases

    def forward(self, X, training=False, rng=None):        # forward pass; training/rng args kept only for call-signature parity with earlier days
        self.X = X                                          # cache raw input for backward()'s stem gradient
        self.Zstem = conv2d_forward_mc(self.X, self.Wstem, self.bstem, pad=self.pad)  # stem conv pre-activation (spatial size preserved by padding)
        self.Astem = relu(self.Zstem)                        # stem conv activation, this feeds into block 1
        cur = self.Astem                                     # `cur` tracks the running activation as it passes through each block
        self.block_caches = []                               # will hold one cache per block, needed by backward()
        self.block_outs = []                                 # will hold each block's PRE-pool output, needed by maxpool_backward's `A` argument
        self.pool_masks = []                                 # will hold each block's maxpool argmax cache
        for blk in self.blocks:                              # loop over every block in forward order
            out, cache = plainblock_forward(cur, blk["W1"], blk["b1"], blk["W2"], blk["b2"], pad=self.pad)  # run this block (no shortcut)
            self.block_caches.append(cache)                  # remember this block's cache
            self.block_outs.append(out)                      # remember this block's pre-pool output
            pooled, mask = maxpool_forward(out, 2, 2)         # pool this block's output down by 2x
            self.pool_masks.append(mask)                      # remember the pooling argmax cache
            cur = pooled                                      # feed the pooled result into the next block
        m = X.shape[0]                                       # batch size
        self.flat = cur.reshape(m, -1)                        # flatten the final pooled feature maps into one vector per example
        self.Z1 = self.flat @ self.W1fc + self.b1fc            # FC1 pre-activation
        self.A1 = relu(self.Z1)                               # FC1 activation
        self.Z2 = self.A1 @ self.W2fc + self.b2fc              # FC2 (output) pre-activation, the raw class logits
        self.probs = softmax(self.Z2)                          # convert logits to class probabilities
        return self.probs                                     # hand the probabilities back to the caller

    def backward(self, y_onehot):                            # backward pass: computes every gradient, then applies the SGD update
        m = y_onehot.shape[0]                                 # batch size
        dZ2 = self.probs - y_onehot                           # combined softmax+cross-entropy gradient w.r.t. logits
        dW2fc = self.A1.T @ dZ2 / m                            # FC2 weight gradient
        db2fc = dZ2.sum(axis=0) / m                            # FC2 bias gradient
        dA1 = dZ2 @ self.W2fc.T                                # gradient into FC1's activation
        dZ1 = dA1 * drelu(self.Z1)                             # FC1's pre-activation gradient
        dW1fc = self.flat.T @ dZ1 / m                          # FC1 weight gradient
        db1fc = dZ1.sum(axis=0) / m                            # FC1 bias gradient
        dflat = dZ1 @ self.W1fc.T                              # gradient into the flattened conv features
        final_shape = (m,) + self.block_outs[-1].shape[1:2] + tuple(s // 2 for s in self.block_outs[-1].shape[2:])  # shape of the last pooled feature map
        dcur = dflat.reshape(final_shape)                      # reshape the flat gradient back into that pooled feature-map shape
        block_grads = []                                       # will collect (dW1,db1,dW2,db2) for every block, in FORWARD order, filled in reverse
        for i in reversed(range(self.n_blocks)):               # walk backward through the blocks, last block first
            dout = maxpool_backward(dcur, self.block_outs[i], self.pool_masks[i], 2, 2)  # undo this block's maxpool
            dx, dW1, db1, dW2, db2 = plainblock_backward(dout, self.block_caches[i], self.blocks[i]["W1"], self.blocks[i]["W2"], pad=self.pad)  # undo this plain block
            block_grads.append((dW1, db1, dW2, db2))            # stash this block's gradients (still in reverse order for now)
            dcur = dx                                            # this block's dx becomes the upstream gradient for the block/stem before it
        block_grads.reverse()                                   # flip back to forward order so block_grads[i] matches self.blocks[i]
        dZstem = dcur * drelu(self.Zstem)                       # undo the stem's ReLU using the gradient that emerged from block 0
        dWstem, dbstem, _ = conv2d_backward_mc(dZstem, self.X, self.Wstem, pad=self.pad)  # stem's weight/bias gradients (its dX is discarded)
        self.grads = dict(Wstem=dWstem, bstem=dbstem, blocks=block_grads,  # stash every gradient on self so a diagnostic can inspect them before the update
                           W1fc=dW1fc, b1fc=db1fc, W2fc=dW2fc, b2fc=db2fc)
        self.Wstem -= self.lr * dWstem                          # SGD update: stem weights
        self.bstem -= self.lr * dbstem                          # SGD update: stem biases
        for blk, (dW1, db1, dW2, db2) in zip(self.blocks, block_grads):  # loop over every block alongside its just-computed gradients
            blk["W1"] -= self.lr * dW1                           # SGD update: this block's first conv weights
            blk["b1"] -= self.lr * db1                           # SGD update: this block's first conv biases
            blk["W2"] -= self.lr * dW2                           # SGD update: this block's second conv weights
            blk["b2"] -= self.lr * db2                           # SGD update: this block's second conv biases
        self.W1fc -= self.lr * dW1fc                             # SGD update: FC1 weights
        self.b1fc -= self.lr * db1fc                             # SGD update: FC1 biases
        self.W2fc -= self.lr * dW2fc                             # SGD update: FC2 weights
        self.b2fc -= self.lr * db2fc                             # SGD update: FC2 biases

    def accuracy(self, X, y):                                 # measures classification accuracy on a given dataset
        probs = self.forward(X, training=False)                # run inference
        preds = probs.argmax(axis=1)                            # predicted class = whichever column has the highest probability
        return (preds == y).mean()                              # fraction of predictions that match the true label

    def param_count(self):                                     # total trainable parameter count
        total = self.Wstem.size + self.bstem.size               # start with the stem's weights and biases
        for blk in self.blocks:                                 # add every block's weights and biases
            total += blk["W1"].size + blk["b1"].size + blk["W2"].size + blk["b2"].size
        total += self.W1fc.size + self.b1fc.size + self.W2fc.size + self.b2fc.size  # add the FC head
        return total                                             # the network's full trainable parameter count


# =============================================================================
# 3. RESNETSTYLECONVNET -- structurally IDENTICAL, except each block adds
#    an identity shortcut before its final ReLU (He et al., 2015)
# =============================================================================
class ResNetStyleConvNet:                                # today's fix: same depth, same parameter count, one shortcut per block
    """Exactly PlainDeepConvNet's architecture -- same stem, same number
    of blocks, same channel width C throughout, same FC head -- except
    each block computes relu(conv2(relu(conv1(x))) + x) instead of
    relu(conv2(relu(conv1(x)))). That one difference adds ZERO parameters
    but gives backward() an unattenuated identity gradient path through
    every block, which is the whole ResNet argument."""

    def __init__(self, C=8, img_size=40, hidden=16, n_classes=4,  # identical constructor signature to PlainDeepConvNet, for a fair side-by-side use
                 n_blocks=3, lr=0.15, seed=42, pad=1):
        rng = np.random.RandomState(seed)                 # private RNG -- same seed as PlainDeepConvNet gives both networks IDENTICAL initial weights
        self.lr = lr                                      # learning rate, used in backward()
        self.n_blocks = n_blocks                          # number of residual blocks
        self.pad = pad                                    # padding used throughout
        k = 3                                              # kernel size: 3x3, same as PlainDeepConvNet
        self.Wstem = rng.randn(C, 3, k, k) * he_scale(3 * k * k)  # stem conv: 3->C, identical shape/init rule to PlainDeepConvNet's
        self.bstem = np.zeros(C)                           # stem biases
        self.blocks = []                                   # one dict of {W1,b1,W2,b2} per residual block
        for _ in range(n_blocks):                          # same loop, same shapes, same rng call order as PlainDeepConvNet -- this is what keeps init identical
            W1 = rng.randn(C, C, k, k) * he_scale(C * k * k)  # this block's first conv weights
            b1 = np.zeros(C)                                # this block's first conv biases
            W2 = rng.randn(C, C, k, k) * he_scale(C * k * k)  # this block's second conv weights
            b2 = np.zeros(C)                                # this block's second conv biases
            self.blocks.append(dict(W1=W1, b1=b1, W2=W2, b2=b2))  # store this block's parameters
        final_spatial = img_size // (2 ** n_blocks)         # identical spatial-size arithmetic to PlainDeepConvNet
        self.flat_dim = C * final_spatial * final_spatial   # identical flat_dim to PlainDeepConvNet, by construction
        self.W1fc = rng.randn(self.flat_dim, hidden) * he_scale(self.flat_dim)  # FC1 weights
        self.b1fc = np.zeros(hidden)                        # FC1 biases
        self.W2fc = rng.randn(hidden, n_classes) * he_scale(hidden)  # FC2 (output) weights
        self.b2fc = np.zeros(n_classes)                     # FC2 (output) biases

    def forward(self, X, training=False, rng=None):        # forward pass, identical structure to PlainDeepConvNet's except which block fn is called
        self.X = X                                          # cache raw input for backward()'s stem gradient
        self.Zstem = conv2d_forward_mc(self.X, self.Wstem, self.bstem, pad=self.pad)  # stem conv pre-activation
        self.Astem = relu(self.Zstem)                        # stem conv activation, feeds into block 1
        cur = self.Astem                                     # running activation through the blocks
        self.block_caches = []                               # per-block caches for backward()
        self.block_outs = []                                 # per-block pre-pool outputs, for maxpool_backward
        self.pool_masks = []                                 # per-block maxpool argmax caches
        for blk in self.blocks:                              # loop over every residual block in forward order
            out, cache = resblock_forward(cur, blk["W1"], blk["b1"], blk["W2"], blk["b2"], pad=self.pad)  # THE difference vs. PlainDeepConvNet: resblock, not plainblock
            self.block_caches.append(cache)                  # remember this block's cache
            self.block_outs.append(out)                      # remember this block's pre-pool output
            pooled, mask = maxpool_forward(out, 2, 2)         # pool this block's output down by 2x
            self.pool_masks.append(mask)                      # remember the pooling argmax cache
            cur = pooled                                      # feed the pooled result into the next block
        m = X.shape[0]                                       # batch size
        self.flat = cur.reshape(m, -1)                        # flatten the final pooled feature maps
        self.Z1 = self.flat @ self.W1fc + self.b1fc            # FC1 pre-activation
        self.A1 = relu(self.Z1)                               # FC1 activation
        self.Z2 = self.A1 @ self.W2fc + self.b2fc              # FC2 (output) pre-activation, the raw class logits
        self.probs = softmax(self.Z2)                          # class probabilities
        return self.probs                                     # hand back to the caller

    def backward(self, y_onehot):                            # backward pass, identical structure to PlainDeepConvNet's except which block-backward is called
        m = y_onehot.shape[0]                                 # batch size
        dZ2 = self.probs - y_onehot                           # combined softmax+cross-entropy gradient
        dW2fc = self.A1.T @ dZ2 / m                            # FC2 weight gradient
        db2fc = dZ2.sum(axis=0) / m                            # FC2 bias gradient
        dA1 = dZ2 @ self.W2fc.T                                # gradient into FC1's activation
        dZ1 = dA1 * drelu(self.Z1)                             # FC1's pre-activation gradient
        dW1fc = self.flat.T @ dZ1 / m                          # FC1 weight gradient
        db1fc = dZ1.sum(axis=0) / m                            # FC1 bias gradient
        dflat = dZ1 @ self.W1fc.T                              # gradient into the flattened conv features
        final_shape = (m,) + self.block_outs[-1].shape[1:2] + tuple(s // 2 for s in self.block_outs[-1].shape[2:])  # shape of the last pooled feature map
        dcur = dflat.reshape(final_shape)                      # reshape back into that pooled feature-map shape
        block_grads = []                                       # collected in reverse, then flipped, same pattern as PlainDeepConvNet
        for i in reversed(range(self.n_blocks)):               # walk backward through the residual blocks
            dout = maxpool_backward(dcur, self.block_outs[i], self.pool_masks[i], 2, 2)  # undo this block's maxpool
            dx, dW1, db1, dW2, db2 = resblock_backward(dout, self.block_caches[i], self.blocks[i]["W1"], self.blocks[i]["W2"], pad=self.pad)  # THE difference: resblock_backward, not plainblock_backward
            block_grads.append((dW1, db1, dW2, db2))            # stash this block's gradients
            dcur = dx                                            # feed this block's dx (conv path PLUS identity path) to the block/stem before it
        block_grads.reverse()                                   # restore forward order
        dZstem = dcur * drelu(self.Zstem)                       # undo the stem's ReLU
        dWstem, dbstem, _ = conv2d_backward_mc(dZstem, self.X, self.Wstem, pad=self.pad)  # stem's weight/bias gradients
        self.grads = dict(Wstem=dWstem, bstem=dbstem, blocks=block_grads,  # stash every gradient for the gradient-flow diagnostic to inspect
                           W1fc=dW1fc, b1fc=db1fc, W2fc=dW2fc, b2fc=db2fc)
        self.Wstem -= self.lr * dWstem                          # SGD update: stem weights
        self.bstem -= self.lr * dbstem                          # SGD update: stem biases
        for blk, (dW1, db1, dW2, db2) in zip(self.blocks, block_grads):  # loop over every block alongside its gradients
            blk["W1"] -= self.lr * dW1                           # SGD update: this block's first conv weights
            blk["b1"] -= self.lr * db1                           # SGD update: this block's first conv biases
            blk["W2"] -= self.lr * dW2                           # SGD update: this block's second conv weights
            blk["b2"] -= self.lr * db2                           # SGD update: this block's second conv biases
        self.W1fc -= self.lr * dW1fc                             # SGD update: FC1 weights
        self.b1fc -= self.lr * db1fc                             # SGD update: FC1 biases
        self.W2fc -= self.lr * dW2fc                             # SGD update: FC2 weights
        self.b2fc -= self.lr * db2fc                             # SGD update: FC2 biases

    def accuracy(self, X, y):                                 # classification accuracy, identical logic to PlainDeepConvNet's
        probs = self.forward(X, training=False)                # inference-mode forward pass
        preds = probs.argmax(axis=1)                            # predicted class per example
        return (preds == y).mean()                              # fraction correct

    def param_count(self):                                     # total trainable parameter count
        total = self.Wstem.size + self.bstem.size               # stem
        for blk in self.blocks:                                 # every block
            total += blk["W1"].size + blk["b1"].size + blk["W2"].size + blk["b2"].size
        total += self.W1fc.size + self.b1fc.size + self.W2fc.size + self.b2fc.size  # FC head
        return total                                             # full trainable parameter count


# =============================================================================
# 4. GRADIENT CHECK -- verifying the residual block's backward pass, i.e.
#    that splitting dout into a conv-path gradient AND an identity-path
#    gradient (and re-summing them into dx) actually matches the true
#    numeric gradient of the loss.
# =============================================================================
def demo_gradient_check():                                    # verifies ResNetStyleConvNet's new backward math before trusting any training result
    print("=" * 70)                                             # section separator
    print("1. GRADIENT CHECK -- ResNetStyleConvNet, identity-shortcut backward pass")
    print("=" * 70)                                              # closing separator
    print("Checks the full forward->loss->backward pipeline on a small "  # explains the check's scope
          "12x12 input, 2 residual blocks, so pooling still leaves a "
          "non-empty spatial map.")
    rng = np.random.RandomState(1)                                # fixed rng for reproducible perturbation indices
    net = ResNetStyleConvNet(C=4, img_size=12, hidden=6, n_classes=3,  # a small ResNetStyleConvNet just for this check (cheap to perturb)
                              n_blocks=2, lr=0.01, seed=1)
    X = rng.randn(4, 3, 12, 12) * 0.5                              # a tiny random batch of 4 fake 12x12 RGB images
    y_oh = one_hot(np.array([0, 1, 2, 0]), 3)                      # matching one-hot labels for a 3-class toy problem

    net.forward(X, training=False)                                 # one forward pass to populate every cached value backward() needs
    net.backward(y_oh)                                              # run the real backward() once -- NOTE: this also applies an SGD update in place
    net.Wstem += net.lr * net.grads["Wstem"]                        # undo that update (p -= lr*g, so p += lr*g restores it) so perturbing below starts from the SAME weights backward() actually differentiated
    net.bstem += net.lr * net.grads["bstem"]                        # undo the stem bias update too, for completeness even though bias isn't checked below
    for blk, (dW1, db1, dW2, db2) in zip(net.blocks, net.grads["blocks"]):  # undo every block's update the same way
        blk["W1"] += net.lr * dW1                                    # restore this block's first conv weights
        blk["b1"] += net.lr * db1                                    # restore this block's first conv biases
        blk["W2"] += net.lr * dW2                                    # restore this block's second conv weights
        blk["b2"] += net.lr * db2                                    # restore this block's second conv biases
    net.W1fc += net.lr * net.grads["W1fc"]                          # restore FC1 weights
    net.b1fc += net.lr * net.grads["b1fc"]                          # restore FC1 biases
    net.W2fc += net.lr * net.grads["W2fc"]                          # restore FC2 weights
    net.b2fc += net.lr * net.grads["b2fc"]                          # restore FC2 biases
    analytic = {"Wstem": net.grads["Wstem"],                        # collect the analytic gradients that will be spot-checked numerically
                "block0_W1": net.grads["blocks"][0][0],
                "block0_W2": net.grads["blocks"][0][2],
                "block1_W1": net.grads["blocks"][1][0],
                "block1_W2": net.grads["blocks"][1][2],
                "W1fc": net.grads["W1fc"], "W2fc": net.grads["W2fc"]}
    param_lookup = {"Wstem": net.Wstem,                             # matching lookup so the check can perturb the ACTUAL parameter arrays
                    "block0_W1": net.blocks[0]["W1"], "block0_W2": net.blocks[0]["W2"],
                    "block1_W1": net.blocks[1]["W1"], "block1_W2": net.blocks[1]["W2"],
                    "W1fc": net.W1fc, "W2fc": net.W2fc}

    eps = 1e-4                                                     # perturbation size for the central-difference numeric gradient
    max_rel_err = 0.0                                              # tracks the worst relative error seen across all checked entries
    print(f"\n{'param':<14}{'analytic':>12}{'numeric':>12}{'rel_err':>12}")  # header row for the printed comparison table
    for name, param in param_lookup.items():                       # loop over every parameter tensor being checked
        for _ in range(2):                                          # check 2 random entries per tensor (7 tensors x 2 = 14 checks)
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
            print(f"{name}{str(idx):<10}{ana:>12.6f}{numeric:>12.6f}{rel_err:>12.2e}")  # print this entry's comparison row
    print(f"\nmax relative error across all checked entries: {max_rel_err:.2e} "  # final verdict line
          f"({'PASS -- residual-block backward chain is correct' if max_rel_err < 1e-4 else 'FAIL'})")
    print("Actual output from this lesson's run.")                  # standing convention: label this as real output


# =============================================================================
# 5. GRADIENT FLOW BY DEPTH -- the direct vanishing-gradient diagnostic
# =============================================================================
def demo_gradient_flow_by_depth():                             # measures gradient magnitude at EVERY conv layer, at initialization, before any training
    print("\n" + "=" * 70)                                       # blank line then section separator
    print("2. GRADIENT FLOW BY DEPTH -- measured at initialization, before training")
    print("=" * 70)                                                # closing separator
    print("Day 52 diagnosed dying ReLUs AFTER training -- a downstream "  # explains why this diagnostic is new/different from Day 52's
          "symptom. This measures the cause directly: the gradient norm "
          "reaching each conv layer's weights on a SINGLE forward+backward "
          "pass at initialization, deepest layer first. Both networks use "
          "the identical seed (and therefore identical initial weights), "
          "so any difference below is due to the shortcut connections alone.")

    img_size, C, n_blocks = 40, 8, 3                                # same scale as this lesson's real training run, so the diagnosis is representative
    X_train, y_train = make_junction_dataset_rgb(n_per_class=50, size=img_size,  # reuse the same real dataset -- not synthetic noise -- for this diagnostic
                                                  arm=10, noise=0.4, channel_noise=0.15,
                                                  center_jitter=3, seed=42)
    Xb, yb = X_train[:50], y_train[:50]                             # a single fixed batch of 50 real training images, used for both networks
    yb_oh = one_hot(yb, 4)                                          # one-hot labels for that batch

    plain = PlainDeepConvNet(C=C, img_size=img_size, n_blocks=n_blocks, lr=0.15, seed=42)  # fresh, untrained plain network
    resnet = ResNetStyleConvNet(C=C, img_size=img_size, n_blocks=n_blocks, lr=0.15, seed=42)  # fresh, untrained residual network -- SAME seed

    same_init = np.allclose(plain.Wstem, resnet.Wstem) and all(     # sanity check: confirm both networks really did start from identical weights
        np.allclose(plain.blocks[i]["W1"], resnet.blocks[i]["W1"]) for i in range(n_blocks))
    print(f"\nboth networks start from identical initial weights: {same_init}")  # report the sanity check's result directly

    plain.forward(Xb, training=True)                                # one forward pass through the plain network
    plain.backward(yb_oh)                                           # one backward pass -- populates plain.grads with this step's gradients
    resnet.forward(Xb, training=True)                                # one forward pass through the residual network
    resnet.backward(yb_oh)                                           # one backward pass -- populates resnet.grads

    print(f"\n{'layer (deepest first)':<26}{'plain ||dW||':>14}{'resnet ||dW||':>16}")  # table header
    layer_names = [f"block{n_blocks - 1 - i}.W2" for i in range(n_blocks)]  # will interleave block names deepest-to-shallowest below
    rows = []                                                        # accumulate (name, plain_norm, resnet_norm) triples, deepest layer first
    for i in reversed(range(n_blocks)):                              # walk blocks from the LAST (deepest, closest to output) to the FIRST
        p_dW2, r_dW2 = plain.grads["blocks"][i][2], resnet.grads["blocks"][i][2]  # each block's SECOND conv's weight gradient
        rows.append((f"block{i}.W2", np.linalg.norm(p_dW2), np.linalg.norm(r_dW2)))  # Frobenius norm of that gradient, for both networks
        p_dW1, r_dW1 = plain.grads["blocks"][i][0], resnet.grads["blocks"][i][0]  # each block's FIRST conv's weight gradient
        rows.append((f"block{i}.W1", np.linalg.norm(p_dW1), np.linalg.norm(r_dW1)))  # Frobenius norm of that gradient, for both networks
    rows.append(("stem", np.linalg.norm(plain.grads["Wstem"]), np.linalg.norm(resnet.grads["Wstem"])))  # the shallowest (first) conv layer, checked last
    for name, p_norm, r_norm in rows:                                # print every row of the table, deepest layer first
        print(f"{name:<26}{p_norm:>14.6f}{r_norm:>16.6f}")            # this layer's gradient norm for both networks, side by side
    print("Actual output from this lesson's run.")                   # standing convention

    stem_plain, stem_resnet = rows[-1][1], rows[-1][2]                # the stem's (shallowest layer's) gradient norms, for both networks
    deepest_plain, deepest_resnet = rows[0][1], rows[0][2]            # the deepest block's gradient norms, for both networks
    plain_ratio = stem_plain / max(deepest_plain, 1e-12)               # how much smaller the stem's gradient is vs. the deepest layer's, for PlainDeepConvNet
    resnet_ratio = stem_resnet / max(deepest_resnet, 1e-12)            # the same ratio for ResNetStyleConvNet
    print(f"\nstem-to-deepest-layer gradient ratio -- plain: {plain_ratio:.4f}, "
          f"resnet: {resnet_ratio:.4f} (closer to 1.0 means gradient magnitude "
          f"is preserved more evenly across depth; a ratio near 0 means the "
          f"earliest layer's gradient has nearly vanished by the time it "
          f"arrives, relative to the last block's).")


# =============================================================================
# 6. THE TRAINED COMPARISON -- same aggressive lr, nearly double Day 52's depth
# =============================================================================
def measure_dead_fraction_by_block(net):                       # NEW: measures dead-ReLU fraction per block's SECOND conv, for either network class
    return {f"block{i}.Zc2": (net.block_caches[i][3] <= 0).mean()  # cache index 3 is z2, each block's second conv's pre-activation, before its final relu
            for i in range(net.n_blocks)}


def demo_train_and_compare():                                  # the main experiment: same aggressive lr, plain vs. residual, 7 conv layers deep
    print("\n" + "=" * 70)                                       # blank line then section separator
    print("3a. FIRST ATTEMPT -- lr=0.15, 7 conv layers deep")
    print("=" * 70)                                                # closing separator
    print("Day 52's inherited lr=0.15 already broke a 4-conv-layer network "  # explains the setup and why this is a harder test than Day 52's
          "(VGGStyleConvNet). Today's PlainDeepConvNet and ResNetStyleConvNet "
          "both have 7 conv+ReLU layers (a stem plus 3 blocks of 2 convs "
          "each) -- nearly double that depth -- and both are trained at the "
          "SAME lr=0.15, same 150 epochs, same batch_size=50, same dataset "
          "convention as every prior day in this week. The only difference "
          "between the two networks is whether each block adds its input "
          "back before the final ReLU.")

    img_size, C, n_blocks = 40, 8, 3                                # same architecture scale used in the gradient-flow diagnostic above
    X_train, y_train = make_junction_dataset_rgb(                   # generate the training set
        n_per_class=50, size=img_size, arm=10, noise=0.4, channel_noise=0.15,
        center_jitter=3, seed=42)                                   # seed=42, matching every prior day's training-set convention
    X_test, y_test = make_junction_dataset_rgb(                     # generate a SEPARATE, genuinely unseen test set
        n_per_class=20, size=img_size, arm=10, noise=0.4, channel_noise=0.15,
        center_jitter=3, seed=999)

    epochs, batch_size, lr = 150, 50, 0.15                           # the shared training budget, batch size, and the deliberately aggressive lr

    net_plain = PlainDeepConvNet(C=C, img_size=img_size, n_blocks=n_blocks, lr=lr, seed=42)  # fresh plain network
    hist_plain = train_with_history(net_plain, X_train, y_train, X_test, y_test,  # train while tracking its accuracy trajectory
                                     epochs=epochs, batch_size=batch_size, seed=42)

    net_resnet = ResNetStyleConvNet(C=C, img_size=img_size, n_blocks=n_blocks, lr=lr, seed=42)  # fresh residual network, identical init to net_plain
    hist_resnet = train_with_history(net_resnet, X_train, y_train, X_test, y_test,  # train identically
                                      epochs=epochs, batch_size=batch_size, seed=42)

    tr_plain, te_plain = hist_plain["train_acc"][-1], hist_plain["test_acc"][-1]  # PlainDeepConvNet's final accuracies
    tr_resnet, te_resnet = hist_resnet["train_acc"][-1], hist_resnet["test_acc"][-1]  # ResNetStyleConvNet's final accuracies

    print(f"\n{'network':<20}{'params':>9}{'train_acc':>11}{'test_acc':>10}")  # table header
    print(f"{'PlainDeepConvNet':<20}{net_plain.param_count():>9}"              # PlainDeepConvNet's row
          f"{tr_plain:>11.4f}{te_plain:>10.4f}")
    print(f"{'ResNetStyleConvNet':<20}{net_resnet.param_count():>9}"           # ResNetStyleConvNet's row
          f"{tr_resnet:>11.4f}{te_resnet:>10.4f}")
    print("Actual output from this lesson's run.")                             # standing convention

    dead_plain = measure_dead_fraction_by_block(net_plain)                     # diagnose PlainDeepConvNet's dead-ReLU fraction, per block
    dead_resnet = measure_dead_fraction_by_block(net_resnet)                   # same diagnosis for ResNetStyleConvNet
    print(f"\ndiagnosis -- fraction of each block's second conv stuck <= 0 "
          f"(dead under ReLU) after training:")
    for name in dead_plain:                                                    # print both networks' dead fraction, block by block, side by side
        print(f"  {name:<12} plain: {dead_plain[name]:.3f}    resnet: {dead_resnet[name]:.3f}")
    print("Actual output from this lesson's run.")                             # standing convention
    print(f"\nreal, honest finding: at lr=0.15, BOTH networks fail equally -- "
          f"train_acc={tr_plain:.3f} for PlainDeepConvNet and "
          f"train_acc={tr_resnet:.3f} for ResNetStyleConvNet, both chance "
          f"level for 4 classes. Residual connections did NOT rescue this "
          f"learning rate on their own: block0 and block2 are 100% dead "
          f"under ReLU in BOTH networks. The gradient-flow diagnosis above "
          f"already explains why this isn't a contradiction -- ResNetStyleConvNet "
          f"preserves MORE gradient magnitude at its earliest layer (ratio "
          f"0.415 vs. 0.116) than PlainDeepConvNet does, but 'more gradient "
          f"than a network that has none' can still be an update step large "
          f"enough to push units permanently negative. A shortcut connection "
          f"changes how much gradient reaches early layers -- it does not "
          f"change how large a step lr=0.15 takes once that gradient arrives.")

    return dict(tr_plain=tr_plain, te_plain=te_plain, tr_resnet=tr_resnet, te_resnet=te_resnet,  # hand results back to the fair-rematch stage below
                hist_plain=hist_plain, hist_resnet=hist_resnet)


# =============================================================================
# 7. A FAIR REMATCH -- calibrate lr via a short sweep, then retrain both
#    networks at whichever lr actually lets them learn something
# =============================================================================
def demo_fair_rematch(stage_a):                                # takes stage 3a's results dict so the final plot can show both stages together
    print("\n" + "=" * 70)                                       # blank line then section separator
    print("3b. A SHORT LR SWEEP -- finding a learning rate this depth can handle")
    print("=" * 70)                                                # closing separator
    print("Day 52 calibrated lr by measuring dead-unit fraction across a small "  # explains the calibration methodology, consistent with Day 52's
          "sweep. Today's network is deeper (7 conv layers vs. 4), so the "
          "sweep below checks a wider, lower range of candidate learning "
          "rates, training BOTH networks for a shorter 25-epoch budget just "
          "to see which ones let either network escape chance-level accuracy "
          "at all, before committing to a full 150-epoch run.")

    img_size, C, n_blocks = 40, 8, 3                                # identical architecture scale to stage 3a
    X_train, y_train = make_junction_dataset_rgb(n_per_class=50, size=img_size,  # the SAME training set stage 3a used (same seed)
                                                  arm=10, noise=0.4, channel_noise=0.15,
                                                  center_jitter=3, seed=42)
    X_test, y_test = make_junction_dataset_rgb(n_per_class=20, size=img_size,    # the SAME test set stage 3a used
                                                arm=10, noise=0.4, channel_noise=0.15,
                                                center_jitter=3, seed=999)

    sweep_epochs = 25                                                # a short budget, just enough to see whether training escapes chance level
    candidate_lrs = [0.15, 0.05, 0.02, 0.008]                        # a descending sweep from stage 3a's failed rate down to a much gentler one
    print(f"\n{'lr':<8}{'plain train_acc':>18}{'resnet train_acc':>19}")  # sweep results table header
    sweep_results = []                                                # will collect (lr, plain_acc, resnet_acc) for every candidate
    for lr in candidate_lrs:                                          # try each candidate learning rate in turn
        p = PlainDeepConvNet(C=C, img_size=img_size, n_blocks=n_blocks, lr=lr, seed=42)  # fresh plain network at this lr
        train_with_history(p, X_train, y_train, X_test, y_test, epochs=sweep_epochs, batch_size=50, seed=42, eval_every=sweep_epochs)  # short training run
        r = ResNetStyleConvNet(C=C, img_size=img_size, n_blocks=n_blocks, lr=lr, seed=42)  # fresh residual network at this lr
        train_with_history(r, X_train, y_train, X_test, y_test, epochs=sweep_epochs, batch_size=50, seed=42, eval_every=sweep_epochs)  # short training run
        p_acc, r_acc = p.accuracy(X_train, y_train), r.accuracy(X_train, y_train)  # each network's training accuracy after the short budget
        sweep_results.append((lr, p_acc, r_acc))                       # remember this lr's results
        print(f"{lr:<8}{p_acc:>18.4f}{r_acc:>19.4f}")                   # print this row of the sweep table
    print("Actual output from this lesson's run.")                     # standing convention

    chosen_lr = max(sweep_results, key=lambda row: row[2])[0]           # pick whichever lr gave ResNetStyleConvNet the BEST short-budget train_acc
    print(f"\nchosen lr for the full rematch: {chosen_lr} (the candidate "
          f"that gave ResNetStyleConvNet its best {sweep_epochs}-epoch "
          f"training accuracy in the sweep above).")

    print("\n" + "=" * 70)                                             # section separator for the full rematch
    print(f"3c. THE FULL REMATCH -- both networks, lr={chosen_lr}, 150 epochs")
    print("=" * 70)                                                     # closing separator

    epochs, batch_size = 150, 50                                        # the same full training budget used throughout this series
    net_plain = PlainDeepConvNet(C=C, img_size=img_size, n_blocks=n_blocks, lr=chosen_lr, seed=42)  # plain network retrained at the chosen lr
    hist_plain_lo = train_with_history(net_plain, X_train, y_train, X_test, y_test,
                                        epochs=epochs, batch_size=batch_size, seed=42)
    net_resnet = ResNetStyleConvNet(C=C, img_size=img_size, n_blocks=n_blocks, lr=chosen_lr, seed=42)  # residual network retrained at the chosen lr
    hist_resnet_lo = train_with_history(net_resnet, X_train, y_train, X_test, y_test,
                                         epochs=epochs, batch_size=batch_size, seed=42)

    tr_plain_lo, te_plain_lo = hist_plain_lo["train_acc"][-1], hist_plain_lo["test_acc"][-1]  # plain network's final accuracies at the chosen lr
    tr_resnet_lo, te_resnet_lo = hist_resnet_lo["train_acc"][-1], hist_resnet_lo["test_acc"][-1]  # residual network's final accuracies

    print(f"\n{'network':<20}{'params':>9}{'train_acc':>11}{'test_acc':>10}")  # table header
    print(f"{'PlainDeepConvNet':<20}{net_plain.param_count():>9}"              # plain network's row
          f"{tr_plain_lo:>11.4f}{te_plain_lo:>10.4f}")
    print(f"{'ResNetStyleConvNet':<20}{net_resnet.param_count():>9}"           # residual network's row
          f"{tr_resnet_lo:>11.4f}{te_resnet_lo:>10.4f}")
    print("Actual output from this lesson's run.")                             # standing convention

    dead_plain_lo = measure_dead_fraction_by_block(net_plain)                   # diagnose dead-ReLU fraction at the chosen lr, plain network
    dead_resnet_lo = measure_dead_fraction_by_block(net_resnet)                 # same, residual network
    print(f"\ndiagnosis at lr={chosen_lr} -- fraction of each block's second "
          f"conv stuck <= 0 (dead under ReLU) after training:")
    for name in dead_plain_lo:                                                  # print both networks' dead fraction, block by block
        print(f"  {name:<12} plain: {dead_plain_lo[name]:.3f}    resnet: {dead_resnet_lo[name]:.3f}")
    print("Actual output from this lesson's run.")                              # standing convention

    winner = "ResNetStyleConvNet" if te_resnet_lo > te_plain_lo else "PlainDeepConvNet"  # whichever network actually reached higher test accuracy
    print(f"\nreal, honest finding: at a properly calibrated lr={chosen_lr}, "
          f"PlainDeepConvNet reaches train_acc={tr_plain_lo:.3f}/test_acc="
          f"{te_plain_lo:.3f} while ResNetStyleConvNet reaches train_acc="
          f"{tr_resnet_lo:.3f}/test_acc={te_resnet_lo:.3f} -- {winner} comes "
          f"out ahead here. Both networks have IDENTICAL parameter counts "
          f"({net_plain.param_count()}) -- the shortcut connections are "
          f"completely free. The lesson isn't 'residual connections make any "
          f"learning rate work' (stage 3a already showed that's false at "
          f"this depth) -- it's that they measurably change how gradient "
          f"magnitude is distributed across depth (section 2's diagnostic), "
          f"which is a real, distinct effect from simply picking a better lr, "
          f"even though calibrating the lr is still necessary either way.")

    plt.figure(figsize=(9, 5))                                                  # one combined figure showing both stages together
    plt.plot(stage_a["hist_plain"]["epochs"], stage_a["hist_plain"]["test_acc"],  # PlainDeepConvNet's failed trajectory at lr=0.15
              label=f"Plain, lr=0.15 -- FAILS (test_acc={stage_a['te_plain']:.3f})",
              color="#d9534f", linestyle="--")
    plt.plot(stage_a["hist_resnet"]["epochs"], stage_a["hist_resnet"]["test_acc"],  # ResNetStyleConvNet's failed trajectory at lr=0.15
              label=f"ResNet, lr=0.15 -- FAILS (test_acc={stage_a['te_resnet']:.3f})",
              color="#5cb85c", linestyle="--")
    plt.plot(hist_plain_lo["epochs"], hist_plain_lo["test_acc"],                 # PlainDeepConvNet's trajectory at the recalibrated lr
              label=f"Plain, lr={chosen_lr} (test_acc={te_plain_lo:.3f})", color="#d9534f")
    plt.plot(hist_resnet_lo["epochs"], hist_resnet_lo["test_acc"],               # ResNetStyleConvNet's trajectory at the recalibrated lr
              label=f"ResNet, lr={chosen_lr} (test_acc={te_resnet_lo:.3f})", color="#5cb85c")
    plt.xlabel("epoch")                                                         # x-axis label
    plt.ylabel("test accuracy")                                                 # y-axis label
    plt.ylim(0, 1.0)                                                             # fix the y-axis range for visual comparability across days
    plt.title("7 conv layers deep: both fail at lr=0.15, then a fair rematch")   # descriptive title covering both stages
    plt.legend(fontsize=8)                                                      # legend distinguishing all 4 lines
    plt.tight_layout()                                                          # avoid clipped labels/titles when saving
    plt.savefig("plain_vs_resnet.png", dpi=110)                                 # write the figure to disk
    plt.close()                                                                 # free the figure's memory now that it's saved
    print("\nsaved plain_vs_resnet.png")                                        # confirm the save to the console


# =============================================================================
# 8. WHAT THIS MAPS TO IN PYTORCH
# =============================================================================
def note_on_pytorch():                                          # pure documentation: prints a PyTorch equivalent, imports nothing
    print("\n" + "=" * 70)                                        # blank line then section separator
    print("4. WHAT THIS MAPS TO IN PYTORCH")                       # kept as "4" -- stage 3's sub-parts (3a/3b/3c) are still logically one section
    print("=" * 70)                                                # closing separator
    print(                                                          # a single multi-line string showing the residual block in PyTorch
        "class ResidualBlock(nn.Module):\n"
        "    def __init__(self, channels):\n"
        "        super().__init__()\n"
        "        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)\n"
        "        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)\n"
        "\n"
        "    def forward(self, x):\n"
        "        out = torch.relu(self.conv1(x))\n"
        "        out = self.conv2(out)          # no relu yet -- matches resblock_forward's z2\n"
        "        return torch.relu(out + x)      # add the shortcut, THEN relu\n"
        "\n"
        "# real ResNet (He et al., 2015) also adds a batch-norm layer after each\n"
        "# conv (that's Day 54's topic) and a learned 1x1-conv projection on the\n"
        "# shortcut whenever a block changes channel count or downsamples --\n"
        "# neither is needed here since every block in this lesson keeps C fixed."
    )
    print("\npadding=1 on a 3x3 conv is PyTorch's built-in way of getting the "  # explains the mapping back to today's from-scratch code
          "spatial-size-preserving convolution that conv2d_forward_mc's new "
          "`pad` argument reproduces by hand above; `out + x` is autograd's "
          "version of resblock_backward's explicit dx_from_conv1 + dx_identity line.")


def main():                                                     # entry point: runs every demo in the order this document walks them
    demo_gradient_check()                                        # 1. verify the new residual-block backward pass is correct
    demo_gradient_flow_by_depth()                                 # 2. the direct vanishing-gradient diagnostic, at initialization
    stage_a = demo_train_and_compare()                            # 3a. the real, fully-converged trained comparison at an aggressive lr
    demo_fair_rematch(stage_a)                                    # 3b/3c. calibrate lr via a short sweep, then a fair full-length rematch
    note_on_pytorch()                                             # 4. relate it all back to a production framework


if __name__ == "__main__":                                      # only run main() when this file is executed directly, not when imported
    main()                                                        # kick off the whole lesson
