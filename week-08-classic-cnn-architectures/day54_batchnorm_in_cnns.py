"""
Day 54: Batch Normalization in CNNs -- BN Alone Beats BN+Residual Here

Day 40 introduced batch normalization on DENSE layers: normalize each
feature to zero mean/unit variance ACROSS THE BATCH, then let a learned
gamma/beta rescale and re-shift it. Today extends that to conv layers.
The only real change: a conv layer's output has shape (m, C, H, W), and
"one feature" is now one CHANNEL, not one scalar -- so batch norm here
normalizes each channel using statistics pooled over the batch AND both
spatial axes, with one gamma/beta per channel (not per pixel).

Day 53 left two open questions. First: PlainDeepConvNet and
ResNetStyleConvNet (7 conv+ReLU layers) both failed completely at lr=0.15
-- chance-level accuracy, block0 and block2 100% dead under ReLU in BOTH
networks -- and residual connections alone did not rescue that learning
rate. Second: Day 53's own docstring predicted batch norm would be
"another tool, alongside residual connections, that real deep networks
combine to train reliably at depths far beyond what either idea achieves
alone." Today puts that prediction to a real test at the EXACT SAME
lr=0.15 that broke everything in Day 53.

Three networks are compared: PlainDeepConvNet (Day 53's control, no BN,
no residual, redefined here verbatim), PlainBNConvNet (BN added after
every conv, before its ReLU, no residual), and ResNetBNConvNet (BN AND
residual shortcuts together -- the real ResNet basic block: conv -> BN ->
ReLU -> conv -> BN -> (+x) -> ReLU). The new spatial batch-norm backward
pass is verified with a numeric gradient check first -- catching a real
bug along the way: dgamma/dbeta initially came out scaled by the batch
size m too large, because they weren't divided by m the way every other
parameter gradient in this codebase's conv/FC layers already is. A
gradient-flow-by-depth diagnostic (introduced Day 53) is extended to all
three networks at initialization: BN changes gradient distribution far
more dramatically than residual shortcuts did on their own (stem-to-
deepest ratio 1.53 for BN-only and 1.15 for BN+residual, vs. 0.12 for
the plain network -- BN-only's ratio actually exceeds 1.0, meaning the
stem's gradient is LARGER than the deepest block's).

The trained result is not the tidy "combine both fixes for the best
result" story Day 53's docstring half-predicted. At lr=0.15,
PlainDeepConvNet still fails completely (train_acc=0.250, unchanged from
Day 53). PlainBNConvNet -- batch norm ALONE, no residual shortcuts --
not only trains but reaches train_acc=1.000/test_acc=0.975, the best
result any network in this two-day arc has reached. ResNetBNConvNet
(BN and residual together, the actual published ResNet basic block)
trains too, but far more unevenly and to a lower ceiling
(train_acc=0.750/test_acc=0.738) -- its accuracy trajectory visibly
oscillates rather than climbing smoothly. Adding a raw (unnormalized)
identity shortcut on top of a normalized branch, right before the final
ReLU, changes the distribution the next layer's batch norm has to
renormalize; at this lesson's scale, that combination trained less
smoothly than batch norm by itself, not more. This does not mean
combining the two ideas is a bad one in general -- real ResNet's success
at ImageNet scale, over far more data and far deeper stacks, is well
established -- it means this lesson's own small-scale, honestly-run
comparison doesn't reproduce the "more machinery is strictly better"
story, the same caveat Days 47, 49, 52, and 53 already taught with their
own architectures.

Every line of code below carries its own comment, continuing Day 52's
convention. conv2d_forward_mc/backward_mc keep Day 53's vectorized
kernel-offset-loop implementation.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this
import matplotlib                                      # matplotlib: only its Agg backend and pyplot are used, for saving plots to disk
matplotlib.use("Agg")                                   # "Agg" backend renders to a file, not a screen -- needed since this runs headless
import matplotlib.pyplot as plt                         # pyplot: the plotting API actually called later (plt.plot, plt.savefig, ...)

np.random.seed(42)                                      # fixes numpy's GLOBAL random state so any code that forgets to pass its own rng is still reproducible

# =============================================================================
# 0. SHARED HELPERS -- reused verbatim from Day 53 (including its vectorized,
#    padding-capable conv2d_forward_mc/backward_mc)
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
    for kh in range(kH):                                  # loop over kernel offsets (e.g. 3, not out_H which could be 40) -- Day 53's vectorization
        for kw in range(kW):                              # loop over kernel column offsets
            X_slice = Xp[:, :, kh:kh + out_H, kw:kw + out_W]  # every output position's input pixel AT THIS ONE (kh, kw) offset, for the whole batch/channels/spatial map at once
            Z += np.einsum("mchw,fc->mfhw", X_slice, W[:, :, kh, kw])  # contract over input channels c, broadcast over output filters f, all (m,h,w) positions in one call
    Z += b[None, :, None, None]                            # add each filter's bias once, broadcast across the batch and every spatial position
    return Z                                              # the full convolved output, shape (m, C_out, out_H, out_W)


def conv2d_backward_mc(dZ, X, W, pad=0):                # backward pass for conv2d_forward_mc: returns dW, db, dX -- same kernel-offset-loop vectorization
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


def maxpool_forward(A, size=2, stride=2):               # 2x2 max pooling forward pass (default size/stride) -- unchanged since Day 49
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


def maxpool_backward(dOut, A, idx_cache, size=2, stride=2):  # backward pass for maxpool_forward -- unchanged since Day 49
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
            probs = net.forward(Xb, training=True, rng=rng)  # forward pass in TRAINING mode (matters for batch norm today, which always uses the CURRENT batch's own statistics)
            step_losses.append(cross_entropy_loss(probs, yb))  # record this step's loss
            net.backward(yb)                               # backward pass + parameter update for this step
        if epoch % eval_every == 0 or epoch == epochs - 1:  # only evaluate every eval_every epochs, plus always on the very last epoch
            history_epochs.append(epoch)                    # record which epoch this measurement belongs to
            history_train_acc.append(net.accuracy(X, y))    # measure accuracy on the full training set at this point
            history_test_acc.append(net.accuracy(X_test, y_test))  # and on the held-out test set
    return dict(step_losses=step_losses, epochs=history_epochs,  # bundle everything the caller might want into one dict
                train_acc=history_train_acc, test_acc=history_test_acc)


# =============================================================================
# 1. SPATIAL BATCH NORM -- Day 40's dense-layer batch norm, extended to convs
# =============================================================================
def spatial_batchnorm_forward(Z, gamma, beta, eps=1e-5):  # NEW today: batch norm for a conv layer's (m, C, H, W) output
    mu = Z.mean(axis=(0, 2, 3), keepdims=True)             # per-CHANNEL mean, pooled over the batch AND both spatial axes -- shape (1, C, 1, 1)
    var = Z.var(axis=(0, 2, 3), keepdims=True)             # per-channel variance, same pooling -- Day 40's dense version only pooled over axis=0 (the batch)
    Zhat = (Z - mu) / np.sqrt(var + eps)                   # normalize: zero mean, unit variance, per channel
    out = gamma * Zhat + beta                              # learned per-channel rescale (gamma) and re-shift (beta), broadcasting (1,C,1,1) against (m,C,H,W)
    cache = (Z, mu, var, Zhat, gamma, eps)                 # everything backward() will need
    return out, cache                                      # the normalized+rescaled output, plus its cache


def spatial_batchnorm_backward(dout, cache):             # backward pass for spatial_batchnorm_forward
    Z, mu, var, Zhat, gamma, eps = cache                   # unpack the forward pass's cached values
    m, C, H, W = Z.shape                                   # shape of this layer's pre-normalization input
    N = m * H * W                                          # total number of values POOLED per channel for the batchnorm statistics themselves (mu/var/dvar/dmu math)
    dgamma = np.sum(dout * Zhat, axis=(0, 2, 3), keepdims=True) / m  # gradient for gamma, divided by BATCH SIZE m (not N) -- matches conv2d_backward_mc's own /m on dW/db, so every parameter gradient in this network is consistently scaled to the same "mean loss over the batch" convention
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True) / m  # gradient for beta, same /m normalization as dgamma, for the same reason
    dZhat = dout * gamma                                    # undo the rescale-by-gamma step
    std_inv = 1.0 / np.sqrt(var + eps)                      # reciprocal standard deviation, reused twice below
    dvar = np.sum(dZhat * (Z - mu) * -0.5 * std_inv ** 3,   # gradient flowing into the variance, pooled over the same (batch, H, W) axes as the forward pass
                  axis=(0, 2, 3), keepdims=True)
    dmu = (np.sum(dZhat * -std_inv, axis=(0, 2, 3), keepdims=True)  # gradient flowing into the mean, via the direct path...
           + dvar * np.mean(-2 * (Z - mu), axis=(0, 2, 3), keepdims=True))  # ...plus the indirect path through variance
    dZ = dZhat * std_inv + dvar * 2 * (Z - mu) / N + dmu / N  # combine all three paths into the gradient w.r.t. this layer's ORIGINAL (pre-normalization) input
    return dZ, dgamma, dbeta                               # gradient w.r.t. Z, gamma, and beta


# =============================================================================
# 2. BLOCK-LEVEL HELPERS -- Day 53's plain/residual blocks, each with BN
#    inserted after every conv, BEFORE its ReLU (conv -> BN -> ReLU)
# =============================================================================
def plainblock_forward(x, W1, b1, W2, b2, pad=1):       # Day 53's plain block, UNCHANGED -- kept as today's no-BN control
    z1 = conv2d_forward_mc(x, W1, b1, pad=pad)            # first conv, padded so spatial size is preserved
    a1 = relu(z1)                                          # first conv's activation, no BN
    z2 = conv2d_forward_mc(a1, W2, b2, pad=pad)           # second conv, also padded
    out = relu(z2)                                         # PLAIN block: no identity, no BN
    cache = (x, z1, a1, z2)                                # everything backward() will need
    return out, cache                                     # this block's output, plus its cache


def plainblock_backward(dout, cache, W1, W2, pad=1):    # backward pass for Day 53's plain block, unchanged
    x, z1, a1, z2 = cache                                 # unpack the forward pass's cached values
    dz2 = dout * drelu(z2)                                # undo the block's final ReLU
    dW2, db2, da1 = conv2d_backward_mc(dz2, a1, W2, pad=pad)  # undo the second conv
    dz1 = da1 * drelu(z1)                                 # undo the first conv's ReLU
    dW1, db1, dx = conv2d_backward_mc(dz1, x, W1, pad=pad)  # undo the first conv
    return dx, dW1, db1, dW2, db2                         # gradient into the block's input, plus every weight/bias gradient inside it


def plainblock_bn_forward(x, W1, b1, gamma1, beta1, W2, b2, gamma2, beta2, pad=1):  # NEW today: plain block with BN after each conv
    z1 = conv2d_forward_mc(x, W1, b1, pad=pad)            # first conv
    bn1, bn1_cache = spatial_batchnorm_forward(z1, gamma1, beta1)  # normalize+rescale the first conv's output, per channel
    a1 = relu(bn1)                                         # ReLU AFTER batch norm, not before -- conv -> BN -> ReLU, matching real BN placement
    z2 = conv2d_forward_mc(a1, W2, b2, pad=pad)           # second conv
    bn2, bn2_cache = spatial_batchnorm_forward(z2, gamma2, beta2)  # normalize+rescale the second conv's output
    out = relu(bn2)                                        # PLAIN block: no identity added, just the (now BN'd) conv path
    cache = (x, z1, bn1_cache, bn1, a1, z2, bn2_cache, bn2)  # everything backward() will need, including both BN caches
    return out, cache                                     # this block's output, plus its cache


def plainblock_bn_backward(dout, cache, W1, W2, pad=1):  # backward pass for a plain+BN block
    x, z1, bn1_cache, bn1, a1, z2, bn2_cache, bn2 = cache  # unpack the forward pass's cached values
    dbn2 = dout * drelu(bn2)                              # undo the block's final ReLU (applied to bn2, not z2, since BN sits between conv2 and this ReLU)
    dz2, dgamma2, dbeta2 = spatial_batchnorm_backward(dbn2, bn2_cache)  # undo the second BN layer
    dW2, db2, da1 = conv2d_backward_mc(dz2, a1, W2, pad=pad)  # undo the second conv, get its weight/bias grads plus gradient into a1
    dbn1 = da1 * drelu(bn1)                               # undo the first conv's ReLU (applied to bn1, not z1)
    dz1, dgamma1, dbeta1 = spatial_batchnorm_backward(dbn1, bn1_cache)  # undo the first BN layer
    dW1, db1, dx = conv2d_backward_mc(dz1, x, W1, pad=pad)  # undo the first conv, get its weight/bias grads plus gradient into the block's input
    return dx, dW1, db1, dgamma1, dbeta1, dW2, db2, dgamma2, dbeta2  # gradient into the block's input, plus every weight/bias/gamma/beta gradient inside it


def resblock_bn_forward(x, W1, b1, gamma1, beta1, W2, b2, gamma2, beta2, pad=1):  # NEW today: residual block WITH BN -- the real ResNet basic block
    z1 = conv2d_forward_mc(x, W1, b1, pad=pad)            # first conv
    bn1, bn1_cache = spatial_batchnorm_forward(z1, gamma1, beta1)  # normalize+rescale the first conv's output
    a1 = relu(bn1)                                         # ReLU after BN
    z2 = conv2d_forward_mc(a1, W2, b2, pad=pad)           # second conv
    bn2, bn2_cache = spatial_batchnorm_forward(z2, gamma2, beta2)  # normalize+rescale the second conv's output -- NOTE: no ReLU yet, matching Day 53's resblock_forward
    out = relu(bn2 + x)                                    # add the block's OWN INPUT to the (BN'd) conv path, THEN relu -- the one line that makes this a residual block
    cache = (x, z1, bn1_cache, bn1, a1, z2, bn2_cache, bn2)  # everything backward() will need
    return out, cache                                     # this block's output, plus its cache


def resblock_bn_backward(dout, cache, W1, W2, pad=1):   # backward pass for a residual+BN block
    x, z1, bn1_cache, bn1, a1, z2, bn2_cache, bn2 = cache  # unpack the forward pass's cached values
    dsum = dout * drelu(bn2 + x)                          # undo the outer ReLU (applied to the SUM bn2+x, not bn2 alone)
    dbn2 = dsum                                            # the sum's gradient flows unchanged into bn2 (d(bn2+x)/dbn2 = 1)
    dx_identity = dsum                                     # AND unchanged into x directly (d(bn2+x)/dx = 1) -- the residual shortcut's gradient path, same as Day 53
    dz2, dgamma2, dbeta2 = spatial_batchnorm_backward(dbn2, bn2_cache)  # undo the second BN layer along the conv path
    dW2, db2, da1 = conv2d_backward_mc(dz2, a1, W2, pad=pad)  # undo the second conv
    dbn1 = da1 * drelu(bn1)                               # undo the first conv's ReLU (applied to bn1)
    dz1, dgamma1, dbeta1 = spatial_batchnorm_backward(dbn1, bn1_cache)  # undo the first BN layer
    dW1, db1, dx_from_conv1 = conv2d_backward_mc(dz1, x, W1, pad=pad)  # undo the first conv, get the conv path's contribution to dx
    dx = dx_from_conv1 + dx_identity                       # combine the conv path's gradient with the shortcut's UNATTENUATED gradient, exactly as Day 53 did
    return dx, dW1, db1, dgamma1, dbeta1, dW2, db2, dgamma2, dbeta2  # gradient into the block's input, plus every weight/bias/gamma/beta gradient inside it


# =============================================================================
# 3. PLAINDEEPCONVNET -- Day 53's control network, redefined here verbatim so
#    today's comparison runs all three networks in the same file, same run
# =============================================================================
class PlainDeepConvNet:                                  # today's no-BN, no-residual control: 7 conv+ReLU layers, unchanged from Day 53
    """A stem 3x3 conv (3->C) followed by n_blocks plain blocks (each
    C->C->C, padded so spatial size is preserved), a 2x2 maxpool after
    EACH block, then the standard FC head. Reused verbatim from Day 53 as
    today's baseline: does BN alone, or BN+residual together, beat this
    at the SAME lr that made it fail completely last time?"""

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
        final_spatial = img_size // (2 ** n_blocks)         # spatial size after n_blocks rounds of 2x2 pooling
        self.flat_dim = C * final_spatial * final_spatial   # total flattened feature count feeding into the first FC layer
        self.W1fc = rng.randn(self.flat_dim, hidden) * he_scale(self.flat_dim)  # FC1 weights
        self.b1fc = np.zeros(hidden)                        # FC1 biases
        self.W2fc = rng.randn(hidden, n_classes) * he_scale(hidden)  # FC2 (output) weights
        self.b2fc = np.zeros(n_classes)                     # FC2 (output) biases

    def forward(self, X, training=False, rng=None):        # forward pass; training/rng args kept only for call-signature parity with today's BN networks
        self.X = X                                          # cache raw input for backward()'s stem gradient
        self.Zstem = conv2d_forward_mc(self.X, self.Wstem, self.bstem, pad=self.pad)  # stem conv pre-activation
        self.Astem = relu(self.Zstem)                        # stem conv activation, feeds into block 1
        cur = self.Astem                                     # `cur` tracks the running activation as it passes through each block
        self.block_caches = []                               # will hold one cache per block, needed by backward()
        self.block_outs = []                                 # will hold each block's PRE-pool output, needed by maxpool_backward's `A` argument
        self.pool_masks = []                                 # will hold each block's maxpool argmax cache
        for blk in self.blocks:                              # loop over every block in forward order
            out, cache = plainblock_forward(cur, blk["W1"], blk["b1"], blk["W2"], blk["b2"], pad=self.pad)  # run this block (no shortcut, no BN)
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
# 4. PLAINBNCONVNET -- same architecture, BN inserted after every conv
# =============================================================================
class PlainBNConvNet:                                    # today's first new network: plain blocks, but conv -> BN -> ReLU throughout
    """Identical architecture to PlainDeepConvNet -- same stem, same
    n_blocks, same channel width C, same FC head -- except every conv is
    followed by spatial batch norm BEFORE its ReLU. Adds 2 parameters
    (gamma, beta) per channel per conv layer -- negligible next to the
    conv weights, but enough to test whether BN alone fixes Day 53's
    lr=0.15 training failure."""

    def __init__(self, C=8, img_size=40, hidden=16, n_classes=4,  # identical constructor signature to PlainDeepConvNet
                 n_blocks=3, lr=0.15, seed=42, pad=1):
        rng = np.random.RandomState(seed)                 # same seed as PlainDeepConvNet gives both networks IDENTICAL initial conv/FC weights
        self.lr = lr                                      # learning rate, used in backward()
        self.n_blocks = n_blocks                          # number of blocks
        self.pad = pad                                    # padding used throughout
        k = 3                                              # kernel size: 3x3
        self.Wstem = rng.randn(C, 3, k, k) * he_scale(3 * k * k)  # stem conv weights, identical init rule to PlainDeepConvNet's
        self.bstem = np.zeros(C)                           # stem conv biases
        self.gamma_stem = np.ones((1, C, 1, 1))            # stem's BN scale, initialized to 1 (starts as a no-op rescale, matching Day 40's convention)
        self.beta_stem = np.zeros((1, C, 1, 1))            # stem's BN shift, initialized to 0 (starts as a no-op shift)
        self.blocks = []                                   # one dict of {W1,b1,gamma1,beta1,W2,b2,gamma2,beta2} per block
        for _ in range(n_blocks):                          # same loop, same shapes, same rng call ORDER for conv weights as PlainDeepConvNet
            W1 = rng.randn(C, C, k, k) * he_scale(C * k * k)  # this block's first conv weights
            b1 = np.zeros(C)                                # this block's first conv biases
            gamma1 = np.ones((1, C, 1, 1))                  # this block's first BN scale
            beta1 = np.zeros((1, C, 1, 1))                  # this block's first BN shift
            W2 = rng.randn(C, C, k, k) * he_scale(C * k * k)  # this block's second conv weights
            b2 = np.zeros(C)                                # this block's second conv biases
            gamma2 = np.ones((1, C, 1, 1))                  # this block's second BN scale
            beta2 = np.zeros((1, C, 1, 1))                  # this block's second BN shift
            self.blocks.append(dict(W1=W1, b1=b1, gamma1=gamma1, beta1=beta1,  # store this block's parameters together
                                     W2=W2, b2=b2, gamma2=gamma2, beta2=beta2))
        final_spatial = img_size // (2 ** n_blocks)         # identical spatial-size arithmetic to PlainDeepConvNet
        self.flat_dim = C * final_spatial * final_spatial   # identical flat_dim to PlainDeepConvNet, by construction
        self.W1fc = rng.randn(self.flat_dim, hidden) * he_scale(self.flat_dim)  # FC1 weights
        self.b1fc = np.zeros(hidden)                        # FC1 biases
        self.W2fc = rng.randn(hidden, n_classes) * he_scale(hidden)  # FC2 (output) weights
        self.b2fc = np.zeros(n_classes)                     # FC2 (output) biases

    def forward(self, X, training=False, rng=None):        # forward pass; BN always uses the CURRENT batch's own statistics (Day 40's convention, no running stats)
        self.X = X                                          # cache raw input for backward()'s stem gradient
        self.Zstem = conv2d_forward_mc(self.X, self.Wstem, self.bstem, pad=self.pad)  # stem conv pre-activation
        self.BNstem, self.bn_stem_cache = spatial_batchnorm_forward(self.Zstem, self.gamma_stem, self.beta_stem)  # normalize+rescale the stem's output
        self.Astem = relu(self.BNstem)                       # ReLU AFTER batch norm, feeds into block 1
        cur = self.Astem                                     # running activation through the blocks
        self.block_caches = []                               # per-block caches for backward()
        self.block_outs = []                                 # per-block pre-pool outputs, for maxpool_backward
        self.pool_masks = []                                 # per-block maxpool argmax caches
        for blk in self.blocks:                              # loop over every block in forward order
            out, cache = plainblock_bn_forward(cur, blk["W1"], blk["b1"], blk["gamma1"], blk["beta1"],  # run this block: conv -> BN -> relu, twice, no shortcut
                                                blk["W2"], blk["b2"], blk["gamma2"], blk["beta2"], pad=self.pad)
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

    def backward(self, y_onehot):                            # backward pass, same overall shape as PlainDeepConvNet's, plus gamma/beta grads
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
        block_grads = []                                       # collected in reverse, then flipped
        for i in reversed(range(self.n_blocks)):               # walk backward through the blocks
            dout = maxpool_backward(dcur, self.block_outs[i], self.pool_masks[i], 2, 2)  # undo this block's maxpool
            dx, dW1, db1, dgamma1, dbeta1, dW2, db2, dgamma2, dbeta2 = plainblock_bn_backward(  # undo this plain+BN block
                dout, self.block_caches[i], self.blocks[i]["W1"], self.blocks[i]["W2"], pad=self.pad)
            block_grads.append((dW1, db1, dgamma1, dbeta1, dW2, db2, dgamma2, dbeta2))  # stash this block's gradients
            dcur = dx                                            # feed this block's dx to the block/stem before it
        block_grads.reverse()                                   # restore forward order
        dBNstem = dcur * drelu(self.BNstem)                     # undo the stem's ReLU (applied to BNstem, not Zstem)
        dZstem, dgamma_stem, dbeta_stem = spatial_batchnorm_backward(dBNstem, self.bn_stem_cache)  # undo the stem's BN layer
        dWstem, dbstem, _ = conv2d_backward_mc(dZstem, self.X, self.Wstem, pad=self.pad)  # stem's conv weight/bias gradients
        self.grads = dict(Wstem=dWstem, bstem=dbstem, gamma_stem=dgamma_stem, beta_stem=dbeta_stem,  # stash every gradient for diagnostics
                           blocks=block_grads, W1fc=dW1fc, b1fc=db1fc, W2fc=dW2fc, b2fc=db2fc)
        self.Wstem -= self.lr * dWstem                          # SGD update: stem conv weights
        self.bstem -= self.lr * dbstem                          # SGD update: stem conv biases
        self.gamma_stem -= self.lr * dgamma_stem                # SGD update: stem BN scale
        self.beta_stem -= self.lr * dbeta_stem                  # SGD update: stem BN shift
        for blk, (dW1, db1, dgamma1, dbeta1, dW2, db2, dgamma2, dbeta2) in zip(self.blocks, block_grads):  # loop over every block with its gradients
            blk["W1"] -= self.lr * dW1                           # SGD update: first conv weights
            blk["b1"] -= self.lr * db1                           # SGD update: first conv biases
            blk["gamma1"] -= self.lr * dgamma1                   # SGD update: first BN scale
            blk["beta1"] -= self.lr * dbeta1                     # SGD update: first BN shift
            blk["W2"] -= self.lr * dW2                           # SGD update: second conv weights
            blk["b2"] -= self.lr * db2                           # SGD update: second conv biases
            blk["gamma2"] -= self.lr * dgamma2                   # SGD update: second BN scale
            blk["beta2"] -= self.lr * dbeta2                     # SGD update: second BN shift
        self.W1fc -= self.lr * dW1fc                             # SGD update: FC1 weights
        self.b1fc -= self.lr * db1fc                             # SGD update: FC1 biases
        self.W2fc -= self.lr * dW2fc                             # SGD update: FC2 weights
        self.b2fc -= self.lr * db2fc                             # SGD update: FC2 biases

    def accuracy(self, X, y):                                 # classification accuracy -- calls forward on the WHOLE array as one batch, so BN's
        probs = self.forward(X, training=False)                # "current batch" statistics are just that whole dataset's own per-channel stats
        preds = probs.argmax(axis=1)                            # predicted class per example
        return (preds == y).mean()                              # fraction correct

    def param_count(self):                                     # total trainable parameter count, including every gamma/beta
        total = self.Wstem.size + self.bstem.size + self.gamma_stem.size + self.beta_stem.size  # stem conv + stem BN
        for blk in self.blocks:                                 # every block's conv + BN parameters
            total += (blk["W1"].size + blk["b1"].size + blk["gamma1"].size + blk["beta1"].size +
                      blk["W2"].size + blk["b2"].size + blk["gamma2"].size + blk["beta2"].size)
        total += self.W1fc.size + self.b1fc.size + self.W2fc.size + self.b2fc.size  # FC head
        return total                                             # full trainable parameter count


# =============================================================================
# 5. RESNETBNCONVNET -- residual shortcuts AND batch norm together (the real
#    ResNet basic block: conv -> BN -> ReLU -> conv -> BN -> (+x) -> ReLU)
# =============================================================================
class ResNetBNConvNet:                                   # today's second new network: combines Day 53's residual idea with today's BN
    """Identical architecture to PlainBNConvNet, except each block adds
    its own input back (after the second BN, before the final ReLU) --
    exactly mirroring PlainBNConvNet vs. PlainDeepConvNet's relationship
    from the residual side. Same parameter count as PlainBNConvNet
    (shortcuts are still free); this is the closest this series gets to
    a real ResNet basic block."""

    def __init__(self, C=8, img_size=40, hidden=16, n_classes=4,  # identical constructor signature to PlainBNConvNet, for a fair same-seed comparison
                 n_blocks=3, lr=0.15, seed=42, pad=1):
        rng = np.random.RandomState(seed)                 # same seed as PlainDeepConvNet/PlainBNConvNet gives all three networks IDENTICAL initial conv/FC weights
        self.lr = lr                                      # learning rate, used in backward()
        self.n_blocks = n_blocks                          # number of residual blocks
        self.pad = pad                                    # padding used throughout
        k = 3                                              # kernel size: 3x3
        self.Wstem = rng.randn(C, 3, k, k) * he_scale(3 * k * k)  # stem conv weights, identical init rule to the other two networks
        self.bstem = np.zeros(C)                           # stem conv biases
        self.gamma_stem = np.ones((1, C, 1, 1))            # stem's BN scale
        self.beta_stem = np.zeros((1, C, 1, 1))            # stem's BN shift
        self.blocks = []                                   # one dict of {W1,b1,gamma1,beta1,W2,b2,gamma2,beta2} per residual block
        for _ in range(n_blocks):                          # same loop, same shapes, same rng call order as PlainBNConvNet -- keeps init identical
            W1 = rng.randn(C, C, k, k) * he_scale(C * k * k)  # this block's first conv weights
            b1 = np.zeros(C)                                # this block's first conv biases
            gamma1 = np.ones((1, C, 1, 1))                  # this block's first BN scale
            beta1 = np.zeros((1, C, 1, 1))                  # this block's first BN shift
            W2 = rng.randn(C, C, k, k) * he_scale(C * k * k)  # this block's second conv weights
            b2 = np.zeros(C)                                # this block's second conv biases
            gamma2 = np.ones((1, C, 1, 1))                  # this block's second BN scale
            beta2 = np.zeros((1, C, 1, 1))                  # this block's second BN shift
            self.blocks.append(dict(W1=W1, b1=b1, gamma1=gamma1, beta1=beta1,  # store this block's parameters together
                                     W2=W2, b2=b2, gamma2=gamma2, beta2=beta2))
        final_spatial = img_size // (2 ** n_blocks)         # identical spatial-size arithmetic to the other two networks
        self.flat_dim = C * final_spatial * final_spatial   # identical flat_dim, by construction
        self.W1fc = rng.randn(self.flat_dim, hidden) * he_scale(self.flat_dim)  # FC1 weights
        self.b1fc = np.zeros(hidden)                        # FC1 biases
        self.W2fc = rng.randn(hidden, n_classes) * he_scale(hidden)  # FC2 (output) weights
        self.b2fc = np.zeros(n_classes)                     # FC2 (output) biases

    def forward(self, X, training=False, rng=None):        # forward pass, identical structure to PlainBNConvNet's except which block fn is called
        self.X = X                                          # cache raw input for backward()'s stem gradient
        self.Zstem = conv2d_forward_mc(self.X, self.Wstem, self.bstem, pad=self.pad)  # stem conv pre-activation
        self.BNstem, self.bn_stem_cache = spatial_batchnorm_forward(self.Zstem, self.gamma_stem, self.beta_stem)  # normalize+rescale the stem's output
        self.Astem = relu(self.BNstem)                       # ReLU after BN, feeds into block 1 -- the stem itself has no shortcut (nothing to add yet)
        cur = self.Astem                                     # running activation through the blocks
        self.block_caches = []                               # per-block caches for backward()
        self.block_outs = []                                 # per-block pre-pool outputs
        self.pool_masks = []                                 # per-block maxpool argmax caches
        for blk in self.blocks:                              # loop over every residual block in forward order
            out, cache = resblock_bn_forward(cur, blk["W1"], blk["b1"], blk["gamma1"], blk["beta1"],  # THE difference vs. PlainBNConvNet: resblock_bn, not plainblock_bn
                                              blk["W2"], blk["b2"], blk["gamma2"], blk["beta2"], pad=self.pad)
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

    def backward(self, y_onehot):                            # backward pass, identical structure to PlainBNConvNet's except which block-backward is called
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
        block_grads = []                                       # collected in reverse, then flipped
        for i in reversed(range(self.n_blocks)):               # walk backward through the residual blocks
            dout = maxpool_backward(dcur, self.block_outs[i], self.pool_masks[i], 2, 2)  # undo this block's maxpool
            dx, dW1, db1, dgamma1, dbeta1, dW2, db2, dgamma2, dbeta2 = resblock_bn_backward(  # undo this residual+BN block
                dout, self.block_caches[i], self.blocks[i]["W1"], self.blocks[i]["W2"], pad=self.pad)
            block_grads.append((dW1, db1, dgamma1, dbeta1, dW2, db2, dgamma2, dbeta2))  # stash this block's gradients
            dcur = dx                                            # feed this block's dx (conv path PLUS identity path) to the block/stem before it
        block_grads.reverse()                                   # restore forward order
        dBNstem = dcur * drelu(self.BNstem)                     # undo the stem's ReLU
        dZstem, dgamma_stem, dbeta_stem = spatial_batchnorm_backward(dBNstem, self.bn_stem_cache)  # undo the stem's BN layer
        dWstem, dbstem, _ = conv2d_backward_mc(dZstem, self.X, self.Wstem, pad=self.pad)  # stem's conv weight/bias gradients
        self.grads = dict(Wstem=dWstem, bstem=dbstem, gamma_stem=dgamma_stem, beta_stem=dbeta_stem,  # stash every gradient for diagnostics
                           blocks=block_grads, W1fc=dW1fc, b1fc=db1fc, W2fc=dW2fc, b2fc=db2fc)
        self.Wstem -= self.lr * dWstem                          # SGD update: stem conv weights
        self.bstem -= self.lr * dbstem                          # SGD update: stem conv biases
        self.gamma_stem -= self.lr * dgamma_stem                # SGD update: stem BN scale
        self.beta_stem -= self.lr * dbeta_stem                  # SGD update: stem BN shift
        for blk, (dW1, db1, dgamma1, dbeta1, dW2, db2, dgamma2, dbeta2) in zip(self.blocks, block_grads):  # loop over every block with its gradients
            blk["W1"] -= self.lr * dW1                           # SGD update: first conv weights
            blk["b1"] -= self.lr * db1                           # SGD update: first conv biases
            blk["gamma1"] -= self.lr * dgamma1                   # SGD update: first BN scale
            blk["beta1"] -= self.lr * dbeta1                     # SGD update: first BN shift
            blk["W2"] -= self.lr * dW2                           # SGD update: second conv weights
            blk["b2"] -= self.lr * db2                           # SGD update: second conv biases
            blk["gamma2"] -= self.lr * dgamma2                   # SGD update: second BN scale
            blk["beta2"] -= self.lr * dbeta2                     # SGD update: second BN shift
        self.W1fc -= self.lr * dW1fc                             # SGD update: FC1 weights
        self.b1fc -= self.lr * db1fc                             # SGD update: FC1 biases
        self.W2fc -= self.lr * dW2fc                             # SGD update: FC2 weights
        self.b2fc -= self.lr * db2fc                             # SGD update: FC2 biases

    def accuracy(self, X, y):                                 # classification accuracy, identical logic to the other two networks'
        probs = self.forward(X, training=False)                # inference-mode forward pass
        preds = probs.argmax(axis=1)                            # predicted class per example
        return (preds == y).mean()                              # fraction correct

    def param_count(self):                                     # total trainable parameter count
        total = self.Wstem.size + self.bstem.size + self.gamma_stem.size + self.beta_stem.size  # stem conv + stem BN
        for blk in self.blocks:                                 # every block's conv + BN parameters
            total += (blk["W1"].size + blk["b1"].size + blk["gamma1"].size + blk["beta1"].size +
                      blk["W2"].size + blk["b2"].size + blk["gamma2"].size + blk["beta2"].size)
        total += self.W1fc.size + self.b1fc.size + self.W2fc.size + self.b2fc.size  # FC head
        return total                                             # full trainable parameter count


# =============================================================================
# 6. GRADIENT CHECK -- verifying the spatial batch-norm backward pass, inside
#    the most complex of today's networks (residual shortcuts AND BN together)
# =============================================================================
def demo_gradient_check():                                    # verifies ResNetBNConvNet's backward math before trusting any training result
    print("=" * 70)                                             # section separator
    print("1. GRADIENT CHECK -- ResNetBNConvNet, spatial batch norm + residual shortcuts")
    print("=" * 70)                                              # closing separator
    print("Checks the full forward->loss->backward pipeline on a small "  # explains the check's scope
          "12x12 input, 2 residual+BN blocks, so pooling still leaves a "
          "non-empty spatial map and each channel still pools enough "
          "values (batch x H x W) for batch-norm statistics to be meaningful.")
    rng = np.random.RandomState(1)                                # fixed rng for reproducible perturbation indices
    net = ResNetBNConvNet(C=4, img_size=12, hidden=6, n_classes=3,  # a small ResNetBNConvNet just for this check (cheap to perturb)
                           n_blocks=2, lr=0.01, seed=1)
    X = rng.randn(4, 3, 12, 12) * 0.5                              # a tiny random batch of 4 fake 12x12 RGB images
    y_oh = one_hot(np.array([0, 1, 2, 0]), 3)                      # matching one-hot labels for a 3-class toy problem

    net.forward(X, training=False)                                 # one forward pass to populate every cached value backward() needs
    net.backward(y_oh)                                              # run the real backward() once -- NOTE: this also applies an SGD update in place
    net.Wstem += net.lr * net.grads["Wstem"]                        # undo that update so perturbing below starts from the SAME weights backward() actually differentiated
    net.bstem += net.lr * net.grads["bstem"]                        # undo the stem conv bias update
    net.gamma_stem += net.lr * net.grads["gamma_stem"]              # undo the stem BN scale update
    net.beta_stem += net.lr * net.grads["beta_stem"]                # undo the stem BN shift update
    for blk, g in zip(net.blocks, net.grads["blocks"]):              # undo every block's update the same way
        dW1, db1, dgamma1, dbeta1, dW2, db2, dgamma2, dbeta2 = g      # unpack this block's 8 gradients
        blk["W1"] += net.lr * dW1                                    # restore first conv weights
        blk["b1"] += net.lr * db1                                    # restore first conv biases
        blk["gamma1"] += net.lr * dgamma1                            # restore first BN scale
        blk["beta1"] += net.lr * dbeta1                              # restore first BN shift
        blk["W2"] += net.lr * dW2                                    # restore second conv weights
        blk["b2"] += net.lr * db2                                    # restore second conv biases
        blk["gamma2"] += net.lr * dgamma2                            # restore second BN scale
        blk["beta2"] += net.lr * dbeta2                              # restore second BN shift
    net.W1fc += net.lr * net.grads["W1fc"]                          # restore FC1 weights
    net.b1fc += net.lr * net.grads["b1fc"]                          # restore FC1 biases
    net.W2fc += net.lr * net.grads["W2fc"]                          # restore FC2 weights
    net.b2fc += net.lr * net.grads["b2fc"]                          # restore FC2 biases

    analytic = {"Wstem": net.grads["Wstem"], "gamma_stem": net.grads["gamma_stem"],  # collect the analytic gradients that will be spot-checked
                "block0_W1": net.grads["blocks"][0][0], "block0_gamma1": net.grads["blocks"][0][2],
                "block0_W2": net.grads["blocks"][0][4], "block0_gamma2": net.grads["blocks"][0][6],
                "block1_W2": net.grads["blocks"][1][4], "W1fc": net.grads["W1fc"], "W2fc": net.grads["W2fc"]}
    param_lookup = {"Wstem": net.Wstem, "gamma_stem": net.gamma_stem,             # matching lookup so the check can perturb the ACTUAL parameter arrays
                    "block0_W1": net.blocks[0]["W1"], "block0_gamma1": net.blocks[0]["gamma1"],
                    "block0_W2": net.blocks[0]["W2"], "block0_gamma2": net.blocks[0]["gamma2"],
                    "block1_W2": net.blocks[1]["W2"], "W1fc": net.W1fc, "W2fc": net.W2fc}

    eps = 1e-4                                                     # perturbation size for the central-difference numeric gradient
    max_rel_err = 0.0                                              # tracks the worst relative error seen across all checked entries
    print(f"\n{'param':<16}{'analytic':>12}{'numeric':>12}{'rel_err':>12}")  # header row for the printed comparison table
    for name, param in param_lookup.items():                       # loop over every parameter tensor being checked
        for _ in range(2):                                          # check 2 random entries per tensor (9 tensors x 2 = 18 checks)
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
            print(f"{name}{str(idx):<12}{ana:>12.6f}{numeric:>12.6f}{rel_err:>12.2e}")  # print this entry's comparison row
    print(f"\nmax relative error across all checked entries: {max_rel_err:.2e} "  # final verdict line
          f"({'PASS -- spatial batch norm + residual backward chain is correct' if max_rel_err < 1e-4 else 'FAIL'})")
    print("Actual output from this lesson's run.")                  # standing convention: label this as real output


# =============================================================================
# 7. GRADIENT FLOW BY DEPTH -- extending Day 53's diagnostic to all three
#    networks: no-BN/no-residual, BN-only, and BN+residual together
# =============================================================================
def demo_gradient_flow_by_depth():                             # measures gradient magnitude at EVERY conv layer, at initialization, for all 3 networks
    print("\n" + "=" * 70)                                       # blank line then section separator
    print("2. GRADIENT FLOW BY DEPTH -- measured at initialization, before training")
    print("=" * 70)                                                # closing separator
    print("Day 53 showed residual shortcuts change how gradient magnitude "  # explains why this diagnostic is repeated today with a third network added
          "is distributed across depth. This repeats that exact diagnostic "
          "with a THIRD network added: BN with no shortcut. All three "
          "networks share identical initial conv/FC weights (same seed, "
          "same rng call order), so any difference in the numbers below is "
          "attributable to BN and/or the shortcut connections alone.")

    img_size, C, n_blocks = 40, 8, 3                                # same scale as this lesson's real training run
    X_train, y_train = make_junction_dataset_rgb(n_per_class=50, size=img_size,  # reuse the same real dataset for this diagnostic
                                                  arm=10, noise=0.4, channel_noise=0.15,
                                                  center_jitter=3, seed=42)
    Xb, yb = X_train[:50], y_train[:50]                             # a single fixed batch of 50 real training images, used for all three networks
    yb_oh = one_hot(yb, 4)                                          # one-hot labels for that batch

    plain = PlainDeepConvNet(C=C, img_size=img_size, n_blocks=n_blocks, lr=0.15, seed=42)  # fresh, untrained plain network (no BN, no residual)
    bn_only = PlainBNConvNet(C=C, img_size=img_size, n_blocks=n_blocks, lr=0.15, seed=42)  # fresh, untrained BN-only network -- SAME seed
    bn_res = ResNetBNConvNet(C=C, img_size=img_size, n_blocks=n_blocks, lr=0.15, seed=42)  # fresh, untrained BN+residual network -- SAME seed

    same_init = (np.allclose(plain.Wstem, bn_only.Wstem) and np.allclose(plain.Wstem, bn_res.Wstem) and  # sanity check: all three start identical
                 all(np.allclose(plain.blocks[i]["W1"], bn_only.blocks[i]["W1"]) and
                     np.allclose(plain.blocks[i]["W1"], bn_res.blocks[i]["W1"]) for i in range(n_blocks)))
    print(f"\nall three networks start from identical initial conv/FC weights: {same_init}")  # report the sanity check's result directly

    plain.forward(Xb, training=True); plain.backward(yb_oh)          # one forward+backward through the plain network
    bn_only.forward(Xb, training=True); bn_only.backward(yb_oh)      # one forward+backward through the BN-only network
    bn_res.forward(Xb, training=True); bn_res.backward(yb_oh)        # one forward+backward through the BN+residual network

    print(f"\n{'layer (deepest first)':<26}{'plain ||dW||':>14}{'BN-only ||dW||':>16}{'BN+res ||dW||':>16}")  # table header
    rows = []                                                        # accumulate rows, deepest layer first
    for i in reversed(range(n_blocks)):                              # walk blocks from the LAST (deepest) to the FIRST
        p2, b2, r2 = plain.grads["blocks"][i][2], bn_only.grads["blocks"][i][4], bn_res.grads["blocks"][i][4]  # each network's block-i second-conv weight gradient (index differs since BN networks' block_grads tuples are longer)
        rows.append((f"block{i}.W2", np.linalg.norm(p2), np.linalg.norm(b2), np.linalg.norm(r2)))  # Frobenius norm of that gradient, all three networks
        p1, b1_, r1 = plain.grads["blocks"][i][0], bn_only.grads["blocks"][i][0], bn_res.grads["blocks"][i][0]  # each network's block-i first-conv weight gradient
        rows.append((f"block{i}.W1", np.linalg.norm(p1), np.linalg.norm(b1_), np.linalg.norm(r1)))  # Frobenius norm, all three networks
    rows.append(("stem", np.linalg.norm(plain.grads["Wstem"]), np.linalg.norm(bn_only.grads["Wstem"]), np.linalg.norm(bn_res.grads["Wstem"])))  # shallowest layer, checked last
    for name, p, b, r in rows:                                       # print every row of the table, deepest layer first
        print(f"{name:<26}{p:>14.6f}{b:>16.6f}{r:>16.6f}")            # this layer's gradient norm for all three networks, side by side
    print("Actual output from this lesson's run.")                   # standing convention

    stem_row, deep_row = rows[-1], rows[0]                            # the stem's (shallowest) and the deepest block's rows
    ratios = [stem_row[k] / max(deep_row[k], 1e-12) for k in (1, 2, 3)]  # stem-to-deepest ratio for plain, BN-only, BN+res respectively
    print(f"\nstem-to-deepest-layer gradient ratio -- plain: {ratios[0]:.4f}, "
          f"BN-only: {ratios[1]:.4f}, BN+residual: {ratios[2]:.4f} (closer to "
          f"1.0 means gradient magnitude is preserved more evenly across depth).")


# =============================================================================
# 8. THE TRAINED COMPARISON -- all three networks, Day 53's SAME failing lr
# =============================================================================
def measure_dead_fraction_plain(net):                          # dead-ReLU fraction per block's second conv, for PlainDeepConvNet (no BN)
    return {f"block{i}.Zc2": (net.block_caches[i][3] <= 0).mean() for i in range(net.n_blocks)}  # cache index 3 is z2 for plainblock_forward


def measure_dead_fraction_bn(net):                              # dead-ReLU fraction per block's second conv, for either BN network (checks the POST-BN value)
    return {f"block{i}.bn2": (net.block_caches[i][7] <= 0).mean() for i in range(net.n_blocks)}  # cache index 7 is bn2 for plainblock_bn/resblock_bn forward


def demo_train_and_compare():                                  # the main experiment: all 3 networks, identical lr=0.15, 150 epochs
    print("\n" + "=" * 70)                                       # blank line then section separator
    print("3. THE TRAINED COMPARISON -- lr=0.15, does BN alone (or BN+residual) fix Day 53's failure?")
    print("=" * 70)                                                # closing separator
    print("Day 53's PlainDeepConvNet and ResNetStyleConvNet (no BN) both "  # explains the setup, citing Day 53's already-verified real numbers
          "failed completely at lr=0.15 -- train_acc=0.250 for both, chance "
          "level. Today retrains that SAME PlainDeepConvNet (redefined here, "
          "unchanged) alongside two new networks -- PlainBNConvNet (BN, no "
          "residual) and ResNetBNConvNet (BN AND residual together) -- all "
          "at the identical lr=0.15, same 150 epochs, same batch_size=50, "
          "same dataset convention.")

    img_size, C, n_blocks = 40, 8, 3                                # same architecture scale used throughout this lesson
    X_train, y_train = make_junction_dataset_rgb(                   # generate the training set
        n_per_class=50, size=img_size, arm=10, noise=0.4, channel_noise=0.15,
        center_jitter=3, seed=42)                                   # seed=42, matching every prior day's training-set convention
    X_test, y_test = make_junction_dataset_rgb(                     # generate a SEPARATE, genuinely unseen test set
        n_per_class=20, size=img_size, arm=10, noise=0.4, channel_noise=0.15,
        center_jitter=3, seed=999)

    epochs, batch_size, lr = 150, 50, 0.15                           # the shared training budget, batch size, and the SAME aggressive lr Day 53 used

    net_plain = PlainDeepConvNet(C=C, img_size=img_size, n_blocks=n_blocks, lr=lr, seed=42)  # fresh plain network (today's control)
    hist_plain = train_with_history(net_plain, X_train, y_train, X_test, y_test,
                                     epochs=epochs, batch_size=batch_size, seed=42)

    net_bn = PlainBNConvNet(C=C, img_size=img_size, n_blocks=n_blocks, lr=lr, seed=42)  # fresh BN-only network, identical init to net_plain
    hist_bn = train_with_history(net_bn, X_train, y_train, X_test, y_test,
                                  epochs=epochs, batch_size=batch_size, seed=42)

    net_bnres = ResNetBNConvNet(C=C, img_size=img_size, n_blocks=n_blocks, lr=lr, seed=42)  # fresh BN+residual network, identical init to the other two
    hist_bnres = train_with_history(net_bnres, X_train, y_train, X_test, y_test,
                                     epochs=epochs, batch_size=batch_size, seed=42)

    tr_plain, te_plain = hist_plain["train_acc"][-1], hist_plain["test_acc"][-1]  # plain network's final accuracies
    tr_bn, te_bn = hist_bn["train_acc"][-1], hist_bn["test_acc"][-1]              # BN-only network's final accuracies
    tr_bnres, te_bnres = hist_bnres["train_acc"][-1], hist_bnres["test_acc"][-1]  # BN+residual network's final accuracies

    print(f"\n{'network':<20}{'params':>9}{'train_acc':>11}{'test_acc':>10}")  # table header
    print(f"{'PlainDeepConvNet':<20}{net_plain.param_count():>9}{tr_plain:>11.4f}{te_plain:>10.4f}")  # plain network's row
    print(f"{'PlainBNConvNet':<20}{net_bn.param_count():>9}{tr_bn:>11.4f}{te_bn:>10.4f}")             # BN-only network's row
    print(f"{'ResNetBNConvNet':<20}{net_bnres.param_count():>9}{tr_bnres:>11.4f}{te_bnres:>10.4f}")   # BN+residual network's row
    print("Actual output from this lesson's run.")                             # standing convention

    dead_plain = measure_dead_fraction_plain(net_plain)                        # diagnose PlainDeepConvNet's dead-ReLU fraction
    dead_bn = measure_dead_fraction_bn(net_bn)                                 # same diagnosis for PlainBNConvNet (checks post-BN values)
    dead_bnres = measure_dead_fraction_bn(net_bnres)                           # same diagnosis for ResNetBNConvNet
    print(f"\ndiagnosis -- fraction of each block's second conv (post-BN where applicable) stuck <= 0 after training:")
    for i in range(n_blocks):                                                  # print all three networks' dead fraction, block by block
        print(f"  block{i}   plain: {dead_plain[f'block{i}.Zc2']:.3f}    "
              f"BN-only: {dead_bn[f'block{i}.bn2']:.3f}    "
              f"BN+res: {dead_bnres[f'block{i}.bn2']:.3f}")
    print("Actual output from this lesson's run.")                             # standing convention

    best_acc = max(te_plain, te_bn, te_bnres)                                   # whichever network reached the highest test accuracy
    winner = ("PlainDeepConvNet" if best_acc == te_plain else
              "PlainBNConvNet" if best_acc == te_bn else "ResNetBNConvNet")     # name that network
    print(f"\nreal, honest finding: at the SAME lr=0.15 that made Day 53's "
          f"PlainDeepConvNet and ResNetStyleConvNet both fail completely "
          f"(train_acc=0.250 for both), today's PlainDeepConvNet reaches "
          f"train_acc={tr_plain:.3f}/test_acc={te_plain:.3f}, PlainBNConvNet "
          f"reaches train_acc={tr_bn:.3f}/test_acc={te_bn:.3f}, and "
          f"ResNetBNConvNet reaches train_acc={tr_bnres:.3f}/test_acc="
          f"{te_bnres:.3f} -- {winner} comes out ahead here.")

    plt.figure(figsize=(9, 5))                                                  # one figure comparing all three networks' trajectories
    plt.plot(hist_plain["epochs"], hist_plain["test_acc"],                      # PlainDeepConvNet's trajectory
              label=f"PlainDeepConvNet, no BN (test_acc={te_plain:.3f})", color="#d9534f")
    plt.plot(hist_bn["epochs"], hist_bn["test_acc"],                            # PlainBNConvNet's trajectory
              label=f"PlainBNConvNet, BN only (test_acc={te_bn:.3f})", color="#f0ad4e")
    plt.plot(hist_bnres["epochs"], hist_bnres["test_acc"],                      # ResNetBNConvNet's trajectory
              label=f"ResNetBNConvNet, BN+residual (test_acc={te_bnres:.3f})", color="#5cb85c")
    plt.xlabel("epoch")                                                         # x-axis label
    plt.ylabel("test accuracy")                                                 # y-axis label
    plt.ylim(0, 1.0)                                                             # fix the y-axis range for visual comparability across days
    plt.title(f"Same lr=0.15 that broke Day 53: plain vs. BN vs. BN+residual")   # descriptive title
    plt.legend(fontsize=9)                                                      # show the legend distinguishing all 3 lines
    plt.tight_layout()                                                          # avoid clipped labels/titles when saving
    plt.savefig("batchnorm_vs_plain_vs_resnet.png", dpi=110)                    # write the figure to disk
    plt.close()                                                                 # free the figure's memory now that it's saved
    print("\nsaved batchnorm_vs_plain_vs_resnet.png")                           # confirm the save to the console


# =============================================================================
# 9. WHAT THIS MAPS TO IN PYTORCH
# =============================================================================
def note_on_pytorch():                                          # pure documentation: prints a PyTorch equivalent, imports nothing
    print("\n" + "=" * 70)                                        # blank line then section separator
    print("4. WHAT THIS MAPS TO IN PYTORCH")
    print("=" * 70)                                                # closing separator
    print(                                                          # a single multi-line string showing the real ResNet basic block in PyTorch
        "class ResNetBasicBlock(nn.Module):\n"
        "    def __init__(self, channels):\n"
        "        super().__init__()\n"
        "        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)\n"
        "        self.bn1 = nn.BatchNorm2d(channels)\n"
        "        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)\n"
        "        self.bn2 = nn.BatchNorm2d(channels)\n"
        "\n"
        "    def forward(self, x):\n"
        "        out = torch.relu(self.bn1(self.conv1(x)))\n"
        "        out = self.bn2(self.conv2(out))     # no relu yet -- matches resblock_bn_forward's bn2\n"
        "        return torch.relu(out + x)           # add the shortcut, THEN relu\n"
        "\n"
        "# this IS the real ResNet (He et al., 2015) basic block, exactly as\n"
        "# published -- today's ResNetBNConvNet is the closest this series gets\n"
        "# to reproducing it. nn.BatchNorm2d also tracks running mean/variance\n"
        "# for use at inference time on a SINGLE example -- today's spatial_\n"
        "# batchnorm_forward always uses the current batch's own statistics,\n"
        "# matching Day 40's simplification (fine here since this series' own\n"
        "# accuracy() always evaluates a full array at once, never one example)."
    )
    print("\nnn.BatchNorm2d(channels) tracks a running mean/variance during "  # explains the mapping back to today's from-scratch code, and its one simplification
          "training (a detail this lesson's spatial_batchnorm_forward skips, "
          "following Day 40's convention) so it can normalize a single "
          "example at inference time without needing a batch of its own -- "
          "otherwise, conv -> BN -> ReLU, twice, then an identity add before "
          "the final ReLU, is exactly resblock_bn_forward's structure above.")


def main():                                                     # entry point: runs every demo in the order this document walks them
    demo_gradient_check()                                        # 1. verify the new spatial batch-norm + residual backward pass is correct
    demo_gradient_flow_by_depth()                                 # 2. the vanishing-gradient diagnostic, extended to plain/BN-only/BN+residual
    demo_train_and_compare()                                      # 3. the real, fully-converged trained comparison at Day 53's failing lr
    note_on_pytorch()                                             # 4. relate it all back to a production framework, and the real ResNet block


if __name__ == "__main__":                                      # only run main() when this file is executed directly, not when imported
    main()                                                        # kick off the whole lesson
