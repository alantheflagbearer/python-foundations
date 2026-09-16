"""
Day 56 Exercise -- The Position-Generalization Failure, Isolated

The main lesson's core finding, minimal: train a plain dense network on
an order-detection task with both marker positions confined to the FIRST
half of the sequence, then see how it does on the SECOND half. No CNN,
no gradient check, no plots -- just the one result this week's Day 62
integration will need as its "baseline to beat".
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def relu(z):                                            # ReLU activation
    return np.maximum(0, z)                              # elementwise max(0, z)


def drelu(z):                                           # derivative of ReLU
    return (z > 0).astype(z.dtype)                       # 1 where z>0, else 0


def softmax(z):                                         # turns logits into a probability distribution per row
    z_shifted = z - z.max(axis=1, keepdims=True)          # subtract each row's max for numerical stability
    exp_z = np.exp(z_shifted)                             # exponentiate
    return exp_z / exp_z.sum(axis=1, keepdims=True)       # normalize each row to sum to 1


def one_hot(y, n_classes):                              # integer labels -> one-hot rows
    out = np.zeros((y.shape[0], n_classes))               # start all-zero
    out[np.arange(y.shape[0]), y] = 1                     # set the true class's column to 1
    return out                                            # the one-hot matrix


def make_order_dataset(n_per_class, T, pos_lo, pos_hi, seed):  # same task as the main lesson: does the high spike come before the low spike?
    rng = np.random.RandomState(seed)                     # this function's own private RNG
    X, y = [], []                                          # accumulators
    for label in [0, 1]:                                   # both classes
        for _ in range(n_per_class):                        # n_per_class examples each
            seq = rng.randn(T) * 0.3                         # start from small noise
            p1, p2 = rng.choice(np.arange(pos_lo, pos_hi + 1), size=2, replace=False)  # two distinct positions in [pos_lo, pos_hi]
            p_high, p_low = (p1, p2) if p1 < p2 else (p2, p1)  # sort so p_high is earlier
            if label == 0:                                   # for label 0, swap the order
                p_high, p_low = p_low, p_high                  # low now comes first
            seq[p_high] += 3.0                                # place the high spike
            seq[p_low] -= 3.0                                 # place the low spike
            X.append(seq)                                     # store the sequence
            y.append(label)                                   # store its label
    X, y = np.array(X), np.array(y)                        # stack into arrays
    idx = rng.permutation(len(y))                           # shuffle
    return X[idx], y[idx]                                   # return shuffled pair


class PlainFCNet:                                        # the same minimal dense network as the main lesson
    def __init__(self, T, hidden=16, n_classes=2, lr=0.2, seed=42):  # constructor
        rng = np.random.RandomState(seed)                   # this network's own RNG
        self.W1 = rng.randn(T, hidden) * np.sqrt(2.0 / T)     # FC1 weights, He-scaled
        self.b1 = np.zeros(hidden)                            # FC1 biases
        self.W2 = rng.randn(hidden, n_classes) * np.sqrt(2.0 / hidden)  # FC2 weights
        self.b2 = np.zeros(n_classes)                         # FC2 biases
        self.lr = lr                                          # learning rate

    def forward(self, X):                                  # forward pass
        self.X = X                                            # cache input
        self.Z1 = X @ self.W1 + self.b1                       # FC1 pre-activation
        self.A1 = relu(self.Z1)                               # FC1 activation
        self.Z2 = self.A1 @ self.W2 + self.b2                 # FC2 pre-activation (logits)
        self.probs = softmax(self.Z2)                         # class probabilities
        return self.probs                                     # hand back

    def backward(self, y_onehot):                          # backward pass + SGD update
        m = y_onehot.shape[0]                                 # batch size
        dZ2 = self.probs - y_onehot                           # combined softmax+CE gradient
        dW2 = self.A1.T @ dZ2 / m                              # FC2 weight grad
        db2 = dZ2.sum(axis=0) / m                              # FC2 bias grad
        dA1 = dZ2 @ self.W2.T                                 # grad into FC1 activation
        dZ1 = dA1 * drelu(self.Z1)                             # FC1 pre-activation grad
        dW1 = self.X.T @ dZ1 / m                               # FC1 weight grad
        db1 = dZ1.sum(axis=0) / m                              # FC1 bias grad
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1      # SGD update FC1
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2      # SGD update FC2

    def accuracy(self, X, y):                              # classification accuracy
        return (self.forward(X).argmax(axis=1) == y).mean()  # fraction correct


T = 30                                                    # sequence length
X_train, y_train = make_order_dataset(100, T, 0, 14, seed=42)   # train: markers in first half only
X_test_seen, y_test_seen = make_order_dataset(40, T, 0, 14, seed=123)   # held-out, SAME half
X_test_shift, y_test_shift = make_order_dataset(40, T, 15, 29, seed=999)  # held-out, SHIFTED half

net = PlainFCNet(T, lr=0.2, seed=7)                       # fresh network
rng = np.random.RandomState(7)                            # training loop's own RNG
for epoch in range(200):                                   # 200 epochs, matching the main lesson's budget
    order = rng.permutation(len(y_train))                    # shuffle each epoch
    for start in range(0, len(order), 40):                    # mini-batches of 40
        idx = order[start:start + 40]                            # this batch's indices
        net.forward(X_train[idx])                                # forward
        net.backward(one_hot(y_train[idx], 2))                    # backward + update

print(f"train_acc (seen positions):        {net.accuracy(X_train, y_train):.4f}")           # training accuracy
print(f"test_acc (seen positions, held out): {net.accuracy(X_test_seen, y_test_seen):.4f}")  # generalization within seen range
print(f"test_acc (SHIFTED positions):        {net.accuracy(X_test_shift, y_test_shift):.4f}")  # the actual failure being isolated
