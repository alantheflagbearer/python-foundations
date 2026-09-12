# Day 36 — Line-by-Line Syntax Explanation

Pure syntax focus: what each line *does mechanically*, not the neural-network theory (that's in the PDF). Line numbers match `day36_neural_network_fundamentals.py`.

---

## Imports and setup (lines 32–38)

```python
import numpy as np
```
Imports the NumPy library, aliased as `np`. Every array operation in this file goes through this alias.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```
`matplotlib.use("Agg")` must run **before** `import matplotlib.pyplot`. It tells matplotlib to render to a file instead of a screen window (no GUI backend). This is why running the script never pops up a chart window — it just writes PNGs to disk. `plt` is the standard alias for the plotting submodule.

```python
from sklearn.datasets import make_moons
```
Imports one specific function (`make_moons`) directly, rather than importing all of `sklearn.datasets`. `make_moons` generates a synthetic 2D dataset shaped like two interleaving crescents.

```python
np.random.seed(42)
```
Fixes NumPy's global random number generator to a starting point. Any code that calls `np.random.*` after this line (weight initialization, `make_moons`'s noise) produces the exact same "random" numbers every run — this is why your output matches the PDF exactly.

---

## Section 1: The Perceptron class (lines 45–73)

```python
class Perceptron:
    """A single linear unit..."""
```
Defines a new class. The triple-quoted string right after the `class` line is a **docstring** — Python stores it as `Perceptron.__doc__`, viewable via `help(Perceptron)`.

```python
    def __init__(self, n_inputs, lr=0.1):
        self.w = np.zeros(n_inputs)
        self.b = 0.0
        self.lr = lr
```
`__init__` is the constructor — runs automatically when you write `Perceptron(2)`. `self` is the instance being built (every method needs it as the first parameter). `lr=0.1` is a **default argument** — callers can omit it.
- `np.zeros(n_inputs)` creates a 1D array of that length filled with `0.0` — e.g. `np.zeros(2)` → `[0.0, 0.0]`. This becomes the weight vector.
- `self.w = ...` attaches the array to the instance as an attribute, so it persists between method calls.
- `self.b = 0.0` — a plain float, the bias term.

```python
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return 1 if z >= 0 else 0
```
`np.dot(x, self.w)` computes the dot product — elementwise multiply then sum: `x[0]*w[0] + x[1]*w[1]`. Adding `self.b` gives the raw weighted sum `z`.
`return 1 if z >= 0 else 0` is a **conditional expression** (ternary) — shorthand for an if/else that evaluates to a value in one line. This is the perceptron's step activation function.

```python
    def train(self, X, y, epochs=20):
        history = []
        for _ in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
```
`history = []` — empty list, will collect one number per epoch.
`for _ in range(epochs)` — loops `epochs` times. The underscore `_` is a Python convention meaning "I don't need this loop variable's value."
`for xi, yi in zip(X, y)` — `zip` pairs up elements from two sequences positionally: first element of `X` with first element of `y`, second with second, etc. This lets you loop over both arrays in lockstep. `xi` is one row of `X` (a length-2 array, e.g. `[0, 1]`), `yi` is the matching label.

```python
                pred = self.forward(xi)
                error = yi - pred
                if error != 0:
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    errors += 1
```
`self.w += self.lr * error * xi` — this is the **perceptron learning rule**. `+=` is augmented assignment (`self.w = self.w + ...`). Because `xi` is a NumPy array, `self.lr * error * xi` produces an array (scalar × scalar × array → array), and `self.w += (that array)` updates all weights element-by-element in one line — no explicit loop over the 2 weights needed, NumPy broadcasts the scalar multiplication automatically.
`errors += 1` — plain integer counter increment.

```python
            history.append(errors)
            if errors == 0:
                break
        return history
```
`history.append(errors)` — adds this epoch's error count to the end of the list.
`if errors == 0: break` — `break` exits the nearest enclosing `for` loop immediately. Once a full pass makes zero mistakes, training is done — no point running the remaining epochs.
`return history` — sends the list of per-epoch error counts back to the caller (used to print "converged in N epochs").

---

## `demo_perceptron_and_gate()` (lines 76–91)

```python
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
```
`np.array([[...], [...], ...])` — a list of lists becomes a 2D NumPy array (4 rows, 2 columns here). Each inner list is one training example's two inputs. `y_and` is a 1D array of the matching AND-gate outputs.

```python
    p = Perceptron(n_inputs=2)
    history = p.train(X_and, y_and)
```
`Perceptron(n_inputs=2)` — calls `__init__` with a **keyword argument**, explicit about which parameter `2` fills (clearer than positional `Perceptron(2)`). `p.train(...)` calls the instance method, mutating `p`'s internal `w`/`b` and returning the error history.

```python
    print(f"Converged in {len(history)} epochs (errors per epoch: {history})")
```
An **f-string** (`f"..."`) — anything inside `{}` is evaluated as a Python expression and inserted into the string. `len(history)` counts how many epochs actually ran (stopped early via `break`).

```python
    for xi, yi in zip(X_and, y_and):
        print(f"  {xi} -> predicted {p.forward(xi)}, actual {yi}")
```
Same `zip` pattern as before, this time just for printing each input/prediction/actual triple after training is finished.

---

## Section 2: XOR failure (lines 98–114)

Structurally identical to Section 1's demo — same `Perceptron` class, different data (`y_xor`). Two syntax points worth calling out:

```python
    correct = sum(p.forward(xi) == yi for xi, yi in zip(X_xor, y_xor))
```
`(p.forward(xi) == yi for xi, yi in zip(...))` is a **generator expression** — like a list comprehension but with `()` instead of `[]`, so it produces values one at a time instead of building a full list in memory. `p.forward(xi) == yi` evaluates to `True`/`False`; in Python, `True` behaves as `1` and `False` as `0` when summed, so `sum(...)` counts how many predictions were correct.

```python
    print(f"Ran {len(history)} epochs, never converged (errors per epoch: {history[:10]}...)")
```
`history[:10]` — **slicing**. Takes the first 10 elements of the list (or fewer, if the list is shorter). Prevents printing all 50 epochs' worth of error counts.

---

## Section 3: Activation functions (lines 121–159)

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
A plain **module-level function** (no class, no `self`) — these are called directly as `sigmoid(some_array)`. `np.exp(-z)` applies `e^(-z)` **elementwise** if `z` is an array — no loop needed, NumPy vectorizes it automatically. The whole expression `1 / (1 + np.exp(-z))` also broadcasts elementwise.

```python
def relu(z):
    return np.maximum(0, z)
```
`np.maximum(0, z)` compares `0` against every element of `z` and keeps the larger of the two, elementwise — this is ReLU (negative values become 0, positive values pass through unchanged). Different from Python's built-in `max()`, which only compares single values, not arrays.

```python
def softmax(z):
    ez = np.exp(z - np.max(z))  # subtract max for numerical stability
    return ez / ez.sum()
```
`np.max(z)` finds the single largest value in the array. `z - np.max(z)` subtracts that scalar from every element (broadcasting again) — this shifts all values so the largest becomes 0, preventing `np.exp()` from overflowing on large inputs, without changing the final softmax result mathematically. `ez.sum()` adds up all elements; dividing normalizes them so they sum to 1 (a probability distribution).

```python
    for name, fn in [("sigmoid", sigmoid), ("tanh", tanh), ("relu", relu)]:
        vals = [round(float(fn(np.array([zv]))[0]), 4) for zv in [-2, 0, 2]]
        print(f"  {name:8s}: {vals}")
```
`[("sigmoid", sigmoid), ...]` — a list of tuples pairing a display name (string) with the actual function object. In Python, functions are first-class values — `sigmoid` here (no parentheses) refers to the function itself, not a call to it.
`for name, fn in [...]` — unpacks each tuple into two loop variables per iteration.
`[round(...) for zv in [-2, 0, 2]]` — a **list comprehension**: builds a new list by evaluating the expression on the left once per item in `[-2, 0, 2]`.
`np.array([zv])[0]` — wraps the single number `zv` into a 1-element array (so it matches the array-shaped inputs `fn` expects), calls `fn` on it, then `[0]` pulls the single resulting number back out.
`round(float(...), 4)` — `float()` converts a NumPy scalar to a plain Python float; `round(..., 4)` rounds to 4 decimal places for clean printing.
`f"  {name:8s}: {vals}"` — `{name:8s}` is an f-string **format spec**: pad the string to 8 characters wide (left-aligned) so the columns line up.

```python
    z_vals = np.linspace(-5, 5, 200)
```
`np.linspace(start, stop, num)` generates `num` evenly spaced numbers between `start` and `stop` inclusive — here, 200 points from -5 to 5, used as the x-axis for the activation function plots.

```python
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
```
`plt.subplots(1, 3, ...)` creates one figure containing a 1-row-by-3-column grid of subplots, returning `fig` (the whole figure) and `axes` (an array of 3 individual subplot objects). `figsize=(12, 3.2)` sets the overall image size in inches.

```python
    for ax, (name, fn) in zip(axes, [("Sigmoid", sigmoid), ("Tanh", tanh), ("ReLU", relu)]):
        ax.plot(z_vals, fn(z_vals))
        ax.set_title(name)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
```
`zip(axes, [...])` pairs each of the 3 subplot axes with a (name, function) tuple. `(name, fn)` inside the `for` unpacks the tuple in the same step as unpacking `ax`.
`ax.plot(z_vals, fn(z_vals))` draws the curve on that specific subplot — `fn(z_vals)` calls the activation function on the whole 200-element array at once (vectorized), producing the y-values.
`ax.axhline(0, ...)` / `ax.axvline(0, ...)` draw a thin horizontal/vertical reference line through 0 on each subplot.

```python
    plt.tight_layout()
    plt.savefig("activation_functions.png", dpi=110)
    plt.close()
```
`plt.tight_layout()` auto-adjusts spacing so subplot titles/labels don't overlap. `plt.savefig(path, dpi=110)` writes the current figure to disk as a PNG at 110 dots-per-inch. `plt.close()` frees the figure from memory — good practice once you're done with it, especially in a script generating multiple plots.

---

## Section 4: Hand-crafted XOR network (lines 166–196)

```python
    W1 = np.array([[20, -20], [20, -20]])
    b1 = np.array([-10, 30])
    W2 = np.array([[20], [20]])
    b2 = np.array([-30])
```
Hard-coded weight matrices and bias vectors — not learned, just typed in directly. `W1` is 2×2 (2 inputs → 2 hidden units), `W2` is 2×1 (2 hidden units → 1 output).

```python
    def forward(x):
        h = sigmoid(x @ W1 + b1)
        out = sigmoid(h @ W2 + b2)
        return out[0]
```
This `forward` is a **local function**, defined inside `demo_handcrafted_xor_network()` — it only exists within that function's scope and can see `W1`, `b1`, `W2`, `b2` from the enclosing scope (a closure) without them being passed in as arguments.
`x @ W1` — the `@` operator is **matrix multiplication** (distinct from `*`, which would multiply elementwise). `x` is shape `(2,)`, `W1` is shape `(2,2)`, so `x @ W1` produces shape `(2,)` — the weighted sum into each hidden unit, all at once via one matrix multiply instead of a loop.
`+ b1` broadcasts the bias vector onto the result. `sigmoid(...)` then squashes it elementwise.
`out[0]` — `out` comes out as a 1-element array (`b2` has shape `(1,)`, so the whole computation stays array-shaped); `[0]` extracts the plain scalar for printing.

```python
    for xi, yi in zip(X_xor, y_xor):
        pred = forward(xi)
        pred_class = 1 if pred >= 0.5 else 0
        correct += (pred_class == yi)
```
`pred_class = 1 if pred >= 0.5 else 0` — thresholds the sigmoid's continuous 0–1 output into a hard class label at the standard 0.5 cutoff.
`correct += (pred_class == yi)` — again relies on `True`/`False` acting as `1`/`0` under addition.

---

## Section 5: TinyNet — manual backprop (lines 203–254)

```python
class TinyNet:
    def __init__(self, n_in, n_hidden, lr=0.5):
        self.W1 = np.random.randn(n_in, n_hidden) * 0.5
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, 1) * 0.5
        self.b2 = np.zeros(1)
        self.lr = lr
```
`np.random.randn(n_in, n_hidden)` draws random numbers from a standard normal distribution (mean 0, std 1) into a `(n_in, n_hidden)`-shaped array — random weight initialization. `* 0.5` scales them down so initial weights are small. `np.zeros(n_hidden)` initializes biases at exactly 0, a common convention (unlike weights, biases don't need randomness to break symmetry).

```python
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
```
Same `@` matrix-multiply pattern as Section 4, but now `X` is the **entire dataset at once** — shape `(300, 2)` for the moons data — so `X @ self.W1` computes the hidden layer's weighted sums for all 300 examples simultaneously, no Python loop over examples at all. Storing results as `self.z1`, `self.a1`, etc. (not just local variables) matters: `backward()` needs these cached intermediate values to compute gradients, since they can't be recomputed cheaply without redoing the forward pass.

```python
    def backward(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)
        dz2 = self.a2 - y
        dW2 = self.a1.T @ dz2 / m
        db2 = dz2.mean(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.a1 * (1 - self.a1)
        dW1 = X.T @ dz1 / m
        db1 = dz1.mean(axis=0)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
```
`X.shape[0]` — `.shape` returns a tuple of an array's dimensions, e.g. `(300, 2)`; `[0]` grabs the first number, 300 — the number of training examples, used to average gradients.
`y.reshape(-1, 1)` — reshapes a 1D array of shape `(300,)` into a 2D column of shape `(300, 1)`. The `-1` means "figure out this dimension automatically" from the total element count. This matters because `self.a2` (the prediction) has shape `(300, 1)`, and subtracting a `(300,)` array from a `(300, 1)` array without reshaping would silently broadcast wrong (this is the exact bug documented in the comment at line 242–249 — it happened in `train()`, not here, because `train()` originally forgot this reshape before computing the loss).
`self.a1.T` — `.T` transposes a matrix (swaps rows and columns). Needed here so the matrix dimensions line up for `@` to produce a gradient matrix the same shape as `W2`.
`dz2.mean(axis=0)` — averages across `axis=0` (down the rows, i.e., across all 300 examples), collapsing the per-example gradient into one gradient per bias term.
`self.W1 -= self.lr * dW1` — gradient descent update: `-=` because we move *against* the gradient (downhill) to reduce loss, scaled by the learning rate `self.lr`.

```python
    def train(self, X, y, epochs=3000):
        losses = []
        eps = 1e-9
        for _ in range(epochs):
            pred = self.forward(X)
            y_col = y.reshape(-1, 1)
            loss = -np.mean(y_col * np.log(pred + eps) + (1 - y_col) * np.log(1 - pred + eps))
            losses.append(loss)
            self.backward(X, y)
        return losses
```
`eps = 1e-9` — a tiny constant (`1e-9` = 0.000000001, scientific notation) added inside the `log()` calls purely to prevent `log(0)`, which is `-infinity` and would crash or corrupt the loss if a prediction ever hit exactly 0 or 1.
`y_col = y.reshape(-1, 1)` — the exact fix described in the earlier comment, reshaping before the loss computation.
The loss line is the binary cross-entropy formula written directly in NumPy: `np.mean(...)` averages over all 300 examples, `-` negates it (cross-entropy is conventionally reported as a positive number, but the raw log-likelihood is negative).
`losses.append(loss)` then `self.backward(X, y)` — record this epoch's loss *before* updating weights, so `losses[0]` reflects the network's performance before any training happened.

---

## `demo_moons_comparison()` (lines 257–305)

```python
    X_moons, y_moons = make_moons(n_samples=300, noise=0.2, random_state=42)
```
`make_moons(...)` returns two things at once — unpacked directly into `X_moons` (the 300×2 coordinates) and `y_moons` (the 300 class labels). `noise=0.2` adds random jitter so the two crescents aren't perfectly clean. `random_state=42` seeds *this specific function's* randomness independently of the earlier global `np.random.seed(42)` — scikit-learn functions that accept `random_state` don't rely on NumPy's global seed.

```python
    net_pred = (net.forward(X_moons) >= 0.5).astype(int).flatten()
    net_acc = (net_pred == y_moons).mean()
```
`net.forward(X_moons) >= 0.5` — comparing an array against a scalar produces a **boolean array** (elementwise `True`/`False`), one entry per prediction, shape `(300, 1)`.
`.astype(int)` converts `True`/`False` to `1`/`0`.
`.flatten()` collapses the `(300, 1)` shape down to plain `(300,)`, so it can be compared directly against `y_moons` (also shape `(300,)`).
`(net_pred == y_moons).mean()` — elementwise equality gives another boolean array; `.mean()` on booleans treats `True` as 1.0 and `False` as 0.0, so the mean is literally the fraction of correct predictions — i.e., accuracy — in one line.

```python
    xx, yy = np.meshgrid(
        np.linspace(X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5, 200),
        np.linspace(X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5, 200),
    )
```
`X_moons[:, 0]` — NumPy 2D indexing: `:` means "all rows," `0` means "column 0." So this pulls out just the x-coordinates of every point (column 1 similarly gives y-coordinates via `X_moons[:, 1]`).
`.min() - 0.5` / `.max() + 0.5` — finds the data's range and pads it by 0.5 on each side, so the plotted background covers a bit beyond the actual data points.
`np.meshgrid(a, b)` takes two 1D arrays and produces two 2D grids (`xx`, `yy`) representing every combination of x and y — the standard way to build a background grid for a decision-boundary plot.

```python
    grid = np.c_[xx.ravel(), yy.ravel()]
```
`.ravel()` flattens a 2D array back into 1D. `np.c_[a, b]` stacks two 1D arrays as columns into a 2D array — so `grid` ends up as a `(40000, 2)` array of every (x, y) point on the grid, in the same `(n_points, 2)` shape the network's `forward()` expects.

```python
    net_zz = net.forward(grid).reshape(xx.shape)
```
Runs the trained network on all 40,000 grid points in one call, then `.reshape(xx.shape)` folds the flat result back into the 2D grid shape so it can be plotted as a filled contour.

```python
    perc_zz = np.array([perc.forward(pt) for pt in grid]).reshape(xx.shape)
```
The perceptron's `forward()` only accepts one row at a time (no vectorized batch support in this class), so this uses a list comprehension to loop over every point in `grid` individually, wraps the resulting list back into an array, then reshapes it the same way.

```python
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = [
        (axes[0], perc_zz, "Single perceptron (linear)", perc_acc),
        (axes[1], net_zz, "2-layer network (non-linear)", net_acc),
    ]
    for ax, zz, title, acc in panels:
        ax.contourf(xx, yy, zz, levels=[-1, 0.5, 2], colors=["#F5C4B3", "#9FE1CB"], alpha=0.6)
        ax.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap="coolwarm",
                   s=12, edgecolors="k", linewidths=0.3)
        ax.set_title(f"{title}\naccuracy={acc:.3f}")
```
`panels` — a list of 4-element tuples bundling everything needed to draw each subplot, so the loop below can handle both panels identically instead of duplicating the drawing code twice.
`ax.contourf(xx, yy, zz, levels=[-1, 0.5, 2], colors=[...])` — draws filled regions: wherever `zz` falls between -1 and 0.5 it's colored the first color, between 0.5 and 2 the second — this is what paints the pink/green decision-boundary background.
`ax.scatter(..., c=y_moons, cmap="coolwarm", s=12, edgecolors="k", linewidths=0.3)` — plots each data point; `c=y_moons` colors points by their class label using the `"coolwarm"` color map, `s=12` sets marker size, `edgecolors`/`linewidths` add a thin black outline per dot for visibility against the background.
`f"{title}\naccuracy={acc:.3f}"` — `\n` inside the f-string is a newline character, so the title wraps onto two lines. `{acc:.3f}` formats the float to exactly 3 decimal places.

---

## `main()` and the entry point (lines 329–345)

```python
def main():
    demo_perceptron_and_gate()
    demo_perceptron_xor_failure()
    demo_activation_functions()
    demo_handcrafted_xor_network()
    demo_moons_comparison()
    note_on_pytorch()
```
Just calls every demo function in order — this is the single place that defines "what running this script does."

```python
if __name__ == "__main__":
    main()
```
Standard Python idiom. `__name__` is a special variable Python sets automatically: it equals `"__main__"` only when the file is run directly (`python day36_neural_network_fundamentals.py`), but equals the module's name if the file is instead *imported* by another script (like Day 34 importing `add_family_size` from Day 33). This guard means `main()` only fires on direct execution — importing this file elsewhere wouldn't re-run all the demos.

---

## Quick syntax glossary (things that showed up repeatedly)

`@` — matrix multiplication between two arrays (not elementwise; use `*` for elementwise).
`.T` — transpose (swap rows/columns) of a 2D array.
`array[:, i]` — select column `i`, all rows.
`.reshape(-1, 1)` — reshape into a column vector; `-1` auto-infers that dimension.
`.mean(axis=0)` — average down the rows (per-column average); omit `axis` to average everything into one number.
`f"{x:.3f}"` — format a float to 3 decimal places inside an f-string; `f"{x:8s}"` pads a string to width 8.
`zip(a, b)` — iterate over two sequences in parallel, pairing elements by position.
`[expr for item in iterable]` — list comprehension, builds a new list.
`(expr for item in iterable)` — generator expression, same idea but lazy (used inside `sum(...)`).
`True == 1`, `False == 0` — booleans are integers under the hood in Python, so `.mean()` on a boolean array gives a fraction, and `sum()` over comparisons counts `True`s.
