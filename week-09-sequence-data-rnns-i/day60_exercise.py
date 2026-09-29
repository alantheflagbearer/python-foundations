"""
Day 60 Exercise -- Gradient Clipping, Isolated

The main lesson's one new function, on its own: clip a set of gradients
by their GLOBAL norm, then check the three properties that make it safe
-- a no-op below the threshold, the norm lands exactly on the threshold
above it, and the direction never changes. Element-wise clipping is run
alongside for contrast. No network, no training.
"""
import numpy as np                                    # numpy: every array/matrix operation below is built on this


def clip_by_global_norm(grads, max_norm):               # scale ALL gradients by one factor if their combined length is too big
    norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))  # global norm: every gradient treated as one long vector
    if norm <= max_norm:                                   # already short enough
        return grads, norm                                   # untouched -- an exact no-op
    scale = max_norm / norm                                 # the factor that makes the norm exactly max_norm
    return {k: g * scale for k, g in grads.items()}, norm   # SAME factor for every entry -> direction preserved


def clip_by_value(grads, limit):                         # the alternative: clip every entry to [-limit, limit] independently
    return {k: np.clip(g, -limit, limit) for k, g in grads.items()}


def as_vector(grads):                                   # every gradient concatenated into one vector
    return np.concatenate([g.ravel() for g in grads.values()])


def cosine(a, b):                                       # 1.0 means pointing in exactly the same direction
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


rng = np.random.RandomState(0)                          # fixed rng
grads = {"Wxh": rng.randn(1, 16),                         # a fake gradient shaped like the RNN's parameters...
         "Whh": rng.randn(16, 16) * 20,                   # ...with an EXPLODING recurrent part, 20x larger than the rest
         "bh": rng.randn(16)}
g0 = as_vector(grads)                                     # the original, as one vector
print(f"original global norm: {np.linalg.norm(g0):.2f}")

for c in [1000.0, 5.0]:                                   # one threshold above the norm, one far below it
    clipped, _ = clip_by_global_norm(grads, c)
    g1 = as_vector(clipped)
    print(f"global-norm clip at c={c:>6}: new norm {np.linalg.norm(g1):8.4f}  cosine {cosine(g0, g1):.8f}  "
          f"untouched: {all(clipped[k] is grads[k] for k in grads)}")

g2 = as_vector(clip_by_value(grads, 0.5))                 # element-wise clipping for comparison
print(f"value clip at +-0.5:        new norm {np.linalg.norm(g2):8.4f}  cosine {cosine(g0, g2):.8f}")

def whh_dominance(g):                                   # how much bigger the recurrent gradient is than the input gradient
    return np.linalg.norm(g["Whh"]) / np.linalg.norm(g["Wxh"])


print(f"\n||dWhh|| / ||dWxh||: original {whh_dominance(grads):.1f}, "          # measured, not assumed, for all three
      f"after global-norm clip {whh_dominance(clip_by_global_norm(grads, 5.0)[0]):.1f}, "
      f"after value clip {whh_dominance(clip_by_value(grads, 0.5)):.1f}")
