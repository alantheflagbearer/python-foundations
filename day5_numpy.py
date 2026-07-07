# day5_numpy.py
# Week 1 Day 5 — Md Mutasim Billah
# Topic: NumPy arrays, vectorized math, statistics

import numpy as np   # np is the universal alias — always use this

print("NumPy version:", np.__version__)
print("NumPy is working!")


#Creating NumPy arrays
#Arrays are the core data structure — everything else builds on them
import numpy as np
scores = np.array([72, 88, 65, 95, 81, 77])

#zeros and ones
zeros = np.zeros(5)
ones = np.ones(5)

#range of numbers
r = np.arange(0, 10, 2)

#evenly spaced between two values
s = np.linspace(0, 1, 5)

print("scores:", scores)
print("zeros: ", zeros)
print("arange:", r)
print("linspace:", s)
print("shape:", scores.shape)
print("dtype:", scores.dtype)

print("*" * 40)
# ── VECTORIZED OPERATIONS ────────────────────
scores = np.array([77, 88, 65, 95, 81, 77])

# math on every element — no loop needed
print(scores + 10)
print(scores * 2)
print(scores / 100)
print(scores ** 2)

#operations between two arrays
weights = np.array([0.3, 0.3, 0.4, 0.4, 0.3, 0.3])
weighted = scores * weights
print("weighted:", weighted)

#comparison — returns array of True/False
passed = scores >= 75
print("passed 75:", passed) #[False True False True True True]

#count how many passed
print("how many passed:", passed.sum())
print("*" * 40)

# Indexing and boolean masking
# Select exactly the elements you want — no loops, no if statements

# ── INDEXING AND BOOLEAN MASKING ─────────────
scores = np.array([72, 88, 65, 95, 81, 77])
# regular indexing — same as lists
print(scores[0])
print(scores[-1])
print(scores[1:4])

# boolean masking — select by condition
high_scores = scores[scores > 80]
print("avobe 80:", high_scores)

low_scores = scores[scores < 75]
print("below 75:", low_scores)


# combine conditions with & (and) | (or)
mid = scores[(scores >= 70) & (scores <= 85)]
print("between 70-85:", mid)

# where — returns indices of matching elements
idx = np.where(scores > 80)
print("indices above 80:", idx[0])
print("*" * 40)

# Statistics with NumPy
# Everything pfds07 did with loops — in one line each
# ── STATISTICS ───────────────────────────────
scores = np.array([72, 88, 65, 95, 81, 77])

print(f"mean : {np.mean(scores):.2f}")
print(f"median: {np.median(scores):.2f}")
print(f"std : {np.std(scores):.2f}")
print(f"min :{np.min(scores)}")
print(f"max :{np.max(scores)}")
print(f"sum :{np.sum(scores)}")
print(f"range :{np.max(scores) - np.min(scores)}")


# percentiles — tells you the distribution
print(f"25th % : {np.percentile(scores, 25):.2f}")
print(f"75th % : {np.percentile(scores, 75):.2f}")

# these also work as array methods
print(f"mean (method): {scores.mean():.2f}")
print(f"std (method): {scores.std():.2f}")
print("*" * 40)

# 2D arrays — rows and columns
#The foundation of DataFrames, matrices, and image data
# ── 2D ARRAYS ────────────────────────────────
# 3 students, 3 subjects each
grades = np.array([
    [80, 58, 77], # mutasim
    [95, 90, 88], # kh
    [40, 45, 38], # alan
])

print("shape:", grades.shape) # (3, 3) — 3 rows, 3 cols
print("ndim:", grades.ndim) # 2

# access rows and columns
print("rows 0 (Mutasim):", grades[0])       # [80 58 77]
print("col 0 (subject 1):", grades[:, 0])   # [80 95 40]
print("one cell [1,2]:", grades[1, 2])      # 88 (Sara, Subject 3)


# statistics along axes
# axis=1 means across columns (per student average)
student_avgs = grades.mean(axis=1)
print("student averages:", student_avgs)  # [71.7 91.  41. ]

#axis=0 means across rows (per subject average)
subject_avgs = grades.mean(axis=0)
print("subjects averages:", subject_avgs)  # [71.7 64.3 67.7]
print("*" * 40)

# Rewrite pfds07 with NumPy
# See exactly how much less code you need now
# ── REWRITE pfds07 WITH NUMPY ────────────────
print("\== pfds07 rewritten with with numpy ===")

scores = np.array([72, 88, 65, 95, 81, 77])

for score in scores:
    print(f"score: {score}")

print(f"score: {score}")
print(f"total: {scores.sum()}")
print(f"average: {scores.mean():.2f}")
print(f"highest:{scores.max()}")
print(f"lowest:{scores.min()}")
print(f"std dev: {scores.std():.2f}")

# grade curve — add 5 points to everyone
curved = scores + 5
print(f"\nafter +5 curve: {curved}")
print(f"new average : {curved.mean():.2f}")

# who passed after curve? (>=75)
passed = curved[curved >= 75]
print(f"scors passing after curve: {passed}")
print(f"number who passed: {(curved >= 75).sum()}")