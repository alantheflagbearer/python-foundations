# Week 1 Day 7 — Md Mutasim Billah
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("Matplotlib version:", plt.matplotlib.__version__)

# Step 2 — Line chart (trends over time)
# ── LINE CHART ───────────────────────────────

weeks = [1, 2, 3, 4, 5, 6, 7]
scores = [55, 62, 68, 71, 75, 79, 85]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(weeks, scores, color="steelblue", linewidth=2,
        marker="o", markersize=6, label="weekly score")
ax.set_title("learning progress - week by week", fontsize=14)
ax.set_xlabel("week")
ax.set_ylabel("score")
ax.set_ylim(0, 100)
ax.axhline(y=70, color="red", linestyle="--", label="pass line")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("line_chart.png", dpi=150)
plt.show()
print("saved line_chart.png")


# ── BAR CHART ────────────────────────────────
students = ["Mutasim", "Ali", "Sara", "John", "Priya"]
grades = [88, 45, 95, 38, 77]
colors = ["steelblue" if g >= 50 else "salmon" for g in grades]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(students, grades, color=colors, edgecolor="white")

for bar, grade in zip(bars, grades):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            str(grade), ha="center", fontsize=10, fontweight="bold")

ax.set_title("student grades", fontsize=14)
ax.set_xlabel("student")
ax.set_ylabel("grade")
ax.set_ylim(0, 110)
ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="pass line")
ax.legend()
plt.tight_layout()
plt.savefig("bar_chart.png", dpi=150)
plt.show()
print("saved bar_chart.png")


#Step 4 — Histogram (distribution)
# ── HISTOGRAM ────────────────────────────────

np.random.seed(42)
exam_scores = np.random.normal(loc=70, scale=15, size=200)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(exam_scores, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(x=exam_scores.mean(), color="red", linestyle="--",
           label=f"Mean: {exam_scores.mean():.1f}")
ax.axvline(x=np.median(exam_scores), color="orange",
           label=f"median: {np.median(exam_scores):.1f}")
ax.set_title("distribution of exam scores (n=200)", fontsize=14)
ax.set_xlabel("score")
ax.set_ylabel("number of students")
ax.legend()
plt.tight_layout()
plt.savefig("histogram.png", dpi=150)
plt.show()
print("saved histogram.png")


#Step 5 — Scatter plot (relationship between two variables)
# ── SCATTER PLOT ─────────────────────────────

np.random.seed(42)
study_hours = np.random.uniform(1, 10, 30)
exam_grade = study_hours * 7 + np.random.normal(0, 8, 30)
exam_grade = np.clip(exam_grade, 0, 100)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(study_hours, exam_grade, color="steelblue", alpha=0.7, s=60)

z = np.polyfit(study_hours, exam_grade, 1)
p = np.poly1d(z)
x_line = np.linspace(study_hours.min(), study_hours.max(), 100)
ax.plot(x_line, p(x_line), color="red", linewidth=1.5,
        linestyle="--", label="Trend line")

ax.set_title("study hours vs exam grade", fontsize=14)
ax.set_xlabel("study hours")
ax.set_ylabel("exam grade")
ax.legend()
plt.tight_layout()
plt.savefig("scatter.png", dpi=150)
plt.show()
print("saved scatter.png")


# Step 6 — Subplots dashboard (2×2 layout)
# ── SUBPLOTS 2×2 ─────────────────────────────

students = ["Mutasim", "Ali", "Sara", "John", "Priya"]
grades   = [88, 45, 95, 38, 77]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("student performance dashboard", fontsize=16, fontweight="bold")

axes[0,0].bar(students, grades, color="steelblue")
axes[0,0].set_title("Grades by student")
axes[0,0].axhline(50, color="red", linestyle="--", alpha=0.5)

axes[0,1].pie([3, 2], labels=["pass","Fail"],
              colors=["steelblue","salmon"], autopct="%1.0f%%")
axes[0,1].set_title("pass / fail split")

axes[1,0].barh(students, grades, color="mediumseagreen")
axes[1,0].set_title("grades - horizontal")

sorted_grades = sorted(grades, reverse=True)
axes[1,1].plot(range(1,6), sorted_grades, marker="o",
               color="darkorange", linewidth=2)
axes[1,1].set_title("grade ranking")

plt.tight_layout()
plt.savefig("dashboard.png", dpi=150)
plt.show()
print("saved dashboard.png")


# Step 7 — Pandas built-in plots (fastest in real work)
# # ── PANDAS PLOTS ─────────────────────────────

df = pd.DataFrame({
    "name"   : ["Mutasim","Ali","Sara","John","Priya"],
    "grade"  : [88, 45, 95, 38, 77],
    "study_h": [6, 2, 8, 1, 5]
})

ax = df.plot(kind="bar", x="name", y="grade",
             title="Grades (pandas)", color="steelblue", legend=False)
ax.set_xlabel("students")
plt.tight_layout()
plt.savefig("pandas_bar.png", dpi=150)
plt.show()

ax2 = df.plot(kind="scatter", x="study_h", y="grade",
              title="study hours vs grade", color="darkorange", s=80)
plt.tight_layout()
plt.savefig("pandas_scatter.png", dpi=150)
plt.show()

print("all 7 charts saved")