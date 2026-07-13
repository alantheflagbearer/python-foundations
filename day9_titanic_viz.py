# Week 2 Day 9 — Md Mutasim Billah
# Topic: Titanic EDA — visualizing survival patterns

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load and clean (same as Day 8)
df = pd.read_csv("titanic.csv")
df["Age"] = df["Age"].fillna(df["Age"].median())
df = df.drop(columns=["Cabin"])
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)


# color palette — use consistently throughout
BLUE = "steelblue"
ORANGE = "darkorange"
RED = "firebrick"
GREEN = "seagreen"

print("data loaded and clean. shape:", df.shape)
print("ready to chart!")

'''Chart 1 — overall survival overview
The first chart any analyst makes — what happened overall?'''

# ── CHART 1: SURVIVAL OVERVIEW ───────────────
survived = df["Survived"].sum()
not_survived = len(df) - survived


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("titanic - overall Survival", fontsize=14, fontweight="bold")

#bar chart--------
ax1.bar(["Died", "Survived"],
        [not_survived, survived],
        color=[RED, GREEN], edgecolor="white")

ax1.set_title("count")
ax1.set_ylabel("number of passenger")
for i, v in enumerate([not_survived, survived]):
    ax1.text(i, v + 5, str(v), ha="center", fontweight="bold")


#pie chart-----------
ax2.pie([not_survived, survived],
        labels=["Died (61.6%)", "Survived (38.4%)"],
        colors=[RED, GREEN],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2})
ax2.set_title("Proportion")

plt.tight_layout()
plt.savefig("titanic_survival_overview.png", dpi=150)
plt.show()
print("Saved titanic_survival_overview.png")

'''Chart 2 — survival by sex
Visualize the most powerful predictor of survival'''

# ── CHART 2: SURVIVAL BY SEX ─────────────────
survival_sex = df.groupby("Sex")["Survived"].mean() * 100

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(survival_sex.index,
              survival_sex.values,
     	      color=[ORANGE, BLUE],
 	      edgecolor="white", width=0.5)

#add percentage labels on bars
for bar, val in zip(bars, survival_sex.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center", fontsize=13, fontweight="bold")

ax.set_title("Survival Rate by sex", fontsize=14, fontweight="bold")
ax.set_xlabel("Sex")
ax.set_ylabel("Survival rate (%)")
ax.set_ylim(0, 100)

ax.axhline(y=38.4, color="gray", linestyle="--",
          alpha=0.6, label="Overall rate (38.4%)")

ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("titanic_survival_sex.png", dpi=150)
plt.show()
print("Saved titanic_survival_sex.png")

# ── CHART 2: SURVIVAL BY SEX ─────────────────
survival_sex = df.groupby("Sex")["Survived"].mean() * 100

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(survival_sex.index, survival_sex.values,
              color=[ORANGE, BLUE],
              edgecolor="white", width=0.5)

for bar, val in zip(bars, survival_sex.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center", fontsize=13, fontweight="bold")

ax.set_title("Survival Rate by Sex", fontsize=14, fontweight="bold")
ax.set_xlabel("Sex")
ax.set_ylabel("Survival rate (%)")
ax.set_ylim(0, 100)
ax.axhline(y=38.4, color="gray", linestyle="--",
           alpha=0.6, label="Overall rate (38.4%)")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("titanic_survival_sex.png", dpi=150)
plt.show()
print("Saved titanic_survival_sex.png")


# ── CHART 3: SURVIVAL BY CLASS ───────────────
survival_class = df.groupby("Pclass")["Survived"].mean() * 100
count_class = df.groupby("Pclass")["Survived"].count()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Passenger class analysis", fontsize=14, fontweight="bold")

colors = [GREEN, BLUE, RED]
bars = ax1.bar(["1st class", "2nd class", "3rd class"],
               survival_class.values,
               color=colors, edgecolor="white", width=0.5)

for bar, val in zip(bars, survival_class.values):
    ax1.text(bar.get_x() + bar.get_width()/2,
         bar.get_height() + 1,
         f"{val:.1f}%", ha="center", fontweight="bold")

ax1.set_title("survival rate by class")
ax1.set_ylabel("survival rate (%)")
ax1.set_ylim(0, 100)
ax1.grid(axis="y", alpha=0.3)

ax2.bar(["1st Class", "2nd Class", "3rd Class"],
        count_class.values,
        color=colors, edgecolor="white", width=0.5)

for i, v in enumerate(count_class.values):
    ax2.text(i, v + 5, str(v), ha="center", fontweight="bold")
ax2.set_title("Passenger Count by Class")
ax2.set_ylabel("Number of passengers")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("titanic_class.png", dpi=150)
plt.show()
print("Saved titanic_class.png")

# ── CHART 4: AGE DISTRIBUTION ────────────────
survived_ages     = df[df["Survived"] == 1]["Age"]
not_survived_ages = df[df["Survived"] == 0]["Age"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(not_survived_ages, bins=25, alpha=0.6,
        color=RED, label="Died", edgecolor="white")
ax.hist(survived_ages, bins=25, alpha=0.6,
        color=GREEN, label="Survived", edgecolor="white")
ax.axvline(x=survived_ages.mean(), color=GREEN, linestyle="--",
           linewidth=1.5, label=f"Survived mean: {survived_ages.mean():.1f}")
ax.axvline(x=not_survived_ages.mean(), color=RED, linestyle="--",
           linewidth=1.5, label=f"Died mean: {not_survived_ages.mean():.1f}")
ax.set_title("Age Distribution — Survivors vs Non-Survivors",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Number of passengers")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("titanic_age.png", dpi=150)
plt.show()
print("Saved titanic_age.png")

# ── CHART 5: FARE BY CLASS ───────────────────
fare_by_class = [
    df[df["Pclass"] == 1]["Fare"],
    df[df["Pclass"] == 2]["Fare"],
    df[df["Pclass"] == 3]["Fare"],
]

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(fare_by_class,
                tick_labels=["1st Class", "2nd Class", "3rd Class"],
                patch_artist=True,
                medianprops={"color": "white", "linewidth": 2})
for patch, color in zip(bp["boxes"], [GREEN, BLUE, RED]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title("Fare Distribution by Passenger Class",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Fare ($)")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("titanic_fare.png", dpi=150)
plt.show()
print("Saved titanic_fare.png")

# ── FINAL DASHBOARD ──────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Titanic Survival Analysis Dashboard",
             fontsize=16, fontweight="bold", y=1.01)

survived = df["Survived"].sum()
axes[0,0].pie([len(df)-survived, survived],
              labels=["Died", "Survived"],
              colors=[RED, GREEN], autopct="%1.1f%%",
              startangle=90,
              wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[0,0].set_title("Overall Survival")

surv_sex = df.groupby("Sex")["Survived"].mean() * 100
axes[0,1].bar(surv_sex.index, surv_sex.values,
              color=[ORANGE, BLUE], edgecolor="white", width=0.5)
axes[0,1].set_title("Survival Rate by Sex")
axes[0,1].set_ylabel("Rate (%)")
axes[0,1].set_ylim(0, 100)
for i, v in enumerate(surv_sex.values):
    axes[0,1].text(i, v+2, f"{v:.1f}%", ha="center", fontweight="bold")

surv_cls = df.groupby("Pclass")["Survived"].mean() * 100
axes[0,2].bar(["1st", "2nd", "3rd"], surv_cls.values,
              color=[GREEN, BLUE, RED], edgecolor="white", width=0.5)
axes[0,2].set_title("Survival Rate by Class")
axes[0,2].set_ylabel("Rate (%)")
axes[0,2].set_ylim(0, 100)
for i, v in enumerate(surv_cls.values):
    axes[0,2].text(i, v+2, f"{v:.1f}%", ha="center", fontweight="bold")

axes[1,0].hist(df[df["Survived"]==0]["Age"], bins=20,
               alpha=0.6, color=RED, label="Died")
axes[1,0].hist(df[df["Survived"]==1]["Age"], bins=20,
               alpha=0.6, color=GREEN, label="Survived")
axes[1,0].set_title("Age Distribution")
axes[1,0].set_xlabel("Age")
axes[1,0].legend()

axes[1,1].boxplot([df[df["Pclass"]==1]["Fare"],
                   df[df["Pclass"]==2]["Fare"],
                   df[df["Pclass"]==3]["Fare"]],
                  tick_labels=["1st", "2nd", "3rd"],
                  patch_artist=True)
axes[1,1].set_title("Fare by Class")
axes[1,1].set_ylabel("Fare ($)")

surv_fam = df.groupby("IsAlone")["Survived"].mean() * 100
axes[1,2].bar(["With Family", "Alone"],
              [surv_fam[0], surv_fam[1]],
              color=[GREEN, RED], edgecolor="white", width=0.5)
axes[1,2].set_title("Survival: Family vs Alone")
axes[1,2].set_ylabel("Rate (%)")
axes[1,2].set_ylim(0, 100)
for i, v in enumerate([surv_fam[0], surv_fam[1]]):
    axes[1,2].text(i, v+2, f"{v:.1f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("titanic_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved titanic_dashboard.png")
print("All charts saved!")