
"""
Day 13: Full EDA Portfolio Project
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
 
Ties together everything from Days 1-12 into one shippable project:
  1. Load + clean (Day 8)
  2. Feature engineering (Day 8)
  3. Statistical analysis: t-test, chi-square, correlation (Day 10)
  4. SQL summary tables: GROUP BY, CASE WHEN, window functions (Day 11-12)
  5. A polished 6-panel dashboard image (Day 7, 9)
  6. An auto-generated README.md with the key findings
 
Run this once. It produces:
  - day13_dashboard.png
  - README.md
  - prints every finding to the terminal along the way
"""

import os
import sqlite3
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN + FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def load_and_clean(path=DATA_PATH, url=DATA_URL):
    print("=" * 70)
    print("1. LOAD, CLEAN, FEATURE ENGINEERING")
    print("=" * 70)

    if not os.path.exists(path):
        print(f"{path} not found locally - downloading...")
        urllib.request.urlretrieve(url, path)

    df = pd.read_csv(path)
    print(f"loadd: {df.shape[0]} rows, {df.shape[1]} columns")

    # --- cleaning ---
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df = df.drop(columns=["Cabin"])
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    print("Cleaned: Age -> median, Cabin dropped, Embarked -> mode")
    
    # --- feature engineering ---
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["AgeGroup"] = pd.cut(
         df["Age"],
         bins=[0, 12, 18, 35, 60, 100],
         labels=["Child", "Teen", "Adult", "MiddleAge", "Senior"],)
    df["Title"] = df["Name"].str.extract(r",\s*([^.]+)\.")
    print("Engineered: FamilySize, IsAlone, AgeGroup, Title")
    print(f"Final shape: {df.shape}\n")
    return df

# ---------------------------------------------------------------------------
# 2. STATISTICAL ANALYSIS
# ---------------------------------------------------------------------------

def run_statistical_summary(df):
    print("=" * 70)
    print("2. STATISTICAL SUMMARY")
    print("=" * 70)
 
    findings = []

    # t-test: Fare, survived vs died
    fare_survived = df.loc[df["Survived"] == 1, "Fare"]
    fare_died = df.loc[df["Survived"] == 0, "Fare"]
    t_stat, t_p = stats.ttest_ind(fare_survived, fare_died, equal_var=False)
    finding = (
        f"survivors paid ${fare_survived.mean():.2f} avg fare vs "
        f"${fare_died.mean():.2f} for non-survivors (t={t_stat:.2f}, p={t_p:.2e}) "
        f"- {'statistically significant' if t_p < 0.05 else 'not significant'}.")
    print(finding)
    findings.append(finding)

    # chi-square: Sex vs Survived  
    ct_sex = pd.crosstab(df["Sex"], df["Survived"])
    chi2, p_sex, _, _ = stats.chi2_contingency(ct_sex)
    female_rate = df.loc[df["Sex"] == "female", "Survived"].mean()
    male_rate = df.loc[df["Sex"] == "male", "Survived"].mean()
    finding = (
        f"Female survival rate {female_rate:.1%} vs male {male_rate:.1%} "
        f"(chi2={chi2:.1f}, p={p_sex:.2e}) — "
        f"{'statistically significant' if p_sex < 0.05 else 'not significant'}."
    )
    print(finding)
    findings.append(finding)

    # chi-square: Pclass vs Survived
    ct_class = pd.crosstab(df["Pclass"], df["Survived"])
    chi2_c, p_class, _, _ = stats.chi2_contingency(ct_class)
    rates_by_class = df.groupby("Pclass")["Survived"].mean()
    finding = (
        f"Survival rate by class: 1st {rates_by_class[1]:.1%}, "
        f"2nd {rates_by_class[2]:.1%}, 3rd {rates_by_class[3]:.1%} "
        f"(chi2={chi2_c:.1f}, p={p_class:.2e}) — "
        f"{'statistically significant' if p_class < 0.05 else 'not significant'}."
    )
    print(finding)
    findings.append(finding)
    
     # correlation
    numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
    corr = df[numeric_cols].corr()
    strongest = (
        corr["Survived"].drop("Survived").abs().sort_values(ascending=False).index[0]
    )
    finding = (
        f"Strongest correlate of Survived is {strongest} "
        f"(r={corr['Survived'][strongest]:.2f}) — correlation, not causation: "
        f"{strongest} is a proxy for cabin location and lifeboat access, not a direct cause."
    )
    print(finding)
    findings.append(finding)
 
    print()
    return {"findings": findings, "correlation": corr}
 
 
# ---------------------------------------------------------------------------
# 3. SQL SUMMARY TABLES
# ---------------------------------------------------------------------------
 
def run_sql_summary(df):
    print("=" * 70)
    print("3. SQL SUMMARY TABLES")
    print("=" * 70)
 
    conn = sqlite3.connect(":memory:")
    df.to_sql("titanic", conn, index=False)
 
    survival_by_group = pd.read_sql("""
        SELECT Pclass, Sex,
               COUNT(*) AS n_passengers,
               ROUND(AVG(Survived), 3) AS survival_rate
        FROM titanic
        GROUP BY Pclass, Sex
        ORDER BY Pclass, Sex;
    """, conn)
    print("--- Survival rate by Pclass x Sex ---")
    print(survival_by_group.to_string(index=False))
 
    fare_tiers = pd.read_sql("""
        SELECT
          CASE
            WHEN Fare < 15 THEN 'Budget'
            WHEN Fare < 60 THEN 'Standard'
            ELSE 'Premium'
          END AS fare_tier,
          COUNT(*) AS n_passengers,
          ROUND(AVG(Survived), 3) AS survival_rate
        FROM titanic
        GROUP BY fare_tier
        ORDER BY survival_rate DESC;
    """, conn)
    print("\n--- Survival rate by fare tier (CASE WHEN) ---")
    print(fare_tiers.to_string(index=False))
 
    top_fares_per_class = pd.read_sql("""
        SELECT Name, Pclass, Fare, fare_rank FROM (
            SELECT Name, Pclass, Fare,
                RANK() OVER (PARTITION BY Pclass ORDER BY Fare DESC) AS fare_rank
            FROM titanic
        )
        WHERE fare_rank <= 3
        ORDER BY Pclass, fare_rank;
    """, conn)
    print("\n--- Top 3 fares per class (window function) ---")
    print(top_fares_per_class.to_string(index=False))
    print()
 
    conn.close()
    return {
        "survival_by_group": survival_by_group,
        "fare_tiers": fare_tiers,
        "top_fares_per_class": top_fares_per_class,
    }
 
 
# ---------------------------------------------------------------------------
# 4. DASHBOARD
# ---------------------------------------------------------------------------
 
def build_dashboard(df, corr, out_path="day13_dashboard.png"):
    print("=" * 70)
    print("4. DASHBOARD")
    print("=" * 70)
 
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
 
    # Panel 1: overall survival
    survived_counts = df["Survived"].value_counts().sort_index()
    axes[0, 0].bar(["Died", "Survived"], survived_counts.values,
                   color=["#D85A30", "#1D9E75"])
    axes[0, 0].set_title("Overall survival")
    for i, v in enumerate(survived_counts.values):
        axes[0, 0].text(i, v + 5, str(v), ha="center")
 
    # Panel 2: survival by sex
    sex_survival = df.groupby("Sex")["Survived"].mean()
    axes[0, 1].bar(sex_survival.index, sex_survival.values, color="#378ADD")
    axes[0, 1].set_title("Survival rate by sex")
    axes[0, 1].set_ylim(0, 1)
 
    # Panel 3: survival by class
    class_survival = df.groupby("Pclass")["Survived"].mean()
    axes[0, 2].bar(class_survival.index.astype(str), class_survival.values, color="#7F77DD")
    axes[0, 2].set_title("Survival rate by class")
    axes[0, 2].set_ylim(0, 1)
 
    # Panel 4: age distribution
    axes[1, 0].hist(df["Age"], bins=30, color="#D85A30", edgecolor="white")
    axes[1, 0].set_title("Age distribution")
 
    # Panel 5: fare distribution (log-scaled)
    axes[1, 1].hist(np.log1p(df["Fare"]), bins=30, color="#1D9E75", edgecolor="white")
    axes[1, 1].set_title("Fare distribution (log1p)")
 
    # Panel 6: correlation heatmap
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=axes[1, 2], square=True, cbar=False, annot_kws={"size": 8})
    axes[1, 2].set_title("Correlation matrix")
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved dashboard: {out_path}\n")
 
 
# ---------------------------------------------------------------------------
# 5. README
# ---------------------------------------------------------------------------
 
def write_readme(df, stats_results, sql_results, out_path="README.md"):
    print("=" * 70)
    print("5. README")
    print("=" * 70)
 
    findings_md = "\n".join(f"- {f}" for f in stats_results["findings"])
 
    content = f"""# Titanic EDA — full portfolio project
 
Day 13 of a 52-week data science to ML/AI roadmap. This project answers one
question: **what determined who survived the Titanic disaster, and can it be
shown consistently across pandas, statistical tests, and raw SQL?**
 
## Dataset
 
{df.shape[0]} passengers, {df.shape[1]} columns after cleaning and feature
engineering (FamilySize, IsAlone, AgeGroup, Title added; Age imputed with
median, Cabin dropped, Embarked imputed with mode).
 
## Key findings
 
{findings_md}
 
## Dashboard
 
![Titanic EDA dashboard](day13_dashboard.png)
 
Six panels: overall survival, survival by sex, survival by class, age
distribution, log-scaled fare distribution, and the full correlation matrix.
 
## SQL cross-check
 
The same survival patterns are reproduced with raw SQL (GROUP BY, CASE WHEN
fare-tier bucketing, and a RANK() window function for top fares per class) —
see `day13_eda_portfolio.py` for the queries and their output.
 
## What's next
 
With more time: engineer a Title-based social class proxy, test interaction
effects between Sex and Pclass more formally, and move this analysis into a
baseline classification model.
"""
 
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
 
    print(f"Wrote {out_path}\n")
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    df = load_and_clean()
    stats_results = run_statistical_summary(df)
    sql_results = run_sql_summary(df)
    build_dashboard(df, stats_results["correlation"])
    write_readme(df, stats_results, sql_results)
 
    print("=" * 70)
    print("Day 13 complete. Deliverables: day13_dashboard.png, README.md")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
 