
"""Day 10: Statistics Fundamentals
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
 
Covers:
  1. Manual descriptive statistics (mean, median, mode, variance, std, IQR, outliers)
  2. Distributions (skew, kurtosis, log transform)
  3. Probability basics (marginal, joint, conditional, Bayes' theorem)
  4. Correlation vs causation
  5. Hypothesis testing (t-test, chi-square)
 
Expects titanic.csv in the same folder. If not found, downloads a copy.
"""
 
import os
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
 
 
def load_titanic(path=DATA_PATH, url=DATA_URL):
    """Load titanic.csv locally, or download it if missing."""
    if not os.path.exists(path):
        print(f"{path} not found locally — downloading...")
        urllib.request.urlretrieve(url, path)
    df = pd.read_csv(path)
    return df
 
 
# ---------------------------------------------------------------------------
# 1. DESCRIPTIVE STATISTICS (manual, no pandas .describe())
# ---------------------------------------------------------------------------
 
def describe_manual(series):
    """
    Compute descriptive statistics by hand using plain Python/NumPy.
    Returns a dict comparable to series.describe() plus variance/IQR/outliers.
    """
    data = series.dropna().to_numpy()
    n = len(data)
    sorted_data = np.sort(data)
 
    mean = data.sum() / n
 
    # median
    mid = n // 2
    median = sorted_data[mid] if n % 2 == 1 else (sorted_data[mid - 1] + sorted_data[mid]) / 2
 
    # mode (most frequent value; ties -> smallest value)
    values, counts = np.unique(data, return_counts=True)
    mode = values[np.argmax(counts)]
 
    # variance / std (sample, ddof=1) and population (ddof=0)
    sq_diffs = (data - mean) ** 2
    pop_variance = sq_diffs.sum() / n
    sample_variance = sq_diffs.sum() / (n - 1)
    pop_std = pop_variance ** 0.5
    sample_std = sample_variance ** 0.5
 
    # quartiles via linear interpolation (matches numpy default / pandas)
    def percentile(p):
        idx = (n - 1) * p
        lo = int(np.floor(idx))
        hi = int(np.ceil(idx))
        if lo == hi:
            return sorted_data[lo]
        frac = idx - lo
        return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])
 
    q1 = percentile(0.25)
    q2 = percentile(0.50)
    q3 = percentile(0.75)
    iqr = q3 - q1
 
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = data[(data < lower_fence) | (data > upper_fence)]
 
    return {
        "count": n,
        "mean": mean,
        "median": median,
        "mode": mode,
        "min": sorted_data[0],
        "max": sorted_data[-1],
        "range": sorted_data[-1] - sorted_data[0],
        "pop_variance": pop_variance,
        "sample_variance": sample_variance,
        "pop_std": pop_std,
        "sample_std": sample_std,
        "q1": q1,
        "q2_median": q2,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "n_outliers": len(outliers),
        "outlier_values": outliers,
    }
 
 
def run_descriptive_stats(df):
    print("\n" + "=" * 70)
    print("1. DESCRIPTIVE STATISTICS")
    print("=" * 70)
 
    for col in ["Fare", "Age"]:
        print(f"\n--- {col} (manual) ---")
        stats_dict = describe_manual(df[col])
        for k, v in stats_dict.items():
            if k == "outlier_values":
                print(f"{k:>18}: {len(v)} values (showing first 5): {np.round(v[:5], 2)}")
            else:
                print(f"{k:>18}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"{k:>18}: {v}")
 
        print(f"\n--- {col} via pandas.describe() (sanity check) ---")
        print(df[col].describe())
 
    return {col: describe_manual(df[col]) for col in ["Fare", "Age"]}
 
 
# ---------------------------------------------------------------------------
# 2. DISTRIBUTIONS
# ---------------------------------------------------------------------------
 
def run_distributions(df):
    print("\n" + "=" * 70)
    print("2. DISTRIBUTIONS")
    print("=" * 70)
 
    fare_skew = df["Fare"].skew()
    fare_kurt = df["Fare"].kurt()
    age_skew = df["Age"].skew()
    age_kurt = df["Age"].kurt()
 
    print(f"Fare skew: {fare_skew:.3f} (kurtosis: {fare_kurt:.3f}) -> "
          f"{'right-skewed (long tail of high fares)' if fare_skew > 0.5 else 'roughly symmetric'}")
    print(f"Age  skew: {age_skew:.3f} (kurtosis: {age_kurt:.3f}) -> "
          f"{'right-skewed' if age_skew > 0.5 else 'roughly symmetric / slightly skewed'}")
 
    log_fare = np.log1p(df["Fare"])  # log1p handles Fare == 0 safely
    log_fare_skew = log_fare.skew()
    print(f"log1p(Fare) skew: {log_fare_skew:.3f} -> skew reduced from {fare_skew:.3f}")
 
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
 
    axes[0, 0].hist(df["Fare"].dropna(), bins=40, color="#378ADD", edgecolor="white")
    axes[0, 0].set_title(f"Fare (raw) — skew={fare_skew:.2f}")
    axes[0, 0].set_xlabel("Fare")
 
    axes[0, 1].hist(log_fare.dropna(), bins=40, color="#1D9E75", edgecolor="white")
    axes[0, 1].set_title(f"log1p(Fare) — skew={log_fare_skew:.2f}")
    axes[0, 1].set_xlabel("log1p(Fare)")
 
    axes[1, 0].hist(df["Age"].dropna(), bins=30, color="#D85A30", edgecolor="white")
    axes[1, 0].set_title(f"Age — skew={age_skew:.2f}")
    axes[1, 0].set_xlabel("Age")
 
    sns.kdeplot(df["Fare"].dropna(), ax=axes[1, 1], color="#378ADD", label="Fare (raw)")
    sns.kdeplot(log_fare.dropna(), ax=axes[1, 1], color="#1D9E75", label="log1p(Fare)")
    axes[1, 1].set_title("Fare density: raw vs log-transformed")
    axes[1, 1].legend()
 
    plt.tight_layout()
    plt.savefig("day10_distributions.png", dpi=150)
    plt.close()
    print("Saved plot: day10_distributions.png")
 
 
# ---------------------------------------------------------------------------
# 3. PROBABILITY BASICS
# ---------------------------------------------------------------------------
 
def run_probability(df):
    print("\n" + "=" * 70)
    print("3. PROBABILITY BASICS")
    print("=" * 70)
 
    n = len(df)
    p_survived = df["Survived"].sum() / n
 
    n_female = (df["Sex"] == "female").sum()
    p_survived_given_female = df.loc[df["Sex"] == "female", "Survived"].sum() / n_female
 
    n_pclass1 = (df["Pclass"] == 1).sum()
    p_survived_given_pclass1 = df.loc[df["Pclass"] == 1, "Survived"].sum() / n_pclass1
 
    p_female = n_female / n
 
    # Bayes' theorem: P(female | survived) = P(survived | female) * P(female) / P(survived)
    p_female_given_survived_bayes = (p_survived_given_female * p_female) / p_survived
 
    # direct calculation, to verify Bayes gives the same answer
    n_survived = df["Survived"].sum()
    p_female_given_survived_direct = df.loc[df["Survived"] == 1, "Sex"].eq("female").sum() / n_survived
 
    print(f"P(Survived)                     = {p_survived:.4f}")
    print(f"P(Survived | Sex=female)        = {p_survived_given_female:.4f}")
    print(f"P(Survived | Pclass=1)          = {p_survived_given_pclass1:.4f}")
    print(f"P(Sex=female)                   = {p_female:.4f}")
    print(f"P(Sex=female | Survived), Bayes = {p_female_given_survived_bayes:.4f}")
    print(f"P(Sex=female | Survived), direct= {p_female_given_survived_direct:.4f}")
    print(f"Match: {np.isclose(p_female_given_survived_bayes, p_female_given_survived_direct)}")
 
 
# ---------------------------------------------------------------------------
# 4. CORRELATION VS CAUSATION
# ---------------------------------------------------------------------------
 
def run_correlation(df):
    print("\n" + "=" * 70)
    print("4. CORRELATION VS CAUSATION")
    print("=" * 70)
 
    numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    corr = df[numeric_cols].corr()
    print(corr.round(2))
 
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation matrix — Titanic numeric features")
    plt.tight_layout()
    plt.savefig("day10_correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved plot: day10_correlation_heatmap.png")
 
    print(
        "\nNote: Pclass and Fare correlate strongly (higher class -> higher fare), "
        "but Pclass does not CAUSE Fare in a mechanical sense — both are driven by a "
        "third factor: how much the passenger paid at booking, which itself determined "
        "class. Similarly, Fare correlates with Survived, but paying more didn't directly "
        "save anyone; it's a proxy for cabin location and lifeboat access, which is the "
        "real causal link. Correlation here reflects shared upstream causes, not one "
        "variable driving the other."
    )
 
 
# ---------------------------------------------------------------------------
# 5. HYPOTHESIS TESTING
# ---------------------------------------------------------------------------
 
def run_hypothesis_tests(df):
    print("\n" + "=" * 70)
    print("5. HYPOTHESIS TESTING")
    print("=" * 70)
 
    print(
        "\nConcepts:\n"
        "  H0 (null): no difference / no relationship exists\n"
        "  H1 (alt) : a difference / relationship exists\n"
        "  alpha = 0.05: threshold for 'statistically significant'\n"
        "  p-value: probability of seeing data this extreme IF H0 were true\n"
        "  Type I error: rejecting H0 when it's actually true (false positive)\n"
        "  Type II error: failing to reject H0 when it's actually false (false negative)\n"
    )
 
    # --- t-test: Fare, survivors vs non-survivors ---
    fare_survived = df.loc[df["Survived"] == 1, "Fare"].dropna()
    fare_died = df.loc[df["Survived"] == 0, "Fare"].dropna()
 
    t_stat, t_pvalue = stats.ttest_ind(fare_survived, fare_died, equal_var=False)
 
    print("--- Two-sample t-test: Fare (survived vs died) ---")
    print(f"H0: mean Fare is the same for survivors and non-survivors")
    print(f"Survivors mean Fare: {fare_survived.mean():.2f} | Non-survivors mean Fare: {fare_died.mean():.2f}")
    print(f"t-statistic = {t_stat:.4f}, p-value = {t_pvalue:.6f}")
    if t_pvalue < 0.05:
        print("Conclusion: p < 0.05, reject H0 — Fare differs significantly between "
              "survivors and non-survivors (survivors paid significantly more on average).")
    else:
        print("Conclusion: p >= 0.05, fail to reject H0 — no significant difference in Fare.")
 
    # --- chi-square test: Sex vs Survived ---
    contingency = pd.crosstab(df["Sex"], df["Survived"])
    chi2, chi_pvalue, dof, expected = stats.chi2_contingency(contingency)
 
    print("\n--- Chi-square test: Sex vs Survived ---")
    print("H0: Sex and Survived are independent (no relationship)")
    print(contingency)
    print(f"chi2 = {chi2:.4f}, p-value = {chi_pvalue:.6e}, dof = {dof}")
    if chi_pvalue < 0.05:
        print("Conclusion: p < 0.05, reject H0 — Sex and survival are significantly "
              "related (being female was strongly associated with survival).")
    else:
        print("Conclusion: p >= 0.05, fail to reject H0 — no significant relationship.")
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    df = load_titanic()
    print(f"Loaded titanic.csv: {df.shape[0]} rows, {df.shape[1]} columns")
 
    run_descriptive_stats(df)
    run_distributions(df)
    run_probability(df)
    run_correlation(df)
    run_hypothesis_tests(df)
 
    print("\n" + "=" * 70)
    print("Day 10 complete. Plots saved: day10_distributions.png, day10_correlation_heatmap.png")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
 