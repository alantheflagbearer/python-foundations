"""
Day 30: Partial Dependence and ICE Curves
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 5, Day 3
 
Covers:
  1. How PDP/ICE differ from Day 29's SHAP dependence plots (simulation vs
     observed attribution)
  2. Partial Dependence Plot (PDP) — the model's average predicted response
     as one feature is swept across a grid, holding everything else fixed
  3. Individual Conditional Expectation (ICE) — one curve per row, showing
     the heterogeneity a PDP's average line hides
  4. Running PDP/ICE directly on the fitted pipeline, on raw feature names
  5. Two-feature (2D) partial dependence — interaction between two features
  6. Comparing what PDP/SHAP dependence agree and disagree on
 
Requires: pip install scikit-learn matplotlib xgboost
Expects titanic.csv in the same folder. If not found, downloads a copy.
"""
 
import os
import urllib.request
 
import matplotlib
matplotlib.use("Agg")  # headless-safe backend — saves files without opening a window
import matplotlib.pyplot as plt
 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
 
from xgboost import XGBClassifier
 
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
 
DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
 
 
# ---------------------------------------------------------------------------
# 0. LOAD + SPLIT (same as Day 22-29 — raw, no manual cleaning)
# ---------------------------------------------------------------------------
 
def load_and_split(path=DATA_PATH, url=DATA_URL):
    print("=" * 70)
    print("0. LOAD + SPLIT")
    print("=" * 70)
 
    if not os.path.exists(path):
        print(f"{path} not found locally — downloading...")
        urllib.request.urlretrieve(url, path)
 
    df = pd.read_csv(path)
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[feature_cols]
    y = df["Survived"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    print(f"Loaded {df.shape[0]} passengers. Train: {X_train.shape[0]}, Test: {X_test.shape[0]}\n")
    return X_train, X_test, y_train, y_test
 
 
# ---------------------------------------------------------------------------
# 1. SAME PREPROCESSOR + XGBOOST MODEL FROM DAY 21-29
# ---------------------------------------------------------------------------
 
def add_family_size(X):
    X = X.copy()
    X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
    return X
 
 
def build_preprocessor():
    numeric_cols = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
    categorical_cols = ["Pclass", "Sex", "Embarked"]
 
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])
 
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])
    return preprocessor
 
 
def build_and_fit_model(X_train, y_train):
    print("=" * 70)
    print("1. FITTING THE DAY 25 XGBOOST PIPELINE")
    print("=" * 70)
 
    engineer = FunctionTransformer(add_family_size)
    preprocessor = build_preprocessor()
 
    pipeline = Pipeline(steps=[
        ("engineer", engineer),
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)
 
    print("Same fitted pipeline as Day 28/29. PDP/ICE work directly on the whole pipeline,")
    print("taking RAW column names (Pclass, Age, Fare...) — not the transformed/encoded")
    print("columns Day 28/29's SHAP work operated on.\n")
    return pipeline


# ---------------------------------------------------------------------------
# 1b. BUILD A PDP-SAFE REFERENCE TABLE (fills raw missing values)
# ---------------------------------------------------------------------------
 
def build_pdp_reference(X):
    """
    PartialDependenceDisplay/partial_dependence build their grid by taking
    percentiles of the RAW column passed in — BEFORE it ever reaches the
    pipeline's own imputer. If that raw column still has missing values,
    np.percentile propagates NaN straight into the grid, silently producing
    an all-NaN, invisible plot even though no error is raised.
 
    This fills missing values the same way the pipeline's own imputer would
    (median for numeric, most-frequent for categorical) purely so PDP has a
    finite range to build its grid from — the pipeline's internal imputer
    still runs on every prediction and is unaffected, since a column with no
    remaining NaNs simply passes through it unchanged.
    """
    X_ref = X.copy()
    for col in ["Age", "Fare"]:
        if X_ref[col].isna().any():
            X_ref[col] = X_ref[col].fillna(X_ref[col].median())
    for col in ["Embarked"]:
        if X_ref[col].isna().any():
            X_ref[col] = X_ref[col].fillna(X_ref[col].mode()[0])
    return X_ref

 
# ---------------------------------------------------------------------------
# 2. WHY PDP/ICE ARE DIFFERENT FROM SHAP DEPENDENCE PLOTS
# ---------------------------------------------------------------------------
 
def explain_pdp_vs_shap():
    print("=" * 70)
    print("2. PDP/ICE VS DAY 29'S SHAP DEPENDENCE PLOT")
    print("=" * 70)
    print("SHAP dependence plot (Day 29): for each ACTUAL row, shows that row's ACTUAL")
    print("SHAP contribution — real, observed attributions from real data.")
    print("PDP/ICE (today): a simulation. Take every row, artificially set one feature to")
    print("a fixed value across ALL of them, re-predict, and see how the average (or each")
    print("individual) prediction changes. It answers 'what WOULD happen if,' not 'what DID")
    print("happen' — a genuinely different question with a genuinely different computation.\n")
 
 
# ---------------------------------------------------------------------------
# 3. PARTIAL DEPENDENCE PLOT (PDP) — the average simulated effect
# ---------------------------------------------------------------------------

def plot_pdp(pipeline, X_train, feature, out_path=None):
    if out_path is None:
        out_path = f"pdp_{feature}.png"

    print("=" * 70)
    print(f"3. PARTIAL DEPENDENCE PLOT — {feature}")
    print("=" * 70)
 
    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        pipeline, X_train, features=[feature], kind="average", ax=ax
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
 
    print(f"Saved: {out_path}")
    print(f"For a grid of {feature} values, every row in X_train has its {feature} column")
    print("temporarily overwritten with that grid value, the model re-predicts on all rows,")
    print("and the AVERAGE predicted probability at that grid value is plotted — one point")
    print("per grid value, connected into a curve.\n")
 
 
# ---------------------------------------------------------------------------
# 4. INDIVIDUAL CONDITIONAL EXPECTATION (ICE) — per-row curves
# ---------------------------------------------------------------------------
 
def plot_ice(pipeline, X_train, feature, out_path=None):
    if out_path is None:
        out_path = f"ice_{feature}.png"
 
    print("=" * 70)
    print(f"4. ICE PLOT — {feature}")
    print("=" * 70)
 
    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        pipeline, X_train, features=[feature], kind="both",
        ax=ax, ice_lines_kw={"alpha": 0.15, "linewidth": 0.5},
        pd_line_kw={"linewidth": 2.5, "color": "black"},
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
 
    print(f"Saved: {out_path}")
    print(f"kind=\"both\" draws one thin line PER ROW (its own {feature}-swept prediction")
    print("curve) plus the thick black PDP average line on top — the individual lines are")
    print("the ICE curves; the PDP line is simply their average at every grid point.\n")
 
 
# ---------------------------------------------------------------------------
# 5. TWO-FEATURE (2D) PARTIAL DEPENDENCE — interaction effects
# ---------------------------------------------------------------------------
 
def plot_2d_pdp(pipeline, X_train, feature_pair, out_path="pdp_2d_age_fare.png"):
    print("=" * 70)
    print(f"5. 2D PARTIAL DEPENDENCE — {feature_pair}")
    print("=" * 70)
 
    fig, ax = plt.subplots(figsize=(6, 5))
    PartialDependenceDisplay.from_estimator(
        pipeline, X_train, features=[feature_pair], kind="average", ax=ax
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
 
    print(f"Saved: {out_path}")
    print(f"Sweeps BOTH {feature_pair[0]} and {feature_pair[1]} across a 2D grid together,")
    print("showing whether their combined effect is just the sum of their individual")
    print("effects, or genuinely different depending on the combination — an interaction")
    print("SHAP's single-feature dependence plot (Day 29) cannot show directly.\n")
 
 
# ---------------------------------------------------------------------------
# 6. COMPARING PDP'S NUMBERS AGAINST THE RAW SIMULATION
# ---------------------------------------------------------------------------
 
def show_pdp_values(pipeline, X_train, feature):
    print("=" * 70)
    print(f"6. RAW PDP VALUES — {feature}")
    print("=" * 70)   

    result = partial_dependence(pipeline, X_train, features=[feature], kind="average")
    grid = result["grid_values"][0]
    avg_pred = result["average"][0]

    table = pd.DataFrame({feature: grid, "avg_predicted_probability": avg_pred})
    print(table.round(3).to_string(index=False))

    print(f"\n{feature} rises from {grid.min():.1f} to {grid.max():.1f} across the grid;")
    print(f"average predicted survival probability moves from {avg_pred[0]:.3f} to "
          f"{avg_pred[-1]:.3f} as a direct result of sweeping only {feature}.\n")
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    X_train, X_test, y_train, y_test = load_and_split()
    pipeline = build_and_fit_model(X_train, y_train)
 
    explain_pdp_vs_shap()
 
    X_train_pdp = build_pdp_reference(X_train)
    print(f"Age had {X_train['Age'].isna().sum()} missing values in the raw training data —")
    print("filled for PDP's grid construction only, so the grid isn't silently all-NaN.\n")
 
    plot_pdp(pipeline, X_train_pdp, "Age")
    plot_pdp(pipeline, X_train_pdp, "Fare")
    plot_ice(pipeline, X_train_pdp, "Age")
    plot_2d_pdp(pipeline, X_train_pdp, ("Age", "Fare"))
    show_pdp_values(pipeline, X_train_pdp, "Fare")
 
    print("=" * 70)
    print("7. SUMMARY")
    print("=" * 70)
    print("Five outputs saved in this folder:")
    print("  pdp_Age.png / pdp_Fare.png   — average simulated effect per feature")
    print("  ice_Age.png                  — per-row curves + PDP average overlaid")
    print("  pdp_2d_age_fare.png          — Age x Fare interaction surface")
    print("  (Fare PDP values also printed as a table above)")
    print("Day 29's SHAP dependence plot showed observed attribution from real rows;")
    print("today's PDP/ICE show simulated 'what if' behavior — both are legitimate and")
    print("complementary views of the same fitted model, not competing answers.\n")
 
    print("=" * 70)
    print("Day 30 complete. Model behavior explored by simulation (PDP/ICE), not just by")
    print("attributing outcomes already observed in the data (SHAP).")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
 


            