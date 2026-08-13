"""
Day 29: Visual SHAP — Beeswarm, Bar, Dependence, and Waterfall Plots
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 5, Day 2
 
Covers:
  1. Why Day 28's printed SHAP tables aren't what stakeholders actually want to see
  2. shap.Explanation objects — the modern container the plotting API expects
  3. Global bar plot — mean |SHAP value| per feature, visually
  4. Beeswarm plot — every row's SHAP value per feature, colored by feature value,
     showing both magnitude AND direction across the whole dataset at once
  5. Dependence/scatter plot — how one feature's SHAP value changes as that
     feature's own value changes, revealing non-linear relationships
  6. Waterfall plot — the single-prediction breakdown from Day 28, drawn instead
     of printed
  7. Saving every plot to a PNG file for a report or dashboard
 
Requires: pip install shap xgboost matplotlib
Expects titanic.csv in the same folder. If not found, downloads a copy.
"""

"""
Day 29: Visual SHAP — Beeswarm, Bar, Dependence, and Waterfall Plots
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 5, Day 2
 
Covers:
  1. Why Day 28's printed SHAP tables aren't what stakeholders actually want to see
  2. shap.Explanation objects — the modern container the plotting API expects
  3. Global bar plot — mean |SHAP value| per feature, visually
  4. Beeswarm plot — every row's SHAP value per feature, colored by feature value,
     showing both magnitude AND direction across the whole dataset at once
  5. Dependence/scatter plot — how one feature's SHAP value changes as that
     feature's own value changes, revealing non-linear relationships
  6. Waterfall plot — the single-prediction breakdown from Day 28, drawn instead
     of printed
  7. Saving every plot to a PNG file for a report or dashboard
 
Requires: pip install shap xgboost matplotlib
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
 
from xgboost import XGBClassifier
import shap
 
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
 
DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
 
 
# ---------------------------------------------------------------------------
# COMPATIBILITY PATCH: same fix from Day 28 — xgboost >= 2.0 serializes
# base_score as "[value]" (bracketed), which crashes shap's model loader.
# Patches shap's model loader to strip the brackets before parsing.
# ---------------------------------------------------------------------------
import shap.explainers._tree as _shap_tree_module
 
_original_decode_ubjson = _shap_tree_module.decode_ubjson_buffer
 
 
def _strip_base_score_brackets(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "base_score" and isinstance(value, str) and value.strip().startswith("["):
                obj[key] = value.strip().strip("[]")
            else:
                _strip_base_score_brackets(value)
    elif isinstance(obj, list):
        for item in obj:
            _strip_base_score_brackets(item)
 
 
def _patched_decode_ubjson(fd):
    jmodel = _original_decode_ubjson(fd)
    _strip_base_score_brackets(jmodel)
    return jmodel
 
 
_shap_tree_module.decode_ubjson_buffer = _patched_decode_ubjson
 
 
# ---------------------------------------------------------------------------
# 0. LOAD + SPLIT (same as Day 22-28 — raw, no manual cleaning)
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
# 1. SAME PREPROCESSOR + XGBOOST MODEL FROM DAY 21-28
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
 
    print("Same fitted pipeline as Day 28 — today's work is entirely about how the")
    print("SHAP values already being computed get turned into pictures.\n")
    return pipeline

 
# ---------------------------------------------------------------------------
# 2. BUILDING A SHAP EXPLANATION OBJECT (NOT JUST A RAW ARRAY)
# ---------------------------------------------------------------------------
 
def build_explanation(pipeline, X):
    print("=" * 70)
    print("2. BUILDING A SHAP EXPLANATION OBJECT")
    print("=" * 70)
 
    engineer = pipeline.named_steps["engineer"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
 
    X_transformed = preprocessor.transform(engineer.transform(X))
    feature_names = preprocessor.get_feature_names_out()
 
    explainer = shap.TreeExplainer(classifier)
    explanation = explainer(X_transformed)
    explanation.feature_names = list(feature_names)
 
    print("Day 28 used explainer.shap_values(X) — a plain numpy array of numbers.")
    print("Today uses explainer(X) instead, which returns a shap.Explanation object:")
    print("it bundles the SHAP values, the base value, AND the original feature values")
    print("together, which is exactly what every plotting function below needs.\n")
 
    return explanation, X_transformed, feature_names
 
 
# ---------------------------------------------------------------------------
# 3. GLOBAL BAR PLOT — mean |SHAP value| per feature, visually
# ---------------------------------------------------------------------------
 
def plot_global_bar(explanation, out_path="shap_bar_global.png"):
    print("=" * 70)
    print("3. GLOBAL BAR PLOT")
    print("=" * 70)
 
    plt.figure()
    shap.plots.bar(explanation, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
 
    print(f"Saved: {out_path}")
    print("The visual version of Day 28's mean |SHAP value| table — same ranking,")
    print("now a horizontal bar chart instead of a printed pandas Series.\n")
 
 
# ---------------------------------------------------------------------------
# 4. BEESWARM PLOT — every row's SHAP value, colored by feature value
# ---------------------------------------------------------------------------
 
def plot_beeswarm(explanation, out_path="shap_beeswarm.png"):
    print("=" * 70)
    print("4. BEESWARM PLOT")
    print("=" * 70)
 
    plt.figure()
    shap.plots.beeswarm(explanation, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
 
    print(f"Saved: {out_path}")
    print("Every training row becomes one dot per feature — dot position shows the")
    print("SHAP value (pushed prediction up or down), dot color shows whether that")
    print("row's actual feature value was high or low. This is the single plot that")
    print("replaces both the Day 28 gain-importance table AND the mean |SHAP| table,")
    print("while also showing direction — something neither table could show.\n")
 
 
# ---------------------------------------------------------------------------
# 5. DEPENDENCE / SCATTER PLOT — one feature's SHAP value vs its own value
# ---------------------------------------------------------------------------
 
def plot_dependence(explanation, feature_name, out_path=None):
    if out_path is None:
        out_path = f"shap_dependence_{feature_name.replace('__', '_')}.png"
 
    print("=" * 70)
    print(f"5. DEPENDENCE PLOT — {feature_name}")
    print("=" * 70)
 
    plt.figure()
    shap.plots.scatter(explanation[:, feature_name], show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
 
    print(f"Saved: {out_path}")
    print(f"Plots {feature_name}'s actual value (x-axis) against its SHAP contribution")
    print("(y-axis) for every row — reveals whether the relationship is a straight")
    print("line, a threshold effect, or something non-monotonic that a single")
    print("importance number could never show.\n")
 
 
# ---------------------------------------------------------------------------
# 6. WATERFALL PLOT — the Day 28 single-prediction breakdown, drawn
# ---------------------------------------------------------------------------
 
def plot_waterfall(explanation, row_index=0, out_path="shap_waterfall_row0.png"):
    print("=" * 70)
    print(f"6. WATERFALL PLOT (test row {row_index})")
    print("=" * 70)
 
    plt.figure()
    shap.plots.waterfall(explanation[row_index], show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
 
    print(f"Saved: {out_path}")
    print("This is Day 28's manually-printed base value + contributions table, as a")
    print("chart: it starts at the base value, adds/subtracts each feature's push,")
    print("and lands exactly on the final predicted value — the same additive")
    print("guarantee from Day 28, now visually traceable top to bottom.\n")
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    X_train, X_test, y_train, y_test = load_and_split()
    pipeline = build_and_fit_model(X_train, y_train)
 
    explanation, X_test_transformed, feature_names = build_explanation(pipeline, X_test)
 
    plot_global_bar(explanation)
    plot_beeswarm(explanation)
    plot_dependence(explanation, "num__Fare")
    plot_dependence(explanation, "cat__Sex_male")
    plot_waterfall(explanation, row_index=0)
 
    print("=" * 70)
    print("7. SUMMARY")
    print("=" * 70)
    print("Five PNG files saved in this folder:")
    print("  shap_bar_global.png        — global feature ranking")
    print("  shap_beeswarm.png          — global ranking + direction, every row")
    print("  shap_dependence_num_Fare.png     — Fare's effect across its own range")
    print("  shap_dependence_cat_Sex_male.png — Sex's effect across its own range")
    print("  shap_waterfall_row0.png    — one passenger's full prediction breakdown")
    print("Every number Day 28 printed to the terminal now exists as a shareable image.\n")
 
    print("=" * 70)
    print("Day 29 complete. Same SHAP values as Day 28 — this time built into the")
    print("visuals a stakeholder or a report would actually expect to see.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()