"""
Day 28: Model Interpretability with SHAP
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 5, Day 1
 
Covers:
  1. Why gain-based feature importance (Day 23/25) isn't enough on its own
  2. SHAP values — additive, per-prediction feature attribution
  3. TreeExplainer for tree-based models (works with RF, GB, XGBoost)
  4. Explaining one individual prediction, not just the whole model
  5. Global SHAP importance (mean |SHAP value|) vs XGBoost's built-in gain importance
  6. Reading a SHAP explanation: base value + contributions = final prediction
  7. Practical use: explaining why one specific passenger's prediction came out the way it did
 
Requires: pip install shap xgboost
Expects titanic.csv in the same folder. If not found, downloads a copy.
"""

import os
import urllib.request
 
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

# ---------------------------------------------------------------------------
# COMPATIBILITY PATCH: xgboost >= 2.0 serializes base_score as "[value]"
# (bracketed, to support multi-target models). shap's TreeExplainer calls
# float() directly on that string and crashes. This patches shap's model
# loader to strip the brackets before parsing — a known version-mismatch
# issue between recent xgboost and shap, not a bug in this script's logic.
# ---------------------------------------------------------------------------

import shap.explainers._tree as _shap_tree_module

_original_decode_ubjson = _shap_tree_module.decode_ubjson_buffer

def _strip_base_score_brackets(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "base score" and isinstance(value, str) and value.strip().startswith("["):
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

DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
 
 
# ---------------------------------------------------------------------------
# 0. LOAD + SPLIT (same as Day 22-26 — raw, no manual cleaning)
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
# 1. SAME PREPROCESSOR + XGBOOST MODEL FROM DAY 21-25
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
 
    print("Same XGBoost pipeline from Day 25 — SHAP explains this exact fitted model,")
    print("it doesn't require retraining anything.\n")
    return pipeline
 

# ---------------------------------------------------------------------------
# 2. WHY GAIN-BASED IMPORTANCE ISN'T ENOUGH
# ---------------------------------------------------------------------------
 
def show_gain_importance(pipeline):
    print("=" * 70)
    print("2. RECAP: GAIN-BASED IMPORTANCE (DAY 25) — GLOBAL, NOT PER-PREDICTION")
    print("=" * 70)

    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    gain_importance = pd.Series(
        classifier.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    print(gain_importance.round(3).to_string())
    print("\nThis ranks features across the WHOLE model. It cannot answer:")
    print("'why did THIS specific passenger get predicted survived?' — that needs SHAP.\n")


# ---------------------------------------------------------------------------
# 3. BUILDING THE SHAP TREEEXPLAINER
# ---------------------------------------------------------------------------
 
def build_shap_explainer(pipeline, X_train):
    print("=" * 70)
    print("3. BUILDING A SHAP TREEEXPLAINER")
    print("=" * 70)
 
    engineer = pipeline.named_steps["engineer"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    
    # SHAP's TreeExplainer needs already-preprocessed (numeric) data —
    # it operates on the classifier directly, not the full pipeline object.
    X_train_transformed = preprocessor.transform(engineer.transform(X_train))
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    print("TreeExplainer is optimized specifically for tree ensembles (random forest,")
    print("gradient boosting, XGBoost, LightGBM) — it computes exact SHAP values fast,")
    print("instead of the slower model-agnostic KernelExplainer needed for other model types.\n")

    return explainer, X_train_transformed, feature_names


# ---------------------------------------------------------------------------
# 4. EXPLAINING ONE INDIVIDUAL PREDICTION
# ---------------------------------------------------------------------------

def explain_one_prediction(pipeline, explainer, X_test, feature_names, row_index=0):
    print("=" * 70)
    print(f"4. EXPLAINING ONE PREDICTION (test row {row_index})")
    print("=" * 70)

    engineer = pipeline.named_steps["engineer"]
    preprocessor = pipeline.named_steps["preprocessor"]

    X_test_transformed = preprocessor.transform(engineer.transform(X_test))
    row = X_test_transformed[row_index:row_index + 1]

    shap_values = explainer.shap_values(row)
    base_value = explainer.expected_value

    prediction = pipeline.named_steps["classifier"].predict_proba(row)[0][1]
    
    print(f"Base value (average model output, in log-odds/margin space): {base_value:.3f}")

    contributions = pd.Series(shap_values[0], index=feature_names).sort_values(
        key=abs, ascending=False
    )
    print("\nFeature contributions for this one passenger, largest impact first:")
    print(contributions.round(3).to_string())

    margin_sum = base_value + contributions.sum()
    reconstructed_prob = 1 / (1 + np.exp(-margin_sum))

    print(f"\nBase value + sum(contributions) = {margin_sum:.3f}  (still in log-odds space)")
    print(f"sigmoid(that log-odds sum) = {reconstructed_prob:.3f}")
    print(f"Model's actual predicted probability of survival: {prediction:.3f}")
    print("(these two should match closely — SHAP values are additive in log-odds space,")
    print("and the sigmoid function converts log-odds back to a 0-1 probability)\n")
 
    return contributions


# ---------------------------------------------------------------------------
# 5. GLOBAL SHAP IMPORTANCE VS GAIN-BASED IMPORTANCE
# ---------------------------------------------------------------------------
 
def compare_global_importance(pipeline, explainer, X_train_transformed, feature_names):
    print("=" * 70)
    print("5. GLOBAL SHAP IMPORTANCE VS GAIN-BASED IMPORTANCE")
    print("=" * 70)

    shap_values_all = explainer.shap_values(X_train_transformed)
    mean_abs_shap = pd.Series(
        np.abs(shap_values_all).mean(axis=0), index=feature_names).sort_values(ascending=False)

    classifier = pipeline.named_steps["classifier"]
    gain_importance = pd.Series(classifier.feature_importances_, index=feature_names).sort_values(ascending=False)

    comparison = pd.DataFrame({
        "gain_rank": gain_importance.rank(ascending=False).astype(int),
        "shap_rank": mean_abs_shap.reindex(gain_importance.index).rank(ascending=False).astype(int),})
    print(comparison.to_string())
    print("\nGain measures how much a feature reduced loss when splits used it.")
    print("Mean |SHAP value| measures how much a feature moved individual predictions,")
    print("averaged across every row — the two often agree on top features but can rank")
    print("middle-of-the-pack features differently.\n")
 
    return mean_abs_shap

 
# ---------------------------------------------------------------------------
# 6. COMPARING TWO PASSENGERS WITH OPPOSITE PREDICTIONS
# ---------------------------------------------------------------------------
 
def compare_two_predictions(pipeline, explainer, X_test, y_test, feature_names):
    print("=" * 70)
    print("6. COMPARING SHAP EXPLANATIONS FOR TWO DIFFERENT PASSENGERS")
    print("=" * 70)
 
    engineer = pipeline.named_steps["engineer"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    
    X_test_transformed = preprocessor.transform(engineer.transform(X_test))
    probs = classifier.predict_proba(X_test_transformed)[:, 1]
 
    highest_idx = int(np.argmax(probs))
    lowest_idx = int(np.argmin(probs))
 
    for label, idx in [("highest predicted survival probability", highest_idx),
                        ("lowest predicted survival probability", lowest_idx)]:
        row = X_test_transformed[idx:idx + 1]
        shap_values = explainer.shap_values(row)
        contributions = pd.Series(shap_values[0], index=feature_names).sort_values(
            key=abs, ascending=False
        ).head(3)
        print(f"\nPassenger with {label} (p={probs[idx]:.3f}):")
        print(contributions.round(3).to_string())
 
    print("\nEach passenger gets their own explanation — this is what a global feature")
    print("importance ranking alone can never show.\n")
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    X_train, X_test, y_train, y_test = load_and_split()
    pipeline = build_and_fit_model(X_train, y_train)
 
    show_gain_importance(pipeline)
 
    explainer, X_train_transformed, feature_names = build_shap_explainer(pipeline, X_train)
    explain_one_prediction(pipeline, explainer, X_test, feature_names, row_index=0)
    compare_global_importance(pipeline, explainer, X_train_transformed, feature_names)
    compare_two_predictions(pipeline, explainer, X_test, y_test, feature_names)
 
    print("=" * 70)
    print("Day 28 complete. Moved from 'which features matter overall' to 'why did the model")
    print("make THIS specific prediction' — the question stakeholders actually ask in practice.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()