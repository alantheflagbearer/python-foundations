"""
Day 24: Handling Imbalanced Classes
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 4, Day 4
 
Covers:
  1. Why accuracy is misleading on imbalanced data
  2. class_weight='balanced' — reweighting the loss during training
  3. Oversampling vs undersampling
  4. SMOTE (Synthetic Minority Oversampling Technique)
  5. Why resampling must happen only on training folds (imblearn.Pipeline)
  6. Comparing baseline vs class_weight vs SMOTE on minority-class metrics
 
Requires: pip install imbalanced-learn
Expects titanic.csv in the same folder. If not found, downloads a copy.
"""
 
import os
import urllib.request
 
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
 
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
 
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
 
DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
 
 
# ---------------------------------------------------------------------------
# 0. LOAD + SPLIT + CLASS BALANCE CHECK
# ---------------------------------------------------------------------------
 
def load_and_split(path=DATA_PATH, url=DATA_URL):
    print("=" * 70)
    print("0. LOAD + SPLIT + CLASS BALANCE CHECK")
    print("=" * 70)
 
    if not os.path.exists(path):
        print(f"{path} not found locally — downloading...")
        urllib.request.urlretrieve(url, path)
 
    df = pd.read_csv(path)
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[feature_cols]
    y = df["Survived"]
 
    print("Class balance:")
    print(y.value_counts(normalize=True).round(3).rename({0: "died", 1: "survived"}))
    print(f"\nA model that always predicts 'died' would score "
          f"{(y == 0).mean():.1%} accuracy while catching zero survivors.\n")
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows\n")
    return X_train, X_test, y_train, y_test
 
 
# ---------------------------------------------------------------------------
# 1. SAME PREPROCESSOR FROM DAY 21-23
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
 
 
# ---------------------------------------------------------------------------
# 2. THREE PIPELINES: BASELINE, CLASS_WEIGHT, SMOTE
# ---------------------------------------------------------------------------
 
def build_pipelines(preprocessor):
    print("=" * 70)
    print("1-2. BUILDING THREE PIPELINES: BASELINE, CLASS_WEIGHT, SMOTE")
    print("=" * 70)
 
    engineer = FunctionTransformer(add_family_size)
 
    baseline_pipeline = Pipeline(steps=[
        ("engineer", engineer),
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
 
    weighted_pipeline = Pipeline(steps=[
        ("engineer", engineer),
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )),
    ])
 
    # imblearn's Pipeline — the only kind that can include a sampling step like SMOTE
    smote_pipeline = ImbPipeline(steps=[
        ("engineer", engineer),
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
 
    print("baseline_pipeline: no adjustment for imbalance")
    print("weighted_pipeline: class_weight='balanced' reweights the loss during training")
    print("smote_pipeline:    imblearn.Pipeline, SMOTE generates synthetic minority rows")
    print("                   — but ONLY during .fit() on training folds, never during .predict()\n")
 
    return baseline_pipeline, weighted_pipeline, smote_pipeline
 
 
# ---------------------------------------------------------------------------
# 3. FIT + EVALUATE, JUDGED ON MINORITY-CLASS METRICS
# ---------------------------------------------------------------------------
 
def fit_and_evaluate(name, pipeline, X_train, y_train, X_test, y_test):
    print("=" * 70)
    print(f"3. FIT + EVALUATE: {name}")
    print("=" * 70)
 
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
 
    print(f"Overall accuracy: {accuracy_score(y_test, preds):.3f}  (misleading on its own — see below)")
    report = classification_report(
        y_test, preds, target_names=["died", "survived"], output_dict=True
    )
    print(classification_report(y_test, preds, target_names=["died", "survived"]))
 
    survived_recall = report["survived"]["recall"]
    survived_precision = report["survived"]["precision"]
    print(f"Survived-class recall:    {survived_recall:.3f}  (of real survivors, how many did we catch?)")
    print(f"Survived-class precision: {survived_precision:.3f}  (of our 'survived' calls, how many were right?)\n")
 
    return report
 
 
# ---------------------------------------------------------------------------
# 4. SIDE-BY-SIDE COMPARISON ON THE MINORITY CLASS
# ---------------------------------------------------------------------------
 
def compare_reports(reports):
    print("=" * 70)
    print("4. SIDE-BY-SIDE COMPARISON (survived class only — this is what matters here)")
    print("=" * 70)
 
    rows = []
    for name, report in reports.items():
        rows.append({
            "approach": name,
            "survived_precision": report["survived"]["precision"],
            "survived_recall": report["survived"]["recall"],
            "survived_f1": report["survived"]["f1-score"],
        })
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    best = table.loc[table["survived_f1"].idxmax()]
    print(f"\nBest survived-class F1: {best['approach']} ({best['survived_f1']:.3f})")
    print("Whether class_weight or SMOTE actually helps varies by dataset — that's exactly")
    print("why both get compared here instead of assuming one is always better.\n")
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    X_train, X_test, y_train, y_test = load_and_split()
    preprocessor = build_preprocessor()
    baseline_pipeline, weighted_pipeline, smote_pipeline = build_pipelines(preprocessor)
 
    reports = {}
    reports["Baseline"] = fit_and_evaluate(
        "Baseline (no adjustment)", baseline_pipeline, X_train, y_train, X_test, y_test
    )
    reports["class_weight"] = fit_and_evaluate(
        "class_weight='balanced'", weighted_pipeline, X_train, y_train, X_test, y_test
    )
    reports["SMOTE"] = fit_and_evaluate(
        "SMOTE (imblearn.Pipeline)", smote_pipeline, X_train, y_train, X_test, y_test
    )
 
    compare_reports(reports)
 
    print("=" * 70)
    print("Day 24 complete. Imbalance addressed at training time, judged on the metric that")
    print("actually matters for the minority class — not overall accuracy.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()