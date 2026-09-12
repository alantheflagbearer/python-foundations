"""
Day 33: Model Persistence with joblib
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 5, Day 6
 
Covers:
  1. Why persistence matters — training a model once, using it many times
     without rerunning the entire training script
  2. joblib.dump/joblib.load — saving and loading the fitted pipeline object
  3. Why the WHOLE pipeline must be saved, not just the classifier — proven
     by deliberately showing the failure mode when preprocessing is skipped
  4. Saving metadata alongside the model (versions, feature names, CV score,
     timestamp) so a loaded model's provenance is never a mystery
  5. Loading the pipeline fresh and predicting on brand-new raw data, with
     no access to the original training script or DataFrame
  6. joblib vs XGBoost's native save_model() — a version-stability tradeoff
 
Requires: pip install scikit-learn xgboost joblib
Expects titanic.csv in the same folder. If not found, downloads a copy.
"""
 
import json
import os
import platform
import sys
import urllib.request
from datetime import datetime, timezone
 
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
 
from xgboost import XGBClassifier
 
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
 
DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
MODEL_PATH = "titanic_pipeline.joblib"
METADATA_PATH = "titanic_pipeline_metadata.json"
 
 
# ---------------------------------------------------------------------------
# 0. LOAD + SPLIT (same as Day 22-32 — raw, no manual cleaning)
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
# 1. SAME PREPROCESSOR + XGBOOST PIPELINE FROM DAY 21-32
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
 
 
def build_and_fit_pipeline(X_train, y_train):
    print("=" * 70)
    print("1. FITTING THE PIPELINE (final tuned settings from Day 25/31/32)")
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
 
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print(f"Pipeline fitted. 5-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})\n")
    return pipeline, cv_scores.mean()


# ---------------------------------------------------------------------------
# 2. SAVING THE WHOLE PIPELINE + METADATA
# ---------------------------------------------------------------------------
 
def save_pipeline_with_metadata(pipeline, cv_score, X_train):
    print("=" * 70)
    print("2. SAVING THE PIPELINE + METADATA")
    print("=" * 70)
 
    joblib.dump(pipeline, MODEL_PATH)
    file_size_kb = os.path.getsize(MODEL_PATH) / 1024
 
    metadata = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgb.__version__,
        "joblib_version": joblib.__version__,
        "feature_columns": list(X_train.columns),
        "training_rows": int(X_train.shape[0]),
        "cv_accuracy_mean": round(float(cv_score), 4),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
 
    print(f"Saved: {MODEL_PATH} ({file_size_kb:.1f} KB)")
    print(f"Saved: {METADATA_PATH}")
    print(json.dumps(metadata, indent=2))
    print("\nThe WHOLE pipeline is saved as one object — engineer, preprocessor (with its")
    print("fitted imputer medians/modes, scaler mean/std, and encoder categories), and the")
    print("classifier — not just the classifier alone. The metadata file records exactly which")
    print("library versions produced this file, so a future 'why won't this load' has an answer")
    print("without guessing.\n")
 
 
# ---------------------------------------------------------------------------
# 3. LOADING FRESH AND PREDICTING ON BRAND-NEW RAW DATA
# ---------------------------------------------------------------------------
 
def load_and_predict_new_passengers():
    print("=" * 70)
    print("3. LOADING FRESH + PREDICTING ON NEW, UNSEEN RAW DATA")
    print("=" * 70)
 
    loaded_pipeline = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
 
    print(f"Loaded {MODEL_PATH}, trained with sklearn=={metadata['sklearn_version']}, "
          f"xgboost=={metadata['xgboost_version']}")
    print(f"Currently running sklearn=={sklearn.__version__}, xgboost=={xgb.__version__}")
    if metadata["sklearn_version"] != sklearn.__version__ or metadata["xgboost_version"] != xgb.__version__:
        print("VERSION MISMATCH — predictions below may still work, but aren't guaranteed to be")
        print("identical to what the original training environment would have produced.\n")
    else:
        print("Versions match — no environment drift between training and this load.\n")
 
    # Brand-new raw passengers — never seen during training, no manual cleaning applied.
    # NOTE: missing values use np.nan, not Python's None — np.nan is what pandas actually
    # produces for missing cells read from a real CSV, and it's what SimpleImputer's
    # missing_values=np.nan default is built to detect. Python's None does NOT reliably
    # match that check for object-dtype columns and can silently slip through the imputer
    # unfilled, hitting the encoder as an "unknown category" instead of being imputed —
    # worth knowing, since it's an easy mistake when hand-building test data like this.
    new_passengers = pd.DataFrame([
        {"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 100.0, "Embarked": "S"},
        {"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"},
        {"Pclass": 2, "Sex": "female", "Age": np.nan, "SibSp": 1, "Parch": 2, "Fare": 26.0, "Embarked": np.nan},
    ])
 
    predictions = loaded_pipeline.predict(new_passengers)
    probabilities = loaded_pipeline.predict_proba(new_passengers)[:, 1]
 
    results = new_passengers.copy()
    results["predicted_survived"] = predictions
    results["survival_probability"] = probabilities.round(3)
    print(results.to_string(index=False))
    print("\nRow 3 has a missing Age and a missing Embarked — the loaded pipeline's own imputer")
    print("(fit during training, saved as part of the pipeline object) handles both correctly,")
    print("with zero manual preprocessing written in this function.\n")
 
    return loaded_pipeline
 
 
# ---------------------------------------------------------------------------
# 4. WHY THE WHOLE PIPELINE MUST BE SAVED — THE FAILURE MODE
# ---------------------------------------------------------------------------
 
def demonstrate_classifier_only_failure(loaded_pipeline):
    print("=" * 70)
    print("4. THE FAILURE MODE: SAVING ONLY THE CLASSIFIER, NOT THE PIPELINE")
    print("=" * 70)
 
    classifier_only = loaded_pipeline.named_steps["classifier"]
    raw_row = pd.DataFrame([
        {"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 100.0, "Embarked": "S"},
    ])
 
    print("Attempting classifier_only.predict(raw_row) — raw, unprocessed data, no engineer,")
    print("no imputer, no scaler, no one-hot encoding applied:")
    try:
        classifier_only.predict(raw_row)
        print("(unexpectedly succeeded — this should not happen with unencoded string columns)")
    except Exception as exc:
        print(f"FAILED as expected: {type(exc).__name__}: {str(exc)[:150]}")
 
    print("\nThis is exactly why Topic 2 saved the ENTIRE pipeline object, not just")
    print("pipeline.named_steps['classifier']. The classifier alone has no idea how to turn")
    print("'Sex': 'female' into the numeric, encoded, scaled array it was actually trained on —")
    print("only the full pipeline remembers every preprocessing step.\n")
 
 
# ---------------------------------------------------------------------------
# 5. JOBLIB VS XGBOOST'S NATIVE SAVE FORMAT
# ---------------------------------------------------------------------------
 
def compare_persistence_formats(pipeline):
    print("=" * 70)
    print("5. JOBLIB VS XGBOOST'S NATIVE save_model()")
    print("=" * 70)
 
    classifier = pipeline.named_steps["classifier"]
    native_path = "xgboost_classifier_only.json"
    classifier.save_model(native_path)
    native_size_kb = os.path.getsize(native_path) / 1024
    joblib_size_kb = os.path.getsize(MODEL_PATH) / 1024
 
    print(f"joblib (whole pipeline):        {MODEL_PATH} — {joblib_size_kb:.1f} KB")
    print(f"XGBoost native (classifier only): {native_path} — {native_size_kb:.1f} KB")
    print("\njoblib pickles the ENTIRE pipeline, including scikit-learn objects (ColumnTransformer,")
    print("SimpleImputer, OneHotEncoder) — convenient, but pickle files can break across sklearn")
    print("version changes, since pickle serializes Python object internals, not a stable format.")
    print("XGBoost's own save_model() writes a documented, version-stable JSON format for the")
    print("classifier ONLY — the preprocessing steps still need to be saved separately (e.g. with")
    print("joblib, just for the ColumnTransformer alone) if this format is used.\n")
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    X_train, X_test, y_train, y_test = load_and_split()
    pipeline, cv_score = build_and_fit_pipeline(X_train, y_train)
 
    save_pipeline_with_metadata(pipeline, cv_score, X_train)
    loaded_pipeline = load_and_predict_new_passengers()
    demonstrate_classifier_only_failure(loaded_pipeline)
    compare_persistence_formats(pipeline)
 
    print("=" * 70)
    print("Day 33 complete. A trained pipeline now exists as a file that can be loaded and used")
    print("without ever rerunning training — the first piece a real deployment needs.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
 