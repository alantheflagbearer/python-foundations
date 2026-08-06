"""
Day 19: Full Classification Portfolio Project
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 3, Day 5
 
Brings Days 15-18 together into one end-to-end pipeline:
  1. Load, clean, encode (same pipeline as Day 15-18)
  2. Select AND tune a model with GridSearchCV across 3 candidates
  3. Choose a decision threshold using out-of-fold CV predictions (no test-set peeking)
  4. Evaluate once, honestly, on the held-out test set
  5. Save the final model (+ threshold + feature columns) to disk with joblib
  6. Reload the saved file and prove it reproduces the same predictions
 
Expects titanic.csv in the same folder. If not found, downloads a copy.
Saves day19_final_evaluation.png and day19_final_model.joblib.
"""

import os
import urllib.request
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    confusion_matrix, classification_report, accuracy_score
)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
 
DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


 # ---------------------------------------------------------------------------
# 0. LOAD, CLEAN, ENCODE, SPLIT (same pipeline as Day 15-18)
# ---------------------------------------------------------------------------
 
def prepare_data(path=DATA_PATH, url=DATA_URL):
    print("=" * 70)
    print("0. LOAD, CLEAN, ENCODE, SPLIT")
    print("=" * 70)
 
    if not os.path.exists(path):
        print(f"{path} not found locally — downloading...")
        urllib.request.urlretrieve(url, path)
 
    df = pd.read_csv(path)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df = df.drop(columns=["Cabin"])
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
 
    y = df["Survived"]
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
                     "Embarked", "FamilySize"]
    X = pd.get_dummies(df[feature_cols], columns=["Sex", "Embarked"], drop_first=True)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Loaded {df.shape[0]} passengers, {X.shape[1]} features after encoding")
    print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows (held out until step 3)\n")
 
    return X_train, X_test, y_train, y_test, list(X.columns) 

# ---------------------------------------------------------------------------
# 1. SELECT + TUNE — GridSearchCV across three candidate models
# ---------------------------------------------------------------------------

def select_and_tune_model(X_train, y_train):
    print("=" * 70)
    print("1. MODEL SELECTION + TUNING (GridSearchCV across 3 candidates)")
    print("=" * 70)

    candidates = {
      "Logistic Regression": (
            LogisticRegression(max_iter=1000),
            {"C": [0.1, 1, 10]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [3, 5, None], "min_samples_leaf": [5, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [4, 8]},)
    }
    best_name, best_model, best_score = None, None, -1
    for name, (estimator, grid) in candidates.items():
        search = GridSearchCV(estimator, grid, cv=5, n_jobs=-1)
        search.fit(X_train, y_train)
        print(f"{name}: best params = {search.best_params_}, CV score = {search.best_score_:.3f}")
        if search.best_score_ > best_score:
            best_name, best_model, best_score = name, search.best_estimator_, search.best_score_
 
    print(f"\nWinner: {best_name} (CV score {best_score:.3f})\n")
    return best_name, best_model, best_score

# ---------------------------------------------------------------------------
# 2. THRESHOLD SELECTION — out-of-fold predictions, no test-set peeking
# ---------------------------------------------------------------------------
 
def choose_threshold(best_model, X_train, y_train):
    print("=" * 70)
    print("2. THRESHOLD SELECTION (cross_val_predict, training data only)")
    print("=" * 70)
 
    oof_probs = cross_val_predict(
        best_model, X_train, y_train, cv=5, method="predict_proba"
    )[:, 1]
 
    precision, recall, thresholds = precision_recall_curve(y_train, oof_probs)
    denom = precision[:-1] + recall[:-1]
    numer = 2 * precision[:-1] * recall[:-1]
    f1_scores = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
 
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
 
    print(f"Out-of-fold F1-optimal threshold: {best_threshold:.3f}")
    print(f"At that threshold (on out-of-fold data): precision = {precision[best_idx]:.3f}, "
          f"recall = {recall[best_idx]:.3f}, F1 = {f1_scores[best_idx]:.3f}")
    print("This threshold was chosen without the test set being involved at all.\n")
 
    return best_threshold

# ---------------------------------------------------------------------------
# 3. FINAL TEST-SET EVALUATION — the one and only look at X_test
# ---------------------------------------------------------------------------
 
def final_evaluation(best_model, best_threshold, X_train, y_train, X_test, y_test):
    print("=" * 70)
    print("3. FINAL EVALUATION (the one and only look at the held-out test set)")
    print("=" * 70)
 
    best_model.fit(X_train, y_train)
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)
 
    print(f"Test accuracy: {accuracy_score(y_test, test_preds):.3f}")
    print(f"Test AUC:      {roc_auc_score(y_test, test_probs):.3f}\n")
 
    cm = confusion_matrix(y_test, test_preds)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(f"                predicted died  predicted survived")
    print(f"actual died          {cm[0][0]:>4}              {cm[0][1]:>4}")
    print(f"actual survived      {cm[1][0]:>4}              {cm[1][1]:>4}\n")
 
    print("Classification report:")
    print(classification_report(y_test, test_preds, target_names=["died", "survived"]))
 
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    precision, recall, _ = precision_recall_curve(y_test, test_probs)
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(fpr, tpr, color="#378ADD", label=f"AUC={roc_auc_score(y_test, test_probs):.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve (final model, test set)")
    axes[0].legend(fontsize=8)
 
    axes[1].plot(recall, precision, color="#1D9E75")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve (final model, test set)")
 
    plt.tight_layout()
    plt.savefig("day19_final_evaluation.png", dpi=150)
    plt.close()
    print("Saved plot: day19_final_evaluation.png\n")
 
    return best_model
 
 
# ---------------------------------------------------------------------------
# 4. SAVE MODEL — bundle model + threshold + feature columns together
# ---------------------------------------------------------------------------
 
def save_model(model, threshold, feature_columns, path="day19_final_model.joblib"):
    print("=" * 70)
    print("4. SAVING THE MODEL")
    print("=" * 70)
 
    bundle = {
        "model": model,
        "threshold": threshold,
        "feature_columns": feature_columns,
    }
    joblib.dump(bundle, path)
    print(f"Saved model bundle to {path}")
    print(f"Bundle contains: model, threshold ({threshold:.3f}), "
          f"and {len(feature_columns)} feature column names\n")
 
 
# ---------------------------------------------------------------------------
# 5. RELOAD AND VERIFY — prove the saved file actually works
# ---------------------------------------------------------------------------
 
def verify_reload(original_model, threshold, X_test, y_test, path="day19_final_model.joblib"):
    print("=" * 70)
    print("5. RELOADING AND VERIFYING THE SAVED MODEL")
    print("=" * 70)
 
    bundle = joblib.load(path)
    reloaded_model = bundle["model"]
    reloaded_threshold = bundle["threshold"]
 
    sample = X_test.iloc[:10]
    original_probs = original_model.predict_proba(sample)[:, 1]
    reloaded_probs = reloaded_model.predict_proba(sample)[:, 1]
 
    original_preds = (original_probs >= threshold).astype(int)
    reloaded_preds = (reloaded_probs >= reloaded_threshold).astype(int)
 
    match = np.array_equal(original_preds, reloaded_preds)
    print(f"Original model predictions on 10 sample passengers: {original_preds.tolist()}")
    print(f"Reloaded model predictions on the same passengers:  {reloaded_preds.tolist()}")
    print(f"Match: {match} — {'the saved file is a verified, working deliverable' if match else 'MISMATCH — investigate before shipping this file'}\n")
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    X_train, X_test, y_train, y_test, feature_columns = prepare_data()
 
    best_name, best_model, best_cv_score = select_and_tune_model(X_train, y_train)
    best_threshold = choose_threshold(best_model, X_train, y_train)
    fitted_model = final_evaluation(best_model, best_threshold, X_train, y_train, X_test, y_test)
    save_model(fitted_model, best_threshold, feature_columns)
    verify_reload(fitted_model, best_threshold, X_test, y_test)
 
    print("=" * 70)
    print(f"Day 19 complete. Final model: {best_name}, threshold: {best_threshold:.3f}")
    print("Full pipeline: selected, tuned, thresholded, evaluated, saved, and verified.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()