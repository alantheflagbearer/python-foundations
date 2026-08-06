"""
Day 18: ROC Curves, AUC, and Decision Thresholds
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 3, Day 4
 
Covers:
  1. predict_proba() vs predict()
  2. ROC curve (roc_curve) and AUC (roc_auc_score)
  3. The precision/recall tradeoff
  4. Precision-recall curve
  5. Choosing a threshold on purpose (F1-optimal, not just 0.5)
 
Expects titanic.csv in the same folder. If not found, downloads a copy.
Saves day18_roc_pr_curves.png.
"""

import os
from tkinter import _test
import urllib.request

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"



# ---------------------------------------------------------------------------
# 0. LOAD, CLEAN, ENCODE, SPLIT, TRAIN (same pipeline as Day 15-17)
# ---------------------------------------------------------------------------

def prepare_and_train(path=DATA_PATH, url=DATA_URL):
    print("=" * 70)
    print("0. LOAD, CLEAN, ENCODE, SPLIT, TRAIN")
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

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)

    forest = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    forest.fit(X_train, y_train)

    print(f"loaded {df.shape[0]} passengers. trained logistic regression and random forest. \n")

    return {
        "X_test": X_test, "y_test": y_test,
        "logreg": logreg, "forest": forest
    }

# ---------------------------------------------------------------------------
# 1. PREDICT_PROBA VS PREDICT
# ---------------------------------------------------------------------------


def show_predict_proba(models):
    print("=" * 70)
    print("1. PREDICT_PROBA VS PREDICT")
    print("=" * 70)
 
    logreg = models["logreg"]
    X_test = models["X_test"]
 
    labels = logreg.predict(X_test)
    probs = logreg.predict_proba(X_test)[:, 1]
 
    preview = pd.DataFrame({
        "P(survived)": probs[:8].round(3),
        ".predict() label": labels[:8],
        "matches (prob > 0.5)?": (probs[:8] > 0.5) == (labels[:8] == 1),
    })
    print(preview.to_string())
    print("\n.predict() is exactly '.predict_proba() > 0.5' under the hood — confirmed above.\n")
 
    return probs

# ---------------------------------------------------------------------------
# 2. ROC CURVE + AUC
# ---------------------------------------------------------------------------


def compute_roc(models, logreg_probs):
    print("=" * 70)
    print("2. ROC CURVE + AUC")
    print("=" * 70)

    y_test = models["y_test"]
    forest_probs = models["forest"].predict_proba(models["X_test"])[:, 1]

    fpr_lr, tpr_lr, _ = roc_curve(y_test, logreg_probs)
    auc_lr = roc_auc_score(y_test, logreg_probs)

    fpr_rf, tpr_rf, _ = roc_curve(y_test, forest_probs)
    auc_rf = roc_auc_score(y_test, forest_probs)

    print(f"Logistic Regression AUC = {auc_lr:.3f}")
    print(f"Random Forest AUC = {auc_rf:.3f}\n")
    print(f"{'Logistic Regression' if auc_lr > auc_rf else 'Random Forest'} ranks survivors vs "
          f"non_survivors better on this test set (higher AUC = better ranking, independent of threshold).\n")

    return {
        "fpr_lr": fpr_lr, "tpr_lr": tpr_lr, "auc_lr": auc_lr,
        "fpr_rf": fpr_rf, "tpr_rf": tpr_rf, "auc_rf": auc_rf,
        "forest_probs": forest_probs,
    }

# ---------------------------------------------------------------------------
# 3-4. PRECISION/RECALL TRADEOFF + PR CURVE
# ---------------------------------------------------------------------------

def show_threshold_tradeoff(models, probs):
    print("=" * 70)
    print("3. THE PRECISION/RECALL TRADEOFF (logistic regression)")
    print("=" * 70)

    y_test = models["y_test"]
    rows = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds_at_t = (probs > t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_test, preds_at_t, zero_division=0),
            "recall": recall_score(y_test, preds_at_t, zero_division=0),
            "accuracy": accuracy_score(y_test, preds_at_t),
        })
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    print("\nLower threshold -> higer recall, lower precision. raise it -> the reverse.")
    print("Neither direction improves both at once - thats the tradeoff, not a bug.\n")

    return table

def compute_pr_curve(models, probs):
    print("=" * 70)
    print("4. PRECISION-RECALL CURVE + F1-OPTIMAL THRESHOLD")
    print("=" * 70)

    y_test = models["y_test"]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)

    # precision/recall arrays are 1 longer than thresholds; align them
    denom = precision[:-1] + recall[:-1]
    numer = 2 * precision[:-1] * recall[:-1]
    f1_scores = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"F1-optimal threshold found by sweeping the PR curve: {best_threshold:.3f}")
    print(f"At that threshold: precision = {precision[best_idx]:.3f}, "
          f"recall = {recall[best_idx]:.3f}, F1 = {f1_scores[best_idx]:.3f}")
    print("(compare this to the default 0.5 in the table above - it may or maay not match)\n")

    return precision, recall, thresholds, best_threshold

# ---------------------------------------------------------------------------
# 5. PLOT BOTH CURVES
# ---------------------------------------------------------------------------
 
def plot_curves(roc_data, pr_data, models, logreg_probs):
    print("=" * 70)
    print("5. PLOTTING ROC AND PRECISION-RECALL CURVES")
    print("=" * 70)

    precision, recall, thresholds, best_threshold = pr_data
    y_test = models["y_test"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].plot(roc_data["fpr_lr"], roc_data["tpr_lr"],
                 label=f"Logistic Regression (AUC={roc_data['auc_lr']:.3f})", color="#378ADD")

    axes[0].plot(roc_data["fpr_rf"], roc_data["tpr_rf"],
                 label=f"Random Forest (AUC={roc_data['auc_rf']:.3f})", color="#D85A30")

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess (AUC=0.5)")

    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(fontsize=8)

    axes[1].plot(recall, precision, color="#1D9E75", label="Logistic Regression")
    axes[1].scatter(
        [recall[np.argmin(np.abs(thresholds - best_threshold))]],
        [precision[np.argmin(np.abs(thresholds - best_threshold))]],
        color="#D85A30", zorder=5, label=f"F1-optimal (t={best_threshold:.2f})"
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve")
    axes[1].legend(fontsize=8)
 
    plt.tight_layout()
    plt.savefig("day18_roc_pr_curves.png", dpi=150)
    plt.close()
    print("Saved plot: day18_roc_pr_curves.png\n") 

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    models = prepare_and_train()
    logreg_probs = show_predict_proba(models)
    roc_data = compute_roc(models, logreg_probs)
    show_threshold_tradeoff(models, logreg_probs)
    pr_data = compute_pr_curve(models, logreg_probs)
    plot_curves(roc_data, pr_data, models, logreg_probs)
 
    print("=" * 70)
    print("Day 18 complete. Thresholds chosen on purpose, not defaulted to 0.5.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()