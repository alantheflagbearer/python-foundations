"""
Day 16: Decision Trees and Random Forests
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 3, Day 2
 
Covers:
  1. A single decision tree (DecisionTreeClassifier)
  2. Overfitting: train vs test accuracy, max_depth / min_samples_leaf
  3. Random Forest (ensemble of trees)
  4. Feature importance
  5. Comparing logistic regression vs tree vs forest
 
Expects titanic.csv in the same folder. If not found, downloads a copy.
"""
 
import os
import urllib.request
 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
 
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
 
DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
 
 
# ---------------------------------------------------------------------------
# 0. LOAD, CLEAN, FEATURES, ENCODE, SPLIT (same pipeline as Day 15)
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
    print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows\n")
 
    return X_train, X_test, y_train, y_test
 
 
# ---------------------------------------------------------------------------
# 1. LOGISTIC REGRESSION (baseline from Day 15, for comparison)
# ---------------------------------------------------------------------------
 
def run_logistic_regression(X_train, X_test, y_train, y_test):
    print("=" * 70)
    print("1. LOGISTIC REGRESSION (Day 15 baseline)")
    print("=" * 70)
 
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
 
    print(f"Test accuracy: {test_acc:.3f}\n")
    return test_acc
 
 
# ---------------------------------------------------------------------------
# 2. A SINGLE DECISION TREE + OVERFITTING DEMONSTRATION
# ---------------------------------------------------------------------------
 
def run_decision_tree(X_train, X_test, y_train, y_test):
    print("=" * 70)
    print("2. DECISION TREE — UNRESTRICTED (watch for overfitting)")
    print("=" * 70)
 
    tree_unrestricted = DecisionTreeClassifier(random_state=42)
    tree_unrestricted.fit(X_train, y_train)
 
    train_acc = accuracy_score(y_train, tree_unrestricted.predict(X_train))
    test_acc = accuracy_score(y_test, tree_unrestricted.predict(X_test))
 
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")
    print(f"Gap (train - test): {train_acc - test_acc:.3f} "
          f"— {'large gap, likely overfitting' if train_acc - test_acc > 0.1 else 'reasonably small gap'}")
 
    print("\n--- Same tree, but depth-limited (max_depth=4, min_samples_leaf=10) ---")
    tree_limited = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=10, random_state=42
    )
    tree_limited.fit(X_train, y_train)
 
    train_acc_ltd = accuracy_score(y_train, tree_limited.predict(X_train))
    test_acc_ltd = accuracy_score(y_test, tree_limited.predict(X_test))
 
    print(f"Train accuracy: {train_acc_ltd:.3f}")
    print(f"Test accuracy:  {test_acc_ltd:.3f}")
    print(f"Gap (train - test): {train_acc_ltd - test_acc_ltd:.3f}")
    print()
 
    return test_acc_ltd, train_acc - test_acc
 
 
# ---------------------------------------------------------------------------
# 3. RANDOM FOREST
# ---------------------------------------------------------------------------
 
def run_random_forest(X_train, X_test, y_train, y_test, unrestricted_tree_gap):
    print("=" * 70)
    print("3. RANDOM FOREST")
    print("=" * 70)
 
    forest = RandomForestClassifier(n_estimators=200, random_state=42)
    forest.fit(X_train, y_train)
 
    train_acc = accuracy_score(y_train, forest.predict(X_train))
    test_acc = accuracy_score(y_test, forest.predict(X_test))
    forest_gap = train_acc - test_acc
 
    print(f"Trees in forest: 200")
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")
    if forest_gap < unrestricted_tree_gap:
        print(f"Gap (train - test): {forest_gap:.3f} — smaller than the unrestricted "
              f"single tree's gap ({unrestricted_tree_gap:.3f}), as averaging usually predicts")
    else:
        print(f"Gap (train - test): {forest_gap:.3f} — not smaller than the unrestricted "
              f"single tree's gap ({unrestricted_tree_gap:.3f}) this time. Averaging reduces "
              f"variance on average, not on every single run — with a small dataset like this "
              f"one split can easily be the exception")
    print()
 
    return forest, test_acc
 
 
# ---------------------------------------------------------------------------
# 4. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
 
def show_feature_importance(forest, X_train):
    print("=" * 70)
    print("4. FEATURE IMPORTANCE (random forest)")
    print("=" * 70)
 
    importances = pd.Series(
        forest.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
 
    print(importances.round(3).to_string())
    print(f"\nMost important feature: {importances.index[0]} "
          f"({importances.iloc[0]:.1%} of total importance)\n")
 
 
# ---------------------------------------------------------------------------
# 5. COMPARISON TABLE
# ---------------------------------------------------------------------------
 
def print_comparison(logreg_acc, tree_acc, forest_acc):
    print("=" * 70)
    print("5. MODEL COMPARISON")
    print("=" * 70)
 
    results = pd.DataFrame({
        "model": ["Logistic Regression", "Decision Tree (depth-limited)", "Random Forest"],
        "test_accuracy": [logreg_acc, tree_acc, forest_acc],
    }).sort_values("test_accuracy", ascending=False)
 
    print(results.to_string(index=False))
    best = results.iloc[0]
    print(f"\nBest performer on this split: {best['model']} ({best['test_accuracy']:.3f})")
    print("Note: on a small test set, differences this close can shift with a different")
    print("random_state — this is exactly why cross-validation exists (coming up next).\n")
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    X_train, X_test, y_train, y_test = prepare_data()
 
    logreg_acc = run_logistic_regression(X_train, X_test, y_train, y_test)
    tree_acc, unrestricted_tree_gap = run_decision_tree(X_train, X_test, y_train, y_test)
    forest, forest_acc = run_random_forest(X_train, X_test, y_train, y_test, unrestricted_tree_gap)
    show_feature_importance(forest, X_train)
    print_comparison(logreg_acc, tree_acc, forest_acc)
 
    print("=" * 70)
    print("Day 16 complete. Three models trained and compared.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()