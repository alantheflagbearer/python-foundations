"""
Day 17: Cross-Validation and Hyperparameter Tuning
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 3, Day 3
 
Covers:
  1. Why a single train/test split isn't a reliable evaluation
  2. K-Fold cross-validation via cross_val_score
  3. Hyperparameters vs learned parameters
  4. GridSearchCV
  5. Tuned vs default comparison, reported honestly (mean +/- std)
 
Expects titanic.csv in the same folder. If not found, downloads a copy.
"""
 
import os
import urllib.request
 
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
 
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
 
DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
 
 
# ---------------------------------------------------------------------------
# 0. LOAD, CLEAN, ENCODE, SPLIT (same pipeline as Day 15-16)
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
    print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows (held out, untouched until step 5)\n")
 
    return X_train, X_test, y_train, y_test
 
 
# ---------------------------------------------------------------------------
# 1. WHY ONE SPLIT ISN'T ENOUGH — same model, different random splits
# ---------------------------------------------------------------------------
 
def demonstrate_split_instability(X_train, y_train):
    print("=" * 70)
    print("1. WHY ONE SPLIT ISN'T ENOUGH")
    print("=" * 70)
 
    accs = []
    for seed in range(5):
        X_a, X_b, y_a, y_b = train_test_split(
            X_train, y_train, test_size=0.25, random_state=seed, stratify=y_train
        )
        model = LogisticRegression(max_iter=1000)
        model.fit(X_a, y_a)
        acc = accuracy_score(y_b, model.predict(X_b))
        accs.append(acc)
        print(f"random_state={seed}: accuracy = {acc:.3f}")
 
    print(f"\nRange across 5 different splits: {min(accs):.3f} to {max(accs):.3f} "
          f"(spread of {max(accs) - min(accs):.3f})")
    print("Same model, same data, same code — only the split changed. That spread is why\n"
          "a single accuracy number from one split shouldn't be trusted on its own.\n")
 
 
# ---------------------------------------------------------------------------
# 2-3. K-FOLD CROSS-VALIDATION
# ---------------------------------------------------------------------------
 
def run_cross_validation(X_train, y_train):
    print("=" * 70)
    print("2-3. 5-FOLD CROSS-VALIDATION (all three Day 16 models)")
    print("=" * 70)
 
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree (max_depth=4)": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=10, random_state=42
        ),
        "Random Forest (default)": RandomForestClassifier(n_estimators=200, random_state=42),
    }
 
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5)
        results[name] = scores
        print(f"{name}:")
        print(f"  fold scores: {scores.round(3)}")
        print(f"  mean = {scores.mean():.3f}  +/-  std = {scores.std():.3f}\n")
 
    return results
 
 
# ---------------------------------------------------------------------------
# 4-5. HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# ---------------------------------------------------------------------------
 
def run_grid_search(X_train, y_train):
    print("=" * 70)
    print("4-5. GRIDSEARCHCV (tuning the random forest)")
    print("=" * 70)
 
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 8, None],
        "min_samples_leaf": [1, 5],
    }
    print(f"Grid: {param_grid}")
    print(f"Total combinations: "
          f"{len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_leaf'])}, "
          f"each evaluated with 5-fold CV\n")
 
    forest = RandomForestClassifier(random_state=42)
    search = GridSearchCV(forest, param_grid, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
 
    print(f"Best params: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.3f}\n")
 
    return search.best_estimator_, search.best_score_
 
 
# ---------------------------------------------------------------------------
# 6-7. TUNED VS DEFAULT COMPARISON
# ---------------------------------------------------------------------------
 
def compare_tuned_vs_default(cv_results, tuned_cv_score, best_model, X_train, y_train, X_test, y_test):
    print("=" * 70)
    print("6-7. TUNED VS DEFAULT — reported honestly")
    print("=" * 70)
 
    default_forest_scores = cv_results["Random Forest (default)"]
    print(f"Default random forest CV: {default_forest_scores.mean():.3f} +/- {default_forest_scores.std():.3f}")
    print(f"Tuned random forest CV:   {tuned_cv_score:.3f}")
 
    diff = tuned_cv_score - default_forest_scores.mean()
    if abs(diff) < 0.01:
        print(f"Difference: {diff:+.3f} — essentially no improvement from tuning on this dataset.")
        print("That's a legitimate result, not a failure: it means the default settings were")
        print("already close to as good as this grid could find, given this much data.")
    else:
        print(f"Difference: {diff:+.3f} — tuning {'helped' if diff > 0 else 'hurt'} noticeably.")
 
    # final held-out test check — the one number that matters most, checked last and once
    best_model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, best_model.predict(X_test))
    print(f"\nFinal check on the held-out test set (touched for the first time just now): "
          f"{test_acc:.3f}")
    print("This is the only evaluation on X_test in this entire script — everything above")
    print("was decided using cross-validation on the training data alone.\n")
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    X_train, X_test, y_train, y_test = prepare_data()
    demonstrate_split_instability(X_train, y_train)
    cv_results = run_cross_validation(X_train, y_train)
    best_model, tuned_cv_score = run_grid_search(X_train, y_train)
    compare_tuned_vs_default(cv_results, tuned_cv_score, best_model, X_train, y_train, X_test, y_test)
 
    print("=" * 70)
    print("Day 17 complete. Cross-validated, tuned, and honestly evaluated.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
 