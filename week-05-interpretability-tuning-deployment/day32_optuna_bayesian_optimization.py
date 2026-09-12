"""
Day 32: Bayesian Optimization with Optuna
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 5, Day 5
 
Covers:
  1. Why RandomizedSearchCV's independent sampling leaves information on
     the table (Day 31)
  2. Optuna's TPE sampler — using every completed trial's result to choose
     the NEXT combination to try, instead of sampling blindly
  3. Defining a search space with trial.suggest_* instead of scipy
     distributions
  4. A three-way, same-budget comparison: GridSearchCV vs RandomizedSearchCV
     vs Optuna, all at 90 model fits
  5. Optimization history — watching the best score improve trial by trial,
     something random search's plot (Day 31) never showed
  6. Parameter importance — which hyperparameters Optuna's surrogate model
     found actually moved the score
 
Requires: pip install optuna scikit-learn scipy xgboost matplotlib
Expects titanic.csv in the same folder. If not found, downloads a copy.
"""
 
import os
import time
import urllib.request
 
import matplotlib
matplotlib.use("Agg")  # headless-safe backend — saves files without opening a window
import matplotlib.pyplot as plt
 
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
 
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
 
from xgboost import XGBClassifier
 
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress the default per-trial log spam
 
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
 
DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
 
 
# ---------------------------------------------------------------------------
# 0. LOAD + SPLIT (same as Day 22-31 — raw, no manual cleaning)
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
# 1. SAME PREPROCESSOR + XGBOOST PIPELINE FROM DAY 21-31
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
 
 
def build_pipeline():
    engineer = FunctionTransformer(add_family_size)
    preprocessor = build_preprocessor()
    return Pipeline(steps=[
        ("engineer", engineer),
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(eval_metric="logloss", random_state=42)),
    ])
 

# ---------------------------------------------------------------------------
# 2. WHY RANDOM SEARCH LEAVES INFORMATION ON THE TABLE
# ---------------------------------------------------------------------------
 
def explain_random_vs_bayesian():
    print("=" * 70)
    print("2. RANDOM SEARCH (DAY 31) VS BAYESIAN OPTIMIZATION (TODAY)")
    print("=" * 70)
    print("RandomizedSearchCV samples every combination independently — trial 15 knows nothing")
    print("about what happened in trials 1-14, even if 5 of them already scored badly in the")
    print("same corner of the space. Optuna's default sampler (TPE — Tree-structured Parzen")
    print("Estimator) builds a running model of which regions tend to score well, and biases")
    print("each new suggestion toward those regions — using every trial's result to inform the")
    print("next one, instead of sampling blind every time.\n")
 
 
# ---------------------------------------------------------------------------
# 3a. GRIDSEARCHCV — same small grid as Day 31 (for the 3-way comparison)
# ---------------------------------------------------------------------------
 
def run_grid_search(pipeline, X_train, y_train):
    print("=" * 70)
    print("3a. GRIDSEARCHCV (same grid as Day 31)")
    print("=" * 70)
 
    param_grid = {
        "classifier__max_depth": [2, 3, 4],
        "classifier__learning_rate": [0.05, 0.1, 0.2],
        "classifier__subsample": [0.7, 1.0],
    }
    start = time.time()
    search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    elapsed = time.time() - start
 
    print(f"Best CV score: {search.best_score_:.3f}  |  Time: {elapsed:.1f}s  |  Fits: 90\n")
    return search.best_score_, elapsed
 
 
# ---------------------------------------------------------------------------
# 3b. RANDOMIZEDSEARCHCV — same distributions and n_iter as Day 31
# ---------------------------------------------------------------------------
 
def run_randomized_search(pipeline, X_train, y_train, n_iter=18):
    print("=" * 70)
    print("3b. RANDOMIZEDSEARCHCV (same distributions as Day 31)")
    print("=" * 70)
 
    param_distributions = {
        "classifier__n_estimators": randint(100, 400),
        "classifier__max_depth": randint(2, 8),
        "classifier__learning_rate": loguniform(0.01, 0.3),
        "classifier__subsample": uniform(0.5, 0.5),
        "classifier__colsample_bytree": uniform(0.5, 0.5),
        "classifier__reg_alpha": uniform(0.0, 2.0),
        "classifier__reg_lambda": uniform(0.5, 2.0),
    }
    start = time.time()
    search = RandomizedSearchCV(
        pipeline, param_distributions, n_iter=n_iter, cv=5, n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    elapsed = time.time() - start
 
    print(f"Best CV score: {search.best_score_:.3f}  |  Time: {elapsed:.1f}s  |  Fits: {n_iter * 5}\n")
    return search.best_score_, elapsed
 
 
# ---------------------------------------------------------------------------
# 3c. OPTUNA — same search space, TPE sampler, same trial budget
# ---------------------------------------------------------------------------
 
def build_objective(X_train, y_train):
    def objective(trial):
        params = {
            "classifier__n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "classifier__max_depth": trial.suggest_int("max_depth", 2, 8),
            "classifier__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "classifier__subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "classifier__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "classifier__reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "classifier__reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.5),
        }
        pipeline = build_pipeline()
        pipeline.set_params(**params)
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1)
        return scores.mean()
    return objective
 
 
def run_optuna_search(X_train, y_train, n_trials=18):
    print("=" * 70)
    print("3c. OPTUNA (TPE sampler, same search space as Day 31's distributions)")
    print("=" * 70)
 
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
 
    start = time.time()
    study.optimize(build_objective(X_train, y_train), n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - start
 
    readable_params = {k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in study.best_params.items()}
    print(f"Best params: {readable_params}")
    print(f"Best CV score: {study.best_value:.3f}  |  Time: {elapsed:.1f}s  |  "
          f"Fits: {n_trials * 5}\n")
 
    return study, study.best_value, elapsed
 
 
# ---------------------------------------------------------------------------
# 4. THREE-WAY, SAME-BUDGET COMPARISON
# ---------------------------------------------------------------------------
 
def compare_three_way(grid_result, random_result, optuna_result):
    print("=" * 70)
    print("4. THREE-WAY SAME-BUDGET COMPARISON (90 fits each)")
    print("=" * 70)
 
    grid_score, grid_time = grid_result
    random_score, random_time = random_result
    _, optuna_score, optuna_time = optuna_result
 
    table = pd.DataFrame([
        {"method": "GridSearchCV", "strategy": "exhaustive, 3 params",
         "time_s": round(grid_time, 1), "best_cv_score": round(grid_score, 3)},
        {"method": "RandomizedSearchCV", "strategy": "independent random, 7 params",
         "time_s": round(random_time, 1), "best_cv_score": round(random_score, 3)},
        {"method": "Optuna (TPE)", "strategy": "adaptive Bayesian, 7 params",
         "time_s": round(optuna_time, 1), "best_cv_score": round(optuna_score, 3)},
    ])
    print(table.to_string(index=False))
    print("\nAll three trained the same number of models (90). Optuna's TPE sampler used every")
    print("completed trial's score to bias where it looked next, instead of sampling every point")
    print("independently — the comparison isolates that one difference in search strategy.\n")
 
 
# ---------------------------------------------------------------------------
# 5. OPTIMIZATION HISTORY — best score improving trial by trial
# ---------------------------------------------------------------------------
 
def plot_optimization_history(study, out_path="optuna_optimization_history.png"):
    print("=" * 70)
    print("5. OPTIMIZATION HISTORY")
    print("=" * 70)
 
    trial_scores = [t.value for t in study.trials]
    best_so_far = np.maximum.accumulate(trial_scores)
 
    plt.figure(figsize=(6, 4.5))
    plt.scatter(range(1, len(trial_scores) + 1), trial_scores,
                color="#7A4EC9", alpha=0.6, label="Each trial's score")
    plt.plot(range(1, len(best_so_far) + 1), best_so_far,
              color="black", linewidth=2, label="Best score so far")
    plt.xlabel("Trial number")
    plt.ylabel("Mean CV accuracy")
    plt.title("Optuna: best score so far, trial by trial")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
 
    print(f"Saved: {out_path}")
    print(f"Trial scores ranged {min(trial_scores):.3f} to {max(trial_scores):.3f}; the running")
    print("best-so-far line should trend upward and flatten as later trials, informed by earlier")
    print("ones, cluster closer to the best region found — unlike Day 31's random search, where")
    print("there's no reason for later trials to outperform earlier ones on average.\n")
 
 
# ---------------------------------------------------------------------------
# 6. PARAMETER IMPORTANCE — which hyperparameters actually moved the score
# ---------------------------------------------------------------------------
 
def plot_param_importance(study, out_path="optuna_param_importance.png"):
    print("=" * 70)
    print("6. PARAMETER IMPORTANCE")
    print("=" * 70)
 
    importances = optuna.importance.get_param_importances(study)
    names = list(importances.keys())
    values = list(importances.values())
 
    plt.figure(figsize=(6, 4))
    plt.barh(names[::-1], values[::-1], color="#7A4EC9")
    plt.xlabel("Importance (fraction of variance explained)")
    plt.title("Optuna: which hyperparameters mattered most")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
 
    print(f"Saved: {out_path}")
    top_param = names[0]
    print(f"Most important hyperparameter: {top_param} ({importances[top_param]:.1%})")
    print("Computed from a surrogate model fit to all completed trials — an entirely different")
    print("kind of importance than Day 23/25's gain-based feature importance, since this ranks")
    print("HYPERPARAMETERS by their effect on CV score, not FEATURES by their effect on a")
    print("prediction.\n")
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    X_train, X_test, y_train, y_test = load_and_split()
    pipeline = build_pipeline()
 
    explain_random_vs_bayesian()
 
    grid_result = run_grid_search(pipeline, X_train, y_train)
    random_result = run_randomized_search(pipeline, X_train, y_train, n_iter=18)
    study, optuna_score, optuna_time = run_optuna_search(X_train, y_train, n_trials=18)
 
    compare_three_way(grid_result, random_result, (study, optuna_score, optuna_time))
    plot_optimization_history(study)
    plot_param_importance(study)
 
    print("=" * 70)
    print("Day 32 complete. Same fit budget as Day 31, one more search strategy added — an")
    print("optimizer that learns from its own trials instead of sampling blind.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
 