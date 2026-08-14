"""
Day 31: RandomizedSearchCV — Smarter Hyperparameter Search
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 5, Day 4
 
Covers:
  1. Why GridSearchCV (Day 17/21/23/25) stops scaling once the grid grows
  2. RandomizedSearchCV — sampling a fixed number of random combinations
     instead of exhaustively enumerating every one
  3. Parameter distributions (scipy.stats) — continuous ranges instead of
     discrete lists, letting the search explore values a hand-picked grid
     would never include
  4. A fair, same-budget comparison: GridSearchCV on a small grid vs
     RandomizedSearchCV on a much larger space, same number of model fits
  5. n_iter — the dial that trades search thoroughness for compute cost
  6. Visualizing which sampled hyperparameter region actually performed best
 
Requires: pip install scikit-learn scipy xgboost matplotlib
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
 
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
 
from xgboost import XGBClassifier
 
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
 
DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
 
 
# ---------------------------------------------------------------------------
# 0. LOAD + SPLIT (same as Day 22-30 — raw, no manual cleaning)
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
# 1. SAME PREPROCESSOR + XGBOOST PIPELINE FROM DAY 21-30
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
    print("=" * 70)
    print("1. BUILDING THE DAY 25 XGBOOST PIPELINE (unfitted — the search fits it)")
    print("=" * 70)
 
    engineer = FunctionTransformer(add_family_size)
    preprocessor = build_preprocessor()
 
    pipeline = Pipeline(steps=[
        ("engineer", engineer),
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(eval_metric="logloss", random_state=42)),
    ])
 
    print("Same engineer + preprocessor as Day 25-30 — today the classifier's hyperparameters")
    print("are what's being searched, not fixed by hand.\n")
    return pipeline

 # ---------------------------------------------------------------------------
# 2. WHY GRIDSEARCHCV STOPS SCALING
# ---------------------------------------------------------------------------
 
def show_grid_explosion():
    print("=" * 70)
    print("2. WHY GRIDSEARCHCV STOPS SCALING")
    print("=" * 70)

    full_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [0.5, 1, 2],
    }

    n_combinations = 1
    for values in full_grid.values():
        n_combinations *= len(values)
    cv_folds = 5
    total_fits = n_combinations * cv_folds

    
    print(f"A realistic grid across {len(full_grid)} hyperparameters, each with only "
          f"{sum(len(v) for v in full_grid.values()) // len(full_grid)}-4 values:")

    for name, values in full_grid.items():
        print(f" {name}: {values}")
    print(f"\nTotal combinations: {n_combinations}")
    print(f"With cv={cv_folds}: {total_fits} total model fits — GridSearchCV would need to")
    print("train every single one before returning an answer. Day 17/21/23/25's grids stayed")
    print("small (18-27 combos) specifically to avoid this; a genuinely thorough grid does not.\n")
 
    return full_grid, n_combinations, total_fits
 
 
# ---------------------------------------------------------------------------
# 3. FAIR, SAME-BUDGET COMPARISON: GRIDSEARCHCV VS RANDOMIZEDSEARCHCV
# ---------------------------------------------------------------------------
 
def run_grid_search(pipeline, X_train, y_train):
    print("=" * 70)
    print("3a. GRIDSEARCHCV — small grid, exhaustive")
    print("=" * 70)
 
    param_grid = {
        "classifier__max_depth": [2, 3, 4],
        "classifier__learning_rate": [0.05, 0.1, 0.2],
        "classifier__subsample": [0.7, 1.0],
    }
    n_combos = 3 * 3 * 2
    print(f"Grid: {param_grid}")
    print(f"{n_combos} combinations x cv=5 = {n_combos * 5} model fits, all of them, every time.\n")    

    start = time.time()
    search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    elapsed = time.time() - start
 
    print(f"Best params: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.3f}")
    print(f"Time: {elapsed:.1f}s, fits: {n_combos * 5}\n")
 
    return search, elapsed, n_combos * 5
 
 
def run_randomized_search(pipeline, X_train, y_train, n_iter=18):
    print("=" * 70)
    print("3b. RANDOMIZEDSEARCHCV — same fit budget, much larger space")
    print("=" * 70)
 
    param_distributions = {
        "classifier__n_estimators": randint(100, 400),
        "classifier__max_depth": randint(2, 8),
        "classifier__learning_rate": loguniform(0.01, 0.3),
        "classifier__subsample": uniform(0.5, 0.5),        # 0.5 to 1.0
        "classifier__colsample_bytree": uniform(0.5, 0.5),  # 0.5 to 1.0
        "classifier__reg_alpha": uniform(0.0, 2.0),
        "classifier__reg_lambda": uniform(0.5, 2.0),
    }
    print(f"Distributions across {len(param_distributions)} hyperparameters (continuous, not")
    print(f"fixed lists) — n_iter={n_iter} means only {n_iter} combinations get sampled and")
    print(f"tried, x cv=5 = {n_iter * 5} model fits: the SAME fit budget as 3a's grid, across a")
    print("space that would need thousands of combinations to cover exhaustively.\n")
 
    start = time.time()
    search = RandomizedSearchCV(
        pipeline, param_distributions, n_iter=n_iter, cv=5,
        n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    elapsed = time.time() - start
 
    # best_params_ holds numpy scalars (np.float64/np.int64) from the sampled
    # distributions — rounding them to plain Python floats makes the printed
    # output readable instead of showing "np.float64(0.0288...)" verbatim.
    readable_params = {
        key: (round(float(value), 3) if isinstance(value, (float, np.floating)) else value)
        for key, value in search.best_params_.items()
    }
    print(f"Best params: {readable_params}")
    print(f"Best CV score: {search.best_score_:.3f}")
    print(f"Time: {elapsed:.1f}s, fits: {n_iter * 5}\n")
 
    return search, elapsed, n_iter * 5
 
 
def compare_searches(grid_result, random_result):
    print("=" * 70)
    print("4. SAME-BUDGET COMPARISON")
    print("=" * 70)
 
    grid_search, grid_time, grid_fits = grid_result
    random_search, random_time, random_fits = random_result
 
    table = pd.DataFrame([
        {"search": "GridSearchCV", "space": "3 params, 18 fixed combos",
         "fits": grid_fits, "time_s": round(grid_time, 1), "best_cv_score": round(grid_search.best_score_, 3)},
        {"search": "RandomizedSearchCV", "space": "7 params, continuous",
         "fits": random_fits, "time_s": round(random_time, 1), "best_cv_score": round(random_search.best_score_, 3)},
    ])
    print(table.to_string(index=False))
    print("\nSame number of model fits, but RandomizedSearchCV searched a far larger,")
    print("continuous 7-parameter space instead of a fixed 3-parameter grid — the tradeoff")
    print("is that it might miss the single best point a fine grid would have hit exactly,")
    print("in exchange for covering far more of the space per fit spent.\n")
 
 
# ---------------------------------------------------------------------------
# 5. VISUALIZING THE SEARCH — which sampled region performed best
# ---------------------------------------------------------------------------
 
def plot_search_results(random_search, out_path="randomized_search_results.png"):
    print("=" * 70)
    print("5. VISUALIZING THE RANDOM SEARCH")
    print("=" * 70)
 
    results = pd.DataFrame(random_search.cv_results_)
    x = results["param_classifier__learning_rate"].astype(float)
    y = results["param_classifier__max_depth"].astype(float)
    scores = results["mean_test_score"]
 
    plt.figure(figsize=(6, 4.5))
    scatter = plt.scatter(x, y, c=scores, cmap="viridis", s=80, edgecolors="black")
    plt.colorbar(scatter, label="Mean CV accuracy")
    plt.xlabel("learning_rate (sampled)")
    plt.ylabel("max_depth (sampled)")
    plt.title("RandomizedSearchCV: sampled points, colored by CV score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
 
    print(f"Saved: {out_path}")
    best_idx = scores.idxmax()
    print(f"Best sampled point: learning_rate={x[best_idx]:.3f}, max_depth={y[best_idx]:.0f}, "
          f"score={scores[best_idx]:.3f}")
    print("Every dot is one of the n_iter randomly sampled combinations actually tried — the")
    print("plot shows which region of the space the search happened to land its best score in.\n")
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    X_train, X_test, y_train, y_test = load_and_split()
    pipeline = build_pipeline()
 
    show_grid_explosion()
 
    grid_result = run_grid_search(pipeline, X_train, y_train)
    random_result = run_randomized_search(pipeline, X_train, y_train, n_iter=18)
    compare_searches(grid_result, random_result)
 
    random_search = random_result[0]
    plot_search_results(random_search)
 
    print("=" * 70)
    print("Day 31 complete. Same compute budget, a far larger search space — the practical")
    print("answer to what happens once a hyperparameter grid grows past what GridSearchCV")
    print("can exhaustively cover in reasonable time.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()