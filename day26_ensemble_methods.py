"""
Day 26: Ensemble Methods — Voting and Stacking
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 4, Day 6
 
Covers:
  1. Why combine models trained on the same data (different error patterns)
  2. VotingClassifier — hard voting vs soft voting
  3. StackingClassifier — a meta-learner trained on base models' out-of-fold predictions
  4. Building both on top of the Day 25 base learners (RF, sklearn GB, XGBoost)
  5. Comparing individual models vs voting vs stacking on the same test set
  6. Why StackingClassifier uses cross-validation internally to avoid leakage
  7. Tuning the stacking meta-learner
 
Requires: pip install xgboost
Expects titanic.csv in the same folder. If not found, downloads a copy.
"""
 
import os
import urllib.request
 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier,
)
from sklearn.metrics import accuracy_score, classification_report
 
from xgboost import XGBClassifier
 
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
 
DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
 
 
# ---------------------------------------------------------------------------
# 0. LOAD + SPLIT (same as Day 22-25 — raw, no manual cleaning)
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
# 1. SAME PREPROCESSOR FROM DAY 21-25
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
# 2. THREE BASE LEARNERS, EACH WRAPPED AS A FULL PIPELINE
# ---------------------------------------------------------------------------
 
def build_base_pipelines(preprocessor):
    print("=" * 70)
    print("1-2. BUILDING THE THREE BASE LEARNERS FROM DAY 25")
    print("=" * 70)

    engineer = FunctionTransformer(add_family_size)
 
    forest_pipeline = Pipeline(steps=[
        ("engineer", engineer),
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
 
    gb_pipeline = Pipeline(steps=[
        ("engineer", engineer),
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )),
    ])
 
    xgb_pipeline = Pipeline(steps=[
        ("engineer", engineer),
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
        )),
    ])
 
    print("Three base learners, each a full pipeline: random forest, sklearn gradient")
    print("boosting, and XGBoost — the same three compared individually on Day 25.\n")
 
    return forest_pipeline, gb_pipeline, xgb_pipeline
 
 
# ---------------------------------------------------------------------------
# 3. VOTING CLASSIFIER — HARD VS SOFT
# ---------------------------------------------------------------------------
 
def build_voting_classifiers(forest_pipeline, gb_pipeline, xgb_pipeline):
    print("=" * 70)
    print("3. VOTING CLASSIFIER (HARD VS SOFT)")
    print("=" * 70)
 
    estimators = [
        ("forest", forest_pipeline),
        ("gb", gb_pipeline),
        ("xgb", xgb_pipeline),
    ]
 
    hard_voting = VotingClassifier(estimators=estimators, voting="hard")
    soft_voting = VotingClassifier(estimators=estimators, voting="soft")
 
    print("hard_voting: each model casts one vote (0 or 1), majority wins")
    print("soft_voting: each model's predicted probabilities are averaged, then thresholded")
    print("Soft voting needs every base estimator to support predict_proba — all three do.\n")
 
    return hard_voting, soft_voting
 
 
# ---------------------------------------------------------------------------
# 4. STACKING CLASSIFIER — A META-LEARNER ON TOP OF BASE PREDICTIONS
# ---------------------------------------------------------------------------
 
def build_stacking_classifier(forest_pipeline, gb_pipeline, xgb_pipeline):
    print("=" * 70)
    print("4. STACKING CLASSIFIER (META-LEARNER)")
    print("=" * 70)
 
    estimators = [
        ("forest", forest_pipeline),
        ("gb", gb_pipeline),
        ("xgb", xgb_pipeline),
    ]
 
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
    )
 
    print("final_estimator: a logistic regression trained on the three base models' outputs,")
    print("                 not on the raw features directly")
    print("cv=5:            StackingClassifier internally cross-validates each base learner and")
    print("                 trains the meta-learner on out-of-fold predictions — this stops the")
    print("                 meta-learner from 'cheating' by seeing predictions on rows the base")
    print("                 models were themselves trained on\n")
 
    return stacking
 
 
# ---------------------------------------------------------------------------
# 5. FIT + COMPARE EVERYTHING ON THE SAME TEST SET
# ---------------------------------------------------------------------------
 
def fit_and_compare(name, model, X_train, y_train, X_test, y_test):
    print("=" * 70)
    print(f"5. FIT + EVALUATE: {name}")
    print("=" * 70)
 
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
 
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=["died", "survived"]))
 
    return acc
 
 
# ---------------------------------------------------------------------------
# 6. TUNING THE STACKING META-LEARNER
# ---------------------------------------------------------------------------
 
def tune_stacking(stacking, X_train, y_train):
    print("=" * 70)
    print("6. TUNING THE STACKING META-LEARNER")
    print("=" * 70)
 
    param_grid = {
        "final_estimator__C": [0.1, 1.0, 10.0],
    }
    print(f"Grid: {param_grid}")
    print("Only the meta-learner's regularization strength is tuned here — the base learners")
    print("keep their Day 25 settings; tuning everything jointly would be far more expensive.\n")
 
    search = GridSearchCV(stacking, param_grid, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
 
    print(f"Best params: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.3f}\n")
 
    return search.best_estimator_, search.best_score_
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    X_train, X_test, y_train, y_test = load_and_split()
    preprocessor = build_preprocessor()
    forest_pipeline, gb_pipeline, xgb_pipeline = build_base_pipelines(preprocessor)
 
    results = {}
    results["Random Forest"] = fit_and_compare(
        "Random Forest (base)", forest_pipeline, X_train, y_train, X_test, y_test
    )
    results["Sklearn GB"] = fit_and_compare(
        "Sklearn Gradient Boosting (base)", gb_pipeline, X_train, y_train, X_test, y_test
    )
    results["XGBoost"] = fit_and_compare(
        "XGBoost (base)", xgb_pipeline, X_train, y_train, X_test, y_test
    )
 
    hard_voting, soft_voting = build_voting_classifiers(forest_pipeline, gb_pipeline, xgb_pipeline)
    results["Hard Voting"] = fit_and_compare(
        "Hard Voting", hard_voting, X_train, y_train, X_test, y_test
    )
    results["Soft Voting"] = fit_and_compare(
        "Soft Voting", soft_voting, X_train, y_train, X_test, y_test
    )
 
    stacking = build_stacking_classifier(forest_pipeline, gb_pipeline, xgb_pipeline)
    results["Stacking"] = fit_and_compare(
        "Stacking", stacking, X_train, y_train, X_test, y_test
    )
 
    best_stack, best_cv_score = tune_stacking(stacking, X_train, y_train)
 
    print("=" * 70)
    print("7. SUMMARY")
    print("=" * 70)
    table = pd.DataFrame(
        [{"model": k, "test_accuracy": v} for k, v in results.items()]
    ).sort_values("test_accuracy", ascending=False)
    print(table.round(3).to_string(index=False))
    print(f"\nTuned stacking CV score: {best_cv_score:.3f}")
    print("Ensembling rarely produces a dramatic jump when the base models already agree most")
    print("of the time — its value shows up more clearly when base models make genuinely")
    print("different kinds of errors on different rows.\n")
 
    print("=" * 70)
    print("Day 26 complete. Combined Day 25's three base learners with voting and stacking,")
    print("compared fairly against each base model alone.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
 