"""
Day 23: Gradient Boosting
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 4, Day 3
 
Covers:
  1. Bagging (Day 16's random forest) vs boosting (sequential error correction)
  2. GradientBoostingClassifier, dropped into the Day 21-22 pipeline
  3. learning_rate vs n_estimators tradeoff
  4. Comparing gradient boosting against random forest, same preprocessing, same test set
  5. Feature importances from boosting
  6. Tuning learning_rate/n_estimators/max_depth together with GridSearchCV
 
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
 
DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
 
 
# ---------------------------------------------------------------------------
# 0. LOAD + SPLIT (same as Day 22 — raw, no manual cleaning)
# ---------------------------------------------------------------------------
 
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
 
DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
 
 
# ---------------------------------------------------------------------------
# 0. LOAD + SPLIT (same as Day 22 — raw, no manual cleaning)
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
# 1. SAME PREPROCESSOR FROM DAY 21-22 (engineer + impute/scale/encode)
# ---------------------------------------------------------------------------
 
def add_family_size(X):
    X = X.copy()
    X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
    return X
 
 
def build_preprocessor():
    print("=" * 70)
    print("1. REUSING THE DAY 21-22 PREPROCESSOR")
    print("=" * 70)
 
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
 
    print("Identical engineer + preprocessor as Day 22 — only the classifier changes today.\n")
    return preprocessor


# ---------------------------------------------------------------------------
# 2. BAGGING VS BOOSTING — build both pipelines, same preprocessing
# ---------------------------------------------------------------------------
 
def build_pipelines(preprocessor):
    print("=" * 70)
    print("2. BAGGING (RANDOM FOREST) VS BOOSTING (GRADIENT BOOSTING)")
    print("=" * 70)
 
    engineer = FunctionTransformer(add_family_size)
 
    forest_pipeline = Pipeline(steps=[
        ("engineer", engineer),
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
 
    boosting_pipeline = Pipeline(steps=[
        ("engineer", engineer),
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )),
    ])
 
    print("forest_pipeline:   trees trained in parallel, votes averaged (bagging)")
    print("boosting_pipeline: trees trained in sequence, each correcting the last (boosting)")
    print("Both share the exact same engineer + preprocessor — only the classifier differs.\n")
 
    return forest_pipeline, boosting_pipeline
 
 
# ---------------------------------------------------------------------------
# 3. FIT + COMPARE ON THE SAME TEST SET
# ---------------------------------------------------------------------------
 
def fit_and_compare(name, pipeline, X_train, y_train, X_test, y_test):
    print("=" * 70)
    print(f"3. FIT + EVALUATE: {name}")
    print("=" * 70)
 
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
 
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=["died", "survived"]))
 
    return acc
 
 
# ---------------------------------------------------------------------------
# 4. FEATURE IMPORTANCE FROM BOOSTING
# ---------------------------------------------------------------------------
 
def show_feature_importance(boosting_pipeline):
    print("=" * 70)
    print("4. FEATURE IMPORTANCE (gradient boosting)")
    print("=" * 70)
 
    classifier = boosting_pipeline.named_steps["classifier"]
    preprocessor = boosting_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
 
    importances = pd.Series(
        classifier.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
 
    print(importances.round(3).to_string())
    print(f"\nMost important feature: {importances.index[0]} ({importances.iloc[0]:.1%})\n")
 
 
# ---------------------------------------------------------------------------
# 5. TUNING learning_rate / n_estimators / max_depth TOGETHER
# ---------------------------------------------------------------------------
 
def tune_boosting(boosting_pipeline, X_train, y_train):
    print("=" * 70)
    print("5. TUNING GRADIENT BOOSTING (learning_rate + n_estimators + max_depth)")
    print("=" * 70)
 
    param_grid = {
        "classifier__learning_rate": [0.01, 0.1, 0.3],
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [2, 3, 4],
    }
    print(f"Grid: {param_grid}")
    print("Tuned together, not independently — learning_rate and n_estimators trade off.\n")
 
    search = GridSearchCV(boosting_pipeline, param_grid, cv=5, n_jobs=-1)
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
    forest_pipeline, boosting_pipeline = build_pipelines(preprocessor)
 
    forest_acc = fit_and_compare("Random Forest (bagging)", forest_pipeline,
                                  X_train, y_train, X_test, y_test)
    boosting_acc = fit_and_compare("Gradient Boosting (boosting)", boosting_pipeline,
                                    X_train, y_train, X_test, y_test)
 
    show_feature_importance(boosting_pipeline)
    best_model, best_cv_score = tune_boosting(boosting_pipeline, X_train, y_train)
 
    print("=" * 70)
    print("6. SUMMARY")
    print("=" * 70)
    print(f"Random Forest test accuracy:      {forest_acc:.3f}")
    print(f"Gradient Boosting test accuracy:  {boosting_acc:.3f}")
    print(f"Tuned Gradient Boosting CV score: {best_cv_score:.3f}")
    print("Note: on a small test set, differences this close can shift with a different")
    print("random_state — this is exactly the same caveat from Day 16 and Day 17.\n")
 
    print("=" * 70)
    print("Day 23 complete. Bagging and boosting compared, fairly, on identical preprocessing.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()