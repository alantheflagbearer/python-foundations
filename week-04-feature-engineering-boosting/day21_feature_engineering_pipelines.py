"""
Day 21: Feature Engineering at Scale — Pipeline and ColumnTransformer
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 4, Day 1
 
Covers:
  1. Why manual pd.get_dummies() doesn't scale to production
  2. StandardScaler for numeric features
  3. OneHotEncoder (stateful, production-safe categorical encoding)
  4. ColumnTransformer — different preprocessing per column type
  5. Pipeline — chaining preprocessing + model into one object
  6. Why this prevents data leakage
  7. GridSearchCV over a full pipeline (preprocessing + model together)
 
Expects titanic.csv in the same folder. If not found, downloads a copy.
"""

import os
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
 
DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# ---------------------------------------------------------------------------
# 0. LOAD + MINIMAL CLEAN (no manual encoding this time — the pipeline does it)
# ---------------------------------------------------------------------------
 
def load_and_clean(path=DATA_PATH, url=DATA_URL):
    print("=" * 70)
    print("0. LOAD + MINIMAL CLEAN")
    print("=" * 70)

    df = pd.read_csv(path)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df = df.drop(columns=["Cabin"])
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    print (f"loaded {df.shape[0]} passenger. ")
    print("Note: Sex/Embarked are left as raw text columns on purpose - no")
    print("pd.get_dummies() here. The pipeline itself will encode them.\n")
 
    return df

# ---------------------------------------------------------------------------
# 1. SPLIT (before any scaling/encoding is fit — this is what prevents leakage)
# ---------------------------------------------------------------------------
 
def split_data(df):
    print("=" * 70)
    print("1. TRAIN/TEST SPLIT")
    print("=" * 70)

    numeric_cols = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
    categorical_cols = ["Pclass", "Sex", "Embarked"]
    feature_cols = numeric_cols + categorical_cols

    X = df[feature_cols]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Numeric columns:     {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
    print("Nothing has been scaled or encoded yet — X_train/X_test are still raw.\n")
 
    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols

# ---------------------------------------------------------------------------
# 2. BUILD THE COLUMNTRANSFORMER
# ---------------------------------------------------------------------------

def build_preprocessor(numeric_cols, categorical_cols):
    print("=" * 70)
    print("2. BUILDING THE COLUMNTRANSFORMER")
    print("=" * 70)

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
    ])

    print("ColumnTransformer built:")
    print(f" 'num' -> StandardScaler()     applied to {numeric_cols}")

    print(f"  'cat' -> OneHotEncoder(...)         applied to {categorical_cols}")
    print("Not fit yet — this object only describes what to do, not the result.\n")
 
    return preprocessor

# ---------------------------------------------------------------------------
# 3. BUILD PIPELINES (preprocessing + model, two model choices to compare)
# ---------------------------------------------------------------------------
 
def build_pipelines(preprocessor):
    print("=" * 70)
    print("3. BUILDING PIPELINES (preprocessing + model as one object)")
    print("=" * 70)

    logreg_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    forest_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])

    print("Two pipelines built, each with its own classifier step:")
    print("  logreg_pipeline: preprocessor -> LogisticRegression")
    print("  forest_pipeline: preprocessor -> RandomForestClassifier")
    print("Both share the exact same preprocessing definition, applied consistently.\n")
 
    return logreg_pipeline, forest_pipeline

# ---------------------------------------------------------------------------
# 4. FIT + EVALUATE (one .fit() call does preprocessing AND training)
# ---------------------------------------------------------------------------
 
def fit_and_evaluate(name, pipeline, X_train, y_train, X_test, y_test):
    print("=" * 70)
    print(f"4. FIT + EVALUATE: {name}")
    print("=" * 70)

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=["died", "survived"]))

    return acc

# ---------------------------------------------------------------------------
# 5. GRIDSEARCHCV OVER A FULL PIPELINE
# ---------------------------------------------------------------------------
 
def tune_pipeline(forest_pipeline, X_train, y_train):
    print("=" * 70)
    print("5. GRIDSEARCHCV OVER THE FULL PIPELINE")
    print("=" * 70)

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [4, 8, None],
    }

    search = GridSearchCV(forest_pipeline, param_grid, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.3f}")
    print("This search tuned the classifier while the preprocessing steps stayed")
    print("fixed and identical across every fold — no separate scaling step to")
    print("accidentally leave out or apply inconsistently.\n")
 
    return search.best_estimator_, search.best_score_


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    df = load_and_clean()
    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = split_data(df)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    logreg_pipeline, forest_pipeline = build_pipelines(preprocessor)
 
    logreg_acc = fit_and_evaluate("Logistic Regression pipeline", logreg_pipeline,
                                   X_train, y_train, X_test, y_test)
    forest_acc = fit_and_evaluate("Random Forest pipeline", forest_pipeline,
                                   X_train, y_train, X_test, y_test)
 
    best_model, best_cv_score = tune_pipeline(forest_pipeline, X_train, y_train)
 
    print("=" * 70)
    print("6. SUMMARY")
    print("=" * 70)
    print(f"Logistic Regression pipeline test accuracy: {logreg_acc:.3f}")
    print(f"Random Forest pipeline test accuracy:        {forest_acc:.3f}")
    print(f"Tuned Random Forest pipeline CV score:       {best_cv_score:.3f}")
 
    print("\n" + "=" * 70)
    print("Day 21 complete. Preprocessing and modeling now travel as one object.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
 
 