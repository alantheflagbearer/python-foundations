"""
Day 22: Imputation and Custom Features Inside the Pipeline
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 4, Day 2
 
Covers:
  1. Why manual fillna() before the split is still a leakage risk
  2. SimpleImputer — a fittable, train-only fillna()
  3. Nesting a Pipeline inside a ColumnTransformer branch (impute -> scale)
  4. The same pattern for categorical columns (impute -> encode)
  5. FunctionTransformer — custom feature engineering as a pipeline step
  6. Assembling the full nested pipeline
  7. GridSearchCV over a nested pipeline (deeper double-underscore chains)
 
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
 
DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# ---------------------------------------------------------------------------
# 0. LOAD ONLY — no cleaning here. Missing values stay missing on purpose.
# ---------------------------------------------------------------------------

def load_raw(path=DATA_PATH, url=DATA_URL):
    print("=" * 70)
    print("0. LOAD (no cleaning — missing values are left as NaN on purpose)")
    print("=" * 70)

    if not os.path.exists(path):
        print(f"{path} not found locally — downloading...")
        urllib.request.urlretrieve(url, path)
 
    df = pd.read_csv(path)
    missing = df[["Age", "Embarked"]].isna().sum()
    print(f"Loaded {df.shape[0]} passengers.")
    print(f"Missing values BEFORE any split or imputation:\n{missing}")
    print("These will be filled inside the pipeline, fit on X_train only.\n")
 
    return df
# ---------------------------------------------------------------------------
# 1. SPLIT (still on raw, unimputed, unencoded data)
# ---------------------------------------------------------------------------
 
def split_data(df):
    print("=" * 70)
    print("1. TRAIN/TEST SPLIT")
    print("=" * 70)

    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[feature_cols]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
    print(f"Missing Age in X_train: {X_train['Age'].isna().sum()}, in X_test: {X_test['Age'].isna().sum()}")
    print("Both still have missing values — nothing has been imputed yet.\n")
 
    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------------------------
# 2. FUNCTIONTRANSFORMER — custom feature engineering as a pipeline step
# ---------------------------------------------------------------------------
 
def add_family_size(X):
    """Custom feature engineering function, wrapped by FunctionTransformer below."""
    X = X.copy()
    X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
    return X

def build_feature_engineer():
    print("=" * 70)
    print("2. FUNCTIONTRANSFORMER (custom feature engineering)")
    print("=" * 70)
 
    engineer = FunctionTransformer(add_family_size)
    print("FunctionTransformer built around add_family_size().")
    print("This wraps a plain function so it behaves like any other pipeline step —")
    print("it will run automatically every time the pipeline is fit or used to predict.\n")
 
    return engineer

 
# ---------------------------------------------------------------------------
# 3-4. NESTED SUB-PIPELINES FOR NUMERIC AND CATEGORICAL COLUMNS
# ---------------------------------------------------------------------------
 
def build_preprocessor():
    print("=" * 70)
    print("3-4. NESTED SUB-PIPELINES INSIDE THE COLUMNTRANSFORMER")
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
 
    print("numeric_pipeline:     SimpleImputer(median) -> StandardScaler")
    print("categorical_pipeline: SimpleImputer(most_frequent) -> OneHotEncoder")
    print("Both wrapped into one ColumnTransformer named 'preprocessor'.\n")
 
    return preprocessor
 
 
# ---------------------------------------------------------------------------
# 5. ASSEMBLE THE FULL NESTED PIPELINE
# ---------------------------------------------------------------------------
 
def build_full_pipeline(engineer, preprocessor):
    print("=" * 70)
    print("5. ASSEMBLING THE FULL PIPELINE")
    print("=" * 70)
 
    full_pipeline = Pipeline(steps=[
        ("engineer", engineer),
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
 
    print("Full pipeline: engineer -> preprocessor -> classifier")
    print("One object. One .fit(). One .predict(). Every learned statistic (medians,")
    print("modes, encoder vocabulary) will come from X_train alone.\n")
 
    return full_pipeline
 
 
# ---------------------------------------------------------------------------
# 6. FIT + EVALUATE
# ---------------------------------------------------------------------------
 
def fit_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    print("=" * 70)
    print("6. FIT + EVALUATE")
    print("=" * 70)
 
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
 
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=["died", "survived"]))
    print("Note: X_test still had missing Age values right up until this call —")
    print("the pipeline imputed them internally using X_train's median, automatically.\n")
 
    return acc
 
 
# ---------------------------------------------------------------------------
# 7. GRIDSEARCHCV OVER THE NESTED PIPELINE
# ---------------------------------------------------------------------------
 
def tune_pipeline(full_pipeline, X_train, y_train):
    print("=" * 70)
    print("7. GRIDSEARCHCV OVER A NESTED PIPELINE")
    print("=" * 70)
 
    param_grid = {
        "preprocessor__num__imputer__strategy": ["median", "mean"],
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [4, 8, None],
    }
    print("Grid (note the nested double-underscore chain reaching 3 levels deep):")
    print(f"  {param_grid}\n")
 
    search = GridSearchCV(full_pipeline, param_grid, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
 
    print(f"Best params: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.3f}")
    print("Every candidate in this grid was evaluated as a full pipeline, including")
    print("imputation strategy — not just the classifier's hyperparameters.\n")
 
    return search.best_estimator_, search.best_score_
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    df = load_raw()
    X_train, X_test, y_train, y_test = split_data(df)
    engineer = build_feature_engineer()
    preprocessor = build_preprocessor()
    full_pipeline = build_full_pipeline(engineer, preprocessor)
 
    acc = fit_and_evaluate(full_pipeline, X_train, y_train, X_test, y_test)
    best_model, best_cv_score = tune_pipeline(full_pipeline, X_train, y_train)
 
    print("=" * 70)
    print("8. SUMMARY")
    print("=" * 70)
    print(f"Default pipeline test accuracy: {acc:.3f}")
    print(f"Tuned pipeline CV score:        {best_cv_score:.3f}")
 
    print("\n" + "=" * 70)
    print("Day 22 complete. Imputation, feature engineering, encoding, and the model")
    print("all now live inside one leak-proof, tunable object.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()