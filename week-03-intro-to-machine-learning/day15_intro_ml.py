
"""
Day 15: Intro to Machine Learning
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
Week 3, Day 1 — first supervised learning model
 
Covers:
  1. Features (X) vs target (y)
  2. Train/test split
  3. Encoding categorical variables
  4. Baseline model: Logistic Regression
  5. Evaluation: accuracy, confusion matrix, classification report
 
Expects titanic.csv in the same folder. If not found, downloads a copy.
"""

import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN + FEATURE ENGINEERING (same as Day 13)
# ---------------------------------------------------------------------------
def load_and_clean(path=DATA_PATH, url=DATA_URL):
    print("=" * 70)
    print("1. LOAD + CLEAN")
    print("=" * 70)
    
    if not os.path.exists(path):
        print(f"{path} not found locally - downloading...")
        urllib.request.urlretrieve(url, path)

    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df = df.drop(columns=["Cabin"])
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    
    print(f"Cleaned.shape: {df.shape}\n")
    return df

# ---------------------------------------------------------------------------
# 2. FEATURES (X) VS TARGET (y)
# ---------------------------------------------------------------------------

def build_features_and_target(df):
    print("=" * 70)
    print("2. FEATURE (X) VS TARGET (y)")
    print("=" * 70)

    y = df["Survived"]
    
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
                     "Embarked", "FamilySize"]
    X = df[feature_cols].copy()
    print(f"target (y): Survived - {y.sum()} survived out of {len(y)} "
        f"({y.mean():.1%})")
    print(f"features (X), before encoding: {list(X.columns)}")
    print("Dropped as non-predictive identifiers: passengerID, Name, Ticket\n")
    
    return X, y

# ---------------------------------------------------------------------------
# 3. ENCODING CATEGORICAL VARIABLES
# ---------------------------------------------------------------------------

def encode_features(X):
    print("=" * 70)
    print("3. ENCODING CATEGORICAL VARIABLES")
    print("=" * 70)

    X_encoded = pd.get_dummies(X, columns=["Sex", "Embarked"], drop_first=True)
    print(f"Before encoding: {X.shape[1]} columns")
    print(f"after encoding: {X_encoded.shape[1]} columns")
    print(f"new columns from encoding: "
          f"{[c for c in X_encoded.columns if c not in X.columns]}\n")

    return X_encoded

# ---------------------------------------------------------------------------
# 4. TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------
 
def split_data(X, y):
    print("=" * 70)
    print("4. TRAIN/TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"train: {X_train.shape[0]} rows ({X_train.shape[0] / len(X):.0%})")
    print(f"test: {X_test.shape[0]} rows ({X_test.shape[0] / len(X):.0%})")
    print(f"Train survival rate: {y_train.mean():.1%}")
    print(f"Test survival rate:  {y_test.mean():.1%}")
    print("(stratify=y keeps the survival ratio consistent across both splits)\n")

    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------------------------
# 5. BASELINE MODEL: LOGISTIC REGRESSION
# ---------------------------------------------------------------------------
 
def train_model(X_train, y_train):
    print("=" * 70)
    print("5. TRAINING: LOGISTIC REGRESSION")
    print("=" * 70)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Model trained on", X_train.shape[0], "passengers")

    coefs = pd.Series(model.coef_[0], index=X_train.columns).sort_values()
    print("\nLearned coefficients (negative = pushes toward 'died',")
    print("positive = pushed toward 'survived'):")
    print(coefs.round(3).to_string())
    print()

    return model

# ---------------------------------------------------------------------------
# 6. EVALUATION
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    print("=" * 70)
    print("6. EVALUATION")
    print("=" * 70)
    
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    
    print(f"Accuracy: {acc:.3f}  ({acc:.1%} of test passengers correctly classified)\n")
    
    cm = confusion_matrix(y_test, predictions)
    print("Confusion matrix:")
    print("                 Predicted: Died   Predicted: Survived")
    print(f"Actual: Died         {cm[0][0]:>4}                {cm[0][1]:>4}")
    print(f"Actual: Survived     {cm[1][0]:>4}                {cm[1][1]:>4}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["Died", "Survived"]))
 
    # baseline comparison: what if we just guessed "always died"?
    baseline_acc = 1 - y_test.mean()
    print(f"For comparison, always predicting 'died' would score "
          f"{baseline_acc:.3f} accuracy.")
    print(f"The model beats that naive baseline by {acc - baseline_acc:+.3f}.")
 
    return predictions
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    df = load_and_clean()
    X, y = build_features_and_target(df)
    X_encoded = encode_features(X)
    X_train, X_test, y_train, y_test = split_data(X_encoded, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
 
    print("\n" + "=" * 70)
    print("Day 15 complete. First trained model, evaluated on unseen data.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
    