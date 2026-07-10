# Load the dataset and check its shape
# First 5 things every data scientist does with a new dataset

# day8_titanic.py
# Week 2 Day 8 — Md Mutasim Billah
# Topic: Titanic dataset — load, explore, clean

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── LOAD ─────────────────────────────────────
df = pd.read_csv("titanic.csv")

# first 5 things you always do
print("shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nBasic stats:")
print(df.describe())

# ── EXPLORE ──────────────────────────────────
print("\n--- info() ---")
df.info()

print("\n--- survival counts ---")
print(df["Survived"].value_counts())
print(f"Survival rate: {df['Survived'].mean():.1%}")

print("\n--- passenger class ---")
print(df["Pclass"].value_counts().sort_index())

print("\n--- gender split ---")
print(df["Sex"].value_counts())

print("\n--- age stats ---")
print(f"min age : {df['Age'].min()}")
print(f"max age : {df['Age'].max()}")
print(f"mean age : {df['Age'].mean():.1f}")

print("\n--- embarked ports ---")
print(df["Embarked"].value_counts())

print("\n--- fare stats ---")
print(f"min fare : {df['Fare'].min():.2f}")
print(f"max fare : {df['Fare'].max():.2f}")
print(f"mean fare : {df['Fare'].mean():.2f}")

# ── MISSING VALUES ───────────────────────────
print("\n--- missing value counts ---")
missing = df.isnull().sum()
print(missing[missing > 0]) # only show columns with missing

print("\n--- missing value percentages ---")
missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
print(missing_pct[missing_pct > 0])

print("\n--- what to do with each ---")
print("Age : 20% missing → fill with median age")
print("Cabin : 77% missing → too much, drop this column")
print("Embarked : 2 missing → fill with most common (s)")

# ── CLEAN ────────────────────────────────────
# 1. Fill Age with median

median_age = df["Age"].median()
df["Age"] = df["Age"].fillna(median_age)
print(f"Filled Age missing values with median: {median_age}")

# 2. Drop Cabin — too many missing
df = df.drop(columns=['Cabin'])
print("Dropped cabin column")

# 3. Fill Embarked with most common value (mode)
most_common = df["Embarked"].mode()[0]
df["Embarked"]=df["Embarked"].fillna(most_common)
print(f"Filled Embarked with mode: {most_common}")

#confirm no missing values remain
print("\nmissing values after cleaning:")
print(f"{df.isnull().sum().sum()} total missing values")
print(f"new shape: {df.shape}")


# ── FEATURE ENGINEERING ──────────────────────
# family size: siblings + parents + self
df["FamilySize"]=df["SibSp"] + df["Parch"] + 1

#travelling alone
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

#age group
def age_group(age):
    if age < 12: return "child"
    if age < 18: return "teen"
    if age < 60: return "adult"
    else:        return "senior"

df["AgeGroup"] = df["Age"].apply(age_group)

# title from name (Mr, Mrs, Miss, Master etc)
df["Title"] = df["Name"].str.extract(r"([A-Za-z]+)\.")

print("New column added:")
print(df[["Name", "FamilySize", "IsAlone",
           "AgeGroup", "Title"]].head(8))

print("\nTitle count:")
print(df["Title"].value_counts())

# ── ANSWER QUESTIONS ─────────────────────────
print("\n=== Q1: Did woman survive more than man? ===")
survival_by_sex = df.groupby("Sex")["Survived"].mean()
print(survival_by_sex)
print(f"Women: {survival_by_sex['female']:.1%}")
print(f"Men: {survival_by_sex['male']:.1%}")

print("\n=== Q2: Did class affect survival? ===")
survival_by_class = df.groupby("Pclass")["Survived"].mean()
print(survival_by_class)

print("\n=== Q3: Did children survive more? ===")
survival_by_age = df.groupby("AgeGroup")["Survived"].mean()
print(survival_by_age)

print("\n=== Q4: Did family size matter? ===")
survival_by_family = df.groupby("IsAlone")["Survived"].mean()
print("Travelling alone:", f"{survival_by_family[1]:.1%}")
print("traveling with family: ", f"{survival_by_family[0]:.1%}")

print("\n=== Q5: Average fare by class ===")
fare_by_class = df.groupby("Pclass")["Fare"].mean()
for cls, fare in fare_by_class.items():
    print(f"class {cls}: ${fare:.2f}")

print("\n=== Summary ===")
print(f"Total passengers : {len(df)}")
print(f"Survived         : {df['Survived'].sum()}")
print(f"Overall rate     : {df['Survived'].mean():.1%}")
print(f"Columns now      : {df.shape[1]}")