# Week 1 Day 6 — Md Mutasim Billah
# Topic: Pandas DataFrames — the core data science tool
import numpy as np
import pandas as pd   # pd is the universal alias — always use this

print("Pandas version:", pd.__version__)
print("Pandas is working!")
print("*" * 40)

# ── CREATE DATAFRAME FROM pfds12 DATA ────────
data = {
    "name" : ["Mutasim", "Ali", "Sara", "John", "Priya"],
    "grade": [88, 45, 95, 38, 77],
    "passed": [True, False, True, False, True],
    "city" : ["Brooklyn", "Queens", "Brooklyn", "Manhattan", "Queens"]
}

df = pd.DataFrame(data)
print(df)
print("\nshape:", df.shape) # (5, 4) — 5 rows, 4 columns
print("columns:", df.columns.tolist())
print("index:", df.index.tolist())
print("*" * 40)

'''Every data science project starts with exploration
Before you clean, filter, or model data — you need to
 understand it. These 5 methods are the first thing every data
scientist runs on any new dataset. Learn them in order.'''

# ── EXPLORING THE DATAFRAME ──────────────────
print("\n--- head() ---")
print(df.head()) # first 5 rows (default)
print(df.head(3)) #first 3 rows
print("*" * 40)

print("\n--- tail() ---")
print(df.tail(2)) #last 2 rows
print("*" * 40)

print("\n--- info() ---")
print(df.info())   # column types, null counts, memory
print("*" * 40)

print("\n --- describe() ---")
print(df.describe()) # stats for numeric columns only
print("*" * 40)

print("\n--- value_count() ---")
print(df["city"].value_counts())   # count per unique city
print(df["passed"].value_counts()) # count True/False
print("*" * 40)
'''Selecting columns and rows
Get exactly the data you need'''

# ── SELECTING COLUMNS AND ROWS ───────────────
# select one column - series

grades = df["grade"]
print(type(grades))   # pandas.Series
print(grades)
print("*" * 40)

# select multiple columns → DataFrame
subset = df[["name", "grade"]]
print(subset)
print("*" * 40)

# select rows by label: loc
print(df.loc[0])              # row 0 — Mutasim
print(df.loc[1:3])            # rows 1 to 3 inclusive
print("*" * 40)

# select rows by position: iloc
print(df.iloc[0])                 # first row
print(df.iloc[-1])                # last row
print(df.iloc[0:3, 0:2])          # rows 0-2, cols 0-1
print("*" * 40)

'''Step 5 of 9 — Day 6
Filtering rows — boolean masking in pandas
The same concept from NumPy Day 5 — now on labeled data'''

# ── FILTERING ROWS ───────────────────────────

#student who passed
passed_df = df[df["passed"] == True]
print("passed students: \n", passed_df)
print("*" * 40)

#grades above 80
high = df[df["grade"] > 80]
print("\ngrade above 80:\n", high)
print("*" * 40)

# from Brooklyn only
brooklyn = df[df["city"] == "brooklyn"]
print("\nbrooklyn students: \n", brooklyn)
print("*" * 40)

# combine conditions with & and |
high_pass = df[(df["grade"] > 80) & (df["passed"] == True)]
print("\nhigh grade and passed:\n", high_pass)
print("*" * 40)

queens_or_brooklyn = df[df["city"].isin(["brooklyn", "queens"])]
print("\nqueens or brooklyn:\n", queens_or_brooklyn)
print("*" * 40)

'''Creating new columns
Feature engineering — creating new information from existing data'''

# ── CREATING NEW COLUMNS ─────────────────────
# grade out of 100 → letter grade
def get_letter(grade):
    if grade >= 90: return "A"
    elif grade >= 80: return "B"
    elif grade >= 70: return "C"
    elif grade >= 60: return "D"
    else: return "F"

#simple math column
df["grade_pct"] = df["grade"] / 100

# apply a function to each row
df["letter"] = df["grade"].apply(get_letter)

# string operation
df["name_upper"] = df["name"].str.upper()

# boolean column
df["high_achiever"] = df["grade"] >= 85

print(df[["name", "grade", "grade_pct", "letter", "high_achiever"]])

'''groupby — summarize data by category
The most powerful aggregation tool in pandas'''
# ── GROUPBY ──────────────────────────────────
# average grade by city
city_avg = df.groupby("city")["grade"].mean()
print("average grade by city:\n", city_avg)
print("*" * 40)

# count students per city
city_count = df.groupby("city")["name"].count()
print("\nstudents per city:\n", city_count)
print("*" * 40)

# multiple stats at once with agg()
city_states = df.groupby("city")["grade"].agg(
    ["mean", "min", "max", "count"]
)
print("\ncity states:\n", city_states)
print("*" * 40)

# pass rate by city
pass_rate = df.groupby("city")["passed"].mean()
print("\npass rate by city:\n", pass_rate)
# True=1, False=0, so mean = fraction who passed
print("*" * 40)

'''Read and write CSV files
Real data science starts here — loading actual files'''
# ── SAVE AND LOAD CSV ────────────────────────

# save DataFrame to CSV
df.to_csv("students.csv", index=False)
print("saved students.csv")
print("*" * 40)

# read it back
df2 = pd.read_csv("students.csv")
print("\nload from csv:")
print(df2.head())
print("shape:", df2.shape)
print("*" * 40)

# sort by grade descending
sorted_df = df2.sort_values("grade", ascending=False)
print("\nranked by grade:\n", sorted_df[["name", "grade", "letter"]])
print("*" * 40)

#missing value check
print("\nmissing values:\n", df2.isnull().sum())
print("*" * 40)

#summery stats
print("\ngrade summery:")
print(f" mean : {df2['grade'].mean():.1f}")
print(f"  Median: {df2['grade'].median():.1f}")
print(f"  Std   : {df2['grade'].std():.1f}")
print("*" * 40)