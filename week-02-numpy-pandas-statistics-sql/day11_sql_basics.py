"""
Day 11: SQL Basics
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
 
Covers:
  1. Loading Titanic data into a real SQLite database
  2. SELECT, FROM, WHERE
  3. ORDER BY, LIMIT, DISTINCT
  4. Aggregate functions + GROUP BY
  5. WHERE vs HAVING
  6. INNER JOIN / LEFT JOIN
 
Expects titanic.csv in the same folder. If not found, downloads a copy.
Every query is run through sqlite3 + pandas.read_sql so you see the SQL
AND the resulting table together.
"""

import os
import sqlite3
import urllib.request

import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 120)

DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

DB_PATH = "titanic.db"

def load_titanic(path=DATA_PATH, url=DATA_URL):
    """Load titanic.csv locally, or download it if missing."""
    if not os.path.exists(path):
        print(f"{path} not found locally - downloading...")
        urllib.request.urlretrieve(url, path)
    return pd.read_csv(path)

def run_query(conn, label, sql):
    """Run a SQL query, print the SQL, print the result, return the DataFrame."""

    print(f"\n--- {label} ---")
    print(sql.strip())
    print()
    result = pd.read_sql(sql, conn)
    print(result.to_string(index=False))
    return result


# ---------------------------------------------------------------------------
# 1. LOAD DATA INTO A REAL SQL DATABASE
# ---------------------------------------------------------------------------

def setup_database():
    print("=" * 70)
    print("1. LOADING DATA INTO SQLITE")
    print("=" * 70)
    
    df = load_titanic()
    print(f"Loaded titanic.csv: {df.shape[0]} rows, {df.shape[1]} columns")

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("titanic", conn, if_exists="replace", index=False)
    print(f"Wrote table 'titanic' into {DB_PATH}")

    # small lookup table for the JOIN section later
    class_lookup = pd.DataFrame({
        "Pclass": [1, 2, 3],
         "class_label": ["First", "Second", "Third"],
    })
    class_lookup.to_sql("class_lookup", conn, if_exists="replace", index=False)
    print("Wrote lookup table 'class_lookup' (Pclass -> class_label)")
    return conn

# ---------------------------------------------------------------------------
# 2. SELECT, FROM, WHERE
# ---------------------------------------------------------------------------

def run_select_where(conn):
    print("\n" + "=" * 70)
    print("2. SELECT, FROM, WHERE")
    print("=" * 70)
    
    run_query(conn, "First class survivors: name, sex, age, fare", """
    SELECT Name, Sex, Age, Fare
    FROM titanic
    WHERE Pclass = 1 AND Survived = 1
    LIMIT 8;
    """)

    run_query(conn, "Female passengers under 18", """
    SELECT Name, Age, Pclass, Survived
    FROM titanic
    WHERE Sex = 'female' AND Age < 18
    LIMIT 8;
    """)

# ---------------------------------------------------------------------------
# 3. ORDER BY, LIMIT, DISTINCT
# ---------------------------------------------------------------------------

def run_order_limit_distinct(conn):
    print("\n" + "=" * 70)
    print("3. ORDER BY, LIMIT, DISTINCT")
    print("=" * 70)

    run_query(conn, "Distinct embarkation ports", """
      SELECT DISTINCT Embarked
      FROM titanic;
      """)
    
    run_query(conn, "Top 5 highest fares paid", """
      SELECT Name, Fare, Pclass
      FROM titanic
      ORDER BY Fare DESC
      LIMIT 5;""")

    run_query(conn, "5 oldest survivors", """
      SELECT Name, Age, Sex, Pclass
      FROM titanic
      WHERE Survived = 1
      ORDER BY Age DESC
      LIMIT 5;""")

# ---------------------------------------------------------------------------
# 4. AGGREGATES + GROUP BY
# ---------------------------------------------------------------------------

def run_aggregates_group_by(conn):
    print("\n" + "=" * 70)
    print("4. AGGREGATE FUNCTIONS + GROUP BY")
    print("=" * 70)

    run_query(conn, "Passenger count, avg fare, survival rate by Pclass",
"""
      SELECT
          Pclass,
          COUNT(*) AS n_passengers,
          ROUND(AVG(Fare), 2) AS avg_fare,
          ROUND(AVG(Survived), 3) AS survived_rate
      FROM titanic
      GROUP BY Pclass
      ORDER BY Pclass;
""")

    run_query(conn, "Survival rate and avg age by Sex", """
        SELECT
            Sex,
            COUNT(*) AS n_passengers,
            ROUND(AVG(Age), 1) AS avg_age,
            ROUND(AVG(Survived), 3) AS survival_rate
        FROM titanic
        GROUP BY Sex;
""")

# ---------------------------------------------------------------------------
# 5. WHERE VS HAVING
# ---------------------------------------------------------------------------

def run_where_vs_having(conn):
    print("\n" + "=" * 70)
    print("5. WHERE VS HAVING")
    print("=" * 70)
 
    run_query(conn, "HAVING: filter groups AFTER aggregating (ports with 50+ passengers)", """
      SELECT Embarked, COUNT(*) AS n
      FROM titanic
      GROUP BY Embarked
      HAVING COUNT(*) > 50;
""")

    run_query(conn, "HAVING: Pclass/Sex groups with avg fare over 50", """ 
      SELECT Pclass, Sex, ROUND(AVG(Fare), 2) AS avg_fare
      FROM titanic
      GROUP BY Pclass, Sex
      HAVING AVG(Fare) > 50
      ORDER BY avg_fare DESC;
""")

# ---------------------------------------------------------------------------
# 6. JOINS
# ---------------------------------------------------------------------------

def run_joins(conn):
    print("\n" + "=" * 70)
    print("6. JOINS")
    print("=" * 70)

    run_query(conn, "INNER JOIN titanic with class_lookup", """
        SELECT t.Name, t.Pclass, c.class_label, t.Fare
        FROM titanic t
        INNER JOIN class_lookup c
            ON t.Pclass = c.Pclass
        LIMIT 8;
    """)
    
    run_query(conn, "LEFT JOIN: same query, nothing should be missing here"
                    " (every Pclass has a lookup row) — count check", """
        SELECT c.class_label, COUNT(t.Name) AS n_passengers
        FROM class_lookup c
        LEFT JOIN titanic t
            ON t.Pclass = c.Pclass
        GROUP BY c.class_label;
    """)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    conn = setup_database()
    run_select_where(conn)
    run_order_limit_distinct(conn)
    run_aggregates_group_by(conn)
    run_where_vs_having(conn)
    run_joins(conn)

    conn.close()

    print("\n" + "=" * 70)
    print("Day 11 complete. Database saved: titanic.db")
    print("=" * 70)
 
if __name__ == "__main__":
    main()