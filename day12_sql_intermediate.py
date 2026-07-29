
"""
Day 12: SQL Intermediate
MD Mutasim Billah - 52-week Data Science to ML/AI roadmap
 
Covers:
  1. Subqueries (scalar and IN-list)
  2. CASE WHEN (conditional bucketing)
  3. CTEs (WITH clause)
  4. Window functions (RANK, ROW_NUMBER, PARTITION BY)
  5. Multiple JOINs (3 tables)
  6. UNION vs UNION ALL
 
Reuses titanic.db built in Day 11. If it's missing, rebuilds it from
titanic.csv (downloading titanic.csv if that's missing too).
"""

import os
import sqlite3
import urllib.request

import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

DATA_PATH = "titanic.csv"
DATA_URL =  "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

DB_PATH = "titanic.db"

def load_titanic(path=DATA_PATH, url=DATA_URL):
    if not os.path.exists(path):
        print(f"{path} not found locally - downloading...")
        urllib.request.urlretrieve(url, path)
    return pd.read_csv(path)

def run_query(conn, label, sql):
    print(f"\n--- {label} ---")
    print(sql.strip())
    print()
    result = pd.read_sql(sql, conn)
    print(result.to_string(index=False))
    return result

# ---------------------------------------------------------------------------
# 0. SETUP — reuse Day 11's database, add embarked_lookup for multi-joins
# ---------------------------------------------------------------------------

def setup_database():
    print("=" * 70)
    print("0. SETUP")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='tables';", conn)["name"].tolist()
    if "titanic" not in tables:
        df = load_titanic()
        df.to_sql("titanic", conn, if_exists="replace", index=False)
        print("Rebuild 'titanic' table (day 11 wasn't found).")
    else:
        print("rebuild existing 'titanic' table from day 11's titanic.db")
    # ensure class_lookup exists
    if "class_lookup" not in tables:
        class_lookup = pd.DataFrame({
            "Pclass": [1, 2, 3],
            "class_label": ["First", "Second", "Third"],
        })
        class_lookup.to_sql("class_lookup", conn, if_exists="replace", index=False)
        print("Rebuilt 'class_lookup' table.")
    else:
        print("Reusing existing 'class_lookup' table.")

    # ensure embarked_lookup exists
    if "embarked_lookup" not in tables:
        embarked_lookup = pd.DataFrame({
            "Embarked": ["S", "C", "Q"],
            "port_name": ["Southampton", "Cherbourg", "Queenstown"],
        })
        embarked_lookup.to_sql("embarked_lookup", conn, if_exists="replace", index=False)
        print("Rebuilt 'embarked_lookup' table.")
    else:
        print("Reusing existing 'embarked_lookup' table.")

    return conn

# ---------------------------------------------------------------------------
# 1. SUBQUERIES
# ---------------------------------------------------------------------------

def run_subqueries(conn):
    print("\n" + "=" * 70)
    print("1. SUBQUERIES")
    print("=" * 70)

    run_query(conn, "Passengers who paid more than the overall average fare", """
        SELECT Name, Pclass, Fare
        FROM titanic
        WHERE Fare > (SELECT AVG(Fare) FROM titanic)
        ORDER BY Fare DESC
        LIMIT 8;
       """)

    run_query(conn, "Passengaer in classes better than Third (IN-list subquery)", """
        SELECT Name, Pclass, Fare
        FROM titanic
        WHERE Pclass IN (
            SELECT Pclass FROM class_lookup WHERE class_label != 'Third'
        )
        LIMIT 8;
    """)
    
    run_query(conn, "Passengers who paid more than THEIR OWN class's average fare", """
        SELECT t.Name, t.Pclass, t.Fare
        FROM titanic t
        WHERE t.Fare > (
            SELECT AVG(t2.Fare)
            FROM titanic t2
            WHERE t2.Pclass = t.Pclass)
        ORDER BY t.Pclass, t.Fare DESC
        LIMIT 10;
        """)

# ---------------------------------------------------------------------------
# 2. CASE WHEN
# ---------------------------------------------------------------------------

def run_case_when(conn):
    print("\n" + "=" * 70)
    print("2. CASE WHEN")
    print("=" * 70)

    run_query(conn, "Age bucketed into Child / Adult / Senior", """
        SELECT Name, Age,
          CASE
            WHEN Age < 18 THEN 'Child'
            WHEN Age < 60 THEN 'Adult'
            WHEN Age >= 60 THEN 'Senior'
            ELSE 'Unknown'
          END AS age_group
        FROM titanic
        LIMIT 10;
        """)
    
    run_query(conn, "Fare bucketed into Budget / Standard / Premium, with counts", """
        SELECT
          CASE
            WHEN Fare < 15 THEN 'Budget'
            WHEN Fare < 60 THEN 'Standard'
            ELSE 'Premium'
          END AS fare_tier,
          COUNT(*) AS n_passengers,
          ROUND(AVG(Survived), 3) AS survival_rate
        FROM titanic
        GROUP BY fare_tier
        ORDER BY n_passengers DESC;
        """)

# ---------------------------------------------------------------------------
# 3. CTEs (WITH)
# ---------------------------------------------------------------------------
 
def run_ctes(conn):
    print("\n" + "=" * 70)
    print("3. CTEs (WITH)")
    print("=" * 70)

    run_query(conn, "Passengers who paid above their class average, via CTE", """
        WITH class_avg AS (
            SELECT Pclass, AVG(Fare) AS avg_fare
            FROM titanic
            GROUP BY Pclass)
        SELECT t.Name, t.Pclass, t.Fare, ROUND(c.avg_fare, 2) AS class_avg_fare
        FROM titanic t
        JOIN class_avg c ON t.Pclass = c.Pclass
        WHERE t.Fare > c.avg_fare
        ORDER BY t.Pclass, t.Fare DESC
        LIMIT 10;
        """)

    run_query(conn, "Two chained CTEs: class survival rates, then best class", """
        WITH class_stats AS (
            SELECT Pclass, AVG(Survived) AS survival_rate
            FROM titanic
            GROUP BY Pclass),
        ranked AS (
            SELECT Pclass, survival_rate,
                   RANK() OVER (ORDER BY survival_rate DESC) AS rnk
            FROM class_stats)
        SELECT Pclass, ROUND(survival_rate, 3) AS survival_rate, rnk
        FROM ranked;
        """)

# ---------------------------------------------------------------------------
# 4. WINDOW FUNCTIONS
# ---------------------------------------------------------------------------

def run_window_functions(conn):
    print("\n" + "=" * 70)
    print("4. WINDOW FUNCTIONS")
    print("=" * 70)
 
    run_query(conn, "Top 3 highest fares within each Pclass", """
        SELECT Name, Pclass, Fare, fare_rank FROM (
            SELECT Name, Pclass, Fare,
                RANK() OVER (
                    PARTITION BY Pclass ORDER BY Fare DESC) AS fare_rank
            FROM titanic)
        WHERE fare_rank <= 3
        ORDER BY Pclass, fare_rank;
        """)

    run_query(conn, "ROW_NUMBER vs RANK on tied fares (first 15 rows, Pclass 3)", """

        SELECT Name, Pclass, Fare,
            ROW_NUMBER() OVER (PARTITION BY Pclass ORDER BY Fare DESC) AS ROW_num,
            RANK() OVER (PARTITION BY Pclass ORDER BY Fare DESC) AS rank_num 
        FROM titanic
        WHERE Pclass = 3
        ORDER BY Fare DESC
        LIMIT 15;
        """)

# ---------------------------------------------------------------------------
# 5. MULTIPLE JOINS
# ---------------------------------------------------------------------------
def run_multiple_joins(conn):
    print("\n" + "=" * 70)
    print("5. MULTIPLE JOINS (3 TABLES)")
    print("=" * 70)
 
    run_query(conn, "titanic + class_lookup + embarked_lookup, joined together", """
        SELECT t.Name, c.class_label, e.port_name, t.Fare
        FROM titanic t
        JOIN class_lookup c ON t.Pclass = c.Pclass
        JOIN embarked_lookup e ON t.Embarked = e.Embarked
        LIMIT 10;
    """)
 
 
# ---------------------------------------------------------------------------
# 6. UNION vs UNION ALL
# ---------------------------------------------------------------------------
 
def run_union(conn):
    print("\n" + "=" * 70)
    print("6. UNION vs UNION ALL")
    print("=" * 70)
 
    run_query(conn, "UNION: children under 12 and elders over 65, deduped", """
        SELECT Name, Age, 'Child' AS note FROM titanic WHERE Age < 12
        UNION
        SELECT Name, Age, 'Elder' AS note FROM titanic WHERE Age > 65
        ORDER BY Age;
    """)
 
    run_query(conn, "UNION ALL: same query, keeps duplicates, no dedup cost", """
        SELECT 'Under 12' AS note, COUNT(*) AS n FROM titanic WHERE Age < 12
        UNION ALL
        SELECT 'Over 65' AS note, COUNT(*) AS n FROM titanic WHERE Age > 65;
    """)
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    conn = setup_database()
 
    run_subqueries(conn)
    run_case_when(conn)
    run_ctes(conn)
    run_window_functions(conn)
    run_multiple_joins(conn)
    run_union(conn)
 
    conn.close()
 
    print("\n" + "=" * 70)
    print("Day 12 complete. Database updated: titanic.db (added embarked_lookup)")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
