"""
database.py
Loads the CSV into an SQLite database and runs SQL queries for analysis.
"""

import sqlite3
import pandas as pd


DB_PATH = "data/student_performance.db"


def load_to_sqlite(csv_path: str = "data/student_performance.csv") -> None:
    """Load CSV into SQLite table 'students'."""
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("students", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ Data loaded into SQLite → {DB_PATH}")


def run_queries() -> dict:
    """Run SQL queries and return results as DataFrames."""
    conn = sqlite3.connect(DB_PATH)
    results = {}

    queries = {
        "avg_score_by_gender": """
            SELECT gender,
                   ROUND(AVG(avg_score), 2)   AS avg_score,
                   ROUND(AVG(math_score), 2)  AS avg_math,
                   ROUND(AVG(science_score),2) AS avg_science,
                   ROUND(AVG(english_score),2) AS avg_english
            FROM students
            GROUP BY gender;
        """,
        "grade_distribution": """
            SELECT grade, COUNT(*) AS count,
                   ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM students), 2) AS percentage
            FROM students
            GROUP BY grade
            ORDER BY grade;
        """,
        "top_10_students": """
            SELECT student_id, gender, avg_score, grade, attendance_pct, study_hours_per_day
            FROM students
            ORDER BY avg_score DESC
            LIMIT 10;
        """,
        "avg_by_parent_education": """
            SELECT parent_education,
                   COUNT(*) AS total_students,
                   ROUND(AVG(avg_score), 2) AS avg_score
            FROM students
            GROUP BY parent_education
            ORDER BY avg_score DESC;
        """,
        "internet_vs_no_internet": """
            SELECT internet_access,
                   ROUND(AVG(avg_score), 2)  AS avg_score,
                   ROUND(AVG(attendance_pct), 2) AS avg_attendance
            FROM students
            GROUP BY internet_access;
        """,
        "pass_fail_count": """
            SELECT pass_fail, COUNT(*) AS count
            FROM students
            GROUP BY pass_fail;
        """,
        "high_study_low_score": """
            SELECT student_id, study_hours_per_day, avg_score, attendance_pct, sleep_hours
            FROM students
            WHERE study_hours_per_day > 7 AND avg_score < 50
            ORDER BY avg_score ASC
            LIMIT 10;
        """,
    }

    for name, sql in queries.items():
        df = pd.read_sql_query(sql, conn)
        results[name] = df
        print(f"\n📋 Query → {name}:\n{df.to_string(index=False)}")

    conn.close()
    return results


if __name__ == "__main__":
    load_to_sqlite()
    run_queries()
