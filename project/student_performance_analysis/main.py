# -*- coding: utf-8 -*-
"""
main.py
Orchestrates the full Student Performance Analysis pipeline:
  1. Generate synthetic dataset
  2. Load into SQLite & run SQL queries
  3. Statistical analysis + insights
  4. Generate all Matplotlib / Seaborn visualizations
"""

import os
import sys

# Fix Windows console encoding so emojis don't crash
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure we always run relative to this file's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Make sure output dirs exist
os.makedirs("data",          exist_ok=True)
os.makedirs("output",        exist_ok=True)
os.makedirs("output/charts", exist_ok=True)

# ── Step 1: Generate Dataset ──────────────────────────────────────────────────
print("=" * 55)
print("  STUDENT PERFORMANCE ANALYSIS - FULL PIPELINE")
print("=" * 55)

print("\n[STEP 1] Data Generation")
from data_generator import generate_student_data
df_raw = generate_student_data(500)
df_raw.to_csv("data/student_performance.csv", index=False)
print(f"   OK  Dataset saved -> data/student_performance.csv")
print(f"   Rows: {len(df_raw)}   |   Columns: {len(df_raw.columns)}")

# ── Step 2: SQLite + SQL Queries ──────────────────────────────────────────────
print("\n[STEP 2] Database & SQL Queries")
from database import load_to_sqlite, run_queries
load_to_sqlite()
sql_results = run_queries()

# ── Step 3: Statistical Analysis ──────────────────────────────────────────────
print("\n[STEP 3] Statistical Analysis")
from analysis import load_data, clean_data, compute_statistics, compute_correlations, generate_insights, export_for_powerbi
df = load_data()
df = clean_data(df)
stats = compute_statistics(df)
compute_correlations(df)
generate_insights(df, stats)
export_for_powerbi(df)

# ── Step 4: Visualizations ────────────────────────────────────────────────────
print("\n[STEP 4] Generating Visualizations")
from visualizations import generate_all_charts
generate_all_charts(df)

# ── Done ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  PIPELINE COMPLETE")
print("=" * 55)
print("\nOutput files generated:")
print("   data/student_performance.csv   - raw dataset")
print("   data/student_performance.db    - SQLite database")
print("   output/statistical_summary.csv - stats per subject")
print("   output/correlation_matrix.csv  - Pearson correlations")
print("   output/insights.txt            - generated insights")
print("   output/powerbi_export.csv      - Power BI ready CSV")
print("   output/charts/                 - 10 PNG charts")
print()
