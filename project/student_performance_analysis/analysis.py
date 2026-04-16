"""
analysis.py
Statistical analysis of student performance:
mean, median, variance, std, correlation and insight generation.
"""

import numpy as np
import pandas as pd


def load_data(path: str = "data/student_performance.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# ── Data Cleaning ──────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🧹 DATA CLEANING")
    print(f"   Shape before: {df.shape}")

    # Check duplicates
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"   Duplicates removed: {dupes}")

    # Check nulls
    nulls = df.isnull().sum().sum()
    df = df.dropna()
    print(f"   Null rows removed: {nulls}")

    # Fix outlier study hours (cap at 16h/day)
    df['study_hours_per_day'] = df['study_hours_per_day'].clip(0, 16)

    # Fix score range
    score_cols = ['math_score', 'science_score', 'english_score', 'history_score', 'cs_score']
    for col in score_cols:
        df[col] = df[col].clip(0, 100)

    df['avg_score'] = df[score_cols].mean(axis=1).round(2)
    print(f"   Shape after : {df.shape}")
    return df


# ── Statistical Analysis ───────────────────────────────────────────────────────
def compute_statistics(df: pd.DataFrame) -> dict:
    score_cols = ['math_score', 'science_score', 'english_score', 'history_score', 'cs_score', 'avg_score']

    print("\n📊 STATISTICAL ANALYSIS")
    stats = {}
    for col in score_cols:
        data = df[col].dropna()
        stat = {
            'mean'    : round(float(np.mean(data)), 2),
            'median'  : round(float(np.median(data)), 2),
            'variance': round(float(np.var(data, ddof=1)), 2),
            'std_dev' : round(float(np.std(data, ddof=1)), 2),
            'min'     : round(float(data.min()), 2),
            'max'     : round(float(data.max()), 2),
            'q1'      : round(float(np.percentile(data, 25)), 2),
            'q3'      : round(float(np.percentile(data, 75)), 2),
        }
        stats[col] = stat

    stats_df = pd.DataFrame(stats).T
    print(stats_df.to_string())
    stats_df.to_csv("output/statistical_summary.csv")
    print("\n   ✅ Saved → output/statistical_summary.csv")
    return stats


# ── Correlation Analysis ───────────────────────────────────────────────────────
def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df[['study_hours_per_day', 'attendance_pct', 'sleep_hours',
                      'math_score', 'science_score', 'english_score',
                      'history_score', 'cs_score', 'avg_score']]
    corr = numeric_df.corr(method='pearson').round(3)

    print("\n🔗 CORRELATION MATRIX (with avg_score):")
    print(corr['avg_score'].sort_values(ascending=False).to_string())
    corr.to_csv("output/correlation_matrix.csv")
    print("   ✅ Saved → output/correlation_matrix.csv")
    return corr


# ── Insight Generation ─────────────────────────────────────────────────────────
def generate_insights(df: pd.DataFrame, stats: dict) -> None:
    print("\n💡 GENERATED INSIGHTS")
    insights = []

    # 1. Overall pass rate
    pass_rate = (df['pass_fail'] == 'Pass').mean() * 100
    insights.append(f"✅ Overall pass rate: {pass_rate:.1f}%")

    # 2. Best subject
    subjects = {'Math': 'math_score', 'Science': 'science_score',
                'English': 'english_score', 'History': 'history_score', 'CS': 'cs_score'}
    best_subj   = max(subjects, key=lambda s: stats[subjects[s]]['mean'])
    lowest_subj = min(subjects, key=lambda s: stats[subjects[s]]['mean'])
    insights.append(f"📈 Highest avg subject: {best_subj} ({stats[subjects[best_subj]]['mean']})")
    insights.append(f"📉 Lowest avg subject : {lowest_subj} ({stats[subjects[lowest_subj]]['mean']})")

    # 3. Study hours vs score correlation
    corr_study = df['study_hours_per_day'].corr(df['avg_score'])
    insights.append(f"📚 Study hours ↔ Score correlation: {corr_study:.3f}")

    # 4. Attendance impact
    high_att = df[df['attendance_pct'] >= 85]['avg_score'].mean()
    low_att  = df[df['attendance_pct'] <  60]['avg_score'].mean()
    insights.append(f"🏫 Avg score (attendance ≥85%): {high_att:.1f} vs (<60%): {low_att:.1f}")

    # 5. Gender comparison
    gender_avg = df.groupby('gender')['avg_score'].mean()
    for g, v in gender_avg.items():
        insights.append(f"👤 {g} avg score: {v:.2f}")

    # 6. Internet access impact
    inet = df.groupby('internet_access')['avg_score'].mean()
    insights.append(f"🌐 Internet (Yes/No) avg: {inet.get('Yes', 0):.1f} / {inet.get('No', 0):.1f}")

    for i in insights:
        print(f"   {i}")

    with open("output/insights.txt", "w") as f:
        f.write("STUDENT PERFORMANCE — KEY INSIGHTS\n")
        f.write("=" * 45 + "\n\n")
        f.write("\n".join(insights))
    print("\n   ✅ Saved → output/insights.txt")


# ── Power BI Export ────────────────────────────────────────────────────────────
def export_for_powerbi(df: pd.DataFrame) -> None:
    """Export cleaned enriched dataset ready for Power BI import."""
    df_export = df.copy()
    df_export['score_band'] = pd.cut(
        df_export['avg_score'],
        bins=[0, 40, 55, 70, 85, 100],
        labels=['Below 40', '40–55', '55–70', '70–85', '85–100']
    )
    df_export.to_csv("output/powerbi_export.csv", index=False)
    print("\n📊 Power BI Export:")
    print(f"   ✅ Saved → output/powerbi_export.csv  ({len(df_export)} rows)")
    print("   ℹ️  Import this CSV into Power BI Desktop via Get Data → Text/CSV")


if __name__ == "__main__":
    df  = load_data()
    df  = clean_data(df)
    stats = compute_statistics(df)
    compute_correlations(df)
    generate_insights(df, stats)
    export_for_powerbi(df)
