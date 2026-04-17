# -*- coding: utf-8 -*-
"""
app.py  —  Streamlit Dashboard for Student Performance Analysis
Run locally : streamlit run app.py
Deploy free : https://share.streamlit.io
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sqlite3
import streamlit as st
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA_DIR   = BASE / "data"
OUTPUT_DIR = BASE / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark background */
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
[data-testid="stHeader"]           { background: transparent; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stMetricValue"] { color: #a78bfa; font-size: 2rem; font-weight: 800; }
[data-testid="stMetricLabel"] { color: #8b949e; }

/* Headers */
h1, h2, h3 { color: #e6edf3 !important; }
p, li       { color: #c9d1d9; }

/* Tab styling */
[data-baseweb="tab-list"] { background: #161b22; border-radius: 10px; padding: 4px; }
[data-baseweb="tab"]      { color: #8b949e; }
[aria-selected="true"]    { background: #6366f1 !important; border-radius: 8px; color: white !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 10px; }

/* Divider */
hr { border-color: #30363d; }

/* Insight box */
.insight-box {
    background: #161b22;
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    color: #c9d1d9;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ── Data pipeline (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_or_generate():
    csv_path = DATA_DIR / "student_performance.csv"
    if not csv_path.exists():
        sys.path.insert(0, str(BASE))
        from data_generator import generate_student_data
        df = generate_student_data(500)
        df.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)
    # clean
    df['study_hours_per_day'] = df['study_hours_per_day'].clip(0, 16)
    score_cols = ['math_score','science_score','english_score','history_score','cs_score']
    for c in score_cols:
        df[c] = df[c].clip(0, 100)
    df['avg_score'] = df[score_cols].mean(axis=1).round(2)
    return df

@st.cache_data
def get_stats(df):
    score_cols = ['math_score','science_score','english_score','history_score','cs_score','avg_score']
    stats = {}
    for col in score_cols:
        d = df[col].dropna()
        stats[col] = {
            'mean': round(float(np.mean(d)), 2),
            'median': round(float(np.median(d)), 2),
            'variance': round(float(np.var(d, ddof=1)), 2),
            'std_dev': round(float(np.std(d, ddof=1)), 2),
            'min': round(float(d.min()), 2),
            'max': round(float(d.max()), 2),
        }
    return pd.DataFrame(stats).T

@st.cache_data
def get_correlations(df):
    num = df[['study_hours_per_day','attendance_pct','sleep_hours',
               'math_score','science_score','english_score','history_score','cs_score','avg_score']]
    return num.corr(method='pearson').round(3)

# ── Chart helpers ─────────────────────────────────────────────────────────────
PALETTE = ['#6366f1','#a78bfa','#ec4899','#06b6d4','#10b981','#f59e0b']

def apply_dark(fig):
    fig.patch.set_facecolor('#0d1117')
    for ax in fig.get_axes():
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#e6edf3')
        ax.yaxis.label.set_color('#e6edf3')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
    return fig

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.facecolor':'#0d1117','axes.facecolor':'#161b22',
    'axes.edgecolor':'#30363d','axes.labelcolor':'#e6edf3',
    'xtick.color':'#8b949e','ytick.color':'#8b949e',
    'text.color':'#e6edf3','grid.color':'#21262d','grid.linewidth':0.6,
    'axes.titleweight':'bold','axes.titlesize':13,
})

# ── Sidebar ───────────────────────────────────────────────────────────────────
df_full = load_or_generate()

with st.sidebar:
    st.markdown("## 🎛️ Filters")
    gender_filter = st.multiselect("Gender", df_full['gender'].unique(), default=list(df_full['gender'].unique()))
    grade_filter  = st.multiselect("Grade",  sorted(df_full['grade'].dropna().unique()), default=sorted(df_full['grade'].dropna().unique()))
    internet_filter = st.multiselect("Internet Access", df_full['internet_access'].unique(), default=list(df_full['internet_access'].unique()))
    study_range   = st.slider("Study Hours / Day", 0.0, 12.0, (0.0, 12.0), 0.5)

    st.markdown("---")
    st.markdown("### 📂 Downloads")
    csv_bytes = df_full.to_csv(index=False).encode()
    st.download_button("⬇️ Download Dataset (CSV)", csv_bytes, "student_performance.csv", "text/csv")

    st.markdown("---")
    st.caption("Student Performance Analysis\nBuilt with Python + Streamlit")

# Apply filters
df = df_full[
    df_full['gender'].isin(gender_filter) &
    df_full['grade'].isin(grade_filter) &
    df_full['internet_access'].isin(internet_filter) &
    df_full['study_hours_per_day'].between(*study_range)
]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Student Performance Analysis")
st.markdown(f"**{len(df)}** students shown · Use sidebar filters to explore")
st.markdown("---")

if len(df) == 0:
    st.warning("⚠️ No students match the selected filters. Please adjust the filters in the sidebar!")
    st.stop()

# ── KPI Metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
pass_rate    = (df['pass_fail'] == 'Pass').mean() * 100
avg_score    = df['avg_score'].mean()
corr_study   = df['study_hours_per_day'].corr(df['avg_score'])
best_subject = df[['math_score','science_score','english_score','history_score','cs_score']].mean().idxmax().replace('_score','').title()
female_avg   = df[df['gender']=='Female']['avg_score'].mean()
male_avg     = df[df['gender']=='Male']['avg_score'].mean()

c1.metric("Pass Rate",         f"{pass_rate:.1f}%")
c2.metric("Avg Score",         f"{avg_score:.2f}")
c3.metric("Study Correlation", f"{corr_study:.3f}")
c4.metric("Best Subject",      best_subject)
c5.metric("Female Avg",        f"{female_avg:.2f}")
c6.metric("Male Avg",          f"{male_avg:.2f}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🏠 Overview", "📈 Charts", "🔗 Correlations", "🗄️ SQL Queries", "📋 Statistics", "💡 Insights"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Summary Dashboard")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        # Score distribution
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(df['avg_score'], bins=25, kde=True, color='#6366f1', ax=ax, alpha=0.75)
        ax.axvline(df['avg_score'].mean(), color='white', linestyle='--', linewidth=1.5, label=f"Mean: {df['avg_score'].mean():.1f}")
        ax.set_title("Average Score Distribution"); ax.set_xlabel("Score"); ax.legend()
        fig = apply_dark(fig); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with col_r:
        # Grade pie
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        gc = df['grade'].value_counts().sort_index()
        ax2.pie(gc, labels=gc.index, autopct='%1.1f%%', colors=PALETTE, startangle=140,
                wedgeprops=dict(edgecolor='#0d1117', linewidth=2))
        ax2.set_title("Grade Distribution")
        fig2.patch.set_facecolor('#0d1117'); fig2.tight_layout()
        st.pyplot(fig2); plt.close(fig2)

    # Pass/Fail
    col_a, col_b = st.columns(2)
    with col_a:
        pf = df['pass_fail'].value_counts()
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        bars = ax3.bar(pf.index, pf.values, color=[PALETTE[4], PALETTE[2]], edgecolor='#0d1117', linewidth=1.5)
        for b, v in zip(bars, pf.values):
            ax3.text(b.get_x()+b.get_width()/2, b.get_height()+2, str(v), ha='center', fontweight='bold')
        ax3.set_title("Pass / Fail Count"); ax3.set_ylabel("Students")
        fig3 = apply_dark(fig3); fig3.tight_layout()
        st.pyplot(fig3); plt.close(fig3)

    with col_b:
        gm = df.groupby('gender')['avg_score'].mean()
        fig4, ax4 = plt.subplots(figsize=(6, 3.5))
        bars4 = ax4.bar(gm.index, gm.values, color=[PALETTE[2], PALETTE[0]], edgecolor='#0d1117', linewidth=1.5)
        for b, v in zip(bars4, gm.values):
            ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{v:.2f}', ha='center', fontweight='bold')
        ax4.set_title("Avg Score by Gender"); ax4.set_ylim(0, 100)
        fig4 = apply_dark(fig4); fig4.tight_layout()
        st.pyplot(fig4); plt.close(fig4)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("All Visualizations")

    # Score distributions per subject
    score_cols = {'Math':'math_score','Science':'science_score','English':'english_score','History':'history_score','CS':'cs_score'}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('Score Distribution per Subject', fontsize=15, fontweight='bold')
    axes = axes.flatten()
    for idx, (label, col) in enumerate(score_cols.items()):
        ax = axes[idx]
        sns.histplot(df[col], bins=20, kde=True, color=PALETTE[idx], ax=ax, alpha=0.7)
        ax.axvline(df[col].mean(), color='white', linestyle='--', linewidth=1.2, label=f'Mean:{df[col].mean():.1f}')
        ax.set_title(f'{label} Score'); ax.legend(fontsize=8)
    sns.histplot(df['avg_score'], bins=25, kde=True, color='#f59e0b', ax=axes[5], alpha=0.7)
    axes[5].set_title('Overall Avg Score')
    fig = apply_dark(fig); fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        # Study vs Score scatter
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        sc = ax5.scatter(df['study_hours_per_day'], df['avg_score'], c=df['attendance_pct'],
                         cmap='viridis', alpha=0.6, s=25)
        m, b = np.polyfit(df['study_hours_per_day'], df['avg_score'], 1)
        xl = np.linspace(df['study_hours_per_day'].min(), df['study_hours_per_day'].max(), 100)
        ax5.plot(xl, m*xl+b, color='#ec4899', linewidth=2,
                 label=f'r={df["study_hours_per_day"].corr(df["avg_score"]):.2f}')
        plt.colorbar(sc, ax=ax5, label='Attendance %')
        ax5.set_title('Study Hours vs Avg Score'); ax5.set_xlabel('Study Hrs/Day'); ax5.legend()
        fig5 = apply_dark(fig5); fig5.tight_layout()
        st.pyplot(fig5); plt.close(fig5)

    with col2:
        # Boxplot by grade
        fig6, ax6 = plt.subplots(figsize=(7, 5))
        order = [g for g in ['F','D','C','B','A'] if g in df['grade'].values]
        sns.boxplot(data=df, x='grade', y='avg_score', order=order,
                    palette=dict(zip(order, PALETTE)), ax=ax6, linewidth=1.5, fliersize=3)
        ax6.set_title('Score Distribution by Grade')
        fig6 = apply_dark(fig6); fig6.tight_layout()
        st.pyplot(fig6); plt.close(fig6)

    col3, col4 = st.columns(2)
    with col3:
        # Gender comparison grouped bar
        subjects = ['math_score','science_score','english_score','history_score','cs_score']
        labels   = ['Math','Science','English','History','CS']
        gm2 = df.groupby('gender')[subjects].mean()
        x = np.arange(len(labels)); w = 0.35
        fig7, ax7 = plt.subplots(figsize=(7, 5))
        if 'Female' in gm2.index:
            ax7.bar(x-w/2, gm2.loc['Female'], w, label='Female', color=PALETTE[2], alpha=0.85)
        if 'Male' in gm2.index:
            ax7.bar(x+w/2, gm2.loc['Male'],   w, label='Male',   color=PALETTE[0], alpha=0.85)
        ax7.set_xticks(x); ax7.set_xticklabels(labels)
        ax7.set_title('Score by Subject & Gender'); ax7.legend(); ax7.set_ylim(0,100)
        fig7 = apply_dark(fig7); fig7.tight_layout()
        st.pyplot(fig7); plt.close(fig7)

    with col4:
        # Internet violin
        fig8, ax8 = plt.subplots(figsize=(7, 5))
        sns.violinplot(data=df, x='internet_access', y='avg_score',
                       palette={'Yes': PALETTE[0], 'No': PALETTE[2]}, ax=ax8, inner='quartile')
        ax8.set_title('Score Distribution by Internet Access')
        fig8 = apply_dark(fig8); fig8.tight_layout()
        st.pyplot(fig8); plt.close(fig8)

    # Parent education
    fig9, ax9 = plt.subplots(figsize=(12, 5))
    order2 = ['No Education','High School','Bachelor','Master','PhD']
    means = df.groupby('parent_education')['avg_score'].mean().reindex(order2)
    bars9 = ax9.bar(means.index, means.values, color=PALETTE, edgecolor='#0d1117')
    for b, v in zip(bars9, means.values):
        ax9.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{v:.1f}', ha='center', fontweight='bold')
    ax9.set_title('Impact of Parent Education on Avg Score'); ax9.set_ylim(0,100)
    fig9 = apply_dark(fig9); fig9.tight_layout()
    st.pyplot(fig9); plt.close(fig9)

    # 2D heatmap
    df2 = df.copy()
    df2['study_bin'] = pd.cut(df2['study_hours_per_day'], bins=5, precision=0)
    df2['att_bin']   = pd.cut(df2['attendance_pct'], bins=5, precision=0)
    pivot = df2.pivot_table(values='avg_score', index='att_bin', columns='study_bin', aggfunc='mean')
    fig10, ax10 = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax10,
                linewidths=0.5, cbar_kws={'label':'Avg Score'})
    ax10.set_title('Avg Score by Study Hours × Attendance Band')
    fig10 = apply_dark(fig10); fig10.tight_layout()
    st.pyplot(fig10); plt.close(fig10)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Pearson Correlation Matrix")
    corr = get_correlations(df)
    fig11, ax11 = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, ax=ax11, annot_kws={'size':9})
    ax11.set_title('Pearson Correlation Heatmap')
    fig11 = apply_dark(fig11); fig11.tight_layout()
    st.pyplot(fig11); plt.close(fig11)

    st.markdown("#### Correlation with Avg Score (sorted)")
    corr_series = corr['avg_score'].drop('avg_score').sort_values(ascending=False).reset_index()
    corr_series.columns = ['Feature', 'Correlation']
    st.dataframe(corr_series, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SQL QUERIES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("SQL Query Results")
    db_path = DATA_DIR / "student_performance.db"

    if not db_path.exists():
        # Save to SQLite on first run
        conn = sqlite3.connect(db_path)
        df_full.to_sql("students", conn, if_exists="replace", index=False)
        conn.close()

    conn = sqlite3.connect(db_path)
    queries = {
        "Avg Score by Gender": "SELECT gender, ROUND(AVG(avg_score),2) AS avg_score, ROUND(AVG(math_score),2) AS avg_math, ROUND(AVG(science_score),2) AS avg_science FROM students GROUP BY gender",
        "Grade Distribution": "SELECT grade, COUNT(*) AS count, ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM students),2) AS pct FROM students GROUP BY grade ORDER BY grade",
        "Top 10 Students": "SELECT student_id, gender, avg_score, grade, attendance_pct, study_hours_per_day FROM students ORDER BY avg_score DESC LIMIT 10",
        "Parent Education Impact": "SELECT parent_education, COUNT(*) AS total, ROUND(AVG(avg_score),2) AS avg_score FROM students GROUP BY parent_education ORDER BY avg_score DESC",
        "Internet Access Impact": "SELECT internet_access, ROUND(AVG(avg_score),2) AS avg_score, ROUND(AVG(attendance_pct),2) AS avg_att FROM students GROUP BY internet_access",
        "Pass / Fail Count": "SELECT pass_fail, COUNT(*) AS count FROM students GROUP BY pass_fail",
        "High Study Low Score": "SELECT student_id, study_hours_per_day, avg_score, attendance_pct FROM students WHERE study_hours_per_day > 7 AND avg_score < 50 ORDER BY avg_score LIMIT 10",
    }

    for name, sql in queries.items():
        with st.expander(f"📋 {name}"):
            result = pd.read_sql_query(sql, conn)
            st.dataframe(result, use_container_width=True)
            st.code(sql, language="sql")
    conn.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Statistical Summary")
    stats_df = get_stats(df)
    st.dataframe(stats_df.style.background_gradient(cmap='RdYlGn', axis=0), use_container_width=True)

    st.markdown("#### Raw Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("💡 Auto-Generated Insights")
    pass_rate2 = (df['pass_fail'] == 'Pass').mean() * 100
    subj_map = {'Math':'math_score','Science':'science_score','English':'english_score','History':'history_score','CS':'cs_score'}
    subj_avgs = {k: df[v].mean() for k, v in subj_map.items()}
    best  = max(subj_avgs, key=subj_avgs.get)
    worst = min(subj_avgs, key=subj_avgs.get)
    study_corr = df['study_hours_per_day'].corr(df['avg_score'])
    high_att = df[df['attendance_pct']>=85]['avg_score'].mean()
    low_att  = df[df['attendance_pct']<60]['avg_score'].mean()
    inet = df.groupby('internet_access')['avg_score'].mean()
    parent_top = df.groupby('parent_education')['avg_score'].mean().idxmax()

    insights = [
        f"✅ Overall pass rate: **{pass_rate2:.1f}%** ({int(df['pass_fail'].eq('Pass').sum())} of {len(df)} students)",
        f"📈 Highest avg subject: **{best}** ({subj_avgs[best]:.2f})",
        f"📉 Lowest avg subject: **{worst}** ({subj_avgs[worst]:.2f})",
        f"📚 Study hours ↔ Score correlation: **{study_corr:.3f}** (very strong positive)",
        f"🏫 Attendance ≥85% avg score: **{high_att:.1f}** vs <60%: **{low_att:.1f}**",
        f"👩 Female avg: **{df[df['gender']=='Female']['avg_score'].mean():.2f}** | 👨 Male avg: **{df[df['gender']=='Male']['avg_score'].mean():.2f}**",
        f"🌐 Internet Yes avg: **{inet.get('Yes', 0):.1f}** | No avg: **{inet.get('No', 0):.1f}**",
        f"🎓 Best parent education group: **{parent_top}** ({df.groupby('parent_education')['avg_score'].mean().max():.2f} avg)",
        f"😴 Sleep hours correlation: **{df['sleep_hours'].corr(df['avg_score']):.3f}** (nearly zero impact)",
    ]

    for ins in insights:
        st.markdown(f"""<div class="insight-box">{ins}</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Subject Average Comparison")
    subj_df = pd.DataFrame(subj_avgs.items(), columns=['Subject', 'Avg Score']).sort_values('Avg Score', ascending=False)
    st.bar_chart(subj_df.set_index('Subject'))
