"""
visualizations.py
All Matplotlib & Seaborn charts for the Student Performance Analysis Dashboard.
Charts are saved to output/charts/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ── Global Style ───────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    'figure.facecolor' : '#0d1117',
    'axes.facecolor'   : '#161b22',
    'axes.edgecolor'   : '#30363d',
    'axes.labelcolor'  : '#e6edf3',
    'xtick.color'      : '#8b949e',
    'ytick.color'      : '#8b949e',
    'text.color'       : '#e6edf3',
    'grid.color'       : '#21262d',
    'grid.linewidth'   : 0.6,
    'font.family'      : 'DejaVu Sans',
    'axes.titleweight' : 'bold',
    'axes.titlesize'   : 13,
})

PALETTE  = ['#6366f1', '#a78bfa', '#ec4899', '#06b6d4', '#10b981', '#f59e0b']
OUT_DIR  = "output/charts"
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, name: str):
    path = f"{OUT_DIR}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   ✅ Saved → {path}")


# ── 1. Score Distribution (Histogram + KDE) ────────────────────────────────────
def plot_score_distribution(df: pd.DataFrame):
    score_cols = {
        'Math': 'math_score', 'Science': 'science_score',
        'English': 'english_score', 'History': 'history_score', 'CS': 'cs_score'
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Score Distribution per Subject', fontsize=16, fontweight='bold', y=1.01)
    axes = axes.flatten()

    for idx, (label, col) in enumerate(score_cols.items()):
        ax = axes[idx]
        sns.histplot(df[col], bins=20, kde=True, color=PALETTE[idx], ax=ax, alpha=0.7)
        ax.axvline(df[col].mean(), color='white', linestyle='--', linewidth=1.2, label=f'Mean: {df[col].mean():.1f}')
        ax.axvline(df[col].median(), color=PALETTE[-1], linestyle=':', linewidth=1.2, label=f'Median: {df[col].median():.1f}')
        ax.set_title(f'{label} Score')
        ax.set_xlabel('Score'); ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    # Overall avg in last panel
    ax = axes[5]
    sns.histplot(df['avg_score'], bins=25, kde=True, color='#f59e0b', ax=ax, alpha=0.7)
    ax.axvline(df['avg_score'].mean(), color='white', linestyle='--', linewidth=1.2)
    ax.set_title('Overall Average Score')
    ax.set_xlabel('Score'); ax.set_ylabel('Count')
    fig.tight_layout()
    _save(fig, '01_score_distributions')


# ── 2. Correlation Heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    numeric = df[['study_hours_per_day', 'attendance_pct', 'sleep_hours',
                   'math_score', 'science_score', 'english_score',
                   'history_score', 'cs_score', 'avg_score']]
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, ax=ax, annot_kws={'size': 9})
    ax.set_title('Pearson Correlation Heatmap', fontsize=14)
    fig.tight_layout()
    _save(fig, '02_correlation_heatmap')


# ── 3. Study Hours vs Avg Score (Scatter) ────────────────────────────────────
def plot_study_vs_score(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter
    ax = axes[0]
    scatter = ax.scatter(df['study_hours_per_day'], df['avg_score'],
                         c=df['attendance_pct'], cmap='viridis', alpha=0.6, s=25)
    m, b = np.polyfit(df['study_hours_per_day'], df['avg_score'], 1)
    x_line = np.linspace(df['study_hours_per_day'].min(), df['study_hours_per_day'].max(), 100)
    ax.plot(x_line, m * x_line + b, color='#ec4899', linewidth=2, label=f'Trend (r={df["study_hours_per_day"].corr(df["avg_score"]):.2f})')
    plt.colorbar(scatter, ax=ax, label='Attendance %')
    ax.set_xlabel('Study Hours/Day'); ax.set_ylabel('Avg Score')
    ax.set_title('Study Hours vs Average Score\n(color = attendance %)')
    ax.legend()

    # Attendance vs Score
    ax = axes[1]
    ax.scatter(df['attendance_pct'], df['avg_score'],
               c=df['study_hours_per_day'], cmap='plasma', alpha=0.6, s=25)
    m2, b2 = np.polyfit(df['attendance_pct'], df['avg_score'], 1)
    x2 = np.linspace(df['attendance_pct'].min(), df['attendance_pct'].max(), 100)
    ax.plot(x2, m2 * x2 + b2, color='#06b6d4', linewidth=2, label=f'Trend (r={df["attendance_pct"].corr(df["avg_score"]):.2f})')
    ax.set_xlabel('Attendance %'); ax.set_ylabel('Avg Score')
    ax.set_title('Attendance % vs Average Score\n(color = study hours)')
    ax.legend()
    fig.tight_layout()
    _save(fig, '03_study_attendance_vs_score')


# ── 4. Grade Distribution (Pie + Bar) ────────────────────────────────────────
def plot_grade_distribution(df: pd.DataFrame):
    grade_counts = df['grade'].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Pie
    wedges, texts, autotexts = axes[0].pie(
        grade_counts, labels=grade_counts.index,
        autopct='%1.1f%%', colors=PALETTE[:len(grade_counts)],
        startangle=140, pctdistance=0.8,
        wedgeprops=dict(edgecolor='#0d1117', linewidth=2)
    )
    for t in autotexts: t.set_fontsize(10); t.set_color('white')
    axes[0].set_title('Grade Distribution (Pie)')

    # Bar
    bars = axes[1].bar(grade_counts.index, grade_counts.values,
                       color=PALETTE[:len(grade_counts)], edgecolor='#0d1117', linewidth=1.5)
    for bar, val in zip(bars, grade_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                     str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Grade'); axes[1].set_ylabel('Number of Students')
    axes[1].set_title('Grade Distribution (Count)')
    fig.tight_layout()
    _save(fig, '04_grade_distribution')


# ── 5. Gender Comparison (Grouped Bar) ────────────────────────────────────────
def plot_gender_comparison(df: pd.DataFrame):
    subjects = ['math_score', 'science_score', 'english_score', 'history_score', 'cs_score']
    labels   = ['Math', 'Science', 'English', 'History', 'CS']
    gender_means = df.groupby('gender')[subjects].mean()

    x = np.arange(len(labels)); width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, gender_means.loc['Female'], width, label='Female', color=PALETTE[2], alpha=0.85)
    bars2 = ax.bar(x + width/2, gender_means.loc['Male'],   width, label='Male',   color=PALETTE[0], alpha=0.85)
    for bar in list(bars1) + list(bars2):
        ax.annotate(f'{bar.get_height():.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Average Score'); ax.set_title('Subject-wise Score Comparison by Gender')
    ax.legend(); ax.set_ylim(0, 100)
    fig.tight_layout()
    _save(fig, '05_gender_comparison')


# ── 6. Boxplot — Score by Grade ───────────────────────────────────────────────
def plot_boxplot_by_grade(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    order = ['F', 'D', 'C', 'B', 'A']
    present = [g for g in order if g in df['grade'].values]
    palette_map = dict(zip(present, PALETTE))
    sns.boxplot(data=df, x='grade', y='avg_score', order=present,
                palette=palette_map, ax=ax, linewidth=1.5, fliersize=3)
    ax.set_xlabel('Grade'); ax.set_ylabel('Avg Score')
    ax.set_title('Average Score Distribution by Grade (Boxplot)')
    fig.tight_layout()
    _save(fig, '06_boxplot_score_by_grade')


# ── 7. Heatmap — Study Hours × Attendance Avg Score ──────────────────────────
def plot_2d_heatmap(df: pd.DataFrame):
    df2 = df.copy()
    df2['study_bin'] = pd.cut(df2['study_hours_per_day'], bins=5, precision=0)
    df2['att_bin']   = pd.cut(df2['attendance_pct'],      bins=5, precision=0)
    pivot = df2.pivot_table(values='avg_score', index='att_bin', columns='study_bin', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Avg Score'})
    ax.set_title('Avg Score by Study Hours × Attendance Band')
    ax.set_xlabel('Study Hours per Day (bin)'); ax.set_ylabel('Attendance % (bin)')
    fig.tight_layout()
    _save(fig, '07_heatmap_study_attendance_grid')


# ── 8. Parent Education Impact ────────────────────────────────────────────────
def plot_parent_education(df: pd.DataFrame):
    order = ['No Education', 'High School', 'Bachelor', 'Master', 'PhD']
    fig, ax = plt.subplots(figsize=(12, 6))
    means = df.groupby('parent_education')['avg_score'].mean().reindex(order)
    bars = ax.bar(means.index, means.values, color=PALETTE, edgecolor='#0d1117', linewidth=1.5)
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xlabel('Parent Education Level'); ax.set_ylabel('Average Score')
    ax.set_title('Impact of Parent Education on Student Avg Score')
    ax.set_ylim(0, 100); fig.tight_layout()
    _save(fig, '08_parent_education_impact')


# ── 9. Violin — Score by Internet Access ──────────────────────────────────────
def plot_internet_violin(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x='internet_access', y='avg_score',
                   palette={'Yes': PALETTE[0], 'No': PALETTE[2]}, ax=ax, inner='quartile')
    ax.set_xlabel('Internet Access'); ax.set_ylabel('Avg Score')
    ax.set_title('Score Distribution by Internet Access (Violin)')
    fig.tight_layout()
    _save(fig, '09_internet_access_violin')


# ── 10. Summary Dashboard (Multi-panel) ───────────────────────────────────────
def plot_summary_dashboard(df: pd.DataFrame):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Student Performance Analysis — Summary Dashboard', fontsize=18, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1 — avg score histogram
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(df['avg_score'], bins=20, kde=True, color=PALETTE[0], ax=ax1)
    ax1.axvline(df['avg_score'].mean(), color='white', linestyle='--', linewidth=1.5, label=f"Mean {df['avg_score'].mean():.1f}")
    ax1.set_title('Avg Score Distribution'); ax1.set_xlabel('Score'); ax1.legend(fontsize=8)

    # Panel 2 — grade pie
    ax2 = fig.add_subplot(gs[0, 1])
    gc = df['grade'].value_counts().sort_index()
    ax2.pie(gc, labels=gc.index, autopct='%1.1f%%', colors=PALETTE, startangle=140, pctdistance=0.82)
    ax2.set_title('Grade Distribution')

    # Panel 3 — gender bar
    ax3 = fig.add_subplot(gs[0, 2])
    df.groupby('gender')['avg_score'].mean().plot(kind='bar', ax=ax3, color=[PALETTE[2], PALETTE[0]], edgecolor='#0d1117')
    ax3.set_title('Avg Score by Gender')
    ax3.set_xlabel(''); ax3.set_ylabel('Avg Score'); ax3.set_ylim(0, 100)
    ax3.tick_params(axis='x', rotation=0)

    # Panel 4 — study hours scatter
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(df['study_hours_per_day'], df['avg_score'], alpha=0.4, s=15, color=PALETTE[1])
    m, b = np.polyfit(df['study_hours_per_day'], df['avg_score'], 1)
    x_l = np.linspace(df['study_hours_per_day'].min(), df['study_hours_per_day'].max(), 100)
    ax4.plot(x_l, m * x_l + b, color=PALETTE[2], linewidth=2)
    ax4.set_xlabel('Study Hours/Day'); ax4.set_ylabel('Avg Score')
    ax4.set_title('Study Hours vs Score')

    # Panel 5 — attendance vs score
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(df['attendance_pct'], df['avg_score'], alpha=0.4, s=15, color=PALETTE[3])
    m2, b2 = np.polyfit(df['attendance_pct'], df['avg_score'], 1)
    x2 = np.linspace(df['attendance_pct'].min(), df['attendance_pct'].max(), 100)
    ax5.plot(x2, m2 * x2 + b2, color=PALETTE[4], linewidth=2)
    ax5.set_xlabel('Attendance %'); ax5.set_ylabel('Avg Score')
    ax5.set_title('Attendance vs Score')

    # Panel 6 — subject means bar
    ax6 = fig.add_subplot(gs[1, 2])
    subj_means = df[['math_score','science_score','english_score','history_score','cs_score']].mean()
    subj_means.index = ['Math','Science','English','History','CS']
    bars = ax6.bar(subj_means.index, subj_means.values, color=PALETTE, edgecolor='#0d1117')
    for bar, v in zip(bars, subj_means.values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{v:.1f}',
                 ha='center', fontsize=9, fontweight='bold')
    ax6.set_title('Avg Score by Subject'); ax6.set_ylabel('Score'); ax6.set_ylim(0, 100)

    _save(fig, '10_summary_dashboard')


# ── Run all ────────────────────────────────────────────────────────────────────
def generate_all_charts(df: pd.DataFrame):
    print("\n🎨 GENERATING VISUALIZATIONS")
    plot_score_distribution(df)
    plot_correlation_heatmap(df)
    plot_study_vs_score(df)
    plot_grade_distribution(df)
    plot_gender_comparison(df)
    plot_boxplot_by_grade(df)
    plot_2d_heatmap(df)
    plot_parent_education(df)
    plot_internet_violin(df)
    plot_summary_dashboard(df)
    print(f"\n✅ All charts saved to → {OUT_DIR}/")


if __name__ == "__main__":
    import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from analysis import load_data, clean_data
    df = clean_data(load_data())
    generate_all_charts(df)
