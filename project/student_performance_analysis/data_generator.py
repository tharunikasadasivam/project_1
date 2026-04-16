"""
data_generator.py
Generates a realistic synthetic student performance dataset.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_student_data(n=500):
    subjects         = ['Math', 'Science', 'English', 'History', 'Computer Science']
    study_hours_base = np.random.normal(5, 2, n).clip(1, 12)
    attendance_base  = np.random.normal(78, 12, n).clip(40, 100)
    sleep_hours      = np.random.normal(7, 1.2, n).clip(4, 10)

    def make_score(base_hours, base_att, noise=8):
        score = (base_hours * 4) + (base_att * 0.3) + np.random.normal(0, noise, n)
        return score.clip(0, 100).round(1)

    data = {
        'student_id'          : [f'STU{str(i).zfill(4)}' for i in range(1, n + 1)],
        'gender'              : np.random.choice(['Male', 'Female'], n, p=[0.48, 0.52]),
        'age'                 : np.random.randint(15, 20, n),
        'study_hours_per_day' : study_hours_base.round(1),
        'attendance_pct'      : attendance_base.round(1),
        'sleep_hours'         : sleep_hours.round(1),
        'extra_curricular'    : np.random.choice(['Yes', 'No'], n, p=[0.4, 0.6]),
        'internet_access'     : np.random.choice(['Yes', 'No'], n, p=[0.85, 0.15]),
        'parent_education'    : np.random.choice(
            ['No Education', 'High School', 'Bachelor', 'Master', 'PhD'], n,
            p=[0.05, 0.35, 0.40, 0.15, 0.05]
        ),
        'math_score'          : make_score(study_hours_base, attendance_base),
        'science_score'       : make_score(study_hours_base, attendance_base, 9),
        'english_score'       : make_score(study_hours_base, attendance_base, 7),
        'history_score'       : make_score(study_hours_base, attendance_base, 10),
        'cs_score'            : make_score(study_hours_base, attendance_base, 8),
    }

    df = pd.DataFrame(data)
    df['avg_score']  = df[['math_score','science_score','english_score','history_score','cs_score']].mean(axis=1).round(2)
    df['grade']      = pd.cut(df['avg_score'],
                              bins=[0, 40, 55, 70, 85, 100],
                              labels=['F', 'D', 'C', 'B', 'A'])
    df['pass_fail']  = df['avg_score'].apply(lambda x: 'Pass' if x >= 40 else 'Fail')
    return df


if __name__ == '__main__':
    df = generate_student_data()
    df.to_csv('data/student_performance.csv', index=False)
    print(f"✅ Dataset generated: {len(df)} students, {len(df.columns)} features")
    print(df.head())
