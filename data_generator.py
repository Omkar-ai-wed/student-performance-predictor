"""
data_generator.py
=================
Generates a reproducible synthetic dataset for the Student Performance Predictor.

Run directly to create sample CSV and JSON files:
    python data_generator.py
"""

import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_STUDENTS   = 500          # total rows
MISSING_HW_RATE = 0.15      # 15 % of homework_completion_rate values are NaN


def generate_dataset(n: int = N_STUDENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Produce a synthetic DataFrame with realistic student-performance data.

    Feature distributions
    ---------------------
    attendance_rate         : Beta(8, 2)  → skewed toward high attendance
    weekly_study_hours      : Normal(12, 4) clipped to [1, 40]
    past_exam_scores        : Normal(68, 12) clipped to [0, 100]
    homework_completion_rate: Beta(7, 2) × 100, with 15 % missing
    final_grade (target)    : deterministic formula + Gaussian noise
    """
    rng = np.random.default_rng(seed)

    attendance_rate          = rng.beta(8, 2, n) * 100                      # 0–100
    weekly_study_hours       = rng.normal(12, 4, n).clip(1, 40)
    past_exam_scores         = rng.normal(68, 12, n).clip(0, 100)
    homework_completion_rate = rng.beta(7, 2, n) * 100                      # 0–100

    # Target: weighted combination + interaction + noise
    noise       = rng.normal(0, 3, n)
    final_grade = (
        0.30 * attendance_rate
        + 0.25 * (past_exam_scores / 100 * 100)
        + 0.25 * weekly_study_hours * (100 / 40)   # scale to 0–100 contribution
        + 0.20 * homework_completion_rate
        + noise
    ).clip(0, 100)

    df = pd.DataFrame({
        "attendance_rate":          attendance_rate.round(2),
        "weekly_study_hours":       weekly_study_hours.round(2),
        "past_exam_scores":         past_exam_scores.round(2),
        "homework_completion_rate": homework_completion_rate.round(2),
        "final_grade":              final_grade.round(2),
    })

    # Introduce realistic missingness in homework_completion_rate
    missing_idx = rng.choice(n, size=int(n * MISSING_HW_RATE), replace=False)
    df.loc[missing_idx, "homework_completion_rate"] = np.nan

    return df


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, path: str = "data/students.csv") -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[✓] CSV saved → {path}  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# JSON schema helper
# ---------------------------------------------------------------------------
JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "StudentRecord",
    "type": "object",
    "required": ["attendance_rate", "weekly_study_hours", "past_exam_scores"],
    "properties": {
        "attendance_rate": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": "Percentage of classes attended (0–100)."
        },
        "weekly_study_hours": {
            "type": "number",
            "minimum": 0,
            "description": "Average study hours per week."
        },
        "past_exam_scores": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": "Average score on all prior exams (0–100)."
        },
        "homework_completion_rate": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 100,
            "description": "Percentage of homework submitted (0–100). Optional – may be null."
        },
        "final_grade": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 100,
            "description": "Target: final course grade percentage. Null at inference time."
        }
    }
}


def save_json_schema(path: str = "data/schema.json") -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(JSON_SCHEMA, f, indent=2)
    print(f"[✓] JSON schema saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = generate_dataset()
    print(df.describe().round(2))
    save_csv(df)
    save_json_schema()
    print("\nSample rows:")
    print(df.head(5).to_string(index=False))
