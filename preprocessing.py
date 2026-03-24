"""
preprocessing.py
================
Data loading, quality checks, missing-value imputation, feature engineering,
and scaling for the Student Performance Predictor.

Usage example (standalone):
    from preprocessing import build_pipeline, load_data
    df = load_data("data/students.csv")
    X_scaled, y, pipeline = build_pipeline(df, fit=True)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
REQUIRED_COLS = ["attendance_rate", "weekly_study_hours", "past_exam_scores"]
OPTIONAL_COLS = ["homework_completion_rate"]
TARGET_COL    = "final_grade"
ALL_FEATURE_COLS = REQUIRED_COLS + OPTIONAL_COLS


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV file and validate its schema.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError  : if the CSV file doesn't exist.
    ValueError         : if required columns are missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(p)
    print(f"[✓] Loaded {len(df)} rows from '{path}'")

    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Raise if required columns are absent."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from dataset: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    print(f"[✓] Schema validation passed.  Optional cols present: "
          f"{[c for c in OPTIONAL_COLS if c in df.columns]}")


# ---------------------------------------------------------------------------
# 2. Data-quality report
# ---------------------------------------------------------------------------
def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print and return a per-column data-quality summary.

    Checks:
    - dtype
    - null count / null %
    - out-of-range values for known bounded features
    - duplicate rows
    """
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    bounds = {
        "attendance_rate":          (0, 100),
        "weekly_study_hours":       (0, 168),  # max hours/week is 168
        "past_exam_scores":         (0, 100),
        "homework_completion_rate": (0, 100),
        "final_grade":              (0, 100),
    }

    rows = []
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct   = null_count / len(df) * 100
        lo, hi     = bounds.get(col, (None, None))
        if lo is not None:
            out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        else:
            out_of_range = 0
        rows.append({
            "column":       col,
            "dtype":        str(df[col].dtype),
            "null_count":   null_count,
            "null_pct":     f"{null_pct:.1f}%",
            "out_of_range": out_of_range,
        })

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))

    dup_rows = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup_rows}")
    print("=" * 60 + "\n")

    return report


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features before scaling.

    New features
    ------------
    study_attendance_interaction :  weekly_study_hours × attendance_rate / 100
        Captures students who study AND attend; a synergy effect.
    avg_academic_performance     : mean of past_exam_scores and attendance_rate
        A single composite measure of historical diligence.
    effort_score                 : (weekly_study_hours / 40) × 100
        Normalises study hours to a 0–100 effort percentage.
    """
    df = df.copy()

    df["study_attendance_interaction"] = (
        df["weekly_study_hours"] * df["attendance_rate"] / 100
    ).round(4)

    df["avg_academic_performance"] = (
        (df["past_exam_scores"] + df["attendance_rate"]) / 2
    ).round(4)

    df["effort_score"] = (df["weekly_study_hours"] / 40 * 100).clip(0, 100).round(4)

    return df


# ---------------------------------------------------------------------------
# 4. scikit-learn preprocessing pipeline
# ---------------------------------------------------------------------------
def make_sklearn_pipeline() -> Pipeline:
    """
    Return a scikit-learn Pipeline that:
      1. Imputes missing values with the column median (handles optional HW col).
      2. Standardises all features (zero mean, unit variance).
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


# ---------------------------------------------------------------------------
# 5. Full build_pipeline helper (convenience entry-point)
# ---------------------------------------------------------------------------
def build_pipeline(
    df:        pd.DataFrame,
    fit:       bool = True,
    pipeline:  Pipeline | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], Pipeline]:
    """
    Run the complete preprocessing sequence:
        load → validate → engineer features → impute → scale.

    Parameters
    ----------
    df       : raw DataFrame (must include all REQUIRED_COLS and TARGET_COL)
    fit      : if True, fit_transform the pipeline; else only transform.
    pipeline : optional pre-fitted pipeline to reuse during inference.

    Returns
    -------
    X_processed : np.ndarray  – scaled feature matrix
    y           : np.ndarray  – target vector (empty array if target absent)
    feature_names : list[str] – ordered list of feature column names
    pipe        : fitted Pipeline
    """
    # Quality report (informational only)
    data_quality_report(df)

    # --- Feature engineering ---
    df_eng = engineer_features(df)

    # Determine features present (optional col may be absent in inference)
    engineered_extras = [
        "study_attendance_interaction",
        "avg_academic_performance",
        "effort_score",
    ]
    feature_cols = [
        c for c in (ALL_FEATURE_COLS + engineered_extras) if c in df_eng.columns
    ]

    X_raw = df_eng[feature_cols].values

    # --- Target ---
    if TARGET_COL in df_eng.columns:
        y = df_eng[TARGET_COL].values
    else:
        y = np.array([])

    # --- Impute + scale ---
    if pipeline is None:
        pipeline = make_sklearn_pipeline()

    if fit:
        X_processed = pipeline.fit_transform(X_raw)
    else:
        X_processed = pipeline.transform(X_raw)

    print(f"[✓] Preprocessing complete. Feature matrix shape: {X_processed.shape}")
    return X_processed, y, feature_cols, pipeline


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_generator import generate_dataset

    df_raw = generate_dataset(n=200, seed=0)
    X, y, feat_cols, pipe = build_pipeline(df_raw, fit=True)
    print(f"Features used : {feat_cols}")
    print(f"X shape       : {X.shape}")
    print(f"y shape       : {y.shape}")
    print(f"X mean (≈0)   : {X.mean(axis=0).round(4)}")
    print(f"X std  (≈1)   : {X.std(axis=0).round(4)}")
