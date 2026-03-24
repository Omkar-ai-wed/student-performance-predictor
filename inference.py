"""
inference.py
============
Production-ready inference module for the Student Performance Predictor.

Given a trained model + preprocessing pipeline, this module:
  1. Accepts a single student record or a batch (dict / DataFrame / JSON).
  2. Runs the same preprocessing steps used at training time.
  3. Returns predicted final grade(s) as a float (or list of floats).

Usage examples
--------------
  # Single student
  from inference import predict_single
  result = predict_single(
      model, pipe, feat_cols,
      attendance_rate=85,
      weekly_study_hours=12,
      past_exam_scores=72,
      homework_completion_rate=90,
  )
  print(result)  # e.g. {'predicted_grade': 76.43, 'confidence_note': '...'}

  # Batch (DataFrame or list of dicts)
  from inference import predict_batch
  preds = predict_batch(model, pipe, feat_cols, records_df)
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd

from preprocessing import (
    OPTIONAL_COLS,
    REQUIRED_COLS,
    build_pipeline,
    engineer_features,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------
def _validate_input_record(record: dict) -> None:
    """
    Raise ValueError if a required input feature is missing or out of range.

    This is a lightweight guard at inference time; it does NOT replace
    server-side validation in a production API.
    """
    missing = [c for c in REQUIRED_COLS if c not in record]
    if missing:
        raise ValueError(
            f"Missing required features for inference: {missing}. "
            f"Required: {REQUIRED_COLS}"
        )

    bounds = {
        "attendance_rate":          (0, 100),
        "weekly_study_hours":       (0, 168),
        "past_exam_scores":         (0, 100),
        "homework_completion_rate": (0, 100),
    }
    for col, (lo, hi) in bounds.items():
        if col in record and record[col] is not None:
            val = record[col]
            if not (lo <= val <= hi):
                raise ValueError(
                    f"Feature '{col}' value {val} is outside valid range [{lo}, {hi}]."
                )


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------
def _preprocess_for_inference(
    records:      list[dict],
    pipe,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Apply feature engineering + trained pipeline.transform() to raw records.

    Parameters
    ----------
    records      : list of dicts (one per student)
    pipe         : fitted sklearn Pipeline (imputer + scaler)
    feature_cols : ordered list of feature names the model was trained on

    Returns
    -------
    np.ndarray of shape (n_records, n_features)
    """
    df = pd.DataFrame(records)

    # Add optional columns as NaN if absent (will be median-imputed)
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df_eng = engineer_features(df)

    # Align columns exactly to training-time order; fill any extras with NaN
    df_aligned = pd.DataFrame(index=df_eng.index)
    for col in feature_cols:
        df_aligned[col] = df_eng[col] if col in df_eng.columns else np.nan

    return pipe.transform(df_aligned.values)


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------
def predict_single(
    model,
    pipe,
    feature_cols: list[str],
    **kwargs: Any,
) -> dict:
    """
    Predict the final grade for ONE student.

    Parameters
    ----------
    model, pipe, feature_cols : from the trained pipeline
    **kwargs : feature values  e.g. attendance_rate=85, weekly_study_hours=10

    Returns
    -------
    dict : {
        'predicted_grade'  : float (0–100, rounded to 2 dp),
        'input_features'   : dict of raw inputs provided,
        'confidence_note'  : str – a plain-language caveat
    }
    """
    record = dict(kwargs)
    _validate_input_record(record)

    X = _preprocess_for_inference([record], pipe, feature_cols)
    raw_pred = float(model.predict(X)[0])
    predicted_grade = round(np.clip(raw_pred, 0, 100), 2)

    return {
        "predicted_grade": predicted_grade,
        "input_features":  record,
        "confidence_note": (
            "This estimate is based on historical patterns and should be used "
            "as a supplementary indicator only. Individual circumstances vary."
        ),
    }


def predict_batch(
    model,
    pipe,
    feature_cols: list[str],
    records:      pd.DataFrame | list[dict],
) -> pd.DataFrame:
    """
    Predict grades for a batch of students.

    Parameters
    ----------
    model, pipe, feature_cols : from the trained pipeline
    records : pd.DataFrame or list of dicts with feature columns

    Returns
    -------
    pd.DataFrame with original data plus a 'predicted_grade' column.
    """
    if isinstance(records, pd.DataFrame):
        records_list = records.to_dict(orient="records")
        result_df    = records.copy()
    else:
        records_list = records
        result_df    = pd.DataFrame(records)

    for rec in records_list:
        _validate_input_record(rec)

    X = _preprocess_for_inference(records_list, pipe, feature_cols)
    raw_preds = model.predict(X)
    result_df["predicted_grade"] = np.clip(raw_preds, 0, 100).round(2)

    return result_df


# ---------------------------------------------------------------------------
# Load-model convenience wrapper
# ---------------------------------------------------------------------------
def load_production_model(model_name: str = "gradient_boosting", models_dir: str = "models"):
    """
    Load a saved model + pipeline from disk.

    This is the typical entry-point for a deployed inference service.

    Parameters
    ----------
    model_name  : name of the model file (without .joblib extension)
    models_dir  : path to the directory containing model files

    Returns
    -------
    (model, pipe, feature_cols)
    """
    import json
    import joblib
    from pathlib import Path

    mdir   = Path(models_dir)
    model  = joblib.load(mdir / f"{model_name}.joblib")
    pipe   = joblib.load(mdir / "preprocessing_pipeline.joblib")

    meta_path = mdir / "feature_cols.json"
    with open(meta_path) as f:
        feature_cols = json.load(f)

    print(f"[OK] Loaded '{model_name}' model from '{mdir}'")
    return model, pipe, feature_cols


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data_generator import generate_dataset
    from model_training  import split_data, train_gradient_boosting
    from preprocessing   import build_pipeline

    print("=== Inference Demo ===\n")

    # --- Train a quick model for demo purposes ---
    df = generate_dataset(n=300)
    X, y, feat_cols, pipe = build_pipeline(df, fit=True)
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    model = train_gradient_boosting(X_tr, y_tr)

    # --- Single-record inference ---
    result = predict_single(
        model, pipe, feat_cols,
        attendance_rate=88.5,
        weekly_study_hours=14.0,
        past_exam_scores=75.0,
        homework_completion_rate=92.0,
    )
    print("\n--- Single prediction ---")
    print(f"  Predicted grade : {result['predicted_grade']} %")
    print(f"  Confidence note : {result['confidence_note']}")

    # --- Batch inference ---
    sample_batch = df[REQUIRED_COLS + OPTIONAL_COLS].head(5)
    batch_preds  = predict_batch(model, pipe, feat_cols, sample_batch)
    print("\n--- Batch predictions (first 5 students) ---")
    print(batch_preds[["attendance_rate", "past_exam_scores", "predicted_grade"]].to_string(index=False))

    # --- Inference with missing optional feature ---
    result2 = predict_single(
        model, pipe, feat_cols,
        attendance_rate=70.0,
        weekly_study_hours=8.0,
        past_exam_scores=60.0,
        # homework_completion_rate intentionally omitted
    )
    print("\n--- Without optional feature ---")
    print(f"  Predicted grade : {result2['predicted_grade']} %")
