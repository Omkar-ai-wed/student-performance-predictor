"""
main.py
=======
End-to-end orchestration for the Student Performance Predictor.

Running this file executes the FULL pipeline in three phases:
  Phase 1 : Generate data → Preprocess → Train baseline (LinearRegression)
  Phase 2 : Regularised + tree models, cross-validation, comparison table
  Phase 3 : Save best model + pipeline, demo inference

Usage:
    python main.py [--phase {1,2,all}] [--data path/to/data.csv]

Dependencies:
    pip install -r requirements.txt
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")
from data_generator  import generate_dataset, save_csv, save_json_schema
from evaluation      import compare_models, print_feature_importance, residual_summary
from inference       import predict_batch, predict_single
from model_training  import (
    cross_validate_model,
    save_model,
    split_data,
    train_gradient_boosting,
    train_linear,
    train_lasso,
    train_random_forest,
    train_ridge,
)
from preprocessing   import build_pipeline, REQUIRED_COLS, OPTIONAL_COLS

MODELS_DIR = Path("models")
DATA_CSV   = "data/students.csv"


# ---------------------------------------------------------------------------
# Phase 1 – MVP baseline
# ---------------------------------------------------------------------------
def run_phase1(df, verbose: bool = True) -> tuple:
    """
    Phase 1: Essential preprocessing + Linear Regression baseline.

    Steps
    -----
    1. Preprocess (impute + scale + feature engineering)
    2. Train/test split (80/20)
    3. Fit LinearRegression
    4. Evaluate on held-out test set

    Returns
    -------
    (X_train, X_test, y_train, y_test, feature_cols, pipe, lr_model)
    """
    print("\n" + "▓" * 60)
    print("  PHASE 1 — MVP BASELINE  (LinearRegression)")
    print("▓" * 60)

    X, y, feat_cols, pipe = build_pipeline(df, fit=True)
    X_tr, X_te, y_tr, y_te = split_data(X, y)

    lr = train_linear(X_tr, y_tr)

    print("\n--- Phase 1 Test-Set Metrics ---")
    compare_models({"LinearRegression (baseline)": lr}, X_te, y_te)

    return X_tr, X_te, y_tr, y_te, feat_cols, pipe, lr


# ---------------------------------------------------------------------------
# Phase 2 – Enhanced models
# ---------------------------------------------------------------------------
def run_phase2(X_tr, X_te, y_tr, y_te, X_full, y_full, feat_cols) -> dict:
    """
    Phase 2: Regularised linear + tree-based models with cross-validation.

    Steps
    -----
    1. Train Ridge, Lasso, RandomForest, GradientBoosting
    2. Run 5-fold cross-validation on each (full dataset)
    3. Compare all models on the held-out test set
    4. Display feature importances for the best tree model
    5. Residual diagnostics for the best model

    Returns
    -------
    dict of {model_name: fitted_estimator}
    """
    print("\n" + "▓" * 60)
    print("  PHASE 2 — ENHANCED MODELS")
    print("▓" * 60)

    rdg = train_ridge(X_tr, y_tr, alpha=1.0)
    lso = train_lasso(X_tr, y_tr, alpha=0.1)
    rf  = train_random_forest(X_tr, y_tr, n_estimators=200)
    gb  = train_gradient_boosting(X_tr, y_tr, n_estimators=200, learning_rate=0.05)

    # --- Cross-validation (on full dataset; informational) ---
    print("\n--- 5-Fold Cross-Validation (full dataset) ---")
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble      import RandomForestRegressor, GradientBoostingRegressor

    for name, est in [
        ("Ridge",            Ridge(alpha=1.0)),
        ("Lasso",            Lasso(alpha=0.1, max_iter=5000)),
        ("RandomForest",     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42)),
    ]:
        cross_validate_model(est, X_full, y_full, cv=5)

    # --- Comparison table ---
    all_models = {
        "Ridge":            rdg,
        "Lasso":            lso,
        "RandomForest":     rf,
        "GradientBoosting": gb,
    }
    print("\n--- Phase 2 Test-Set Comparison ---")
    report = compare_models(all_models, X_te, y_te)

    # --- Feature importances ---
    print_feature_importance(gb, feat_cols)

    # --- Residuals for best model ---
    best_name = report.iloc[0]["label"]
    best_model = all_models[best_name]
    residual_summary(y_te, best_model.predict(X_te), label=best_name)

    return all_models


# ---------------------------------------------------------------------------
# Phase 3 – Save models and demo inference
# ---------------------------------------------------------------------------
def run_phase3(
    best_model,
    pipe,
    feat_cols: list[str],
    model_name: str = "gradient_boosting",
) -> None:
    """
    Phase 3: Persist best model + pipeline; run inference demo.

    Files written
    -------------
    models/<model_name>.joblib
    models/preprocessing_pipeline.joblib
    models/feature_cols.json
    """
    print("\n" + "▓" * 60)
    print("  PHASE 3 — DEPLOYMENT")
    print("▓" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    save_model(best_model, model_name)

    # Save preprocessing pipeline
    pipe_path = MODELS_DIR / "preprocessing_pipeline.joblib"
    joblib.dump(pipe, pipe_path)
    print(f"[✓] Pipeline saved → {pipe_path}")

    # Save feature column list (needed for inference alignment)
    meta_path = MODELS_DIR / "feature_cols.json"
    with open(meta_path, "w") as f:
        json.dump(feat_cols, f)
    print(f"[✓] Feature list saved → {meta_path}")

    # --- Inference demo ---
    print("\n--- Inference Demo ---")
    res = predict_single(
        best_model, pipe, feat_cols,
        attendance_rate=82.0,
        weekly_study_hours=10.5,
        past_exam_scores=68.0,
        homework_completion_rate=85.0,
    )
    print(f"  Student A → Predicted grade: {res['predicted_grade']} %")

    res2 = predict_single(
        best_model, pipe, feat_cols,
        attendance_rate=55.0,
        weekly_study_hours=4.0,
        past_exam_scores=50.0,
        # homework_completion_rate omitted (will be imputed)
    )
    print(f"  Student B (missing HW) → Predicted grade: {res2['predicted_grade']} %")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Student Performance Predictor – full pipeline"
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "all"],
        default="all",
        help="Which phase(s) to run (default: all)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to existing CSV (default: generate synthetic data)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  STUDENT PERFORMANCE PREDICTOR")
    print("=" * 60)

    # ── Data loading / generation ──────────────────────────────────────────
    if args.data:
        from preprocessing import load_data
        df = load_data(args.data)
    else:
        print("\n[*] No data provided – generating synthetic dataset …")
        df = generate_dataset(n=500, seed=42)
        save_csv(df, DATA_CSV)
        save_json_schema()

    # ── Phase 1 ────────────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te, feat_cols, pipe, lr = run_phase1(df)
    X_full, y_full = np.vstack([X_tr, X_te]), np.concatenate([y_tr, y_te])

    if args.phase == "1":
        print("\n[Done] Phase 1 complete.")
        return

    # ── Phase 2 ────────────────────────────────────────────────────────────
    all_models = run_phase2(X_tr, X_te, y_tr, y_te, X_full, y_full, feat_cols)

    if args.phase == "2":
        print("\n[Done] Phase 2 complete.")
        return

    # ── Phase 3 ────────────────────────────────────────────────────────────
    # Use GradientBoosting as the production model
    best_model = all_models["GradientBoosting"]
    run_phase3(best_model, pipe, feat_cols, model_name="gradient_boosting")

    print("\n" + "=" * 60)
    print("  ALL PHASES COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
