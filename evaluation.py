"""
evaluation.py
=============
Model evaluation utilities for the Student Performance Predictor.

Metrics computed
----------------
- RMSE  – Root Mean Squared Error (primary; penalises large errors more)
- MAE   – Mean Absolute Error     (robust; in same units as grade %)
- R²    – Coefficient of Determination (fraction of variance explained)

Ethical note: Always disaggregate metrics by subgroups (if available) to
detect hidden bias.  A model accurate on average may still systematically
mis-predict for specific cohorts.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label:  str = "Model",
) -> dict:
    """
    Compute RMSE, MAE, and R² for a set of predictions.

    Parameters
    ----------
    y_true : ground-truth target values
    y_pred : predicted values from the model
    label  : descriptive name shown in the printed report

    Returns
    -------
    dict with keys: 'label', 'rmse', 'mae', 'r2'
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    metrics = {"label": label, "rmse": rmse, "mae": mae, "r2": r2}
    _print_metrics(metrics)
    return metrics


def _print_metrics(m: dict) -> None:
    print(
        f"  [{m['label']:30s}]  "
        f"RMSE={m['rmse']:.4f}  "
        f"MAE={m['mae']:.4f}  "
        f"R²={m['r2']:.4f}"
    )


# ---------------------------------------------------------------------------
# Multi-model comparison table
# ---------------------------------------------------------------------------
def compare_models(
    models:  dict,              # {"model_name": fitted_estimator, ...}
    X_test:  np.ndarray,
    y_test:  np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate all provided models on the same test set and return a
    ranked comparison DataFrame.

    Parameters
    ----------
    models  : dict mapping model label → fitted scikit-learn estimator
    X_test  : scaled feature matrix (test split)
    y_test  : ground-truth target (test split)

    Returns
    -------
    pd.DataFrame sorted by RMSE (ascending = better first)
    """
    print("\n" + "=" * 65)
    print("MODEL COMPARISON TABLE")
    print("=" * 65)

    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        m = compute_metrics(y_test, y_pred, label=name)
        rows.append(m)

    df_report = (
        pd.DataFrame(rows)
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    print("\nRanked by RMSE (lower is better):")
    print(df_report.to_string(index=False))
    print("=" * 65 + "\n")
    return df_report


# ---------------------------------------------------------------------------
# Residual summary (for diagnostics)
# ---------------------------------------------------------------------------
def residual_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label:  str = "Model",
) -> pd.DataFrame:
    """
    Return descriptive statistics of residuals (y_true − y_pred).

    Large skew or heavy tails in residuals may indicate non-linearity
    not captured by the model.
    """
    residuals = y_true - y_pred
    summary = {
        "model":  label,
        "mean":   residuals.mean(),
        "median": np.median(residuals),
        "std":    residuals.std(),
        "min":    residuals.min(),
        "max":    residuals.max(),
        "p5":     np.percentile(residuals, 5),
        "p95":    np.percentile(residuals, 95),
    }
    df = pd.DataFrame([summary])
    print(f"\nResidual summary [{label}]:")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Feature importance (tree models only)
# ---------------------------------------------------------------------------
def print_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """
    Print sorted feature importances for tree-based estimators.
    Returns a DataFrame for downstream use.
    """
    if not hasattr(model, "feature_importances_"):
        print("[!] Model does not expose feature_importances_. Skipping.")
        return pd.DataFrame()

    df = (
        pd.DataFrame({
            "feature":    feature_names,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("\nFeature Importances:")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Ethical fairness stub
# ---------------------------------------------------------------------------
def fairness_check(
    df_test:        pd.DataFrame,
    y_pred:         np.ndarray,
    protected_col:  str,
) -> pd.DataFrame:
    """
    Compute per-group RMSE to surface potential disparate impact.

    Parameters
    ----------
    df_test       : original (unscaled) test DataFrame with the protected column
    y_pred        : model predictions (aligned with df_test rows)
    protected_col : column name of demographic group (e.g. 'gender', 'cohort')

    Returns
    -------
    pd.DataFrame with RMSE per subgroup.

    Note: This is a *minimum viable* fairness check.  Production systems
    should use dedicated libraries (Fairlearn, AIF360) for full audits.
    """
    if protected_col not in df_test.columns:
        print(f"[!] Protected column '{protected_col}' not found. Skipping fairness check.")
        return pd.DataFrame()

    df_check = df_test.copy()
    df_check["_pred"] = y_pred

    rows = []
    for group, subdf in df_check.groupby(protected_col):
        rmse = np.sqrt(mean_squared_error(subdf["final_grade"], subdf["_pred"]))
        rows.append({"group": group, "n": len(subdf), "rmse": round(rmse, 4)})

    result = pd.DataFrame(rows)
    print(f"\nFairness breakdown by '{protected_col}':")
    print(result.to_string(index=False))
    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data_generator import generate_dataset
    from model_training import (
        split_data,
        train_gradient_boosting,
        train_linear,
        train_random_forest,
        train_ridge,
    )
    from preprocessing import build_pipeline

    df                   = generate_dataset()
    X, y, feat_cols, _   = build_pipeline(df, fit=True)
    X_tr, X_te, y_tr, y_te = split_data(X, y)

    lr  = train_linear(X_tr, y_tr)
    rdg = train_ridge(X_tr,  y_tr)
    rf  = train_random_forest(X_tr, y_tr)
    gb  = train_gradient_boosting(X_tr, y_tr)

    models = {
        "LinearRegression": lr,
        "Ridge":            rdg,
        "RandomForest":     rf,
        "GradientBoosting": gb,
    }
    report = compare_models(models, X_te, y_te)

    # Residuals for the best model
    y_pred_gb = gb.predict(X_te)
    residual_summary(y_te, y_pred_gb, label="GradientBoosting")

    # Feature importances
    print_feature_importance(rf, feat_cols)
