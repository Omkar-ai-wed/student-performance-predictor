"""
hyperparameter_tuning.py
========================
Phase 2 – Grid search and randomized search for Ridge, Lasso, RandomForest,
and GradientBoosting. Demonstrates best-practice CV-based tuning.

Usage:
    python hyperparameter_tuning.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error


# ---------------------------------------------------------------------------
# Scorer helper (RMSE)
# ---------------------------------------------------------------------------
def rmse_scorer():
    """Return a negative-RMSE scorer for sklearn grid search."""
    def _neg_rmse(y_true, y_pred):
        return -np.sqrt(mean_squared_error(y_true, y_pred))
    return make_scorer(_neg_rmse)


# ---------------------------------------------------------------------------
# Grid search helpers
# ---------------------------------------------------------------------------
def tune_ridge(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> Ridge:
    """
    Exhaustive grid search over Ridge alpha values.

    Grid: alpha ∈ {0.01, 0.1, 1, 10, 100}
    Reasonable range: smaller alpha → closer to OLS; larger alpha → stronger L2 penalty.
    """
    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    gs = GridSearchCV(
        Ridge(),
        param_grid,
        scoring=rmse_scorer(),
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    print(f"[✓] Ridge best params : {gs.best_params_}  |  CV RMSE: {-gs.best_score_:.4f}")
    return gs.best_estimator_


def tune_lasso(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> Lasso:
    """
    Exhaustive grid search over Lasso alpha values.

    Grid: alpha ∈ {0.001, 0.01, 0.05, 0.1, 0.5}
    High alpha drives many coefficients to zero; use LassoCV for finer search.
    """
    param_grid = {"alpha": [0.001, 0.01, 0.05, 0.1, 0.5]}
    gs = GridSearchCV(
        Lasso(max_iter=5000),
        param_grid,
        scoring=rmse_scorer(),
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    non_zero = (gs.best_estimator_.coef_ != 0).sum()
    print(
        f"[✓] Lasso best params : {gs.best_params_}  |  "
        f"CV RMSE: {-gs.best_score_:.4f}  |  non-zero coefs: {non_zero}"
    )
    return gs.best_estimator_


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter:  int = 20,
    cv:      int = 3,
) -> RandomForestRegressor:
    """
    Randomised search over RandomForest hyperparameters.

    Key parameters
    --------------
    n_estimators  : more trees → lower variance, but slower training.
    max_depth     : limit to avoid memorising noise; None = grow until pure.
    min_samples_leaf: minimum observations per leaf; higher → smoother model.
    max_features  : subset of features per split; 'sqrt' is standard default.
    """
    param_dist = {
        "n_estimators":    [50, 100, 200, 300],
        "max_depth":       [None, 5, 10, 20],
        "min_samples_leaf":[1, 2, 5, 10],
        "max_features":    ["sqrt", "log2", 0.5],
    }
    rs = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_dist,
        n_iter=n_iter,
        scoring=rmse_scorer(),
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    rs.fit(X_train, y_train)
    print(f"[✓] RF   best params : {rs.best_params_}  |  CV RMSE: {-rs.best_score_:.4f}")
    return rs.best_estimator_


def tune_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter:  int = 20,
    cv:      int = 3,
) -> GradientBoostingRegressor:
    """
    Randomised search over GradientBoosting hyperparameters.

    Key parameters
    --------------
    n_estimators   : number of boosting stages; more = lower bias, risk overfit.
    learning_rate  : step size per stage; small lr + many stages = best results.
    max_depth      : depth of individual trees; 3–5 is typical for boosting.
    subsample      : fraction of training data used per stage (< 1 = stochastic GB).
    min_samples_leaf: minimum examples at a leaf.
    """
    param_dist = {
        "n_estimators":     [100, 200, 300, 500],
        "learning_rate":    [0.01, 0.03, 0.05, 0.1],
        "max_depth":        [2, 3, 4, 5],
        "subsample":        [0.7, 0.8, 1.0],
        "min_samples_leaf": [1, 2, 5],
    }
    rs = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_dist,
        n_iter=n_iter,
        scoring=rmse_scorer(),
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    rs.fit(X_train, y_train)
    print(f"[✓] GB   best params : {rs.best_params_}  |  CV RMSE: {-rs.best_score_:.4f}")
    return rs.best_estimator_


# ---------------------------------------------------------------------------
# Run all tuning
# ---------------------------------------------------------------------------
def run_full_tuning(X_train, y_train) -> dict:
    """
    Tune all four model types and return a dict of best estimators.
    May take 2–5 minutes on 400 training samples.
    """
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING (this may take a few minutes)")
    print("=" * 60)
    return {
        "Ridge (tuned)":            tune_ridge(X_train, y_train),
        "Lasso (tuned)":            tune_lasso(X_train, y_train),
        "RandomForest (tuned)":     tune_random_forest(X_train, y_train),
        "GradientBoosting (tuned)": tune_gradient_boosting(X_train, y_train),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_generator import generate_dataset
    from preprocessing  import build_pipeline
    from evaluation     import compare_models

    df = generate_dataset(n=500)
    X, y, feat_cols, _ = build_pipeline(df, fit=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    best_models = run_full_tuning(X_tr, y_tr)
    compare_models(best_models, X_te, y_te)
