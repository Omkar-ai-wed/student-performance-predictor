"""
model_training.py
=================
Trains and compares multiple regression models for student grade prediction.

Models covered
--------------
Phase 1  – LinearRegression (baseline)
Phase 2a – Ridge / Lasso (regularized linear)
Phase 2b – RandomForestRegressor, GradientBoostingRegressor (tree-based)

Usage example:
    python model_training.py
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split

# ---------------------------------------------------------------------------
# Train/test split helper
# ---------------------------------------------------------------------------
def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified random split of (X, y) into train and test portions.

    Parameters
    ----------
    X, y        : full feature matrix and target vector
    test_size   : fraction reserved for testing (default 20 %)
    random_state: reproducibility seed

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(
        f"[✓] Split → train: {len(X_train)} samples | test: {len(X_test)} samples"
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Phase 1 – Baseline linear regression
# ---------------------------------------------------------------------------
def train_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LinearRegression:
    """
    Fit a plain OLS (Ordinary Least Squares) linear regression — the Phase-1
    baseline.  No regularisation; all features treated with equal weight.

    When to use
    -----------
    Always start here. Provides a performance floor and interpretable
    coefficients that reveal which features drive the prediction most.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[✓] Trained  LinearRegression (baseline).")
    return model


# ---------------------------------------------------------------------------
# Phase 2a – Regularised linear models
# ---------------------------------------------------------------------------
def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
) -> Ridge:
    """
    Ridge regression (L2 penalty).

    When to use
    -----------
    When multicollinearity exists between features (e.g. attendance and study
    hours are correlated).  Ridge shrinks all coefficients but keeps them
    non-zero — good when most features are believed to contribute.

    alpha : regularisation strength (larger → stronger shrinkage).
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    print(f"[✓] Trained  Ridge    (alpha={alpha}).")
    return model


def train_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 0.1,
) -> Lasso:
    """
    Lasso regression (L1 penalty).

    When to use
    -----------
    When you suspect only a subset of features truly matter.  Lasso drives
    irrelevant coefficients to exactly zero, performing automatic feature
    selection.

    alpha : regularisation strength.
    """
    model = Lasso(alpha=alpha, max_iter=5000)
    model.fit(X_train, y_train)
    non_zero = (model.coef_ != 0).sum()
    print(f"[✓] Trained  Lasso    (alpha={alpha}) → {non_zero} non-zero coefficients.")
    return model


# ---------------------------------------------------------------------------
# Phase 2b – Tree-based models
# ---------------------------------------------------------------------------
def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Random Forest regressor — ensemble of decision trees.

    When to use
    -----------
    • When relationships between features and the target are non-linear
      (e.g. attendance below a threshold causes sharp grade drop-off).
    • When you have enough data (≥ 300 samples) for trees to generalise.
    • Built-in feature-importance scores are useful for explainability.

    Key trade-offs vs linear models
    --------------------------------
    + Handles non-linearity and interactions automatically.
    + Robust to outliers and noisy features.
    – Less interpretable (black box).
    – Can overfit on small datasets (use cross-validation).
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"[✓] Trained  RandomForestRegressor  (n_estimators={n_estimators}).")
    return model


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators:  int   = 200,
    learning_rate: float = 0.05,
    max_depth:     int   = 3,
    random_state:  int   = 42,
) -> GradientBoostingRegressor:
    """
    Gradient Boosting regressor — sequential ensemble that minimises residuals.

    When to use
    -----------
    • Usually the strongest out-of-the-box performer when data is tabular.
    • When you care about top predictive accuracy more than interpretability.
    • Works well even with moderate feature collinearity.

    Hyper-parameter guidance
    ------------------------
    n_estimators  : more trees → lower bias but slower; 100–500 is typical.
    learning_rate : smaller value + more trees = better generalisation.
    max_depth     : shallow trees (2–4) reduce overfitting.
    """
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    print(
        f"[✓] Trained  GradientBoostingRegressor "
        f"(n={n_estimators}, lr={learning_rate}, depth={max_depth})."
    )
    return model


# ---------------------------------------------------------------------------
# Cross-validation helper (Phase 2 optional)
# ---------------------------------------------------------------------------
def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv:      int = 5,
    scoring: str = "neg_root_mean_squared_error",
) -> dict:
    """
    Run k-fold cross-validation and return mean / std of the metric.

    Parameters
    ----------
    model   : unfitted scikit-learn estimator
    X, y    : full feature matrix and targets (NOT split beforehand)
    cv      : number of folds
    scoring : scikit-learn scoring string

    Returns
    -------
    dict with keys: 'scores', 'mean', 'std'
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    # For neg-RMSE, flip sign
    if scoring.startswith("neg_"):
        scores = -scores
    result = {"scores": scores, "mean": scores.mean(), "std": scores.std()}
    metric_name = scoring.replace("neg_", "")
    print(
        f"[✓] {cv}-fold CV  {metric_name}: "
        f"{result['mean']:.4f} ± {result['std']:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")


def save_model(model, name: str) -> str:
    """Persist a fitted model to disk using joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"[✓] Model saved → {path}")
    return str(path)


def load_model(name: str):
    """Load a previously saved model from disk."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model found at {path}")
    model = joblib.load(path)
    print(f"[✓] Model loaded ← {path}")
    return model


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data_generator import generate_dataset
    from preprocessing import build_pipeline

    df  = generate_dataset()
    X, y, _, pipe = build_pipeline(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train all models
    lr  = train_linear(X_train, y_train)
    rdg = train_ridge(X_train, y_train, alpha=1.0)
    lso = train_lasso(X_train, y_train, alpha=0.1)
    rf  = train_random_forest(X_train, y_train)
    gb  = train_gradient_boosting(X_train, y_train)

    # Save models
    save_model(lr,  "linear_regression")
    save_model(rdg, "ridge")
    save_model(rf,  "random_forest")
    save_model(gb,  "gradient_boosting")

    # Cross-validation demo on full data (linear baseline)
    print("\n--- 5-fold CV (LinearRegression) ---")
    cross_validate_model(LinearRegression(), X, y, cv=5)
