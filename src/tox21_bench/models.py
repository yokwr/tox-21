"""Models, plus the control models that make the real model's score interpretable."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


def calibrated_rf(seed: int = 42, n_estimators: int = 200) -> CalibratedClassifierCV:
    """Random forest with Platt-scaled probabilities (5-fold internal CV).

    Calibration is fitted inside the training fold only -- CalibratedClassifierCV
    cross-fits, so no test information reaches the calibrator.
    """
    return CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1),
        method="sigmoid",
        cv=5,
    )


def fit_predict(model, X_train, y_train, X_test) -> np.ndarray:
    model.fit(X_train, np.asarray(y_train).ravel())
    return model.predict_proba(X_test)[:, 1]


def shuffled_label_control(
    model_factory, X_train, y_train, X_test, seed: int = 0
) -> np.ndarray:
    """Train the real pipeline on permuted labels.

    Any signal that survives label destruction is coming from the pipeline, not the
    chemistry -- this is the check that catches indexing bugs and leaked features.
    Expected result: ROC-AUC ~= 0.5, PR-AUC ~= base rate.
    """
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(np.asarray(y_train).ravel())
    return fit_predict(model_factory(), X_train, y_shuffled, X_test)


def single_feature_ranker(X_test: np.ndarray, col: int) -> np.ndarray:
    """Rank test compounds by one descriptor, unfitted.

    The cheapest possible 'model'. If a trained model does not clearly beat the best
    of these, it has not learned anything a lookup table could not do.
    """
    x = X_test[:, col].astype(float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
