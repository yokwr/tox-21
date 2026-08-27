"""Metrics, and the null baselines each one has to be read against.

Every metric here is reported next to the score a trivial predictor achieves on
the *same* test set. Accuracy and PR-AUC both move with the class balance, so a
number quoted without its floor is not interpretable -- and the two splits in
this benchmark have different base rates (0.147 random vs 0.231 scaffold), so
their raw scores are not comparable to each other either.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Standard binned ECE: mean |accuracy - confidence| weighted by bin population."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def enrichment(y: np.ndarray, scores: np.ndarray, frac: float = 0.10) -> dict:
    """Top-fraction enrichment: the metric a screening campaign actually cares about.

    A screen does not apply a 0.5 threshold; it takes the top N and tests them.
    Enrichment is (hit rate in the top slice) / (hit rate of picking at random).
    """
    n = len(y)
    n_top = max(1, int(np.ceil(frac * n)))
    top = np.argsort(scores)[::-1][:n_top]
    actives_total = int(y.sum())
    actives_top = int(y[top].sum())
    expected = actives_total * n_top / n
    return {
        "frac": frac,
        "n_top": n_top,
        "actives_top": actives_top,
        "expected_actives_if_random": round(expected, 1),
        "enrichment_x": actives_top / expected if expected > 0 else float("nan"),
        "precision_at_frac": actives_top / n_top,
        "recall_at_frac": actives_top / actives_total if actives_total else float("nan"),
    }


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """Full metric set for one set of predicted probabilities."""
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    pred = (p >= threshold).astype(float)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    return {
        "n": int(len(y)),
        "n_actives": int(y.sum()),
        "base_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "recall_at_0.5": tp / (tp + fn) if (tp + fn) else float("nan"),
        "precision_at_0.5": tp / (tp + fp) if (tp + fp) else float("nan"),
        "enrichment_top10pct": enrichment(y, p, 0.10),
    }


def null_baselines(y: np.ndarray) -> dict:
    """What a model has to beat before any of its scores mean anything.

    - majority class: predict the commoner label for everything
    - constant base rate: predict P(active) = prevalence for everything
    - random ranker: PR-AUC of a random ordering equals the base rate
    """
    y = np.asarray(y, dtype=float).ravel()
    br = float(y.mean())
    return {
        "majority_class_accuracy": max(br, 1 - br),
        "random_ranker_pr_auc": br,
        "random_ranker_roc_auc": 0.5,
        "constant_base_rate_brier": br * (1 - br),
        "random_ranker_enrichment_x": 1.0,
    }


def margin_over_null(result: dict, nulls: dict) -> dict:
    """The only numbers worth putting in a summary: model minus floor."""
    return {
        "accuracy_over_majority_pp": 100
        * (result["accuracy"] - nulls["majority_class_accuracy"]),
        "pr_auc_over_random_x": result["pr_auc"] / nulls["random_ranker_pr_auc"],
        "brier_vs_base_rate": result["brier"] - nulls["constant_base_rate_brier"],
        "enrichment_x": result["enrichment_top10pct"]["enrichment_x"],
    }
