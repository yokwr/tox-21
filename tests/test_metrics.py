"""Tests for the metric and null-baseline layer.

These mostly pin down the properties that make the numbers interpretable: that
the nulls really are what a trivial predictor scores, and that enrichment behaves
sanely at the extremes.
"""
import numpy as np
import pytest

from tox21_bench import applicability as ad
from tox21_bench import metrics as M


@pytest.fixture
def imbalanced():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.2).astype(float)
    return y


def test_null_baselines_match_trivial_predictors(imbalanced):
    y = imbalanced
    nulls = M.null_baselines(y)
    # majority-class classifier
    assert nulls["majority_class_accuracy"] == pytest.approx(1 - y.mean())
    # constant base-rate predictor's Brier score
    p = np.full_like(y, y.mean())
    assert nulls["constant_base_rate_brier"] == pytest.approx(
        np.mean((p - y) ** 2), rel=1e-9
    )


def test_random_ranker_pr_auc_equals_base_rate():
    """PR-AUC has a floor that moves with prevalence. This is why a PR-AUC quoted
    on its own says nothing, and why the two splits here are not comparable."""
    rng = np.random.default_rng(1)
    for rate in (0.05, 0.2, 0.5):
        y = (rng.random(20000) < rate).astype(float)
        scores = rng.random(20000)
        assert M.evaluate(y, scores)["pr_auc"] == pytest.approx(y.mean(), abs=0.02)


def test_perfect_and_inverted_rankers():
    y = np.array([0, 0, 1, 1, 0, 1.0])
    assert M.evaluate(y, y)["roc_auc"] == 1.0
    assert M.evaluate(y, 1 - y)["roc_auc"] == 0.0


def test_enrichment_bounds():
    y = np.zeros(100)
    y[:10] = 1
    perfect = np.concatenate([np.ones(10), np.zeros(90)])
    res = M.enrichment(y, perfect, frac=0.10)
    assert res["precision_at_frac"] == 1.0
    assert res["enrichment_x"] == pytest.approx(10.0)

    flat = np.zeros(100)  # no ranking information
    assert M.enrichment(y, flat, frac=0.10)["enrichment_x"] <= 10.0


def test_ece_zero_for_perfectly_calibrated_constant():
    y = np.concatenate([np.ones(200), np.zeros(800)])
    p = np.full(1000, 0.2)
    assert M.expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_overconfidence():
    y = np.concatenate([np.ones(200), np.zeros(800)])
    p = np.concatenate([np.full(200, 0.9), np.full(800, 0.9)])
    assert M.expected_calibration_error(y, p) > 0.5


def test_thresholds_are_derived_from_the_reference_distribution():
    self_sim = np.linspace(0.0, 1.0, 1001)
    th = ad.derive_thresholds(self_sim, in_pct=25.0, out_pct=5.0)
    assert th["in_domain_min_similarity"] == pytest.approx(0.25, abs=0.01)
    assert th["out_of_domain_max_similarity"] == pytest.approx(0.05, abs=0.01)


def test_assign_domain_bands_are_ordered():
    th = {"in_domain_min_similarity": 0.5, "out_of_domain_max_similarity": 0.3}
    bands = ad.assign_domain(np.array([0.9, 0.4, 0.1]), th)
    assert list(bands) == ["in_domain", "borderline", "out_of_domain"]


def test_max_similarity_of_identical_molecule_is_one():
    sims = ad.max_similarity_to_reference(["c1ccccc1O"], ["CCCC", "c1ccccc1O"])
    assert sims[0] == pytest.approx(1.0)
