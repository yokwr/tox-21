"""Tanimoto applicability domain, with thresholds derived rather than guessed.

An applicability domain answers: *is this query the kind of molecule the training
set can speak for?* The usual implementation picks round-number similarity cutoffs
(0.5, 0.25) with no justification. Those numbers are not properties of chemistry;
they are properties of how densely the training set happens to cover chemical space,
which differs per dataset and per fingerprint.

Here the cutoffs are read off the training set's own internal density: for each
training compound, its nearest neighbour among the *other* training compounds. If a
query is less similar to the training set than the great majority of training
compounds are to each other, the training set is thin there.

The domain is then *validated*, not asserted: test performance is reported per
similarity band, so the claim "predictions degrade outside the domain" is
something the numbers either support or don't.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")


def _fps(smiles: list[str], radius: int = 2, n_bits: int = 2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"unparseable SMILES: {s!r}")
        out.append(gen.GetFingerprint(mol))
    return out


def max_similarity_to_reference(
    query_smiles: list[str], reference_smiles: list[str]
) -> np.ndarray:
    """For each query, its highest Tanimoto similarity to any reference compound."""
    ref = _fps(reference_smiles)
    qry = _fps(query_smiles)
    return np.array([max(DataStructs.BulkTanimotoSimilarity(q, ref)) for q in qry])


def training_self_similarity(reference_smiles: list[str]) -> np.ndarray:
    """Leave-one-out nearest-neighbour similarity within the training set.

    This is the reference distribution the cutoffs are derived from.
    """
    ref = _fps(reference_smiles)
    out = np.empty(len(ref))
    for i, f in enumerate(ref):
        sims = DataStructs.BulkTanimotoSimilarity(f, ref)
        sims[i] = -1.0  # exclude self
        out[i] = max(sims)
    return out


def derive_thresholds(
    self_sim: np.ndarray, in_pct: float = 25.0, out_pct: float = 5.0
) -> dict:
    """Cutoffs as percentiles of the training set's own nearest-neighbour distribution.

    - in_domain: at or above the `in_pct`-th percentile -- the query sits in a region
      at least as well covered as three quarters of the training set itself.
    - out_of_domain: below the `out_pct`-th percentile -- sparser than all but the
      most isolated 5% of training compounds.

    Percentiles, not similarities, are the tunable choice here, and they are stated
    rather than buried.
    """
    return {
        "in_domain_min_similarity": float(np.percentile(self_sim, in_pct)),
        "out_of_domain_max_similarity": float(np.percentile(self_sim, out_pct)),
        "in_percentile": in_pct,
        "out_percentile": out_pct,
        "training_self_similarity_median": float(np.median(self_sim)),
    }


def assign_domain(sims: np.ndarray, thresholds: dict) -> np.ndarray:
    """Label each query in / borderline / out of domain."""
    lo = thresholds["out_of_domain_max_similarity"]
    hi = thresholds["in_domain_min_similarity"]
    out = np.full(len(sims), "borderline", dtype=object)
    out[sims >= hi] = "in_domain"
    out[sims < lo] = "out_of_domain"
    return out


def stratify(y: np.ndarray, p: np.ndarray, sims: np.ndarray, thresholds: dict) -> dict:
    """Metrics computed separately per domain band.

    Bands with very few actives are reported with their counts so a reader can see
    when a metric is resting on a handful of compounds.
    """
    from .metrics import evaluate, null_baselines

    bands = assign_domain(sims, thresholds)
    out = {}
    for band in ("in_domain", "borderline", "out_of_domain"):
        m = bands == band
        n_act = int(y[m].sum())
        if m.sum() < 20 or n_act < 5 or n_act == m.sum():
            out[band] = {
                "n": int(m.sum()),
                "n_actives": n_act,
                "note": "too few compounds or actives for a stable metric",
            }
            continue
        res = evaluate(y[m], p[m])
        res["nulls"] = null_baselines(y[m])
        res["mean_similarity_to_train"] = float(sims[m].mean())
        out[band] = res
    return out


def similarity_quartiles(y: np.ndarray, p: np.ndarray, sims: np.ndarray) -> list[dict]:
    """A threshold-free view: metrics by quartile of similarity to the training set.

    Useful as a cross-check -- if the trend here matches the banded result, the
    conclusion does not depend on where the cutoffs were placed.
    """
    from .metrics import evaluate

    edges = np.percentile(sims, [0, 25, 50, 75, 100])
    rows = []
    for q in range(4):
        lo, hi = edges[q], edges[q + 1]
        m = (sims >= lo) & (sims <= hi if q == 3 else sims < hi)
        if m.sum() < 20 or y[m].sum() < 5 or y[m].sum() == m.sum():
            rows.append({"quartile": q + 1, "n": int(m.sum()), "note": "too few actives"})
            continue
        res = evaluate(y[m], p[m])
        rows.append(
            {
                "quartile": q + 1,
                "similarity_range": [float(lo), float(hi)],
                "n": res["n"],
                "base_rate": res["base_rate"],
                "pr_auc": res["pr_auc"],
                "roc_auc": res["roc_auc"],
                "brier": res["brier"],
                "ece": res["ece"],
                "pr_auc_over_random_x": res["pr_auc"] / res["base_rate"],
            }
        )
    return rows
