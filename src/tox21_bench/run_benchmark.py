"""Run the full benchmark and write results/results.json.

    python -m tox21_bench.run_benchmark --data-dir data --out results

Every number in README.md and DATA_CARD.md comes from this script.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from . import applicability as ad
from . import data as D
from . import metrics as M
from . import models as Mo
from . import splits as S

DESIGNS = {
    "scaffold": lambda smiles: S.scaffold_split(smiles, frac_train=0.8),
    "random": lambda smiles: S.random_split(len(smiles), frac_train=0.8, seed=42),
}
FEATURES = {"ecfp4": D.ecfp4, "descriptors": D.descriptors}


def _versions() -> dict:
    import sklearn
    import rdkit
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "rdkit": rdkit.__version__,
    }


def main(data_dir: str, out_dir: str) -> dict:
    t0 = time.time()
    ds = D.load_local(data_dir)
    smiles, y = ds.smiles, ds.y

    results: dict = {
        "dataset": {
            "n": len(ds),
            "n_actives": int(y.sum()),
            "base_rate": ds.base_rate,
            "assay": D.ASSAY,
        },
        "duplicates": D.duplicate_report(smiles, y),
        "versions": _versions(),
        "designs": {},
    }

    feats = {name: fn(smiles) for name, fn in FEATURES.items()}

    for design_name, splitter in DESIGNS.items():
        tr, te = splitter(smiles)
        y_tr, y_te = y[tr], y[te]
        block: dict = {
            "n_train": len(tr),
            "n_test": len(te),
            "train_base_rate": float(y_tr.mean()),
            "test_base_rate": float(y_te.mean()),
            "scaffold_overlap": S.scaffold_overlap(smiles, tr, te),
            "scaffold_composition": S.scaffold_composition(smiles, te),
            "nulls": M.null_baselines(y_te),
            "models": {},
        }

        for feat_name, X in feats.items():
            print(f"[{time.time()-t0:6.1f}s] {design_name} / {feat_name}: fitting")
            p = Mo.fit_predict(Mo.calibrated_rf(), X[tr], y_tr, X[te])
            res = M.evaluate(y_te, p)
            res["margin_over_null"] = M.margin_over_null(res, block["nulls"])
            block["models"][f"calibrated_rf__{feat_name}"] = res

            if feat_name == "ecfp4" and design_name == "scaffold":
                np.save(Path(out_dir) / "scaffold_ecfp4_test_probs.npy", p)
                block["_probs_for_ad"] = True

        # --- controls, on ECFP4 ---
        print(f"[{time.time()-t0:6.1f}s] {design_name}: shuffled-label control")
        p_shuf = Mo.shuffled_label_control(
            Mo.calibrated_rf, feats["ecfp4"][tr], y_tr, feats["ecfp4"][te]
        )
        block["controls"] = {
            "shuffled_labels__ecfp4": M.evaluate(y_te, p_shuf),
        }
        for i, name in enumerate(D.DESCRIPTOR_NAMES):
            p1 = Mo.single_feature_ranker(feats["descriptors"][te], i)
            block["controls"][f"single_descriptor__{name}"] = {
                "roc_auc": float(M.evaluate(y_te, p1)["roc_auc"]),
                "pr_auc": float(M.evaluate(y_te, p1)["pr_auc"]),
            }

        results["designs"][design_name] = block

    # --- applicability domain, on the scaffold split + ECFP4 model ---
    print(f"[{time.time()-t0:6.1f}s] applicability domain")
    tr, te = DESIGNS["scaffold"](smiles)
    tr_smiles = [smiles[i] for i in tr]
    te_smiles = [smiles[i] for i in te]
    p = np.load(Path(out_dir) / "scaffold_ecfp4_test_probs.npy")

    self_sim = ad.training_self_similarity(tr_smiles)
    thresholds = ad.derive_thresholds(self_sim)
    sims = ad.max_similarity_to_reference(te_smiles, tr_smiles)

    results["applicability_domain"] = {
        "thresholds": thresholds,
        "test_similarity_summary": {
            "mean": float(sims.mean()),
            "median": float(np.median(sims)),
            "p10": float(np.percentile(sims, 10)),
            "p90": float(np.percentile(sims, 90)),
        },
        "band_counts": {
            b: int((ad.assign_domain(sims, thresholds) == b).sum())
            for b in ("in_domain", "borderline", "out_of_domain")
        },
        "by_band": ad.stratify(y[te], p, sims, thresholds),
        "by_similarity_quartile": ad.similarity_quartiles(y[te], p, sims),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"[{time.time()-t0:6.1f}s] wrote {out/'results.json'}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    main(a.data_dir, a.out)
