"""Render results/results.json into results/tables.md.

Keeping this separate from run_benchmark means the tables in README.md are
generated from the stored results rather than retyped, so they cannot drift
away from the run that produced them.

    python -m tox21_bench.make_tables --results results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _f(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


def headline_table(r: dict) -> str:
    rows = [
        "| Split | Features | Accuracy | vs majority | PR-AUC | vs random | ROC-AUC | Brier | ECE | Recall @0.5 | Enrichment @10% |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for split in ("random", "scaffold"):
        d = r["designs"][split]
        for key, res in d["models"].items():
            feat = key.split("__")[1]
            m = res["margin_over_null"]
            rows.append(
                f"| {split} | {feat} | {_f(res['accuracy'])} | "
                f"{m['accuracy_over_majority_pp']:+.2f} pp | {_f(res['pr_auc'])} | "
                f"{m['pr_auc_over_random_x']:.2f}x | {_f(res['roc_auc'])} | "
                f"{_f(res['brier'])} | {_f(res['ece'])} | {_f(res['recall_at_0.5'])} | "
                f"{res['enrichment_top10pct']['enrichment_x']:.2f}x |"
            )
    return "\n".join(rows)


def nulls_table(r: dict) -> str:
    rows = [
        "| Split | Test n | Base rate | Majority-class accuracy | Random-ranker PR-AUC | Base-rate Brier |",
        "|---|---|---|---|---|---|",
    ]
    for split in ("random", "scaffold"):
        d = r["designs"][split]
        n = d["nulls"]
        rows.append(
            f"| {split} | {d['n_test']} | {_f(d['test_base_rate'])} | "
            f"{_f(n['majority_class_accuracy'])} | {_f(n['random_ranker_pr_auc'])} | "
            f"{_f(n['constant_base_rate_brier'])} |"
        )
    return "\n".join(rows)


def ad_table(r: dict) -> str:
    a = r["applicability_domain"]
    rows = [
        "| Band | n | Mean similarity to train | Base rate | PR-AUC | vs random | ROC-AUC | Brier | ECE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for band in ("in_domain", "borderline", "out_of_domain"):
        v = a["by_band"][band]
        if "note" in v:
            rows.append(f"| {band} | {v['n']} | - | - | - | - | - | - | {v['note']} |")
            continue
        rows.append(
            f"| {band} | {v['n']} | {_f(v['mean_similarity_to_train'])} | "
            f"{_f(v['base_rate'])} | {_f(v['pr_auc'])} | "
            f"{v['pr_auc'] / v['base_rate']:.2f}x | {_f(v['roc_auc'])} | "
            f"{_f(v['brier'])} | {_f(v['ece'])} |"
        )
    return "\n".join(rows)


def quartile_table(r: dict) -> str:
    rows = [
        "| Similarity quartile | Range | n | Base rate | PR-AUC | vs random | ECE |",
        "|---|---|---|---|---|---|---|",
    ]
    for q in r["applicability_domain"]["by_similarity_quartile"]:
        if "note" in q:
            continue
        lo, hi = q["similarity_range"]
        rows.append(
            f"| Q{q['quartile']} | {lo:.2f}-{hi:.2f} | {q['n']} | {_f(q['base_rate'])} | "
            f"{_f(q['pr_auc'])} | {q['pr_auc_over_random_x']:.2f}x | {_f(q['ece'])} |"
        )
    return "\n".join(rows)


def controls_table(r: dict) -> str:
    rows = ["| Split | Control | ROC-AUC | PR-AUC |", "|---|---|---|---|"]
    for split in ("random", "scaffold"):
        c = r["designs"][split]["controls"]
        s = c["shuffled_labels__ecfp4"]
        rows.append(
            f"| {split} | shuffled training labels (ECFP4) | {_f(s['roc_auc'])} | {_f(s['pr_auc'])} |"
        )
        singles = {
            k.split("__")[1]: v for k, v in c.items() if k.startswith("single_descriptor")
        }
        best = max(singles.items(), key=lambda kv: kv[1]["roc_auc"])
        rows.append(
            f"| {split} | best single descriptor ({best[0]}) | "
            f"{_f(best[1]['roc_auc'])} | {_f(best[1]['pr_auc'])} |"
        )
    return "\n".join(rows)


def main(results_dir: str) -> str:
    p = Path(results_dir)
    r = json.loads((p / "results.json").read_text())
    parts = [
        "# Results\n",
        f"Dataset: {r['dataset']['n']} compounds, {r['dataset']['n_actives']} actives "
        f"(base rate {r['dataset']['base_rate']:.3f}), assay {r['dataset']['assay']}.\n",
        "## What a trivial predictor scores\n",
        nulls_table(r),
        "\n## Headline results\n",
        headline_table(r),
        "\n## Controls\n",
        controls_table(r),
        "\n## Applicability domain (scaffold split, ECFP4)\n",
        ad_table(r),
        "\n### Threshold-free cross-check\n",
        quartile_table(r),
        "",
    ]
    perm = p / "permutation_null.json"
    if perm.exists():
        pn = json.loads(perm.read_text())
        parts.insert(
            -1,
            "\n## Permutation null (scaffold split, ECFP4, 20 label permutations)\n\n"
            f"- observed ROC-AUC {pn['real']['roc_auc']:.4f} vs null "
            f"{pn['null_roc_auc']['mean']:.4f} +/- {pn['null_roc_auc']['sd']:.4f} "
            f"(z = {pn['z_roc']:.1f}, empirical p = {pn['empirical_p_roc']:.2f})\n"
            f"- observed PR-AUC {pn['real']['pr_auc']:.4f} vs null "
            f"{pn['null_pr_auc']['mean']:.4f} +/- {pn['null_pr_auc']['sd']:.4f} "
            f"(z = {pn['z_pr']:.1f}, empirical p = {pn['empirical_p_pr']:.2f})\n",
        )
    text = "\n".join(parts)
    (p / "tables.md").write_text(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    print(main(ap.parse_args().results))
