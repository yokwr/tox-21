# tox21-sr-are-benchmark

An evaluation harness for the Tox21 SR-ARE assay (5,825 compounds, 942 actives).

The model is a calibrated random forest on ECFP4 fingerprints. It is deliberately
ordinary. The point of this repo is the evaluation around it: scaffold splits,
null baselines for every metric, a permutation control, and an applicability
domain with thresholds derived from the training data.

Methodology, provenance and dataset defects are in [DATA_CARD.md](DATA_CARD.md).
Full numbers are in [results/tables.md](results/tables.md), generated from
`results/results.json`.

## Running

```bash
pip install -r requirements.txt
python -m tox21_bench.run_benchmark --data-dir data --out results
python -m tox21_bench.make_tables --results results
pytest
```

Versions are pinned in `requirements.txt`. An earlier XGBoost-based version of
this work reported ROC-AUC 0.7650; the same inputs on a newer release of the
library gave 0.7697.

## Results

### Accuracy against null baselines

Scaffold split. Majority-class accuracy on this test set is 0.769.

| Model | Accuracy | vs majority | Recall @0.5 | Enrichment @10% |
|---|---|---|---|---|
| Calibrated RF, ECFP4 | 0.778 | +0.86 pp | 0.074 | 2.07x |
| Calibrated RF, 6 descriptors | 0.767 | -0.17 pp | 0.052 | 2.04x |

The descriptor model scores below a constant predictor. The fingerprint model
identifies 7% of actives at a 0.5 threshold, but ranks at 2.07x enrichment with
47.9% precision in the top decile. It works as a ranker and not as a classifier,
and accuracy shows neither.

### Effect of the split

Featuriser held fixed at ECFP4, only the split changes.

| Split | Test compounds sharing a scaffold with train | ROC-AUC | PR-AUC vs random | Enrichment @10% |
|---|---|---|---|---|
| Random (seed 42) | 927 / 1,165 (79.6%) | 0.763 | 2.68x | 2.97x |
| Bemis-Murcko scaffold | 0 (0%) | 0.708 | 1.89x | 2.07x |

Base rates differ between the two test sets (0.187 vs 0.231), so only the margins
over each split's own null are comparable, not the raw scores.

### Applicability domain

Maximum Tanimoto similarity (ECFP4) to any training compound. Cutoffs are
percentiles of the training set's own leave-one-out nearest-neighbour
distribution rather than round numbers.

| Band | n | PR-AUC vs random | ECE |
|---|---|---|---|
| in-domain (sim >= 0.50) | 305 | 2.85x | 0.040 |
| borderline | 439 | 1.70x | 0.063 |
| out-of-domain (sim < 0.33) | 421 | 1.33x | 0.074 |

The aggregate 1.89x averages over a model that ranks well on familiar chemistry
and close to randomly on unfamiliar chemistry. Calibration error roughly doubles
across the same range, so out-of-domain predictions are both worse and more
confident. A similarity-quartile breakdown in `results/tables.md` shows the same
trend without using the thresholds at all.

### Permutation null

Pipeline retrained on permuted training labels, 20 times, scaffold split, ECFP4.

| | Observed | Permutation null | Nominal floor |
|---|---|---|---|
| ROC-AUC | 0.692 | 0.544 +/- 0.023 | 0.500 |
| PR-AUC | 0.424 | 0.256 +/- 0.017 | 0.231 (base rate) |

Both nulls sit above their nominal floors and the null ROC-AUC never fell below
0.496 across the 20 runs. The observed result clears either reference (z = 6.3
and 10.0, empirical p = 0), but margins quoted against 0.5 and against the base
rate are optimistic by roughly 0.04 and 0.03 respectively.

Why the null is displaced is untested. One possibility is that a forest fitted to
permuted labels still encodes local training-set density, and density correlates
with activity here. That is a guess.

## Notes on the split

Bemis-Murcko maps acyclic molecules to the empty scaffold, which collects 1,513
compounds (26% of the dataset). Benzene collects another 1,224. The DeepChem
splitter fills training largest-group-first, so both go to training, and all
1,165 test compounds end up in singleton scaffold classes. Train base rate is
0.144, test is 0.231.

Scaffold-split scores here are therefore a novel-chemotype estimate rather than
an average-case one.

Six InChIKey-skeleton groups carry conflicting SR-ARE labels. Five are
protonation-state or stereochemistry duplicates (pyruvate/pyruvic acid,
2-phenylphenol/phenolate, selenite/selenous acid, and two others), which are the
same species at assay pH. None are split across train and test here. Details in
[DATA_CARD.md](DATA_CARD.md) section 2.

`splits.scaffold_split` reimplements `deepchem.splits.ScaffoldSplitter` so the
DeepChem dependency can be dropped. A regression test pins it to the published
partition (4,660 / 1,165, 896 negatives, 269 actives). The split is deterministic
and takes no seed.

## Layout

```
src/tox21_bench/
  data.py            loading, ECFP4 + descriptor featurisation, InChIKey duplicate audit
  splits.py          scaffold and random splits, overlap and composition diagnostics
  metrics.py         metrics, null baselines, ECE, enrichment
  models.py          calibrated RF, shuffled-label control, single-feature rankers
  applicability.py   Tanimoto AD, derived thresholds, stratified evaluation
  run_benchmark.py   entrypoint, writes results/results.json
  make_tables.py     results.json -> results/tables.md
tests/               17 tests
app/app.py           demo UI, shows the domain band alongside every score
data/                frozen SMILES and SR-ARE labels
```

## Limitations

The main untested question is whether the model detects SR-ARE activity or
general cytotoxicity. Tox21 stress-response panels ship with paired viability
readouts and no counter-screen control is run here, so the supported claim is
that the model ranks SR-ARE-labelled compounds above others, and nothing
stronger.

Also: one assay, one split per design, no hyperparameter search, no confidence
intervals on the headline numbers, and assay noise is unquantified so the
distance to the achievable ceiling is unknown. See
[DATA_CARD.md](DATA_CARD.md) section 8.
