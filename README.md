# Tox21 SR-ARE: a leakage-audited benchmark

A small benchmark built to answer one question honestly: **does a model trained on
the Tox21 SR-ARE assay actually work, and on what?**

The modelling here is deliberately ordinary — a calibrated random forest on ECFP4
fingerprints. The work is in the evaluation: leakage-safe splits, null baselines
for every metric, a permutation control, and a validated applicability domain.
Several of the findings are about the *benchmark* rather than the model.

Full methodology, provenance and known defects: **[DATA_CARD.md](DATA_CARD.md)**.
All numbers: **[results/tables.md](results/tables.md)**, regenerated from
`results/results.json` rather than typed by hand.

---

## Findings

### 1. Accuracy on this task is worse than uninformative

On the scaffold split, a classifier that predicts "inactive" for everything scores
**0.769**.

| Model | Accuracy | vs majority | Recall @0.5 |
|---|---|---|---|
| Calibrated RF, ECFP4 | 0.778 | **+0.86 pp** | 0.074 |
| Calibrated RF, 6 descriptors | 0.767 | **−0.17 pp** | 0.052 |

The descriptor model is *worse than a constant predictor*, and the fingerprint
model finds 7% of actives. Yet the same fingerprint model reaches **2.07x
enrichment at 47.9% precision** in the top decile — genuinely useful for ranking a
library. Accuracy communicates the opposite of both facts.

### 2. Random splits leak, and here is the mechanism and the size of it

Holding the featuriser fixed at ECFP4 and changing only the split:

| Split | Test compounds sharing a scaffold with train | ROC-AUC | PR-AUC vs random | Enrichment @10% |
|---|---|---|---|---|
| Random (seed 42) | **927 / 1,165 (79.6%)** | 0.763 | 2.68x | 2.97x |
| Bemis–Murcko scaffold | **0 (0%)** | 0.708 | 1.89x | 2.07x |

An earlier iteration of this project compared a random-split descriptor model
against a scaffold-split fingerprint model and read the gap as leakage. That
comparison changed two things at once and could not support the conclusion. The
table above changes one.

### 3. The applicability domain is real, and the aggregate score hides it

Maximum Tanimoto similarity to the training set, with cutoffs derived from the
training set's own leave-one-out nearest-neighbour distribution (not hand-picked):

| Band | n | PR-AUC vs random | ECE |
|---|---|---|---|
| in-domain (sim ≥ 0.50) | 305 | **2.85x** | 0.040 |
| borderline | 439 | 1.70x | 0.063 |
| out-of-domain (sim < 0.33) | 421 | **1.33x** | 0.074 |

The headline 1.89x is an average over a model that works on familiar chemistry and
one that barely beats random on unfamiliar chemistry. **Calibration error nearly
doubles in the same direction** — so the model is not merely less accurate
out-of-domain, it is more confidently wrong there. A threshold-free
similarity-quartile breakdown reproduces the same monotonic trend, so the
conclusion does not depend on where the cutoffs were placed.

### 4. The permutation null is not centred at 0.5

Retraining the pipeline on permuted training labels, 20 times:

| | Observed | Permutation null | Textbook floor |
|---|---|---|---|
| ROC-AUC | 0.692 | **0.544 ± 0.023** | 0.500 |
| PR-AUC | 0.424 | **0.256 ± 0.017** | 0.231 (base rate) |

Both nulls sit meaningfully above their textbook floors, and across 20 permutations
the null ROC-AUC never dropped below 0.496. Scoring against 0.5 and against the
base rate would overstate the result on both metrics. The observed result survives
either way (z = 6.3 and 10.0, empirical p = 0), but the margin is smaller than the
naive comparison suggests.

A single permutation is not a control — it is one draw from a distribution whose
spread turns out to be non-trivial. An earlier iteration of this project ran
exactly one shuffle, got 0.553, and had no way to tell whether that indicated a
leak or was ordinary variation. It was ordinary variation around a displaced null.

**Why the null is displaced is not established here.** The plausible route is that
a forest trained on destroyed labels still encodes local training-set density, and
density is not independent of activity in this dataset. That is a hypothesis, not
a result, and it is the second-most interesting open question in the repo.

### 5. Bemis–Murcko does not do what its name suggests on this dataset

| | |
|---|---|
| Compounds mapping to the **empty** (acyclic) scaffold | **1,513 (26%)** |
| Compounds on the benzene scaffold | 1,224 (21%) |
| Scaffolds appearing exactly once | 1,202 of 1,576 |
| **Test compounds sitting in singleton scaffold classes** | **1,165 of 1,165** |

Murcko strips side chains to the ring system, so every acyclic molecule collapses
into one pseudo-group sharing nothing but the absence of rings. Two "groups" hold
47% of the data, the splitter sends both to train wholesale, and the test set ends
up composed entirely of scaffold singletons — the structural tail, not a random
sample of harder chemistry. Class balance shifts from 0.144 in train to 0.231 in test.

The scaffold-split score is therefore a **novel-chemotype worst case**, not "the
honest number". That distinction matters when quoting it.

### 6. Six compounds are labelled against themselves

Five of the six InChIKey-skeleton groups with conflicting SR-ARE labels are
protonation-state or stereo-annotation duplicates — pyruvate vs pyruvic acid,
2-phenylphenol vs its phenolate, selenite vs selenous acid — the same species at
assay pH with opposite labels. They cap achievable accuracy. None cross the
train/test boundary in this split; under a random split they would leak directly.
Details in [DATA_CARD.md §2.1](DATA_CARD.md).

---

## Layout

```
src/tox21_bench/
  data.py            load, featurise (ECFP4 + descriptors), InChIKey duplicate audit
  splits.py          scaffold + random splits, overlap and composition diagnostics
  metrics.py         metrics, null baselines, ECE, enrichment
  models.py          calibrated RF, shuffled-label control, single-feature rankers
  applicability.py   Tanimoto AD, derived thresholds, stratified evaluation
  run_benchmark.py   entrypoint -> results/results.json
  make_tables.py     results.json -> results/tables.md
tests/               17 tests, including a regression test pinning the split
app/app.py           demo UI that shows the domain alongside every score
data/                frozen SMILES + SR-ARE labels
DATA_CARD.md         provenance, defects, splits, metric rationale, limitations
```

`splits.scaffold_split` reimplements `deepchem.splits.ScaffoldSplitter` and is
pinned to its published partition (4,660 / 1,165, 896 negatives / 269 actives) by
`tests/test_splits.py::test_matches_published_deepchem_split`. This drops the
DeepChem dependency, which pins an old RDKit and does not install cleanly on
current Python.

Note that a scaffold split is **deterministic** — it takes no seed. Code passing
`seed=` to a scaffold splitter is passing an argument that does nothing.

---

## Reproducing

```bash
pip install -r requirements.txt
python -m tox21_bench.run_benchmark --data-dir data --out results
python -m tox21_bench.make_tables --results results
pytest
```

Versions are pinned because the numbers move without them: an earlier
XGBoost-based run of this project reported ROC-AUC 0.7650, which came back as
0.7697 on a newer release of the same library with identical inputs.

---

## What this does not establish

The biggest open question is a **counter-screen control**. Tox21 stress-response
panels ship with paired viability readouts, and nothing here tests whether the
model is detecting SR-ARE activity specifically or general cytotoxicity. Until
that is run, "the model finds SR-ARE actives" is not supported — only "the model
ranks SR-ARE-labelled compounds above others."

Other limits — single assay, single split per design, no hyperparameter search,
no confidence intervals on the headline numbers, unquantified assay noise — are in
[DATA_CARD.md §8](DATA_CARD.md).
