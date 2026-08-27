# Data card — Tox21 SR-ARE benchmark

Everything below is either quoted from the source or produced by
`python -m tox21_bench.run_benchmark`. Nothing is estimated.

---

## 1. Source and scope

| | |
|---|---|
| Dataset | Tox21, DeepChem mirror: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz` |
| Assay used | **SR-ARE** — antioxidant response element, stress-response panel |
| Compounds after filtering | **5,825** |
| Actives | **942** (base rate **0.162**) |
| Representation | ECFP4 (Morgan, radius 2, 2,048 bits); plus six RDKit physicochemical descriptors as a low-capacity contrast |
| Label | Binary. `1` = active in the SR-ARE assay |

**Filtering applied.** The raw file has 7,823 rows. Rows where SR-ARE is `NaN`
are dropped, leaving 5,825. This is the standard treatment for Tox21's sparse
multi-task matrix, but it is *not* a neutral operation: absence of a measurement
in Tox21 is not random. Compounds were prioritised for testing on chemical and
regulatory grounds, so the surviving 5,825 are the compounds someone chose to
run in this assay. Any performance figure here describes that population, not
chemical space.

**What the label does and does not mean.** SR-ARE activity is a cell-based
reporter readout, not a toxicity endpoint. Calling a positive "toxic" — as an
earlier iteration of this project did in its UI — is wrong, and the naming has
been corrected throughout.

---

## 2. Known defects

### 2.1 Duplicate and near-duplicate compounds

| Check | Count |
|---|---|
| Exact duplicate SMILES | 0 |
| Duplicate InChIKeys (full) | 2 |
| InChIKey groups with conflicting labels | 0 |
| Duplicate InChIKey **skeletons** (first block = 2D connectivity) | 128 |
| Skeleton groups with **conflicting** labels | **6** |

Five of those six conflicts are annotation artifacts rather than biology — the
same substance entered twice in different protonation or stereochemical states,
with opposite labels:

| Skeleton | Labelled 0 | Labelled 1 | Nature of the conflict |
|---|---|---|---|
| `LCTONWCANYUPML` | `CC(=O)C(=O)[O-]` | `CC(=O)C(=O)O` | pyruvate vs pyruvic acid |
| `LLEMOWNGBBNAJR` | `Oc1ccccc1-c1ccccc1` | `[O-]c1ccccc1-c1ccccc1` | 2-phenylphenol vs its phenolate |
| `MCAHWIHFGHIESP` | `O=[Se]([O-])[O-]` | `O=[Se](O)O` | selenite vs selenous acid |
| `BACYUWVYYTXETD` | carboxylate form | acid form | lauroyl sarcosine pair |
| `VEMKTZHHVJILDY` | stereochemistry specified | stereochemistry unspecified | same pyrethroid ester |
| `VZCYOOQTPOCHFL` | maleic acid (Z) | fumaric acid (E) | **genuine** distinct compounds |

At assay pH these pairs are the same species. They put a hard ceiling on
achievable accuracy — no model can get both members of a pair right — and under a
random split they would leak directly.

**In the scaffold split used here, zero of the six conflicting groups are split
across train and test**, so they do not cause leakage in the reported numbers.
They are documented because a different split would not be so lucky.

### 2.2 The scaffold vocabulary is not what the name suggests

This is the most important caveat in this document.

| | |
|---|---|
| Distinct Bemis–Murcko scaffolds | 1,576 |
| Scaffolds appearing exactly once | 1,202 (76%) |
| Compounds mapping to the **empty** scaffold (acyclic) | **1,513 (26% of the dataset)** |
| Compounds on the benzene scaffold | 1,224 (21%) |

Bemis–Murcko strips side chains to the ring system. Every acyclic molecule
therefore maps to the *empty* scaffold and collapses into a single pseudo-group
of 1,513 compounds that share no structural feature at all beyond having no rings.
Together with benzene, two "groups" absorb 47% of the dataset.

The DeepChem splitter fills the training set largest-group-first, so both of those
groups go to train wholesale. The consequence:

> **All 1,165 test compounds sit in singleton scaffold classes.**

The scaffold-split test set is therefore not a random sample of harder chemistry —
it is precisely the structural tail of the dataset. That is visible in the class
balance, which shifts from **0.144 in train to 0.231 in test**: singleton
scaffolds are enriched in actives.

**How to read the scaffold-split numbers because of this.** They are a
worst-case, novel-chemotype estimate, not an average-deployment estimate. They
are the right number to quote when asking "will this work on a scaffold we have
never seen"; they are the wrong number to quote as "the model's accuracy". The
random-split numbers are the wrong number for essentially every purpose (see §3)
and are reported only as a contrast.

---

## 3. Splits

| Split | Train | Test | Test base rate | Test compounds sharing a scaffold with train |
|---|---|---|---|---|
| Scaffold (Bemis–Murcko) | 4,660 | 1,165 | 0.231 | **0 (0%)** |
| Random (seed 42) | 4,660 | 1,165 | 0.187 | **927 (79.6%)** |

The scaffold split is **deterministic** — it takes no random seed. Grouping by
scaffold and ordering largest-first fully determines the partition. Code that
passes a `seed` to a scaffold splitter is passing an argument that does nothing;
an earlier iteration of this project did exactly that.

`splits.scaffold_split` reimplements `deepchem.splits.ScaffoldSplitter` and is
pinned to the published partition by a regression test
(`tests/test_splits.py::test_matches_published_deepchem_split`: 4,660 / 1,165 with
896 negatives and 269 actives). The reimplementation exists so that the heavy
DeepChem dependency, which pins an old RDKit, can be dropped.

**The two splits' scores are not directly comparable in absolute terms.** Their
test base rates differ (0.187 vs 0.231), and both accuracy and PR-AUC have floors
that move with prevalence. Only the *margin over each split's own null baseline*
is comparable across the two. That is why every table in this repo reports the
margin, not just the raw score.

---

## 4. Metrics and why each is here

| Metric | Null baseline on the same test set | Why included |
|---|---|---|
| Accuracy | majority-class classifier | Included **to demonstrate that it is uninformative here** (§5) |
| ROC-AUC | 0.5 (nominal) — but see the permutation null | Standard, prevalence-insensitive ranking measure |
| PR-AUC | base rate | The honest ranking measure under class imbalance |
| Brier | constant base-rate predictor | Probability quality, not just ordering |
| ECE | 0 for a calibrated model | Whether a stated confidence means anything |
| Enrichment @10% | 1.0x | What a screening campaign actually experiences |

**Enrichment is the metric that matches the use case.** Nobody screening compounds
applies a 0.5 threshold; they take the top N and test them. Enrichment reports the
hit rate in that slice relative to picking at random.

---

## 5. Why accuracy is reported but should not be used

On the scaffold split, majority-class accuracy is **0.769**. The ECFP4 model
scores **0.778** (+0.86 pp) and the descriptor model scores **0.767**, which is
*below* the trivial baseline. At the 0.5 threshold the ECFP4 model recovers **7.4%
of actives**.

The same model reaches **2.07x enrichment** in the top decile with **47.9%
precision**. It is a poor classifier and a usable ranker, and any report quoting
its accuracy would communicate the opposite of both facts.

---

## 6. Calibration

Probabilities come from `CalibratedClassifierCV(RandomForest, method="sigmoid",
cv=5)`. The calibrator is cross-fitted **inside the training fold only**, so no
test information reaches it.

Calibration quality is not uniform across the test set — it degrades with
distance from the training set (§7). A single aggregate ECE hides that.

---

## 7. Applicability domain

Defined as maximum Tanimoto similarity (ECFP4) from a query to any training
compound.

**Thresholds are derived, not chosen.** For each training compound its
leave-one-out nearest neighbour among the other training compounds is computed;
that distribution describes how densely the training set covers its own space
(median 0.606). Cutoffs are then percentiles of that distribution:

| Band | Rule | Similarity | Test compounds |
|---|---|---|---|
| in-domain | ≥ 25th percentile of training self-similarity | ≥ 0.500 | 305 |
| borderline | between | 0.333–0.500 | 439 |
| out-of-domain | < 5th percentile | < 0.333 | 421 |

The percentiles (25 / 5) are the tunable choice and are stated rather than buried.
An earlier iteration of this project used hand-picked cutoffs of 0.49 / 0.25; the
derived values land at 0.50 / 0.333, so the original guess was close — but it was
a guess, and on a different dataset it would not have been.

The domain is **validated rather than asserted**: performance is reported per band,
and cross-checked against a threshold-free similarity-quartile breakdown that
reproduces the same monotonic trend. See `results/tables.md`.

---

## 8. Limitations

1. **Single assay.** SR-ARE only. Nothing here generalises to the other 11 Tox21 tasks.
2. **Single split per design.** No repeated resampling, so no confidence intervals on the headline numbers. The permutation null (20 label shuffles) is the only uncertainty estimate present.
3. **No counter-screen control.** Tox21 stress-response panels ship with paired viability readouts. Whether this model is detecting SR-ARE activity specifically or general cytotoxicity is **not tested here**, and it is the most important open question about the result.
4. **No hyperparameter search.** Model settings are inherited from the earlier iteration of this project so that the split comparison is the only thing changing. Absolute performance is therefore not a ceiling.
5. **2D representation only.** ECFP4 encodes connectivity, not conformation, tautomer state or protonation — which is exactly what §2.1 shows the labels are sensitive to.
6. **Measurement noise unquantified.** Tox21 replicate agreement is not modelled, so the irreducible error is unknown and the gap between model and ceiling cannot be attributed.

---

## 9. Reproducing

```bash
pip install -r requirements.txt
python -m tox21_bench.run_benchmark --data-dir data --out results
python -m tox21_bench.make_tables --results results
pytest
```

Versions are pinned in `requirements.txt` because the numbers move without them:
an earlier XGBoost-based run of this project reported ROC-AUC 0.7650, which came
back as 0.7697 on a newer release of the same library with identical inputs.

---

## 10. Permutation null

The nominal floors for ROC-AUC (0.5) and PR-AUC (the base rate) assume a ranker
carrying no information. Retraining the actual pipeline on permuted training
labels tests that assumption instead of asserting it. Over 20 permutations
(scaffold split, ECFP4, uncalibrated forest):

| | Observed | Permutation null (mean ± sd) | Nominal floor |
|---|---|---|---|
| ROC-AUC | 0.692 | 0.544 ± 0.023 (min 0.496, max 0.584) | 0.500 |
| PR-AUC | 0.424 | 0.256 ± 0.017 (min 0.221, max 0.282) | 0.231 |

z = 6.3 and 10.0 respectively; empirical p = 0 in both cases.

Both nulls are displaced upward. Any margin computed against the nominal floors —
including the "vs random" columns in `results/tables.md` — is therefore slightly
generous, and should be read as an upper bound on the model's advantage.

Calibration is omitted from this control for compute reasons. It does not affect
ranking metrics, and the observed values quoted above come from the same
uncalibrated forest, so the comparison is like-for-like.

**Cause not established.** The likely mechanism is that a forest fitted to
destroyed labels still encodes local training-set density, and density is not
independent of activity here. This is untested.
