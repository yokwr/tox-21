# Results

Dataset: 5825 compounds, 942 actives (base rate 0.162), assay SR-ARE.

## What a trivial predictor scores

| Split | Test n | Base rate | Majority-class accuracy | Random-ranker PR-AUC | Base-rate Brier |
|---|---|---|---|---|---|
| random | 1165 | 0.187 | 0.813 | 0.187 | 0.152 |
| scaffold | 1165 | 0.231 | 0.769 | 0.231 | 0.178 |

## Headline results

| Split | Features | Accuracy | vs majority | PR-AUC | vs random | ROC-AUC | Brier | ECE | Recall @0.5 | Enrichment @10% |
|---|---|---|---|---|---|---|---|---|---|---|
| random | ecfp4 | 0.835 | +2.23 pp | 0.502 | 2.68x | 0.763 | 0.127 | 0.045 | 0.183 | 2.97x |
| random | descriptors | 0.820 | +0.69 pp | 0.407 | 2.18x | 0.736 | 0.137 | 0.045 | 0.124 | 2.33x |
| scaffold | ecfp4 | 0.778 | +0.86 pp | 0.435 | 1.89x | 0.707 | 0.163 | 0.052 | 0.074 | 2.07x |
| scaffold | descriptors | 0.767 | -0.17 pp | 0.398 | 1.72x | 0.700 | 0.165 | 0.050 | 0.052 | 2.04x |

## Controls

| Split | Control | ROC-AUC | PR-AUC |
|---|---|---|---|
| random | shuffled training labels (ECFP4) | 0.409 | 0.157 |
| random | best single descriptor (molecular weight) | 0.634 | 0.245 |
| scaffold | shuffled training labels (ECFP4) | 0.553 | 0.257 |
| scaffold | best single descriptor (lipophilicity) | 0.677 | 0.355 |

## Applicability domain (scaffold split, ECFP4)

| Band | n | Mean similarity to train | Base rate | PR-AUC | vs random | ROC-AUC | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| in_domain | 305 | 0.628 | 0.213 | 0.606 | 2.85x | 0.800 | 0.127 | 0.040 |
| borderline | 439 | 0.400 | 0.221 | 0.375 | 1.70x | 0.688 | 0.164 | 0.062 |
| out_of_domain | 421 | 0.269 | 0.254 | 0.339 | 1.33x | 0.637 | 0.188 | 0.074 |

### Threshold-free cross-check

| Similarity quartile | Range | n | Base rate | PR-AUC | vs random | ECE |
|---|---|---|---|---|---|---|
| Q1 | 0.08-0.30 | 291 | 0.247 | 0.326 | 1.32x | 0.058 |
| Q2 | 0.30-0.38 | 291 | 0.265 | 0.399 | 1.51x | 0.105 |
| Q3 | 0.38-0.50 | 278 | 0.198 | 0.355 | 1.79x | 0.045 |
| Q4 | 0.50-1.00 | 305 | 0.213 | 0.606 | 2.85x | 0.040 |

## Permutation null (scaffold split, ECFP4, 20 label permutations)

- observed ROC-AUC 0.6924 vs null 0.5444 +/- 0.0234 (z = 6.3, empirical p = 0.00)
- observed PR-AUC 0.4240 vs null 0.2561 +/- 0.0167 (z = 10.0, empirical p = 0.00)

