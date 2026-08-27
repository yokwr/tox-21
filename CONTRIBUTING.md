# Working on this repo

## Adding a metric

Every metric must ship with the score a trivial predictor achieves on the *same*
test set. Add it to `metrics.null_baselines` at the same time you add it to
`metrics.evaluate`, and surface the margin in `metrics.margin_over_null`. A metric
without its floor is not reportable here.

## Adding a split

New splits go in `splits.py` and must come with a leakage diagnostic that
quantifies what the split does and does not separate — `scaffold_overlap` is the
model. Register the split in `run_benchmark.DESIGNS` so it is evaluated against
every featuriser, and never compare a new split's raw score against an existing
one: compare margins over each split's own null, because the base rates differ.

## Changing the split logic

`tests/test_splits.py::test_matches_published_deepchem_split` pins the scaffold
partition to 4,660 / 1,165 with 896 negatives and 269 actives. If a change breaks
it, every number in README.md and DATA_CARD.md is stale — regenerate them with
`run_benchmark` and `make_tables` in the same commit, do not edit the tables by hand.

## Reporting a result

State the null alongside it. "PR-AUC 0.435" is not a finding; "PR-AUC 0.435 against
a 0.231 random-ranker floor and a 0.256 permutation null" is.
