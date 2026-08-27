"""Tests for split correctness.

The important one is test_matches_published_deepchem_split: this repo reimplements
deepchem.splits.ScaffoldSplitter so the deepchem dependency can be dropped, and
that reimplementation has to produce the identical partition or the historical
numbers in README.md stop being comparable.
"""
from pathlib import Path

import numpy as np
import pytest

from tox21_bench import data as D
from tox21_bench import splits as S

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def ds():
    return D.load_local(DATA)


def test_murcko_scaffold_basics():
    assert S.murcko_scaffold("c1ccccc1O") == "c1ccccc1"
    # acyclic molecules have no ring system, so Murcko returns the empty scaffold
    assert S.murcko_scaffold("CCCCO") == ""
    assert S.murcko_scaffold("not a molecule") == ""


def test_scaffold_split_is_disjoint_by_scaffold(ds):
    tr, te = S.scaffold_split(ds.smiles)
    train_scaffolds = {S.murcko_scaffold(ds.smiles[i]) for i in tr}
    test_scaffolds = {S.murcko_scaffold(ds.smiles[i]) for i in te}
    assert train_scaffolds & test_scaffolds == set()


def test_scaffold_split_partitions_every_compound(ds):
    tr, te = S.scaffold_split(ds.smiles)
    assert len(tr) + len(te) == len(ds)
    assert set(tr.tolist()) & set(te.tolist()) == set()


def test_scaffold_split_is_deterministic(ds):
    a1, b1 = S.scaffold_split(ds.smiles)
    a2, b2 = S.scaffold_split(ds.smiles)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)


def test_matches_published_deepchem_split(ds):
    """Regression against the deepchem ScaffoldSplitter partition this replaces.

    Reference figures were produced with deepchem.splits.ScaffoldSplitter
    .train_test_split(test_size=0.2) on this exact dataset.
    """
    tr, te = S.scaffold_split(ds.smiles)
    assert len(tr) == 4660
    assert len(te) == 1165
    y_te = ds.y[te]
    assert int(y_te.sum()) == 269
    assert int((y_te == 0).sum()) == 896


def test_random_split_leaks_scaffolds_and_scaffold_split_does_not(ds):
    """The contrast the whole benchmark rests on."""
    tr_s, te_s = S.scaffold_split(ds.smiles)
    tr_r, te_r = S.random_split(len(ds), seed=42)
    assert S.scaffold_overlap(ds.smiles, tr_s, te_s)["fraction"] == 0.0
    assert S.scaffold_overlap(ds.smiles, tr_r, te_r)["fraction"] > 0.5


def test_random_split_is_seed_stable(ds):
    a1, b1 = S.random_split(len(ds), seed=7)
    a2, b2 = S.random_split(len(ds), seed=7)
    a3, _ = S.random_split(len(ds), seed=8)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert not np.array_equal(a1, a3)


def test_scaffold_composition_flags_acyclic_pseudo_group(ds):
    """The empty scaffold is not a scaffold, and it is huge. Fail loudly if that
    ever silently stops being true, because the split's meaning depends on it."""
    comp = S.scaffold_composition(ds.smiles)
    assert comp["acyclic_empty_scaffold_compounds"] > 1000
    assert comp["singleton_scaffolds"] > 0.5 * comp["n_scaffolds"]
