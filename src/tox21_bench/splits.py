"""Train/test splits.

Two splits are provided so that the *effect of the split* can be measured
independently of the effect of the features. Comparing a random-split number
against a scaffold-split number computed on different features tells you
nothing about leakage, because two things changed at once.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Bemis-Murcko scaffold SMILES. Unparseable input maps to the empty scaffold."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_split(
    smiles: list[str], frac_train: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic Bemis-Murcko scaffold split.

    Groups compounds by scaffold, orders the groups largest-first (ties broken by
    lowest member index), and fills train until it would overflow frac_train;
    everything else goes to test. No scaffold appears on both sides.

    This reimplements deepchem.splits.ScaffoldSplitter exactly -- it is verified
    against the published split in tests/test_splits.py -- but without the
    deepchem dependency, which pins an old RDKit and does not install cleanly
    on current Python.

    Note the split is *deterministic*: it takes no seed. Anything that looks like
    a seed on a scaffold splitter is decoration.
    """
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        groups.setdefault(murcko_scaffold(s), []).append(i)

    ordered = [
        members
        for _, members in sorted(
            ((k, sorted(v)) for k, v in groups.items()),
            key=lambda kv: (len(kv[1]), kv[1][0]),
            reverse=True,
        )
    ]

    cutoff = frac_train * len(smiles)
    train: list[int] = []
    test: list[int] = []
    for members in ordered:
        if len(train) + len(members) > cutoff:
            test += members
        else:
            train += members
    return np.array(sorted(train)), np.array(sorted(test))


def random_split(
    n: int, frac_train: float = 0.8, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Plain random split. Included as the *contrast* to the scaffold split, not
    as a recommended evaluation setting."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(round(frac_train * n))
    return np.sort(idx[:cut]), np.sort(idx[cut:])


def scaffold_composition(smiles: list[str], test_idx: np.ndarray | None = None) -> dict:
    """Diagnostic on the scaffold vocabulary itself.

    Bemis-Murcko maps every acyclic molecule to the *empty* scaffold, so all acyclic
    compounds collapse into one enormous pseudo-group. Because the splitter fills
    train largest-group-first, that group -- and the benzene group -- go to train
    wholesale, and the test set ends up composed of scaffold singletons.

    That is worth knowing before quoting a scaffold-split score as "the honest
    number": it is a tail-of-distribution estimate, not an average-case one.
    """
    scaffolds = [murcko_scaffold(s) for s in smiles]
    counts: dict[str, int] = {}
    for s in scaffolds:
        counts[s] = counts.get(s, 0) + 1
    singletons = sum(1 for c in counts.values() if c == 1)
    out = {
        "n_compounds": len(smiles),
        "n_scaffolds": len(counts),
        "singleton_scaffolds": singletons,
        "acyclic_empty_scaffold_compounds": counts.get("", 0),
        "largest_scaffold_classes": sorted(
            ((k or "<acyclic>", v) for k, v in counts.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )[:8],
    }
    if test_idx is not None:
        test_scaffolds = [scaffolds[i] for i in test_idx]
        out["test_compounds"] = len(test_idx)
        out["distinct_test_scaffolds"] = len(set(test_scaffolds))
        out["test_compounds_in_singleton_scaffolds"] = sum(
            1 for s in test_scaffolds if counts[s] == 1
        )
    return out


def scaffold_overlap(
    smiles: list[str], train_idx: np.ndarray, test_idx: np.ndarray
) -> dict:
    """How much scaffold sharing a split leaves behind.

    For a scaffold split this must be zero. For a random split it is large, and
    that number is the mechanism behind the optimistic random-split score.
    """
    tr = {murcko_scaffold(smiles[i]) for i in train_idx}
    te_scaffolds = [murcko_scaffold(smiles[i]) for i in test_idx]
    shared = sum(1 for s in te_scaffolds if s in tr)
    return {
        "n_test": len(test_idx),
        "test_compounds_whose_scaffold_is_in_train": shared,
        "fraction": shared / len(test_idx) if len(test_idx) else float("nan"),
        "n_train_scaffolds": len(tr),
        "n_test_scaffolds": len(set(te_scaffolds)),
    }
