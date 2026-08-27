"""Loading and featurisation for the Tox21 SR-ARE benchmark.

The raw source is the DeepChem mirror of Tox21:
    https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz

Only the SR-ARE assay column is used. Rows where SR-ARE is unmeasured (NaN)
are dropped -- see DATA_CARD.md for why that matters.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
ASSAY = "SR-ARE"

DESCRIPTOR_NAMES = [
    "molecular weight",
    "number of hydrogen donors",
    "number of hydrogen acceptors",
    "polar surface area",
    "lipophilicity",
    "rotatable bonds",
]


@dataclass(frozen=True)
class Dataset:
    smiles: list[str]
    y: np.ndarray  # shape (n,), float 0/1

    def __len__(self) -> int:
        return len(self.smiles)

    @property
    def base_rate(self) -> float:
        return float(self.y.mean())


def load_from_source(url: str = TOX21_URL) -> Dataset:
    """Download Tox21 and keep rows with a measured SR-ARE label."""
    raw = pd.read_csv(url, compression="gzip")
    df = raw[["smiles", ASSAY]].dropna(how="any").reset_index(drop=True)
    keep = [i for i, s in enumerate(df["smiles"]) if Chem.MolFromSmiles(s) is not None]
    df = df.iloc[keep].reset_index(drop=True)
    return Dataset(smiles=df["smiles"].astype(str).tolist(), y=df[ASSAY].to_numpy(float))


def load_local(data_dir: str | Path) -> Dataset:
    """Load the frozen copy committed to this repo (data/smiles.csv, data/labels.csv)."""
    data_dir = Path(data_dir)
    smiles = pd.read_csv(data_dir / "smiles.csv")["smiles"].astype(str).tolist()
    y = pd.read_csv(data_dir / "labels.csv")[ASSAY].to_numpy(float)
    if len(smiles) != len(y):
        raise ValueError(f"smiles ({len(smiles)}) and labels ({len(y)}) length mismatch")
    return Dataset(smiles=smiles, y=y)


def ecfp4(smiles: list[str], n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """ECFP4 bit vectors as a dense uint8 array.

    Uses rdFingerprintGenerator (the non-deprecated API). This is bit-for-bit
    equivalent to the older AllChem.GetMorganFingerprintAsBitVect and to
    deepchem.feat.CircularFingerprint(size=n_bits, radius=radius); see
    tests/test_featurisers.py.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"unparseable SMILES at index {i}: {s!r}")
        out[i] = gen.GetFingerprintAsNumPy(mol)
    return out


def descriptors(smiles: list[str]) -> np.ndarray:
    """The six physicochemical descriptors used in the first iteration of this project."""
    fns = [
        Descriptors.MolWt,
        Descriptors.NumHDonors,
        Descriptors.NumHAcceptors,
        Descriptors.TPSA,
        Descriptors.MolLogP,
        Descriptors.NumRotatableBonds,
    ]
    out = np.zeros((len(smiles), len(fns)), dtype=float)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"unparseable SMILES at index {i}: {s!r}")
        out[i] = [f(mol) for f in fns]
    return out


def inchikeys(smiles: list[str]) -> list[str]:
    """Standard InChIKeys, used to audit for duplicate / near-duplicate compounds."""
    keys = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        keys.append("" if mol is None else MolToInchiKey(mol))
    return keys


def duplicate_report(smiles: list[str], y: np.ndarray) -> dict:
    """Count exact-SMILES and InChIKey-collapsed duplicates, and label conflicts.

    An InChIKey collision with disagreeing labels is a benchmark defect: whichever
    copy lands in test is partly predictable from the copy in train, and the
    "correct" answer is ambiguous.
    """
    keys = inchikeys(smiles)
    skeleton = [k.split("-")[0] if k else "" for k in keys]

    df = pd.DataFrame({"smiles": smiles, "key": keys, "skeleton": skeleton, "y": y})
    conflicts = (
        df[df["key"] != ""]
        .groupby("key")["y"]
        .nunique()
        .pipe(lambda s: int((s > 1).sum()))
    )
    skeleton_conflicts = (
        df[df["skeleton"] != ""]
        .groupby("skeleton")["y"]
        .nunique()
        .pipe(lambda s: int((s > 1).sum()))
    )
    return {
        "n": len(smiles),
        "exact_smiles_duplicates": int(df["smiles"].duplicated().sum()),
        "inchikey_duplicates": int(df[df["key"] != ""]["key"].duplicated().sum()),
        "inchikey_groups_with_conflicting_labels": conflicts,
        "skeleton_duplicates": int(df[df["skeleton"] != ""]["skeleton"].duplicated().sum()),
        "skeleton_groups_with_conflicting_labels": skeleton_conflicts,
        "unparseable": int(sum(1 for k in keys if k == "")),
    }
