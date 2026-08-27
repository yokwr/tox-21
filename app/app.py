"""Minimal demo UI for the SR-ARE model.

This replaces an earlier Streamlit app that had a fatal inconsistency: it trained
on 2,048-bit ECFP4 fingerprints but called predict() with a six-element
physicochemical descriptor vector, so every prediction either crashed on a shape
mismatch or was meaningless. It also called .corr() on a dataframe of SMILES
strings, and it labelled positives "TOXIC" when SR-ARE is a stress-response
reporter readout, not a toxicity endpoint.

Design decisions here, all of which follow from results/tables.md:

1. One featuriser. The same tox21_bench.data.ecfp4 used to train the model is
   used at inference. There is no second code path.
2. The applicability domain is shown on every prediction, not hidden in a tab.
   Out-of-domain, this model ranks at 1.33x random and is measurably worse
   calibrated (ECE 0.074 vs 0.040), so a bare probability is misleading.
3. No 0.5 threshold and no accuracy figure anywhere. At 0.5 the model recovers
   7.4% of actives and scores below a majority-class baseline. The output is a
   rank-style score with its band-specific reliability attached.

    streamlit run app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402

from tox21_bench import applicability as ad  # noqa: E402
from tox21_bench import data as D  # noqa: E402
from tox21_bench import models as Mo  # noqa: E402
from tox21_bench import splits as S  # noqa: E402

# Band-specific reliability, measured on the held-out scaffold-split test set.
# Source: results/tables.md. These are not decoration -- they are the reason a
# single probability is not a sufficient answer.
BAND_EVIDENCE = {
    "in_domain": ("PR-AUC 2.85x random, ECE 0.040 (n=305)", "The model has support here."),
    "borderline": ("PR-AUC 1.70x random, ECE 0.063 (n=439)", "Treat as a weak prior."),
    "out_of_domain": (
        "PR-AUC 1.33x random, ECE 0.074 (n=421)",
        "Barely better than random, and overconfident. Do not act on this score alone.",
    ),
}


@st.cache_resource(show_spinner="Fitting model on the scaffold-split training fold...")
def load_pipeline():
    ds = D.load_local(ROOT / "data")
    train_idx, _ = S.scaffold_split(ds.smiles)
    train_smiles = [ds.smiles[i] for i in train_idx]
    X = D.ecfp4(ds.smiles)

    model = Mo.calibrated_rf()
    model.fit(X[train_idx], ds.y[train_idx])

    self_sim = ad.training_self_similarity(train_smiles)
    thresholds = ad.derive_thresholds(self_sim)
    return model, train_smiles, thresholds


def main() -> None:
    st.set_page_config(page_title="SR-ARE activity — scored with its domain", layout="centered")
    st.title("Tox21 SR-ARE activity score")
    st.caption(
        "SR-ARE is an antioxidant-response reporter assay. A positive means activity "
        "in that assay — it does not mean the compound is toxic."
    )

    model, train_smiles, thresholds = load_pipeline()

    query = st.text_input("SMILES", value="c1ccc(cc1)C(=O)O")
    if not query:
        return

    mol = Chem.MolFromSmiles(query)
    if mol is None:
        st.error("RDKit could not parse that SMILES.")
        return

    score = float(model.predict_proba(D.ecfp4([query]))[0, 1])
    sim = float(ad.max_similarity_to_reference([query], train_smiles)[0])
    band = str(ad.assign_domain(np.array([sim]), thresholds)[0])
    evidence, advice = BAND_EVIDENCE[band]

    left, right = st.columns([2, 1])
    with left:
        st.metric("SR-ARE activity score", f"{score:.3f}")
        st.metric("Nearest training compound (Tanimoto, ECFP4)", f"{sim:.3f}")
        st.write(f"**Applicability domain: {band.replace('_', ' ')}**")
        st.write(f"Measured reliability in this band — {evidence}")
        (st.success if band == "in_domain" else st.warning)(advice)
        st.caption(
            f"Band cutoffs derived from the training set's own nearest-neighbour "
            f"distribution: in-domain ≥ {thresholds['in_domain_min_similarity']:.3f}, "
            f"out-of-domain < {thresholds['out_of_domain_max_similarity']:.3f}."
        )
    with right:
        st.image(Draw.MolToImage(mol, size=(260, 260)))

    st.divider()
    st.markdown(
        "**How to use this number.** It is a ranking score, not a decision. "
        "Applying a 0.5 cut-off recovers 7.4% of actives and scores below a "
        "majority-class baseline. Used as a ranker on the top decile it reaches "
        "2.07x enrichment at 47.9% precision. Rank a library with it; do not "
        "classify a single compound with it."
    )


if __name__ == "__main__":
    main()
