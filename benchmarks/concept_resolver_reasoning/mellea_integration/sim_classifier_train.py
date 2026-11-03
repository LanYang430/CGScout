"""Builds the persisted Stage 3 experiment-vs-simulation classifier
artifact (sim_classifier_model.joblib, loaded at runtime by
sim_classifier.py). NOT imported by production code -- run manually
whenever the training CSV/label corrections below change:

    .venv-mellea/bin/python benchmarks/concept_resolver_reasoning/mellea_integration/sim_classifier_train.py

Ports the exact cleaning logic (DOI-keyed relabel/exclude sets, review-note
exclusion) and model architecture validated in this session's standalone
diagnostic (scratchpad stage3_logreg_v5_clean_train.py): PCA(60) +
StandardScaler + LogisticRegression(C=1.0), fit on ALL 424 cleaned training
rows (reviews/tool-papers excluded from training entirely, not relabeled --
empirically the best of three tested strategies). 5-fold GroupKFold CV
ROC-AUC on this training set: 0.9160. External 427-row held-out test set
(review/tool papers left IN, per instruction, to see real-world behavior):
ROC-AUC 0.8477, at threshold 0.5 -- TP=166 FP=105 FN=20 TN=136, recall
89.2%, precision 61.3%.

TRAIN_CSV is the user's own labeled export (beam_stage3_training_ready_
yes_no.csv, 557 rows: title/doi/abstract/simulation_by_authors/label_note),
not checked into the repo -- point TRAIN_CSV at wherever that CSV lives, or
override via the BEAM_STAGE3_TRAIN_CSV env var.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))  # sim_classifier.py, same directory
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent))  # repo root, for beam_agent.models

from sim_classifier import EMBED_MODEL, MODEL_PATH  # noqa: E402

TRAIN_CSV = Path(os.environ.get(
    "BEAM_STAGE3_TRAIN_CSV",
    "~/Downloads/beam_stage3_training_ready_yes_no.csv",
))
EMBED_CACHE_PATH = Path(__file__).resolve().parent / "sim_classifier_embed_cache.json"
N_PCA_COMPONENTS = 60
SEED = 0

# --- Manual audit, all rounds -- FINAL union, each DOI in exactly ONE set
# (mirrors scratchpad stage3_logreg_v5_clean_train.py's validated state) ---

RELABEL_YES_TO_NO_DOIS = {
    "10.3390/molecules27051647",          # Thermoresponsive Ionic Liquid/Water Mixtures -- review
    "10.1038/s41586-022-04574-8",         # Structure deformation and curvature sensing of PIEZO1 -- experimental/cryo-EM
    "10.1021/acsomega.4c03344",           # In Silico View of Crowding -- Mini-Review
    "10.1103/physrevlett.119.188102",     # RNA Base Pairing Determines Conformations of RNA Inside Spherical Viruses -- theoretical/numerical, not a simulation trajectory
    "10.1002/polb.23168",                 # Polymer brushes on flat and curved surfaces -- review
    "10.1021/acs.macromol.8b02476",       # Impact of Macromolecular Crowding and Compression on Protein-Protein Interactions and LLPS -- round-2 audit: NO
}

RELABEL_NO_TO_YES_DOIS = {
    "10.1002/pro.70284",                  # Accurate prediction of thermoresponsive phase behavior of disordered proteins -- this IS the Mpipi-T paper
    "10.1002/jcc.25797",                  # Modeling large protein-glycosaminoglycan complexes using a fragment-based approach
    "10.1038/s41598-018-37034-3",         # Can enzyme proximity accelerate cascade reactions?
    "10.1021/acscatal.8b02760",           # Queueing Theory-Based Perspective of the Kinetics of Channeled Enzyme Cascade Reactions
    "10.1016/j.bpj.2017.01.035",          # Gramicidin A Channel Formation Induces Local Lipid Redistribution II
    "10.1085/jgp.202213260",              # Microscopic mechanism of PIEZO1 activation by pressure-induced membrane stretch
    "10.1016/j.molcel.2021.01.039",       # Transient non-specific DNA binding dominates the target search of bacterial DNA-binding proteins
    "10.1016/j.jmb.2013.04.006",          # Integrative Modelling Coupled with Ion Mobility Mass Spectrometry
    "10.1021/acs.jpcb.1c00910",           # GlycanDock
    "10.1371/journal.pone.0018934",       # Rosetta FlexPepDock ab-initio
    "10.1038/s41567-021-01462-2",         # Sequence-dependent surface condensation of a pioneer transcription factor on DNA
    "10.1016/j.jmb.2004.09.029",          # A Simple Physical Model for the Prediction and Design of Protein-DNA Interactions
    "10.3389/fchem.2021.753002",          # OnionNet-2
    "10.1093/nar/gki875",                 # Protein-DNA binding specificity predictions with structural models
    "10.1016/j.bpj.2026.01.011",          # The cyanobacterial ESCRT-III protein IM30 forms biomolecular condensates
    "10.1101/424358",                     # Narrow equilibrium window for complex coacervation of tau and RNA under cellular conditions
    "10.1101/2025.07.27.666995",          # Identification and Characterization of Outer Membrane Proteins -- Brucella melitensis
    "10.1038/s41565-025-02077-x",         # Nanoscale domains govern local diffusion and ageing within fused-in-sarcoma condensates
    "10.1016/j.bpj.2024.01.023",          # Ensemble structure of the N-terminal domain (1-267) of FUS -- EPR + ensemble modeling
}

EXCLUDE_DOIS = {
    # --- round 1 ---
    "10.1007/978-94-011-0497-5_33", "10.1016/j.sbi.2012.06.002", "10.1016/j.coviro.2016.02.012",
    "10.3390/polym5030890", "10.1002/jcc.21287", "10.1021/bi971709u", "10.1021/acs.jcim.4c01372",
    # --- round 2 ---
    "10.1002/bmb.2003.494031040239", "10.1016/j.bbamem.2011.07.048", "10.1016/j.bbamem.2008.10.006",
    "10.1007/s12551-025-01276-z", "10.1371/journal.pcbi.1003720", "10.1016/j.imu.2022.100880",
    "10.1002/cpbi.3", "10.1002/wcms.1410", "10.1201/9781420059564", "10.1002/wcms.1169",
    "10.1101/2024.05.26.595895", "10.1146/annurev-chembioeng-092319-083243", "10.1021/acs.jcim.4c01520",
    "10.2174/138920312799277901", "10.1146/annurev-physchem-071819-113553", "10.1021/acs.biochem.2c00210",
    # --- round 3: high-confidence-only cheap review filter (review_filter.py) ---
    "10.1021/acs.chemrev.2c00814", "10.1016/j.pnmrs.2026.101611", "10.3389/fmolb.2019.00021",
    "10.1021/acs.chemrev.4c00138", "10.1016/j.cocis.2020.101416", "10.1111/febs.15617",
    "10.1002/pro.4968", "10.1146/annurev-biophys-062215-010915", "10.1016/j.addr.2024.115418",
    "10.1016/j.sbi.2020.12.012", "10.1016/j.sbi.2009.03.002", "10.1016/j.sbi.2007.03.004",
    "10.1016/j.sbi.2010.07.005", "10.1016/j.sbi.2022.102356", "10.1016/j.coviro.2021.11.014",
    "10.1002/bies.950140102", "10.1016/j.tibs.2017.12.003", "10.1016/j.sbi.2017.03.011",
    "10.1016/j.sbi.2019.04.004", "10.1038/s41580-024-00773-5",
}
EXCLUDE_TITLE_SUBSTRINGS = [
    "characterizing the biophysical effects of mutations on the xpf-ercc1",
    "modélisation à gros grains de la séparation de phases liquide-liquide",
]


def is_train_review(note: str) -> bool:
    n = (note or "").lower()
    return "review" in n and not n.startswith("manual/web review")


def load_train_rows() -> list[dict]:
    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kept, n_review, n_relabel_to_no, n_relabel_to_yes, n_excluded_doi = [], 0, 0, 0, 0
    for r in rows:
        doi = (r["doi"] or "").strip()
        title_lower = r["title"].strip().lower()
        if doi in EXCLUDE_DOIS or any(s in title_lower for s in EXCLUDE_TITLE_SUBSTRINGS):
            n_excluded_doi += 1
            continue
        if doi in RELABEL_YES_TO_NO_DOIS:
            r = {**r, "simulation_by_authors": "NO"}
            n_relabel_to_no += 1
        if doi in RELABEL_NO_TO_YES_DOIS:
            r = {**r, "simulation_by_authors": "YES"}
            n_relabel_to_yes += 1
        if r["simulation_by_authors"] not in ("YES", "NO"):
            continue
        if is_train_review(r.get("label_note", "")):
            n_review += 1
            continue
        kept.append(r)
    print(f"train: excluded {n_excluded_doi} rows by DOI/title, excluded {n_review} more flagged review "
          f"by label_note, relabeled {n_relabel_to_no} YES->NO, relabeled {n_relabel_to_yes} NO->YES; "
          f"kept {len(kept)}")
    return kept


def get_embeddings(texts: list[str], keys: list[str]) -> np.ndarray:
    cache = json.loads(EMBED_CACHE_PATH.read_text()) if EMBED_CACHE_PATH.exists() else {}
    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    print(f"  embedding cache: {len(keys) - len(missing_idx)}/{len(keys)} hits, {len(missing_idx)} new calls")
    if missing_idx:
        client = OpenAI()
        BATCH = 200
        for start in range(0, len(missing_idx), BATCH):
            batch_idx = missing_idx[start:start + BATCH]
            resp = client.embeddings.create(model=EMBED_MODEL, input=[texts[i] for i in batch_idx])
            for i, d in zip(batch_idx, resp.data):
                cache[keys[i]] = d.embedding
        EMBED_CACHE_PATH.write_text(json.dumps(cache))
    return np.array([cache[k] for k in keys])


def main() -> None:
    train_rows = load_train_rows()
    n_yes = sum(1 for r in train_rows if r["simulation_by_authors"] == "YES")
    print(f"final training set: {len(train_rows)} rows  YES={n_yes} NO={len(train_rows) - n_yes}")

    texts = [f"{r['title']}\n\n{r['abstract']}" for r in train_rows]
    keys = [r["doi"] or r["title"] for r in train_rows]
    y = np.array([1 if r["simulation_by_authors"] == "YES" else 0 for r in train_rows])

    print("embedding training set...")
    X = get_embeddings(texts, keys)

    n_comp = min(N_PCA_COMPONENTS, len(train_rows) - 10)
    pipe = Pipeline([
        ("pca", PCA(n_components=n_comp, random_state=SEED)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, C=1.0)),
    ])
    pipe.fit(X, y)

    import joblib
    joblib.dump(pipe, MODEL_PATH)
    print(f"wrote {MODEL_PATH} (n_pca_components={n_comp}, trained on {len(train_rows)} rows)")


if __name__ == "__main__":
    main()
