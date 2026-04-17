"""
Convert scripts/data/processed_data.pkl → data/tcga_redo_mlomicZ.pkl

The training scripts (train_autoencoders.py, train_shared.py) expect a pickle
with this structure:

    {
        "rna":         pd.DataFrame  [n_samples x n_rna_features],   index=case_barcode
        "methylation": pd.DataFrame  [n_samples x n_meth_features],  index=case_barcode
    }

Run from the project root:
    python scripts/convert_to_training_format.py
"""

import os
import pickle
import numpy as np
import pandas as pd

PROCESSED_PKL = os.path.join(os.path.dirname(__file__), "data", "processed_data.pkl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_PKL = os.path.join(OUT_DIR, "tcga_redo_mlomicZ.pkl")


def main():
    print(f"Loading {PROCESSED_PKL} ...")
    with open(PROCESSED_PKL, "rb") as f:
        merged_df = pickle.load(f)

    print(f"  Shape: {merged_df.shape}")
    print(f"  Columns: {list(merged_df.columns)}")

    # ── RNA matrix ────────────────────────────────────────────────────────────
    rna_matrix = np.stack(merged_df["tpm_unstranded"].values).astype(np.float32)
    index = merged_df["case_barcode"].values

    rna_df = pd.DataFrame(
        rna_matrix,
        index=index,
        columns=[f"rna_{i}" for i in range(rna_matrix.shape[1])],
    )
    rna_df.index.name = "case_barcode"

    # ── Methylation matrix ────────────────────────────────────────────────────
    meth_matrix = np.stack(merged_df["beta_value"].values).astype(np.float32)

    meth_df = pd.DataFrame(
        meth_matrix,
        index=index,
        columns=[f"meth_{i}" for i in range(meth_matrix.shape[1])],
    )
    meth_df.index.name = "case_barcode"

    print(f"\nRNA matrix:         {rna_df.shape}")
    print(f"Methylation matrix: {meth_df.shape}")

    multi_omic = {"rna": rna_df, "methylation": meth_df}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(multi_omic, f)

    print(f"\nSaved → {OUT_PKL}")
    print("\nYou can now run:")
    print("  python train_autoencoders.py")
    print("  python train_shared.py")


if __name__ == "__main__":
    main()
