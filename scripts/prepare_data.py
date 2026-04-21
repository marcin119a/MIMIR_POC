"""
Script to download and prepare data from Kaggle
"""
import os
import sys
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import kagglehub
from sklearn.preprocessing import LabelEncoder
from config import Config

def download_datasets():
    """Download datasets from Kaggle"""
    print("Downloading RNA and mutations dataset...")
    rna_path = kagglehub.dataset_download('martininf1n1ty/rna-mutations-all-datasets')
    print(f"RNA dataset downloaded to: {rna_path}")
    
    print("\nDownloading DNA methylation dataset...")
    dna_path = kagglehub.dataset_download('martininf1n1ty/dna-methylation-final-adnotated')
    print(f"DNA methylation dataset downloaded to: {dna_path}")
    
    return rna_path, dna_path


def prepare_rna_data(rna_path):
    """Prepare RNA expression data"""
    print("\nPreparing RNA expression data...")
    df_expressions = pd.read_parquet(f'{rna_path}/expression_onko_db.parquet')
    
    # Sort by gene_name before grouping
    df_expressions_sorted = df_expressions.sort_values(by='gene_name')
    df_expressions_sorted = df_expressions_sorted.drop_duplicates(subset=['case_barcode', 'gene_name'])
    
    # Group by case_barcode and aggregate tpm_unstranded into a list, keep primary_site
    grouped_expressions_df = df_expressions_sorted.groupby('case_barcode').agg({
        'tpm_unstranded': list,
        'primary_site': 'first'  # Take first primary_site (should be same for all genes in a case)
    }).reset_index()
    
    # Filter to keep only rows where the list length is Config.INPUT_DIM_A
    filtered_grouped_expressions_df = grouped_expressions_df[
        grouped_expressions_df['tpm_unstranded'].apply(len) == Config.INPUT_DIM_A
    ]
    
    print(f"RNA data shape: {filtered_grouped_expressions_df.shape}")
    return filtered_grouped_expressions_df




def prepare_dna_methylation_data(dna_path):
    """Prepare DNA methylation data"""
    print("\nPreparing DNA methylation data...")
    df = pd.read_parquet(f'{dna_path}/part-00000-db52fd1e-039e-43fd-9eef-5f241ff75754-c000.snappy.parquet')
    
    # Sort by probe_id before grouping
    df_sorted = df.sort_values(by='probe_id_id')
    grouped_df = df_sorted.groupby('case_barcode')['beta_value'].apply(list).reset_index()

    filtered_grouped_methylation_df = grouped_df[
        grouped_df['beta_value'].apply(len) == Config.INPUT_DIM_B
    ]
    print(f"DNA methylation data shape: {filtered_grouped_methylation_df.shape}")
    return filtered_grouped_methylation_df


def merge_and_normalize_data(rna_df, dna_df, top_n_sites=24):
    """Merge all datasets and normalize"""
    print("\nMerging datasets...")
    
    # Merge RNA expression with DNA methylation using outer join to capture unmatched records
    merged_df = pd.merge(rna_df, dna_df, on='case_barcode', how='outer', indicator=True)
    
    # Identify and save unmatched records
    print("\nIdentifying unmatched records...")
    
    # RNA only (no matching DNA) - right side has NaN
    rna_only = merged_df[merged_df['_merge'] == 'left_only'].copy()
    if len(rna_only) > 0:
        print(f"Found {len(rna_only)} RNA samples without matching DNA methylation data")
        rna_only = rna_only[['case_barcode', 'tpm_unstranded', 'primary_site']]
        os.makedirs('data', exist_ok=True)
        rna_only.to_pickle('data/rna_only_unmatched.pkl')
        print(f"  Saved to: data/rna_only_unmatched.pkl")
    else:
        print("No RNA-only samples found")
    
    # DNA only (no matching RNA) - left side has NaN
    dna_only = merged_df[merged_df['_merge'] == 'right_only'].copy()
    if len(dna_only) > 0:
        print(f"Found {len(dna_only)} DNA methylation samples without matching RNA expression data")
        dna_only = dna_only[['case_barcode', 'beta_value']]
        dna_only.to_pickle('data/dna_only_unmatched.pkl')
        print(f"  Saved to: data/dna_only_unmatched.pkl")
    else:
        print("No DNA-only samples found")
    
    # Keep only successfully merged records
    merged_df = merged_df[merged_df['_merge'] == 'both'].copy()
    merged_df = merged_df.drop(columns=['_merge'])
    rna_only = rna_only.drop(columns=['_merge'], errors='ignore')
    dna_only = dna_only.drop(columns=['_merge'], errors='ignore')
    
    print(f"\nMerged data shape before filtering: {merged_df.shape}")
    
    # Filter to keep only top N most common primary sites
    print(f"\nFiltering to keep only top {top_n_sites} most common primary sites...")
    site_counts = merged_df['primary_site'].value_counts()
    print(f"Total number of unique primary sites: {len(site_counts)}")
    
    top_sites = site_counts.head(top_n_sites).index.tolist()
    print(f"\nTop {top_n_sites} primary sites:")
    for i, (site, count) in enumerate(site_counts.head(top_n_sites).items(), 1):
        print(f"  {i}. {site}: {count} samples")
    
    # Filter dataframe to keep only top sites
    merged_df = merged_df[merged_df['primary_site'].isin(top_sites)].reset_index(drop=True)
    print(f"\nMerged data shape after filtering: {merged_df.shape}")
    
    # Normalize tpm_unstranded data
    print("\nNormalizing RNA expression data...")
    merged_df["tpm_unstranded"] = merged_df["tpm_unstranded"].apply(
        lambda x: np.log1p(np.array(x))
    )
    
    # Encode primary site labels
    print("\nEncoding primary site labels...")
    label_encoder = LabelEncoder()
    merged_df['primary_site_encoded'] = label_encoder.fit_transform(merged_df['primary_site'])
    
    print(f"\nPrimary site encoding (all {len(label_encoder.classes_)} classes):")
    for cls, code in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        count = (merged_df['primary_site'] == cls).sum()
        print(f"  {code}: {cls} ({count} samples)")
    
    return merged_df, label_encoder, rna_only, dna_only


def build_multi_omic_dict(merged_df, rna_only_df=None, dna_only_df=None):
    """
    Convert the flat merged DataFrame into the dict format expected by
    train_autoencoders.py and train_shared.py:

        {
            "rna":         pd.DataFrame  [n_samples x n_rna_features],   index=case_barcode
            "methylation": pd.DataFrame  [n_samples x n_meth_features],  index=case_barcode
        }

    rna_only_df / dna_only_df: optional DataFrames of samples present in only
    one modality (already normalized). They are appended so that the union of all
    samples is represented — missing modality rows simply won't appear in the
    other DataFrame, enabling union-based splits.
    """
    print("\nBuilding multi-omic dict for training scripts...")

    rna_matrix = np.stack(merged_df["tpm_unstranded"].values)   # (N, 1177)
    meth_matrix = np.stack(merged_df["beta_value"].values)       # (N, 1211)
    index = merged_df["case_barcode"].values

    rna_df_out  = pd.DataFrame(rna_matrix,  index=index,
                               columns=[f"rna_{i}"  for i in range(rna_matrix.shape[1])])
    meth_df_out = pd.DataFrame(meth_matrix, index=index,
                               columns=[f"meth_{i}" for i in range(meth_matrix.shape[1])])

    rna_df_out.index.name  = "case_barcode"
    meth_df_out.index.name = "case_barcode"

    if rna_only_df is not None and len(rna_only_df) > 0:
        extra = np.stack(rna_only_df["tpm_unstranded"].values).astype(np.float32)
        extra_df = pd.DataFrame(extra, index=rna_only_df["case_barcode"].values,
                                columns=rna_df_out.columns)
        extra_df.index.name = "case_barcode"
        rna_df_out = pd.concat([rna_df_out, extra_df])
        print(f"  + {len(extra_df)} RNA-only samples appended")

    if dna_only_df is not None and len(dna_only_df) > 0:
        extra = np.stack(dna_only_df["beta_value"].values).astype(np.float32)
        extra_df = pd.DataFrame(extra, index=dna_only_df["case_barcode"].values,
                                columns=meth_df_out.columns)
        extra_df.index.name = "case_barcode"
        meth_df_out = pd.concat([meth_df_out, extra_df])
        print(f"  + {len(extra_df)} DNA-only samples appended")

    print(f"  RNA matrix:         {rna_df_out.shape}")
    print(f"  Methylation matrix: {meth_df_out.shape}")

    return {"rna": rna_df_out, "methylation": meth_df_out}


def main():
    """Main data preparation pipeline"""
    # Download datasets
    rna_path, dna_path = download_datasets()

    # Prepare individual datasets
    rna_df = prepare_rna_data(rna_path)
    dna_df = prepare_dna_methylation_data(dna_path)

    # Merge and normalize
    merged_df, label_encoder, rna_only, dna_only = merge_and_normalize_data(rna_df, dna_df)

    # Save processed data
    print("\nSaving processed data...")
    os.makedirs('data', exist_ok=True)
    merged_df.to_pickle('data/processed_data.pkl')

    # Save label encoder
    import pickle
    with open('data/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Normalize rna_only (log1p, same as matched RNA)
    rna_only = rna_only.copy()
    rna_only["tpm_unstranded"] = rna_only["tpm_unstranded"].apply(
        lambda x: np.log1p(np.array(x))
    )

    # Build and save multi-omic dict for train_autoencoders.py / train_shared.py
    multi_omic = build_multi_omic_dict(merged_df, rna_only_df=rna_only, dna_only_df=dna_only)
    multi_omic_path = 'data/tcga_redo_mlomicZ.pkl'
    with open(multi_omic_path, 'wb') as f:
        pickle.dump(multi_omic, f)
    print(f"Multi-omic dict saved to: {multi_omic_path}")

    print("\nData preparation complete!")
    print(f"Processed data saved to: data/processed_data.pkl")
    print(f"Label encoder saved to: data/label_encoder.pkl")
    print(f"Multi-omic dict saved to: {multi_omic_path}")
    print(f"\nAdditional files (if any unmatched records):")
    print(f"  - data/rna_only_unmatched.pkl (RNA samples without DNA)")
    print(f"  - data/dna_only_unmatched.pkl (DNA samples without RNA)")


if __name__ == "__main__":
    main()
