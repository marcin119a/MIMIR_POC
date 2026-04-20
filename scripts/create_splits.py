"""
Script to generate a train/val/test split JSON file from the processed multi-omic data.

Usage:
    python scripts/create_splits.py
    python scripts/create_splits.py --data data/tcga_redo_mlomicZ.pkl --output data/splits.json
    python scripts/create_splits.py --val-size 0.1 --test-size 0.2 --seed 42
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils import compute_shared_splits


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test split JSON")
    parser.add_argument("--data", default="data/tcga_redo_mlomicZ.pkl",
                        help="Path to multi-omic pickle file (default: data/tcga_redo_mlomicZ.pkl)")
    parser.add_argument("--output", default="data/splits.json",
                        help="Output JSON path (default: data/splits.json)")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Fraction of samples for validation (default: 0.1)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of samples for test (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--all-modalities", action="store_true", default=False,
                        help="Only include samples present in ALL modalities (intersection). "
                             "Default: include samples with at least one modality (union).")
    args = parser.parse_args()

    print(f"Loading data from: {args.data}")
    with open(args.data, "rb") as f:
        data_dict = pickle.load(f)

    print(f"Modalities: {list(data_dict.keys())}")
    for mod, df in data_dict.items():
        print(f"  {mod}: {df.shape}")

    require_all = args.all_modalities
    print(f"\nSample pool: {'intersection (all modalities)' if require_all else 'union (≥1 modality)'}")

    all_samples, train_idx, val_idx, test_idx = compute_shared_splits(
        data_dict,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        require_all_modalities=require_all,
    )

    train_samples = [all_samples[i] for i in train_idx]
    val_samples   = [all_samples[i] for i in val_idx]
    test_samples  = [all_samples[i] for i in test_idx]

    split_dict = {
        "train": train_samples,
        "val":   val_samples,
        "test":  test_samples,
    }

    print(f"\nSplit sizes:")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val:   {len(val_samples)}")
    print(f"  Test:  {len(test_samples)}")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(split_dict, f, indent=2)

    print(f"\nSplits saved to: {args.output}")


if __name__ == "__main__":
    main()
