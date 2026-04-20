import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
from typing import Dict, List, Tuple
import pickle
from torch.optim import Adam
import pandas as pd
import json

###############################################
# Datasets & Dataloaders
###############################################

class MultiOmicDataset(Dataset):
    """
    Takes a dict(modality -> pandas.DataFrame [samples x features]) and aligns on common samples.
    __getitem__ returns a dict(modality -> tensor) for multi-modal finetuning.
    """
    def __init__(self, data_dict):
        self.modalities = list(data_dict.keys())
        sample_sets = [set(df.index) for df in data_dict.values()]
        self.common_samples = sorted(set.intersection(*sample_sets))
        self.data = {mod: torch.tensor(df.loc[self.common_samples].values, dtype=torch.float32)
                     for mod, df in data_dict.items()}

    def __len__(self):
        return len(self.common_samples)

    def __getitem__(self, idx):
        return {mod: self.data[mod][idx] for mod in self.modalities}

    def get_split_indices(self, test_size: float=0.2, seed: int=42) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(seed)
        idxs = np.arange(len(self.common_samples))
        np.random.shuffle(idxs)
        split = int(len(idxs) * (1 - test_size))
        return idxs[:split], idxs[split:]


def get_dataloader(dataset: Dataset, batch_size: int=64, shuffle: bool=True, split_idx=None):
    ds = dataset if split_idx is None else Subset(dataset, split_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class SingleModalityDataset(Dataset):
    """
    For pretraining a *single* modality autoencoder.
    Expects a pandas.DataFrame (samples x features) for one modality.
    """
    def __init__(self, df):
        self.X = torch.tensor(df.values, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


class SingleModalityDatasetAligned(Dataset):
    """
    Ensures *shared splits across modalities* by enforcing a common sample order.
    Pass the full modality DataFrame and a list of `common_samples` (ordered) shared by all modalities.
    """
    def __init__(self, df, common_samples: List[str]):
        X = df.loc[common_samples].values
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


def compute_shared_splits(
    data_dict: Dict[str, pd.DataFrame],
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
    require_all_modalities: bool = True,
):
    """
    Compute a single train/val/test split from the sample pool.

    Args:
        data_dict: dict(modality -> DataFrame with samples as index)
        val_size: fraction of samples for validation
        test_size: fraction of samples for test
        seed: random seed
        require_all_modalities: if True (default) use intersection (only samples
            present in every modality); if False use union (include samples with
            at least one modality).

    Returns:
        common_samples (list[str]),
        train_idx (np.ndarray),
        val_idx (np.ndarray),
        test_idx (np.ndarray)
    """
    sample_sets = [set(df.index) for df in data_dict.values()]
    if require_all_modalities:
        common_samples = sorted(set.intersection(*sample_sets))
    else:
        common_samples = sorted(set.union(*sample_sets))

    rng = np.random.RandomState(seed)
    idxs = np.arange(len(common_samples))
    rng.shuffle(idxs)

    n_total = len(idxs)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)

    test_idx = idxs[:n_test]
    val_idx = idxs[n_test:n_test+n_val]
    train_idx = idxs[n_test+n_val:]

    return common_samples, train_idx, val_idx, test_idx

def load_shared_splits_from_json(
    data_dict: Dict[str, pd.DataFrame],
    json_path: str
) -> Tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-defined train/val/test splits from a JSON file and
    restrict to the intersection of sample IDs across all modalities.

    Args:
        data_dict: dict(modality -> DataFrame with samples as index)
        json_path: path to JSON file with keys ["train", "val", "test"]
                   and values being lists of sample IDs

    Returns:
        common_samples (list[str]),
        train_idx (np.ndarray),
        val_idx (np.ndarray),
        test_idx (np.ndarray)
    """
    with open(json_path, "r") as f:
        split_dict = json.load(f)

    # Convert to sets
    train_set = set(split_dict.get("train", []))
    val_set = set(split_dict.get("val", []))
    test_set = set(split_dict.get("test", []))

    # Compute common samples across modalities
    sample_sets = [set(df.index) for df in data_dict.values()]
    common_samples = sorted(set.intersection(*sample_sets))

    # Keep only samples that exist in the JSON splits and in all modalities
    train_samples = sorted(list(train_set.intersection(common_samples)))
    val_samples = sorted(list(val_set.intersection(common_samples)))
    test_samples = sorted(list(test_set.intersection(common_samples)))

    # Map to indices in `common_samples`
    sample_to_idx = {s: i for i, s in enumerate(common_samples)}
    train_idx = np.array([sample_to_idx[s] for s in train_samples])
    val_idx = np.array([sample_to_idx[s] for s in val_samples])
    test_idx = np.array([sample_to_idx[s] for s in test_samples])

    return common_samples, train_idx, val_idx, test_idx

def make_loaders_from_splits(multi_omic_data: Dict[str, pd.DataFrame],
                             common_samples: list,
                             train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
                             batch_size: int = 64):
    multi_ds = MultiOmicDataset({m: df.loc[common_samples] for m, df in multi_omic_data.items()})
    train_loader = get_dataloader(multi_ds, batch_size=batch_size, shuffle=True,  split_idx=train_idx)
    val_loader   = get_dataloader(multi_ds, batch_size=batch_size, shuffle=False, split_idx=val_idx)
    test_loader  = get_dataloader(multi_ds, batch_size=batch_size, shuffle=False, split_idx=test_idx)
    return train_loader, val_loader, test_loader
