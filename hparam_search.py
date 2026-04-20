"""
Hyperparameter search for the new-loss MIMIR model.

Trains a grid of named configs, evaluates LOO imputation on the test set,
and prints a comparison table. Already-trained configs are skipped (resume
by re-running the script).

Variants explore:
  - nan_fix  : exclude NaN sentinel positions from cross-modal imputation loss
  - lambda_cross : weight on the LOO imputation term (1 / 2 / 5 / 10)
  - lambda_contrast : weight on InfoNCE term (0 / 0.5 / 1)
  - scheduler : cosine-annealing LR vs constant
  - alpha_recon : weight on masked vs. overall reconstruction term

Usage:
    uv run python hparam_search.py
    uv run python hparam_search.py --configs baseline nan_fix lc5
    uv run python hparam_search.py --eval_only       # skip training, re-eval all
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from phase2_model import MIMIRPhase2
from train_phase2 import (
    MultiOmicDataset,
    Phase2Config,
    contrastive_loss,
    masked_recon_loss,
    sample_obs_mask,
)


# ── Extended config ────────────────────────────────────────────────────────────

@dataclass
class SearchConfig(Phase2Config):
    name: str = "baseline"
    nan_fix: bool = False           # exclude NaN positions from cross-modal loss
    use_scheduler: bool = False     # cosine-annealing LR scheduler
    checkpoint_dir: str = field(init=False)

    def __post_init__(self):
        self.checkpoint_dir = f"checkpoints_search/{self.name}"


# ── Patched cross-modal imputation loss ───────────────────────────────────────

def cross_modal_imputation_loss(
    model: MIMIRPhase2,
    orig: dict,
    nan_mask: dict,
    obs_mask: torch.Tensor,
    modalities: list,
    device: torch.device,
    fix_nan: bool = False,
) -> torch.Tensor:
    """
    LOO imputation loss. With fix_nan=True, sentinel positions (true NaNs)
    are excluded from the MSE so the model is not penalised for them.
    """
    total = torch.tensor(0.0, device=device)
    n_targets = 0

    multi_obs = obs_mask.sum(dim=1) >= 2
    if not multi_obs.any():
        return total

    other_col_indices = {
        i: [j for j in range(len(modalities)) if j != i]
        for i in range(len(modalities))
    }

    for i, m_target in enumerate(modalities):
        eligible = multi_obs & obs_mask[:, i]
        if not eligible.any():
            continue

        other_mods = [modalities[j] for j in other_col_indices[i]]
        other_obs_sub = obs_mask[eligible][:, other_col_indices[i]]

        z_other = {
            m: model.project(model.encode(orig[m][eligible], m), m)
            for m in other_mods
        }

        stacked  = torch.stack([z_other[m] for m in other_mods], dim=1)
        mask_sub = other_obs_sub.float().unsqueeze(-1)
        counts   = mask_sub.sum(dim=1).clamp(min=1.0)
        z_shared = (stacked * mask_sub).sum(dim=1) / counts

        x_hat    = model.decode(z_shared, m_target)
        x_orig_t = orig[m_target][eligible]

        if fix_nan:
            obs = ~nan_mask[m_target][eligible]
            if obs.any():
                total = total + F.mse_loss(x_hat[obs], x_orig_t[obs])
        else:
            total = total + F.mse_loss(x_hat, x_orig_t)

        n_targets += 1

    return total / n_targets if n_targets > 0 else total


# ── Batch step (extended) ─────────────────────────────────────────────────────

def run_batch(
    model: MIMIRPhase2,
    batch: dict,
    config: SearchConfig,
    device: torch.device,
    training: bool,
) -> tuple:
    modalities = list(batch.keys())
    M = len(modalities)
    B = next(iter(batch.values()))["orig"].shape[0]

    orig      = {m: batch[m]["orig"].to(device)      for m in modalities}
    masked    = {m: batch[m]["masked"].to(device)    for m in modalities}
    feat_mask = {m: batch[m]["feat_mask"].to(device) for m in modalities}
    nan_mask  = {m: batch[m]["nan_mask"].to(device)  for m in modalities}

    dropout_p = config.modality_dropout_prob if training else 0.0
    obs_mask  = sample_obs_mask(B, M, dropout_p, device)

    h_dict = {m: model.encode(masked[m], m) for m in modalities}
    z_proj = {m: model.project(h_dict[m], m) for m in modalities}
    z_shared = model.aggregate(z_proj, obs_mask)
    recons = {m: model.decode(z_shared, m) for m in modalities}

    recon_loss = torch.tensor(0.0, device=device)
    for i, m in enumerate(modalities):
        obs_idx = obs_mask[:, i]
        if obs_idx.any():
            recon_loss = recon_loss + masked_recon_loss(
                recons[m][obs_idx],
                orig[m][obs_idx],
                feat_mask[m][obs_idx],
                nan_mask[m][obs_idx],
                alpha=config.alpha_recon,
            )

    c_loss = contrastive_loss(z_proj, obs_mask, modalities, config.temperature)

    lomo_loss = cross_modal_imputation_loss(
        model, orig, nan_mask, obs_mask, modalities, device,
        fix_nan=config.nan_fix,
    )

    total = (
        recon_loss
        + config.lambda_contrast * c_loss
        + config.lambda_cross    * lomo_loss
    )
    return total, recon_loss.item(), c_loss.item(), lomo_loss.item()


# ── Train / eval epochs ───────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, config, device):
    model.train()
    sums = [0.0, 0.0, 0.0, 0.0]
    for batch in loader:
        total, r, c, lomo = run_batch(model, batch, config, device, training=True)
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        sums[0] += total.item(); sums[1] += r; sums[2] += c; sums[3] += lomo
    n = len(loader)
    return [s / n for s in sums]


@torch.no_grad()
def eval_epoch(model, loader, config, device):
    model.eval()
    sums = [0.0, 0.0, 0.0, 0.0]
    for batch in loader:
        total, r, c, lomo = run_batch(model, batch, config, device, training=False)
        sums[0] += total.item(); sums[1] += r; sums[2] += c; sums[3] += lomo
    n = len(loader)
    return [s / n for s in sums]


# ── Training loop ─────────────────────────────────────────────────────────────

def train_config(
    config: SearchConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    modality_dims: dict,
    device: torch.device,
) -> str:
    """Train one config. Returns path to best checkpoint."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(config.checkpoint_dir, "best_model.pt")

    if os.path.exists(best_ckpt):
        print(f"  [skip] {config.name}: checkpoint exists → {best_ckpt}")
        return best_ckpt

    print(f"\n{'='*62}")
    print(f"  Training: {config.name}")
    print(f"  nan_fix={config.nan_fix}  lc={config.lambda_cross}"
          f"  lcontr={config.lambda_contrast}  alpha={config.alpha_recon}"
          f"  sched={config.use_scheduler}")
    print(f"{'='*62}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = MIMIRPhase2(
        modality_dims=modality_dims,
        latent_dim=config.latent_dim,
        shared_dim=config.shared_dim,
        encoder_hidden=config.encoder_hidden,
        decoder_hidden=config.decoder_hidden,
        dropout=config.dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config.learning_rate,
                     weight_decay=config.weight_decay)
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)
        if config.use_scheduler else None
    )

    best_val = float("inf")
    patience_ctr = 0
    hdr = f"{'Ep':>4} | {'TrLoss':>8} | {'VaLoss':>8} | {'Recon':>8} | {'Contr':>8} | {'LOMO':>8} | Time"
    print(hdr)
    print("-" * len(hdr))

    for epoch in range(1, config.num_epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, config, device)
        vl = eval_epoch(model, val_loader, config, device)
        if scheduler:
            scheduler.step()
        dt = time.time() - t0

        print(f"{epoch:>4} | {tr[0]:>8.4f} | {vl[0]:>8.4f} | "
              f"{vl[1]:>8.4f} | {vl[2]:>8.4f} | {vl[3]:>8.4f} | {dt:.1f}s")

        if vl[0] < best_val:
            best_val = vl[0]
            patience_ctr = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val,
                    "config": config,
                    "modality_dims": modality_dims,
                },
                best_ckpt,
            )
        else:
            patience_ctr += 1
            if patience_ctr >= config.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"Best val loss: {best_val:.4f}  →  {best_ckpt}")
    return best_ckpt


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def impute(model, data, present_mods, target_mod, samples, batch_size, device):
    tensors = {m: torch.tensor(data[m].loc[samples].values, dtype=torch.float32)
               for m in present_mods}
    n = len(samples)
    preds = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        zs = []
        for m in present_mods:
            x = tensors[m][start:end].to(device)
            zs.append(model.project(model.encode(x, m), m))
        z_shared = torch.stack(zs, dim=1).mean(dim=1)
        preds.append(model.decode(z_shared, target_mod).cpu().numpy())
    return np.concatenate(preds, axis=0)


def eval_loo(ckpt_path: str, data: dict, test_samples: list,
             device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    modality_dims = ckpt["modality_dims"]
    cfg = ckpt["config"]
    model = MIMIRPhase2(
        modality_dims=modality_dims,
        latent_dim=cfg.latent_dim,
        shared_dim=cfg.shared_dim,
        encoder_hidden=cfg.encoder_hidden,
        decoder_hidden=cfg.decoder_hidden,
        dropout=cfg.dropout,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    modalities = list(modality_dims.keys())
    results = {}
    for target in modalities:
        present = [m for m in modalities if m != target]
        pred = impute(model, data, present, target, test_samples, 256, device)
        true = data[target].loc[test_samples].values.astype(np.float32)
        flat_pred, flat_true = pred.ravel(), true.ravel()
        r,   _ = pearsonr(flat_pred, flat_true)
        rho, _ = spearmanr(flat_pred, flat_true)
        mse    = float(np.mean((pred - true) ** 2))
        results[f"{'+'.join(present)}→{target}"] = {
            "mse": mse, "pearson": float(r), "spearman": float(rho),
            "epoch": ckpt["epoch"], "val_loss": ckpt["val_loss"],
        }
    return results


# ── Config grid ───────────────────────────────────────────────────────────────

def build_grid() -> list[SearchConfig]:
    base = dict(
        num_epochs=200,
        patience=20,
        latent_dim=128,
        shared_dim=256,
        encoder_hidden=(512, 256),
        decoder_hidden=(256, 512),
        dropout=0.1,
        batch_size=256,
        learning_rate=3e-4,
        weight_decay=1e-5,
        grad_clip=1.0,
        mask_rate=0.20,
        modality_dropout_prob=0.40,
        sentinel=-1.0,
        temperature=0.1,
        seed=42,
    )
    grid = [
        # ── ablations on the nan-fix ──────────────────────────────────────────
        SearchConfig(**base, name="baseline",
                     nan_fix=False, lambda_cross=1.0, lambda_contrast=1.0,
                     alpha_recon=0.5, use_scheduler=False),

        SearchConfig(**base, name="nan_fix",
                     nan_fix=True,  lambda_cross=1.0, lambda_contrast=1.0,
                     alpha_recon=0.5, use_scheduler=False),

        # ── lambda_cross sweep (with nan_fix) ─────────────────────────────────
        SearchConfig(**base, name="lc2",
                     nan_fix=True,  lambda_cross=2.0, lambda_contrast=1.0,
                     alpha_recon=0.5, use_scheduler=False),

        SearchConfig(**base, name="lc5",
                     nan_fix=True,  lambda_cross=5.0, lambda_contrast=1.0,
                     alpha_recon=0.5, use_scheduler=False),

        SearchConfig(**base, name="lc10",
                     nan_fix=True,  lambda_cross=10.0, lambda_contrast=1.0,
                     alpha_recon=0.5, use_scheduler=False),

        # ── no contrastive loss (ablation) ────────────────────────────────────
        SearchConfig(**base, name="no_contrast",
                     nan_fix=True,  lambda_cross=2.0, lambda_contrast=0.0,
                     alpha_recon=0.5, use_scheduler=False),

        # ── cosine-annealing LR ───────────────────────────────────────────────
        SearchConfig(**base, name="lc2_sched",
                     nan_fix=True,  lambda_cross=2.0, lambda_contrast=1.0,
                     alpha_recon=0.5, use_scheduler=True),

        # ── alpha_recon ablation ──────────────────────────────────────────────
        SearchConfig(**base, name="lc2_alpha03",
                     nan_fix=True,  lambda_cross=2.0, lambda_contrast=1.0,
                     alpha_recon=0.3, use_scheduler=False),

        SearchConfig(**base, name="lc2_alpha07",
                     nan_fix=True,  lambda_cross=2.0, lambda_contrast=1.0,
                     alpha_recon=0.7, use_scheduler=False),
    ]
    return grid


# ── Main ──────────────────────────────────────────────────────────────────────

def load_search_assets(data_path: str, splits_path: str) -> tuple[dict, dict, dict, list]:
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(splits_path) as f:
        splits = json.load(f)

    modality_dims = {m: data[m].shape[1] for m in data}
    test_samples = [s for s in splits["test"] if all(s in data[m].index for m in data)]
    return data, splits, modality_dims, test_samples


def select_configs(config_names: list[str] | None) -> list[SearchConfig]:
    grid = build_grid()
    if config_names:
        grid = [cfg for cfg in grid if cfg.name in config_names]
    if not grid:
        raise ValueError("No matching configs selected.")
    return grid


def build_loaders(
    data: dict,
    splits: dict,
    config: SearchConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    train_ds = MultiOmicDataset(
        data, splits["train"], mask_rate=config.mask_rate, sentinel=config.sentinel
    )
    val_ds = MultiOmicDataset(
        data, splits["val"], mask_rate=config.mask_rate, sentinel=config.sentinel
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )
    return train_loader, val_loader


def train_config_subprocess(config_name: str, args) -> str:
    config = next(cfg for cfg in build_grid() if cfg.name == config_name)
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker_config",
        config_name,
        "--data",
        args.data,
        "--splits",
        args.splits,
        "--out",
        args.out,
    ]

    print(f"\n{'='*62}")
    print(f"  Launching subprocess: {config_name}")
    print(f"{'='*62}")
    subprocess.run(command, check=True)

    best_ckpt = os.path.join(config.checkpoint_dir, "best_model.pt")
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(
            f"Training subprocess for {config_name} finished without checkpoint: {best_ckpt}"
        )
    return best_ckpt


def run_training_worker(args) -> None:
    config = select_configs([args.worker_config])[0]
    device = get_device()
    print(f"Device: {device}")

    data, splits, modality_dims, _ = load_search_assets(args.data, args.splits)
    train_loader, val_loader = build_loaders(data, splits, config, device)
    train_config(config, train_loader, val_loader, modality_dims, device)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="*", default=None,
                   help="Names of configs to run (default: all)")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training, only re-evaluate existing checkpoints")
    p.add_argument("--data",   default="data/tcga_redo_mlomicZ.pkl")
    p.add_argument("--splits", default="data/splits.json")
    p.add_argument("--out",    default="results/hparam_search")
    p.add_argument("--worker_config", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    if args.worker_config:
        run_training_worker(args)
        return

    device = get_device()
    print(f"Device: {device}")

    data, splits, modality_dims, test_samples = load_search_assets(
        args.data, args.splits
    )
    print(f"Modalities: {list(modality_dims.keys())}")
    print(f"Test samples: {len(test_samples)}")

    grid = select_configs(args.configs)

    # ── Train ─────────────────────────────────────────────────────────────────
    ckpt_paths = {}
    for cfg in grid:
        if args.eval_only:
            ckpt = os.path.join(cfg.checkpoint_dir, "best_model.pt")
            if os.path.exists(ckpt):
                ckpt_paths[cfg.name] = ckpt
            else:
                print(f"  [warn] {cfg.name}: no checkpoint found, skipping")
        else:
            ckpt_paths[cfg.name] = train_config_subprocess(cfg.name, args)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n{'='*62}\n  LOO Imputation Comparison\n{'='*62}")
    all_rows = []
    for name, ckpt in ckpt_paths.items():
        print(f"\nEvaluating {name} …")
        res = eval_loo(ckpt, data, test_samples, device)
        for task, m in res.items():
            all_rows.append({
                "config": name, "task": task,
                "MSE": m["mse"], "r": m["pearson"], "rho": m["spearman"],
                "epoch": m["epoch"], "val_loss": m["val_loss"],
            })

    df = pd.DataFrame(all_rows)

    # ── Print table ───────────────────────────────────────────────────────────
    print()
    cfg_col_w   = max(len(c) for c in df["config"]) + 2
    task_col_w  = max(len(t) for t in df["task"])   + 2
    hdr = (f"{'Config':<{cfg_col_w}} {'Task':<{task_col_w}}"
           f" {'MSE':>8} {'r':>7} {'rho':>7} {'epoch':>6} {'val_loss':>9}")
    print(hdr)
    print("-" * len(hdr))

    grid_names = [c.name for c in grid]
    df_sorted = df.set_index("config").loc[[n for n in grid_names if n in df["config"].values]]
    df_sorted = df_sorted.reset_index()
    for _, row in df_sorted.iterrows():
        print(f"{row['config']:<{cfg_col_w}} {row['task']:<{task_col_w}}"
              f" {row['MSE']:>8.4f} {row['r']:>7.4f} {row['rho']:>7.4f}"
              f" {int(row['epoch']):>6} {row['val_loss']:>9.4f}")

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved → {out_csv}")

    # ── Best config per task ──────────────────────────────────────────────────
    print("\nBest config per task (by Pearson r):")
    for task in df["task"].unique():
        sub = df[df["task"] == task].sort_values("r", ascending=False).iloc[0]
        print(f"  {task:40s}  {sub['config']}  r={sub['r']:.4f}  rho={sub['rho']:.4f}")


if __name__ == "__main__":
    main()
