"""
Training script for the Cross-Modal Attention Imputer (CMAI).

Loss:
    L = L_self_recon  +  λ_contrast · L_contrast  +  λ_cross · L_cross

    L_self_recon  – masked MSE: each modality reconstructs itself from its own
                    compressed representation (verifies encoder/decoder quality)
    L_contrast    – InfoNCE across modality pairs (reused from train_phase2)
    L_cross       – LOO imputation MSE via cross-modal attention (the key term)

Usage:
    python train_crossmodal.py
"""

import json
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from crossmodal_model import CrossModalAttentionImputer
from train_phase2 import (
    MultiOmicDataset,
    contrastive_loss,
    masked_recon_loss,
    sample_obs_mask,
    get_device,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class CrossModalConfig:
    # Paths
    data_path: str = "data/tcga_redo_mlomicZ.pkl"
    splits_path: str = "data/splits.json"
    checkpoint_dir: str = "checkpoints_crossmodal"

    # Model
    d_model: int = 64
    n_latents: int = 32
    n_heads: int = 4
    n_within_layers: int = 2
    n_cross_layers: int = 2
    decoder_hidden: tuple = (256, 512)
    proj_dim: int = 256
    dropout: float = 0.1

    # Optimiser
    batch_size: int = 256
    num_epochs: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    patience: int = 20
    grad_clip: float = 1.0

    # Masking / dropout
    mask_rate: float = 0.20
    modality_dropout_prob: float = 0.40
    sentinel: float = -1.0

    # Loss weights
    lambda_contrast: float = 5.0
    lambda_cross: float = 1.0
    temperature: float = 0.1

    seed: int = 42


# ── Loss functions ────────────────────────────────────────────────────────────

def cross_modal_imputation_loss_cmai(
    model: CrossModalAttentionImputer,
    orig: dict,
    obs_mask: torch.Tensor,
    modalities: list,
    device: torch.device,
    z_cache: dict | None = None,
) -> torch.Tensor:
    """
    LOO imputation loss using proper cross-modal attention.

    For each target modality m observed in a sample:
        1. Compress all OTHER observed modalities.
        2. Impute m via cross-modal attention from those compressed tokens.
        3. MSE against ground truth.

    This is the key loss that distinguishes CMAI from MIMIRPhase2.
    """
    total = torch.tensor(0.0, device=device)
    n_targets = 0

    multi_obs = obs_mask.sum(dim=1) >= 2
    if not multi_obs.any():
        return total

    other_indices = {
        i: [j for j in range(len(modalities)) if j != i]
        for i in range(len(modalities))
    }

    for i, m_target in enumerate(modalities):
        eligible = multi_obs & obs_mask[:, i]
        if not eligible.any():
            continue

        other_mods = [modalities[j] for j in other_indices[i]]
        other_obs_sub = obs_mask[eligible][:, other_indices[i]]  # (N, M-1)

        # Reuse cached compressed representations when available
        if z_cache is not None:
            z_other = {m: z_cache[m][eligible] for m in other_mods}
        else:
            z_other = {m: model.compress(orig[m][eligible], m) for m in other_mods}

        # Per-sample: gather compressed tokens of OBSERVED other modalities.
        # We handle variable availability by zeroing out absent modalities then
        # concatenating — absent contributions are masked to zero before concat.
        N = eligible.sum().item()
        K = model.n_latents
        d = model.d_model

        # Build masked source: only include tokens from observed other modalities
        # Stack to (N, M-1, K, d), mask, then reshape to (N, M-1*K, d)
        stacked = torch.stack(
            [z_other[m] for m in other_mods], dim=1
        )  # (N, M-1, K, d)
        mask = other_obs_sub.float().unsqueeze(-1).unsqueeze(-1)  # (N, M-1, 1, 1)
        stacked = stacked * mask  # zero out absent modalities
        source = stacked.reshape(N, len(other_mods) * K, d)  # (N, M-1*K, d)

        # Build target queries and run cross-attention layers
        q = model.target_queries[m_target].unsqueeze(0).expand(N, -1, -1).contiguous()
        for layer in model.cross_tf[m_target]:
            q = layer(q, source)

        x_hat = model.decoders[m_target](q.mean(dim=1))
        total = total + F.mse_loss(x_hat, orig[m_target][eligible])
        n_targets += 1

    return total / n_targets if n_targets > 0 else total


# ── Batch step ────────────────────────────────────────────────────────────────

def run_batch(
    model: CrossModalAttentionImputer,
    batch: dict,
    config: CrossModalConfig,
    device: torch.device,
    training: bool,
) -> tuple:
    modalities = list(batch.keys())
    M = len(modalities)
    B = next(iter(batch.values()))["orig"].shape[0]

    orig      = {m: batch[m]["orig"].to(device)      for m in modalities}
    masked    = {m: batch[m]["masked"].to(device)    for m in modalities}
    feat_mask = {m: batch[m]["feat_mask"].to(device) for m in modalities}

    dropout_p = config.modality_dropout_prob if training else 0.0
    obs_mask  = sample_obs_mask(B, M, dropout_p, device)

    # ── Compress all modalities (using masked inputs for self-recon) ──
    z = {m: model.compress(masked[m], m) for m in modalities}

    # ── Self-reconstruction loss (observed modalities only) ──
    recon_loss = torch.tensor(0.0, device=device)
    for i, m in enumerate(modalities):
        obs_idx = obs_mask[:, i]
        if obs_idx.any():
            x_hat = model.reconstruct(z[m][obs_idx], m)
            recon_loss = recon_loss + masked_recon_loss(
                x_hat, orig[m][obs_idx], feat_mask[m][obs_idx]
            )

    # ── Contrastive loss (reused from phase2) ──
    z_proj = {m: model.project(z[m], m) for m in modalities}
    c_loss = contrastive_loss(z_proj, obs_mask, modalities, config.temperature)

    # ── Cross-modal LOO imputation loss ──
    lomo_loss = cross_modal_imputation_loss_cmai(
        model, orig, obs_mask, modalities, device, z_cache=z
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
        sums[0] += total.item()
        sums[1] += r; sums[2] += c; sums[3] += lomo
    n = len(loader)
    return [s / n for s in sums]


@torch.no_grad()
def eval_epoch(model, loader, config, device):
    model.eval()
    sums = [0.0, 0.0, 0.0, 0.0]
    for batch in loader:
        total, r, c, lomo = run_batch(model, batch, config, device, training=False)
        sums[0] += total.item()
        sums[1] += r; sums[2] += c; sums[3] += lomo
    n = len(loader)
    return [s / n for s in sums]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config = CrossModalConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = get_device()
    print(f"Device: {device}")

    with open(config.data_path, "rb") as f:
        data_dict = pickle.load(f)
    with open(config.splits_path) as f:
        splits = json.load(f)

    modality_dims = {m: data_dict[m].shape[1] for m in data_dict}
    print(f"Modalities: {modality_dims}")

    train_ds = MultiOmicDataset(
        data_dict, splits["train"], config.mask_rate, config.sentinel
    )
    val_ds = MultiOmicDataset(
        data_dict, splits["val"], config.mask_rate, config.sentinel
    )
    print(f"Samples  train={len(train_ds)}  val={len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=pin,
    )

    model = CrossModalAttentionImputer(
        modality_dims=modality_dims,
        d_model=config.d_model,
        n_latents=config.n_latents,
        n_heads=config.n_heads,
        n_within_layers=config.n_within_layers,
        n_cross_layers=config.n_cross_layers,
        decoder_hidden=config.decoder_hidden,
        proj_dim=config.proj_dim,
        dropout=config.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(config.checkpoint_dir, "best_model.pt")
    best_val = float("inf")
    patience_ctr = 0

    header = (
        f"{'Epoch':>5} | {'TrLoss':>8} | {'VaLoss':>8} | "
        f"{'Recon':>8} | {'Contr':>8} | {'LOMO':>8} | Time"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, config.num_epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, config, device)
        vl = eval_epoch(model, val_loader, config, device)
        dt = time.time() - t0

        print(
            f"{epoch:>5} | {tr[0]:>8.4f} | {vl[0]:>8.4f} | "
            f"{vl[1]:>8.4f} | {vl[2]:>8.4f} | {vl[3]:>8.4f} | {dt:.1f}s"
        )

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
                    "model_type": "crossmodal",
                },
                best_ckpt,
            )
        else:
            patience_ctr += 1
            if patience_ctr >= config.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest val loss: {best_val:.4f}  →  {best_ckpt}")


if __name__ == "__main__":
    main()
