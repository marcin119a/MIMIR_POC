"""
Phase 2 training: multi-modal shared representation.

Jointly optimises three objectives:
    L = Σ_m L_recon(m)  +  λ_contrast · L_contrast  +  λ_cross · L_cross

where
  L_recon     – α·L_masked + (1-α)·L_overall per modality (α=0.5); NaN positions excluded
  L_contrast  – InfoNCE across ordered modality pairs (τ = 0.1)
  L_cross     – leave-one-modality-out MSE imputation loss

Encoders and decoders are randomly initialised (no Phase 1 pretraining).
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
from torch.utils.data import DataLoader, Dataset

from phase2_model import MIMIRPhase2


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Phase2Config:
    # Paths
    data_path: str = "data/tcga_redo_mlomicZ.pkl"
    splits_path: str = "data/splits.json"
    checkpoint_dir: str = "checkpoints_phase2"

    # Model
    latent_dim: int = 128
    shared_dim: int = 256
    encoder_hidden: tuple = (512, 256)
    decoder_hidden: tuple = (256, 512)
    dropout: float = 0.1

    # Optimiser (paper: lr=3e-4, wd=1e-5)
    batch_size: int = 256
    num_epochs: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    patience: int = 20
    grad_clip: float = 1.0

    # Masking / dropout
    mask_rate: float = 0.20          # fraction of features randomly masked per sample
    modality_dropout_prob: float = 0.40  # per-sample prob of dropping each modality
    sentinel: float = -1.0           # sentinel value for masked features

    # Loss weights (paper: both = 1)
    lambda_contrast: float = 1.0
    lambda_cross: float = 1.0
    temperature: float = 0.1         # τ for InfoNCE
    alpha_recon: float = 0.5         # weight on masked term vs overall term

    seed: int = 42


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiOmicDataset(Dataset):
    """
    Returns per-sample dicts:
        {modality: {"orig": Tensor, "masked": Tensor,
                    "feat_mask": BoolTensor, "nan_mask": BoolTensor}}

    nan_mask  – positions truly missing in the raw data (NaN → sentinel).
    feat_mask – positions artificially masked during denoising (non-NaN only).
    Feature masking is applied lazily in __getitem__ so each epoch gets fresh masks.
    Only barcodes present in every modality are kept.
    """

    def __init__(
        self,
        data_dict: dict,
        barcodes: list,
        mask_rate: float = 0.2,
        sentinel: float = -1.0,
    ):
        self.modalities = list(data_dict.keys())
        valid = [b for b in barcodes if all(b in data_dict[m].index for m in self.modalities)]
        self.tensors = {
            m: torch.tensor(data_dict[m].loc[valid].values, dtype=torch.float32)
            for m in self.modalities
        }
        self.n = len(valid)
        self.mask_rate = mask_rate
        self.sentinel = sentinel

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        out = {}
        for m in self.modalities:
            x = self.tensors[m][idx]
            nan_mask = torch.isnan(x)
            x = x.clone()
            x[nan_mask] = self.sentinel                             # NaN → sentinel
            feat_mask = (~nan_mask) & (torch.rand_like(x) < self.mask_rate)  # mask only observed
            x_masked = x.clone()
            x_masked[feat_mask] = self.sentinel
            out[m] = {"orig": x, "masked": x_masked, "feat_mask": feat_mask, "nan_mask": nan_mask}
        return out


# ── Loss functions ────────────────────────────────────────────────────────────

def masked_recon_loss(
    x_hat: torch.Tensor,
    x_orig: torch.Tensor,
    feat_mask: torch.Tensor,
    nan_mask: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    L_recon = α · L_masked + (1-α) · L_overall   (α = 0.5 per paper)

    L_overall – MSE over all observed (non-NaN) features.
    L_masked  – MSE restricted to artificially masked positions.
    Truly missing values (nan_mask) are excluded from both terms.
    """
    obs = ~nan_mask                              # truly observed positions
    l_overall = F.mse_loss(x_hat[obs], x_orig[obs]) if obs.any() else x_hat.new_tensor(0.0)

    masked_obs = feat_mask & obs                 # artificially masked & non-NaN
    if masked_obs.any():
        l_masked = F.mse_loss(x_hat[masked_obs], x_orig[masked_obs])
    else:
        l_masked = l_overall                     # no masked positions: fall back to overall

    return alpha * l_masked + (1.0 - alpha) * l_overall


def contrastive_loss(
    z_proj: dict,
    obs_mask: torch.Tensor,
    modalities: list,
    temperature: float,
) -> torch.Tensor:
    """
    InfoNCE (NT-Xent) loss averaged over all ordered modality pairs.

    For each ordered pair (m, m') we:
      1. Restrict to samples where both modalities are observed.
      2. Build cosine-similarity matrix S_ab = cos(z_m^a, z_m'^b) / τ.
      3. Treat the diagonal as positives (cross-entropy with identity labels).

    Only contributes when ≥2 samples have a given pair of modalities observed.
    """
    device = next(iter(z_proj.values())).device
    total = torch.tensor(0.0, device=device)
    n_pairs = 0

    for i, m in enumerate(modalities):
        for j, mp in enumerate(modalities):
            if i == j:
                continue
            both = obs_mask[:, i] & obs_mask[:, j]
            if both.sum() < 2:
                continue

            zm  = F.normalize(z_proj[m][both],  dim=-1)
            zmp = F.normalize(z_proj[mp][both], dim=-1)

            sim = torch.mm(zm, zmp.T) / temperature          # (N, N)
            labels = torch.arange(sim.shape[0], device=device)
            total = total + F.cross_entropy(sim, labels)
            n_pairs += 1

    return total / n_pairs if n_pairs > 0 else total


def cross_modal_imputation_loss(
    model: MIMIRPhase2,
    orig: dict,
    feat_mask: dict,
    obs_mask: torch.Tensor,
    modalities: list,
    device: torch.device,
) -> torch.Tensor:
    """
    Leave-one-modality-out imputation loss.

    For each target modality m that is observed:
      1. Aggregate shared z from all OTHER observed modalities.
      2. Decode m from that z.
      3. Penalise with MSE against the original (unmasked) target values.

    Averaged across all (sample, target-modality) pairs with ≥2 observed modalities.
    """
    total = torch.tensor(0.0, device=device)
    n_targets = 0

    multi_obs = obs_mask.sum(dim=1) >= 2  # samples with ≥2 observed modalities
    if not multi_obs.any():
        return total

    other_col_indices = {
        i: [j for j in range(len(modalities)) if j != i]
        for i in range(len(modalities))
    }

    for i, m_target in enumerate(modalities):
        # Samples where this modality is observed AND at least one other exists
        eligible = multi_obs & obs_mask[:, i]
        if not eligible.any():
            continue

        other_mods = [modalities[j] for j in other_col_indices[i]]
        other_obs_sub = obs_mask[eligible][:, other_col_indices[i]]  # (N, M-1) bool

        # Encode + project other modalities for eligible samples
        z_other = {
            m: model.project(model.encode(orig[m][eligible], m), m)
            for m in other_mods
        }

        # Masked aggregate using only the observed-other modalities
        stacked  = torch.stack([z_other[m] for m in other_mods], dim=1)  # (N, M-1, D)
        mask_sub = other_obs_sub.float().unsqueeze(-1)                    # (N, M-1, 1)
        counts   = mask_sub.sum(dim=1).clamp(min=1.0)                    # (N, 1)
        z_shared = (stacked * mask_sub).sum(dim=1) / counts              # (N, D)

        x_hat    = model.decode(z_shared, m_target)
        x_orig_t = orig[m_target][eligible]

        total = total + F.mse_loss(x_hat, x_orig_t)
        n_targets += 1

    return total / n_targets if n_targets > 0 else total


# ── Modality-dropout mask ─────────────────────────────────────────────────────

def sample_obs_mask(
    B: int,
    M: int,
    dropout_prob: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample (B, M) bool observation mask.
    Each modality is independently dropped with `dropout_prob`.
    Samples where ALL modalities are dropped get one random modality restored.
    """
    obs = torch.rand(B, M, device=device) >= dropout_prob
    all_dropped = ~obs.any(dim=1)
    if all_dropped.any():
        n_dropped = int(all_dropped.sum().item())
        rand_mods = torch.randint(0, M, (n_dropped,), device=device)
        obs[all_dropped] = False
        dropped_indices = all_dropped.nonzero(as_tuple=True)[0]
        for k, idx in enumerate(dropped_indices):
            obs[idx, rand_mods[k]] = True
    return obs


# ── Batch step ────────────────────────────────────────────────────────────────

def run_batch(
    model: MIMIRPhase2,
    batch: dict,
    config: Phase2Config,
    device: torch.device,
    training: bool,
) -> tuple:
    """
    Single forward pass + loss computation.

    Returns: (total_loss, recon_loss, contrast_loss, cross_loss) as floats
    except total_loss which retains the computation graph during training.
    """
    modalities = list(batch.keys())
    M = len(modalities)
    B = next(iter(batch.values()))["orig"].shape[0]

    orig      = {m: batch[m]["orig"].to(device)      for m in modalities}
    masked    = {m: batch[m]["masked"].to(device)    for m in modalities}
    feat_mask = {m: batch[m]["feat_mask"].to(device) for m in modalities}
    nan_mask  = {m: batch[m]["nan_mask"].to(device)  for m in modalities}

    dropout_p = config.modality_dropout_prob if training else 0.0
    obs_mask  = sample_obs_mask(B, M, dropout_p, device)   # (B, M) bool

    # ── Encode all modalities, project to shared space ──
    h_dict = {m: model.encode(masked[m], m) for m in modalities}
    z_proj = {m: model.project(h_dict[m], m) for m in modalities}

    # ── Aggregate with per-sample modality dropout ──
    z_shared = model.aggregate(z_proj, obs_mask)

    # ── Decode all modalities ──
    recons = {m: model.decode(z_shared, m) for m in modalities}

    # ── Reconstruction loss (observed modalities only) ──
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

    # ── Contrastive loss ──
    c_loss = contrastive_loss(z_proj, obs_mask, modalities, config.temperature)

    # ── Cross-modal imputation loss ──
    lomo_loss = cross_modal_imputation_loss(
        model, orig, feat_mask, obs_mask, modalities, device
    )

    total = (
        recon_loss
        + config.lambda_contrast * c_loss
        + config.lambda_cross    * lomo_loss
    )
    return total, recon_loss.item(), c_loss.item(), lomo_loss.item()


# ── Train / eval epochs ───────────────────────────────────────────────────────

def train_epoch(
    model: MIMIRPhase2,
    loader: DataLoader,
    optimizer: Adam,
    config: Phase2Config,
    device: torch.device,
) -> list:
    model.train()
    sums = [0.0, 0.0, 0.0, 0.0]
    for batch in loader:
        total, r, c, lomo = run_batch(model, batch, config, device, training=True)
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        sums[0] += total.item()
        sums[1] += r
        sums[2] += c
        sums[3] += lomo
    n = len(loader)
    return [s / n for s in sums]


@torch.no_grad()
def eval_epoch(
    model: MIMIRPhase2,
    loader: DataLoader,
    config: Phase2Config,
    device: torch.device,
) -> list:
    model.eval()
    sums = [0.0, 0.0, 0.0, 0.0]
    for batch in loader:
        total, r, c, lomo = run_batch(model, batch, config, device, training=False)
        sums[0] += total.item()
        sums[1] += r
        sums[2] += c
        sums[3] += lomo
    n = len(loader)
    return [s / n for s in sums]


# ── Main ──────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    config = Phase2Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = get_device()
    print(f"Device: {device}")

    # ── Load data ──
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
        num_workers=0, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=pin
    )

    # ── Model ──
    model = MIMIRPhase2(
        modality_dims=modality_dims,
        latent_dim=config.latent_dim,
        shared_dim=config.shared_dim,
        encoder_hidden=config.encoder_hidden,
        decoder_hidden=config.decoder_hidden,
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
    best_val   = float("inf")
    patience_ctr = 0

    header = f"{'Epoch':>5} | {'TrLoss':>8} | {'VaLoss':>8} | {'Recon':>8} | {'Contr':>8} | {'LOMO':>8} | Time"
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, config.num_epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, config, device)
        vl = eval_epoch(model, val_loader,   config, device)
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
