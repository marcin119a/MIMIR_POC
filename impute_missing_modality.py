"""
Phase 3: Imputation with Missing Modalities

Loads a pretrained MIMIRPhase2 checkpoint and evaluates imputation under
two missingness scenarios:
  1. Leave-one-modality-out (LOO)
  2. All possible missingness patterns

Usage:
    python impute_missing_modality.py
    python impute_missing_modality.py --data data/tcga_redo_mlomicZ.pkl --splits data/splits.json
    python impute_missing_modality.py --checkpoint checkpoints_phase2/best_model.pt
    python impute_missing_modality.py --skip_all_possible
"""

import argparse
import itertools
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

import sys
from phase2_model import MIMIRPhase2
from train_phase2 import Phase2Config as _Phase2Config
from crossmodal_model import CrossModalAttentionImputer
from train_crossmodal import CrossModalConfig as _CrossModalConfig

# Checkpoint was saved from __main__, so unpickling requires config classes there
sys.modules[__name__].Phase2Config = _Phase2Config
sys.modules[__name__].CrossModalConfig = _CrossModalConfig
if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]
sys.modules["__main__"].Phase2Config = _Phase2Config
sys.modules["__main__"].CrossModalConfig = _CrossModalConfig


# ─── Display names ────────────────────────────────────────────────────────────

DISPLAY_NAME = {
    "rna": "mRNA",
    "methylation": "Methylation",
}


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    modality_dims = ckpt["modality_dims"]
    cfg = ckpt["config"]
    model_type = ckpt.get("model_type", "phase2")

    if model_type == "crossmodal":
        model = CrossModalAttentionImputer(
            modality_dims=modality_dims,
            d_model=cfg.d_model,
            n_latents=cfg.n_latents,
            n_heads=cfg.n_heads,
            n_within_layers=cfg.n_within_layers,
            n_cross_layers=cfg.n_cross_layers,
            decoder_hidden=cfg.decoder_hidden,
            proj_dim=cfg.proj_dim,
            dropout=cfg.dropout,
        )
    else:
        model = MIMIRPhase2(
            modality_dims=modality_dims,
            latent_dim=cfg.latent_dim,
            shared_dim=cfg.shared_dim,
            encoder_hidden=cfg.encoder_hidden,
            decoder_hidden=cfg.decoder_hidden,
            dropout=cfg.dropout,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded {model_type} checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return model, modality_dims, model_type


# ─── Imputation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def impute(
    model,
    data: dict,
    present_mods: list[str],
    target_mod: str,
    samples: list[str],
    batch_size: int,
    device: torch.device,
    model_type: str = "phase2",
) -> np.ndarray:
    """Impute `target_mod` from `present_mods` for the given samples."""
    tensors = {
        m: torch.tensor(data[m].loc[samples].values, dtype=torch.float32)
        for m in present_mods
    }
    n = len(samples)
    preds = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        if model_type == "crossmodal":
            z_list = [
                model.compress(tensors[m][start:end].to(device), m)
                for m in present_mods
            ]
            x_hat = model.impute(z_list, target_mod)
        else:
            z_proj_list = []
            for m in present_mods:
                x = tensors[m][start:end].to(device)
                h = model.encode(x, m)
                z = model.project(h, m)
                z_proj_list.append(z)
            z_shared = torch.stack(z_proj_list, dim=1).mean(dim=1)
            x_hat = model.decode(z_shared, target_mod)

        preds.append(x_hat.cpu().numpy())

    return np.concatenate(preds, axis=0)


def leave_one_out_imputation(
    model,
    data: dict,
    samples: list[str],
    batch_size: int,
    device: torch.device,
    model_type: str = "phase2",
) -> dict:
    """LOO: for each modality, impute it from all others."""
    modalities = list(data.keys())
    pred_dict = {}

    for target in modalities:
        present = [m for m in modalities if m != target]
        print(f"  LOO: present={present} → target={target}")
        pred = impute(model, data, present, target, samples, batch_size, device, model_type)
        pred_dict[(tuple(present), target)] = pred

    return pred_dict


def all_possible_imputation(
    model,
    data: dict,
    samples: list[str],
    batch_size: int,
    device: torch.device,
    model_type: str = "phase2",
) -> dict:
    """Enumerate all non-empty proper subsets of observed modalities."""
    modalities = list(data.keys())
    M = len(modalities)
    pred_dict = {}

    for r in range(1, M):
        for present in itertools.combinations(modalities, r):
            for target in modalities:
                if target in present:
                    continue
                print(f"  AP: present={list(present)} → target={target}")
                pred = impute(
                    model, data, list(present), target, samples, batch_size, device, model_type
                )
                pred_dict[(present, target)] = pred

    return pred_dict


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_imputations(pred_dict: dict, data: dict, samples: list[str]) -> dict:
    metrics = {}
    for (present_mods, target), pred in pred_dict.items():
        true = data[target].loc[samples].values.astype(np.float32)
        mse = float(np.mean((pred - true) ** 2))
        flat_pred = pred.ravel()
        flat_true = true.ravel()
        r, _   = pearsonr(flat_pred, flat_true)
        rho, _ = spearmanr(flat_pred, flat_true)
        metrics[(present_mods, target)] = {
            "mse": mse,
            "pearson": float(r),
            "spearman": float(rho),
            "n_points": flat_true.size,
        }
    return metrics


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_metrics(metrics: dict, label: str = "") -> None:
    if label:
        print(f"\n{'='*60}\n  {label}\n{'='*60}")
    header = f"{'Present':40s} {'Target':14s} {'MSE':>8s} {'r':>7s} {'rho':>7s} {'N pts':>10s}"
    print(header)
    print("-" * len(header))
    for (present_mods, target), m in metrics.items():
        present_str = ", ".join(present_mods)
        print(
            f"{present_str:40s} {target:14s} "
            f"{m['mse']:8.4f} {m['pearson']:7.4f} {m['spearman']:7.4f} {m['n_points']:10,d}"
        )


def metrics_to_upset_df(metrics: dict, score_key: str = "pearson") -> pd.DataFrame:
    rows = []
    for (present_mods, target), vals in metrics.items():
        rows.append({
            "target": target,
            "present": set(present_mods),
            "n_present": len(present_mods),
            "score": vals[score_key],
        })
    return pd.DataFrame(rows)


def plot_upset_for_target(
    df: pd.DataFrame,
    target: str,
    all_modalities: list[str],
    score_label: str = "Pearson r",
    save_path: str | None = None,
) -> None:
    sub = df[df["target"] == target].copy()
    sub = sub.sort_values(by="score", ascending=False).reset_index(drop=True)
    n = len(sub)
    if n == 0:
        print(f"[WARN] No rows for target={target}; skipping plot.")
        return

    mods = [m for m in all_modalities if m != target]

    fig = plt.figure(figsize=(max(6, n * 0.6), 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax_bar = fig.add_subplot(gs[0])
    ax_bar.bar(range(n), sub["score"], color="tab:orange")
    ymax = sub["score"].max()
    ax_bar.set_ylim(0, ymax * 1.1)
    ax_bar.set_ylabel(score_label)
    ax_bar.set_title(f"Imputing {DISPLAY_NAME.get(target, target)}")
    for i, k in enumerate(sub["n_present"]):
        ax_bar.text(i, sub["score"].iloc[i] + 0.01, str(k),
                    ha="center", va="bottom", fontsize=9)
    ax_bar.set_xticks([])

    ax_mat = fig.add_subplot(gs[1], sharex=ax_bar)
    for i, present in enumerate(sub["present"]):
        for j, mod in enumerate(mods):
            ax_mat.scatter(
                i, j, s=50,
                color="black" if mod in present else "white",
                edgecolor="black",
                zorder=3,
            )
    ax_mat.set_yticks(range(len(mods)))
    ax_mat.set_yticklabels([DISPLAY_NAME.get(m, m) for m in mods])
    ax_mat.set_xlabel("Available modalities")
    ax_mat.set_xlim(-0.5, n - 0.5)
    ax_mat.set_ylim(-0.5, len(mods) - 0.5)
    for spine in ["top", "right", "left"]:
        ax_mat.spines[spine].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3: Impute missing modalities")
    p.add_argument("--data",       default="data/tcga_redo_mlomicZ.pkl",      help="Path to multi-omic pickle")
    p.add_argument("--splits",     default="data/splits.json",                help="Path to splits JSON")
    p.add_argument("--checkpoint", default="checkpoints_phase2/best_model.pt",help="Model checkpoint (.pt) — auto-detects model type")
    p.add_argument("--out",        default="results/imputation_modality",      help="Output directory")
    p.add_argument("--device",     default=None,                               help="cuda / mps / cpu (auto if omitted)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--skip_all_possible", action="store_true",
                   help="Skip all-possible-missingness evaluation (faster)")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Data & splits ──────────────────────────────────────────────────────────
    print(f"\nLoading data from {args.data} …")
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    print(f"Modalities: {list(data.keys())}")

    with open(args.splits) as f:
        splits = json.load(f)
    test_samples = splits["test"]
    print(
        f"Samples: train={len(splits['train'])} | val={len(splits['val'])} | test={len(test_samples)}"
    )

    # Keep only modalities present in this checkpoint
    model, modality_dims, model_type = load_model(args.checkpoint, device)
    data = {k: v for k, v in data.items() if k in modality_dims}
    print(f"Active modalities: {list(data.keys())}  |  model_type={model_type}")

    # Filter test samples to those present in every modality
    test_samples = [s for s in test_samples if all(s in data[m].index for m in data)]
    print(f"Test samples with full coverage: {len(test_samples)}")

    # ── LOO imputation ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Leave-one-modality-out imputation")
    print("="*60)

    pred_loo = leave_one_out_imputation(
        model, data, test_samples, args.batch_size, device, model_type
    )
    metrics_loo = evaluate_imputations(pred_loo, data, test_samples)
    print_metrics(metrics_loo, label="LOO Imputation Metrics")

    with open(os.path.join(args.out, "metrics_loo.pkl"), "wb") as f:
        pickle.dump(metrics_loo, f)

    # ── All-possible imputation ────────────────────────────────────────────────
    if not args.skip_all_possible:
        print("\n" + "="*60)
        print("  All-possible-missingness imputation")
        print("="*60)

        pred_ap = all_possible_imputation(
            model, data, test_samples, args.batch_size, device, model_type
        )
        metrics_ap = evaluate_imputations(pred_ap, data, test_samples)
        print_metrics(metrics_ap, label="All-Possible Imputation Metrics")

        with open(os.path.join(args.out, "metrics_all_possible.pkl"), "wb") as f:
            pickle.dump(metrics_ap, f)

        plots_dir = os.path.join(args.out, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        df_upset = metrics_to_upset_df(metrics_ap, score_key="pearson")

        for target in data.keys():
            plot_upset_for_target(
                df_upset,
                target=target,
                all_modalities=list(data.keys()),
                score_label="Pearson r",
                save_path=os.path.join(plots_dir, f"upset_{target}.png"),
            )

    print(f"\nDone. Results saved to {args.out}/")


if __name__ == "__main__":
    main()
