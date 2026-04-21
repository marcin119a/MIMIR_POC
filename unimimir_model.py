import torch
import torch.nn as nn


def _mlp(dims: list, dropout: float = 0.1) -> nn.Sequential:
    layers: list = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(dims[i + 1]), nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class UniMIMIR(nn.Module):
    """
    UniMIMIR: unified multi-modal shared representation model.

    Per modality m:
        h_m = encoder_m(x_m)              input_dim -> latent_dim
        z_m = proj_m(h_m)                 latent_dim -> shared_dim
        z   = mean_{m in obs}(z_m)        shared aggregation
        h_m'= inv_proj_m(z)               shared_dim -> latent_dim
        x_m = decoder_m(h_m')             latent_dim -> input_dim

    Args:
        modality_dims: {name: feature_dim}, e.g. {"rna": 1177, "methylation": 1211}
        latent_dim:    encoder output / decoder input dimension
        shared_dim:    dimension of the shared latent space (256 per paper)
        encoder_hidden / decoder_hidden: hidden layer sizes for encoder/decoder MLPs
        dropout:       dropout probability applied between hidden layers
    """

    def __init__(
        self,
        modality_dims: dict,
        latent_dim: int = 128,
        shared_dim: int = 256,
        encoder_hidden: tuple = (512, 256),
        decoder_hidden: tuple = (256, 512),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.shared_dim = shared_dim
        self.latent_dim = latent_dim

        enc = list(encoder_hidden)
        dec = list(decoder_hidden)

        self.encoders = nn.ModuleDict({
            m: _mlp([dim] + enc + [latent_dim], dropout)
            for m, dim in modality_dims.items()
        })
        self.decoders = nn.ModuleDict({
            m: _mlp([latent_dim] + dec + [dim], dropout)
            for m, dim in modality_dims.items()
        })
        # Modality-specific projection heads: latent -> shared
        self.proj_heads = nn.ModuleDict({
            m: _mlp([latent_dim, shared_dim, shared_dim], dropout)
            for m in modality_dims
        })
        # Inverse projection heads: shared -> latent
        self.inv_proj_heads = nn.ModuleDict({
            m: _mlp([shared_dim, shared_dim, latent_dim], dropout)
            for m in modality_dims
        })

    def encode(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        return self.encoders[modality](x)

    def project(self, h: torch.Tensor, modality: str) -> torch.Tensor:
        return self.proj_heads[modality](h)

    def decode(self, z_shared: torch.Tensor, modality: str) -> torch.Tensor:
        h = self.inv_proj_heads[modality](z_shared)
        return self.decoders[modality](h)

    def aggregate(self, z_proj: dict, obs_mask: torch.Tensor) -> torch.Tensor:
        """
        Per-sample mean of projected embeddings for observed modalities.

        Args:
            z_proj:   {modality: (B, shared_dim)} – all modalities encoded & projected
            obs_mask: (B, M) bool – True where modality m is observed for sample b,
                      columns ordered by self.modality_names

        Returns:
            z_shared: (B, shared_dim)
        """
        stacked = torch.stack(
            [z_proj[m] for m in self.modality_names], dim=1
        )  # (B, M, D)
        mask = obs_mask.float().unsqueeze(-1)        # (B, M, 1)
        counts = mask.sum(dim=1).clamp(min=1.0)     # (B, 1)
        return (stacked * mask).sum(dim=1) / counts  # (B, D)
