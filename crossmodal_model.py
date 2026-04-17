"""
Cross-Modal Attention Imputer (CMAI)

Architecture per modality m:
    tokens_m = FeatureTokenizer_m(x_m)              (B, D_m, d_model)
    z_m      = PerceiverCompressor_m(tokens_m)       (B, K, d_model)
    z_m      = WithinModalTransformer_m(z_m)         (B, K, d_model)

To impute target t from observed set S:
    source = concat([z_m for m in S], dim=1)         (B, |S|·K, d_model)
    q_t    = target_queries_t (learned)              (B, K, d_model)
    q_t    = CrossModalTransformer_t(Q=q_t, KV=src)  (B, K, d_model)
    x_hat  = Decoder_t(q_t.mean(dim=1))              (B, D_t)

Key advantage over MIMIRPhase2: cross-modal attention replaces mean-pooling,
allowing the model to learn which features in the source modality are most
informative for each feature in the target modality.
"""

import torch
import torch.nn as nn


def _mlp(dims: list, dropout: float = 0.1) -> nn.Sequential:
    layers: list = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(dims[i + 1]), nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class FeatureTokenizer(nn.Module):
    """
    Maps raw feature values (B, D) to token sequence (B, D, d_model).
    Each position gets: value_projection(x_i) + learned_feature_embedding_i
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.feat_embed = nn.Embedding(n_features, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D) → (B, D, d_model)
        val_tok = self.value_proj(x.unsqueeze(-1))        # (B, D, d_model)
        pos_tok = self.feat_embed.weight.unsqueeze(0)     # (1, D, d_model)
        return self.norm(val_tok + pos_tok)


class PerceiverCompressor(nn.Module):
    """
    Perceiver-style cross-attention: compress D feature tokens to K latent tokens.
    K learnable queries attend to all D feature tokens → O(K·D) instead of O(D²).
    """

    def __init__(self, n_latents: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = _mlp([d_model, d_model * 2, d_model], dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D, d) → (B, K, d)
        B = x.shape[0]
        q = self.latents.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.cross_attn(q, x, x)
        q = self.norm1(q + out)
        q = self.norm2(q + self.ff(q))
        return q


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block with self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = _mlp([d_model, d_model * 2, d_model], dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block: query attends to separate key/value sequence."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = _mlp([d_model, d_model * 2, d_model], dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(q, kv, kv)
        q = self.norm1(q + out)
        q = self.norm2(q + self.ff(q))
        return q


class CrossModalAttentionImputer(nn.Module):
    """
    Cross-Modal Attention Imputer (CMAI).

    Args:
        modality_dims:    {name: feature_dim}
        d_model:          token embedding dimension
        n_latents:        K — number of latent tokens per modality after compression
        n_heads:          attention heads (must divide d_model)
        n_within_layers:  transformer layers applied within each modality's K tokens
        n_cross_layers:   cross-modal attention layers for imputation
        decoder_hidden:   MLP hidden dims for feature decoder
        proj_dim:         dimension of projection head (for contrastive loss)
        dropout:          dropout probability
    """

    def __init__(
        self,
        modality_dims: dict,
        d_model: int = 64,
        n_latents: int = 32,
        n_heads: int = 4,
        n_within_layers: int = 2,
        n_cross_layers: int = 2,
        decoder_hidden: tuple = (256, 512),
        proj_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.modality_dims = modality_dims
        self.d_model = d_model
        self.n_latents = n_latents

        self.tokenizers = nn.ModuleDict(
            {m: FeatureTokenizer(dim, d_model) for m, dim in modality_dims.items()}
        )
        self.compressors = nn.ModuleDict(
            {
                m: PerceiverCompressor(n_latents, d_model, n_heads, dropout)
                for m in modality_dims
            }
        )
        self.within_tf = nn.ModuleDict(
            {
                m: nn.Sequential(
                    *[TransformerBlock(d_model, n_heads, dropout) for _ in range(n_within_layers)]
                )
                for m in modality_dims
            }
        )

        # Learned query tokens for each target modality during imputation
        self.target_queries = nn.ParameterDict(
            {
                m: nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
                for m in modality_dims
            }
        )
        self.cross_tf = nn.ModuleDict(
            {
                m: nn.Sequential(
                    *[CrossAttentionBlock(d_model, n_heads, dropout) for _ in range(n_cross_layers)]
                )
                for m in modality_dims
            }
        )

        dec = list(decoder_hidden)
        self.decoders = nn.ModuleDict(
            {m: _mlp([d_model] + dec + [dim], dropout) for m, dim in modality_dims.items()}
        )

        # Projection head: compressed repr → proj_dim (for InfoNCE contrastive loss)
        self.proj_heads = nn.ModuleDict(
            {
                m: _mlp([d_model, proj_dim, proj_dim], dropout)
                for m in modality_dims
            }
        )

    # ── Core operations ────────────────────────────────────────────────────────

    def compress(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """(B, D_m) → (B, K, d_model)"""
        tokens = self.tokenizers[modality](x)       # (B, D, d_model)
        z = self.compressors[modality](tokens)      # (B, K, d_model)
        z = self.within_tf[modality](z)             # (B, K, d_model)
        return z

    def project(self, z: torch.Tensor, modality: str) -> torch.Tensor:
        """(B, K, d_model) → (B, proj_dim) — used for contrastive loss."""
        return self.proj_heads[modality](z.mean(dim=1))

    def impute(self, z_observed: list, target: str) -> torch.Tensor:
        """
        Impute target modality via cross-modal attention.

        Args:
            z_observed: list of (B, K, d_model) tensors from observed modalities
            target:     name of the modality to impute

        Returns:
            (B, D_target) imputed features
        """
        B = z_observed[0].shape[0]
        source = torch.cat(z_observed, dim=1)   # (B, |S|·K, d_model)
        q = self.target_queries[target].unsqueeze(0).expand(B, -1, -1).contiguous()
        for layer in self.cross_tf[target]:
            q = layer(q, source)                # (B, K, d_model)
        return self.decoders[target](q.mean(dim=1))  # (B, D_target)

    def reconstruct(self, z: torch.Tensor, modality: str) -> torch.Tensor:
        """Self-reconstruct from compressed representation (B, K, d) → (B, D)."""
        return self.decoders[modality](z.mean(dim=1))
