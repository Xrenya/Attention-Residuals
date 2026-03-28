"""
Attention Residuals (AttnRes)
Paper: "Attention Residuals" - Kimi Team / MoonshotAI, arXiv:2603.15031 (2026)

Two variants are provided:
- FullAttnResStack  - exact formulation; O(L * B * T * D) history memory.
- BlockAttnResStack - memory-efficient; applies AttnRes only at block boundaries;
                        O(N * B * T * D) memory, N = ceil(L / block_size)

Usage
    model = FullAttnResStack(d_model=256, num_layers=12)
    x = torch.randn(2, 128, 256)  # (batch, seq_len, d_model)
    out  = model(x)  # (2, 128, 256)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# FF block
def _ff_block(d_model: int, expansion: int = 4) -> nn.Sequential:
    """PreNorm feed-forward sublayer: LN - Linear - GELU - Linear"""
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model * expansion),
        nn.GELU(),
        nn.Linear(d_model * expansion, d_model),
    )


# Full Attention Residuals
class FullAttnResStack(nn.Module):
    """
    Input / output shape: (B, T, D) => (B, T, D)

    Memory: O(L * B * T * D) - full hidden-state history is kept
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 12,
        ff_expand: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.scale = math.sqrt(d_model) ** -1

        self.queries = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_model)) for _ in range(num_layers)]
        )
        self.key_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )
        self.ff_blocks = nn.ModuleList(
            [_ff_block(d_model, ff_expand) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input embeddings (h_0)
        Returns:
            (B, T, D) output of the final layer
        """
        history = [x]  # h_0 = input embeddings
        h = x

        for query, key_norm, ff in zip(self.queries, self.key_norms, self.ff_blocks):
            # Stack all prior hidden states: (L_prev, B, T, D)
            V = torch.stack(history, dim=0)
            K = key_norm(V)

            # Depth-wise attention scores: w_l * LN(h_i) / sqrt(D)
            scores = torch.einsum("d, l b t d -> l b t", query, K) * self.scale
            alpha = scores.softmax(dim=0)  # (L_prev, B, T)

            # Attention-weighted residual (replaces the fixed + h_{l-1} skip)
            residual = torch.einsum("l b t, l b t d -> b t d", alpha, V)

            # Standard sublayer output + AttnRes skip
            h = residual + ff(h)
            history.append(h)

        return h  # (B, T, D)

# Block Attention Residuals
class BlockAttnResStack(nn.Module):
    """
    Block Attention Residuals - memory-efficient AttnRes variant

    Input / output shape: (B, T, D) => (B, T, D)

    Memory: O(N * B * T * D), N = ceil(L / block_size)
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 24,
        block_size: int = 4,
        ff_expand: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.scale = math.sqrt(d_model) ** -1

        self.ff_blocks = nn.ModuleList(
            [_ff_block(d_model, ff_expand) for _ in range(num_layers)]
        )
        num_blocks = (num_layers + block_size - 1) // block_size
        # Zero-init so each block boundary starts with uniform aggregation
        self.block_queries = nn.Parameter(torch.zeros(num_blocks, d_model))
        self.block_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input embeddings.
        Returns:
            (B, T, D) output after all layers.
        """
        h = x
        block_reps: list[torch.Tensor] = []
        b_idx = 0

        for i, ff in enumerate(self.ff_blocks):
            h = h + ff(h)  # standard residual within block

            if (i + 1) % self.block_size == 0:  # block boundary
                block_reps.append(h)
                if len(block_reps) > 1:
                    V = torch.stack(block_reps, dim=0)  # (N, B, T, D)
                    K = self.block_norm(V)
                    scores = (
                        torch.einsum("d, n b t d -> n b t", self.block_queries[b_idx], K)
                        * self.scale
                    )
                    alpha = scores.softmax(dim=0)
                    h = torch.einsum("n b t, n b t d -> b t d", alpha, V)
                b_idx += 1

        return h  # (B, T, D)


# ── Convenience factory ───────────────────────────────────────────────────────

def build_attn_res(
    variant: str,
    d_model: int,
    num_layers: int,
    *,
    block_size: int = 4,
    ff_expand: int = 4,
) -> nn.Module:
    if variant == "full":
        return FullAttnResStack(d_model, num_layers, ff_expand=ff_expand)
    elif variant == "block":
        return BlockAttnResStack(d_model, num_layers, block_size=block_size, ff_expand=ff_expand)
    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose 'full' or 'block'.")

# Test
if __name__ == "__main__":
    import sys

    torch.manual_seed(42)
    B, T, D = 2, 32, 128

    x = torch.randn(B, T, D)
    print(f"Input x.shape = {x.shape}")
    for variant, model in [
        ("full", FullAttnResStack(D, num_layers=6)),
        ("block", BlockAttnResStack(D, num_layers=12, block_size=4)),
    ]:
        out = model(x)
        params = sum(p.numel() for p in model.parameters())
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
        print(f"[{variant:5s}] output {tuple(out.shape)} | params {params:,}")
