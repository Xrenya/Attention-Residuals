# Residual Attention Reproduction

## Variants

| Class | Memory | Description |
|---|---|---|
| `FullAttnResStack` | O(L x B x T x D) | Exact formulation; stores full hidden-state history |
| `BlockAttnResStack` | O(N x B x T x D) | AttnRes only at block boundaries; N = [L / block_size] |

---

## Install
```bash
pip install torch
pip install -e .
```

## Usage
```python
import torch

from attn_res import FullAttnResStack, BlockAttnResStack, build_attn_res


x = torch.randn(2, 128, 256) # (batch, seq_len, d_model)

# Full variant
model = FullAttnResStack(d_model=256, num_layers=12)
out   = model(x)  # (2, 128, 256)

# Block variant (memory-efficient)
model = BlockAttnResStack(d_model=256, num_layers=24, block_size=4)
out   = model(x)  # (2, 128, 256)

# Factory
model = build_attn_res('block', d_model=256, num_layers=24, block_size=4)
```

Reference:
1. [Technical Report of Residual Attention](https://arxiv.org/pdf/2603.15031)
2. [When does Kimi's "Attention Residuals" work?](https://kindxiaoming.github.io/blog/2026/attention-residual/)
