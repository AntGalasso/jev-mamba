"""
Backbone SSM selettivo in puro PyTorch (stile Mamba-1), per CPU/test/esperimenti piccoli.

NON sostituisce `mamba-ssm`: la scansione e' un ciclo sequenziale sui token (lenta, ma identica nella struttura:
proiezione in ingresso, conv causale depthwise, parametri Delta/B/C dipendenti dall'input, stato nascosto
ricorrente, gating). Serve a provare l'intero flusso Stage 1 (SGD) -> Stage 2 (teste in forma chiusa) senza GPU.
Interfaccia compatibile con `MambaJevModel(cfg, backbone=...)`: forward(input_ids) -> (B, L, d_model) post norm finale.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, d_conv: int = 4) -> None:
        super().__init__()
        di = expand * d_model
        self.d_inner, self.d_state, self.dt_rank = di, d_state, max(d_model // 16, 1)
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * di, bias=False)
        self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        self.x_proj = nn.Linear(di, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        xi, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        xi = F.silu(self.conv(xi.transpose(1, 2))[..., :length].transpose(1, 2))  # conv causale
        dt, bm, cm = self.x_proj(xi).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, di)
        a = -torch.exp(self.A_log.float())  # (di, N)
        da = torch.exp(dt.unsqueeze(-1) * a)  # (B, L, di, N)
        dbx = dt.unsqueeze(-1) * bm.unsqueeze(2) * xi.unsqueeze(-1)  # (B, L, di, N)
        h = x.new_zeros(b, self.d_inner, self.d_state)
        ys = []
        for t in range(length):  # scansione ricorrente: lo stato al passo t dipende solo dai token <= t
            h = da[:, t] * h + dbx[:, t]
            ys.append((h * cm[:, t].unsqueeze(1)).sum(-1))
        y = (torch.stack(ys, 1) + xi * self.D) * F.silu(z)
        return x + self.out_proj(y)


class TinySSMBackbone(nn.Module):
    """Embedding + n_layer blocchi SSM + RMSNorm finale. forward(input_ids) -> (B, L, d_model)."""

    def __init__(self, vocab_size: int, d_model: int = 128, n_layer: int = 2, d_state: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(SSMBlock(d_model, d_state) for _ in range(n_layer))
        self.norm_f = RMSNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.embedding(input_ids))
        for layer in self.layers:
            h = self.drop(layer(h))
        return self.norm_f(h)
