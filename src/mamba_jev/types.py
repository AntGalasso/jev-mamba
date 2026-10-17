"""Container tipizzati per output e target delle tre teste."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class JevOutput:
    choice_logits: torch.Tensor  # (B, num_choices)
    score_logit: torch.Tensor  # (B,)   -> sigmoid per [0, 1]
    noul_logit: torch.Tensor  # (B,)   -> sigmoid per P(out-of-domain)


@dataclass
class JevTargets:
    choice: torch.Tensor  # (B,) long, ignore_index dove non definito
    score: torch.Tensor  # (B,) float in [0, 1]
    noul: torch.Tensor  # (B,) float in {0, 1}
    score_mask: Optional[torch.Tensor] = None  # (B,) bool: righe con score valido
