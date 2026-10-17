"""Loss combinata per le tre teste, con pesi bilanciabili e mascheramento per riga."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import JevOutput, JevTargets


class CombinedJevLoss(nn.Module):
    """L = w_c * CE(choice) + w_s * BCE|MSE(score) + w_n * BCE(noul), con mascheramento per riga."""

    def __init__(
        self,
        w_choice: float = 1.0,
        w_score: float = 1.0,
        w_noul: float = 1.0,
        score_loss: str = "bce",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if score_loss not in ("bce", "mse"):
            raise ValueError("score_loss deve essere 'bce' o 'mse'")
        self.w_choice, self.w_score, self.w_noul = w_choice, w_score, w_noul
        self.score_loss = score_loss
        self.ignore_index = ignore_index

    def forward(self, out: JevOutput, tgt: JevTargets) -> Tuple[torch.Tensor, Dict[str, float]]:
        zero = out.choice_logits.sum() * 0.0  # mantiene il grafo anche se una testa non ha righe valide

        valid_c = tgt.choice != self.ignore_index
        l_choice = (
            F.cross_entropy(out.choice_logits.float()[valid_c], tgt.choice[valid_c]) if valid_c.any() else zero
        )

        valid_s = tgt.score_mask if tgt.score_mask is not None else torch.ones_like(tgt.score, dtype=torch.bool)
        if valid_s.any():
            logit, y = out.score_logit.float()[valid_s], tgt.score.float()[valid_s]
            l_score = (
                F.binary_cross_entropy_with_logits(logit, y)
                if self.score_loss == "bce"
                else F.mse_loss(logit.sigmoid(), y)
            )
        else:
            l_score = zero

        l_noul = F.binary_cross_entropy_with_logits(out.noul_logit.float(), tgt.noul.float())

        total = self.w_choice * l_choice + self.w_score * l_score + self.w_noul * l_noul
        parts = {
            "loss": float(total.detach()),
            "choice": float(l_choice.detach()),
            "score": float(l_score.detach()),
            "noul": float(l_noul.detach()),
        }
        return total, parts
