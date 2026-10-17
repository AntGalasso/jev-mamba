"""Configurazione di Mamba-Jev: parametri del backbone Mamba + dimensioni delle 3 teste."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

_DTYPES: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass
class MambaJevConfig:
    """Wrappa i parametri del backbone Mamba (MixerModel) + dimensioni delle 3 teste."""

    # --- backbone (rispecchia mamba_ssm.models.config_mamba.MambaConfig) ---
    d_model: int = 768
    n_layer: int = 24
    d_intermediate: int = 0
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 8
    ssm_cfg: Dict[str, Any] = field(default_factory=dict)  # es. {"layer": "Mamba2"}
    norm_epsilon: float = 1e-5
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    initializer_cfg: Optional[Dict[str, Any]] = None

    # --- teste di output ---
    num_choices: int = 8  # dimensione di choice_head
    score_dim: int = 1  # score_head: 1 logit -> sigmoide
    noul_dim: int = 1  # noul_head: 1 logit binario
    head_bias: bool = True  # intercetta nelle teste (il solver la stima non penalizzata); con False le teste sono bias-free

    # --- tokenizzazione / pooling ---
    pad_token_id: Optional[int] = 0  # usato solo se attention_mask non e' fornita

    # --- precisione ---
    param_dtype: str = "bfloat16"  # dtype dei pesi del backbone
    # Le teste restano SEMPRE in fp32: servono per precisione del solver e logit stabili.

    @property
    def padded_vocab_size(self) -> int:
        m = self.pad_vocab_size_multiple
        return self.vocab_size + (-self.vocab_size) % m if m > 1 else self.vocab_size

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.param_dtype]
