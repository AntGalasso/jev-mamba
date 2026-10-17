"""MambaJevModel: backbone Mamba + pooling sull'ultimo token + 3 teste lineari (+ calibrazione e modello di dominio)."""
from __future__ import annotations

import inspect
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

try:
    from mamba_ssm.models.mixer_seq_simple import MixerModel

    _HAS_MAMBA = True
except ImportError:  # pragma: no cover
    MixerModel = None  # type: ignore[assignment]
    _HAS_MAMBA = False

from .config import MambaJevConfig
from .types import JevOutput
from .uncertainty import Calibration, DomainModel


def _supported_kwargs(fn: Any, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Filtra i kwargs non supportati: la firma di MixerModel cambia tra versioni di mamba-ssm."""
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _build_mamba_backbone(cfg: MambaJevConfig, device: Optional[torch.device | str]) -> nn.Module:
    if not _HAS_MAMBA:
        raise ImportError("mamba-ssm non trovato: `pip install mamba-ssm causal-conv1d` (richiede CUDA).")
    kwargs = dict(
        d_model=cfg.d_model,
        n_layer=cfg.n_layer,
        d_intermediate=cfg.d_intermediate,
        vocab_size=cfg.padded_vocab_size,
        ssm_cfg=cfg.ssm_cfg,
        norm_epsilon=cfg.norm_epsilon,
        rms_norm=cfg.rms_norm,
        initializer_cfg=cfg.initializer_cfg,
        fused_add_norm=cfg.fused_add_norm,
        residual_in_fp32=cfg.residual_in_fp32,
        device=device,
        dtype=cfg.torch_dtype,
    )
    return MixerModel(**_supported_kwargs(MixerModel.__init__, kwargs))


class MambaJevModel(nn.Module):
    def __init__(
        self,
        cfg: MambaJevConfig,
        device: Optional[torch.device | str] = None,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        """
        `backbone` e' opzionale: qualunque nn.Module con forward(input_ids) -> (B, L, d_model).
        Se None viene costruito il MixerModel di mamba-ssm. L'iniezione serve per i test su CPU
        e per esperimenti con backbone alternativi.
        """
        super().__init__()
        self.cfg = cfg
        self._backbone_frozen = False
        self.backbone = backbone if backbone is not None else _build_mamba_backbone(cfg, device)

        # Tre teste indipendenti, fp32 (peso nn.Linear: (out, D); il solver produce (D, out)).
        d, hb = cfg.d_model, cfg.head_bias
        self.choice_head = nn.Linear(d, cfg.num_choices, bias=hb, device=device, dtype=torch.float32)
        self.score_head = nn.Linear(d, cfg.score_dim, bias=hb, device=device, dtype=torch.float32)
        self.noul_head = nn.Linear(d, cfg.noul_dim, bias=hb, device=device, dtype=torch.float32)

        # Calibrazione post-hoc (identita' finche' non viene fittata) e modello di dominio per il rejection.
        self.calibration = Calibration().to(device)
        self.domain: Optional[DomainModel] = None

    # ---------------------------------------------------------------- pooling / feature
    @staticmethod
    def _last_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
        """Indice dell'ultimo token con mask==1. Corretto sia con right- che con left-padding."""
        positions = torch.arange(1, attention_mask.shape[1] + 1, device=attention_mask.device)
        return (attention_mask.long() * positions).argmax(dim=1)

    def extract_features(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Restituisce X: (B, d_model), hidden state (post norm_f) dell'ultimo token non-pad.
        E' esattamente l'input delle teste lineari, quindi la matrice di design del solver.

        NB: Mamba e' causale, quindi con RIGHT-padding lo stato dell'ultimo token reale non e'
        contaminato dai pad. Con left-padding i pad entrano nello stato ricorrente: sconsigliato.
        Se `attention_mask` e' None viene inferita da `cfg.pad_token_id`; se anche questo e' None
        si usa l'ultima posizione.
        """
        hidden = self.backbone(input_ids)  # (B, L, D), dtype del backbone
        if attention_mask is None and self.cfg.pad_token_id is not None:
            attention_mask = input_ids != self.cfg.pad_token_id
        if attention_mask is None:
            return hidden[:, -1]
        idx = self._last_token_index(attention_mask)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]

    def apply_heads(self, feats: torch.Tensor) -> JevOutput:
        # Autocast disattivato: le teste lavorano in fp32 pieno (coerenza col solver closed-form).
        with torch.autocast(device_type=feats.device.type, enabled=False):
            f = feats.to(torch.float32)
            return JevOutput(
                choice_logits=self.choice_head(f),
                score_logit=self.score_head(f).squeeze(-1),
                noul_logit=self.noul_head(f).squeeze(-1),
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> JevOutput:
        return self.apply_heads(self.extract_features(input_ids, attention_mask))

    @torch.no_grad()
    def predict(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inferenza single-pass: un forward del backbone, zero decoding autoregressivo.

        Le probabilita' passano per `self.calibration` (identita' se non calibrata). Se e' presente un
        DomainModel si aggiungono `ood_leverage` e `ood_maha` (alto = fuori dominio).
        """
        feats = self.extract_features(input_ids, attention_mask)
        return self.predict_from_features(feats)

    @torch.no_grad()
    def predict_from_features(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.apply_heads(feats)
        probs = self.calibration.choice_probs(out.choice_logits)
        res = {
            "choice": out.choice_logits.argmax(-1),
            "choice_probs": probs,
            "score": probs.max(-1).values if bool(self.calibration.score_from_msp) else self.calibration.score_prob(out.score_logit),
            "noul_prob": self.calibration.noul_prob(out.noul_logit),
        }
        if self.domain is not None:
            f = feats.to(torch.float64)
            res["ood_leverage"] = self.domain.leverage(f).float()
            if self.domain.has_classes:
                res["ood_maha"] = self.domain.maha_tied(f).float()
        return res

    # ---------------------------------------------------------------- freeze / inject
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        self._backbone_frozen = False

    def train(self, mode: bool = True) -> "MambaJevModel":
        super().train(mode)
        if self._backbone_frozen:  # model.train() non deve "scongelare" il backbone
            self.backbone.eval()
        return self

    def _heads(self) -> Tuple[nn.Linear, nn.Linear, nn.Linear]:
        return self.choice_head, self.score_head, self.noul_head

    def freeze_heads(self) -> None:
        for h in self._heads():
            h.weight.requires_grad_(False)
            h.weight.grad = None

    def unfreeze_heads(self) -> None:
        for h in self._heads():
            h.weight.requires_grad_(True)

    @torch.no_grad()
    def inject_exact_weights(
        self,
        choice_w: torch.Tensor,
        score_w: torch.Tensor,
        noul_w: torch.Tensor,
        choice_b: Optional[torch.Tensor] = None,
        score_b: Optional[torch.Tensor] = None,
        noul_b: Optional[torch.Tensor] = None,
        freeze_heads: bool = True,
    ) -> None:
        """
        Carica i pesi closed-form (e le intercette) nelle teste, senza passare da autograd.

        Convenzione: i pesi arrivano con shape (D, out) (formato del solver); nn.Linear
        memorizza (out, D), quindi si trasporta. Un tensore 1-D (D,) e' trattato come (D, 1).
        Le intercette `*_b` hanno shape (out,); se la testa ha il bias e l'intercetta manca, il bias va a zero;
        se la testa e' bias-free e l'intercetta e' non nulla si solleva un errore (usa cfg.head_bias=True).
        """
        for name, head, w, b in (
            ("choice", self.choice_head, choice_w, choice_b),
            ("score", self.score_head, score_w, score_b),
            ("noul", self.noul_head, noul_w, noul_b),
        ):
            if w.dim() == 1:
                w = w.unsqueeze(1)
            expected = (head.in_features, head.out_features)
            if tuple(w.shape) != expected:
                raise ValueError(f"{name}_w: shape {tuple(w.shape)} != attesa {expected} (formato (D, out)).")
            head.weight.copy_(w.t().to(device=head.weight.device, dtype=head.weight.dtype))
            head.weight.grad = None
            if head.bias is not None:
                if b is None:
                    head.bias.zero_()
                else:
                    head.bias.copy_(b.reshape(-1).to(device=head.bias.device, dtype=head.bias.dtype))
                head.bias.grad = None
            elif b is not None and bool((b.abs() > 0).any()):
                raise ValueError(f"{name}: intercetta non nulla ma la testa e' bias-free (cfg.head_bias=False).")
        if freeze_heads:
            self.freeze_heads()

    @torch.no_grad()
    def load_fit(self, fit, freeze_heads: bool = True) -> None:
        """Carica pesi, intercette, calibrazione e DomainModel di un `pipeline.JevFit`."""
        self.inject_exact_weights(**fit.weights, freeze_heads=freeze_heads)
        self.calibration.load_state_dict(fit.calibration.state_dict())
        self.domain = fit.domain.to(self.choice_head.weight.device) if fit.domain is not None else None

    @torch.no_grad()
    def expand_choice_head(self, extra: int) -> None:
        """
        Allarga la testa choice di `extra` classi (le righe nuove partono da zero). Va usato con
        `ExactHeadSolver.expand_choices(extra)` e poi `inject_exact_weights` con la soluzione aggiornata.
        """
        old = self.choice_head
        new = nn.Linear(old.in_features, old.out_features + extra, bias=old.bias is not None,
                        device=old.weight.device, dtype=old.weight.dtype)
        new.weight.zero_()
        new.weight[: old.out_features].copy_(old.weight)
        if old.bias is not None:
            new.bias.zero_()
            new.bias[: old.out_features].copy_(old.bias)
        new.weight.requires_grad_(old.weight.requires_grad)
        self.choice_head = new
        self.cfg.num_choices += extra
