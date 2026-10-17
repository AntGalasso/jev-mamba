"""Training loop con branch standard (SGD) e branch exact-solver (teste in forma chiusa)."""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch

from .loss import CombinedJevLoss
from .model import MambaJevModel
from .pipeline import fit_jev_heads
from .solver import ExactHeadSolver
from .types import JevTargets

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 3
    use_exact_solver: bool = False
    grad_clip: float = 1.0
    log_every: int = 50
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16  # None = nessun autocast
    # iperparametri del solver
    solver_ridge: Union[float, str] = 1e-4  # rho relativo, oppure "gcv" (scelto dalle statistiche, nessun passaggio extra)
    solver_score_eps: float = 0.02
    solver_logit_margin: float = 3.0
    solver_fit_intercept: bool = True
    # calibrazione LOO + DomainModel (richiede di tenere le feature in memoria: N x D, fp32 su CPU).
    # ATTENZIONE: il LOO e' valido solo se il backbone NON e' stato addestrato sulle righe del fit (vedi docs/RESULTS.md).
    loo_calibration: bool = False
    score_mode: str = "msp"  # "msp" | "user" | "loo_correct" (vedi pipeline.fit_jev_heads)


def _autocast(device: torch.device, dtype: Optional[torch.dtype]):
    """torch.autocast, oppure nullcontext se dtype e' None (utile su CPU dove bf16 puo' essere lento)."""
    return torch.autocast(device.type, dtype=dtype) if dtype is not None else contextlib.nullcontext()


def _cat_targets(parts: List[JevTargets]) -> JevTargets:
    has_mask = any(t.score_mask is not None for t in parts)
    mask = (
        torch.cat([t.score_mask if t.score_mask is not None else torch.ones_like(t.score, dtype=torch.bool) for t in parts])
        if has_mask
        else None
    )
    return JevTargets(
        choice=torch.cat([t.choice for t in parts]),
        score=torch.cat([t.score for t in parts]),
        noul=torch.cat([t.noul for t in parts]),
        score_mask=mask,
    )


def _unpack_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor], JevTargets]:
    ids = batch["input_ids"].to(device, non_blocking=True)
    mask = batch.get("attention_mask")
    mask = mask.to(device, non_blocking=True) if mask is not None else None
    smask = batch.get("score_mask")
    tgt = JevTargets(
        choice=batch["choice_label"].to(device),
        score=batch["score_target"].to(device).float(),
        noul=batch["noul_label"].to(device).float(),
        score_mask=smask.to(device) if smask is not None else None,
    )
    return ids, mask, tgt


@torch.no_grad()
def evaluate(model: MambaJevModel, loss_fn: CombinedJevLoss, loader: Iterable, device: torch.device, autocast_dtype: Optional[torch.dtype] = torch.bfloat16) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    agg: Dict[str, float] = {}
    correct = total = n_batches = 0
    for batch in loader:
        ids, mask, tgt = _unpack_batch(batch, device)
        with _autocast(device, autocast_dtype):
            out = model(ids, mask)
        _, parts = loss_fn(out, tgt)
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        valid = tgt.choice != loss_fn.ignore_index
        correct += int((out.choice_logits.argmax(-1)[valid] == tgt.choice[valid]).sum())
        total += int(valid.sum())
        n_batches += 1
    model.train(was_training)
    res = {k: v / max(n_batches, 1) for k, v in agg.items()}
    res["choice_acc"] = correct / max(total, 1)
    return res


def train(
    model: MambaJevModel,
    loss_fn: CombinedJevLoss,
    loader: Iterable,
    cfg: TrainConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device | str = "cuda",
    val_loader: Optional[Iterable] = None,
) -> None:
    device = torch.device(device)
    model.to(device)
    if not cfg.use_exact_solver and optimizer is None:
        raise ValueError("Il branch standard richiede un optimizer.")

    solver = (
        ExactHeadSolver(
            feature_dim=model.cfg.d_model,
            num_choices=model.cfg.num_choices,
            ridge=cfg.solver_ridge,
            score_eps=cfg.solver_score_eps,
            logit_margin=cfg.solver_logit_margin,
            fit_intercept=cfg.solver_fit_intercept,
            ignore_index=loss_fn.ignore_index,
            device=device,
        )
        if cfg.use_exact_solver
        else None
    )

    # Con backbone congelato la soluzione closed-form e' deterministica: un solo passaggio basta.
    n_epochs = 1 if cfg.use_exact_solver else cfg.epochs

    for epoch in range(n_epochs):
        if cfg.use_exact_solver:
            # ============================ BRANCH EXACT SOLVER ============================
            # 1) Nessun gradiente per le teste: azzeriamo i grad residui e le escludiamo da autograd.
            #    Il backbone viene congelato: X deve restare FISSA durante tutto il passaggio,
            #    altrimenti (X^T X, X^T Y) accumulate su batch diversi non descriverebbero
            #    la stessa mappa feature -> pesi.
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            model.freeze_backbone()
            model.freeze_heads()
            assert solver is not None
            solver.reset()

            # 2) Passaggio di sola lettura sul dataset: estrazione feature + accumulo streaming.
            #    Si accumulano solo A = X^T X e B = X^T Y (memoria O(D^2)), mai X per intero,
            #    salvo cfg.loo_calibration=True: li' le feature vengono tenute (CPU) per il leave-one-out esatto.
            model.eval()
            cache_x: List[torch.Tensor] = []
            cache_t: List[JevTargets] = []
            with torch.no_grad():
                for batch in loader:
                    ids, mask, tgt = _unpack_batch(batch, device)
                    with _autocast(device, cfg.autocast_dtype):
                        feats = model.extract_features(ids, mask)  # (B, D), pre-teste
                    if cfg.loo_calibration:
                        cache_x.append(feats.detach().float().cpu())
                        cache_t.append(JevTargets(*(None if v is None else v.cpu() for v in (tgt.choice, tgt.score, tgt.noul, tgt.score_mask))))
                    else:
                        # >>> PUNTO DI INIEZIONE #1: qui puoi sostituire solver.accumulate(...) con
                        #     il tuo Exact QP Solver (stessa interfaccia: feats + target per testa).
                        solver.accumulate(feats, tgt)

            if cfg.loo_calibration:
                logger.warning(
                    "loo_calibration: le predizioni LOO sono ottimiste se il backbone e' stato addestrato sulle stesse righe "
                    "(Stage 1). Valido con backbone preaddestrato/esterno; altrimenti calibra su un held-out (pipeline.fit_jev_heads(holdout=...))."
                )
                # Pipeline completa: ridge (rho per GCV se richiesto) -> LOO -> calibrazione -> DomainModel.
                fit = fit_jev_heads(
                    torch.cat(cache_x).to(device),
                    _cat_targets(cache_t),
                    model.cfg.num_choices,
                    ridge=cfg.solver_ridge,
                    score_mode=cfg.score_mode,
                    solver_kwargs=dict(
                        score_eps=cfg.solver_score_eps,
                        logit_margin=cfg.solver_logit_margin,
                        fit_intercept=cfg.solver_fit_intercept,
                        ignore_index=loss_fn.ignore_index,
                    ),
                )
                model.load_fit(fit)
                logger.info("[exact] diagnostica LOO | %s", {k: v for k, v in fit.diagnostics.items()})
            else:
                # 3) Multi-GPU: le statistiche sono additive, la somma tra rank da' la soluzione
                #    esatta sul dataset globale.
                solver.all_reduce()

                # 4) Risoluzione closed-form  W = (A_c + lambda I)^-1 B_c  per ciascuna testa.
                weights = solver.solve()  # {"choice_w": (D,C), ..., "choice_b": (C,), ...}

                # >>> PUNTO DI INIEZIONE #2: i pesi entrano nelle teste via copy_ sotto no_grad,
                #     senza forward/backward e senza optimizer.step().
                model.inject_exact_weights(**weights)

            metrics = evaluate(model, loss_fn, loader, device, cfg.autocast_dtype)
            logger.info("[exact] soluzione iniettata (train) | %s", {k: round(v, 4) for k, v in metrics.items()})
            if val_loader is not None:
                vm = evaluate(model, loss_fn, val_loader, device, cfg.autocast_dtype)
                logger.info("[exact] soluzione iniettata (val)   | %s", {k: round(v, 4) for k, v in vm.items()})
        else:
            # ============================ BRANCH STANDARD (SGD) ============================
            model.train()
            for step, batch in enumerate(loader):
                ids, mask, tgt = _unpack_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
                with _autocast(device, cfg.autocast_dtype):
                    out = model(ids, mask)
                loss, parts = loss_fn(out, tgt)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()  # type: ignore[union-attr]
                if step % cfg.log_every == 0:
                    logger.info("[sgd] ep %d step %d | %s", epoch, step, {k: round(v, 4) for k, v in parts.items()})
