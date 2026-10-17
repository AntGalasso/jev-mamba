"""
Pipeline completa a partire da feature in memoria: ridge in forma chiusa -> LOO esatto -> calibrazione -> rejection.

    fit = fit_jev_heads(X, targets, num_choices=C)
    model.load_fit(fit)            # pesi + intercette + calibrazione + DomainModel

Tutto senza validation set: temperatura e Platt sono stimati sulle predizioni leave-one-out del training set,
che per la ridge si ottengono dallo stesso fattore di Cholesky (HeadFit.loo_predict), senza alcun refit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .solver import ExactHeadSolver, HeadFit
from .types import JevTargets
from .uncertainty import Calibration, DomainModel, fit_platt, fit_temperature

logger = logging.getLogger(__name__)


@dataclass
class JevFit:
    weights: Dict[str, torch.Tensor]  # {choice_w, score_w, noul_w, choice_b, score_b, noul_b}
    calibration: Calibration
    domain: Optional[DomainModel]
    solver: ExactHeadSolver
    heads: Dict[str, HeadFit]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _holdout_rho(solver: ExactHeadSolver, hx: torch.Tensor, ht: JevTargets) -> Dict[str, float]:
    """rho per testa: choice sull'errore di classificazione held-out, noul sull'MSE held-out; lo score resta su GCV."""
    x = hx.to(device=solver.device, dtype=solver.accum_dtype)
    rho: Dict[str, float] = {"score": solver.select_ridge("score")[0] if int(solver.stats["score"].n) > 1 else 1.0}
    ch = ht.choice.to(solver.device)
    v = ch != solver.ignore_index
    rho["choice"] = solver.select_ridge_holdout(
        "choice", x[v], F.one_hot(ch[v], solver.num_choices).to(x) * solver.choice_scale, criterion="acc"
    )[0]
    nl = ht.noul.to(solver.device).float() > 0.5
    yn = torch.where(nl, solver.logit_margin, -solver.logit_margin).to(x).unsqueeze(1)
    rho["noul"] = solver.select_ridge_holdout("noul", x, yn)[0] if 0 < int(nl.sum()) < len(nl) else solver.select_ridge("noul")[0]
    return rho


def _calibrate_on_holdout(solver, fits, cal, diag, holdout, num_choices) -> None:
    hx, ht = holdout
    x = hx.to(device=solver.device, dtype=solver.accum_dtype)
    ch = ht.choice.to(solver.device)
    v = ch != solver.ignore_index
    if v.any():
        logits = fits["choice"].predict(x[v])
        t = fit_temperature(logits, ch[v])
        cal.choice_temperature.fill_(t)
        diag.update(choice_temperature=t, choice_holdout_acc=float((logits.argmax(1) == ch[v]).float().mean()))
    nl = ht.noul.to(solver.device).float()
    if 0 < float(nl.sum()) < len(nl):
        a, b = fit_platt(fits["noul"].predict(x).squeeze(1), nl)
        cal.noul_a.fill_(a)
        cal.noul_b.fill_(b)
        diag.update(noul_platt=(a, b))


def fit_jev_heads(
    feats: torch.Tensor,
    tgt: JevTargets,
    num_choices: int,
    *,
    ridge: Union[float, str] = "gcv",  # numero | "gcv" | "holdout"
    score_mode: str = "msp",
    loo_calibration: bool = True,
    holdout: Optional[Tuple[torch.Tensor, JevTargets]] = None,
    domain: bool = True,
    solver_kwargs: Optional[Dict[str, Any]] = None,
) -> JevFit:
    """
    Fit esatto delle tre teste + calibrazione LOO + modello di dominio, da feature gia' estratte (N, D).

    holdout: (feats, targets) NON usati nel fit. Se presente, ha la precedenza sul LOO: la temperatura e il Platt
      sono stimati sulle predizioni held-out e `ridge="holdout"` sceglie rho per testa (choice: accuratezza, noul: MSE).
      Obbligatorio nella pratica se il backbone e' stato addestrato sulle stesse righe del fit (Stage 1 sul training
      set): in quel caso i residui LOO sono ottimisti perche' misurano solo il refit delle teste, non quello del
      backbone (vedi docs/RESULTS.md).

    score_mode:
      "user"        : lo score e' regredito sui target forniti in `tgt.score` (come ExactHeadSolver di base).
      "loo_correct" : il target dello score e' la CORRETTEZZA LOO della choice (1 se la predizione leave-one-out
                      coincide con la label): niente ottimismo in-sample, nessun target esterno richiesto.
                      Lo score finale e' P(choice corretta | x), calibrata con Platt sul LOO della testa score.
      "msp"         : nessuna testa score: lo score e' la probabilita' massima della choice, calibrata con la
                      temperatura LOO (vedi docs/RESULTS.md: su CLINC150 ordina gli errori meglio della testa lineare).
    """
    if score_mode not in ("user", "loo_correct", "msp"):
        raise ValueError("score_mode deve essere 'user', 'loo_correct' o 'msp'")
    if holdout is not None and score_mode == "loo_correct":
        raise ValueError("score_mode='loo_correct' usa il LOO e non e' compatibile con holdout: usa 'msp'.")
    d = feats.shape[1]
    solver = ExactHeadSolver(d, num_choices, ridge=ridge, device=feats.device, **(solver_kwargs or {}))
    solver.accumulate(feats, tgt)
    if holdout is not None and ridge == "holdout":
        solver.ridge = ridge = _holdout_rho(solver, *holdout)
    elif ridge == "holdout":
        raise ValueError("ridge='holdout' richiede holdout=(feats, targets).")
    fits = solver.fit()
    cal = Calibration()
    diag: Dict[str, Any] = {"rho": {h: f.rho for h, f in fits.items()}}

    x64 = feats.to(device=solver.device, dtype=solver.accum_dtype)
    choice = tgt.choice.to(solver.device)
    vc = choice != solver.ignore_index

    if holdout is not None:
        _calibrate_on_holdout(solver, fits, cal, diag, holdout, num_choices)
    elif loo_calibration and vc.any():
        # ---- choice: temperatura sulle predizioni LOO
        y1h = F.one_hot(choice[vc], num_choices).to(solver.accum_dtype) * solver.choice_scale
        logits_loo, _ = fits["choice"].loo_predict(x64[vc], y1h)
        t = fit_temperature(logits_loo, choice[vc])
        cal.choice_temperature.fill_(t)
        correct = (logits_loo.argmax(1) == choice[vc]).float()
        diag.update(choice_temperature=t, choice_loo_acc=float(correct.mean()))
        diag["choice_train_acc"] = float((fits["choice"].predict(x64[vc]).argmax(1) == choice[vc]).float().mean())

        # ---- score: target = correttezza LOO, poi Platt sul LOO della testa score
        if score_mode == "loo_correct":
            solver.reset_head("score")
            solver.accumulate_score(x64[vc], correct)
            fits["score"] = solver.fit_head("score")
            ys = solver._score_target(correct)
            s_loo, _ = fits["score"].loo_predict(x64[vc], ys)
            a, b = fit_platt(s_loo.squeeze(1), correct)
            cal.score_a.fill_(a)
            cal.score_b.fill_(b)
            diag.update(score_platt=(a, b))

        # ---- noul: Platt sul LOO della testa noul (tutte le righe)
        noul = tgt.noul.to(solver.device).float()
        yn = torch.where(noul > 0.5, solver.logit_margin, -solver.logit_margin).to(solver.accum_dtype).unsqueeze(1)
        n_loo, _ = fits["noul"].loo_predict(x64, yn)
        if 0 < float(noul.sum()) < len(noul):
            a, b = fit_platt(n_loo.squeeze(1), noul)
            cal.noul_a.fill_(a)
            cal.noul_b.fill_(b)
            diag.update(noul_platt=(a, b))
    elif score_mode != "user":
        logger.warning("score_mode='%s' richiede loo_calibration=True e label choice valide: score lasciato sui target forniti.", score_mode)
    if score_mode == "msp" and (loo_calibration or holdout is not None) and vc.any():
        cal.score_from_msp.fill_(True)

    dom = solver.domain_model() if domain and int(solver.domain_stats.n) >= 2 else None
    return JevFit(solver.solve(), cal, dom, solver, fits, diag)
