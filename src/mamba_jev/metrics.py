"""
Metriche per calibrazione, selective classification e rejection.

Solo numpy/torch (nessuna dipendenza da scikit-learn), cosi' il pacchetto resta leggero.
Convenzione per le metriche di rejection: la classe POSITIVA e' "fuori dominio" (noul = 1) e uno
score ALTO significa "piu' fuori dominio".
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch


def _np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().double().numpy()
    return np.asarray(x, dtype=np.float64)


# ------------------------------------------------------------------ calibrazione
def ece(conf, correct, n_bins: int = 15) -> float:
    """Expected Calibration Error con bin equi-spaziati su [0, 1]."""
    conf, correct = _np(conf), _np(correct)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def brier_binary(p, y) -> float:
    p, y = _np(p), _np(y)
    return float(np.mean((p - y) ** 2))


def brier_multiclass(probs, labels) -> float:
    probs, labels = _np(probs), np.asarray(_np(labels), dtype=int)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll_multiclass(probs, labels, eps: float = 1e-12) -> float:
    probs, labels = _np(probs), np.asarray(_np(labels), dtype=int)
    return float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + eps)))


# ------------------------------------------------------------------ rejection
def auroc(scores, labels) -> float:
    """AUROC (Mann-Whitney, ranghi medi sui pareggi). labels==1 e' la classe positiva."""
    s, y = _np(scores), _np(labels) > 0.5
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0  # ranghi 1-based, media sui pareggi
    r = avg_rank[inv]
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fpr_at_tpr(scores, labels, tpr: float = 0.95) -> float:
    """FPR dei negativi (in-domain) alla soglia che cattura la frazione `tpr` dei positivi (OOD)."""
    s, y = _np(scores), _np(labels) > 0.5
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    thr = np.quantile(s[y], 1.0 - tpr)
    return float(np.mean(s[~y] >= thr))


# ------------------------------------------------------------------ selective classification
def aurc(conf, correct) -> float:
    """Area sotto la curva rischio-copertura (piu' basso = meglio). Ordina per confidenza decrescente."""
    conf, correct = _np(conf), _np(correct)
    order = np.argsort(-conf, kind="mergesort")
    err = 1.0 - correct[order]
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risk.mean())


# ------------------------------------------------------------------ bootstrap
def bootstrap_ci(
    fn: Callable[..., float], *arrays, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    """Intervallo bootstrap percentile per `fn(*arrays)`; i campioni sono ricampionati in modo appaiato."""
    arrs = [_np(a) for a in arrays]
    n = len(arrs[0])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(*[a[idx] for a in arrs])
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
