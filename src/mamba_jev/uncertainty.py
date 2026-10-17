"""
Calibrazione e rejection derivate dalle STESSE statistiche sufficienti del solver ridge.

Idea: per una ridge con matrice A_c = sum (x - mu)(x - mu)^T, il fattore di Cholesky L di (A_c + lam I)
serve gia' per risolvere i pesi. Lo stesso fattore da' gratis:
  * leverage    h(x) = 1/n + ||L^-1 (x - mu)||^2          (distanza dal dominio, stile Mahalanobis)
  * LOO exact   e_loo = e / (1 - h)                        (residuo leave-one-out, nessun refit)
  * Mahalanobis a covarianza condivisa per classe (LDA): le somme per classe sono le colonne di X^T Y_onehot.

Qui vivono: DomainModel (rejection), Calibration (temperatura / Platt) e i fit dei parametri di calibrazione.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================== rejection
class DomainModel(nn.Module):
    """
    Modello gaussiano del dominio, costruito dalle statistiche del solver (nessun passaggio extra sui dati).

    - `leverage(x)`: leverage ridge rispetto alle righe in-domain (media e covarianza globali).
    - `maha_tied(x)`: distanza di Mahalanobis minima verso le medie di classe, covarianza within-class condivisa
      (Lee et al. 2018). Richiede le statistiche della testa choice.
    Entrambi: valore ALTO = piu' lontano dal dominio.
    """

    def __init__(
        self,
        mu_in: torch.Tensor,
        chol_in: torch.Tensor,
        n_in: int,
        class_means: Optional[torch.Tensor] = None,
        chol_within: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("mu_in", mu_in)
        self.register_buffer("chol_in", chol_in)
        self.n_in = int(n_in)
        self.has_classes = class_means is not None and chol_within is not None
        if self.has_classes:
            self.register_buffer("class_means", class_means)
            self.register_buffer("chol_within", chol_within)

    @staticmethod
    def _whiten(chol: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        # chol e xc sono gia' in fp64; restituisce L^-1 xc con shape (n, D)
        return torch.linalg.solve_triangular(chol, xc.t(), upper=False).t()

    @torch.no_grad()
    def leverage(self, x: torch.Tensor) -> torch.Tensor:
        z = self._whiten(self.chol_in, x.to(self.mu_in) - self.mu_in)
        return (z * z).sum(1) + 1.0 / max(self.n_in, 1)

    @torch.no_grad()
    def maha_tied(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_classes:
            raise RuntimeError("Statistiche per-classe assenti: serve la testa choice per la Mahalanobis.")
        z = self._whiten(self.chol_within, x.to(self.class_means))  # (n, D)
        zc = self._whiten(self.chol_within, self.class_means)  # (C, D)
        d2 = (z * z).sum(1, keepdim=True) - 2.0 * z @ zc.t() + (zc * zc).sum(1)[None]
        return d2.clamp_min(0).min(dim=1).values


# ====================================================================== calibrazione
class Calibration(nn.Module):
    """
    Parametri di calibrazione post-hoc (pochi scalari), tutti buffer: seguono model.to(device) e state_dict.

        choice: softmax(logits / T)
        score : sigmoid(a_s * logit + b_s)            oppure, se score_from_msp, la MSP calibrata della choice
        noul  : sigmoid(a_n * logit + b_n)
    I valori di default sono l'identita': prima di calibrare il comportamento coincide con quello senza modulo.
    """

    def __init__(self) -> None:
        super().__init__()
        for name, val in (("choice_temperature", 1.0), ("score_a", 1.0), ("score_b", 0.0), ("noul_a", 1.0), ("noul_b", 0.0)):
            self.register_buffer(name, torch.tensor(val, dtype=torch.float32))
        self.register_buffer("score_from_msp", torch.tensor(False))

    def choice_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits / self.choice_temperature).softmax(-1)

    def score_prob(self, logit: torch.Tensor) -> torch.Tensor:
        return (self.score_a * logit + self.score_b).sigmoid()

    def noul_prob(self, logit: torch.Tensor) -> torch.Tensor:
        return (self.noul_a * logit + self.noul_b).sigmoid()


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Temperatura T che minimizza la NLL di softmax(logits / T). Ricerca su griglia logaritmica + raffinamento L-BFGS:
    i logit LS hanno scala ~1 (target one-hot), quindi T ottimale e' tipicamente << 1 e un avvio ingenuo da T=1 diverge.
    """
    z, y = logits.detach().double(), labels.detach().long()
    grid = torch.logspace(-3, 2, 51, dtype=torch.float64)
    nll = torch.stack([F.cross_entropy(z / t, y) for t in grid])
    log_t = torch.log(grid[nll.argmin()]).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def fit_platt(logit: torch.Tensor, y: torch.Tensor, l2: float = 1e-8, max_iter: int = 100) -> tuple[float, float]:
    """Platt scaling p = sigmoid(a * logit + b) via Newton con backtracking. Restituisce (a, b)."""
    s, t = logit.detach().double(), y.detach().double()
    xmat = torch.stack([s, torch.ones_like(s)], 1)
    w = torch.tensor([1.0, 0.0], dtype=torch.float64)
    reg = l2 * torch.eye(2, dtype=torch.float64)

    def nll(w_: torch.Tensor) -> float:
        z = xmat @ w_
        return float(F.binary_cross_entropy_with_logits(z, t, reduction="sum") + 0.5 * l2 * (w_ @ w_))

    cur = nll(w)
    for _ in range(max_iter):
        p = (xmat @ w).sigmoid()
        grad = xmat.t() @ (p - t) + l2 * w
        hess = (xmat * (p * (1 - p)).unsqueeze(1)).t() @ xmat + reg
        step = torch.linalg.solve(hess + 1e-12 * torch.eye(2, dtype=torch.float64), grad)
        scale = 1.0
        while scale > 1e-6:
            cand = nll(w - scale * step)
            if cand <= cur + 1e-12:
                break
            scale *= 0.5
        w, new = w - scale * step, nll(w - scale * step)
        if abs(cur - new) < 1e-12 * max(1.0, abs(cur)):
            break
        cur = new
    return float(w[0]), float(w[1])
