"""
Solver closed-form (ridge, in streaming) per le teste lineari.

Interfaccia di riferimento (reset / accumulate / all_reduce / solve): e' quella che un Exact QP Solver
alternativo deve rispettare per essere sostituito senza toccare il resto.

Oltre ai pesi, le statistiche sufficienti (A = sum x x^T, B = sum x y^T, somme, conteggi) bastano a
ottenere, senza altri passaggi sui dati:
  * la scelta di lambda per GCV                      (`select_ridge`)
  * il leverage e la Mahalanobis per il rejection      (`domain_model`)
  * l'aggiunta/rimozione esatta di righe e di classi   (`accumulate(sign=-1)`, `expand_choices`)
e, con le feature in memoria, il residuo leave-one-out esatto (`HeadFit.loo_predict`).
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .types import JevTargets
from .uncertainty import DomainModel

logger = logging.getLogger(__name__)

DEFAULT_RIDGE_GRID: Tuple[float, ...] = tuple(10.0 ** e for e in range(-8, 4))  # 1e-8 ... 1e3, relativo a tr(A_c)/D


@contextlib.contextmanager
def _highest_matmul_precision() -> Iterator[None]:
    """Disabilita TF32 durante l'accumulo di X^T X: la Gram matrix deve restare in precisione piena."""
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


def _cholesky_jittered(a: torch.Tensor, lam: float, floor: float) -> Tuple[torch.Tensor, float]:
    """Cholesky di (a + lam I); se fallisce (a non PD con lam troppo piccolo) aumenta lam di 10x e avvisa."""
    eye = torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    lam_eff = lam
    for _ in range(10):
        chol, info = torch.linalg.cholesky_ex(a + lam_eff * eye)
        if int(info.item()) == 0:
            if lam_eff != lam:
                logger.warning("Cholesky: lambda aumentato da %.3e a %.3e (matrice non PD).", lam, lam_eff)
            return chol, lam_eff
        lam_eff = max(lam_eff * 10.0, floor)
    raise RuntimeError("Cholesky fallita anche con lambda molto grande: statistiche non finite?")


class _Stats:
    """Statistiche sufficienti di una testa: A = sum x x^T, B = sum x y^T, sx, sy, sum y^2, n."""

    def __init__(self, d: int, out: int, device: torch.device, dtype: torch.dtype) -> None:
        kw = dict(device=device, dtype=dtype)
        self.A = torch.zeros(d, d, **kw)
        self.B = torch.zeros(d, out, **kw)
        self.sx = torch.zeros(d, **kw)
        self.sy = torch.zeros(out, **kw)
        self.yty = torch.zeros((), **kw)
        self.n = torch.zeros((), device=device, dtype=torch.int64)

    def tensors(self) -> Tuple[torch.Tensor, ...]:
        return self.A, self.B, self.sx, self.sy, self.yty, self.n

    def add(self, x: torch.Tensor, y: torch.Tensor, sign: int = 1) -> None:
        self.A.addmm_(x.t(), x, alpha=sign)
        self.B.addmm_(x.t(), y, alpha=sign)
        self.sx += sign * x.sum(0)
        self.sy += sign * y.sum(0)
        self.yty += sign * (y * y).sum()
        self.n += sign * x.shape[0]
        if int(self.n) < 0:
            raise ValueError("Rimosse piu' righe di quante ne fossero state accumulate.")

    def centered(self, fit_intercept: bool):
        """(n, mu, ybar, A_c, B_c, ytyc): statistiche centrate. Con intercetta non penalizzata la ridge
        sulle statistiche centrate da' le stesse pendenze e rende la penalita' uno scalare (utile per eigh/GCV)."""
        n = float(self.n)
        if fit_intercept and n > 0:
            mu, ybar = self.sx / n, self.sy / n
            a = self.A - n * torch.outer(mu, mu)
            b = self.B - n * torch.outer(mu, ybar)
            yty = self.yty - n * (ybar * ybar).sum()
        else:
            mu, ybar = torch.zeros_like(self.sx), torch.zeros_like(self.sy)
            a, b, yty = self.A, self.B, self.yty
        return n, mu, ybar, (a + a.t()) / 2, b, yty


@dataclass
class HeadFit:
    """Soluzione ridge di una testa, con il fattore di Cholesky per leverage e LOO (tutto in fp64)."""

    head: str
    W: torch.Tensor  # (D, out) pendenze
    b: torch.Tensor  # (out,) intercetta (zeri se fit_intercept=False)
    mu: torch.Tensor  # (D,) media delle feature usata per centrare
    chol: torch.Tensor  # (D, D) Cholesky inferiore di (A_c + lam I)
    lam: float
    rho: float
    n: int
    fit_intercept: bool

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.W) @ self.W + self.b

    def leverage(self, x: torch.Tensor) -> torch.Tensor:
        """h_i = 1/n + (x_i - mu)^T (A_c + lam I)^-1 (x_i - mu): diagonale della hat matrix ridge, per qualunque x."""
        xc = x.to(self.W) - self.mu
        z = torch.linalg.solve_triangular(self.chol, xc.t(), upper=False)
        h = (z * z).sum(0)
        return h + (1.0 / max(self.n, 1) if self.fit_intercept else 0.0)

    def loo_predict(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predizione leave-one-out ESATTA per righe usate nel fit: y - (y - y_hat) / (1 - h).
        Vale per lambda fisso (quello di questo fit); y ha shape (n, out). Restituisce (pred_loo, h).
        """
        y = y.to(self.W)
        h = self.leverage(x)
        resid = y - self.predict(x)
        return y - resid / (1.0 - h).unsqueeze(1), h


class ExactHeadSolver:
    """
    Least-squares ridge in forma chiusa, in streaming, per le tre teste.

    Per ogni testa si accumulano A = sum x x^T, B = sum x y^T (+ somme e conteggi) batch dopo batch
    (memoria O(D^2), indipendente da N); poi, sulle statistiche centrate (intercetta non penalizzata):

        W = (A_c + lam I)^-1 B_c ,   b = ybar - mu^T W ,   lam = rho * tr(A_c) / D   (invariante di scala)

    `ridge` puo' essere un numero (rho) oppure "gcv": in quel caso rho e' scelto per testa minimizzando
    il Generalized Cross-Validation su una griglia, usando solo le statistiche accumulate (zero passaggi extra).
    Il sistema e' risolto via Cholesky in fp64 (con jitter se A_c non e' PD).

    Target (spazio dei LOGIT, perche' le teste sono lineari e la sigmoide e' applicata dopo):
      choice : one-hot * choice_scale (argmax invariante alla scala)
      score  : logit(clamp(s, eps, 1-eps))
      noul   : +/- logit_margin
    Nota: e' una regressione LS su questi target, NON il minimizzatore di CE/BCE: ottima per
    ranking/argmax, non calibrata. Per probabilita' usare `Calibration` (vedi pipeline.fit_jev_heads).
    """

    HEADS = ("choice", "score", "noul")

    def __init__(
        self,
        feature_dim: int,
        num_choices: int,
        ridge: Union[float, str, Dict[str, float]] = 1e-4,
        score_eps: float = 0.02,
        logit_margin: float = 3.0,
        choice_scale: float = 1.0,
        ignore_index: int = -100,
        fit_intercept: bool = True,
        accum_dtype: torch.dtype = torch.float64,
        ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.D, self.num_choices = feature_dim, num_choices
        self.ridge, self.score_eps = ridge, score_eps
        self.logit_margin, self.choice_scale = logit_margin, choice_scale
        self.ignore_index = ignore_index
        self.fit_intercept = fit_intercept
        self.accum_dtype = accum_dtype
        self.ridge_grid = tuple(ridge_grid)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self._out_dim = {"choice": num_choices, "score": 1, "noul": 1}
        self.reset()

    # ------------------------------------------------------------------ stato
    def reset(self) -> None:
        mk = lambda out: _Stats(self.D, out, self.device, self.accum_dtype)  # noqa: E731
        self.stats: Dict[str, _Stats] = {h: mk(self._out_dim[h]) for h in self.HEADS}
        # righe in-domain (noul == 0), senza target: base del leverage per il rejection
        self.domain_stats = mk(0)
        self.fits: Dict[str, HeadFit] = {}

    def reset_head(self, head: str) -> None:
        self.stats[head] = _Stats(self.D, self._out_dim[head], self.device, self.accum_dtype)
        self.fits.pop(head, None)

    @property
    def count(self) -> Dict[str, torch.Tensor]:
        return {h: self.stats[h].n for h in self.HEADS}

    # ------------------------------------------------------------------ accumulo
    def _score_target(self, s: torch.Tensor) -> torch.Tensor:
        return torch.logit(s.to(self.accum_dtype).clamp(self.score_eps, 1 - self.score_eps)).unsqueeze(1)

    @torch.no_grad()
    def accumulate(self, feats: torch.Tensor, tgt: JevTargets, sign: int = 1) -> None:
        """
        Aggiorna le statistiche con un batch di feature X (B, D) e i relativi target.
        `sign=-1` RIMUOVE le stesse righe (sottrazione esatta delle statistiche): vedi `remove`.
        """
        x = feats.detach().to(device=self.device, dtype=self.accum_dtype)  # bf16/fp32 -> fp64 e' esatto
        with _highest_matmul_precision():
            # choice: solo righe con label valida (es. escluse quelle out-of-domain)
            choice = tgt.choice.to(self.device)
            vc = choice != self.ignore_index
            if vc.any():
                y = F.one_hot(choice[vc], self.num_choices).to(self.accum_dtype) * self.choice_scale
                self.stats["choice"].add(x[vc], y, sign)

            # score: regressione sul logit del target (clamp per evitare +/-inf)
            score = tgt.score.to(self.device)
            vs = tgt.score_mask.to(self.device) if tgt.score_mask is not None else torch.ones_like(score, dtype=torch.bool)
            if vs.any():
                self.stats["score"].add(x[vs], self._score_target(score[vs]), sign)

            # noul: target binario mappato su +/- margin
            noul = tgt.noul.to(self.device).float() > 0.5
            y = torch.where(noul, self.logit_margin, -self.logit_margin).to(self.accum_dtype).unsqueeze(1)
            self.stats["noul"].add(x, y, sign)

            # dominio: righe non-OOD
            if (~noul).any():
                self.domain_stats.add(x[~noul], x.new_zeros(int((~noul).sum()), 0), sign)

    def remove(self, feats: torch.Tensor, tgt: JevTargets) -> None:
        """Rimozione esatta di righe gia' accumulate (machine unlearning in forma chiusa)."""
        self.accumulate(feats, tgt, sign=-1)

    @torch.no_grad()
    def accumulate_score(self, feats: torch.Tensor, s: torch.Tensor) -> None:
        """Accumula nella sola testa score target s in [0, 1] forniti a parte (es. correttezza LOO della choice)."""
        x = feats.detach().to(device=self.device, dtype=self.accum_dtype)
        self.stats["score"].add(x, self._score_target(s.to(self.device)), 1)

    def expand_choices(self, extra: int) -> None:
        """
        Aggiunge `extra` classi alla testa choice SENZA rivedere i dati vecchi: le righe gia' accumulate hanno
        zero nelle nuove colonne del target, quindi B (e la soluzione) restano esatti. Poi si accumulano
        solo le righe delle nuove classi con le label num_choices_old ... num_choices_old+extra-1.
        """
        st = self.stats["choice"]
        st.B = torch.cat([st.B, st.B.new_zeros(self.D, extra)], 1)
        st.sy = torch.cat([st.sy, st.sy.new_zeros(extra)])
        self.num_choices += extra
        self._out_dim["choice"] += extra
        self.fits.pop("choice", None)

    def all_reduce(self) -> None:
        """Multi-GPU: somma le statistiche tra i rank. Le statistiche sono additive -> risultato esatto."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        for st in (*self.stats.values(), self.domain_stats):
            for t in st.tensors():
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

    # ------------------------------------------------------------------ scelta di lambda (GCV)
    def select_ridge(self, head: str, grid: Optional[Sequence[float]] = None) -> Tuple[float, List[Tuple[float, float]]]:
        """
        rho ottimo per GCV, calcolato dalle sole statistiche: una `eigh` di A_c, poi per ogni rho
        RSS(rho) = y'y - 2 tr(W'B) + tr(W'AW) e gradi di liberta' tr(H) = sum s_j/(s_j + lam).
        Restituisce (rho*, [(rho, gcv), ...]).
        """
        st = self.stats[head]
        n, _, _, a, b, yty = st.centered(self.fit_intercept)
        if n < 2:
            raise RuntimeError(f"Testa '{head}': campioni insufficienti per il GCV.")
        evals, q = torch.linalg.eigh(a)
        evals = evals.clamp_min(0)
        bq = q.t() @ b
        scale = evals.sum() / self.D
        table: List[Tuple[float, float]] = []
        for rho in grid or self.ridge_grid:
            lam = rho * scale
            coef = 1.0 / (evals + lam)
            wq = coef.unsqueeze(1) * bq
            rss = yty - 2 * (wq * bq).sum() + (evals.unsqueeze(1) * wq * wq).sum()
            dof = (evals * coef).sum() + (1.0 if self.fit_intercept else 0.0)
            denom = n - dof
            gcv = float(n * rss.clamp_min(0) / denom**2) if denom > 1e-9 else float("inf")
            table.append((float(rho), gcv))
        best = min(table, key=lambda t: t[1])[0]
        if best == min(r for r, _ in table):
            logger.warning("GCV testa '%s': rho=%g sul bordo inferiore della griglia, ampliarla.", head, best)
        elif best == max(r for r, _ in table):
            logger.info("GCV testa '%s': rho=%g sul bordo superiore (segnale debole: shrinkage massimo).", head, best)
        return best, table

    def select_ridge_holdout(
        self,
        head: str,
        x: torch.Tensor,
        y: torch.Tensor,
        grid: Optional[Sequence[float]] = None,
        criterion: str = "mse",
    ) -> Tuple[float, List[Tuple[float, float]]]:
        """
        rho ottimo su un set HELD-OUT (x: (m, D), y: (m, out) nello spazio dei target della testa).
        criterion="mse": errore quadratico medio (cio' che la ridge ottimizza); "acc": errore di classificazione
        argmax (per la testa choice con y one-hot) - a parita' di errore vince il rho piu' grande.
        Stessa `eigh` del GCV: costo O(m D^2) in piu'. Da preferire a GCV/LOO quando il backbone e' stato
        addestrato sulle righe del fit (vedi docs/RESULTS.md: in quel caso i residui di training sono ottimisti).
        La tabella restituita contiene (rho, errore).
        """
        if criterion not in ("mse", "acc"):
            raise ValueError("criterion deve essere 'mse' o 'acc'")
        n, mu, ybar, a, b, _ = self.stats[head].centered(self.fit_intercept)
        evals, q = torch.linalg.eigh(a)
        evals = evals.clamp_min(0)
        bq = q.t() @ b
        scale = evals.sum() / self.D
        xq = (x.to(a) - mu) @ q
        table: List[Tuple[float, float]] = []
        for rho in grid or self.ridge_grid:
            coef = 1.0 / (evals + rho * scale)
            pred = ybar + (xq * coef) @ bq
            if criterion == "mse":
                err = float(((pred - y.to(a)) ** 2).mean())
            else:
                err = float((pred.argmax(1) != y.argmax(1)).to(a).mean())
            table.append((float(rho), err))
        return min(table, key=lambda t: (t[1], -t[0]))[0], table

    # ------------------------------------------------------------------ soluzione
    def fit_head(self, head: str, ridge: Optional[Union[float, str]] = None) -> HeadFit:
        st = self.stats[head]
        n = int(st.n.item())
        if n == 0 and head == "score":
            # lo score puo' non essere addestrato (score_mask tutta False): testa nulla, in attesa di score_mode='loo_correct'
            logger.warning("Testa score senza righe valide: pesi nulli.")
            eye = torch.eye(self.D, dtype=self.accum_dtype, device=self.device)
            z = torch.zeros(self.D, self._out_dim[head], dtype=self.accum_dtype, device=self.device)
            return HeadFit(head, z, z.new_zeros(self._out_dim[head]), z.new_zeros(self.D), eye, 1.0, float("nan"), 0, self.fit_intercept)
        if n == 0:
            raise RuntimeError(f"Nessuna riga valida accumulata per la testa '{head}'.")
        if n < self.D:
            logger.warning("Testa %s: %d campioni < D=%d, il sistema e' sottodeterminato (conta solo la ridge).", head, n, self.D)
        ridge = self.ridge if ridge is None else ridge
        if isinstance(ridge, dict):
            ridge = ridge[head]
        rho = self.select_ridge(head)[0] if ridge == "gcv" else float(ridge)
        _, mu, ybar, a, b, _ = st.centered(self.fit_intercept)
        scale = float(a.diagonal().mean().clamp_min(1e-12))
        chol, lam = _cholesky_jittered(a, rho * scale, floor=1e-10 * scale)
        w = torch.cholesky_solve(b, chol)
        bias = ybar - mu @ w
        return HeadFit(head, w, bias, mu, chol, lam, rho, n, self.fit_intercept)

    def fit(
        self, ridge: Optional[Union[float, str, Dict[str, float]]] = None, heads: Optional[Sequence[str]] = None
    ) -> Dict[str, HeadFit]:
        """`ridge` puo' essere un numero, "gcv", oppure un dict {testa: rho} (es. scelto su un held-out)."""
        for h in heads or self.HEADS:
            self.fits[h] = self.fit_head(h, ridge)
        return self.fits

    @torch.no_grad()
    def solve(self) -> Dict[str, torch.Tensor]:
        """
        Restituisce {choice_w, score_w, noul_w, choice_b, score_b, noul_b}: pesi (D, out) e intercette (out,).
        Passa dritto a `MambaJevModel.inject_exact_weights(**...)`. Le intercette sono zero se fit_intercept=False.
        """
        fits = self.fit()
        out: Dict[str, torch.Tensor] = {}
        for h, f in fits.items():
            out[f"{h}_w"] = f.W.float()
            out[f"{h}_b"] = f.b.float()
        return out

    # ------------------------------------------------------------------ rejection
    @torch.no_grad()
    def domain_model(self, ridge: float = 1e-3) -> DomainModel:
        """
        DomainModel (leverage + Mahalanobis a covarianza condivisa) dalle statistiche gia' accumulate.
        Le medie di classe sono le colonne di X^T Y_onehot / n_c, e lo scatter within-class e'
        A_choice - sum_c S_c S_c^T / n_c: nessun dato in piu' rispetto alla ridge.
        """
        dom = self.domain_stats
        n_in = int(dom.n.item())
        if n_in < 2:
            raise RuntimeError("Servono almeno 2 righe in-domain (noul == 0) per il DomainModel.")
        _, mu, _, a, _, _ = dom.centered(True)
        scale = float(a.diagonal().mean().clamp_min(1e-12))
        chol_in, _ = _cholesky_jittered(a, ridge * scale, floor=1e-10 * scale)

        means = chol_w = None
        ch = self.stats["choice"]
        if int(ch.n) > 0:
            nc = ch.sy / self.choice_scale
            sc = ch.B / self.choice_scale  # (D, C): somme di x per classe
            ok = nc > 0.5
            if int(ok.sum()) >= 1:
                sc_v, nc_v = sc[:, ok], nc[ok]
                means = (sc_v / nc_v).t()  # (C_v, D)
                within = (ch.A - (sc_v / nc_v) @ sc_v.t()) / float(ch.n)
                within = (within + within.t()) / 2
                s_w = float(within.diagonal().mean().clamp_min(1e-12))
                chol_w, _ = _cholesky_jittered(within, ridge * s_w, floor=1e-10 * s_w)
        return DomainModel(mu, chol_in, n_in, means, chol_w)
