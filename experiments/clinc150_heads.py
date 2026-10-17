"""
Esperimento sulle TESTE (backbone-agnostico) su CLINC150: 150 intent in-scope + richieste out-of-scope.

Domande (vedi docs/RESULTS.md):
  Q1  la ridge LS e' vicina a una testa CE (regressione logistica) in accuratezza?               (choice)
  Q2  la temperatura stimata sul LOO, SENZA validation, calibra quanto una stimata sulla val?    (calibrazione)
  Q3  lo score "correttezza LOO" e la MSP calibrata ordinano gli errori meglio di MSP grezza?     (selective cl.)
  Q4  leverage / Mahalanobis (stesse statistiche, nessun dato OOD) vs MSP/energy e vs testa noul? (rejection)
  Q5  GCV sceglie rho come la validation?                                                          (ablation)
  Q6  quanto costa il fit rispetto a LR (L-BFGS) e a un probe SGD?                                 (H2)
  Q7  aggiunta di classi e rimozione di righe sono esatte?                                          (incrementale)

Il backbone e' un ENCODER CONGELATO qualunque (--encoder lsa|spacy|npz): le conclusioni valgono per le teste, non
dicono nulla sulla qualita' di Mamba. Per usare feature Mamba: scripts/extract_features.py -> --encoder npz.

Uso:  python experiments/clinc150_heads.py --encoder lsa --out results/
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mamba_jev import ExactHeadSolver, JevTargets, fit_jev_heads, fit_temperature
from mamba_jev.metrics import aurc, auroc, bootstrap_ci, brier_multiclass, ece, fpr_at_tpr, nll_multiclass

DATA_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
DT = torch.float64
logging.getLogger("mamba_jev").setLevel(logging.ERROR)  # niente rumore dai warning del solver durante gli sweep


# ===================================================================== dati / encoder
def load_clinc(path: Path) -> Dict[str, List]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)  # noqa: S310 - URL fisso
    return json.loads(path.read_text())


def encode(encoder: str, splits: Dict[str, List[str]], dim: int, npz: str | None) -> Dict[str, np.ndarray]:
    if encoder == "npz":  # feature precalcolate (es. Mamba): npz con una chiave per split
        z = np.load(npz)
        return {k: z[k] for k in splits}
    if encoder == "lsa":
        from scipy.sparse import hstack
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        w = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3)
        fit_txt = splits["train"]  # solo testo in-scope di training
        w.fit(fit_txt), c.fit(fit_txt)
        feat = lambda t: hstack([w.transform(t), c.transform(t)]).tocsr()  # noqa: E731
        svd = TruncatedSVD(dim, random_state=0).fit(feat(fit_txt))
        return {k: svd.transform(feat(v)) for k, v in splits.items()}
    if encoder == "spacy":
        import spacy

        nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "lemmatizer", "tagger", "attribute_ruler"])
        return {k: np.stack([d.vector for d in nlp.pipe(v, batch_size=512)]) for k, v in splits.items()}
    raise ValueError(encoder)


# ===================================================================== baseline CE
def fit_lr(x: torch.Tensor, y: torch.Tensor, c: int, reg: float, max_iter: int = 300) -> Tuple[torch.Tensor, torch.Tensor]:
    """Regressione logistica multiclasse L2 (intercetta libera), L-BFGS a batch intero: B4 del piano."""
    w = torch.zeros(x.shape[1], c, dtype=x.dtype, requires_grad=True)
    b = torch.zeros(c, dtype=x.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-7)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ w + b, y) + 0.5 * reg * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def fit_lr_binary(x: torch.Tensor, y: torch.Tensor, reg: float, max_iter: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
    w = torch.zeros(x.shape[1], dtype=x.dtype, requires_grad=True)
    b = torch.zeros((), dtype=x.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ w + b, y) + 0.5 * reg * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def fit_sgd_probe(x, y, c, epochs=30, lr=3e-3, wd=1e-4, bs=256, seed=0):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(x.shape[1], c).to(x.dtype)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    n = len(x)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            F.cross_entropy(lin(x[idx]), y[idx]).backward()
            opt.step()
    return lin.weight.detach().t().contiguous(), lin.bias.detach()


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


# ===================================================================== helpers metriche
def conf_metrics(probs: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    p = probs.double()
    conf, pred = p.max(1)
    correct = (pred == y).double()
    return dict(
        acc=float(correct.mean()),
        ece=ece(conf, correct),
        nll=nll_multiclass(p, y),
        brier=brier_multiclass(p, y),
    )


def ci(fn, *arrs, n_boot=300):
    lo, hi = bootstrap_ci(fn, *arrs, n_boot=n_boot)
    return [round(lo, 4), round(hi, 4)]


# ===================================================================== analisi
def analyze(feats_np: Dict[str, np.ndarray], raw: Dict[str, List], name: str, b1: Dict | None = None) -> Dict:
    """
    Analisi completa delle teste su feature congelate. `feats_np`: array (N, D) per split
    (train, val, test, oos_train, oos_val, oos_test). `b1` (opzionale): teste CE addestrate end-to-end insieme
    al backbone, {"choice": (W (D,C), b (C,)), "noul": (w (D,), b ())}, valutate sulle feature NON standardizzate.
    """
    intents = sorted({lab for _, lab in raw["train"]})
    cid = {k: i for i, k in enumerate(intents)}
    C = len(intents)

    # z-score fittato sul TRAIN in-scope (stessa preprocessing per tutti i metodi)
    mu, sd = feats_np["train"].mean(0), feats_np["train"].std(0) + 1e-8
    X = {k: torch.tensor((v - mu) / sd, dtype=DT) for k, v in feats_np.items()}
    XR = {k: torch.tensor(v, dtype=DT) for k, v in feats_np.items()}  # non standardizzate (per B1)
    Y = {k: torch.tensor([cid[l] for _, l in raw[k]]) for k in ("train", "val", "test")}
    D = X["train"].shape[1]
    print(f"[{name}] D={D}  N_train={len(X['train'])}  C={C}", flush=True)

    res: Dict = {"encoder": name, "D": D, "C": C}

    # ---- training set per le teste LS: in-scope (label = intent, noul=0) + oos_train (choice ignorata, noul=1)
    n_oos = len(X["oos_train"])
    xtr = torch.cat([X["train"], X["oos_train"]])
    ytr = torch.cat([Y["train"], torch.full((n_oos,), -100)])
    noul_tr = torch.cat([torch.zeros(len(Y["train"])), torch.ones(n_oos)])
    tgt_tr = JevTargets(ytr, torch.zeros(len(xtr)), noul_tr)

    # ============================================================ Q1/Q2: choice + calibrazione
    ls, t_ls = timed(lambda: fit_jev_heads(xtr, tgt_tr, C, ridge="gcv", score_mode="loo_correct"))
    res["fit_time_closed_form_s"] = round(t_ls, 3)
    res["rho_gcv"] = ls.diagnostics["rho"]
    res["diag"] = {k: v for k, v in ls.diagnostics.items() if k != "rho"}
    W, b = ls.weights["choice_w"].double(), ls.weights["choice_b"].double()
    ls_logits = lambda x: x @ W + b  # noqa: E731
    T_loo = ls.diagnostics["choice_temperature"]

    # temperatura sulla validation (riferimento "con dati extra")
    T_val = fit_temperature(ls_logits(X["val"]), Y["val"])
    zt = ls_logits(X["test"])
    ytest = Y["test"]
    rows = {}
    rows["LS raw softmax"] = conf_metrics(zt.softmax(1), ytest)
    rows["LS + T(LOO)  [ours, no val]"] = conf_metrics((zt / T_loo).softmax(1), ytest)
    rows["LS + T(val)"] = conf_metrics((zt / T_val).softmax(1), ytest)
    res["T_loo"], res["T_val"] = T_loo, T_val

    # protocollo B (con held-out): rho scelto sull'MSE della val, temperatura sulla val (stesso budget dati dei baseline LR)
    hx = torch.cat([X["val"], X["oos_val"]])
    h_tgt = JevTargets(
        torch.cat([Y["val"], torch.full((len(X["oos_val"]),), -100)]),
        torch.zeros(len(hx)),
        torch.cat([torch.zeros(len(X["val"])), torch.ones(len(X["oos_val"]))]),
    )
    lsh, t_lsh = timed(lambda: fit_jev_heads(xtr, tgt_tr, C, ridge="holdout", holdout=(hx, h_tgt), score_mode="msp"))
    Wh, bh = lsh.weights["choice_w"].double(), lsh.weights["choice_b"].double()
    zh = X["test"] @ Wh + bh
    T_h = lsh.diagnostics["choice_temperature"]
    rows["LS rho=holdout + T(holdout)"] = conf_metrics((zh / T_h).softmax(1), ytest)
    res["holdout_protocol"] = dict(rho=lsh.diagnostics["rho"], T=T_h, fit_time_s=round(t_lsh, 3))

    # baseline CE: LR con reg scelta sulla val (accuracy), poi raw e T(val)
    best = None
    t_lr_total = 0.0
    lr_fits = {}
    for reg in (1e-5, 1e-4, 1e-3, 1e-2):
        (w_lr, b_lr), dt = timed(lambda reg=reg: fit_lr(X["train"], Y["train"], C, reg))
        t_lr_total += dt
        acc = float(((X["val"] @ w_lr + b_lr).argmax(1) == Y["val"]).double().mean())
        lr_fits[reg] = (w_lr, b_lr, dt)
        if best is None or acc > best[0]:
            best = (acc, reg)
    reg_lr = best[1]
    w_lr, b_lr, t_lr_one = lr_fits[reg_lr]
    lr_logits = lambda x: x @ w_lr + b_lr  # noqa: E731
    T_lr = fit_temperature(lr_logits(X["val"]), Y["val"])
    rows["LR (CE) raw"] = conf_metrics(lr_logits(X["test"]).softmax(1), ytest)
    rows["LR (CE) + T(val)"] = conf_metrics((lr_logits(X["test"]) / T_lr).softmax(1), ytest)

    # B1: teste CE addestrate end-to-end col backbone (se fornite)
    if b1 is not None:
        wb, bb = torch.as_tensor(b1["choice"][0], dtype=DT), torch.as_tensor(b1["choice"][1], dtype=DT)
        b1_logits = lambda x: x @ wb + bb  # noqa: E731
        T_b1 = fit_temperature(b1_logits(XR["val"]), Y["val"])
        rows["B1 end-to-end CE raw"] = conf_metrics(b1_logits(XR["test"]).softmax(1), ytest)
        rows["B1 end-to-end CE + T(val)"] = conf_metrics((b1_logits(XR["test"]) / T_b1).softmax(1), ytest)

    # LDA a covarianza condivisa dalle STESSE statistiche della ridge (nessun dato in piu'): solo accuratezza
    ch = ls.solver.stats["choice"]
    nc = ch.sy
    means = (ch.B / nc).t()
    within = (ch.A - (ch.B / nc) @ ch.B.t()) / float(ch.n)
    within = (within + within.t()) / 2
    prec = torch.linalg.inv(within + 1e-2 * within.diagonal().mean() * torch.eye(D, dtype=DT))
    lda = X["test"] @ (prec @ means.t()) - 0.5 * (means @ prec * means).sum(1) + torch.log(nc / ch.n)
    res["lda_tied_acc"] = round(float((lda.argmax(1) == ytest).double().mean()), 4)

    # probe SGD (B2): 3 seed
    sgd_accs, sgd_times = [], []
    for s in range(3):
        (w_s, b_s), dt = timed(lambda s=s: fit_sgd_probe(X["train"], Y["train"], C, seed=s))
        sgd_accs.append(float(((X["test"] @ w_s + b_s).argmax(1) == ytest).double().mean()))
        sgd_times.append(dt)
    res["sgd_probe"] = dict(acc_mean=round(float(np.mean(sgd_accs)), 4), acc_std=round(float(np.std(sgd_accs)), 4), time_s=round(float(np.mean(sgd_times)), 2))

    for k, m in rows.items():
        m["acc_ci"] = None
    # CI bootstrap sugli indicatori per-esempio (acc e ECE) per i metodi principali
    for name, probs in (
        ("LS + T(LOO)  [ours, no val]", (zt / T_loo).softmax(1)),
        ("LS + T(val)", (zt / T_val).softmax(1)),
        ("LR (CE) + T(val)", (lr_logits(X["test"]) / T_lr).softmax(1)),
    ):
        conf, pred = probs.max(1)
        corr = (pred == ytest).double()
        rows[name]["ece_ci"] = ci(ece, conf, corr)
        rows[name]["acc_ci"] = ci(lambda c_: float(np.mean(c_)), corr)
    res["choice_calibration"] = rows
    res["fit_time_lr_one_s"] = round(t_lr_one, 2)
    res["fit_time_lr_grid4_s"] = round(t_lr_total, 2)
    res["reg_lr"] = reg_lr

    # ============================================================ Q3: selective classification (in-scope test)
    p_ls_T = (zt / T_loo).softmax(1)
    p_lr_T = (lr_logits(X["test"]) / T_lr).softmax(1)
    pred_ls, pred_lr = zt.argmax(1), lr_logits(X["test"]).argmax(1)
    ok_ls, ok_lr = (pred_ls == ytest).double(), (pred_lr == ytest).double()
    s_head = ls.weights  # score head: logit -> Platt (a, b) dal LOO
    sc_logit = X["test"] @ s_head["score_w"].double() + s_head["score_b"].double()
    a_s, b_s_ = ls.diagnostics["score_platt"]
    p_score = torch.sigmoid(a_s * sc_logit.squeeze(1) + b_s_)
    conf_sets = {
        "LS MSP raw": (zt.softmax(1).max(1).values, ok_ls),
        "LS MSP + T(LOO)": (p_ls_T.max(1).values, ok_ls),
        "LS score head (LOO-correct, Platt)": (p_score, ok_ls),
        "LS max-logit": (zt.max(1).values, ok_ls),
        "LS rho=holdout MSP + T(holdout)": ((zh / T_h).softmax(1).max(1).values, (zh.argmax(1) == ytest).double()),
        "LR MSP raw": (lr_logits(X["test"]).softmax(1).max(1).values, ok_lr),
        "LR MSP + T(val)": (p_lr_T.max(1).values, ok_lr),
    }
    sel = {}
    for k, (conf, ok) in conf_sets.items():
        cc = conf.clamp(0, 1) if "max-logit" not in k else conf
        sel[k] = dict(
            auroc_err=round(auroc(-cc, 1 - ok), 4),
            aurc=round(aurc(cc, ok), 5),
            ece=round(ece(cc, ok), 4) if "max-logit" not in k else None,
            aurc_ci=ci(aurc, cc, ok),
        )
    res["selective"] = sel

    # ============================================================ Q4: rejection (in-scope test vs oos_test)
    x_id, x_oos = X["test"], X["oos_test"]
    xe = torch.cat([x_id, x_oos])
    lab = torch.cat([torch.zeros(len(x_id)), torch.ones(len(x_oos))])

    def head_scores(x):
        d = {}
        z = ls_logits(x)
        zl = lr_logits(x)
        d["MSP  (LR, CE)"] = 1 - zl.softmax(1).max(1).values
        d["max-logit (LR)"] = -zl.max(1).values
        d["energy (LR)"] = -torch.logsumexp(zl, 1)
        d["MSP + T(LOO) (LS)"] = 1 - (z / T_loo).softmax(1).max(1).values
        d["max-logit (LS)"] = -z.max(1).values
        d["MSP + T(holdout) (LS, rho=holdout)"] = 1 - ((x @ Wh + bh) / T_h).softmax(1).max(1).values
        d["leverage (stats ridge)  [no OOD data]"] = ls.domain.leverage(x)
        d["Mahalanobis tied (stats ridge) [no OOD data]"] = ls.domain.maha_tied(x)
        nz = x @ ls.weights["noul_w"].double() + ls.weights["noul_b"].double()
        d["noul head LS (100 OOS train)"] = nz.squeeze(1)
        return d

    sc = head_scores(xe)
    if b1 is not None:
        xr = torch.cat([XR["test"], XR["oos_test"]])
        zb = xr @ torch.as_tensor(b1["choice"][0], dtype=DT) + torch.as_tensor(b1["choice"][1], dtype=DT)
        sc["MSP  (B1 end-to-end)"] = 1 - zb.softmax(1).max(1).values
        sc["energy (B1 end-to-end)"] = -torch.logsumexp(zb, 1)
        sc["noul head B1 end-to-end (100 OOS train)"] = xr @ torch.as_tensor(b1["noul"][0], dtype=DT) + torch.as_tensor(b1["noul"][1], dtype=DT)
    # baseline testa noul CE: LR binaria su in-scope + oos_train, reg scelta su val(+oos_val)
    xn = torch.cat([X["train"], X["oos_train"]])
    yn = torch.cat([torch.zeros(len(X["train"])), torch.ones(n_oos)]).to(DT)
    xv = torch.cat([X["val"], X["oos_val"]])
    yv = torch.cat([torch.zeros(len(X["val"])), torch.ones(len(X["oos_val"]))])
    best_n = None
    for reg in (1e-4, 1e-3, 1e-2, 1e-1):
        wn, bn = fit_lr_binary(xn, yn, reg)
        a = auroc(xv @ wn + bn, yv)
        if best_n is None or a > best_n[0]:
            best_n = (a, wn, bn)
    sc["noul head LR (CE, 100 OOS train)"] = xe @ best_n[1] + best_n[2]

    rej = {}
    for k, v in sc.items():
        rej[k] = dict(auroc=round(auroc(v, lab), 4), fpr95=round(fpr_at_tpr(v, lab), 4), auroc_ci=ci(auroc, v, lab))
    res["rejection"] = rej

    # sensibilita' del rejection: ridge del DomainModel e numero di esempi OOS visti dalla testa noul
    sens = {}
    for r in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        dm = ls.solver.domain_model(ridge=r)
        sens[f"domain_ridge={r:g}"] = dict(
            leverage=round(auroc(dm.leverage(xe), lab), 4), maha=round(auroc(dm.maha_tied(xe), lab), 4)
        )
    res["domain_ridge_sensitivity"] = sens
    sweep = {}
    g = torch.Generator().manual_seed(0)
    for m in (0, 5, 10, 25, 50, 100):
        aucs = []
        for rep in range(1 if m in (0, 100) else 5):
            sel_idx = torch.randperm(n_oos, generator=g)[:m]
            xs = torch.cat([X["train"], X["oos_train"][sel_idx]])
            ys = torch.cat([Y["train"], torch.full((m,), -100)])
            ns = torch.cat([torch.zeros(len(X["train"])), torch.ones(m)])
            if m == 0:
                aucs.append(float("nan"))
                continue
            fm = fit_jev_heads(xs, JevTargets(ys, torch.zeros(len(xs)), ns), C, ridge="gcv", loo_calibration=False, domain=False)
            z = xe @ fm.weights["noul_w"].double() + fm.weights["noul_b"].double()
            aucs.append(auroc(z.squeeze(1), lab))
        sweep[str(m)] = [round(float(np.nanmean(aucs)), 4) if m else None, round(float(np.nanstd(aucs)), 4) if m else None]
    res["noul_head_auroc_vs_n_oos_train"] = sweep

    # ============================================================ Q5: GCV vs validation vs rho fisso
    abl = {}
    for rho in (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        f = fit_jev_heads(xtr, tgt_tr, C, ridge=rho, loo_calibration=False, domain=False)
        wr, br = f.weights["choice_w"].double(), f.weights["choice_b"].double()
        abl[f"{rho:g}"] = dict(
            val_acc=round(float(((X["val"] @ wr + br).argmax(1) == Y["val"]).double().mean()), 4),
            test_acc=round(float(((X["test"] @ wr + br).argmax(1) == ytest).double().mean()), 4),
        )
    res["ridge_sweep"] = abl
    res["gcv_rho_choice"] = ls.diagnostics["rho"]["choice"]
    gcv_best = max(abl.items(), key=lambda kv: kv[1]["val_acc"])[0]
    res["best_rho_on_val"] = gcv_best

    # ============================================================ Q7: incremental / unlearning su dati reali
    keep = Y["train"] < C - 10
    sub = lambda m: JevTargets(Y["train"][m], torch.zeros(int(m.sum())), torch.zeros(int(m.sum())))  # noqa: E731
    inc = ExactHeadSolver(D, C - 10, ridge=1e-3)
    inc.accumulate(X["train"][keep], sub(keep))
    inc.expand_choices(10)
    inc.accumulate(X["train"][~keep], sub(~keep))
    full = ExactHeadSolver(D, C, ridge=1e-3)
    full.accumulate(X["train"], sub(torch.ones_like(keep)))
    res["incremental_max_abs_weight_diff"] = float((inc.solve()["choice_w"] - full.solve()["choice_w"]).abs().max())
    rm = torch.arange(len(keep)) % 7 == 0
    unl = ExactHeadSolver(D, C, ridge=1e-3)
    unl.accumulate(X["train"], sub(torch.ones_like(keep)))
    unl.remove(X["train"][rm], sub(rm))
    ref = ExactHeadSolver(D, C, ridge=1e-3)
    ref.accumulate(X["train"][~rm], sub(~rm))
    res["unlearning_max_abs_weight_diff"] = float((unl.solve()["choice_w"] - ref.solve()["choice_w"]).abs().max())

    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/clinc150/data_full.json")
    ap.add_argument("--encoder", default="lsa", choices=["lsa", "spacy", "npz"])
    ap.add_argument("--npz", default=None)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    torch.set_num_threads(2)

    raw = load_clinc(Path(args.data))
    texts = {k: [t for t, _ in v] for k, v in raw.items()}
    feats_np = encode(args.encoder, texts, args.dim, args.npz)
    res = analyze(feats_np, raw, args.encoder)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"clinc150_{args.encoder}.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
