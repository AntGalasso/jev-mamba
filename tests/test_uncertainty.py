"""Verifica numerica delle formule chiuse (LOO, leverage, GCV, Mahalanobis, unlearning) contro calcoli espliciti."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mamba_jev import ExactHeadSolver, JevTargets, MambaJevConfig, MambaJevModel, TrainConfig, fit_jev_heads, train
from mamba_jev import CombinedJevLoss, fit_platt, fit_temperature
from mamba_jev.metrics import aurc, auroc, ece, fpr_at_tpr

from conftest import StubBackbone, make_batches

DT = torch.float64


def _data(n=300, d=12, c=4, seed=0, shift=2.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=DT) + shift  # media lontana da zero: stress per la centratura
    y = torch.randint(0, c, (n,), generator=g)
    x[:, 0] += y.to(DT)  # un po' di segnale
    t = JevTargets(y, torch.rand(n, generator=g), (torch.rand(n, generator=g) > 0.7).float())
    return x, t


def _explicit_ridge(x, y, lam):
    """Ridge con intercetta NON penalizzata, via matrice aumentata (riferimento indipendente dal solver)."""
    n, d = x.shape
    aug = torch.cat([x, torch.ones(n, 1, dtype=x.dtype)], 1)
    pen = torch.diag(torch.tensor([lam] * d + [0.0], dtype=x.dtype))
    return torch.linalg.solve(aug.t() @ aug + pen, aug.t() @ y)


# ------------------------------------------------------------------ LOO / leverage
def test_loo_matches_explicit_refit():
    x, t = _data()
    solver = ExactHeadSolver(x.shape[1], 4, ridge=1e-2)
    solver.accumulate(x, t)
    fit = solver.fit_head("choice")
    y = F.one_hot(t.choice, 4).to(DT)
    pred_loo, h = fit.loo_predict(x, y)
    for i in (0, 7, 123, 299):
        keep = torch.arange(len(x)) != i
        w = _explicit_ridge(x[keep], y[keep], fit.lam)  # STESSO lambda assoluto
        ref = torch.cat([x[i], torch.ones(1, dtype=DT)]) @ w
        assert torch.allclose(pred_loo[i], ref, atol=1e-9), i


def test_leverage_equals_hat_matrix_diagonal():
    x, t = _data()
    solver = ExactHeadSolver(x.shape[1], 4, ridge=1e-2)
    solver.accumulate(x, t)
    fit = solver.fit_head("choice")
    aug = torch.cat([x, torch.ones(len(x), 1, dtype=DT)], 1)
    pen = torch.diag(torch.tensor([fit.lam] * x.shape[1] + [0.0], dtype=DT))
    hat = aug @ torch.linalg.solve(aug.t() @ aug + pen, aug.t())
    assert torch.allclose(fit.leverage(x), hat.diagonal(), atol=1e-10)


def test_leverage_grows_for_far_points():
    x, t = _data()
    solver = ExactHeadSolver(x.shape[1], 4)
    solver.accumulate(x, t)
    fit = solver.fit_head("choice")
    far = x[:5] + 50.0
    assert fit.leverage(far).min() > 10 * fit.leverage(x).max()


# ------------------------------------------------------------------ GCV
def test_gcv_matches_explicit_hat_matrix():
    x, t = _data(n=200)
    solver = ExactHeadSolver(x.shape[1], 4)
    solver.accumulate(x, t)
    grid = (1e-4, 1e-2, 1.0)
    best, table = solver.select_ridge("choice", grid)
    y = F.one_hot(t.choice, 4).to(DT)
    _, _, _, a, _, _ = solver.stats["choice"].centered(True)
    scale = float(a.diagonal().mean())
    for rho, gcv in table:
        lam = rho * float(a.diagonal().sum() / x.shape[1])
        aug = torch.cat([x, torch.ones(len(x), 1, dtype=DT)], 1)
        pen = torch.diag(torch.tensor([lam] * x.shape[1] + [0.0], dtype=DT))
        hat = aug @ torch.linalg.solve(aug.t() @ aug + pen, aug.t())
        rss = ((y - hat @ y) ** 2).sum()
        ref = len(x) * rss / (len(x) - hat.trace()) ** 2
        assert abs(gcv - float(ref)) < 1e-6 * abs(float(ref))
    assert best == min(table, key=lambda r: r[1])[0]
    assert scale > 0


def test_gcv_ridge_option_runs_and_is_finite():
    x, t = _data()
    solver = ExactHeadSolver(x.shape[1], 4, ridge="gcv")
    solver.accumulate(x, t)
    for w in solver.solve().values():
        assert torch.isfinite(w).all()


# ------------------------------------------------------------------ unlearning / classi incrementali
def test_remove_rows_equals_never_adding():
    x, t = _data()
    keep = torch.arange(len(x)) < 220
    sub = lambda m: JevTargets(t.choice[m], t.score[m], t.noul[m])  # noqa: E731
    a = ExactHeadSolver(x.shape[1], 4)
    a.accumulate(x, t)
    a.remove(x[~keep], sub(~keep))
    b = ExactHeadSolver(x.shape[1], 4)
    b.accumulate(x[keep], sub(keep))
    wa, wb = a.solve(), b.solve()
    for k in wa:
        assert torch.allclose(wa[k], wb[k], atol=1e-5), k


def test_remove_more_than_added_raises():
    x, t = _data(n=20)
    s = ExactHeadSolver(x.shape[1], 4)
    s.accumulate(x[:5], JevTargets(t.choice[:5], t.score[:5], t.noul[:5]))
    with pytest.raises(ValueError):
        s.remove(x, t)


def test_expand_choices_is_exact():
    x, t = _data(n=400, c=4)
    old = t.choice < 3
    sub = lambda m: JevTargets(t.choice[m], t.score[m], t.noul[m])  # noqa: E731
    inc = ExactHeadSolver(x.shape[1], 3)
    inc.accumulate(x[old], sub(old))
    inc.expand_choices(1)
    inc.accumulate(x[~old], sub(~old))  # solo righe della nuova classe
    full = ExactHeadSolver(x.shape[1], 4)
    full.accumulate(x, t)
    assert torch.allclose(inc.solve()["choice_w"], full.solve()["choice_w"], atol=1e-5)
    assert torch.allclose(inc.solve()["choice_b"], full.solve()["choice_b"], atol=1e-5)


# ------------------------------------------------------------------ DomainModel
def test_domain_model_matches_explicit_gaussian():
    x, t = _data(n=500)
    ridge = 1e-3
    solver = ExactHeadSolver(x.shape[1], 4)
    solver.accumulate(x, t)
    dm = solver.domain_model(ridge=ridge)

    ind = t.noul < 0.5
    xi = x[ind]
    mu = xi.mean(0)
    xc = xi - mu
    a = xc.t() @ xc
    lam = ridge * float(a.diagonal().mean())
    q = x[:6]
    ref_lev = 1.0 / len(xi) + torch.einsum("nd,de,ne->n", q - mu, torch.linalg.inv(a + lam * torch.eye(x.shape[1], dtype=DT)), q - mu)
    assert torch.allclose(dm.leverage(q), ref_lev, atol=1e-9)

    # Mahalanobis a covarianza condivisa (tutte le righe con label choice valida)
    means = torch.stack([x[t.choice == c].mean(0) for c in range(4)])
    within = sum(((x[t.choice == c] - means[c]).t() @ (x[t.choice == c] - means[c])) for c in range(4)) / len(x)
    lam_w = ridge * float(within.diagonal().mean())
    prec = torch.linalg.inv(within + lam_w * torch.eye(x.shape[1], dtype=DT))
    d2 = torch.stack([torch.einsum("nd,de,ne->n", q - means[c], prec, q - means[c]) for c in range(4)], 1).min(1).values
    assert torch.allclose(dm.maha_tied(q), d2, atol=1e-8)


# ------------------------------------------------------------------ calibrazione
def test_fit_temperature_recovers_known_temperature():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(20000, 6, generator=g) * 2.0
    t_true = 0.37
    y = torch.distributions.Categorical(logits=z / t_true).sample()  # etichette generate da softmax(z / T)
    assert abs(fit_temperature(z, y) - t_true) < 0.03


def test_fit_platt_recovers_known_parameters():
    g = torch.Generator().manual_seed(0)
    s = torch.randn(40000, generator=g) * 2.0
    a_true, b_true = 1.7, -0.4
    y = (torch.rand(40000, generator=g) < torch.sigmoid(a_true * s + b_true)).float()
    a, b = fit_platt(s, y)
    assert abs(a - a_true) < 0.06 and abs(b - b_true) < 0.06


# ------------------------------------------------------------------ metriche
def test_auroc_ties_and_known_values():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    s = np.round(rng.normal(size=500), 1)  # molti pareggi
    y = rng.integers(0, 2, 500)
    assert abs(auroc(s, y) - sk.roc_auc_score(y, s)) < 1e-12


def test_ece_perfect_and_miscalibrated():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 200000)
    correct = (rng.uniform(0, 1, conf.shape) < conf).astype(float)
    assert ece(conf, correct) < 0.01
    assert ece(np.full(1000, 0.9), np.zeros(1000)) == pytest.approx(0.9)


def test_fpr_at_tpr_and_aurc_basic():
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert fpr_at_tpr(s, y, 0.95) == 0.0
    assert aurc([0.9, 0.8, 0.1], [1, 1, 0]) < aurc([0.1, 0.8, 0.9], [1, 1, 0])


# ------------------------------------------------------------------ pipeline end-to-end
def _gauss_clusters(n_per=200, c=5, d=16, seed=0):
    centers = torch.randn(c, d, generator=torch.Generator().manual_seed(123)) * 1.2  # centri condivisi tra train e test
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(c).repeat_interleave(n_per)
    x = centers[y] + torch.randn(len(y), d, generator=g)
    return x, y


def test_fit_jev_heads_calibrates_and_rejects():
    c, d = 5, 16
    x, y = _gauss_clusters(c=c, d=d)
    t = JevTargets(y, torch.zeros(len(y)), torch.zeros(len(y)))
    fit = fit_jev_heads(x, t, c, ridge="gcv", score_mode="loo_correct")

    # temperatura < 1 (i logit LS vivono in [0,1]); LOO meno ottimista dell'accuracy in-sample
    assert fit.diagnostics["choice_temperature"] < 1.0
    assert fit.diagnostics["choice_loo_acc"] <= fit.diagnostics["choice_train_acc"] + 1e-9

    # probabilita' calibrate su dati nuovi dalla stessa distribuzione
    xt, yt = _gauss_clusters(seed=1)
    cfg = MambaJevConfig(d_model=d, n_layer=1, vocab_size=8, num_choices=c, param_dtype="float32")
    model = MambaJevModel(cfg, backbone=StubBackbone(8, d))
    model.load_fit(fit)
    out = model.predict_from_features(xt)
    conf, correct = out["choice_probs"].max(-1).values, (out["choice"] == yt).float()
    raw_conf = (model.apply_heads(xt).choice_logits).softmax(-1).max(-1).values
    assert ece(conf, correct) < 0.5 * ece(raw_conf, correct)

    # rejection: punti lontani dal dominio hanno leverage e Mahalanobis molto maggiori
    far = torch.randn(200, d) * 4.0
    for key in ("ood_leverage", "ood_maha"):
        s_in = model.predict_from_features(xt)[key]
        s_far = model.predict_from_features(far)[key]
        assert auroc(torch.cat([s_in, s_far]), torch.cat([torch.zeros(len(s_in)), torch.ones(len(s_far))])) > 0.95


def test_train_exact_branch_with_loo_calibration(cfg):
    torch.manual_seed(0)
    model = MambaJevModel(cfg, backbone=StubBackbone(cfg.vocab_size, cfg.d_model))
    batches = make_batches(n_batches=10)
    train(model, CombinedJevLoss(), batches, TrainConfig(use_exact_solver=True, solver_ridge="gcv", loo_calibration=True), device="cpu")
    assert model.domain is not None
    assert float(model.calibration.choice_temperature) != 1.0
    out = model.predict(batches[0]["input_ids"], batches[0]["attention_mask"])
    assert {"ood_leverage", "ood_maha", "choice_probs", "score", "noul_prob"} <= set(out)


def test_holdout_ridge_and_calibration():
    c, d = 5, 16
    x, y = _gauss_clusters(c=c, d=d, seed=0)
    xh, yh = _gauss_clusters(c=c, d=d, seed=7)
    z = torch.zeros(len(y))
    fit = fit_jev_heads(
        x, JevTargets(y, z, z), c, ridge="holdout", holdout=(xh, JevTargets(yh, torch.zeros(len(yh)), torch.zeros(len(yh)))), score_mode="msp"
    )
    assert fit.diagnostics["choice_temperature"] < 1.0 and "choice_holdout_acc" in fit.diagnostics
    assert bool(fit.calibration.score_from_msp)
    with pytest.raises(ValueError):
        fit_jev_heads(x, JevTargets(y, z, z), c, ridge="holdout")  # manca l'holdout
    with pytest.raises(ValueError):
        fit_jev_heads(x, JevTargets(y, z, z), c, holdout=(xh, JevTargets(yh, z[: len(yh)], z[: len(yh)])), score_mode="loo_correct")


def test_select_ridge_holdout_matches_explicit_mse():
    x, t = _data(n=250)
    xh, th = _data(n=80, seed=5)
    solver = ExactHeadSolver(x.shape[1], 4)
    solver.accumulate(x, t)
    yh = F.one_hot(th.choice, 4).to(DT)
    best, table = solver.select_ridge_holdout("choice", xh, yh, grid=(1e-3, 1e-1, 10.0))
    y = F.one_hot(t.choice, 4).to(DT)
    a = solver.stats["choice"].centered(True)[3]
    for rho, mse in table:
        lam = rho * float(a.diagonal().sum() / x.shape[1])
        w = _explicit_ridge(x, y, lam)
        pred = torch.cat([xh, torch.ones(len(xh), 1, dtype=DT)], 1) @ w
        assert abs(mse - float(((pred - yh) ** 2).mean())) < 1e-9
    assert best == min(table, key=lambda r: r[1])[0]


def test_select_ridge_holdout_accuracy_criterion():
    x, t = _data(n=250)
    xh, th = _data(n=120, seed=5)
    solver = ExactHeadSolver(x.shape[1], 4)
    solver.accumulate(x, t)
    yh = F.one_hot(th.choice, 4).to(DT)
    best, table = solver.select_ridge_holdout("choice", xh, yh, grid=(1e-3, 1e-1, 10.0), criterion="acc")
    assert all(0.0 <= e <= 1.0 for _, e in table)
    assert best == min(table, key=lambda r: (r[1], -r[0]))[0]
    with pytest.raises(ValueError):
        solver.select_ridge_holdout("choice", xh, yh, criterion="boh")
