import pytest
import torch
import torch.nn.functional as F

from mamba_jev import ExactHeadSolver, JevTargets


def _targets(n: int, c: int, seed: int = 0) -> JevTargets:
    g = torch.Generator().manual_seed(seed)
    return JevTargets(
        choice=torch.randint(0, c, (n,), generator=g),
        score=torch.rand(n, generator=g),
        noul=(torch.rand(n, generator=g) > 0.5).float(),
    )


def _x(n: int, d: int, seed: int = 1) -> torch.Tensor:
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def test_matches_lstsq():
    n, d, c = 512, 64, 4
    x, t = _x(n, d), _targets(n, c)
    solver = ExactHeadSolver(d, c, ridge=0.0, fit_intercept=False)
    solver.accumulate(x, t)
    w = solver.solve()["choice_w"]
    ref = torch.linalg.lstsq(x, F.one_hot(t.choice, c).float()).solution
    assert w.shape == (d, c)
    assert torch.allclose(w, ref, atol=1e-3)


def test_intercept_matches_lstsq_on_augmented_matrix():
    n, d, c = 512, 64, 4
    x, t = _x(n, d) + 3.0, _targets(n, c)  # feature con media lontana da zero
    solver = ExactHeadSolver(d, c, ridge=0.0)
    solver.accumulate(x, t)
    sol = solver.solve()
    aug = torch.cat([x, torch.ones(n, 1)], 1)
    ref = torch.linalg.lstsq(aug.double(), F.one_hot(t.choice, c).double()).solution
    assert torch.allclose(sol["choice_w"].double(), ref[:d], atol=1e-4)
    assert torch.allclose(sol["choice_b"].double(), ref[d], atol=1e-4)


def test_streaming_equals_single_pass():
    n, d, c = 300, 32, 4
    x, t = _x(n, d), _targets(n, c)
    whole = ExactHeadSolver(d, c, ridge=1e-4)
    whole.accumulate(x, t)
    chunked = ExactHeadSolver(d, c, ridge=1e-4)
    for sl in (slice(0, 100), slice(100, 220), slice(220, n)):
        chunked.accumulate(x[sl], JevTargets(t.choice[sl], t.score[sl], t.noul[sl]))
    w1, w2 = whole.solve(), chunked.solve()
    for k in w1:
        assert torch.allclose(w1[k], w2[k], atol=1e-4), k


def test_ignored_choice_rows_are_excluded():
    n, d, c = 400, 32, 4
    x, t = _x(n, d), _targets(n, c)
    choice = t.choice.clone()
    choice[::2] = -100
    solver = ExactHeadSolver(d, c, ridge=0.0, fit_intercept=False)
    solver.accumulate(x, JevTargets(choice, t.score, t.noul))
    valid = choice != -100
    assert int(solver.count["choice"]) == int(valid.sum())
    ref = torch.linalg.lstsq(x[valid], F.one_hot(choice[valid], c).float()).solution
    assert torch.allclose(solver.solve()["choice_w"], ref, atol=1e-3)


def test_score_mask_is_respected():
    n, d, c = 100, 16, 3
    x, t = _x(n, d), _targets(n, c)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:40] = True
    solver = ExactHeadSolver(d, c)
    solver.accumulate(x, JevTargets(t.choice, t.score, t.noul, score_mask=mask))
    assert int(solver.count["score"]) == 40
    assert int(solver.count["noul"]) == n


def test_underdetermined_system_is_finite_with_ridge():
    n, d, c = 8, 32, 3  # n < D
    x, t = _x(n, d), _targets(n, c)
    solver = ExactHeadSolver(d, c, ridge=1e-3)
    solver.accumulate(x, t)
    for w in solver.solve().values():
        assert torch.isfinite(w).all()


def test_head_without_valid_rows_raises():
    n, d, c = 20, 8, 3
    x, t = _x(n, d), _targets(n, c)
    solver = ExactHeadSolver(d, c)
    solver.accumulate(x, JevTargets(torch.full((n,), -100), t.score, t.noul))
    with pytest.raises(RuntimeError):
        solver.solve()


def test_noul_head_separates_linearly_separable_data():
    n, d = 2000, 8
    x = _x(n, d)
    noul = (x[:, 0] > 0).float()
    solver = ExactHeadSolver(d, 3, ridge=1e-6)
    solver.accumulate(x, JevTargets(torch.zeros(n, dtype=torch.long), torch.rand(n), noul))
    w = solver.solve()["noul_w"]
    pred = ((x @ w).squeeze(1) > 0).float()
    assert (pred == noul).float().mean() > 0.9


def test_reset_clears_state():
    n, d, c = 50, 8, 3
    x, t = _x(n, d), _targets(n, c)
    solver = ExactHeadSolver(d, c)
    solver.accumulate(x, t)
    solver.reset()
    assert all(int(v) == 0 for v in solver.count.values())


def test_untrained_score_head_gives_zero_weights_instead_of_raising():
    n, d, c = 60, 8, 3
    x, t = _x(n, d), _targets(n, c)
    solver = ExactHeadSolver(d, c)
    solver.accumulate(x, JevTargets(t.choice, t.score, t.noul, score_mask=torch.zeros(n, dtype=torch.bool)))
    sol = solver.solve()
    assert torch.count_nonzero(sol["score_w"]) == 0 and torch.count_nonzero(sol["score_b"]) == 0
    assert torch.count_nonzero(sol["choice_w"]) > 0
