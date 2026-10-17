import torch
import torch.nn.functional as F

from mamba_jev import CombinedJevLoss, JevOutput, JevTargets

B, C = 6, 4


def _out() -> JevOutput:
    g = torch.Generator().manual_seed(0)
    return JevOutput(
        choice_logits=torch.randn(B, C, generator=g).requires_grad_(),
        score_logit=torch.randn(B, generator=g).requires_grad_(),
        noul_logit=torch.randn(B, generator=g).requires_grad_(),
    )


def _tgt(**kw) -> JevTargets:
    base = dict(
        choice=torch.tensor([0, 1, 2, 3, 0, 1]),
        score=torch.tensor([0.1, 0.9, 0.5, 0.3, 0.7, 0.2]),
        noul=torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
    )
    base.update(kw)
    return JevTargets(**base)


def test_all_choice_labels_ignored_is_finite_and_differentiable():
    out = _out()
    loss, _ = CombinedJevLoss()(out, _tgt(choice=torch.full((B,), -100)))
    assert torch.isfinite(loss)
    loss.backward()
    assert out.choice_logits.grad is not None


def test_head_weights_scale_total():
    out = _out()
    l1, _ = CombinedJevLoss(1.0, 0.0, 0.0)(out, _tgt())
    l2, _ = CombinedJevLoss(2.0, 0.0, 0.0)(out, _tgt())
    assert torch.isclose(l2, 2 * l1)


def test_score_mask_ignores_masked_rows():
    out = _out()
    mask = torch.tensor([True, True, True, False, False, False])
    score = torch.tensor([0.1, 0.9, 0.5, 0.99, 0.01, 0.5])
    loss, _ = CombinedJevLoss(0.0, 1.0, 0.0)(out, _tgt(score=score, score_mask=mask))
    expected = F.binary_cross_entropy_with_logits(out.score_logit[:3], score[:3])
    assert torch.isclose(loss, expected)


def test_mse_variant_applies_sigmoid():
    out = _out()
    tgt = _tgt()
    loss, _ = CombinedJevLoss(0.0, 1.0, 0.0, score_loss="mse")(out, tgt)
    assert torch.isclose(loss, F.mse_loss(out.score_logit.sigmoid(), tgt.score))


def test_invalid_score_loss_name_raises():
    import pytest

    with pytest.raises(ValueError):
        CombinedJevLoss(score_loss="huber")
