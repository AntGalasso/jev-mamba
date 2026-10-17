import pytest
import torch

from mamba_jev import MambaJevModel

from conftest import C, D


def test_last_token_index_right_and_left_padding():
    right = torch.tensor([[1, 1, 1, 0, 0]])
    left = torch.tensor([[0, 0, 1, 1, 1]])
    assert MambaJevModel._last_token_index(right).item() == 2
    assert MambaJevModel._last_token_index(left).item() == 4


def test_extract_features_picks_last_non_pad_token(model):
    ids = torch.tensor([[3, 4, 5, 0, 0], [0, 0, 6, 7, 8]])
    feats = model.extract_features(ids, ids != 0)
    emb = model.backbone.emb.weight
    assert torch.allclose(feats[0], emb[5])
    assert torch.allclose(feats[1], emb[8])


def test_mask_is_inferred_from_pad_token_id(model):
    ids = torch.tensor([[3, 4, 5, 0, 0]])
    assert torch.allclose(model.extract_features(ids), model.extract_features(ids, ids != 0))


def test_forward_shapes(model):
    ids = torch.randint(1, 12, (5, 7))
    out = model(ids)
    assert out.choice_logits.shape == (5, C)
    assert out.score_logit.shape == (5,)
    assert out.noul_logit.shape == (5,)


def test_head_bias_can_be_disabled(cfg):
    from dataclasses import replace

    from conftest import StubBackbone, VOCAB

    m = MambaJevModel(replace(cfg, head_bias=False), backbone=StubBackbone(VOCAB, D))
    for h in (m.choice_head, m.score_head, m.noul_head):
        assert h.bias is None
    with pytest.raises(ValueError):  # intercetta non nulla su testa bias-free
        m.inject_exact_weights(torch.zeros(D, C), torch.zeros(D), torch.zeros(D), choice_b=torch.ones(C))


def test_heads_have_bias_by_default(model):
    for h in (model.choice_head, model.score_head, model.noul_head):
        assert h.bias is not None


def test_inject_exact_weights_roundtrip(model):
    g = torch.Generator().manual_seed(0)
    wc, ws, wn = torch.randn(D, C, generator=g), torch.randn(D, generator=g), torch.randn(D, 1, generator=g)
    model.inject_exact_weights(wc, ws, wn)
    assert torch.allclose(model.choice_head.weight, wc.t())
    feats = torch.randn(3, D)
    out = model.apply_heads(feats)
    assert torch.allclose(out.choice_logits, feats @ wc, atol=1e-5)
    assert torch.allclose(out.score_logit, feats @ ws, atol=1e-5)
    assert not any(h.weight.requires_grad for h in (model.choice_head, model.score_head, model.noul_head))


def test_inject_wrong_shape_raises(model):
    with pytest.raises(ValueError):
        model.inject_exact_weights(torch.randn(D, C + 1), torch.randn(D, 1), torch.randn(D, 1))


def test_freeze_backbone_survives_train_mode(model):
    model.freeze_backbone()
    model.train()
    assert not model.backbone.training
    assert all(not p.requires_grad for p in model.backbone.parameters())
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.backbone.parameters())
