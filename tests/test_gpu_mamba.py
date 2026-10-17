"""Test sul backbone Mamba reale: saltati se mancano mamba-ssm o CUDA."""
import pytest
import torch

pytest.importorskip("mamba_ssm")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="richiede CUDA")

from mamba_jev import MambaJevConfig, MambaJevModel  # noqa: E402


def test_real_backbone_forward_and_pooling():
    cfg = MambaJevConfig(d_model=128, n_layer=2, vocab_size=1000, num_choices=8)
    model = MambaJevModel(cfg, device="cuda")
    ids = torch.randint(1, 1000, (4, 16), device="cuda")
    mask = torch.ones_like(ids)
    mask[:, 10:] = 0  # right-padding
    feats = model.extract_features(ids, mask)
    assert feats.shape == (4, 128)
    out = model(ids, mask)
    assert out.choice_logits.shape == (4, 8)
    assert out.choice_logits.dtype == torch.float32
