import torch

from mamba_jev import CombinedJevLoss, MambaJevConfig, MambaJevModel, TrainConfig, train
from mamba_jev.tiny_ssm import TinySSMBackbone

from conftest import C, VOCAB, make_batches


def test_tiny_ssm_is_causal():
    torch.manual_seed(0)
    m = TinySSMBackbone(VOCAB, d_model=16, n_layer=2, d_state=4).eval()
    ids = torch.randint(1, VOCAB, (3, 10))
    ids2 = ids.clone()
    ids2[:, 6:] = torch.randint(1, VOCAB, (3, 4))
    with torch.no_grad():
        assert torch.equal(m(ids)[:, :6], m(ids2)[:, :6])  # i token futuri non cambiano lo stato passato


def test_stage1_then_exact_stage2_with_ssm_backbone():
    torch.manual_seed(0)
    cfg = MambaJevConfig(d_model=16, n_layer=1, vocab_size=VOCAB, num_choices=C, param_dtype="float32", pad_token_id=0)
    model = MambaJevModel(cfg, backbone=TinySSMBackbone(VOCAB, 16, 1, d_state=4))
    batches = make_batches(n_batches=6)
    loss_fn = CombinedJevLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    train(model, loss_fn, batches, TrainConfig(epochs=3, autocast_dtype=None, log_every=10_000), optimizer=opt, device="cpu")
    train(model, loss_fn, batches, TrainConfig(use_exact_solver=True, solver_ridge="gcv", loo_calibration=True,
                                               autocast_dtype=None), device="cpu")
    assert model.domain is not None
    assert all(not p.requires_grad for p in model.backbone.parameters())
    out = model.predict(batches[0]["input_ids"], batches[0]["attention_mask"])
    assert torch.isfinite(out["choice_probs"]).all() and torch.isfinite(out["ood_maha"]).all()
