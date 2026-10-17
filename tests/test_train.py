import pytest
import torch

from mamba_jev import CombinedJevLoss, TrainConfig, evaluate, train

CPU = torch.device("cpu")


def test_exact_branch_fits_heads_and_freezes_everything(model, batches):
    loss_fn = CombinedJevLoss()
    train(model, loss_fn, batches, TrainConfig(use_exact_solver=True), device="cpu")

    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(not h.weight.requires_grad for h in (model.choice_head, model.score_head, model.noul_head))

    metrics = evaluate(model, loss_fn, batches, CPU)
    assert metrics["choice_acc"] > 0.95

    b = batches[0]
    pred = model.predict(b["input_ids"], b["attention_mask"])
    assert ((pred["noul_prob"] > 0.5).float() == b["noul_label"]).float().mean() > 0.95


def test_standard_branch_reduces_loss(model, batches):
    torch.manual_seed(0)
    loss_fn = CombinedJevLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = evaluate(model, loss_fn, batches, CPU)["loss"]
    train(model, loss_fn, batches, TrainConfig(epochs=15, log_every=10_000), optimizer=opt, device="cpu")
    after = evaluate(model, loss_fn, batches, CPU)["loss"]
    assert after < before


def test_standard_branch_requires_optimizer(model, batches):
    with pytest.raises(ValueError):
        train(model, CombinedJevLoss(), batches, TrainConfig(use_exact_solver=False), device="cpu")
