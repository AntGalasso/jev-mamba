"""Fixture condivise: backbone finto (embedding per token) per testare tutto su CPU senza mamba-ssm."""
import pytest
import torch
import torch.nn as nn

from mamba_jev import MambaJevConfig, MambaJevModel

VOCAB, D, C = 12, 16, 4


class StubBackbone(nn.Module):
    """Sostituto leggero del backbone: forward(input_ids) -> (B, L, D)."""

    def __init__(self, vocab: int, d: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(input_ids)


def make_batches(n_batches: int = 8, batch_size: int = 16, max_len: int = 10, seed: int = 0):
    """Dataset sintetico right-padded: tutti i target sono funzioni dell'ultimo token reale."""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        lengths = torch.randint(2, max_len + 1, (batch_size,), generator=g)
        ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, n in enumerate(lengths.tolist()):
            ids[i, :n] = torch.randint(1, VOCAB, (n,), generator=g)
        mask = (torch.arange(max_len)[None] < lengths[:, None]).long()
        last = ids[torch.arange(batch_size), lengths - 1]
        batches.append(
            {
                "input_ids": ids,
                "attention_mask": mask,
                "choice_label": last % C,
                "score_target": (last % 5).float() / 4,
                "noul_label": (last % 2 == 0).float(),
            }
        )
    return batches


@pytest.fixture
def cfg() -> MambaJevConfig:
    return MambaJevConfig(d_model=D, n_layer=1, vocab_size=VOCAB, num_choices=C, param_dtype="float32", pad_token_id=0)


@pytest.fixture
def model(cfg: MambaJevConfig) -> MambaJevModel:
    torch.manual_seed(0)
    return MambaJevModel(cfg, backbone=StubBackbone(VOCAB, D))


@pytest.fixture
def batches():
    return make_batches()
