"""Smoke test end-to-end su GPU: Stage 1 (SGD) + Stage 2 (teste esatte) con dati sintetici."""
import logging
from typing import Dict

import torch

from mamba_jev import CombinedJevLoss, MambaJevConfig, MambaJevModel, TrainConfig, train


def main() -> None:
    from torch.utils.data import DataLoader, Dataset

    class Synthetic(Dataset):
        def __init__(self, n: int, vocab: int, max_len: int) -> None:
            self.n, self.vocab, self.max_len = n, vocab, max_len

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
            length = int(torch.randint(4, self.max_len, ()))
            ids = torch.zeros(self.max_len, dtype=torch.long)  # right-padding con id 0
            ids[:length] = torch.randint(1, self.vocab, (length,))
            noul = float(torch.rand(()) > 0.8)
            return {
                "input_ids": ids,
                "attention_mask": (torch.arange(self.max_len) < length).long(),
                "choice_label": torch.tensor(-100 if noul else int(torch.randint(0, 8, ())), dtype=torch.long),
                "score_target": torch.rand(()),
                "noul_label": torch.tensor(noul),
            }

    cfg = MambaJevConfig(d_model=128, n_layer=2, vocab_size=1000, num_choices=8)
    model = MambaJevModel(cfg, device="cuda")
    loss_fn = CombinedJevLoss(1.0, 0.5, 0.5)
    loader = DataLoader(Synthetic(256, cfg.vocab_size, 32), batch_size=32)

    # Stage 1: gradient-based
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    train(model, loss_fn, loader, TrainConfig(epochs=1, use_exact_solver=False), optimizer=opt)
    # Stage 2: teste esatte a backbone congelato
    train(model, loss_fn, loader, TrainConfig(use_exact_solver=True), optimizer=opt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA non disponibile: lo smoke test del backbone Mamba richiede una GPU.")
    main()
