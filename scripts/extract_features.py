"""
Estrae le feature (hidden state dell'ultimo token non-pad) di un Mamba PRE-ADDESTRATO su CLINC150 e le salva in un npz
utilizzabile da:  python experiments/clinc150_heads.py --encoder npz --npz <file>

!!! NON TESTATO in questo repository: richiede GPU CUDA, `pip install -e ".[mamba]" transformers` e accesso a HuggingFace.
!!! Verificare che le feature abbiano shape (N, d_model) e che right-padding + attention_mask siano rispettati.

Uso:  python scripts/extract_features.py --model state-spaces/mamba-130m --out feats_mamba130m.npz
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

from mamba_jev import MambaJevConfig, MambaJevModel

DATA_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba-130m")
    ap.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--data", default="data/clinc150/data_full.json")
    ap.add_argument("--out", default="feats_mamba.npz")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=64)
    args = ap.parse_args()

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from transformers import AutoTokenizer

    path = Path(args.data)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)  # noqa: S310 - URL fisso
    raw = json.loads(path.read_text())

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # IMPORTANTE: con left-padding i pad entrano nello stato ricorrente
    lm = MambaLMHeadModel.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
    cfg = MambaJevConfig(d_model=lm.config.d_model, n_layer=lm.config.n_layer, num_choices=2, pad_token_id=None)
    model = MambaJevModel(cfg, device="cuda", backbone=lm.backbone).eval()

    out = {}
    with torch.no_grad():
        for split, rows in raw.items():
            texts = [t for t, _ in rows]
            feats = []
            for i in range(0, len(texts), args.bs):
                enc = tok(texts[i : i + args.bs], return_tensors="pt", padding=True, truncation=True, max_length=args.max_len).to("cuda")
                feats.append(model.extract_features(enc["input_ids"], enc["attention_mask"]).float().cpu())
            out[split] = torch.cat(feats).numpy()
            print(split, out[split].shape, flush=True)
    np.savez(args.out, **out)


if __name__ == "__main__":
    main()
