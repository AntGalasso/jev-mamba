"""
Flusso reale Stage 1 -> Stage 2 su CLINC150 con un backbone SSM selettivo in puro PyTorch (CPU).

  Stage 1: backbone + 3 teste addestrati insieme con CE/BCE (B1)
  Stage 2: backbone congelato, teste ricalcolate in forma chiusa (B3) + calibrazione LOO + DomainModel,
           confrontate con LR (B4) e con le teste di Stage 1, tramite experiments/clinc150_heads.analyze().

Il backbone e' minuscolo e addestrato da zero su 15k frasi: NON e' un risultato su Mamba pre-addestrato.
Serve a verificare che le teste si comportino bene quando le feature sono state addestrate per teste lineari.

Uso:  python experiments/clinc150_tiny_ssm.py --epochs 10 --out results/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from clinc150_heads import analyze, load_clinc  # noqa: E402

from mamba_jev import CombinedJevLoss, MambaJevConfig, MambaJevModel, TrainConfig, train  # noqa: E402
from mamba_jev.metrics import auroc, ece  # noqa: E402
from mamba_jev.tiny_ssm import TinySSMBackbone  # noqa: E402

TOK = re.compile(r"[a-z0-9']+|[^\sa-z0-9]")
MAXLEN = 24


def tokenize(t: str):
    return TOK.findall(t.lower())[:MAXLEN]


def pad_batch(seqs, vocab):
    ids = torch.zeros(len(seqs), max(len(s) for s in seqs), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor([vocab.get(w, 1) for w in s])  # 1 = <unk>, 0 = pad (right-padding)
    return ids, (ids != 0).long()


def make_batches(seqs, vocab, choice, noul, bs, shuffle, seed=0):
    """Batch con bucketing per lunghezza (la scansione e' sequenziale: batch di frasi corte sono molto piu' veloci)."""
    n = len(seqs)
    g = np.random.default_rng(seed)
    order = g.permutation(n) if shuffle else np.arange(n)
    chunk = bs * 50
    out = []
    for i in range(0, n, chunk):
        part = sorted(order[i : i + chunk], key=lambda j: len(seqs[j]))
        for k in range(0, len(part), bs):
            idx = part[k : k + bs]
            ids, mask = pad_batch([seqs[j] for j in idx], vocab)
            out.append(
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "choice_label": torch.tensor([choice[j] for j in idx]),
                    "score_target": torch.zeros(len(idx)),
                    "score_mask": torch.zeros(len(idx), dtype=torch.bool),  # score non addestrato in Stage 1
                    "noul_label": torch.tensor([noul[j] for j in idx], dtype=torch.float32),
                    "idx": torch.tensor(idx),
                }
            )
    if shuffle:
        out = [out[i] for i in g.permutation(len(out))]
    return out


@torch.no_grad()
def extract(model, seqs, vocab, bs=256):
    model.eval()
    feats = np.zeros((len(seqs), model.cfg.d_model), dtype=np.float32)
    for b in make_batches(seqs, vocab, [-100] * len(seqs), [0] * len(seqs), bs, shuffle=False):
        feats[b["idx"].numpy()] = model.extract_features(b["input_ids"], b["attention_mask"]).float().numpy()
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/clinc150/data_full.json")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse", action="store_true", help="carica tiny_ssm_stage1.pt da --out invece di riaddestrare")
    ap.add_argument("--init-embeddings", default="none", choices=["none", "spacy"],
                    help="spacy: embedding inizializzati con i vettori en_core_web_md (d_model forzato a 300)")
    ap.add_argument("--tag", default="tiny_ssm", help="nome dei file di output")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)

    raw = load_clinc(Path(args.data))
    intents = sorted({lab for _, lab in raw["train"]})
    cid = {k: i for i, k in enumerate(intents)}
    seqs = {k: [tokenize(t) for t, _ in v] for k, v in raw.items()}
    counts = {}
    for s in seqs["train"]:
        for w in s:
            counts[w] = counts.get(w, 0) + 1
    vocab = {w: i + 2 for i, w in enumerate(sorted(counts))}
    V = len(vocab) + 2
    C = len(intents)

    if args.init_embeddings == "spacy":
        args.d_model = 300
    cfg = MambaJevConfig(d_model=args.d_model, n_layer=args.n_layer, vocab_size=V, num_choices=C, param_dtype="float32", pad_token_id=0)
    model = MambaJevModel(cfg, backbone=TinySSMBackbone(V, args.d_model, args.n_layer, d_state=8, dropout=args.dropout))
    if args.init_embeddings == "spacy":
        import spacy

        nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "lemmatizer", "tagger", "attribute_ruler"])
        emb = model.backbone.embedding.weight
        with torch.no_grad():
            emb.normal_(0, 0.1)
            emb[0].zero_()
            for w, i in vocab.items():
                lex = nlp.vocab[w]
                if lex.has_vector:
                    emb[i] = torch.tensor(lex.vector) / 5.0  # scala simile all'init casuale
    print(f"vocab={V} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    # ---- Stage 1: end-to-end (in-scope + 100 oos_train per la testa noul)
    s_tr = seqs["train"] + seqs["oos_train"]
    c_tr = [cid[l] for _, l in raw["train"]] + [-100] * len(raw["oos_train"])
    n_tr = [0] * len(raw["train"]) + [1] * len(raw["oos_train"])
    loss_fn = CombinedJevLoss(w_choice=1.0, w_score=0.0, w_noul=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * ((len(s_tr) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total + 10, pct_start=0.1)
    val_b = make_batches(seqs["val"], vocab, [cid[l] for _, l in raw["val"]], [0] * len(seqs["val"]), 256, False)
    ckpt = Path(args.out) / "tiny_ssm_stage1.pt"
    t0 = time.perf_counter()
    if args.reuse:
        model.load_state_dict(torch.load(ckpt))
    for ep in range(0 if args.reuse else args.epochs):
        model.train()
        for b in make_batches(s_tr, vocab, c_tr, n_tr, args.bs, True, seed=ep):
            opt.zero_grad()
            out = model(b["input_ids"], b["attention_mask"])
            from mamba_jev import JevTargets

            loss, _ = loss_fn(out, JevTargets(b["choice_label"], b["score_target"], b["noul_label"], b["score_mask"]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            correct = sum(int((model(b["input_ids"], b["attention_mask"]).choice_logits.argmax(-1) == b["choice_label"]).sum()) for b in val_b)
        print(f"epoch {ep+1}/{args.epochs}  val_acc={correct/len(seqs['val']):.4f}  elapsed={time.perf_counter()-t0:.0f}s", flush=True)
    stage1_time = time.perf_counter() - t0

    out_dir0 = Path(args.out)
    out_dir0.mkdir(parents=True, exist_ok=True)
    if not args.reuse:
        torch.save(model.state_dict(), out_dir0 / "tiny_ssm_stage1.pt")

    # ---- feature congelate + teste di Stage 1 come baseline B1
    feats = {k: extract(model, v, vocab) for k, v in seqs.items()}
    b1 = {
        "choice": (model.choice_head.weight.detach().t().double().numpy(), model.choice_head.bias.detach().double().numpy()),
        "noul": (model.noul_head.weight.detach().squeeze(0).double().numpy(), float(model.noul_head.bias.detach())),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{args.tag}_features.npz", **feats)
    res = analyze(feats, raw, args.tag, b1=b1)
    res["stage1_train_time_s"] = round(stage1_time, 1)

    # ---- percorso "libreria": train(use_exact_solver=True, loo_calibration=True) sul MODELLO, poi predict()
    ex_b = make_batches(s_tr, vocab, c_tr, n_tr, 256, False)
    t0 = time.perf_counter()
    train(
        model, loss_fn, ex_b,
        TrainConfig(use_exact_solver=True, solver_ridge="gcv", loo_calibration=True, score_mode="msp", autocast_dtype=None),
        device="cpu",
    )
    res["library_stage2_time_s"] = round(time.perf_counter() - t0, 2)
    test_b = make_batches(seqs["test"] + seqs["oos_test"], vocab, [0] * (len(seqs["test"]) + len(seqs["oos_test"])), [0] * (len(seqs["test"]) + len(seqs["oos_test"])), 256, False)
    n_id = len(seqs["test"])
    keys = ("choice", "conf", "score", "noul_prob", "ood_leverage", "ood_maha")
    acc = {k: np.zeros(n_id + len(seqs["oos_test"])) for k in keys}
    for b in test_b:
        pr = model.predict(b["input_ids"], b["attention_mask"])
        i = b["idx"].numpy()
        acc["choice"][i] = pr["choice"].numpy()
        acc["conf"][i] = pr["choice_probs"].max(-1).values.numpy()
        for k in keys[2:]:
            acc[k][i] = pr[k].numpy()
    y_te = np.array([cid[l] for _, l in raw["test"]])
    ok = (acc["choice"][:n_id] == y_te).astype(float)
    lab = np.r_[np.zeros(n_id), np.ones(len(seqs["oos_test"]))]
    res["library_path"] = dict(
        test_acc=round(float(ok.mean()), 4),
        ece_msp=round(ece(acc["conf"][:n_id], ok), 4),
        ece_score=round(ece(acc["score"][:n_id], ok), 4),
        auroc_msp=round(auroc(1 - acc["conf"], lab), 4),
        auroc_leverage=round(auroc(acc["ood_leverage"], lab), 4),
        auroc_maha=round(auroc(acc["ood_maha"], lab), 4),
        auroc_noul_head=round(auroc(acc["noul_prob"], lab), 4),
    )
    (out_dir / f"clinc150_{args.tag}.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
