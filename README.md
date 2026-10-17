# Mamba-Jev

Un backbone Mamba codifica l'intero prompt e, in un solo forward pass, tre teste lineari producono: **choice** (N classi), **score** (confidenza in [0,1]) e **noul** (rifiuto per richieste fuori dominio). Le teste si calcolano in **forma chiusa** (ridge ai minimi quadrati, in streaming) invece che con la discesa del gradiente.

Cosa aggiunge questa versione alla sola "testa ridge":

| | cosa | stato verificato |
|---|---|---|
| intercetta non penalizzata | teste con bias, stimato dalle statistiche centrate | test vs `lstsq` aumentato |
| `select_ridge` (GCV) | rho dalle statistiche, nessun passaggio sui dati | test vs matrice hat esplicita |
| `select_ridge_holdout` | rho su held-out (MSE o accuratezza) | test vs calcolo esplicito |
| LOO esatto (`HeadFit.loo_predict`) | `e/(1-h)` dallo stesso Cholesky, nessun refit | test vs refit con riga esclusa |
| leverage + Mahalanobis (`DomainModel`) | rejection dalle stesse statistiche, **senza dati OOD** | test vs gaussiana esplicita; CLINC150 |
| `Calibration` | temperatura (choice), Platt (noul), score = MSP calibrata | recupero di parametri noti |
| `remove()` / `expand_choices()` | rimozione di righe e aggiunta di classi esatte | differenza vs refit <= 4e-15 |
| `fit_jev_heads` | pipeline: ridge -> calibrazione (LOO o held-out) -> dominio | CLINC150, tre backbone |

**Leggi [docs/RESULTS.md](docs/RESULTS.md) prima di fidarti di qualcosa**: contiene anche i risultati negativi. In breve: il LOO senza validation calibra bene solo se il backbone non ha visto le righe del fit; la Mahalanobis "gratis" eguaglia la MSP di una testa CE su un backbone e fallisce su un altro; la testa LS perde 9-16 punti di accuracy contro CE su encoder non addestrati, ma con feature addestrate (Stage 1) e rho su held-out e' alla pari. **Nessun backbone Mamba reale e' stato eseguito.**

## Installazione

```bash
pip install -e ".[dev]"            # package + pytest (CPU: solver, calibrazione, training loop, SSM di riferimento)
pip install -e ".[mamba]"          # backbone reale (richiede CUDA): mamba-ssm, causal-conv1d
pip install -e ".[exp]"            # esperimenti su CLINC150
```

## Layout

```
src/mamba_jev/
  config.py       MambaJevConfig (backbone + dimensioni delle teste, head_bias)
  types.py        JevOutput, JevTargets
  model.py        MambaJevModel: extract_features, predict, inject_exact_weights, load_fit, expand_choice_head
  loss.py         CombinedJevLoss (pesi per testa, mascheramento per riga)
  solver.py       ExactHeadSolver: statistiche in streaming, ridge (GCV/held-out), HeadFit (LOO, leverage), remove, expand_choices
  uncertainty.py  DomainModel (leverage, Mahalanobis), Calibration, fit_temperature, fit_platt
  pipeline.py     fit_jev_heads(): ridge -> calibrazione -> DomainModel da feature in memoria
  metrics.py      ECE, Brier, NLL, AUROC, FPR@TPR, AURC, bootstrap
  tiny_ssm.py     backbone SSM selettivo in puro PyTorch (CPU, per test e prove del flusso Stage 1 -> 2)
  train.py        train(): branch SGD / branch exact-solver (+ loo_calibration)
tests/            pytest su CPU (backbone finto); test_uncertainty.py verifica le formule contro calcoli espliciti
experiments/      clinc150_heads.py (analisi teste), clinc150_tiny_ssm.py (Stage 1 -> 2), make_report.py
scripts/          smoke_test.py (GPU), extract_features.py (Mamba pre-addestrato, NON TESTATO)
docs/             research plan (.tex/.pdf) e RESULTS.md
```

## Quickstart

```python
import torch
from mamba_jev import JevTargets, MambaJevConfig, MambaJevModel, fit_jev_heads

# feature gia' estratte (N, D) da un backbone CONGELATO e target
tgt = JevTargets(choice=y, score=torch.zeros(len(y)), noul=noul)      # choice = -100 sulle righe OOD
fit = fit_jev_heads(X, tgt, num_choices=150, ridge="gcv", score_mode="msp")

# se il backbone e' stato addestrato sulle stesse righe (Stage 1): usa un held-out, NON il LOO
fit = fit_jev_heads(X, tgt, 150, ridge="holdout", holdout=(X_val, tgt_val), score_mode="msp")

model.load_fit(fit)                       # pesi + intercette + calibrazione + DomainModel
out = model.predict(ids, mask)            # choice, choice_probs (calibrate), score, noul_prob, ood_leverage, ood_maha
```

Flusso a due stadi con il training loop (streaming, senza tenere le feature, solo A e B in memoria `O(D^2)`):

```python
from mamba_jev import CombinedJevLoss, TrainConfig, train
loss_fn = CombinedJevLoss(w_choice=1.0, w_score=0.0, w_noul=0.5)
train(model, loss_fn, loader, TrainConfig(epochs=3), optimizer=opt)                      # Stage 1: SGD
train(model, loss_fn, loader, TrainConfig(use_exact_solver=True, solver_ridge="gcv"))    # Stage 2: teste esatte
```
`loo_calibration=True` in `TrainConfig` tiene le feature in memoria e usa il LOO: **valido solo con backbone non addestrato sulle righe** (la libreria lo segnala con un warning).

Formato del batch prodotto dal `loader`:

| chiave | shape / dtype | note |
|---|---|---|
| `input_ids` | (B, L) long | **right-padding** consigliato |
| `attention_mask` | (B, L) | opzionale; se assente e' dedotta da `pad_token_id` |
| `choice_label` | (B,) long | `-100` dove non definita |
| `score_target` | (B,) float in [0,1] | |
| `noul_label` | (B,) float in {0,1} | |
| `score_mask` | (B,) bool | opzionale; se tutta False la testa score resta a zero |

## Test ed esperimenti

```bash
pytest                                              # 48 test CPU (1 skip senza GPU/mamba-ssm)
python experiments/clinc150_heads.py --encoder lsa --out results/
python experiments/clinc150_tiny_ssm.py --out results/ && python experiments/make_report.py results/
python scripts/smoke_test.py                        # GPU + mamba-ssm
```

## Note di design e limiti

- Il solver minimizza `||XW - Y||^2 + lambda ||W||^2` con target nello spazio dei logit (one-hot per choice, `logit(clip(s))` per score, `+/-m` per noul). **Non** e' il minimizzatore di CE/BCE: le probabilita' si ottengono con `Calibration`.
- Le statistiche sono accumulate in fp64 (default) e centrate a posteriori: con feature a media grande l'accumulo fp32 soffre di cancellazione.
- Il LOO esatto vale per `lambda` fisso e solo per righe usate nel fit; non tiene conto di un backbone addestrato su quelle righe.
- `ExactHeadSolver` e' l'implementazione di riferimento; per usare un altro solver basta rispettare `reset / accumulate / all_reduce / solve`.
- Il backbone e' iniettabile (`MambaJevModel(cfg, backbone=...)`): `tiny_ssm.TinySSMBackbone` per CPU, `MixerModel` di mamba-ssm su GPU.
- Non implementato: ridge differenziabile con split support/query, aggiornamenti Sherman-Morrison, garanzie conformal, LOO esatto per il LDA.
