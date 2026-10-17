# Risultati: teste in forma chiusa su CLINC150

Questo documento riporta cosa e' stato **misurato** (non cosa ci si aspetta) sulle teste di Mamba-Jev.
Tutti i numeri sono riproducibili con `experiments/` (vedi in fondo). Limiti dichiarati nella sezione finale:
**nessun backbone Mamba e' stato eseguito** (l'ambiente non aveva GPU), quindi le conclusioni riguardano le teste.

## Setup

- **Dati**: CLINC150 `data_full` (150 intent in-scope; train 15000, val 3000, test 4500; out-of-scope: 100 train, 100 val, 1000 test).
  La testa `noul` vede solo i 100 esempi OOS di train; la `choice` ignora le righe OOS (`-100`).
- **Tre backbone congelati**, scelti per avere geometrie diverse:
  `lsa` (TF-IDF parole+caratteri -> SVD 384, non addestrato su CLINC oltre il fit dell'SVD),
  `spacy` (media dei vettori di `en_core_web_md`, 300-d, preaddestrato su altro corpus),
  `tiny_ssm` (SSM selettivo in puro PyTorch, 2 strati, d=192, 1.5M parametri, **addestrato da zero con CE sul training set**: flusso Stage 1 -> Stage 2 del progetto; accuratezza val Stage 1 = 81.6%).
- **Preprocessing**: z-score fittato sul train in-scope, uguale per tutti i metodi.
- **Baseline**: regressione logistica L2 a batch intero (L-BFGS, reg scelta su val), probe SGD (AdamW, 3 seed), e per `tiny_ssm` le teste CE di Stage 1 (B1).
- **Protocolli di calibrazione**: *LOO* (nessun dato extra: temperatura stimata sulle predizioni leave-one-out del training) e *held-out* (rho e temperatura scelti sulla validation, stesso budget dei baseline).
- Rejection: positivo = OOS; AUROC con IC bootstrap 95% (300 ricampionamenti), FPR@95%TPR = quota di in-scope scartati quando si cattura il 95% degli OOS.
- Un solo training del backbone, un solo split (CLINC ha split ufficiale): gli IC catturano la variabilita' del test set, **non** quella del training. Le differenze piccole (< ~0.01 AUROC) non vanno lette come ordinamento.

## Sintesi

1. **Il LOO esatto e' gratuito ma vale solo se il backbone non ha visto le righe del fit.** Con encoder esterni (`lsa`, `spacy`) la temperatura stimata sul LOO, senza validation, coincide con quella stimata sulla validation (0.0513 vs 0.0527; 0.0352 vs 0.0371) e l'ECE e' identico (0.062 vs 0.060; 0.146 vs 0.144). Con `tiny_ssm`, addestrato sulle stesse righe, **fallisce**: T=0.031 contro 0.055 e ECE 0.144 contro 0.030. Il LOO corregge l'ottimismo del *refit delle teste*, non quello del backbone. Stesso problema per GCV (sceglie rho=1e-3 -> 75.0% di accuracy; rho=1000 scelto su held-out -> 81.6%).
2. **Rejection dalle stesse statistiche, senza dati OOD (Mahalanobis a covarianza condivisa; i medie di classe sono le colonne di `X^T Y`)**: su `spacy` AUROC 0.798 [0.786, 0.810] contro 0.804 [0.790, 0.819] della MSP della regressione logistica (FPR@95: 0.632 vs 0.638), e batte la testa `noul` addestrata su 100 OOS (0.687 LS, 0.742 LR). Su `tiny_ssm` 0.892 [0.884, 0.903] contro 0.912 di MSP/energy LR (leggermente peggio, FPR@95 0.321 vs 0.274) e contro 0.828-0.853 delle teste `noul`. Su `lsa` **fallisce** (leverage 0.231, Mahalanobis 0.511): i testi OOS hanno norma LSA piu' piccola e cadono vicino alla media. Il metodo dipende dalla geometria delle feature: va verificato su ogni backbone.
3. **La testa LS e' competitiva con CE solo se le feature sono state addestrate per teste lineari e rho e' scelto su held-out.** `tiny_ssm`: 81.6% (rho=1000 su held-out) contro 82.1% LR, ECE 0.020 vs 0.016, AUROC errori 0.922 vs 0.931, AUROC rejection 0.911 vs 0.912. Sugli encoder non addestrati il divario e' grande a qualunque rho: `lsa` 82.5% vs 91.4% (-9 punti), `spacy` 62.6% vs 78.2% (-16). Il LDA calcolato dalle stesse statistiche non lo chiude (83.5% e 66.6%).
4. **La testa `score` lineare addestrata sulla correttezza LOO e' dominata dalla MSP calibrata** (AUROC errori 0.733/0.676/0.377 contro 0.868/0.762/0.892 su lsa/spacy/tiny_ssm). Il default di `fit_jev_heads` e' quindi `score_mode="msp"`.
5. **Una testa `noul` lineare con 100 esempi OOS e' un rilevatore debole** (AUROC 0.69-0.85) e migliora lentamente con piu' esempi (0.59 -> 0.70 da 5 a 100 su lsa).
6. **Costo (H2) ridimensionato.** Fit chiuso completo (stats + GCV + LOO + Mahalanobis) 0.9-2.8 s contro 2.7-21.9 s per un fit L-BFGS di LR (3-9.5x) e 20-74 s con una griglia di 4 regolarizzazioni (19-32x), ma solo **1.4-3.4x** rispetto a un probe SGD a 30 epoche (CPU a 2 core, feature in cache, nessun tuning dell'SGD incluso). I vantaggi solidi sono altri: nessun iperparametro da tarare con il gradiente, determinismo, additivita' tra rank e aggiornamenti esatti.
7. **Aggiunta di classi e rimozione di righe sono esatte**: differenza massima sui pesi rispetto al refit completo 0 (aggiunta di 10 classi) e <= 3.6e-15 (rimozione di 1/7 delle righe), in fp64.



## Come leggere i risultati / limiti

- **Nessun Mamba pre-addestrato** e nessun risultato su GPU. `tiny_ssm` e' un backbone SSM di 1.5M parametri addestrato da zero: dimostra il flusso Stage 1 -> Stage 2, non la qualita' di Mamba. Per provare Mamba: `scripts/extract_features.py` (NON TESTATO, richiede GPU e HuggingFace) e `experiments/clinc150_heads.py --encoder npz`.
- Un solo dataset (CLINC150), un solo seed per il backbone, un solo split: la generalita' dei punti 1-3 e' da confermare su Banking77/HWU64/ROSTD e con piu' seed. Le coppie con IC sovrapposti vanno considerate indistinguibili.
- Il rilevatore OOS di CLINC ha 100 esempi: la valutazione `noul` supervisionata e' volutamente quella del caso reale "pochi esempi OOD".
- Il LDA e la Mahalanobis usano covarianza condivisa e ridge relativa 1e-3 (sensibilita' nei JSON: per ridge <= 1e-2 la variazione e' <= 0.01 AUROC su tutti e tre i backbone; a 0.1 il leverage di `spacy` scende di 0.03).
- Non e' stato implementato: ridge differenziabile con split support/query, aggiornamento incrementale con Sherman-Morrison (il re-solve costa `O(D^3)`), conformal prediction, LOO esatto per il LDA.
- La riga "libreria" (`train(..., loo_calibration=True)` su `tiny_ssm`) conferma il limite del punto 1: ECE 0.144 (invece di 0.020 con held-out). `train()` ora lo segnala con un warning.

## Riproduzione

```bash
pip install -e ".[dev,exp]"
python experiments/clinc150_heads.py --encoder lsa  --dim 384 --out results/     # scarica CLINC150 da GitHub
python experiments/clinc150_heads.py --encoder spacy --out results/              # richiede spacy + en_core_web_md
python experiments/clinc150_tiny_ssm.py --epochs 15 --d-model 192 --dropout 0.25 --out results/   # ~8 min su 2 core CPU
python experiments/make_report.py results/                                       # tabelle in Markdown
```
I JSON completi (inclusa la sensibilita' del rejection a rho del dominio e lo sweep del numero di OOS) sono in `results/`.

---

# Tabelle complete (generate da `experiments/make_report.py`)

### Setup

| encoder | D | LS fit closed-form (s) | LR L-BFGS 1 fit (s) | LR grid 4 reg (s) | SGD probe (s) |
|---|---|---|---|---|---|
| lsa | 384 | 2.84 | 9.03 | 53.6 | 4.1 |
| spacy | 300 | 2.31 | 21.87 | 74.4 | 3.8 |
| tiny_ssm | 192 | 0.92 | 2.73 | 20.3 | 3.1 |

### Choice: accuratezza e calibrazione (test in-scope, 4500)

**lsa** (SGD probe acc 0.904 ± 0.001; LDA da stesse statistiche acc 0.8347)

| metodo | acc | ECE | NLL | Brier |
|---|---|---|---|---|
| LS raw softmax | 0.825 | 0.812 | 4.449 | 0.982 |
| LS + T(LOO)  [ours, no val] | 0.825 | 0.062 [0.054,0.070] | 0.777 | 0.263 |
| LS + T(val) | 0.825 | 0.060 [0.054,0.070] | 0.777 | 0.263 |
| LS rho=holdout + T(holdout) | 0.826 | 0.061 | 0.776 | 0.264 |
| LR (CE) raw | 0.914 | 0.057 | 0.386 | 0.137 |
| LR (CE) + T(val) | 0.914 | 0.012 [0.010,0.020] | 0.348 | 0.127 |

T(LOO)=0.0513, T(val)=0.0527; rho GCV={'choice': 0.01, 'score': 1000.0, 'noul': 1.0}; rho_holdout={'choice': 10.0, 'score': 1000.0, 'noul': 1e-08}

**spacy** (SGD probe acc 0.767 ± 0.001; LDA da stesse statistiche acc 0.6656)

| metodo | acc | ECE | NLL | Brier |
|---|---|---|---|---|
| LS raw softmax | 0.626 | 0.616 | 4.724 | 0.988 |
| LS + T(LOO)  [ours, no val] | 0.626 | 0.146 [0.133,0.159] | 1.781 | 0.555 |
| LS + T(val) | 0.626 | 0.144 [0.132,0.158] | 1.777 | 0.556 |
| LS rho=holdout + T(holdout) | 0.626 | 0.144 | 1.756 | 0.555 |
| LR (CE) raw | 0.782 | 0.051 | 0.870 | 0.321 |
| LR (CE) + T(val) | 0.782 | 0.035 [0.028,0.046] | 0.863 | 0.316 |

T(LOO)=0.0352, T(val)=0.0371; rho GCV={'choice': 0.001, 'score': 1000.0, 'noul': 0.001}; rho_holdout={'choice': 0.01, 'score': 1000.0, 'noul': 0.01}

**tiny_ssm** (SGD probe acc 0.818 ± 0.002; LDA da stesse statistiche acc 0.7887)

| metodo | acc | ECE | NLL | Brier |
|---|---|---|---|---|
| LS raw softmax | 0.750 | 0.740 | 4.591 | 0.986 |
| LS + T(LOO)  [ours, no val] | 0.750 | 0.144 [0.136,0.158] | 1.702 | 0.387 |
| LS + T(val) | 0.750 | 0.030 [0.026,0.041] | 1.295 | 0.350 |
| LS rho=holdout + T(holdout) | 0.816 | 0.020 | 0.851 | 0.251 |
| LR (CE) raw | 0.821 | 0.139 | 0.941 | 0.269 |
| LR (CE) + T(val) | 0.821 | 0.016 [0.014,0.027] | 0.834 | 0.240 |
| B1 end-to-end CE raw | 0.823 | 0.088 | 1.157 | 0.260 |
| B1 end-to-end CE + T(val) | 0.823 | 0.022 | 0.846 | 0.238 |

T(LOO)=0.0307, T(val)=0.0546; rho GCV={'choice': 0.001, 'score': 1000.0, 'noul': 0.0001}; rho_holdout={'choice': 1000.0, 'score': 1000.0, 'noul': 10.0}

### Ridge sweep (acc test)

| rho | lsa | spacy | tiny_ssm |
|---|---|---|---|
| 1e-06 | 0.825 | 0.626 | 0.750 |
| 0.0001 | 0.825 | 0.626 | 0.750 |
| 0.01 | 0.825 | 0.626 | 0.755 |
| 0.1 | 0.824 | 0.622 | 0.774 |
| 1 | 0.825 | 0.605 | 0.800 |
| 10 | 0.826 | 0.553 | 0.812 |
| 100 | 0.826 | 0.456 | 0.816 |

### Selective classification (errori in-scope)

**lsa**

| confidenza | AUROC errori | AURC (basso=meglio) | ECE |
|---|---|---|---|
| LS MSP raw | 0.838 | 0.0510 | 0.812 |
| LS MSP + T(LOO) | 0.868 | 0.0439 | 0.062 |
| LS score head (LOO-correct, Platt) | 0.733 | 0.0812 | 0.039 |
| LS max-logit | 0.837 | 0.0510 | - |
| LS rho=holdout MSP + T(holdout) | 0.866 | 0.0439 | 0.061 |
| LR MSP raw | 0.918 | 0.0121 | 0.057 |
| LR MSP + T(val) | 0.925 | 0.0114 | 0.012 |

**spacy**

| confidenza | AUROC errori | AURC (basso=meglio) | ECE |
|---|---|---|---|
| LS MSP raw | 0.723 | 0.2176 | 0.616 |
| LS MSP + T(LOO) | 0.762 | 0.1898 | 0.146 |
| LS score head (LOO-correct, Platt) | 0.676 | 0.2444 | 0.045 |
| LS max-logit | 0.723 | 0.2179 | - |
| LS rho=holdout MSP + T(holdout) | 0.760 | 0.1904 | 0.143 |
| LR MSP raw | 0.828 | 0.0741 | 0.051 |
| LR MSP + T(val) | 0.832 | 0.0728 | 0.035 |

**tiny_ssm**

| confidenza | AUROC errori | AURC (basso=meglio) | ECE |
|---|---|---|---|
| LS MSP raw | 0.853 | 0.0794 | 0.740 |
| LS MSP + T(LOO) | 0.892 | 0.0682 | 0.144 |
| LS score head (LOO-correct, Platt) | 0.377 | 0.4047 | 0.227 |
| LS max-logit | 0.852 | 0.0798 | - |
| LS rho=holdout MSP + T(holdout) | 0.922 | 0.0363 | 0.020 |
| LR MSP raw | 0.927 | 0.0333 | 0.139 |
| LR MSP + T(val) | 0.931 | 0.0326 | 0.016 |

### Rejection: in-scope test (4500) vs out-of-scope test (1000); positivo = OOS

**lsa**

| score | AUROC [CI 95%] | FPR@95TPR |
|---|---|---|
| MSP  (LR, CE) | 0.935 [0.928,0.943] | 0.277 |
| max-logit (LR) | 0.939 [0.932,0.947] | 0.257 |
| energy (LR) | 0.937 [0.929,0.944] | 0.276 |
| MSP + T(LOO) (LS) | 0.894 [0.884,0.903] | 0.439 |
| max-logit (LS) | 0.904 [0.894,0.913] | 0.379 |
| MSP + T(holdout) (LS, rho=holdout) | 0.895 [0.885,0.904] | 0.434 |
| leverage (stats ridge)  [no OOD data] | 0.231 [0.214,0.247] | 0.993 |
| Mahalanobis tied (stats ridge) [no OOD data] | 0.511 [0.492,0.528] | 0.902 |
| noul head LS (100 OOS train) | 0.697 [0.678,0.714] | 0.828 |
| noul head LR (CE, 100 OOS train) | 0.692 [0.673,0.711] | 0.811 |

**spacy**

| score | AUROC [CI 95%] | FPR@95TPR |
|---|---|---|
| MSP  (LR, CE) | 0.804 [0.790,0.819] | 0.638 |
| max-logit (LR) | 0.769 [0.754,0.785] | 0.734 |
| energy (LR) | 0.748 [0.733,0.765] | 0.726 |
| MSP + T(LOO) (LS) | 0.670 [0.652,0.688] | 0.822 |
| max-logit (LS) | 0.647 [0.630,0.666] | 0.835 |
| MSP + T(holdout) (LS, rho=holdout) | 0.673 [0.655,0.692] | 0.823 |
| leverage (stats ridge)  [no OOD data] | 0.765 [0.751,0.779] | 0.760 |
| Mahalanobis tied (stats ridge) [no OOD data] | 0.798 [0.786,0.810] | 0.632 |
| noul head LS (100 OOS train) | 0.687 [0.665,0.708] | 0.972 |
| noul head LR (CE, 100 OOS train) | 0.742 [0.726,0.758] | 0.817 |

**tiny_ssm**

| score | AUROC [CI 95%] | FPR@95TPR |
|---|---|---|
| MSP  (LR, CE) | 0.912 [0.904,0.921] | 0.274 |
| max-logit (LR) | 0.914 [0.906,0.923] | 0.274 |
| energy (LR) | 0.912 [0.904,0.921] | 0.265 |
| MSP + T(LOO) (LS) | 0.799 [0.787,0.810] | 0.398 |
| max-logit (LS) | 0.750 [0.736,0.762] | 0.432 |
| MSP + T(holdout) (LS, rho=holdout) | 0.911 [0.901,0.920] | 0.305 |
| leverage (stats ridge)  [no OOD data] | 0.858 [0.846,0.872] | 0.574 |
| Mahalanobis tied (stats ridge) [no OOD data] | 0.892 [0.884,0.903] | 0.321 |
| noul head LS (100 OOS train) | 0.828 [0.811,0.846] | 0.910 |
| MSP  (B1 end-to-end) | 0.897 [0.887,0.906] | 0.314 |
| energy (B1 end-to-end) | 0.912 [0.904,0.921] | 0.254 |
| noul head B1 end-to-end (100 OOS train) | 0.831 [0.812,0.848] | 0.918 |
| noul head LR (CE, 100 OOS train) | 0.853 [0.837,0.869] | 0.760 |

### Testa noul LS: AUROC vs numero di esempi OOS visti in training (media su 5 sottoinsiemi)

| n OOS | lsa | spacy | tiny_ssm |
|---|---|---|---|
| 5 | 0.589 | 0.548 | 0.825 |
| 10 | 0.615 | 0.614 | 0.824 |
| 25 | 0.656 | 0.633 | 0.825 |
| 50 | 0.679 | 0.646 | 0.826 |
| 100 | 0.697 | 0.687 | 0.828 |

### Incrementale / unlearning (max |dW| vs refit completo, choice)

| encoder | add 10 classi | rimozione 1/7 righe |
|---|---|---|
| lsa | 0.0e+00 | 0.0e+00 |
| spacy | 0.0e+00 | 0.0e+00 |
| tiny_ssm | 0.0e+00 | 3.6e-15 |

### Percorso libreria (train(use_exact_solver=True, loo_calibration=True) + model.predict), tiny_ssm

{"test_acc": 0.7504, "ece_msp": 0.1443, "ece_score": 0.1443, "auroc_msp": 0.7994, "auroc_leverage": 0.8582, "auroc_maha": 0.8923, "auroc_noul_head": 0.8222}
