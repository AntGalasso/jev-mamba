"""Legge results/clinc150_*.json e stampa le tabelle in Markdown (usate in docs/RESULTS.md)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main(res_dir: str = "results") -> None:
    files = {n: json.loads((Path(res_dir) / f"clinc150_{n}.json").read_text()) for n in ("lsa", "spacy", "tiny_ssm") if (Path(res_dir) / f"clinc150_{n}.json").exists()}
    names = list(files)

    print("### Setup\n")
    print("| encoder | D | LS fit closed-form (s) | LR L-BFGS 1 fit (s) | LR grid 4 reg (s) | SGD probe (s) |\n|---|---|---|---|---|---|")
    for n, r in files.items():
        print(f"| {n} | {r['D']} | {f(r['fit_time_closed_form_s'],2)} | {f(r['fit_time_lr_one_s'],2)} | {f(r['fit_time_lr_grid4_s'],1)} | {f(r['sgd_probe']['time_s'],1)} |")

    print("\n### Choice: accuratezza e calibrazione (test in-scope, 4500)\n")
    for n, r in files.items():
        print(f"**{n}** (SGD probe acc {r['sgd_probe']['acc_mean']:.3f} ± {r['sgd_probe']['acc_std']:.3f}; LDA da stesse statistiche acc {r.get('lda_tied_acc')})\n")
        print("| metodo | acc | ECE | NLL | Brier |\n|---|---|---|---|---|")
        for k, m in r["choice_calibration"].items():
            e = f"{m['ece']:.3f}" + (f" [{m['ece_ci'][0]:.3f},{m['ece_ci'][1]:.3f}]" if m.get("ece_ci") else "")
            print(f"| {k} | {m['acc']:.3f} | {e} | {m['nll']:.3f} | {m['brier']:.3f} |")
        print(f"\nT(LOO)={r['T_loo']:.4f}, T(val)={r['T_val']:.4f}; rho GCV={r['rho_gcv']}; rho_holdout={r.get('holdout_protocol',{}).get('rho')}\n")

    print("### Ridge sweep (acc test)\n")
    rhos = list(next(iter(files.values()))["ridge_sweep"])
    print("| rho | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for rho in rhos:
        print(f"| {rho} | " + " | ".join(f(files[n]["ridge_sweep"][rho]["test_acc"]) for n in names) + " |")

    print("\n### Selective classification (errori in-scope)\n")
    for n, r in files.items():
        print(f"**{n}**\n\n| confidenza | AUROC errori | AURC (basso=meglio) | ECE |\n|---|---|---|---|")
        for k, m in r["selective"].items():
            print(f"| {k} | {m['auroc_err']:.3f} | {m['aurc']:.4f} | {f(m['ece'])} |")
        print()

    print("### Rejection: in-scope test (4500) vs out-of-scope test (1000); positivo = OOS\n")
    for n, r in files.items():
        print(f"**{n}**\n\n| score | AUROC [CI 95%] | FPR@95TPR |\n|---|---|---|")
        for k, m in r["rejection"].items():
            print(f"| {k} | {m['auroc']:.3f} [{m['auroc_ci'][0]:.3f},{m['auroc_ci'][1]:.3f}] | {m['fpr95']:.3f} |")
        print()

    print("### Testa noul LS: AUROC vs numero di esempi OOS visti in training (media su 5 sottoinsiemi)\n")
    ms = list(next(iter(files.values()))["noul_head_auroc_vs_n_oos_train"])[1:]
    print("| n OOS | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for m in ms:
        print(f"| {m} | " + " | ".join(f"{files[n]['noul_head_auroc_vs_n_oos_train'][m][0]:.3f}" for n in names) + " |")

    print("\n### Incrementale / unlearning (max |dW| vs refit completo, choice)\n")
    print("| encoder | add 10 classi | rimozione 1/7 righe |\n|---|---|---|")
    for n, r in files.items():
        print(f"| {n} | {r['incremental_max_abs_weight_diff']:.1e} | {r['unlearning_max_abs_weight_diff']:.1e} |")
    if "tiny_ssm" in files and "library_path" in files["tiny_ssm"]:
        print("\n### Percorso libreria (train(use_exact_solver=True, loo_calibration=True) + model.predict), tiny_ssm\n")
        print(json.dumps(files["tiny_ssm"]["library_path"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
