"""Run every model over several seeds and feature sizes; save results to results/.

    python -m src.benchmark              # full run: 5 seeds x k in {4, 8}, then all figures
    python -m src.benchmark --seeds 1    # quick run

Outputs
    results/metrics.csv          one row per (model, k, seed): test metrics, params, train time
    results/summary.csv          mean and std over seeds for each (model, k)
    results/predictions_seed0.csv  test-set probabilities from seed 0 (for ROC curves)
    results/histories_seed0.json   loss curves from seed 0 (for training-curve plots)

What a "seed" changes: weight initialisation and mini-batch order. The train/val/test split is the
fixed official MedMNIST split, so seed-to-seed spread measures training randomness only - not how
much the score would change on a different test set.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.classical import (MLP, SmallCNN, logistic_regression, logistic_regression_trig,
                           majority_baseline)
from src.data import load_data
from src.quantum import HybridQNN
from src.train import compute_metrics, n_params, predict_proba, set_seed, train_model

RESULTS = Path("results")
METRICS = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]


def run_sklearn(model, F, y):
    start = time.time()
    model.fit(F["train"], y["train"])
    seconds = time.time() - start
    final = model[-1] if hasattr(model, "steps") else model  # the classifier at the end of a pipeline
    params = final.coef_.size + final.intercept_.size if hasattr(final, "coef_") else 0
    probs = {split: model.predict_proba(F[split])[:, 1] for split in ("val", "test")}
    return probs, params, seconds, None


def run_torch(model, X, y, seed, lr):
    start = time.time()
    model, history = train_model(model, X["train"], y["train"], X["val"], y["val"], lr=lr, seed=seed)
    seconds = time.time() - start
    probs = {split: predict_proba(model, X[split]) for split in ("val", "test")}
    return probs, n_params(model), seconds, history


def main(seeds=(0, 1, 2, 3, 4), ks=(4, 8)):
    RESULTS.mkdir(exist_ok=True)
    data = {k: load_data(n_components=k) for k in ks}
    y = data[ks[0]].labels
    rows, predictions, histories = [], {"y_true": y["test"]}, {}

    for seed in seeds:
        jobs = []
        for k in ks:
            F = data[k].features
            jobs += [
                ("majority", k, lambda F=F: run_sklearn(majority_baseline(), F, y)),
                ("logreg", k, lambda F=F: run_sklearn(logistic_regression(seed), F, y)),
                ("logreg_trig", k, lambda F=F: run_sklearn(logistic_regression_trig(seed), F, y)),
                ("mlp", k, lambda F=F, k=k: run_torch(MLP(k), F, y, seed, lr=1e-2)),
                ("qnn", k, lambda F=F, k=k: run_torch(HybridQNN(k), F, y, seed, lr=0.02)),
                ("qnn_no_entangle", k,
                 lambda F=F, k=k: run_torch(HybridQNN(k, entangle=False), F, y, seed, lr=0.02)),
            ]
        jobs.append(("cnn", "full", lambda: run_torch(SmallCNN(), data[ks[0]].images, y, seed, lr=1e-3)))

        for name, k, job in jobs:
            set_seed(seed)  # model weights are created inside job(), after this
            probs, params, seconds, history = job()
            prob = probs["test"]
            m = compute_metrics(y["test"], prob)
            val_auc = compute_metrics(y["val"], probs["val"])["roc_auc"]
            rows.append({"model": name, "k": str(k), "seed": seed, **m, "val_roc_auc": val_auc,
                         "params": params,
                         "train_seconds": round(seconds, 2),
                         "best_epoch": history["best_epoch"] if history else None})
            print(f"seed {seed}  {name:16s} k={str(k):4s}  auc {m['roc_auc']:.3f}  "
                  f"bal_acc {m['balanced_accuracy']:.3f}  ({seconds:.0f}s)", flush=True)
            if seed == seeds[0]:
                predictions[f"{name}@{k}"] = prob
                if history:
                    histories[f"{name}@{k}"] = history

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "metrics.csv", index=False)
    summary = (df.groupby(["model", "k"], sort=False)
                 .agg(**{f"{m}_{s}": (m, s) for m in METRICS for s in ("mean", "std")},
                      val_roc_auc_mean=("val_roc_auc", "mean"), params=("params", "first"),
                      train_seconds=("train_seconds", "mean"),
                      n_seeds=("seed", "count"))
                 .reset_index())
    summary.to_csv(RESULTS / "summary.csv", index=False, float_format="%.4f")
    pd.DataFrame(predictions).to_csv(RESULTS / "predictions_seed0.csv", index=False)
    (RESULTS / "histories_seed0.json").write_text(json.dumps(histories, indent=1))

    print("\n" + summary[["model", "k", "params", "val_roc_auc_mean", "roc_auc_mean", "roc_auc_std",
                          "balanced_accuracy_mean", "balanced_accuracy_std", "train_seconds"]]
          .to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds (0, 1, ..., n-1)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    main(seeds=tuple(range(args.seeds)))
    if not args.no_plots:
        from src import plots
        plots.main()
