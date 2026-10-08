"""Make every figure in results/figures/ from the files written by src/benchmark.py.

    python -m src.plots
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve

from src.data import load_split

RESULTS = Path("results")
FIGURES = RESULTS / "figures"

# Colour encodes how much of the image a model sees. The same colour means the same thing in every figure.
INPUT_COLOURS = {"4": "#2a78d6", "8": "#eb6834", "full": "#1baf7a"}
INPUT_LABELS = {"4": "4 PCA features", "8": "8 PCA features", "full": "full 28x28 image"}
MODEL_COLOURS = {"logreg": "#2a78d6", "mlp": "#eb6834", "qnn": "#1baf7a", "cnn": "#eda100",
                 "qnn_no_entangle": "#e87ba4"}
MODEL_LABELS = {"logreg": "Logistic regression", "mlp": "MLP", "qnn": "Hybrid QNN",
                "qnn_no_entangle": "QNN, no entanglement", "logreg_trig": "Log. reg. on cos/sin",
                "cnn": "Small CNN", "majority": "Majority"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def performance(summary):
    """Mean +/- std over seeds, one row per model, one dot per input size."""
    order = ["cnn", "qnn", "qnn_no_entangle", "logreg_trig", "mlp", "logreg"]
    metrics = [("roc_auc", "ROC-AUC"), ("balanced_accuracy", "Balanced accuracy")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    offsets = {"4": 0.15, "8": -0.15, "full": 0.0}

    for ax, (metric, title) in zip(axes, metrics):
        for i, model in enumerate(order):
            for _, row in summary[summary.model == model].iterrows():
                ax.errorbar(row[f"{metric}_mean"], i + offsets[row.k], xerr=row[f"{metric}_std"],
                            fmt="o", ms=6, color=INPUT_COLOURS[row.k], ecolor=INPUT_COLOURS[row.k],
                            elinewidth=2, capsize=0, mec="white", mew=1)
        # Zoom in on the models. (The "always pneumonia" baseline sits at 0.5 and is noted in the caption.)
        shown = summary[summary.model.isin(order)]
        lo = (shown[f"{metric}_mean"] - shown[f"{metric}_std"]).min()
        hi = (shown[f"{metric}_mean"] + shown[f"{metric}_std"]).max()
        ax.set_xlim(lo - 0.01, hi + 0.01)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), [MODEL_LABELS[m] for m in order])
    axes[0].set_ylim(-0.6, len(order) - 0.1)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=INPUT_LABELS[k])
               for k, c in INPUT_COLOURS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.07))
    fig.text(0.01, -0.06, f"Test set (n=624). Dots: mean over {int(summary.n_seeds.max())} training seeds; "
             "bars: +/- 1 std across seeds (often smaller than the dot).\n"
             "Seeds vary only training randomness; test-set sampling adds roughly +/-0.02 AUC "
             "(bootstrap 95% CI). Always-'pneumonia' baseline: 0.50 on both metrics.",
             ha="left", fontsize=8, color=INK_2)
    fig.savefig(FIGURES / "performance.png")
    plt.close(fig)


def roc_curves(predictions, k="4"):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls="--")
    for model in ["logreg", "mlp", "qnn", "cnn"]:
        col = f"{model}@{'full' if model == 'cnn' else k}"
        fpr, tpr, _ = roc_curve(predictions.y_true, predictions[col])
        ax.plot(fpr, tpr, lw=2, color=MODEL_COLOURS[model], label=MODEL_LABELS[model])
    ax.set_xlabel("False positive rate (healthy flagged as pneumonia)")
    ax.set_ylabel("True positive rate (pneumonia caught)")
    ax.set_title(f"ROC curves, seed 0 ({k} PCA features; CNN on full image)", loc="left", fontsize=10)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    fig.savefig(FIGURES / "roc_curves.png")
    plt.close(fig)


def training_curves(histories, k="4"):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for model in ["mlp", "qnn"]:
        h = histories[f"{model}@{k}"]
        epochs = np.arange(1, len(h["val_loss"]) + 1)
        colour = MODEL_COLOURS[model]
        ax.plot(epochs, h["train_loss"], lw=2, ls=":", color=colour)
        ax.plot(epochs, h["val_loss"], lw=2, color=colour, label=MODEL_LABELS[model])
        best = h["best_epoch"]
        ax.plot(best, h["val_loss"][best - 1], "o", ms=8, color=colour, mec="white", mew=1.5)
    ax.plot([], [], color=INK_2, lw=2, label="validation")
    ax.plot([], [], color=INK_2, lw=2, ls=":", label="train")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted BCE loss")
    ax.set_title(f"Training curves, seed 0, {k} features (dot = best epoch, weights restored)",
                 loc="left", fontsize=10)
    ax.legend(ncol=2)
    fig.savefig(FIGURES / "training_curves.png")
    plt.close(fig)


def pca_variance(images, max_k=32):
    flat = images.reshape(len(images), -1)
    cumulative = np.cumsum(PCA(n_components=max_k, svd_solver="full").fit(flat).explained_variance_ratio_)
    ks = np.arange(1, max_k + 1)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(ks, cumulative * 100, lw=2, color=INPUT_COLOURS["4"])
    for k in (4, 8):
        ax.plot(k, cumulative[k - 1] * 100, "o", ms=8, color=INPUT_COLOURS[str(k)], mec="white", mew=1.5)
        ax.annotate(f"k={k}: {cumulative[k - 1]:.0%}", (k, cumulative[k - 1] * 100),
                    xytext=(8, -14), textcoords="offset points", color=INK, fontsize=9)
    ax.set_xlabel("Number of PCA components (= qubits)")
    ax.set_ylabel("Pixel variance kept (%)")
    ax.set_ylim(0, 100)
    ax.set_title("How much of the image survives PCA (fit on training set)", loc="left", fontsize=10)
    fig.savefig(FIGURES / "pca_variance.png")
    plt.close(fig)


def sample_images(images, labels, n=5):
    fig, axes = plt.subplots(2, n, figsize=(n * 1.4, 3.2))
    for row, (label, name) in enumerate([(0, "Normal"), (1, "Pneumonia")]):
        idx = np.flatnonzero(labels == label)[:n]
        for ax, i in zip(axes[row], idx):
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[row, 0].set_ylabel(name, fontsize=10)
    fig.suptitle("PneumoniaMNIST training images (28x28)", x=0.02, ha="left", fontsize=10)
    fig.savefig(FIGURES / "samples.png")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RESULTS / "summary.csv", dtype={"k": str})
    predictions = pd.read_csv(RESULTS / "predictions_seed0.csv")
    histories = json.loads((RESULTS / "histories_seed0.json").read_text())
    images, labels = load_split("train")

    performance(summary)
    roc_curves(predictions)
    training_curves(histories)
    pca_variance(images)
    sample_images(images, labels)
    print(f"figures written to {FIGURES}/")


if __name__ == "__main__":
    main()
