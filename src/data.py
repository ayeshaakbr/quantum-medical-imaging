"""Load PneumoniaMNIST and turn each image into a few numbers a small quantum circuit can use.

Pipeline (all statistics are learned from the TRAINING split only):
    28x28 image -> 784 pixels in [0, 1] -> PCA to k features -> rescale to [0, pi] (rotation angles)
"""
from dataclasses import dataclass

import numpy as np
from medmnist import PneumoniaMNIST
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

DATA_ROOT = "data"
SPLITS = ("train", "val", "test")


def load_split(split, root=DATA_ROOT):
    """Return images as float32 in [0, 1], shape (n, 28, 28), and labels as int64, shape (n,).

    Label 0 = normal, 1 = pneumonia.
    """
    ds = PneumoniaMNIST(split=split, download=True, root=root)
    images = ds.imgs.astype(np.float32) / 255.0
    labels = ds.labels.ravel().astype(np.int64)
    return images, labels


@dataclass
class Dataset:
    """Everything the models need. `images` is for the CNN; `features` is for everything else."""
    images: dict      # split -> (n, 28, 28) float32 in [0, 1]
    features: dict    # split -> (n, k) float32 in [0, pi]
    labels: dict      # split -> (n,) int64
    pca: PCA
    scaler: MinMaxScaler


def load_data(n_components=4, root=DATA_ROOT, seed=0):
    images, labels = {}, {}
    for split in SPLITS:
        images[split], labels[split] = load_split(split, root)

    flat = {split: images[split].reshape(len(images[split]), -1) for split in SPLITS}

    # Fit on train only, so no information about val/test leaks into the features.
    # svd_solver="full" is exact (no randomness), so the result is the same on every run.
    pca = PCA(n_components=n_components, svd_solver="full", random_state=seed).fit(flat["train"])
    scaler = MinMaxScaler(feature_range=(0.0, np.pi)).fit(pca.transform(flat["train"]))

    features = {}
    for split in SPLITS:
        f = scaler.transform(pca.transform(flat[split]))
        # Val/test points can fall slightly outside the training range; clip them back to [0, pi].
        features[split] = np.clip(f, 0.0, np.pi).astype(np.float32)

    return Dataset(images=images, features=features, labels=labels, pca=pca, scaler=scaler)


if __name__ == "__main__":
    for k in (2, 4, 8, 16):
        d = load_data(n_components=k)
        print(f"k={k:2d}  variance explained by PCA = {d.pca.explained_variance_ratio_.sum():.1%}")
