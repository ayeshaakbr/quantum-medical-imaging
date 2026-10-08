import numpy as np
import pytest

from src.data import SPLITS, load_data


@pytest.fixture(scope="module")
def data():
    return load_data(n_components=4)


def test_shapes(data):
    for split in SPLITS:
        n = len(data.labels[split])
        assert data.images[split].shape == (n, 28, 28)
        assert data.features[split].shape == (n, 4)


def test_known_split_sizes(data):
    assert {s: len(data.labels[s]) for s in SPLITS} == {"train": 4708, "val": 524, "test": 624}


def test_labels_are_binary(data):
    for split in SPLITS:
        assert set(np.unique(data.labels[split])) == {0, 1}


def test_value_ranges(data):
    for split in SPLITS:
        assert data.images[split].min() >= 0.0 and data.images[split].max() <= 1.0
        assert data.features[split].min() >= 0.0 and data.features[split].max() <= np.pi + 1e-6


def test_pca_fitted_on_train_only(data):
    # PCA stores the mean it subtracted; it must be the training mean, not a mean over all splits.
    train_mean = data.images["train"].reshape(len(data.images["train"]), -1).mean(axis=0)
    np.testing.assert_allclose(data.pca.mean_, train_mean, atol=1e-5)


def test_train_features_span_full_angle_range(data):
    # The scaler is fit on train, so each train feature should reach exactly 0 and pi.
    np.testing.assert_allclose(data.features["train"].min(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(data.features["train"].max(axis=0), np.pi, atol=1e-5)


def test_deterministic():
    a, b = load_data(n_components=4), load_data(n_components=4)
    for split in SPLITS:
        np.testing.assert_array_equal(a.features[split], b.features[split])
