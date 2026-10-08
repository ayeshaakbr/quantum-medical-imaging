"""Classical baselines. Each is deliberately small so the comparison with the quantum model is fair."""
import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from torch import nn


def majority_baseline():
    """Always predicts the most common training class (pneumonia). Any real model must beat this."""
    return DummyClassifier(strategy="most_frequent")


def logistic_regression(seed=0):
    """A linear model on the PCA features. class_weight='balanced' handles the 3:1 imbalance."""
    return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)


def trig_features(x):
    """[cos(x), sin(x)]: the functions a single RY-encoded qubit can produce from its input angle."""
    return np.hstack([np.cos(x), np.sin(x)])


def logistic_regression_trig(seed=0):
    """Control for the quantum model: a linear model on the same cos/sin encoding, with no qubits.

    If this matches the QNN, the QNN's performance comes from the encoding, not from anything quantum.
    """
    return make_pipeline(FunctionTransformer(trig_features), logistic_regression(seed))


class MLP(nn.Module):
    """k PCA features -> hidden layer -> 1 logit. Sees exactly the same input as the quantum model."""

    def __init__(self, n_features, hidden=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """Works on the full 28x28 image, so it is NOT limited by PCA. Expected to be the strongest model."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 8 x 14 x 14
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16 x 7 x 7
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 1),
        )

    def forward(self, x):
        if x.dim() == 3:          # (n, 28, 28) -> (n, 1, 28, 28): add the single grey channel
            x = x.unsqueeze(1)
        return self.net(x)


if __name__ == "__main__":
    # Quick check: train every classical baseline once with k=4 and print test metrics.
    from src.data import load_data
    from src.train import compute_metrics, n_params, predict_proba, set_seed, train_model

    set_seed(0)
    d = load_data(n_components=4)
    F, I, y = d.features, d.images, d.labels

    rows = []
    for name, sk in [("majority", majority_baseline()), ("logreg", logistic_regression())]:
        sk.fit(F["train"], y["train"])
        params = 0 if name == "majority" else sk.coef_.size + sk.intercept_.size
        rows.append((name, params, compute_metrics(y["test"], sk.predict_proba(F["test"])[:, 1])))

    mlp, h = train_model(MLP(4), F["train"], y["train"], F["val"], y["val"])
    rows.append((f"mlp (best epoch {h['best_epoch']})", n_params(mlp),
                 compute_metrics(y["test"], predict_proba(mlp, F["test"]))))

    cnn, h = train_model(SmallCNN(), I["train"], y["train"], I["val"], y["val"], lr=1e-3)
    rows.append((f"cnn (best epoch {h['best_epoch']})", n_params(cnn),
                 compute_metrics(y["test"], predict_proba(cnn, I["test"]))))

    print(f"{'model':24s} {'params':>7s} {'acc':>6s} {'bal_acc':>8s} {'f1':>6s} {'auc':>6s}")
    for name, p, m in rows:
        print(f"{name:24s} {p:7d} {m['accuracy']:6.3f} {m['balanced_accuracy']:8.3f} "
              f"{m['f1']:6.3f} {m['roc_auc']:6.3f}")
