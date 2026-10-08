"""Shared training loop (used by the MLP, the CNN and the quantum model) plus metrics."""
import copy
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    """Seed every random number generator we use, and ask PyTorch for deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def class_weight(y):
    """pos_weight for BCE loss: makes the 1214 normal and 3494 pneumonia images count equally overall."""
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def train_model(model, X_train, y_train, X_val, y_val, *, epochs=50, lr=1e-2,
                batch_size=64, patience=5, seed=0, verbose=False):
    """Train a model that outputs one logit per example. Returns (best model, history).

    Early stopping: after each epoch we compute the validation loss. We keep a copy of the
    weights with the lowest validation loss so far, and stop if it hasn't improved for
    `patience` epochs. The returned model has those best weights, not the last ones.
    """
    set_seed(seed)
    X_train, X_val = torch.as_tensor(X_train), torch.as_tensor(X_val)
    y_train = torch.as_tensor(y_train, dtype=torch.float32)
    y_val = torch.as_tensor(y_val, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight(y_train.numpy()))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_loss, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val).squeeze(-1), y_val).item()
        history["train_loss"].append(total / len(X_train))
        history["val_loss"].append(val_loss)
        if verbose:
            print(f"  epoch {epoch + 1:3d}  train {history['train_loss'][-1]:.4f}  val {val_loss:.4f}")

        if val_loss < best_loss - 1e-4:
            best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = int(np.argmin(history["val_loss"])) + 1
    return model, history


def predict_proba(model, X):
    """Probability of pneumonia for each example."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X)).squeeze(-1)).numpy()


def compute_metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    # A constant prediction has no ranking information; AUC is defined as 0.5 in that case.
    auc = roc_auc_score(y_true, prob) if np.unique(prob).size > 1 else 0.5
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": auc,
    }


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
