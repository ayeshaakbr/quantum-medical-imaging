"""Hybrid quantum-classical model: a variational quantum circuit followed by a tiny linear layer.

    k features in [0, pi]
        -> encode: one RY rotation per qubit                        (fixed, no parameters)
        -> L variational layers: trainable rotations + CNOT ring    (3 * k * L parameters)
        -> measure <Z> on every qubit: k numbers in [-1, 1]
        -> Linear(k, 1): one logit                                  (k + 1 parameters)

The circuit runs on PennyLane's `default.qubit` simulator, which computes the exact quantum state
with ordinary linear algebra on the CPU. `diff_method="backprop"` lets PyTorch differentiate through
that simulation like any other layer. (On real hardware you would use the parameter-shift rule,
which costs 2 extra circuit runs per parameter per gradient.)
"""
import pennylane as qml
import torch
from torch import nn


def make_circuit(n_qubits, entangle=True):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        if entangle:
            # Each layer: an arbitrary single-qubit rotation Rot(a, b, c) on every qubit,
            # then CNOTs in a ring (0->1, 1->2, ..., k-1->0) so the qubits influence each other.
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        else:
            # Ablation: same trainable rotations, but no CNOTs. Every qubit is then independent,
            # so the circuit can't model any interaction between features.
            for layer in weights:
                for wire in range(n_qubits):
                    qml.Rot(*layer[wire], wires=wire)
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit


class HybridQNN(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, entangle=True):
        super().__init__()
        circuit = make_circuit(n_qubits, entangle)
        self.quantum = qml.qnn.TorchLayer(circuit, weight_shapes={"weights": (n_layers, n_qubits, 3)})
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x):
        return self.head(self.quantum(x))


if __name__ == "__main__":
    import time

    from src.data import load_data
    from src.train import compute_metrics, n_params, predict_proba, set_seed, train_model

    set_seed(0)
    model = HybridQNN(n_qubits=4, n_layers=2)
    print(qml.draw(model.quantum.qnode, decimals=None)(torch.zeros(4), model.quantum.weights))
    print(f"\ntrainable parameters: {n_params(model)} "
          f"(circuit {model.quantum.weights.numel()}, linear head {n_params(model.head)})\n")

    d = load_data(n_components=4)
    F, y = d.features, d.labels
    start = time.time()
    model, h = train_model(model, F["train"], y["train"], F["val"], y["val"], lr=0.02, verbose=True)
    elapsed = time.time() - start
    m = compute_metrics(y["test"], predict_proba(model, F["test"]))
    print(f"\ntrained in {elapsed:.0f}s, best epoch {h['best_epoch']}")
    print("test: " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()))
