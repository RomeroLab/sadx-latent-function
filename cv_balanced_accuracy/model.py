"""Standalone CORAL training implementation used for the retrospective CV analysis.

The embedding lookup is exactly the dense one-hot first layer evaluated sparsely.
Library identity selects output thresholds; it is not an extra sequence feature.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i for i, aa in enumerate(AA_ALPHABET)}
PARENT_TO_INT = {"1VH": 0, "2L": 1, "3VRL": 2}
CATEGORY_TO_INT = {"N": 0, "L": 1, "P": 2, "H": 3}
NUM_CLASSES = 4


@dataclass(frozen=True)
class ModelSpec:
    name: str
    use_sequence: bool
    use_dca: bool
    use_library_bias: bool


MODEL_SPECS = {
    "full": ModelSpec("full", True, True, True),
    "no_dca": ModelSpec("no_dca", True, False, True),
    "no_library_bias": ModelSpec("no_library_bias", True, True, False),
    "sequence_only": ModelSpec("sequence_only", True, False, False),
    "dca_only": ModelSpec("dca_only", False, True, True),
}

def encode_sequences(sequences: pd.Series) -> np.ndarray:
    lengths = sequences.str.len().unique()
    if len(lengths) != 1:
        raise ValueError(f"Expected one sequence length, found {lengths.tolist()}")
    unknown = sorted(set("".join(sequences.tolist())) - set(AA_ALPHABET))
    if unknown:
        raise ValueError(f"Unsupported amino acids: {unknown}")
    return np.asarray([[AA_TO_INT[aa] for aa in seq] for seq in sequences], dtype=np.int64)


def make_folds(df: pd.DataFrame, n_splits: int, split_seed: int) -> np.ndarray:
    # Joint labels preserve parent/class composition as far as rare cells allow.
    # Identical amino-acid sequences are grouped to prevent exact-sequence leakage.
    strata = (df["parent"].astype(str) + "__" + df["category"].astype(str)).to_numpy()
    groups = df["sequence_aa_trim"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    fold = np.full(len(df), -1, dtype=np.int64)
    dummy = np.zeros(len(df), dtype=np.float32)
    for fold_id, (_, heldout_idx) in enumerate(splitter.split(dummy, strata, groups)):
        fold[heldout_idx] = fold_id
    if np.any(fold < 0):
        raise RuntimeError("Some rows did not receive a fold assignment")
    for _, duplicate_rows in df.groupby("sequence_aa_trim").groups.items():
        if len(set(fold[np.asarray(list(duplicate_rows), dtype=int)])) != 1:
            raise RuntimeError("Identical sequences were assigned to different folds")
    return fold


class EfficientCoralMLP(torch.nn.Module):
    """The 6eeae50e MLP, with an exact sparse implementation of layer one."""

    def __init__(
        self,
        sequence_length: int,
        spec: ModelSpec,
        hidden: tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.spec = spec
        self.dropout_p = dropout
        h1, h2 = hidden
        input_size = (sequence_length * len(AA_ALPHABET) if spec.use_sequence else 0) + (1 if spec.use_dca else 0)
        if input_size == 0:
            raise ValueError("A neural ablation must retain at least one input feature")

        if spec.use_sequence:
            self.sequence_embedding = torch.nn.Embedding(sequence_length * len(AA_ALPHABET), h1)
        else:
            self.sequence_embedding = None
        if spec.use_dca:
            self.dca_weight = torch.nn.Parameter(torch.empty(1, h1))
        else:
            self.register_parameter("dca_weight", None)
        self.first_bias = torch.nn.Parameter(torch.zeros(h1))
        self.second = torch.nn.Linear(h1, h2)
        self.coral_weight = torch.nn.Linear(h2, 1, bias=False)
        self.coral_bias = torch.nn.Parameter(
            torch.arange(NUM_CLASSES - 1, 0, -1, dtype=torch.float32)
            .repeat(3 if spec.use_library_bias else 1, 1)
            / (NUM_CLASSES - 1)
        )
        self.reset_parameters(input_size)

    def reset_parameters(self, input_size: int) -> None:
        h1 = self.first_bias.numel()
        bound1 = math.sqrt(6.0) / math.sqrt(input_size + h1)
        if self.sequence_embedding is not None:
            torch.nn.init.uniform_(self.sequence_embedding.weight, -bound1, bound1)
        if self.dca_weight is not None:
            torch.nn.init.uniform_(self.dca_weight, -bound1, bound1)
        torch.nn.init.zeros_(self.first_bias)
        bound2 = math.sqrt(6.0) / math.sqrt(self.second.in_features + self.second.out_features)
        torch.nn.init.uniform_(self.second.weight, -bound2, bound2)
        torch.nn.init.zeros_(self.second.bias)
        bound3 = math.sqrt(6.0) / math.sqrt(self.coral_weight.in_features + 1)
        torch.nn.init.uniform_(self.coral_weight.weight, -bound3, bound3)

    def forward(self, aa: torch.Tensor, dca: torch.Tensor, library: torch.Tensor) -> torch.Tensor:
        h = self.first_bias.unsqueeze(0).expand(aa.shape[0], -1)
        if self.sequence_embedding is not None:
            positions = torch.arange(self.sequence_length, device=aa.device).unsqueeze(0)
            lookup = positions * len(AA_ALPHABET) + aa
            embedded = self.sequence_embedding(lookup)
            if self.training and self.dropout_p:
                keep = torch.rand(aa.shape, device=aa.device) >= self.dropout_p
                embedded = embedded * keep.unsqueeze(-1) / (1.0 - self.dropout_p)
            h = h + embedded.sum(dim=1)
        if self.dca_weight is not None:
            dca_input = dca
            if self.training and self.dropout_p:
                keep = (torch.rand_like(dca_input) >= self.dropout_p).to(dca_input.dtype)
                dca_input = dca_input * keep / (1.0 - self.dropout_p)
            h = h + dca_input * self.dca_weight
        h = torch.relu(h)
        h = torch.nn.functional.dropout(h, p=self.dropout_p, training=self.training)
        h = torch.relu(self.second(h))
        shared_score = self.coral_weight(h)
        bias_rows = library if self.spec.use_library_bias else torch.zeros_like(library)
        return shared_score + self.coral_bias[bias_rows]


def coral_levels(y: torch.Tensor) -> torch.Tensor:
    thresholds = torch.arange(NUM_CLASSES - 1, device=y.device).unsqueeze(0)
    return (thresholds < y.unsqueeze(1)).to(torch.float32)


def coral_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, coral_levels(y), reduction="none"
    ).sum(dim=1).mean()


def logits_to_class_probs(logits: torch.Tensor) -> torch.Tensor:
    cumulative = torch.sigmoid(logits)
    padded = torch.cat(
        [
            torch.ones((len(logits), 1), device=logits.device),
            cumulative,
            torch.zeros((len(logits), 1), device=logits.device),
        ],
        dim=1,
    )
    probs = -torch.diff(padded, dim=1)
    # Numerical guard only; valid CORAL thresholds should already be monotonic.
    probs = torch.clamp(probs, min=0.0)
    return probs / probs.sum(dim=1, keepdim=True)


def train_predict(
    aa: torch.Tensor,
    dca: torch.Tensor,
    library: torch.Tensor,
    y: torch.Tensor,
    train_idx: np.ndarray,
    heldout_idx: np.ndarray,
    spec: ModelSpec,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    hidden: tuple[int, int],
    dropout: float,
) -> tuple[np.ndarray, list[float]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = EfficientCoralMLP(aa.shape[1], spec, hidden, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    train_tensor = torch.as_tensor(train_idx, dtype=torch.int64)
    losses: list[float] = []

    for _ in range(epochs):
        model.train()
        permutation = train_tensor[torch.randperm(len(train_tensor), generator=generator)]
        running = 0.0
        seen = 0
        for start in range(0, len(permutation), batch_size):
            idx = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(aa[idx], dca[idx], library[idx])
            loss = coral_loss(logits, y[idx])
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(idx)
            seen += len(idx)
        losses.append(running / seen)

    model.eval()
    with torch.no_grad():
        idx = torch.as_tensor(heldout_idx, dtype=torch.int64)
        probs = logits_to_class_probs(model(aa[idx], dca[idx], library[idx])).cpu().numpy()
    return probs, losses


def empirical_library_prior(df: pd.DataFrame, train_idx: np.ndarray, heldout_idx: np.ndarray) -> np.ndarray:
    train = df.iloc[train_idx]
    heldout = df.iloc[heldout_idx]
    global_counts = train["y_true"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).to_numpy(float)
    global_probs = global_counts / global_counts.sum()
    result = np.zeros((len(heldout), NUM_CLASSES), dtype=float)
    for i, parent in enumerate(heldout["parent"]):
        counts = (
            train.loc[train["parent"] == parent, "y_true"]
            .value_counts()
            .reindex(range(NUM_CLASSES), fill_value=0)
            .to_numpy(float)
        )
        result[i] = counts / counts.sum() if counts.sum() else global_probs
    return result


def predictions_frame(
    df: pd.DataFrame,
    heldout_idx: np.ndarray,
    probs: np.ndarray,
    model_name: str,
    seed: int,
    fold: str,
    evaluation: str,
) -> pd.DataFrame:
    rows = df.iloc[heldout_idx][
        ["row_index", "parent", "category", "y_true", "sequence_aa_trim", "dca_score"]
    ].copy()
    rows["model"] = model_name
    rows["seed"] = seed
    rows["fold"] = fold
    rows["evaluation"] = evaluation
    for cls, name in enumerate(["N", "L", "P", "H"]):
        rows[f"prob_{name}"] = probs[:, cls]
    rows["predicted_class"] = probs.argmax(axis=1)
    rows["expected_class"] = probs @ np.arange(NUM_CLASSES)
    return rows


