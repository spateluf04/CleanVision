"""Train sequence models for normalized air-drawn letter trajectories.

This module loads trajectory rows from the project CSV dataset, builds a
stratified train/validation split, and trains both an LSTM baseline and a
Transformer encoder. It depends on NumPy, PyTorch, and the shared project
configuration, and it produces model checkpoints such as ``letter_model.pt``.
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_OUTPUT,
    DEFAULT_MODEL_OUTPUT_PATH,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TRAIN_EPOCHS,
    HIDDEN_SIZE,
    INPUT_POINTS,
    INPUT_SIZE,
    LABELS,
    LABEL_TO_INDEX,
    NUM_CLASSES,
    NUM_LAYERS,
    TRAINING_STATE_PATH,
    TRAIN_SPLIT_RATIO,
    TRANSFORMER_EMBED_DIM,
    TRANSFORMER_FF_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
)
from logging_utils import get_logger


logger = get_logger(__name__)


def load_model_version(training_state_path: Path) -> int:
    """Load the current model version number from training state.

    Args:
        training_state_path: Path to the persistent training state JSON file.

    Returns:
        The integer model version from state, or 0 if the file is missing or unreadable.
    """
    if not training_state_path.exists():
        return 0

    try:
        with training_state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Failed to read training state from %s for model version: %s", training_state_path, exc)
        return 0

    try:
        return int(payload.get("model_version", 0))
    except (TypeError, ValueError):
        return 0


def resolve_model_output_path(requested_output: str) -> Path:
    """Resolve a checkpoint output path relative to this script directory.

    Args:
        requested_output: Requested output filename or path from CLI args.

    Returns:
        Absolute output path anchored to the training script directory.
    """
    requested_path = Path(requested_output).expanduser()
    if requested_path.is_absolute():
        return requested_path
    return Path(__file__).resolve().parent / requested_path


class LetterTrajectoryIndex:
    """Lightweight index for trajectory CSV rows.

    The index stores file offsets and encoded labels so the training pipeline
    can stream trajectory rows instead of loading the full dataset into memory.

    Args:
        csv_path: Path to the normalized trajectory CSV file.

    Raises:
        RuntimeError: If the CSV file cannot be read.
        ValueError: If the CSV file is empty or contains no valid samples.
    """

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.row_offsets: List[int] = []
        self.labels: List[int] = []
        self.expected_feature_count = INPUT_POINTS * INPUT_SIZE

        try:
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                header_offset = handle.tell()
                header_line = handle.readline()
                if not header_line:
                    raise ValueError(f"Dataset CSV is empty: {csv_path}")

                header = next(csv.reader([header_line]))
                column_to_index = {name: idx for idx, name in enumerate(header)}
                required_columns = ["label"] + [f"p{point_idx}_{axis}" for point_idx in range(INPUT_POINTS) for axis in ("x", "y")]
                missing_columns = [column for column in required_columns if column not in column_to_index]
                if missing_columns:
                    raise ValueError(f"Dataset CSV is missing required columns: {missing_columns[:5]}")

                while True:
                    row_offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    row_idx = len(self.row_offsets) + 2
                    try:
                        values = next(csv.reader([line]))
                        label = (values[column_to_index["label"]] or "").strip().upper()
                        if label not in LABEL_TO_INDEX:
                            logger.warning("Skipping invalid label %r at CSV row %s.", label, row_idx)
                            continue

                        flat_values = []
                        for point_idx in range(INPUT_POINTS):
                            flat_values.append(float(values[column_to_index[f"p{point_idx}_x"]]))
                            flat_values.append(float(values[column_to_index[f"p{point_idx}_y"]]))

                        if len(flat_values) != self.expected_feature_count:
                            logger.warning("Skipping incomplete row %s in %s.", row_idx, csv_path)
                            continue

                        trajectory = np.asarray(flat_values, dtype=np.float32)
                        if not np.isfinite(trajectory).all():
                            logger.warning("Skipping non-finite trajectory values at row %s.", row_idx)
                            continue

                        self.row_offsets.append(row_offset)
                        self.labels.append(LABEL_TO_INDEX[label])
                    except (IndexError, KeyError) as exc:
                        logger.warning("Skipping row %s due to missing column data %s.", row_idx, exc)
                    except (TypeError, ValueError) as exc:
                        logger.warning("Skipping corrupt row %s in %s: %s", row_idx, csv_path, exc)
        except OSError as exc:
            raise RuntimeError(f"Failed to read dataset CSV {csv_path}: {exc}") from exc

        if not self.row_offsets:
            raise ValueError(f"No samples found in dataset: {csv_path}")

    def __len__(self) -> int:
        """Return the number of indexed rows."""
        return len(self.row_offsets)


class LetterTrajectoryIterableDataset(IterableDataset):
    """Stream normalized trajectory rows from disk for model training.

    Args:
        csv_path: Source dataset CSV path.
        row_offsets: File offsets corresponding to rows included in this split.
    """

    def __init__(self, csv_path: Path, row_offsets: Sequence[int]):
        self.csv_path = Path(csv_path)
        self.row_offsets = list(row_offsets)

    def __len__(self) -> int:
        """Return the number of rows in the iterable split."""
        return len(self.row_offsets)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        """Yield normalized trajectory tensors and integer labels.

        Yields:
            Tuples of ``(trajectory_tensor, label_index)`` for each valid row.

        Raises:
            RuntimeError: If the backing CSV cannot be read while streaming.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            offsets = self.row_offsets
        else:
            per_worker = int(np.ceil(len(self.row_offsets) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.row_offsets))
            offsets = self.row_offsets[start:end]

        try:
            with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
                for row_offset in offsets:
                    handle.seek(row_offset)
                    line = handle.readline()
                    if not line:
                        continue
                    try:
                        values = next(csv.reader([line]))
                        label = (values[0] or "").strip().upper()
                        if label not in LABEL_TO_INDEX:
                            continue
                        features = np.asarray(values[1:], dtype=np.float32)
                        if features.size != INPUT_POINTS * INPUT_SIZE or not np.isfinite(features).all():
                            continue
                        trajectory = torch.from_numpy(features.reshape(INPUT_POINTS, INPUT_SIZE))
                        yield trajectory, LABEL_TO_INDEX[label]
                    except (TypeError, ValueError, IndexError):
                        continue
        except OSError as exc:
            raise RuntimeError(f"Failed to stream dataset CSV {self.csv_path}: {exc}") from exc


class LetterLSTMClassifier(nn.Module):
    """LSTM classifier for air-written letters."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the LSTM classifier.

        Args:
            x: Batched trajectory tensor of shape ``(batch, 64, 2)``.

        Returns:
            Logits for each letter class.
        """
        _, (hidden_state, _) = self.lstm(x)
        logits = self.fc(hidden_state[-1])
        return logits


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for trajectory tokens."""

    def __init__(self, embed_dim: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-np.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to an embedded sequence."""
        return x + self.pe[:, : x.size(1)]


class LetterTransformerClassifier(nn.Module):
    """Transformer encoder classifier for air-written letters."""

    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(INPUT_SIZE, TRANSFORMER_EMBED_DIM)
        self.position_encoding = PositionalEncoding(TRANSFORMER_EMBED_DIM, INPUT_POINTS)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_EMBED_DIM,
            nhead=TRANSFORMER_HEADS,
            dim_feedforward=TRANSFORMER_FF_DIM,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_LAYERS)
        self.fc = nn.Linear(TRANSFORMER_EMBED_DIM, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the Transformer classifier."""
        x = self.input_projection(x)
        x = self.position_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


def build_stratified_split(
    dataset: LetterTrajectoryIndex,
    train_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Build a stratified split of dataset row offsets.

    Args:
        dataset: Indexed dataset metadata.
        train_ratio: Fraction of samples assigned to training.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of ``(train_offsets, val_offsets)``.

    Raises:
        ValueError: If a valid train/validation split cannot be formed.
    """
    rng = random.Random(seed)
    label_to_indices = {label_idx: [] for label_idx in range(NUM_CLASSES)}

    for idx, label in enumerate(dataset.labels):
        label_to_indices[label].append(idx)

    train_indices = []
    val_indices = []

    for indices in label_to_indices.values():
        if not indices:
            continue
        rng.shuffle(indices)
        split_idx = max(1, int(len(indices) * train_ratio))
        if split_idx >= len(indices) and len(indices) > 1:
            split_idx = len(indices) - 1

        train_indices.extend(indices[:split_idx])
        val_indices.extend(indices[split_idx:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    if not train_indices or not val_indices:
        raise ValueError(
            "Train/validation split failed. Make sure the dataset has enough samples per class "
            "for an 80/20 split."
        )

    train_offsets = [dataset.row_offsets[idx] for idx in train_indices]
    val_offsets = [dataset.row_offsets[idx] for idx in val_indices]
    return train_offsets, val_offsets


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    """Compute correct predictions and batch size from logits."""
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == targets).sum().item()
    return correct, targets.size(0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> Tuple[float, float]:
    """Run one train or validation epoch.

    Args:
        model: Model being trained or evaluated.
        loader: Data loader for the current split.
        criterion: Loss function.
        optimizer: Optimizer used during training mode.
        device: Target PyTorch device.
        train: Whether to run in training mode.

    Returns:
        A tuple of ``(average_loss, average_accuracy)``.

    Raises:
        RuntimeError: If NaN or infinite loss is encountered.
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Encountered NaN/Inf loss during training. Stopping to avoid saving a broken model.")

            if train:
                loss.backward()
                optimizer.step()

            correct, batch_size = accuracy_from_logits(logits, targets)
            total_loss += loss.item() * batch_size
            total_correct += correct
            total_examples += batch_size

    avg_loss = total_loss / total_examples
    avg_accuracy = total_correct / total_examples
    return avg_loss, avg_accuracy


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    model_out: Optional[Path] = None,
    versioned_model_out: Optional[Path] = None,
) -> float:
    """Train a model and optionally save the best checkpoint.

    Args:
        model: Model instance to optimize.
        model_name: Friendly model name for logging.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Target PyTorch device.
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        model_out: Optional checkpoint output path.
        versioned_model_out: Optional versioned checkpoint output path.

    Returns:
        Best validation accuracy observed during training.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        logger.info(
            f"[{model_name}] Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            if model_out is not None:
                checkpoint = {
                    "model_type": model_name.lower(),
                    "model_state_dict": model.state_dict(),
                    "label_to_index": LABEL_TO_INDEX,
                    "input_points": INPUT_POINTS,
                    "input_size": INPUT_SIZE,
                    "num_classes": NUM_CLASSES,
                    "best_val_accuracy": best_val_accuracy,
                }
                if model_name.lower() == "lstm":
                    checkpoint.update(
                        {
                            "hidden_size": HIDDEN_SIZE,
                            "num_layers": NUM_LAYERS,
                        }
                    )
                else:
                    checkpoint.update(
                        {
                            "embed_dim": TRANSFORMER_EMBED_DIM,
                            "num_heads": TRANSFORMER_HEADS,
                            "num_layers": TRANSFORMER_LAYERS,
                            "ff_dim": TRANSFORMER_FF_DIM,
                        }
                    )
                torch.save(checkpoint, model_out)
                logger.info(
                    "Saved new best %s model to %s (val_acc=%.4f)",
                    model_name,
                    model_out.resolve(),
                    best_val_accuracy,
                )
                print(f"Saved best {model_name} checkpoint to: {model_out.resolve()}")
                if versioned_model_out is not None:
                    torch.save(checkpoint, versioned_model_out)
                    logger.info(
                        "Saved versioned %s model to %s (val_acc=%.4f)",
                        model_name,
                        versioned_model_out.resolve(),
                        best_val_accuracy,
                    )
                    print(f"Saved versioned {model_name} checkpoint to: {versioned_model_out.resolve()}")

    return best_val_accuracy


def main() -> None:
    """Parse CLI arguments and train the sequence models."""
    parser = argparse.ArgumentParser(
        description="Train a Transformer classifier on normalized air-drawn letter trajectories and compare it to the LSTM baseline."
    )
    parser.add_argument("csv_path", help="Path to the trajectory CSV file.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS, help=f"Number of training epochs. Default: {DEFAULT_TRAIN_EPOCHS}")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size. Default: {DEFAULT_BATCH_SIZE}")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help=f"Learning rate. Default: {DEFAULT_LEARNING_RATE}")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help=f"Random seed. Default: {DEFAULT_RANDOM_SEED}")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_OUTPUT, help=f"Best-model output path. Default: {DEFAULT_MODEL_OUTPUT}")
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_index = LetterTrajectoryIndex(csv_path)
    train_offsets, val_offsets = build_stratified_split(dataset_index, train_ratio=TRAIN_SPLIT_RATIO, seed=args.seed)
    train_set = LetterTrajectoryIterableDataset(csv_path, train_offsets)
    val_set = LetterTrajectoryIterableDataset(csv_path, val_offsets)

    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    use_loader_workers = device.type in {"cuda", "mps"}
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 4 if use_loader_workers else 0,
        "pin_memory": use_loader_workers,
    }
    train_loader = DataLoader(train_set, **loader_kwargs)
    val_loader = DataLoader(val_set, **loader_kwargs)

    model_out = resolve_model_output_path(args.model_out)
    model_version = load_model_version(TRAINING_STATE_PATH)
    versioned_model_out = model_out.with_name(f"letter_model_v{model_version}.pt")

    logger.info("Loaded %s total samples from %s", len(dataset_index), csv_path)
    logger.info("Train samples: %s | Val samples: %s", len(train_set), len(val_set))
    logger.info("Training on device: %s", device)

    logger.info("Training LSTM baseline...")
    lstm_model = LetterLSTMClassifier().to(device)
    lstm_best_val_accuracy = train_model(
        model=lstm_model,
        model_name="LSTM",
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_out=None,
    )

    logger.info("Training Transformer encoder...")
    transformer_model = LetterTransformerClassifier().to(device)
    transformer_best_val_accuracy = train_model(
        model=transformer_model,
        model_name="Transformer",
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_out=model_out,
        versioned_model_out=versioned_model_out,
    )

    logger.info("Validation accuracy comparison")
    logger.info("LSTM best val_acc: %.4f", lstm_best_val_accuracy)
    logger.info("Transformer best val_acc: %.4f", transformer_best_val_accuracy)
    logger.info("Delta (Transformer-LSTM): %+0.4f", transformer_best_val_accuracy - lstm_best_val_accuracy)
    logger.info("Best Transformer checkpoint saved to: %s", model_out.resolve())
    logger.info("Versioned Transformer checkpoint saved to: %s", versioned_model_out.resolve())
    if model_out.exists():
        print(f"letter_model.pt saved successfully at: {model_out.resolve()}")
    if versioned_model_out.exists():
        print(f"letter_model_v{model_version}.pt saved successfully at: {versioned_model_out.resolve()}")


if __name__ == "__main__":
    main()
