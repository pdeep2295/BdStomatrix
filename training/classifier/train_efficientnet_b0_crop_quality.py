"""
Train the three-class EfficientNet-B0 stomatal crop-quality classifier.

Classes
-------
0 = full_good
1 = guard_only
2 = bad

Scientific safeguards
---------------------
- Uses only the locked ``train`` and ``val`` partitions.
- Never loads test-image pixels and never calculates test performance.
- Groups were already separated by slide when the dataset was constructed.
- Preserves every crop by median-padding to a square before resizing.
- Does not use random/centre crops, blur, cutout, hue, saturation, or CLAHE.
- Does not edit, move, rename, or delete any crop TIFF.
- Selects ``best.pt`` using validation macro-F1 only.

Run this script yourself in the ``stoma`` environment. The script is designed
for Apple MPS and deliberately refuses to fall back silently to CPU.

Normal first run
----------------
    python train_efficientnet_b0_crop_quality.py

Resume after an interrupted run
-------------------------------
1. Set RESUME_FROM near the top of this file to the existing ``last.pt``.
2. Run the same command again.

The independent classifier test partition is reserved for a later, separate
evaluation script after the model and probability thresholds have been locked.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageEnhance, ImageOps
import torch
import torchvision
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


# =============================================================================
# 1. Confirmed paths and locked dataset identity
# =============================================================================

PROJECT_ROOT = Path(
    "/Users/pawandeepsingh/Documents/Postdoc/YOLO_model"
).resolve()
CLASSIFIER_ROOT = (PROJECT_ROOT / "classifier_main").resolve()
DATASET_ROOT = (
    CLASSIFIER_ROOT / "classifier_dataset_slide_grouped_v1"
).resolve()
SPLIT_MANIFEST_CSV = (DATASET_ROOT / "split_manifest.csv").resolve()
DATASET_PROVENANCE_JSON = (DATASET_ROOT / "dataset_provenance.json").resolve()

# A new directory is created only when you run this script. An existing run is
# never overwritten unless RESUME_FROM explicitly points to its last.pt.
OUTPUT_PARENT = (CLASSIFIER_ROOT / "classifier_training_results").resolve()
RUN_NAME = "efficientnet_b0_three_class_v1"
RUN_DIR = (OUTPUT_PARENT / RUN_NAME).resolve()

# Keep as None for a new run. To resume, replace None with the full last.pt path.
RESUME_FROM: Path | None = None
# Example:
# RESUME_FROM = RUN_DIR / "last.pt"

EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "e3497555303c0c4f930b7b4c35798cdedb69ab3be07854c920f7942ec84cd34e"
)
EXPECTED_DATASET_COUNTS = {"train": 3187, "val": 702, "test": 657}
EXPECTED_CLASS_COUNTS = {
    "train": {"full_good": 1549, "guard_only": 613, "bad": 1025},
    "val": {"full_good": 342, "guard_only": 132, "bad": 228},
    "test": {"full_good": 316, "guard_only": 126, "bad": 215},
}

CLASS_NAMES = ("full_good", "guard_only", "bad")
CLASS_TO_ID = {name: class_id for class_id, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {class_id: name for name, class_id in CLASS_TO_ID.items()}
ALLOWED_TRAINING_SPLITS = {"train", "val"}


# =============================================================================
# 2. Training settings
# =============================================================================


@dataclass(frozen=True)
class TrainingConfig:
    architecture: str = "efficientnet_b0"
    pretrained_weights: str = "EfficientNet_B0_Weights.DEFAULT"
    image_size: int = 224
    batch_size: int = 32
    workers: int = 2
    head_only_epochs: int = 5
    maximum_epochs: int = 50
    head_learning_rate: float = 1.0e-3
    full_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1.0e-4
    random_seed: int = 42
    device: str = "mps"
    label_smoothing: float = 0.0
    dropout: float = 0.2
    brightness_jitter: float = 0.10
    contrast_jitter: float = 0.10
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    right_angle_rotation_probability: float = 0.75
    verify_train_val_hashes: bool = True
    use_mixed_precision: bool = False


CONFIG = TrainingConfig()


# =============================================================================
# 3. General utilities and reproducibility
# =============================================================================


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 checksum without changing the file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temporary file and replace the destination safely."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def save_checkpoint_atomic(path: Path, checkpoint: dict[str, Any]) -> None:
    """Save a PyTorch checkpoint atomically."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and DataLoader random number generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Some MPS operations may not have a deterministic implementation. Warnings
    # are recorded instead of aborting an otherwise valid training run.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("high")


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker a deterministic but distinct seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def require_mps_device() -> torch.device:
    """Return MPS or fail clearly rather than silently training on CPU."""
    if CONFIG.device != "mps":
        raise ValueError("This locked implementation expects CONFIG.device='mps'.")
    if not torch.backends.mps.is_built():
        raise RuntimeError(
            "This PyTorch installation was not built with Apple MPS support."
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "Apple MPS is not available in this Python session. Confirm that "
            "you are running the same stoma environment used for detector "
            "training and that macOS/PyTorch can access the Apple GPU. The "
            "script will not silently fall back to CPU."
        )
    return torch.device("mps")


def safe_divide(numerator: float, denominator: float) -> float:
    """Return zero when a metric denominator is zero."""
    return float(numerator / denominator) if denominator else 0.0


# =============================================================================
# 4. Dataset validation
# =============================================================================


def load_and_validate_split_manifest() -> tuple[pd.DataFrame, str]:
    """Validate the locked slide-grouped split without loading test pixels."""
    if not DATASET_ROOT.is_dir():
        raise NotADirectoryError(f"Classifier dataset directory not found:\n{DATASET_ROOT}")
    if not SPLIT_MANIFEST_CSV.is_file():
        raise FileNotFoundError(f"Split manifest not found:\n{SPLIT_MANIFEST_CSV}")
    if not DATASET_PROVENANCE_JSON.is_file():
        raise FileNotFoundError(
            f"Dataset provenance record not found:\n{DATASET_PROVENANCE_JSON}"
        )

    observed_manifest_hash = sha256_file(SPLIT_MANIFEST_CSV)
    if observed_manifest_hash != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise RuntimeError(
            "The split manifest has changed since it was locked.\n"
            f"Expected: {EXPECTED_SPLIT_MANIFEST_SHA256}\n"
            f"Observed: {observed_manifest_hash}"
        )

    manifest = pd.read_csv(SPLIT_MANIFEST_CSV, keep_default_na=False)
    required_columns = {
        "split",
        "class_id",
        "destination_relative_path",
        "stoma_id",
        "crop_filename",
        "crop_sha256",
        "source_image_name",
        "surface",
        "slide_id",
        "biological_unit",
        "quality_label",
    }
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Split manifest is missing columns: {sorted(missing_columns)}"
        )

    if manifest["stoma_id"].duplicated().any():
        raise ValueError("Duplicated stoma_id values detected in the split manifest.")
    if manifest["destination_relative_path"].duplicated().any():
        raise ValueError("Duplicated destination paths detected in the split manifest.")
    if set(manifest["split"]) != set(EXPECTED_DATASET_COUNTS):
        raise ValueError(f"Unexpected split values: {sorted(set(manifest['split']))}")
    if set(manifest["quality_label"]) != set(CLASS_NAMES):
        raise ValueError(
            f"Unexpected class labels: {sorted(set(manifest['quality_label']))}"
        )
    if set(manifest["surface"]) != {"AB", "AD"}:
        raise ValueError(f"Unexpected surface values: {sorted(set(manifest['surface']))}")

    expected_ids = manifest["quality_label"].map(CLASS_TO_ID).astype(int)
    observed_ids = pd.to_numeric(manifest["class_id"], errors="raise").astype(int)
    if not expected_ids.equals(observed_ids):
        raise ValueError("class_id values do not match the locked class-name mapping.")

    for split, expected_count in EXPECTED_DATASET_COUNTS.items():
        observed_count = int(manifest["split"].eq(split).sum())
        if observed_count != expected_count:
            raise ValueError(
                f"Unexpected {split} crop count: {observed_count}; "
                f"expected {expected_count}."
            )
        observed_class_counts = (
            manifest.loc[manifest["split"] == split, "quality_label"]
            .value_counts()
            .to_dict()
        )
        if observed_class_counts != EXPECTED_CLASS_COUNTS[split]:
            raise ValueError(
                f"Unexpected {split} class counts: {observed_class_counts}; "
                f"expected {EXPECTED_CLASS_COUNTS[split]}."
            )

    for grouping_column in ("slide_id", "biological_unit", "source_image_name"):
        maximum_split_count = manifest.groupby(grouping_column)["split"].nunique().max()
        if maximum_split_count != 1:
            raise ValueError(
                f"Leakage detected: {grouping_column} occurs in multiple splits."
            )

    for row in manifest.itertuples(index=False):
        relative_path = Path(row.destination_relative_path)
        expected_relative_path = Path(row.split) / row.quality_label / row.crop_filename
        if relative_path != expected_relative_path:
            raise ValueError(
                f"Unexpected destination path for {row.stoma_id}: {relative_path}"
            )

    # Validate file existence for all splits as dataset bookkeeping, but only
    # train and val pixels are read or hashed below. Test remains image-hidden.
    missing_paths = []
    for relative_path in manifest["destination_relative_path"]:
        candidate = (DATASET_ROOT / relative_path).resolve()
        try:
            candidate.relative_to(DATASET_ROOT)
        except ValueError as error:
            raise ValueError(f"Dataset path escapes DATASET_ROOT: {candidate}") from error
        if not candidate.is_file():
            missing_paths.append(str(candidate))
    if missing_paths:
        raise FileNotFoundError(
            f"Missing {len(missing_paths)} copied crops; first examples:\n"
            + "\n".join(missing_paths[:5])
        )

    if CONFIG.verify_train_val_hashes:
        development_rows = manifest[manifest["split"].isin(ALLOWED_TRAINING_SPLITS)]
        checksum_failures = []
        for row in development_rows.itertuples(index=False):
            crop_path = DATASET_ROOT / row.destination_relative_path
            if sha256_file(crop_path) != row.crop_sha256:
                checksum_failures.append(row.crop_filename)
        if checksum_failures:
            raise RuntimeError(
                f"Checksum failure for {len(checksum_failures)} train/val crops; "
                f"first examples: {checksum_failures[:5]}"
            )

    return manifest, observed_manifest_hash


# =============================================================================
# 5. Full-crop-preserving image transformations
# =============================================================================


class MedianPadToSquare:
    """Centre-pad a PIL RGB image to a square using its per-channel median."""

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        if width == height:
            return image

        array = np.asarray(image, dtype=np.uint8)
        medians = np.rint(np.median(array.reshape(-1, 3), axis=0)).astype(np.uint8)
        fill = tuple(int(value) for value in medians)

        difference = abs(width - height)
        first = difference // 2
        second = difference - first
        if width < height:
            border = (first, 0, second, 0)
        else:
            border = (0, first, 0, second)
        return ImageOps.expand(image, border=border, fill=fill)


class RandomRightAngleRotation:
    """Apply a lossless 90, 180, or 270 degree rotation with probability p."""

    def __init__(self, probability: float) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Rotation probability must be between zero and one.")
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        operation = random.choice(
            (
                Image.Transpose.ROTATE_90,
                Image.Transpose.ROTATE_180,
                Image.Transpose.ROTATE_270,
            )
        )
        return image.transpose(operation)


class MildBrightnessContrast:
    """Apply mild grayscale-compatible brightness and contrast variation."""

    def __init__(self, brightness: float, contrast: float) -> None:
        self.brightness = float(brightness)
        self.contrast = float(contrast)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.brightness > 0:
            factor = random.uniform(1.0 - self.brightness, 1.0 + self.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast > 0:
            factor = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image


class CropTransform:
    """Apply the complete train or validation preprocessing pipeline."""

    def __init__(self, training: bool, config: TrainingConfig) -> None:
        self.training = training
        self.config = config
        self.pad_to_square = MedianPadToSquare()
        self.rotate = RandomRightAngleRotation(
            config.right_angle_rotation_probability
        )
        self.intensity = MildBrightnessContrast(
            config.brightness_jitter,
            config.contrast_jitter,
        )
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = self.pad_to_square(image)
        if self.training:
            if random.random() < self.config.horizontal_flip_probability:
                image = ImageOps.mirror(image)
            if random.random() < self.config.vertical_flip_probability:
                image = ImageOps.flip(image)
            image = self.rotate(image)
            image = self.intensity(image)

        image = TF.resize(
            image,
            [self.config.image_size, self.config.image_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, mean=self.mean, std=self.std)


# =============================================================================
# 6. Manifest-backed dataset and DataLoaders
# =============================================================================


class CropQualityDataset(Dataset):
    """Read only one approved development split from the locked manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        transform: CropTransform,
    ) -> None:
        if split not in ALLOWED_TRAINING_SPLITS:
            raise ValueError(
                f"CropQualityDataset can load only {sorted(ALLOWED_TRAINING_SPLITS)}; "
                f"received {split!r}. Test pixels are intentionally inaccessible."
            )
        self.split = split
        self.transform = transform
        self.records = (
            manifest.loc[manifest["split"] == split]
            .sort_values(["slide_id", "source_image_name", "crop_filename"])
            .reset_index(drop=True)
        )
        if self.records.empty:
            raise ValueError(f"No records found for split {split!r}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.records.iloc[index]
        crop_path = DATASET_ROOT / row["destination_relative_path"]
        with Image.open(crop_path) as image:
            tensor = self.transform(image.copy())
        return tensor, int(row["class_id"]), index


def build_dataloaders(
    manifest: pd.DataFrame,
) -> tuple[DataLoader, DataLoader, CropQualityDataset, CropQualityDataset]:
    """Build deterministic train and validation DataLoaders."""
    train_dataset = CropQualityDataset(
        manifest,
        split="train",
        transform=CropTransform(training=True, config=CONFIG),
    )
    val_dataset = CropQualityDataset(
        manifest,
        split="val",
        transform=CropTransform(training=False, config=CONFIG),
    )

    generator = torch.Generator()
    generator.manual_seed(CONFIG.random_seed)

    common_arguments = {
        "batch_size": CONFIG.batch_size,
        "num_workers": CONFIG.workers,
        "pin_memory": False,
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_arguments,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_arguments,
    )
    return train_loader, val_loader, train_dataset, val_dataset


def calculate_class_weights(train_dataset: CropQualityDataset) -> torch.Tensor:
    """Calculate normalized inverse-square-root weights from training data only."""
    counts = (
        train_dataset.records["class_id"]
        .value_counts()
        .reindex(range(len(CLASS_NAMES)), fill_value=0)
        .to_numpy(dtype=float)
    )
    if np.any(counts <= 0):
        raise ValueError(f"Every training class must be represented; counts={counts}.")
    total = counts.sum()
    weights = np.sqrt(total / (len(CLASS_NAMES) * counts))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# 7. Model construction and training stages
# =============================================================================


def build_model() -> nn.Module:
    """Load ImageNet-pretrained EfficientNet-B0 and install a three-class head."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=CONFIG.dropout, inplace=False),
        nn.Linear(input_features, len(CLASS_NAMES)),
    )
    return model


def stage_for_epoch(epoch: int) -> str:
    """Return the training stage for a one-based epoch number."""
    return "head_only" if epoch <= CONFIG.head_only_epochs else "full_finetune"


def configure_stage(
    model: nn.Module,
    stage: str,
) -> tuple[AdamW, CosineAnnealingLR]:
    """Freeze/unfreeze the model and construct the stage optimizer/scheduler."""
    if stage == "head_only":
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        learning_rate = CONFIG.head_learning_rate
        stage_epochs = CONFIG.head_only_epochs
    elif stage == "full_finetune":
        for parameter in model.parameters():
            parameter.requires_grad = True
        learning_rate = CONFIG.full_learning_rate
        stage_epochs = CONFIG.maximum_epochs - CONFIG.head_only_epochs
    else:
        raise ValueError(f"Unknown training stage: {stage}")

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=CONFIG.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, stage_epochs),
        eta_min=learning_rate * 0.01,
    )
    return optimizer, scheduler


# =============================================================================
# 8. Metrics
# =============================================================================


def confusion_matrix_array(
    true_ids: np.ndarray,
    predicted_ids: np.ndarray,
) -> np.ndarray:
    """Return a rows=true, columns=predicted confusion matrix."""
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(matrix, (true_ids, predicted_ids), 1)
    return matrix


def metrics_from_confusion_matrix(matrix: np.ndarray) -> dict[str, Any]:
    """Calculate overall and per-class metrics from a confusion matrix."""
    total = int(matrix.sum())
    per_class = {}
    recalls = []
    f1_values = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        true_positive = int(matrix[class_id, class_id])
        false_positive = int(matrix[:, class_id].sum() - true_positive)
        false_negative = int(matrix[class_id, :].sum() - true_positive)
        support = int(matrix[class_id, :].sum())
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)
        recalls.append(recall)
        f1_values.append(f1)
        per_class[class_name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    accuracy = safe_divide(float(np.trace(matrix)), total)
    return {
        "n": total,
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
    }


def calculate_critical_error_rates(predictions: pd.DataFrame) -> dict[str, float]:
    """Calculate downstream-relevant quality-classification error rates."""
    rules = {
        "bad_to_full_good_rate": ("bad", "full_good"),
        "guard_only_to_full_good_rate": ("guard_only", "full_good"),
        "full_good_to_bad_rate": ("full_good", "bad"),
    }
    output = {}
    for name, (true_name, predicted_name) in rules.items():
        true_subset = predictions["true_label"] == true_name
        numerator = int(
            (true_subset & (predictions["predicted_label"] == predicted_name)).sum()
        )
        denominator = int(true_subset.sum())
        output[name] = safe_divide(numerator, denominator)
    return output


# =============================================================================
# 9. Epoch execution
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch and return weighted loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    sample_count = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.detach().cpu()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
        sample_count += batch_size

    return safe_divide(running_loss, sample_count), safe_divide(correct, sample_count)


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    dataset: CropQualityDataset,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    """Validate without augmentation and return metrics plus crop probabilities."""
    model.eval()
    running_loss = 0.0
    sample_count = 0
    true_batches = []
    predicted_batches = []
    probability_batches = []
    index_batches = []

    with torch.inference_mode():
        for images, labels, indices in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            batch_size = labels.size(0)
            running_loss += float(loss.detach().cpu()) * batch_size
            sample_count += batch_size
            true_batches.append(labels.detach().cpu().numpy())
            predicted_batches.append(predictions.detach().cpu().numpy())
            probability_batches.append(probabilities.detach().cpu().numpy())
            index_batches.append(indices.detach().cpu().numpy())

    true_ids = np.concatenate(true_batches).astype(int)
    predicted_ids = np.concatenate(predicted_batches).astype(int)
    probabilities = np.concatenate(probability_batches)
    record_indices = np.concatenate(index_batches).astype(int)

    matrix = confusion_matrix_array(true_ids, predicted_ids)
    metrics = metrics_from_confusion_matrix(matrix)
    validation_loss = safe_divide(running_loss, sample_count)
    metrics["loss"] = validation_loss

    metadata_columns = [
        "stoma_id",
        "crop_filename",
        "source_image_name",
        "surface",
        "slide_id",
        "biological_unit",
    ]
    predictions_table = dataset.records.iloc[record_indices][metadata_columns].reset_index(
        drop=True
    )
    predictions_table["true_class_id"] = true_ids
    predictions_table["true_label"] = [ID_TO_CLASS[value] for value in true_ids]
    predictions_table["predicted_class_id"] = predicted_ids
    predictions_table["predicted_label"] = [
        ID_TO_CLASS[value] for value in predicted_ids
    ]
    for class_id, class_name in enumerate(CLASS_NAMES):
        predictions_table[f"probability_{class_name}"] = probabilities[:, class_id]

    metrics["critical_error_rates"] = calculate_critical_error_rates(
        predictions_table
    )
    return validation_loss, metrics, predictions_table


# =============================================================================
# 10. Validation artifacts and plots
# =============================================================================


def plot_training_history(history: pd.DataFrame, output_path: Path) -> None:
    """Write a compact four-panel training diagnostic figure."""
    if history.empty:
        return
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0, 0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(history["epoch"], history["val_macro_f1"])
    axes[0, 1].set_title("Validation macro-F1")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].plot(
        history["epoch"], history["train_accuracy"], label="train accuracy"
    )
    axes[1, 0].plot(
        history["epoch"], history["val_accuracy"], label="validation accuracy"
    )
    axes[1, 0].plot(
        history["epoch"],
        history["val_balanced_accuracy"],
        label="validation balanced accuracy",
    )
    axes[1, 0].set_title("Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(history["epoch"], history["learning_rate"])
    axes[1, 1].set_title("Learning rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_yscale("log")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    """Plot counts and row-normalized percentages for the validation matrix."""
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Fraction within true class")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=25, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Best-checkpoint validation confusion matrix")

    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            fraction = normalized[row, column]
            color = "white" if fraction > 0.5 else "black"
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n({100 * fraction:.1f}%)",
                ha="center",
                va="center",
                color=color,
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def surface_metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate validation performance separately for AB and AD."""
    rows = []
    for surface in ("AB", "AD"):
        subset = predictions[predictions["surface"] == surface]
        true_ids = subset["true_class_id"].to_numpy(dtype=int)
        predicted_ids = subset["predicted_class_id"].to_numpy(dtype=int)
        matrix = confusion_matrix_array(true_ids, predicted_ids)
        metrics = metrics_from_confusion_matrix(matrix)
        critical = calculate_critical_error_rates(subset)
        row = {
            "surface": surface,
            "n": metrics["n"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            **critical,
        }
        for class_name in CLASS_NAMES:
            for metric_name in ("precision", "recall", "f1", "support"):
                row[f"{class_name}_{metric_name}"] = metrics["per_class"][
                    class_name
                ][metric_name]
        rows.append(row)
    return pd.DataFrame(rows)


def save_best_validation_artifacts(
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
) -> None:
    """Save validation-only tables and figures for the current best checkpoint."""
    predictions.to_csv(RUN_DIR / "validation_predictions_best.csv", index=False)
    write_json_atomic(RUN_DIR / "validation_metrics_best.json", metrics)

    matrix = np.asarray(
        metrics["confusion_matrix_rows_true_columns_predicted"], dtype=int
    )
    matrix_table = pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"predicted_{name}" for name in CLASS_NAMES],
    )
    matrix_table.to_csv(RUN_DIR / "validation_confusion_matrix_best.csv")
    plot_confusion_matrix(
        matrix,
        RUN_DIR / "validation_confusion_matrix_best.png",
    )

    per_class_rows = []
    for class_name, class_metrics in metrics["per_class"].items():
        per_class_rows.append({"class_name": class_name, **class_metrics})
    pd.DataFrame(per_class_rows).to_csv(
        RUN_DIR / "validation_metrics_by_class_best.csv",
        index=False,
    )
    surface_metrics_table(predictions).to_csv(
        RUN_DIR / "validation_metrics_by_surface_best.csv",
        index=False,
    )


# =============================================================================
# 11. Provenance and run-directory handling
# =============================================================================


def environment_record() -> dict[str, Any]:
    """Collect software and hardware information without invoking subprocesses."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": PIL.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }


def initialize_or_validate_run_directory() -> None:
    """Protect existing results and validate an explicitly requested resume."""
    if RESUME_FROM is None:
        if RUN_DIR.exists():
            raise FileExistsError(
                "The output directory already exists and will not be overwritten:\n"
                f"{RUN_DIR}\nUse a new RUN_NAME, or explicitly set RESUME_FROM "
                "to its last.pt after confirming that you intend to resume."
            )
        RUN_DIR.mkdir(parents=True, exist_ok=False)
    else:
        resume_path = Path(RESUME_FROM).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found:\n{resume_path}")
        if resume_path.parent != RUN_DIR:
            raise ValueError(
                "RESUME_FROM must point to last.pt inside the configured RUN_DIR."
            )
        if not RUN_DIR.is_dir():
            raise NotADirectoryError(f"Resume run directory not found:\n{RUN_DIR}")


def make_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    stage: str,
    history: list[dict[str, Any]],
    best_macro_f1: float,
    best_epoch: int,
    epochs_without_improvement: int,
    class_weights: torch.Tensor,
    manifest_hash: str,
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build a complete restartable and auditable checkpoint."""
    return {
        "architecture": CONFIG.architecture,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "stage": stage,
        "history": history,
        "best_macro_f1": best_macro_f1,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "class_names": list(CLASS_NAMES),
        "class_to_id": CLASS_TO_ID,
        "class_weights": class_weights.tolist(),
        "training_config": asdict(CONFIG),
        "split_manifest_sha256": manifest_hash,
        "validation_metrics_at_checkpoint": validation_metrics,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# 12. Complete training workflow
# =============================================================================


def train_efficientnet_b0_classifier() -> Path:
    """Run validation-controlled training without accessing test pixels."""
    if CONFIG.maximum_epochs <= CONFIG.head_only_epochs:
        raise ValueError("maximum_epochs must be greater than head_only_epochs.")
    if CONFIG.batch_size <= 0:
        raise ValueError("batch_size must be a positive fixed integer.")
    if CONFIG.use_mixed_precision:
        raise ValueError(
            "This version deliberately locks mixed precision off for MPS stability."
        )

    set_reproducible_seed(CONFIG.random_seed)
    device = require_mps_device()
    manifest, manifest_hash = load_and_validate_split_manifest()
    initialize_or_validate_run_directory()

    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(
        manifest
    )
    class_weights = calculate_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=CONFIG.label_smoothing,
    )
    model = build_model().to(device)

    configuration_record = {
        "training_config": asdict(CONFIG),
        "class_names": list(CLASS_NAMES),
        "class_to_id": CLASS_TO_ID,
        "train_class_counts": EXPECTED_CLASS_COUNTS["train"],
        "validation_class_counts": EXPECTED_CLASS_COUNTS["val"],
        "class_weights_inverse_square_root_normalized": {
            name: float(class_weights[class_id])
            for name, class_id in CLASS_TO_ID.items()
        },
        "dataset_root": str(DATASET_ROOT),
        "split_manifest": str(SPLIT_MANIFEST_CSV),
        "split_manifest_sha256": manifest_hash,
        "run_directory": str(RUN_DIR),
        "test_pixels_loaded": False,
        "test_evaluation_performed": False,
        "preprocessing": {
            "convert_to_rgb": True,
            "median_pad_to_square": True,
            "resize": [CONFIG.image_size, CONFIG.image_size],
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "random_crop": False,
            "center_crop": False,
            "clahe": False,
            "blur": False,
            "random_erasing": False,
            "hue_or_saturation_change": False,
        },
    }
    write_json_atomic(RUN_DIR / "run_configuration.json", configuration_record)
    write_json_atomic(RUN_DIR / "software_environment.json", environment_record())

    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_macro_f1 = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    current_stage = stage_for_epoch(start_epoch)
    optimizer, scheduler = configure_stage(model, current_stage)

    if RESUME_FROM is not None:
        checkpoint = torch.load(
            Path(RESUME_FROM).resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("split_manifest_sha256") != manifest_hash:
            raise RuntimeError("Resume checkpoint belongs to a different split manifest.")
        if checkpoint.get("class_to_id") != CLASS_TO_ID:
            raise RuntimeError("Resume checkpoint class mapping is incompatible.")
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history", []))
        best_macro_f1 = float(checkpoint.get("best_macro_f1", -math.inf))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        saved_stage = str(checkpoint["stage"])
        next_stage = stage_for_epoch(start_epoch)
        optimizer, scheduler = configure_stage(model, saved_stage)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        current_stage = saved_stage
        if next_stage != saved_stage:
            current_stage = next_stage
            optimizer, scheduler = configure_stage(model, current_stage)
            epochs_without_improvement = 0

    if start_epoch > CONFIG.maximum_epochs:
        raise RuntimeError(
            f"Checkpoint already reached epoch {start_epoch - 1}; "
            f"maximum_epochs is {CONFIG.maximum_epochs}."
        )

    print(f"Device: {device}")
    print(f"Run directory: {RUN_DIR}")
    print(f"Training crops: {len(train_dataset)}")
    print(f"Validation crops: {len(val_dataset)}")
    print("Test pixels are not loaded by this script.")
    print(
        "Class weights: "
        + ", ".join(
            f"{name}={class_weights[class_id]:.4f}"
            for name, class_id in CLASS_TO_ID.items()
        )
    )

    for epoch in range(start_epoch, CONFIG.maximum_epochs + 1):
        required_stage = stage_for_epoch(epoch)
        if required_stage != current_stage:
            current_stage = required_stage
            optimizer, scheduler = configure_stage(model, current_stage)
            epochs_without_improvement = 0
            print(f"\nUnfreezing the full model at epoch {epoch}.")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_metrics, val_predictions = validate_one_epoch(
            model,
            val_loader,
            val_dataset,
            criterion,
            device,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        row = {
            "epoch": epoch,
            "stage": current_stage,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_bad_to_full_good_rate": val_metrics["critical_error_rates"][
                "bad_to_full_good_rate"
            ],
            "val_guard_only_to_full_good_rate": val_metrics[
                "critical_error_rates"
            ]["guard_only_to_full_good_rate"],
            "val_full_good_to_bad_rate": val_metrics["critical_error_rates"][
                "full_good_to_bad_rate"
            ],
        }
        for class_name in CLASS_NAMES:
            row[f"val_{class_name}_precision"] = val_metrics["per_class"][
                class_name
            ]["precision"]
            row[f"val_{class_name}_recall"] = val_metrics["per_class"][
                class_name
            ]["recall"]
            row[f"val_{class_name}_f1"] = val_metrics["per_class"][class_name][
                "f1"
            ]
        history.append(row)

        history_table = pd.DataFrame(history)
        history_table.to_csv(RUN_DIR / "training_history.csv", index=False)
        plot_training_history(history_table, RUN_DIR / "training_curves.png")

        improved = (
            val_metrics["macro_f1"]
            > best_macro_f1 + CONFIG.early_stopping_min_delta
        )
        if improved:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = make_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            stage=current_stage,
            history=history,
            best_macro_f1=best_macro_f1,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            class_weights=class_weights,
            manifest_hash=manifest_hash,
            validation_metrics=val_metrics,
        )
        save_checkpoint_atomic(RUN_DIR / "last.pt", checkpoint)

        if improved:
            save_checkpoint_atomic(RUN_DIR / "best.pt", checkpoint)
            save_best_validation_artifacts(val_metrics, val_predictions)

        print(
            f"Epoch {epoch:02d}/{CONFIG.maximum_epochs} | "
            f"stage={current_stage} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_accuracy={val_metrics['accuracy']:.4f} | "
            f"best_epoch={best_epoch}"
        )

        if (
            current_stage == "full_finetune"
            and epochs_without_improvement >= CONFIG.early_stopping_patience
        ):
            print(
                "Early stopping: validation macro-F1 did not improve by at "
                f"least {CONFIG.early_stopping_min_delta} for "
                f"{CONFIG.early_stopping_patience} full-fine-tuning epochs."
            )
            break

    completion_record = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_epoch": int(history[-1]["epoch"]),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "best_checkpoint": str(RUN_DIR / "best.pt"),
        "last_checkpoint": str(RUN_DIR / "last.pt"),
        "test_pixels_loaded": False,
        "test_evaluation_performed": False,
        "next_step": (
            "Inspect validation diagnostics and lock any probability thresholds "
            "before running a separate independent test evaluation."
        ),
    }
    write_json_atomic(RUN_DIR / "training_completion.json", completion_record)
    print(f"\nTraining finished. Candidate checkpoint:\n{RUN_DIR / 'best.pt'}")
    print("Independent test evaluation has NOT been performed.")
    return RUN_DIR


if __name__ == "__main__":
    train_efficientnet_b0_classifier()
