"""Train the final YOLO26n detector on the combined AB + AD dataset.

This script trains only against the train and validation splits. The test split
is checked for structural integrity but is not evaluated during training.

Run this script yourself from the Python environment containing Ultralytics,
PyTorch with Apple MPS support, OpenCV, NumPy, and PyYAML.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# Paths and identifiers
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(
    "/Users/pawandeepsingh/Documents/Postdoc/YOLO_model"
).resolve()

DATASET_YAML = PROJECT_ROOT / "dataset_A_boxes" / "dataset.yaml"
OUTPUT_PROJECT = PROJECT_ROOT / "Training_Results"
RUN_NAME = "yolo26n_Stomata_DIC_Final"
MODEL_WEIGHTS = "yolo26n.pt"


# -----------------------------------------------------------------------------
# Dataset assumptions established during dataset QC
# -----------------------------------------------------------------------------

EXPECTED_CLASSES = {0: "stoma", 1: "hair_cell"}
EXPECTED_SPLIT_COUNTS = {"train": 407, "val": 48, "test": 58}
EXPECTED_IMAGE_SHAPE = (2048, 3072, 3)
EXPECTED_IMAGE_DTYPE = np.uint8
IMAGE_SUFFIXES = {".tif", ".tiff"}

# This checks every training copy before training begins. It does not modify it.
VERIFY_ALL_THREE_CHANNEL_IMAGES = True

# Disk caching avoids consuming unified memory as a RAM cache. It can create
# .npy cache files beside the dataset copies. Set this explicitly to False if
# SSD space is insufficient; do not change it to "ram" on this computer.
CACHE_MODE: str | bool = "disk"
MINIMUM_FREE_DISK_GIB = 15.0


# -----------------------------------------------------------------------------
# Final training configuration
# -----------------------------------------------------------------------------

TRAINING_ARGUMENTS: dict[str, Any] = {
    # Dataset and output
    "data": str(DATASET_YAML),
    "project": str(OUTPUT_PROJECT),
    "name": RUN_NAME,
    "exist_ok": False,

    # Model fitting
    "epochs": 100,
    "imgsz": 1536,
    "batch": 8,
    "device": "mps",
    "workers": 4,
    "rect": True,
    "cache": CACHE_MODE,

    # Optimization
    "optimizer": "AdamW",
    "lr0": 0.001,
    "warmup_epochs": 3.0,
    "patience": 25,

    # Reproducibility
    "seed": 42,
    "deterministic": True,

    # Checkpoints and diagnostic plots
    "save": True,
    "save_period": 10,
    "plots": True,
    "val": True,

    # Keep the two detection classes separate
    "single_cls": False,

    # Custom pilot loss weights retained for scientific comparability
    "box": 10.0,
    "cls": 2.0,
    "label_smoothing": 0.0,

    # DIC augmentation
    "degrees": 45.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.2,
    "mosaic": 0.0,
    "translate": 0.05,
    "scale": 0.2,
}


def load_dataset_configuration() -> dict[str, Any]:
    """Load the dataset YAML and verify the required class mapping."""
    if not DATASET_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATASET_YAML}")

    with DATASET_YAML.open("r", encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)

    if not isinstance(configuration, dict):
        raise ValueError("The dataset YAML must contain a YAML mapping.")

    required_fields = {"path", "train", "val", "test", "names"}
    missing_fields = required_fields.difference(configuration)
    if missing_fields:
        raise ValueError(
            "Dataset YAML is missing: " + ", ".join(sorted(missing_fields))
        )

    raw_names = configuration["names"]
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(index): str(name) for index, name in raw_names.items()}
    else:
        raise ValueError("The YAML 'names' field must be a list or mapping.")

    if names != EXPECTED_CLASSES:
        raise ValueError(
            f"Incorrect class mapping. Expected {EXPECTED_CLASSES}; "
            f"observed {names}."
        )

    return configuration


def resolve_dataset_root(configuration: dict[str, Any]) -> Path:
    """Resolve and validate the dataset root recorded in the YAML."""
    dataset_root = Path(str(configuration["path"])).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = DATASET_YAML.parent / dataset_root
    dataset_root = dataset_root.resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    expected_root = (PROJECT_ROOT / "dataset_A_boxes").resolve()
    if dataset_root != expected_root:
        raise ValueError(
            "The YAML does not point to the reviewed final dataset.\n"
            f"Expected: {expected_root}\nObserved: {dataset_root}"
        )

    return dataset_root


def resolve_split_directory(
    dataset_root: Path,
    split_value: Any,
    split_name: str,
) -> Path:
    """Resolve one image split directory from the YAML."""
    if not isinstance(split_value, str):
        raise ValueError(f"YAML field '{split_name}' must be one directory.")

    split_directory = Path(split_value).expanduser()
    if not split_directory.is_absolute():
        split_directory = dataset_root / split_directory
    split_directory = split_directory.resolve()

    if not split_directory.is_dir():
        raise FileNotFoundError(
            f"Image directory for '{split_name}' not found: {split_directory}"
        )

    return split_directory


def corresponding_label_directory(image_directory: Path) -> Path:
    """Convert .../images/<split> into .../labels/<split>."""
    path_parts = list(image_directory.parts)
    image_positions = [
        index for index, value in enumerate(path_parts) if value == "images"
    ]
    if not image_positions:
        raise ValueError(f"No 'images' component in: {image_directory}")

    path_parts[image_positions[-1]] = "labels"
    label_directory = Path(*path_parts)
    if not label_directory.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_directory}")

    return label_directory


def list_images(image_directory: Path) -> list[Path]:
    """Return only TIFF images, excluding JSON annotation sidecars."""
    return sorted(
        path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_yolo_label(label_path: Path) -> None:
    """Check one YOLO-HBB annotation file without modifying it."""
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing YOLO label: {label_path}")

    annotation_count = 0
    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            annotation_count += 1

            if len(fields) != 5:
                raise ValueError(
                    f"{label_path}, line {line_number}: expected five values."
                )

            try:
                class_value = float(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError as error:
                raise ValueError(
                    f"{label_path}, line {line_number}: non-numeric value."
                ) from error

            if not class_value.is_integer():
                raise ValueError(
                    f"{label_path}, line {line_number}: non-integer class ID."
                )

            class_id = int(class_value)
            if class_id not in EXPECTED_CLASSES:
                raise ValueError(
                    f"{label_path}, line {line_number}: class ID {class_id}."
                )

            x_center, y_center, width, height = coordinates
            if not (
                0.0 <= x_center <= 1.0
                and 0.0 <= y_center <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
            ):
                raise ValueError(
                    f"{label_path}, line {line_number}: invalid normalized box."
                )

    if annotation_count == 0:
        raise ValueError(f"Unexpected empty annotation file: {label_path}")


def validate_three_channel_image(image_path: Path) -> None:
    """Require an 8-bit image with three identical grayscale channels."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read: {image_path}")

    if image.shape != EXPECTED_IMAGE_SHAPE:
        raise ValueError(
            f"{image_path}: expected shape {EXPECTED_IMAGE_SHAPE}, "
            f"observed {image.shape}."
        )
    if image.dtype != EXPECTED_IMAGE_DTYPE:
        raise ValueError(
            f"{image_path}: expected uint8, observed {image.dtype}."
        )
    if not (
        np.array_equal(image[:, :, 0], image[:, :, 1])
        and np.array_equal(image[:, :, 0], image[:, :, 2])
    ):
        raise ValueError(f"Image channels are not identical: {image_path}")


def validate_dataset() -> dict[str, Any]:
    """Run final structural and pixel checks before training."""
    configuration = load_dataset_configuration()
    dataset_root = resolve_dataset_root(configuration)
    split_summary: dict[str, Any] = {}
    observed_stems: dict[str, set[str]] = {}

    for split_name in ("train", "val", "test"):
        image_directory = resolve_split_directory(
            dataset_root,
            configuration[split_name],
            split_name,
        )
        label_directory = corresponding_label_directory(image_directory)
        images = list_images(image_directory)

        expected_count = EXPECTED_SPLIT_COUNTS[split_name]
        if len(images) != expected_count:
            raise ValueError(
                f"{split_name}: expected {expected_count} images, "
                f"observed {len(images)}."
            )

        surface_counts = {"AB": 0, "AD": 0}
        stems: set[str] = set()

        for image_path in images:
            stems.add(image_path.stem)
            if "_AB_" in image_path.name:
                surface_counts["AB"] += 1
            elif "_AD_" in image_path.name:
                surface_counts["AD"] += 1
            else:
                raise ValueError(
                    f"Cannot identify AB/AD surface: {image_path.name}"
                )

            label_path = label_directory / f"{image_path.stem}.txt"
            validate_yolo_label(label_path)

        images_to_check = images if VERIFY_ALL_THREE_CHANNEL_IMAGES else images[:1]
        for image_path in images_to_check:
            validate_three_channel_image(image_path)

        observed_stems[split_name] = stems
        split_summary[split_name] = {
            "images": len(images),
            "surfaces": surface_counts,
            "image_directory": str(image_directory),
            "label_directory": str(label_directory),
        }

    overlaps = {
        "train_val": observed_stems["train"] & observed_stems["val"],
        "train_test": observed_stems["train"] & observed_stems["test"],
        "val_test": observed_stems["val"] & observed_stems["test"],
    }
    if any(overlaps.values()):
        raise ValueError(f"Image names overlap between splits: {overlaps}")

    if CACHE_MODE == "disk":
        free_gib = shutil.disk_usage(dataset_root).free / (1024**3)
        if free_gib < MINIMUM_FREE_DISK_GIB:
            raise RuntimeError(
                f"Only {free_gib:.1f} GiB is free. Set CACHE_MODE=False "
                "or free more SSD space before training."
            )
    else:
        free_gib = shutil.disk_usage(dataset_root).free / (1024**3)

    if not torch.backends.mps.is_built():
        raise RuntimeError("This PyTorch installation was not built with MPS.")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this environment.")

    output_directory = (OUTPUT_PROJECT / RUN_NAME).resolve()
    if output_directory.exists():
        raise FileExistsError(
            f"Output already exists: {output_directory}\n"
            "Choose a new RUN_NAME rather than overwriting it."
        )

    return {
        "dataset_root": str(dataset_root),
        "dataset_yaml": str(DATASET_YAML),
        "classes": EXPECTED_CLASSES,
        "splits": split_summary,
        "free_disk_gib": round(free_gib, 2),
        "expected_output_directory": str(output_directory),
    }


def make_json_safe(value: Any) -> Any:
    """Convert nested Python objects into JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def train_final_model() -> Path:
    """Run preflight checks, train the detector, and save provenance."""
    dataset_summary = validate_dataset()

    software_versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }

    print("Final dataset preflight passed.")
    print(json.dumps(dataset_summary, indent=2))
    print("Software versions:")
    print(json.dumps(software_versions, indent=2))
    print(f"Training arguments:\n{json.dumps(TRAINING_ARGUMENTS, indent=2)}")
    print("The test split will not be evaluated during training.")

    # Loading a .pt checkpoint initializes from pretrained model weights.
    model = YOLO(MODEL_WEIGHTS, task="detect")
    model.train(**TRAINING_ARGUMENTS)

    actual_output_directory = Path(model.trainer.save_dir).resolve()
    provenance = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "model": MODEL_WEIGHTS,
        "task": "detect",
        "candidate_final_checkpoint": str(
            actual_output_directory / "weights" / "best.pt"
        ),
        "actual_output_directory": str(actual_output_directory),
        "software_versions": software_versions,
        "dataset": dataset_summary,
        "training_arguments": TRAINING_ARGUMENTS,
    }

    provenance_path = actual_output_directory / "training_provenance.json"
    with provenance_path.open("x", encoding="utf-8") as handle:
        json.dump(make_json_safe(provenance), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Training completed: {actual_output_directory}")
    print(
        "Candidate final checkpoint: "
        f"{actual_output_directory / 'weights' / 'best.pt'}"
    )
    return actual_output_directory


if __name__ == "__main__":
    train_final_model()
