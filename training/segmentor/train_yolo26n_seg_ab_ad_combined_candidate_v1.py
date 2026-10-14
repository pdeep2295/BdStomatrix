"""Train the combined AB+AD YOLO26n stomatal instance-segmentation model.

This script is intended to be run manually in the ``stoma`` conda
environment. It validates the combined slide-grouped dataset, trains using
train and validation crops only, and keeps the independent test split hidden.

Dataset composition
-------------------
The combined dataset contains 438 fully annotated crops:

* 303 training crops from 36 slides (147 AB and 156 AD)
* 66 validation crops from 8 slides (31 AB and 35 AD)
* 69 reserved test crops from 8 slides (33 AB and 36 AD)

Every included crop has a matching JSON annotation and a non-empty YOLO-seg
label containing exactly two ``guard_cell`` and two ``subsidary_cell``
instances. Empty labels for skipped fallback or guard-only crops are not part
of this dataset.

The run starts from the original pretrained ``yolo26n-seg.pt`` weights. It
does not start from the AB-only pilot checkpoint, avoiding an undocumented AB
initialization advantage and keeping the combined experiment interpretable.

Preflight without training
--------------------------
    /Users/pawandeepsingh/miniconda3/envs/stoma/bin/python \
        /Users/pawandeepsingh/Documents/Postdoc/YOLO_model/segmentation/train_yolo26n_seg_ab_ad_combined_candidate_v1.py \
        --preflight-only

Training
--------
    /Users/pawandeepsingh/miniconda3/envs/stoma/bin/python \
        /Users/pawandeepsingh/Documents/Postdoc/YOLO_model/segmentation/train_yolo26n_seg_ab_ad_combined_candidate_v1.py

This file does not evaluate the independent test split. AB and AD validation
analysis and the final independent test require separate scripts after the
checkpoint, confidence threshold, structured selection and QC rules are
locked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

# Do not silently execute unsupported MPS operations on the CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import cv2
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO


# =============================================================================
# 1. Confirmed paths and locked dataset identity
# =============================================================================

PROJECT_ROOT = Path(
    "/Users/pawandeepsingh/Documents/Postdoc/YOLO_model"
).resolve()
SEGMENTATION_ROOT = (PROJECT_ROOT / "segmentation").resolve()
DATASET_ROOT = (
    SEGMENTATION_ROOT / "ab_ad_yoloseg_combined_v1"
).resolve()
TRAINING_YAML = (DATASET_ROOT / "dataset_trainval.yaml").resolve()
DATASET_YAML_WITH_TEST = (DATASET_ROOT / "dataset.yaml").resolve()
SPLIT_MANIFEST = (DATASET_ROOT / "split_manifest.csv").resolve()

OUTPUT_PROJECT = (SEGMENTATION_ROOT / "Training_Results").resolve()
RUN_NAME = "yolo26n_seg_AB_AD_combined_candidate_v1"
EXPECTED_RUN_DIRECTORY = (OUTPUT_PROJECT / RUN_NAME).resolve()

MODEL_WEIGHTS = "yolo26n-seg.pt"
REVIEWED_ULTRALYTICS_VERSION = "8.4.6"

EXPECTED_CLASSES = {0: "guard_cell", 1: "subsidary_cell"}
EXPECTED_INSTANCES_PER_IMAGE = {0: 2, 1: 2}
EXPECTED_SPLIT_COUNTS = {"train": 303, "val": 66, "test": 69}
EXPECTED_SLIDE_COUNTS = {"train": 36, "val": 8, "test": 8}
EXPECTED_SURFACE_COUNTS = {
    "train": {"AB": 147, "AD": 156},
    "val": {"AB": 31, "AD": 35},
    "test": {"AB": 33, "AD": 36},
}
EXPECTED_SOURCE_GROUP_COUNTS = {
    "AB_full_good_standardized": 200,
    "AD_full_good_standardized_original": 200,
    "AD_fallback_json_annotated_only": 21,
    "AB_guard_only_reaudited_full_good_json_only": 11,
    "AD_guard_only_reaudited_full_good_json_only": 6,
}
IMAGE_SUFFIXES = {".tif", ".tiff"}

# Leave as None for a new run. If an interrupted run must be resumed, replace
# None with this run's exact .../weights/last.pt path. Do not resume from
# best.pt, the AB-only pilot, or another experiment.
RESUME_FROM: Path | None = None
# Example:
# RESUME_FROM = EXPECTED_RUN_DIRECTORY / "weights" / "last.pt"


# =============================================================================
# 2. Training settings
# =============================================================================

TRAINING_ARGUMENTS: dict[str, Any] = {
    # Dataset and output
    "data": str(TRAINING_YAML),
    "project": str(OUTPUT_PROJECT),
    "name": RUN_NAME,
    "exist_ok": False,

    # Core training. These reproduce the successful AB version-2 experiment.
    "epochs": 100,
    "imgsz": 320,
    "batch": 8,
    "device": "mps",
    "workers": 2,
    "cache": False,
    "rect": False,
    "fraction": 1.0,

    # Optimization
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "patience": 25,
    "cos_lr": False,

    # Reproducibility and MPS stability
    "seed": 42,
    "deterministic": True,
    "amp": False,
    "multi_scale": 0.0,
    "compile": False,

    # Instance segmentation
    # Keep every cell as a separate training instance.
    "overlap_mask": False,
    # A finer mask target than the Ultralytics default of 4 is retained because
    # the downstream objective is calibrated morphometry.
    "mask_ratio": 2,
    "single_cls": False,

    # Installed Ultralytics 8.4.6 defaults, stated for provenance. The detector
    # experiment's custom box=10 and cls=2 do not transfer to this balanced
    # instance-segmentation problem.
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,

    # Conservative DIC augmentation
    "degrees": 45.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.2,
    "translate": 0.05,
    "scale": 0.2,
    "shear": 0.0,
    "perspective": 0.0,
    "bgr": 0.0,

    # Disable augmentations that would create artificial multi-stoma scenes or
    # synthetic cell boundaries.
    "mosaic": 0.0,
    "close_mosaic": 0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,

    # Validation, checkpoints and plots
    "val": True,
    "split": "val",
    "save": True,
    "save_period": 10,
    "plots": True,
    "profile": False,
    "pretrained": True,
    "verbose": True,
}


# =============================================================================
# 3. Dataset validation helpers
# =============================================================================


def sha256_file(path: Path) -> str:
    """Return a SHA-256 checksum without modifying the file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def list_images(directory: Path) -> list[Path]:
    """Return TIFF images in deterministic filename order."""
    if not directory.is_dir():
        raise NotADirectoryError(f"Image directory not found:\n{directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def list_labels(directory: Path) -> list[Path]:
    """Return YOLO text labels in deterministic filename order."""
    if not directory.is_dir():
        raise NotADirectoryError(f"Label directory not found:\n{directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    )


def slide_and_surface_from_stem(stem: str) -> tuple[str, str]:
    """Extract the slide ID and AB/AD surface from a crop filename."""
    fields = stem.split("_")
    if (
        len(fields) < 5
        or not fields[0].startswith("S")
        or fields[1] not in {"AB", "AD"}
    ):
        raise ValueError(f"Unexpected crop filename: {stem}")
    return fields[0], fields[1]


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Calculate normalized polygon area with the shoelace formula."""
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )


def validate_yolo_seg_label(label_path: Path) -> dict[str, Any]:
    """Validate one four-instance YOLO-seg label without changing it."""
    class_counts: Counter[int] = Counter()
    duplicate_point_rows: list[int] = []
    point_counts: list[int] = []

    with label_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if len(lines) != 4:
        raise ValueError(
            f"{label_path}: expected four polygon rows, observed {len(lines)}."
        )

    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2 != 0:
            raise ValueError(
                f"{label_path}, line {line_number}: expected a class ID "
                "followed by at least three x/y coordinate pairs."
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
                f"{label_path}, line {line_number}: unexpected class {class_id}."
            )
        if any(not math.isfinite(value) for value in coordinates):
            raise ValueError(
                f"{label_path}, line {line_number}: non-finite coordinate."
            )
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(
                f"{label_path}, line {line_number}: coordinates must be in [0, 1]."
            )

        points = list(zip(coordinates[0::2], coordinates[1::2]))
        if polygon_area(points) <= 0.0:
            raise ValueError(
                f"{label_path}, line {line_number}: polygon has zero area."
            )
        if any(
            points[index] == points[(index + 1) % len(points)]
            for index in range(len(points))
        ):
            duplicate_point_rows.append(line_number)

        class_counts[class_id] += 1
        point_counts.append(len(points))

    if dict(class_counts) != EXPECTED_INSTANCES_PER_IMAGE:
        raise ValueError(
            f"{label_path}: expected {EXPECTED_INSTANCES_PER_IMAGE}, "
            f"observed {dict(class_counts)}."
        )

    return {
        "class_counts": dict(class_counts),
        "duplicate_point_rows": duplicate_point_rows,
        "minimum_points": min(point_counts),
        "maximum_points": max(point_counts),
    }


def validate_three_channel_crop(image_path: Path) -> tuple[int, int]:
    """Require an 8-bit TIFF containing three identical grayscale channels."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read:\n{image_path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"{image_path}: expected three channels, observed {image.shape}."
        )
    if image.dtype != np.uint8:
        raise ValueError(
            f"{image_path}: expected uint8, observed {image.dtype}."
        )
    if not (
        np.array_equal(image[:, :, 0], image[:, :, 1])
        and np.array_equal(image[:, :, 0], image[:, :, 2])
    ):
        raise ValueError(f"Image channels are not identical:\n{image_path}")
    height, width = image.shape[:2]
    return width, height


def load_manifest() -> list[dict[str, str]]:
    """Read and validate the immutable combined-dataset manifest."""
    if not SPLIT_MANIFEST.is_file():
        raise FileNotFoundError(f"Split manifest not found:\n{SPLIT_MANIFEST}")
    with SPLIT_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    expected_total = sum(EXPECTED_SPLIT_COUNTS.values())
    if len(rows) != expected_total:
        raise ValueError(
            f"Manifest expected {expected_total} rows, observed {len(rows)}."
        )
    stems = [row["crop_id"] for row in rows]
    if len(stems) != len(set(stems)):
        duplicates = sorted(
            stem for stem, count in Counter(stems).items() if count > 1
        )
        raise ValueError(f"Duplicate crop IDs in manifest: {duplicates}")

    source_groups = Counter(row["inclusion_group"] for row in rows)
    if dict(source_groups) != EXPECTED_SOURCE_GROUP_COUNTS:
        raise ValueError(
            "Manifest inclusion-group counts differ from the locked dataset.\n"
            f"Expected: {EXPECTED_SOURCE_GROUP_COUNTS}\n"
            f"Observed: {dict(source_groups)}"
        )
    if any(row["json_annotation_present"] != "True" for row in rows):
        raise ValueError("Manifest contains a crop without a JSON annotation.")
    return rows


def validate_split(
    split: str,
    manifest_rows: list[dict[str, str]],
    inspect_contents: bool,
) -> dict[str, Any]:
    """Validate one split while leaving test pixels and polygons unread."""
    image_directory = DATASET_ROOT / "images" / split
    label_directory = DATASET_ROOT / "labels" / split
    images = list_images(image_directory)
    labels = list_labels(label_directory)
    expected_count = EXPECTED_SPLIT_COUNTS[split]

    if len(images) != expected_count or len(labels) != expected_count:
        raise ValueError(
            f"{split}: expected {expected_count} images and labels; observed "
            f"{len(images)} images and {len(labels)} labels."
        )

    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    manifest_stems = {
        row["crop_id"] for row in manifest_rows if row["split"] == split
    }
    if image_stems != label_stems or image_stems != manifest_stems:
        raise ValueError(
            f"{split}: image, label and manifest membership do not agree."
        )

    slide_ids: set[str] = set()
    surface_counts: Counter[str] = Counter()
    for stem in sorted(image_stems):
        slide_id, surface = slide_and_surface_from_stem(stem)
        slide_ids.add(slide_id)
        surface_counts[surface] += 1

    if len(slide_ids) != EXPECTED_SLIDE_COUNTS[split]:
        raise ValueError(
            f"{split}: expected {EXPECTED_SLIDE_COUNTS[split]} slides, "
            f"observed {len(slide_ids)}."
        )
    if dict(surface_counts) != EXPECTED_SURFACE_COUNTS[split]:
        raise ValueError(
            f"{split}: expected surface counts {EXPECTED_SURFACE_COUNTS[split]}, "
            f"observed {dict(surface_counts)}."
        )

    summary: dict[str, Any] = {
        "images": len(images),
        "labels": len(labels),
        "slides": len(slide_ids),
        "slide_ids": sorted(slide_ids),
        "surface_counts": dict(surface_counts),
        "image_directory": str(image_directory),
        "label_directory": str(label_directory),
    }

    if not inspect_contents:
        summary["content_policy"] = (
            "Reserved test pixels and polygon coordinates were not read. "
            "Only filenames, pairing, counts and slide/surface membership "
            "were checked."
        )
        return summary

    manifest_by_stem = {
        row["crop_id"]: row
        for row in manifest_rows
        if row["split"] == split
    }
    duplicate_point_warnings: list[dict[str, Any]] = []
    widths: list[int] = []
    heights: list[int] = []
    total_class_counts: Counter[int] = Counter()

    for image_path in images:
        label_path = label_directory / f"{image_path.stem}.txt"
        label_summary = validate_yolo_seg_label(label_path)
        total_class_counts.update(label_summary["class_counts"])
        if label_summary["duplicate_point_rows"]:
            duplicate_point_warnings.append(
                {
                    "label": str(label_path),
                    "rows": label_summary["duplicate_point_rows"],
                }
            )
        width, height = validate_three_channel_crop(image_path)
        widths.append(width)
        heights.append(height)

        manifest_row = manifest_by_stem[image_path.stem]
        if sha256_file(image_path) != manifest_row["image_sha256"]:
            raise ValueError(f"Image checksum mismatch: {image_path}")
        if sha256_file(label_path) != manifest_row["label_sha256"]:
            raise ValueError(f"Label checksum mismatch: {label_path}")

    expected_instances = {
        class_id: expected_count * count
        for class_id, count in EXPECTED_INSTANCES_PER_IMAGE.items()
    }
    if dict(total_class_counts) != expected_instances:
        raise ValueError(
            f"{split}: expected instance totals {expected_instances}; "
            f"observed {dict(total_class_counts)}."
        )

    summary.update(
        {
            "instances": {
                EXPECTED_CLASSES[class_id]: total_class_counts[class_id]
                for class_id in EXPECTED_CLASSES
            },
            "width_range_pixels": [min(widths), max(widths)],
            "height_range_pixels": [min(heights), max(heights)],
            "duplicate_point_warnings": duplicate_point_warnings,
            "destination_checksums_match_manifest": True,
        }
    )
    return summary


def verify_training_yaml() -> None:
    """Require a train/validation-only YAML that does not expose test."""
    if not TRAINING_YAML.is_file():
        raise FileNotFoundError(f"Training YAML not found:\n{TRAINING_YAML}")
    with TRAINING_YAML.open("r", encoding="utf-8") as handle:
        observed = yaml.safe_load(handle)
    expected = {
        "path": str(DATASET_ROOT),
        "train": "images/train",
        "val": "images/val",
        "names": EXPECTED_CLASSES,
    }
    if observed != expected:
        raise ValueError(
            f"Training YAML differs from the reviewed content:\n{TRAINING_YAML}"
        )
    if "test" in observed:
        raise ValueError("Test must not appear in the training YAML.")


def validate_dataset_and_environment() -> dict[str, Any]:
    """Run safeguards without training or evaluating the test split."""
    required = [
        DATASET_ROOT,
        TRAINING_YAML,
        DATASET_YAML_WITH_TEST,
        SPLIT_MANIFEST,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required combined-dataset files missing:\n" + "\n".join(missing))

    if ultralytics.__version__ != REVIEWED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            "Ultralytics changed since this script was reviewed.\n"
            f"Reviewed: {REVIEWED_ULTRALYTICS_VERSION}\n"
            f"Installed: {ultralytics.__version__}\n"
            "Review compatibility before training rather than continuing silently."
        )

    verify_training_yaml()
    manifest_rows = load_manifest()
    split_summaries = {
        "train": validate_split("train", manifest_rows, inspect_contents=True),
        "val": validate_split("val", manifest_rows, inspect_contents=True),
        "test": validate_split("test", manifest_rows, inspect_contents=False),
    }

    slide_sets = {
        split: set(summary["slide_ids"])
        for split, summary in split_summaries.items()
    }
    overlaps = {
        "train_val": sorted(slide_sets["train"] & slide_sets["val"]),
        "train_test": sorted(slide_sets["train"] & slide_sets["test"]),
        "val_test": sorted(slide_sets["val"] & slide_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Slide leakage detected: {overlaps}")

    if not torch.backends.mps.is_built():
        raise RuntimeError("This PyTorch installation was not built with MPS.")
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "Apple MPS is unavailable in this Python session. Run in the "
            "stoma environment used previously. CPU fallback is disabled."
        )

    if RESUME_FROM is None:
        if EXPECTED_RUN_DIRECTORY.exists():
            raise FileExistsError(
                f"Output directory already exists:\n{EXPECTED_RUN_DIRECTORY}\n"
                "Choose a new RUN_NAME or explicitly configure RESUME_FROM."
            )
    else:
        resume_path = RESUME_FROM.expanduser().resolve()
        expected_last = (EXPECTED_RUN_DIRECTORY / "weights" / "last.pt").resolve()
        if resume_path != expected_last:
            raise ValueError(
                "RESUME_FROM must be this combined run's exact last.pt.\n"
                f"Expected: {expected_last}\nObserved: {resume_path}"
            )
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found:\n{resume_path}")

    return {
        "dataset_root": str(DATASET_ROOT),
        "classes": EXPECTED_CLASSES,
        "splits": split_summaries,
        "slide_overlap": overlaps,
        "split_manifest": str(SPLIT_MANIFEST),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "training_yaml": str(TRAINING_YAML),
        "training_yaml_sha256": sha256_file(TRAINING_YAML),
        "training_test_policy": (
            "Training receives dataset_trainval.yaml only. Test pixels, test "
            "polygon coordinates and test metrics are not accessed."
        ),
        "expected_output_directory": str(EXPECTED_RUN_DIRECTORY),
    }


def make_json_safe(value: Any) -> Any:
    """Convert nested values into JSON-compatible types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def software_versions() -> dict[str, str]:
    """Collect software versions for display and provenance."""
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
    }


# =============================================================================
# 4. Training entry point
# =============================================================================


def train_combined_ab_ad_candidate(preflight_only: bool = False) -> Path | None:
    """Validate, optionally train from pretrained weights, and document it."""
    dataset_summary = validate_dataset_and_environment()
    versions = software_versions()

    print("Combined AB+AD YOLO-seg preflight passed.")
    print(json.dumps(make_json_safe(dataset_summary), indent=2))
    print("Software versions:")
    print(json.dumps(versions, indent=2))
    print("Training arguments:")
    print(json.dumps(make_json_safe(TRAINING_ARGUMENTS), indent=2))
    print("The independent test split is not available to this training run.")

    if preflight_only:
        print("Preflight-only mode completed. Training was not started.")
        return None

    if RESUME_FROM is None:
        model = YOLO(MODEL_WEIGHTS, task="segment")
        model.train(**TRAINING_ARGUMENTS)
    else:
        model = YOLO(str(RESUME_FROM), task="segment")
        model.train(resume=True)

    actual_output_directory = Path(model.trainer.save_dir).resolve()
    if actual_output_directory != EXPECTED_RUN_DIRECTORY:
        raise RuntimeError(
            "Ultralytics wrote to an unexpected directory.\n"
            f"Expected: {EXPECTED_RUN_DIRECTORY}\n"
            f"Observed: {actual_output_directory}"
        )

    best_checkpoint = actual_output_directory / "weights" / "best.pt"
    last_checkpoint = actual_output_directory / "weights" / "last.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(
            f"Training returned without best.pt:\n{best_checkpoint}"
        )
    if not last_checkpoint.is_file():
        raise FileNotFoundError(
            f"Training returned without last.pt:\n{last_checkpoint}"
        )

    provenance = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "model": MODEL_WEIGHTS,
        "task": "segment",
        "training_scope": (
            "Joint AB+AD segmentation of two guard-cell and two subsidiary-cell "
            "instances per classifier-selected crop. Includes JSON-annotated "
            "full_good, fallback-origin and re-audited guard_only-origin crops."
        ),
        "initialization": (
            "New run from pretrained yolo26n-seg.pt; not initialized from the "
            "AB-only pilot checkpoint."
        ),
        "candidate_checkpoint": str(best_checkpoint),
        "candidate_checkpoint_sha256": sha256_file(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "last_checkpoint_sha256": sha256_file(last_checkpoint),
        "actual_output_directory": str(actual_output_directory),
        "software_versions": versions,
        "dataset": dataset_summary,
        "training_arguments": TRAINING_ARGUMENTS,
        "resumed_from": str(RESUME_FROM) if RESUME_FROM else None,
        "test_evaluated": False,
    }

    provenance_path = actual_output_directory / "training_provenance.json"
    with provenance_path.open("x", encoding="utf-8") as handle:
        json.dump(make_json_safe(provenance), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Training completed: {actual_output_directory}")
    print(f"Candidate joint AB+AD checkpoint: {best_checkpoint}")
    print("The independent test split remains unevaluated.")
    return actual_output_directory


def parse_arguments() -> argparse.Namespace:
    """Parse the optional safe preflight mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate paths, labels, images, MPS and settings without training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    train_combined_ab_ad_candidate(preflight_only=arguments.preflight_only)
