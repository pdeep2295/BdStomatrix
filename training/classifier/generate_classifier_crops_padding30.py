"""
Generate 30-pixel stomatal crops for crop-quality classifier annotation.

The script uses the completed, locked detector predictions for the 120 copied
classifier-training images. It does NOT rerun the detector or NMS. It:

- reads only ``classifier_main/images_120`` as image input;
- reads the post-NMS class-0 predictions made at confidence 0.40 and NMS 0.70;
- verifies the prediction provenance and all source-image SHA-256 checksums;
- creates one unmodified, uncompressed TIFF crop per prediction;
- uses 30 pixels of requested padding around the integer detector box;
- clips crops at source-image boundaries and records the actual padding;
- records neighboring predicted stomata that could complicate classification or
  later segmentation;
- compares predictions with the copied manual YOLO-HBB labels at IoU >= 0.50
  for descriptive audit metadata only;
- retains unmatched predictions as crops because they are informative possible
  ``bad`` examples for the crop-quality classifier;
- writes manifests, matching summaries, and a provenance record;
- never reads the XAnyLabeling JSON files;
- never modifies source images, predictions, or manual annotations;
- refuses to overwrite an existing final output directory.

Derived crops are ordinary, uncompressed RGB TIFF files. The original/source
OME-TIFF files and their metadata remain untouched. No crop is resized,
normalized, contrast-enhanced, or otherwise altered.

Run this file yourself in the same Python environment used for the previous
crop-pilot scripts. Required packages: numpy, opencv-python, and tifffile.
This file performs crop generation only when run directly.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile


# =====================================================================
# Confirmed project paths and locked settings
# =====================================================================

PROJECT_ROOT = Path(
    "/Users/pawandeepsingh/Documents/Postdoc/YOLO_model"
).resolve()

CLASSIFIER_ROOT = (PROJECT_ROOT / "classifier_main").resolve()
SOURCE_IMAGE_DIR = (CLASSIFIER_ROOT / "images_120").resolve()

PREDICTION_DIR = (
    CLASSIFIER_ROOT / "predictions_conf040_nms070"
).resolve()
PREDICTION_CSV = (
    PREDICTION_DIR / "post_nms_stoma_predictions.csv"
).resolve()
PREDICTION_JSON = (
    PREDICTION_DIR / "post_nms_stoma_predictions.json"
).resolve()
PREDICTION_PROVENANCE = (
    PREDICTION_DIR / "prediction_provenance.json"
).resolve()
PREDICTION_INPUT_MANIFEST = (
    PREDICTION_DIR / "input_image_manifest.csv"
).resolve()

MANUAL_ANNOTATION_ROOT = (
    CLASSIFIER_ROOT / "manual_annotations"
).resolve()
YOLO_LABEL_DIR = (
    MANUAL_ANNOTATION_ROOT / "yolo_hbb_labels"
).resolve()
CLASSES_FILE = (MANUAL_ANNOTATION_ROOT / "classes.txt").resolve()

# This directory is created only when the script is run successfully.
# Change it deliberately before running if you want a different new run name.
FINAL_OUTPUT_DIR = (
    CLASSIFIER_ROOT / "crops_padding30_conf040_nms070"
).resolve()
CROP_SUBDIRECTORY_NAME = "prediction_crops"
AUDIT_SUBDIRECTORY_NAME = "manual_annotation_audit"


# Detector/crop settings already locked in this project.
STOMA_CLASS_ID = 0
STOMA_CLASS_NAME = "stoma"
HAIR_CELL_CLASS_ID = 1
HAIR_CELL_CLASS_NAME = "hair_cell"

CONFIDENCE_THRESHOLD = 0.40
MANUAL_NMS_IOU_THRESHOLD = 0.70
MANUAL_MATCH_IOU_THRESHOLD = 0.50
PADDING_PIXELS = 30

EXPECTED_MODEL_SHA256 = (
    "6fa85a2f895f79777b188a69c9b384b730da3feb6956f4bbd2beb626ad4d2862"
)
EXPECTED_IMAGE_COUNT = 120
EXPECTED_AB_IMAGE_COUNT = 60
EXPECTED_AD_IMAGE_COUNT = 60
EXPECTED_PREDICTION_COUNT = 4546
EXPECTED_AB_PREDICTION_COUNT = 1711
EXPECTED_AD_PREDICTION_COUNT = 2835
EXPECTED_IMAGE_WIDTH = 3072
EXPECTED_IMAGE_HEIGHT = 2048
EXPECTED_CHANNELS = 3
EXPECTED_DTYPE = "uint8"

# Provisional only; this value is recorded, not used to rescale crops.
MICROMETERS_PER_PIXEL = 0.065
CALIBRATION_STATUS = "provisional_unverified_stage_micrometer_required"

FILENAME_PATTERN = re.compile(
    r"^(S\d{3})_(AB|AD)_P([1-5])_EDF\.ome\.tif$"
)


# Output filenames.
CROP_MANIFEST_NAME = "crop_manifest.csv"
PER_IMAGE_SUMMARY_NAME = "crop_summary_per_image.csv"
OVERALL_SUMMARY_NAME = "crop_summary_overall_and_by_surface.csv"
VERIFIED_SOURCE_MANIFEST_NAME = "verified_source_image_manifest.csv"
MANUAL_LABEL_MANIFEST_NAME = "manual_label_manifest.csv"
PREDICTION_MATCHES_NAME = "prediction_annotation_matches.csv"
MISSED_ANNOTATIONS_NAME = "missed_manual_stoma_annotations.csv"
MATCHING_SUMMARY_NAME = "descriptive_matching_summary.csv"
PROVENANCE_NAME = "crop_generation_provenance.json"
RUN_REPORT_NAME = "CLASSIFIER_CROP_GENERATION_REPORT.txt"


# =====================================================================
# General utilities
# =====================================================================

def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 checksum without changing the file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    """Read and validate a JSON object."""
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found:\n{path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object:\n{path}")
    return value


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV file and return its header and rows."""
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV file not found:\n{path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header:\n{path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Write a CSV with a stable column order, even when rows are empty."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def ensure_inside(path: Path, parent: Path) -> None:
    """Reject a resolved path outside the expected parent directory."""
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as error:
        raise ValueError(
            f"Path lies outside the permitted directory:\n{path}\n{parent}"
        ) from error


def require_columns(
    observed: list[str],
    required: set[str],
    path: Path,
) -> None:
    """Check that a CSV contains all required columns."""
    missing = sorted(required - set(observed))
    if missing:
        raise ValueError(f"Missing columns in {path}:\n{missing}")


def parse_bool(value: str, description: str) -> bool:
    """Parse a CSV Boolean value strictly."""
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid Boolean for {description}: {value!r}")


def parse_image_name(image_name: str) -> dict[str, Any]:
    """Extract slide, surface, replicate, and biological-unit fields."""
    match = FILENAME_PATTERN.fullmatch(image_name)
    if match is None:
        raise ValueError(
            "Unexpected source-image filename convention:\n"
            f"{image_name}"
        )
    slide_id = match.group(1)
    surface = match.group(2)
    plant_number = int(match.group(3))
    plant_replicate = f"P{plant_number}"
    return {
        "slide_id": slide_id,
        "surface": surface,
        "plant_number": plant_number,
        "plant_replicate": plant_replicate,
        "biological_unit": f"{slide_id}_{plant_replicate}",
        "source_base_name": image_name.removesuffix(".ome.tif"),
    }


def safe_ratio(numerator: int, denominator: int) -> float:
    """Return a ratio, or zero when its denominator is zero."""
    return float(numerator / denominator) if denominator else 0.0


# =====================================================================
# Prediction and source-image validation
# =====================================================================

def validate_classes_file() -> str:
    """Verify the copied class order: class 0 stoma, class 1 hair_cell."""
    if not CLASSES_FILE.is_file():
        raise FileNotFoundError(f"Classes file not found:\n{CLASSES_FILE}")
    names = [
        line.strip()
        for line in CLASSES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = [STOMA_CLASS_NAME, HAIR_CELL_CLASS_NAME]
    if names != expected:
        raise ValueError(
            f"Expected classes {expected}; observed {names}."
        )
    return sha256_file(CLASSES_FILE)


def validate_prediction_provenance() -> dict[str, Any]:
    """Verify that the input prediction run used the locked settings."""
    provenance = load_json_object(PREDICTION_PROVENANCE)

    model = provenance.get("model", {})
    settings = provenance.get("operational_settings", {})
    counts = provenance.get("result_counts", {})
    restrictions = provenance.get("input_restrictions", {})

    if model.get("sha256") != EXPECTED_MODEL_SHA256:
        raise ValueError("Prediction provenance does not use locked best.pt.")
    if model.get("classes") != {"0": "stoma", "1": "hair_cell"}:
        raise ValueError("Prediction provenance has unexpected class names.")
    if int(settings.get("stoma_class_id", -1)) != STOMA_CLASS_ID:
        raise ValueError("Prediction provenance does not use class 0 stoma.")
    if not math.isclose(
        float(settings.get("confidence_threshold", -1.0)),
        CONFIDENCE_THRESHOLD,
        abs_tol=1e-12,
    ):
        raise ValueError("Prediction confidence threshold is not 0.40.")
    if not math.isclose(
        float(settings.get("manual_nms_iou_threshold", -1.0)),
        MANUAL_NMS_IOU_THRESHOLD,
        abs_tol=1e-12,
    ):
        raise ValueError("Prediction manual-NMS threshold is not 0.70.")

    expected_counts = {
        "predictions_after_nms": EXPECTED_PREDICTION_COUNT,
        "AB_predictions_after_nms": EXPECTED_AB_PREDICTION_COUNT,
        "AD_predictions_after_nms": EXPECTED_AD_PREDICTION_COUNT,
    }
    for key, expected in expected_counts.items():
        if int(counts.get(key, -1)) != expected:
            raise ValueError(
                f"Prediction provenance {key} is {counts.get(key)!r}; "
                f"expected {expected}."
            )

    if int(restrictions.get("image_count", -1)) != EXPECTED_IMAGE_COUNT:
        raise ValueError("Prediction provenance does not describe 120 images.")
    if restrictions.get("test_images_accessed") is not False:
        raise ValueError("Prediction provenance says test images were accessed.")
    if restrictions.get("validation_images_accessed") is not False:
        raise ValueError(
            "Prediction provenance says validation images were accessed."
        )
    if restrictions.get("hidden_phenotyping_images_accessed") is not False:
        raise ValueError(
            "Prediction provenance says hidden phenotyping images were accessed."
        )

    return provenance


def load_prediction_input_manifest() -> dict[str, dict[str, Any]]:
    """Load the source-image manifest created during detector inference."""
    header, raw_rows = read_csv_rows(PREDICTION_INPUT_MANIFEST)
    required = {
        "image_name",
        "surface",
        "slide_id",
        "plant_replicate",
        "biological_unit",
        "image_sha256",
        "image_width",
        "image_height",
        "channels",
        "dtype",
        "three_channels_identical",
    }
    require_columns(header, required, PREDICTION_INPUT_MANIFEST)

    if len(raw_rows) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            "Prediction input manifest does not contain exactly 120 rows."
        )

    by_name: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        image_name = row["image_name"]
        if image_name in by_name:
            raise ValueError(f"Duplicate input-manifest image: {image_name}")
        metadata = parse_image_name(image_name)
        if row["surface"] != metadata["surface"]:
            raise ValueError(f"Surface mismatch for {image_name}.")
        if row["slide_id"] != metadata["slide_id"]:
            raise ValueError(f"Slide mismatch for {image_name}.")
        if row["plant_replicate"] != metadata["plant_replicate"]:
            raise ValueError(f"Replicate mismatch for {image_name}.")
        if row["biological_unit"] != metadata["biological_unit"]:
            raise ValueError(f"Biological-unit mismatch for {image_name}.")

        parsed = {
            **metadata,
            "image_name": image_name,
            "image_sha256": row["image_sha256"],
            "image_width": int(row["image_width"]),
            "image_height": int(row["image_height"]),
            "channels": int(row["channels"]),
            "dtype": row["dtype"],
            "three_channels_identical": parse_bool(
                row["three_channels_identical"],
                f"three_channels_identical for {image_name}",
            ),
        }
        expected_dimensions = (
            EXPECTED_IMAGE_WIDTH,
            EXPECTED_IMAGE_HEIGHT,
            EXPECTED_CHANNELS,
            EXPECTED_DTYPE,
            True,
        )
        observed_dimensions = (
            parsed["image_width"],
            parsed["image_height"],
            parsed["channels"],
            parsed["dtype"],
            parsed["three_channels_identical"],
        )
        if observed_dimensions != expected_dimensions:
            raise ValueError(
                f"Unexpected input-manifest image properties for {image_name}: "
                f"{observed_dimensions}."
            )
        by_name[image_name] = parsed

    surfaces = Counter(row["surface"] for row in by_name.values())
    if surfaces != Counter(
        {"AB": EXPECTED_AB_IMAGE_COUNT, "AD": EXPECTED_AD_IMAGE_COUNT}
    ):
        raise ValueError(f"Unexpected image surface counts: {dict(surfaces)}")
    return by_name


def load_and_validate_predictions(
    input_manifest: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Load typed post-NMS stoma predictions and validate their identities."""
    header, raw_rows = read_csv_rows(PREDICTION_CSV)
    required = {
        "image_name",
        "surface",
        "slide_id",
        "plant_replicate",
        "biological_unit",
        "source_image_sha256",
        "stoma_number",
        "stoma_id",
        "class_id",
        "class_name",
        "confidence",
        "x1_pixels",
        "y1_pixels",
        "x2_pixels",
        "y2_pixels",
        "center_x_pixels",
        "center_y_pixels",
        "box_width_pixels",
        "box_height_pixels",
        "source_image_width",
        "source_image_height",
    }
    require_columns(header, required, PREDICTION_CSV)

    if len(raw_rows) != EXPECTED_PREDICTION_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PREDICTION_COUNT} predictions; "
            f"observed {len(raw_rows)}."
        )

    predictions: list[dict[str, Any]] = []
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()

    for raw in raw_rows:
        image_name = raw["image_name"]
        if image_name not in input_manifest:
            raise ValueError(f"Prediction refers to unknown image: {image_name}")
        image_metadata = input_manifest[image_name]

        record = {
            "image_name": image_name,
            "surface": raw["surface"],
            "slide_id": raw["slide_id"],
            "plant_replicate": raw["plant_replicate"],
            "biological_unit": raw["biological_unit"],
            "source_image_sha256": raw["source_image_sha256"],
            "stoma_number": int(raw["stoma_number"]),
            "stoma_id": raw["stoma_id"],
            "class_id": int(raw["class_id"]),
            "class_name": raw["class_name"],
            "confidence": float(raw["confidence"]),
            "x1": float(raw["x1_pixels"]),
            "y1": float(raw["y1_pixels"]),
            "x2": float(raw["x2_pixels"]),
            "y2": float(raw["y2_pixels"]),
            "center_x": float(raw["center_x_pixels"]),
            "center_y": float(raw["center_y_pixels"]),
            "box_width": float(raw["box_width_pixels"]),
            "box_height": float(raw["box_height_pixels"]),
            "source_width": int(raw["source_image_width"]),
            "source_height": int(raw["source_image_height"]),
        }

        identity_fields = [
            "surface",
            "slide_id",
            "plant_replicate",
            "biological_unit",
        ]
        for field in identity_fields:
            if record[field] != image_metadata[field]:
                raise ValueError(
                    f"Prediction {field} mismatch for {record['stoma_id']}."
                )
        if record["source_image_sha256"] != image_metadata["image_sha256"]:
            raise ValueError(
                f"Prediction source checksum mismatch for {record['stoma_id']}."
            )
        if record["class_id"] != STOMA_CLASS_ID:
            raise ValueError(f"Non-stoma class in prediction CSV: {record}")
        if record["class_name"] != STOMA_CLASS_NAME:
            raise ValueError(f"Unexpected prediction class name: {record}")
        if not CONFIDENCE_THRESHOLD <= record["confidence"] <= 1.0:
            raise ValueError(
                f"Prediction confidence outside [0.40, 1]: {record}"
            )
        if (record["source_width"], record["source_height"]) != (
            EXPECTED_IMAGE_WIDTH,
            EXPECTED_IMAGE_HEIGHT,
        ):
            raise ValueError(f"Unexpected prediction dimensions: {record}")
        if not (
            0.0 <= record["x1"] < record["x2"] <= record["source_width"]
            and 0.0
            <= record["y1"]
            < record["y2"]
            <= record["source_height"]
        ):
            raise ValueError(f"Invalid prediction box: {record}")

        expected_center_x = (record["x1"] + record["x2"]) / 2.0
        expected_center_y = (record["y1"] + record["y2"]) / 2.0
        expected_width = record["x2"] - record["x1"]
        expected_height = record["y2"] - record["y1"]
        numeric_checks = [
            (record["center_x"], expected_center_x, "center_x"),
            (record["center_y"], expected_center_y, "center_y"),
            (record["box_width"], expected_width, "box_width"),
            (record["box_height"], expected_height, "box_height"),
        ]
        for observed, expected, description in numeric_checks:
            if not math.isclose(observed, expected, abs_tol=1e-5):
                raise ValueError(
                    f"Inconsistent {description} for {record['stoma_id']}."
                )

        expected_id = (
            f"{image_metadata['source_base_name']}_stoma_"
            f"{record['stoma_number']:04d}"
        )
        if record["stoma_id"] != expected_id:
            raise ValueError(
                f"Unexpected deterministic stoma ID: {record['stoma_id']}"
            )
        if record["stoma_id"] in seen_ids:
            raise ValueError(f"Duplicate stoma ID: {record['stoma_id']}")
        seen_ids.add(record["stoma_id"])
        predictions.append(record)
        by_image[image_name].append(record)

    if set(by_image) != set(input_manifest):
        missing = sorted(set(input_manifest) - set(by_image))
        extra = sorted(set(by_image) - set(input_manifest))
        raise ValueError(
            f"Prediction/source image-set mismatch. Missing={missing}; extra={extra}"
        )

    for image_name, image_predictions in by_image.items():
        image_predictions.sort(key=lambda item: item["stoma_number"])
        observed_numbers = [item["stoma_number"] for item in image_predictions]
        expected_numbers = list(range(1, len(image_predictions) + 1))
        if observed_numbers != expected_numbers:
            raise ValueError(
                f"Nonconsecutive stoma numbers for {image_name}."
            )

    surface_counts = Counter(item["surface"] for item in predictions)
    expected_surface_counts = Counter(
        {
            "AB": EXPECTED_AB_PREDICTION_COUNT,
            "AD": EXPECTED_AD_PREDICTION_COUNT,
        }
    )
    if surface_counts != expected_surface_counts:
        raise ValueError(
            f"Unexpected prediction surface counts: {dict(surface_counts)}"
        )

    # Confirm the completed NMS result has no residual pair above IoU 0.70.
    for image_name, image_predictions in by_image.items():
        for first_index, first in enumerate(image_predictions):
            first_box = np.array(
                [first["x1"], first["y1"], first["x2"], first["y2"]],
                dtype=np.float64,
            )
            for second in image_predictions[first_index + 1 :]:
                second_box = np.array(
                    [second["x1"], second["y1"], second["x2"], second["y2"]],
                    dtype=np.float64,
                )
                overlap = calculate_iou(first_box, second_box)
                if overlap > MANUAL_NMS_IOU_THRESHOLD + 1e-12:
                    raise ValueError(
                        "Residual prediction pair exceeds manual NMS 0.70 in "
                        f"{image_name}: {first['stoma_id']} and "
                        f"{second['stoma_id']}, IoU={overlap:.6f}."
                    )

    return predictions, dict(by_image)


def verify_source_images(
    input_manifest: dict[str, dict[str, Any]],
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """Verify source identities, file checksums, shape, channels, and dtype."""
    if not SOURCE_IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Source-image directory not found:\n{SOURCE_IMAGE_DIR}"
        )
    visible_files = sorted(
        path
        for path in SOURCE_IMAGE_DIR.iterdir()
        if path.is_file() and not path.name.startswith(".")
    )
    non_tiffs = [
        path.name
        for path in visible_files
        if path.suffix.lower() not in {".tif", ".tiff"}
    ]
    if non_tiffs:
        raise ValueError(f"Unexpected non-TIFF files in images_120: {non_tiffs}")

    image_paths = sorted(
        [path.resolve() for path in visible_files],
        key=lambda item: item.name,
    )
    if len(image_paths) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"Expected 120 source TIFFs; observed {len(image_paths)}."
        )
    if {path.name for path in image_paths} != set(input_manifest):
        raise ValueError("Current source image set differs from inference input.")

    path_by_name: dict[str, Path] = {}
    verified_rows: list[dict[str, Any]] = []
    for image_path in image_paths:
        ensure_inside(image_path, SOURCE_IMAGE_DIR)
        expected = input_manifest[image_path.name]
        observed_hash = sha256_file(image_path)
        if observed_hash != expected["image_sha256"]:
            raise ValueError(
                "Source image changed after prediction (SHA-256 mismatch):\n"
                f"{image_path}"
            )

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"OpenCV could not read:\n{image_path}")
        expected_shape = (
            EXPECTED_IMAGE_HEIGHT,
            EXPECTED_IMAGE_WIDTH,
            EXPECTED_CHANNELS,
        )
        if image.shape != expected_shape:
            raise ValueError(
                f"Unexpected shape for {image_path.name}: {image.shape}."
            )
        if str(image.dtype) != EXPECTED_DTYPE:
            raise ValueError(
                f"Unexpected dtype for {image_path.name}: {image.dtype}."
            )
        identical = bool(
            np.array_equal(image[:, :, 0], image[:, :, 1])
            and np.array_equal(image[:, :, 0], image[:, :, 2])
        )
        if not identical:
            raise ValueError(
                f"Source image channels are not identical: {image_path.name}"
            )

        path_by_name[image_path.name] = image_path
        verified_rows.append(
            {
                "image_name": image_path.name,
                "surface": expected["surface"],
                "slide_id": expected["slide_id"],
                "plant_replicate": expected["plant_replicate"],
                "biological_unit": expected["biological_unit"],
                "source_path": str(image_path),
                "source_image_sha256": observed_hash,
                "image_width": EXPECTED_IMAGE_WIDTH,
                "image_height": EXPECTED_IMAGE_HEIGHT,
                "channels": EXPECTED_CHANNELS,
                "dtype": EXPECTED_DTYPE,
                "three_channels_identical": True,
                "source_file_modified": False,
            }
        )
        del image

    return path_by_name, verified_rows


# =====================================================================
# Manual YOLO-HBB annotations: descriptive audit only
# =====================================================================

def load_manual_stoma_annotations(
    input_manifest: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Read copied YOLO-HBB labels and retain class-0 boxes for audit only."""
    if not YOLO_LABEL_DIR.is_dir():
        raise FileNotFoundError(f"YOLO label directory not found:\n{YOLO_LABEL_DIR}")

    label_paths = sorted(
        path.resolve()
        for path in YOLO_LABEL_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    )
    expected_label_names = {
        f"{Path(image_name).stem}.txt" for image_name in input_manifest
    }
    observed_label_names = {path.name for path in label_paths}
    if observed_label_names != expected_label_names:
        missing = sorted(expected_label_names - observed_label_names)
        extra = sorted(observed_label_names - expected_label_names)
        raise ValueError(
            f"Manual-label image set mismatch. Missing={missing}; extra={extra}"
        )

    annotations_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_manifest_rows: list[dict[str, Any]] = []

    for image_name in sorted(input_manifest):
        label_path = YOLO_LABEL_DIR / f"{Path(image_name).stem}.txt"
        ensure_inside(label_path, YOLO_LABEL_DIR)
        stoma_count = 0
        hair_cell_count = 0
        nonempty_lines = 0

        with label_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                nonempty_lines += 1
                parts = stripped.split()
                if len(parts) != 5:
                    raise ValueError(
                        f"Expected 5 YOLO-HBB fields in {label_path.name}, "
                        f"line {line_number}; observed {len(parts)}."
                    )
                try:
                    class_id = int(parts[0])
                    center_x_n, center_y_n, width_n, height_n = map(
                        float, parts[1:]
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Invalid numeric label in {label_path.name}, "
                        f"line {line_number}."
                    ) from error

                if class_id not in {STOMA_CLASS_ID, HAIR_CELL_CLASS_ID}:
                    raise ValueError(
                        f"Unexpected class {class_id} in {label_path.name}."
                    )
                if not (
                    0.0 <= center_x_n <= 1.0
                    and 0.0 <= center_y_n <= 1.0
                    and 0.0 < width_n <= 1.0
                    and 0.0 < height_n <= 1.0
                ):
                    raise ValueError(
                        f"Invalid normalized box in {label_path.name}, "
                        f"line {line_number}."
                    )

                if class_id == HAIR_CELL_CLASS_ID:
                    hair_cell_count += 1
                    continue

                stoma_count += 1
                width = width_n * EXPECTED_IMAGE_WIDTH
                height = height_n * EXPECTED_IMAGE_HEIGHT
                center_x = center_x_n * EXPECTED_IMAGE_WIDTH
                center_y = center_y_n * EXPECTED_IMAGE_HEIGHT
                x1 = max(0.0, center_x - width / 2.0)
                y1 = max(0.0, center_y - height / 2.0)
                x2 = min(float(EXPECTED_IMAGE_WIDTH), center_x + width / 2.0)
                y2 = min(float(EXPECTED_IMAGE_HEIGHT), center_y + height / 2.0)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(
                        f"Degenerate annotation in {label_path.name}, "
                        f"line {line_number}."
                    )

                source_base = input_manifest[image_name]["source_base_name"]
                annotations_by_image[image_name].append(
                    {
                        "annotation_id": (
                            f"{source_base}_manual_stoma_{stoma_count:04d}"
                        ),
                        "image_name": image_name,
                        "surface": input_manifest[image_name]["surface"],
                        "line_number": line_number,
                        "manual_stoma_number": stoma_count,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "center_x": center_x,
                        "center_y": center_y,
                    }
                )

        label_manifest_rows.append(
            {
                "image_name": image_name,
                "surface": input_manifest[image_name]["surface"],
                "label_filename": label_path.name,
                "label_path": str(label_path.resolve()),
                "label_sha256": sha256_file(label_path),
                "nonempty_annotation_lines": nonempty_lines,
                "stoma_annotations": stoma_count,
                "hair_cell_annotations": hair_cell_count,
                "label_file_modified": False,
            }
        )

    return dict(annotations_by_image), label_manifest_rows


def calculate_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Calculate intersection-over-union for xyxy boxes."""
    intersection_x1 = max(float(box_a[0]), float(box_b[0]))
    intersection_y1 = max(float(box_a[1]), float(box_b[1]))
    intersection_x2 = min(float(box_a[2]), float(box_b[2]))
    intersection_y2 = min(float(box_a[3]), float(box_b[3]))
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
        0.0, float(box_a[3] - box_a[1])
    )
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
        0.0, float(box_b[3] - box_b[1])
    )
    union = area_a + area_b - intersection_area
    return float(intersection_area / union) if union > 0.0 else 0.0


def match_predictions_to_annotations(
    predictions_by_image: dict[str, list[dict[str, Any]]],
    annotations_by_image: dict[str, list[dict[str, Any]]],
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Perform deterministic one-to-one highest-IoU matching at IoU 0.50."""
    match_by_stoma_id: dict[str, dict[str, Any]] = {}
    prediction_match_rows: list[dict[str, Any]] = []
    missed_annotation_rows: list[dict[str, Any]] = []

    for image_name in sorted(predictions_by_image):
        predictions = predictions_by_image[image_name]
        annotations = annotations_by_image.get(image_name, [])
        candidates: list[tuple[float, int, int]] = []

        for prediction_index, prediction in enumerate(predictions):
            prediction_box = np.array(
                [
                    prediction["x1"],
                    prediction["y1"],
                    prediction["x2"],
                    prediction["y2"],
                ],
                dtype=np.float64,
            )
            for annotation_index, annotation in enumerate(annotations):
                annotation_box = np.array(
                    [
                        annotation["x1"],
                        annotation["y1"],
                        annotation["x2"],
                        annotation["y2"],
                    ],
                    dtype=np.float64,
                )
                overlap = calculate_iou(prediction_box, annotation_box)
                if overlap >= MANUAL_MATCH_IOU_THRESHOLD:
                    candidates.append(
                        (overlap, prediction_index, annotation_index)
                    )

        candidates.sort(
            key=lambda item: (
                -item[0],
                predictions[item[1]]["stoma_number"],
                annotations[item[2]]["manual_stoma_number"],
            )
        )
        used_predictions: set[int] = set()
        used_annotations: set[int] = set()

        for overlap, prediction_index, annotation_index in candidates:
            if prediction_index in used_predictions:
                continue
            if annotation_index in used_annotations:
                continue
            used_predictions.add(prediction_index)
            used_annotations.add(annotation_index)
            prediction = predictions[prediction_index]
            annotation = annotations[annotation_index]
            match_by_stoma_id[prediction["stoma_id"]] = {
                "match_status": "matched",
                "matched_annotation_id": annotation["annotation_id"],
                "match_iou": overlap,
                "matched_annotation": annotation,
            }

        for prediction_index, prediction in enumerate(predictions):
            match = match_by_stoma_id.get(prediction["stoma_id"])
            if match is None:
                match = {
                    "match_status": "unmatched_prediction",
                    "matched_annotation_id": "",
                    "match_iou": "",
                    "matched_annotation": None,
                }
                match_by_stoma_id[prediction["stoma_id"]] = match
            prediction_match_rows.append(
                {
                    "image_name": image_name,
                    "surface": prediction["surface"],
                    "stoma_id": prediction["stoma_id"],
                    "confidence": prediction["confidence"],
                    "match_status": match["match_status"],
                    "matched_annotation_id": match["matched_annotation_id"],
                    "match_iou": match["match_iou"],
                    "audit_only": True,
                    "crop_retained_regardless_of_match": True,
                }
            )

        for annotation_index, annotation in enumerate(annotations):
            if annotation_index not in used_annotations:
                missed_annotation_rows.append(
                    {
                        "image_name": image_name,
                        "surface": annotation["surface"],
                        "annotation_id": annotation["annotation_id"],
                        "annotation_line_number": annotation["line_number"],
                        "x1_pixels": annotation["x1"],
                        "y1_pixels": annotation["y1"],
                        "x2_pixels": annotation["x2"],
                        "y2_pixels": annotation["y2"],
                        "audit_only": True,
                        "no_crop_generated_from_manual_box": True,
                    }
                )

    return match_by_stoma_id, prediction_match_rows, missed_annotation_rows


# =====================================================================
# Crop geometry and generation
# =====================================================================

def integer_detection_box(
    prediction: dict[str, Any],
) -> tuple[int, int, int, int]:
    """Convert a float detector box using floor for minima and ceil for maxima."""
    width = int(prediction["source_width"])
    height = int(prediction["source_height"])
    x1 = max(0, min(width, math.floor(prediction["x1"])))
    y1 = max(0, min(height, math.floor(prediction["y1"])))
    x2 = max(0, min(width, math.ceil(prediction["x2"])))
    y2 = max(0, min(height, math.ceil(prediction["y2"])))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Degenerate integer box for {prediction['stoma_id']}."
        )
    return x1, y1, x2, y2


def calculate_crop_geometry(prediction: dict[str, Any]) -> dict[str, Any]:
    """Add fixed padding to the integer box and clip to source boundaries."""
    box_x1, box_y1, box_x2, box_y2 = integer_detection_box(prediction)
    requested_x1 = box_x1 - PADDING_PIXELS
    requested_y1 = box_y1 - PADDING_PIXELS
    requested_x2 = box_x2 + PADDING_PIXELS
    requested_y2 = box_y2 + PADDING_PIXELS
    source_width = int(prediction["source_width"])
    source_height = int(prediction["source_height"])

    crop_x1 = max(0, requested_x1)
    crop_y1 = max(0, requested_y1)
    crop_x2 = min(source_width, requested_x2)
    crop_y2 = min(source_height, requested_y2)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError(f"Empty crop for {prediction['stoma_id']}.")

    actual_left = box_x1 - crop_x1
    actual_top = box_y1 - crop_y1
    actual_right = crop_x2 - box_x2
    actual_bottom = crop_y2 - box_y2
    edge_clipped = any(
        value < PADDING_PIXELS
        for value in (actual_left, actual_top, actual_right, actual_bottom)
    )

    return {
        "integer_box_x1": box_x1,
        "integer_box_y1": box_y1,
        "integer_box_x2": box_x2,
        "integer_box_y2": box_y2,
        "requested_crop_x1": requested_x1,
        "requested_crop_y1": requested_y1,
        "requested_crop_x2": requested_x2,
        "requested_crop_y2": requested_y2,
        "crop_x1": crop_x1,
        "crop_y1": crop_y1,
        "crop_x2": crop_x2,
        "crop_y2": crop_y2,
        "crop_width": crop_x2 - crop_x1,
        "crop_height": crop_y2 - crop_y1,
        "actual_padding_left": actual_left,
        "actual_padding_top": actual_top,
        "actual_padding_right": actual_right,
        "actual_padding_bottom": actual_bottom,
        "edge_clipped": edge_clipped,
    }


def center_inside_crop(
    center_x: float,
    center_y: float,
    geometry: dict[str, Any],
) -> bool:
    """Test a center against the half-open crop pixel interval."""
    return bool(
        geometry["crop_x1"] <= center_x < geometry["crop_x2"]
        and geometry["crop_y1"] <= center_y < geometry["crop_y2"]
    )


def box_intersects_crop(
    box: dict[str, Any],
    geometry: dict[str, Any],
) -> bool:
    """Test whether a float box has positive-area intersection with a crop."""
    overlap_width = min(box["x2"], geometry["crop_x2"]) - max(
        box["x1"], geometry["crop_x1"]
    )
    overlap_height = min(box["y2"], geometry["crop_y2"]) - max(
        box["y1"], geometry["crop_y1"]
    )
    return bool(overlap_width > 0.0 and overlap_height > 0.0)


def fully_contained(
    box: dict[str, Any],
    geometry: dict[str, Any],
) -> bool:
    """Test whether a box lies completely inside crop boundaries."""
    return bool(
        box["x1"] >= geometry["crop_x1"]
        and box["y1"] >= geometry["crop_y1"]
        and box["x2"] <= geometry["crop_x2"]
        and box["y2"] <= geometry["crop_y2"]
    )


def prepare_crop_records(
    predictions: list[dict[str, Any]],
    predictions_by_image: dict[str, list[dict[str, Any]]],
    annotations_by_image: dict[str, list[dict[str, Any]]],
    match_by_stoma_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Calculate geometry and neighboring-stoma audit fields for every crop."""
    records: list[dict[str, Any]] = []

    for prediction in predictions:
        geometry = calculate_crop_geometry(prediction)
        image_predictions = predictions_by_image[prediction["image_name"]]
        other_center_ids: list[str] = []
        other_intersection_ids: list[str] = []

        for other in image_predictions:
            if other["stoma_id"] == prediction["stoma_id"]:
                continue
            if center_inside_crop(other["center_x"], other["center_y"], geometry):
                other_center_ids.append(other["stoma_id"])
            if box_intersects_crop(other, geometry):
                other_intersection_ids.append(other["stoma_id"])

        match = match_by_stoma_id[prediction["stoma_id"]]
        matched_annotation = match["matched_annotation"]
        matched_annotation_id = match["matched_annotation_id"]
        other_annotation_center_ids: list[str] = []

        for annotation in annotations_by_image.get(prediction["image_name"], []):
            if annotation["annotation_id"] == matched_annotation_id:
                continue
            if center_inside_crop(
                annotation["center_x"], annotation["center_y"], geometry
            ):
                other_annotation_center_ids.append(annotation["annotation_id"])

        if matched_annotation is None:
            annotation_fully_contained: bool | str = ""
        else:
            annotation_fully_contained = fully_contained(
                matched_annotation, geometry
            )

        target_fully_contained = fully_contained(prediction, geometry)
        if not target_fully_contained:
            raise ValueError(
                f"Target detector box is not contained for {prediction['stoma_id']}."
            )

        crop_filename = f"{prediction['stoma_id']}.tif"
        final_crop_path = (
            FINAL_OUTPUT_DIR / CROP_SUBDIRECTORY_NAME / crop_filename
        ).resolve()
        records.append(
            {
                "crop_filename": crop_filename,
                "crop_path": str(final_crop_path),
                "crop_sha256": "",
                "source_image_name": prediction["image_name"],
                "source_image_path": str(
                    (SOURCE_IMAGE_DIR / prediction["image_name"]).resolve()
                ),
                "source_image_sha256": prediction["source_image_sha256"],
                "surface": prediction["surface"],
                "slide_id": prediction["slide_id"],
                "plant_replicate": prediction["plant_replicate"],
                "biological_unit": prediction["biological_unit"],
                "stoma_number": prediction["stoma_number"],
                "stoma_id": prediction["stoma_id"],
                "crop_origin": "post_nms_detector_prediction",
                "use_for_density": True,
                "crop_quality_status": "unreviewed",
                "segmentation_status": "not_attempted",
                "class_id": prediction["class_id"],
                "class_name": prediction["class_name"],
                "confidence": prediction["confidence"],
                "prediction_x1_pixels": prediction["x1"],
                "prediction_y1_pixels": prediction["y1"],
                "prediction_x2_pixels": prediction["x2"],
                "prediction_y2_pixels": prediction["y2"],
                "prediction_center_x_pixels": prediction["center_x"],
                "prediction_center_y_pixels": prediction["center_y"],
                "prediction_width_pixels": prediction["box_width"],
                "prediction_height_pixels": prediction["box_height"],
                **geometry,
                "target_center_crop_x_pixels": (
                    prediction["center_x"] - geometry["crop_x1"]
                ),
                "target_center_crop_y_pixels": (
                    prediction["center_y"] - geometry["crop_y1"]
                ),
                "target_box_crop_x1_pixels": (
                    prediction["x1"] - geometry["crop_x1"]
                ),
                "target_box_crop_y1_pixels": (
                    prediction["y1"] - geometry["crop_y1"]
                ),
                "target_box_crop_x2_pixels": (
                    prediction["x2"] - geometry["crop_x1"]
                ),
                "target_box_crop_y2_pixels": (
                    prediction["y2"] - geometry["crop_y1"]
                ),
                "target_box_fully_contained": target_fully_contained,
                "requested_padding_pixels": PADDING_PIXELS,
                "match_status": match["match_status"],
                "matched_annotation_id": matched_annotation_id,
                "match_iou": match["match_iou"],
                "matched_annotation_fully_contained": (
                    annotation_fully_contained
                ),
                "manual_annotation_match_audit_only": True,
                "other_detection_centers_in_crop": len(other_center_ids),
                "other_detection_center_in_crop": bool(other_center_ids),
                "other_detection_center_ids": ";".join(other_center_ids),
                "other_detection_boxes_intersect_crop": len(
                    other_intersection_ids
                ),
                "other_detection_box_intersects_crop": bool(
                    other_intersection_ids
                ),
                "other_detection_intersecting_ids": ";".join(
                    other_intersection_ids
                ),
                "other_manual_stoma_centers_in_crop": len(
                    other_annotation_center_ids
                ),
                "other_manual_stoma_center_in_crop": bool(
                    other_annotation_center_ids
                ),
                "other_manual_stoma_center_ids": ";".join(
                    other_annotation_center_ids
                ),
                "source_image_width": prediction["source_width"],
                "source_image_height": prediction["source_height"],
                "source_channels": EXPECTED_CHANNELS,
                "source_dtype": EXPECTED_DTYPE,
                "crop_channels": EXPECTED_CHANNELS,
                "crop_dtype": EXPECTED_DTYPE,
                "crop_resized": False,
                "crop_normalized": False,
                "crop_contrast_enhanced": False,
                "crop_compression": "none",
                "pixel_identity_verified": False,
                "micrometers_per_pixel": MICROMETERS_PER_PIXEL,
                "calibration_status": CALIBRATION_STATUS,
            }
        )

    if len(records) != EXPECTED_PREDICTION_COUNT:
        raise RuntimeError("Prepared crop count differs from prediction count.")
    filenames = [record["crop_filename"] for record in records]
    if len(set(filenames)) != len(filenames):
        raise RuntimeError("Duplicate crop filenames were generated.")
    return records


def write_and_verify_crops(
    crop_records: list[dict[str, Any]],
    source_path_by_name: dict[str, Path],
    temporary_crop_dir: Path,
) -> None:
    """Write every uncompressed TIFF crop and verify exact pixel identity."""
    records_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in crop_records:
        records_by_image[record["source_image_name"]].append(record)

    for image_number, image_name in enumerate(sorted(records_by_image), start=1):
        source_path = source_path_by_name[image_name]
        source = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if source is None:
            raise RuntimeError(f"OpenCV could not read:\n{source_path}")
        if source.shape != (
            EXPECTED_IMAGE_HEIGHT,
            EXPECTED_IMAGE_WIDTH,
            EXPECTED_CHANNELS,
        ):
            raise RuntimeError(f"Source shape changed for {image_name}.")

        for record in records_by_image[image_name]:
            crop = source[
                record["crop_y1"] : record["crop_y2"],
                record["crop_x1"] : record["crop_x2"],
            ].copy()
            expected_shape = (
                record["crop_height"],
                record["crop_width"],
                EXPECTED_CHANNELS,
            )
            if crop.shape != expected_shape:
                raise RuntimeError(
                    f"Unexpected crop shape for {record['stoma_id']}: "
                    f"{crop.shape}; expected {expected_shape}."
                )
            if str(crop.dtype) != EXPECTED_DTYPE:
                raise RuntimeError(
                    f"Unexpected crop dtype for {record['stoma_id']}."
                )

            output_path = temporary_crop_dir / record["crop_filename"]
            ensure_inside(output_path, temporary_crop_dir)
            tifffile.imwrite(
                output_path,
                crop,
                photometric="rgb",
                compression=None,
                metadata=None,
            )
            saved = tifffile.imread(output_path)
            if saved.shape != crop.shape or saved.dtype != crop.dtype:
                raise RuntimeError(
                    f"Saved TIFF properties changed for {record['stoma_id']}."
                )
            if not np.array_equal(saved, crop):
                raise RuntimeError(
                    f"Saved TIFF pixels changed for {record['stoma_id']}."
                )
            record["crop_sha256"] = sha256_file(output_path)
            record["pixel_identity_verified"] = True

        print(
            f"Verified source image {image_number:03d}/{EXPECTED_IMAGE_COUNT}: "
            f"{image_name} ({len(records_by_image[image_name])} crops)"
        )
        del source

    written_paths = sorted(temporary_crop_dir.glob("*.tif"))
    if len(written_paths) != EXPECTED_PREDICTION_COUNT:
        raise RuntimeError(
            f"Expected 4546 written crops; observed {len(written_paths)}."
        )
    if not all(record["pixel_identity_verified"] for record in crop_records):
        raise RuntimeError("One or more crops failed pixel-identity validation.")


# =====================================================================
# Summaries, provenance, and main workflow
# =====================================================================

CROP_MANIFEST_FIELDS = [
    "crop_filename",
    "crop_path",
    "crop_sha256",
    "source_image_name",
    "source_image_path",
    "source_image_sha256",
    "surface",
    "slide_id",
    "plant_replicate",
    "biological_unit",
    "stoma_number",
    "stoma_id",
    "crop_origin",
    "use_for_density",
    "crop_quality_status",
    "segmentation_status",
    "class_id",
    "class_name",
    "confidence",
    "prediction_x1_pixels",
    "prediction_y1_pixels",
    "prediction_x2_pixels",
    "prediction_y2_pixels",
    "prediction_center_x_pixels",
    "prediction_center_y_pixels",
    "prediction_width_pixels",
    "prediction_height_pixels",
    "integer_box_x1",
    "integer_box_y1",
    "integer_box_x2",
    "integer_box_y2",
    "requested_crop_x1",
    "requested_crop_y1",
    "requested_crop_x2",
    "requested_crop_y2",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "crop_width",
    "crop_height",
    "target_center_crop_x_pixels",
    "target_center_crop_y_pixels",
    "target_box_crop_x1_pixels",
    "target_box_crop_y1_pixels",
    "target_box_crop_x2_pixels",
    "target_box_crop_y2_pixels",
    "target_box_fully_contained",
    "requested_padding_pixels",
    "actual_padding_left",
    "actual_padding_top",
    "actual_padding_right",
    "actual_padding_bottom",
    "edge_clipped",
    "match_status",
    "matched_annotation_id",
    "match_iou",
    "matched_annotation_fully_contained",
    "manual_annotation_match_audit_only",
    "other_detection_centers_in_crop",
    "other_detection_center_in_crop",
    "other_detection_center_ids",
    "other_detection_boxes_intersect_crop",
    "other_detection_box_intersects_crop",
    "other_detection_intersecting_ids",
    "other_manual_stoma_centers_in_crop",
    "other_manual_stoma_center_in_crop",
    "other_manual_stoma_center_ids",
    "source_image_width",
    "source_image_height",
    "source_channels",
    "source_dtype",
    "crop_channels",
    "crop_dtype",
    "crop_resized",
    "crop_normalized",
    "crop_contrast_enhanced",
    "crop_compression",
    "pixel_identity_verified",
    "micrometers_per_pixel",
    "calibration_status",
]


def summarize_matching(
    prediction_match_rows: list[dict[str, Any]],
    missed_annotation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize descriptive matching overall and separately for AB/AD."""
    rows: list[dict[str, Any]] = []
    for group in ["overall", "AB", "AD"]:
        predictions = [
            row
            for row in prediction_match_rows
            if group == "overall" or row["surface"] == group
        ]
        missed = [
            row
            for row in missed_annotation_rows
            if group == "overall" or row["surface"] == group
        ]
        true_positive = sum(row["match_status"] == "matched" for row in predictions)
        false_positive = len(predictions) - true_positive
        false_negative = len(missed)
        manual_annotations = true_positive + false_negative
        precision = safe_ratio(true_positive, true_positive + false_positive)
        recall = safe_ratio(true_positive, manual_annotations)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        rows.append(
            {
                "surface_group": group,
                "predictions": len(predictions),
                "manual_stoma_annotations": manual_annotations,
                "matched_predictions": true_positive,
                "unmatched_predictions": false_positive,
                "missed_manual_annotations": false_negative,
                "precision_descriptive": precision,
                "recall_descriptive": recall,
                "f1_descriptive": f1,
                "match_iou_threshold": MANUAL_MATCH_IOU_THRESHOLD,
                "independent_evaluation": False,
                "interpretation": (
                    "descriptive audit on detector-training images only"
                ),
            }
        )
    return rows


def summarize_crops_per_image(
    crop_records: list[dict[str, Any]],
    annotations_by_image: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Create a compact per-source-image crop summary."""
    records_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in crop_records:
        records_by_image[record["source_image_name"]].append(record)

    rows: list[dict[str, Any]] = []
    for image_name in sorted(records_by_image):
        records = records_by_image[image_name]
        first = records[0]
        rows.append(
            {
                "image_name": image_name,
                "surface": first["surface"],
                "slide_id": first["slide_id"],
                "plant_replicate": first["plant_replicate"],
                "biological_unit": first["biological_unit"],
                "prediction_crops": len(records),
                "manual_stoma_annotations": len(
                    annotations_by_image.get(image_name, [])
                ),
                "matched_predictions": sum(
                    record["match_status"] == "matched" for record in records
                ),
                "unmatched_predictions": sum(
                    record["match_status"] == "unmatched_prediction"
                    for record in records
                ),
                "edge_clipped_crops": sum(
                    bool(record["edge_clipped"]) for record in records
                ),
                "crops_with_other_detection_center": sum(
                    bool(record["other_detection_center_in_crop"])
                    for record in records
                ),
                "crops_intersecting_other_detection_box": sum(
                    bool(record["other_detection_box_intersects_crop"])
                    for record in records
                ),
                "all_pixels_verified": all(
                    bool(record["pixel_identity_verified"]) for record in records
                ),
            }
        )
    return rows


def summarize_crops_overall(
    crop_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize crop geometry overall and separately for AB/AD."""
    rows: list[dict[str, Any]] = []
    for group in ["overall", "AB", "AD"]:
        records = [
            record
            for record in crop_records
            if group == "overall" or record["surface"] == group
        ]
        rows.append(
            {
                "surface_group": group,
                "crop_count": len(records),
                "edge_clipped_count": sum(
                    bool(record["edge_clipped"]) for record in records
                ),
                "edge_clipped_fraction": safe_ratio(
                    sum(bool(record["edge_clipped"]) for record in records),
                    len(records),
                ),
                "other_detection_center_count": sum(
                    bool(record["other_detection_center_in_crop"])
                    for record in records
                ),
                "other_detection_center_fraction": safe_ratio(
                    sum(
                        bool(record["other_detection_center_in_crop"])
                        for record in records
                    ),
                    len(records),
                ),
                "other_detection_box_intersection_count": sum(
                    bool(record["other_detection_box_intersects_crop"])
                    for record in records
                ),
                "other_detection_box_intersection_fraction": safe_ratio(
                    sum(
                        bool(record["other_detection_box_intersects_crop"])
                        for record in records
                    ),
                    len(records),
                ),
                "matched_prediction_count": sum(
                    record["match_status"] == "matched" for record in records
                ),
                "unmatched_prediction_count": sum(
                    record["match_status"] == "unmatched_prediction"
                    for record in records
                ),
                "requested_padding_pixels": PADDING_PIXELS,
                "pixel_identity_verified_count": sum(
                    bool(record["pixel_identity_verified"]) for record in records
                ),
            }
        )
    return rows


def make_provenance(
    script_path: Path,
    crop_records: list[dict[str, Any]],
    overall_summary_rows: list[dict[str, Any]],
    matching_summary_rows: list[dict[str, Any]],
    output_paths: dict[str, Path],
    classes_sha256: str,
) -> dict[str, Any]:
    """Build the crop-generation provenance record."""
    overall = overall_summary_rows[0]
    matching_overall = matching_summary_rows[0]
    return {
        "analysis": "classifier_training_crop_generation",
        "created_at": datetime.now().astimezone().isoformat(),
        "script": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
        },
        "inputs": {
            "source_image_directory": str(SOURCE_IMAGE_DIR),
            "prediction_csv": str(PREDICTION_CSV),
            "prediction_csv_sha256": sha256_file(PREDICTION_CSV),
            "prediction_json": str(PREDICTION_JSON),
            "prediction_json_sha256": sha256_file(PREDICTION_JSON),
            "prediction_provenance": str(PREDICTION_PROVENANCE),
            "prediction_provenance_sha256": sha256_file(
                PREDICTION_PROVENANCE
            ),
            "prediction_input_manifest": str(PREDICTION_INPUT_MANIFEST),
            "prediction_input_manifest_sha256": sha256_file(
                PREDICTION_INPUT_MANIFEST
            ),
            "manual_yolo_label_directory": str(YOLO_LABEL_DIR),
            "classes_file": str(CLASSES_FILE),
            "classes_file_sha256": classes_sha256,
        },
        "input_restrictions": {
            "source_image_count": EXPECTED_IMAGE_COUNT,
            "AB_source_image_count": EXPECTED_AB_IMAGE_COUNT,
            "AD_source_image_count": EXPECTED_AD_IMAGE_COUNT,
            "manual_yolo_labels_accessed_for_audit": True,
            "manual_xanylabeling_json_accessed": False,
            "test_images_accessed": False,
            "validation_images_accessed": False,
            "hidden_phenotyping_images_accessed": False,
            "detector_inference_rerun": False,
        },
        "locked_detector_settings_in_input_predictions": {
            "model_sha256": EXPECTED_MODEL_SHA256,
            "class_id": STOMA_CLASS_ID,
            "class_name": STOMA_CLASS_NAME,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "manual_nms_iou_threshold": MANUAL_NMS_IOU_THRESHOLD,
        },
        "crop_settings": {
            "requested_padding_pixels": PADDING_PIXELS,
            "float_box_to_integer_rule": (
                "floor x1/y1; ceil x2/y2; then clip to image"
            ),
            "crop_boundary_rule": "half-open NumPy [y1:y2, x1:x2]",
            "crop_resized": False,
            "crop_normalized": False,
            "crop_contrast_enhanced": False,
            "crop_file_format": "TIFF",
            "crop_photometric": "RGB",
            "crop_compression": None,
            "manual_annotation_match_iou_threshold": (
                MANUAL_MATCH_IOU_THRESHOLD
            ),
            "manual_annotation_matching_purpose": "descriptive audit only",
            "unmatched_predictions_retained": True,
            "classifier_status_on_creation": "unreviewed",
            "micrometers_per_pixel": MICROMETERS_PER_PIXEL,
            "calibration_status": CALIBRATION_STATUS,
        },
        "results": {
            "crop_count": len(crop_records),
            "AB_crop_count": sum(
                record["surface"] == "AB" for record in crop_records
            ),
            "AD_crop_count": sum(
                record["surface"] == "AD" for record in crop_records
            ),
            "edge_clipped_crop_count": overall["edge_clipped_count"],
            "crops_with_other_detection_center": (
                overall["other_detection_center_count"]
            ),
            "crops_intersecting_other_detection_box": (
                overall["other_detection_box_intersection_count"]
            ),
            "matched_predictions_audit": (
                matching_overall["matched_predictions"]
            ),
            "unmatched_predictions_audit": (
                matching_overall["unmatched_predictions"]
            ),
            "missed_manual_annotations_audit": (
                matching_overall["missed_manual_annotations"]
            ),
            "pixel_identity_verified_count": sum(
                bool(record["pixel_identity_verified"])
                for record in crop_records
            ),
        },
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in output_paths.items()
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "tifffile": tifffile.__version__,
        },
        "scientific_notes": [
            "Every post-NMS detector prediction generated exactly one crop.",
            "Unmatched predictions were retained; classifier review may label them bad.",
            "Manual annotations were used only for descriptive overlap metadata.",
            "The crop-quality classifier must not alter detector-based density counts.",
            "Edge clipping is a geometry flag and does not automatically mean bad.",
            "A second predicted center is recorded for later classifier/segmentation QC.",
            "Any partial neighboring-box intersection is recorded separately from center inclusion.",
            "AB and AD crops remain identifiable and must be split by source image, never by crop.",
            "The 0.065 micrometers-per-pixel calibration remains provisional.",
            "Source OME-TIFF images and metadata were not modified.",
        ],
    }


def make_report(
    overall_summary_rows: list[dict[str, Any]],
    matching_summary_rows: list[dict[str, Any]],
) -> str:
    """Create a concise human-readable completion report."""
    overall = overall_summary_rows[0]
    ab = overall_summary_rows[1]
    ad = overall_summary_rows[2]
    matching = matching_summary_rows[0]
    return "\n".join(
        [
            "CLASSIFIER CROP GENERATION REPORT",
            "=================================",
            "",
            f"Source images: {EXPECTED_IMAGE_COUNT} (60 AB, 60 AD)",
            f"Crops written: {overall['crop_count']}",
            f"AB crops: {ab['crop_count']}",
            f"AD crops: {ad['crop_count']}",
            f"Requested padding: {PADDING_PIXELS} pixels",
            "Input predictions: confidence >= 0.40, manual stoma-only NMS IoU 0.70",
            "Detector inference rerun: no",
            "Crop resizing/normalization/contrast enhancement: no/no/no",
            "TIFF compression: none",
            "All saved pixels verified against source slices: yes",
            "",
            "GEOMETRY FLAGS",
            f"Edge-clipped crops: {overall['edge_clipped_count']} "
            f"({overall['edge_clipped_fraction']:.3%})",
            "Crops containing another predicted center: "
            f"{overall['other_detection_center_count']} "
            f"({overall['other_detection_center_fraction']:.3%})",
            "Crops intersecting any other predicted box: "
            f"{overall['other_detection_box_intersection_count']} "
            f"({overall['other_detection_box_intersection_fraction']:.3%})",
            "",
            "MANUAL-ANNOTATION AUDIT (DESCRIPTIVE, NOT INDEPENDENT)",
            f"Matched predictions: {matching['matched_predictions']}",
            f"Unmatched predictions retained as crops: {matching['unmatched_predictions']}",
            f"Missed manual annotations: {matching['missed_manual_annotations']}",
            f"Matching IoU threshold: {MANUAL_MATCH_IOU_THRESHOLD:.2f}",
            "",
            "SCIENTIFIC INTERPRETATION",
            "- Edge clipping alone does not make a crop bad.",
            "- Another-box intersection includes small partial overlaps and does not automatically make a crop bad.",
            "- Another predicted center identifies a higher-priority multi-stoma review case.",
            "- Crop-quality labels must never be used to recalculate detector-based density.",
            "- Future classifier splits must be made by source image/biological unit, not individual crop.",
            "- Calibration 0.065 um/pixel is provisional until stage-micrometer verification.",
            "",
            f"Output directory: {FINAL_OUTPUT_DIR}",
            "",
        ]
    )


def run_crop_generation() -> Path:
    """Validate inputs, create crops, write audits, and finalize atomically."""
    script_path = Path(__file__).resolve()

    required_paths = [
        CLASSIFIER_ROOT,
        SOURCE_IMAGE_DIR,
        PREDICTION_DIR,
        PREDICTION_CSV,
        PREDICTION_JSON,
        PREDICTION_PROVENANCE,
        PREDICTION_INPUT_MANIFEST,
        YOLO_LABEL_DIR,
        CLASSES_FILE,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found:\n{path}")
    if FINAL_OUTPUT_DIR.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing crop run:\n"
            f"{FINAL_OUTPUT_DIR}\n"
            "Inspect or deliberately change FINAL_OUTPUT_DIR before rerunning."
        )

    classes_sha256 = validate_classes_file()
    validate_prediction_provenance()
    input_manifest = load_prediction_input_manifest()
    predictions, predictions_by_image = load_and_validate_predictions(
        input_manifest
    )
    source_path_by_name, verified_source_rows = verify_source_images(
        input_manifest
    )
    annotations_by_image, label_manifest_rows = (
        load_manual_stoma_annotations(input_manifest)
    )
    (
        match_by_stoma_id,
        prediction_match_rows,
        missed_annotation_rows,
    ) = match_predictions_to_annotations(
        predictions_by_image,
        annotations_by_image,
    )
    crop_records = prepare_crop_records(
        predictions,
        predictions_by_image,
        annotations_by_image,
        match_by_stoma_id,
    )

    FINAL_OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{FINAL_OUTPUT_DIR.name}_INCOMPLETE_",
            dir=str(FINAL_OUTPUT_DIR.parent),
        )
    ).resolve()
    temporary_crop_dir = temporary_output_dir / CROP_SUBDIRECTORY_NAME
    temporary_audit_dir = temporary_output_dir / AUDIT_SUBDIRECTORY_NAME
    temporary_crop_dir.mkdir()
    temporary_audit_dir.mkdir()

    try:
        write_and_verify_crops(
            crop_records,
            source_path_by_name,
            temporary_crop_dir,
        )

        matching_summary_rows = summarize_matching(
            prediction_match_rows,
            missed_annotation_rows,
        )
        per_image_summary_rows = summarize_crops_per_image(
            crop_records,
            annotations_by_image,
        )
        overall_summary_rows = summarize_crops_overall(crop_records)

        crop_manifest_path = temporary_output_dir / CROP_MANIFEST_NAME
        per_image_summary_path = (
            temporary_output_dir / PER_IMAGE_SUMMARY_NAME
        )
        overall_summary_path = temporary_output_dir / OVERALL_SUMMARY_NAME
        verified_source_path = (
            temporary_output_dir / VERIFIED_SOURCE_MANIFEST_NAME
        )
        manual_label_manifest_path = (
            temporary_audit_dir / MANUAL_LABEL_MANIFEST_NAME
        )
        prediction_matches_path = (
            temporary_audit_dir / PREDICTION_MATCHES_NAME
        )
        missed_annotations_path = (
            temporary_audit_dir / MISSED_ANNOTATIONS_NAME
        )
        matching_summary_path = (
            temporary_audit_dir / MATCHING_SUMMARY_NAME
        )

        write_csv(crop_manifest_path, crop_records, CROP_MANIFEST_FIELDS)
        write_csv(
            per_image_summary_path,
            per_image_summary_rows,
            list(per_image_summary_rows[0]),
        )
        write_csv(
            overall_summary_path,
            overall_summary_rows,
            list(overall_summary_rows[0]),
        )
        write_csv(
            verified_source_path,
            verified_source_rows,
            list(verified_source_rows[0]),
        )
        write_csv(
            manual_label_manifest_path,
            label_manifest_rows,
            list(label_manifest_rows[0]),
        )
        write_csv(
            prediction_matches_path,
            prediction_match_rows,
            list(prediction_match_rows[0]),
        )
        write_csv(
            missed_annotations_path,
            missed_annotation_rows,
            [
                "image_name",
                "surface",
                "annotation_id",
                "annotation_line_number",
                "x1_pixels",
                "y1_pixels",
                "x2_pixels",
                "y2_pixels",
                "audit_only",
                "no_crop_generated_from_manual_box",
            ],
        )
        write_csv(
            matching_summary_path,
            matching_summary_rows,
            list(matching_summary_rows[0]),
        )

        # Paths recorded in provenance must point to the finalized directory.
        finalized_output_paths = {
            "crop_manifest": FINAL_OUTPUT_DIR / CROP_MANIFEST_NAME,
            "per_image_summary": FINAL_OUTPUT_DIR / PER_IMAGE_SUMMARY_NAME,
            "overall_summary": FINAL_OUTPUT_DIR / OVERALL_SUMMARY_NAME,
            "verified_source_manifest": (
                FINAL_OUTPUT_DIR / VERIFIED_SOURCE_MANIFEST_NAME
            ),
            "manual_label_manifest": (
                FINAL_OUTPUT_DIR
                / AUDIT_SUBDIRECTORY_NAME
                / MANUAL_LABEL_MANIFEST_NAME
            ),
            "prediction_annotation_matches": (
                FINAL_OUTPUT_DIR
                / AUDIT_SUBDIRECTORY_NAME
                / PREDICTION_MATCHES_NAME
            ),
            "missed_manual_annotations": (
                FINAL_OUTPUT_DIR
                / AUDIT_SUBDIRECTORY_NAME
                / MISSED_ANNOTATIONS_NAME
            ),
            "matching_summary": (
                FINAL_OUTPUT_DIR
                / AUDIT_SUBDIRECTORY_NAME
                / MATCHING_SUMMARY_NAME
            ),
        }
        temporary_output_paths = {
            name: temporary_output_dir
            / path.relative_to(FINAL_OUTPUT_DIR)
            for name, path in finalized_output_paths.items()
        }

        provenance = make_provenance(
            script_path,
            crop_records,
            overall_summary_rows,
            matching_summary_rows,
            temporary_output_paths,
            classes_sha256,
        )
        # Replace temporary paths inside the output file map with final paths.
        for name, final_path in finalized_output_paths.items():
            provenance["outputs"][name]["path"] = str(final_path.resolve())

        provenance_path = temporary_output_dir / PROVENANCE_NAME
        with provenance_path.open("w", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True)
            handle.write("\n")

        report_path = temporary_output_dir / RUN_REPORT_NAME
        report_path.write_text(
            make_report(overall_summary_rows, matching_summary_rows),
            encoding="utf-8",
        )

        if FINAL_OUTPUT_DIR.exists():
            raise FileExistsError(
                "Final output appeared during the run; refusing overwrite:\n"
                f"{FINAL_OUTPUT_DIR}"
            )
        temporary_output_dir.rename(FINAL_OUTPUT_DIR)

    except Exception:
        print(
            "\nCrop generation did not finalize. Partial diagnostic output was "
            "left at:\n"
            f"{temporary_output_dir}\n"
            "The source images, predictions, and annotations were not modified.",
            file=sys.stderr,
        )
        raise

    print("\nCrop generation completed and verified.")
    print(f"Final output: {FINAL_OUTPUT_DIR}")
    print(f"Crops: {EXPECTED_PREDICTION_COUNT}")
    print(f"Manifest: {FINAL_OUTPUT_DIR / CROP_MANIFEST_NAME}")
    print(f"Report: {FINAL_OUTPUT_DIR / RUN_REPORT_NAME}")
    return FINAL_OUTPUT_DIR


if __name__ == "__main__":
    run_crop_generation()
