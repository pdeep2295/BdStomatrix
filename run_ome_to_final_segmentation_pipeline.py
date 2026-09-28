"""
Run the compiled Brachypodium stomatal computer-vision pipeline.

Scope
-----
This script starts from one original single-channel OME-TIFF, or an explicit
directory of OME-TIFFs, and stops after final cell-instance segmentation.
It intentionally does NOT calculate density, morphometric traits, image means,
plant means, accession means, or GWAS phenotypes. Those operations will be
added only after this integrated inference workflow has been audited.

Pipeline
--------
1. Read the original OME-TIFF without modifying it.
2. Inspect OME physical-pixel metadata and compare it with the verified
   microscope calibration of 4.5848 pixels/um. The micrometer calibration was
   completed when the microscope was installed.
3. Duplicate grayscale intensity into three identical channels in memory.
4. Run the locked final detector once for both classes. Retain stoma class 0
   at confidence 0.40 and hair_cell class 1 at confidence 0.34.
5. Apply confidence-ranked, class-specific manual NMS at IoU 0.70. Hair-cell
   detections are counted and indexed but never enter the crop pipeline.
6. Generate 30-pixel crops from retained stomata only, in memory.
7. Run the locked EfficientNet-B0 three-class crop-quality classifier.
8. Select confident full-good crops, minimum-five ranked fallbacks, and the
   explicit guard-only route.
9. Run the locked combined AB+AD YOLO26n segmentor.
10. Select at most two guard-cell and two subsidiary-cell masks. A guard pair
    must pass hard overlap and bilateral-separation gates before target-centre
    ranking; otherwise only the best single guard is retained and flagged.
    A guard_only crop is upgraded to full 2+2 morphometry only when its two
    subsidiary masks pass the locked area-similarity gate.
11. If fewer than three preliminary usable 2+2 complexes remain, segment
    additional unused crops in descending P(full_good) order until three pass
    or the detected-crop pool is exhausted.
12. Save compressed selected masks, compact masked-crop references, one indexed
    detector overview containing differently coloured stoma and hair-cell
    boxes, class-specific manifests and complete provenance.

Safety
------
- Input images are opened read-only and are never overwritten or re-encoded.
- Three-channel images and ordinary unmasked crops exist only in memory.
- The requested output directory must not already exist.
- A failed run leaves an ``.incomplete`` directory for inspection; it is not
  silently deleted.
- The three selected checkpoint hashes are verified before inference.
- There is no silent CPU fallback when MPS is requested.

Example
-------
python run_ome_to_final_segmentation_pipeline.py \
    --input "/full/path/to/S001_AB_P1_EDF.ome.tif" \
    --output-dir "/full/path/to/pipeline_trial_S001_AB_P1"

Directory input is also supported. Add ``--recursive`` only when TIFFs should
be discovered in nested subdirectories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import re
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import tifffile
import torch
import torchvision
import ultralytics
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import efficientnet_b0
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from ultralytics import YOLO


# =============================================================================
# Curated checkpoints and locked model identities
# =============================================================================


FINAL_MODEL_ROOT = Path(__file__).resolve().parent

DETECTOR_CHECKPOINT = (
    FINAL_MODEL_ROOT / "detector" / "model_training_record" / "best.pt"
).resolve()
CLASSIFIER_CHECKPOINT = (
    FINAL_MODEL_ROOT / "classifier" / "model_training_record" / "best.pt"
).resolve()
SEGMENTOR_CHECKPOINT = (
    FINAL_MODEL_ROOT / "segmentor" / "model_training_record" / "best.pt"
).resolve()

EXPECTED_DETECTOR_SHA256 = (
    "6fa85a2f895f79777b188a69c9b384b730da3feb6956f4bbd2beb626ad4d2862"
)
EXPECTED_CLASSIFIER_SHA256 = (
    "e253dae9064c4a6ab1f84ac0aee204ae604c4dcf2afddd38f6818d3d238a249d"
)
EXPECTED_SEGMENTOR_SHA256 = (
    "d52c91a2f7aab9341aae2d964ef492dbe2186ab0d09a354b5cc63347a058e9b4"
)
EXPECTED_ULTRALYTICS_VERSION = "8.4.6"


# =============================================================================
# Locked operational settings
# =============================================================================


EXPECTED_IMAGE_WIDTH = 3072
EXPECTED_IMAGE_HEIGHT = 2048
EXPECTED_IMAGE_DTYPE = np.dtype("uint8")

REFERENCE_PIXELS_PER_MICROMETRE = 4.5848
# This is a numerical metadata-comparison tolerance, not a biological QC
# threshold. It accepts the agreed rounded form 4.585 pixels/um.
CALIBRATION_ABSOLUTE_TOLERANCE = 0.001
STAGE_MICROMETER_VERIFIED = True

STOMA_CLASS_ID = 0
HAIR_CELL_CLASS_ID = 1
DETECTOR_CLASS_NAMES = {0: "stoma", 1: "hair_cell"}
DETECTOR_IMAGE_SIZE = 1536
STOMA_DETECTOR_CONFIDENCE = 0.40
STOMA_DETECTOR_MANUAL_NMS_IOU = 0.70
HAIR_CELL_DETECTOR_CONFIDENCE = 0.34
HAIR_CELL_DETECTOR_MANUAL_NMS_IOU = 0.70
DETECTOR_INFERENCE_CONFIDENCE_FLOOR = min(
    STOMA_DETECTOR_CONFIDENCE,
    HAIR_CELL_DETECTOR_CONFIDENCE,
)
DETECTOR_INTERNAL_NMS_IOU = 0.70
DETECTOR_MAX_DETECTIONS = 300

# OpenCV uses BGR order. Hair cells are shown as bright yellow and labelled
# H1, H2, ... so they cannot be confused with numbered stoma boxes.
HAIR_CELL_OVERVIEW_COLOR_BGR = (0, 255, 255)

CROP_PADDING_PIXELS = 30

CLASSIFIER_CLASS_NAMES = ("full_good", "guard_only", "bad")
CLASSIFIER_CLASS_TO_ID = {
    name: class_id for class_id, name in enumerate(CLASSIFIER_CLASS_NAMES)
}
CLASSIFIER_IMAGE_SIZE = 224
CLASSIFIER_DROPOUT = 0.20
FULL_GOOD_THRESHOLD = 0.75
MINIMUM_FULL_COMPLEX_CANDIDATES = 5
MINIMUM_USABLE_COMPLEXES_AFTER_SEGMENTATION = 3

SEGMENTOR_CLASS_NAMES = {0: "guard_cell", 1: "subsidary_cell"}
SEGMENTOR_IMAGE_SIZE = 320
SEGMENTOR_BASE_CONFIDENCE = 0.05
SEGMENTOR_ACCEPTANCE_CONFIDENCE = 0.20
SEGMENTOR_INTERNAL_NMS_IOU = 0.70
SEGMENTOR_MAX_DETECTIONS = 20

# Guard cells are separate biological instances. Candidate pairs that overlap
# strongly or fail to occupy opposite sides of the stomatal axis are treated
# as duplicate/same-side predictions, never as a complete guard-cell pair.
MAX_GUARD_PAIR_MASK_IOU = 0.10
MIN_GUARD_PAIR_BILATERAL_ARRANGEMENT = 0.50

# A guard_only classifier result warns that subsidiary boundaries are weak.
# Such a crop is upgraded to full 2+2 morphometry only when the two selected
# subsidiary masks have at least this much smaller/larger area similarity.
MIN_GUARD_ONLY_SUBSIDIARY_PAIR_AREA_SIMILARITY = 0.50

PAIR_SCORE_WEIGHTS = {
    "mean_confidence": 0.50,
    "area_similarity": 0.15,
    "non_overlap": 0.15,
    "longitudinal_alignment": 0.10,
    "bilateral_arrangement": 0.10,
}

TIFF_SUFFIXES = (".ome.tif", ".ome.tiff", ".tif", ".tiff")

FILENAME_PATTERNS = (
    re.compile(
        r"^(?P<slide>S\d{3})_(?P<surface>AB|AD)_P(?P<plant>[1-5])(?:_|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<slide>S\d{3})_P(?P<plant>[1-5])_(?P<surface>AB|AD)(?:_|$)",
        re.IGNORECASE,
    ),
)


# =============================================================================
# Small data structures
# =============================================================================


@dataclass
class MaskInstance:
    """One predicted cell mask mapped to the original crop dimensions."""

    class_id: int
    instance_index: int
    confidence: float
    mask: np.ndarray


@dataclass
class CropContext:
    """In-memory crop plus its evolving detector/classifier/segmentor record."""

    record: dict[str, Any]
    crop_gray: np.ndarray
    crop_rgb: np.ndarray


# =============================================================================
# General utilities
# =============================================================================


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ordered_fieldnames(
    rows: Sequence[dict[str, Any]],
    preferred: Sequence[str],
) -> list[str]:
    observed: list[str] = []
    seen: set[str] = set()
    for name in preferred:
        if name not in seen:
            observed.append(name)
            seen.add(name)
    for row in rows:
        for name in row:
            if name not in seen:
                observed.append(name)
                seen.add(name)
    return observed


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    preferred_fields: Sequence[str],
) -> None:
    fieldnames = ordered_fieldnames(rows, preferred_fields)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def normalized_model_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    raise TypeError(f"Unsupported model class-name container: {type(names)!r}")


def require_requested_device(device_name: str, cuda_device: int) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError("This PyTorch installation was not built with MPS.")
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS is unavailable. The pipeline will not silently use CPU. "
                "Run with --device cpu only if CPU inference is intentional."
            )
        return torch.device("mps")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Install a CUDA-enabled PyTorch build and "
                "a compatible NVIDIA driver, or explicitly use --device cpu."
            )
        device_count = int(torch.cuda.device_count())
        if not 0 <= cuda_device < device_count:
            raise RuntimeError(
                f"CUDA device index {cuda_device} is invalid; "
                f"available device count is {device_count}."
            )
        torch.cuda.set_device(cuda_device)
        return torch.device(f"cuda:{cuda_device}")
    raise ValueError(f"Unsupported device: {device_name}")


def empty_device_cache(device_name: str) -> None:
    if device_name == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device_name.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def source_base_name(path: Path) -> str:
    lower = path.name.lower()
    for suffix in TIFF_SUFFIXES:
        if lower.endswith(suffix):
            return path.name[: -len(suffix)]
    raise ValueError(f"Not a recognized TIFF filename: {path.name}")


def is_tiff(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in TIFF_SUFFIXES)


def parse_biological_name(path: Path) -> dict[str, str]:
    base = source_base_name(path)
    for pattern in FILENAME_PATTERNS:
        match = pattern.match(base)
        if match:
            slide = match.group("slide").upper()
            surface = match.group("surface").upper()
            plant = f"P{int(match.group('plant'))}"
            return {
                "source_base_name": base,
                "slide_id": slide,
                "surface": surface,
                "plant_replicate": plant,
                "biological_unit": f"{slide}_{plant}",
            }
    raise ValueError(
        "The filename does not match either supported biological pattern:\n"
        "  S001_AB_P1_... or S001_P1_AB_...\n"
        f"Observed: {path.name}\n"
        "Rename/copying is intentionally not performed. Confirm the filename "
        "or add an explicit parser before processing it."
    )


def resolve_inside(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(parent.resolve()):
        raise RuntimeError(f"Output path escapes its run directory: {resolved}")
    return resolved


# =============================================================================
# Command-line configuration and preflight
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked detector, crop-quality classifier and combined "
            "cell segmentor on original single-channel OME-TIFF image(s)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Full path to one OME-TIFF or a directory containing OME-TIFFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory. It must not already exist.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search nested input subdirectories. Off by default.",
    )
    parser.add_argument(
        "--device",
        choices=("mps", "cpu", "cuda"),
        default="mps",
        help="Inference device. MPS is default; there is no silent fallback.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="Zero-based NVIDIA CUDA device index; used only with --device cuda.",
    )
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=32,
        help="Classifier inference batch size. Reduce if MPS memory is unstable.",
    )
    parser.add_argument(
        "--calibration-reference",
        type=float,
        default=REFERENCE_PIXELS_PER_MICROMETRE,
        help=(
            "Verified microscope calibration in pixels per micrometre for "
            "checking each image's OME metadata."
        ),
    )
    parser.add_argument(
        "--calibration-absolute-tolerance",
        type=float,
        default=CALIBRATION_ABSOLUTE_TOLERANCE,
        help=(
            "Allowed absolute metadata difference in pixels/micrometre. "
            "Default 0.001 accepts 4.5848 and rounded 4.585."
        ),
    )
    parser.add_argument(
        "--calibration-policy",
        choices=("strict-reference", "use-image-metadata"),
        default="strict-reference",
        help=(
            "Strict mode requires metadata to match the reference. Metadata "
            "mode uses any valid positive X/Y OME calibration and records a "
            "warning when it differs from the reference."
        ),
    )
    parser.add_argument(
        "--stop-on-calibration-problem",
        action="store_true",
        help=(
            "Stop if OME calibration is missing, anisotropic or outside the "
            "specified tolerance. Without this flag, segmentation proceeds "
            "but the problem is recorded. No traits are calculated by this "
            "segmentation-only script."
        ),
    )
    parser.add_argument(
        "--overview-jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality for compact indexed full-image references (70-100).",
    )
    parser.add_argument(
        "--expected-image-width",
        type=int,
        default=EXPECTED_IMAGE_WIDTH,
        help=f"Expected image width in pixels; default {EXPECTED_IMAGE_WIDTH}.",
    )
    parser.add_argument(
        "--expected-image-height",
        type=int,
        default=EXPECTED_IMAGE_HEIGHT,
        help=f"Expected image height in pixels; default {EXPECTED_IMAGE_HEIGHT}.",
    )
    parser.add_argument(
        "--allow-ultralytics-version-mismatch",
        action="store_true",
        help=(
            "Allow inference with an Ultralytics version other than 8.4.6. "
            "The mismatch will be recorded. Use only for a deliberate test."
        ),
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".incomplete")

    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist:\n{input_path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite:\n{output_dir}")
    if staging_dir.exists():
        raise FileExistsError(
            "An incomplete run directory already exists. Inspect it before "
            f"deciding what to do:\n{staging_dir}"
        )
    if args.classifier_batch_size < 1:
        raise ValueError("--classifier-batch-size must be at least 1.")
    if args.cuda_device < 0:
        raise ValueError("--cuda-device cannot be negative.")
    if args.expected_image_width < 1 or args.expected_image_height < 1:
        raise ValueError("Expected image width and height must be positive.")
    if args.calibration_reference <= 0:
        raise ValueError("--calibration-reference must be positive.")
    if args.calibration_absolute_tolerance < 0:
        raise ValueError("Calibration tolerance cannot be negative.")
    if not 70 <= args.overview_jpeg_quality <= 100:
        raise ValueError("--overview-jpeg-quality must be between 70 and 100.")
    if input_path.is_dir() and output_dir.is_relative_to(input_path):
        raise ValueError(
            "Place the output directory outside the input image directory so "
            "future image discovery cannot mistake outputs for source data."
        )

    return input_path, output_dir, staging_dir


def discover_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if not is_tiff(input_path):
            raise ValueError(f"Input file is not a recognized TIFF: {input_path}")
        images = [input_path]
    elif input_path.is_dir():
        candidates = input_path.rglob("*") if recursive else input_path.iterdir()
        images = sorted(
            (
                path.resolve()
                for path in candidates
                if path.is_file()
                and not path.name.startswith(".")
                and is_tiff(path)
            ),
            key=lambda path: str(path),
        )
    else:
        raise ValueError(f"Input is neither a file nor a directory: {input_path}")

    if not images:
        raise FileNotFoundError(f"No TIFF images found under:\n{input_path}")

    base_names = [source_base_name(path) for path in images]
    duplicates = sorted(
        name for name, count in Counter(base_names).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            "Source base names must be unique across one run. Duplicates: "
            f"{duplicates}"
        )

    for path in images:
        parse_biological_name(path)
    return images


def verify_checkpoints_and_versions(
    allow_version_mismatch: bool,
) -> dict[str, Any]:
    expected = {
        "detector": (DETECTOR_CHECKPOINT, EXPECTED_DETECTOR_SHA256),
        "classifier": (CLASSIFIER_CHECKPOINT, EXPECTED_CLASSIFIER_SHA256),
        "segmentor": (SEGMENTOR_CHECKPOINT, EXPECTED_SEGMENTOR_SHA256),
    }
    model_record: dict[str, Any] = {}
    for stage, (path, expected_hash) in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing curated {stage} checkpoint:\n{path}")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"Curated {stage} checkpoint hash changed.\n"
                f"Expected: {expected_hash}\nObserved: {observed_hash}"
            )
        model_record[stage] = {
            "path": str(path),
            "sha256": observed_hash,
        }

    version_matches = ultralytics.__version__ == EXPECTED_ULTRALYTICS_VERSION
    if not version_matches and not allow_version_mismatch:
        raise RuntimeError(
            "Ultralytics differs from the locked segmentor environment.\n"
            f"Expected: {EXPECTED_ULTRALYTICS_VERSION}\n"
            f"Observed: {ultralytics.__version__}\n"
            "Use --allow-ultralytics-version-mismatch only for a deliberate "
            "compatibility test."
        )
    model_record["software"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "tifffile": tifffile.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "ultralytics_matches_locked_version": version_matches,
    }
    return model_record


# =============================================================================
# OME metadata and pixel reading
# =============================================================================


def local_xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def unit_to_micrometre_factor(unit: str | None) -> float | None:
    if unit is None:
        return None
    normalized = (
        str(unit)
        .strip()
        .lower()
        .replace("μ", "µ")
        .replace("micrometers", "micrometer")
        .replace("micrometres", "micrometre")
    )
    factors = {
        "µm": 1.0,
        "um": 1.0,
        "micrometer": 1.0,
        "micrometre": 1.0,
        "nm": 0.001,
        "nanometer": 0.001,
        "nanometre": 0.001,
        "mm": 1000.0,
        "millimeter": 1000.0,
        "millimetre": 1000.0,
        "m": 1_000_000.0,
        "meter": 1_000_000.0,
        "metre": 1_000_000.0,
    }
    return factors.get(normalized)


def read_ome_metadata(path: Path) -> dict[str, Any]:
    with tifffile.TiffFile(path) as tif:
        ome_xml = tif.ome_metadata
        series_shapes = [list(series.shape) for series in tif.series]
        page_count = len(tif.pages)

    record: dict[str, Any] = {
        "ome_metadata_present": bool(ome_xml),
        "ome_xml_sha256": (
            hashlib.sha256(ome_xml.encode("utf-8")).hexdigest()
            if ome_xml
            else ""
        ),
        "tiff_series_shapes": json.dumps(series_shapes),
        "tiff_page_count": page_count,
        "physical_size_x_raw": "",
        "physical_size_x_unit_raw": "",
        "physical_size_y_raw": "",
        "physical_size_y_unit_raw": "",
        "physical_size_x_um": None,
        "physical_size_y_um": None,
    }
    if not ome_xml:
        return record

    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError as error:
        record["ome_xml_parse_error"] = str(error)
        return record

    pixels_element = next(
        (
            element
            for element in root.iter()
            if local_xml_name(element.tag) == "Pixels"
        ),
        None,
    )
    if pixels_element is None:
        record["ome_xml_parse_error"] = "No Pixels element found."
        return record

    for axis in ("X", "Y"):
        raw_size = pixels_element.attrib.get(f"PhysicalSize{axis}")
        raw_unit = pixels_element.attrib.get(f"PhysicalSize{axis}Unit")
        record[f"physical_size_{axis.lower()}_raw"] = raw_size or ""
        record[f"physical_size_{axis.lower()}_unit_raw"] = raw_unit or ""
        if raw_size is None:
            continue
        try:
            numeric_size = float(raw_size)
        except ValueError:
            continue
        factor = unit_to_micrometre_factor(raw_unit)
        if factor is not None and numeric_size > 0:
            record[f"physical_size_{axis.lower()}_um"] = numeric_size * factor
    return record


def assess_calibration(
    metadata: dict[str, Any],
    reference_pixels_per_um: float,
    absolute_tolerance: float,
    calibration_policy: str,
) -> dict[str, Any]:
    size_x = metadata.get("physical_size_x_um")
    size_y = metadata.get("physical_size_y_um")
    pixels_x = 1.0 / size_x if isinstance(size_x, (int, float)) and size_x > 0 else None
    pixels_y = 1.0 / size_y if isinstance(size_y, (int, float)) and size_y > 0 else None

    result = {
        "calibration_reference_pixels_per_um": reference_pixels_per_um,
        "calibration_reference_source": (
            "microscope micrometer calibration verified at installation and recorded in OME metadata"
        ),
        "calibration_absolute_tolerance_pixels_per_um": absolute_tolerance,
        "metadata_pixels_per_um_x": pixels_x,
        "metadata_pixels_per_um_y": pixels_y,
        "calibration_difference_x": (
            abs(pixels_x - reference_pixels_per_um) if pixels_x is not None else None
        ),
        "calibration_difference_y": (
            abs(pixels_y - reference_pixels_per_um) if pixels_y is not None else None
        ),
        "calibration_isotropic": None,
        "calibration_policy": calibration_policy,
        "calibration_reference_match": None,
        "calibration_metadata_used_for_traits": False,
        "calibration_warning": "",
        "calibration_provisional": False,
        "stage_micrometer_verified": STAGE_MICROMETER_VERIFIED,
        "calibrated_traits_permitted": False,
        "calibration_status": "",
    }
    if pixels_x is None or pixels_y is None:
        result["calibration_status"] = "missing_or_unreadable_ome_physical_size"
        return result

    isotropic = math.isclose(
        pixels_x,
        pixels_y,
        rel_tol=0.0,
        abs_tol=absolute_tolerance,
    )
    result["calibration_isotropic"] = isotropic
    reference_match = bool(
        abs(pixels_x - reference_pixels_per_um) <= absolute_tolerance
        and abs(pixels_y - reference_pixels_per_um) <= absolute_tolerance
    )
    result["calibration_reference_match"] = reference_match

    if isotropic and reference_match:
        result["calibration_status"] = "verified_metadata_calibration_matches_reference"
        result["calibrated_traits_permitted"] = True
        result["calibration_metadata_used_for_traits"] = True
    elif calibration_policy == "use-image-metadata":
        result["calibrated_traits_permitted"] = True
        result["calibration_metadata_used_for_traits"] = True
        result["calibration_provisional"] = True
        if not isotropic:
            result["calibration_status"] = "metadata_used_anisotropic_calibration"
            result["calibration_warning"] = (
                "OME X/Y calibration is anisotropic and differs from the "
                "locked isotropic reference; X and Y metadata values were "
                "used separately for calibrated traits."
            )
        else:
            result["calibration_status"] = "metadata_used_reference_mismatch"
            result["calibration_warning"] = (
                "OME calibration differs from the locked reference; the "
                "per-image metadata value was used for calibrated traits."
            )
    elif not isotropic:
        result["calibration_status"] = "anisotropic_metadata_calibration"
    else:
        result["calibration_status"] = "metadata_reference_mismatch"
    return result


def read_original_grayscale(
    path: Path,
    expected_width: int = EXPECTED_IMAGE_WIDTH,
    expected_height: int = EXPECTED_IMAGE_HEIGHT,
) -> tuple[np.ndarray, str]:
    reader = "tifffile"
    try:
        array = tifffile.imread(path)
    except Exception as tif_error:
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        reader = "opencv_fallback"
        if array is None:
            raise RuntimeError(
                f"Neither tifffile nor OpenCV could decode {path}.\n"
                f"tifffile error: {tif_error}\n"
                "For LZW-compressed TIFFs, install imagecodecs in the active "
                "environment, then retry."
            ) from tif_error

    while array.ndim > 2 and 1 in array.shape:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(
            "The production source must be a single two-dimensional grayscale "
            f"image. Observed shape {array.shape} for {path.name}."
        )
    if array.dtype != EXPECTED_IMAGE_DTYPE:
        raise ValueError(
            "The production source must be 8-bit grayscale. No automatic "
            f"intensity conversion will be applied. Observed {array.dtype}."
        )
    if array.shape != (expected_height, expected_width):
        raise ValueError(
            f"Expected {expected_width} x {expected_height}; "
            f"observed {array.shape[1]} x {array.shape[0]} for {path.name}."
        )
    return np.ascontiguousarray(array), reader


def grayscale_to_three_channels(gray: np.ndarray) -> np.ndarray:
    if gray.ndim != 2 or gray.dtype != EXPECTED_IMAGE_DTYPE:
        raise ValueError("Three-channel conversion requires a 2D uint8 array.")
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    if not (
        np.array_equal(rgb[:, :, 0], gray)
        and np.array_equal(rgb[:, :, 1], gray)
        and np.array_equal(rgb[:, :, 2], gray)
    ):
        raise RuntimeError("In-memory channel duplication changed pixel values.")
    return np.ascontiguousarray(rgb)


# =============================================================================
# Detector and manual NMS
# =============================================================================


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
        0.0, float(box_a[3] - box_a[1])
    )
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
        0.0, float(box_b[3] - box_b[1])
    )
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def apply_manual_nms(
    boxes: np.ndarray,
    confidences: np.ndarray,
    iou_threshold: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    if boxes.shape != (len(boxes), 4):
        raise ValueError(f"Expected detector boxes shaped (n,4); got {boxes.shape}.")
    if len(boxes) != len(confidences):
        raise ValueError("Detector box and confidence counts differ.")
    remaining = sorted(
        range(len(boxes)),
        key=lambda index: (-float(confidences[index]), index),
    )
    kept: list[int] = []
    removed: list[dict[str, Any]] = []
    while remaining:
        selected = remaining.pop(0)
        kept.append(selected)
        next_remaining: list[int] = []
        for candidate in remaining:
            overlap = box_iou(boxes[selected], boxes[candidate])
            if overlap > iou_threshold:
                removed.append(
                    {
                        "removed_prediction_index": candidate,
                        "removed_confidence": float(confidences[candidate]),
                        "removed_box_xyxy": boxes[candidate].tolist(),
                        "retained_prediction_index": selected,
                        "retained_confidence": float(confidences[selected]),
                        "retained_box_xyxy": boxes[selected].tolist(),
                        "overlap_iou": overlap,
                    }
                )
            else:
                next_remaining.append(candidate)
        remaining = next_remaining
    return kept, removed


def load_detector() -> YOLO:
    model = YOLO(str(DETECTOR_CHECKPOINT), task="detect")
    observed_names = normalized_model_names(model.names)
    if observed_names != DETECTOR_CLASS_NAMES:
        raise RuntimeError(
            f"Detector class mapping changed: {observed_names}; "
            f"expected {DETECTOR_CLASS_NAMES}."
        )
    return model


def detect_stomata_and_hair_cells(
    model: YOLO,
    image_rgb: np.ndarray,
    source_metadata: dict[str, str],
    source_sha256: str,
    device_name: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Run the joint detector once and post-process each class separately.

    Only retained stoma records are passed to crop generation. Hair-cell
    records remain full-image detections for count/density and overview use.
    """
    # Three channels are identical, so Ultralytics' numpy BGR convention does
    # not alter intensities relative to RGB.
    results = model.predict(
        source=image_rgb,
        imgsz=DETECTOR_IMAGE_SIZE,
        device=device_name,
        conf=DETECTOR_INFERENCE_CONFIDENCE_FLOOR,
        iou=DETECTOR_INTERNAL_NMS_IOU,
        max_det=DETECTOR_MAX_DETECTIONS,
        rect=True,
        half=False,
        augment=False,
        save=False,
        save_txt=False,
        save_conf=False,
        verbose=False,
        stream=False,
    )
    if len(results) != 1:
        raise RuntimeError(f"Expected one detector result; observed {len(results)}.")
    result = results[0]
    height, width = image_rgb.shape[:2]
    if tuple(result.orig_shape) != (height, width):
        raise RuntimeError(
            f"Detector returned original shape {result.orig_shape}; "
            f"expected {(height, width)}."
        )

    if result.boxes is None or len(result.boxes) == 0:
        all_boxes = np.empty((0, 4), dtype=float)
        all_confidences = np.empty((0,), dtype=float)
        all_classes = np.empty((0,), dtype=int)
    else:
        all_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
        all_confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
        all_classes = result.boxes.cls.detach().cpu().numpy().astype(int)

    if not set(all_classes.tolist()).issubset(DETECTOR_CLASS_NAMES):
        raise RuntimeError(
            f"Detector returned an unexpected class ID: {sorted(set(all_classes))}."
        )
    if np.any(all_confidences < DETECTOR_INFERENCE_CONFIDENCE_FLOOR - 1e-6):
        raise RuntimeError(
            "Detector returned a prediction below the joint confidence floor."
        )
    if len(all_boxes) > DETECTOR_MAX_DETECTIONS:
        raise RuntimeError("Detector returned more than 300 predictions.")
    if len(all_boxes):
        if not np.isfinite(all_boxes).all() or not np.isfinite(
            all_confidences
        ).all():
            raise RuntimeError("Detector returned non-finite values.")
        all_boxes[:, [0, 2]] = np.clip(
            all_boxes[:, [0, 2]], 0.0, float(width)
        )
        all_boxes[:, [1, 3]] = np.clip(
            all_boxes[:, [1, 3]], 0.0, float(height)
        )
        if np.any(all_boxes[:, 2] <= all_boxes[:, 0]) or np.any(
            all_boxes[:, 3] <= all_boxes[:, 1]
        ):
            raise RuntimeError("Detector returned a degenerate box.")

    def retain_one_class(
        class_id: int,
        confidence_threshold: float,
        manual_nms_iou: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        class_mask = (
            (all_classes == class_id)
            & (all_confidences >= confidence_threshold)
        )
        original_indices = np.flatnonzero(class_mask)
        boxes = all_boxes[class_mask]
        confidences = all_confidences[class_mask]
        kept_indices, removed = apply_manual_nms(
            boxes,
            confidences,
            manual_nms_iou,
        )

        retained_records: list[dict[str, Any]] = []
        class_name = DETECTOR_CLASS_NAMES[class_id]
        for class_relative_index in kept_indices:
            x1, y1, x2, y2 = boxes[class_relative_index]
            retained_records.append(
                {
                    "source_image_name": source_metadata["source_image_name"],
                    "source_base_name": source_metadata["source_base_name"],
                    "source_image_sha256": source_sha256,
                    "surface": source_metadata["surface"],
                    "slide_id": source_metadata["slide_id"],
                    "plant_replicate": source_metadata["plant_replicate"],
                    "biological_unit": source_metadata["biological_unit"],
                    "pre_nms_prediction_index": int(
                        original_indices[class_relative_index]
                    ),
                    "detector_class_id": class_id,
                    "detector_class_name": class_name,
                    "detector_confidence": float(
                        confidences[class_relative_index]
                    ),
                    "prediction_x1_pixels": float(x1),
                    "prediction_y1_pixels": float(y1),
                    "prediction_x2_pixels": float(x2),
                    "prediction_y2_pixels": float(y2),
                    "prediction_center_x_pixels": float((x1 + x2) / 2.0),
                    "prediction_center_y_pixels": float((y1 + y2) / 2.0),
                    "prediction_width_pixels": float(x2 - x1),
                    "prediction_height_pixels": float(y2 - y1),
                    "source_image_width": width,
                    "source_image_height": height,
                }
            )

        for item in removed:
            item.update(
                {
                    "source_image_name": source_metadata["source_image_name"],
                    "surface": source_metadata["surface"],
                    "detector_class_id": class_id,
                    "detector_class_name": class_name,
                    "removed_prediction_index": int(
                        original_indices[item["removed_prediction_index"]]
                    ),
                    "retained_prediction_index": int(
                        original_indices[item["retained_prediction_index"]]
                    ),
                }
            )
        return retained_records, removed, len(boxes)

    retained_stomata, removed_stomata, stomata_before_nms = retain_one_class(
        STOMA_CLASS_ID,
        STOMA_DETECTOR_CONFIDENCE,
        STOMA_DETECTOR_MANUAL_NMS_IOU,
    )
    retained_hair_cells, removed_hair_cells, hair_cells_before_nms = (
        retain_one_class(
            HAIR_CELL_CLASS_ID,
            HAIR_CELL_DETECTOR_CONFIDENCE,
            HAIR_CELL_DETECTOR_MANUAL_NMS_IOU,
        )
    )

    retained_stomata.sort(
        key=lambda record: (
            record["prediction_center_y_pixels"],
            record["prediction_center_x_pixels"],
            -record["detector_confidence"],
            record["pre_nms_prediction_index"],
        )
    )
    for stoma_number, record in enumerate(retained_stomata, start=1):
        record["stoma_number"] = stoma_number
        record["stoma_id"] = (
            f"{source_metadata['source_base_name']}_stoma_{stoma_number:04d}"
        )

    retained_hair_cells.sort(
        key=lambda record: (
            record["prediction_center_y_pixels"],
            record["prediction_center_x_pixels"],
            -record["detector_confidence"],
            record["pre_nms_prediction_index"],
        )
    )
    for hair_cell_number, record in enumerate(retained_hair_cells, start=1):
        record["hair_cell_number"] = hair_cell_number
        record["hair_cell_id"] = (
            f"{source_metadata['source_base_name']}_hair_cell_"
            f"{hair_cell_number:04d}"
        )

    detection_summary = {
        "joint_predictions_at_confidence_floor": len(all_boxes),
        "detector_max_det_reached": len(all_boxes) >= DETECTOR_MAX_DETECTIONS,
        "stoma_predictions_before_manual_nms": stomata_before_nms,
        "stoma_predictions_after_manual_nms": len(retained_stomata),
        "stoma_manual_nms_duplicates_removed": len(removed_stomata),
        "hair_cell_predictions_before_manual_nms": hair_cells_before_nms,
        "hair_cell_predictions_after_manual_nms": len(retained_hair_cells),
        "hair_cell_manual_nms_duplicates_removed": len(removed_hair_cells),
    }
    return (
        retained_stomata,
        retained_hair_cells,
        [*removed_stomata, *removed_hair_cells],
        detection_summary,
    )


# =============================================================================
# In-memory crop generation
# =============================================================================


def integer_detection_box(record: dict[str, Any]) -> tuple[int, int, int, int]:
    width = int(record["source_image_width"])
    height = int(record["source_image_height"])
    x1 = max(0, min(width, math.floor(record["prediction_x1_pixels"])))
    y1 = max(0, min(height, math.floor(record["prediction_y1_pixels"])))
    x2 = max(0, min(width, math.ceil(record["prediction_x2_pixels"])))
    y2 = max(0, min(height, math.ceil(record["prediction_y2_pixels"])))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate integer crop box for {record['stoma_id']}.")
    return x1, y1, x2, y2


def calculate_crop_geometry(record: dict[str, Any]) -> dict[str, Any]:
    box_x1, box_y1, box_x2, box_y2 = integer_detection_box(record)
    requested_x1 = box_x1 - CROP_PADDING_PIXELS
    requested_y1 = box_y1 - CROP_PADDING_PIXELS
    requested_x2 = box_x2 + CROP_PADDING_PIXELS
    requested_y2 = box_y2 + CROP_PADDING_PIXELS
    crop_x1 = max(0, requested_x1)
    crop_y1 = max(0, requested_y1)
    crop_x2 = min(int(record["source_image_width"]), requested_x2)
    crop_y2 = min(int(record["source_image_height"]), requested_y2)
    actual_left = box_x1 - crop_x1
    actual_top = box_y1 - crop_y1
    actual_right = crop_x2 - box_x2
    actual_bottom = crop_y2 - box_y2
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
        "requested_padding_pixels": CROP_PADDING_PIXELS,
        "actual_padding_left": actual_left,
        "actual_padding_top": actual_top,
        "actual_padding_right": actual_right,
        "actual_padding_bottom": actual_bottom,
        "edge_clipped": any(
            value < CROP_PADDING_PIXELS
            for value in (actual_left, actual_top, actual_right, actual_bottom)
        ),
    }


def center_inside_crop(
    center_x: float,
    center_y: float,
    geometry: dict[str, Any],
) -> bool:
    return bool(
        geometry["crop_x1"] <= center_x < geometry["crop_x2"]
        and geometry["crop_y1"] <= center_y < geometry["crop_y2"]
    )


def box_intersects_crop(record: dict[str, Any], geometry: dict[str, Any]) -> bool:
    overlap_width = min(
        record["prediction_x2_pixels"], geometry["crop_x2"]
    ) - max(record["prediction_x1_pixels"], geometry["crop_x1"])
    overlap_height = min(
        record["prediction_y2_pixels"], geometry["crop_y2"]
    ) - max(record["prediction_y1_pixels"], geometry["crop_y1"])
    return bool(overlap_width > 0 and overlap_height > 0)


def prepare_crops(
    gray: np.ndarray,
    retained: list[dict[str, Any]],
    calibration: dict[str, Any],
) -> list[CropContext]:
    contexts: list[CropContext] = []
    for record in retained:
        geometry = calculate_crop_geometry(record)
        other_centres: list[str] = []
        other_intersections: list[str] = []
        for other in retained:
            if other["stoma_id"] == record["stoma_id"]:
                continue
            if center_inside_crop(
                other["prediction_center_x_pixels"],
                other["prediction_center_y_pixels"],
                geometry,
            ):
                other_centres.append(other["stoma_id"])
            if box_intersects_crop(other, geometry):
                other_intersections.append(other["stoma_id"])

        crop_gray = gray[
            geometry["crop_y1"] : geometry["crop_y2"],
            geometry["crop_x1"] : geometry["crop_x2"],
        ].copy()
        if crop_gray.shape != (
            geometry["crop_height"],
            geometry["crop_width"],
        ):
            raise RuntimeError(f"Crop shape mismatch for {record['stoma_id']}.")
        crop_rgb = grayscale_to_three_channels(crop_gray)
        record.update(geometry)
        record.update(
            {
                "crop_origin": "post_nms_detector_prediction",
                "crop_saved_unmasked": False,
                "target_center_crop_x_pixels": (
                    record["prediction_center_x_pixels"] - geometry["crop_x1"]
                ),
                "target_center_crop_y_pixels": (
                    record["prediction_center_y_pixels"] - geometry["crop_y1"]
                ),
                "target_box_crop_x1_pixels": (
                    record["prediction_x1_pixels"] - geometry["crop_x1"]
                ),
                "target_box_crop_y1_pixels": (
                    record["prediction_y1_pixels"] - geometry["crop_y1"]
                ),
                "target_box_crop_x2_pixels": (
                    record["prediction_x2_pixels"] - geometry["crop_x1"]
                ),
                "target_box_crop_y2_pixels": (
                    record["prediction_y2_pixels"] - geometry["crop_y1"]
                ),
                "other_detection_centers_in_crop": len(other_centres),
                "other_detection_center_in_crop": bool(other_centres),
                "other_detection_center_ids": ";".join(other_centres),
                "other_detection_boxes_intersect_crop": len(other_intersections),
                "other_detection_box_intersects_crop": bool(other_intersections),
                "other_detection_intersecting_ids": ";".join(
                    other_intersections
                ),
                "calibration_status": calibration["calibration_status"],
                "calibration_provisional": calibration["calibration_provisional"],
                "stage_micrometer_verified": calibration[
                    "stage_micrometer_verified"
                ],
                "calibrated_traits_permitted": calibration[
                    "calibrated_traits_permitted"
                ],
                "metadata_pixels_per_um_x": calibration[
                    "metadata_pixels_per_um_x"
                ],
                "metadata_pixels_per_um_y": calibration[
                    "metadata_pixels_per_um_y"
                ],
            }
        )
        contexts.append(CropContext(record, crop_gray, crop_rgb))
    return contexts


# =============================================================================
# Locked classifier preprocessing, model and selection
# =============================================================================


def median_pad_to_square(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    if width == height:
        return image
    array = np.asarray(image, dtype=np.uint8)
    medians = np.rint(np.median(array.reshape(-1, 3), axis=0)).astype(np.uint8)
    difference = abs(width - height)
    first = difference // 2
    second = difference - first
    border = (
        (first, 0, second, 0)
        if width < height
        else (0, first, 0, second)
    )
    return ImageOps.expand(
        image,
        border=border,
        fill=tuple(int(value) for value in medians),
    )


def classifier_tensor(crop_gray: np.ndarray) -> torch.Tensor:
    image = median_pad_to_square(Image.fromarray(crop_gray, mode="L"))
    image = TF.resize(
        image,
        [CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    tensor = TF.to_tensor(image)
    return TF.normalize(
        tensor,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def load_classifier(device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(
        CLASSIFIER_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("class_to_id") != CLASSIFIER_CLASS_TO_ID:
        raise RuntimeError(
            "Classifier class mapping changed. Expected "
            f"{CLASSIFIER_CLASS_TO_ID}; observed {checkpoint.get('class_to_id')}."
        )
    model = efficientnet_b0(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=CLASSIFIER_DROPOUT, inplace=False),
        nn.Linear(input_features, len(CLASSIFIER_CLASS_NAMES)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    metadata = {
        "epoch": int(checkpoint["epoch"]),
        "stage": str(checkpoint["stage"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_macro_f1": float(checkpoint["best_macro_f1"]),
        "training_config": checkpoint.get("training_config"),
        "training_split_manifest_sha256": checkpoint.get(
            "split_manifest_sha256"
        ),
    }
    return model, metadata


def classify_crops(
    model: nn.Module,
    contexts: list[CropContext],
    device: torch.device,
    batch_size: int,
) -> None:
    if not contexts:
        return
    tensors = [classifier_tensor(context.crop_gray) for context in contexts]
    probability_rows: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[start : start + batch_size]).to(device)
                probabilities = torch.softmax(model(batch), dim=1)
                probability_rows.extend(probabilities.cpu().numpy())
                del batch, probabilities
    except RuntimeError as error:
        message = str(error).lower()
        if "out of memory" in message or "mps" in message:
            raise RuntimeError(
                "Classifier inference failed on MPS. If the message indicates "
                "memory pressure, rerun in a new output directory with a lower "
                "--classifier-batch-size such as 16 or 8. Original images and "
                "the incomplete output remain unchanged."
            ) from error
        raise

    if len(probability_rows) != len(contexts):
        raise RuntimeError("Classifier did not return one result per crop.")
    for context, probability_row in zip(contexts, probability_rows, strict=True):
        class_id = int(np.argmax(probability_row))
        context.record.update(
            {
                "classifier_predicted_class_id": class_id,
                "classifier_predicted_label": CLASSIFIER_CLASS_NAMES[class_id],
                "probability_full_good": float(probability_row[0]),
                "probability_guard_only": float(probability_row[1]),
                "probability_bad": float(probability_row[2]),
                "full_good_threshold": FULL_GOOD_THRESHOLD,
            }
        )


def apply_classifier_routing(contexts: list[CropContext]) -> dict[str, Any]:
    ranked = sorted(
        contexts,
        key=lambda context: (
            -context.record["probability_full_good"],
            context.record["stoma_id"],
        ),
    )
    for rank, context in enumerate(ranked, start=1):
        context.record.update(
            {
                "full_good_probability_rank_within_image": rank,
                "selected_for_segmentation": False,
                "initially_selected_for_segmentation": False,
                "segmentation_route": "not_selected",
                "full_complex_candidate": False,
                "post_segmentation_top_up": False,
                "post_segmentation_top_up_rank": "",
                "usable_complete_count_before_top_up_attempt": "",
            }
        )

    confident = [
        context
        for context in ranked
        if context.record["probability_full_good"] >= FULL_GOOD_THRESHOLD
    ]
    for context in confident:
        context.record.update(
            {
                "selected_for_segmentation": True,
                "initially_selected_for_segmentation": True,
                "segmentation_route": "confident_full_good",
                "full_complex_candidate": True,
            }
        )

    confident_object_ids = {id(context) for context in confident}
    remaining = [
        context for context in ranked if id(context) not in confident_object_ids
    ]
    number_needed = max(0, MINIMUM_FULL_COMPLEX_CANDIDATES - len(confident))
    fallback = remaining[:number_needed]
    for context in fallback:
        context.record.update(
            {
                "selected_for_segmentation": True,
                "initially_selected_for_segmentation": True,
                "segmentation_route": "fallback_ranked",
                "full_complex_candidate": True,
            }
        )

    guard_only = [
        context
        for context in remaining[number_needed:]
        if context.record["classifier_predicted_label"] == "guard_only"
    ]
    for context in guard_only:
        context.record.update(
            {
                "selected_for_segmentation": True,
                "initially_selected_for_segmentation": True,
                "segmentation_route": "guard_only_route",
                "full_complex_candidate": False,
            }
        )

    return {
        "available_crop_count": len(contexts),
        "confident_full_good_count": len(confident),
        "fallback_candidate_count": len(fallback),
        "guard_only_route_count": len(guard_only),
        "full_complex_candidate_count": len(confident) + len(fallback),
        "total_selected_for_segmentation": len(confident) + len(fallback) + len(guard_only),
        "minimum_full_complex_candidates_requested": MINIMUM_FULL_COMPLEX_CANDIDATES,
        "minimum_candidate_target_met": (
            len(confident) + len(fallback) >= MINIMUM_FULL_COMPLEX_CANDIDATES
        ),
        "fewer_than_minimum_crops_available": (
            len(contexts) < MINIMUM_FULL_COMPLEX_CANDIDATES
        ),
        "requires_fallback": bool(fallback),
    }


# =============================================================================
# Segmentor prediction and structured pair selection
# =============================================================================


def load_segmentor() -> YOLO:
    model = YOLO(str(SEGMENTOR_CHECKPOINT), task="segment")
    observed_names = normalized_model_names(model.names)
    if observed_names != SEGMENTOR_CLASS_NAMES:
        raise RuntimeError(
            f"Segmentor class mapping changed: {observed_names}; "
            f"expected {SEGMENTOR_CLASS_NAMES}."
        )
    return model


def resize_binary_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    if mask.shape == (height, width):
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def predict_cell_masks(
    model: YOLO,
    crop_rgb: np.ndarray,
    device_name: str,
) -> list[MaskInstance]:
    results = model.predict(
        source=crop_rgb,
        imgsz=SEGMENTOR_IMAGE_SIZE,
        conf=SEGMENTOR_BASE_CONFIDENCE,
        iou=SEGMENTOR_INTERNAL_NMS_IOU,
        max_det=SEGMENTOR_MAX_DETECTIONS,
        device=device_name,
        retina_masks=True,
        agnostic_nms=False,
        classes=sorted(SEGMENTOR_CLASS_NAMES),
        save=False,
        save_txt=False,
        save_crop=False,
        verbose=False,
        stream=False,
    )
    if len(results) != 1:
        raise RuntimeError(f"Expected one segmentor result; observed {len(results)}.")
    result = results[0]
    height, width = crop_rgb.shape[:2]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    if result.masks is None:
        raise RuntimeError("Segmentor returned boxes without masks.")

    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
    masks = result.masks.data.detach().cpu().numpy()
    if not (len(class_ids) == len(confidences) == len(masks)):
        raise RuntimeError("Segmentor box, class, confidence and mask counts differ.")
    counters: Counter[int] = Counter()
    predictions: list[MaskInstance] = []
    for class_id, confidence, mask in zip(
        class_ids,
        confidences,
        masks,
        strict=True,
    ):
        class_id = int(class_id)
        if class_id not in SEGMENTOR_CLASS_NAMES:
            continue
        counters[class_id] += 1
        predictions.append(
            MaskInstance(
                class_id=class_id,
                instance_index=counters[class_id],
                confidence=float(confidence),
                mask=resize_binary_mask(mask > 0.5, (height, width)),
            )
        )
    return predictions


def mask_pixels_xy(mask: np.ndarray) -> np.ndarray:
    rows_y, columns_x = np.nonzero(mask)
    return np.column_stack((columns_x, rows_y)).astype(float)


def mask_centroid(mask: np.ndarray) -> np.ndarray:
    pixels = mask_pixels_xy(mask)
    if not len(pixels):
        raise ValueError("Cannot calculate centroid of an empty mask.")
    return pixels.mean(axis=0)


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def major_axis_from_masks(
    masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    pixels = np.vstack([mask_pixels_xy(mask) for mask in masks])
    if len(pixels) < 3:
        raise ValueError("Insufficient mask pixels for major-axis estimation.")
    center = pixels.mean(axis=0)
    covariance = np.cov(pixels - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    axis = axis / np.linalg.norm(axis)
    projections = (pixels - center) @ axis
    length = float(projections.max() - projections.min() + 1.0)
    return center, axis, length


def area_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_area = float(first.sum())
    second_area = float(second.sum())
    largest = max(first_area, second_area)
    return min(first_area, second_area) / largest if largest else 0.0


def pair_geometry(
    first: MaskInstance,
    second: MaskInstance,
    reference_guard_pair: list[MaskInstance] | None,
) -> dict[str, float]:
    confidence = float(np.mean([first.confidence, second.confidence]))
    similarity = area_similarity(first.mask, second.mask)
    overlap = mask_iou(first.mask, second.mask)
    non_overlap = max(0.0, 1.0 - min(overlap / 0.50, 1.0))
    pair_center, pair_axis, pair_length = major_axis_from_masks(
        [first.mask, second.mask]
    )
    first_centroid = mask_centroid(first.mask)
    second_centroid = mask_centroid(second.mask)
    longitudinal_difference = abs(
        float((first_centroid - second_centroid) @ pair_axis)
    )
    alignment = max(
        0.0,
        1.0 - longitudinal_difference / max(0.35 * pair_length, 1.0),
    )
    if reference_guard_pair is None:
        normal = np.array([-pair_axis[1], pair_axis[0]], dtype=float)
        transverse_difference = abs(
            float((first_centroid - second_centroid) @ normal)
        )
        bilateral = min(
            transverse_difference / max(0.12 * pair_length, 1.0),
            1.0,
        )
    else:
        guard_center, guard_axis, _ = major_axis_from_masks(
            [item.mask for item in reference_guard_pair]
        )
        guard_normal = np.array([-guard_axis[1], guard_axis[0]], dtype=float)
        side_first = float((first_centroid - guard_center) @ guard_normal)
        side_second = float((second_centroid - guard_center) @ guard_normal)
        bilateral = 1.0 if side_first * side_second < 0.0 else 0.0
        longitudinal_difference = abs(
            float((first_centroid - second_centroid) @ guard_axis)
        )
        alignment = max(
            0.0,
            1.0 - longitudinal_difference / max(0.35 * pair_length, 1.0),
        )
    score = sum(
        (
            PAIR_SCORE_WEIGHTS["mean_confidence"] * confidence,
            PAIR_SCORE_WEIGHTS["area_similarity"] * similarity,
            PAIR_SCORE_WEIGHTS["non_overlap"] * non_overlap,
            PAIR_SCORE_WEIGHTS["longitudinal_alignment"] * alignment,
            PAIR_SCORE_WEIGHTS["bilateral_arrangement"] * bilateral,
        )
    )
    return {
        "score": score,
        "mean_confidence": confidence,
        "area_similarity": similarity,
        "mask_iou_between_candidates": overlap,
        "non_overlap": non_overlap,
        "longitudinal_alignment": alignment,
        "bilateral_arrangement": bilateral,
        "pair_center_x": float(pair_center[0]),
        "pair_center_y": float(pair_center[1]),
    }


def point_inside_box(point_xy: np.ndarray, box_xyxy: np.ndarray) -> bool:
    return bool(
        box_xyxy[0] <= point_xy[0] <= box_xyxy[2]
        and box_xyxy[1] <= point_xy[1] <= box_xyxy[3]
    )


def guard_pair_geometry_is_valid(features: dict[str, Any]) -> bool:
    return bool(
        features["mask_iou_between_candidates"] <= MAX_GUARD_PAIR_MASK_IOU
        and features["bilateral_arrangement"]
        >= MIN_GUARD_PAIR_BILATERAL_ARRANGEMENT
    )


def add_guard_pair_geometry_fields(features: dict[str, Any]) -> dict[str, Any]:
    return {
        **features,
        "guard_pair_geometry_required": True,
        "guard_pair_geometry_valid": guard_pair_geometry_is_valid(features),
        "maximum_guard_pair_mask_iou": MAX_GUARD_PAIR_MASK_IOU,
        "minimum_guard_pair_bilateral_arrangement": (
            MIN_GUARD_PAIR_BILATERAL_ARRANGEMENT
        ),
    }


def select_single_guard_fallback(
    candidates: list[MaskInstance],
    selection_mode: str,
    candidate_pair_count: int,
    best_rejected_pair_features: dict[str, Any] | None,
    target_center: np.ndarray | None = None,
    target_box: np.ndarray | None = None,
) -> tuple[list[MaskInstance], dict[str, Any]]:
    """Retain at most one guard when no biologically valid pair exists."""
    if not candidates:
        selected: list[MaskInstance] = []
        single_features: dict[str, Any] = {}
    elif target_center is None or target_box is None:
        best = min(
            candidates,
            key=lambda item: (-item.confidence, item.instance_index),
        )
        selected = [best]
        centroid = mask_centroid(best.mask)
        single_features = {
            "selected_single_guard_instance_index": best.instance_index,
            "selected_single_guard_confidence": best.confidence,
            "selected_single_guard_centroid_x": float(centroid[0]),
            "selected_single_guard_centroid_y": float(centroid[1]),
        }
    else:
        ranked_singles: list[
            tuple[tuple[Any, ...], MaskInstance, np.ndarray]
        ] = []
        for item in candidates:
            centroid = mask_centroid(item.mask)
            inside = point_inside_box(centroid, target_box)
            distance = float(np.linalg.norm(centroid - target_center))
            ranking_key = (
                -int(inside),
                -item.confidence,
                distance,
                item.instance_index,
            )
            ranked_singles.append((ranking_key, item, centroid))
        ranked_singles.sort(key=lambda item: item[0])
        _, best, centroid = ranked_singles[0]
        selected = [best]
        single_features = {
            "selected_single_guard_instance_index": best.instance_index,
            "selected_single_guard_confidence": best.confidence,
            "selected_single_guard_centroid_x": float(centroid[0]),
            "selected_single_guard_centroid_y": float(centroid[1]),
            "selected_single_guard_centroid_inside_target_box": (
                point_inside_box(centroid, target_box)
            ),
            "selected_single_guard_distance_to_target_pixels": float(
                np.linalg.norm(centroid - target_center)
            ),
        }

    rejection_reason = (
        "fewer_than_two_guard_candidates"
        if candidate_pair_count == 0
        else "no_guard_pair_passed_overlap_and_bilateral_gates"
    )
    summary = {
        "selection_mode": selection_mode,
        "guard_pair_geometry_required": True,
        "guard_pair_geometry_valid": False,
        "guard_pair_candidate_count": candidate_pair_count,
        "guard_pair_valid_candidate_count": 0,
        "guard_pair_rejection_reason": rejection_reason,
        "maximum_guard_pair_mask_iou": MAX_GUARD_PAIR_MASK_IOU,
        "minimum_guard_pair_bilateral_arrangement": (
            MIN_GUARD_PAIR_BILATERAL_ARRANGEMENT
        ),
        "best_rejected_pair_features": best_rejected_pair_features or {},
        **single_features,
    }
    return selected, summary


def select_structured_pair(
    candidates: list[MaskInstance],
    reference_guard_pair: list[MaskInstance] | None,
    *,
    require_distinct_guard_pair: bool = False,
) -> tuple[list[MaskInstance], dict[str, Any] | None]:
    if len(candidates) <= 1:
        if require_distinct_guard_pair:
            return select_single_guard_fallback(
                candidates,
                "locked_structured_single_guard_fallback",
                candidate_pair_count=0,
                best_rejected_pair_features=None,
            )
        return candidates, None

    ranked: list[
        tuple[float, tuple[int, int], list[MaskInstance], dict[str, Any]]
    ] = []
    for first, second in itertools.combinations(candidates, 2):
        features = pair_geometry(first, second, reference_guard_pair)
        if require_distinct_guard_pair:
            features = add_guard_pair_geometry_fields(features)
        identifiers = tuple(sorted((first.instance_index, second.instance_index)))
        ranked.append((features["score"], identifiers, [first, second], features))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    if require_distinct_guard_pair:
        valid_ranked = [item for item in ranked if item[3]["guard_pair_geometry_valid"]]
        if not valid_ranked:
            return select_single_guard_fallback(
                candidates,
                "locked_structured_single_guard_fallback",
                candidate_pair_count=len(ranked),
                best_rejected_pair_features=ranked[0][3],
            )
        ranked = valid_ranked

    _, _, selected, features = ranked[0]
    features = {
        **features,
        "selection_mode": "locked_structured_pair_score",
    }
    if require_distinct_guard_pair:
        features.update(
            {
                "guard_pair_candidate_count": math.comb(len(candidates), 2),
                "guard_pair_valid_candidate_count": len(ranked),
                "guard_pair_rejection_reason": "",
            }
        )
    return selected, features


def select_target_aware_pair(
    candidates: list[MaskInstance],
    target_center: np.ndarray,
    target_box: np.ndarray,
    reference_guard_pair: list[MaskInstance] | None,
    *,
    require_distinct_guard_pair: bool = False,
) -> tuple[list[MaskInstance], dict[str, Any] | None]:
    """Associate an ambiguous multi-mask crop with its detector target.

    This deterministic extension is intentionally transparent. It first
    prefers a pair whose two mask centroids lie within the target detector box,
    then a pair centered inside that box, then the smallest center distance,
    and finally the validation-selected structured pair score. It is activated
    only for overlap/extra-candidate cases and is flagged as not independently
    validated.
    """
    if len(candidates) <= 1:
        if require_distinct_guard_pair:
            return select_single_guard_fallback(
                candidates,
                "target_aware_single_guard_fallback",
                candidate_pair_count=0,
                best_rejected_pair_features=None,
                target_center=target_center,
                target_box=target_box,
            )
        return candidates, None
    target_diagonal = max(
        float(np.linalg.norm(target_box[2:] - target_box[:2])),
        1.0,
    )
    ranked: list[
        tuple[
            tuple[Any, ...],
            list[MaskInstance],
            dict[str, Any],
        ]
    ] = []
    for first, second in itertools.combinations(candidates, 2):
        features = pair_geometry(first, second, reference_guard_pair)
        if require_distinct_guard_pair:
            features = add_guard_pair_geometry_fields(features)
        first_centroid = mask_centroid(first.mask)
        second_centroid = mask_centroid(second.mask)
        pair_center = (first_centroid + second_centroid) / 2.0
        both_inside = point_inside_box(first_centroid, target_box) and point_inside_box(
            second_centroid, target_box
        )
        center_inside = point_inside_box(pair_center, target_box)
        distance = float(np.linalg.norm(pair_center - target_center))
        normalized_distance = distance / target_diagonal
        identifiers = tuple(sorted((first.instance_index, second.instance_index)))
        features.update(
            {
                "both_centroids_inside_target_box": both_inside,
                "pair_center_inside_target_box": center_inside,
                "pair_center_distance_to_target_pixels": distance,
                "pair_center_distance_to_target_box_diagonal": normalized_distance,
                "selection_mode": "target_aware_unvalidated",
            }
        )
        ranking_key = (
            -int(both_inside),
            -int(center_inside),
            normalized_distance,
            -features["score"],
            identifiers,
        )
        ranked.append((ranking_key, [first, second], features))
    ranked.sort(key=lambda item: item[0])

    if require_distinct_guard_pair:
        valid_ranked = [item for item in ranked if item[2]["guard_pair_geometry_valid"]]
        if not valid_ranked:
            return select_single_guard_fallback(
                candidates,
                "target_aware_single_guard_fallback",
                candidate_pair_count=len(ranked),
                best_rejected_pair_features=ranked[0][2],
                target_center=target_center,
                target_box=target_box,
            )
        ranked = valid_ranked

    _, selected, features = ranked[0]
    if require_distinct_guard_pair:
        features.update(
            {
                "guard_pair_candidate_count": math.comb(len(candidates), 2),
                "guard_pair_valid_candidate_count": len(ranked),
                "guard_pair_rejection_reason": "",
            }
        )
    return selected, features


def mask_touches_border(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def select_cell_instances(
    predictions: list[MaskInstance],
    context: CropContext,
) -> tuple[list[MaskInstance], dict[str, Any]]:
    eligible = {
        class_id: sorted(
            (
                item
                for item in predictions
                if item.class_id == class_id
                and item.confidence >= SEGMENTOR_ACCEPTANCE_CONFIDENCE
            ),
            key=lambda item: (-item.confidence, item.instance_index),
        )
        for class_id in SEGMENTOR_CLASS_NAMES
    }
    record = context.record
    target_center = np.array(
        [
            record["target_center_crop_x_pixels"],
            record["target_center_crop_y_pixels"],
        ],
        dtype=float,
    )
    target_box = np.array(
        [
            record["target_box_crop_x1_pixels"],
            record["target_box_crop_y1_pixels"],
            record["target_box_crop_x2_pixels"],
            record["target_box_crop_y2_pixels"],
        ],
        dtype=float,
    )
    target_aware = bool(
        record["other_detection_center_in_crop"]
        or record["other_detection_box_intersects_crop"]
        or len(eligible[0]) > 2
    )

    pair_selector = select_target_aware_pair if target_aware else None
    if pair_selector is None:
        guards, guard_features = select_structured_pair(
            eligible[0],
            None,
            require_distinct_guard_pair=True,
        )
    else:
        guards, guard_features = pair_selector(
            eligible[0],
            target_center,
            target_box,
            None,
            require_distinct_guard_pair=True,
        )

    if target_aware:
        subsidiaries, subsidiary_features = select_target_aware_pair(
            eligible[1],
            target_center,
            target_box,
            guards if len(guards) == 2 else None,
        )
    else:
        subsidiaries, subsidiary_features = select_structured_pair(
            eligible[1],
            guards if len(guards) == 2 else None,
        )

    selected = [*guards, *subsidiaries]
    guard_features = guard_features or {}
    guard_pair_rejection_reason = str(
        guard_features.get("guard_pair_rejection_reason", "")
    )
    guard_pair_geometry_valid = bool(
        len(guards) == 2
        and guard_features.get("guard_pair_geometry_valid", False)
    )
    guard_pair_geometry_rejected = (
        guard_pair_rejection_reason
        == "no_guard_pair_passed_overlap_and_bilateral_gates"
    )
    reported_guard_pair_features = guard_features
    if "mask_iou_between_candidates" not in reported_guard_pair_features:
        reported_guard_pair_features = guard_features.get(
            "best_rejected_pair_features",
            {},
        )
    guard_centroids = [mask_centroid(item.mask) for item in guards]
    both_guard_centroids_inside = bool(
        len(guard_centroids) == 2
        and all(point_inside_box(point, target_box) for point in guard_centroids)
    )
    guard_pair_center_inside = bool(
        len(guard_centroids) == 2
        and point_inside_box(np.mean(guard_centroids, axis=0), target_box)
    )
    if guard_pair_geometry_rejected:
        target_association_status = "guard_pair_geometry_rejected"
    elif len(guards) < 2:
        target_association_status = "insufficient_guard_pair"
    elif both_guard_centroids_inside and guard_pair_center_inside:
        target_association_status = "supported_by_target_box"
    else:
        target_association_status = "review_target_association"

    guard_border = any(mask_touches_border(item.mask) for item in guards)
    subsidiary_border = any(mask_touches_border(item.mask) for item in subsidiaries)
    subsidiary_pair_area_similarity: float | str = ""
    subsidiary_pair_max_to_min_area_ratio: float | str = ""
    if len(subsidiaries) == 2:
        subsidiary_pair_area_similarity = area_similarity(
            subsidiaries[0].mask,
            subsidiaries[1].mask,
        )
        subsidiary_pair_max_to_min_area_ratio = (
            1.0 / subsidiary_pair_area_similarity
            if subsidiary_pair_area_similarity > 0.0
            else math.inf
        )

    guard_only_subsidiary_similarity_gate_required = (
        record["classifier_predicted_label"] == "guard_only"
    )
    if len(subsidiaries) != 2:
        subsidiary_pair_qc_valid = False
        subsidiary_pair_qc_reason = "fewer_than_two_selected_subsidiary_masks"
    elif (
        guard_only_subsidiary_similarity_gate_required
        and subsidiary_pair_area_similarity
        < MIN_GUARD_ONLY_SUBSIDIARY_PAIR_AREA_SIMILARITY
    ):
        subsidiary_pair_qc_valid = False
        subsidiary_pair_qc_reason = (
            "guard_only_subsidiary_area_similarity_below_threshold"
        )
    else:
        subsidiary_pair_qc_valid = True
        subsidiary_pair_qc_reason = "passed"

    if len(guards) == 2 and len(subsidiaries) == 2:
        structure_status = "complete_2_guard_2_subsidiary"
    elif len(guards) == 2:
        structure_status = "guard_pair_only_or_incomplete_subsidiary"
    elif selected:
        structure_status = "partial_instances"
    else:
        structure_status = "no_accepted_masks"

    route = record["segmentation_route"]
    if (
        len(guards) == 2
        and len(subsidiaries) == 2
        and not subsidiary_pair_qc_valid
    ):
        evidence_tier = "guard_only_subsidiary_pair_rejected"
    elif len(guards) == 2 and len(subsidiaries) == 2:
        if route == "confident_full_good":
            evidence_tier = "full_good_complete"
        elif route == "fallback_ranked":
            evidence_tier = "fallback_complete"
        elif route == "post_segmentation_top_up":
            evidence_tier = "post_segmentation_top_up_complete"
        else:
            evidence_tier = "guard_only_complete_conservative"
    elif guard_pair_geometry_rejected:
        evidence_tier = "duplicate_or_same_side_guard_pair_rejected"
    elif len(guards) == 2:
        evidence_tier = "guard_only_guard_valid"
    else:
        evidence_tier = "segmentation_rejected_or_partial"

    if not selected:
        preliminary_qc = "no_accepted_masks"
    elif guard_pair_geometry_rejected:
        preliminary_qc = "review_duplicate_or_same_side_guard_predictions"
    elif guard_border or subsidiary_border:
        preliminary_qc = "review_border_touching"
    elif target_association_status == "review_target_association":
        preliminary_qc = "review_target_association"
    elif (
        len(guards) == 2
        and len(subsidiaries) == 2
        and not subsidiary_pair_qc_valid
    ):
        preliminary_qc = "review_guard_only_subsidiary_area_imbalance"
    elif len(guards) == 2 and len(subsidiaries) == 2:
        preliminary_qc = "complete_candidate_for_later_morphometry_qc"
    elif len(guards) == 2:
        preliminary_qc = "guard_candidate_for_later_morphometry_qc"
    else:
        preliminary_qc = "partial_not_morphometry_ready"

    audit = {
        "raw_guard_predictions": sum(item.class_id == 0 for item in predictions),
        "raw_subsidiary_predictions": sum(item.class_id == 1 for item in predictions),
        "eligible_guard_predictions": len(eligible[0]),
        "eligible_subsidiary_predictions": len(eligible[1]),
        "selected_guard_instances": len(guards),
        "selected_subsidiary_instances": len(subsidiaries),
        "guard_pair_geometry_valid": guard_pair_geometry_valid,
        "guard_pair_geometry_rejected": guard_pair_geometry_rejected,
        "guard_pair_rejection_reason": guard_pair_rejection_reason,
        "guard_pair_candidate_count": int(
            guard_features.get("guard_pair_candidate_count", 0)
        ),
        "guard_pair_valid_candidate_count": int(
            guard_features.get("guard_pair_valid_candidate_count", 0)
        ),
        "selected_guard_pair_mask_iou": reported_guard_pair_features.get(
            "mask_iou_between_candidates",
            "",
        ),
        "selected_guard_pair_bilateral_arrangement": (
            reported_guard_pair_features.get("bilateral_arrangement", "")
        ),
        "guard_only_subsidiary_similarity_gate_required": (
            guard_only_subsidiary_similarity_gate_required
        ),
        "minimum_guard_only_subsidiary_pair_area_similarity": (
            MIN_GUARD_ONLY_SUBSIDIARY_PAIR_AREA_SIMILARITY
        ),
        "subsidiary_pair_area_similarity": subsidiary_pair_area_similarity,
        "subsidiary_pair_max_to_min_area_ratio": (
            subsidiary_pair_max_to_min_area_ratio
        ),
        "subsidiary_pair_qc_valid": subsidiary_pair_qc_valid,
        "subsidiary_pair_qc_reason": subsidiary_pair_qc_reason,
        "segmentation_structure_status": structure_status,
        "segmentation_evidence_tier": evidence_tier,
        "preliminary_segmentation_qc": preliminary_qc,
        "target_association_mode": (
            "target_aware_unvalidated" if target_aware else "locked_structured"
        ),
        "target_association_status": target_association_status,
        "both_guard_centroids_inside_target_box": both_guard_centroids_inside,
        "guard_pair_center_inside_target_box": guard_pair_center_inside,
        "selected_guard_mask_touches_border": guard_border,
        "selected_subsidiary_mask_touches_border": subsidiary_border,
        "selected_any_mask_touches_border": guard_border or subsidiary_border,
        "guard_pair_selection_features_json": json.dumps(
            json_safe(guard_features or {}), sort_keys=True
        ),
        "subsidiary_pair_selection_features_json": json.dumps(
            json_safe(subsidiary_features or {}), sort_keys=True
        ),
    }
    return selected, audit


# =============================================================================
# Mask and reference-image outputs
# =============================================================================


def save_selected_mask_arrays(
    path: Path,
    selected: list[MaskInstance],
    context: CropContext,
) -> None:
    masks = np.stack([item.mask.astype(np.uint8) for item in selected], axis=0)
    np.savez_compressed(
        path,
        masks=masks,
        class_ids=np.array([item.class_id for item in selected], dtype=np.int8),
        confidences=np.array(
            [item.confidence for item in selected], dtype=np.float32
        ),
        instance_indices=np.array(
            [item.instance_index for item in selected], dtype=np.int16
        ),
        crop_origin_xy=np.array(
            [context.record["crop_x1"], context.record["crop_y1"]],
            dtype=np.int32,
        ),
        crop_shape_hw=np.array(context.crop_gray.shape, dtype=np.int32),
        source_shape_hw=np.array(
            [
                context.record["source_image_height"],
                context.record["source_image_width"],
            ],
            dtype=np.int32,
        ),
        target_center_crop_xy=np.array(
            [
                context.record["target_center_crop_x_pixels"],
                context.record["target_center_crop_y_pixels"],
            ],
            dtype=np.float32,
        ),
        target_box_crop_xyxy=np.array(
            [
                context.record["target_box_crop_x1_pixels"],
                context.record["target_box_crop_y1_pixels"],
                context.record["target_box_crop_x2_pixels"],
                context.record["target_box_crop_y2_pixels"],
            ],
            dtype=np.float32,
        ),
    )


def make_mask_overlay(
    context: CropContext,
    selected: list[MaskInstance],
) -> np.ndarray:
    overlay = grayscale_to_three_channels(context.crop_gray).copy()
    colors_bgr = {
        0: np.array([255, 170, 40], dtype=np.float32),
        1: np.array([150, 90, 255], dtype=np.float32),
    }
    alpha = 0.35
    for item in selected:
        mask = item.mask.astype(bool)
        color = colors_bgr[item.class_id]
        overlay[mask] = np.clip(
            (1.0 - alpha) * overlay[mask].astype(np.float32) + alpha * color,
            0,
            255,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        bgr_color = tuple(int(value) for value in color)
        cv2.drawContours(overlay, contours, -1, bgr_color, 1, cv2.LINE_AA)

    target = (
        int(round(context.record["target_center_crop_x_pixels"])),
        int(round(context.record["target_center_crop_y_pixels"])),
    )
    # Drawing functions treat the array as BGR; the underlying grayscale is
    # identical and mask colors are display-only.
    cv2.drawMarker(
        overlay,
        target,
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=8,
        thickness=1,
    )
    return overlay


def save_png(path: Path, image: np.ndarray) -> None:
    success = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not success:
        raise RuntimeError(f"Failed to save PNG: {path}")


def save_overview(
    path: Path,
    gray: np.ndarray,
    contexts: list[CropContext],
    hair_cells: list[dict[str, Any]],
    jpeg_quality: int,
) -> None:
    overview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Draw hair cells first so the richer stoma QC boxes remain visible where
    # detections overlap. H-prefixed labels distinguish hair cells from the
    # unprefixed deterministic stoma indices.
    for record in hair_cells:
        x1 = int(round(record["prediction_x1_pixels"]))
        y1 = int(round(record["prediction_y1_pixels"]))
        x2 = int(round(record["prediction_x2_pixels"]))
        y2 = int(round(record["prediction_y2_pixels"]))
        cv2.rectangle(
            overview,
            (x1, y1),
            (x2, y2),
            HAIR_CELL_OVERVIEW_COLOR_BGR,
            2,
        )
        label = f"H{record['hair_cell_number']}"
        text_origin = (x1, max(14, y1 - 4))
        cv2.putText(
            overview,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overview,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            HAIR_CELL_OVERVIEW_COLOR_BGR,
            1,
            cv2.LINE_AA,
        )

    for context in contexts:
        record = context.record
        status = record.get("segmentation_structure_status", "not_attempted")
        if record.get("preliminary_usable_complete", False):
            color = (70, 190, 70)
        elif status == "complete_2_guard_2_subsidiary":
            # Raw 2+2 count exists, but at least one geometry, association or
            # border QC rule rejected it as a usable complete complex.
            color = (180, 80, 180)
        elif status == "guard_pair_only_or_incomplete_subsidiary":
            color = (220, 150, 30)
        elif record.get("selected_for_segmentation"):
            color = (40, 140, 240)
        else:
            color = (150, 150, 150)
        x1 = int(round(record["prediction_x1_pixels"]))
        y1 = int(round(record["prediction_y1_pixels"]))
        x2 = int(round(record["prediction_x2_pixels"]))
        y2 = int(round(record["prediction_y2_pixels"]))
        cv2.rectangle(overview, (x1, y1), (x2, y2), color, 2)
        label = str(record["stoma_number"])
        text_origin = (x1, max(14, y1 - 4))
        cv2.putText(
            overview,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overview,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    success = cv2.imwrite(
        str(path),
        overview,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not success:
        raise RuntimeError(f"Failed to save overview JPEG: {path}")


def instance_rows(
    selected: list[MaskInstance],
    context: CropContext,
    mask_archive_relative: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    record = context.record
    target_center = np.array(
        [
            context.record["target_center_crop_x_pixels"],
            context.record["target_center_crop_y_pixels"],
        ],
        dtype=float,
    )
    for stored_index, item in enumerate(selected):
        centroid = mask_centroid(item.mask)
        if item.class_id == 0:
            future_trait_scope = "guard_cell_pair"
            if not record["guard_pair_geometry_valid"]:
                future_trait_eligible = False
                future_trait_exclusion_reason = "guard_pair_geometry_invalid"
            elif record["target_association_status"] != "supported_by_target_box":
                future_trait_eligible = False
                future_trait_exclusion_reason = "target_association_not_supported"
            elif record["selected_guard_mask_touches_border"]:
                future_trait_eligible = False
                future_trait_exclusion_reason = "guard_mask_touches_crop_border"
            else:
                future_trait_eligible = True
                future_trait_exclusion_reason = ""
        else:
            future_trait_scope = "paired_subsidiary_cells"
            if not record["guard_pair_geometry_valid"]:
                future_trait_eligible = False
                future_trait_exclusion_reason = "guard_pair_geometry_invalid"
            elif not record["subsidiary_pair_qc_valid"]:
                future_trait_eligible = False
                future_trait_exclusion_reason = record[
                    "subsidiary_pair_qc_reason"
                ]
            elif record["target_association_status"] != "supported_by_target_box":
                future_trait_eligible = False
                future_trait_exclusion_reason = "target_association_not_supported"
            elif record["selected_subsidiary_mask_touches_border"]:
                future_trait_eligible = False
                future_trait_exclusion_reason = (
                    "subsidiary_mask_touches_crop_border"
                )
            else:
                future_trait_eligible = True
                future_trait_exclusion_reason = ""
        rows.append(
            {
                "source_image_name": context.record["source_image_name"],
                "surface": context.record["surface"],
                "stoma_id": context.record["stoma_id"],
                "stoma_number": context.record["stoma_number"],
                "classifier_predicted_label": context.record[
                    "classifier_predicted_label"
                ],
                "probability_full_good": context.record["probability_full_good"],
                "segmentation_route": context.record["segmentation_route"],
                "post_segmentation_top_up": context.record[
                    "post_segmentation_top_up"
                ],
                "segmentation_evidence_tier": context.record[
                    "segmentation_evidence_tier"
                ],
                "guard_pair_geometry_valid": context.record[
                    "guard_pair_geometry_valid"
                ],
                "guard_pair_geometry_rejected": context.record[
                    "guard_pair_geometry_rejected"
                ],
                "subsidiary_pair_area_similarity": context.record[
                    "subsidiary_pair_area_similarity"
                ],
                "subsidiary_pair_qc_valid": context.record[
                    "subsidiary_pair_qc_valid"
                ],
                "subsidiary_pair_qc_reason": context.record[
                    "subsidiary_pair_qc_reason"
                ],
                "preliminary_usable_complete": context.record[
                    "preliminary_usable_complete"
                ],
                "target_association_status": context.record[
                    "target_association_status"
                ],
                "stored_mask_index": stored_index,
                "model_instance_index": item.instance_index,
                "class_id": item.class_id,
                "model_class_name": SEGMENTOR_CLASS_NAMES[item.class_id],
                "display_class_name": (
                    "subsidiary_cell" if item.class_id == 1 else "guard_cell"
                ),
                "confidence": item.confidence,
                "centroid_crop_x_pixels": float(centroid[0]),
                "centroid_crop_y_pixels": float(centroid[1]),
                "centroid_source_x_pixels": float(
                    centroid[0] + context.record["crop_x1"]
                ),
                "centroid_source_y_pixels": float(
                    centroid[1] + context.record["crop_y1"]
                ),
                "centroid_distance_to_target_pixels": float(
                    np.linalg.norm(centroid - target_center)
                ),
                "mask_touches_crop_border": mask_touches_border(item.mask),
                "future_trait_scope": future_trait_scope,
                "future_trait_eligible": future_trait_eligible,
                "future_trait_exclusion_reason": future_trait_exclusion_reason,
                "mask_archive_path": mask_archive_relative,
                "morphometry_calculated": False,
            }
        )
    return rows


def initialise_segmentation_record(context: CropContext) -> None:
    """Create explicit defaults before any segmentation attempt."""
    context.record.update(
        {
            "segmentation_attempted": False,
            "segmentation_status": "not_selected_by_classifier",
            "segmentation_structure_status": "not_attempted",
            "segmentation_evidence_tier": "not_segmented",
            "preliminary_segmentation_qc": "not_segmented",
            "preliminary_usable_complete": False,
            "raw_guard_predictions": 0,
            "raw_subsidiary_predictions": 0,
            "eligible_guard_predictions": 0,
            "eligible_subsidiary_predictions": 0,
            "selected_guard_instances": 0,
            "selected_subsidiary_instances": 0,
            "guard_pair_geometry_valid": False,
            "guard_pair_geometry_rejected": False,
            "guard_pair_rejection_reason": "not_attempted",
            "guard_pair_candidate_count": 0,
            "guard_pair_valid_candidate_count": 0,
            "selected_guard_pair_mask_iou": "",
            "selected_guard_pair_bilateral_arrangement": "",
            "guard_only_subsidiary_similarity_gate_required": False,
            "minimum_guard_only_subsidiary_pair_area_similarity": (
                MIN_GUARD_ONLY_SUBSIDIARY_PAIR_AREA_SIMILARITY
            ),
            "subsidiary_pair_area_similarity": "",
            "subsidiary_pair_max_to_min_area_ratio": "",
            "subsidiary_pair_qc_valid": False,
            "subsidiary_pair_qc_reason": "not_attempted",
            "target_association_mode": "not_attempted",
            "target_association_status": "not_attempted",
            "selected_guard_mask_touches_border": False,
            "selected_subsidiary_mask_touches_border": False,
            "selected_any_mask_touches_border": False,
            "mask_archive_path": "",
            "masked_crop_reference_path": "",
            "morphometry_calculated": False,
        }
    )


def is_preliminary_usable_complete(record: dict[str, Any]) -> bool:
    """Apply the conservative, pre-morphometry definition used for top-up."""
    return bool(
        record["segmentation_attempted"]
        and record["segmentation_structure_status"]
        == "complete_2_guard_2_subsidiary"
        and record["guard_pair_geometry_valid"]
        and record["subsidiary_pair_qc_valid"]
        and record["target_association_status"] == "supported_by_target_box"
        and not record["selected_any_mask_touches_border"]
    )


def segment_and_save_context(
    context: CropContext,
    segmentor: YOLO,
    device_name: str,
    staging_dir: Path,
    mask_rows: list[dict[str, Any]],
) -> None:
    """Run segmentation once for one crop and save only accepted masks."""
    record = context.record
    if record["segmentation_attempted"]:
        raise RuntimeError(
            f"Segmentation was requested twice for {record['stoma_id']}."
        )

    record["segmentation_attempted"] = True
    predictions = predict_cell_masks(segmentor, context.crop_rgb, device_name)
    selected, audit = select_cell_instances(predictions, context)
    record.update(audit)
    record["preliminary_usable_complete"] = is_preliminary_usable_complete(record)
    record["segmentation_status"] = (
        "accepted_masks_saved" if selected else "no_accepted_masks"
    )

    if selected:
        archive_relative = f"masks/{record['stoma_id']}_masks.npz"
        overlay_relative = f"masked_crops/{record['stoma_id']}_masked.png"
        archive_path = resolve_inside(staging_dir / archive_relative, staging_dir)
        overlay_path = resolve_inside(staging_dir / overlay_relative, staging_dir)
        save_selected_mask_arrays(archive_path, selected, context)
        save_png(overlay_path, make_mask_overlay(context, selected))
        record["mask_archive_path"] = archive_relative
        record["masked_crop_reference_path"] = overlay_relative
        mask_rows.extend(instance_rows(selected, context, archive_relative))
    empty_device_cache(device_name)


# =============================================================================
# Per-image integrated processing
# =============================================================================


def process_one_image(
    image_path: Path,
    detector: YOLO,
    classifier: nn.Module,
    segmentor: YOLO,
    classifier_device: torch.device,
    args: argparse.Namespace,
    staging_dir: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    biological = parse_biological_name(image_path)
    source_sha = sha256_file(image_path)
    ome_metadata = read_ome_metadata(image_path)
    calibration = assess_calibration(
        ome_metadata,
        args.calibration_reference,
        args.calibration_absolute_tolerance,
        args.calibration_policy,
    )
    if (
        args.stop_on_calibration_problem
        and not calibration["calibrated_traits_permitted"]
    ):
        raise RuntimeError(
            f"Calibration preflight failed for {image_path.name}: "
            f"{calibration['calibration_status']}"
        )

    gray, pixel_reader = read_original_grayscale(
        image_path,
        expected_width=args.expected_image_width,
        expected_height=args.expected_image_height,
    )
    image_rgb = grayscale_to_three_channels(gray)
    source_metadata = {
        **biological,
        "source_image_name": image_path.name,
    }

    (
        retained_stomata,
        retained_hair_cells,
        removed_nms,
        detection_summary,
    ) = detect_stomata_and_hair_cells(
        detector,
        image_rgb,
        source_metadata,
        source_sha,
        args.runtime_device,
    )
    empty_device_cache(args.runtime_device)

    detector_limit_reached = bool(detection_summary["detector_max_det_reached"])
    for record in retained_hair_cells:
        touches_image_edge = bool(
            record["prediction_x1_pixels"] <= 0.0
            or record["prediction_y1_pixels"] <= 0.0
            or record["prediction_x2_pixels"] >= gray.shape[1]
            or record["prediction_y2_pixels"] >= gray.shape[0]
        )
        is_ab = record["surface"] == "AB"
        primary_eligible = bool(
            is_ab
            and calibration["calibrated_traits_permitted"]
            and not detector_limit_reached
        )
        qc_flags: list[str] = []
        if is_ab:
            qc_flags.append("AB_HAIR_DENSITY_CANDIDATE")
        else:
            qc_flags.append("AD_HAIR_VISIBILITY_LIMITED_BY_GROOVES")
        if detector_limit_reached:
            qc_flags.append("DETECTOR_MAX_DET_REACHED")
        if touches_image_edge:
            qc_flags.append("HAIR_BOX_TOUCHES_IMAGE_EDGE")
        if not calibration["calibrated_traits_permitted"]:
            qc_flags.append("CALIBRATION_QC_FAILED")
        record.update(
            {
                "hair_cell_box_touches_image_edge": touches_image_edge,
                "hair_cell_visibility_interpretation": (
                    "AB_primary_density_candidate"
                    if is_ab
                    else "AD_visible_hair_cells_only_groove_limited"
                ),
                "hair_cell_density_primary_trait_eligible": primary_eligible,
                "hair_cell_qc_flags": ";".join(qc_flags),
                "calibration_status": calibration["calibration_status"],
                "calibrated_traits_permitted": calibration[
                    "calibrated_traits_permitted"
                ],
                "hair_cell_density_calculated": False,
            }
        )

    contexts = prepare_crops(gray, retained_stomata, calibration)
    classify_crops(
        classifier,
        contexts,
        classifier_device,
        args.classifier_batch_size,
    )
    routing_summary = apply_classifier_routing(contexts)
    empty_device_cache(args.runtime_device)

    mask_rows: list[dict[str, Any]] = []
    for context in contexts:
        initialise_segmentation_record(context)

    initially_selected = [
        context
        for context in contexts
        if context.record["initially_selected_for_segmentation"]
    ]
    for context in initially_selected:
        segment_and_save_context(
            context,
            segmentor,
            args.runtime_device,
            staging_dir,
            mask_rows,
        )

    initial_usable_count = sum(
        bool(context.record["preliminary_usable_complete"])
        for context in initially_selected
    )
    current_usable_count = initial_usable_count
    top_up_candidates = sorted(
        (
            context
            for context in contexts
            if not context.record["segmentation_attempted"]
        ),
        key=lambda context: (
            -context.record["probability_full_good"],
            context.record["stoma_id"],
        ),
    )
    top_up_attempt_count = 0
    top_up_usable_gain = 0
    for context in top_up_candidates:
        if current_usable_count >= MINIMUM_USABLE_COMPLEXES_AFTER_SEGMENTATION:
            break

        top_up_attempt_count += 1
        context.record.update(
            {
                "selected_for_segmentation": True,
                "segmentation_route": "post_segmentation_top_up",
                "full_complex_candidate": True,
                "post_segmentation_top_up": True,
                "post_segmentation_top_up_rank": top_up_attempt_count,
                "usable_complete_count_before_top_up_attempt": (
                    current_usable_count
                ),
            }
        )
        segment_and_save_context(
            context,
            segmentor,
            args.runtime_device,
            staging_dir,
            mask_rows,
        )
        if context.record["preliminary_usable_complete"]:
            current_usable_count += 1
            top_up_usable_gain += 1

    remaining_unsegmented_count = sum(
        not context.record["segmentation_attempted"] for context in contexts
    )
    top_up_requested = (
        initial_usable_count < MINIMUM_USABLE_COMPLEXES_AFTER_SEGMENTATION
    )
    top_up_target_met = (
        current_usable_count >= MINIMUM_USABLE_COMPLEXES_AFTER_SEGMENTATION
    )
    top_up_exhausted = bool(
        top_up_requested and not top_up_target_met and remaining_unsegmented_count == 0
    )

    overview_relative = f"overviews/{biological['source_base_name']}_detections.jpg"
    save_overview(
        resolve_inside(staging_dir / overview_relative, staging_dir),
        gray,
        contexts,
        retained_hair_cells,
        args.overview_jpeg_quality,
    )

    input_row = {
        "source_image_name": image_path.name,
        "source_image_path": str(image_path),
        "source_image_sha256": source_sha,
        **biological,
        "image_width": gray.shape[1],
        "image_height": gray.shape[0],
        "configured_expected_image_width": args.expected_image_width,
        "configured_expected_image_height": args.expected_image_height,
        "configured_resolution_is_default": bool(
            args.expected_image_width == EXPECTED_IMAGE_WIDTH
            and args.expected_image_height == EXPECTED_IMAGE_HEIGHT
        ),
        "source_channels": 1,
        "source_dtype": str(gray.dtype),
        "pixel_reader": pixel_reader,
        "three_channel_conversion": "in_memory_identical_channels",
        "source_file_modified": False,
        **ome_metadata,
        **calibration,
        "indexed_overview_path": overview_relative,
    }

    stoma_rows = [context.record for context in contexts]
    image_summary = {
        "source_image_name": image_path.name,
        **biological,
        "calibration_status": calibration["calibration_status"],
        "calibration_policy": calibration["calibration_policy"],
        "calibration_reference_match": calibration["calibration_reference_match"],
        "calibration_metadata_used_for_traits": calibration[
            "calibration_metadata_used_for_traits"
        ],
        "calibration_warning": calibration["calibration_warning"],
        "calibration_provisional": calibration["calibration_provisional"],
        "stage_micrometer_verified": calibration["stage_micrometer_verified"],
        "calibrated_traits_permitted": calibration[
            "calibrated_traits_permitted"
        ],
        # These three historical fields remain stoma aliases so downstream
        # readers of earlier trial outputs are not silently broken.
        "detector_predictions_before_manual_nms": detection_summary[
            "stoma_predictions_before_manual_nms"
        ],
        "detector_predictions_after_manual_nms": detection_summary[
            "stoma_predictions_after_manual_nms"
        ],
        "manual_nms_duplicates_removed": detection_summary[
            "stoma_manual_nms_duplicates_removed"
        ],
        **detection_summary,
        "hair_cell_count": len(retained_hair_cells),
        "hair_cell_density_calculated": False,
        "hair_cell_density_primary_trait_eligible": bool(
            biological["surface"] == "AB"
            and calibration["calibrated_traits_permitted"]
            and not detector_limit_reached
        ),
        "hair_cell_edge_touching_box_count": sum(
            bool(record["hair_cell_box_touches_image_edge"])
            for record in retained_hair_cells
        ),
        "hair_cell_visibility_interpretation": (
            "AB_primary_density_candidate"
            if biological["surface"] == "AB"
            else "AD_visible_hair_cells_only_groove_limited"
        ),
        "hair_cell_image_qc_flag": ";".join(
            [
                (
                    "AB_HAIR_DENSITY_CANDIDATE"
                    if biological["surface"] == "AB"
                    else "AD_HAIR_VISIBILITY_LIMITED_BY_GROOVES"
                ),
                *(
                    ["DETECTOR_MAX_DET_REACHED"]
                    if detector_limit_reached
                    else []
                ),
                *(
                    ["CALIBRATION_QC_FAILED"]
                    if not calibration["calibrated_traits_permitted"]
                    else []
                ),
            ]
        ),
        **routing_summary,
        "initial_full_complex_candidate_count": routing_summary[
            "full_complex_candidate_count"
        ],
        "initial_total_selected_for_segmentation": routing_summary[
            "total_selected_for_segmentation"
        ],
        "full_complex_candidate_count": sum(
            bool(row["full_complex_candidate"]) for row in stoma_rows
        ),
        "total_selected_for_segmentation": sum(
            bool(row["selected_for_segmentation"]) for row in stoma_rows
        ),
        "minimum_usable_complexes_after_segmentation_requested": (
            MINIMUM_USABLE_COMPLEXES_AFTER_SEGMENTATION
        ),
        "initial_preliminary_usable_complete_count": initial_usable_count,
        "post_segmentation_top_up_requested": top_up_requested,
        "post_segmentation_top_up_triggered": top_up_attempt_count > 0,
        "post_segmentation_top_up_attempt_count": top_up_attempt_count,
        "post_segmentation_top_up_usable_gain": top_up_usable_gain,
        "final_preliminary_usable_complete_count": current_usable_count,
        "post_segmentation_top_up_target_met": top_up_target_met,
        "post_segmentation_top_up_exhausted": top_up_exhausted,
        "remaining_unsegmented_crop_count": remaining_unsegmented_count,
        "segmentation_attempted_count": sum(
            bool(row["segmentation_attempted"]) for row in stoma_rows
        ),
        "segmentation_with_saved_masks_count": sum(
            bool(row["mask_archive_path"]) for row in stoma_rows
        ),
        "complete_2plus2_count": sum(
            row["segmentation_structure_status"]
            == "complete_2_guard_2_subsidiary"
            for row in stoma_rows
        ),
        "complete_2plus2_but_qc_flagged_count": sum(
            row["segmentation_structure_status"]
            == "complete_2_guard_2_subsidiary"
            and not row["preliminary_usable_complete"]
            for row in stoma_rows
        ),
        "guard_pair_only_count": sum(
            row["segmentation_structure_status"]
            == "guard_pair_only_or_incomplete_subsidiary"
            for row in stoma_rows
        ),
        "partial_instances_count": sum(
            row["segmentation_structure_status"] == "partial_instances"
            for row in stoma_rows
        ),
        "guard_pair_geometry_rejected_count": sum(
            bool(row["guard_pair_geometry_rejected"]) for row in stoma_rows
        ),
        "guard_only_subsidiary_pair_similarity_rejected_count": sum(
            bool(row["guard_only_subsidiary_similarity_gate_required"])
            and row["selected_subsidiary_instances"] == 2
            and not row["subsidiary_pair_qc_valid"]
            for row in stoma_rows
        ),
        "no_accepted_masks_count": sum(
            row["segmentation_status"] == "no_accepted_masks"
            for row in stoma_rows
        ),
        "target_association_review_count": sum(
            row["target_association_status"] == "review_target_association"
            for row in stoma_rows
        ),
        "target_aware_unvalidated_count": sum(
            row["target_association_mode"] == "target_aware_unvalidated"
            for row in stoma_rows
        ),
        "border_touching_selected_mask_count": sum(
            bool(row["selected_any_mask_touches_border"]) for row in stoma_rows
        ),
        "desired_three_complete_complexes_met": sum(
            bool(row["preliminary_usable_complete"]) for row in stoma_rows
        )
        >= MINIMUM_USABLE_COMPLEXES_AFTER_SEGMENTATION,
        "traits_calculated": False,
        "indexed_overview_path": overview_relative,
    }
    return (
        input_row,
        stoma_rows,
        retained_hair_cells,
        mask_rows,
        image_summary,
        removed_nms,
    )


# =============================================================================
# Run-level output and provenance
# =============================================================================


INPUT_FIELDS = (
    "source_image_name",
    "source_image_path",
    "source_image_sha256",
    "surface",
    "slide_id",
    "plant_replicate",
    "biological_unit",
    "image_width",
    "image_height",
    "configured_expected_image_width",
    "configured_expected_image_height",
    "configured_resolution_is_default",
    "source_channels",
    "source_dtype",
    "pixel_reader",
    "ome_metadata_present",
    "physical_size_x_raw",
    "physical_size_x_unit_raw",
    "physical_size_y_raw",
    "physical_size_y_unit_raw",
    "physical_size_x_um",
    "physical_size_y_um",
    "metadata_pixels_per_um_x",
    "metadata_pixels_per_um_y",
    "calibration_reference_pixels_per_um",
    "calibration_reference_source",
    "calibration_policy",
    "calibration_reference_match",
    "calibration_metadata_used_for_traits",
    "calibration_warning",
    "calibration_provisional",
    "stage_micrometer_verified",
    "calibrated_traits_permitted",
    "calibration_status",
    "indexed_overview_path",
)

STOMA_FIELDS = (
    "source_image_name",
    "surface",
    "slide_id",
    "plant_replicate",
    "biological_unit",
    "stoma_number",
    "stoma_id",
    "detector_confidence",
    "prediction_x1_pixels",
    "prediction_y1_pixels",
    "prediction_x2_pixels",
    "prediction_y2_pixels",
    "prediction_center_x_pixels",
    "prediction_center_y_pixels",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "edge_clipped",
    "other_detection_center_in_crop",
    "other_detection_center_ids",
    "other_detection_box_intersects_crop",
    "classifier_predicted_label",
    "probability_full_good",
    "probability_guard_only",
    "probability_bad",
    "segmentation_route",
    "selected_for_segmentation",
    "initially_selected_for_segmentation",
    "full_complex_candidate",
    "post_segmentation_top_up",
    "post_segmentation_top_up_rank",
    "usable_complete_count_before_top_up_attempt",
    "segmentation_attempted",
    "segmentation_status",
    "selected_guard_instances",
    "selected_subsidiary_instances",
    "guard_pair_geometry_valid",
    "guard_pair_geometry_rejected",
    "guard_pair_rejection_reason",
    "guard_pair_candidate_count",
    "guard_pair_valid_candidate_count",
    "selected_guard_pair_mask_iou",
    "selected_guard_pair_bilateral_arrangement",
    "guard_only_subsidiary_similarity_gate_required",
    "minimum_guard_only_subsidiary_pair_area_similarity",
    "subsidiary_pair_area_similarity",
    "subsidiary_pair_max_to_min_area_ratio",
    "subsidiary_pair_qc_valid",
    "subsidiary_pair_qc_reason",
    "segmentation_structure_status",
    "segmentation_evidence_tier",
    "preliminary_segmentation_qc",
    "preliminary_usable_complete",
    "target_association_mode",
    "target_association_status",
    "selected_any_mask_touches_border",
    "mask_archive_path",
    "masked_crop_reference_path",
    "calibration_status",
    "calibration_provisional",
    "stage_micrometer_verified",
    "calibrated_traits_permitted",
    "morphometry_calculated",
)

HAIR_CELL_FIELDS = (
    "source_image_name",
    "surface",
    "slide_id",
    "plant_replicate",
    "biological_unit",
    "hair_cell_number",
    "hair_cell_id",
    "detector_class_id",
    "detector_class_name",
    "detector_confidence",
    "pre_nms_prediction_index",
    "prediction_x1_pixels",
    "prediction_y1_pixels",
    "prediction_x2_pixels",
    "prediction_y2_pixels",
    "prediction_center_x_pixels",
    "prediction_center_y_pixels",
    "prediction_width_pixels",
    "prediction_height_pixels",
    "source_image_width",
    "source_image_height",
    "hair_cell_box_touches_image_edge",
    "hair_cell_visibility_interpretation",
    "hair_cell_density_primary_trait_eligible",
    "hair_cell_qc_flags",
    "calibration_status",
    "calibrated_traits_permitted",
    "hair_cell_density_calculated",
)

MASK_FIELDS = (
    "source_image_name",
    "surface",
    "stoma_id",
    "stoma_number",
    "classifier_predicted_label",
    "probability_full_good",
    "segmentation_route",
    "post_segmentation_top_up",
    "segmentation_evidence_tier",
    "guard_pair_geometry_valid",
    "guard_pair_geometry_rejected",
    "subsidiary_pair_area_similarity",
    "subsidiary_pair_qc_valid",
    "subsidiary_pair_qc_reason",
    "preliminary_usable_complete",
    "target_association_status",
    "stored_mask_index",
    "model_instance_index",
    "class_id",
    "model_class_name",
    "display_class_name",
    "confidence",
    "centroid_crop_x_pixels",
    "centroid_crop_y_pixels",
    "centroid_source_x_pixels",
    "centroid_source_y_pixels",
    "centroid_distance_to_target_pixels",
    "mask_touches_crop_border",
    "future_trait_scope",
    "future_trait_eligible",
    "future_trait_exclusion_reason",
    "mask_archive_path",
    "morphometry_calculated",
)

IMAGE_SUMMARY_FIELDS = (
    "source_image_name",
    "surface",
    "slide_id",
    "plant_replicate",
    "biological_unit",
    "calibration_status",
    "calibration_policy",
    "calibration_reference_match",
    "calibration_metadata_used_for_traits",
    "calibration_warning",
    "calibration_provisional",
    "stage_micrometer_verified",
    "calibrated_traits_permitted",
    "detector_predictions_before_manual_nms",
    "detector_predictions_after_manual_nms",
    "manual_nms_duplicates_removed",
    "joint_predictions_at_confidence_floor",
    "detector_max_det_reached",
    "stoma_predictions_before_manual_nms",
    "stoma_predictions_after_manual_nms",
    "stoma_manual_nms_duplicates_removed",
    "hair_cell_predictions_before_manual_nms",
    "hair_cell_predictions_after_manual_nms",
    "hair_cell_manual_nms_duplicates_removed",
    "hair_cell_count",
    "hair_cell_density_calculated",
    "hair_cell_density_primary_trait_eligible",
    "hair_cell_edge_touching_box_count",
    "hair_cell_visibility_interpretation",
    "hair_cell_image_qc_flag",
    "confident_full_good_count",
    "fallback_candidate_count",
    "guard_only_route_count",
    "initial_full_complex_candidate_count",
    "initial_total_selected_for_segmentation",
    "full_complex_candidate_count",
    "total_selected_for_segmentation",
    "minimum_usable_complexes_after_segmentation_requested",
    "initial_preliminary_usable_complete_count",
    "post_segmentation_top_up_requested",
    "post_segmentation_top_up_triggered",
    "post_segmentation_top_up_attempt_count",
    "post_segmentation_top_up_usable_gain",
    "final_preliminary_usable_complete_count",
    "post_segmentation_top_up_target_met",
    "post_segmentation_top_up_exhausted",
    "remaining_unsegmented_crop_count",
    "segmentation_attempted_count",
    "segmentation_with_saved_masks_count",
    "complete_2plus2_count",
    "complete_2plus2_but_qc_flagged_count",
    "guard_pair_only_count",
    "partial_instances_count",
    "guard_pair_geometry_rejected_count",
    "guard_only_subsidiary_pair_similarity_rejected_count",
    "no_accepted_masks_count",
    "target_association_review_count",
    "target_aware_unvalidated_count",
    "border_touching_selected_mask_count",
    "desired_three_complete_complexes_met",
    "traits_calculated",
    "indexed_overview_path",
)

NMS_FIELDS = (
    "source_image_name",
    "surface",
    "detector_class_id",
    "detector_class_name",
    "removed_prediction_index",
    "removed_confidence",
    "removed_box_xyxy",
    "retained_prediction_index",
    "retained_confidence",
    "retained_box_xyxy",
    "overlap_iou",
)


def make_run_report(
    image_rows: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    return "\n".join(
        (
            "Brachypodium OME-TIFF to cell-segmentation pipeline",
            "",
            f"Completed: {utc_now()}",
            f"Output: {output_dir}",
            f"Input images: {len(image_rows)}",
            "Joint detector inference confidence floor: 0.34",
            "Stoma confidence: 0.40; manual stoma-only NMS IoU: 0.70",
            "Hair-cell confidence: 0.34; manual hair-only NMS IoU: 0.70",
            "Crop padding: 30 pixels",
            "Classifier full-good gate: 0.75",
            "Minimum full-complex candidates per image when available: 5",
            "Guard-only classifier route: enabled",
            "Segmentor accepted-instance confidence: 0.20",
            "Segmentor internal NMS IoU: 0.70",
            "Maximum valid guard-pair mask IoU: 0.10",
            "Minimum valid guard-pair bilateral arrangement: 0.50",
            "Invalid guard pairs are reduced to the best single guard mask",
            "Minimum guard_only subsidiary-pair area similarity: 0.50",
            "Failed guard_only subsidiary pairs remain saved but are excluded from future subsidiary traits",
            "Post-segmentation top-up target: 3 preliminary usable complete complexes",
            "Top-up order: descending full_good probability among unused crops",
            "Usable definition: geometry-valid guard pair, accepted subsidiary pair, complete 2+2, target-box supported, no selected mask touching the crop border",
            "",
            f"Detected stomata after manual NMS: {sum(row['detector_predictions_after_manual_nms'] for row in image_rows)}",
            f"Detected hair cells after manual NMS: {sum(row['hair_cell_count'] for row in image_rows)}",
            f"Detected AB hair cells: {sum(row['hair_cell_count'] for row in image_rows if row['surface'] == 'AB')}",
            f"Detected AD visible hair cells: {sum(row['hair_cell_count'] for row in image_rows if row['surface'] == 'AD')}",
            f"Segmentation attempts: {sum(row['segmentation_attempted_count'] for row in image_rows)}",
            f"Post-segmentation top-up attempts: {sum(row['post_segmentation_top_up_attempt_count'] for row in image_rows)}",
            f"Usable complete results gained by top-up: {sum(row['post_segmentation_top_up_usable_gain'] for row in image_rows)}",
            f"Images meeting the three-usable-complex target: {sum(row['post_segmentation_top_up_target_met'] for row in image_rows)}/{len(image_rows)}",
            f"Images exhausting all detected crops below target: {sum(row['post_segmentation_top_up_exhausted'] for row in image_rows)}",
            f"Stomata with saved masks: {sum(row['segmentation_with_saved_masks_count'] for row in image_rows)}",
            f"Complete 2+2 results: {sum(row['complete_2plus2_count'] for row in image_rows)}",
            f"Raw 2+2 results rejected by QC: {sum(row['complete_2plus2_but_qc_flagged_count'] for row in image_rows)}",
            f"Guard-pair-only results: {sum(row['guard_pair_only_count'] for row in image_rows)}",
            f"Partial-instance results: {sum(row['partial_instances_count'] for row in image_rows)}",
            f"Duplicate/same-side guard pairs rejected: {sum(row['guard_pair_geometry_rejected_count'] for row in image_rows)}",
            f"Guard-only subsidiary pairs rejected for area imbalance: {sum(row['guard_only_subsidiary_pair_similarity_rejected_count'] for row in image_rows)}",
            f"Target-association review cases: {sum(row['target_association_review_count'] for row in image_rows)}",
            "Overview colours: yellow H#=hair cell; green=QC-accepted complete stoma; purple=raw 2+2 but QC-flagged stoma; blue=valid guard pair with incomplete subsidiaries; orange=other attempted partial stoma; grey=stoma not segmented",
            "",
            "AB hair-cell counts are retained as candidates for later calibrated density extraction.",
            "AD hair-cell counts represent visible cells only and are flagged as groove-limited; they are not primary density traits.",
            "Micrometer calibration was verified when the microscope was installed.",
            "OME metadata is checked against the verified 4.5848 pixels/um reference for every image.",
            f"Images passing per-image calibration QC: {sum(row['calibrated_traits_permitted'] for row in image_rows)}/{len(image_rows)}",
            "Passing images provide calibrated inputs to the trait-extraction stage.",
            "Original OME-TIFFs were not modified.",
            "Ordinary unmasked crops were not saved.",
            "Compressed mask arrays are retained for later trait extraction.",
            "",
        )
    )


def run_pipeline() -> Path:
    args = parse_arguments()
    input_path, output_dir, staging_dir = validate_arguments(args)
    images = discover_images(input_path, args.recursive)
    checkpoint_record = verify_checkpoints_and_versions(
        args.allow_ultralytics_version_mismatch
    )
    classifier_device = require_requested_device(args.device, args.cuda_device)
    args.runtime_device = str(classifier_device)

    detector = load_detector()
    classifier, classifier_metadata = load_classifier(classifier_device)
    segmentor = load_segmentor()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=False, exist_ok=False)
    for name in ("overviews", "masked_crops", "masks"):
        (staging_dir / name).mkdir()

    input_rows: list[dict[str, Any]] = []
    stoma_rows: list[dict[str, Any]] = []
    hair_cell_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    nms_rows: list[dict[str, Any]] = []

    try:
        for image_number, image_path in enumerate(images, start=1):
            print(f"[{image_number}/{len(images)}] {image_path.name}", flush=True)
            (
                input_row,
                image_stomata,
                image_hair_cells,
                image_masks,
                image_summary,
                image_nms,
            ) = process_one_image(
                image_path=image_path,
                detector=detector,
                classifier=classifier,
                segmentor=segmentor,
                classifier_device=classifier_device,
                args=args,
                staging_dir=staging_dir,
            )
            input_rows.append(input_row)
            stoma_rows.extend(image_stomata)
            hair_cell_rows.extend(image_hair_cells)
            mask_rows.extend(image_masks)
            image_rows.append(image_summary)
            nms_rows.extend(image_nms)

        write_csv(staging_dir / "input_image_manifest.csv", input_rows, INPUT_FIELDS)
        write_csv(
            staging_dir / "stoma_segmentation_manifest.csv",
            stoma_rows,
            STOMA_FIELDS,
        )
        write_csv(
            staging_dir / "hair_cell_detection_manifest.csv",
            hair_cell_rows,
            HAIR_CELL_FIELDS,
        )
        write_csv(staging_dir / "selected_mask_instances.csv", mask_rows, MASK_FIELDS)
        write_csv(
            staging_dir / "image_pipeline_summary.csv",
            image_rows,
            IMAGE_SUMMARY_FIELDS,
        )
        write_csv(
            staging_dir / "manual_nms_removed_duplicates.csv",
            nms_rows,
            NMS_FIELDS,
        )

        provenance = {
            "pipeline": "Brachypodium_OME_to_final_cell_segmentation",
            "scope": "inference_through_segmentation_without_trait_extraction",
            "started_from_original_ome_tiff": True,
            "completed_at_utc": utc_now(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "input_argument": str(input_path),
            "input_location_policy": (
                "explicit_user_supplied_path; hidden collection is not hard-coded"
            ),
            "output_directory": str(output_dir),
            "input_images": len(input_rows),
            "source_images_modified": False,
            "three_channel_conversion": "temporary_in_memory_only",
            "ordinary_unmasked_crops_saved": False,
            "trait_extraction_performed": False,
            "models_and_software": checkpoint_record,
            "classifier_checkpoint_metadata": classifier_metadata,
            "locked_settings": {
                "detector_image_size": DETECTOR_IMAGE_SIZE,
                "detector_joint_inference_confidence_floor": (
                    DETECTOR_INFERENCE_CONFIDENCE_FLOOR
                ),
                "detector_internal_nms_iou": DETECTOR_INTERNAL_NMS_IOU,
                "detector_stoma_confidence": STOMA_DETECTOR_CONFIDENCE,
                "detector_manual_stoma_nms_iou": (
                    STOMA_DETECTOR_MANUAL_NMS_IOU
                ),
                "detector_hair_cell_confidence": (
                    HAIR_CELL_DETECTOR_CONFIDENCE
                ),
                "detector_manual_hair_cell_nms_iou": (
                    HAIR_CELL_DETECTOR_MANUAL_NMS_IOU
                ),
                "detector_max_detections": DETECTOR_MAX_DETECTIONS,
                "crop_padding_pixels": CROP_PADDING_PIXELS,
                "classifier_image_size": CLASSIFIER_IMAGE_SIZE,
                "classifier_full_good_threshold": FULL_GOOD_THRESHOLD,
                "minimum_full_complex_candidates": MINIMUM_FULL_COMPLEX_CANDIDATES,
                "minimum_usable_complexes_after_segmentation": (
                    MINIMUM_USABLE_COMPLEXES_AFTER_SEGMENTATION
                ),
                "guard_only_route_enabled": True,
                "segmentor_image_size": SEGMENTOR_IMAGE_SIZE,
                "segmentor_base_confidence": SEGMENTOR_BASE_CONFIDENCE,
                "segmentor_acceptance_confidence": SEGMENTOR_ACCEPTANCE_CONFIDENCE,
                "segmentor_internal_nms_iou": SEGMENTOR_INTERNAL_NMS_IOU,
                "segmentor_max_detections": SEGMENTOR_MAX_DETECTIONS,
                "maximum_guard_pair_mask_iou": MAX_GUARD_PAIR_MASK_IOU,
                "minimum_guard_pair_bilateral_arrangement": (
                    MIN_GUARD_PAIR_BILATERAL_ARRANGEMENT
                ),
                "minimum_guard_only_subsidiary_pair_area_similarity": (
                    MIN_GUARD_ONLY_SUBSIDIARY_PAIR_AREA_SIMILARITY
                ),
                "pair_score_weights": PAIR_SCORE_WEIGHTS,
                "default_expected_image_width": EXPECTED_IMAGE_WIDTH,
                "default_expected_image_height": EXPECTED_IMAGE_HEIGHT,
            },
            "run_arguments": {
                "requested_device": args.device,
                "runtime_device": args.runtime_device,
                "cuda_device_index": args.cuda_device,
                "classifier_batch_size": args.classifier_batch_size,
                "recursive": args.recursive,
                "expected_image_width": args.expected_image_width,
                "expected_image_height": args.expected_image_height,
                "calibration_reference_pixels_per_um": args.calibration_reference,
                "calibration_absolute_tolerance_pixels_per_um": (
                    args.calibration_absolute_tolerance
                ),
                "calibration_policy": args.calibration_policy,
                "stop_on_calibration_problem": args.stop_on_calibration_problem,
                "overview_jpeg_quality": args.overview_jpeg_quality,
                "allow_ultralytics_version_mismatch": (
                    args.allow_ultralytics_version_mismatch
                ),
            },
            "calibration": {
                "verified_reference_pixels_per_um": args.calibration_reference,
                "micrometer_calibrated_at_microscope_installation": True,
                "calibration_provisional": False,
                "stage_micrometer_verified": STAGE_MICROMETER_VERIFIED,
                "calibrated_traits_permitted_for_images_passing_metadata_qc": True,
                "images_passing_calibration_qc": sum(
                    bool(row["calibrated_traits_permitted"]) for row in input_rows
                ),
                "all_images_pass_calibration_qc": all(
                    bool(row["calibrated_traits_permitted"]) for row in input_rows
                ),
                "images_matching_reference": sum(
                    bool(row.get("calibration_reference_match"))
                    for row in input_rows
                ),
                "images_using_nonreference_metadata": sum(
                    bool(row.get("calibrated_traits_permitted"))
                    and not bool(row.get("calibration_reference_match"))
                    for row in input_rows
                ),
                "policy": (
                    "Strict mode requires valid isotropic OME calibration that "
                    "matches the verified reference. The opt-in metadata mode "
                    "uses valid positive per-image X/Y metadata even when it "
                    "differs, and records a provisional warning."
                ),
            },
            "post_segmentation_top_up": {
                "target_preliminary_usable_complete_complexes_per_image": (
                    MINIMUM_USABLE_COMPLEXES_AFTER_SEGMENTATION
                ),
                "trigger": (
                    "fewer than three preliminary usable complete complexes "
                    "after all initially routed crops have been segmented"
                ),
                "candidate_order": (
                    "unused detector crops ranked by descending classifier "
                    "full_good probability, then deterministic stoma_id"
                ),
                "stop_rule": (
                    "stop after reaching the target or exhausting all retained "
                    "detector crops"
                ),
                "preliminary_usable_definition": (
                    "geometry-valid distinct guard pair, accepted subsidiary "
                    "pair, exactly two guard and two subsidiary instances, "
                    "guard-pair association supported by the target detector "
                    "box, and no selected mask touching the crop border"
                ),
                "top_up_is_flagged_per_stoma": True,
                "top_up_does_not_override_classifier_label": True,
            },
            "guard_pair_selection": {
                "hard_geometry_gate_precedes_target_ranking": True,
                "maximum_mask_iou": MAX_GUARD_PAIR_MASK_IOU,
                "minimum_bilateral_arrangement": (
                    MIN_GUARD_PAIR_BILATERAL_ARRANGEMENT
                ),
                "invalid_pair_action": (
                    "retain only the best single target-associated guard mask "
                    "and flag the crop as incomplete"
                ),
                "rationale": (
                    "guard cells are separate biological instances; target "
                    "proximity cannot override duplicate/same-side geometry"
                ),
            },
            "guard_only_subsidiary_selection": {
                "minimum_pair_area_similarity": (
                    MIN_GUARD_ONLY_SUBSIDIARY_PAIR_AREA_SIMILARITY
                ),
                "maximum_corresponding_area_ratio": (
                    1.0 / MIN_GUARD_ONLY_SUBSIDIARY_PAIR_AREA_SIMILARITY
                ),
                "failed_pair_action": (
                    "retain masks for audit, preserve eligible guard-cell "
                    "measurements, mark both subsidiary masks ineligible for "
                    "primary paired-subsidiary trait extraction, and prevent "
                    "the crop from counting as a fully usable complex"
                ),
                "classifier_scope": "guard_only only",
            },
            "target_association": {
                "ordinary_crop_rule": "locked_structured_conf_020_pair_score",
                "ambiguous_crop_rule": "target_aware_unvalidated",
                "activation": (
                    "neighboring retained box/centre or more than two eligible "
                    "guard candidates"
                ),
                "warning": (
                    "Target-aware production association is applied only after "
                    "hard guard-pair geometry filtering. It still requires "
                    "broader independent audit, so every activation is flagged."
                ),
            },
            "hair_cell_detection": {
                "single_joint_detector_inference": True,
                "class_id": HAIR_CELL_CLASS_ID,
                "confidence_threshold": HAIR_CELL_DETECTOR_CONFIDENCE,
                "manual_class_specific_nms_iou": (
                    HAIR_CELL_DETECTOR_MANUAL_NMS_IOU
                ),
                "validation_evidence": (
                    "detector/validation_inference/"
                    "hair_cell_threshold_nms_validation_v1"
                ),
                "independent_test_evidence": (
                    "detector/test_inference/"
                    "hair_cell_independent_test_conf034_nms070_v1"
                ),
                "AB_policy": (
                    "retain count as a candidate for calibrated primary "
                    "hair-cell density after trait extraction and biological "
                    "repeatability QC"
                ),
                "AD_policy": (
                    "retain visible count for audit/exploration but flag as "
                    "groove-limited and ineligible for primary hair-cell "
                    "density"
                ),
                "density_calculated_in_this_script": False,
                "hair_cells_enter_crop_classifier_or_segmentor": False,
            },
            "overview_colour_legend": {
                "yellow_H_prefixed_box": "retained_hair_cell_detection",
                "green": "preliminary_usable_complete",
                "purple": "raw_complete_2plus2_but_qc_flagged",
                "blue": "valid_guard_pair_with_incomplete_subsidiaries",
                "orange": "other_selected_partial_or_rejected_result",
                "grey": "detected_but_not_selected_for_segmentation",
            },
            "outputs": {
                "stoma_manifest": "stoma_segmentation_manifest.csv",
                "hair_cell_manifest": "hair_cell_detection_manifest.csv",
                "mask_instance_manifest": "selected_mask_instances.csv",
                "image_qc_summary": "image_pipeline_summary.csv",
                "input_manifest": "input_image_manifest.csv",
                "manual_nms_removed": "manual_nms_removed_duplicates.csv",
                "mask_archives": "masks/*.npz",
                "masked_crop_references": "masked_crops/*_masked.png",
                "indexed_overviews": "overviews/*_detections.jpg",
            },
        }
        write_json(staging_dir / "pipeline_provenance.json", provenance)
        (staging_dir / "RUN_REPORT.txt").write_text(
            make_run_report(image_rows, output_dir),
            encoding="utf-8",
        )
        os.rename(staging_dir, output_dir)
    except Exception:
        failure_path = staging_dir / "PIPELINE_FAILURE.txt"
        if staging_dir.exists() and not failure_path.exists():
            failure_path.write_text(
                "The pipeline stopped before completion. No source image was "
                "modified. The incomplete directory was retained.\n\n"
                + traceback.format_exc(),
                encoding="utf-8",
            )
        raise

    print("Pipeline completed successfully.")
    print(f"Output directory: {output_dir}")
    return output_dir


if __name__ == "__main__":
    run_pipeline()
