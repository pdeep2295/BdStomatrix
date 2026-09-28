"""
Extract calibrated Brachypodium stomatal traits from a completed inference run.

This is a separate, read-only consumer of the locked detector -> crop-quality
classifier -> cell-segmentation pipeline outputs. It does not load or rerun any
neural-network model, and it never edits the original OME-TIFF or inference
results.

The script produces three principal biological tables:

1. ``stoma_traits.csv``: one row per retained detector stoma.
2. ``image_traits.csv``: one row per source image.
3. ``image_gwas_traits.csv``: one compact row per source image containing only
   GWAS-relevant scalar or mean phenotypes plus essential sample-size and QC
   fields. Distributional SD, CV, SE, confidence interval, minimum, maximum and
   median columns remain in ``image_traits.csv`` and are omitted here.

It also produces audit-only cell-instance, row and band tables plus a spatial
overlay. These supplemental tables make it possible to verify complex masks and
the distinction between a single longitudinal stomatal row and a stomatal band
that may contain one, two, three, four, five or more rows.

Important biological definitions
--------------------------------
* A stomatal row/file is one longitudinal sequence of stomata.
* A stomatal band is a spatially inferred vein/groove-associated group of one
  or more parallel stomatal rows. Veins are not segmented in this workflow, so
  ``band`` is deliberately used instead of claiming a directly observed vein.
* Rows are expected to run vertically in these images. Vertical is verified for
  every image and a small estimated tilt is removed before spatial analysis.
* All retained detector stomata contribute to count, density and spatial
  traits. The crop-quality classifier never changes density.
* Guard-cell morphometry requires a QC-valid guard pair. Subsidiary-cell and
  complete-complex morphometry require a QC-valid subsidiary pair as well.
* Individual-cell solidity uses pixel-count area for both the mask and its
  filled convex hull. Stomatal-complex solidity is the component-union area
  divided by the convex-envelope area; it therefore measures how completely
  the four accepted cell masks fill the complex envelope.
* No pavement-cell dimensions are assumed. ICN, FSN, stomatal index and true
  pavement-cell counts are intentionally absent.

Spatial analysis
----------------
This version saves first-to-fifth nearest-neighbour coordinates and calculates
transparent, directly observable row-, band- and neighbour-based traits. The
same biological definitions and inference rules are applied to every image;
the detected positions of rows and bands are allowed to vary among images.

Example
-------
python "/Users/pawandeepsingh/Documents/Postdoc/YOLO_model/Final_model/extract_traits_from_pipeline_output.py" \
    --pipeline-output-dir "/full/path/to/completed_pipeline_output" \
    --output-dir "/full/path/to/trait_extraction_v1"

Safety
------
* ``--pipeline-output-dir`` must already contain completed pipeline manifests.
* ``--output-dir`` must not already exist.
* Files below the inference directory are opened read-only.
* A failed extraction retains ``<output>.incomplete`` for diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


SCRIPT_VERSION = "1.3.0"


@dataclass(frozen=True)
class TraitSettings:
    """Provisional spatial settings that must be audited before full use."""

    expected_vertical_axis_deg: float = 0.0
    maximum_expected_axis_deviation_deg: float = 35.0
    minimum_stomata_per_resolved_row: int = 1
    maximum_rows_per_band: int = 5
    minimum_band_gap_um: float = 25.0
    band_gap_detector_width_factor: float = 2.0
    minimum_row_separation_um: float = 4.0
    row_separation_detector_width_factor: float = 0.30
    row_model_minimum_delta_bic: float = 6.0
    row_model_minimum_rss_reduction: float = 0.20
    row_model_additional_row_minimum_delta_bic: float = 10.0
    row_model_residual_floor_width_factor: float = 0.05
    row_model_simple_fit_stop_width_factor: float = 0.05
    two_row_rescue_minimum_rss_reduction: float = 0.45
    two_row_rescue_minimum_separation_width_factor: float = 0.35
    two_row_rescue_minimum_points_per_row: int = 2
    long_gap_factor: float = 2.5
    nearest_neighbor_ranks: int = 5
    local_neighbor_radius_um: float = 100.0
    minimum_origins_for_probability_traits: int = 10


@dataclass
class AxisTransform:
    """Image-centred rotation from source X/Y into lateral/longitudinal axes."""

    angle_from_vertical_deg: float
    origin_x_um: float
    origin_y_um: float
    source: str
    confidence: str
    candidate_count: int
    angular_mad_deg: float | None
    warning: str

    @property
    def radians(self) -> float:
        return math.radians(self.angle_from_vertical_deg)

    def forward(self, x_um: float, y_um: float) -> tuple[float, float]:
        """Return lateral and longitudinal coordinates in micrometres."""

        x = x_um - self.origin_x_um
        y = y_um - self.origin_y_um
        sine = math.sin(self.radians)
        cosine = math.cos(self.radians)
        lateral = cosine * x - sine * y
        longitudinal = sine * x + cosine * y
        return lateral, longitudinal

    def inverse(self, lateral_um: float, longitudinal_um: float) -> tuple[float, float]:
        """Map rotated coordinates back to source X/Y micrometres."""

        sine = math.sin(self.radians)
        cosine = math.cos(self.radians)
        x = cosine * lateral_um + sine * longitudinal_um + self.origin_x_um
        y = -sine * lateral_um + cosine * longitudinal_um + self.origin_y_um
        return x, y


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or str(value).strip() == "":
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot interpret Boolean value: {value!r}")


def optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def optional_int(value: Any) -> int | None:
    number = optional_float(value)
    return None if number is None else int(round(number))


def safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def finite_values(values: Iterable[Any]) -> np.ndarray:
    cleaned: list[float] = []
    for value in values:
        number = optional_float(value)
        if number is not None:
            cleaned.append(number)
    return np.asarray(cleaned, dtype=float)


def sample_summary(values: Iterable[Any], prefix: str) -> dict[str, Any]:
    array = finite_values(values)
    result: dict[str, Any] = {f"{prefix}_n": int(array.size)}
    if array.size == 0:
        for suffix in (
            "mean",
            "median",
            "sd",
            "cv_percent",
            "se",
            "ci95_low",
            "ci95_high",
            "min",
            "max",
        ):
            result[f"{prefix}_{suffix}"] = None
        return result

    mean = float(np.mean(array))
    median = float(np.median(array))
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if array.size >= 2:
        sd = float(np.std(array, ddof=1))
        se = sd / math.sqrt(array.size)
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se
    else:
        sd = se = ci_low = ci_high = None
    result.update(
        {
            f"{prefix}_mean": mean,
            f"{prefix}_median": median,
            f"{prefix}_sd": sd,
            f"{prefix}_cv_percent": (
                100.0 * sd / abs(mean)
                if sd is not None and not math.isclose(mean, 0.0)
                else None
            ),
            f"{prefix}_se": se,
            f"{prefix}_ci95_low": ci_low,
            f"{prefix}_ci95_high": ci_high,
            f"{prefix}_min": minimum,
            f"{prefix}_max": maximum,
        }
    )
    return result


def relative_asymmetry(first: float | None, second: float | None) -> float | None:
    """Return 2*absolute difference/sum; zero means a perfectly equal pair."""

    if first is None or second is None or first + second <= 0:
        return None
    return 2.0 * abs(first - second) / (first + second)


def axial_angle_difference(first_deg: float | None, second_deg: float | None) -> float | None:
    """Smallest difference between unoriented axes, in degrees."""

    if first_deg is None or second_deg is None:
        return None
    difference = abs(first_deg - second_deg) % 180.0
    return min(difference, 180.0 - difference)


def normalize_axial_angle(angle_deg: float) -> float:
    """Normalize an unoriented line angle to [-90, 90)."""

    return ((angle_deg + 90.0) % 180.0) - 90.0


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(json_safe(value), sort_keys=True)
    return value


def ordered_fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return fields


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    empty_table_fields: Sequence[str] | None = None,
) -> None:
    fields = ordered_fieldnames(rows)
    if not fields and empty_table_fields:
        fields = list(empty_table_fields)
    if not fields:
        raise ValueError(f"Cannot write a table with no columns: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def group_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(grouped)


def require_fields(rows: Sequence[dict[str, Any]], fields: Sequence[str], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} contains no data rows.")
    missing = [field for field in fields if field not in rows[0]]
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")


def load_audited_row_count_overrides(
    path: Path | None,
    valid_image_names: set[str],
    maximum_rows_per_band: int,
) -> tuple[dict[tuple[str, str], int], dict[str, Any]]:
    """Load optional, explicit band-level row counts without changing source data.

    An override is appropriate only after visual biological audit shows that
    center coordinates alone cannot distinguish the rows reliably. The exact
    source-image name is required so an override can never leak across images.
    """

    if path is None:
        return {}, {
            "available": False,
            "path": None,
            "sha256": None,
            "override_count": 0,
        }

    resolved = path.expanduser().resolve()
    rows = read_csv(resolved)
    require_fields(
        rows,
        ("source_image_name", "band_id", "row_count"),
        str(resolved),
    )
    overrides: dict[tuple[str, str], int] = {}
    for line_number, row in enumerate(rows, start=2):
        image_name = str(row.get("source_image_name", "")).strip()
        band_id = str(row.get("band_id", "")).strip().upper()
        raw_row_count = optional_float(row.get("row_count"))
        row_count = (
            int(round(raw_row_count))
            if raw_row_count is not None
            and math.isclose(raw_row_count, round(raw_row_count))
            else None
        )
        if image_name not in valid_image_names:
            raise ValueError(
                f"Unknown source_image_name on line {line_number} of {resolved}: "
                f"{image_name!r}"
            )
        if not (
            len(band_id) == 4
            and band_id.startswith("B")
            and band_id[1:].isdigit()
        ):
            raise ValueError(
                f"Invalid band_id on line {line_number} of {resolved}: "
                f"{band_id!r}; expected B001-style identifiers."
            )
        if row_count is None or not 1 <= row_count <= maximum_rows_per_band:
            raise ValueError(
                f"Invalid row_count on line {line_number} of {resolved}: "
                f"{row.get('row_count')!r}; expected 1..{maximum_rows_per_band}."
            )
        key = (image_name, band_id)
        if key in overrides:
            raise ValueError(
                f"Duplicate audited row-count override for {image_name}, "
                f"{band_id} in {resolved}."
            )
        overrides[key] = row_count

    return overrides, {
        "available": True,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "override_count": len(overrides),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract calibrated stomatal, spatial row/band and mask traits "
            "from a completed final-pipeline output directory."
        )
    )
    parser.add_argument("--pipeline-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows-per-band", type=int, default=5)
    parser.add_argument("--min-stomata-per-row", type=int, default=1)
    parser.add_argument("--maximum-axis-deviation-deg", type=float, default=35.0)
    parser.add_argument("--minimum-band-gap-um", type=float, default=25.0)
    parser.add_argument("--band-gap-width-factor", type=float, default=2.0)
    parser.add_argument("--minimum-row-separation-um", type=float, default=4.0)
    parser.add_argument("--row-separation-width-factor", type=float, default=0.30)
    parser.add_argument("--row-model-min-delta-bic", type=float, default=6.0)
    parser.add_argument("--row-model-min-rss-reduction", type=float, default=0.20)
    parser.add_argument(
        "--row-model-additional-row-min-delta-bic", type=float, default=10.0
    )
    parser.add_argument(
        "--row-model-residual-floor-width-factor", type=float, default=0.05
    )
    parser.add_argument(
        "--row-model-simple-fit-stop-width-factor", type=float, default=0.05
    )
    parser.add_argument(
        "--two-row-rescue-min-rss-reduction", type=float, default=0.45
    )
    parser.add_argument(
        "--two-row-rescue-min-separation-width-factor", type=float, default=0.35
    )
    parser.add_argument(
        "--two-row-rescue-min-points-per-row", type=int, default=2
    )
    parser.add_argument("--long-gap-factor", type=float, default=2.5)
    parser.add_argument("--nearest-neighbor-k", type=int, default=5)
    parser.add_argument("--local-radius-um", type=float, default=100.0)
    parser.add_argument("--minimum-probability-origins", type=int, default=10)
    parser.add_argument(
        "--no-audit-overlays",
        action="store_true",
        help="Do not create row/band spatial audit PNGs.",
    )
    parser.add_argument(
        "--audited-row-count-overrides",
        type=Path,
        default=None,
        help=(
            "Optional CSV with source_image_name, band_id and row_count. "
            "Use only for biologically audited band-level corrections; all "
            "automatic candidate evidence remains recorded."
        ),
    )
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> TraitSettings:
    settings = TraitSettings(
        maximum_expected_axis_deviation_deg=args.maximum_axis_deviation_deg,
        minimum_stomata_per_resolved_row=args.min_stomata_per_row,
        maximum_rows_per_band=args.max_rows_per_band,
        minimum_band_gap_um=args.minimum_band_gap_um,
        band_gap_detector_width_factor=args.band_gap_width_factor,
        minimum_row_separation_um=args.minimum_row_separation_um,
        row_separation_detector_width_factor=args.row_separation_width_factor,
        row_model_minimum_delta_bic=args.row_model_min_delta_bic,
        row_model_minimum_rss_reduction=args.row_model_min_rss_reduction,
        row_model_additional_row_minimum_delta_bic=(
            args.row_model_additional_row_min_delta_bic
        ),
        row_model_residual_floor_width_factor=(
            args.row_model_residual_floor_width_factor
        ),
        row_model_simple_fit_stop_width_factor=(
            args.row_model_simple_fit_stop_width_factor
        ),
        two_row_rescue_minimum_rss_reduction=(
            args.two_row_rescue_min_rss_reduction
        ),
        two_row_rescue_minimum_separation_width_factor=(
            args.two_row_rescue_min_separation_width_factor
        ),
        two_row_rescue_minimum_points_per_row=(
            args.two_row_rescue_min_points_per_row
        ),
        long_gap_factor=args.long_gap_factor,
        nearest_neighbor_ranks=args.nearest_neighbor_k,
        local_neighbor_radius_um=args.local_radius_um,
        minimum_origins_for_probability_traits=args.minimum_probability_origins,
    )
    if settings.minimum_stomata_per_resolved_row < 1:
        raise ValueError("--min-stomata-per-row must be at least 1.")
    if settings.maximum_rows_per_band < 1:
        raise ValueError("--max-rows-per-band must be positive.")
    for name in (
        "minimum_band_gap_um",
        "band_gap_detector_width_factor",
        "minimum_row_separation_um",
        "row_separation_detector_width_factor",
        "row_model_minimum_delta_bic",
        "row_model_minimum_rss_reduction",
        "row_model_additional_row_minimum_delta_bic",
        "row_model_residual_floor_width_factor",
        "row_model_simple_fit_stop_width_factor",
        "two_row_rescue_minimum_rss_reduction",
        "two_row_rescue_minimum_separation_width_factor",
        "long_gap_factor",
        "local_neighbor_radius_um",
        "maximum_expected_axis_deviation_deg",
    ):
        if getattr(settings, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if settings.nearest_neighbor_ranks < 1:
        raise ValueError("--nearest-neighbor-k must be positive.")
    if settings.two_row_rescue_minimum_points_per_row < 2:
        raise ValueError("--two-row-rescue-min-points-per-row must be at least 2.")
    return settings


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    input_dir = args.pipeline_output_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".incomplete")
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError(
            "Trait output must be outside the inference directory so no "
            "pipeline result can be overwritten."
        )
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(
            f"Output or incomplete output already exists: {output_dir} / {staging_dir}"
        )
    for filename in (
        "input_image_manifest.csv",
        "stoma_segmentation_manifest.csv",
        "selected_mask_instances.csv",
        "image_pipeline_summary.csv",
        "pipeline_provenance.json",
    ):
        if not (input_dir / filename).is_file():
            raise FileNotFoundError(input_dir / filename)
    return input_dir, output_dir, staging_dir


# =============================================================================
# Mask morphometry
# =============================================================================


def mask_xy(mask: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(mask.astype(bool))
    return np.column_stack((x, y)).astype(float)


def pca_axes(points_um: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_um.shape[0] < 3:
        raise ValueError("Fewer than three mask pixels.")
    center = np.mean(points_um, axis=0)
    covariance = np.cov(points_um - center, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    if major[1] < 0:
        major = -major
    minor = np.array([-major[1], major[0]], dtype=float)
    return center, major, minor


def metric_contour_perimeter(mask: np.ndarray, px_per_um_x: float, px_per_um_y: float) -> float:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    perimeter = 0.0
    for contour in contours:
        points = contour[:, 0, :].astype(np.float32)
        points[:, 0] /= px_per_um_x
        points[:, 1] /= px_per_um_y
        perimeter += float(cv2.arcLength(points.reshape(-1, 1, 2), True))
    return perimeter


def metric_convex_hull_area(mask: np.ndarray, px_per_um_x: float, px_per_um_y: float) -> float:
    """Return filled convex-hull area using the mask's pixel-area convention.

    Both the numerator used for mask area and this hull area are based on
    counts of raster pixels. Mixing pixel-count mask area with the vector area
    of pixel-centre coordinates can yield the physically impossible result
    ``solidity > 1`` for small masks.
    """

    points = mask_xy(mask).astype(np.int32)
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    hull_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull[:, 0, :], 1)
    return float(np.count_nonzero(hull_mask) / (px_per_um_x * px_per_um_y))


def central_guard_region_length(
    points_um: np.ndarray,
    center: np.ndarray,
    major: np.ndarray,
    minor: np.ndarray,
) -> tuple[float | None, str, float | None]:
    """Estimate the narrow central shaft length of a dumbbell guard cell.

    This is a transparent geometry heuristic, not a learned boundary. It is
    returned only when both terminal regions are detectably wider than the
    central region.
    """

    longitudinal = (points_um - center) @ major
    lateral = (points_um - center) @ minor
    total_length = float(np.ptp(longitudinal))
    if len(points_um) < 30 or total_length <= 0:
        return None, "insufficient_mask_pixels", None

    bins = int(np.clip(round(math.sqrt(len(points_um)) * 1.5), 25, 160))
    edges = np.linspace(float(np.min(longitudinal)), float(np.max(longitudinal)), bins + 1)
    widths = np.full(bins, np.nan, dtype=float)
    for index in range(bins):
        if index == bins - 1:
            selected = (longitudinal >= edges[index]) & (longitudinal <= edges[index + 1])
        else:
            selected = (longitudinal >= edges[index]) & (longitudinal < edges[index + 1])
        if np.count_nonzero(selected) >= 2:
            widths[index] = float(np.ptp(lateral[selected]))

    valid = np.isfinite(widths)
    if np.count_nonzero(valid) < max(12, bins // 2):
        return None, "insufficient_cross_sections", None
    indices = np.arange(bins)
    widths = np.interp(indices, indices[valid], widths[valid])
    kernel = np.ones(5, dtype=float) / 5.0
    smooth = np.convolve(np.pad(widths, (2, 2), mode="edge"), kernel, mode="valid")

    core = smooth[int(0.30 * bins) : max(int(0.70 * bins), int(0.30 * bins) + 1)]
    terminals = np.concatenate(
        (smooth[: max(2, int(0.20 * bins))], smooth[min(bins - 2, int(0.80 * bins)) :])
    )
    core_width = float(np.median(core))
    terminal_width = float(np.median(terminals))
    terminal_to_core = safe_divide(terminal_width, core_width)
    if terminal_to_core is None or terminal_to_core < 1.10:
        return None, "no_reliable_wide_to_narrow_transition", terminal_to_core

    threshold = core_width + 0.35 * (terminal_width - core_width)
    middle = int(np.argmin(np.abs((edges[:-1] + edges[1:]) / 2.0)))
    left = middle
    right = middle
    while left > 0 and smooth[left - 1] <= threshold:
        left -= 1
    while right < bins - 1 and smooth[right + 1] <= threshold:
        right += 1
    length = float(edges[right + 1] - edges[left])
    if length <= 0 or length >= 0.95 * total_length:
        return None, "transition_did_not_bound_central_region", terminal_to_core
    return length, "passed", terminal_to_core


def measure_mask(
    mask: np.ndarray,
    px_per_um_x: float,
    px_per_um_y: float,
    calculate_guard_central_region: bool,
) -> dict[str, Any]:
    binary = mask.astype(bool)
    points_px = mask_xy(binary)
    if len(points_px) < 3:
        return {"measurement_status": "insufficient_mask_pixels"}
    points_um = points_px.copy()
    points_um[:, 0] /= px_per_um_x
    points_um[:, 1] /= px_per_um_y
    center, major, minor = pca_axes(points_um)
    major_projection = (points_um - center) @ major
    minor_projection = (points_um - center) @ minor
    length_um = float(np.ptp(major_projection))
    width_um = float(np.ptp(minor_projection))
    area_um2 = float(np.count_nonzero(binary) / (px_per_um_x * px_per_um_y))
    perimeter_um = metric_contour_perimeter(binary, px_per_um_x, px_per_um_y)
    hull_area_um2 = metric_convex_hull_area(binary, px_per_um_x, px_per_um_y)
    orientation_vertical = normalize_axial_angle(math.degrees(math.atan2(major[0], major[1])))
    result: dict[str, Any] = {
        "measurement_status": "passed",
        "area_um2": area_um2,
        "perimeter_um": perimeter_um,
        "length_um": length_um,
        "width_um": width_um,
        "aspect_ratio": safe_divide(length_um, width_um),
        "circularity": (
            4.0 * math.pi * area_um2 / (perimeter_um * perimeter_um)
            if perimeter_um > 0
            else None
        ),
        "solidity": safe_divide(area_um2, hull_area_um2),
        "centroid_crop_x_pixels": float(np.mean(points_px[:, 0])),
        "centroid_crop_y_pixels": float(np.mean(points_px[:, 1])),
        "orientation_from_vertical_deg": orientation_vertical,
    }
    if calculate_guard_central_region:
        central_length, status, ratio = central_guard_region_length(
            points_um, center, major, minor
        )
        result.update(
            {
                "central_narrow_length_um": central_length,
                "central_narrow_length_status": status,
                "terminal_to_central_width_ratio": ratio,
            }
        )
    return result


def convex_envelope_mask(masks: Sequence[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("No masks supplied for complex envelope.")
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise ValueError("Selected masks for a stoma do not share one crop shape.")
    union = np.logical_or.reduce([mask.astype(bool) for mask in masks])
    points = mask_xy(union).astype(np.int32)
    envelope = np.zeros(shape, dtype=np.uint8)
    if len(points) >= 3:
        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        cv2.fillConvexPoly(envelope, hull[:, 0, :], 1)
    return envelope.astype(bool)


def mask_eligibility(stoma: dict[str, Any], class_id: int, class_count: int) -> tuple[bool, str]:
    target_supported = stoma.get("target_association_status") == "supported_by_target_box"
    if class_id == 0:
        if class_count != 2:
            return False, "guard_instance_count_not_two"
        if not parse_bool(stoma.get("guard_pair_geometry_valid")):
            return False, "guard_pair_geometry_invalid"
        if not target_supported:
            return False, "target_association_not_supported"
        if parse_bool(stoma.get("selected_guard_mask_touches_border")):
            return False, "guard_mask_touches_crop_border"
        return True, "passed"

    if class_count != 2:
        return False, "subsidiary_instance_count_not_two"
    if not parse_bool(stoma.get("guard_pair_geometry_valid")):
        return False, "guard_pair_geometry_invalid"
    if not target_supported:
        return False, "target_association_not_supported"
    if parse_bool(stoma.get("selected_subsidiary_mask_touches_border")):
        return False, "subsidiary_mask_touches_crop_border"
    if "subsidiary_pair_qc_valid" in stoma and str(stoma.get("subsidiary_pair_qc_valid", "")).strip():
        if not parse_bool(stoma.get("subsidiary_pair_qc_valid")):
            return False, str(stoma.get("subsidiary_pair_qc_reason") or "subsidiary_pair_qc_invalid")
    elif not parse_bool(stoma.get("preliminary_usable_complete")):
        return False, "legacy_output_without_positive_complete_complex_qc"
    return True, "passed"


PAIR_METRICS = (
    "length_um",
    "width_um",
    "area_um2",
    "perimeter_um",
    "aspect_ratio",
    "circularity",
    "solidity",
    "central_narrow_length_um",
    "terminal_to_central_width_ratio",
)


def populate_pair_traits(
    target: dict[str, Any], prefix: str, instances: list[dict[str, Any]]
) -> None:
    ordered = sorted(instances, key=lambda row: row["centroid_source_x_pixels"])
    for number, instance in enumerate(ordered, start=1):
        for metric in PAIR_METRICS:
            if metric in instance:
                target[f"{prefix}_{number}_{metric}"] = instance.get(metric)
        target[f"{prefix}_{number}_orientation_from_vertical_deg"] = instance.get(
            "orientation_from_vertical_deg"
        )
        target[f"{prefix}_{number}_confidence"] = instance.get("confidence")

    for metric in PAIR_METRICS:
        values = [optional_float(instance.get(metric)) for instance in ordered]
        if all(value is not None for value in values) and values:
            target[f"{prefix}_mean_{metric}"] = float(np.mean(values))
            target[f"{prefix}_absolute_difference_{metric}"] = abs(values[0] - values[1])
            target[f"{prefix}_relative_asymmetry_{metric}"] = relative_asymmetry(
                values[0], values[1]
            )
        else:
            target[f"{prefix}_mean_{metric}"] = None
            target[f"{prefix}_absolute_difference_{metric}"] = None
            target[f"{prefix}_relative_asymmetry_{metric}"] = None
    # Additive pair totals are defined only for dimensional measurements.
    # Aspect ratio, circularity and solidity are intentionally not summed.
    for metric in ("length_um", "width_um", "perimeter_um"):
        values = [optional_float(instance.get(metric)) for instance in ordered]
        target[f"{prefix}_total_{metric}"] = (
            float(sum(values))
            if len(values) == 2 and all(value is not None for value in values)
            else None
        )
    target[f"{prefix}_total_area_um2"] = sum(
        optional_float(instance.get("area_um2")) or 0.0 for instance in ordered
    )
    target[f"{prefix}_angular_difference_deg"] = axial_angle_difference(
        optional_float(ordered[0].get("orientation_from_vertical_deg")),
        optional_float(ordered[1].get("orientation_from_vertical_deg")),
    )


def extract_image_morphometry(
    input_dir: Path,
    stoma_rows: list[dict[str, Any]],
    px_per_um_x: float,
    px_per_um_y: float,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    stoma_traits: dict[str, dict[str, Any]] = {}
    cell_rows: list[dict[str, Any]] = []
    for stoma in stoma_rows:
        stoma_id = stoma["stoma_id"]
        summary: dict[str, Any] = {
            "guard_morphometry_eligible": False,
            "subsidiary_morphometry_eligible": False,
            "complex_morphometry_eligible": False,
            "guard_morphometry_exclusion_reason": "no_saved_masks",
            "subsidiary_morphometry_exclusion_reason": "no_saved_masks",
            "complex_morphometry_exclusion_reason": "no_saved_masks",
        }
        archive_relative = str(stoma.get("mask_archive_path", "")).strip()
        if not archive_relative:
            stoma_traits[stoma_id] = summary
            continue
        archive_path = (input_dir / archive_relative).resolve()
        if input_dir not in archive_path.parents or not archive_path.is_file():
            raise FileNotFoundError(f"Invalid mask archive for {stoma_id}: {archive_path}")

        with np.load(archive_path, allow_pickle=False) as archive:
            masks = archive["masks"].astype(bool)
            class_ids = archive["class_ids"].astype(int)
            confidences = archive["confidences"].astype(float)
            crop_origin = archive["crop_origin_xy"].astype(float)
            source_shape = archive["source_shape_hw"].astype(int)
        if len(masks) != len(class_ids) or len(masks) != len(confidences):
            raise ValueError(f"Mask array length mismatch in {archive_path}")
        expected_shape = (
            optional_int(stoma.get("source_image_height")),
            optional_int(stoma.get("source_image_width")),
        )
        if all(value is not None for value in expected_shape) and tuple(source_shape) != expected_shape:
            raise ValueError(f"Source-shape mismatch for {stoma_id}")

        instances_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for stored_index, (mask, class_id, confidence) in enumerate(
            zip(masks, class_ids, confidences, strict=True)
        ):
            measured = measure_mask(
                mask,
                px_per_um_x,
                px_per_um_y,
                calculate_guard_central_region=int(class_id) == 0,
            )
            instance = {
                "source_image_name": stoma["source_image_name"],
                "surface": stoma["surface"],
                "stoma_id": stoma_id,
                "stoma_number": optional_int(stoma.get("stoma_number")),
                "stored_mask_index": stored_index,
                "class_id": int(class_id),
                "cell_class": "guard_cell" if int(class_id) == 0 else "subsidiary_cell",
                "confidence": float(confidence),
                "mask_archive_path": archive_relative,
                **measured,
            }
            if "centroid_crop_x_pixels" in measured:
                instance["centroid_source_x_pixels"] = (
                    measured["centroid_crop_x_pixels"] + float(crop_origin[0])
                )
                instance["centroid_source_y_pixels"] = (
                    measured["centroid_crop_y_pixels"] + float(crop_origin[1])
                )
            instances_by_class[int(class_id)].append(instance)

        guard_eligible, guard_reason = mask_eligibility(
            stoma, 0, len(instances_by_class.get(0, []))
        )
        subsidiary_eligible, subsidiary_reason = mask_eligibility(
            stoma, 1, len(instances_by_class.get(1, []))
        )
        for class_id, instances in instances_by_class.items():
            eligible = guard_eligible if class_id == 0 else subsidiary_eligible
            reason = guard_reason if class_id == 0 else subsidiary_reason
            for instance in instances:
                instance["primary_trait_eligible"] = eligible
                instance["primary_trait_exclusion_reason"] = "" if eligible else reason
                cell_rows.append(instance)

        summary.update(
            {
                "guard_morphometry_eligible": guard_eligible,
                "subsidiary_morphometry_eligible": subsidiary_eligible,
                "complex_morphometry_eligible": guard_eligible and subsidiary_eligible,
                "guard_morphometry_exclusion_reason": "" if guard_eligible else guard_reason,
                "subsidiary_morphometry_exclusion_reason": (
                    "" if subsidiary_eligible else subsidiary_reason
                ),
                "complex_morphometry_exclusion_reason": (
                    "" if guard_eligible and subsidiary_eligible else ";".join(
                        reason
                        for reason in (guard_reason, subsidiary_reason)
                        if reason != "passed"
                    )
                ),
            }
        )
        if guard_eligible:
            guards = instances_by_class[0]
            populate_pair_traits(summary, "guard_cell", guards)
            guard_masks = [masks[row["stored_mask_index"]] for row in guards]
            guard_union = np.logical_or.reduce(guard_masks)
            guard_pair_metric = measure_mask(
                guard_union, px_per_um_x, px_per_um_y, False
            )
            for key, value in guard_pair_metric.items():
                summary[f"guard_pair_{key}"] = value

        if subsidiary_eligible:
            subsidiaries = instances_by_class[1]
            populate_pair_traits(summary, "subsidiary_cell", subsidiaries)
            subsidiary_masks = [masks[row["stored_mask_index"]] for row in subsidiaries]
            subsidiary_union = np.logical_or.reduce(subsidiary_masks)
            subsidiary_pair_metric = measure_mask(
                subsidiary_union, px_per_um_x, px_per_um_y, False
            )
            for key, value in subsidiary_pair_metric.items():
                summary[f"subsidiary_pair_{key}"] = value

        if guard_eligible and subsidiary_eligible:
            selected_instances = instances_by_class[0] + instances_by_class[1]
            selected_masks = [masks[row["stored_mask_index"]] for row in selected_instances]
            union = np.logical_or.reduce(selected_masks)
            envelope = convex_envelope_mask(selected_masks)
            envelope_metrics = measure_mask(
                envelope, px_per_um_x, px_per_um_y, False
            )
            for key, value in envelope_metrics.items():
                summary[f"stomatal_complex_{key}"] = value
            summary["stomatal_complex_component_union_area_um2"] = float(
                np.count_nonzero(union) / (px_per_um_x * px_per_um_y)
            )
            component_fill_fraction = safe_divide(
                summary["stomatal_complex_component_union_area_um2"],
                optional_float(summary.get("stomatal_complex_area_um2")),
            )
            summary["stomatal_complex_component_fill_fraction"] = (
                component_fill_fraction
            )
            # The complex mask is deliberately a convex envelope. Its own
            # envelope solidity would therefore be identically one and would
            # contain no useful biological information. Define complex
            # solidity instead as the accepted component-union area divided by
            # the convex-envelope area. This is bounded by [0, 1] and records
            # how completely the four selected cell masks fill the envelope.
            summary["stomatal_complex_solidity"] = component_fill_fraction

        guard_area = optional_float(summary.get("guard_cell_total_area_um2"))
        subsidiary_area = optional_float(summary.get("subsidiary_cell_total_area_um2"))
        complex_area = optional_float(summary.get("stomatal_complex_area_um2"))
        summary["subsidiary_to_guard_area_ratio"] = safe_divide(
            subsidiary_area, guard_area
        )
        summary["guard_pair_fraction_complex_area"] = safe_divide(
            guard_area, complex_area
        )
        summary["subsidiary_pair_fraction_complex_area"] = safe_divide(
            subsidiary_area, complex_area
        )
        summary["subsidiary_to_guard_mean_length_ratio"] = safe_divide(
            optional_float(summary.get("subsidiary_cell_mean_length_um")),
            optional_float(summary.get("guard_cell_mean_length_um")),
        )
        summary["subsidiary_to_guard_mean_width_ratio"] = safe_divide(
            optional_float(summary.get("subsidiary_cell_mean_width_um")),
            optional_float(summary.get("guard_cell_mean_width_um")),
        )
        stoma_traits[stoma_id] = summary
    return stoma_traits, cell_rows


# =============================================================================
# Axis, row and band inference
# =============================================================================


def robust_median_and_mad(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return median, mad


def estimate_vertical_axis(
    stoma_rows: list[dict[str, Any]],
    morphometry: dict[str, dict[str, Any]],
    width_um: float,
    height_um: float,
    settings: TraitSettings,
) -> AxisTransform:
    mask_angles: list[float] = []
    for stoma in stoma_rows:
        traits = morphometry.get(stoma["stoma_id"], {})
        angle = optional_float(
            traits.get("stomatal_complex_orientation_from_vertical_deg")
        )
        if angle is None:
            angle = optional_float(traits.get("guard_pair_orientation_from_vertical_deg"))
        if angle is not None and abs(angle) <= 45.0:
            mask_angles.append(angle)

    source = "segmentation_mask_major_axes"
    angles = mask_angles
    if len(angles) < 3:
        source = "detector_center_local_vertical_neighbors"
        x = np.asarray([float(row["center_x_um"]) for row in stoma_rows])
        y = np.asarray([float(row["center_y_um"]) for row in stoma_rows])
        widths = finite_values(row.get("prediction_width_um") for row in stoma_rows)
        heights = finite_values(row.get("prediction_height_um") for row in stoma_rows)
        lateral_limit = 2.5 * (float(np.median(widths)) if widths.size else 20.0)
        longitudinal_minimum = 0.75 * (
            float(np.median(heights)) if heights.size else 20.0
        )
        angles = []
        for index in range(len(x)):
            dx = x - x[index]
            dy = y - y[index]
            candidates = np.where(
                (np.arange(len(x)) != index)
                & (np.abs(dx) <= lateral_limit)
                & (np.abs(dy) >= longitudinal_minimum)
            )[0]
            if not len(candidates):
                continue
            neighbor = candidates[int(np.argmin(np.abs(dy[candidates])))]
            angle = normalize_axial_angle(
                math.degrees(math.atan2(float(dx[neighbor]), float(dy[neighbor])))
            )
            if abs(angle) <= 35.0:
                angles.append(angle)

    if angles:
        angle, mad = robust_median_and_mad(angles)
        confidence = "high" if len(angles) >= 5 and mad <= 5.0 else "moderate"
    else:
        angle = 0.0
        mad = None
        confidence = "low"
        source = "declared_vertical_image_orientation"
    warning = ""
    if abs(angle) > settings.maximum_expected_axis_deviation_deg:
        warning = "estimated_axis_exceeds_expected_vertical_deviation"
        confidence = "low"
    return AxisTransform(
        angle_from_vertical_deg=angle,
        origin_x_um=width_um / 2.0,
        origin_y_um=height_um / 2.0,
        source=source,
        confidence=confidence,
        candidate_count=len(angles),
        angular_mad_deg=mad,
        warning=warning,
    )


def kmeans_1d(values: np.ndarray, k: int, maximum_iterations: int = 100) -> np.ndarray | None:
    if k < 1 or len(values) < k:
        return None
    quantiles = np.linspace(0.0, 1.0, k + 2)[1:-1]
    centers = np.quantile(values, quantiles) if k > 1 else np.array([np.mean(values)])
    labels = np.zeros(len(values), dtype=int)
    for _ in range(maximum_iterations):
        new_labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        if any(np.count_nonzero(new_labels == group) == 0 for group in range(k)):
            return None
        new_centers = np.array(
            [np.mean(values[new_labels == group]) for group in range(k)], dtype=float
        )
        if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers):
            labels = new_labels
            break
        labels = new_labels
        centers = new_centers
    return labels


def fit_parallel_row_model(
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    k: int,
    minimum_points: int,
    minimum_separation_um: float,
    fixed_slope: float | None = None,
    residual_floor_um: float = 0.0,
) -> dict[str, Any] | None:
    local_lateral = lateral[point_indices]
    local_longitudinal = longitudinal[point_indices]
    long_reference = float(np.median(local_longitudinal))
    z = local_longitudinal - long_reference
    clustering_values = (
        local_lateral - float(fixed_slope) * z
        if fixed_slope is not None
        else local_lateral
    )
    labels = kmeans_1d(clustering_values, k)
    if labels is None:
        return None
    previous: np.ndarray | None = None
    slope = float(fixed_slope) if fixed_slope is not None else 0.0
    intercepts = np.zeros(k, dtype=float)
    for _ in range(100):
        if any(np.count_nonzero(labels == group) < minimum_points for group in range(k)):
            return None
        if fixed_slope is None:
            design = np.zeros((len(point_indices), k + 1), dtype=float)
            design[:, 0] = z
            design[np.arange(len(point_indices)), labels + 1] = 1.0
            beta, *_ = np.linalg.lstsq(design, local_lateral, rcond=None)
            slope = float(beta[0])
            intercepts = beta[1:].astype(float)
        else:
            adjusted = local_lateral - slope * z
            intercepts = np.asarray(
                [float(np.mean(adjusted[labels == group])) for group in range(k)]
            )
        predictions = slope * z[:, None] + intercepts[None, :]
        new_labels = np.argmin(np.abs(local_lateral[:, None] - predictions), axis=1)
        if previous is not None and np.array_equal(new_labels, labels):
            break
        previous = labels.copy()
        labels = new_labels

    if any(np.count_nonzero(labels == group) < minimum_points for group in range(k)):
        return None
    order = np.argsort(intercepts)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.asarray([remap[int(label)] for label in labels], dtype=int)
    intercepts = intercepts[order]
    if k > 1 and np.any(np.diff(intercepts) < minimum_separation_um):
        return None
    fitted = slope * z + intercepts[labels]
    residuals = local_lateral - fitted
    rss = float(np.sum(residuals * residuals))
    n = len(point_indices)
    raw_variance = rss / max(n, 1)
    bic_variance = max(raw_variance, residual_floor_um**2, 1e-9)
    # Two effective parameters per row is a deliberately conservative penalty,
    # even though the fitted slope is shared.
    bic = float(n * math.log(bic_variance) + (2 * k) * math.log(max(n, 2)))
    return {
        "k": k,
        "point_indices": list(point_indices),
        "labels": labels,
        "long_reference_um": long_reference,
        "slope": slope,
        "slope_fixed_for_candidate": fixed_slope is not None,
        "intercepts_um": intercepts,
        "rss": rss,
        "rms_residual_um": math.sqrt(max(raw_variance, 0.0)),
        "bic_residual_floor_um": residual_floor_um,
        "bic": bic,
        "cluster_sizes": [int(np.count_nonzero(labels == group)) for group in range(k)],
    }


def select_row_model(
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettings,
    audited_row_count: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    minimum_separation = max(
        settings.minimum_row_separation_um,
        settings.row_separation_detector_width_factor * median_detector_width_um,
    )
    maximum_k = min(
        settings.maximum_rows_per_band,
        max(1, len(point_indices) // settings.minimum_stomata_per_resolved_row),
    )
    residual_floor_um = (
        settings.row_model_residual_floor_width_factor
        * median_detector_width_um
    )
    one_row = fit_parallel_row_model(
        point_indices,
        lateral,
        longitudinal,
        1,
        1,
        minimum_separation,
        fixed_slope=None,
        residual_floor_um=residual_floor_um,
    )
    if one_row is None:
        raise RuntimeError("Even the one-row model failed unexpectedly.")
    candidates: list[dict[str, Any]] = [one_row]
    for k in range(2, maximum_k + 1):
        candidate = fit_parallel_row_model(
            point_indices,
            lateral,
            longitudinal,
            k,
            settings.minimum_stomata_per_resolved_row if k > 1 else 1,
            minimum_separation,
            fixed_slope=float(one_row["slope"]),
            residual_floor_um=residual_floor_um,
        )
        if candidate is not None:
            candidates.append(candidate)

    chosen = one_row
    selection_reason = "one_row_model_retained"
    simple_fit_stop_tolerance_um = (
        settings.row_model_simple_fit_stop_width_factor
        * median_detector_width_um
    )
    if one_row["rms_residual_um"] <= simple_fit_stop_tolerance_um:
        selection_reason = "one_row_fit_within_tilt_corrected_tolerance"
    else:
        for candidate in candidates[1:]:
            # A missing intermediate model is a hard stop. In particular, a
            # one-row band can no longer jump directly to four singleton rows.
            if candidate["k"] != chosen["k"] + 1:
                selection_reason = "missing_intermediate_row_model"
                break
            delta_bic = chosen["bic"] - candidate["bic"]
            rss_reduction = safe_divide(
                chosen["rss"] - candidate["rss"], chosen["rss"]
            )
            delta_threshold = (
                settings.row_model_minimum_delta_bic
                if candidate["k"] == 2
                else settings.row_model_additional_row_minimum_delta_bic
            )
            ordinary_acceptance = bool(
                delta_bic >= delta_threshold
                and rss_reduction is not None
                and rss_reduction >= settings.row_model_minimum_rss_reduction
            )
            separation = (
                float(np.min(np.diff(candidate["intercepts_um"])))
                if candidate["k"] > 1
                else 0.0
            )
            two_row_rescue = bool(
                chosen["k"] == 1
                and candidate["k"] == 2
                and min(candidate["cluster_sizes"])
                >= settings.two_row_rescue_minimum_points_per_row
                and separation
                >= settings.two_row_rescue_minimum_separation_width_factor
                * median_detector_width_um
                and rss_reduction is not None
                and rss_reduction
                >= settings.two_row_rescue_minimum_rss_reduction
            )
            if ordinary_acceptance or two_row_rescue:
                chosen = candidate
                selection_reason = (
                    "two_row_geometric_rescue"
                    if two_row_rescue and not ordinary_acceptance
                    else "sequential_bic_and_rss_support"
                )
            else:
                selection_reason = "next_row_model_lacked_required_support"
                break

    automatic_k = int(chosen["k"])
    automatic_selection_reason = selection_reason
    automatic_model = chosen
    if automatic_k == 1:
        automatic_alternative = next(
            (item for item in candidates if item["k"] == 2), None
        )
        automatic_ambiguity_delta = (
            automatic_model["bic"] - automatic_alternative["bic"]
            if automatic_alternative is not None
            else None
        )
        automatic_one_vs_two_ambiguous = bool(
            automatic_ambiguity_delta is not None
            and 0 < automatic_ambiguity_delta
            < settings.row_model_minimum_delta_bic
        )
    else:
        automatic_simpler = next(
            (item for item in candidates if item["k"] == automatic_k - 1), None
        )
        automatic_delta = (
            automatic_simpler["bic"] - automatic_model["bic"]
            if automatic_simpler is not None
            else None
        )
        automatic_one_vs_two_ambiguous = bool(
            automatic_k == 2
            and automatic_delta is not None
            and automatic_delta < 10.0
        )
    chosen = dict(chosen)
    if audited_row_count is not None:
        audited_candidate = next(
            (item for item in candidates if item["k"] == audited_row_count),
            None,
        )
        if audited_candidate is None:
            raise ValueError(
                f"Audited row_count={audited_row_count} has no valid candidate "
                f"model for a band containing {len(point_indices)} stomata."
            )
        chosen = dict(audited_candidate)
        selection_reason = "audited_row_count_override"

    simpler = next((item for item in candidates if item["k"] == chosen["k"] - 1), None)
    chosen["delta_bic_vs_simpler"] = (
        simpler["bic"] - chosen["bic"] if simpler is not None else None
    )
    delta = optional_float(chosen.get("delta_bic_vs_simpler"))
    if chosen["k"] == 1:
        alternative = next((item for item in candidates if item["k"] == 2), None)
        ambiguity_delta = (
            chosen["bic"] - alternative["bic"] if alternative is not None else None
        )
        chosen["one_vs_two_ambiguous"] = bool(
            ambiguity_delta is not None
            and 0 < ambiguity_delta < settings.row_model_minimum_delta_bic
        )
        chosen["model_confidence"] = "low" if chosen["one_vs_two_ambiguous"] else "moderate"
    else:
        chosen["one_vs_two_ambiguous"] = bool(chosen["k"] == 2 and delta is not None and delta < 10.0)
        chosen["model_confidence"] = "high" if delta is not None and delta >= 10.0 else "moderate"
    # A single detected stoma is allowed to preserve sparse AD rows, but it is
    # not enough evidence to call that row high-confidence by itself.
    chosen["sparse_row_support"] = any(
        size < 3 for size in chosen["cluster_sizes"]
    )
    if chosen["sparse_row_support"]:
        chosen["model_confidence"] = "low"
    chosen["automatic_row_count"] = automatic_k
    chosen["automatic_selection_reason"] = automatic_selection_reason
    chosen["automatic_one_vs_two_ambiguous"] = (
        automatic_one_vs_two_ambiguous
    )
    chosen["audited_row_count_override"] = audited_row_count is not None
    chosen["row_count_selection_reason"] = selection_reason
    chosen["tilt_slope_source"] = "one_row_model_fixed_before_row_clustering"
    chosen["one_row_rms_residual_um"] = one_row["rms_residual_um"]
    chosen["simple_fit_stop_tolerance_um"] = simple_fit_stop_tolerance_um
    if audited_row_count is not None:
        chosen["one_vs_two_ambiguous"] = False
        chosen["model_confidence"] = "audited"
    return chosen, candidates


def infer_rows_and_bands(
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettings,
    audited_row_counts: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    audited_row_counts = audited_row_counts or {}
    lateral = np.asarray([row["lateral_coordinate_um"] for row in stoma_traits], dtype=float)
    longitudinal = np.asarray(
        [row["longitudinal_coordinate_um"] for row in stoma_traits], dtype=float
    )
    detector_widths = finite_values(row.get("prediction_width_um") for row in stoma_traits)
    median_width = float(np.median(detector_widths)) if detector_widths.size else 20.0
    band_gap_threshold = max(
        settings.minimum_band_gap_um,
        settings.band_gap_detector_width_factor * median_width,
    )

    ordered_points = sorted(range(len(stoma_traits)), key=lambda index: lateral[index])
    provisional_bands: list[list[int]] = []
    current: list[int] = []
    previous_lateral: float | None = None
    for index in ordered_points:
        if (
            previous_lateral is not None
            and lateral[index] - previous_lateral > band_gap_threshold
            and current
        ):
            provisional_bands.append(current)
            current = []
        current.append(index)
        previous_lateral = float(lateral[index])
    if current:
        provisional_bands.append(current)

    row_records: list[dict[str, Any]] = []
    band_records: list[dict[str, Any]] = []
    point_assignment: dict[int, dict[str, Any]] = {}
    global_row_number = 0
    model_audit: list[dict[str, Any]] = []
    observed_band_ids: set[str] = set()
    for band_number, point_indices in enumerate(provisional_bands, start=1):
        band_id = f"B{band_number:03d}"
        observed_band_ids.add(band_id)
        chosen, candidates = select_row_model(
            point_indices,
            lateral,
            longitudinal,
            median_width,
            settings,
            audited_row_count=audited_row_counts.get(band_id),
        )
        local_labels = {
            point_index: int(label)
            for point_index, label in zip(point_indices, chosen["labels"], strict=True)
        }
        band_row_ids: list[str] = []
        for local_row in range(chosen["k"]):
            global_row_number += 1
            row_id = f"R{global_row_number:03d}"
            band_row_ids.append(row_id)
            members = [index for index in point_indices if local_labels[index] == local_row]
            y_values = np.sort(longitudinal[members])
            gaps = np.diff(y_values)
            median_gap = float(np.median(gaps)) if len(gaps) else None
            long_gap_count = (
                int(np.count_nonzero(gaps > settings.long_gap_factor * median_gap))
                if median_gap is not None and median_gap > 0
                else 0
            )
            continuity = (
                1.0 - long_gap_count / len(gaps) if len(gaps) else None
            )
            row_record = {
                "row_id": row_id,
                "band_id": band_id,
                "row_position_within_band": local_row + 1,
                "rows_in_band": chosen["k"],
                "automatic_rows_in_band": chosen["automatic_row_count"],
                "audited_row_count_override": chosen["audited_row_count_override"],
                "row_count_selection_reason": chosen["row_count_selection_reason"],
                "automatic_row_count_selection_reason": chosen[
                    "automatic_selection_reason"
                ],
                "stoma_count": len(members),
                "longitudinal_min_um": float(np.min(y_values)),
                "longitudinal_max_um": float(np.max(y_values)),
                "observed_longitudinal_span_um": float(np.ptp(y_values)),
                "row_intercept_lateral_um": float(chosen["intercepts_um"][local_row]),
                "row_common_slope_lateral_per_longitudinal": chosen["slope"],
                "row_tilt_slope_source": chosen["tilt_slope_source"],
                "row_orientation_relative_leaf_axis_deg": math.degrees(
                    math.atan(chosen["slope"])
                ),
                "row_longitudinal_reference_um": chosen["long_reference_um"],
                "row_rms_lateral_residual_um": chosen["rms_residual_um"],
                "row_model_bic": chosen["bic"],
                "row_model_delta_bic_vs_simpler": chosen["delta_bic_vs_simpler"],
                "row_model_confidence": chosen["model_confidence"],
                "sparse_row_support": len(members) < 3,
                "ambiguous_one_vs_two_rows": chosen["one_vs_two_ambiguous"],
                "automatic_ambiguous_one_vs_two_rows": chosen[
                    "automatic_one_vs_two_ambiguous"
                ],
                "median_within_row_spacing_um": median_gap,
                "long_gap_count": long_gap_count,
                "row_continuity": continuity,
                "member_stoma_ids": [stoma_traits[index]["stoma_id"] for index in members],
                "_point_indices": members,
            }
            row_record.update(sample_summary(gaps, "within_row_spacing_um"))
            row_records.append(row_record)
            for index in members:
                point_assignment[index] = {
                    "row_id": row_id,
                    "band_id": band_id,
                    "row_position_within_band": local_row + 1,
                    "rows_in_band": chosen["k"],
                    "automatic_rows_in_band": chosen["automatic_row_count"],
                    "audited_row_count_override": chosen[
                        "audited_row_count_override"
                    ],
                    "row_count_selection_reason": chosen[
                        "row_count_selection_reason"
                    ],
                    "row_assignment_confidence": chosen["model_confidence"],
                    "sparse_row_support": len(members) < 3,
                    "band_assignment_confidence": (
                        "low" if chosen["one_vs_two_ambiguous"] else chosen["model_confidence"]
                    ),
                }

        sorted_intercepts = np.asarray(chosen["intercepts_um"], dtype=float)
        intercepts_at_image_center = sorted_intercepts + chosen["slope"] * (
            0.0 - chosen["long_reference_um"]
        )
        interrow_gaps = np.diff(sorted_intercepts)
        band_center = float(np.mean(intercepts_at_image_center))
        row_center_span = float(np.ptp(sorted_intercepts)) if len(sorted_intercepts) else 0.0
        band_width = row_center_span + median_width
        row_labels_in_longitudinal_order = [
            local_labels[index]
            for index in sorted(point_indices, key=lambda item: longitudinal[item])
        ]
        change_fraction = (
            sum(
                first != second
                for first, second in zip(
                    row_labels_in_longitudinal_order[:-1],
                    row_labels_in_longitudinal_order[1:],
                )
            )
            / (len(row_labels_in_longitudinal_order) - 1)
            if len(row_labels_in_longitudinal_order) >= 2 and chosen["k"] > 1
            else None
        )
        longest_alternating = None
        if chosen["k"] == 2 and row_labels_in_longitudinal_order:
            longest = current_length = 1
            for first, second in zip(
                row_labels_in_longitudinal_order[:-1],
                row_labels_in_longitudinal_order[1:],
            ):
                current_length = current_length + 1 if first != second else 1
                longest = max(longest, current_length)
            longest_alternating = longest

        phase_offsets: list[float] = []
        pooled_row_gaps: list[float] = []
        for local_row in range(chosen["k"]):
            y_row = np.sort(
                longitudinal[
                    [index for index in point_indices if local_labels[index] == local_row]
                ]
            )
            pooled_row_gaps.extend(np.diff(y_row).tolist())
        median_pooled_gap = float(np.median(pooled_row_gaps)) if pooled_row_gaps else None
        if chosen["k"] > 1 and median_pooled_gap and median_pooled_gap > 0:
            for local_row in range(chosen["k"] - 1):
                first_y = longitudinal[
                    [index for index in point_indices if local_labels[index] == local_row]
                ]
                second_y = longitudinal[
                    [index for index in point_indices if local_labels[index] == local_row + 1]
                ]
                offsets = []
                for value in first_y:
                    offsets.append(float(np.min(np.abs(second_y - value))))
                for value in second_y:
                    offsets.append(float(np.min(np.abs(first_y - value))))
                if offsets:
                    phase_offsets.append(float(np.median(offsets)) / median_pooled_gap)

        band_record = {
            "band_id": band_id,
            "band_order_left_to_right": band_number,
            "stoma_count": len(point_indices),
            "row_count": chosen["k"],
            "automatic_row_count": chosen["automatic_row_count"],
            "audited_row_count_override": chosen["audited_row_count_override"],
            "row_count_selection_reason": chosen["row_count_selection_reason"],
            "automatic_row_count_selection_reason": chosen[
                "automatic_selection_reason"
            ],
            "row_ids": band_row_ids,
            "band_center_lateral_um": band_center,
            "band_row_center_span_um": row_center_span,
            "band_width_um": band_width,
            "band_width_basis": "row_center_span_plus_median_detector_box_width",
            "within_band_interrow_spacing_mean_um": (
                float(np.mean(interrow_gaps)) if len(interrow_gaps) else None
            ),
            "within_band_interrow_spacing_median_um": (
                float(np.median(interrow_gaps)) if len(interrow_gaps) else None
            ),
            "successive_stoma_row_change_fraction": change_fraction,
            "two_row_alternation_index": change_fraction if chosen["k"] == 2 else None,
            "two_row_longest_alternating_sequence": longest_alternating,
            "adjacent_row_phase_offset_normalized_mean": (
                float(np.mean(phase_offsets)) if phase_offsets else None
            ),
            "row_model_confidence": chosen["model_confidence"],
            "sparse_row_support": chosen["sparse_row_support"],
            "ambiguous_one_vs_two_rows": chosen["one_vs_two_ambiguous"],
            "automatic_ambiguous_one_vs_two_rows": chosen[
                "automatic_one_vs_two_ambiguous"
            ],
            "row_model_delta_bic_vs_simpler": chosen["delta_bic_vs_simpler"],
            "one_row_rms_residual_um": chosen["one_row_rms_residual_um"],
            "simple_fit_stop_tolerance_um": chosen[
                "simple_fit_stop_tolerance_um"
            ],
            "row_tilt_slope_source": chosen["tilt_slope_source"],
            "member_stoma_ids": [stoma_traits[index]["stoma_id"] for index in point_indices],
            "_point_indices": list(point_indices),
        }
        band_record.update(sample_summary(interrow_gaps, "within_band_interrow_spacing_um"))
        band_records.append(band_record)
        for candidate in candidates:
            model_audit.append(
                {
                    "band_id": band_id,
                    "candidate_row_count": candidate["k"],
                    "bic": candidate["bic"],
                    "rss": candidate["rss"],
                    "rms_residual_um": candidate["rms_residual_um"],
                    "bic_residual_floor_um": candidate[
                        "bic_residual_floor_um"
                    ],
                    "slope_lateral_per_longitudinal": candidate["slope"],
                    "slope_fixed_for_candidate": candidate[
                        "slope_fixed_for_candidate"
                    ],
                    "cluster_sizes": candidate["cluster_sizes"],
                    "selected": candidate["k"] == chosen["k"],
                    "automatic_selected": (
                        candidate["k"] == chosen["automatic_row_count"]
                    ),
                }
            )

    unknown_override_bands = sorted(set(audited_row_counts) - observed_band_ids)
    if unknown_override_bands:
        raise ValueError(
            "Audited row-count override refers to band(s) not found in this "
            f"image: {unknown_override_bands}. Available bands: "
            f"{sorted(observed_band_ids)}"
        )

    row_lookup = {row["row_id"]: row for row in row_records}
    for band in band_records:
        for row_id in band["row_ids"]:
            row_lookup[row_id]["band_row_count"] = band["row_count"]
    audit = {
        "median_detector_box_width_um": median_width,
        "band_gap_threshold_um": band_gap_threshold,
        "row_minimum_separation_um": max(
            settings.minimum_row_separation_um,
            settings.row_separation_detector_width_factor * median_width,
        ),
        "candidate_models": model_audit,
        "audited_row_count_overrides": dict(audited_row_counts),
        "automatic_row_count_policy": (
            "fit one shared band slope first; detrend before clustering; "
            "accept only consecutive k-to-k+1 models; stop well-fit one-row "
            "bands; allow a supported two-row geometric rescue"
        ),
    }
    return row_records, band_records, point_assignment, audit


def row_lateral_at(row: dict[str, Any], longitudinal_um: float) -> float:
    return float(row["row_intercept_lateral_um"]) + float(
        row["row_common_slope_lateral_per_longitudinal"]
    ) * (longitudinal_um - float(row["row_longitudinal_reference_um"]))


# =============================================================================
# Per-stoma and image spatial traits
# =============================================================================


def add_row_neighbors(
    stoma_traits: list[dict[str, Any]],
    row_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
) -> None:
    row_lookup = {row["row_id"]: row for row in row_records}
    rows_ordered = sorted(row_records, key=lambda row: row_lateral_at(row, 0.0))
    row_order = {row["row_id"]: index for index, row in enumerate(rows_ordered)}

    for index, assignment in point_assignment.items():
        stoma_traits[index].update(assignment)

    for row in row_records:
        members = sorted(
            row["_point_indices"],
            key=lambda index: stoma_traits[index]["longitudinal_coordinate_um"],
        )
        for position, index in enumerate(members):
            current = stoma_traits[index]
            upper = stoma_traits[members[position - 1]] if position > 0 else None
            lower = stoma_traits[members[position + 1]] if position + 1 < len(members) else None
            upper_distance = (
                current["longitudinal_coordinate_um"] - upper["longitudinal_coordinate_um"]
                if upper is not None
                else None
            )
            lower_distance = (
                lower["longitudinal_coordinate_um"] - current["longitudinal_coordinate_um"]
                if lower is not None
                else None
            )
            current.update(
                {
                    "upper_same_row_stoma_id": upper["stoma_id"] if upper else "",
                    "upper_same_row_distance_um": upper_distance,
                    "lower_same_row_stoma_id": lower["stoma_id"] if lower else "",
                    "lower_same_row_distance_um": lower_distance,
                    "nearest_same_row_distance_um": (
                        min(value for value in (upper_distance, lower_distance) if value is not None)
                        if upper_distance is not None or lower_distance is not None
                        else None
                    ),
                    "mean_two_sided_row_spacing_um": (
                        (upper_distance + lower_distance) / 2.0
                        if upper_distance is not None and lower_distance is not None
                        else None
                    ),
                    "longitudinal_spacing_asymmetry": relative_asymmetry(
                        upper_distance, lower_distance
                    ),
                }
            )

    for index, current in enumerate(stoma_traits):
        row_id = current.get("row_id")
        if not row_id:
            continue
        position = row_order[row_id]
        longitudinal = current["longitudinal_coordinate_um"]
        left_row = rows_ordered[position - 1] if position > 0 else None
        right_row = rows_ordered[position + 1] if position + 1 < len(rows_ordered) else None
        focal_lateral = row_lateral_at(row_lookup[row_id], longitudinal)

        def lateral_distance(other: dict[str, Any] | None) -> float | None:
            if other is None:
                return None
            return abs(row_lateral_at(other, longitudinal) - focal_lateral)

        left_distance = lateral_distance(left_row)
        right_distance = lateral_distance(right_row)
        available = [value for value in (left_distance, right_distance) if value is not None]
        same_band_distances = [
            abs(row_lateral_at(row, longitudinal) - focal_lateral)
            for row in rows_ordered
            if row["row_id"] != row_id and row["band_id"] == current["band_id"]
        ]
        other_band_distances = [
            abs(row_lateral_at(row, longitudinal) - focal_lateral)
            for row in rows_ordered
            if row["band_id"] != current["band_id"]
        ]
        current.update(
            {
                "left_adjacent_row_id": left_row["row_id"] if left_row else "",
                "left_adjacent_row_distance_um": left_distance,
                "right_adjacent_row_id": right_row["row_id"] if right_row else "",
                "right_adjacent_row_distance_um": right_distance,
                "nearest_lateral_row_distance_um": min(available) if available else None,
                "other_side_lateral_row_distance_um": max(available) if len(available) == 2 else None,
                "lateral_row_spacing_asymmetry": relative_asymmetry(
                    left_distance, right_distance
                ),
                "nearest_other_row_same_band_distance_um": (
                    min(same_band_distances) if same_band_distances else None
                ),
                "nearest_other_band_row_distance_um": (
                    min(other_band_distances) if other_band_distances else None
                ),
            }
        )


def neighbor_relation(first: dict[str, Any], second: dict[str, Any]) -> str:
    if first.get("row_id") == second.get("row_id"):
        return "same_row"
    if first.get("band_id") == second.get("band_id"):
        return "same_band_other_row"
    first_band = optional_int(str(first.get("band_id", "")).lstrip("B"))
    second_band = optional_int(str(second.get("band_id", "")).lstrip("B"))
    if first_band is not None and second_band is not None and abs(first_band - second_band) == 1:
        return "adjacent_band"
    return "other_band"


def add_general_nearest_neighbors(
    stoma_traits: list[dict[str, Any]],
    hair_rows: list[dict[str, Any]] | None,
    width_um: float,
    height_um: float,
    settings: TraitSettings,
) -> dict[str, Any]:
    n = len(stoma_traits)
    coordinates = np.asarray(
        [
            [row["lateral_coordinate_um"], row["longitudinal_coordinate_um"]]
            for row in stoma_traits
        ],
        dtype=float,
    )
    if n:
        differences = coordinates[:, None, :] - coordinates[None, :, :]
        distances = np.sqrt(np.sum(differences * differences, axis=2))
        np.fill_diagonal(distances, np.inf)
    else:
        distances = np.empty((0, 0), dtype=float)

    relationship_counts: Counter[str] = Counter()
    relationship_offsets: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    eligible_origin_count = 0
    hair_coordinates = None
    hair_ids: list[str] = []
    if hair_rows is not None:
        hair_coordinates = np.asarray(
            [
                [float(row["prediction_center_x_pixels"]), float(row["prediction_center_y_pixels"])]
                for row in hair_rows
            ],
            dtype=float,
        )
        hair_ids = [
            str(row.get("hair_cell_id") or row.get("hair_cell_number") or "")
            for row in hair_rows
        ]

    for index, row in enumerate(stoma_traits):
        x_um = row["center_x_um"]
        y_um = row["center_y_um"]
        edge_distance = min(x_um, width_um - x_um, y_um, height_um - y_um)
        row["distance_to_image_edge_um"] = edge_distance
        if n >= 2:
            order = np.argsort(distances[index])[: min(settings.nearest_neighbor_ranks, n - 1)]
        else:
            order = np.asarray([], dtype=int)
        for rank in range(1, settings.nearest_neighbor_ranks + 1):
            if rank <= len(order):
                neighbor_index = int(order[rank - 1])
                neighbor = stoma_traits[neighbor_index]
                relation = neighbor_relation(row, neighbor)
                row[f"neighbor_{rank}_stoma_id"] = neighbor["stoma_id"]
                row[f"neighbor_{rank}_distance_um"] = float(distances[index, neighbor_index])
                row[f"neighbor_{rank}_lateral_offset_um"] = float(
                    neighbor["lateral_coordinate_um"] - row["lateral_coordinate_um"]
                )
                row[f"neighbor_{rank}_longitudinal_offset_um"] = float(
                    neighbor["longitudinal_coordinate_um"] - row["longitudinal_coordinate_um"]
                )
                row[f"neighbor_{rank}_relationship"] = relation
            else:
                row[f"neighbor_{rank}_stoma_id"] = ""
                row[f"neighbor_{rank}_distance_um"] = None
                row[f"neighbor_{rank}_lateral_offset_um"] = None
                row[f"neighbor_{rank}_longitudinal_offset_um"] = None
                row[f"neighbor_{rank}_relationship"] = ""

        nearest = optional_float(row.get("neighbor_1_distance_um"))
        kth = optional_float(
            row.get(f"neighbor_{min(settings.nearest_neighbor_ranks, max(n - 1, 1))}_distance_um")
        )
        row["nearest_neighbor_distance_um"] = nearest
        row["edge_safe_for_ranked_neighbors"] = bool(kth is not None and edge_distance >= kth)
        row["spatial_eligible"] = bool(row.get("row_id") and row["edge_safe_for_ranked_neighbors"])
        if row["spatial_eligible"]:
            eligible_origin_count += 1
            for rank in range(1, min(settings.nearest_neighbor_ranks, n - 1) + 1):
                relation = row[f"neighbor_{rank}_relationship"]
                relationship_counts[relation] += 1
                dx = abs(float(row[f"neighbor_{rank}_lateral_offset_um"]))
                dy = abs(float(row[f"neighbor_{rank}_longitudinal_offset_um"]))
                distance = float(row[f"neighbor_{rank}_distance_um"])
                relationship_offsets[relation].append((dx, dy, distance))

        if n:
            local_count = int(
                np.count_nonzero(distances[index] <= settings.local_neighbor_radius_um)
            )
        else:
            local_count = 0
        row["local_neighbor_radius_um"] = settings.local_neighbor_radius_um
        row["local_stoma_neighbor_count"] = local_count
        row["local_radius_fully_inside_image"] = edge_distance >= settings.local_neighbor_radius_um

        if hair_coordinates is None:
            row["nearest_hair_cell_distance_um"] = None
            row["nearest_hair_cell_id"] = ""
            row["hair_spatial_analysis_eligible"] = False
            row["hair_spatial_primary_eligible"] = False
        elif len(hair_coordinates) == 0:
            row["nearest_hair_cell_distance_um"] = None
            row["nearest_hair_cell_id"] = ""
            row["hair_spatial_analysis_eligible"] = True
            row["hair_spatial_primary_eligible"] = row.get("surface") == "AB"
        else:
            source_x = row["center_x_pixels"]
            source_y = row["center_y_pixels"]
            dx_um = (hair_coordinates[:, 0] - source_x) / row["pixels_per_um_x"]
            dy_um = (hair_coordinates[:, 1] - source_y) / row["pixels_per_um_y"]
            hair_distances = np.sqrt(dx_um * dx_um + dy_um * dy_um)
            nearest_hair_index = int(np.argmin(hair_distances))
            row["nearest_hair_cell_distance_um"] = float(
                hair_distances[nearest_hair_index]
            )
            row["nearest_hair_cell_id"] = hair_ids[nearest_hair_index]
            row["hair_spatial_analysis_eligible"] = True
            row["hair_spatial_primary_eligible"] = row.get("surface") == "AB"

    total_relationships = sum(relationship_counts.values())
    enough_origins = eligible_origin_count >= settings.minimum_origins_for_probability_traits
    result: dict[str, Any] = {
        "spatial_probability_origin_count": eligible_origin_count,
        "spatial_probability_neighbor_relationship_count": total_relationships,
        "empirical_neighbor_probabilities_calculated": enough_origins,
        "empirical_neighbor_probability_exclusion_reason": (
            "" if enough_origins else "insufficient_edge_safe_ranked_neighbor_origins"
        ),
    }
    for relation in ("same_row", "same_band_other_row", "adjacent_band", "other_band"):
        count = relationship_counts[relation]
        result[f"{relation}_neighbor_count"] = count
        result[f"{relation}_empirical_probability"] = (
            count / total_relationships if enough_origins and total_relationships else None
        )
        offsets = relationship_offsets[relation]
        result.update(sample_summary((item[0] for item in offsets), f"{relation}_absolute_lateral_offset_um"))
        result.update(sample_summary((item[1] for item in offsets), f"{relation}_absolute_longitudinal_offset_um"))
        result.update(sample_summary((item[2] for item in offsets), f"{relation}_neighbor_distance_um"))
    return result


def add_size_normalized_spacing(stoma_traits: list[dict[str, Any]]) -> None:
    for row in stoma_traits:
        complex_length = optional_float(row.get("stomatal_complex_length_um"))
        complex_width = optional_float(row.get("stomatal_complex_width_um"))
        row["relative_longitudinal_spacing"] = safe_divide(
            optional_float(row.get("mean_two_sided_row_spacing_um")), complex_length
        )
        row["relative_nearest_same_row_spacing"] = safe_divide(
            optional_float(row.get("nearest_same_row_distance_um")), complex_length
        )
        row["relative_lateral_row_spacing"] = safe_divide(
            optional_float(row.get("nearest_lateral_row_distance_um")), complex_width
        )
        row["relative_within_band_row_spacing"] = safe_divide(
            optional_float(row.get("nearest_other_row_same_band_distance_um")),
            complex_width,
        )
        row["relative_between_band_spacing"] = safe_divide(
            optional_float(row.get("nearest_other_band_row_distance_um")),
            complex_width,
        )
        row["relative_nearest_neighbor_distance"] = safe_divide(
            optional_float(row.get("nearest_neighbor_distance_um")),
            math.sqrt(complex_length * complex_width)
            if complex_length is not None and complex_width is not None
            else None,
        )


# =============================================================================
# Image summaries and audit overlays
# =============================================================================


SUMMARY_STOMA_TRAITS = (
    "nearest_neighbor_distance_um",
    "nearest_same_row_distance_um",
    "mean_two_sided_row_spacing_um",
    "nearest_lateral_row_distance_um",
    "nearest_other_row_same_band_distance_um",
    "nearest_other_band_row_distance_um",
    "nearest_hair_cell_distance_um",
    "relative_longitudinal_spacing",
    "relative_lateral_row_spacing",
    "stomatal_complex_length_um",
    "stomatal_complex_width_um",
    "stomatal_complex_area_um2",
    "stomatal_complex_perimeter_um",
    "stomatal_complex_aspect_ratio",
    "stomatal_complex_circularity",
    "stomatal_complex_solidity",
    "guard_cell_mean_length_um",
    "guard_cell_mean_width_um",
    "guard_cell_mean_area_um2",
    "guard_cell_total_length_um",
    "guard_cell_total_width_um",
    "guard_cell_total_area_um2",
    "guard_cell_total_perimeter_um",
    "guard_cell_mean_central_narrow_length_um",
    "guard_cell_relative_asymmetry_length_um",
    "guard_cell_relative_asymmetry_width_um",
    "guard_cell_relative_asymmetry_area_um2",
    "subsidiary_cell_mean_length_um",
    "subsidiary_cell_mean_width_um",
    "subsidiary_cell_mean_area_um2",
    "subsidiary_cell_total_length_um",
    "subsidiary_cell_total_width_um",
    "subsidiary_cell_total_area_um2",
    "subsidiary_cell_total_perimeter_um",
    "subsidiary_cell_relative_asymmetry_length_um",
    "subsidiary_cell_relative_asymmetry_width_um",
    "subsidiary_cell_relative_asymmetry_area_um2",
    "subsidiary_to_guard_area_ratio",
    "subsidiary_to_guard_mean_length_ratio",
    "subsidiary_to_guard_mean_width_ratio",
    "guard_pair_fraction_complex_area",
    "subsidiary_pair_fraction_complex_area",
)


# This compact export intentionally contains one estimate per image for each
# phenotype. Distributional statistics remain available in image_traits.csv,
# but only means/scalar estimates and essential audit fields enter this table.
GWAS_IMAGE_IDENTIFIER_FIELDS = (
    "source_image_name",
    "surface",
    "slide_id",
    "plant_replicate",
    "biological_unit",
)

GWAS_IMAGE_TRAIT_FIELDS = (
    # Counts and calibrated density
    "stoma_count",
    "stomatal_density_per_mm2",
    "hair_cell_count",
    "hair_cell_density_per_mm2",
    "hair_to_stoma_ratio",
    "nearest_hair_cell_distance_um_mean",
    # Row and band organisation
    "stomatal_row_count",
    "stomatal_band_count",
    "stomatal_row_density_per_mm",
    "stomatal_band_density_per_mm",
    "within_row_stomatal_frequency_per_mm",
    "rows_per_band_mean",
    "single_row_band_fraction",
    "two_row_band_fraction",
    "three_row_band_fraction",
    "four_plus_row_band_fraction",
    "multirow_band_fraction",
    "two_row_alternation_index_mean",
    "adjacent_row_phase_offset_normalized_mean",
    # Directly observed neighbour relationships
    "same_row_empirical_probability",
    "same_band_other_row_empirical_probability",
    "adjacent_band_empirical_probability",
    "other_band_empirical_probability",
    "within_row_spacing_um_mean",
    "within_band_interrow_spacing_um_mean",
    "between_band_spacing_um_mean",
    "band_width_um_mean",
    "nearest_neighbor_distance_um_mean",
    "nearest_same_row_distance_um_mean",
    "mean_two_sided_row_spacing_um_mean",
    "nearest_lateral_row_distance_um_mean",
    "nearest_other_row_same_band_distance_um_mean",
    "nearest_other_band_row_distance_um_mean",
    "stomata_per_row_mean",
    "stomata_per_band_mean",
    "row_continuity_mean",
    "row_orientation_relative_leaf_axis_deg_mean",
    "nearest_neighbor_regularity_index_csr",
    "longitudinal_to_lateral_spacing_ratio",
    "neighbor_relationship_entropy",
    "neighbor_probability_longitudinal_lateral_anisotropy",
    "relative_longitudinal_spacing_mean",
    "relative_lateral_row_spacing_mean",
    # Complete stomatal-complex morphology
    "stomatal_complex_length_um_mean",
    "stomatal_complex_width_um_mean",
    "stomatal_complex_area_um2_mean",
    "stomatal_complex_perimeter_um_mean",
    "stomatal_complex_aspect_ratio_mean",
    "stomatal_complex_circularity_mean",
    "stomatal_complex_solidity_mean",
    # Guard-cell morphology
    "guard_cell_mean_length_um_mean",
    "guard_cell_mean_width_um_mean",
    "guard_cell_mean_area_um2_mean",
    "guard_cell_total_length_um_mean",
    "guard_cell_total_width_um_mean",
    "guard_cell_total_area_um2_mean",
    "guard_cell_total_perimeter_um_mean",
    "guard_cell_mean_central_narrow_length_um_mean",
    "guard_cell_relative_asymmetry_length_um_mean",
    "guard_cell_relative_asymmetry_width_um_mean",
    "guard_cell_relative_asymmetry_area_um2_mean",
    # Subsidiary-cell morphology
    "subsidiary_cell_mean_length_um_mean",
    "subsidiary_cell_mean_width_um_mean",
    "subsidiary_cell_mean_area_um2_mean",
    "subsidiary_cell_total_length_um_mean",
    "subsidiary_cell_total_width_um_mean",
    "subsidiary_cell_total_area_um2_mean",
    "subsidiary_cell_total_perimeter_um_mean",
    "subsidiary_cell_relative_asymmetry_length_um_mean",
    "subsidiary_cell_relative_asymmetry_width_um_mean",
    "subsidiary_cell_relative_asymmetry_area_um2_mean",
    # Relative allocation and coverage
    "subsidiary_to_guard_area_ratio_mean",
    "subsidiary_to_guard_mean_length_ratio_mean",
    "subsidiary_to_guard_mean_width_ratio_mean",
    "guard_pair_fraction_complex_area_mean",
    "subsidiary_pair_fraction_complex_area_mean",
    "stomatal_coverage_fraction",
    "guard_cell_area_fraction_of_field",
    "subsidiary_cell_area_fraction_of_field",
)

GWAS_IMAGE_QC_FIELDS = (
    "field_area_mm2",
    "pixels_per_um_x",
    "pixels_per_um_y",
    "calibrated_traits_permitted",
    "calibration_status",
    "calibration_provisional",
    "stage_micrometer_verified",
    "hair_manifest_available",
    "hair_cell_density_primary_trait_eligible",
    "hair_cell_visibility_interpretation",
    "morphometry_usable_guard_count",
    "morphometry_usable_subsidiary_count",
    "morphometry_usable_complete_complex_count",
    "complete_complex_sampling_fraction",
    "guard_cell_mean_length_um_n",
    "guard_cell_mean_central_narrow_length_um_n",
    "subsidiary_cell_mean_length_um_n",
    "stomatal_complex_length_um_n",
    "within_row_spacing_um_n",
    "within_band_interrow_spacing_um_n",
    "between_band_spacing_um_n",
    "longitudinal_axis_angle_from_vertical_deg",
    "axis_deviation_from_vertical_deg",
    "axis_estimation_source",
    "axis_estimation_confidence",
    "axis_candidate_count",
    "axis_angular_mad_deg",
    "axis_warning",
    "ambiguous_band_count",
    "automatic_ambiguous_band_count",
    "audited_row_count_override_band_count",
    "automatic_vs_final_row_count_difference_band_count",
    "sparse_support_row_count",
    "sparse_support_band_count",
    "spatial_probability_origin_count",
    "spatial_probability_neighbor_relationship_count",
    "empirical_neighbor_probabilities_calculated",
    "empirical_neighbor_probability_exclusion_reason",
    "nearest_neighbor_regularity_edge_safe_origin_count",
    "traits_calculated",
)

def image_gwas_trait_summary(image_traits: dict[str, Any]) -> dict[str, Any]:
    """Return a compact image-level phenotype row for later GWAS aggregation.

    Visible hair-cell traits are retained for both AB and AD images. The
    eligibility flag and visibility interpretation must remain attached so AD
    values can be treated as groove-limited exploratory measurements during
    downstream analysis.
    """

    return {
        field: image_traits.get(field)
        for field in (
            *GWAS_IMAGE_IDENTIFIER_FIELDS,
            *GWAS_IMAGE_TRAIT_FIELDS,
            *GWAS_IMAGE_QC_FIELDS,
        )
    }


def image_trait_summary(
    image_metadata: dict[str, Any],
    stoma_traits: list[dict[str, Any]],
    hair_rows: list[dict[str, Any]] | None,
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    axis: AxisTransform,
    spatial_probability: dict[str, Any],
    row_band_audit: dict[str, Any],
    width_um: float,
    height_um: float,
) -> dict[str, Any]:
    area_mm2 = width_um * height_um / 1_000_000.0
    width_mm = width_um / 1000.0
    height_mm = height_um / 1000.0
    stoma_count = len(stoma_traits)
    hair_count = len(hair_rows) if hair_rows is not None else None
    hair_primary_eligible = bool(
        hair_rows is not None and image_metadata.get("surface") == "AB"
    )
    if hair_rows:
        manifest_eligibility = [
            parse_bool(row.get("hair_cell_density_primary_trait_eligible"))
            for row in hair_rows
            if str(row.get("hair_cell_density_primary_trait_eligible", "")).strip()
        ]
        if manifest_eligibility:
            hair_primary_eligible = all(manifest_eligibility)
    hair_visibility_interpretation = (
        "manifest_missing_hair_density_not_calculated"
        if hair_rows is None
        else (
            "AB_candidate_primary_after_repeatability_QC"
            if image_metadata.get("surface") == "AB"
            else "AD_visible_hair_only_groove_limited"
        )
    )
    row_count = len(row_records)
    band_count = len(band_records)
    rows_per_band = [record["row_count"] for record in band_records]
    within_row_gaps: list[float] = []
    for record in row_records:
        members = sorted(
            record["_point_indices"],
            key=lambda index: stoma_traits[index]["longitudinal_coordinate_um"],
        )
        within_row_gaps.extend(
            np.diff([stoma_traits[index]["longitudinal_coordinate_um"] for index in members]).tolist()
        )
    within_band_gaps = finite_values(
        record.get("within_band_interrow_spacing_mean_um") for record in band_records
    )
    bands_ordered = sorted(band_records, key=lambda row: row["band_center_lateral_um"])
    between_band_gaps: list[float] = []
    for first, second in zip(bands_ordered[:-1], bands_ordered[1:]):
        first_rows = [
            row for row in row_records if row["band_id"] == first["band_id"]
        ]
        second_rows = [
            row for row in row_records if row["band_id"] == second["band_id"]
        ]
        if first_rows and second_rows:
            first_right = max(row_lateral_at(row, 0.0) for row in first_rows)
            second_left = min(row_lateral_at(row, 0.0) for row in second_rows)
            between_band_gaps.append(float(second_left - first_right))

    row_density = safe_divide(float(row_count), width_mm)
    band_density = safe_divide(float(band_count), width_mm)
    within_row_frequency = safe_divide(
        float(stoma_count), row_count * height_mm if row_count else None
    )
    mean_rows_per_band = safe_divide(float(row_count), float(band_count))
    reconstructed_row_density = (
        row_density * within_row_frequency
        if row_density is not None and within_row_frequency is not None
        else None
    )
    reconstructed_band_density = (
        band_density * mean_rows_per_band * within_row_frequency
        if band_density is not None
        and mean_rows_per_band is not None
        and within_row_frequency is not None
        else None
    )

    result: dict[str, Any] = {
        "source_image_name": image_metadata["source_image_name"],
        "surface": image_metadata["surface"],
        "slide_id": image_metadata.get("slide_id", ""),
        "plant_replicate": image_metadata.get("plant_replicate", ""),
        "biological_unit": image_metadata.get("biological_unit", ""),
        "image_width_pixels": optional_int(image_metadata.get("image_width")),
        "image_height_pixels": optional_int(image_metadata.get("image_height")),
        "pixels_per_um_x": optional_float(image_metadata.get("metadata_pixels_per_um_x")),
        "pixels_per_um_y": optional_float(image_metadata.get("metadata_pixels_per_um_y")),
        "calibrated_traits_permitted": parse_bool(
            image_metadata.get("calibrated_traits_permitted")
        ),
        "calibration_status": image_metadata.get("calibration_status", ""),
        "calibration_provisional": parse_bool(
            image_metadata.get("calibration_provisional")
        ),
        "stage_micrometer_verified": parse_bool(
            image_metadata.get("stage_micrometer_verified")
        ),
        "field_width_um": width_um,
        "field_height_um": height_um,
        "field_area_mm2": area_mm2,
        "stoma_count": stoma_count,
        "stomatal_density_per_mm2": safe_divide(float(stoma_count), area_mm2),
        "hair_manifest_available": hair_rows is not None,
        "hair_cell_count": hair_count,
        "hair_cell_density_per_mm2": (
            safe_divide(float(hair_count), area_mm2) if hair_count is not None else None
        ),
        "hair_cell_density_primary_trait_eligible": hair_primary_eligible,
        "hair_cell_visibility_interpretation": hair_visibility_interpretation,
        "hair_to_stoma_ratio": (
            safe_divide(float(hair_count), float(stoma_count)) if hair_count is not None else None
        ),
        "morphometry_usable_guard_count": sum(
            parse_bool(row.get("guard_morphometry_eligible")) for row in stoma_traits
        ),
        "morphometry_usable_subsidiary_count": sum(
            parse_bool(row.get("subsidiary_morphometry_eligible")) for row in stoma_traits
        ),
        "morphometry_usable_complete_complex_count": sum(
            parse_bool(row.get("complex_morphometry_eligible")) for row in stoma_traits
        ),
        "complete_complex_sampling_fraction": safe_divide(
            sum(parse_bool(row.get("complex_morphometry_eligible")) for row in stoma_traits),
            stoma_count,
        ),
        "longitudinal_axis_angle_from_vertical_deg": axis.angle_from_vertical_deg,
        "axis_deviation_from_vertical_deg": abs(axis.angle_from_vertical_deg),
        "axis_estimation_source": axis.source,
        "axis_estimation_confidence": axis.confidence,
        "axis_candidate_count": axis.candidate_count,
        "axis_angular_mad_deg": axis.angular_mad_deg,
        "axis_warning": axis.warning,
        "stomatal_row_count": row_count,
        "stomatal_band_count": band_count,
        "stomatal_row_density_per_mm": row_density,
        "stomatal_band_density_per_mm": band_density,
        "within_row_stomatal_frequency_per_mm": within_row_frequency,
        "rows_per_band_mean": mean_rows_per_band,
        "rows_per_band_median": float(np.median(rows_per_band)) if rows_per_band else None,
        "rows_per_band_min": min(rows_per_band) if rows_per_band else None,
        "rows_per_band_max": max(rows_per_band) if rows_per_band else None,
        "single_row_band_fraction": safe_divide(sum(value == 1 for value in rows_per_band), band_count),
        "two_row_band_fraction": safe_divide(sum(value == 2 for value in rows_per_band), band_count),
        "three_row_band_fraction": safe_divide(sum(value == 3 for value in rows_per_band), band_count),
        "four_plus_row_band_fraction": safe_divide(sum(value >= 4 for value in rows_per_band), band_count),
        "multirow_band_fraction": safe_divide(sum(value > 1 for value in rows_per_band), band_count),
        "ambiguous_band_count": sum(
            parse_bool(record.get("ambiguous_one_vs_two_rows")) for record in band_records
        ),
        "automatic_ambiguous_band_count": sum(
            parse_bool(record.get("automatic_ambiguous_one_vs_two_rows"))
            for record in band_records
        ),
        "audited_row_count_override_band_count": sum(
            parse_bool(record.get("audited_row_count_override"))
            for record in band_records
        ),
        "automatic_vs_final_row_count_difference_band_count": sum(
            optional_int(record.get("automatic_row_count"))
            != optional_int(record.get("row_count"))
            for record in band_records
        ),
        "sparse_support_row_count": sum(
            parse_bool(record.get("sparse_row_support")) for record in row_records
        ),
        "sparse_support_band_count": sum(
            parse_bool(record.get("sparse_row_support")) for record in band_records
        ),
        "two_row_alternation_index_mean": (
            float(np.mean(finite_values(record.get("two_row_alternation_index") for record in band_records)))
            if finite_values(record.get("two_row_alternation_index") for record in band_records).size
            else None
        ),
        "adjacent_row_phase_offset_normalized_mean": (
            float(np.mean(finite_values(record.get("adjacent_row_phase_offset_normalized_mean") for record in band_records)))
            if finite_values(record.get("adjacent_row_phase_offset_normalized_mean") for record in band_records).size
            else None
        ),
        "density_reconstructed_from_row_components_per_mm2": reconstructed_row_density,
        "row_density_reconstruction_residual_per_mm2": (
            safe_divide(float(stoma_count), area_mm2) - reconstructed_row_density
            if reconstructed_row_density is not None
            else None
        ),
        "density_reconstructed_from_band_components_per_mm2": reconstructed_band_density,
        "band_density_reconstruction_residual_per_mm2": (
            safe_divide(float(stoma_count), area_mm2) - reconstructed_band_density
            if reconstructed_band_density is not None
            else None
        ),
        "band_gap_threshold_um": row_band_audit["band_gap_threshold_um"],
        "row_minimum_separation_um": row_band_audit["row_minimum_separation_um"],
        **spatial_probability,
    }
    result.update(sample_summary(within_row_gaps, "within_row_spacing_um"))
    result.update(sample_summary(rows_per_band, "rows_per_band"))
    result.update(sample_summary(within_band_gaps, "within_band_interrow_spacing_um"))
    result.update(sample_summary(between_band_gaps, "between_band_spacing_um"))
    result.update(sample_summary((record.get("band_width_um") for record in band_records), "band_width_um"))
    result.update(sample_summary((row.get("nearest_neighbor_distance_um") for row in stoma_traits), "nearest_neighbor_distance_um"))
    result.update(sample_summary((row.get("stoma_count") for row in row_records), "stomata_per_row"))
    result.update(sample_summary((band.get("stoma_count") for band in band_records), "stomata_per_band"))
    result.update(sample_summary((row.get("row_continuity") for row in row_records), "row_continuity"))
    result.update(
        sample_summary(
            (row.get("row_orientation_relative_leaf_axis_deg") for row in row_records),
            "row_orientation_relative_leaf_axis_deg",
        )
    )

    edge_safe_nn = finite_values(
        row.get("nearest_neighbor_distance_um")
        for row in stoma_traits
        if parse_bool(row.get("edge_safe_for_ranked_neighbors"))
    )
    observed_nn = float(np.mean(edge_safe_nn)) if edge_safe_nn.size else None
    density_per_um2 = safe_divide(float(stoma_count), width_um * height_um)
    expected_nn_csr = (
        0.5 / math.sqrt(density_per_um2)
        if density_per_um2 is not None and density_per_um2 > 0
        else None
    )
    result["nearest_neighbor_expected_under_csr_um"] = expected_nn_csr
    result["nearest_neighbor_regularity_index_csr"] = safe_divide(
        observed_nn, expected_nn_csr
    )
    result["nearest_neighbor_regularity_edge_safe_origin_count"] = int(
        edge_safe_nn.size
    )
    mean_longitudinal = optional_float(result.get("within_row_spacing_um_mean"))
    mean_lateral = optional_float(
        sample_summary(
            (row.get("nearest_lateral_row_distance_um") for row in stoma_traits),
            "temporary",
        ).get("temporary_mean")
    )
    result["longitudinal_to_lateral_spacing_ratio"] = safe_divide(
        mean_longitudinal, mean_lateral
    )
    empirical_probabilities = finite_values(
        result.get(key)
        for key in (
            "same_row_empirical_probability",
            "same_band_other_row_empirical_probability",
            "adjacent_band_empirical_probability",
            "other_band_empirical_probability",
        )
    )
    result["neighbor_relationship_entropy"] = (
        float(-np.sum(empirical_probabilities * np.log(empirical_probabilities)))
        if empirical_probabilities.size and np.all(empirical_probabilities > 0)
        else (
            float(-np.sum(empirical_probabilities[empirical_probabilities > 0] * np.log(empirical_probabilities[empirical_probabilities > 0])))
            if empirical_probabilities.size
            else None
        )
    )
    same_row_probability = optional_float(result.get("same_row_empirical_probability"))
    lateral_probability = sum(
        optional_float(result.get(key)) or 0.0
        for key in (
            "same_band_other_row_empirical_probability",
            "adjacent_band_empirical_probability",
            "other_band_empirical_probability",
        )
    )
    result["neighbor_probability_longitudinal_lateral_anisotropy"] = (
        safe_divide(
            same_row_probability - lateral_probability,
            same_row_probability + lateral_probability,
        )
        if same_row_probability is not None
        else None
    )

    for trait in SUMMARY_STOMA_TRAITS:
        result.update(sample_summary((row.get(trait) for row in stoma_traits), trait))

    density = optional_float(result.get("stomatal_density_per_mm2"))
    mean_complex_area = optional_float(result.get("stomatal_complex_area_um2_mean"))
    mean_guard_area = optional_float(result.get("guard_cell_total_area_um2_mean"))
    mean_subsidiary_area = optional_float(result.get("subsidiary_cell_total_area_um2_mean"))
    result["stomatal_coverage_fraction"] = (
        density * mean_complex_area / 1_000_000.0
        if density is not None and mean_complex_area is not None
        else None
    )
    result["guard_cell_area_fraction_of_field"] = (
        density * mean_guard_area / 1_000_000.0
        if density is not None and mean_guard_area is not None
        else None
    )
    result["subsidiary_cell_area_fraction_of_field"] = (
        density * mean_subsidiary_area / 1_000_000.0
        if density is not None and mean_subsidiary_area is not None
        else None
    )
    result["traits_calculated"] = True
    return result


BAND_COLORS_BGR = (
    (60, 180, 75),
    (255, 130, 40),
    (70, 70, 230),
    (190, 80, 180),
    (30, 200, 220),
    (210, 170, 50),
    (140, 90, 230),
    (80, 190, 190),
)


def make_spatial_audit_overlay(
    input_dir: Path,
    output_path: Path,
    image_metadata: dict[str, Any],
    stoma_traits: list[dict[str, Any]],
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    axis: AxisTransform,
    px_per_um_x: float,
    px_per_um_y: float,
) -> None:
    overview_relative = str(image_metadata.get("indexed_overview_path", "")).strip()
    overview_path = input_dir / overview_relative if overview_relative else None
    image = cv2.imread(str(overview_path), cv2.IMREAD_COLOR) if overview_path and overview_path.is_file() else None
    if image is None:
        height = optional_int(image_metadata.get("image_height"))
        width = optional_int(image_metadata.get("image_width"))
        if height is None or width is None:
            raise ValueError("Cannot create audit overlay without image dimensions.")
        image = np.full((height, width, 3), 235, dtype=np.uint8)

    band_color = {
        band["band_id"]: BAND_COLORS_BGR[(index - 1) % len(BAND_COLORS_BGR)]
        for index, band in enumerate(band_records, start=1)
    }
    corners_um = [
        axis.forward(x, y)
        for x, y in (
            (0.0, 0.0),
            (image.shape[1] / px_per_um_x, 0.0),
            (0.0, image.shape[0] / px_per_um_y),
            (image.shape[1] / px_per_um_x, image.shape[0] / px_per_um_y),
        )
    ]
    long_min = min(value[1] for value in corners_um)
    long_max = max(value[1] for value in corners_um)
    for row in row_records:
        long_values = np.linspace(long_min, long_max, 120)
        points: list[tuple[int, int]] = []
        for longitudinal in long_values:
            lateral = row_lateral_at(row, float(longitudinal))
            x_um, y_um = axis.inverse(lateral, float(longitudinal))
            x = int(round(x_um * px_per_um_x))
            y = int(round(y_um * px_per_um_y))
            if -50 <= x <= image.shape[1] + 50 and -50 <= y <= image.shape[0] + 50:
                points.append((x, y))
        if len(points) >= 2:
            cv2.polylines(
                image,
                [np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)],
                False,
                band_color[row["band_id"]],
                2,
                cv2.LINE_AA,
            )

    for row in stoma_traits:
        x = int(round(row["center_x_pixels"]))
        y = int(round(row["center_y_pixels"]))
        color = band_color.get(row.get("band_id"), (160, 160, 160))
        cv2.circle(image, (x, y), 5, color, -1, cv2.LINE_AA)
        label = f"{row.get('band_id', '?')}/{row.get('row_id', '?')}"
        cv2.putText(
            image,
            label,
            (x + 5, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (x + 5, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            color,
            1,
            cv2.LINE_AA,
        )

    origin = (80, image.shape[0] - 80)
    arrow_length = 120
    angle = axis.radians
    endpoint = (
        int(round(origin[0] - arrow_length * math.sin(angle))),
        int(round(origin[1] - arrow_length * math.cos(angle))),
    )
    cv2.arrowedLine(image, origin, endpoint, (255, 255, 255), 5, cv2.LINE_AA, tipLength=0.15)
    cv2.arrowedLine(image, origin, endpoint, (0, 0, 0), 2, cv2.LINE_AA, tipLength=0.15)
    cv2.putText(
        image,
        "longitudinal axis",
        (15, image.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    success = cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not success:
        raise RuntimeError(f"Failed to write audit overlay: {output_path}")


# =============================================================================
# Main image and run processing
# =============================================================================


def prepare_stoma_base_rows(
    stoma_source_rows: list[dict[str, Any]],
    image_metadata: dict[str, Any],
    px_per_um_x: float,
    px_per_um_y: float,
    morphometry: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    base: list[dict[str, Any]] = []
    for source in stoma_source_rows:
        x_pixels = float(source["prediction_center_x_pixels"])
        y_pixels = float(source["prediction_center_y_pixels"])
        row = {
            "source_image_name": source["source_image_name"],
            "surface": source["surface"],
            "slide_id": source.get("slide_id", image_metadata.get("slide_id", "")),
            "plant_replicate": source.get(
                "plant_replicate", image_metadata.get("plant_replicate", "")
            ),
            "biological_unit": source.get(
                "biological_unit", image_metadata.get("biological_unit", "")
            ),
            "stoma_id": source["stoma_id"],
            "stoma_number": optional_int(source.get("stoma_number")),
            "detector_confidence": optional_float(source.get("detector_confidence")),
            "center_x_pixels": x_pixels,
            "center_y_pixels": y_pixels,
            "center_x_um": x_pixels / px_per_um_x,
            "center_y_um": y_pixels / px_per_um_y,
            "pixels_per_um_x": px_per_um_x,
            "pixels_per_um_y": px_per_um_y,
            "prediction_x1_pixels": optional_float(source.get("prediction_x1_pixels")),
            "prediction_y1_pixels": optional_float(source.get("prediction_y1_pixels")),
            "prediction_x2_pixels": optional_float(source.get("prediction_x2_pixels")),
            "prediction_y2_pixels": optional_float(source.get("prediction_y2_pixels")),
            "prediction_width_um": safe_divide(
                optional_float(source.get("prediction_width_pixels"))
                or (
                    float(source["prediction_x2_pixels"])
                    - float(source["prediction_x1_pixels"])
                ),
                px_per_um_x,
            ),
            "prediction_height_um": safe_divide(
                optional_float(source.get("prediction_height_pixels"))
                or (
                    float(source["prediction_y2_pixels"])
                    - float(source["prediction_y1_pixels"])
                ),
                px_per_um_y,
            ),
            "density_eligible": True,
            "classifier_predicted_label": source.get("classifier_predicted_label", ""),
            "probability_full_good": optional_float(source.get("probability_full_good")),
            "probability_guard_only": optional_float(source.get("probability_guard_only")),
            "probability_bad": optional_float(source.get("probability_bad")),
            "segmentation_route": source.get("segmentation_route", ""),
            "post_segmentation_top_up": parse_bool(source.get("post_segmentation_top_up")),
            "segmentation_evidence_tier": source.get("segmentation_evidence_tier", ""),
            "edge_clipped_crop": parse_bool(source.get("edge_clipped")),
            "other_detection_center_in_crop": parse_bool(
                source.get("other_detection_center_in_crop")
            ),
            "mask_archive_path": source.get("mask_archive_path", ""),
        }
        row.update(morphometry.get(source["stoma_id"], {}))
        base.append(row)
    return base


def process_image(
    input_dir: Path,
    staging_dir: Path,
    image_metadata: dict[str, Any],
    stoma_source_rows: list[dict[str, Any]],
    hair_rows: list[dict[str, Any]] | None,
    settings: TraitSettings,
    audited_row_counts: dict[str, int],
    create_overlay: bool,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    px_per_um_x = optional_float(image_metadata.get("metadata_pixels_per_um_x"))
    px_per_um_y = optional_float(image_metadata.get("metadata_pixels_per_um_y"))
    width_pixels = optional_int(image_metadata.get("image_width"))
    height_pixels = optional_int(image_metadata.get("image_height"))
    if not parse_bool(image_metadata.get("calibrated_traits_permitted")):
        raise ValueError(
            f"Calibrated traits are not permitted for {image_metadata['source_image_name']}: "
            f"{image_metadata.get('calibration_status')}"
        )
    if px_per_um_x is None or px_per_um_y is None or width_pixels is None or height_pixels is None:
        raise ValueError(f"Missing calibration or dimensions for {image_metadata['source_image_name']}")
    width_um = width_pixels / px_per_um_x
    height_um = height_pixels / px_per_um_y

    morphometry, cell_rows = extract_image_morphometry(
        input_dir,
        stoma_source_rows,
        px_per_um_x,
        px_per_um_y,
    )
    stoma_traits = prepare_stoma_base_rows(
        stoma_source_rows,
        image_metadata,
        px_per_um_x,
        px_per_um_y,
        morphometry,
    )
    axis = estimate_vertical_axis(
        stoma_traits,
        morphometry,
        width_um,
        height_um,
        settings,
    )
    for row in stoma_traits:
        lateral, longitudinal = axis.forward(row["center_x_um"], row["center_y_um"])
        row["lateral_coordinate_um"] = lateral
        row["longitudinal_coordinate_um"] = longitudinal
        row["longitudinal_axis_angle_from_vertical_deg"] = axis.angle_from_vertical_deg
        orientation_keys = [
            key for key in row if key.endswith("_orientation_from_vertical_deg")
        ]
        for key in orientation_keys:
            orientation = optional_float(row.get(key))
            relative_key = key.replace(
                "_orientation_from_vertical_deg",
                "_orientation_relative_leaf_axis_deg",
            )
            row[relative_key] = (
                normalize_axial_angle(orientation - axis.angle_from_vertical_deg)
                if orientation is not None
                else None
            )
    for cell in cell_rows:
        orientation = optional_float(cell.get("orientation_from_vertical_deg"))
        cell["orientation_relative_leaf_axis_deg"] = (
            normalize_axial_angle(orientation - axis.angle_from_vertical_deg)
            if orientation is not None
            else None
        )

    row_records, band_records, assignments, row_band_audit = infer_rows_and_bands(
        stoma_traits,
        settings,
        audited_row_counts=audited_row_counts,
    )
    for row in row_records:
        row["source_image_name"] = image_metadata["source_image_name"]
        row["surface"] = image_metadata["surface"]
    for band in band_records:
        band["source_image_name"] = image_metadata["source_image_name"]
        band["surface"] = image_metadata["surface"]
    add_row_neighbors(stoma_traits, row_records, assignments)
    spatial_probability = add_general_nearest_neighbors(
        stoma_traits,
        hair_rows,
        width_um,
        height_um,
        settings,
    )
    add_size_normalized_spacing(stoma_traits)
    image_traits = image_trait_summary(
        image_metadata,
        stoma_traits,
        hair_rows,
        row_records,
        band_records,
        axis,
        spatial_probability,
        row_band_audit,
        width_um,
        height_um,
    )
    if create_overlay:
        safe_name = Path(str(image_metadata["source_image_name"])).name
        for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
            if safe_name.lower().endswith(suffix):
                safe_name = safe_name[: -len(suffix)]
                break
        overlay_relative = f"spatial_audit_overlays/{safe_name}_row_band_audit.png"
        make_spatial_audit_overlay(
            input_dir,
            staging_dir / overlay_relative,
            image_metadata,
            stoma_traits,
            row_records,
            band_records,
            axis,
            px_per_um_x,
            px_per_um_y,
        )
        image_traits["spatial_audit_overlay_path"] = overlay_relative
    else:
        image_traits["spatial_audit_overlay_path"] = ""

    for row in row_records:
        row.pop("_point_indices", None)
    for band in band_records:
        band.pop("_point_indices", None)
    return (
        stoma_traits,
        image_traits,
        cell_rows,
        row_records,
        band_records,
        row_band_audit,
    )


def run_trait_extraction() -> Path:
    args = parse_arguments()
    settings = build_settings(args)
    input_dir, output_dir, staging_dir = validate_paths(args)

    input_rows = read_csv(input_dir / "input_image_manifest.csv")
    stoma_rows = read_csv(input_dir / "stoma_segmentation_manifest.csv")
    image_pipeline_rows = read_csv(input_dir / "image_pipeline_summary.csv")
    selected_mask_rows = read_csv(input_dir / "selected_mask_instances.csv")
    hair_path = input_dir / "hair_cell_detection_manifest.csv"
    hair_rows = read_csv(hair_path) if hair_path.is_file() else None

    require_fields(
        input_rows,
        (
            "source_image_name",
            "source_image_path",
            "surface",
            "image_width",
            "image_height",
            "metadata_pixels_per_um_x",
            "metadata_pixels_per_um_y",
            "calibrated_traits_permitted",
        ),
        "input_image_manifest.csv",
    )
    require_fields(
        stoma_rows,
        (
            "source_image_name",
            "surface",
            "stoma_id",
            "stoma_number",
            "prediction_center_x_pixels",
            "prediction_center_y_pixels",
            "prediction_x1_pixels",
            "prediction_y1_pixels",
            "prediction_x2_pixels",
            "prediction_y2_pixels",
            "mask_archive_path",
        ),
        "stoma_segmentation_manifest.csv",
    )
    image_names = [row["source_image_name"] for row in input_rows]
    if len(set(image_names)) != len(image_names):
        raise ValueError("input_image_manifest.csv contains duplicate image names.")
    audited_overrides, audited_override_provenance = (
        load_audited_row_count_overrides(
            args.audited_row_count_overrides,
            set(image_names),
            settings.maximum_rows_per_band,
        )
    )
    overrides_by_image: dict[str, dict[str, int]] = defaultdict(dict)
    for (image_name, band_id), row_count in audited_overrides.items():
        overrides_by_image[image_name][band_id] = row_count

    stoma_by_image = group_by(stoma_rows, "source_image_name")
    hair_by_image = group_by(hair_rows, "source_image_name") if hair_rows is not None else {}
    pipeline_summary_by_image = {
        row["source_image_name"]: row for row in image_pipeline_rows
    }
    all_stoma_traits: list[dict[str, Any]] = []
    all_image_traits: list[dict[str, Any]] = []
    all_cell_traits: list[dict[str, Any]] = []
    all_row_traits: list[dict[str, Any]] = []
    all_band_traits: list[dict[str, Any]] = []
    row_band_audits: dict[str, Any] = {}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=False, exist_ok=False)
    if not args.no_audit_overlays:
        (staging_dir / "spatial_audit_overlays").mkdir()
    try:
        for image_number, metadata in enumerate(input_rows, start=1):
            image_name = metadata["source_image_name"]
            print(f"[{image_number}/{len(input_rows)}] {image_name}", flush=True)
            image_stomata = stoma_by_image.get(image_name, [])
            image_hairs = hair_by_image.get(image_name, []) if hair_rows is not None else None
            (
                stoma_traits,
                image_traits,
                cell_traits,
                row_traits,
                band_traits,
                audit,
            ) = process_image(
                input_dir,
                staging_dir,
                metadata,
                image_stomata,
                image_hairs,
                settings,
                overrides_by_image.get(image_name, {}),
                create_overlay=not args.no_audit_overlays,
            )
            pipeline_summary = pipeline_summary_by_image.get(image_name, {})
            image_traits["detector_predictions_after_manual_nms"] = optional_int(
                pipeline_summary.get("detector_predictions_after_manual_nms")
            )
            image_traits["final_preliminary_usable_complete_count"] = optional_int(
                pipeline_summary.get("final_preliminary_usable_complete_count")
            )
            image_traits["source_pipeline_indexed_overview_path"] = pipeline_summary.get(
                "indexed_overview_path", metadata.get("indexed_overview_path", "")
            )
            all_stoma_traits.extend(stoma_traits)
            all_image_traits.append(image_traits)
            all_cell_traits.extend(cell_traits)
            all_row_traits.extend(row_traits)
            all_band_traits.extend(band_traits)
            row_band_audits[image_name] = audit

        if len(all_cell_traits) != len(selected_mask_rows):
            raise ValueError(
                "The number of measured masks does not match "
                "selected_mask_instances.csv: "
                f"measured={len(all_cell_traits)}, "
                f"manifest={len(selected_mask_rows)}."
            )

        all_image_gwas_traits = [
            image_gwas_trait_summary(row) for row in all_image_traits
        ]

        write_csv(
            staging_dir / "stoma_traits.csv",
            all_stoma_traits,
            empty_table_fields=(
                "source_image_name",
                "surface",
                "stoma_id",
                "stoma_number",
                "density_eligible",
                "spatial_eligible",
                "guard_morphometry_eligible",
                "subsidiary_morphometry_eligible",
                "complex_morphometry_eligible",
            ),
        )
        write_csv(staging_dir / "image_traits.csv", all_image_traits)
        write_csv(
            staging_dir / "image_gwas_traits.csv",
            all_image_gwas_traits,
            empty_table_fields=(
                "source_image_name",
                "surface",
                "slide_id",
                "plant_replicate",
                "biological_unit",
                "stoma_count",
                "stomatal_density_per_mm2",
                "traits_calculated",
            ),
        )
        write_csv(
            staging_dir / "cell_instance_traits_qc.csv",
            all_cell_traits,
            empty_table_fields=(
                "source_image_name",
                "surface",
                "stoma_id",
                "stored_mask_index",
                "cell_class",
                "primary_trait_eligible",
                "primary_trait_exclusion_reason",
            ),
        )
        write_csv(
            staging_dir / "row_traits_qc.csv",
            all_row_traits,
            empty_table_fields=(
                "source_image_name",
                "surface",
                "row_id",
                "band_id",
                "stoma_count",
            ),
        )
        write_csv(
            staging_dir / "band_traits_qc.csv",
            all_band_traits,
            empty_table_fields=(
                "source_image_name",
                "surface",
                "band_id",
                "row_count",
                "stoma_count",
            ),
        )

        source_manifest_hashes = {
            filename: sha256_file(input_dir / filename)
            for filename in (
                "input_image_manifest.csv",
                "stoma_segmentation_manifest.csv",
                "selected_mask_instances.csv",
                "image_pipeline_summary.csv",
                "pipeline_provenance.json",
            )
        }
        if hair_path.is_file():
            source_manifest_hashes[hair_path.name] = sha256_file(hair_path)
        provenance = {
            "trait_extractor": "Brachypodium_final_trait_extraction",
            "script_version": SCRIPT_VERSION,
            "completed_at_utc": utc_now(),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "pipeline_output_directory": str(input_dir),
            "trait_output_directory": str(output_dir),
            "input_pipeline_files_modified": False,
            "original_ome_tiffs_modified": False,
            "models_rerun": False,
            "settings": asdict(settings),
            "audited_row_count_overrides": audited_override_provenance,
            "source_manifest_sha256": source_manifest_hashes,
            "image_count": len(all_image_traits),
            "stoma_count": len(all_stoma_traits),
            "cell_instance_count": len(all_cell_traits),
            "row_count": len(all_row_traits),
            "band_count": len(all_band_traits),
            "hair_manifest_available": hair_rows is not None,
            "surface_policy": (
                "same schema per image; surface retained as metadata; AB and AD "
                "must remain separate during plant/accession aggregation and GWAS"
            ),
            "density_policy": (
                "all retained post-NMS detector stomata contribute; classifier "
                "and segmentor never change density"
            ),
            "morphometry_policy": (
                "guard traits require a QC-valid guard pair; subsidiary and "
                "complex traits additionally require a QC-valid subsidiary pair"
            ),
            "shape_metric_policy": {
                "individual_cell_solidity": (
                    "mask pixel-count area divided by filled convex-hull "
                    "pixel-count area"
                ),
                "stomatal_complex_solidity": (
                    "accepted guard-plus-subsidiary component-union area divided "
                    "by convex-envelope area"
                ),
                "stomatal_complex_component_fill_fraction": (
                    "same defined quantity as stomatal_complex_solidity; retained "
                    "as an explicit audit field"
                ),
            },
            "pavement_cell_policy": (
                "no pavement-cell dimensions assumed; ICN, FSN, pavement density "
                "and stomatal index are not calculated"
            ),
            "empirical_neighbor_policy": {
                "ranked_neighbor_coordinates_saved": True,
                "empirical_same_row_same_band_adjacent_band_probabilities": True,
                "definition": (
                    "direct row_id and band_id relationships from the locked "
                    "genotype-independent row/band inference rules"
                ),
            },
            "gwas_image_table_policy": (
                "one row per source image; contains GWAS-relevant scalar and "
                "mean phenotypes plus essential sample-size and QC fields; "
                "distributional SD, CV, SE, CI, median, minimum and maximum "
                "statistics remain in image_traits.csv; visible hair-cell traits "
                "are retained for both surfaces together with their eligibility "
                "and groove-visibility QC fields"
            ),
            "row_band_model_audit": row_band_audits,
            "outputs": {
                "primary_stoma_table": "stoma_traits.csv",
                "primary_image_table": "image_traits.csv",
                "gwas_relevant_image_mean_table": "image_gwas_traits.csv",
                "cell_instance_audit": "cell_instance_traits_qc.csv",
                "row_audit": "row_traits_qc.csv",
                "band_audit": "band_traits_qc.csv",
                "spatial_overlays": (
                    "spatial_audit_overlays/*.png" if not args.no_audit_overlays else "disabled"
                ),
            },
        }
        (staging_dir / "trait_extraction_provenance.json").write_text(
            json.dumps(json_safe(provenance), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_lines = [
            "Brachypodium trait extraction completed",
            "",
            f"Completed: {utc_now()}",
            f"Input inference directory: {input_dir}",
            f"Output directory: {output_dir}",
            f"Images: {len(all_image_traits)}",
            f"Stomata: {len(all_stoma_traits)}",
            f"Rows inferred: {len(all_row_traits)}",
            f"Bands inferred: {len(all_band_traits)}",
            f"Audited band row-count overrides: {len(audited_overrides)}",
            f"Cell masks measured: {len(all_cell_traits)}",
            f"GWAS-ready image phenotype rows: {len(all_image_gwas_traits)}",
            "",
            "No model was rerun and no pipeline result or OME-TIFF was modified.",
            "Row/band settings are provisional until representative AB and AD overlays are audited.",
            "Only directly observable row-, band- and neighbour-based spatial traits were calculated.",
        ]
        (staging_dir / "TRAIT_EXTRACTION_REPORT.txt").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )
        os.rename(staging_dir, output_dir)
    except Exception:
        failure_path = staging_dir / "TRAIT_EXTRACTION_FAILURE.txt"
        if staging_dir.exists() and not failure_path.exists():
            failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise

    print("Trait extraction completed successfully.")
    print(f"Output directory: {output_dir}")
    return output_dir


if __name__ == "__main__":
    run_trait_extraction()
