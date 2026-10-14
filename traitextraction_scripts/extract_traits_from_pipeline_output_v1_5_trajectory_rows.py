#!/usr/bin/env python3
"""Experimental v1.5 trajectory-based Brachypodium spatial trait extractor.

This file is a separate inspection candidate. It imports the locked v1.3
extractor and the experimental v1.4.2 wrapper, then replaces only row-model
selection, biological-band demarcation, and their uncertainty reporting.
Detection, density, calibration, segmentation morphometry, source-image
preservation, output-directory safety, and ordinary nearest-neighbour traits
remain unchanged.

Biological model
----------------
Stomatal rows are treated as nearby, non-crossing trajectories rather than
fixed vertical strips. One small residual trajectory angle is estimated for
the whole image after the established leaf-axis rotation, then applied to every
row before broad neighbourhoods are formed. Lateral separation is measured
between fitted trajectories at matched longitudinal positions and normalized
by median detector-box width.

Rows are inferred before bands:

* <= 1.20 detector-box widths is treated as within-band row spacing;
* >= 1.50 detector-box widths is a strong candidate band boundary;
* 1.20--1.50 widths requires both a large local-gap ratio and supporting
  local raw-image edge evidence relative to parallel flanks;
* asymmetric and multiple partitions are permitted; and
* a final biological band is expected to contain one to five rows.

Sparse singleton-supported rows are never made high-confidence from centre
geometry alone. A special sparse three-row rescue additionally requires strong
local raw-image edge support. This generic edge signal is exploratory and is
not described as a validated vein or groove detector. Deterministic coordinate
perturbations quantify
row-count and assignment stability. Unstable row/band traits remain in the
detailed QC tables but are removed from the compact GWAS table; density,
morphometry and ordinary nearest-neighbour measurements remain available.

Do not integrate this experimental file into the production pipeline until a
representative, biologically audited AB+AD validation set supports its rules.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import extract_traits_from_pipeline_output as base
import extract_traits_from_pipeline_output_v1_4_global_rows as v14


SCRIPT_VERSION = "1.5.0-experimental-trajectory-rows-bands"


@dataclass(frozen=True)
class TraitSettingsV15(v14.TraitSettingsV14):
    """v1.4.2 settings plus trajectory, band-distance and stability controls."""

    # The initial groups are deliberately broader than biological bands.
    trajectory_supergroup_gap_width_factor: float = 3.00
    trajectory_max_rows_per_supergroup: int = 10

    # Every candidate row shares one bounded image-wide residual tilt after the
    # established leaf-axis rotation.
    trajectory_local_tilt_limit_deg: float = 6.0
    trajectory_local_tilt_step_deg: float = 0.25
    trajectory_minimum_separation_um: float = 4.0
    trajectory_minimum_separation_width_factor: float = 0.25
    trajectory_slope_penalty_weight: float = 0.02

    # Residual BIC remains primary. The singleton safeguard is retained, but
    # v1.4's uneven-spacing and large-gap penalties are not: either pattern can
    # be created by two real biological bands inside one broad neighbourhood.
    trajectory_simpler_tie_tolerance: float = 2.0

    # A sparse third trajectory needs centre geometry, detector confidence and
    # persistent raw-image support. This rescue remains explicitly low confidence.
    sparse_trajectory_rescue_min_bic_improvement: float = 6.0
    sparse_trajectory_rescue_min_rss_reduction: float = 0.60
    sparse_trajectory_rescue_max_spacing_cv: float = 0.25
    sparse_trajectory_rescue_min_detector_confidence: float = 0.75
    sparse_trajectory_rescue_min_image_support_ratio: float = 1.25

    # Adjacent fitted trajectories first define biological-band boundaries.
    band_same_max_gap_width_factor: float = 1.20
    band_strong_boundary_gap_width_factor: float = 1.50
    band_ambiguous_min_local_gap_ratio: float = 1.25
    band_ambiguous_min_image_support_ratio: float = 1.05
    band_boundary_min_longitudinal_overlap_fraction: float = 0.20

    # Raw edge evidence must be sampled outside the actual stomatal boxes.
    # Padding is proportional to each box dimension and is applied in source
    # pixels, so a stoma masks only its local two-dimensional neighbourhood.
    raw_evidence_stoma_exclusion_padding_fraction: float = 0.10

    # Deterministic stability assessment; it changes QC only, never density.
    trajectory_stability_iterations: int = 25
    trajectory_stability_jitter_width_fraction: float = 0.03
    trajectory_stability_min_row_count_fraction: float = 0.80
    trajectory_stability_min_assignment_fraction: float = 0.85
    band_boundary_stability_min_fraction: float = 0.80
    trajectory_stability_seed: int = 42


_ACTIVE_AXIS_V15: base.AxisTransform | None = None
_ACTIVE_STOMA_TRAITS_V15: list[dict[str, Any]] | None = None
_ACTIVE_GLOBAL_TRAJECTORY_SLOPE_V15 = 0.0
_ACTIVE_GLOBAL_TRAJECTORY_DETRENDING_V15 = False


def _v15_argument_parser() -> argparse.ArgumentParser:
    """Return a parser containing only v1.5-specific options."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--trajectory-supergroup-gap-width-factor", type=float, default=3.00
    )
    parser.add_argument("--trajectory-max-rows-per-supergroup", type=int, default=10)
    parser.add_argument("--trajectory-local-tilt-limit-deg", type=float, default=6.0)
    parser.add_argument("--trajectory-local-tilt-step-deg", type=float, default=0.25)
    parser.add_argument("--trajectory-minimum-separation-um", type=float, default=4.0)
    parser.add_argument(
        "--trajectory-minimum-separation-width-factor", type=float, default=0.25
    )
    parser.add_argument("--trajectory-slope-penalty-weight", type=float, default=0.02)
    parser.add_argument("--trajectory-simpler-tie-tolerance", type=float, default=2.0)
    parser.add_argument(
        "--sparse-trajectory-rescue-min-bic-improvement", type=float, default=6.0
    )
    parser.add_argument(
        "--sparse-trajectory-rescue-min-rss-reduction", type=float, default=0.60
    )
    parser.add_argument(
        "--sparse-trajectory-rescue-max-spacing-cv", type=float, default=0.25
    )
    parser.add_argument(
        "--sparse-trajectory-rescue-min-detector-confidence",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--sparse-trajectory-rescue-min-image-support-ratio",
        type=float,
        default=1.25,
    )
    parser.add_argument("--band-same-max-gap-width-factor", type=float, default=1.20)
    parser.add_argument(
        "--band-strong-boundary-gap-width-factor", type=float, default=1.50
    )
    parser.add_argument(
        "--band-ambiguous-min-local-gap-ratio", type=float, default=1.25
    )
    parser.add_argument(
        "--band-ambiguous-min-image-support-ratio", type=float, default=1.05
    )
    parser.add_argument(
        "--band-boundary-min-longitudinal-overlap-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--raw-evidence-stoma-exclusion-padding-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument("--trajectory-stability-iterations", type=int, default=25)
    parser.add_argument(
        "--trajectory-stability-jitter-width-fraction", type=float, default=0.03
    )
    parser.add_argument(
        "--trajectory-stability-min-row-count-fraction", type=float, default=0.80
    )
    parser.add_argument(
        "--trajectory-stability-min-assignment-fraction", type=float, default=0.85
    )
    parser.add_argument(
        "--band-boundary-stability-min-fraction", type=float, default=0.80
    )
    parser.add_argument("--trajectory-stability-seed", type=int, default=42)
    return parser


def parse_arguments_v15() -> argparse.Namespace:
    """Parse v1.5 options, then delegate all established options to v1.4.2."""

    if any(value in {"-h", "--help"} for value in sys.argv[1:]):
        print("v1.5 trajectory-specific options:")
        _v15_argument_parser().print_help()
        print("\nEstablished extractor options:")
    v15_args, remaining = _v15_argument_parser().parse_known_args()
    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0], *remaining]
        established = v14.parse_arguments_v14()
    finally:
        sys.argv = original_argv
    for name, value in vars(v15_args).items():
        setattr(established, name, value)
    return established


def build_settings_v15(args: argparse.Namespace) -> TraitSettingsV15:
    """Validate v1.4.2 settings and add the trajectory-specific controls."""

    old = v14.build_settings_v14(args)
    values = {
        name: getattr(args, name)
        for name in (
            "trajectory_supergroup_gap_width_factor",
            "trajectory_max_rows_per_supergroup",
            "trajectory_local_tilt_limit_deg",
            "trajectory_local_tilt_step_deg",
            "trajectory_minimum_separation_um",
            "trajectory_minimum_separation_width_factor",
            "trajectory_slope_penalty_weight",
            "trajectory_simpler_tie_tolerance",
            "sparse_trajectory_rescue_min_bic_improvement",
            "sparse_trajectory_rescue_min_rss_reduction",
            "sparse_trajectory_rescue_max_spacing_cv",
            "sparse_trajectory_rescue_min_detector_confidence",
            "sparse_trajectory_rescue_min_image_support_ratio",
            "band_same_max_gap_width_factor",
            "band_strong_boundary_gap_width_factor",
            "band_ambiguous_min_local_gap_ratio",
            "band_ambiguous_min_image_support_ratio",
            "band_boundary_min_longitudinal_overlap_fraction",
            "raw_evidence_stoma_exclusion_padding_fraction",
            "trajectory_stability_iterations",
            "trajectory_stability_jitter_width_fraction",
            "trajectory_stability_min_row_count_fraction",
            "trajectory_stability_min_assignment_fraction",
            "band_boundary_stability_min_fraction",
            "trajectory_stability_seed",
        )
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite; received {value!r}.")
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} cannot be negative.")
    if values["trajectory_max_rows_per_supergroup"] < old.maximum_rows_per_band:
        raise ValueError(
            "trajectory_max_rows_per_supergroup cannot be smaller than "
            "--max-rows-per-band."
        )
    if values["trajectory_local_tilt_step_deg"] <= 0:
        raise ValueError("trajectory_local_tilt_step_deg must be positive.")
    if values["trajectory_stability_iterations"] < 1:
        raise ValueError("trajectory_stability_iterations must be at least 1.")
    for name in (
        "sparse_trajectory_rescue_min_rss_reduction",
        "sparse_trajectory_rescue_max_spacing_cv",
        "band_boundary_min_longitudinal_overlap_fraction",
        "raw_evidence_stoma_exclusion_padding_fraction",
        "trajectory_stability_jitter_width_fraction",
        "trajectory_stability_min_row_count_fraction",
        "trajectory_stability_min_assignment_fraction",
        "band_boundary_stability_min_fraction",
    ):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must lie between 0 and 1.")
    if (
        values["band_strong_boundary_gap_width_factor"]
        <= values["band_same_max_gap_width_factor"]
    ):
        raise ValueError(
            "band_strong_boundary_gap_width_factor must exceed "
            "band_same_max_gap_width_factor."
        )
    if (
        values["trajectory_supergroup_gap_width_factor"]
        <= values["band_strong_boundary_gap_width_factor"]
    ):
        raise ValueError(
            "trajectory_supergroup_gap_width_factor must exceed the strong "
            "band-boundary factor so nearby biological bands can be considered "
            "together before partitioning."
        )

    inherited = asdict(old)
    # The original grouping now creates broad supergroups, not final bands.
    inherited["band_gap_detector_width_factor"] = max(
        float(inherited["band_gap_detector_width_factor"]),
        float(values["trajectory_supergroup_gap_width_factor"]),
    )
    return TraitSettingsV15(**inherited, **values)


def _estimate_image_wide_trajectory_slope(
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    settings: TraitSettingsV15,
) -> dict[str, Any]:
    """Estimate one residual row angle from repeated longitudinal alignments.

    For each allowed angle, stomatal centres are projected back to the image
    centre. Pairs that are longitudinally separated contribute most when their
    projected lateral positions coincide. Genuine repeated observations from
    the same row therefore vote for one shared angle across the image, whereas
    unrelated cross-row pairs contribute little through the narrow kernel.
    """

    if len(stoma_traits) < 3:
        return {
            "slope": 0.0,
            "angle_deg": 0.0,
            "evidence_available": False,
            "confidence": "insufficient_stomata_established_axis_retained",
            "eligible_pair_count": 0,
            "support_score": None,
            "support_gain_over_zero": None,
            "score_margin": None,
            "candidate_scores": [],
        }
    lateral = np.asarray(
        [float(row["lateral_coordinate_um"]) for row in stoma_traits],
        dtype=float,
    )
    longitudinal = np.asarray(
        [float(row["longitudinal_coordinate_um"]) for row in stoma_traits],
        dtype=float,
    )
    first, second = np.triu_indices(len(stoma_traits), k=1)
    longitudinal_difference = np.abs(
        longitudinal[first] - longitudinal[second]
    )
    minimum_longitudinal_difference = max(
        10.0, median_detector_width_um
    )
    eligible = longitudinal_difference >= minimum_longitudinal_difference
    first = first[eligible]
    second = second[eligible]
    longitudinal_difference = longitudinal_difference[eligible]
    if len(first) < 3:
        return {
            "slope": 0.0,
            "angle_deg": 0.0,
            "evidence_available": False,
            "confidence": "insufficient_longitudinal_pair_support_established_axis_retained",
            "eligible_pair_count": int(len(first)),
            "support_score": None,
            "support_gain_over_zero": None,
            "score_margin": None,
            "candidate_scores": [],
        }

    kernel_width_um = max(1.0, 0.12 * median_detector_width_um)
    pair_weights = np.minimum(
        1.0,
        longitudinal_difference / max(4.0 * median_detector_width_um, 1e-9),
    )
    candidate_scores: list[dict[str, float]] = []
    for angle_deg in np.arange(
        -settings.trajectory_local_tilt_limit_deg,
        settings.trajectory_local_tilt_limit_deg
        + 0.5 * settings.trajectory_local_tilt_step_deg,
        settings.trajectory_local_tilt_step_deg,
        dtype=float,
    ):
        slope = math.tan(math.radians(float(angle_deg)))
        projected = lateral - slope * longitudinal
        projected_difference = np.abs(projected[first] - projected[second])
        raw_support = float(
            np.sum(
                pair_weights
                * np.exp(
                    -0.5 * (projected_difference / kernel_width_um) ** 2
                )
            )
        )
        angle_fraction = abs(float(angle_deg)) / max(
            settings.trajectory_local_tilt_limit_deg, 1e-9
        )
        prior_penalty = (
            settings.trajectory_slope_penalty_weight
            * max(raw_support, 1.0)
            * angle_fraction**2
        )
        candidate_scores.append(
            {
                "angle_deg": float(angle_deg),
                "slope": slope,
                "raw_support": raw_support,
                "prior_penalty": prior_penalty,
                "score": raw_support - prior_penalty,
            }
        )
    best_score = max(item["score"] for item in candidate_scores)
    tie_tolerance = max(0.50, 0.01 * abs(best_score))
    tied = [
        item
        for item in candidate_scores
        if item["score"] >= best_score - tie_tolerance
    ]
    selected = min(
        tied,
        key=lambda item: (abs(item["angle_deg"]), -item["score"]),
    )
    alternatives = sorted(
        (
            item
            for item in candidate_scores
            if abs(item["angle_deg"] - selected["angle_deg"])
            >= settings.trajectory_local_tilt_step_deg
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    margin = (
        float(selected["score"] - alternatives[0]["score"])
        if alternatives
        else None
    )
    zero = min(candidate_scores, key=lambda item: abs(item["angle_deg"]))
    gain_over_zero = float(selected["raw_support"] - zero["raw_support"])
    proposed = selected
    if abs(proposed["angle_deg"]) < 1e-10:
        confidence = "established_axis_retained_within_score_tie"
    elif (
        gain_over_zero >= max(2.0, 0.05 * max(zero["raw_support"], 1.0))
        and margin is not None
        and margin > 0
    ):
        confidence = "supported_image_wide_residual_angle"
    else:
        confidence = "weak_residual_angle_support_established_axis_retained"
        selected = zero
    return {
        "slope": float(selected["slope"]),
        "angle_deg": float(selected["angle_deg"]),
        "proposed_angle_deg": float(proposed["angle_deg"]),
        "proposed_slope": float(proposed["slope"]),
        "evidence_available": True,
        "confidence": confidence,
        "eligible_pair_count": int(len(first)),
        "minimum_longitudinal_pair_difference_um": (
            minimum_longitudinal_difference
        ),
        "projection_kernel_width_um": kernel_width_um,
        "support_score": float(selected["raw_support"]),
        "proposed_support_score": float(proposed["raw_support"]),
        "support_gain_over_zero": float(
            selected["raw_support"] - zero["raw_support"]
        ),
        "proposed_support_gain_over_zero": gain_over_zero,
        "score_margin": margin,
        "candidate_scores": candidate_scores,
    }


def _supergroup_membership_sensitivity(
    detrended_lateral: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV15,
) -> dict[str, Any]:
    """Test whether broad-group breaks persist near their gap threshold."""

    threshold_um = max(
        settings.minimum_band_gap_um,
        settings.band_gap_detector_width_factor * median_detector_width_um,
    )
    if len(detrended_lateral) < 2:
        return {
            "stable": True,
            "minimum_classification_agreement_fraction": 1.0,
            "gap_threshold_um": threshold_um,
            "adjacent_gap_diagnostics": [],
        }
    ordered = np.sort(np.asarray(detrended_lateral, dtype=float))
    gaps = np.diff(ordered)
    rng = np.random.default_rng(settings.trajectory_stability_seed + 104729)
    diagnostics: list[dict[str, Any]] = []
    for gap_index, gap_um in enumerate(gaps, start=1):
        original_boundary = bool(gap_um > threshold_um)
        agreement = 0
        for _ in range(settings.trajectory_stability_iterations):
            perturbed_gap = max(
                0.0,
                float(gap_um)
                + float(
                    rng.normal(
                        0.0,
                        math.sqrt(2.0)
                        * settings.trajectory_stability_jitter_width_fraction
                        * median_detector_width_um,
                    )
                ),
            )
            agreement += int((perturbed_gap > threshold_um) == original_boundary)
        fraction = agreement / settings.trajectory_stability_iterations
        diagnostics.append(
            {
                "ordered_gap_index": gap_index,
                "gap_um": float(gap_um),
                "gap_width_factor": float(gap_um)
                / max(median_detector_width_um, 1e-9),
                "original_supergroup_boundary": original_boundary,
                "classification_agreement_fraction": fraction,
            }
        )
    minimum_fraction = min(
        item["classification_agreement_fraction"] for item in diagnostics
    )
    return {
        "stable": bool(
            minimum_fraction >= settings.trajectory_stability_min_row_count_fraction
        ),
        "minimum_classification_agreement_fraction": minimum_fraction,
        "gap_threshold_um": threshold_um,
        "adjacent_gap_diagnostics": diagnostics,
    }


def _slope_angles(settings: TraitSettingsV15) -> list[float]:
    """Return the residual grid, or zero after image-wide detrending."""

    if _ACTIVE_GLOBAL_TRAJECTORY_DETRENDING_V15:
        return [0.0]

    limit = float(settings.trajectory_local_tilt_limit_deg)
    step = float(settings.trajectory_local_tilt_step_deg)
    angles = np.arange(-limit, limit + step * 0.5, step, dtype=float).tolist()
    angles.append(0.0)
    return sorted({round(float(angle), 10) for angle in angles})


def _spatial_stoma_exclusion_boxes_pixels(
    stoma_traits: list[dict[str, Any]],
    point_indices: list[int],
    padding_fraction: float,
) -> list[tuple[float, float, float, float]]:
    """Return padded source-pixel boxes for local raw-evidence masking.

    The boxes remain two-dimensional. Consequently, a stoma masks a raw-image
    sample only when that sample is nearby in both lateral and longitudinal
    position; it does not remove the full longitudinal slice of the image.
    """

    boxes: list[tuple[float, float, float, float]] = []
    for point_index in point_indices:
        stoma = stoma_traits[point_index]
        x1 = base.optional_float(stoma.get("prediction_x1_pixels"))
        y1 = base.optional_float(stoma.get("prediction_y1_pixels"))
        x2 = base.optional_float(stoma.get("prediction_x2_pixels"))
        y2 = base.optional_float(stoma.get("prediction_y2_pixels"))
        if None in (x1, y1, x2, y2):
            center_x = base.optional_float(stoma.get("center_x_pixels"))
            center_y = base.optional_float(stoma.get("center_y_pixels"))
            width_um = base.optional_float(stoma.get("prediction_width_um"))
            height_um = base.optional_float(stoma.get("prediction_height_um"))
            px_x = base.optional_float(stoma.get("pixels_per_um_x"))
            px_y = base.optional_float(stoma.get("pixels_per_um_y"))
            if None in (center_x, center_y, width_um, height_um, px_x, px_y):
                continue
            half_width = 0.5 * float(width_um) * float(px_x)
            half_height = 0.5 * float(height_um) * float(px_y)
            x1 = float(center_x) - half_width
            x2 = float(center_x) + half_width
            y1 = float(center_y) - half_height
            y2 = float(center_y) + half_height
        left, right = sorted((float(x1), float(x2)))
        top, bottom = sorted((float(y1), float(y2)))
        pad_x = padding_fraction * max(right - left, 1.0)
        pad_y = padding_fraction * max(bottom - top, 1.0)
        boxes.append(
            (left - pad_x, top - pad_y, right + pad_x, bottom + pad_y)
        )
    return boxes


def _inside_any_exclusion_box(
    x_pixels: int,
    y_pixels: int,
    boxes: list[tuple[float, float, float, float]] | None,
) -> bool:
    """Return whether a source-pixel sample lies in a padded stomatal box."""

    return any(
        left <= x_pixels <= right and top <= y_pixels <= bottom
        for left, top, right, bottom in (boxes or [])
    )


def _trajectory_gradient_evidence(
    intercept_um: float,
    slope: float,
    longitudinal_reference_um: float,
    median_detector_width_um: float,
    longitudinal_min_um: float | None = None,
    longitudinal_max_um: float | None = None,
    exclusion_boxes_pixels: list[tuple[float, float, float, float]] | None = None,
) -> dict[str, Any]:
    """Return local target-line evidence relative to parallel flanks."""

    if (
        _ACTIVE_AXIS_V15 is None
        or v14._ACTIVE_HORIZONTAL_GRADIENT is None
        or v14._ACTIVE_IMAGE_METADATA is None
    ):
        return {
            "available": False,
            "unavailable_reason": "raw_source_image_or_axis_unavailable",
            "target_gradient_median": None,
            "flank_gradient_median": None,
            "target_to_flank_ratio": None,
            "sample_count": 0,
        }
    px_x = base.optional_float(
        v14._ACTIVE_IMAGE_METADATA.get("metadata_pixels_per_um_x")
    )
    px_y = base.optional_float(
        v14._ACTIVE_IMAGE_METADATA.get("metadata_pixels_per_um_y")
    )
    if px_x is None or px_y is None:
        return {
            "available": False,
            "unavailable_reason": "pixel_calibration_unavailable",
            "target_gradient_median": None,
            "flank_gradient_median": None,
            "target_to_flank_ratio": None,
            "sample_count": 0,
        }
    gradient = v14._ACTIVE_HORIZONTAL_GRADIENT
    height, width = gradient.shape[:2]
    corners = [
        _ACTIVE_AXIS_V15.forward(x, y)
        for x, y in (
            (0.0, 0.0),
            (width / px_x, 0.0),
            (0.0, height / px_y),
            (width / px_x, height / px_y),
        )
    ]
    image_long_min = min(value[1] for value in corners)
    image_long_max = max(value[1] for value in corners)
    requested_min = (
        image_long_min
        if longitudinal_min_um is None
        else float(longitudinal_min_um)
    )
    requested_max = (
        image_long_max
        if longitudinal_max_um is None
        else float(longitudinal_max_um)
    )
    if requested_max < requested_min:
        requested_min, requested_max = requested_max, requested_min
    minimum_span = 4.0 * median_detector_width_um
    if requested_max - requested_min < minimum_span:
        midpoint = 0.5 * (requested_min + requested_max)
        requested_min = midpoint - 0.5 * minimum_span
        requested_max = midpoint + 0.5 * minimum_span
    long_min = max(image_long_min, requested_min)
    long_max = min(image_long_max, requested_max)
    if long_max <= long_min:
        return {
            "available": False,
            "unavailable_reason": "requested_trajectory_region_outside_image",
            "target_gradient_median": None,
            "flank_gradient_median": None,
            "target_to_flank_ratio": None,
            "sample_count": 0,
        }

    target_offsets = np.linspace(
        -0.12 * median_detector_width_um,
        0.12 * median_detector_width_um,
        5,
    )
    flank_centers = (-0.45, 0.45)
    flank_offsets = np.linspace(
        -0.08 * median_detector_width_um,
        0.08 * median_detector_width_um,
        3,
    )
    target_samples: list[float] = []
    flank_samples: list[float] = []
    masked_sample_points = 0
    global_slope = (
        _ACTIVE_GLOBAL_TRAJECTORY_SLOPE_V15
        if _ACTIVE_GLOBAL_TRAJECTORY_DETRENDING_V15
        else 0.0
    )
    original_intercept_um = intercept_um + global_slope * longitudinal_reference_um
    original_slope = slope + global_slope
    for longitudinal_um in np.linspace(long_min, long_max, 160):
        lateral_um = original_intercept_um + original_slope * (
            float(longitudinal_um) - longitudinal_reference_um
        )
        target_local: list[float] = []
        for offset in target_offsets:
            x_um, y_um = _ACTIVE_AXIS_V15.inverse(
                lateral_um + float(offset), float(longitudinal_um)
            )
            x = int(round(x_um * px_x))
            y = int(round(y_um * px_y))
            if 0 <= x < width and 0 <= y < height:
                if _inside_any_exclusion_box(x, y, exclusion_boxes_pixels):
                    masked_sample_points += 1
                    continue
                target_local.append(float(gradient[y, x]))
        left_flank_local: list[float] = []
        right_flank_local: list[float] = []
        for flank_center in flank_centers:
            flank_local = (
                left_flank_local if flank_center < 0 else right_flank_local
            )
            for offset in flank_offsets:
                flank_lateral = lateral_um + (
                    flank_center * median_detector_width_um + float(offset)
                )
                x_um, y_um = _ACTIVE_AXIS_V15.inverse(
                    flank_lateral, float(longitudinal_um)
                )
                x = int(round(x_um * px_x))
                y = int(round(y_um * px_y))
                if 0 <= x < width and 0 <= y < height:
                    if _inside_any_exclusion_box(
                        x, y, exclusion_boxes_pixels
                    ):
                        masked_sample_points += 1
                        continue
                    flank_local.append(float(gradient[y, x]))
        # Require evidence on both sides of the proposed trajectory so a box
        # masking only one flank cannot create an asymmetric comparison.
        if target_local and left_flank_local and right_flank_local:
            target_samples.append(max(target_local))
            flank_samples.append(
                0.5
                * (
                    float(np.median(left_flank_local))
                    + float(np.median(right_flank_local))
                )
            )
    if len(target_samples) < 12 or len(flank_samples) < 12:
        return {
            "available": False,
            "unavailable_reason": (
                "insufficient_local_spatially_unmasked_line_samples"
            ),
            "target_gradient_median": None,
            "flank_gradient_median": None,
            "target_to_flank_ratio": None,
            "sample_count": min(len(target_samples), len(flank_samples)),
            "masked_sample_point_count": masked_sample_points,
        }
    target = float(np.median(target_samples))
    flank = float(np.median(flank_samples))
    return {
        "available": True,
        "unavailable_reason": "",
        "target_gradient_median": target,
        "flank_gradient_median": flank,
        "target_to_flank_ratio": target / max(flank, 1e-9),
        "sample_count": len(target_samples),
        "masked_sample_point_count": masked_sample_points,
        "spatial_exclusion_box_count": len(exclusion_boxes_pixels or []),
        "longitudinal_min_um": long_min,
        "longitudinal_max_um": long_max,
        "target_offset_half_width_factor": 0.12,
        "flank_center_width_factors": list(flank_centers),
    }


def _trajectory_gradient_score(
    intercept_um: float,
    slope: float,
    longitudinal_reference_um: float,
    median_detector_width_um: float,
    longitudinal_min_um: float | None = None,
    longitudinal_max_um: float | None = None,
) -> float | None:
    """Compatibility helper returning only the local target score."""

    evidence = _trajectory_gradient_evidence(
        intercept_um,
        slope,
        longitudinal_reference_um,
        median_detector_width_um,
        longitudinal_min_um=longitudinal_min_um,
        longitudinal_max_um=longitudinal_max_um,
    )
    return evidence["target_gradient_median"]


def _score_trajectory_candidate(
    candidate: dict[str, Any],
    median_detector_width_um: float,
    slope_angle_deg: float,
    settings: TraitSettingsV15,
) -> dict[str, Any]:
    """Score a row model without treating a potential band gap as an error."""

    diagnostic = v14.score_global_row_candidate(
        candidate, median_detector_width_um, settings
    )
    slope_penalty = settings.trajectory_slope_penalty_weight * (
        abs(float(slope_angle_deg))
        / max(settings.trajectory_local_tilt_limit_deg, 1e-9)
    ) ** 2
    adjusted = (
        float(diagnostic["original_bic"])
        + float(diagnostic["singleton_penalty"])
        + slope_penalty
    )
    diagnostic.update(
        {
            "v14_spacing_penalty_ignored_in_v15": diagnostic[
                "spacing_penalty"
            ],
            "v14_large_gap_penalty_ignored_in_v15": diagnostic[
                "large_gap_penalty"
            ],
            "v14_pair_partition_bonus_ignored_in_v15": diagnostic[
                "pair_partition_score_bonus"
            ],
            "spacing_penalty": 0.0,
            "large_gap_penalty": 0.0,
            "pair_partition_score_bonus": 0.0,
            "trajectory_slope_lateral_per_longitudinal": float(
                candidate["slope"]
            ),
            "trajectory_slope_angle_deg": float(slope_angle_deg),
            "trajectory_slope_penalty": slope_penalty,
            "trajectory_adjusted_score": adjusted,
            "adjusted_score": adjusted,
        }
    )
    return diagnostic


def _fit_trajectory_candidates(
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV15,
    separation_scale: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[int, tuple[dict[str, Any], dict[str, Any]]]]:
    """Fit every row count across the bounded shared-slope grid."""

    minimum_separation = separation_scale * max(
        settings.trajectory_minimum_separation_um,
        settings.trajectory_minimum_separation_width_factor
        * median_detector_width_um,
    )
    residual_floor_um = (
        settings.row_model_residual_floor_width_factor
        * median_detector_width_um
    )
    maximum_k = min(
        int(settings.trajectory_max_rows_per_supergroup), len(point_indices)
    )
    scored: list[dict[str, Any]] = []
    best_by_k: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for angle_deg in _slope_angles(settings):
        slope = math.tan(math.radians(angle_deg))
        for k in range(1, maximum_k + 1):
            candidate = base.fit_parallel_row_model(
                point_indices,
                lateral,
                longitudinal,
                k,
                minimum_points=1,
                minimum_separation_um=minimum_separation,
                fixed_slope=slope,
                residual_floor_um=residual_floor_um,
            )
            if candidate is None:
                continue
            diagnostic = _score_trajectory_candidate(
                candidate, median_detector_width_um, angle_deg, settings
            )
            record = {
                "candidate": candidate,
                "diagnostic": diagnostic,
                "angle_deg": angle_deg,
            }
            scored.append(record)
            existing = best_by_k.get(k)
            if existing is None or (
                diagnostic["adjusted_score"],
                abs(angle_deg),
                diagnostic["original_bic"],
            ) < (
                existing[1]["adjusted_score"],
                abs(existing[1]["trajectory_slope_angle_deg"]),
                existing[1]["original_bic"],
            ):
                best_by_k[k] = (candidate, diagnostic)
    if 1 not in best_by_k:
        raise RuntimeError("No valid one-trajectory candidate was produced.")
    return scored, best_by_k


def _candidate_image_support(
    candidate: dict[str, Any],
    median_detector_width_um: float,
    settings: TraitSettingsV15,
) -> tuple[list[float | None], float | None, list[dict[str, Any]]]:
    """Return local per-trajectory evidence and the singleton/flank ratio."""

    labels = np.asarray(candidate["labels"], dtype=int)
    point_indices = list(candidate["point_indices"])
    exclusion_boxes = (
        _spatial_stoma_exclusion_boxes_pixels(
            _ACTIVE_STOMA_TRAITS_V15,
            point_indices,
            settings.raw_evidence_stoma_exclusion_padding_fraction,
        )
        if _ACTIVE_STOMA_TRAITS_V15 is not None
        else []
    )

    evidences: list[dict[str, Any]] = []
    for group, intercept in enumerate(candidate["intercepts_um"]):
        member_positions = np.flatnonzero(labels == group)
        member_indices = [point_indices[int(position)] for position in member_positions]
        if _ACTIVE_STOMA_TRAITS_V15 is not None and member_indices:
            member_longitudinal = [
                float(
                    _ACTIVE_STOMA_TRAITS_V15[index][
                        "longitudinal_coordinate_um"
                    ]
                )
                for index in member_indices
            ]
            long_min = min(member_longitudinal)
            long_max = max(member_longitudinal)
        else:
            long_min = None
            long_max = None
        evidences.append(
            _trajectory_gradient_evidence(
                float(intercept),
                float(candidate["slope"]),
                float(candidate["long_reference_um"]),
                median_detector_width_um,
                longitudinal_min_um=long_min,
                longitudinal_max_um=long_max,
                exclusion_boxes_pixels=exclusion_boxes,
            )
        )
    supports = [
        evidence.get("target_gradient_median") for evidence in evidences
    ]
    singleton_groups = [
        index for index, size in enumerate(candidate["cluster_sizes"]) if size == 1
    ]
    if len(singleton_groups) != 1:
        return supports, None, evidences
    ratio = evidences[singleton_groups[0]].get("target_to_flank_ratio")
    return supports, base.optional_float(ratio), evidences


def _choose_trajectory_model_once(
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV15,
    separation_scale: float = 1.0,
    allow_sparse_image_rescue: bool = True,
) -> tuple[
    dict[str, Any],
    dict[int, tuple[dict[str, Any], dict[str, Any]]],
    dict[str, Any],
]:
    """Choose one trajectory model without calculating perturbation stability."""

    scored, best_by_k = _fit_trajectory_candidates(
        point_indices,
        lateral,
        longitudinal,
        median_detector_width_um,
        settings,
        separation_scale=separation_scale,
    )
    ranked = sorted(
        (
            (int(k), candidate, diagnostic)
            for k, (candidate, diagnostic) in best_by_k.items()
        ),
        key=lambda item: (
            item[2]["adjusted_score"],
            item[0],
            abs(item[2]["trajectory_slope_angle_deg"]),
        ),
    )
    strict_k, strict_candidate, strict_diagnostic = ranked[0]
    strict_score = float(strict_diagnostic["adjusted_score"])
    tied = [
        item
        for item in ranked
        if float(item[2]["adjusted_score"])
        <= strict_score + settings.trajectory_simpler_tie_tolerance
    ]
    automatic_k, automatic_candidate, automatic_diagnostic = min(
        tied, key=lambda item: item[0]
    )
    reason = "image_wide_detrended_trajectory_candidate_comparison"

    direct_two_row_evaluated = bool(automatic_k == 2 and 1 in best_by_k)
    direct_two_row_validated = False
    direct_two_row_rejected = False
    direct_two_row_score_improvement: float | None = None
    direct_two_row_strong_score_support = False
    direct_two_row_structural_support = False
    direct_evidence = {
        "direct_two_row_alternation": None,
        "direct_two_row_phase_offset_normalized": None,
        "direct_two_row_longest_alternating_sequence": None,
    }
    if direct_two_row_evaluated:
        one_diagnostic = best_by_k[1][1]
        direct_evidence = v14._direct_two_row_structural_evidence(
            automatic_candidate
        )
        direct_two_row_score_improvement = float(
            one_diagnostic["adjusted_score"]
            - automatic_diagnostic["adjusted_score"]
        )
        direct_two_row_strong_score_support = bool(
            direct_two_row_score_improvement
            >= settings.direct_two_row_min_score_improvement
        )
        alternation = direct_evidence["direct_two_row_alternation"]
        phase = direct_evidence["direct_two_row_phase_offset_normalized"]
        longest = direct_evidence[
            "direct_two_row_longest_alternating_sequence"
        ]
        direct_two_row_structural_support = bool(
            alternation is not None
            and alternation >= settings.direct_two_row_min_alternation
            and phase is not None
            and phase <= settings.direct_two_row_max_phase_offset_normalized
            and longest is not None
            and longest
            >= settings.direct_two_row_min_longest_alternating_sequence
        )
        direct_two_row_validated = bool(
            direct_two_row_strong_score_support
            or direct_two_row_structural_support
        )
        direct_two_row_rejected = not direct_two_row_validated
        automatic_diagnostic.update(
            {
                **direct_evidence,
                "direct_two_row_score_improvement_vs_one": (
                    direct_two_row_score_improvement
                ),
                "direct_two_row_strong_score_support": (
                    direct_two_row_strong_score_support
                ),
                "direct_two_row_structural_support": (
                    direct_two_row_structural_support
                ),
                "direct_two_row_validated": direct_two_row_validated,
                "direct_two_row_rejected_to_one_row": direct_two_row_rejected,
            }
        )
        if direct_two_row_rejected:
            automatic_k = 1
            automatic_candidate, automatic_diagnostic = best_by_k[1]
            reason += "__two_row_rejected_after_tilt_correction"
        else:
            reason += "__two_row_validated_after_tilt_correction"

    sparse_rescue_applied = False
    sparse_rescue_image_support_ratio: float | None = None
    sparse_rescue_image_supports: list[float | None] = []
    sparse_rescue_image_support_evidence: list[dict[str, Any]] = []
    sparse_rescue_bic_improvement: float | None = None
    sparse_rescue_rss_reduction: float | None = None
    sparse_rescue_comparison_row_count: int | None = None
    sparse_geometry_candidate_without_image_confirmation = False
    sparse_geometry_candidate_image_support_ratio: float | None = None
    sparse_geometry_candidate_image_support_unavailable = False
    sparse_rescue_candidates: list[
        tuple[
            float,
            float,
            dict[str, Any],
            dict[str, Any],
            list[float | None],
            list[dict[str, Any]],
            int,
        ]
    ] = []
    if allow_sparse_image_rescue and automatic_k <= 2:
        for record in scored:
            candidate = record["candidate"]
            diagnostic = record["diagnostic"]
            if (
                int(candidate["k"]) != 3
                or sum(int(size) == 1 for size in candidate["cluster_sizes"]) != 1
                or any(
                    int(size) < 2
                    for size in candidate["cluster_sizes"]
                    if int(size) != 1
                )
            ):
                continue
            same_slope_simpler = [
                other["candidate"]
                for other in scored
                if int(other["candidate"]["k"]) < 3
                and math.isclose(
                    float(other["angle_deg"]),
                    float(record["angle_deg"]),
                    rel_tol=0.0,
                    abs_tol=1e-10,
                )
            ]
            if not same_slope_simpler:
                continue
            # Compare the proposed third trajectory with the closest simpler
            # model (normally k=2), not merely with k=1. Otherwise a two-row
            # configuration can make an unsupported third row look beneficial.
            comparison = max(
                same_slope_simpler, key=lambda item: int(item["k"])
            )
            bic_improvement = float(
                comparison["bic"] - candidate["bic"]
            )
            rss_reduction = base.safe_divide(
                float(comparison["rss"] - candidate["rss"]),
                float(comparison["rss"]),
            )
            singleton_confidences = diagnostic[
                "singleton_detector_confidences"
            ]
            singleton_confidence = (
                float(singleton_confidences[0])
                if len(singleton_confidences) == 1
                else None
            )
            if (
                bic_improvement
                < settings.sparse_trajectory_rescue_min_bic_improvement
                or rss_reduction is None
                or rss_reduction
                < settings.sparse_trajectory_rescue_min_rss_reduction
                or float(diagnostic["spacing_cv"])
                > settings.sparse_trajectory_rescue_max_spacing_cv
                or singleton_confidence is None
                or singleton_confidence
                < settings.sparse_trajectory_rescue_min_detector_confidence
            ):
                continue
            supports, image_ratio, image_evidence = _candidate_image_support(
                candidate, median_detector_width_um, settings
            )
            if (
                image_ratio is None
                or image_ratio
                < settings.sparse_trajectory_rescue_min_image_support_ratio
            ):
                sparse_geometry_candidate_without_image_confirmation = True
                if image_ratio is not None:
                    candidate_ratio = float(image_ratio)
                    if (
                        sparse_geometry_candidate_image_support_ratio is None
                        or candidate_ratio
                        > sparse_geometry_candidate_image_support_ratio
                    ):
                        sparse_geometry_candidate_image_support_ratio = (
                            candidate_ratio
                        )
                else:
                    sparse_geometry_candidate_image_support_unavailable = True
                continue
            sparse_rescue_candidates.append(
                (
                    float(image_ratio),
                    bic_improvement,
                    candidate,
                    diagnostic,
                    supports,
                    image_evidence,
                    int(comparison["k"]),
                )
            )

    if sparse_rescue_candidates:
        (
            sparse_rescue_image_support_ratio,
            sparse_rescue_bic_improvement,
            automatic_candidate,
            automatic_diagnostic,
            sparse_rescue_image_supports,
            sparse_rescue_image_support_evidence,
            sparse_rescue_comparison_row_count,
        ) = max(
            sparse_rescue_candidates,
            key=lambda item: (item[0], item[1], -item[3]["adjusted_score"]),
        )
        same_slope_comparison = next(
            record["candidate"]
            for record in scored
            if int(record["candidate"]["k"])
            == sparse_rescue_comparison_row_count
            and math.isclose(
                float(record["angle_deg"]),
                float(automatic_diagnostic["trajectory_slope_angle_deg"]),
                rel_tol=0.0,
                abs_tol=1e-10,
            )
        )
        sparse_rescue_rss_reduction = base.safe_divide(
            float(
                same_slope_comparison["rss"] - automatic_candidate["rss"]
            ),
            float(same_slope_comparison["rss"]),
        )
        automatic_k = 3
        sparse_rescue_applied = True
        reason += "__sparse_three_trajectory_raw_image_rescue"
        best_by_k[3] = (automatic_candidate, automatic_diagnostic)

    metadata = {
        "strict_best_row_count": int(strict_k),
        "strict_best_score": strict_score,
        "automatic_row_count": int(automatic_k),
        "automatic_selection_reason": reason,
        "direct_two_row_candidate_evaluated": direct_two_row_evaluated,
        "direct_two_row_validated": direct_two_row_validated,
        "direct_two_row_rejected_to_one_row": direct_two_row_rejected,
        "direct_two_row_score_improvement_vs_one": direct_two_row_score_improvement,
        "direct_two_row_strong_score_support": direct_two_row_strong_score_support,
        "direct_two_row_structural_support": direct_two_row_structural_support,
        **direct_evidence,
        "sparse_trajectory_rescue_applied": sparse_rescue_applied,
        "sparse_trajectory_rescue_image_support_ratio": (
            sparse_rescue_image_support_ratio
        ),
        "sparse_trajectory_rescue_image_supports": sparse_rescue_image_supports,
        "sparse_trajectory_rescue_image_support_evidence": (
            sparse_rescue_image_support_evidence
        ),
        "sparse_trajectory_rescue_bic_improvement": sparse_rescue_bic_improvement,
        "sparse_trajectory_rescue_rss_reduction": sparse_rescue_rss_reduction,
        "sparse_trajectory_rescue_comparison_row_count": (
            sparse_rescue_comparison_row_count
        ),
        "sparse_trajectory_geometry_candidate_without_image_confirmation": (
            sparse_geometry_candidate_without_image_confirmation
            and not sparse_rescue_applied
        ),
        "sparse_trajectory_geometry_candidate_image_support_ratio": (
            sparse_geometry_candidate_image_support_ratio
        ),
        "sparse_trajectory_geometry_candidate_image_support_unavailable": (
            sparse_geometry_candidate_image_support_unavailable
            and not sparse_rescue_applied
        ),
        "trajectory_slope_grid_candidate_count": len(scored),
    }
    return automatic_candidate, best_by_k, metadata


def select_row_model_v15(
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV15,
    audited_row_count: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Infer image-wide-aligned row trajectories with stability QC."""

    legacy, _ = v14._BASE_SELECT_ROW_MODEL(
        point_indices,
        lateral,
        longitudinal,
        median_detector_width_um,
        settings,
        audited_row_count=None,
    )
    automatic, best_by_k, metadata = _choose_trajectory_model_once(
        point_indices,
        lateral,
        longitudinal,
        median_detector_width_um,
        settings,
    )
    automatic_k = int(automatic["k"])
    trajectory_maximum_k_reached = bool(
        automatic_k == settings.trajectory_max_rows_per_supergroup
        and len(point_indices) > settings.trajectory_max_rows_per_supergroup
    )
    singleton_group_indices = [
        group
        for group, size in enumerate(automatic["cluster_sizes"])
        if int(size) == 1
    ]
    singleton_validation_required = bool(singleton_group_indices)
    singleton_detector_confidences = v14._singleton_detector_confidences(
        automatic
    )
    singleton_support_ratios: list[float | None] = []
    singleton_support_available: list[bool] = []
    if singleton_validation_required:
        _, _, automatic_image_evidence = _candidate_image_support(
            automatic, median_detector_width_um, settings
        )
        for group in singleton_group_indices:
            evidence = automatic_image_evidence[group]
            singleton_support_ratios.append(
                base.optional_float(evidence.get("target_to_flank_ratio"))
            )
            singleton_support_available.append(bool(evidence.get("available")))
    singleton_rows_validated = bool(
        not singleton_validation_required
        or (
            len(singleton_detector_confidences) == len(singleton_group_indices)
            and all(
                value
                >= settings.sparse_trajectory_rescue_min_detector_confidence
                for value in singleton_detector_confidences
            )
            and len(singleton_support_ratios) == len(singleton_group_indices)
            and all(singleton_support_available)
            and all(
                value is not None
                and value
                >= settings.sparse_trajectory_rescue_min_image_support_ratio
                for value in singleton_support_ratios
            )
        )
    )
    singleton_failure_reasons: list[str] = []
    if singleton_validation_required and not singleton_rows_validated:
        if not all(singleton_support_available):
            singleton_failure_reasons.append("local_raw_edge_support_unavailable")
        if any(
            value
            < settings.sparse_trajectory_rescue_min_detector_confidence
            for value in singleton_detector_confidences
        ) or len(singleton_detector_confidences) != len(singleton_group_indices):
            singleton_failure_reasons.append("singleton_detector_confidence_too_low")
        if any(
            value is None
            or value < settings.sparse_trajectory_rescue_min_image_support_ratio
            for value in singleton_support_ratios
        ) or len(singleton_support_ratios) != len(singleton_group_indices):
            singleton_failure_reasons.append("singleton_local_edge_ratio_too_low")

    seed_offset = sum(
        (position + 1) * (int(index) + 1)
        for position, index in enumerate(point_indices)
    )
    rng = np.random.default_rng(
        int(settings.trajectory_stability_seed) + seed_offset
    )
    stability_counts: dict[int, int] = {}
    assignment_agreements: list[float] = []
    automatic_labels = np.asarray(automatic["labels"], dtype=int)
    jitter_um = (
        settings.trajectory_stability_jitter_width_fraction
        * median_detector_width_um
    )
    for _ in range(settings.trajectory_stability_iterations):
        jittered = np.asarray(lateral, dtype=float).copy()
        jittered[point_indices] += rng.normal(0.0, jitter_um, len(point_indices))
        separation_scale = float(rng.uniform(0.90, 1.10))
        perturbed, _, _ = _choose_trajectory_model_once(
            point_indices,
            jittered,
            longitudinal,
            median_detector_width_um,
            settings,
            separation_scale=separation_scale,
            allow_sparse_image_rescue=True,
        )
        perturbed_k = int(perturbed["k"])
        stability_counts[perturbed_k] = stability_counts.get(perturbed_k, 0) + 1
        if perturbed_k == automatic_k:
            perturbed_labels = np.asarray(perturbed["labels"], dtype=int)
            assignment_agreements.append(
                float(np.mean(perturbed_labels == automatic_labels))
            )
    row_count_stability = (
        stability_counts.get(automatic_k, 0)
        / settings.trajectory_stability_iterations
    )
    assignment_stability = (
        float(np.mean(assignment_agreements))
        if assignment_agreements
        else 0.0
    )
    trajectory_stable = bool(
        row_count_stability
        >= settings.trajectory_stability_min_row_count_fraction
        and assignment_stability
        >= settings.trajectory_stability_min_assignment_fraction
    )
    unresolved_sparse_third_trajectory = bool(
        metadata.get(
            "sparse_trajectory_geometry_candidate_without_image_confirmation"
        )
    )

    chosen = dict(automatic)
    selection_reason = metadata["automatic_selection_reason"]
    if audited_row_count is not None:
        if audited_row_count not in best_by_k:
            raise ValueError(
                f"Audited row_count={audited_row_count} has no valid trajectory "
                f"candidate for a supergroup containing {len(point_indices)} stomata."
            )
        chosen = dict(best_by_k[audited_row_count][0])
        selection_reason = "audited_row_count_override"

    chosen_k = int(chosen["k"])
    chosen_diagnostic = best_by_k[chosen_k][1]
    alternatives = sorted(
        (
            diagnostic
            for k, (_, diagnostic) in best_by_k.items()
            if int(k) != automatic_k
        ),
        key=lambda item: (
            item["adjusted_score"], item["candidate_row_count"]
        ),
    )
    alternative = alternatives[0] if alternatives else None
    raw_margin = (
        float(alternative["adjusted_score"])
        - float(best_by_k[automatic_k][1]["adjusted_score"])
        if alternative is not None
        else math.inf
    )
    confidence_margin = max(0.0, raw_margin)
    strict_best = bool(
        automatic_k == int(metadata["strict_best_row_count"])
        and not metadata["sparse_trajectory_rescue_applied"]
    )
    singleton_count = sum(int(size) == 1 for size in automatic["cluster_sizes"])
    if (
        trajectory_maximum_k_reached
        or unresolved_sparse_third_trajectory
        or not singleton_rows_validated
    ):
        model_confidence = "low"
    elif (
        trajectory_stable
        and strict_best
        and confidence_margin >= settings.global_row_high_confidence_margin
        and singleton_count == 0
    ):
        model_confidence = "high"
    elif trajectory_stable and singleton_count == 0:
        model_confidence = "moderate"
    else:
        model_confidence = "low"

    simpler = best_by_k.get(chosen_k - 1)
    chosen["delta_bic_vs_simpler"] = (
        float(simpler[0]["bic"] - chosen["bic"])
        if simpler is not None
        else None
    )
    one_score = best_by_k.get(1, (None, {}))[1].get("adjusted_score")
    two_score = best_by_k.get(2, (None, {}))[1].get("adjusted_score")
    one_vs_two_ambiguous = bool(
        one_score is not None
        and two_score is not None
        and (
            abs(float(one_score) - float(two_score))
            <= settings.trajectory_simpler_tie_tolerance
            or metadata["direct_two_row_rejected_to_one_row"]
        )
    )
    chosen.update(
        {
            "automatic_row_count": automatic_k,
            "automatic_selection_reason": metadata[
                "automatic_selection_reason"
            ],
            "automatic_one_vs_two_ambiguous": one_vs_two_ambiguous,
            "audited_row_count_override": audited_row_count is not None,
            "row_count_selection_reason": selection_reason,
            "tilt_slope_source": "image_wide_residual_trajectory_detrending",
            "one_row_rms_residual_um": float(best_by_k[1][0]["rms_residual_um"]),
            "simple_fit_stop_tolerance_um": (
                settings.row_model_simple_fit_stop_width_factor
                * median_detector_width_um
            ),
            "one_vs_two_ambiguous": one_vs_two_ambiguous,
            "sparse_row_support": any(
                int(size) < 3 for size in chosen["cluster_sizes"]
            ),
            "model_confidence": (
                "audited" if audited_row_count is not None else model_confidence
            ),
            "trajectory_slope_angle_deg": math.degrees(
                math.atan(float(chosen["slope"]))
            ),
            "trajectory_row_count_stability_fraction": row_count_stability,
            "trajectory_assignment_stability_fraction": assignment_stability,
            "trajectory_conditional_row_count_sensitivity_fraction": (
                row_count_stability
            ),
            "trajectory_conditional_assignment_sensitivity_fraction": (
                assignment_stability
            ),
            "trajectory_row_count_stable": trajectory_stable,
            "trajectory_maximum_k_reached": trajectory_maximum_k_reached,
            "trajectory_unresolved_sparse_third_trajectory": (
                unresolved_sparse_third_trajectory
            ),
            "trajectory_singleton_validation_required": (
                singleton_validation_required
            ),
            "trajectory_singleton_rows_validated": singleton_rows_validated,
            "trajectory_singleton_detector_confidences": (
                singleton_detector_confidences
            ),
            "trajectory_singleton_local_edge_support_ratios": (
                singleton_support_ratios
            ),
            "trajectory_singleton_validation_failure_reason": ";".join(
                singleton_failure_reasons
            ),
            "trajectory_stability_count_distribution": stability_counts,
            "trajectory_exact_row_traits_eligible": bool(
                (
                    trajectory_stable
                    and not trajectory_maximum_k_reached
                    and not unresolved_sparse_third_trajectory
                    and singleton_rows_validated
                )
                or audited_row_count is not None
            ),
            **metadata,
        }
    )

    candidate_diagnostics: list[dict[str, Any]] = []
    candidates_for_base: list[dict[str, Any]] = []
    for k in sorted(best_by_k):
        candidate, diagnostic = best_by_k[k]
        candidates_for_base.append(candidate)
        candidate_diagnostics.append(diagnostic)
    automatic_diagnostic = best_by_k[automatic_k][1]
    diagnostic_record = {
        "legacy_sequential_row_count": int(legacy["k"]),
        "legacy_sequential_selection_reason": legacy[
            "row_count_selection_reason"
        ],
        "global_automatic_row_count": automatic_k,
        "row_count_used_for_traits": chosen_k,
        "row_count_used_for_traits_source": (
            "audited_override"
            if audited_row_count is not None
            else "trajectory_automatic"
        ),
        "alternative_row_count": (
            int(alternative["candidate_row_count"])
            if alternative is not None
            else None
        ),
        "global_row_selection_score": float(
            automatic_diagnostic["adjusted_score"]
        ),
        "global_row_score_margin": confidence_margin,
        "global_row_raw_margin": raw_margin,
        "global_row_automatic_is_strict_best": strict_best,
        "global_row_model_confidence": model_confidence,
        "singleton_supported_row_count": singleton_count,
        "singleton_detector_confidence_mean": automatic_diagnostic[
            "singleton_detector_confidence_mean"
        ],
        "global_row_count_changed_from_legacy": automatic_k != int(legacy["k"]),
        "potential_row_undersplit_in_legacy": automatic_k > int(legacy["k"]),
        "potential_row_oversplit_in_legacy": automatic_k < int(legacy["k"]),
        "candidate_diagnostics": candidate_diagnostics,
        "trajectory_slope_angle_deg": chosen["trajectory_slope_angle_deg"],
        "trajectory_row_count_stability_fraction": row_count_stability,
        "trajectory_assignment_stability_fraction": assignment_stability,
        "trajectory_conditional_row_count_sensitivity_fraction": (
            row_count_stability
        ),
        "trajectory_conditional_assignment_sensitivity_fraction": (
            assignment_stability
        ),
        "trajectory_row_count_stable": trajectory_stable,
        "trajectory_maximum_k_reached": trajectory_maximum_k_reached,
        "trajectory_unresolved_sparse_third_trajectory": (
            unresolved_sparse_third_trajectory
        ),
        "trajectory_singleton_validation_required": singleton_validation_required,
        "trajectory_singleton_rows_validated": singleton_rows_validated,
        "trajectory_singleton_detector_confidences": (
            singleton_detector_confidences
        ),
        "trajectory_singleton_local_edge_support_ratios": (
            singleton_support_ratios
        ),
        "trajectory_singleton_validation_failure_reason": ";".join(
            singleton_failure_reasons
        ),
        "trajectory_stability_count_distribution": stability_counts,
        "trajectory_exact_row_traits_eligible": chosen[
            "trajectory_exact_row_traits_eligible"
        ],
        **metadata,
    }
    v14._ACTIVE_BAND_DIAGNOSTICS.append(diagnostic_record)
    return chosen, candidates_for_base


def _matched_trajectory_gap(
    first: dict[str, Any], second: dict[str, Any]
) -> tuple[float, float]:
    """Median lateral gap and longitudinal-overlap fraction for two rows."""

    first_min = float(first["longitudinal_min_um"])
    first_max = float(first["longitudinal_max_um"])
    second_min = float(second["longitudinal_min_um"])
    second_max = float(second["longitudinal_max_um"])
    overlap_min = max(first_min, second_min)
    overlap_max = min(first_max, second_max)
    union_span = max(first_max, second_max) - min(first_min, second_min)
    overlap_span = max(0.0, overlap_max - overlap_min)
    overlap_fraction = (
        overlap_span / union_span if union_span > 0 else 0.0
    )
    if overlap_span > 0:
        positions = np.linspace(overlap_min, overlap_max, 31)
    else:
        positions = np.asarray(
            [0.5 * (max(first_min, second_min) + min(first_max, second_max))]
        )
    gaps = [
        abs(
            base.row_lateral_at(second, float(longitudinal))
            - base.row_lateral_at(first, float(longitudinal))
        )
        for longitudinal in positions
    ]
    return float(np.median(gaps)), float(overlap_fraction)


def _partition_trajectory_bands_v15(
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    settings: TraitSettingsV15,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Demarcate bands after fitting rows using matched lateral distances."""

    row_lookup = {str(row["row_id"]): row for row in row_records}
    if len(v14._ACTIVE_BAND_DIAGNOSTICS) != len(band_records):
        raise RuntimeError(
            "Trajectory diagnostics do not align with broad provisional "
            f"groups: diagnostics={len(v14._ACTIVE_BAND_DIAGNOSTICS)}, "
            f"groups={len(band_records)}."
        )
    # Version 1.4 attaches only its established fields before calling the band
    # partition hook. Add the v1.5 stability fields here so every rebuilt band
    # inherits the correct trajectory evidence.
    for original, diagnostic in zip(
        band_records, v14._ACTIVE_BAND_DIAGNOSTICS, strict=True
    ):
        original.update(
            {
                "trajectory_slope_angle_deg": diagnostic.get(
                    "trajectory_slope_angle_deg"
                ),
                "trajectory_row_count_stability_fraction": diagnostic.get(
                    "trajectory_row_count_stability_fraction"
                ),
                "trajectory_assignment_stability_fraction": diagnostic.get(
                    "trajectory_assignment_stability_fraction"
                ),
                "trajectory_row_count_stable": bool(
                    diagnostic.get("trajectory_row_count_stable")
                ),
                "trajectory_maximum_k_reached": bool(
                    diagnostic.get("trajectory_maximum_k_reached")
                ),
                "trajectory_unresolved_sparse_third_trajectory": bool(
                    diagnostic.get(
                        "trajectory_unresolved_sparse_third_trajectory"
                    )
                ),
                "trajectory_stability_count_distribution": diagnostic.get(
                    "trajectory_stability_count_distribution", {}
                ),
                "trajectory_exact_row_traits_eligible": bool(
                    diagnostic.get("trajectory_exact_row_traits_eligible")
                ),
                "sparse_trajectory_rescue_applied": bool(
                    diagnostic.get("sparse_trajectory_rescue_applied")
                ),
                "sparse_trajectory_rescue_image_support_ratio": diagnostic.get(
                    "sparse_trajectory_rescue_image_support_ratio"
                ),
                "sparse_trajectory_rescue_bic_improvement": diagnostic.get(
                    "sparse_trajectory_rescue_bic_improvement"
                ),
                "sparse_trajectory_rescue_rss_reduction": diagnostic.get(
                    "sparse_trajectory_rescue_rss_reduction"
                ),
                "sparse_trajectory_rescue_comparison_row_count": diagnostic.get(
                    "sparse_trajectory_rescue_comparison_row_count"
                ),
            }
        )
    rebuilt: list[dict[str, Any]] = []
    partition_audit: list[dict[str, Any]] = []
    for original in band_records:
        rows = sorted(
            [row_lookup[str(row_id)] for row_id in original["row_ids"]],
            key=lambda row: base.row_lateral_at(row, 0.0),
        )
        boundary_exclusion_boxes = _spatial_stoma_exclusion_boxes_pixels(
            stoma_traits,
            list(original["_point_indices"]),
            settings.raw_evidence_stoma_exclusion_padding_fraction,
        )
        gap_measurements = [
            _matched_trajectory_gap(first, second)
            for first, second in zip(rows[:-1], rows[1:])
        ]
        raw_gaps = [item[0] for item in gap_measurements]
        boundary_evidences: list[dict[str, Any]] = []
        for first, second in zip(rows[:-1], rows[1:]):
            boundary_at_zero = 0.5 * (
                base.row_lateral_at(first, 0.0)
                + base.row_lateral_at(second, 0.0)
            )
            boundary_slope = 0.5 * (
                float(first["row_common_slope_lateral_per_longitudinal"])
                + float(second["row_common_slope_lateral_per_longitudinal"])
            )
            overlap_min = max(
                float(first["longitudinal_min_um"]),
                float(second["longitudinal_min_um"]),
            )
            overlap_max = min(
                float(first["longitudinal_max_um"]),
                float(second["longitudinal_max_um"]),
            )
            boundary_evidences.append(
                _trajectory_gradient_evidence(
                    boundary_at_zero,
                    boundary_slope,
                    0.0,
                    median_detector_width_um,
                    longitudinal_min_um=overlap_min,
                    longitudinal_max_um=overlap_max,
                    exclusion_boxes_pixels=boundary_exclusion_boxes,
                )
            )

        accepted_cuts: list[int] = []
        candidates_for_original: list[dict[str, Any]] = []
        for index, ((gap_um, overlap_fraction), image_evidence) in enumerate(
            zip(gap_measurements, boundary_evidences, strict=True), start=1
        ):
            factor = gap_um / max(median_detector_width_um, 1e-9)
            other_gaps = [
                value for position, value in enumerate(raw_gaps) if position != index - 1
            ]
            within_band_reference_gaps = [
                value
                for value in other_gaps
                if value / max(median_detector_width_um, 1e-9)
                <= settings.band_same_max_gap_width_factor
            ]
            local_reference = (
                float(np.median(within_band_reference_gaps))
                if within_band_reference_gaps
                else float(np.median(other_gaps)) if other_gaps else None
            )
            local_gap_ratio = (
                gap_um / max(local_reference, 1e-9)
                if local_reference is not None
                else math.inf
            )
            image_score = image_evidence.get("target_gradient_median")
            image_control = image_evidence.get("flank_gradient_median")
            image_ratio = image_evidence.get("target_to_flank_ratio")
            overlap_supported = bool(
                overlap_fraction
                >= settings.band_boundary_min_longitudinal_overlap_fraction
            )
            same_band_geometry = bool(
                factor <= settings.band_same_max_gap_width_factor
            )
            strong_boundary = bool(
                factor >= settings.band_strong_boundary_gap_width_factor
                and overlap_supported
            )
            ambiguous_boundary_supported = bool(
                settings.band_same_max_gap_width_factor
                < factor
                < settings.band_strong_boundary_gap_width_factor
                and overlap_supported
                and local_gap_ratio
                >= settings.band_ambiguous_min_local_gap_ratio
                and image_ratio is not None
                and image_ratio
                >= settings.band_ambiguous_min_image_support_ratio
            )

            seed_offset = sum(ord(char) for char in str(original["band_id"])) + index
            rng = np.random.default_rng(
                settings.trajectory_stability_seed + seed_offset
            )
            stable_votes = 0
            same_band_votes = 0
            for _ in range(settings.trajectory_stability_iterations):
                perturbed_gap = max(
                    0.0,
                    gap_um
                    + float(
                        rng.normal(
                            0.0,
                            math.sqrt(2.0)
                            * settings.trajectory_stability_jitter_width_fraction
                            * median_detector_width_um,
                        )
                    ),
                )
                perturbed_factor = perturbed_gap / max(
                    median_detector_width_um, 1e-9
                )
                perturbed_strong = bool(
                    perturbed_factor
                    >= settings.band_strong_boundary_gap_width_factor
                    and overlap_supported
                )
                perturbed_ambiguous = bool(
                    settings.band_same_max_gap_width_factor
                    < perturbed_factor
                    < settings.band_strong_boundary_gap_width_factor
                    and overlap_supported
                    and local_gap_ratio
                    >= settings.band_ambiguous_min_local_gap_ratio
                    and image_ratio is not None
                    and image_ratio
                    >= settings.band_ambiguous_min_image_support_ratio
                )
                stable_votes += int(perturbed_strong or perturbed_ambiguous)
                same_band_votes += int(
                    perturbed_factor
                    <= settings.band_same_max_gap_width_factor
                )
            stability_fraction = (
                stable_votes / settings.trajectory_stability_iterations
            )
            stable_boundary_gap_decision = bool(
                stability_fraction
                >= settings.band_boundary_stability_min_fraction
            )
            same_band_stability_fraction = (
                same_band_votes / settings.trajectory_stability_iterations
            )
            stable_same_band = bool(
                same_band_geometry
                and same_band_stability_fraction
                >= settings.band_boundary_stability_min_fraction
            )
            accepted = bool(
                (strong_boundary or ambiguous_boundary_supported)
                and stable_boundary_gap_decision
            )
            candidate = {
                "provisional_band_id": original["band_id"],
                "split": index,
                "boundary_gap_um": gap_um,
                "gap_ratio": local_gap_ratio,
                "gap_width_factor": factor,
                "longitudinal_overlap_fraction": overlap_fraction,
                "left_alternation": v14._row_group_alternation(
                    rows[:index], stoma_traits
                ),
                "right_alternation": v14._row_group_alternation(
                    rows[index:], stoma_traits
                ),
                "raw_image_boundary_gradient": image_score,
                "raw_image_control_gradient_median": (
                    image_control
                ),
                "raw_image_control_source": "parallel_local_flanks",
                "raw_image_support_available": bool(
                    image_evidence.get("available")
                ),
                "raw_image_support_unavailable_reason": image_evidence.get(
                    "unavailable_reason", ""
                ),
                "raw_image_support_sample_count": image_evidence.get(
                    "sample_count"
                ),
                "image_support_ratio": image_ratio,
                "same_band_geometry": same_band_geometry,
                "same_band_stability_fraction": same_band_stability_fraction,
                "same_band_stable": stable_same_band,
                "strong_boundary_geometry": strong_boundary,
                "ambiguous_boundary_image_supported": (
                    ambiguous_boundary_supported
                ),
                "boundary_gap_threshold_stability_fraction": stability_fraction,
                "boundary_gap_threshold_stable": stable_boundary_gap_decision,
                # Compatibility aliases; the explicit names above define the
                # narrower interpretation used by v1.5.
                "boundary_stability_fraction": stability_fraction,
                "boundary_stable": stable_boundary_gap_decision,
                "accepted": accepted,
            }
            candidates_for_original.append(candidate)
            partition_audit.append(candidate)
            if accepted:
                accepted_cuts.append(index)

        boundaries = [0, *sorted(set(accepted_cuts)), len(rows)]
        groups = [
            rows[start:stop]
            for start, stop in zip(boundaries[:-1], boundaries[1:])
            if stop > start
        ]
        unresolved_boundary = any(
            (
                candidate["same_band_geometry"]
                and not candidate["same_band_stable"]
            )
            or (
                not candidate["same_band_geometry"]
                and not candidate["accepted"]
            )
            for candidate in candidates_for_original
        )
        for group_index, group in enumerate(groups, start=1):
            nearby_partition: dict[str, Any] | None = None
            if accepted_cuts:
                group_start = boundaries[group_index - 1]
                group_stop = boundaries[group_index]
                nearby_cut = (
                    group_stop
                    if group_stop in accepted_cuts
                    else group_start if group_start in accepted_cuts else accepted_cuts[0]
                )
                nearby_partition = next(
                    candidate
                    for candidate in candidates_for_original
                    if int(candidate["split"]) == int(nearby_cut)
                )
            record = v14._rebuild_band_record(
                original,
                group,
                stoma_traits,
                median_detector_width_um,
                nearby_partition,
                group_index,
            )
            matched_group_gaps = [
                _matched_trajectory_gap(first, second)[0]
                for first, second in zip(group[:-1], group[1:])
            ]
            record.update(
                base.sample_summary(
                    matched_group_gaps,
                    "within_band_interrow_spacing_um",
                )
            )
            record["within_band_interrow_spacing_mean_um"] = (
                float(np.mean(matched_group_gaps))
                if matched_group_gaps
                else None
            )
            record["within_band_interrow_spacing_median_um"] = (
                float(np.median(matched_group_gaps))
                if matched_group_gaps
                else None
            )
            record["within_band_spacing_basis"] = (
                "matched_longitudinal_positions_between_fitted_trajectories"
            )
            row_count_within_expected_range = bool(
                1 <= len(group) <= settings.maximum_rows_per_band
            )
            record.update(
                {
                    "trajectory_band_partition_method": (
                        "matched_lateral_trajectory_distance"
                    ),
                    "trajectory_multiple_boundaries_permitted": True,
                    "trajectory_unresolved_boundary_candidate": unresolved_boundary,
                    "trajectory_rows_within_expected_band_range": (
                        row_count_within_expected_range
                    ),
                    "trajectory_band_structure_stable": bool(
                        row_count_within_expected_range
                        and not unresolved_boundary
                        and bool(original.get("trajectory_row_count_stable", False))
                    ),
                    "trajectory_exact_row_band_traits_eligible": bool(
                        row_count_within_expected_range
                        and not unresolved_boundary
                        and bool(
                            original.get("trajectory_exact_row_traits_eligible", False)
                        )
                    ),
                }
            )
            rebuilt.append(record)

    rebuilt.sort(key=lambda band: float(band["band_center_lateral_um"]))
    for band_number, band in enumerate(rebuilt, start=1):
        final_band_id = f"B{band_number:03d}"
        band["band_id"] = final_band_id
        band["band_order_left_to_right"] = band_number
        rows = sorted(
            [row_lookup[str(row_id)] for row_id in band["row_ids"]],
            key=lambda row: base.row_lateral_at(row, 0.0),
        )
        for position, row in enumerate(rows, start=1):
            row["provisional_band_id"] = row.get("band_id")
            row["band_id"] = final_band_id
            row["row_position_within_band"] = position
            row["rows_in_band"] = len(rows)
            row["automatic_rows_in_band"] = len(rows)
            row["band_row_count"] = len(rows)
            row["band_partition_applied"] = bool(band["band_partition_applied"])
            row["band_partition_group_index"] = band[
                "band_partition_group_index"
            ]
            row["trajectory_band_structure_stable"] = band[
                "trajectory_band_structure_stable"
            ]
            row["trajectory_exact_row_band_traits_eligible"] = band[
                "trajectory_exact_row_band_traits_eligible"
            ]
            for index in row["_point_indices"]:
                assignment = point_assignment[index]
                assignment["provisional_band_id"] = assignment.get("band_id")
                assignment["band_id"] = final_band_id
                assignment["row_position_within_band"] = position
                assignment["rows_in_band"] = len(rows)
                assignment["automatic_rows_in_band"] = len(rows)
                assignment["band_partition_applied"] = bool(
                    band["band_partition_applied"]
                )
                assignment["trajectory_band_structure_stable"] = band[
                    "trajectory_band_structure_stable"
                ]
                assignment["trajectory_exact_row_band_traits_eligible"] = band[
                    "trajectory_exact_row_band_traits_eligible"
                ]

    return row_records, rebuilt, point_assignment, partition_audit


def _trajectory_diagnostic_fields(
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """Return the v1.5 diagnostic fields copied into CSV-level records."""

    return {
        "row_model_version": SCRIPT_VERSION,
        "trajectory_slope_angle_deg": diagnostic.get(
            "trajectory_slope_angle_deg"
        ),
        "trajectory_residual_slope_angle_deg": diagnostic.get(
            "trajectory_residual_slope_angle_deg"
        ),
        "trajectory_image_wide_residual_angle_deg": diagnostic.get(
            "trajectory_image_wide_residual_angle_deg"
        ),
        "trajectory_row_count_stability_fraction": diagnostic.get(
            "trajectory_row_count_stability_fraction"
        ),
        "trajectory_assignment_stability_fraction": diagnostic.get(
            "trajectory_assignment_stability_fraction"
        ),
        "trajectory_conditional_row_count_sensitivity_fraction": diagnostic.get(
            "trajectory_conditional_row_count_sensitivity_fraction"
        ),
        "trajectory_conditional_assignment_sensitivity_fraction": diagnostic.get(
            "trajectory_conditional_assignment_sensitivity_fraction"
        ),
        "trajectory_row_count_stable": bool(
            diagnostic.get("trajectory_row_count_stable")
        ),
        "trajectory_maximum_k_reached": bool(
            diagnostic.get("trajectory_maximum_k_reached")
        ),
        "trajectory_unresolved_sparse_third_trajectory": bool(
            diagnostic.get("trajectory_unresolved_sparse_third_trajectory")
        ),
        "trajectory_singleton_validation_required": bool(
            diagnostic.get("trajectory_singleton_validation_required")
        ),
        "trajectory_singleton_rows_validated": bool(
            diagnostic.get("trajectory_singleton_rows_validated")
        ),
        "trajectory_singleton_detector_confidences": diagnostic.get(
            "trajectory_singleton_detector_confidences", []
        ),
        "trajectory_singleton_local_edge_support_ratios": diagnostic.get(
            "trajectory_singleton_local_edge_support_ratios", []
        ),
        "trajectory_singleton_validation_failure_reason": diagnostic.get(
            "trajectory_singleton_validation_failure_reason", ""
        ),
        "trajectory_exact_row_traits_eligible": bool(
            diagnostic.get("trajectory_exact_row_traits_eligible")
        ),
        "sparse_trajectory_rescue_applied": bool(
            diagnostic.get("sparse_trajectory_rescue_applied")
        ),
        "sparse_trajectory_rescue_image_support_ratio": diagnostic.get(
            "sparse_trajectory_rescue_image_support_ratio"
        ),
        "sparse_trajectory_rescue_bic_improvement": diagnostic.get(
            "sparse_trajectory_rescue_bic_improvement"
        ),
        "sparse_trajectory_rescue_rss_reduction": diagnostic.get(
            "sparse_trajectory_rescue_rss_reduction"
        ),
        "sparse_trajectory_rescue_comparison_row_count": diagnostic.get(
            "sparse_trajectory_rescue_comparison_row_count"
        ),
        "sparse_trajectory_geometry_candidate_without_image_confirmation": bool(
            diagnostic.get(
                "sparse_trajectory_geometry_candidate_without_image_confirmation"
            )
        ),
        "sparse_trajectory_geometry_candidate_image_support_ratio": diagnostic.get(
            "sparse_trajectory_geometry_candidate_image_support_ratio"
        ),
        "sparse_trajectory_geometry_candidate_image_support_unavailable": bool(
            diagnostic.get(
                "sparse_trajectory_geometry_candidate_image_support_unavailable"
            )
        ),
    }


def _remark_partial_band_eligibility_v15(
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    trajectory_slope: float,
    settings: TraitSettingsV15,
) -> list[dict[str, Any]]:
    """Recalculate field coverage along the fitted image-wide trajectory."""

    axis = v14._active_axis(stoma_traits)
    metadata = v14._ACTIVE_IMAGE_METADATA
    if axis is None or metadata is None:
        # Preserve inherited results, but make the fallback explicit.
        return [
            {
                "band_id": band["band_id"],
                "provisional_band_id": band.get("provisional_band_id"),
                "spatial_eligible": bool(band.get("band_spatial_eligible")),
                "exclusion_reason": band.get("band_spatial_exclusion_reason", ""),
                "v15_recalculation_available": False,
                "v15_unavailable_reason": "image_axis_or_metadata_unavailable",
            }
            for band in band_records
        ]
    px_x = base.optional_float(metadata.get("metadata_pixels_per_um_x"))
    px_y = base.optional_float(metadata.get("metadata_pixels_per_um_y"))
    width_pixels = base.optional_int(metadata.get("image_width"))
    height_pixels = base.optional_int(metadata.get("image_height"))
    if px_x is None or px_y is None or width_pixels is None or height_pixels is None:
        return [
            {
                "band_id": band["band_id"],
                "provisional_band_id": band.get("provisional_band_id"),
                "spatial_eligible": bool(band.get("band_spatial_eligible")),
                "exclusion_reason": band.get("band_spatial_exclusion_reason", ""),
                "v15_recalculation_available": False,
                "v15_unavailable_reason": "calibration_or_dimensions_unavailable",
            }
            for band in band_records
        ]
    width_um = width_pixels / px_x
    height_um = height_pixels / px_y
    corners = [
        axis.forward(x, y)
        for x, y in (
            (0.0, 0.0),
            (width_um, 0.0),
            (0.0, height_um),
            (width_um, height_um),
        )
    ]
    image_long_min = min(item[1] for item in corners)
    image_long_max = max(item[1] for item in corners)

    measurements: list[dict[str, Any]] = []
    for band in band_records:
        observed_values = [
            float(stoma_traits[index]["longitudinal_coordinate_um"])
            for index in band["_point_indices"]
        ]
        observed = float(np.ptp(observed_values)) if observed_values else 0.0
        visible: list[float] = []
        center_at_zero = float(band["band_center_lateral_um"])
        for longitudinal_um in np.linspace(image_long_min, image_long_max, 1200):
            lateral_um = center_at_zero + trajectory_slope * float(longitudinal_um)
            x_um, y_um = axis.inverse(lateral_um, float(longitudinal_um))
            if 0.0 <= x_um <= width_um and 0.0 <= y_um <= height_um:
                visible.append(float(longitudinal_um))
        accessible = (
            float(max(visible) - min(visible)) if len(visible) >= 2 else None
        )
        edge_factor = base.optional_float(
            band.get("band_detector_side_edge_margin_box_width_factor")
        )
        edge_source = "detector_bbox_source_pixels"
        if edge_factor is None:
            edge_factor = base.optional_float(
                band.get("band_rotated_lateral_edge_margin_box_width_factor")
            )
            edge_source = "inherited_rotated_center_fallback"
        measurements.append(
            {
                "band": band,
                "observed": observed,
                "accessible": accessible,
                "coverage": base.safe_divide(observed, accessible),
                "edge_factor": edge_factor,
                "edge_source": edge_source,
            }
        )

    reference_pool = [
        item["observed"]
        for item in measurements
        if item["observed"] > 0
        and (
            item["edge_factor"] is None
            or item["edge_factor"]
            > settings.partial_band_edge_margin_box_width_factor
        )
    ]
    if not reference_pool:
        reference_pool = [
            item["observed"] for item in measurements if item["observed"] > 0
        ]
    reference_span = float(np.median(reference_pool)) if reference_pool else None
    row_lookup = {str(row["row_id"]): row for row in row_records}
    eligibility_audit: list[dict[str, Any]] = []
    for item in measurements:
        band = item["band"]
        observed = float(item["observed"])
        coverage = item["coverage"]
        relative_span = base.safe_divide(observed, reference_span)
        edge_factor = item["edge_factor"]
        low_absolute = bool(
            coverage is not None
            and coverage < settings.partial_band_min_longitudinal_coverage_fraction
        )
        low_relative = bool(
            relative_span is not None
            and relative_span < settings.partial_band_min_relative_observed_span
        )
        edge_risk = bool(
            edge_factor is not None
            and edge_factor <= settings.partial_band_edge_margin_box_width_factor
        )
        excluded = bool(
            (low_absolute and (low_relative or observed == 0.0))
            or (edge_risk and low_relative)
        )
        reason = ""
        if excluded:
            reason = (
                "field_truncated_band_short_relative_span_at_lateral_edge"
                if edge_risk
                else "incomplete_band_low_absolute_and_relative_longitudinal_span"
            )
        band.update(
            {
                "band_spatial_eligible": not excluded,
                "band_fov_truncated": excluded,
                "band_spatial_exclusion_reason": reason,
                "band_observed_longitudinal_span_um": observed,
                "band_accessible_longitudinal_span_um": item["accessible"],
                "band_longitudinal_coverage_fraction": coverage,
                "band_relative_observed_span": relative_span,
                "partial_band_reference_span_um": reference_span,
                "partial_band_accessible_span_basis": (
                    "fitted_image_wide_trajectory_clipped_to_source_field"
                ),
                "band_lateral_edge_margin_source": item["edge_source"],
            }
        )
        for row_id in band["row_ids"]:
            row = row_lookup[str(row_id)]
            row["row_band_spatial_eligible"] = not excluded
            row["row_band_spatial_exclusion_reason"] = reason
            for index in row["_point_indices"]:
                point_assignment[index]["row_band_spatial_eligible"] = not excluded
                point_assignment[index]["row_band_spatial_exclusion_reason"] = reason
        eligibility_audit.append(
            {
                "band_id": band["band_id"],
                "provisional_band_id": band.get("provisional_band_id"),
                "spatial_eligible": not excluded,
                "exclusion_reason": reason,
                "observed_longitudinal_span_um": observed,
                "accessible_longitudinal_span_um": item["accessible"],
                "coverage_fraction": coverage,
                "relative_observed_span": relative_span,
                "lateral_edge_margin_box_width_factor": edge_factor,
                "lateral_edge_margin_source": item["edge_source"],
                "v15_recalculation_available": True,
                "accessible_span_basis": (
                    "fitted_image_wide_trajectory_clipped_to_source_field"
                ),
            }
        )
    return eligibility_audit


def infer_rows_and_bands_v15(
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettingsV15,
    audited_row_counts: dict[str, int] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    """Infer trajectories first and demarcate biological bands second."""

    global _ACTIVE_AXIS_V15, _ACTIVE_STOMA_TRAITS_V15
    global _ACTIVE_GLOBAL_TRAJECTORY_SLOPE_V15
    global _ACTIVE_GLOBAL_TRAJECTORY_DETRENDING_V15
    if audited_row_counts:
        raise ValueError(
            "Version 1.5 does not apply inherited biological-band row-count "
            "overrides to broad pre-partition trajectory groups. Run without "
            "--audited-row-count-overrides; any later v1.5 audit manifest must "
            "use separately defined supergroup/final-band semantics."
        )
    _ACTIVE_AXIS_V15 = v14._active_axis(stoma_traits)
    _ACTIVE_STOMA_TRAITS_V15 = stoma_traits
    detector_widths = base.finite_values(
        row.get("prediction_width_um") for row in stoma_traits
    )
    median_detector_width_um = (
        float(np.median(detector_widths)) if detector_widths.size else 20.0
    )
    global_angle = _estimate_image_wide_trajectory_slope(
        stoma_traits, median_detector_width_um, settings
    )
    _ACTIVE_GLOBAL_TRAJECTORY_SLOPE_V15 = float(global_angle["slope"])
    _ACTIVE_GLOBAL_TRAJECTORY_DETRENDING_V15 = True
    original_lateral = [
        float(row["lateral_coordinate_um"]) for row in stoma_traits
    ]
    for row in stoma_traits:
        row["lateral_coordinate_um"] = float(row["lateral_coordinate_um"]) - (
            _ACTIVE_GLOBAL_TRAJECTORY_SLOPE_V15
            * float(row["longitudinal_coordinate_um"])
        )
    supergroup_sensitivity = _supergroup_membership_sensitivity(
        np.asarray(
            [float(row["lateral_coordinate_um"]) for row in stoma_traits],
            dtype=float,
        ),
        median_detector_width_um,
        settings,
    )
    try:
        row_records, band_records, point_assignment, audit = (
            v14.infer_rows_and_bands_v14(
                stoma_traits,
                settings,
                audited_row_counts=audited_row_counts,
            )
        )
        for row in row_records:
            reference = float(row["row_longitudinal_reference_um"])
            residual_slope = float(
                row["row_common_slope_lateral_per_longitudinal"]
            )
            row["row_intercept_lateral_um"] = float(
                row["row_intercept_lateral_um"]
            ) + _ACTIVE_GLOBAL_TRAJECTORY_SLOPE_V15 * reference
            row["row_common_slope_lateral_per_longitudinal"] = (
                residual_slope + _ACTIVE_GLOBAL_TRAJECTORY_SLOPE_V15
            )
            row["row_orientation_relative_leaf_axis_deg"] = math.degrees(
                math.atan(
                    float(row["row_common_slope_lateral_per_longitudinal"])
                )
            )
            row["row_tilt_slope_source"] = (
                "one_image_wide_residual_trajectory_angle"
            )
            row["trajectory_residual_slope_after_image_detrending"] = (
                residual_slope
            )
            row["trajectory_image_wide_residual_angle_deg"] = global_angle[
                "angle_deg"
            ]

        for diagnostic in audit.get("band_selection_diagnostics", {}).values():
            residual_angle = diagnostic.get("trajectory_slope_angle_deg")
            diagnostic["trajectory_residual_slope_angle_deg"] = residual_angle
            diagnostic["trajectory_image_wide_residual_angle_deg"] = (
                global_angle["angle_deg"]
            )
            diagnostic["trajectory_slope_angle_deg"] = math.degrees(
                math.atan(
                    math.tan(math.radians(float(residual_angle or 0.0)))
                    + _ACTIVE_GLOBAL_TRAJECTORY_SLOPE_V15
                )
            )
        for candidate in audit.get("candidate_models", []):
            residual_slope = float(
                candidate.get("slope_lateral_per_longitudinal") or 0.0
            )
            candidate["residual_slope_after_image_detrending"] = residual_slope
            candidate["image_wide_residual_angle_deg"] = global_angle[
                "angle_deg"
            ]
            candidate["slope_lateral_per_longitudinal"] = (
                residual_slope + _ACTIVE_GLOBAL_TRAJECTORY_SLOPE_V15
            )
        eligibility_audit = _remark_partial_band_eligibility_v15(
            row_records,
            band_records,
            point_assignment,
            stoma_traits,
            median_detector_width_um,
            _ACTIVE_GLOBAL_TRAJECTORY_SLOPE_V15,
            settings,
        )
        audit["band_spatial_eligibility"] = eligibility_audit
        audit["spatially_eligible_band_count"] = sum(
            bool(band.get("band_spatial_eligible")) for band in band_records
        )
        audit["spatially_eligible_row_count"] = sum(
            bool(row.get("row_band_spatial_eligible")) for row in row_records
        )
        audit["partial_band_accessible_span_basis"] = (
            "fitted_image_wide_trajectory_clipped_to_source_field"
        )
    finally:
        for row, lateral_value in zip(stoma_traits, original_lateral, strict=True):
            row["lateral_coordinate_um"] = lateral_value
        _ACTIVE_AXIS_V15 = None
        _ACTIVE_STOMA_TRAITS_V15 = None
        _ACTIVE_GLOBAL_TRAJECTORY_SLOPE_V15 = 0.0
        _ACTIVE_GLOBAL_TRAJECTORY_DETRENDING_V15 = False

    diagnostics = audit.get("band_selection_diagnostics", {})
    for band in band_records:
        provisional_id = str(
            band.get("provisional_band_id") or band.get("band_id")
        )
        diagnostic = diagnostics.get(provisional_id, {})
        band.update(_trajectory_diagnostic_fields(diagnostic))
        band["trajectory_exact_row_band_traits_eligible"] = bool(
            band.get("trajectory_exact_row_band_traits_eligible")
            and band.get("trajectory_exact_row_traits_eligible")
        )

    for row in row_records:
        provisional_id = str(
            row.get("provisional_band_id") or row.get("band_id")
        )
        diagnostic = diagnostics.get(provisional_id, {})
        row.update(_trajectory_diagnostic_fields(diagnostic))
        row["trajectory_exact_row_band_traits_eligible"] = bool(
            row.get("trajectory_exact_row_band_traits_eligible")
            and row.get("trajectory_exact_row_traits_eligible")
        )

    for assignment in point_assignment.values():
        provisional_id = str(
            assignment.get("provisional_band_id")
            or assignment.get("band_id")
        )
        diagnostic = diagnostics.get(provisional_id, {})
        assignment.update(_trajectory_diagnostic_fields(diagnostic))
        assignment["trajectory_exact_row_band_traits_eligible"] = bool(
            assignment.get("trajectory_exact_row_band_traits_eligible")
            and assignment.get("trajectory_exact_row_traits_eligible")
        )

    if row_records and stoma_traits:
        longitudinal_min = min(
            float(row["longitudinal_coordinate_um"]) for row in stoma_traits
        )
        longitudinal_max = max(
            float(row["longitudinal_coordinate_um"]) for row in stoma_traits
        )
        order_at_minimum = [
            str(row["row_id"])
            for row in sorted(
                row_records,
                key=lambda item: base.row_lateral_at(item, longitudinal_min),
            )
        ]
        order_at_maximum = [
            str(row["row_id"])
            for row in sorted(
                row_records,
                key=lambda item: base.row_lateral_at(item, longitudinal_max),
            )
        ]
        global_noncrossing = order_at_minimum == order_at_maximum
    else:
        order_at_minimum = []
        order_at_maximum = []
        global_noncrossing = True
    exact_structure_global_qc = bool(
        global_noncrossing and supergroup_sensitivity["stable"]
    )
    if not exact_structure_global_qc:
        for record in band_records + row_records:
            record["trajectory_exact_row_band_traits_eligible"] = False
            record["trajectory_global_crossing_detected"] = not global_noncrossing
            record["trajectory_supergroup_membership_stable"] = bool(
                supergroup_sensitivity["stable"]
            )
        for assignment in point_assignment.values():
            assignment["trajectory_exact_row_band_traits_eligible"] = False
            assignment["trajectory_global_crossing_detected"] = not global_noncrossing
            assignment["trajectory_supergroup_membership_stable"] = bool(
                supergroup_sensitivity["stable"]
            )
    else:
        for record in band_records + row_records:
            record["trajectory_global_crossing_detected"] = False
            record["trajectory_supergroup_membership_stable"] = True
        for assignment in point_assignment.values():
            assignment["trajectory_global_crossing_detected"] = False
            assignment["trajectory_supergroup_membership_stable"] = True

    audit.update(
        {
            "row_model_version": SCRIPT_VERSION,
            "trajectory_image_wide_residual_angle": global_angle,
            "trajectory_image_wide_orientation_enforced": True,
            "trajectory_global_noncrossing": global_noncrossing,
            "trajectory_noncrossing_constrained": True,
            "trajectory_supergroup_membership_sensitivity": (
                supergroup_sensitivity
            ),
            "trajectory_supergroup_membership_stable": bool(
                supergroup_sensitivity["stable"]
            ),
            "trajectory_row_order_at_longitudinal_minimum": order_at_minimum,
            "trajectory_row_order_at_longitudinal_maximum": order_at_maximum,
            "automatic_row_count_policy": (
                "estimate one bounded residual trajectory angle from the whole "
                "image; apply it to every stoma before forming broad lateral "
                "neighbourhoods; fit every valid one-to-ten non-crossing parallel-"
                "trajectory model; compare adjusted scores globally; retain the direct "
                "one-versus-two structural safeguard; permit a sparse third row "
                "only with detector and raw-image support"
            ),
            "band_partition_policy": (
                "infer rows before bands; measure adjacent-trajectory lateral "
                "distance at matched longitudinal positions; keep <=1.20 detector "
                "widths together; treat >=1.50 widths as strong boundaries; require "
                "local-gap and local target-versus-flank raw edge support in the "
                "intermediate zone; allow "
                "asymmetric and multiple partitions"
            ),
            "trajectory_stability_policy": (
                "within each fixed, image-wide-detrended supergroup, repeat trajectory "
                "inference after deterministic lateral jitter and minimum-separation "
                "perturbation; separately test the upstream supergroup gap threshold; "
                "separately perturb each matched "
                "lateral gap to test sensitivity to the band-decision thresholds; "
                "exact row/band GWAS phenotypes require stable row count, stable "
                "assignments and a stable gap-threshold decision"
            ),
            "trajectory_settings": {
                key: value
                for key, value in asdict(settings).items()
                if key.startswith(("trajectory_", "sparse_trajectory_", "band_"))
            },
            "trajectory_unstable_provisional_group_count": sum(
                not bool(item.get("trajectory_row_count_stable"))
                for item in diagnostics.values()
            ),
            "trajectory_sparse_three_row_rescue_group_count": sum(
                bool(item.get("sparse_trajectory_rescue_applied"))
                for item in diagnostics.values()
            ),
            "trajectory_maximum_k_reached_group_count": sum(
                bool(item.get("trajectory_maximum_k_reached"))
                for item in diagnostics.values()
            ),
            "trajectory_unresolved_sparse_third_group_count": sum(
                bool(item.get("trajectory_unresolved_sparse_third_trajectory"))
                for item in diagnostics.values()
            ),
            "trajectory_unvalidated_singleton_group_count": sum(
                bool(item.get("trajectory_singleton_validation_required"))
                and not bool(item.get("trajectory_singleton_rows_validated"))
                for item in diagnostics.values()
            ),
            "trajectory_unresolved_final_band_count": sum(
                bool(band.get("trajectory_unresolved_boundary_candidate"))
                for band in band_records
            ),
            "trajectory_exact_row_band_eligible_final_band_count": sum(
                bool(band.get("trajectory_exact_row_band_traits_eligible"))
                for band in band_records
            ),
        }
    )
    return row_records, band_records, point_assignment, audit


def image_trait_summary_v15(
    image_metadata: dict[str, Any],
    stoma_traits: list[dict[str, Any]],
    hair_rows: list[dict[str, Any]] | None,
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    axis: base.AxisTransform,
    spatial_probability: dict[str, Any],
    row_band_audit: dict[str, Any],
    width_um: float,
    height_um: float,
) -> dict[str, Any]:
    """Keep all measurements, while adding image-level trajectory eligibility."""

    result = v14.image_trait_summary_v14(
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
    eligible_bands = [
        band for band in band_records if bool(band.get("band_spatial_eligible"))
    ]
    provisional: dict[str, dict[str, Any]] = {}
    for band in band_records:
        provisional.setdefault(
            str(band.get("provisional_band_id") or band.get("band_id")), band
        )
    unstable_groups = [
        band
        for band in provisional.values()
        if not bool(band.get("trajectory_row_count_stable"))
    ]
    unresolved_bands = [
        band
        for band in eligible_bands
        if bool(band.get("trajectory_unresolved_boundary_candidate"))
    ]
    ineligible_trajectory_bands = [
        band
        for band in eligible_bands
        if not bool(band.get("trajectory_exact_row_band_traits_eligible"))
    ]
    unstable_eligible_groups = {
        str(band.get("provisional_band_id") or band.get("band_id"))
        for band in ineligible_trajectory_bands
        if not bool(band.get("trajectory_row_count_stable"))
    }
    ceiling_limited_eligible_groups = {
        str(band.get("provisional_band_id") or band.get("band_id"))
        for band in ineligible_trajectory_bands
        if bool(band.get("trajectory_maximum_k_reached"))
    }
    unresolved_sparse_eligible_groups = {
        str(band.get("provisional_band_id") or band.get("band_id"))
        for band in ineligible_trajectory_bands
        if bool(band.get("trajectory_unresolved_sparse_third_trajectory"))
    }
    unvalidated_singleton_eligible_groups = {
        str(band.get("provisional_band_id") or band.get("band_id"))
        for band in ineligible_trajectory_bands
        if bool(band.get("trajectory_singleton_validation_required"))
        and not bool(band.get("trajectory_singleton_rows_validated"))
    }
    exact_gwas_eligible = bool(
        eligible_bands and not ineligible_trajectory_bands
    )
    global_angle = row_band_audit.get(
        "trajectory_image_wide_residual_angle", {}
    )
    reasons: list[str] = []
    if not eligible_bands:
        reasons.append("no_spatially_complete_band")
    if unstable_eligible_groups:
        reasons.append("unstable_trajectory_row_count_or_assignment")
    if ceiling_limited_eligible_groups:
        reasons.append("trajectory_model_selected_configured_k_ceiling")
    if unresolved_sparse_eligible_groups:
        reasons.append("sparse_third_trajectory_lacks_local_raw_image_confirmation")
    if unvalidated_singleton_eligible_groups:
        reasons.append("singleton_supported_row_lacks_detector_or_local_edge_validation")
    if not bool(row_band_audit.get("trajectory_global_noncrossing", True)):
        reasons.append("fitted_trajectory_order_crosses_within_field")
    if not bool(
        row_band_audit.get("trajectory_supergroup_membership_stable", True)
    ):
        reasons.append("broad_trajectory_supergroup_membership_near_threshold")
    if unresolved_bands:
        reasons.append("unresolved_trajectory_band_boundary")
    if any(
        not bool(band.get("trajectory_rows_within_expected_band_range"))
        for band in eligible_bands
    ):
        reasons.append("final_band_outside_expected_one_to_five_row_range")

    result.update(
        {
            "row_model_version": SCRIPT_VERSION,
            "trajectory_image_wide_orientation_enforced": True,
            "trajectory_image_wide_residual_angle_deg": global_angle.get(
                "angle_deg"
            ),
            "trajectory_image_wide_angle_confidence": global_angle.get(
                "confidence"
            ),
            "trajectory_image_wide_angle_eligible_pair_count": global_angle.get(
                "eligible_pair_count"
            ),
            "trajectory_image_wide_angle_support_gain_over_zero": global_angle.get(
                "support_gain_over_zero"
            ),
            "trajectory_global_noncrossing": bool(
                row_band_audit.get("trajectory_global_noncrossing", True)
            ),
            "trajectory_noncrossing_constrained": True,
            "trajectory_supergroup_membership_stable": bool(
                row_band_audit.get(
                    "trajectory_supergroup_membership_stable", True
                )
            ),
            "trajectory_supergroup_minimum_gap_classification_agreement": (
                row_band_audit.get(
                    "trajectory_supergroup_membership_sensitivity", {}
                ).get("minimum_classification_agreement_fraction")
            ),
            "biological_band_partition_count": max(
                0, len(band_records) - len(provisional)
            ),
            "trajectory_provisional_group_count": len(provisional),
            "trajectory_unstable_provisional_group_count": len(unstable_groups),
            "trajectory_stable_provisional_group_fraction": base.safe_divide(
                float(len(provisional) - len(unstable_groups)),
                float(len(provisional)),
            ),
            "trajectory_unresolved_spatially_eligible_band_count": len(
                unresolved_bands
            ),
            "trajectory_ceiling_limited_spatially_eligible_group_count": len(
                ceiling_limited_eligible_groups
            ),
            "trajectory_unresolved_sparse_spatially_eligible_group_count": len(
                unresolved_sparse_eligible_groups
            ),
            "trajectory_unvalidated_singleton_spatially_eligible_group_count": len(
                unvalidated_singleton_eligible_groups
            ),
            "trajectory_ineligible_spatially_eligible_band_count": len(
                ineligible_trajectory_bands
            ),
            "trajectory_sparse_three_row_rescue_group_count": sum(
                bool(band.get("sparse_trajectory_rescue_applied"))
                for band in provisional.values()
            ),
            "trajectory_exact_row_band_traits_gwas_eligible": exact_gwas_eligible,
            "trajectory_row_band_gwas_exclusion_reason": ";".join(
                dict.fromkeys(reasons)
            ),
            "trajectory_row_band_trait_policy": (
                "detailed CSV retains experimental estimates; compact GWAS CSV "
                "blanks exact row/band phenotypes when any spatially eligible band "
                "has unstable row assignments, an unresolved band boundary, a "
                "ceiling-limited model, or an unconfirmed sparse third trajectory; "
                "density, morphometry and ordinary nearest-neighbour traits remain"
            ),
        }
    )
    return result


_ROW_BAND_GWAS_TRAITS = (
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
    "same_row_empirical_probability",
    "same_band_other_row_empirical_probability",
    "adjacent_band_empirical_probability",
    "other_band_empirical_probability",
    "within_row_spacing_um_mean",
    "within_band_interrow_spacing_um_mean",
    "between_band_spacing_um_mean",
    "band_width_um_mean",
    "nearest_same_row_distance_um_mean",
    "mean_two_sided_row_spacing_um_mean",
    "nearest_lateral_row_distance_um_mean",
    "nearest_other_row_same_band_distance_um_mean",
    "nearest_other_band_row_distance_um_mean",
    "stomata_per_row_mean",
    "stomata_per_band_mean",
    "row_continuity_mean",
    "row_orientation_relative_leaf_axis_deg_mean",
    "longitudinal_to_lateral_spacing_ratio",
    "neighbor_relationship_entropy",
    "neighbor_probability_longitudinal_lateral_anisotropy",
    "relative_longitudinal_spacing_mean",
    "relative_lateral_row_spacing_mean",
    "row_band_spatial_stomatal_density_per_mm2",
    "stomatal_band_density_complete_intervals_per_mm",
    "stomatal_row_density_complete_intervals_per_mm",
)


def image_gwas_trait_summary_v15(
    image_traits: dict[str, Any],
) -> dict[str, Any]:
    """Write trajectory QC and suppress only unstable row/band phenotypes."""

    summary = v14.image_gwas_trait_summary_v14(image_traits)
    for field in (
        "trajectory_provisional_group_count",
        "trajectory_unstable_provisional_group_count",
        "trajectory_stable_provisional_group_fraction",
        "trajectory_unresolved_spatially_eligible_band_count",
        "trajectory_ceiling_limited_spatially_eligible_group_count",
        "trajectory_unresolved_sparse_spatially_eligible_group_count",
        "trajectory_unvalidated_singleton_spatially_eligible_group_count",
        "trajectory_ineligible_spatially_eligible_band_count",
        "trajectory_sparse_three_row_rescue_group_count",
        "trajectory_exact_row_band_traits_gwas_eligible",
        "trajectory_row_band_gwas_exclusion_reason",
        "trajectory_row_band_trait_policy",
        "trajectory_image_wide_orientation_enforced",
        "trajectory_image_wide_residual_angle_deg",
        "trajectory_image_wide_angle_confidence",
        "trajectory_image_wide_angle_eligible_pair_count",
        "trajectory_image_wide_angle_support_gain_over_zero",
        "trajectory_global_noncrossing",
        "trajectory_noncrossing_constrained",
        "trajectory_supergroup_membership_stable",
        "trajectory_supergroup_minimum_gap_classification_agreement",
    ):
        summary[field] = image_traits.get(field)
    if not bool(
        image_traits.get("trajectory_exact_row_band_traits_gwas_eligible")
    ):
        for field in _ROW_BAND_GWAS_TRAITS:
            summary[field] = None
    return summary


def add_row_neighbors_v15(
    stoma_traits: list[dict[str, Any]],
    row_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
) -> None:
    """Propagate trajectory QC to every stoma, then retain v1.4 eligibility."""

    for index, assignment in point_assignment.items():
        trajectory_qc = {
            key: value
            for key, value in assignment.items()
            if key == "row_model_version"
            or key.startswith("trajectory_")
            or key.startswith("sparse_trajectory_")
        }
        stoma_traits[index].update(trajectory_qc)
    v14.add_row_neighbors_v14(stoma_traits, row_records, point_assignment)


def make_spatial_audit_overlay_v15(
    input_dir: Path,
    output_path: Path,
    image_metadata: dict[str, Any],
    stoma_traits: list[dict[str, Any]],
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    axis: base.AxisTransform,
    px_per_um_x: float,
    px_per_um_y: float,
) -> None:
    """Retain the established overlay and mark trajectory-uncertain bands."""

    v14.make_spatial_audit_overlay_v14(
        input_dir,
        output_path,
        image_metadata,
        stoma_traits,
        row_records,
        band_records,
        axis,
        px_per_um_x,
        px_per_um_y,
    )
    uncertain = [
        band
        for band in band_records
        if bool(band.get("band_spatial_eligible"))
        and not bool(band.get("trajectory_exact_row_band_traits_eligible"))
    ]
    if not uncertain:
        return
    image = base.cv2.imread(str(output_path), base.cv2.IMREAD_COLOR)
    if image is None:
        return
    orange = (0, 165, 255)
    for band in uncertain:
        members = list(band.get("_point_indices", []))
        for index in members:
            x = int(round(stoma_traits[index]["center_x_pixels"]))
            y = int(round(stoma_traits[index]["center_y_pixels"]))
            base.cv2.circle(image, (x, y), 15, orange, 2, base.cv2.LINE_AA)
        if members:
            x = int(
                round(
                    np.mean(
                        [stoma_traits[index]["center_x_pixels"] for index in members]
                    )
                )
            )
            y = max(
                24,
                int(
                    round(
                        min(
                            stoma_traits[index]["center_y_pixels"]
                            for index in members
                        )
                    )
                )
                - 12,
            )
            base.cv2.putText(
                image,
                f"{band['band_id']} TRAJECTORY QC",
                (max(5, x - 105), y),
                base.cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                orange,
                2,
                base.cv2.LINE_AA,
            )
    if not base.cv2.imwrite(
        str(output_path), image, [base.cv2.IMWRITE_PNG_COMPRESSION, 6]
    ):
        raise RuntimeError(f"Failed to update v1.5 audit overlay: {output_path}")


def process_image_v15(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Retain v1.4 raw-image context and add final v1.5 summary counts."""

    result = list(v14.process_image_v14(*args, **kwargs))
    image_traits = result[1]
    image_traits["row_model_version"] = SCRIPT_VERSION
    return tuple(result)


def install_v15_trajectory_selector() -> None:
    """Install v1.5 extension points without changing either parent file."""

    v14.install_v14_row_selector()
    v14.SCRIPT_VERSION = SCRIPT_VERSION
    v14._partition_provisional_bands = _partition_trajectory_bands_v15

    base.parse_arguments = parse_arguments_v15
    base.build_settings = build_settings_v15
    base.select_row_model = select_row_model_v15
    base.infer_rows_and_bands = infer_rows_and_bands_v15
    base.add_row_neighbors = add_row_neighbors_v15
    base.neighbor_relation = v14.neighbor_relation_v14
    base.image_trait_summary = image_trait_summary_v15
    base.make_spatial_audit_overlay = make_spatial_audit_overlay_v15
    base.process_image = process_image_v15
    base.image_gwas_trait_summary = image_gwas_trait_summary_v15
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base.__file__ = __file__


def run_trait_extraction() -> Path:
    """Run the separate v1.5 inspection candidate with inherited safeguards."""

    install_v15_trajectory_selector()
    return base.run_trait_extraction()


if __name__ == "__main__":
    run_trait_extraction()
