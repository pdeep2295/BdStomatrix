#!/usr/bin/env python3
"""Experimental Brachypodium trait extractor with improved row counting.

Version 1.4.2 is an inspection candidate.  It imports the locked version-1.3
trait extractor and revises row-model selection, biological band partitioning,
partial-field eligibility and their QC reporting. Calibration, masks,
non-row morphometry and output-directory safety remain those of version 1.3.
Generic density still uses every detected stoma.

Why this version exists
-----------------------
The version-1.3 row selector examines one to five parallel-row models but
accepts them sequentially.  If a three-row model is not sufficiently better
than a two-row model, selection stops and a strongly supported four-row model
is never considered.  Sparse rows supported by one real stoma are also
penalized heavily by that sequential decision.

Version 1.4 therefore:

1. fits every valid model from one to ``--max-rows-per-band``;
2. compares all models directly instead of stopping at an intermediate count;
3. permits one-stoma rows;
4. uses detector confidence as graded evidence, never as a hard singleton
   rejection threshold;
5. favours reasonably regular within-band row spacing without imposing one
   fixed absolute spacing across genotypes;
6. chooses the simpler model when adjusted scores are nearly tied unless a
   supported two-band row arrangement is the strict best model;
7. requires either a strong adjusted-score gain or reproducible stagger
   structure before changing a directly competing one-row band to two rows;
8. shares one image-wide orientation across every inferred row; and
9. writes the legacy count, alternative count, score margin, singleton support,
   biological band partitions and field-coverage decisions into the QC tables.

It additionally applies the biological rules established during overlay audit:

* one image-wide longitudinal orientation is shared by every complete,
  incomplete and singleton-supported row;
* an initially broad group of rows can be divided into two close biological
  bands when row-pair geometry, alternation and the raw image support a boundary;
* bands incompletely sampled at the field-of-view edge remain available for
  stomatal density and morphometry but are excluded from aggregated row/band
  spatial traits; and
* total detected and spatially eligible row/band counts are both retained.

This file does not modify version 1.3.  Use a new output directory whenever it
is tested.  The defaults below are deliberately provisional and must be
validated on biologically audited bands before production integration.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import extract_traits_from_pipeline_output as base


SCRIPT_VERSION = "1.4.2-experimental-global-rows-bands"


@dataclass(frozen=True)
class TraitSettingsV14(base.TraitSettings):
    """Version-1.3 settings plus transparent global row-selection settings."""

    # A singleton remains eligible.  This is a modest false-positive safeguard,
    # not a rule requiring two observations per row.
    global_row_singleton_penalty: float = 4.0

    # Penalize very unequal consecutive row gaps.  The penalty is based on the
    # coefficient of variation, so it does not impose one absolute spacing on
    # every genotype.
    global_row_spacing_cv_penalty_weight: float = 30.0

    # A gap wider than this multiple of median stomatal-box width can indicate
    # that an intervening row has been merged into a simpler model.
    global_row_large_gap_start_width_factor: float = 0.90
    global_row_large_gap_penalty_weight: float = 12.0

    # If several candidate scores are this close, retain the smaller row count
    # and report low confidence instead of choosing the most complex model.
    global_row_simpler_tie_tolerance: float = 2.0

    # Adjusted-score margins used only for QC confidence labels.
    global_row_moderate_confidence_margin: float = 2.0
    global_row_high_confidence_margin: float = 6.0

    # A direct one-row to two-row change must have either a strong adjusted-
    # score improvement or the alternating/phase-shifted arrangement expected
    # when two genuine stomatal rows share one biological band.
    direct_two_row_min_score_improvement: float = 6.0
    direct_two_row_min_alternation: float = 0.60
    direct_two_row_max_phase_offset_normalized: float = 0.75
    direct_two_row_min_longest_alternating_sequence: int = 3

    # A central gap can support partitioning a provisional four- or five-row
    # group into two close biological bands.  Both sides must contain at least
    # two inferred rows.  The bonus is applied during row-count selection so a
    # supported 2+2 arrangement is not lost to a nearly tied three-row model.
    band_partition_min_rows_per_side: int = 2
    band_partition_min_boundary_gap_width_factor: float = 0.55
    band_partition_min_boundary_to_internal_gap_ratio: float = 1.15
    band_partition_strong_geometry_gap_ratio: float = 1.20
    band_partition_min_within_group_alternation: float = 0.60
    band_partition_image_support_ratio: float = 1.05
    band_partition_row_model_bonus: float = 5.0

    # A partial band is excluded from row/band summaries only when both its
    # absolute and image-relative longitudinal coverage are low.  Proximity to
    # an image side strengthens the truncation decision. Detector boxes provide
    # the primary side-edge distance in source pixels; the rotated band-centre
    # distance is retained as auxiliary QC. These rules never remove stomata
    # from density or morphometry.
    partial_band_edge_margin_box_width_factor: float = 1.0
    partial_band_min_longitudinal_coverage_fraction: float = 0.45
    partial_band_min_relative_observed_span: float = 0.60


_BASE_PARSE_ARGUMENTS = base.parse_arguments
_BASE_BUILD_SETTINGS = base.build_settings
_BASE_SELECT_ROW_MODEL = base.select_row_model
_BASE_INFER_ROWS_AND_BANDS = base.infer_rows_and_bands
_BASE_PROCESS_IMAGE = base.process_image
_BASE_IMAGE_GWAS_TRAIT_SUMMARY = base.image_gwas_trait_summary
_BASE_IMAGE_TRAIT_SUMMARY = base.image_trait_summary
_BASE_MAKE_SPATIAL_AUDIT_OVERLAY = base.make_spatial_audit_overlay
_BASE_ADD_ROW_NEIGHBORS = base.add_row_neighbors
_BASE_NEIGHBOR_RELATION = base.neighbor_relation

# These variables exist only while one image is being processed.  The trait
# extractor is single-process at image level, so this deterministic context is
# safe and avoids changing the established version-1.3 function signatures.
_ACTIVE_DETECTOR_CONFIDENCES: np.ndarray | None = None
_ACTIVE_LONGITUDINAL_COORDINATES: np.ndarray | None = None
_ACTIVE_BAND_DIAGNOSTICS: list[dict[str, Any]] = []
_ACTIVE_SOURCE_GRAY: np.ndarray | None = None
_ACTIVE_HORIZONTAL_GRADIENT: np.ndarray | None = None
_ACTIVE_IMAGE_METADATA: dict[str, Any] | None = None
_ACTIVE_SETTINGS: TraitSettingsV14 | None = None


def parse_arguments_v14() -> argparse.Namespace:
    """Expose every version-1.3 option plus the new row-scoring controls."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract calibrated Brachypodium traits with experimental global "
            "one-to-five-row comparison and singleton-supported rows."
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
            "Supported for validation only; it is not required by the new "
            "automatic selector."
        ),
    )

    parser.add_argument(
        "--global-row-singleton-penalty",
        type=float,
        default=4.0,
        help=(
            "Modest adjusted-score penalty for a row supported by one stoma. "
            "Singleton rows remain eligible."
        ),
    )
    parser.add_argument(
        "--global-row-spacing-cv-penalty-weight",
        type=float,
        default=30.0,
        help="Weight for unequal consecutive within-band row spacing.",
    )
    parser.add_argument(
        "--global-row-large-gap-start-width-factor",
        type=float,
        default=0.90,
        help=(
            "Begin penalizing a candidate when its largest inter-row gap "
            "exceeds this multiple of median stomatal-box width."
        ),
    )
    parser.add_argument(
        "--global-row-large-gap-penalty-weight",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--global-row-simpler-tie-tolerance",
        type=float,
        default=2.0,
        help="Choose the smaller row count when scores differ by no more than this.",
    )
    parser.add_argument(
        "--global-row-moderate-confidence-margin",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--global-row-high-confidence-margin",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--direct-two-row-min-score-improvement",
        type=float,
        default=6.0,
        help=(
            "Accept a direct one-row to two-row change from score evidence "
            "alone only when the adjusted score improves by at least this value."
        ),
    )
    parser.add_argument(
        "--direct-two-row-min-alternation",
        type=float,
        default=0.60,
        help="Minimum successive-stoma row alternation for structural support.",
    )
    parser.add_argument(
        "--direct-two-row-max-phase-offset-normalized",
        type=float,
        default=0.75,
        help=(
            "Maximum median cross-row phase offset divided by typical "
            "within-row spacing for structural support."
        ),
    )
    parser.add_argument(
        "--direct-two-row-min-longest-alternating-sequence",
        type=int,
        default=3,
        help="Minimum longest alternating run for structural two-row support.",
    )
    parser.add_argument("--band-partition-min-rows-per-side", type=int, default=2)
    parser.add_argument(
        "--band-partition-min-boundary-gap-width-factor", type=float, default=0.55
    )
    parser.add_argument(
        "--band-partition-min-boundary-to-internal-gap-ratio",
        type=float,
        default=1.15,
    )
    parser.add_argument(
        "--band-partition-strong-geometry-gap-ratio", type=float, default=1.20
    )
    parser.add_argument(
        "--band-partition-min-within-group-alternation", type=float, default=0.60
    )
    parser.add_argument(
        "--band-partition-image-support-ratio", type=float, default=1.05
    )
    parser.add_argument("--band-partition-row-model-bonus", type=float, default=5.0)
    parser.add_argument(
        "--partial-band-edge-margin-box-width-factor", type=float, default=1.0
    )
    parser.add_argument(
        "--partial-band-min-longitudinal-coverage-fraction",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--partial-band-min-relative-observed-span", type=float, default=0.60
    )
    return parser.parse_args()


def build_settings_v14(args: argparse.Namespace) -> TraitSettingsV14:
    """Validate old settings with v1.3 and add the experimental settings."""

    old = _BASE_BUILD_SETTINGS(args)
    if old.minimum_stomata_per_resolved_row != 1:
        raise ValueError(
            "Version 1.4 requires --min-stomata-per-row 1 because a genuine "
            "biological row may be supported by only one detected stoma."
        )
    new_values = {
        "global_row_singleton_penalty": args.global_row_singleton_penalty,
        "global_row_spacing_cv_penalty_weight": (
            args.global_row_spacing_cv_penalty_weight
        ),
        "global_row_large_gap_start_width_factor": (
            args.global_row_large_gap_start_width_factor
        ),
        "global_row_large_gap_penalty_weight": (
            args.global_row_large_gap_penalty_weight
        ),
        "global_row_simpler_tie_tolerance": (
            args.global_row_simpler_tie_tolerance
        ),
        "global_row_moderate_confidence_margin": (
            args.global_row_moderate_confidence_margin
        ),
        "global_row_high_confidence_margin": (
            args.global_row_high_confidence_margin
        ),
        "direct_two_row_min_score_improvement": (
            args.direct_two_row_min_score_improvement
        ),
        "direct_two_row_min_alternation": args.direct_two_row_min_alternation,
        "direct_two_row_max_phase_offset_normalized": (
            args.direct_two_row_max_phase_offset_normalized
        ),
        "direct_two_row_min_longest_alternating_sequence": (
            args.direct_two_row_min_longest_alternating_sequence
        ),
        "band_partition_min_rows_per_side": args.band_partition_min_rows_per_side,
        "band_partition_min_boundary_gap_width_factor": (
            args.band_partition_min_boundary_gap_width_factor
        ),
        "band_partition_min_boundary_to_internal_gap_ratio": (
            args.band_partition_min_boundary_to_internal_gap_ratio
        ),
        "band_partition_strong_geometry_gap_ratio": (
            args.band_partition_strong_geometry_gap_ratio
        ),
        "band_partition_min_within_group_alternation": (
            args.band_partition_min_within_group_alternation
        ),
        "band_partition_image_support_ratio": args.band_partition_image_support_ratio,
        "band_partition_row_model_bonus": args.band_partition_row_model_bonus,
        "partial_band_edge_margin_box_width_factor": (
            args.partial_band_edge_margin_box_width_factor
        ),
        "partial_band_min_longitudinal_coverage_fraction": (
            args.partial_band_min_longitudinal_coverage_fraction
        ),
        "partial_band_min_relative_observed_span": (
            args.partial_band_min_relative_observed_span
        ),
    }
    for name, value in new_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite; received {value!r}.")
    for name in (
        "global_row_singleton_penalty",
        "global_row_spacing_cv_penalty_weight",
        "global_row_large_gap_penalty_weight",
        "global_row_simpler_tie_tolerance",
        "global_row_moderate_confidence_margin",
        "global_row_high_confidence_margin",
        "direct_two_row_min_score_improvement",
        "direct_two_row_min_alternation",
        "direct_two_row_max_phase_offset_normalized",
        "band_partition_min_boundary_gap_width_factor",
        "band_partition_min_boundary_to_internal_gap_ratio",
        "band_partition_strong_geometry_gap_ratio",
        "band_partition_min_within_group_alternation",
        "band_partition_image_support_ratio",
        "band_partition_row_model_bonus",
        "partial_band_edge_margin_box_width_factor",
        "partial_band_min_longitudinal_coverage_fraction",
        "partial_band_min_relative_observed_span",
    ):
        if new_values[name] < 0:
            raise ValueError(f"{name} cannot be negative.")
    if new_values["global_row_large_gap_start_width_factor"] <= 0:
        raise ValueError(
            "global_row_large_gap_start_width_factor must be positive."
        )
    if new_values["band_partition_min_rows_per_side"] < 2:
        raise ValueError("band_partition_min_rows_per_side must be at least 2.")
    if new_values["direct_two_row_min_longest_alternating_sequence"] < 2:
        raise ValueError(
            "direct_two_row_min_longest_alternating_sequence must be at least 2."
        )
    for name in (
        "direct_two_row_min_alternation",
        "band_partition_min_within_group_alternation",
        "partial_band_min_longitudinal_coverage_fraction",
        "partial_band_min_relative_observed_span",
    ):
        if not 0.0 <= new_values[name] <= 1.0:
            raise ValueError(f"{name} must lie between 0 and 1.")
    if (
        new_values["global_row_high_confidence_margin"]
        < new_values["global_row_moderate_confidence_margin"]
    ):
        raise ValueError(
            "The high-confidence margin cannot be smaller than the "
            "moderate-confidence margin."
        )
    return TraitSettingsV14(**asdict(old), **new_values)


def _singleton_detector_confidences(candidate: dict[str, Any]) -> list[float]:
    """Return confidence values only for candidate rows containing one stoma."""

    labels = np.asarray(candidate["labels"], dtype=int)
    point_indices = list(candidate["point_indices"])
    confidences: list[float] = []
    for group, size in enumerate(candidate["cluster_sizes"]):
        if size != 1:
            continue
        local_positions = np.flatnonzero(labels == group)
        if len(local_positions) != 1:
            continue
        point_index = point_indices[int(local_positions[0])]
        if (
            _ACTIVE_DETECTOR_CONFIDENCES is not None
            and 0 <= point_index < len(_ACTIVE_DETECTOR_CONFIDENCES)
        ):
            confidence = float(_ACTIVE_DETECTOR_CONFIDENCES[point_index])
        else:
            confidence = 0.5
        confidences.append(min(max(confidence, 0.0), 1.0))
    return confidences


def score_global_row_candidate(
    candidate: dict[str, Any],
    median_detector_width_um: float,
    settings: TraitSettingsV14,
) -> dict[str, Any]:
    """Calculate a transparent adjusted score; smaller values are preferred."""

    intercepts = np.asarray(candidate["intercepts_um"], dtype=float)
    gaps = np.diff(intercepts)
    if len(gaps) >= 2 and float(np.mean(gaps)) > 0:
        spacing_cv = float(np.std(gaps, ddof=0) / np.mean(gaps))
    else:
        spacing_cv = 0.0
    spacing_penalty = (
        settings.global_row_spacing_cv_penalty_weight * spacing_cv
    )

    width = max(float(median_detector_width_um), 1e-9)
    maximum_gap_width_factor = (
        float(np.max(gaps)) / width if len(gaps) else 0.0
    )
    large_gap_excess = max(
        0.0,
        maximum_gap_width_factor
        - settings.global_row_large_gap_start_width_factor,
    )
    large_gap_penalty = (
        settings.global_row_large_gap_penalty_weight * large_gap_excess**2
    )

    singleton_confidences = _singleton_detector_confidences(candidate)
    # Higher detector confidence reduces, but never removes, the safeguard.
    singleton_penalty = sum(
        settings.global_row_singleton_penalty * (1.25 - 0.50 * confidence)
        for confidence in singleton_confidences
    )

    partition_split: int | None = None
    partition_gap_ratio: float | None = None
    partition_gap_width_factor: float | None = None
    partition_left_alternation: float | None = None
    partition_right_alternation: float | None = None
    labels = np.asarray(candidate["labels"], dtype=int)
    candidate_points = list(candidate["point_indices"])

    def candidate_alternation(first_group: int, last_group: int) -> float | None:
        local_positions = [
            position
            for position, label in enumerate(labels)
            if first_group <= int(label) < last_group
        ]
        if len(local_positions) < 2 or last_group - first_group < 2:
            return None
        if _ACTIVE_LONGITUDINAL_COORDINATES is not None:
            local_positions.sort(
                key=lambda position: _ACTIVE_LONGITUDINAL_COORDINATES[
                    candidate_points[position]
                ]
            )
        sequence = [int(labels[position]) for position in local_positions]
        return sum(
            first != second for first, second in zip(sequence[:-1], sequence[1:])
        ) / (len(sequence) - 1)

    if int(candidate["k"]) >= 2 * settings.band_partition_min_rows_per_side:
        for split in range(
            settings.band_partition_min_rows_per_side,
            int(candidate["k"]) - settings.band_partition_min_rows_per_side + 1,
        ):
            boundary_gap = float(gaps[split - 1])
            internal_gaps = np.delete(gaps, split - 1)
            if not len(internal_gaps):
                continue
            internal_reference = float(np.median(internal_gaps))
            ratio = boundary_gap / max(internal_reference, 1e-9)
            width_factor = boundary_gap / width
            left_alternation = candidate_alternation(0, split)
            right_alternation = candidate_alternation(split, int(candidate["k"]))
            if (
                width_factor
                >= settings.band_partition_min_boundary_gap_width_factor
                and ratio
                >= settings.band_partition_min_boundary_to_internal_gap_ratio
                and left_alternation is not None
                and right_alternation is not None
                and left_alternation
                >= settings.band_partition_min_within_group_alternation
                and right_alternation
                >= settings.band_partition_min_within_group_alternation
                and (
                    partition_gap_ratio is None
                    or ratio > partition_gap_ratio
                )
            ):
                partition_split = split
                partition_gap_ratio = ratio
                partition_gap_width_factor = width_factor
                partition_left_alternation = left_alternation
                partition_right_alternation = right_alternation
    partition_bonus = (
        settings.band_partition_row_model_bonus
        if partition_split is not None
        else 0.0
    )

    adjusted_score = (
        float(candidate["bic"])
        + spacing_penalty
        + large_gap_penalty
        + singleton_penalty
        - partition_bonus
    )
    return {
        "candidate_row_count": int(candidate["k"]),
        "adjusted_score": adjusted_score,
        "original_bic": float(candidate["bic"]),
        "spacing_cv": spacing_cv,
        "spacing_penalty": spacing_penalty,
        "maximum_gap_width_factor": maximum_gap_width_factor,
        "large_gap_penalty": large_gap_penalty,
        "singleton_row_count": len(singleton_confidences),
        "singleton_detector_confidences": singleton_confidences,
        "singleton_detector_confidence_mean": (
            float(np.mean(singleton_confidences))
            if singleton_confidences
            else None
        ),
        "singleton_penalty": singleton_penalty,
        "pair_partition_candidate": partition_split is not None,
        "pair_partition_split_after_row": partition_split,
        "pair_partition_boundary_gap_ratio": partition_gap_ratio,
        "pair_partition_boundary_gap_width_factor": partition_gap_width_factor,
        "pair_partition_left_alternation": partition_left_alternation,
        "pair_partition_right_alternation": partition_right_alternation,
        "pair_partition_score_bonus": partition_bonus,
        "cluster_sizes": list(candidate["cluster_sizes"]),
        "interrow_gaps_um": gaps.tolist(),
    }


def _direct_two_row_structural_evidence(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Describe whether a fitted two-row model has biological stagger structure."""

    labels = np.asarray(candidate["labels"], dtype=int)
    point_indices = list(candidate["point_indices"])
    evidence: dict[str, Any] = {
        "direct_two_row_alternation": None,
        "direct_two_row_phase_offset_normalized": None,
        "direct_two_row_longest_alternating_sequence": None,
    }
    if (
        int(candidate["k"]) != 2
        or len(labels) < 2
        or _ACTIVE_LONGITUDINAL_COORDINATES is None
    ):
        return evidence

    longitudinal = np.asarray(
        [_ACTIVE_LONGITUDINAL_COORDINATES[index] for index in point_indices],
        dtype=float,
    )
    order = np.argsort(longitudinal, kind="stable")
    sequence = labels[order].tolist()
    evidence["direct_two_row_alternation"] = float(
        sum(first != second for first, second in zip(sequence[:-1], sequence[1:]))
        / (len(sequence) - 1)
    )

    longest = current = 1
    for first, second in zip(sequence[:-1], sequence[1:]):
        current = current + 1 if first != second else 1
        longest = max(longest, current)
    evidence["direct_two_row_longest_alternating_sequence"] = int(longest)

    row_coordinates = [
        np.sort(longitudinal[labels == group]) for group in (0, 1)
    ]
    pooled_within_row_gaps = np.concatenate(
        [np.diff(values) for values in row_coordinates if len(values) >= 2]
    ) if any(len(values) >= 2 for values in row_coordinates) else np.asarray([])
    if (
        len(pooled_within_row_gaps)
        and len(row_coordinates[0])
        and len(row_coordinates[1])
    ):
        typical_spacing = float(np.median(pooled_within_row_gaps))
        if typical_spacing > 0:
            first, second = row_coordinates
            offsets = [float(np.min(np.abs(second - value))) for value in first]
            offsets.extend(float(np.min(np.abs(first - value))) for value in second)
            evidence["direct_two_row_phase_offset_normalized"] = float(
                np.median(offsets) / typical_spacing
            )
    return evidence


def select_row_model_v14(
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV14,
    audited_row_count: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select globally among all valid row counts, including singleton rows."""

    # Retain the v1.3 sequential answer only as an explicit comparison field.
    # New v1.4 candidates are then fitted with zero residual slope because the
    # centres have already been rotated into the one image-wide leaf axis.
    legacy, _legacy_candidates = _BASE_SELECT_ROW_MODEL(
        point_indices,
        lateral,
        longitudinal,
        median_detector_width_um,
        settings,
        audited_row_count=None,
    )
    minimum_separation = max(
        settings.minimum_row_separation_um,
        settings.row_separation_detector_width_factor
        * median_detector_width_um,
    )
    maximum_k = min(settings.maximum_rows_per_band, len(point_indices))
    residual_floor_um = (
        settings.row_model_residual_floor_width_factor
        * median_detector_width_um
    )
    candidates: list[dict[str, Any]] = []
    for k in range(1, maximum_k + 1):
        candidate = base.fit_parallel_row_model(
            point_indices,
            lateral,
            longitudinal,
            k,
            minimum_points=(
                1 if k == 1 else settings.minimum_stomata_per_resolved_row
            ),
            minimum_separation_um=minimum_separation,
            fixed_slope=0.0,
            residual_floor_um=residual_floor_um,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        raise RuntimeError("No valid row candidate was produced.")

    diagnostics = [
        score_global_row_candidate(candidate, median_detector_width_um, settings)
        for candidate in candidates
    ]
    by_k = {int(candidate["k"]): candidate for candidate in candidates}
    diagnostic_by_k = {
        int(item["candidate_row_count"]): item for item in diagnostics
    }

    strict_best = min(
        diagnostics,
        key=lambda item: (item["adjusted_score"], item["candidate_row_count"]),
    )
    strict_best_score = float(strict_best["adjusted_score"])
    tied_counts = [
        item["candidate_row_count"]
        for item in diagnostics
        if item["adjusted_score"]
        <= strict_best_score + settings.global_row_simpler_tie_tolerance
    ]
    # A strong 2+2-style configuration is a biological boundary hypothesis,
    # not gratuitous model complexity, so it is not simplified away merely
    # because a lower-k model falls inside the generic tie window.
    automatic_k = int(
        strict_best["candidate_row_count"]
        if strict_best["pair_partition_candidate"]
        else min(tied_counts)
    )

    # A small residual improvement can divide a visibly single, slightly tilted
    # row into two parallel tracks.  When the global competition is specifically
    # one versus two rows, demand either a large adjusted-score improvement or
    # the staggered longitudinal structure expected for two biological rows.
    direct_two_row_evaluated = bool(automatic_k == 2 and 1 in by_k)
    direct_two_row_validated = False
    direct_two_row_rejected = False
    direct_two_row_score_improvement: float | None = None
    direct_two_row_strong_score_support = False
    direct_two_row_structural_support = False
    direct_two_row_evidence = {
        "direct_two_row_alternation": None,
        "direct_two_row_phase_offset_normalized": None,
        "direct_two_row_longest_alternating_sequence": None,
    }
    if direct_two_row_evaluated:
        direct_two_row_evidence = _direct_two_row_structural_evidence(by_k[2])
        direct_two_row_score_improvement = float(
            diagnostic_by_k[1]["adjusted_score"]
            - diagnostic_by_k[2]["adjusted_score"]
        )
        direct_two_row_strong_score_support = bool(
            direct_two_row_score_improvement
            >= settings.direct_two_row_min_score_improvement
        )
        alternation = direct_two_row_evidence["direct_two_row_alternation"]
        phase = direct_two_row_evidence[
            "direct_two_row_phase_offset_normalized"
        ]
        longest = direct_two_row_evidence[
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
        diagnostic_by_k[2].update(
            {
                **direct_two_row_evidence,
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

    automatic = dict(by_k[automatic_k])
    automatic_diagnostic = diagnostic_by_k[automatic_k]

    ranked_alternatives = sorted(
        (
            item
            for item in diagnostics
            if int(item["candidate_row_count"]) != automatic_k
        ),
        key=lambda item: (item["adjusted_score"], item["candidate_row_count"]),
    )
    alternative = ranked_alternatives[0] if ranked_alternatives else None
    raw_margin = (
        float(alternative["adjusted_score"])
        - float(automatic_diagnostic["adjusted_score"])
        if alternative is not None
        else math.inf
    )
    confidence_margin = max(0.0, raw_margin)
    automatic_is_strict_best = bool(
        math.isclose(
            automatic_diagnostic["adjusted_score"],
            strict_best_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )

    singleton_count = int(automatic_diagnostic["singleton_row_count"])
    if (
        automatic_is_strict_best
        and confidence_margin >= settings.global_row_high_confidence_margin
        and singleton_count == 0
    ):
        automatic_confidence = "high"
    elif (
        automatic_is_strict_best
        and confidence_margin >= settings.global_row_moderate_confidence_margin
    ):
        automatic_confidence = "moderate"
    else:
        automatic_confidence = "low"
    # A singleton-supported result can be useful and selected, but it should
    # never be labelled high confidence from centre geometry alone.
    if singleton_count and automatic_confidence == "high":
        automatic_confidence = "moderate"

    automatic_reason = (
        "global_all_k_geometry_singleton_supported"
        if singleton_count
        else "global_all_k_geometry_supported"
    )
    if direct_two_row_rejected:
        automatic_reason += "__two_row_rejected_insufficient_stagger_support"
    elif direct_two_row_validated:
        automatic_reason += "__two_row_validated"
    automatic_reason += f"__legacy_sequential_k{int(legacy['k'])}"

    chosen = automatic
    selection_reason = automatic_reason
    if audited_row_count is not None:
        if audited_row_count not in by_k:
            raise ValueError(
                f"Audited row_count={audited_row_count} has no valid candidate "
                f"model for a band containing {len(point_indices)} stomata."
            )
        chosen = dict(by_k[audited_row_count])
        selection_reason = "audited_row_count_override"

    chosen_k = int(chosen["k"])
    chosen_diagnostic = diagnostic_by_k[chosen_k]
    simpler = by_k.get(chosen_k - 1)
    chosen["delta_bic_vs_simpler"] = (
        float(simpler["bic"] - chosen["bic"])
        if simpler is not None
        else None
    )

    one_score = diagnostic_by_k.get(1, {}).get("adjusted_score")
    two_score = diagnostic_by_k.get(2, {}).get("adjusted_score")
    one_vs_two_ambiguous = bool(
        one_score is not None
        and two_score is not None
        and (
            abs(float(one_score) - float(two_score))
            <= settings.global_row_simpler_tie_tolerance
            or direct_two_row_rejected
        )
    )

    chosen["automatic_row_count"] = automatic_k
    chosen["automatic_selection_reason"] = automatic_reason
    chosen["automatic_one_vs_two_ambiguous"] = one_vs_two_ambiguous
    chosen["audited_row_count_override"] = audited_row_count is not None
    chosen["row_count_selection_reason"] = selection_reason
    chosen["tilt_slope_source"] = "image_axis_fixed_zero_after_rotation"
    chosen["one_row_rms_residual_um"] = float(candidates[0]["rms_residual_um"])
    chosen["simple_fit_stop_tolerance_um"] = (
        settings.row_model_simple_fit_stop_width_factor
        * median_detector_width_um
    )
    chosen["one_vs_two_ambiguous"] = one_vs_two_ambiguous
    chosen["sparse_row_support"] = any(
        int(size) < 3 for size in chosen["cluster_sizes"]
    )
    chosen["model_confidence"] = (
        "audited" if audited_row_count is not None else automatic_confidence
    )
    chosen.update(
        {
            "direct_two_row_candidate_evaluated": direct_two_row_evaluated,
            "direct_two_row_validated": direct_two_row_validated,
            "direct_two_row_rejected_to_one_row": direct_two_row_rejected,
            "direct_two_row_score_improvement_vs_one": (
                direct_two_row_score_improvement
            ),
            "direct_two_row_strong_score_support": (
                direct_two_row_strong_score_support
            ),
            "direct_two_row_structural_support": (
                direct_two_row_structural_support
            ),
            **direct_two_row_evidence,
        }
    )

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
            else "global_automatic"
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
        "global_row_automatic_is_strict_best": automatic_is_strict_best,
        "global_row_model_confidence": automatic_confidence,
        "singleton_supported_row_count": singleton_count,
        "singleton_detector_confidence_mean": automatic_diagnostic[
            "singleton_detector_confidence_mean"
        ],
        "global_row_count_changed_from_legacy": automatic_k != int(legacy["k"]),
        "potential_row_undersplit_in_legacy": automatic_k > int(legacy["k"]),
        "potential_row_oversplit_in_legacy": automatic_k < int(legacy["k"]),
        "direct_two_row_candidate_evaluated": direct_two_row_evaluated,
        "direct_two_row_validated": direct_two_row_validated,
        "direct_two_row_rejected_to_one_row": direct_two_row_rejected,
        "direct_two_row_score_improvement_vs_one": (
            direct_two_row_score_improvement
        ),
        "direct_two_row_strong_score_support": direct_two_row_strong_score_support,
        "direct_two_row_structural_support": direct_two_row_structural_support,
        **direct_two_row_evidence,
        "candidate_diagnostics": diagnostics,
    }
    _ACTIVE_BAND_DIAGNOSTICS.append(diagnostic_record)
    return chosen, candidates


def _row_group_alternation(
    rows: list[dict[str, Any]], stoma_traits: list[dict[str, Any]]
) -> float | None:
    """Fraction of successive longitudinal observations that switch rows."""

    observations = sorted(
        (
            (float(stoma_traits[index]["longitudinal_coordinate_um"]), row["row_id"])
            for row in rows
            for index in row["_point_indices"]
        ),
        key=lambda item: item[0],
    )
    if len(observations) < 2 or len(rows) < 2:
        return None
    changes = sum(
        first[1] != second[1]
        for first, second in zip(observations[:-1], observations[1:])
    )
    return changes / (len(observations) - 1)


def _active_axis(stoma_traits: list[dict[str, Any]]) -> base.AxisTransform | None:
    """Reconstruct the image-wide axis already estimated by version 1.3."""

    if not stoma_traits or _ACTIVE_IMAGE_METADATA is None:
        return None
    px_x = base.optional_float(_ACTIVE_IMAGE_METADATA.get("metadata_pixels_per_um_x"))
    px_y = base.optional_float(_ACTIVE_IMAGE_METADATA.get("metadata_pixels_per_um_y"))
    width = base.optional_int(_ACTIVE_IMAGE_METADATA.get("image_width"))
    height = base.optional_int(_ACTIVE_IMAGE_METADATA.get("image_height"))
    if px_x is None or px_y is None or width is None or height is None:
        return None
    angle = base.optional_float(
        stoma_traits[0].get("longitudinal_axis_angle_from_vertical_deg")
    )
    if angle is None:
        return None
    return base.AxisTransform(
        angle_from_vertical_deg=angle,
        origin_x_um=(width / px_x) / 2.0,
        origin_y_um=(height / px_y) / 2.0,
        source="reconstructed_image_axis_for_v1_4_QC",
        confidence="not_reestimated",
        candidate_count=0,
        angular_mad_deg=None,
        warning="",
    )


def _line_gradient_score(
    lateral_um: float,
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
) -> float | None:
    """Median raw-image edge evidence along one image-axis-parallel line."""

    if _ACTIVE_HORIZONTAL_GRADIENT is None or _ACTIVE_IMAGE_METADATA is None:
        return None
    axis = _active_axis(stoma_traits)
    if axis is None:
        return None
    px_x = base.optional_float(_ACTIVE_IMAGE_METADATA.get("metadata_pixels_per_um_x"))
    px_y = base.optional_float(_ACTIVE_IMAGE_METADATA.get("metadata_pixels_per_um_y"))
    if px_x is None or px_y is None:
        return None
    height, width = _ACTIVE_HORIZONTAL_GRADIENT.shape[:2]
    corners = [
        axis.forward(x, y)
        for x, y in (
            (0.0, 0.0),
            (width / px_x, 0.0),
            (0.0, height / px_y),
            (width / px_x, height / px_y),
        )
    ]
    long_min = min(value[1] for value in corners)
    long_max = max(value[1] for value in corners)
    lateral_offsets = np.linspace(
        -0.15 * median_detector_width_um,
        0.15 * median_detector_width_um,
        5,
    )
    samples: list[float] = []
    for longitudinal_um in np.linspace(long_min, long_max, 240):
        local: list[float] = []
        for offset in lateral_offsets:
            x_um, y_um = axis.inverse(lateral_um + float(offset), float(longitudinal_um))
            x = int(round(x_um * px_x))
            y = int(round(y_um * px_y))
            if 0 <= x < width and 0 <= y < height:
                local.append(float(_ACTIVE_HORIZONTAL_GRADIENT[y, x]))
        if local:
            samples.append(max(local))
    return float(np.median(samples)) if samples else None


def _rebuild_band_record(
    original: dict[str, Any],
    rows: list[dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    partition: dict[str, Any] | None,
    group_index: int,
) -> dict[str, Any]:
    """Recalculate a band record after an accepted biological partition."""

    rows = sorted(rows, key=lambda row: base.row_lateral_at(row, 0.0))
    members = sorted(
        {index for row in rows for index in row["_point_indices"]}
    )
    centers = np.asarray([base.row_lateral_at(row, 0.0) for row in rows], dtype=float)
    gaps = np.diff(centers)
    alternation = _row_group_alternation(rows, stoma_traits)
    ordered_row_ids = [
        row_id
        for _, row_id in sorted(
            (
                (float(stoma_traits[index]["longitudinal_coordinate_um"]), row["row_id"])
                for row in rows
                for index in row["_point_indices"]
            ),
            key=lambda item: item[0],
        )
    ]
    longest_alternating: int | None = None
    if len(rows) == 2 and ordered_row_ids:
        longest_alternating = current = 1
        for first, second in zip(ordered_row_ids[:-1], ordered_row_ids[1:]):
            current = current + 1 if first != second else 1
            longest_alternating = max(longest_alternating, current)

    pooled_gaps: list[float] = []
    for row in rows:
        values = np.sort(
            [
                stoma_traits[index]["longitudinal_coordinate_um"]
                for index in row["_point_indices"]
            ]
        )
        pooled_gaps.extend(np.diff(values).tolist())
    pooled_reference = float(np.median(pooled_gaps)) if pooled_gaps else None
    phase_offsets: list[float] = []
    if len(rows) > 1 and pooled_reference and pooled_reference > 0:
        for first, second in zip(rows[:-1], rows[1:]):
            first_y = np.asarray(
                [stoma_traits[index]["longitudinal_coordinate_um"] for index in first["_point_indices"]],
                dtype=float,
            )
            second_y = np.asarray(
                [stoma_traits[index]["longitudinal_coordinate_um"] for index in second["_point_indices"]],
                dtype=float,
            )
            offsets = [float(np.min(np.abs(second_y - value))) for value in first_y]
            offsets.extend(float(np.min(np.abs(first_y - value))) for value in second_y)
            if offsets:
                phase_offsets.append(float(np.median(offsets)) / pooled_reference)

    record = dict(original)
    record.update(
        {
            "provisional_band_id": original["band_id"],
            "band_partition_applied": partition is not None,
            "band_partition_group_index": group_index if partition is not None else None,
            "band_partition_split_after_row": (
                partition["split"] if partition is not None else None
            ),
            "band_partition_boundary_gap_um": (
                partition["boundary_gap_um"] if partition is not None else None
            ),
            "band_partition_boundary_to_internal_gap_ratio": (
                partition["gap_ratio"] if partition is not None else None
            ),
            "band_partition_boundary_gap_width_factor": (
                partition["gap_width_factor"] if partition is not None else None
            ),
            "band_partition_left_alternation": (
                partition["left_alternation"] if partition is not None else None
            ),
            "band_partition_right_alternation": (
                partition["right_alternation"] if partition is not None else None
            ),
            "band_partition_image_support_ratio": (
                partition["image_support_ratio"] if partition is not None else None
            ),
            "stoma_count": len(members),
            "row_count": len(rows),
            "automatic_row_count": len(rows),
            "row_count_used_for_traits": len(rows),
            "row_ids": [row["row_id"] for row in rows],
            "band_center_lateral_um": float(np.mean(centers)),
            "band_row_center_span_um": float(np.ptp(centers)) if len(centers) else 0.0,
            "band_width_um": (
                float(np.ptp(centers)) + median_detector_width_um
                if len(centers)
                else median_detector_width_um
            ),
            "within_band_interrow_spacing_mean_um": (
                float(np.mean(gaps)) if len(gaps) else None
            ),
            "within_band_interrow_spacing_median_um": (
                float(np.median(gaps)) if len(gaps) else None
            ),
            "successive_stoma_row_change_fraction": alternation,
            "two_row_alternation_index": alternation if len(rows) == 2 else None,
            "two_row_longest_alternating_sequence": longest_alternating,
            "adjacent_row_phase_offset_normalized_mean": (
                float(np.mean(phase_offsets)) if phase_offsets else None
            ),
            "sparse_row_support": any(int(row["stoma_count"]) < 3 for row in rows),
            "singleton_supported_row_count": sum(
                int(row["stoma_count"]) == 1 for row in rows
            ),
            "member_stoma_ids": [stoma_traits[index]["stoma_id"] for index in members],
            "_point_indices": members,
        }
    )
    record.update(base.sample_summary(gaps, "within_band_interrow_spacing_um"))
    return record


def _partition_provisional_bands(
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    settings: TraitSettingsV14,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Split a broad provisional group only when two-band evidence agrees."""

    row_lookup = {str(row["row_id"]): row for row in row_records}
    rebuilt: list[dict[str, Any]] = []
    partition_audit: list[dict[str, Any]] = []
    for original in band_records:
        rows = sorted(
            [row_lookup[str(row_id)] for row_id in original["row_ids"]],
            key=lambda row: base.row_lateral_at(row, 0.0),
        )
        best: dict[str, Any] | None = None
        if len(rows) >= 2 * settings.band_partition_min_rows_per_side:
            centers = np.asarray([base.row_lateral_at(row, 0.0) for row in rows])
            gaps = np.diff(centers)
            for split in range(
                settings.band_partition_min_rows_per_side,
                len(rows) - settings.band_partition_min_rows_per_side + 1,
            ):
                boundary_gap = float(gaps[split - 1])
                internal = np.delete(gaps, split - 1)
                internal_reference = float(np.median(internal)) if len(internal) else 0.0
                ratio = boundary_gap / max(internal_reference, 1e-9)
                width_factor = boundary_gap / max(median_detector_width_um, 1e-9)
                left_alternation = _row_group_alternation(rows[:split], stoma_traits)
                right_alternation = _row_group_alternation(rows[split:], stoma_traits)
                boundary_lateral = float((centers[split - 1] + centers[split]) / 2.0)
                boundary_score = _line_gradient_score(
                    boundary_lateral, stoma_traits, median_detector_width_um
                )
                control_scores = [
                    _line_gradient_score(
                        float((centers[index] + centers[index + 1]) / 2.0),
                        stoma_traits,
                        median_detector_width_um,
                    )
                    for index in range(len(gaps))
                    if index != split - 1
                ]
                controls = [value for value in control_scores if value is not None]
                image_ratio = (
                    boundary_score / max(float(np.median(controls)), 1e-9)
                    if boundary_score is not None and controls
                    else None
                )
                alternation_supported = bool(
                    left_alternation is not None
                    and right_alternation is not None
                    and left_alternation
                    >= settings.band_partition_min_within_group_alternation
                    and right_alternation
                    >= settings.band_partition_min_within_group_alternation
                )
                geometry_supported = bool(
                    width_factor
                    >= settings.band_partition_min_boundary_gap_width_factor
                    and ratio
                    >= settings.band_partition_min_boundary_to_internal_gap_ratio
                )
                image_supported = bool(
                    image_ratio is not None
                    and image_ratio >= settings.band_partition_image_support_ratio
                )
                strong_geometry = ratio >= settings.band_partition_strong_geometry_gap_ratio
                accepted = bool(
                    geometry_supported
                    and alternation_supported
                    and (image_supported or strong_geometry)
                )
                candidate = {
                    "provisional_band_id": original["band_id"],
                    "split": split,
                    "boundary_gap_um": boundary_gap,
                    "gap_ratio": ratio,
                    "gap_width_factor": width_factor,
                    "left_alternation": left_alternation,
                    "right_alternation": right_alternation,
                    "raw_image_boundary_gradient": boundary_score,
                    "raw_image_control_gradient_median": (
                        float(np.median(controls)) if controls else None
                    ),
                    "image_support_ratio": image_ratio,
                    "geometry_supported": geometry_supported,
                    "alternation_supported": alternation_supported,
                    "image_supported": image_supported,
                    "strong_geometry_supported": strong_geometry,
                    "accepted": accepted,
                }
                partition_audit.append(candidate)
                if accepted and (
                    best is None
                    or (ratio, image_ratio or 0.0)
                    > (best["gap_ratio"], best["image_support_ratio"] or 0.0)
                ):
                    best = candidate

        groups = [rows]
        if best is not None:
            groups = [rows[: int(best["split"])], rows[int(best["split"]):]]
        for group_index, group in enumerate(groups, start=1):
            rebuilt.append(
                _rebuild_band_record(
                    original,
                    group,
                    stoma_traits,
                    median_detector_width_um,
                    best,
                    group_index,
                )
            )

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
            row["band_partition_group_index"] = band["band_partition_group_index"]
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

    return row_records, rebuilt, point_assignment, partition_audit


def _accessible_longitudinal_span_um(
    lateral_um: float,
    stoma_traits: list[dict[str, Any]],
) -> float | None:
    """Approximate the visible length of a band-centre line inside the image."""

    if _ACTIVE_IMAGE_METADATA is None:
        return None
    axis = _active_axis(stoma_traits)
    if axis is None:
        return None
    px_x = base.optional_float(_ACTIVE_IMAGE_METADATA.get("metadata_pixels_per_um_x"))
    px_y = base.optional_float(_ACTIVE_IMAGE_METADATA.get("metadata_pixels_per_um_y"))
    width = base.optional_int(_ACTIVE_IMAGE_METADATA.get("image_width"))
    height = base.optional_int(_ACTIVE_IMAGE_METADATA.get("image_height"))
    if px_x is None or px_y is None or width is None or height is None:
        return None
    corners = [
        axis.forward(x, y)
        for x, y in (
            (0.0, 0.0),
            (width / px_x, 0.0),
            (0.0, height / px_y),
            (width / px_x, height / px_y),
        )
    ]
    candidates = np.linspace(
        min(value[1] for value in corners),
        max(value[1] for value in corners),
        1200,
    )
    visible: list[float] = []
    for longitudinal_um in candidates:
        x_um, y_um = axis.inverse(lateral_um, float(longitudinal_um))
        if 0.0 <= x_um <= width / px_x and 0.0 <= y_um <= height / px_y:
            visible.append(float(longitudinal_um))
    return float(max(visible) - min(visible)) if len(visible) >= 2 else None


def _mark_partial_band_eligibility(
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    settings: TraitSettingsV14,
) -> list[dict[str, Any]]:
    """Flag truncated bands without removing their stomata or morphometry."""

    axis = _active_axis(stoma_traits)
    image_width_pixels = (
        base.optional_int(_ACTIVE_IMAGE_METADATA.get("image_width"))
        if _ACTIVE_IMAGE_METADATA is not None
        else None
    )
    detector_box_widths_pixels = [
        x2 - x1
        for stoma in stoma_traits
        if (x1 := base.optional_float(stoma.get("prediction_x1_pixels"))) is not None
        and (x2 := base.optional_float(stoma.get("prediction_x2_pixels"))) is not None
        and x2 > x1
    ]
    median_detector_width_pixels = (
        float(np.median(detector_box_widths_pixels))
        if detector_box_widths_pixels
        else None
    )
    lateral_limits: tuple[float, float] | None = None
    if axis is not None and _ACTIVE_IMAGE_METADATA is not None:
        px_x = base.optional_float(_ACTIVE_IMAGE_METADATA.get("metadata_pixels_per_um_x"))
        px_y = base.optional_float(_ACTIVE_IMAGE_METADATA.get("metadata_pixels_per_um_y"))
        width = base.optional_int(_ACTIVE_IMAGE_METADATA.get("image_width"))
        height = base.optional_int(_ACTIVE_IMAGE_METADATA.get("image_height"))
        if px_x and px_y and width and height:
            lateral_corners = [
                axis.forward(x, y)[0]
                for x, y in (
                    (0.0, 0.0),
                    (width / px_x, 0.0),
                    (0.0, height / px_y),
                    (width / px_x, height / px_y),
                )
            ]
            lateral_limits = (min(lateral_corners), max(lateral_corners))

    measurements: list[dict[str, Any]] = []
    for band in band_records:
        longitudinal = [
            float(stoma_traits[index]["longitudinal_coordinate_um"])
            for index in band["_point_indices"]
        ]
        observed = float(np.ptp(longitudinal)) if longitudinal else 0.0
        accessible = _accessible_longitudinal_span_um(
            float(band["band_center_lateral_um"]), stoma_traits
        )
        coverage = base.safe_divide(observed, accessible)
        rotated_lateral_margin_um = None
        if lateral_limits is not None:
            center = float(band["band_center_lateral_um"])
            rotated_lateral_margin_um = min(
                center - lateral_limits[0], lateral_limits[1] - center
            )
        rotated_edge_margin_factor = base.safe_divide(
            rotated_lateral_margin_um, max(median_detector_width_um, 1e-9)
        )

        detector_side_margins_pixels: list[float] = []
        if image_width_pixels is not None:
            for index in band["_point_indices"]:
                x1 = base.optional_float(
                    stoma_traits[index].get("prediction_x1_pixels")
                )
                x2 = base.optional_float(
                    stoma_traits[index].get("prediction_x2_pixels")
                )
                if x1 is not None and x2 is not None:
                    detector_side_margins_pixels.append(
                        min(x1, float(image_width_pixels) - x2)
                    )
        detector_side_margin_pixels = (
            min(detector_side_margins_pixels)
            if detector_side_margins_pixels
            else None
        )
        detector_edge_margin_factor = base.safe_divide(
            detector_side_margin_pixels, median_detector_width_pixels
        )
        # Source-pixel detector geometry is the primary side-edge test. The
        # rotated centre-line estimate is retained as a transparent fallback.
        effective_edge_margin_factor = (
            detector_edge_margin_factor
            if detector_edge_margin_factor is not None
            else rotated_edge_margin_factor
        )
        measurements.append(
            {
                "band": band,
                "observed_span_um": observed,
                "accessible_span_um": accessible,
                "coverage_fraction": coverage,
                "rotated_lateral_edge_margin_um": rotated_lateral_margin_um,
                "rotated_lateral_edge_margin_box_width_factor": (
                    rotated_edge_margin_factor
                ),
                "detector_side_edge_margin_pixels": detector_side_margin_pixels,
                "median_detector_box_width_pixels": median_detector_width_pixels,
                "detector_side_edge_margin_box_width_factor": (
                    detector_edge_margin_factor
                ),
                "lateral_edge_margin_box_width_factor": (
                    effective_edge_margin_factor
                ),
                "lateral_edge_margin_source": (
                    "detector_bbox_source_pixels"
                    if detector_edge_margin_factor is not None
                    else "rotated_band_center_fallback"
                ),
            }
        )

    reference_pool = [
        item["observed_span_um"]
        for item in measurements
        if item["observed_span_um"] > 0
        and (
            item["lateral_edge_margin_box_width_factor"] is None
            or item["lateral_edge_margin_box_width_factor"]
            > settings.partial_band_edge_margin_box_width_factor
        )
    ]
    if not reference_pool:
        reference_pool = [
            item["observed_span_um"]
            for item in measurements
            if item["observed_span_um"] > 0
        ]
    reference_span = float(np.median(reference_pool)) if reference_pool else None

    row_lookup = {str(row["row_id"]): row for row in row_records}
    eligibility_audit: list[dict[str, Any]] = []
    for item in measurements:
        band = item["band"]
        observed = float(item["observed_span_um"])
        coverage = item["coverage_fraction"]
        relative_span = base.safe_divide(observed, reference_span)
        edge_factor = item["lateral_edge_margin_box_width_factor"]
        low_absolute = bool(
            coverage is not None
            and coverage
            < settings.partial_band_min_longitudinal_coverage_fraction
        )
        low_relative = bool(
            relative_span is not None
            and relative_span < settings.partial_band_min_relative_observed_span
        )
        edge_risk = bool(
            edge_factor is not None
            and edge_factor <= settings.partial_band_edge_margin_box_width_factor
        )
        # A tilted row near a side edge can have a short geometrically
        # accessible segment and therefore an apparently high coverage ratio.
        # The independent relative-span check catches that field truncation.
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
                "band_accessible_longitudinal_span_um": item["accessible_span_um"],
                "band_longitudinal_coverage_fraction": coverage,
                "band_relative_observed_span": relative_span,
                "band_rotated_lateral_edge_margin_um": item[
                    "rotated_lateral_edge_margin_um"
                ],
                "band_rotated_lateral_edge_margin_box_width_factor": item[
                    "rotated_lateral_edge_margin_box_width_factor"
                ],
                "band_detector_side_edge_margin_pixels": item[
                    "detector_side_edge_margin_pixels"
                ],
                "band_median_detector_box_width_pixels": item[
                    "median_detector_box_width_pixels"
                ],
                "band_detector_side_edge_margin_box_width_factor": item[
                    "detector_side_edge_margin_box_width_factor"
                ],
                "band_lateral_edge_margin_box_width_factor": edge_factor,
                "band_lateral_edge_margin_source": item[
                    "lateral_edge_margin_source"
                ],
                "partial_band_reference_span_um": reference_span,
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
                "accessible_longitudinal_span_um": item["accessible_span_um"],
                "coverage_fraction": coverage,
                "relative_observed_span": relative_span,
                "lateral_edge_margin_box_width_factor": edge_factor,
                "lateral_edge_margin_source": item["lateral_edge_margin_source"],
                "detector_side_edge_margin_pixels": item[
                    "detector_side_edge_margin_pixels"
                ],
                "median_detector_box_width_pixels": item[
                    "median_detector_box_width_pixels"
                ],
                "detector_side_edge_margin_box_width_factor": item[
                    "detector_side_edge_margin_box_width_factor"
                ],
                "rotated_lateral_edge_margin_um": item[
                    "rotated_lateral_edge_margin_um"
                ],
                "rotated_lateral_edge_margin_box_width_factor": item[
                    "rotated_lateral_edge_margin_box_width_factor"
                ],
            }
        )
    return eligibility_audit


def infer_rows_and_bands_v14(
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettingsV14,
    audited_row_counts: dict[str, int] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    """Infer globally oriented rows, biological bands, and spatial eligibility."""

    global _ACTIVE_DETECTOR_CONFIDENCES, _ACTIVE_LONGITUDINAL_COORDINATES
    _ACTIVE_BAND_DIAGNOSTICS.clear()
    detector_confidences = [
        base.optional_float(row.get("detector_confidence")) for row in stoma_traits
    ]
    _ACTIVE_DETECTOR_CONFIDENCES = np.asarray(
        [value if value is not None else 0.5 for value in detector_confidences],
        dtype=float,
    )
    _ACTIVE_LONGITUDINAL_COORDINATES = np.asarray(
        [float(row["longitudinal_coordinate_um"]) for row in stoma_traits],
        dtype=float,
    )
    try:
        row_records, band_records, point_assignment, audit = _BASE_INFER_ROWS_AND_BANDS(
            stoma_traits,
            settings,
            audited_row_counts=audited_row_counts,
        )
    finally:
        _ACTIVE_DETECTOR_CONFIDENCES = None
        _ACTIVE_LONGITUDINAL_COORDINATES = None

    if len(_ACTIVE_BAND_DIAGNOSTICS) != len(band_records):
        raise RuntimeError(
            "Global row diagnostics do not align with provisional bands: "
            f"diagnostics={len(_ACTIVE_BAND_DIAGNOSTICS)}, bands={len(band_records)}."
        )

    diagnostics_by_band: dict[str, dict[str, Any]] = {}
    for band, diagnostic in zip(band_records, _ACTIVE_BAND_DIAGNOSTICS, strict=True):
        provisional_id = str(band["band_id"])
        diagnostics_by_band[provisional_id] = diagnostic
        band.update(
            {
                "row_model_version": SCRIPT_VERSION,
                "provisional_band_id": provisional_id,
                "legacy_sequential_row_count": diagnostic["legacy_sequential_row_count"],
                "legacy_sequential_selection_reason": diagnostic["legacy_sequential_selection_reason"],
                "global_automatic_row_count": diagnostic["global_automatic_row_count"],
                "row_count_alternative": diagnostic["alternative_row_count"],
                "row_count_used_for_traits": diagnostic["row_count_used_for_traits"],
                "row_count_used_for_traits_source": diagnostic["row_count_used_for_traits_source"],
                "global_row_selection_score": diagnostic["global_row_selection_score"],
                "global_row_score_margin": diagnostic["global_row_score_margin"],
                "global_row_automatic_is_strict_best": diagnostic["global_row_automatic_is_strict_best"],
                "global_row_count_changed_from_legacy": diagnostic["global_row_count_changed_from_legacy"],
                "potential_row_undersplit_in_legacy": diagnostic["potential_row_undersplit_in_legacy"],
                "potential_row_oversplit_in_legacy": diagnostic["potential_row_oversplit_in_legacy"],
                "singleton_supported_row_count": diagnostic["singleton_supported_row_count"],
                "singleton_detector_confidence_mean": diagnostic["singleton_detector_confidence_mean"],
                "direct_two_row_candidate_evaluated": diagnostic["direct_two_row_candidate_evaluated"],
                "direct_two_row_validated": diagnostic["direct_two_row_validated"],
                "direct_two_row_rejected_to_one_row": diagnostic["direct_two_row_rejected_to_one_row"],
                "direct_two_row_score_improvement_vs_one": diagnostic["direct_two_row_score_improvement_vs_one"],
                "direct_two_row_strong_score_support": diagnostic["direct_two_row_strong_score_support"],
                "direct_two_row_structural_support": diagnostic["direct_two_row_structural_support"],
                "direct_two_row_alternation": diagnostic["direct_two_row_alternation"],
                "direct_two_row_phase_offset_normalized": diagnostic["direct_two_row_phase_offset_normalized"],
                "direct_two_row_longest_alternating_sequence": diagnostic["direct_two_row_longest_alternating_sequence"],
            }
        )

    for row in row_records:
        provisional_id = str(row["band_id"])
        diagnostic = diagnostics_by_band[provisional_id]
        row.update(
            {
                "row_model_version": SCRIPT_VERSION,
                "provisional_band_id": provisional_id,
                "legacy_sequential_rows_in_band": diagnostic["legacy_sequential_row_count"],
                "global_automatic_rows_in_band": diagnostic["global_automatic_row_count"],
                "row_count_alternative": diagnostic["alternative_row_count"],
                "global_row_score_margin": diagnostic["global_row_score_margin"],
                "potential_row_undersplit_in_legacy": diagnostic["potential_row_undersplit_in_legacy"],
                "singleton_supported_row": int(row["stoma_count"]) == 1,
                "direct_two_row_rejected_to_one_row": diagnostic["direct_two_row_rejected_to_one_row"],
                "direct_two_row_score_improvement_vs_one": diagnostic["direct_two_row_score_improvement_vs_one"],
            }
        )
    for assignment in point_assignment.values():
        provisional_id = str(assignment["band_id"])
        diagnostic = diagnostics_by_band[provisional_id]
        assignment.update(
            {
                "row_model_version": SCRIPT_VERSION,
                "provisional_band_id": provisional_id,
                "legacy_sequential_rows_in_band": diagnostic["legacy_sequential_row_count"],
                "global_automatic_rows_in_band": diagnostic["global_automatic_row_count"],
                "row_count_alternative": diagnostic["alternative_row_count"],
                "global_row_score_margin": diagnostic["global_row_score_margin"],
                "potential_row_undersplit_in_legacy": diagnostic["potential_row_undersplit_in_legacy"],
                "direct_two_row_rejected_to_one_row": diagnostic["direct_two_row_rejected_to_one_row"],
                "direct_two_row_score_improvement_vs_one": diagnostic["direct_two_row_score_improvement_vs_one"],
            }
        )

    candidate_lookup = {
        (band_id, int(candidate["candidate_row_count"])): candidate
        for band_id, diagnostic in diagnostics_by_band.items()
        for candidate in diagnostic["candidate_diagnostics"]
    }
    for candidate in audit.get("candidate_models", []):
        key = (str(candidate["band_id"]), int(candidate["candidate_row_count"]))
        extra = candidate_lookup.get(key)
        if extra is None:
            continue
        candidate.update(
            {
                "provisional_band_id": key[0],
                "global_adjusted_score": extra["adjusted_score"],
                "global_spacing_cv": extra["spacing_cv"],
                "global_spacing_penalty": extra["spacing_penalty"],
                "global_maximum_gap_width_factor": extra["maximum_gap_width_factor"],
                "global_large_gap_penalty": extra["large_gap_penalty"],
                "global_singleton_row_count": extra["singleton_row_count"],
                "global_singleton_detector_confidences": extra["singleton_detector_confidences"],
                "global_singleton_penalty": extra["singleton_penalty"],
                "global_interrow_gaps_um": extra["interrow_gaps_um"],
                "pair_partition_candidate": extra["pair_partition_candidate"],
                "pair_partition_split_after_row": extra["pair_partition_split_after_row"],
                "pair_partition_boundary_gap_ratio": extra["pair_partition_boundary_gap_ratio"],
                "pair_partition_boundary_gap_width_factor": extra["pair_partition_boundary_gap_width_factor"],
                "pair_partition_left_alternation": extra["pair_partition_left_alternation"],
                "pair_partition_right_alternation": extra["pair_partition_right_alternation"],
                "pair_partition_score_bonus": extra["pair_partition_score_bonus"],
                "direct_two_row_alternation": extra.get("direct_two_row_alternation"),
                "direct_two_row_phase_offset_normalized": extra.get(
                    "direct_two_row_phase_offset_normalized"
                ),
                "direct_two_row_longest_alternating_sequence": extra.get(
                    "direct_two_row_longest_alternating_sequence"
                ),
                "direct_two_row_score_improvement_vs_one": extra.get(
                    "direct_two_row_score_improvement_vs_one"
                ),
                "direct_two_row_strong_score_support": extra.get(
                    "direct_two_row_strong_score_support"
                ),
                "direct_two_row_structural_support": extra.get(
                    "direct_two_row_structural_support"
                ),
                "direct_two_row_validated": extra.get("direct_two_row_validated"),
                "direct_two_row_rejected_to_one_row": extra.get(
                    "direct_two_row_rejected_to_one_row"
                ),
                "legacy_sequential_selected": (
                    int(candidate["candidate_row_count"])
                    == diagnostics_by_band[key[0]]["legacy_sequential_row_count"]
                ),
            }
        )

    median_width = float(audit["median_detector_box_width_um"])
    row_records, band_records, point_assignment, partition_audit = (
        _partition_provisional_bands(
            row_records,
            band_records,
            point_assignment,
            stoma_traits,
            median_width,
            settings,
        )
    )
    eligibility_audit = _mark_partial_band_eligibility(
        row_records,
        band_records,
        point_assignment,
        stoma_traits,
        median_width,
        settings,
    )

    audit.update(
        {
            "automatic_row_count_policy": (
                "fit all valid one-to-five candidates after the single image-wide "
                "axis rotation; fix residual row slope to zero; compare adjusted "
                "scores globally; allow singleton-supported rows; retain a supported "
                "2+2 biological-band hypothesis; require strong score evidence or "
                "alternating phase structure before a direct one-to-two-row change"
            ),
            "band_partition_policy": (
                "partition a provisional group only when central-gap geometry and "
                "within-group alternation agree, with either raw-image boundary "
                "support or strong geometry"
            ),
            "partial_band_policy": (
                "retain every stoma for density and morphometry; exclude a band and "
                "its rows from row/band summaries only when observed longitudinal "
                "coverage is incomplete"
            ),
            "row_model_version": SCRIPT_VERSION,
            "image_wide_row_orientation_enforced": True,
            "global_row_selection_settings": {
                key: value
                for key, value in asdict(settings).items()
                if key.startswith(
                    ("global_row_", "direct_two_row_", "band_partition_", "partial_band_")
                )
            },
            "band_selection_diagnostics": diagnostics_by_band,
            "band_partition_diagnostics": partition_audit,
            "band_spatial_eligibility": eligibility_audit,
            "provisional_band_count": len(diagnostics_by_band),
            "final_detected_band_count": len(band_records),
            "final_detected_row_count": len(row_records),
            "spatially_eligible_band_count": sum(
                bool(band["band_spatial_eligible"]) for band in band_records
            ),
            "spatially_eligible_row_count": sum(
                bool(row["row_band_spatial_eligible"]) for row in row_records
            ),
            "direct_two_row_candidate_rejected_band_count": sum(
                bool(item["direct_two_row_rejected_to_one_row"])
                for item in diagnostics_by_band.values()
            ),
        }
    )
    return row_records, band_records, point_assignment, audit


def add_row_neighbors_v14(
    stoma_traits: list[dict[str, Any]],
    row_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
) -> None:
    """Calculate row traits only for bands with adequate field coverage."""

    eligible_rows = [
        row for row in row_records if bool(row.get("row_band_spatial_eligible"))
    ]
    eligible_ids = {str(row["row_id"]) for row in eligible_rows}
    eligible_assignment = {
        index: assignment
        for index, assignment in point_assignment.items()
        if str(assignment["row_id"]) in eligible_ids
    }
    for index, assignment in point_assignment.items():
        if index in eligible_assignment:
            continue
        stoma_traits[index].update(
            {
                "row_band_spatial_eligible": False,
                "row_band_spatial_exclusion_reason": assignment.get(
                    "row_band_spatial_exclusion_reason", ""
                ),
                "excluded_inferred_row_id": assignment.get("row_id", ""),
                "excluded_inferred_band_id": assignment.get("band_id", ""),
                "excluded_inferred_provisional_band_id": assignment.get(
                    "provisional_band_id", ""
                ),
            }
        )
    _BASE_ADD_ROW_NEIGHBORS(stoma_traits, eligible_rows, eligible_assignment)
    for row in stoma_traits:
        row.setdefault("row_band_spatial_eligible", False)
        row.setdefault("row_band_spatial_exclusion_reason", "")


def neighbor_relation_v14(
    first: dict[str, Any], second: dict[str, Any]
) -> str:
    """Prevent incomplete bands from being classified as valid row relations."""

    if not bool(first.get("row_band_spatial_eligible")) or not bool(
        second.get("row_band_spatial_eligible")
    ):
        return "row_band_ineligible"
    return _BASE_NEIGHBOR_RELATION(first, second)


def _eligible_spatial_probability(
    stoma_traits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recalculate row/band relationship probabilities from eligible pairs."""

    settings = _ACTIVE_SETTINGS or TraitSettingsV14()
    stoma_by_id = {str(row["stoma_id"]): row for row in stoma_traits}
    relations = ("same_row", "same_band_other_row", "adjacent_band", "other_band")
    counts = {relation: 0 for relation in relations}
    offsets: dict[str, list[tuple[float, float, float]]] = {
        relation: [] for relation in relations
    }
    origin_count = 0
    excluded_neighbor_relationship_count = 0
    for row in stoma_traits:
        if not (
            bool(row.get("row_band_spatial_eligible"))
            and bool(row.get("edge_safe_for_ranked_neighbors"))
        ):
            continue
        origin_count += 1
        for rank in range(1, settings.nearest_neighbor_ranks + 1):
            neighbor_id = str(row.get(f"neighbor_{rank}_stoma_id", ""))
            neighbor = stoma_by_id.get(neighbor_id)
            if neighbor is None:
                continue
            if not bool(neighbor.get("row_band_spatial_eligible")):
                excluded_neighbor_relationship_count += 1
                continue
            relation = str(row.get(f"neighbor_{rank}_relationship", ""))
            if relation not in offsets:
                continue
            distance = base.optional_float(row.get(f"neighbor_{rank}_distance_um"))
            dx = base.optional_float(row.get(f"neighbor_{rank}_lateral_offset_um"))
            dy = base.optional_float(row.get(f"neighbor_{rank}_longitudinal_offset_um"))
            if distance is None or dx is None or dy is None:
                continue
            counts[relation] += 1
            offsets[relation].append((abs(dx), abs(dy), distance))
    total = sum(counts.values())
    enough = origin_count >= settings.minimum_origins_for_probability_traits
    result: dict[str, Any] = {
        "spatial_probability_origin_count": origin_count,
        "spatial_probability_neighbor_relationship_count": total,
        "spatial_probability_ineligible_neighbor_relationship_count": (
            excluded_neighbor_relationship_count
        ),
        "empirical_neighbor_probabilities_calculated": enough,
        "empirical_neighbor_probability_exclusion_reason": (
            "" if enough else "insufficient_edge_safe_spatially_eligible_origins"
        ),
    }
    for relation in relations:
        result[f"{relation}_neighbor_count"] = counts[relation]
        result[f"{relation}_empirical_probability"] = (
            counts[relation] / total if enough and total else None
        )
        result.update(
            base.sample_summary(
                (item[0] for item in offsets[relation]),
                f"{relation}_absolute_lateral_offset_um",
            )
        )
        result.update(
            base.sample_summary(
                (item[1] for item in offsets[relation]),
                f"{relation}_absolute_longitudinal_offset_um",
            )
        )
        result.update(
            base.sample_summary(
                (item[2] for item in offsets[relation]),
                f"{relation}_neighbor_distance_um",
            )
        )
    return result


def image_trait_summary_v14(
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
    """Summarize row/band traits from eligible bands and all other traits from all stomata."""

    eligible_rows = [
        row for row in row_records if bool(row.get("row_band_spatial_eligible"))
    ]
    eligible_bands = [
        band for band in band_records if bool(band.get("band_spatial_eligible"))
    ]
    cleaned_probability = _eligible_spatial_probability(stoma_traits)
    result = _BASE_IMAGE_TRAIT_SUMMARY(
        image_metadata,
        stoma_traits,
        hair_rows,
        eligible_rows,
        eligible_bands,
        axis,
        cleaned_probability,
        row_band_audit,
        width_um,
        height_um,
    )

    eligible_stoma_indices = {
        index for band in eligible_bands for index in band["_point_indices"]
    }
    eligible_stoma_count = len(eligible_stoma_indices)
    height_mm = height_um / 1000.0
    width_mm = width_um / 1000.0
    area_mm2 = width_um * height_um / 1_000_000.0
    eligible_row_count = len(eligible_rows)
    eligible_band_count = len(eligible_bands)
    row_density = base.safe_divide(float(eligible_row_count), width_mm)
    band_density = base.safe_divide(float(eligible_band_count), width_mm)
    within_row_frequency = base.safe_divide(
        float(eligible_stoma_count),
        eligible_row_count * height_mm if eligible_row_count else None,
    )
    eligible_density = base.safe_divide(float(eligible_stoma_count), area_mm2)
    reconstructed_row_density = (
        row_density * within_row_frequency
        if row_density is not None and within_row_frequency is not None
        else None
    )
    mean_rows_per_band = base.safe_divide(
        float(eligible_row_count), float(eligible_band_count)
    )
    reconstructed_band_density = (
        band_density * mean_rows_per_band * within_row_frequency
        if band_density is not None
        and mean_rows_per_band is not None
        and within_row_frequency is not None
        else None
    )

    eligible_band_centers = sorted(
        float(band["band_center_lateral_um"]) for band in eligible_bands
    )
    eligible_row_centers = sorted(
        base.row_lateral_at(row, 0.0) for row in eligible_rows
    )
    complete_band_intervals = np.diff(eligible_band_centers)
    complete_row_intervals = np.diff(eligible_row_centers)
    result.update(
        {
            "detected_row_count_total": len(row_records),
            "detected_band_count_total": len(band_records),
            "spatially_eligible_row_count": eligible_row_count,
            "spatially_eligible_band_count": eligible_band_count,
            "spatially_excluded_row_count": len(row_records) - eligible_row_count,
            "spatially_excluded_band_count": len(band_records) - eligible_band_count,
            "row_band_spatial_stoma_count": eligible_stoma_count,
            "row_band_spatial_stomatal_density_per_mm2": eligible_density,
            "partial_band_detected": len(eligible_bands) != len(band_records),
            "partial_band_exclusion_reasons": sorted(
                {
                    str(band.get("band_spatial_exclusion_reason"))
                    for band in band_records
                    if not bool(band.get("band_spatial_eligible"))
                    and str(band.get("band_spatial_exclusion_reason", ""))
                }
            ),
            "biological_band_partition_count": sum(
                bool(band.get("band_partition_applied")) for band in band_records
            )
            // 2,
            "within_row_stomatal_frequency_per_mm": within_row_frequency,
            "density_reconstructed_from_row_components_per_mm2": reconstructed_row_density,
            "row_density_reconstruction_residual_per_mm2": (
                eligible_density - reconstructed_row_density
                if eligible_density is not None and reconstructed_row_density is not None
                else None
            ),
            "density_reconstructed_from_band_components_per_mm2": reconstructed_band_density,
            "band_density_reconstruction_residual_per_mm2": (
                eligible_density - reconstructed_band_density
                if eligible_density is not None and reconstructed_band_density is not None
                else None
            ),
            "stomatal_band_density_complete_intervals_per_mm": (
                1000.0 / float(np.mean(complete_band_intervals))
                if len(complete_band_intervals)
                and float(np.mean(complete_band_intervals)) > 0
                else None
            ),
            "stomatal_row_density_complete_intervals_per_mm": (
                1000.0 / float(np.mean(complete_row_intervals))
                if len(complete_row_intervals)
                and float(np.mean(complete_row_intervals)) > 0
                else None
            ),
            "row_band_spatial_eligibility_policy": (
                "density_and_morphometry_use_all_stomata; row/band organization "
                "uses only bands with adequate longitudinal field coverage"
            ),
        }
    )
    return result


def make_spatial_audit_overlay_v14(
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
    """Draw the normal overlay, then visibly mark spatially excluded bands."""

    _BASE_MAKE_SPATIAL_AUDIT_OVERLAY(
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
    excluded = [
        band for band in band_records if not bool(band.get("band_spatial_eligible"))
    ]
    if not excluded:
        return
    image = base.cv2.imread(str(output_path), base.cv2.IMREAD_COLOR)
    if image is None:
        return
    for band in excluded:
        for index in band["_point_indices"]:
            x = int(round(stoma_traits[index]["center_x_pixels"]))
            y = int(round(stoma_traits[index]["center_y_pixels"]))
            base.cv2.line(image, (x - 11, y - 11), (x + 11, y + 11), (0, 0, 255), 3)
            base.cv2.line(image, (x - 11, y + 11), (x + 11, y - 11), (0, 0, 255), 3)
        members = band["_point_indices"]
        if members:
            x = int(round(np.mean([stoma_traits[index]["center_x_pixels"] for index in members])))
            y = max(24, int(round(min(stoma_traits[index]["center_y_pixels"] for index in members))) - 12)
            base.cv2.putText(
                image,
                f"{band['band_id']} EXCLUDED: partial field",
                (max(5, x - 110), y),
                base.cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 255),
                2,
                base.cv2.LINE_AA,
            )
    if not base.cv2.imwrite(str(output_path), image, [base.cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise RuntimeError(f"Failed to update v1.4 audit overlay: {output_path}")


def process_image_v14(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Provide raw-image context and add image-level v1.4 QC counts."""

    global _ACTIVE_SOURCE_GRAY, _ACTIVE_HORIZONTAL_GRADIENT
    global _ACTIVE_IMAGE_METADATA, _ACTIVE_SETTINGS
    image_metadata = kwargs.get("image_metadata")
    if image_metadata is None and len(args) >= 3:
        image_metadata = args[2]
    settings = kwargs.get("settings")
    if settings is None and len(args) >= 6:
        settings = args[5]
    _ACTIVE_IMAGE_METADATA = image_metadata
    _ACTIVE_SETTINGS = settings
    _ACTIVE_SOURCE_GRAY = None
    _ACTIVE_HORIZONTAL_GRADIENT = None
    if image_metadata is not None:
        source_path_text = str(image_metadata.get("source_image_path", "")).strip()
        if source_path_text:
            source_path = Path(source_path_text).expanduser()
            if source_path.is_file():
                _ACTIVE_SOURCE_GRAY = base.cv2.imread(
                    str(source_path), base.cv2.IMREAD_GRAYSCALE
                )
                if _ACTIVE_SOURCE_GRAY is not None:
                    gradient_x = base.cv2.Sobel(
                        _ACTIVE_SOURCE_GRAY, base.cv2.CV_32F, 1, 0, ksize=3
                    )
                    gradient_y = base.cv2.Sobel(
                        _ACTIVE_SOURCE_GRAY, base.cv2.CV_32F, 0, 1, ksize=3
                    )
                    _ACTIVE_HORIZONTAL_GRADIENT = base.cv2.magnitude(
                        gradient_x, gradient_y
                    )
    try:
        result = list(_BASE_PROCESS_IMAGE(*args, **kwargs))
    finally:
        _ACTIVE_SOURCE_GRAY = None
        _ACTIVE_HORIZONTAL_GRADIENT = None
        _ACTIVE_IMAGE_METADATA = None
        _ACTIVE_SETTINGS = None
    image_traits = result[1]
    row_records = result[3]
    band_records = result[4]
    provisional_bands: dict[str, dict[str, Any]] = {}
    for band in band_records:
        provisional_bands.setdefault(
            str(band.get("provisional_band_id") or band.get("band_id")), band
        )
    diagnostic_bands = list(provisional_bands.values())

    image_traits.update(
        {
            "row_model_version": SCRIPT_VERSION,
            "global_row_changed_from_legacy_band_count": sum(
                bool(band.get("global_row_count_changed_from_legacy"))
                for band in diagnostic_bands
            ),
            "potential_row_undersplit_in_legacy_band_count": sum(
                bool(band.get("potential_row_undersplit_in_legacy"))
                for band in diagnostic_bands
            ),
            "potential_row_oversplit_in_legacy_band_count": sum(
                bool(band.get("potential_row_oversplit_in_legacy"))
                for band in diagnostic_bands
            ),
            "direct_two_row_candidate_rejected_band_count": sum(
                bool(band.get("direct_two_row_rejected_to_one_row"))
                for band in diagnostic_bands
            ),
            "singleton_supported_band_count": sum(
                int(band.get("singleton_supported_row_count") or 0) > 0
                for band in band_records
            ),
            "singleton_supported_row_count": sum(
                int(row.get("stoma_count") or 0) == 1 for row in row_records
            ),
            "low_confidence_global_row_band_count": sum(
                str(band.get("row_model_confidence")) == "low"
                for band in diagnostic_bands
            ),
            "moderate_confidence_global_row_band_count": sum(
                str(band.get("row_model_confidence")) == "moderate"
                for band in diagnostic_bands
            ),
            "high_confidence_global_row_band_count": sum(
                str(band.get("row_model_confidence")) == "high"
                for band in diagnostic_bands
            ),
            "image_wide_row_orientation_enforced": True,
        }
    )
    return tuple(result)


def image_gwas_trait_summary_v14(
    image_traits: dict[str, Any],
) -> dict[str, Any]:
    """Retain new row-count QC fields beside the established GWAS columns."""

    summary = _BASE_IMAGE_GWAS_TRAIT_SUMMARY(image_traits)
    for field in (
        "row_model_version",
        "global_row_changed_from_legacy_band_count",
        "potential_row_undersplit_in_legacy_band_count",
        "potential_row_oversplit_in_legacy_band_count",
        "direct_two_row_candidate_rejected_band_count",
        "singleton_supported_band_count",
        "singleton_supported_row_count",
        "low_confidence_global_row_band_count",
        "moderate_confidence_global_row_band_count",
        "high_confidence_global_row_band_count",
        "image_wide_row_orientation_enforced",
        "detected_row_count_total",
        "detected_band_count_total",
        "spatially_eligible_row_count",
        "spatially_eligible_band_count",
        "spatially_excluded_row_count",
        "spatially_excluded_band_count",
        "row_band_spatial_stoma_count",
        "row_band_spatial_stomatal_density_per_mm2",
        "partial_band_detected",
        "partial_band_exclusion_reasons",
        "biological_band_partition_count",
        "stomatal_band_density_complete_intervals_per_mm",
        "stomatal_row_density_complete_intervals_per_mm",
        "spatial_probability_ineligible_neighbor_relationship_count",
        "row_band_spatial_eligibility_policy",
    ):
        summary[field] = image_traits.get(field)
    return summary


def install_v14_row_selector() -> None:
    """Patch only the version-1.3 extension points used by this new script."""

    base.parse_arguments = parse_arguments_v14
    base.build_settings = build_settings_v14
    base.select_row_model = select_row_model_v14
    base.infer_rows_and_bands = infer_rows_and_bands_v14
    base.add_row_neighbors = add_row_neighbors_v14
    base.neighbor_relation = neighbor_relation_v14
    base.image_trait_summary = image_trait_summary_v14
    base.make_spatial_audit_overlay = make_spatial_audit_overlay_v14
    base.process_image = process_image_v14
    base.image_gwas_trait_summary = image_gwas_trait_summary_v14
    base.SCRIPT_VERSION = SCRIPT_VERSION

    # The inherited provenance writer should record this wrapper rather than
    # incorrectly identifying the unchanged version-1.3 module as the runner.
    base.__file__ = __file__


def run_trait_extraction() -> Path:
    """Run the experimental extractor with all v1.3 safety checks retained."""

    install_v14_row_selector()
    return base.run_trait_extraction()


if __name__ == "__main__":
    run_trait_extraction()
