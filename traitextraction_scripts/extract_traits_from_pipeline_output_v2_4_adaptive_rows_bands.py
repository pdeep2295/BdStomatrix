#!/usr/bin/env python3
"""Experimental v2.4 Brachypodium row/band topology extractor.

This is a separate successor to the v2.3.2 extractor.  It does not modify the
v2.3.2 script, pipeline inference results, masks, microscopy files, or prior
trait tables.

The rules in this version were motivated by two manually audited 40-image
sets, but every decision is image-name-independent.  V2.4 adds four guarded
operations:

1. Merge adjacent pieces of one locally tilted/curved row when their
   longitudinal supports form a clean continuation.  All other memberships
   remain frozen.
2. Search every sufficiently supported selected row for one strongly
   corroborated bracketed singleton.  The search is no longer limited to a
   candidate nominated by an earlier unconstrained model.
3. Reconsider strong biological-band boundaries after row correction.  A
   split that isolates one multipoint row requires both strong geometry and
   raw-image boundary support.  A weakly supported inherited split can be
   reversed.
4. Assess both lateral half-cut bands and short longitudinal edge fragments.
   Adequate longitudinal support vetoes a proximity-only exclusion.

Density and individual-stoma/cell morphometry are never changed by row/band
eligibility.  Row and band values remain in the audit tables with explicit QC
even when they are excluded from complete-topology summaries.

The thresholds remain provisional.  Run this script only in a fresh Python
process and always use a new output directory.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import extract_traits_from_pipeline_output_v2_3_frozen_bands_extended_singletons as v23


base = v23.base
v14 = v23.v14
v15 = v23.v15
v16 = v23.v16
v19 = v23.v19
v22 = v23.parent

SCRIPT_VERSION = "2.4.0-experimental-adaptive-row-band-topology"
SINGLETON_POLICY_VERSION = SCRIPT_VERSION
ROW_GEOMETRY_MODEL_VERSION = SCRIPT_VERSION
PARTIAL_BAND_ELIGIBILITY_VERSION = SCRIPT_VERSION
TRAIT_EXPORT_POLICY_VERSION = v23.TRAIT_EXPORT_POLICY_VERSION

_PARENT_V23_SCRIPT_PATH = Path(v23.__file__).resolve()

# Adjacent-fragment merge.  Distances are normalized by the local median
# detector-box width, so the rule scales with stomatal size.
V24_FRAGMENT_MIN_MEMBERS_PER_SIDE = 2
V24_FRAGMENT_MAX_CENTER_SEPARATION_WIDTH_FACTOR = 0.35
V24_FRAGMENT_MAX_LONGITUDINAL_OVERLAP_FRACTION = 0.10
V24_FRAGMENT_MIN_ENDPOINT_GAP_RATIO = 0.15
V24_FRAGMENT_MAX_ENDPOINT_GAP_RATIO = 1.75
V24_FRAGMENT_MAX_ENDPOINT_MISMATCH_WIDTH_FACTOR = 0.30
V24_FRAGMENT_MAX_MERGED_RMS_WIDTH_FACTOR = 0.15
V24_FRAGMENT_STRONG_MERGED_RMS_WIDTH_FACTOR = 0.08
V24_FRAGMENT_MIN_OTHER_ROW_CLEARANCE_WIDTH_FACTOR = 0.22
V24_FRAGMENT_MAX_LOCAL_SLOPE_DEG = 12.0
V24_FRAGMENT_UNSTABLE_ROW_COUNT_FRACTION = 0.80
V24_FRAGMENT_MIN_SIMPLER_PERTURBATION_FRACTION = 0.25
V24_FRAGMENT_UNIQUE_SCORE_MARGIN = 0.02

# Frozen-membership singleton recovery.  These conservative values implement
# the audit-supported large-row bracketed route discussed after v2.3.2.
V24_SINGLETON_MIN_SOURCE_MEMBERS = 4
V24_SINGLETON_MIN_REMAINING_MEMBERS = 3
V24_SINGLETON_MIN_DETECTOR_CONFIDENCE = 0.80
V24_SINGLETON_MIN_RESIDUAL_WIDTH_FACTOR = 0.50
V24_SINGLETON_MAX_REMAINING_RMS_WIDTH_FACTOR = 0.06
V24_SINGLETON_MIN_RESIDUAL_DOMINANCE = 6.0
V24_SINGLETON_MIN_FROZEN_SCORE_IMPROVEMENT = 2.0
V24_SINGLETON_MIN_OTHER_ROW_CLEARANCE_WIDTH_FACTOR = 0.35
V24_SINGLETON_MIN_STABILITY_FRACTION = 0.80
V24_SINGLETON_UNIQUE_SCORE_MARGIN = 0.50
V24_SINGLETON_MAX_ADDITIONS_PER_PROVISIONAL_BAND = 1

# Band partitioning.  A split isolating one well-supported row needs a strong
# lateral gap and raw-image boundary evidence.  The 2.0 image-support value
# separated the confirmed S028 and S048 cases and therefore remains explicitly
# audit-derived and provisional.
V24_PARTITION_MIN_GAP_WIDTH_FACTOR = 1.50
V24_PARTITION_MIN_GAP_RATIO = 2.00
V24_PARTITION_MIN_IMAGE_SUPPORT_RATIO = 2.00
V24_PARTITION_MIN_STABILITY_FRACTION = 0.80
V24_PARTITION_MIN_STOMATA_PER_SIDE = 3
V24_PARTITION_REVERSE_SINGLE_ROW_MAX_IMAGE_SUPPORT_RATIO = 2.00

# Final two-axis edge completeness.
V24_EDGE_SHORT_MAX_COVERAGE = 0.60
V24_EDGE_SHORT_MAX_RELATIVE_SPAN = 0.70
V24_EDGE_COMPLETE_MIN_COVERAGE = 0.65
V24_EDGE_COMPLETE_MIN_RELATIVE_SPAN = 0.80
V24_EDGE_COMPLETE_MIN_STOMATA = 3
V24_EDGE_COMPLETE_MIN_ROW_COVERAGE = 0.65
V24_EDGE_MAX_ENDPOINT_CLEARANCE_DETECTOR_HEIGHTS = 1.50
V24_EDGE_LATERAL_SPARSE_MAX_BBOX_CLEARANCE_WIDTH_FACTOR = 0.25


@dataclass(frozen=True)
class TraitSettingsV24(v23.TraitSettingsV23):
    """Immutable v2.4 settings written into provenance."""

    v24_policy_version: str = SCRIPT_VERSION
    v24_parent_v23_script_sha256: str = ""
    v24_fragment_min_members_per_side: int = V24_FRAGMENT_MIN_MEMBERS_PER_SIDE
    v24_fragment_max_center_separation_width_factor: float = (
        V24_FRAGMENT_MAX_CENTER_SEPARATION_WIDTH_FACTOR
    )
    v24_fragment_max_longitudinal_overlap_fraction: float = (
        V24_FRAGMENT_MAX_LONGITUDINAL_OVERLAP_FRACTION
    )
    v24_fragment_min_endpoint_gap_ratio: float = V24_FRAGMENT_MIN_ENDPOINT_GAP_RATIO
    v24_fragment_max_endpoint_gap_ratio: float = V24_FRAGMENT_MAX_ENDPOINT_GAP_RATIO
    v24_fragment_max_endpoint_mismatch_width_factor: float = (
        V24_FRAGMENT_MAX_ENDPOINT_MISMATCH_WIDTH_FACTOR
    )
    v24_fragment_max_merged_rms_width_factor: float = (
        V24_FRAGMENT_MAX_MERGED_RMS_WIDTH_FACTOR
    )
    v24_fragment_strong_merged_rms_width_factor: float = (
        V24_FRAGMENT_STRONG_MERGED_RMS_WIDTH_FACTOR
    )
    v24_fragment_min_other_row_clearance_width_factor: float = (
        V24_FRAGMENT_MIN_OTHER_ROW_CLEARANCE_WIDTH_FACTOR
    )
    v24_fragment_max_local_slope_deg: float = V24_FRAGMENT_MAX_LOCAL_SLOPE_DEG
    v24_fragment_unstable_row_count_fraction: float = (
        V24_FRAGMENT_UNSTABLE_ROW_COUNT_FRACTION
    )
    v24_fragment_min_simpler_perturbation_fraction: float = (
        V24_FRAGMENT_MIN_SIMPLER_PERTURBATION_FRACTION
    )
    v24_fragment_unique_score_margin: float = V24_FRAGMENT_UNIQUE_SCORE_MARGIN
    v24_singleton_min_source_members: int = V24_SINGLETON_MIN_SOURCE_MEMBERS
    v24_singleton_min_remaining_members: int = V24_SINGLETON_MIN_REMAINING_MEMBERS
    v24_singleton_min_detector_confidence: float = (
        V24_SINGLETON_MIN_DETECTOR_CONFIDENCE
    )
    v24_singleton_min_residual_width_factor: float = (
        V24_SINGLETON_MIN_RESIDUAL_WIDTH_FACTOR
    )
    v24_singleton_max_remaining_rms_width_factor: float = (
        V24_SINGLETON_MAX_REMAINING_RMS_WIDTH_FACTOR
    )
    v24_singleton_min_residual_dominance: float = (
        V24_SINGLETON_MIN_RESIDUAL_DOMINANCE
    )
    v24_singleton_min_frozen_score_improvement: float = (
        V24_SINGLETON_MIN_FROZEN_SCORE_IMPROVEMENT
    )
    v24_singleton_min_other_row_clearance_width_factor: float = (
        V24_SINGLETON_MIN_OTHER_ROW_CLEARANCE_WIDTH_FACTOR
    )
    v24_singleton_min_stability_fraction: float = (
        V24_SINGLETON_MIN_STABILITY_FRACTION
    )
    v24_singleton_unique_score_margin: float = V24_SINGLETON_UNIQUE_SCORE_MARGIN
    v24_singleton_max_additions_per_provisional_band: int = (
        V24_SINGLETON_MAX_ADDITIONS_PER_PROVISIONAL_BAND
    )
    v24_partition_min_gap_width_factor: float = V24_PARTITION_MIN_GAP_WIDTH_FACTOR
    v24_partition_min_gap_ratio: float = V24_PARTITION_MIN_GAP_RATIO
    v24_partition_min_image_support_ratio: float = (
        V24_PARTITION_MIN_IMAGE_SUPPORT_RATIO
    )
    v24_partition_min_stability_fraction: float = (
        V24_PARTITION_MIN_STABILITY_FRACTION
    )
    v24_partition_min_stomata_per_side: int = V24_PARTITION_MIN_STOMATA_PER_SIDE
    v24_partition_reverse_single_row_max_image_support_ratio: float = (
        V24_PARTITION_REVERSE_SINGLE_ROW_MAX_IMAGE_SUPPORT_RATIO
    )
    v24_edge_short_max_coverage: float = V24_EDGE_SHORT_MAX_COVERAGE
    v24_edge_short_max_relative_span: float = V24_EDGE_SHORT_MAX_RELATIVE_SPAN
    v24_edge_complete_min_coverage: float = V24_EDGE_COMPLETE_MIN_COVERAGE
    v24_edge_complete_min_relative_span: float = (
        V24_EDGE_COMPLETE_MIN_RELATIVE_SPAN
    )
    v24_edge_complete_min_stomata: int = V24_EDGE_COMPLETE_MIN_STOMATA
    v24_edge_complete_min_row_coverage: float = (
        V24_EDGE_COMPLETE_MIN_ROW_COVERAGE
    )
    v24_edge_max_endpoint_clearance_detector_heights: float = (
        V24_EDGE_MAX_ENDPOINT_CLEARANCE_DETECTOR_HEIGHTS
    )
    v24_edge_lateral_sparse_max_bbox_clearance_width_factor: float = (
        V24_EDGE_LATERAL_SPARSE_MAX_BBOX_CLEARANCE_WIDTH_FACTOR
    )


_V24_DEFAULTS: dict[str, Any] = {
    "trajectory_v24_policy_version": SCRIPT_VERSION,
    "trajectory_v24_fragment_merge_candidate": False,
    "trajectory_v24_fragment_merge_applied": False,
    "trajectory_v24_fragment_merge_status": "not_applicable",
    "trajectory_v24_fragment_merge_reason": "",
    "trajectory_v24_fragment_merge_groups": [],
    "trajectory_v24_fragment_merge_member_counts": [],
    "trajectory_v24_fragment_merge_point_indices": [],
    "trajectory_v24_fragment_merge_center_separation_width_factor": None,
    "trajectory_v24_fragment_merge_longitudinal_overlap_fraction": None,
    "trajectory_v24_fragment_merge_endpoint_gap_ratio": None,
    "trajectory_v24_fragment_merge_endpoint_mismatch_width_factor": None,
    "trajectory_v24_fragment_merge_rms_width_factor": None,
    "trajectory_v24_fragment_merge_simpler_perturbation_fraction": None,
    "trajectory_v24_fragment_merge_unique_margin": None,
    "trajectory_v24_singleton_scan_performed": False,
    "trajectory_v24_singleton_candidate_count": 0,
    "trajectory_v24_singleton_applied": False,
    "trajectory_v24_singleton_status": "not_applicable",
    "trajectory_v24_singleton_reason": "",
    "trajectory_v24_singleton_candidate_point_index": None,
    "trajectory_v24_singleton_candidate_stoma_id": "",
    "trajectory_v24_singleton_source_members_before": None,
    "trajectory_v24_singleton_source_members_after": None,
    "trajectory_v24_singleton_residual_width_factor": None,
    "trajectory_v24_singleton_remaining_rms_width_factor": None,
    "trajectory_v24_singleton_residual_dominance": None,
    "trajectory_v24_singleton_bracketing_gap_ratio": None,
    "trajectory_v24_singleton_frozen_score_improvement": None,
    "trajectory_v24_singleton_other_row_clearance_width_factor": None,
    "trajectory_v24_singleton_stability_fraction": None,
    "trajectory_v24_singleton_unique_margin": None,
    "trajectory_v24_singleton_maximum_rows_in_strong_gap_group": None,
    "trajectory_v24_singleton_strong_gap_threshold_um": None,
    "trajectory_v24_memberships_frozen": False,
    "trajectory_v24_local_row_refit_applied": False,
}

_V24_IMAGE_FIELDS = (
    "trajectory_v24_policy_version",
    "trajectory_v24_fragment_merge_count",
    "trajectory_v24_singleton_rescue_count",
    "trajectory_v24_partition_split_count",
    "trajectory_v24_partition_reversal_count",
    "trajectory_v24_edge_fragment_excluded_band_count",
    "trajectory_v24_lateral_half_cut_excluded_band_count",
    "trajectory_v24_lateral_sparse_border_excluded_band_count",
    "trajectory_v24_longitudinal_edge_fragment_excluded_band_count",
    "trajectory_v24_adequate_support_veto_band_count",
    "detected_row_count_v24",
    "detected_band_count_v24",
    "row_band_v24_qc_status",
    "row_band_v24_qc_flag",
    "row_band_v24_qc_reason",
    "row_band_v24_value_interpretation",
)


def build_settings_v24(args: Any) -> TraitSettingsV24:
    inherited = asdict(v23.build_settings_v23(args))
    inherited.update(
        {
            "v24_policy_version": SCRIPT_VERSION,
            "v24_parent_v23_script_sha256": hashlib.sha256(
                _PARENT_V23_SCRIPT_PATH.read_bytes()
            ).hexdigest(),
        }
    )
    return TraitSettingsV24(**inherited)


def _finite(value: Any) -> float | None:
    parsed = base.optional_float(value)
    if parsed is None or not np.isfinite(parsed):
        return None
    return float(parsed)


def _group_members(candidate: dict[str, Any]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for point_index, label in v22._candidate_label_map(candidate).items():
        grouped.setdefault(int(label), []).append(int(point_index))
    return {key: sorted(value) for key, value in grouped.items()}


def _median_positive_gaps(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    gaps = np.diff(np.sort(np.asarray(values, dtype=float)))
    gaps = gaps[gaps > 1e-9]
    return float(np.median(gaps)) if gaps.size else None


def _fragment_pair_geometry(
    chosen: dict[str, Any],
    left_group: int,
    right_group: int,
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    fallback_width_um: float,
    settings: TraitSettingsV24,
) -> dict[str, Any]:
    grouped = _group_members(chosen)
    first = grouped.get(left_group, [])
    second = grouped.get(right_group, [])
    if min(len(first), len(second)) < int(settings.v24_fragment_min_members_per_side):
        return {"accepted_basic_geometry": False, "reason": "too_few_members"}

    local_width = v23._local_detector_width_um(first + second, fallback_width_um)
    first_long = np.sort(np.asarray([longitudinal[i] for i in first], dtype=float))
    second_long = np.sort(np.asarray([longitudinal[i] for i in second], dtype=float))
    first_span = float(np.ptp(first_long))
    second_span = float(np.ptp(second_long))
    overlap = max(
        0.0,
        min(float(first_long[-1]), float(second_long[-1]))
        - max(float(first_long[0]), float(second_long[0])),
    )
    overlap_reference = max(min(first_span, second_span), 1e-9)
    overlap_fraction = overlap / overlap_reference

    if float(first_long[-1]) <= float(second_long[0]):
        earlier, later = first, second
    elif float(second_long[-1]) <= float(first_long[0]):
        earlier, later = second, first
    else:
        earlier = later = []
    if not earlier:
        endpoint_gap = 0.0
        earlier_endpoint = first[int(np.argmax(longitudinal[first]))]
        later_endpoint = second[int(np.argmin(longitudinal[second]))]
    else:
        earlier_endpoint = max(earlier, key=lambda i: float(longitudinal[i]))
        later_endpoint = min(later, key=lambda i: float(longitudinal[i]))
        endpoint_gap = max(
            0.0,
            float(longitudinal[later_endpoint])
            - float(longitudinal[earlier_endpoint]),
        )
    ordinary_gaps = [
        value
        for value in (
            _median_positive_gaps([float(longitudinal[i]) for i in first]),
            _median_positive_gaps([float(longitudinal[i]) for i in second]),
        )
        if value is not None and value > 0
    ]
    typical_gap = (
        float(np.median(np.asarray(ordinary_gaps, dtype=float)))
        if ordinary_gaps
        else None
    )
    endpoint_gap_ratio = (
        endpoint_gap / typical_gap
        if typical_gap is not None and typical_gap > 0
        else None
    )

    intercept, slope, rms, maximum = v23._fit_local_line(
        first + second, lateral, longitudinal
    )
    endpoint_residuals = [
        float(lateral[index])
        - (intercept + slope * float(longitudinal[index]))
        for index in (earlier_endpoint, later_endpoint)
    ]
    endpoint_mismatch = abs(endpoint_residuals[0] - endpoint_residuals[1])
    intercepts = np.asarray(chosen.get("intercepts_um", []), dtype=float)
    center_separation = (
        abs(float(intercepts[right_group]) - float(intercepts[left_group]))
        if len(intercepts) > right_group
        else math.inf
    )

    midpoint = float(
        np.median(np.asarray([longitudinal[i] for i in first + second], dtype=float))
    )
    merged_at_midpoint = intercept + slope * midpoint
    reference = float(chosen.get("long_reference_um") or 0.0)
    other_clearances = [
        abs(
            merged_at_midpoint
            - (
                float(intercepts[group])
                + float(chosen.get("slope") or 0.0) * (midpoint - reference)
            )
        )
        for group in range(int(chosen.get("k", 0)))
        if group not in {left_group, right_group} and len(intercepts) > group
    ]
    other_clearance = min(other_clearances) if other_clearances else math.inf

    fields = {
        "left_group": left_group,
        "right_group": right_group,
        "first_members": first,
        "second_members": second,
        "local_width_um": local_width,
        "center_separation_width_factor": center_separation / max(local_width, 1e-9),
        "longitudinal_overlap_fraction": overlap_fraction,
        "endpoint_gap_ratio": endpoint_gap_ratio,
        "endpoint_mismatch_width_factor": endpoint_mismatch / max(local_width, 1e-9),
        "merged_rms_width_factor": rms / max(local_width, 1e-9),
        "merged_max_residual_width_factor": maximum / max(local_width, 1e-9),
        "merged_slope_deg": math.degrees(math.atan(slope)),
        "other_row_clearance_width_factor": other_clearance / max(local_width, 1e-9),
    }
    failures: list[str] = []
    if fields["center_separation_width_factor"] > float(
        settings.v24_fragment_max_center_separation_width_factor
    ):
        failures.append("center_separation")
    if overlap_fraction > float(settings.v24_fragment_max_longitudinal_overlap_fraction):
        failures.append("longitudinal_overlap")
    if endpoint_gap_ratio is None or not (
        float(settings.v24_fragment_min_endpoint_gap_ratio)
        <= endpoint_gap_ratio
        <= float(settings.v24_fragment_max_endpoint_gap_ratio)
    ):
        failures.append("endpoint_gap")
    if fields["endpoint_mismatch_width_factor"] > float(
        settings.v24_fragment_max_endpoint_mismatch_width_factor
    ):
        failures.append("endpoint_mismatch")
    if fields["merged_rms_width_factor"] > float(
        settings.v24_fragment_max_merged_rms_width_factor
    ):
        failures.append("merged_rms")
    if abs(fields["merged_slope_deg"]) > float(
        settings.v24_fragment_max_local_slope_deg
    ):
        failures.append("local_slope")
    if fields["other_row_clearance_width_factor"] < float(
        settings.v24_fragment_min_other_row_clearance_width_factor
    ):
        failures.append("duplicates_other_row")
    fields["accepted_basic_geometry"] = not failures
    fields["reason"] = ";".join(failures)
    return fields


def _merge_fixed_membership_candidate(
    chosen: dict[str, Any],
    left_group: int,
    right_group: int,
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV24,
) -> tuple[dict[str, Any], dict[str, Any]]:
    point_indices = [int(value) for value in chosen.get("point_indices", [])]
    original_map = v22._candidate_label_map(chosen)
    labels: list[int] = []
    for point_index in point_indices:
        group = int(original_map[point_index])
        if group == right_group:
            group = left_group
        if group > right_group:
            group -= 1
        labels.append(group)

    limit = float(settings.v24_fragment_max_local_slope_deg)
    step = max(float(settings.trajectory_local_tilt_step_deg), 0.5)
    angles = np.arange(-limit, limit + step * 0.5, step, dtype=float).tolist()
    angles.extend([0.0, math.degrees(math.atan(float(chosen.get("slope") or 0.0)))])
    best: tuple[float, float, dict[str, Any], dict[str, Any]] | None = None
    for angle in sorted({round(float(value), 10) for value in angles}):
        template = dict(chosen)
        template.update(
            {
                "k": int(chosen["k"]) - 1,
                "labels": np.asarray(labels, dtype=int),
                "slope": math.tan(math.radians(angle)),
            }
        )
        refitted = v22._refit_fixed_membership_candidate(
            template, lateral, longitudinal
        )
        diagnostic = v15._score_trajectory_candidate(
            refitted, median_width_um, angle, settings
        )
        rank = (
            float(diagnostic["adjusted_score"]),
            abs(angle),
            refitted,
            diagnostic,
        )
        if best is None or rank[:2] < best[:2]:
            best = rank
    if best is None:
        raise RuntimeError("v2.4 failed to refit a frozen fragment merge.")
    return best[2], best[3]


def _apply_v24_fragment_merge(
    chosen: dict[str, Any],
    candidates: list[dict[str, Any]],
    diagnostic: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV24,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if int(chosen.get("k", 0)) < 2 or bool(
        chosen.get("trajectory_v20_singleton_rescue_applied")
    ):
        return chosen, candidates, diagnostic

    stability_distribution = diagnostic.get("trajectory_stability_count_distribution") or {}
    total_votes = sum(int(value) for value in stability_distribution.values())
    simpler_votes = int(stability_distribution.get(str(int(chosen["k"]) - 1), 0))
    if simpler_votes == 0:
        simpler_votes = int(stability_distribution.get(int(chosen["k"]) - 1, 0))
    simpler_fraction = simpler_votes / max(total_votes, 1)
    row_count_stability = _finite(
        diagnostic.get("trajectory_row_count_stability_fraction")
    )
    instability_supported = bool(
        row_count_stability is not None
        and row_count_stability
        < float(settings.v24_fragment_unstable_row_count_fraction)
        and simpler_fraction
        >= float(settings.v24_fragment_min_simpler_perturbation_fraction)
    )

    assessed: list[dict[str, Any]] = []
    for left_group in range(int(chosen["k"]) - 1):
        geometry = _fragment_pair_geometry(
            chosen,
            left_group,
            left_group + 1,
            lateral,
            longitudinal,
            median_width_um,
            settings,
        )
        if not bool(geometry.get("accepted_basic_geometry")):
            continue
        strong_geometry = bool(
            float(geometry["merged_rms_width_factor"])
            <= float(settings.v24_fragment_strong_merged_rms_width_factor)
            and float(geometry["endpoint_mismatch_width_factor"]) <= 0.20
        )
        if not (instability_supported or strong_geometry):
            continue
        quality = (
            float(geometry["merged_rms_width_factor"])
            + 0.25 * float(geometry["endpoint_mismatch_width_factor"])
            + 0.05 * abs(float(geometry["endpoint_gap_ratio"]) - 1.0)
            + 0.05 * float(geometry["center_separation_width_factor"])
        )
        assessed.append(
            {
                **geometry,
                "strong_geometry": strong_geometry,
                "instability_supported": instability_supported,
                "quality": quality,
            }
        )

    if not assessed:
        return chosen, candidates, diagnostic
    assessed.sort(key=lambda item: (float(item["quality"]), item["left_group"]))
    unique_margin = (
        float(assessed[1]["quality"]) - float(assessed[0]["quality"])
        if len(assessed) > 1
        else math.inf
    )
    best = assessed[0]
    if unique_margin < float(settings.v24_fragment_unique_score_margin):
        fields = {
            "trajectory_v24_fragment_merge_candidate": True,
            "trajectory_v24_fragment_merge_applied": False,
            "trajectory_v24_fragment_merge_status": "rejected_ambiguous_pairs",
            "trajectory_v24_fragment_merge_reason": "no_unique_best_adjacent_fragment_pair",
            "trajectory_v24_fragment_merge_unique_margin": unique_margin,
        }
        chosen.update(fields)
        diagnostic.update(fields)
        return chosen, candidates, diagnostic

    merged, merged_diagnostic = _merge_fixed_membership_candidate(
        chosen,
        int(best["left_group"]),
        int(best["right_group"]),
        lateral,
        longitudinal,
        median_width_um,
        settings,
    )
    merged_points = sorted(
        int(value) for value in best["first_members"] + best["second_members"]
    )
    merged_map = v22._candidate_label_map(merged)
    merged_groups = {merged_map[index] for index in merged_points}
    if len(merged_groups) != 1:
        raise RuntimeError(
            "v2.4 fragment merge refit did not preserve the requested membership."
        )
    merged_group = next(iter(merged_groups))
    reference = float(merged.get("long_reference_um") or 0.0)
    exported_slope = float(merged.get("slope") or 0.0)
    exported_intercept = float(merged["intercepts_um"][merged_group])
    exported_residuals = np.asarray(
        [
            float(lateral[index])
            - (
                exported_intercept
                + exported_slope * (float(longitudinal[index]) - reference)
            )
            for index in merged_points
        ],
        dtype=float,
    )
    exported_rms_factor = float(
        math.sqrt(float(np.mean(exported_residuals * exported_residuals)))
        / max(float(best["local_width_um"]), 1e-9)
    )
    support_longitudinal = np.asarray(
        [float(longitudinal[index]) for index in merged_points], dtype=float
    )
    if support_longitudinal.size:
        support_samples = np.unique(
            np.concatenate(
                [
                    support_longitudinal,
                    np.linspace(
                        float(np.min(support_longitudinal)),
                        float(np.max(support_longitudinal)),
                        101,
                        dtype=float,
                    ),
                ]
            )
        )
    else:
        support_samples = np.asarray([reference], dtype=float)
    exported_other_clearance = min(
        [
            min(
                abs(
                    (
                        exported_intercept
                        + exported_slope * (float(value) - reference)
                    )
                    - (
                        float(intercept)
                        + exported_slope * (float(value) - reference)
                    )
                )
                for value in support_samples
            )
            / max(float(best["local_width_um"]), 1e-9)
            for group, intercept in enumerate(merged.get("intercepts_um", []))
            if group != merged_group
        ]
        or [math.inf]
    )
    if (
        exported_rms_factor
        > float(settings.v24_fragment_max_merged_rms_width_factor)
        or exported_other_clearance
        < float(settings.v24_fragment_min_other_row_clearance_width_factor)
    ):
        rejection = {
            "trajectory_v24_fragment_merge_candidate": True,
            "trajectory_v24_fragment_merge_applied": False,
            "trajectory_v24_fragment_merge_status": "rejected_exported_hybrid_geometry",
            "trajectory_v24_fragment_merge_reason": (
                "frozen_shared_slope_hybrid_failed_rms_or_nonduplication_gate"
            ),
            "trajectory_v24_fragment_merge_rms_width_factor": exported_rms_factor,
            "trajectory_v24_fragment_merge_unique_margin": (
                unique_margin if np.isfinite(unique_margin) else None
            ),
        }
        chosen.update(rejection)
        diagnostic.update(rejection)
        return chosen, candidates, diagnostic
    old_k = int(chosen["k"])
    new_k = int(merged["k"])
    fields = {
        "trajectory_v24_fragment_merge_candidate": True,
        "trajectory_v24_fragment_merge_applied": True,
        "trajectory_v24_fragment_merge_status": "accepted_provisional",
        "trajectory_v24_fragment_merge_reason": (
            "adjacent_nonoverlapping_fragments_form_one_smooth_local_trajectory"
        ),
        "trajectory_v24_fragment_merge_groups": [
            int(best["left_group"]),
            int(best["right_group"]),
        ],
        "trajectory_v24_fragment_merge_member_counts": [
            len(best["first_members"]),
            len(best["second_members"]),
        ],
        "trajectory_v24_fragment_merge_point_indices": merged_points,
        "trajectory_v24_fragment_merge_center_separation_width_factor": best[
            "center_separation_width_factor"
        ],
        "trajectory_v24_fragment_merge_longitudinal_overlap_fraction": best[
            "longitudinal_overlap_fraction"
        ],
        "trajectory_v24_fragment_merge_endpoint_gap_ratio": best[
            "endpoint_gap_ratio"
        ],
        "trajectory_v24_fragment_merge_endpoint_mismatch_width_factor": best[
            "endpoint_mismatch_width_factor"
        ],
        "trajectory_v24_fragment_merge_rms_width_factor": exported_rms_factor,
        "trajectory_v24_fragment_merge_simpler_perturbation_fraction": (
            simpler_fraction
        ),
        "trajectory_v24_fragment_merge_unique_margin": (
            unique_margin if np.isfinite(unique_margin) else None
        ),
        "trajectory_v24_memberships_frozen": True,
        "automatic_row_count": new_k,
        "automatic_selection_reason": (
            str(chosen.get("automatic_selection_reason") or "")
            + "__v24_local_continuation_fragment_merge"
        ),
        "row_count_selection_reason": "v24_local_continuation_fragment_merge",
        "model_confidence": "low",
        "sparse_row_support": any(int(value) < 3 for value in merged["cluster_sizes"]),
        "trajectory_exact_row_traits_eligible": False,
        "trajectory_pre_validation_row_count": old_k,
        "trajectory_post_validation_row_count": new_k,
        "trajectory_row_count_min": new_k,
        "trajectory_row_count_max": old_k,
        "trajectory_row_count_range_reason": "v24_fragment_merge_provisional",
        "trajectory_row_count_stable": False,
        "potential_row_oversplit_in_legacy": True,
    }
    result = dict(chosen)
    result.update(merged)
    result.update(fields)
    replaced = False
    updated_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        if int(candidate.get("k", -1)) == new_k and not replaced:
            updated_candidates.append(merged)
            replaced = True
        else:
            updated_candidates.append(candidate)
    if not replaced:
        updated_candidates.append(merged)
        updated_candidates.sort(key=lambda item: int(item["k"]))
    diagnostic.update(
        {
            **fields,
            "global_automatic_row_count": new_k,
            "row_count_used_for_traits": new_k,
            "row_count_used_for_traits_source": "v24_local_continuation_fragment_merge",
            "alternative_row_count": old_k,
            "global_row_selection_score": merged_diagnostic.get("adjusted_score"),
            "global_row_score_margin": 0.0,
            "global_row_automatic_is_strict_best": False,
            "global_row_model_confidence": "low",
            "singleton_supported_row_count": sum(
                int(value) == 1 for value in merged["cluster_sizes"]
            ),
        }
    )
    candidate_diagnostics = list(diagnostic.get("candidate_diagnostics", []))
    replaced_diagnostic = False
    for position, item in enumerate(candidate_diagnostics):
        if base.optional_int(item.get("candidate_row_count")) == new_k:
            merged_diagnostic = dict(merged_diagnostic)
            merged_diagnostic["candidate_row_count"] = new_k
            candidate_diagnostics[position] = merged_diagnostic
            replaced_diagnostic = True
            break
    if not replaced_diagnostic:
        merged_diagnostic = dict(merged_diagnostic)
        merged_diagnostic["candidate_row_count"] = new_k
        candidate_diagnostics.append(merged_diagnostic)
    diagnostic["candidate_diagnostics"] = candidate_diagnostics
    return result, updated_candidates, diagnostic


def _singleton_other_row_clearance_factor(
    chosen: dict[str, Any],
    point_index: int,
    source_group: int,
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    local_width_um: float,
) -> float:
    intercepts = np.asarray(chosen.get("intercepts_um", []), dtype=float)
    if len(intercepts) <= 1:
        return math.inf
    reference = float(chosen.get("long_reference_um") or 0.0)
    adjusted = float(lateral[point_index]) - float(chosen.get("slope") or 0.0) * (
        float(longitudinal[point_index]) - reference
    )
    distances = [
        abs(adjusted - float(intercepts[group]))
        for group in range(len(intercepts))
        if group != source_group
    ]
    return min(distances) / max(local_width_um, 1e-9) if distances else math.inf


def _v24_singleton_candidate(
    chosen: dict[str, Any],
    point_index: int,
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV24,
) -> dict[str, Any]:
    geometry = v23._v23_local_geometry(
        chosen, point_index, lateral, longitudinal, median_width_um
    )
    if not bool(geometry.get("assessable")):
        return {"point_index": point_index, "passed": False, "reason": geometry.get("reason")}
    confidence = v22._detector_confidence(point_index)
    active = v23._active_stoma_record(point_index)
    density_eligible = (
        base.parse_bool(active.get("density_eligible")) if active is not None else None
    )
    score = v23._v23_frozen_score(
        chosen, point_index, lateral, longitudinal, median_width_um, settings
    )
    improvement = _finite(score.get("adjusted_score_improvement"))
    other_clearance = _singleton_other_row_clearance_factor(
        chosen,
        point_index,
        int(geometry["source_group"]),
        lateral,
        longitudinal,
        float(geometry["local_detector_width_um"]),
    )
    remaining_longitudinal = np.sort(
        np.asarray(
            [
                float(longitudinal[index])
                for index in geometry.get("remaining_members", [])
            ],
            dtype=float,
        )
    )
    candidate_longitudinal = float(longitudinal[point_index])
    lower = remaining_longitudinal[remaining_longitudinal < candidate_longitudinal]
    upper = remaining_longitudinal[remaining_longitudinal > candidate_longitudinal]
    remaining_gaps = np.diff(np.unique(remaining_longitudinal))
    remaining_gaps = remaining_gaps[remaining_gaps > 1e-9]
    typical_remaining_gap = (
        float(np.median(remaining_gaps)) if remaining_gaps.size else None
    )
    bracketing_interval = (
        float(np.min(upper) - np.max(lower)) if lower.size and upper.size else None
    )
    bracketing_gap_ratio = (
        bracketing_interval / typical_remaining_gap
        if bracketing_interval is not None
        and typical_remaining_gap is not None
        and typical_remaining_gap > 0
        else None
    )
    failures: list[str] = []
    if int(geometry["source_member_count_before"]) < int(
        settings.v24_singleton_min_source_members
    ):
        failures.append("source_members")
    if int(geometry["source_member_count_after"]) < int(
        settings.v24_singleton_min_remaining_members
    ):
        failures.append("remaining_members")
    if not bool(geometry["longitudinally_bracketed"]):
        failures.append("not_bracketed")
    if (
        bracketing_gap_ratio is None
        or bracketing_gap_ratio > float(settings.v22_loo_max_bracketing_gap_ratio)
    ):
        failures.append("bracketing_gap")
    if confidence is None or confidence < float(
        settings.v24_singleton_min_detector_confidence
    ):
        failures.append("detector_confidence")
    if density_eligible is not True:
        failures.append("density_eligible")
    if float(geometry["candidate_residual_width_factor"]) < float(
        settings.v24_singleton_min_residual_width_factor
    ):
        failures.append("candidate_residual")
    if float(geometry["remaining_row_rms_width_factor"]) > float(
        settings.v24_singleton_max_remaining_rms_width_factor
    ):
        failures.append("remaining_rms")
    if float(geometry["residual_dominance_ratio"]) < float(
        settings.v24_singleton_min_residual_dominance
    ):
        failures.append("residual_dominance")
    if improvement is None or improvement < float(
        settings.v24_singleton_min_frozen_score_improvement
    ):
        failures.append("frozen_score_improvement")
    if other_clearance < float(
        settings.v24_singleton_min_other_row_clearance_width_factor
    ):
        failures.append("duplicates_other_row")
    return {
        "point_index": point_index,
        "passed": not failures,
        "reason": ";".join(failures),
        "geometry": geometry,
        "confidence": confidence,
        "density_eligible": density_eligible,
        "score_improvement": improvement,
        "other_row_clearance_width_factor": other_clearance,
        "bracketing_gap_ratio": bracketing_gap_ratio,
    }


def _v24_singleton_stability(
    chosen: dict[str, Any],
    nominal: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV24,
) -> float:
    point_index = int(nominal["point_index"])
    points = [int(value) for value in chosen.get("point_indices", [])]
    seed_offset = 2401 * (point_index + 1) + sum(
        (position + 1) * (value + 1) for position, value in enumerate(points)
    )
    rng = np.random.default_rng(int(settings.trajectory_stability_seed) + seed_offset)
    jitter = float(settings.trajectory_stability_jitter_width_fraction) * median_width_um
    passes = 0
    for _ in range(int(settings.trajectory_stability_iterations)):
        perturbed = np.asarray(lateral, dtype=float).copy()
        perturbed[points] += rng.normal(0.0, jitter, len(points))
        candidate = _v24_singleton_candidate(
            chosen,
            point_index,
            perturbed,
            longitudinal,
            median_width_um,
            settings,
        )
        passes += int(bool(candidate.get("passed")))
    return passes / max(int(settings.trajectory_stability_iterations), 1)


def _apply_v24_expanded_singleton(
    chosen: dict[str, Any],
    candidates: list[dict[str, Any]],
    diagnostic: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV24,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if bool(chosen.get("trajectory_v20_singleton_rescue_applied")):
        return chosen, candidates, diagnostic
    if int(chosen.get("k", 0)) >= int(settings.trajectory_max_rows_per_supergroup):
        return chosen, candidates, diagnostic

    grouped = _group_members(chosen)
    scanned: list[dict[str, Any]] = []
    for members in grouped.values():
        if len(members) < int(settings.v24_singleton_min_source_members):
            continue
        for point_index in members:
            record = _v24_singleton_candidate(
                chosen,
                point_index,
                lateral,
                longitudinal,
                median_width_um,
                settings,
            )
            if bool(record.get("passed")):
                stability = _v24_singleton_stability(
                    chosen,
                    record,
                    lateral,
                    longitudinal,
                    median_width_um,
                    settings,
                )
                record["stability_fraction"] = stability
                record["passed"] = bool(
                    stability >= float(settings.v24_singleton_min_stability_fraction)
                )
                if not record["passed"]:
                    record["reason"] = "route_stability"
            scanned.append(record)

    passed = [item for item in scanned if bool(item.get("passed"))]
    scan_fields = {
        "trajectory_v24_singleton_scan_performed": True,
        "trajectory_v24_singleton_candidate_count": len(passed),
    }
    chosen.update(scan_fields)
    diagnostic.update(scan_fields)
    if not passed:
        return chosen, candidates, diagnostic

    passed.sort(
        key=lambda item: (
            -float(item.get("stability_fraction") or 0.0),
            -float(item.get("score_improvement") or -math.inf),
            -float(item["geometry"]["residual_dominance_ratio"]),
            int(item["point_index"]),
        )
    )
    unique_margin = (
        float(passed[0].get("score_improvement") or 0.0)
        - float(passed[1].get("score_improvement") or 0.0)
        if len(passed) > 1
        else math.inf
    )
    if len(passed) > 1 and unique_margin < float(
        settings.v24_singleton_unique_score_margin
    ):
        rejection = {
            "trajectory_v24_singleton_applied": False,
            "trajectory_v24_singleton_status": "rejected_ambiguous_candidates",
            "trajectory_v24_singleton_reason": "no_unique_best_frozen_candidate",
            "trajectory_v24_singleton_unique_margin": unique_margin,
        }
        chosen.update(rejection)
        diagnostic.update(rejection)
        return chosen, candidates, diagnostic

    best = passed[0]
    point_index = int(best["point_index"])
    geometry = best["geometry"]
    refitted_template = dict(chosen)
    refitted_template["slope"] = float(geometry["refitted_slope"])
    refitted = v22._refit_fixed_membership_candidate(
        refitted_template, lateral, longitudinal
    )
    hybrid = v22._build_hybrid_singleton_candidate(
        refitted, point_index, lateral, longitudinal
    )
    sorted_intercepts = np.sort(
        np.asarray(hybrid.get("intercepts_um", []), dtype=float)
    )
    strong_gap_threshold = max(
        float(settings.minimum_band_gap_um),
        float(settings.band_gap_detector_width_factor) * median_width_um,
    )
    run_lengths: list[int] = []
    current_run = 1 if sorted_intercepts.size else 0
    for gap in np.diff(sorted_intercepts):
        if float(gap) >= strong_gap_threshold:
            run_lengths.append(current_run)
            current_run = 1
        else:
            current_run += 1
    if current_run:
        run_lengths.append(current_run)
    maximum_rows_in_strong_gap_group = max(run_lengths, default=0)
    if maximum_rows_in_strong_gap_group > int(settings.maximum_rows_per_band):
        rejection = {
            **scan_fields,
            "trajectory_v24_singleton_applied": False,
            "trajectory_v24_singleton_status": "rejected_biological_band_row_maximum",
            "trajectory_v24_singleton_reason": (
                "frozen_singleton_would_create_more_than_maximum_rows_within_"
                "one_strong_gap_group"
            ),
            "trajectory_v24_singleton_candidate_point_index": point_index,
            "trajectory_v24_singleton_candidate_stoma_id": v22._stoma_id(
                point_index
            ),
            "trajectory_v24_singleton_maximum_rows_in_strong_gap_group": (
                maximum_rows_in_strong_gap_group
            ),
            "trajectory_v24_singleton_strong_gap_threshold_um": (
                strong_gap_threshold
            ),
        }
        chosen.update(rejection)
        diagnostic.update(rejection)
        return chosen, candidates, diagnostic
    hybrid_angle = math.degrees(math.atan(float(hybrid["slope"])))
    hybrid_diagnostic = v15._score_trajectory_candidate(
        hybrid, median_width_um, hybrid_angle, settings
    )
    old_k = int(chosen["k"])
    new_k = int(hybrid["k"])
    fields = {
        **scan_fields,
        "trajectory_v24_singleton_applied": True,
        "trajectory_v24_singleton_status": "accepted_provisional",
        "trajectory_v24_singleton_reason": (
            "unique_stable_bracketed_large_row_leave_one_out_candidate"
        ),
        "trajectory_v24_singleton_candidate_point_index": point_index,
        "trajectory_v24_singleton_candidate_stoma_id": v22._stoma_id(point_index),
        "trajectory_v24_singleton_source_members_before": geometry[
            "source_member_count_before"
        ],
        "trajectory_v24_singleton_source_members_after": geometry[
            "source_member_count_after"
        ],
        "trajectory_v24_singleton_residual_width_factor": geometry[
            "candidate_residual_width_factor"
        ],
        "trajectory_v24_singleton_remaining_rms_width_factor": geometry[
            "remaining_row_rms_width_factor"
        ],
        "trajectory_v24_singleton_residual_dominance": geometry[
            "residual_dominance_ratio"
        ],
        "trajectory_v24_singleton_bracketing_gap_ratio": best[
            "bracketing_gap_ratio"
        ],
        "trajectory_v24_singleton_frozen_score_improvement": best[
            "score_improvement"
        ],
        "trajectory_v24_singleton_other_row_clearance_width_factor": best[
            "other_row_clearance_width_factor"
        ],
        "trajectory_v24_singleton_stability_fraction": best[
            "stability_fraction"
        ],
        "trajectory_v24_singleton_unique_margin": (
            unique_margin if np.isfinite(unique_margin) else None
        ),
        "trajectory_v24_memberships_frozen": True,
        "automatic_row_count": new_k,
        "automatic_selection_reason": (
            str(chosen.get("automatic_selection_reason") or "")
            + "__v24_frozen_bracketed_large_row_singleton"
        ),
        "row_count_selection_reason": "v24_frozen_bracketed_large_row_singleton",
        "model_confidence": "low",
        "sparse_row_support": True,
        "trajectory_exact_row_traits_eligible": False,
        "trajectory_singleton_validation_required": True,
        "trajectory_singleton_rows_validated": True,
        "trajectory_singleton_detector_confidences": [best["confidence"]],
        "trajectory_pre_validation_row_count": old_k,
        "trajectory_post_validation_row_count": new_k,
        "trajectory_row_count_min": old_k,
        "trajectory_row_count_max": new_k,
        "trajectory_row_count_range_reason": "v24_singleton_provisional",
        "trajectory_row_count_stable": False,
        "trajectory_v20_singleton_rescue_applied": True,
        "trajectory_v20_singleton_validation_status": "accepted_provisional",
        "trajectory_v20_singleton_validation_reason": "v24_all_safeguards_passed",
        "trajectory_v20_conservative_row_count": old_k,
        "trajectory_v20_augmented_row_count": new_k,
        "trajectory_v20_added_row_count": 1,
        "trajectory_v20_singleton_route": "within_existing_band",
        "trajectory_v20_singleton_band_boundary_ambiguous": False,
        "trajectory_v20_traits_involving_singleton_provisional": True,
        "trajectory_v22_candidate_point_index": point_index,
        "trajectory_v22_leave_one_out_applied": True,
        "trajectory_v22_leave_one_out_status": "accepted_provisional_v24_route",
        "trajectory_v22_leave_one_out_reason": "v24_all_safeguards_passed",
        "trajectory_v22_augmented_row_count": new_k,
        "trajectory_v22_stability_gate_passed": True,
        "trajectory_v22_route_stability_fraction": best["stability_fraction"],
        "trajectory_v22_adds_new_band": False,
        "trajectory_v22_nonrecursive_single_pass": True,
    }
    result = dict(chosen)
    result.update(hybrid)
    result.update(fields)
    candidates = v23._replace_candidate_and_diagnostic(
        candidates, diagnostic, hybrid, hybrid_diagnostic
    )
    diagnostic.update(
        {
            **fields,
            "global_automatic_row_count": new_k,
            "row_count_used_for_traits": new_k,
            "row_count_used_for_traits_source": "v24_frozen_bracketed_large_row_singleton",
            "alternative_row_count": old_k,
            "global_row_selection_score": hybrid_diagnostic.get("adjusted_score"),
            "global_row_score_margin": best["score_improvement"],
            "global_row_automatic_is_strict_best": False,
            "global_row_model_confidence": "low",
            "singleton_supported_row_count": 1,
            "singleton_detector_confidence_mean": best["confidence"],
        }
    )
    return result, candidates, diagnostic


def select_row_model_v24(
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV24,
    audited_row_count: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run v2.3 selection, merge continuations, then scan frozen singletons."""

    diagnostic_count = len(v14._ACTIVE_BAND_DIAGNOSTICS)
    chosen, candidates = v23.select_row_model_v23(
        point_indices,
        lateral,
        longitudinal,
        median_detector_width_um,
        settings,
        audited_row_count=audited_row_count,
    )
    if len(v14._ACTIVE_BAND_DIAGNOSTICS) != diagnostic_count + 1:
        raise RuntimeError("v2.4 expected one provisional-band diagnostic record.")
    diagnostic = v14._ACTIVE_BAND_DIAGNOSTICS[-1]
    if audited_row_count is None:
        chosen, candidates, diagnostic = _apply_v24_fragment_merge(
            chosen,
            candidates,
            diagnostic,
            lateral,
            longitudinal,
            median_detector_width_um,
            settings,
        )
        chosen, candidates, diagnostic = _apply_v24_expanded_singleton(
            chosen,
            candidates,
            diagnostic,
            lateral,
            longitudinal,
            median_detector_width_um,
            settings,
        )
    for key, default in _V24_DEFAULTS.items():
        value = chosen.get(key, diagnostic.get(key, default))
        chosen[key] = value
        diagnostic[key] = value
    return chosen, candidates


def _trajectory_diagnostic_fields_v24(
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    fields = v23._trajectory_diagnostic_fields_v23(diagnostic)
    fields["row_model_version"] = SCRIPT_VERSION
    for key, default in _V24_DEFAULTS.items():
        fields[key] = diagnostic.get(key, default)
    return fields


def _partition_template(
    provisional_band_id: str,
    split: int,
    gap_um: float,
    gap_ratio: float,
    gap_width_factor: float,
    image_support_ratio: float | None,
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "provisional_band_id": provisional_band_id,
        "split": split,
        "boundary_gap_um": gap_um,
        "gap_ratio": gap_ratio,
        "gap_width_factor": gap_width_factor,
        "left_alternation": v14._row_group_alternation(left_rows, stoma_traits),
        "right_alternation": v14._row_group_alternation(right_rows, stoma_traits),
        "image_support_ratio": image_support_ratio,
    }


def _rebuild_band_v24(
    template: dict[str, Any],
    rows: list[dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_width_um: float,
    partition: dict[str, Any] | None,
    group_index: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    provisional = str(
        template.get("provisional_band_id") or template.get("band_id") or ""
    )
    rebuilt = v14._rebuild_band_record(
        template,
        rows,
        stoma_traits,
        median_width_um,
        partition,
        group_index,
    )
    gaps = [
        v15._matched_trajectory_gap(first, second)[0]
        for first, second in zip(rows[:-1], rows[1:], strict=True)
    ]
    rebuilt.update(base.sample_summary(gaps, "within_band_interrow_spacing_um"))
    rebuilt.update(
        {
            "provisional_band_id": provisional,
            "within_band_interrow_spacing_mean_um": (
                float(np.mean(gaps)) if gaps else None
            ),
            "within_band_interrow_spacing_median_um": (
                float(np.median(gaps)) if gaps else None
            ),
            "within_band_spacing_basis": (
                "matched_longitudinal_positions_between_locally_refitted_trajectories"
            ),
            **extra,
        }
    )
    return rebuilt


def _renumber_final_bands(
    rows: list[dict[str, Any]],
    bands: list[dict[str, Any]],
    assignments: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    row_lookup = {str(row["row_id"]): row for row in rows}
    bands.sort(key=lambda band: float(band["band_center_lateral_um"]))
    for number, band in enumerate(bands, start=1):
        band_id = f"B{number:03d}"
        linked = sorted(
            [row_lookup[str(row_id)] for row_id in band.get("row_ids", [])],
            key=lambda row: base.row_lateral_at(row, 0.0),
        )
        members = sorted(
            {int(index) for row in linked for index in row.get("_point_indices", [])}
        )
        band.update(
            {
                "band_id": band_id,
                "band_order_left_to_right": number,
                "row_ids": [str(row["row_id"]) for row in linked],
                "row_count": len(linked),
                "automatic_row_count": len(linked),
                "row_count_used_for_traits": len(linked),
                "stoma_count": len(members),
                "_point_indices": members,
            }
        )
        partition_fields = {
            key: value
            for key, value in band.items()
            if key.startswith("trajectory_v24_partition_")
            or key.startswith("trajectory_v24_singleton_band_")
            or key
            in {
                "band_partition_applied",
                "band_partition_group_index",
                "band_partition_split_after_row",
                "band_partition_boundary_gap_um",
                "band_partition_boundary_to_internal_gap_ratio",
                "band_partition_boundary_gap_width_factor",
            }
        }
        for position, row in enumerate(linked, start=1):
            row.update(
                {
                    "band_id": band_id,
                    "row_position_within_band": position,
                    "rows_in_band": len(linked),
                    "automatic_rows_in_band": len(linked),
                    "band_row_count": len(linked),
                    **partition_fields,
                }
            )
            for point_index in row.get("_point_indices", []):
                assignments[int(point_index)].update(
                    {
                        "row_id": row["row_id"],
                        "band_id": band_id,
                        "row_position_within_band": position,
                        "rows_in_band": len(linked),
                        "automatic_rows_in_band": len(linked),
                        "band_row_count": len(linked),
                        **partition_fields,
                    }
                )
    return rows, bands, assignments


def _closest_partition_record(
    audit: list[dict[str, Any]],
    provisional_band_id: str,
    gap_um: float,
    split: int | None = None,
) -> dict[str, Any] | None:
    records = [
        record
        for record in audit
        if str(record.get("provisional_band_id") or "") == provisional_band_id
        and _finite(record.get("boundary_gap_um")) is not None
    ]
    supported = [
        record
        for record in records
        if _finite(record.get("image_support_ratio")) is not None
    ]
    if supported:
        records = supported
    if split is not None:
        identity_matched = [
            record
            for record in records
            if base.optional_int(record.get("split")) == int(split)
            or base.optional_int(record.get("band_partition_split_after_row"))
            == int(split)
        ]
        if identity_matched:
            records = identity_matched
    return (
        min(records, key=lambda record: abs(float(record["boundary_gap_um"]) - gap_um))
        if records
        else None
    )


def _partition_trajectory_bands_v24(
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    settings: TraitSettingsV24,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Apply v2.2 partitioning, reverse weak isolated cuts, add strong cuts."""

    rows, bands, assignments, partition_audit = v22._partition_trajectory_bands_v20(
        row_records,
        band_records,
        point_assignment,
        stoma_traits,
        median_detector_width_um,
        settings,
    )
    row_lookup = {str(row["row_id"]): row for row in rows}

    # V2.2 is allowed to isolate a rescued singleton as a separate band.  A
    # v2.4 large-row singleton is explicitly a lane within its frozen source
    # band, so reattach it before the inherited v2.2 invariant is evaluated.
    reattached = True
    while reattached:
        reattached = False
        bands.sort(key=lambda band: float(band["band_center_lateral_um"]))
        for position, band in enumerate(bands):
            linked = [row_lookup[str(value)] for value in band.get("row_ids", [])]
            if not (
                len(linked) == 1
                and int(linked[0].get("stoma_count") or 0) == 1
                and bool(linked[0].get("trajectory_v24_singleton_applied"))
            ):
                continue
            provisional = str(
                band.get("provisional_band_id") or band.get("band_id")
            )
            neighbors: list[tuple[float, int]] = []
            for neighbor_position in (position - 1, position + 1):
                if not 0 <= neighbor_position < len(bands):
                    continue
                neighbor = bands[neighbor_position]
                neighbor_provisional = str(
                    neighbor.get("provisional_band_id") or neighbor.get("band_id")
                )
                if neighbor_provisional != provisional:
                    continue
                neighbor_rows = [
                    row_lookup[str(value)] for value in neighbor.get("row_ids", [])
                ]
                if not neighbor_rows:
                    continue
                distance = min(
                    float(v15._matched_trajectory_gap(linked[0], row)[0])
                    for row in neighbor_rows
                )
                neighbors.append((distance, neighbor_position))
            if not neighbors:
                raise RuntimeError(
                    "v2.4 singleton invariant failed: rescued singleton has no "
                    "same-provisional-band neighbor for frozen reattachment."
                )
            _, neighbor_position = min(neighbors)
            neighbor = bands[neighbor_position]
            neighbor_rows = [
                row_lookup[str(value)] for value in neighbor.get("row_ids", [])
            ]
            combined = sorted(
                linked + neighbor_rows,
                key=lambda row: base.row_lateral_at(row, 0.0),
            )
            template = neighbor
            merged = _rebuild_band_v24(
                template,
                combined,
                stoma_traits,
                median_detector_width_um,
                None,
                1,
                {
                    "trajectory_v24_singleton_band_reattached": True,
                    "trajectory_v24_singleton_band_reattachment_reason": (
                        "frozen_large_row_singleton_cannot_form_isolated_band"
                    ),
                    "v20_singleton_band": False,
                    "v20_singleton_partition_applied": False,
                },
            )
            for row in combined:
                row["v20_singleton_band"] = False
                row["v20_singleton_partition_applied"] = False
                for point_index in row.get("_point_indices", []):
                    assignments[int(point_index)]["v20_singleton_band"] = False
                    assignments[int(point_index)]["v20_singleton_partition_applied"] = False
            low = min(position, neighbor_position)
            high = max(position, neighbor_position)
            bands[low : high + 1] = [merged]
            partition_audit.append(
                {
                    "provisional_band_id": provisional,
                    "accepted": False,
                    "trajectory_v24_singleton_band_reattached": True,
                    "trajectory_v24_singleton_band_reattachment_reason": (
                        "frozen_large_row_singleton_cannot_form_isolated_band"
                    ),
                }
            )
            reattached = True
            break

    # Reverse an inherited cut only when it separates two pieces of the same
    # provisional band, one side is a supported one-row band, and the raw-image
    # boundary evidence does not reach the v2.4 threshold.
    changed = True
    while changed:
        changed = False
        bands.sort(key=lambda band: float(band["band_center_lateral_um"]))
        for position in range(len(bands) - 1):
            first = bands[position]
            second = bands[position + 1]
            first_provisional = str(
                first.get("provisional_band_id") or first.get("band_id")
            )
            second_provisional = str(
                second.get("provisional_band_id") or second.get("band_id")
            )
            if first_provisional != second_provisional:
                continue
            first_rows = sorted(
                [row_lookup[str(value)] for value in first.get("row_ids", [])],
                key=lambda row: base.row_lateral_at(row, 0.0),
            )
            second_rows = sorted(
                [row_lookup[str(value)] for value in second.get("row_ids", [])],
                key=lambda row: base.row_lateral_at(row, 0.0),
            )
            if min(len(first_rows), len(second_rows)) != 1:
                continue
            isolated = first_rows if len(first_rows) == 1 else second_rows
            isolated_stomata = sum(
                int(row.get("stoma_count") or 0) for row in isolated
            )
            if isolated_stomata < int(settings.v24_partition_min_stomata_per_side):
                continue
            boundary_gap = v15._matched_trajectory_gap(first_rows[-1], second_rows[0])[0]
            inherited_split = base.optional_int(
                first.get("band_partition_split_after_row")
            )
            inherited = _closest_partition_record(
                partition_audit,
                first_provisional,
                float(boundary_gap),
                inherited_split,
            )
            if inherited is None or not bool(inherited.get("accepted")):
                continue
            image_support = _finite(inherited.get("image_support_ratio"))
            if image_support is not None and image_support >= float(
                settings.v24_partition_reverse_single_row_max_image_support_ratio
            ):
                inherited.update(
                    {
                        "trajectory_v24_partition_parent_accepted": True,
                        "trajectory_v24_partition_final_accepted": True,
                        "trajectory_v24_partition_split_applied": True,
                        "trajectory_v24_partition_split_reason": (
                            "inherited_strong_isolated_row_cut_confirmed_by_raw_image_support"
                        ),
                        "trajectory_v24_single_row_side_requires_image_support": True,
                        "trajectory_v24_partition_one_decision_limit": True,
                    }
                )
                for record in (first, second):
                    record.update(
                        {
                            "trajectory_v24_partition_parent_accepted": True,
                            "trajectory_v24_partition_final_accepted": True,
                            "trajectory_v24_partition_split_applied": True,
                            "trajectory_v24_partition_split_reason": (
                                "inherited_strong_isolated_row_cut_confirmed_by_raw_image_support"
                            ),
                            "trajectory_v24_partition_image_support_ratio": image_support,
                            "trajectory_v24_single_row_side_requires_image_support": True,
                            "trajectory_v24_partition_one_decision_limit": True,
                        }
                    )
                continue
            inherited.update(
                {
                    "trajectory_v24_partition_parent_accepted": True,
                    "trajectory_v24_partition_final_accepted": False,
                    "trajectory_v24_partition_reversal_applied": True,
                    "trajectory_v24_partition_reversal_reason": (
                        "raw_image_support_below_v24_isolated_row_threshold"
                    ),
                }
            )
            linked = sorted(
                first_rows + second_rows,
                key=lambda row: base.row_lateral_at(row, 0.0),
            )
            merged = _rebuild_band_v24(
                first,
                linked,
                stoma_traits,
                median_detector_width_um,
                None,
                1,
                {
                    "trajectory_v24_partition_parent_accepted": True,
                    "trajectory_v24_partition_final_accepted": False,
                    "trajectory_v24_partition_reversal_applied": True,
                    "trajectory_v24_partition_reversal_reason": (
                        "isolated_multipoint_row_split_lacked_required_raw_image_support"
                    ),
                    "trajectory_v24_partition_image_support_ratio": image_support,
                    "trajectory_v24_single_row_side_requires_image_support": True,
                    "trajectory_v24_partition_one_decision_limit": True,
                },
            )
            partition_audit.append(
                {
                    **_partition_template(
                        first_provisional,
                        len(first_rows),
                        float(boundary_gap),
                        _finite(inherited.get("gap_ratio")) or math.inf,
                        float(boundary_gap) / max(median_detector_width_um, 1e-9),
                        image_support,
                        first_rows,
                        second_rows,
                        stoma_traits,
                    ),
                    "accepted": False,
                    "trajectory_v24_partition_reversal_applied": True,
                    "trajectory_v24_partition_reversal_reason": (
                        "raw_image_support_below_v24_isolated_row_threshold"
                    ),
                }
            )
            bands[position : position + 2] = [merged]
            changed = True
            break

    # Add one strong, image-supported cut per remaining band.  The route is
    # deliberately limited to a cut that isolates one multipoint row.
    rebuilt_bands: list[dict[str, Any]] = []
    for band in bands:
        linked = sorted(
            [row_lookup[str(value)] for value in band.get("row_ids", [])],
            key=lambda row: base.row_lateral_at(row, 0.0),
        )
        if len(linked) < 2 or bool(
            band.get("trajectory_v24_partition_reversal_applied")
        ):
            rebuilt_bands.append(band)
            continue
        gaps = [
            float(v15._matched_trajectory_gap(first, second)[0])
            for first, second in zip(linked[:-1], linked[1:], strict=True)
        ]
        candidates: list[dict[str, Any]] = []
        provisional = str(band.get("provisional_band_id") or band.get("band_id"))
        for cut, gap_um in enumerate(gaps, start=1):
            left = linked[:cut]
            right = linked[cut:]
            if min(len(left), len(right)) != 1:
                continue
            left_stomata = sum(int(row.get("stoma_count") or 0) for row in left)
            right_stomata = sum(int(row.get("stoma_count") or 0) for row in right)
            if min(left_stomata, right_stomata) < int(
                settings.v24_partition_min_stomata_per_side
            ):
                continue
            other = [value for index, value in enumerate(gaps) if index != cut - 1]
            internal_reference = (
                float(np.median(np.asarray(other, dtype=float))) if other else 0.0
            )
            gap_ratio = gap_um / max(internal_reference, 1e-9)
            gap_factor = gap_um / max(median_detector_width_um, 1e-9)
            inherited = _closest_partition_record(
                partition_audit, provisional, gap_um, cut
            )
            image_support = (
                _finite(inherited.get("image_support_ratio"))
                if inherited is not None
                else None
            )
            stability = v22._strong_gap_stability(
                gap_um,
                median_detector_width_um,
                settings,
                sum(ord(char) for char in provisional) + cut + 2400,
            )
            if not (
                gap_factor >= float(settings.v24_partition_min_gap_width_factor)
                and gap_ratio >= float(settings.v24_partition_min_gap_ratio)
                and image_support is not None
                and image_support >= float(settings.v24_partition_min_image_support_ratio)
                and stability >= float(settings.v24_partition_min_stability_fraction)
            ):
                continue
            candidates.append(
                {
                    "cut": cut,
                    "gap_um": gap_um,
                    "gap_ratio": gap_ratio,
                    "gap_factor": gap_factor,
                    "image_support": image_support,
                    "stability": stability,
                    "left": left,
                    "right": right,
                }
            )
        if not candidates:
            rebuilt_bands.append(band)
            continue
        chosen_cut = max(
            candidates,
            key=lambda item: (
                float(item["gap_factor"]),
                float(item["image_support"]),
            ),
        )
        partition = _partition_template(
            provisional,
            int(chosen_cut["cut"]),
            float(chosen_cut["gap_um"]),
            float(chosen_cut["gap_ratio"]),
            float(chosen_cut["gap_factor"]),
            float(chosen_cut["image_support"]),
            chosen_cut["left"],
            chosen_cut["right"],
            stoma_traits,
        )
        for group_index, group in enumerate(
            [chosen_cut["left"], chosen_cut["right"]], start=1
        ):
            rebuilt_bands.append(
                _rebuild_band_v24(
                    band,
                    group,
                    stoma_traits,
                    median_detector_width_um,
                    partition,
                    group_index,
                    {
                        "trajectory_v24_partition_parent_accepted": bool(
                            _closest_partition_record(
                                partition_audit,
                                provisional,
                                float(chosen_cut["gap_um"]),
                                int(chosen_cut["cut"]),
                            )
                            and _closest_partition_record(
                                partition_audit,
                                provisional,
                                float(chosen_cut["gap_um"]),
                                int(chosen_cut["cut"]),
                            ).get("accepted")
                        ),
                        "trajectory_v24_partition_final_accepted": True,
                        "trajectory_v24_partition_split_applied": True,
                        "trajectory_v24_partition_split_reason": (
                            "strong_gap_plus_raw_image_support_isolates_supported_row_band"
                        ),
                        "trajectory_v24_partition_image_support_ratio": chosen_cut[
                            "image_support"
                        ],
                        "trajectory_v24_partition_stability_fraction": chosen_cut[
                            "stability"
                        ],
                        "trajectory_v24_single_row_side_requires_image_support": True,
                        "trajectory_v24_partition_one_decision_limit": True,
                    },
                )
            )
        partition_audit.append(
            {
                **partition,
                "accepted": True,
                "boundary_stability_fraction": chosen_cut["stability"],
                "trajectory_v24_partition_final_accepted": True,
                "trajectory_v24_partition_split_applied": True,
                "trajectory_v24_partition_split_reason": (
                    "strong_gap_plus_raw_image_support_isolates_supported_row_band"
                ),
                "trajectory_v24_partition_one_decision_limit": True,
            }
        )

    rows, rebuilt_bands, assignments = _renumber_final_bands(
        rows, rebuilt_bands, assignments
    )
    return rows, rebuilt_bands, assignments, partition_audit


def _linear_interval(
    coefficient: float,
    intercept: float,
    lower: float,
    upper: float,
) -> tuple[float, float] | None:
    if abs(coefficient) <= 1e-12:
        return (-math.inf, math.inf) if lower <= intercept <= upper else None
    first = (lower - intercept) / coefficient
    second = (upper - intercept) / coefficient
    return min(first, second), max(first, second)


def _visible_row_longitudinal_interval(
    row: dict[str, Any],
    axis: base.AxisTransform,
    width_um: float,
    height_um: float,
) -> tuple[float, float] | None:
    slope = float(row.get("row_common_slope_lateral_per_longitudinal") or 0.0)
    reference = float(row.get("row_longitudinal_reference_um") or 0.0)
    intercept = float(row["row_intercept_lateral_um"]) - slope * reference
    sine = math.sin(axis.radians)
    cosine = math.cos(axis.radians)
    x_coefficient = cosine * slope + sine
    x_intercept = cosine * intercept + axis.origin_x_um
    y_coefficient = -sine * slope + cosine
    y_intercept = -sine * intercept + axis.origin_y_um
    x_interval = _linear_interval(x_coefficient, x_intercept, 0.0, width_um)
    y_interval = _linear_interval(y_coefficient, y_intercept, 0.0, height_um)
    if x_interval is None or y_interval is None:
        return None
    lower = max(x_interval[0], y_interval[0])
    upper = min(x_interval[1], y_interval[1])
    return (lower, upper) if lower <= upper else None


def _row_endpoint_evidence(
    row: dict[str, Any],
    stoma_traits: list[dict[str, Any]],
    axis: base.AxisTransform | None,
    width_um: float | None,
    height_um: float | None,
) -> dict[str, Any]:
    members = [int(value) for value in row.get("_point_indices", [])]
    if not members or axis is None or width_um is None or height_um is None:
        return {"assessable": False}
    visible = _visible_row_longitudinal_interval(row, axis, width_um, height_um)
    if visible is None:
        return {"assessable": False}
    observed = np.asarray(
        [float(stoma_traits[index]["longitudinal_coordinate_um"]) for index in members],
        dtype=float,
    )
    detector_heights = [
        value
        for value in (
            _finite(stoma_traits[index].get("prediction_height_um"))
            or _finite(stoma_traits[index].get("prediction_width_um"))
            for index in members
        )
        if value is not None and value > 0
    ]
    detector_height = (
        float(np.median(np.asarray(detector_heights, dtype=float)))
        if detector_heights
        else None
    )
    visible_span = max(0.0, float(visible[1]) - float(visible[0]))
    observed_span = float(np.ptp(observed)) if observed.size else 0.0
    observed_coverage = (
        observed_span / visible_span if visible_span > 1e-12 else None
    )
    low_clearance = max(0.0, float(np.min(observed)) - visible[0])
    high_clearance = max(0.0, visible[1] - float(np.max(observed)))
    if low_clearance <= high_clearance:
        nearest = low_clearance
        side = "low_longitudinal"
    else:
        nearest = high_clearance
        side = "high_longitudinal"
    return {
        "assessable": detector_height is not None,
        "visible_longitudinal_min_um": visible[0],
        "visible_longitudinal_max_um": visible[1],
        "observed_longitudinal_min_um": float(np.min(observed)),
        "observed_longitudinal_max_um": float(np.max(observed)),
        "observed_longitudinal_span_um": observed_span,
        "visible_longitudinal_span_um": visible_span,
        "observed_coverage_of_visible_trajectory": observed_coverage,
        "member_count": len(members),
        "low_endpoint_clearance_um": low_clearance,
        "high_endpoint_clearance_um": high_clearance,
        "nearest_endpoint_side": side,
        "nearest_endpoint_clearance_um": nearest,
        "median_detector_height_um": detector_height,
        "nearest_endpoint_clearance_detector_heights": (
            nearest / detector_height if detector_height is not None else None
        ),
    }


def _lateral_sparse_border_evidence(
    band: dict[str, Any],
    is_outermost: bool,
    row_count: int,
    stoma_count: int,
    settings: TraitSettingsV24,
) -> dict[str, Any]:
    """Corroborate a one-stoma outer-band fragment on the relevant side."""

    assessed_side = str(band.get("band_lateral_assessed_side") or "")
    scale = _finite(band.get("band_edge_scale_detector_box_width_pixels"))
    left = _finite(band.get("band_detector_box_left_edge_clearance_pixels"))
    right = _finite(band.get("band_detector_box_right_edge_clearance_pixels"))
    candidates: list[tuple[str, float]] = []
    if assessed_side == "left" and left is not None:
        candidates.append(("left", left))
    elif assessed_side == "right" and right is not None:
        candidates.append(("right", right))
    else:
        if bool(band.get("band_lateral_is_leftmost_band")) and left is not None:
            candidates.append(("left", left))
        if bool(band.get("band_lateral_is_rightmost_band")) and right is not None:
            candidates.append(("right", right))
    if not candidates or scale is None or scale <= 0:
        return {
            "assessable": False,
            "candidate": False,
            "assessed_side": assessed_side or "unknown",
            "detector_box_clearance_pixels": None,
            "detector_box_clearance_width_factor": None,
        }
    side, clearance = min(candidates, key=lambda item: item[1])
    ratio = clearance / scale
    threshold = float(
        settings.v24_edge_lateral_sparse_max_bbox_clearance_width_factor
    )
    candidate = bool(
        is_outermost
        and row_count == 1
        and stoma_count == 1
        and ratio <= threshold
    )
    return {
        "assessable": True,
        "candidate": candidate,
        "assessed_side": side,
        "detector_box_clearance_pixels": clearance,
        "detector_box_clearance_width_factor": ratio,
        "maximum_clearance_width_factor": threshold,
        "decision_source": (
            "outermost_single_row_single_stoma_detector_box_clearance"
        ),
    }


def _remark_partial_band_eligibility_v24(
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    trajectory_slope: float,
    settings: TraitSettingsV24,
    parent_audit: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Recompute final eligibility using explicit two-axis edge evidence."""

    if parent_audit is None:
        parent_audit = v19._remark_partial_band_eligibility_v19(
            row_records,
            band_records,
            point_assignment,
            stoma_traits,
            median_detector_width_um,
            trajectory_slope,
            settings,
        )
    audit_by_band = {str(item.get("band_id")): item for item in parent_audit}
    row_lookup = {str(row["row_id"]): row for row in row_records}
    ordered = sorted(
        band_records, key=lambda band: int(band.get("band_order_left_to_right") or 0)
    )
    outermost = {
        str(ordered[0]["band_id"]) if ordered else "",
        str(ordered[-1]["band_id"]) if ordered else "",
    }
    metadata = v14._ACTIVE_IMAGE_METADATA
    axis = v14._active_axis(stoma_traits)
    px_x = _finite(metadata.get("metadata_pixels_per_um_x")) if metadata else None
    px_y = _finite(metadata.get("metadata_pixels_per_um_y")) if metadata else None
    width_pixels = base.optional_int(metadata.get("image_width")) if metadata else None
    height_pixels = base.optional_int(metadata.get("image_height")) if metadata else None
    width_um = width_pixels / px_x if width_pixels is not None and px_x else None
    height_um = height_pixels / px_y if height_pixels is not None and px_y else None

    revised: list[dict[str, Any]] = []
    for band in band_records:
        band_id = str(band["band_id"])
        rows = [row_lookup[str(value)] for value in band.get("row_ids", [])]
        row_evidence = [
            {"row_id": row["row_id"], **_row_endpoint_evidence(
                row, stoma_traits, axis, width_um, height_um
            )}
            for row in rows
        ]
        coverage = _finite(band.get("band_longitudinal_coverage_fraction"))
        relative_span = _finite(band.get("band_relative_observed_span"))
        observed_span = _finite(band.get("band_observed_longitudinal_span_um"))
        stoma_count = int(band.get("stoma_count") or 0)
        row_count = len(rows)
        row_level_complete = bool(
            rows
            and all(
                bool(item.get("assessable"))
                and int(item.get("member_count") or 0) >= 2
                and _finite(item.get("observed_coverage_of_visible_trajectory"))
                is not None
                and float(item["observed_coverage_of_visible_trajectory"])
                >= float(settings.v24_edge_complete_min_row_coverage)
                for item in row_evidence
            )
        )
        complete_support = bool(
            coverage is not None
            and relative_span is not None
            and coverage >= float(settings.v24_edge_complete_min_coverage)
            and relative_span >= float(settings.v24_edge_complete_min_relative_span)
            and stoma_count >= int(settings.v24_edge_complete_min_stomata)
            and (row_count <= 1 or row_level_complete)
        )
        severe_short = bool(
            observed_span is not None
            and (
                observed_span <= 0.0
                or (
                    coverage is not None
                    and relative_span is not None
                    and coverage < float(settings.v24_edge_short_max_coverage)
                    and relative_span < float(settings.v24_edge_short_max_relative_span)
                )
            )
        )
        endpoint_rows = [
            item
            for item in row_evidence
            if bool(item.get("assessable"))
            and _finite(item.get("nearest_endpoint_clearance_detector_heights"))
            is not None
            and float(item["nearest_endpoint_clearance_detector_heights"])
            <= float(settings.v24_edge_max_endpoint_clearance_detector_heights)
        ]
        endpoint_corroborated = False
        endpoint_side = ""
        if len(rows) == 1:
            endpoint_corroborated = len(endpoint_rows) == 1
            endpoint_side = (
                str(endpoint_rows[0].get("nearest_endpoint_side") or "")
                if endpoint_rows
                else ""
            )
        elif rows and len(endpoint_rows) == len(rows):
            sides = {str(item.get("nearest_endpoint_side") or "") for item in endpoint_rows}
            endpoint_corroborated = len(sides) == 1
            endpoint_side = next(iter(sides)) if len(sides) == 1 else ""

        is_outermost = band_id in outermost
        previous_half_cut = bool(band.get("band_lateral_half_cut"))
        previous_lateral_state = str(
            band.get("band_lateral_edge_assessment_state") or ""
        )
        previous_lateral_source = str(
            band.get("band_lateral_half_cut_decision_source") or ""
        )
        previous_half_cut_side = str(
            band.get("band_lateral_half_cut_side") or "none"
        )
        sparse_lateral = _lateral_sparse_border_evidence(
            band,
            is_outermost,
            row_count,
            stoma_count,
            settings,
        )
        lateral_sparse_fragment = bool(sparse_lateral.get("candidate"))
        longitudinal_fragment = bool(
            is_outermost
            and severe_short
            and endpoint_corroborated
        )
        exclusion_candidate = bool(
            is_outermost
            and (
                previous_half_cut
                or lateral_sparse_fragment
                or longitudinal_fragment
            )
        )
        adequate_veto_applied = bool(complete_support and exclusion_candidate)
        excluded = bool(exclusion_candidate and not adequate_veto_applied)
        effective_lateral_half_cut = bool(
            previous_half_cut
            and not adequate_veto_applied
            and not lateral_sparse_fragment
        )
        lateral_half_cut_excluded = bool(
            excluded and effective_lateral_half_cut
        )
        lateral_sparse_excluded = bool(
            excluded and lateral_sparse_fragment
        )
        longitudinal_excluded = bool(
            excluded
            and longitudinal_fragment
            and not effective_lateral_half_cut
            and not lateral_sparse_excluded
        )
        if adequate_veto_applied:
            state = "complete_supported"
            reason = "adequate_longitudinal_support_vetoed_edge_proximity_exclusion"
        elif lateral_sparse_excluded:
            state = "lateral_sparse_border_fragment"
            reason = (
                "outermost_single_row_single_stoma_detector_box_near_lateral_edge"
            )
        elif effective_lateral_half_cut:
            state = "lateral_half_cut"
            reason = "lateral_band_cross_section_half_cut_by_source_image_edge"
        elif longitudinal_excluded:
            state = "longitudinal_edge_fragment"
            reason = "short_longitudinal_support_with_source_endpoint_corroboration"
        elif is_outermost:
            state = "edge_uncertain"
            reason = "outermost_band_without_confirmed_truncation_retained"
        else:
            state = "internal_not_assessed"
            reason = "internal_band_not_subject_to_edge_exclusion"
        eligible = not excluded
        old_eligible = bool(band.get("band_spatial_eligible"))
        old_reason = str(band.get("band_spatial_exclusion_reason") or "")
        if adequate_veto_applied and previous_half_cut:
            effective_lateral_state = "v24_adequate_support_veto"
            effective_lateral_source = "v24_adequate_longitudinal_support_veto"
        elif lateral_sparse_fragment:
            effective_lateral_state = "v24_lateral_sparse_border_fragment"
            effective_lateral_source = str(
                sparse_lateral.get("decision_source") or "v24_lateral_sparse_border"
            )
        else:
            effective_lateral_state = previous_lateral_state
            effective_lateral_source = previous_lateral_source
        completeness_eligible = bool(
            eligible
            and base.parse_bool(
                band.get("band_longitudinal_sampling_complete")
            )
            is True
        )
        band.update(
            {
                "partial_band_eligibility_version": SCRIPT_VERSION,
                "band_previous_v19_spatial_eligible": old_eligible,
                "band_previous_v19_spatial_exclusion_reason": old_reason,
                "band_previous_v19_lateral_half_cut": previous_half_cut,
                "band_previous_v19_lateral_half_cut_side": previous_half_cut_side,
                "band_previous_v19_lateral_edge_assessment_state": (
                    previous_lateral_state
                ),
                "band_previous_v19_lateral_half_cut_decision_source": (
                    previous_lateral_source
                ),
                "band_v24_effective_lateral_half_cut": effective_lateral_half_cut,
                "band_lateral_half_cut": effective_lateral_half_cut,
                "band_lateral_half_cut_side": (
                    previous_half_cut_side if effective_lateral_half_cut else "none"
                ),
                "band_confirmed_lateral_half_cut": effective_lateral_half_cut,
                "band_lateral_edge_assessment_state": effective_lateral_state,
                "band_lateral_half_cut_decision_source": effective_lateral_source,
                "band_complete_topology_eligible": eligible,
                "band_edge_topology_excluded": excluded,
                "band_edge_fragment_excluded": excluded,
                "band_lateral_half_cut_excluded": lateral_half_cut_excluded,
                "band_lateral_sparse_border_excluded": lateral_sparse_excluded,
                "band_longitudinal_edge_fragment_excluded": longitudinal_excluded,
                "band_edge_completeness_state": state,
                "band_edge_completeness_reason": reason,
                "band_edge_exclusion_candidate": exclusion_candidate,
                "band_edge_complete_support_available": complete_support,
                "band_edge_complete_row_level_support": row_level_complete,
                "band_edge_adequate_support_veto_applied": adequate_veto_applied,
                "band_edge_endpoint_corroborated": endpoint_corroborated,
                "band_edge_endpoint_side": endpoint_side,
                "band_edge_lateral_sparse_border_candidate": (
                    lateral_sparse_fragment
                ),
                "band_edge_lateral_sparse_border_evidence": sparse_lateral,
                "band_edge_row_endpoint_evidence": row_evidence,
                "band_edge_short_max_coverage_threshold": float(
                    settings.v24_edge_short_max_coverage
                ),
                "band_edge_short_max_relative_span_threshold": float(
                    settings.v24_edge_short_max_relative_span
                ),
                "band_edge_complete_min_coverage_threshold": float(
                    settings.v24_edge_complete_min_coverage
                ),
                "band_edge_complete_min_relative_span_threshold": float(
                    settings.v24_edge_complete_min_relative_span
                ),
                "band_edge_complete_min_stomata_threshold": int(
                    settings.v24_edge_complete_min_stomata
                ),
                "band_edge_complete_min_row_coverage_threshold": float(
                    settings.v24_edge_complete_min_row_coverage
                ),
                "band_edge_lateral_sparse_max_bbox_clearance_width_factor": float(
                    settings.v24_edge_lateral_sparse_max_bbox_clearance_width_factor
                ),
                "band_edge_max_endpoint_clearance_detector_heights": float(
                    settings.v24_edge_max_endpoint_clearance_detector_heights
                ),
                "band_row_structure_eligible": eligible,
                "band_spatial_eligible": eligible,
                "band_fov_truncated": excluded,
                "band_spatial_exclusion_reason": reason if excluded else "",
                "band_completeness_dependent_traits_eligible": completeness_eligible,
            }
        )
        evidence_by_row = {str(item["row_id"]): item for item in row_evidence}
        for row in rows:
            endpoint = evidence_by_row.get(str(row["row_id"]), {})
            row.update(
                {
                    "partial_band_eligibility_version": SCRIPT_VERSION,
                    "row_band_previous_v19_lateral_half_cut": previous_half_cut,
                    "row_band_v24_effective_lateral_half_cut": (
                        effective_lateral_half_cut
                    ),
                    "row_band_lateral_half_cut": effective_lateral_half_cut,
                    "row_band_lateral_half_cut_side": (
                        previous_half_cut_side if effective_lateral_half_cut else "none"
                    ),
                    "row_complete_topology_eligible": eligible,
                    "row_edge_topology_excluded": excluded,
                    "row_edge_fragment_excluded": excluded,
                    "row_lateral_half_cut_excluded": lateral_half_cut_excluded,
                    "row_lateral_sparse_border_excluded": lateral_sparse_excluded,
                    "row_longitudinal_edge_fragment_excluded": (
                        longitudinal_excluded
                    ),
                    "row_edge_completeness_state": state,
                    "row_edge_completeness_reason": reason,
                    "row_edge_exclusion_candidate": exclusion_candidate,
                    "row_edge_complete_support_available": complete_support,
                    "row_edge_adequate_support_veto_applied": (
                        adequate_veto_applied
                    ),
                    "row_edge_endpoint_clearance_detector_heights": endpoint.get(
                        "nearest_endpoint_clearance_detector_heights"
                    ),
                    "row_structure_eligible": eligible,
                    "row_band_spatial_eligible": eligible,
                    "row_band_spatial_exclusion_reason": reason if excluded else "",
                    "row_completeness_dependent_traits_eligible": (
                        completeness_eligible
                    ),
                }
            )
            for point_index in row.get("_point_indices", []):
                point_assignment[int(point_index)].update(
                    {
                        "partial_band_eligibility_version": SCRIPT_VERSION,
                        "row_band_previous_v19_lateral_half_cut": previous_half_cut,
                        "row_band_v24_effective_lateral_half_cut": (
                            effective_lateral_half_cut
                        ),
                        "row_band_lateral_half_cut": effective_lateral_half_cut,
                        "row_band_lateral_half_cut_side": (
                            previous_half_cut_side
                            if effective_lateral_half_cut
                            else "none"
                        ),
                        "row_complete_topology_eligible": eligible,
                        "row_edge_topology_excluded": excluded,
                        "row_edge_fragment_excluded": excluded,
                        "row_lateral_half_cut_excluded": (
                            lateral_half_cut_excluded
                        ),
                        "row_lateral_sparse_border_excluded": (
                            lateral_sparse_excluded
                        ),
                        "row_longitudinal_edge_fragment_excluded": (
                            longitudinal_excluded
                        ),
                        "row_edge_completeness_state": state,
                        "row_edge_completeness_reason": reason,
                        "row_edge_exclusion_candidate": exclusion_candidate,
                        "row_edge_complete_support_available": complete_support,
                        "row_edge_adequate_support_veto_applied": (
                            adequate_veto_applied
                        ),
                        "row_structure_eligible": eligible,
                        "row_band_spatial_eligible": eligible,
                        "row_band_spatial_exclusion_reason": reason if excluded else "",
                        "row_completeness_dependent_traits_eligible": (
                            completeness_eligible
                        ),
                    }
                )
        record = dict(audit_by_band.get(band_id, {}))
        record.update(
            {
                "band_id": band_id,
                "provisional_band_id": band.get("provisional_band_id"),
                "partial_band_eligibility_version": SCRIPT_VERSION,
                "spatial_eligible": eligible,
                "row_structure_eligible": eligible,
                "complete_topology_eligible": eligible,
                "fov_truncated": excluded,
                "lateral_half_cut": effective_lateral_half_cut,
                "lateral_half_cut_side": (
                    previous_half_cut_side if effective_lateral_half_cut else "none"
                ),
                "confirmed_lateral_half_cut": effective_lateral_half_cut,
                "previous_v19_lateral_half_cut": previous_half_cut,
                "edge_topology_excluded": excluded,
                "edge_fragment_excluded": excluded,
                "lateral_half_cut_excluded": lateral_half_cut_excluded,
                "lateral_sparse_border_excluded": lateral_sparse_excluded,
                "longitudinal_edge_fragment_excluded": longitudinal_excluded,
                "edge_completeness_state": state,
                "edge_completeness_reason": reason,
                "edge_exclusion_candidate": exclusion_candidate,
                "complete_support_available": complete_support,
                "complete_row_level_support": row_level_complete,
                "adequate_support_veto_applied": adequate_veto_applied,
                "endpoint_corroborated": endpoint_corroborated,
                "endpoint_side": endpoint_side,
                "lateral_sparse_border_evidence": sparse_lateral,
                "row_endpoint_evidence": row_evidence,
                "completeness_dependent_traits_eligible": completeness_eligible,
                "exclusion_reason": reason if excluded else "",
            }
        )
        revised.append(record)
    return revised


def _refit_rows_locally(
    rows: list[dict[str, Any]],
    bands: list[dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
) -> None:
    lateral = np.asarray(
        [float(record["lateral_coordinate_um"]) for record in stoma_traits], dtype=float
    )
    longitudinal = np.asarray(
        [float(record["longitudinal_coordinate_um"]) for record in stoma_traits],
        dtype=float,
    )
    row_lookup = {str(row["row_id"]): row for row in rows}
    for band in bands:
        linked = [row_lookup[str(value)] for value in band.get("row_ids", [])]
        fitted_slopes: list[float] = []
        for row in linked:
            members = [int(value) for value in row.get("_point_indices", [])]
            if len(members) < 2:
                continue
            intercept, slope, rms, maximum = v23._fit_local_line(
                members, lateral, longitudinal
            )
            row.update(
                {
                    "row_intercept_lateral_um": intercept,
                    "row_common_slope_lateral_per_longitudinal": slope,
                    "row_longitudinal_reference_um": 0.0,
                    "row_rms_lateral_residual_um": rms,
                    "row_orientation_relative_leaf_axis_deg": math.degrees(
                        math.atan(slope)
                    ),
                    "row_tilt_slope_source": "v24_independent_local_row_refit",
                    "trajectory_v24_local_row_refit_applied": True,
                    "trajectory_v24_local_row_refit_max_residual_um": maximum,
                }
            )
            fitted_slopes.append(slope)
        inherited_slope = (
            float(np.median(np.asarray(fitted_slopes, dtype=float)))
            if fitted_slopes
            else 0.0
        )
        for row in linked:
            members = [int(value) for value in row.get("_point_indices", [])]
            if len(members) != 1:
                continue
            point_index = members[0]
            intercept = float(lateral[point_index]) - inherited_slope * float(
                longitudinal[point_index]
            )
            row.update(
                {
                    "row_intercept_lateral_um": intercept,
                    "row_common_slope_lateral_per_longitudinal": inherited_slope,
                    "row_longitudinal_reference_um": 0.0,
                    "row_rms_lateral_residual_um": 0.0,
                    "row_orientation_relative_leaf_axis_deg": math.degrees(
                        math.atan(inherited_slope)
                    ),
                    "row_tilt_slope_source": "v24_singleton_inherits_band_median_local_slope",
                    "trajectory_v24_local_row_refit_applied": False,
                }
            )


def _refresh_band_geometry(
    rows: list[dict[str, Any]],
    bands: list[dict[str, Any]],
    assignments: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_width_um: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    row_lookup = {str(row["row_id"]): row for row in rows}
    refreshed: list[dict[str, Any]] = []
    for band in bands:
        linked = sorted(
            [row_lookup[str(value)] for value in band.get("row_ids", [])],
            key=lambda row: base.row_lateral_at(row, 0.0),
        )
        preserved = {
            key: value
            for key, value in band.items()
            if key.startswith("trajectory_v24_")
            or key.startswith("band_partition_")
            or key in {"provisional_band_id", "v20_singleton_band"}
        }
        record = _rebuild_band_v24(
            band,
            linked,
            stoma_traits,
            median_width_um,
            None,
            1,
            preserved,
        )
        refreshed.append(record)
    return _renumber_final_bands(rows, refreshed, assignments)


def _assert_v24_invariants(
    stoma_traits: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    bands: list[dict[str, Any]],
    assignments: dict[int, dict[str, Any]],
    eligibility_before: dict[str, list[bool | None]],
    settings: TraitSettingsV24,
) -> None:
    source = (
        str(stoma_traits[0].get("source_image_name") or "unknown_image")
        if stoma_traits
        else "unknown_image"
    )
    if set(assignments) != set(range(len(stoma_traits))):
        raise RuntimeError(
            f"v2.4 invariant failed: incomplete point assignment; source_image={source}."
        )
    row_ids = [str(row["row_id"]) for row in rows]
    band_ids = [str(band["band_id"]) for band in bands]
    if len(row_ids) != len(set(row_ids)) or len(band_ids) != len(set(band_ids)):
        raise RuntimeError(
            f"v2.4 invariant failed: duplicate row/band identifier; source_image={source}."
        )
    expected_bands = [f"B{index:03d}" for index in range(1, len(bands) + 1)]
    if band_ids != expected_bands:
        raise RuntimeError(
            f"v2.4 invariant failed: non-contiguous final band numbering; source_image={source}."
        )
    row_lookup = {str(row["row_id"]): row for row in rows}
    seen_points: list[int] = []
    previous_center = -math.inf
    for band in bands:
        linked = [row_lookup[str(value)] for value in band.get("row_ids", [])]
        declared = len(linked)
        if declared > int(settings.maximum_rows_per_band):
            raise RuntimeError(
                "v2.4 invariant failed: final band exceeds the biological "
                f"row maximum; source_image={source}, band={band['band_id']}, "
                f"rows={declared}, maximum={settings.maximum_rows_per_band}."
            )
        for field in ("row_count", "automatic_row_count", "row_count_used_for_traits"):
            if int(band.get(field) or -1) != declared:
                raise RuntimeError(
                    "v2.4 invariant failed: stale band row-count field; "
                    f"source_image={source}, band={band['band_id']}, field={field}."
                )
        center = float(band["band_center_lateral_um"])
        if center < previous_center:
            raise RuntimeError(
                f"v2.4 invariant failed: bands not ordered left-to-right; source_image={source}."
            )
        previous_center = center
        union = sorted(
            {int(value) for row in linked for value in row.get("_point_indices", [])}
        )
        if union != sorted(int(value) for value in band.get("_point_indices", [])):
            raise RuntimeError(
                "v2.4 invariant failed: band members differ from linked-row union; "
                f"source_image={source}, band={band['band_id']}."
            )
        seen_points.extend(union)
        for position, row in enumerate(linked, start=1):
            if str(row.get("band_id")) != str(band["band_id"]):
                raise RuntimeError("v2.4 invariant failed: row-to-band link mismatch.")
            if not (
                int(row.get("row_position_within_band") or -1) == position
                and int(row.get("rows_in_band") or -1) == declared
                and int(row.get("automatic_rows_in_band") or -1) == declared
                and int(row.get("band_row_count") or -1) == declared
            ):
                raise RuntimeError("v2.4 invariant failed: stale row band-count fields.")
            for point_index in row.get("_point_indices", []):
                assignment = assignments[int(point_index)]
                if (
                    str(assignment.get("row_id")) != str(row["row_id"])
                    or str(assignment.get("band_id")) != str(band["band_id"])
                ):
                    raise RuntimeError("v2.4 invariant failed: point assignment mismatch.")
        if not bool(band.get("band_complete_topology_eligible")):
            if not (
                bool(band.get("band_lateral_is_leftmost_band"))
                or bool(band.get("band_lateral_is_rightmost_band"))
            ):
                raise RuntimeError(
                    "v2.4 invariant failed: internal band excluded as an edge fragment."
                )
        if bool(band.get("band_edge_adequate_support_veto_applied")) and not bool(
            band.get("band_complete_topology_eligible")
        ):
            raise RuntimeError(
                "v2.4 invariant failed: adequate-support veto band was excluded."
            )
        if bool(band.get("band_edge_adequate_support_veto_applied")) and not bool(
            band.get("band_edge_exclusion_candidate")
        ):
            raise RuntimeError(
                "v2.4 invariant failed: adequate-support veto lacked an edge threat."
            )
        excluded = bool(band.get("band_edge_topology_excluded"))
        subtypes = (
            bool(band.get("band_lateral_half_cut_excluded")),
            bool(band.get("band_lateral_sparse_border_excluded")),
            bool(band.get("band_longitudinal_edge_fragment_excluded")),
        )
        if excluded != bool(band.get("band_edge_fragment_excluded")):
            raise RuntimeError(
                "v2.4 invariant failed: generic edge-exclusion flags disagree."
            )
        if excluded and sum(subtypes) != 1:
            raise RuntimeError(
                "v2.4 invariant failed: excluded band lacks one unambiguous edge subtype."
            )
        if not excluded and any(subtypes):
            raise RuntimeError(
                "v2.4 invariant failed: retained band carries an exclusion subtype."
            )
        expected_eligible = bool(band.get("band_complete_topology_eligible"))
        expected_reason = str(band.get("band_spatial_exclusion_reason") or "")
        for row in linked:
            if bool(row.get("row_complete_topology_eligible")) != expected_eligible:
                raise RuntimeError(
                    "v2.4 invariant failed: row/band topology eligibility disagrees."
                )
            if str(row.get("row_band_spatial_exclusion_reason") or "") != expected_reason:
                raise RuntimeError(
                    "v2.4 invariant failed: row/band exclusion reasons disagree."
                )
            for point_index in row.get("_point_indices", []):
                point = assignments[int(point_index)]
                if bool(point.get("row_complete_topology_eligible")) != expected_eligible:
                    raise RuntimeError(
                        "v2.4 invariant failed: point/band topology eligibility disagrees."
                    )
                if str(point.get("row_band_spatial_exclusion_reason") or "") != expected_reason:
                    raise RuntimeError(
                        "v2.4 invariant failed: point/band exclusion reasons disagree."
                    )
        if (
            declared == 1
            and int(band.get("stoma_count") or 0) == 1
            and linked
            and bool(linked[0].get("trajectory_v24_singleton_applied"))
        ):
            raise RuntimeError(
                "v2.4 invariant failed: a rescued singleton remained an isolated band."
            )
    if sorted(seen_points) != list(range(len(stoma_traits))):
        raise RuntimeError(
            f"v2.4 invariant failed: points duplicated or lost; source_image={source}."
        )
    for field, before in eligibility_before.items():
        after = [base.parse_bool(record.get(field)) for record in stoma_traits]
        if after != before:
            raise RuntimeError(
                "v2.4 invariant failed: individual-complex eligibility changed; "
                f"source_image={source}, field={field}."
            )


def infer_rows_and_bands_v24(
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettingsV24,
    audited_row_counts: dict[str, int] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    """Run corrected rows -> final bands -> final two-axis eligibility."""

    eligibility_fields = (
        "density_eligible",
        "guard_morphometry_eligible",
        "subsidiary_morphometry_eligible",
        "complex_morphometry_eligible",
    )
    eligibility_before = {
        field: [base.parse_bool(record.get(field)) for record in stoma_traits]
        for field in eligibility_fields
    }
    rows, bands, assignments, audit = v22.infer_rows_and_bands_v20(
        stoma_traits,
        settings,
        audited_row_counts=audited_row_counts,
    )
    widths = [
        value
        for value in (_finite(record.get("prediction_width_um")) for record in stoma_traits)
        if value is not None and value > 0
    ]
    median_width = float(np.median(np.asarray(widths, dtype=float))) if widths else 20.0
    slopes = [
        float(row.get("row_common_slope_lateral_per_longitudinal") or 0.0)
        for row in rows
        if int(row.get("stoma_count") or 0) >= 2
    ]
    representative_slope = (
        float(np.median(np.asarray(slopes, dtype=float))) if slopes else 0.0
    )
    edge_audit = _remark_partial_band_eligibility_v24(
        rows,
        bands,
        assignments,
        stoma_traits,
        median_width,
        representative_slope,
        settings,
        parent_audit=(
            audit.get("band_spatial_eligibility")
            if isinstance(audit.get("band_spatial_eligibility"), list)
            else None
        ),
    )
    audit["band_spatial_eligibility"] = edge_audit

    diagnostics = list(audit.get("band_selection_diagnostics", {}).values())
    fragment_count = sum(
        bool(item.get("trajectory_v24_fragment_merge_applied")) for item in diagnostics
    )
    singleton_count = sum(
        bool(item.get("trajectory_v24_singleton_applied")) for item in diagnostics
    )
    partition_diagnostics = audit.get("band_partition_diagnostics", [])
    split_count = sum(
        bool(item.get("trajectory_v24_partition_split_applied"))
        and bool(item.get("accepted"))
        for item in partition_diagnostics
    )
    reversal_count = sum(
        bool(item.get("trajectory_v24_partition_reversal_applied"))
        for item in partition_diagnostics
    )
    edge_count = sum(bool(band.get("band_edge_fragment_excluded")) for band in bands)
    lateral_half_cut_count = sum(
        bool(band.get("band_lateral_half_cut_excluded")) for band in bands
    )
    lateral_sparse_count = sum(
        bool(band.get("band_lateral_sparse_border_excluded")) for band in bands
    )
    longitudinal_edge_count = sum(
        bool(band.get("band_longitudinal_edge_fragment_excluded")) for band in bands
    )
    veto_count = sum(
        bool(band.get("band_edge_adequate_support_veto_applied")) for band in bands
    )

    # Populate v2.3 aggregate fields because the v2.4 selector still retains
    # every accepted v2.3 route even though v2.4 bypasses the stale v2.3
    # post-partition reconstruction step.
    audit.update(
        {
            "row_model_version": SCRIPT_VERSION,
            "trajectory_singleton_policy_version_current": SCRIPT_VERSION,
            "trajectory_v23_bracketed_small_rescue_count": sum(
                bool(item.get("trajectory_v23_additional_singleton_applied"))
                and item.get("trajectory_v23_rescue_route") == "bracketed_3_to_2_plus_1"
                for item in diagnostics
            ),
            "trajectory_v23_terminal_rescue_count": sum(
                bool(item.get("trajectory_v23_additional_singleton_applied"))
                and item.get("trajectory_v23_rescue_route")
                == "terminal_refitted_remaining_row"
                for item in diagnostics
            ),
            "trajectory_v23_narrow_two_row_collapse_count": sum(
                bool(item.get("trajectory_v23_narrow_two_row_collapse_applied"))
                for item in diagnostics
            ),
            "trajectory_v23_band_boundary_restored_count": 0,
            "trajectory_v24_policy_version": SCRIPT_VERSION,
            "trajectory_v24_fragment_merge_count": fragment_count,
            "trajectory_v24_singleton_rescue_count": singleton_count,
            "trajectory_v24_partition_split_count": split_count,
            "trajectory_v24_partition_reversal_count": reversal_count,
            "trajectory_v24_edge_fragment_excluded_band_count": edge_count,
            "trajectory_v24_lateral_half_cut_excluded_band_count": (
                lateral_half_cut_count
            ),
            "trajectory_v24_lateral_sparse_border_excluded_band_count": (
                lateral_sparse_count
            ),
            "trajectory_v24_longitudinal_edge_fragment_excluded_band_count": (
                longitudinal_edge_count
            ),
            "trajectory_v24_adequate_support_veto_band_count": veto_count,
            "partial_band_eligibility_version": SCRIPT_VERSION,
            "final_detected_band_count": len(bands),
            "final_detected_row_count": len(rows),
            "spatially_eligible_band_count": sum(
                base.parse_bool(band.get("band_spatial_eligible")) is True
                for band in bands
            ),
            "spatially_excluded_band_count": sum(
                base.parse_bool(band.get("band_spatial_eligible")) is not True
                for band in bands
            ),
            "spatially_eligible_row_count": sum(
                base.parse_bool(row.get("row_band_spatial_eligible")) is True
                for row in rows
            ),
            "spatially_excluded_row_count": sum(
                base.parse_bool(row.get("row_band_spatial_eligible")) is not True
                for row in rows
            ),
            "row_structure_eligible_band_count": sum(
                base.parse_bool(band.get("band_row_structure_eligible")) is True
                for band in bands
            ),
            "lateral_half_cut_band_count": sum(
                base.parse_bool(band.get("band_lateral_half_cut")) is True
                for band in bands
            ),
            "longitudinal_sampling_complete_band_count": sum(
                base.parse_bool(band.get("band_longitudinal_sampling_complete"))
                is True
                for band in bands
            ),
            "longitudinal_sampling_incomplete_band_count": sum(
                base.parse_bool(band.get("band_longitudinal_sampling_complete"))
                is False
                for band in bands
            ),
            "completeness_dependent_traits_eligible_band_count": sum(
                base.parse_bool(
                    band.get("band_completeness_dependent_traits_eligible")
                )
                is True
                for band in bands
            ),
            "automatic_row_count_policy": (
                "v2.4 retains v2.3 selection, merges only unique smooth local "
                "continuation fragments, then permits one unique stable frozen "
                "large-row bracketed singleton"
            ),
            "band_partition_policy": (
                "v2.4 requires strong stable geometry plus raw-image support for "
                "a split isolating one multipoint row; weak inherited isolated-row "
                "splits are reversed; only one new boundary decision is permitted "
                "per provisional band in this experimental version"
            ),
            "partial_band_policy": (
                "v2.4 excludes complete-topology traits for confirmed lateral "
                "half-cuts or short longitudinal edge fragments; adequate support "
                "vetoes proximity-only exclusion; density and morphometry remain"
            ),
            "trajectory_row_band_trait_policy": (
                "all computable row/band values are retained with versioned QC; "
                "complete-topology summaries use final v2.4 eligibility"
            ),
        }
    )
    for collection in (rows, bands, list(assignments.values())):
        for record in collection:
            record["row_model_version"] = SCRIPT_VERSION
            record["trajectory_v24_policy_version"] = SCRIPT_VERSION
    _assert_v24_invariants(
        stoma_traits,
        rows,
        bands,
        assignments,
        eligibility_before,
        settings,
    )
    return rows, bands, assignments, audit


def image_trait_summary_v24(
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
    result = v23.image_trait_summary_v23(
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
    reasons: list[str] = []
    for key, label in (
        ("trajectory_v24_fragment_merge_count", "local_fragment_merge"),
        ("trajectory_v24_singleton_rescue_count", "expanded_singleton_rescue"),
        ("trajectory_v24_partition_split_count", "band_split"),
        ("trajectory_v24_partition_reversal_count", "band_split_reversal"),
        ("trajectory_v24_edge_fragment_excluded_band_count", "edge_fragment_exclusion"),
        ("trajectory_v24_adequate_support_veto_band_count", "adequate_support_veto"),
    ):
        if int(row_band_audit.get(key) or 0):
            reasons.append(f"v24_{label}_provisional")
    result.update(
        {
            key: row_band_audit.get(key)
            for key in _V24_IMAGE_FIELDS
            if key.startswith("trajectory_v24_")
        }
    )
    result.update(
        {
            "detected_row_count_v24": len(row_records),
            "detected_band_count_v24": len(band_records),
            "row_band_v24_qc_status": (
                "flagged_provisional_change" if reasons else "pass_no_v24_change"
            ),
            "row_band_v24_qc_flag": bool(reasons),
            "row_band_v24_qc_reason": ";".join(reasons),
            "row_band_v24_value_interpretation": (
                "v2.4 values include guarded local row-fragment merging, frozen "
                "singleton scanning, image-supported band partitioning and final "
                "two-axis edge-completeness QC; changes remain provisional"
            ),
            "row_geometry_model_version": SCRIPT_VERSION,
            "partial_band_eligibility_version": SCRIPT_VERSION,
            "trajectory_row_band_trait_policy": row_band_audit.get(
                "trajectory_row_band_trait_policy"
            ),
        }
    )
    return result


def image_gwas_trait_summary_v24(image_traits: dict[str, Any]) -> dict[str, Any]:
    summary = v23.image_gwas_trait_summary_v23(image_traits)
    for field in _V24_IMAGE_FIELDS:
        summary[field] = image_traits.get(field)
    summary["row_geometry_model_version"] = SCRIPT_VERSION
    summary["partial_band_eligibility_version"] = SCRIPT_VERSION
    return summary


def add_row_neighbors_v24(
    stoma_traits: list[dict[str, Any]],
    row_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
) -> None:
    v23.add_row_neighbors_v23(stoma_traits, row_records, point_assignment)
    for point_index, assignment in point_assignment.items():
        if not 0 <= int(point_index) < len(stoma_traits):
            continue
        stoma_traits[int(point_index)].update(
            {
                key: value
                for key, value in assignment.items()
                if key.startswith("trajectory_v24_")
                or key.startswith("row_edge_")
                or key == "row_complete_topology_eligible"
            }
        )


def make_spatial_audit_overlay_v24(
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
    # Suppress the inherited generic/V2.2 singleton label only for points
    # introduced by the new v2.4 route.  Their complete provenance is restored
    # immediately and they receive an unambiguous v2.4 label below.
    suppressed: list[tuple[dict[str, Any], Any, Any]] = []
    for row in row_records:
        candidate = base.optional_int(
            row.get("trajectory_v24_singleton_candidate_point_index")
        )
        members = [int(value) for value in row.get("_point_indices", [])]
        if (
            bool(row.get("trajectory_v24_singleton_applied"))
            and candidate is not None
            and candidate in members
        ):
            suppressed.append(
                (
                    row,
                    row.get("trajectory_v20_singleton_rescue_applied"),
                    row.get("trajectory_v22_leave_one_out_applied"),
                )
            )
            row["trajectory_v20_singleton_rescue_applied"] = False
            row["trajectory_v22_leave_one_out_applied"] = False
    try:
        v23.make_spatial_audit_overlay_v23(
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
    finally:
        for row, v20_value, v22_value in suppressed:
            row["trajectory_v20_singleton_rescue_applied"] = v20_value
            row["trajectory_v22_leave_one_out_applied"] = v22_value
    image = base.cv2.imread(str(output_path), base.cv2.IMREAD_COLOR)
    if image is None:
        return
    row_lookup = {str(row["row_id"]): row for row in row_records}
    for band in band_records:
        members = [int(value) for value in band.get("_point_indices", [])]
        if not members:
            continue
        x = int(round(float(np.mean([stoma_traits[i]["center_x_pixels"] for i in members]))))
        y = int(round(float(np.mean([stoma_traits[i]["center_y_pixels"] for i in members]))))
        labels: list[tuple[str, tuple[int, int, int]]] = []
        if bool(band.get("trajectory_v24_partition_split_applied")):
            labels.append((f"{band['band_id']} V2.4 BAND SPLIT", (255, 255, 255)))
        if bool(band.get("trajectory_v24_partition_reversal_applied")):
            labels.append((f"{band['band_id']} V2.4 REATTACHED", (255, 0, 255)))
        if bool(band.get("band_lateral_half_cut_excluded")):
            labels.append((f"{band['band_id']} LATERAL HALF-CUT", (0, 0, 255)))
        elif bool(band.get("band_lateral_sparse_border_excluded")):
            labels.append((f"{band['band_id']} SPARSE LATERAL EDGE", (0, 0, 255)))
        elif bool(band.get("band_longitudinal_edge_fragment_excluded")):
            labels.append((f"{band['band_id']} LONGITUDINAL EDGE", (0, 0, 255)))
        if bool(band.get("band_edge_adequate_support_veto_applied")):
            labels.append((f"{band['band_id']} COMPLETE-SUPPORT RETAIN", (0, 255, 0)))
        for offset, (label, color) in enumerate(labels):
            base.cv2.putText(
                image,
                label,
                (max(5, x - 110), max(22, y - 28 - 18 * offset)),
                base.cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                base.cv2.LINE_AA,
            )
    for row in row_records:
        candidate = base.optional_int(
            row.get("trajectory_v24_singleton_candidate_point_index")
        )
        members = [int(value) for value in row.get("_point_indices", [])]
        if not (
            bool(row.get("trajectory_v24_singleton_applied"))
            and candidate is not None
            and candidate in members
        ):
            continue
        x = int(round(float(stoma_traits[candidate]["center_x_pixels"])))
        y = int(round(float(stoma_traits[candidate]["center_y_pixels"])))
        color = (0, 255, 255)
        base.cv2.circle(image, (x, y), 27, (0, 0, 0), 6, base.cv2.LINE_AA)
        base.cv2.circle(image, (x, y), 24, color, 3, base.cv2.LINE_AA)
        base.cv2.line(image, (x - 20, y), (x + 20, y), color, 2, base.cv2.LINE_AA)
        base.cv2.line(image, (x, y - 20), (x, y + 20), color, 2, base.cv2.LINE_AA)
        base.cv2.putText(
            image,
            f"{row['row_id']} V2.4 EXPANDED SINGLETON ROW",
            (max(5, x - 125), max(22, y - 30)),
            base.cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            2,
            base.cv2.LINE_AA,
        )
    for row in row_records:
        if not bool(row.get("trajectory_v24_fragment_merge_applied")):
            continue
        members = [int(value) for value in row.get("_point_indices", [])]
        merged_members = sorted(
            int(value)
            for value in row.get("trajectory_v24_fragment_merge_point_indices", [])
        )
        if sorted(members) != merged_members:
            continue
        if not members:
            continue
        x = int(round(float(np.mean([stoma_traits[i]["center_x_pixels"] for i in members]))))
        y = int(round(float(np.mean([stoma_traits[i]["center_y_pixels"] for i in members]))))
        base.cv2.putText(
            image,
            f"{row['row_id']} V2.4 MERGED CONTINUATION",
            (max(5, x - 115), max(22, y - 20)),
            base.cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            2,
            base.cv2.LINE_AA,
        )
    if not base.cv2.imwrite(
        str(output_path), image, [base.cv2.IMWRITE_PNG_COMPRESSION, 6]
    ):
        raise RuntimeError(f"Failed to update v2.4 audit overlay: {output_path}")


def install_v24_policy() -> None:
    """Install v2.3, then replace only the v2.4 hooks."""

    v23.install_v23_policy()
    v14._partition_provisional_bands = _partition_trajectory_bands_v24
    v15._trajectory_diagnostic_fields = _trajectory_diagnostic_fields_v24

    base.build_settings = build_settings_v24
    base.select_row_model = select_row_model_v24
    base.infer_rows_and_bands = infer_rows_and_bands_v24
    base.image_trait_summary = image_trait_summary_v24
    base.image_gwas_trait_summary = image_gwas_trait_summary_v24
    base.add_row_neighbors = add_row_neighbors_v24
    base.make_spatial_audit_overlay = make_spatial_audit_overlay_v24
    for module in (v14, v15, v16, base):
        module.SCRIPT_VERSION = SCRIPT_VERSION
    base.__file__ = __file__


def run_trait_extraction() -> Path:
    install_v24_policy()
    return base.run_trait_extraction()


if __name__ == "__main__":
    run_trait_extraction()
