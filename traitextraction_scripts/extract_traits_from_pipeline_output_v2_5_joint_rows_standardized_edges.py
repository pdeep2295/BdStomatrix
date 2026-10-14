#!/usr/bin/env python3
"""Experimental v2.5 Brachypodium row/band topology extractor.

This is a separate successor to v2.4.  It does not modify v2.4, completed
pipeline inference, microscopy files, masks, density eligibility, or cell
morphometry eligibility.

V2.5 addresses the failure modes found in the manually audited 40-image
development set:

1. Row membership and up to two singleton identities are compared jointly,
   rather than adding one leave-one-out singleton to a frozen answer.
2. A bounded band-local residual angle is searched after image-wide axis
   correction.  All rows in a provisional group share that angle.
3. Terminal singletons, two-point -> two-singleton alternatives, multiple
   singleton rows, and same-row-count membership corrections are permitted.
4. Detector confidence is retained as soft evidence; it is not a hard row-
   existence gate.  Crop classification and segmentation never determine
   whether a detected stoma belongs to a row.
5. After the final row assignment, all inherited v2.4 band cuts are archived
   and each complete provisional group is reassessed under one v2.5 rule.
   More than one boundary may be added.  Direct raw-image support is a required
   gate for cuts that isolate a one-row or singleton side.
6. The locked v2.4 single-row edge rules are preserved.  A new, separate rule
   excludes an outer multi-row band from complete topology when any row has at
   least 20% less comparable trajectory than the longest visible row through
   the relevant lateral image boundary.  This detects differential clipping;
   shared absolute clipping of every row remains explicitly unassessable.

The current downstream geometry represents rows by a shared straight slope
and ordered intercepts.  V2.5 therefore does not silently introduce quadratic
coordinates.  Existing local-continuation evidence can reject a singleton
that is explained by a bending row; the pooled residual-curvature diagnostic
separately marks remaining nonlinear uncertainty.  Suspected nonlinear cases
retain a best estimate plus an explicit QC flag rather than exporting curved
geometry as if it were directly supported.

All computable values remain in the detailed tables.  Supported and ambiguous
singleton counts, row-count bounds, candidate diagnostics, and edge evidence
are exported so uncertainty is not silently converted to zero.

Run this script only in a fresh Python process and always use a new output
directory.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import extract_traits_from_pipeline_output_v2_4_adaptive_rows_bands as v24


base = v24.base
v14 = v24.v14
v15 = v24.v15
v16 = v24.v16
v19 = v24.v19
v22 = v24.v22
v23 = v24.v23

SCRIPT_VERSION = "2.5.0-experimental-joint-rows-standardized-edges"
SINGLETON_POLICY_VERSION = SCRIPT_VERSION
ROW_GEOMETRY_MODEL_VERSION = SCRIPT_VERSION
PARTIAL_BAND_ELIGIBILITY_VERSION = SCRIPT_VERSION
TRAIT_EXPORT_POLICY_VERSION = v24.TRAIT_EXPORT_POLICY_VERSION

_PARENT_V24_SCRIPT_PATH = Path(v24.__file__).resolve()

# Joint row search.  These are versioned scientific settings, not hidden
# filename-specific corrections.
V25_LOCAL_RESIDUAL_TILT_LIMIT_DEG = 8.0
V25_LOCAL_RESIDUAL_TILT_COARSE_STEP_DEG = 1.0
V25_LOCAL_RESIDUAL_TILT_FINE_STEP_DEG = 0.25
V25_LOCAL_RESIDUAL_TILT_FINE_HALF_RANGE_DEG = 1.0
V25_MAX_SINGLETONS_PER_PROVISIONAL_GROUP = 2
V25_MAX_SINGLETON_PREFILTER_POINTS = 10
V25_SINGLETON_PREFILTER_MIN_RESIDUAL_WIDTH_FACTOR = 0.20
V25_MIN_ROW_SEPARATION_WIDTH_FACTOR = 0.25
V25_MAX_ORDINARY_ROW_RMS_WIDTH_FACTOR = 0.15
V25_MIN_SINGLETON_RESIDUAL_DOMINANCE = 3.0
V25_SOFT_SINGLETON_PENALTY = 1.50
V25_UNRESOLVED_CURVATURE_PENALTY = 1.00
V25_SLOPE_PENALTY_WEIGHT = 0.10
V25_SIMPLER_TIE_TOLERANCE = 1.0
V25_MIN_SCORE_IMPROVEMENT_NEW_ROW_COUNT = 2.0
V25_MIN_SCORE_IMPROVEMENT_SAME_ROW_COUNT = 1.0
V25_STABILITY_MIN_ROW_COUNT_FRACTION = 0.80
V25_STABILITY_MIN_ASSIGNMENT_FRACTION = 0.80
V25_STABILITY_MIN_SINGLETON_IDENTITY_FRACTION = 0.80
V25_STABILITY_DELTA_SCORE_WINDOW = 6.0
V25_STABILITY_MIN_EVALUATED_FRACTION = 0.80
V25_POSSIBLE_SINGLETON_FREQUENCY = 0.25
V25_CURVATURE_MIN_RSS_REDUCTION = 0.20
V25_CURVATURE_MIN_END_BEND_WIDTH_FACTOR = 0.10

# Post-row biological-band splitting.  The 1.65/1.80 raw-support thresholds
# are development values and must be rechecked on this audit after switching
# to exact-pair trajectory/flank evidence, then frozen before confirmation.
V25_PARTITION_MIN_GAP_WIDTH_FACTOR = 2.30
V25_PARTITION_MIN_LOCAL_GAP_RATIO = 1.50
V25_PARTITION_MIN_IMAGE_SUPPORT_RATIO_ISOLATED_ROW = 1.65
V25_PARTITION_MIN_STABILITY_FRACTION = 0.80
V25_PARTITION_MIN_STOMATA_PER_SIDE = 2
V25_PARTITION_SINGLETON_SIDE_MIN_GAP_WIDTH_FACTOR = 3.00
V25_PARTITION_SINGLETON_SIDE_MIN_IMAGE_SUPPORT_RATIO = 1.80

# New multi-row edge rule.  The earlier single-row rule remains unchanged.
V25_MULTIROW_LATERAL_LOSS_THRESHOLD_FRACTION = 0.20


@dataclass(frozen=True)
class TraitSettingsV25(v24.TraitSettingsV24):
    """Immutable v2.5 settings written into provenance."""

    v25_policy_version: str = SCRIPT_VERSION
    v25_parent_v24_script_sha256: str = ""
    v25_local_residual_tilt_limit_deg: float = V25_LOCAL_RESIDUAL_TILT_LIMIT_DEG
    v25_local_residual_tilt_coarse_step_deg: float = (
        V25_LOCAL_RESIDUAL_TILT_COARSE_STEP_DEG
    )
    v25_local_residual_tilt_fine_step_deg: float = (
        V25_LOCAL_RESIDUAL_TILT_FINE_STEP_DEG
    )
    v25_local_residual_tilt_fine_half_range_deg: float = (
        V25_LOCAL_RESIDUAL_TILT_FINE_HALF_RANGE_DEG
    )
    v25_max_singletons_per_provisional_group: int = (
        V25_MAX_SINGLETONS_PER_PROVISIONAL_GROUP
    )
    v25_max_singleton_prefilter_points: int = V25_MAX_SINGLETON_PREFILTER_POINTS
    v25_singleton_prefilter_min_residual_width_factor: float = (
        V25_SINGLETON_PREFILTER_MIN_RESIDUAL_WIDTH_FACTOR
    )
    v25_min_row_separation_width_factor: float = (
        V25_MIN_ROW_SEPARATION_WIDTH_FACTOR
    )
    v25_max_ordinary_row_rms_width_factor: float = (
        V25_MAX_ORDINARY_ROW_RMS_WIDTH_FACTOR
    )
    v25_min_singleton_residual_dominance: float = (
        V25_MIN_SINGLETON_RESIDUAL_DOMINANCE
    )
    v25_soft_singleton_penalty: float = V25_SOFT_SINGLETON_PENALTY
    v25_unresolved_curvature_penalty: float = V25_UNRESOLVED_CURVATURE_PENALTY
    v25_slope_penalty_weight: float = V25_SLOPE_PENALTY_WEIGHT
    v25_simpler_tie_tolerance: float = V25_SIMPLER_TIE_TOLERANCE
    v25_min_score_improvement_new_row_count: float = (
        V25_MIN_SCORE_IMPROVEMENT_NEW_ROW_COUNT
    )
    v25_min_score_improvement_same_row_count: float = (
        V25_MIN_SCORE_IMPROVEMENT_SAME_ROW_COUNT
    )
    v25_stability_min_row_count_fraction: float = (
        V25_STABILITY_MIN_ROW_COUNT_FRACTION
    )
    v25_stability_min_assignment_fraction: float = (
        V25_STABILITY_MIN_ASSIGNMENT_FRACTION
    )
    v25_stability_min_singleton_identity_fraction: float = (
        V25_STABILITY_MIN_SINGLETON_IDENTITY_FRACTION
    )
    v25_stability_delta_score_window: float = V25_STABILITY_DELTA_SCORE_WINDOW
    v25_stability_min_evaluated_fraction: float = (
        V25_STABILITY_MIN_EVALUATED_FRACTION
    )
    v25_possible_singleton_frequency: float = V25_POSSIBLE_SINGLETON_FREQUENCY
    v25_curvature_min_rss_reduction: float = V25_CURVATURE_MIN_RSS_REDUCTION
    v25_curvature_min_end_bend_width_factor: float = (
        V25_CURVATURE_MIN_END_BEND_WIDTH_FACTOR
    )
    v25_partition_min_gap_width_factor: float = V25_PARTITION_MIN_GAP_WIDTH_FACTOR
    v25_partition_min_local_gap_ratio: float = V25_PARTITION_MIN_LOCAL_GAP_RATIO
    v25_partition_min_image_support_ratio_isolated_row: float = (
        V25_PARTITION_MIN_IMAGE_SUPPORT_RATIO_ISOLATED_ROW
    )
    v25_partition_min_stability_fraction: float = (
        V25_PARTITION_MIN_STABILITY_FRACTION
    )
    v25_partition_min_stomata_per_side: int = (
        V25_PARTITION_MIN_STOMATA_PER_SIDE
    )
    v25_partition_singleton_side_min_gap_width_factor: float = (
        V25_PARTITION_SINGLETON_SIDE_MIN_GAP_WIDTH_FACTOR
    )
    v25_partition_singleton_side_min_image_support_ratio: float = (
        V25_PARTITION_SINGLETON_SIDE_MIN_IMAGE_SUPPORT_RATIO
    )
    v25_multirow_lateral_loss_threshold_fraction: float = (
        V25_MULTIROW_LATERAL_LOSS_THRESHOLD_FRACTION
    )


_V25_DEFAULTS: dict[str, Any] = {
    "trajectory_v25_policy_version": SCRIPT_VERSION,
    "trajectory_v25_joint_search_performed": False,
    "trajectory_v25_joint_candidate_count": 0,
    "trajectory_v25_joint_change_applied": False,
    "trajectory_v25_joint_status": "not_applicable",
    "trajectory_v25_joint_reason": "",
    "trajectory_v25_baseline_row_count": None,
    "trajectory_v25_selected_row_count": None,
    "trajectory_v25_baseline_cluster_sizes": [],
    "trajectory_v25_selected_cluster_sizes": [],
    "trajectory_v25_score_improvement": None,
    "trajectory_v25_pairwise_comembership_vs_baseline": None,
    "trajectory_v25_selected_singleton_count": 0,
    "trajectory_v25_supported_singleton_count": 0,
    "trajectory_v25_ambiguous_singleton_count": 0,
    "trajectory_v25_selected_singleton_point_indices": [],
    "trajectory_v25_selected_singleton_stoma_ids": [],
    "trajectory_v25_supported_singleton_stoma_ids": [],
    "trajectory_v25_ambiguous_singleton_stoma_ids": [],
    "trajectory_v25_possible_singleton_stoma_ids": [],
    "trajectory_v25_singleton_detector_confidences": [],
    "trajectory_v25_singleton_prefilter_diagnostics": [],
    "trajectory_v25_singleton_evidence": [],
    "trajectory_v25_candidate_diagnostics": [],
    "trajectory_v25_row_count_stability_fraction": None,
    "trajectory_v25_assignment_stability_fraction": None,
    "trajectory_v25_singleton_identity_stability_fraction": None,
    "trajectory_v25_stability_evaluated_fraction": None,
    "trajectory_v25_stability_trials_sufficient": False,
    "trajectory_v25_stability_scope": "not_assessed",
    "trajectory_v25_stability_count_distribution": {},
    "trajectory_v25_singleton_frequency_by_stoma": {},
    "trajectory_v25_row_count_lower_bound": None,
    "trajectory_v25_row_count_upper_bound": None,
    "trajectory_v25_topology_confidence_tier": "not_assessed",
    "trajectory_v25_band_local_residual_angle_deg": None,
    "trajectory_v25_residual_angle_at_search_bound": False,
    "trajectory_v25_nonlinear_trajectory_suspected": False,
    "trajectory_v25_curvature_diagnostic": {},
    "trajectory_v25_parent_v24_topology_superseded": False,
}

_V25_IMAGE_FIELDS = (
    "trajectory_v25_policy_version",
    "trajectory_v25_joint_changed_group_count",
    "trajectory_v25_supported_singleton_row_count",
    "trajectory_v25_ambiguous_singleton_row_count",
    "trajectory_v25_possible_singleton_candidate_count",
    "trajectory_v25_residual_angle_bound_hit_group_count",
    "trajectory_v25_added_band_boundary_count",
    "trajectory_v25_capacity_forced_unvalidated_boundary_count",
    "trajectory_v25_unvalidated_inherited_singleton_boundary_count",
    "trajectory_v25_multirow_edge_assessed_band_count",
    "trajectory_v25_multirow_edge_triggered_band_count",
    "trajectory_v25_multirow_edge_newly_excluded_band_count",
    "trajectory_v25_multirow_edge_unassessable_band_count",
    "detected_row_count_v25",
    "detected_band_count_v25",
    "detected_row_count_v25_lower_bound",
    "detected_row_count_v25_upper_bound",
    "supported_singleton_row_fraction_v25",
    "row_band_v25_qc_status",
    "row_band_v25_qc_flag",
    "row_band_v25_qc_reason",
    "row_band_v25_value_interpretation",
)


def build_settings_v25(args: Any) -> TraitSettingsV25:
    inherited = asdict(v24.build_settings_v24(args))
    inherited.update(
        {
            "v25_policy_version": SCRIPT_VERSION,
            "v25_parent_v24_script_sha256": hashlib.sha256(
                _PARENT_V24_SCRIPT_PATH.read_bytes()
            ).hexdigest(),
        }
    )
    return TraitSettingsV25(**inherited)


def _finite(value: Any) -> float | None:
    parsed = base.optional_float(value)
    if parsed is None or not np.isfinite(parsed):
        return None
    return float(parsed)


def _candidate_group_members(candidate: dict[str, Any]) -> list[list[int]]:
    labels = np.asarray(candidate.get("labels", []), dtype=int)
    points = [int(value) for value in candidate.get("point_indices", [])]
    return [
        sorted(points[position] for position in np.flatnonzero(labels == group))
        for group in range(int(candidate.get("k", 0)))
    ]


def _canonical_partition(candidate: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(tuple(group) for group in _candidate_group_members(candidate))
    )


def _singleton_points(candidate: dict[str, Any]) -> list[int]:
    return [group[0] for group in _candidate_group_members(candidate) if len(group) == 1]


def _pairwise_comembership_agreement(
    first: dict[str, Any], second: dict[str, Any]
) -> float:
    """Jaccard agreement among point pairs assigned to the same row.

    Counting the much larger set of pairs separated in both partitions would
    inflate apparent stability when many rows exist.  Same-row Jaccard keeps
    the metric sensitive to the membership changes under audit.
    """

    first_map = v22._candidate_label_map(first)
    second_map = v22._candidate_label_map(second)
    common = sorted(set(first_map) & set(second_map))
    first_pairs: set[tuple[int, int]] = set()
    second_pairs: set[tuple[int, int]] = set()
    for position, left in enumerate(common):
        for right in common[position + 1 :]:
            if first_map[left] == first_map[right]:
                first_pairs.add((left, right))
            if second_map[left] == second_map[right]:
                second_pairs.add((left, right))
    union = first_pairs | second_pairs
    return len(first_pairs & second_pairs) / len(union) if union else 1.0


def _active_record(point_index: int) -> dict[str, Any] | None:
    records = v15._ACTIVE_STOMA_TRAITS_V15
    if records is None or not 0 <= point_index < len(records):
        return None
    return records[point_index]


def _density_eligible(point_index: int) -> bool | None:
    record = _active_record(point_index)
    return base.parse_bool(record.get("density_eligible")) if record is not None else None


def _point_stoma_id(point_index: int) -> str:
    record = _active_record(point_index)
    return str(record.get("stoma_id") or "") if record is not None else ""


def _point_confidence(point_index: int) -> float | None:
    return v22._detector_confidence(point_index)


def _prefix_segment_sse(
    prefix: np.ndarray,
    prefix_square: np.ndarray,
    start: int,
    stop: int,
) -> tuple[float, float]:
    count = stop - start
    total = float(prefix[stop] - prefix[start])
    total_square = float(prefix_square[stop] - prefix_square[start])
    mean = total / count
    sse = max(0.0, total_square - total * total / count)
    return mean, sse


def _exact_contiguous_ordinary_partition(
    ordinary_points: list[int],
    adjusted_lateral: dict[int, float],
    ordinary_row_count: int,
    minimum_separation_um: float,
) -> dict[int, int] | None:
    """Return the minimum-SSE contiguous 1-D partition with >=2 points/row."""

    if ordinary_row_count == 0:
        return {} if not ordinary_points else None
    if len(ordinary_points) < 2 * ordinary_row_count:
        return None
    ordered = sorted(ordinary_points, key=lambda index: (adjusted_lateral[index], index))
    values = np.asarray([adjusted_lateral[index] for index in ordered], dtype=float)
    n = len(values)
    prefix = np.concatenate([np.asarray([0.0]), np.cumsum(values)])
    prefix_square = np.concatenate([np.asarray([0.0]), np.cumsum(values * values)])
    # State includes the start of the last segment.  This permits the
    # adjacent-mean separation rule to be enforced during optimization rather
    # than rejecting the unconstrained optimum after the fact (which could
    # miss a slightly higher-SSE but biologically valid partition).
    # Index states by (number of groups, current endpoint).  Each entry keeps
    # the possible starts of its final segment.  This avoids rescanning every
    # earlier state for every candidate transition during the full v2.5
    # angle × row-count × singleton search.
    states: dict[
        tuple[int, int],
        list[tuple[int, float, tuple[tuple[int, int], ...]]],
    ] = {}
    maximum_first_stop = n - 2 * (ordinary_row_count - 1)
    for stop in range(2, maximum_first_stop + 1):
        _, segment_sse = _prefix_segment_sse(prefix, prefix_square, 0, stop)
        states[(1, stop)] = [(0, segment_sse, ((0, stop),))]

    for group_count in range(2, ordinary_row_count + 1):
        minimum_stop = 2 * group_count
        maximum_stop = n - 2 * (ordinary_row_count - group_count)
        for stop in range(minimum_stop, maximum_stop + 1):
            for start in range(2 * (group_count - 1), stop - 1):
                current_mean, current_sse = _prefix_segment_sse(
                    prefix, prefix_square, start, stop
                )
                eligible_previous = states.get((group_count - 1, start), [])
                best_state: tuple[
                    float, tuple[tuple[int, int], ...]
                ] | None = None
                for previous_start, previous_cost, previous_path in eligible_previous:
                    previous_mean, _ = _prefix_segment_sse(
                        prefix, prefix_square, previous_start, start
                    )
                    if current_mean - previous_mean < minimum_separation_um:
                        continue
                    proposal = (
                        float(previous_cost + current_sse),
                        previous_path + ((start, stop),),
                    )
                    if best_state is None or (
                        proposal[0], proposal[1]
                    ) < (
                        best_state[0], best_state[1]
                    ):
                        best_state = proposal
                if best_state is not None:
                    states.setdefault((group_count, stop), []).append(
                        (start, best_state[0], best_state[1])
                    )

    final_states = [
        (cost, path)
        for _, cost, path in states.get((ordinary_row_count, n), [])
    ]
    if not final_states:
        return None
    _, final_path = min(final_states, key=lambda value: (value[0], value[1]))
    segments = list(final_path)
    labels: dict[int, int] = {}
    for group, (start, stop) in enumerate(segments):
        for position in range(start, stop):
            labels[ordered[position]] = group
    return labels


def _build_joint_candidate(
    point_indices: list[int],
    singleton_subset: tuple[int, ...],
    total_row_count: int,
    slope: float,
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV25,
) -> dict[str, Any] | None:
    singleton_set = {int(value) for value in singleton_subset}
    ordinary = [int(value) for value in point_indices if int(value) not in singleton_set]
    ordinary_k = int(total_row_count) - len(singleton_set)
    if ordinary_k < 0:
        return None
    reference = float(np.median(np.asarray(longitudinal[point_indices], dtype=float)))
    adjusted = {
        int(index): float(lateral[index])
        - slope * (float(longitudinal[index]) - reference)
        for index in point_indices
    }
    minimum_separation = max(
        float(settings.minimum_row_separation_um),
        float(settings.v25_min_row_separation_width_factor) * median_width_um,
    )
    ordinary_labels = _exact_contiguous_ordinary_partition(
        ordinary,
        adjusted,
        ordinary_k,
        minimum_separation,
    )
    if ordinary_labels is None:
        return None
    provisional_groups: list[list[int]] = [
        sorted(index for index, label in ordinary_labels.items() if label == group)
        for group in range(ordinary_k)
    ]
    provisional_groups.extend([[int(index)] for index in singleton_subset])
    if not provisional_groups or len(provisional_groups) != total_row_count:
        return None
    intercepts = [
        float(np.mean([adjusted[index] for index in members]))
        for members in provisional_groups
    ]
    order = np.argsort(np.asarray(intercepts, dtype=float), kind="stable")
    groups = [provisional_groups[int(old)] for old in order]
    sorted_intercepts = np.asarray([intercepts[int(old)] for old in order], dtype=float)
    if len(sorted_intercepts) > 1 and np.any(
        np.diff(sorted_intercepts) < minimum_separation
    ):
        return None
    label_by_point = {
        int(index): group
        for group, members in enumerate(groups)
        for index in members
    }
    labels = np.asarray([label_by_point[int(index)] for index in point_indices], dtype=int)
    local_lateral = np.asarray([float(lateral[index]) for index in point_indices])
    local_longitudinal = np.asarray(
        [float(longitudinal[index]) for index in point_indices]
    )
    fitted = slope * (local_longitudinal - reference) + sorted_intercepts[labels]
    residuals = local_lateral - fitted
    rss = float(np.sum(residuals * residuals))
    n = len(point_indices)
    residual_floor = float(settings.row_model_residual_floor_width_factor) * median_width_um
    variance = max(rss / max(n, 1), residual_floor**2, 1e-9)
    bic = float(n * math.log(variance) + (2 * total_row_count) * math.log(max(n, 2)))
    return {
        "k": int(total_row_count),
        "point_indices": [int(value) for value in point_indices],
        "labels": labels,
        "long_reference_um": reference,
        "slope": float(slope),
        "slope_fixed_for_candidate": True,
        "intercepts_um": sorted_intercepts,
        "rss": rss,
        "rms_residual_um": math.sqrt(max(rss / max(n, 1), 0.0)),
        "bic_residual_floor_um": residual_floor,
        "bic": bic,
        "cluster_sizes": [len(group) for group in groups],
        "trajectory_v25_explicit_singleton_subset": sorted(singleton_set),
    }


def _refit_fixed_partition_free_slope(
    template: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV25,
) -> dict[str, Any] | None:
    points = [int(value) for value in template.get("point_indices", [])]
    labels = np.asarray(template.get("labels", []), dtype=int)
    k = int(template.get("k", 0))
    if not points or len(labels) != len(points) or k < 1:
        return None
    reference = float(np.median(np.asarray(longitudinal[points], dtype=float)))
    z = np.asarray([float(longitudinal[index]) - reference for index in points])
    x = np.asarray([float(lateral[index]) for index in points])
    design = np.zeros((len(points), k + 1), dtype=float)
    design[:, 0] = z
    design[np.arange(len(points)), labels + 1] = 1.0
    beta, *_ = np.linalg.lstsq(design, x, rcond=None)
    slope_limit = math.tan(math.radians(float(settings.v25_local_residual_tilt_limit_deg)))
    slope = float(np.clip(float(beta[0]), -slope_limit, slope_limit))
    adjusted = x - slope * z
    intercepts = np.asarray(
        [float(np.mean(adjusted[labels == group])) for group in range(k)], dtype=float
    )
    order = np.argsort(intercepts, kind="stable")
    remap = {int(old): int(new) for new, old in enumerate(order)}
    remapped = np.asarray([remap[int(label)] for label in labels], dtype=int)
    intercepts = intercepts[order]
    minimum_separation = max(
        float(settings.minimum_row_separation_um),
        float(settings.v25_min_row_separation_width_factor) * median_width_um,
    )
    if len(intercepts) > 1 and np.any(np.diff(intercepts) < minimum_separation):
        return None
    fitted = slope * z + intercepts[remapped]
    residuals = x - fitted
    rss = float(np.sum(residuals * residuals))
    residual_floor = float(settings.row_model_residual_floor_width_factor) * median_width_um
    variance = max(rss / len(points), residual_floor**2, 1e-9)
    result = dict(template)
    result.update(
        {
            "labels": remapped,
            "long_reference_um": reference,
            "slope": slope,
            "slope_fixed_for_candidate": False,
            "trajectory_v25_slope_source": "free_refit_within_bounded_local_angle",
            "intercepts_um": intercepts,
            "rss": rss,
            "rms_residual_um": math.sqrt(max(rss / len(points), 0.0)),
            "bic_residual_floor_um": residual_floor,
            "bic": float(
                len(points) * math.log(variance) + (2 * k) * math.log(max(len(points), 2))
            ),
            "cluster_sizes": [int(np.count_nonzero(remapped == group)) for group in range(k)],
        }
    )
    return result


def _singleton_prefilter(
    baseline: dict[str, Any],
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV25,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Nominate a bounded pool without using detector confidence as a veto."""

    grouped = v24._group_members(baseline)
    group_by_point = {
        int(point): int(group)
        for group, members in grouped.items()
        for point in members
    }
    records: list[dict[str, Any]] = []
    for point_index in point_indices:
        group = group_by_point.get(int(point_index))
        members = grouped.get(group, []) if group is not None else []
        geometry = v23._v23_local_geometry(
            baseline,
            int(point_index),
            lateral,
            longitudinal,
            median_width_um,
        )
        residual_factor = _finite(geometry.get("candidate_residual_width_factor"))
        density = _density_eligible(int(point_index))
        small_parent_row = len(members) <= 3
        residual_nominated = bool(
            residual_factor is not None
            and residual_factor
            >= float(settings.v25_singleton_prefilter_min_residual_width_factor)
        )
        nominated = bool(density is True and (small_parent_row or residual_nominated))
        reasons: list[str] = []
        if density is not True:
            reasons.append("not_density_eligible")
        if small_parent_row:
            reasons.append("parent_row_has_at_most_three_members")
        if residual_nominated:
            reasons.append("leave_one_out_residual_reaches_prefilter_threshold")
        if density is True and not (small_parent_row or residual_nominated):
            reasons.append("residual_below_prefilter_threshold")
        records.append(
            {
                "point_index": int(point_index),
                "stoma_id": _point_stoma_id(int(point_index)),
                "parent_group": group,
                "parent_group_size": len(members),
                "density_eligible": density,
                "detector_confidence": _point_confidence(int(point_index)),
                "leave_one_out_assessable": bool(geometry.get("assessable")),
                "leave_one_out_residual_width_factor": residual_factor,
                "prefilter_nominated_before_cap": nominated,
                "prefilter_reason": ";".join(reasons),
            }
        )
    nominated_records = [record for record in records if record["prefilter_nominated_before_cap"]]
    nominated_records.sort(
        key=lambda record: (
            int(record["parent_group_size"]),
            -float(record["leave_one_out_residual_width_factor"] or -1.0),
            int(record["point_index"]),
        )
    )
    selected = nominated_records[: int(settings.v25_max_singleton_prefilter_points)]
    selected_points = {int(record["point_index"]) for record in selected}
    for record in records:
        record["prefilter_selected"] = int(record["point_index"]) in selected_points
        if record["prefilter_nominated_before_cap"] and not record["prefilter_selected"]:
            record["prefilter_reason"] += ";deterministic_pool_cap"
    return sorted(selected_points), records


def _absorb_singleton_into_nearest_supported_row(
    candidate: dict[str, Any],
    singleton_point: int,
    lateral: np.ndarray,
    longitudinal: np.ndarray,
) -> dict[str, Any] | None:
    groups = _candidate_group_members(candidate)
    singleton_group = next(
        (group for group, members in enumerate(groups) if members == [singleton_point]),
        None,
    )
    if singleton_group is None:
        return None
    supported = [group for group, members in enumerate(groups) if len(members) >= 2]
    if not supported:
        return None
    intercepts = np.asarray(candidate.get("intercepts_um", []), dtype=float)
    nearest = min(
        supported,
        key=lambda group: abs(
            float(intercepts[group]) - float(intercepts[singleton_group])
        ),
    )
    labels_by_point = v22._candidate_label_map(candidate)
    labels_by_point[int(singleton_point)] = int(nearest)
    retained_groups = [group for group in range(int(candidate["k"])) if group != singleton_group]
    remap = {old: new for new, old in enumerate(retained_groups)}
    labels = np.asarray(
        [remap[int(labels_by_point[int(index)])] for index in candidate["point_indices"]],
        dtype=int,
    )
    template = dict(candidate)
    template.update({"k": int(candidate["k"]) - 1, "labels": labels})
    try:
        return v22._refit_fixed_membership_candidate(template, lateral, longitudinal)
    except (RuntimeError, ValueError, KeyError):
        return None


def _pooled_curvature_diagnostic(
    candidate: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV25,
) -> dict[str, Any]:
    """Diagnose shared bending; do not export an incompatible curved geometry."""

    points = [int(value) for value in candidate.get("point_indices", [])]
    labels = np.asarray(candidate.get("labels", []), dtype=int)
    if len(points) < 5 or len(labels) != len(points):
        return {"assessable": False, "suspected": False, "reason": "too_few_points"}
    reference = float(candidate.get("long_reference_um") or 0.0)
    slope = float(candidate.get("slope") or 0.0)
    intercepts = np.asarray(candidate.get("intercepts_um", []), dtype=float)
    z = np.asarray([float(longitudinal[index]) - reference for index in points])
    x = np.asarray([float(lateral[index]) for index in points])
    linear_fitted = slope * z + intercepts[labels]
    residual = x - linear_fitted
    q_basis = z * z
    q_basis = q_basis - float(np.mean(q_basis))
    denominator = float(np.dot(q_basis, q_basis))
    if denominator <= 1e-12:
        return {"assessable": False, "suspected": False, "reason": "no_longitudinal_span"}
    coefficient = float(np.dot(q_basis, residual) / denominator)
    curved_residual = residual - coefficient * q_basis
    linear_rss = float(np.dot(residual, residual))
    curved_rss = float(np.dot(curved_residual, curved_residual))
    reduction = (
        (linear_rss - curved_rss) / linear_rss if linear_rss > 1e-12 else 0.0
    )
    maximum_centered_basis = float(np.max(np.abs(q_basis)))
    bend_um = abs(coefficient) * maximum_centered_basis
    bend_factor = bend_um / max(median_width_um, 1e-9)
    suspected = bool(
        reduction >= float(settings.v25_curvature_min_rss_reduction)
        and bend_factor
        >= float(settings.v25_curvature_min_end_bend_width_factor)
    )
    return {
        "assessable": True,
        "suspected": suspected,
        "reason": "systematic_shared_quadratic_residual" if suspected else "not_material",
        "coefficient_lateral_per_longitudinal_squared": coefficient,
        "linear_rss": linear_rss,
        "curved_diagnostic_rss": curved_rss,
        "rss_reduction_fraction": reduction,
        "maximum_centered_bend_um": bend_um,
        "maximum_centered_bend_width_factor": bend_factor,
        "curved_geometry_exported": False,
    }


def _evaluate_joint_candidate(
    candidate: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV25,
) -> dict[str, Any]:
    groups = _candidate_group_members(candidate)
    points = [int(value) for value in candidate.get("point_indices", [])]
    labels = np.asarray(candidate.get("labels", []), dtype=int)
    reference = float(candidate.get("long_reference_um") or 0.0)
    slope = float(candidate.get("slope") or 0.0)
    intercepts = np.asarray(candidate.get("intercepts_um", []), dtype=float)
    residual_by_group: dict[int, float] = {}
    for group, members in enumerate(groups):
        residuals = np.asarray(
            [
                float(lateral[index])
                - (
                    float(intercepts[group])
                    + slope * (float(longitudinal[index]) - reference)
                )
                for index in members
            ],
            dtype=float,
        )
        residual_by_group[group] = float(
            math.sqrt(float(np.mean(residuals * residuals)))
        )
    ordinary_rms = [
        residual_by_group[group]
        for group, members in enumerate(groups)
        if len(members) >= 2
    ]
    max_ordinary_rms_factor = (
        max(ordinary_rms) / max(median_width_um, 1e-9) if ordinary_rms else 0.0
    )
    configured_residual_floor = (
        float(settings.row_model_residual_floor_width_factor) * median_width_um
    )
    robust_ordinary_residual = max(
        float(np.median(np.asarray(ordinary_rms, dtype=float)))
        if ordinary_rms
        else configured_residual_floor,
        configured_residual_floor,
        1e-9,
    )
    singleton_evidence: list[dict[str, Any]] = []
    failures: list[str] = []
    minimum_separation = max(
        float(settings.minimum_row_separation_um),
        float(settings.v25_min_row_separation_width_factor) * median_width_um,
    )
    if len(intercepts) > 1 and np.any(
        np.diff(np.sort(intercepts)) < minimum_separation
    ):
        failures.append("adjacent_row_separation")
    if max_ordinary_rms_factor > float(settings.v25_max_ordinary_row_rms_width_factor):
        failures.append("ordinary_row_rms")
    for group, members in enumerate(groups):
        if len(members) != 1:
            continue
        point_index = int(members[0])
        other_groups = [other for other in range(len(groups)) if other != group]
        if other_groups:
            nearest_group = min(
                other_groups,
                key=lambda other: abs(float(intercepts[other]) - float(intercepts[group])),
            )
            nearest_gap = abs(float(intercepts[nearest_group]) - float(intercepts[group]))
            nearest_members = groups[nearest_group]
        else:
            nearest_group = None
            nearest_gap = None
            nearest_members = []
        gap_factor = (
            nearest_gap / max(median_width_um, 1e-9)
            if nearest_gap is not None
            else None
        )
        dominance = (
            nearest_gap / max(robust_ordinary_residual, 1e-9)
            if nearest_gap is not None
            else None
        )
        absorbed = _absorb_singleton_into_nearest_supported_row(
            candidate, point_index, lateral, longitudinal
        )
        continuation = (
            v22._local_continuation_geometry(
                point_index,
                absorbed,
                lateral,
                longitudinal,
                median_width_um,
                settings,
            )
            if absorbed is not None
            else {"assessable": False, "continuation_state": "unassessable"}
        )
        continuation_state = str(
            continuation.get("continuation_state") or "unassessable"
        )
        density = _density_eligible(point_index)
        evidence_failures: list[str] = []
        if density is not True:
            evidence_failures.append("not_density_eligible")
        if gap_factor is not None and gap_factor < float(
            settings.v25_min_row_separation_width_factor
        ):
            evidence_failures.append("singleton_lane_too_close")
        if dominance is not None and dominance < float(
            settings.v25_min_singleton_residual_dominance
        ):
            evidence_failures.append("singleton_not_distinct_from_row_noise")
        if continuation_state == "absorbed_existing_row":
            evidence_failures.append("explained_by_local_row_continuation")
        if evidence_failures:
            failures.extend(
                f"singleton_{point_index}:{reason}" for reason in evidence_failures
            )
        singleton_evidence.append(
            {
                "point_index": point_index,
                "stoma_id": _point_stoma_id(point_index),
                "density_eligible": density,
                "detector_confidence": _point_confidence(point_index),
                "nearest_group": nearest_group,
                "nearest_group_size": len(nearest_members),
                "nearest_lane_gap_um": nearest_gap,
                "nearest_lane_gap_width_factor": gap_factor,
                "residual_dominance_ratio": dominance,
                "terminal_status_used_as_veto": False,
                "continuation_state": continuation_state,
                "continuation_diagnostic": continuation,
                "geometry_passed": not evidence_failures,
                "failed_gates": evidence_failures,
            }
        )
    singleton_penalty = 0.0
    for evidence in singleton_evidence:
        confidence = evidence.get("detector_confidence")
        confidence_for_penalty = 0.5 if confidence is None else min(max(float(confidence), 0.0), 1.0)
        singleton_penalty += float(settings.v25_soft_singleton_penalty) * (
            1.15 - 0.30 * confidence_for_penalty
        )
    unresolved_count = sum(
        evidence.get("continuation_state") == "unresolved_keep_conservative"
        for evidence in singleton_evidence
    )
    unresolved_penalty = unresolved_count * float(
        settings.v25_unresolved_curvature_penalty
    )
    angle = math.degrees(math.atan(slope))
    slope_penalty = float(settings.v25_slope_penalty_weight) * (
        abs(angle) / max(float(settings.v25_local_residual_tilt_limit_deg), 1e-9)
    ) ** 2
    curvature = _pooled_curvature_diagnostic(
        candidate, lateral, longitudinal, median_width_um, settings
    )
    candidate_bic = _finite(candidate.get("bic"))
    if candidate_bic is None:
        candidate_bic = math.inf
    adjusted_score = (
        candidate_bic
        + singleton_penalty
        + unresolved_penalty
        + slope_penalty
    )
    return {
        "candidate": candidate,
        "valid": not failures,
        "failed_gates": failures,
        "adjusted_score": adjusted_score,
        "original_bic": candidate_bic,
        "candidate_row_count": int(candidate.get("k", 0)),
        "cluster_sizes": [int(value) for value in candidate.get("cluster_sizes", [])],
        "canonical_partition": _canonical_partition(candidate),
        "singleton_row_count": len(singleton_evidence),
        "singleton_penalty": singleton_penalty,
        "singleton_evidence": singleton_evidence,
        "maximum_ordinary_row_rms_width_factor": max_ordinary_rms_factor,
        "robust_ordinary_row_residual_um": robust_ordinary_residual,
        "unresolved_curvature_penalty": unresolved_penalty,
        "slope_angle_deg": angle,
        "slope_penalty": slope_penalty,
        "curvature_diagnostic": curvature,
        "nonlinear_trajectory_suspected": bool(curvature.get("suspected")),
    }


def _record_for_csv(diagnostic: dict[str, Any]) -> dict[str, Any]:
    """Remove the full NumPy candidate while retaining transparent evidence."""

    return {
        key: value
        for key, value in diagnostic.items()
        if key != "candidate"
    }


def _angle_grid(
    center_deg: float,
    half_range_deg: float,
    step_deg: float,
    limit_deg: float,
) -> list[float]:
    lower = max(-limit_deg, center_deg - half_range_deg)
    upper = min(limit_deg, center_deg + half_range_deg)
    values = np.arange(lower, upper + step_deg * 0.5, step_deg, dtype=float).tolist()
    values.extend([0.0, center_deg])
    return sorted(
        {
            round(float(np.clip(value, -limit_deg, limit_deg)), 10)
            for value in values
        }
    )


def _enumerate_joint_candidates_at_angles(
    point_indices: list[int],
    singleton_pool: list[int],
    row_counts: list[int],
    angles: list[float],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV25,
    retained: dict[tuple[tuple[int, ...], ...], dict[str, Any]],
) -> None:
    subsets: list[tuple[int, ...]] = [()]
    maximum = min(
        int(settings.v25_max_singletons_per_provisional_group),
        len(singleton_pool),
    )
    for count in range(1, maximum + 1):
        subsets.extend(itertools.combinations(singleton_pool, count))
    for angle in angles:
        slope = math.tan(math.radians(float(angle)))
        for singleton_subset in subsets:
            for k in row_counts:
                candidate = _build_joint_candidate(
                    point_indices,
                    singleton_subset,
                    k,
                    slope,
                    lateral,
                    longitudinal,
                    median_width_um,
                    settings,
                )
                if candidate is None:
                    continue
                diagnostic = _evaluate_joint_candidate(
                    candidate, lateral, longitudinal, median_width_um, settings
                )
                canonical = diagnostic["canonical_partition"]
                existing = retained.get(canonical)
                rank = (
                    not bool(diagnostic["valid"]),
                    float(diagnostic["adjusted_score"]),
                    int(diagnostic["singleton_row_count"]),
                    abs(float(diagnostic["slope_angle_deg"])),
                )
                if existing is None:
                    retained[canonical] = diagnostic
                else:
                    existing_rank = (
                        not bool(existing["valid"]),
                        float(existing["adjusted_score"]),
                        int(existing["singleton_row_count"]),
                        abs(float(existing["slope_angle_deg"])),
                    )
                    if rank < existing_rank:
                        retained[canonical] = diagnostic


def _enumerate_joint_candidates(
    baseline: dict[str, Any],
    point_indices: list[int],
    singleton_pool: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV25,
) -> list[dict[str, Any]]:
    maximum_k = min(
        int(settings.trajectory_max_rows_per_supergroup), len(point_indices)
    )
    # Compare the complete biologically permitted row-count range.  A narrow
    # search around the inherited answer cannot correct the audited cases in
    # which a tilted/bent trajectory was fragmented into one row per stoma.
    row_counts = list(range(1, maximum_k + 1))
    retained: dict[tuple[tuple[int, ...], ...], dict[str, Any]] = {}
    limit = float(settings.v25_local_residual_tilt_limit_deg)
    coarse = _angle_grid(
        0.0,
        limit,
        float(settings.v25_local_residual_tilt_coarse_step_deg),
        limit,
    )
    _enumerate_joint_candidates_at_angles(
        point_indices,
        singleton_pool,
        row_counts,
        coarse,
        lateral,
        longitudinal,
        median_width_um,
        settings,
        retained,
    )
    valid = [record for record in retained.values() if bool(record["valid"])]
    if valid:
        best_coarse = min(
            valid,
            key=lambda record: (
                float(record["adjusted_score"]),
                int(record["candidate_row_count"]),
                int(record["singleton_row_count"]),
            ),
        )
        fine = _angle_grid(
            float(best_coarse["slope_angle_deg"]),
            float(settings.v25_local_residual_tilt_fine_half_range_deg),
            float(settings.v25_local_residual_tilt_fine_step_deg),
            limit,
        )
        _enumerate_joint_candidates_at_angles(
            point_indices,
            singleton_pool,
            row_counts,
            fine,
            lateral,
            longitudinal,
            median_width_um,
            settings,
            retained,
        )
    return sorted(
        retained.values(),
        key=lambda record: (
            not bool(record["valid"]),
            float(record["adjusted_score"]),
            int(record["candidate_row_count"]),
            int(record["singleton_row_count"]),
            abs(float(record["slope_angle_deg"])),
            record["canonical_partition"],
        ),
    )


def _choose_scored_record(
    records: list[dict[str, Any]], settings: TraitSettingsV25
) -> dict[str, Any] | None:
    valid = [record for record in records if bool(record.get("valid"))]
    if not valid:
        return None
    strict = min(
        valid,
        key=lambda record: (
            float(record["adjusted_score"]),
            int(record["candidate_row_count"]),
            int(record["singleton_row_count"]),
            abs(float(record["slope_angle_deg"])),
            record["canonical_partition"],
        ),
    )
    threshold = float(strict["adjusted_score"]) + float(
        settings.v25_simpler_tie_tolerance
    )
    tied = [record for record in valid if float(record["adjusted_score"]) <= threshold]
    return min(
        tied,
        key=lambda record: (
            int(record["candidate_row_count"]),
            int(record["singleton_row_count"]),
            float(record["adjusted_score"]),
            abs(float(record["slope_angle_deg"])),
            record["canonical_partition"],
        ),
    )


def _top_candidate_diagnostics(
    records: list[dict[str, Any]], selected: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Keep one transparent best candidate per k plus representative failures."""

    best_by_k: dict[int, dict[str, Any]] = {}
    for record in records:
        k = int(record["candidate_row_count"])
        existing = best_by_k.get(k)
        if existing is None or (
            not bool(record["valid"]),
            float(record["adjusted_score"]),
            int(record["singleton_row_count"]),
        ) < (
            not bool(existing["valid"]),
            float(existing["adjusted_score"]),
            int(existing["singleton_row_count"]),
        ):
            best_by_k[k] = record
    retained = list(best_by_k.values())
    if selected is not None and all(
        not (
            record["canonical_partition"] == selected["canonical_partition"]
            and math.isclose(
                float(record["slope_angle_deg"]),
                float(selected["slope_angle_deg"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(record["adjusted_score"]),
                float(selected["adjusted_score"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        for record in retained
    ):
        retained.append(selected)
    failed = [record for record in records if not bool(record["valid"])][:20]
    retained.extend(
        record
        for record in failed
        if all(
            existing["canonical_partition"] != record["canonical_partition"]
            for existing in retained
        )
    )
    output: list[dict[str, Any]] = []
    selected_partition = selected.get("canonical_partition") if selected else None
    selected_angle = selected.get("slope_angle_deg") if selected else None
    selected_score = selected.get("adjusted_score") if selected else None
    for rank, record in enumerate(
        sorted(
            retained,
            key=lambda item: (
                not bool(item["valid"]),
                float(item["adjusted_score"]),
                int(item["candidate_row_count"]),
            ),
        ),
        start=1,
    ):
        compact = _record_for_csv(record)
        compact["rank_within_retained_diagnostics"] = rank
        compact["selected"] = bool(
            record["canonical_partition"] == selected_partition
            and selected_angle is not None
            and selected_score is not None
            and math.isclose(
                float(record["slope_angle_deg"]),
                float(selected_angle),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(record["adjusted_score"]),
                float(selected_score),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        compact["canonical_partition"] = [list(group) for group in record["canonical_partition"]]
        output.append(compact)
    return output


def _parent_compatible_candidate_diagnostics(
    records: list[dict[str, Any]],
    selected: dict[str, Any],
    median_width_um: float,
    settings: TraitSettingsV25,
) -> list[dict[str, Any]]:
    """Retain the complete v1.4/v1.5 schema consumed downstream.

    The legacy audit builder indexes these records by row count and reads its
    spacing, singleton and pair-partition fields directly.  V2.5 therefore
    augments that schema instead of replacing it with its compact diagnostics.
    The v2.5 adjusted score is the selection score; the inherited score is
    retained under an explicit audit-only name.
    """

    best_by_k: dict[int, dict[str, Any]] = {}
    for record in records:
        if not bool(record.get("valid")):
            continue
        k = int(record["candidate_row_count"])
        existing = best_by_k.get(k)
        if existing is None or (
            float(record["adjusted_score"]),
            int(record["singleton_row_count"]),
            abs(float(record["slope_angle_deg"])),
            record["canonical_partition"],
        ) < (
            float(existing["adjusted_score"]),
            int(existing["singleton_row_count"]),
            abs(float(existing["slope_angle_deg"])),
            existing["canonical_partition"],
        ):
            best_by_k[k] = record

    # The candidate-model audit must describe the exact geometry used to build
    # rows, even when a slightly better same-k alternative did not reach the
    # versioned change threshold.
    best_by_k[int(selected["candidate_row_count"])] = selected

    output: list[dict[str, Any]] = []
    for k in sorted(best_by_k):
        record = best_by_k[k]
        inherited = v15._score_trajectory_candidate(
            record["candidate"],
            median_width_um,
            float(record["slope_angle_deg"]),
            settings,
        )
        inherited_score = _finite(inherited.get("adjusted_score"))
        inherited.update(
            {
                "candidate_row_count": k,
                "v25_parent_schema_adjusted_score": inherited_score,
                "adjusted_score": float(record["adjusted_score"]),
                "trajectory_adjusted_score": float(record["adjusted_score"]),
                "original_bic": float(record["original_bic"]),
                "singleton_row_count": int(record["singleton_row_count"]),
                "trajectory_v25_valid": True,
                "trajectory_v25_failed_gates": list(record["failed_gates"]),
                "trajectory_v25_cluster_sizes": list(record["cluster_sizes"]),
                "trajectory_v25_canonical_partition": [
                    list(group) for group in record["canonical_partition"]
                ],
                "trajectory_v25_singleton_evidence": record[
                    "singleton_evidence"
                ],
                "trajectory_v25_soft_singleton_penalty": float(
                    record["singleton_penalty"]
                ),
                "trajectory_v25_slope_penalty": float(record["slope_penalty"]),
                "trajectory_v25_curvature_diagnostic": record[
                    "curvature_diagnostic"
                ],
            }
        )
        output.append(inherited)
    return output


def _stability_for_selected_topology(
    selected: dict[str, Any],
    competing_records: list[dict[str, Any]],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV25,
) -> dict[str, Any]:
    points = [int(value) for value in selected["candidate"].get("point_indices", [])]
    unique_competitors: dict[tuple[tuple[int, ...], ...], dict[str, Any]] = {}
    for record in [selected, *competing_records]:
        if not bool(record.get("valid")):
            continue
        canonical = record["canonical_partition"]
        existing = unique_competitors.get(canonical)
        if existing is None or float(record["adjusted_score"]) < float(
            existing["adjusted_score"]
        ):
            unique_competitors[canonical] = record
    ranked_competitors = sorted(
        unique_competitors.values(),
        key=lambda record: (
            float(record["adjusted_score"]),
            int(record["candidate_row_count"]),
        ),
    )
    best_score = float(ranked_competitors[0]["adjusted_score"])
    competitors = [
        record
        for record in ranked_competitors
        if float(record["adjusted_score"])
        <= best_score + float(settings.v25_stability_delta_score_window)
    ]
    best_by_k: dict[int, dict[str, Any]] = {}
    for record in ranked_competitors:
        best_by_k.setdefault(int(record["candidate_row_count"]), record)
    for record in best_by_k.values():
        if all(
            existing["canonical_partition"] != record["canonical_partition"]
            for existing in competitors
        ):
            competitors.append(record)
    if all(
        record["canonical_partition"] != selected["canonical_partition"]
        for record in competitors
    ):
        competitors.append(selected)

    seed_offset = sum(
        (position + 1) * (point + 1) for position, point in enumerate(points)
    ) + 2501 * int(selected["candidate_row_count"])
    rng = np.random.default_rng(
        int(settings.trajectory_stability_seed) + seed_offset
    )
    iterations = max(int(settings.trajectory_stability_iterations), 1)
    jitter = float(settings.trajectory_stability_jitter_width_fraction) * median_width_um
    row_count_votes: dict[int, int] = {}
    agreements: list[float] = []
    singleton_votes = {point: 0 for point in points}
    selected_k = int(selected["candidate_row_count"])
    selected_candidate = selected["candidate"]
    for _ in range(iterations):
        perturbed = np.asarray(lateral, dtype=float).copy()
        perturbed[points] += rng.normal(0.0, jitter, len(points))
        refitted_records: list[dict[str, Any]] = []
        for record in competitors:
            refitted = _refit_fixed_partition_free_slope(
                record["candidate"],
                perturbed,
                longitudinal,
                median_width_um,
                settings,
            )
            if refitted is None:
                continue
            evaluated = _evaluate_joint_candidate(
                refitted,
                perturbed,
                longitudinal,
                median_width_um,
                settings,
            )
            if bool(evaluated["valid"]):
                refitted_records.append(evaluated)
        chosen = _choose_scored_record(refitted_records, settings)
        if chosen is None:
            continue
        chosen_k = int(chosen["candidate_row_count"])
        row_count_votes[chosen_k] = row_count_votes.get(chosen_k, 0) + 1
        if chosen_k == selected_k:
            agreements.append(
                _pairwise_comembership_agreement(selected_candidate, chosen["candidate"])
            )
        for point in _singleton_points(chosen["candidate"]):
            singleton_votes[int(point)] = singleton_votes.get(int(point), 0) + 1
    evaluated_iterations = sum(row_count_votes.values())
    denominator = iterations
    evaluated_fraction = evaluated_iterations / denominator
    row_stability = row_count_votes.get(selected_k, 0) / denominator
    assignment_stability = float(sum(agreements) / denominator)
    singleton_frequency = {
        int(point): votes / denominator for point, votes in singleton_votes.items()
    }
    nominal_singletons = _singleton_points(selected_candidate)
    nominal_frequencies = [singleton_frequency.get(point, 0.0) for point in nominal_singletons]
    identity_stability = min(nominal_frequencies) if nominal_frequencies else 1.0
    supported = [
        point
        for point in nominal_singletons
        if singleton_frequency.get(point, 0.0)
        >= float(settings.v25_stability_min_singleton_identity_fraction)
    ]
    ambiguous = [point for point in nominal_singletons if point not in supported]
    possible = [
        point
        for point, frequency in singleton_frequency.items()
        if point not in supported
        and frequency >= float(settings.v25_possible_singleton_frequency)
    ]
    stable_counts = [
        count
        for count, votes in row_count_votes.items()
        if votes / denominator >= 0.10
    ]
    stable_counts.append(selected_k)
    return {
        "evaluated_iterations": evaluated_iterations,
        "requested_iterations": iterations,
        "evaluated_fraction": evaluated_fraction,
        "stability_trials_sufficient": evaluated_fraction
        >= float(settings.v25_stability_min_evaluated_fraction),
        "stability_scope": (
            "fixed_candidate_partitions_within_delta_score_window_plus_best_per_k"
        ),
        "candidate_partition_count": len(competitors),
        "row_count_stability_fraction": row_stability,
        "assignment_stability_fraction": assignment_stability,
        "singleton_identity_stability_fraction": identity_stability,
        "row_count_distribution": {
            str(key): value for key, value in sorted(row_count_votes.items())
        },
        "singleton_frequency_by_point": singleton_frequency,
        "supported_singleton_points": sorted(supported),
        "ambiguous_singleton_points": sorted(ambiguous),
        "possible_singleton_points": sorted(possible),
        "row_count_lower_bound": min(stable_counts),
        "row_count_upper_bound": max(stable_counts),
        "row_count_stable": row_stability
        >= float(settings.v25_stability_min_row_count_fraction),
        "assignment_stable": assignment_stability
        >= float(settings.v25_stability_min_assignment_fraction),
        "singleton_identity_stable": identity_stability
        >= float(settings.v25_stability_min_singleton_identity_fraction),
    }


def _clear_superseded_parent_topology_fields(
    chosen: dict[str, Any], diagnostic: dict[str, Any]
) -> None:
    updates = {
        "trajectory_v20_singleton_rescue_applied": False,
        "trajectory_v20_singleton_validation_status": "superseded_by_v25_joint_topology",
        "trajectory_v21_singleton_rescue_applied": False,
        "trajectory_v22_leave_one_out_applied": False,
        "trajectory_v22_leave_one_out_status": "superseded_by_v25_joint_topology",
        "trajectory_v23_additional_singleton_applied": False,
        "trajectory_v23_rescue_status": "superseded_by_v25_joint_topology",
        "trajectory_v24_fragment_merge_applied": False,
        "trajectory_v24_fragment_merge_status": "superseded_by_v25_joint_topology",
        "trajectory_v24_singleton_applied": False,
        "trajectory_v24_singleton_status": "superseded_by_v25_joint_topology",
        "trajectory_v24_singleton_candidate_point_index": None,
        "trajectory_v24_singleton_candidate_stoma_id": "",
    }
    chosen.update(updates)
    diagnostic.update(updates)


def _apply_v25_joint_search(
    parent_chosen: dict[str, Any],
    parent_candidates: list[dict[str, Any]],
    diagnostic: dict[str, Any],
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV25,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    singleton_pool, prefilter = _singleton_prefilter(
        parent_chosen,
        point_indices,
        lateral,
        longitudinal,
        median_width_um,
        settings,
    )
    enumerated = _enumerate_joint_candidates(
        parent_chosen,
        point_indices,
        singleton_pool,
        lateral,
        longitudinal,
        median_width_um,
        settings,
    )
    # The baseline is the literal inherited v2.4 geometry.  A free local-slope
    # refit is a v2.5 change and must compete explicitly rather than altering
    # the "retained parent" silently.
    baseline_candidate = parent_chosen
    baseline = _evaluate_joint_candidate(
        baseline_candidate,
        lateral,
        longitudinal,
        median_width_um,
        settings,
    )
    all_records = list(enumerated)
    if all(
        record["canonical_partition"] != baseline["canonical_partition"]
        for record in all_records
    ):
        all_records.append(baseline)
    proposed = _choose_scored_record(all_records, settings)
    if proposed is None:
        fields = {
            **_V25_DEFAULTS,
            "trajectory_v25_joint_search_performed": True,
            "trajectory_v25_joint_candidate_count": len(all_records),
            "trajectory_v25_joint_status": "fallback_parent_no_valid_joint_candidate",
            "trajectory_v25_joint_reason": "all_joint_candidates_failed_structural_gates",
            "trajectory_v25_baseline_row_count": int(parent_chosen.get("k", 0)),
            "trajectory_v25_selected_row_count": int(parent_chosen.get("k", 0)),
            "trajectory_v25_singleton_prefilter_diagnostics": prefilter,
            "trajectory_v25_candidate_diagnostics": _top_candidate_diagnostics(
                all_records, None
            ),
        }
        parent_chosen.update(fields)
        diagnostic.update(fields)
        return parent_chosen, parent_candidates, diagnostic

    same_partition = proposed["canonical_partition"] == baseline["canonical_partition"]
    same_k = int(proposed["candidate_row_count"]) == int(baseline["candidate_row_count"])
    score_improvement = float(baseline["adjusted_score"]) - float(
        proposed["adjusted_score"]
    )
    required = float(
        settings.v25_min_score_improvement_same_row_count
        if same_k
        else settings.v25_min_score_improvement_new_row_count
    )
    parent_invalid = not bool(baseline.get("valid"))
    apply_change = bool(
        parent_invalid
        or (not same_partition and score_improvement >= required)
        or (
            same_partition
            and abs(float(proposed["slope_angle_deg"]) - float(baseline["slope_angle_deg"]))
            >= float(settings.v25_local_residual_tilt_fine_step_deg)
            and score_improvement >= float(settings.v25_min_score_improvement_same_row_count)
        )
    )
    selected = proposed if apply_change else baseline
    competitors = [record for record in all_records if record is not selected]
    stability = _stability_for_selected_topology(
        selected,
        competitors,
        lateral,
        longitudinal,
        median_width_um,
        settings,
    )
    supported_points = stability["supported_singleton_points"]
    ambiguous_points = stability["ambiguous_singleton_points"]
    possible_points = stability["possible_singleton_points"]
    topology_stable = bool(
        stability["stability_trials_sufficient"]
        and stability["row_count_stable"]
        and stability["assignment_stable"]
        and stability["singleton_identity_stable"]
    )
    nonlinear = bool(selected.get("nonlinear_trajectory_suspected"))
    confidence_tier = (
        "supported"
        if topology_stable and not nonlinear
        else "ambiguous_insufficient_stability_trials"
        if not stability["stability_trials_sufficient"]
        else "ambiguous_nonlinear"
        if nonlinear
        else "ambiguous_stability"
    )
    selected_candidate = selected["candidate"]
    chosen = dict(parent_chosen)
    chosen.update(selected_candidate)
    selected_singletons = _singleton_points(selected_candidate)
    fields = {
        "trajectory_v25_policy_version": SCRIPT_VERSION,
        "trajectory_v25_joint_search_performed": True,
        "trajectory_v25_joint_candidate_count": len(all_records),
        "trajectory_v25_joint_change_applied": apply_change,
        "trajectory_v25_joint_status": (
            "selected_supported_joint_topology"
            if topology_stable and apply_change
            else "selected_ambiguous_joint_topology"
            if apply_change
            else "parent_topology_retained_after_joint_comparison"
        ),
        "trajectory_v25_joint_reason": (
            "joint_membership_or_band_local_angle_improved_versioned_score"
            if apply_change
            else "joint_alternative_did_not_reach_versioned_improvement_threshold"
        ),
        "trajectory_v25_baseline_row_count": int(parent_chosen.get("k", 0)),
        "trajectory_v25_selected_row_count": int(selected_candidate["k"]),
        "trajectory_v25_baseline_cluster_sizes": [
            int(value) for value in parent_chosen.get("cluster_sizes", [])
        ],
        "trajectory_v25_selected_cluster_sizes": [
            int(value) for value in selected_candidate.get("cluster_sizes", [])
        ],
        "trajectory_v25_score_improvement": score_improvement,
        "trajectory_v25_pairwise_comembership_vs_baseline": (
            _pairwise_comembership_agreement(parent_chosen, selected_candidate)
        ),
        "trajectory_v25_selected_singleton_count": len(selected_singletons),
        "trajectory_v25_supported_singleton_count": len(supported_points),
        "trajectory_v25_ambiguous_singleton_count": len(ambiguous_points),
        "trajectory_v25_selected_singleton_point_indices": selected_singletons,
        "trajectory_v25_selected_singleton_stoma_ids": [
            _point_stoma_id(point) for point in selected_singletons
        ],
        "trajectory_v25_supported_singleton_stoma_ids": [
            _point_stoma_id(point) for point in supported_points
        ],
        "trajectory_v25_ambiguous_singleton_stoma_ids": [
            _point_stoma_id(point) for point in ambiguous_points
        ],
        "trajectory_v25_possible_singleton_stoma_ids": [
            _point_stoma_id(point) for point in possible_points
        ],
        "trajectory_v25_singleton_detector_confidences": [
            _point_confidence(point) for point in selected_singletons
        ],
        "trajectory_v25_singleton_prefilter_diagnostics": prefilter,
        "trajectory_v25_singleton_evidence": selected["singleton_evidence"],
        "trajectory_v25_candidate_diagnostics": _top_candidate_diagnostics(
            all_records, selected
        ),
        "trajectory_v25_row_count_stability_fraction": stability[
            "row_count_stability_fraction"
        ],
        "trajectory_v25_assignment_stability_fraction": stability[
            "assignment_stability_fraction"
        ],
        "trajectory_v25_singleton_identity_stability_fraction": stability[
            "singleton_identity_stability_fraction"
        ],
        "trajectory_v25_stability_evaluated_fraction": stability[
            "evaluated_fraction"
        ],
        "trajectory_v25_stability_trials_sufficient": stability[
            "stability_trials_sufficient"
        ],
        "trajectory_v25_stability_scope": stability["stability_scope"],
        "trajectory_v25_stability_count_distribution": stability[
            "row_count_distribution"
        ],
        "trajectory_v25_singleton_frequency_by_stoma": {
            _point_stoma_id(point): frequency
            for point, frequency in stability["singleton_frequency_by_point"].items()
        },
        "trajectory_v25_row_count_lower_bound": stability["row_count_lower_bound"],
        "trajectory_v25_row_count_upper_bound": stability["row_count_upper_bound"],
        "trajectory_v25_topology_confidence_tier": confidence_tier,
        "trajectory_v25_band_local_residual_angle_deg": selected["slope_angle_deg"],
        "trajectory_v25_residual_angle_at_search_bound": bool(
            abs(float(selected["slope_angle_deg"]))
            >= float(settings.v25_local_residual_tilt_limit_deg) - 1e-9
        ),
        "trajectory_v25_nonlinear_trajectory_suspected": nonlinear,
        "trajectory_v25_curvature_diagnostic": selected["curvature_diagnostic"],
        "trajectory_v25_parent_v24_topology_superseded": bool(
            apply_change and not same_partition
        ),
        "automatic_row_count": int(selected_candidate["k"]),
        "automatic_selection_reason": (
            "v25_joint_band_local_shared_slope_partition"
        ),
        "row_count_selection_reason": (
            "v25_joint_best_estimate_supported"
            if topology_stable
            else "v25_joint_best_estimate_ambiguous"
        ),
        "tilt_slope_source": "image_axis_plus_v25_band_local_shared_residual_angle",
        "sparse_row_support": any(
            int(value) < 3 for value in selected_candidate.get("cluster_sizes", [])
        ),
        "model_confidence": "moderate" if topology_stable and not selected_singletons else "low",
        "trajectory_row_count_stability_fraction": stability[
            "row_count_stability_fraction"
        ],
        "trajectory_assignment_stability_fraction": stability[
            "assignment_stability_fraction"
        ],
        "trajectory_row_count_stable": topology_stable,
        "trajectory_singleton_validation_required": bool(selected_singletons),
        "trajectory_singleton_rows_validated": bool(
            not selected_singletons or not ambiguous_points
        ),
        "trajectory_singleton_detector_confidences": [
            _point_confidence(point) for point in selected_singletons
        ],
        "trajectory_singleton_validation_failure_reason": (
            "" if not ambiguous_points else "v25_singleton_identity_stability_below_threshold"
        ),
        "trajectory_row_count_min": stability["row_count_lower_bound"],
        "trajectory_row_count_max": stability["row_count_upper_bound"],
        "trajectory_row_count_range_reason": "v25_joint_perturbation_bounds",
        "trajectory_exact_row_traits_eligible": bool(topology_stable and not nonlinear),
        "trajectory_maximum_k_reached": bool(
            int(selected_candidate["k"])
            >= int(settings.trajectory_max_rows_per_supergroup)
            and len(point_indices) > int(settings.trajectory_max_rows_per_supergroup)
        ),
        "one_vs_two_ambiguous": bool(
            stability["row_count_lower_bound"] <= 1
            and stability["row_count_upper_bound"] >= 2
        ),
        "automatic_one_vs_two_ambiguous": bool(
            stability["row_count_lower_bound"] <= 1
            and stability["row_count_upper_bound"] >= 2
        ),
        "audited_row_count_override": False,
    }
    # Maintain the core fields consumed by the original row-record builder.
    one_candidates = [
        record for record in all_records if int(record["candidate_row_count"]) == 1
    ]
    if one_candidates:
        fields["one_row_rms_residual_um"] = float(
            min(one_candidates, key=lambda record: float(record["adjusted_score"]))[
                "candidate"
            ]["rms_residual_um"]
        )
    else:
        fields["one_row_rms_residual_um"] = parent_chosen.get(
            "one_row_rms_residual_um"
        )
    simpler = [
        record
        for record in all_records
        if bool(record["valid"])
        and int(record["candidate_row_count"]) == int(selected_candidate["k"]) - 1
    ]
    fields["delta_bic_vs_simpler"] = (
        float(min(simpler, key=lambda record: float(record["adjusted_score"]))["candidate"]["bic"])
        - float(selected_candidate["bic"])
        if simpler
        else None
    )
    chosen.update(fields)
    if bool(fields["trajectory_v25_parent_v24_topology_superseded"]):
        _clear_superseded_parent_topology_fields(chosen, diagnostic)

    diagnostic.update(
        {
            **fields,
            "global_automatic_row_count": int(selected_candidate["k"]),
            "row_count_used_for_traits": int(selected_candidate["k"]),
            "row_count_used_for_traits_source": "v25_joint_best_estimate",
            "alternative_row_count": int(baseline["candidate_row_count"]),
            "global_row_selection_score": float(selected["adjusted_score"]),
            "global_row_score_margin": max(0.0, score_improvement),
            "global_row_raw_margin": score_improvement,
            "global_row_automatic_is_strict_best": topology_stable,
            "global_row_model_confidence": fields["model_confidence"],
            "singleton_supported_row_count": len(supported_points),
            "singleton_detector_confidence_mean": (
                float(
                    np.mean(
                        [
                            value
                            for value in fields[
                                "trajectory_v25_singleton_detector_confidences"
                            ]
                            if value is not None
                        ]
                    )
                )
                if any(
                    value is not None
                    for value in fields[
                        "trajectory_v25_singleton_detector_confidences"
                    ]
                )
                else None
            ),
            "candidate_diagnostics": _parent_compatible_candidate_diagnostics(
                all_records,
                selected,
                median_width_um,
                settings,
            ),
        }
    )
    best_by_k: dict[int, dict[str, Any]] = {}
    for record in all_records:
        if not bool(record["valid"]):
            continue
        k = int(record["candidate_row_count"])
        if k not in best_by_k or float(record["adjusted_score"]) < float(
            best_by_k[k]["adjusted_score"]
        ):
            best_by_k[k] = record
    best_by_k[int(selected["candidate_row_count"])] = selected
    returned_candidates = [best_by_k[k]["candidate"] for k in sorted(best_by_k)]
    return chosen, returned_candidates or parent_candidates, diagnostic


def select_row_model_v25(
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV25,
    audited_row_count: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run v2.4 once, then perform one deterministic joint v2.5 search."""

    diagnostic_count = len(v14._ACTIVE_BAND_DIAGNOSTICS)
    chosen, candidates = v24.select_row_model_v24(
        point_indices,
        lateral,
        longitudinal,
        median_detector_width_um,
        settings,
        audited_row_count=audited_row_count,
    )
    if len(v14._ACTIVE_BAND_DIAGNOSTICS) != diagnostic_count + 1:
        raise RuntimeError("v2.5 expected exactly one provisional-band diagnostic record.")
    diagnostic = v14._ACTIVE_BAND_DIAGNOSTICS[-1]
    if audited_row_count is None:
        chosen, candidates, diagnostic = _apply_v25_joint_search(
            chosen,
            candidates,
            diagnostic,
            point_indices,
            lateral,
            longitudinal,
            median_detector_width_um,
            settings,
        )
    else:
        fields = {
            **_V25_DEFAULTS,
            "trajectory_v25_joint_status": "audited_override_preserved",
            "trajectory_v25_joint_reason": "joint_search_not_applied_to_manual_override",
            "trajectory_v25_baseline_row_count": int(chosen.get("k", 0)),
            "trajectory_v25_selected_row_count": int(chosen.get("k", 0)),
            "trajectory_v25_row_count_lower_bound": int(chosen.get("k", 0)),
            "trajectory_v25_row_count_upper_bound": int(chosen.get("k", 0)),
            "trajectory_v25_topology_confidence_tier": "audited",
        }
        chosen.update(fields)
        diagnostic.update(fields)
    for key, default in _V25_DEFAULTS.items():
        value = chosen.get(key, diagnostic.get(key, default))
        chosen[key] = value
        diagnostic[key] = value
    return chosen, candidates


def _trajectory_diagnostic_fields_v25(
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    fields = v24._trajectory_diagnostic_fields_v24(diagnostic)
    fields["row_model_version"] = SCRIPT_VERSION
    for key, default in _V25_DEFAULTS.items():
        fields[key] = diagnostic.get(key, default)
    return fields


def _v25_partition_candidate(
    band: dict[str, Any],
    linked: list[dict[str, Any]],
    cut: int,
    gaps: list[float],
    boundary_evidence_by_rows: dict[
        tuple[str, str, str], dict[str, Any]
    ],
    median_width_um: float,
    settings: TraitSettingsV25,
) -> dict[str, Any]:
    left = linked[:cut]
    right = linked[cut:]
    gap_um = float(gaps[cut - 1])
    other_gaps = [value for position, value in enumerate(gaps) if position != cut - 1]
    reference = (
        float(np.median(np.asarray(other_gaps, dtype=float))) if other_gaps else None
    )
    gap_ratio = gap_um / max(reference, 1e-9) if reference is not None else math.inf
    gap_factor = gap_um / max(median_width_um, 1e-9)
    provisional = str(band.get("provisional_band_id") or band.get("band_id") or "")
    boundary_key = (
        provisional,
        str(left[-1].get("row_id") or ""),
        str(right[0].get("row_id") or ""),
    )
    # Recursive child cuts reuse local numbers.  Adjacent row identities are
    # invariant, so they are the key for the original raw-image evidence.  If
    # identity is unavailable, support remains unavailable rather than being
    # borrowed from a physically different boundary.
    inherited = boundary_evidence_by_rows.get(boundary_key)
    image_support_identity_exact = bool(
        inherited is not None
        and base.parse_bool(inherited.get("image_support_identity_exact")) is True
        and str(inherited.get("provisional_band_id") or "") == boundary_key[0]
        and str(inherited.get("boundary_left_row_id") or "") == boundary_key[1]
        and str(inherited.get("boundary_right_row_id") or "") == boundary_key[2]
        and str(inherited.get("image_support_evidence_source") or "")
        == "v25_direct_exact_pair_local_trajectory_parallel_flanks"
    )
    image_support = (
        _finite(inherited.get("image_support_ratio"))
        if image_support_identity_exact and inherited is not None
        else None
    )
    stability = v22._strong_gap_stability(
        gap_um,
        median_width_um,
        settings,
        sum(ord(char) for char in provisional) + cut + 2500,
    )
    left_stomata = sum(int(row.get("stoma_count") or 0) for row in left)
    right_stomata = sum(int(row.get("stoma_count") or 0) for row in right)
    isolated_one_row_side = min(len(left), len(right)) == 1
    isolated_stomata = min(left_stomata, right_stomata) if isolated_one_row_side else None
    topology_tiers = {
        str(row.get("trajectory_v25_topology_confidence_tier") or "not_assessed")
        for row in linked
    }
    topology_supported = bool(
        topology_tiers
        and topology_tiers.issubset({"supported", "audited"})
    )
    nonlinear_topology = any(
        base.parse_bool(row.get("trajectory_v25_nonlinear_trajectory_suspected"))
        is True
        for row in linked
    )
    failures: list[str] = []
    if not topology_supported:
        failures.append("row_topology_not_supported")
    if nonlinear_topology:
        failures.append("nonlinear_row_topology")
    if gap_factor < float(settings.v25_partition_min_gap_width_factor):
        failures.append("gap_width")
    if gap_ratio < float(settings.v25_partition_min_local_gap_ratio):
        failures.append("local_gap_ratio")
    if stability < float(settings.v25_partition_min_stability_fraction):
        failures.append("boundary_stability")
    if min(left_stomata, right_stomata) < 1:
        failures.append("empty_side")
    decision_route = "multirow_sides_strong_geometry"
    if isolated_one_row_side:
        if isolated_stomata is not None and isolated_stomata >= int(
            settings.v25_partition_min_stomata_per_side
        ):
            decision_route = "isolated_multipoint_row_geometry_plus_raw_image"
            if image_support is None or image_support < float(
                settings.v25_partition_min_image_support_ratio_isolated_row
            ):
                failures.append("isolated_row_raw_image_support")
        else:
            decision_route = "singleton_side_extra_strong_geometry_plus_raw_image"
            if gap_factor < float(
                settings.v25_partition_singleton_side_min_gap_width_factor
            ):
                failures.append("singleton_side_gap_width")
            if image_support is None or image_support < float(
                settings.v25_partition_singleton_side_min_image_support_ratio
            ):
                failures.append("singleton_side_raw_image_support")
    return {
        "provisional_band_id": provisional,
        "boundary_left_row_id": boundary_key[1],
        "boundary_right_row_id": boundary_key[2],
        "cut": cut,
        "left": left,
        "right": right,
        "gap_um": gap_um,
        "gap_ratio": gap_ratio,
        "gap_factor": gap_factor,
        "image_support": image_support,
        "image_support_identity_exact": image_support_identity_exact,
        "image_support_evidence_source": (
            inherited.get("image_support_evidence_source")
            if image_support_identity_exact and inherited is not None
            else None
        ),
        "raw_image_boundary_gradient": (
            inherited.get("raw_image_boundary_gradient")
            if image_support_identity_exact and inherited is not None
            else None
        ),
        "raw_image_control_gradient_median": (
            inherited.get("raw_image_control_gradient_median")
            if image_support_identity_exact and inherited is not None
            else None
        ),
        "image_support_control_boundary_count": (
            inherited.get("image_support_control_boundary_count")
            if image_support_identity_exact and inherited is not None
            else 0
        ),
        "raw_image_control_source": (
            inherited.get("raw_image_control_source")
            if image_support_identity_exact and inherited is not None
            else None
        ),
        "raw_image_support_sample_count": (
            inherited.get("raw_image_support_sample_count")
            if image_support_identity_exact and inherited is not None
            else 0
        ),
        "raw_image_support_available": bool(
            image_support_identity_exact
            and inherited is not None
            and inherited.get("raw_image_support_available")
        ),
        "raw_image_support_unavailable_reason": (
            inherited.get("raw_image_support_unavailable_reason")
            if inherited is not None
            else "exact_boundary_identity_not_available"
        ),
        "stability": stability,
        "left_stomata": left_stomata,
        "right_stomata": right_stomata,
        "isolated_one_row_side": isolated_one_row_side,
        "row_topology_confidence_tiers": sorted(topology_tiers),
        "row_topology_supported": topology_supported,
        "nonlinear_row_topology": nonlinear_topology,
        "decision_route": decision_route,
        "accepted": not failures,
        "failed_gates": failures,
    }


def _is_v25_selected_singleton_row(
    row: dict[str, Any], stoma_traits: list[dict[str, Any]]
) -> bool:
    members = [int(value) for value in row.get("_point_indices", [])]
    if len(members) != 1 or int(row.get("stoma_count") or 0) != 1:
        return False
    point_index = members[0]
    if not 0 <= point_index < len(stoma_traits):
        return False
    stoma_id = str(stoma_traits[point_index].get("stoma_id") or "")
    selected = {
        str(value)
        for value in row.get("trajectory_v25_selected_singleton_stoma_ids", [])
    }
    return bool(stoma_id and stoma_id in selected)


def _direct_v25_boundary_evidence(
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    settings: TraitSettingsV25,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Measure raw-image support for each exact original adjacent-row pair.

    V2.4 may contain legacy audit records whose support was matched to the
    nearest gap when the exact cut was not evaluated.  Those records are not
    safe for isolated-row decisions.  V2.5 therefore measures every original
    adjacent-row boundary directly and keys the result by the two stable row
    identities.  The score follows the mean local trajectory of the two rows
    over their overlap and compares it with parallel local flanks.  This is
    the established v1.5 raw-image definition, works when the provisional
    group has only two rows, and stays on the scale used by v2.4 thresholds.
    """

    row_lookup = {str(row.get("row_id") or ""): row for row in row_records}
    evidence: dict[tuple[str, str, str], dict[str, Any]] = {}
    for band in band_records:
        provisional = str(
            band.get("provisional_band_id") or band.get("band_id") or ""
        )
        linked = sorted(
            [
                row_lookup[str(row_id)]
                for row_id in band.get("row_ids", [])
                if str(row_id) in row_lookup
            ],
            key=lambda row: base.row_lateral_at(row, 0.0),
        )
        if len(linked) < 2:
            continue
        exclusion_boxes = v15._spatial_stoma_exclusion_boxes_pixels(
            stoma_traits,
            [int(value) for value in band.get("_point_indices", [])],
            settings.raw_evidence_stoma_exclusion_padding_fraction,
        )
        for cut, (left_row, right_row) in enumerate(
            zip(linked[:-1], linked[1:], strict=True), start=1
        ):
            left_id = str(left_row.get("row_id") or "")
            right_id = str(right_row.get("row_id") or "")
            boundary_lateral_um = float(
                (
                    base.row_lateral_at(left_row, 0.0)
                    + base.row_lateral_at(right_row, 0.0)
                )
                / 2.0
            )
            boundary_slope = 0.5 * (
                float(left_row["row_common_slope_lateral_per_longitudinal"])
                + float(right_row["row_common_slope_lateral_per_longitudinal"])
            )
            overlap_min = max(
                float(left_row["longitudinal_min_um"]),
                float(right_row["longitudinal_min_um"]),
            )
            overlap_max = min(
                float(left_row["longitudinal_max_um"]),
                float(right_row["longitudinal_max_um"]),
            )
            image_evidence = v15._trajectory_gradient_evidence(
                boundary_lateral_um,
                boundary_slope,
                0.0,
                median_detector_width_um,
                longitudinal_min_um=overlap_min,
                longitudinal_max_um=overlap_max,
                exclusion_boxes_pixels=exclusion_boxes,
            )
            key = (
                provisional,
                left_id,
                right_id,
            )
            evidence[key] = {
                "provisional_band_id": provisional,
                "boundary_left_row_id": key[1],
                "boundary_right_row_id": key[2],
                "boundary_original_cut": cut,
                "boundary_lateral_um": boundary_lateral_um,
                "boundary_slope_lateral_per_longitudinal": boundary_slope,
                "boundary_longitudinal_overlap_min_um": overlap_min,
                "boundary_longitudinal_overlap_max_um": overlap_max,
                "raw_image_boundary_gradient": image_evidence.get(
                    "target_gradient_median"
                ),
                "raw_image_control_gradient_median": image_evidence.get(
                    "flank_gradient_median"
                ),
                "image_support_ratio": image_evidence.get(
                    "target_to_flank_ratio"
                ),
                "image_support_control_boundary_count": 0,
                "raw_image_control_source": "parallel_local_flanks",
                "raw_image_support_available": bool(
                    image_evidence.get("available")
                ),
                "raw_image_support_unavailable_reason": image_evidence.get(
                    "unavailable_reason", ""
                ),
                "raw_image_support_sample_count": image_evidence.get(
                    "sample_count", 0
                ),
                "image_support_evidence_source": (
                    "v25_direct_exact_pair_local_trajectory_parallel_flanks"
                ),
                "image_support_identity_exact": True,
            }
    return evidence


def _archive_and_neutralize_parent_band_state(
    rows: list[dict[str, Any]],
    assignments: dict[int, dict[str, Any]],
    band_templates: list[dict[str, Any]],
) -> None:
    """Retain parent decisions as audit-only data, never active v2.5 state."""

    exact_active_fields = {
        "v20_singleton_band",
        "v20_singleton_partition_applied",
    }
    active_prefixes = (
        "trajectory_v24_partition_",
        "trajectory_v24_singleton_",
        "band_partition_",
    )
    for record in [*rows, *assignments.values(), *band_templates]:
        archived: dict[str, Any] = {}
        for key in list(record):
            if key not in exact_active_fields and not key.startswith(active_prefixes):
                continue
            value = record.get(key)
            archived[key] = value
            if isinstance(value, bool):
                record[key] = False
            elif isinstance(value, list):
                record[key] = []
            elif isinstance(value, dict):
                record[key] = {}
            elif isinstance(value, str):
                record[key] = ""
            else:
                record[key] = None
        record["trajectory_v25_parent_v24_active_band_state_audit_only"] = archived
        record["trajectory_v25_parent_v24_final_boundary_superseded"] = True


def _partition_trajectory_bands_v25(
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    settings: TraitSettingsV25,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Archive v2.4 cuts, reassess all boundaries, and enforce band capacity."""

    active_diagnostics = list(v14._ACTIVE_BAND_DIAGNOSTICS)
    if len(active_diagnostics) != len(band_records):
        raise RuntimeError(
            "v2.5 partitioning requires one active diagnostic per provisional "
            f"band; diagnostics={len(active_diagnostics)}, "
            f"bands={len(band_records)}."
        )
    diagnostic_fields_by_provisional: dict[str, dict[str, Any]] = {}
    for provisional_band, diagnostic in zip(
        band_records, active_diagnostics, strict=True
    ):
        provisional = str(provisional_band.get("band_id") or "")
        fields = _trajectory_diagnostic_fields_v25(diagnostic)
        diagnostic_fields_by_provisional[provisional] = fields
        provisional_band.update(fields)
        provisional_band["provisional_band_id"] = provisional
    for row in row_records:
        provisional = str(
            row.get("provisional_band_id") or row.get("band_id") or ""
        )
        row.update(diagnostic_fields_by_provisional.get(provisional, {}))
        row["provisional_band_id"] = provisional
    for assignment in point_assignment.values():
        provisional = str(
            assignment.get("provisional_band_id")
            or assignment.get("band_id")
            or ""
        )
        assignment.update(diagnostic_fields_by_provisional.get(provisional, {}))
        assignment["provisional_band_id"] = provisional

    original_band_templates = [dict(band) for band in band_records]

    # Measure every exact adjacent-row boundary before v2.4 can split or
    # reattach the provisional bands.  V2.5 never inherits support from a
    # nearest-but-different legacy cut.
    boundary_evidence_by_rows = _direct_v25_boundary_evidence(
        row_records,
        band_records,
        stoma_traits,
        median_detector_width_um,
        settings,
    )

    rows, bands, assignments, partition_audit = v24._partition_trajectory_bands_v24(
        row_records,
        band_records,
        point_assignment,
        stoma_traits,
        median_detector_width_um,
        settings,
    )
    _archive_and_neutralize_parent_band_state(
        rows,
        assignments,
        original_band_templates,
    )
    row_lookup = {str(row["row_id"]): row for row in rows}

    # V2.4 remains the literal baseline and its audit is preserved, but final
    # v2.5 bands are rebuilt from each complete provisional group.  This makes
    # every possible boundary compete under one v2.5 rule and prevents a
    # legacy inherited cut (including an isolated-row cut) from being silently
    # grandfathered or retained because of left-to-right traversal order.
    parent_v24_band_count = len(bands)
    for record in partition_audit:
        record["trajectory_v25_parent_v24_audit_only"] = True
        record["trajectory_v25_parent_v24_final_boundary_superseded"] = True
    bands = []
    for template in original_band_templates:
        provisional = str(
            template.get("provisional_band_id") or template.get("band_id") or ""
        )
        linked = sorted(
            [
                row_lookup[str(row_id)]
                for row_id in template.get("row_ids", [])
            ],
            key=lambda row: base.row_lateral_at(row, 0.0),
        )
        rebuilt_root = v24._rebuild_band_v24(
            template,
            linked,
            stoma_traits,
            median_detector_width_um,
            None,
            1,
            {
                "trajectory_v25_parent_v24_partition_reassessed_globally": True,
                "trajectory_v25_parent_v24_final_boundary_superseded": True,
                "trajectory_v25_parent_v24_band_count_before_reassessment": (
                    parent_v24_band_count
                ),
                "trajectory_v25_partition_split_applied": False,
                "trajectory_v25_additional_partition_split_applied": False,
                "trajectory_v25_partition_split_depth": 0,
                "trajectory_v25_partition_split_count_in_lineage": 0,
                "trajectory_v25_capacity_forced_unvalidated_boundary": False,
                "trajectory_v25_capacity_forced_unvalidated_boundary_in_lineage": (
                    False
                ),
                "v20_singleton_band": False,
                "v20_singleton_partition_applied": False,
                "trajectory_v24_singleton_applied": False,
            },
        )
        bands.append(rebuilt_root)
        partition_audit.append(
            {
                "provisional_band_id": provisional,
                "accepted": False,
                "trajectory_v25_parent_v24_partition_reassessed_globally": True,
                "trajectory_v25_parent_v24_final_boundary_superseded": True,
                "trajectory_v25_reassessment_root_row_ids": [
                    str(row.get("row_id") or "") for row in linked
                ],
            }
        )

    # The former sequential inherited-boundary reconciliation block remains
    # below only to preserve readable history while v2.5 is audited.  It is
    # deliberately disabled: all parent boundaries were superseded above and
    # are now reconsidered together from one root per provisional group.  This
    # guarantees that final decisions cannot depend on left-to-right traversal.
    validated_singleton_boundaries: set[tuple[str, str, str]] = set()
    rejected_singleton_boundaries: set[tuple[str, str, str]] = set()
    reconciled = False
    while reconciled:
        reconciled = False
        bands.sort(key=lambda band: float(band["band_center_lateral_um"]))
        for position, singleton_container in enumerate(bands):
            container_rows = [
                row_lookup[str(row_id)]
                for row_id in singleton_container.get("row_ids", [])
            ]
            if not any(
                _is_v25_selected_singleton_row(row, stoma_traits)
                for row in container_rows
            ):
                continue
            provisional = str(
                singleton_container.get("provisional_band_id")
                or singleton_container.get("band_id")
                or ""
            )
            for neighbor_position in (position - 1, position + 1):
                if not 0 <= neighbor_position < len(bands):
                    continue
                neighbor = bands[neighbor_position]
                neighbor_provisional = str(
                    neighbor.get("provisional_band_id")
                    or neighbor.get("band_id")
                    or ""
                )
                if neighbor_provisional != provisional:
                    continue
                neighbor_rows = [
                    row_lookup[str(row_id)]
                    for row_id in neighbor.get("row_ids", [])
                ]
                if not neighbor_rows:
                    continue
                combined = sorted(
                    container_rows + neighbor_rows,
                    key=lambda row: base.row_lateral_at(row, 0.0),
                )
                container_ids = {
                    str(row.get("row_id") or "") for row in container_rows
                }
                cuts = [
                    cut
                    for cut in range(1, len(combined))
                    if (
                        str(combined[cut - 1].get("row_id") or "")
                        in container_ids
                    )
                    != (
                        str(combined[cut].get("row_id") or "")
                        in container_ids
                    )
                ]
                if len(cuts) != 1:
                    raise RuntimeError(
                        "v2.5 singleton-boundary reconciliation could not "
                        "identify one contiguous physical boundary."
                    )
                cut = cuts[0]
                if not (
                    _is_v25_selected_singleton_row(
                        combined[cut - 1], stoma_traits
                    )
                    or _is_v25_selected_singleton_row(
                        combined[cut], stoma_traits
                    )
                ):
                    continue
                boundary_key = (
                    provisional,
                    str(combined[cut - 1].get("row_id") or ""),
                    str(combined[cut].get("row_id") or ""),
                )
                if boundary_key in validated_singleton_boundaries:
                    continue
                gaps = [
                    float(v15._matched_trajectory_gap(first, second)[0])
                    for first, second in zip(
                        combined[:-1], combined[1:], strict=True
                    )
                ]
                assessment = _v25_partition_candidate(
                    singleton_container,
                    combined,
                    cut,
                    gaps,
                    boundary_evidence_by_rows,
                    median_detector_width_um,
                    settings,
                )
                compact_assessment = {
                    key: value
                    for key, value in assessment.items()
                    if key not in {"left", "right"}
                }
                if bool(assessment["accepted"]):
                    validated_singleton_boundaries.add(boundary_key)
                    validation_fields = {
                        "trajectory_v25_inherited_singleton_boundary_validated": True,
                        "trajectory_v25_inherited_singleton_boundary_left_row_id": boundary_key[
                            1
                        ],
                        "trajectory_v25_inherited_singleton_boundary_right_row_id": boundary_key[
                            2
                        ],
                        "trajectory_v25_partition_image_support_ratio": assessment[
                            "image_support"
                        ],
                        "trajectory_v25_partition_image_support_identity_exact": assessment[
                            "image_support_identity_exact"
                        ],
                        "trajectory_v25_partition_image_support_evidence_source": assessment[
                            "image_support_evidence_source"
                        ],
                    }
                    boundary_entry = {
                        "provisional_band_id": boundary_key[0],
                        "left_row_id": boundary_key[1],
                        "right_row_id": boundary_key[2],
                    }
                    for boundary_band in (singleton_container, neighbor):
                        existing_boundaries = list(
                            boundary_band.get(
                                "trajectory_v25_validated_inherited_singleton_boundaries",
                                [],
                            )
                        )
                        if boundary_entry not in existing_boundaries:
                            existing_boundaries.append(boundary_entry)
                        boundary_band.update(validation_fields)
                        boundary_band[
                            "trajectory_v25_validated_inherited_singleton_boundaries"
                        ] = existing_boundaries
                        boundary_band[
                            "trajectory_v25_validated_inherited_singleton_boundary_count"
                        ] = len(existing_boundaries)
                    partition_audit.append(
                        {
                            "provisional_band_id": provisional,
                            "accepted": True,
                            **validation_fields,
                            "trajectory_v25_partition_candidate_diagnostics": [
                                compact_assessment
                            ],
                        }
                    )
                    continue

                if len(combined) > int(settings.maximum_rows_per_band):
                    validated_singleton_boundaries.add(boundary_key)
                    unavailable_fields = {
                        "trajectory_v25_inherited_singleton_boundary_revalidation_required": True,
                        "trajectory_v25_inherited_singleton_boundary_revalidation_unavailable": True,
                        "trajectory_v25_inherited_singleton_boundary_retained_due_capacity": True,
                        "trajectory_v25_inherited_singleton_boundary_left_row_id": boundary_key[
                            1
                        ],
                        "trajectory_v25_inherited_singleton_boundary_right_row_id": boundary_key[
                            2
                        ],
                    }
                    singleton_container.update(unavailable_fields)
                    neighbor.update(unavailable_fields)
                    partition_audit.append(
                        {
                            "provisional_band_id": provisional,
                            "accepted": False,
                            "trajectory_v25_inherited_boundary_retained": True,
                            **unavailable_fields,
                            "trajectory_v25_inherited_singleton_boundary_revalidation_reason": (
                                "failed_v25_boundary_could_not_be_reattached_without_"
                                "exceeding_maximum_rows_per_band"
                            ),
                            "trajectory_v25_partition_candidate_diagnostics": [
                                compact_assessment
                            ],
                        }
                    )
                    continue

                merged = v24._rebuild_band_v24(
                    singleton_container,
                    combined,
                    stoma_traits,
                    median_detector_width_um,
                    None,
                    1,
                    {
                        "trajectory_v25_inherited_singleton_boundary_reattached": True,
                        "trajectory_v25_inherited_singleton_boundary_revalidation_required": True,
                        "trajectory_v25_inherited_singleton_boundary_failed_gates": assessment[
                            "failed_gates"
                        ],
                    },
                )
                low = min(position, neighbor_position)
                high = max(position, neighbor_position)
                bands[low : high + 1] = [merged]
                rejected_singleton_boundaries.add(boundary_key)
                partition_audit.append(
                    {
                        "provisional_band_id": provisional,
                        "accepted": False,
                        "trajectory_v25_inherited_singleton_boundary_reattached": True,
                        "trajectory_v25_inherited_singleton_boundary_revalidation_required": True,
                        "trajectory_v25_inherited_singleton_boundary_revalidation_reason": (
                            "inherited_singleton_boundary_failed_v25_standardized_rule"
                        ),
                        "trajectory_v25_partition_candidate_diagnostics": [
                            compact_assessment
                        ],
                    }
                )
                reconciled = True
                break
            if reconciled:
                break

    pending = list(bands)
    finalized: list[dict[str, Any]] = []
    while pending:
        band = pending.pop(0)
        linked = sorted(
            [row_lookup[str(row_id)] for row_id in band.get("row_ids", [])],
            key=lambda row: base.row_lateral_at(row, 0.0),
        )
        if len(linked) < 2:
            band.setdefault("trajectory_v25_partition_split_applied", False)
            finalized.append(band)
            continue
        gaps = [
            float(v15._matched_trajectory_gap(first, second)[0])
            for first, second in zip(linked[:-1], linked[1:], strict=True)
        ]
        assessed = [
            _v25_partition_candidate(
                band,
                linked,
                cut,
                gaps,
                boundary_evidence_by_rows,
                median_detector_width_um,
                settings,
            )
            for cut in range(1, len(linked))
        ]
        for record in assessed:
            boundary_key = (
                str(record.get("provisional_band_id") or ""),
                str(record.get("boundary_left_row_id") or ""),
                str(record.get("boundary_right_row_id") or ""),
            )
            if boundary_key in rejected_singleton_boundaries:
                record["accepted_before_singleton_reconciliation_lock"] = bool(
                    record.get("accepted")
                )
                record["accepted"] = False
                record["failed_gates"] = list(record.get("failed_gates", [])) + [
                    "inherited_singleton_boundary_failed_standardized_revalidation"
                ]
        accepted = [record for record in assessed if bool(record["accepted"])]
        capacity_forced = False
        selection_pool = list(accepted)
        if not selection_pool and len(linked) > int(settings.maximum_rows_per_band):
            capacity_forced = True
            capacity_candidates = [
                record
                for record in assessed
                if len(record["left"]) <= int(settings.maximum_rows_per_band)
                and len(record["right"]) <= int(settings.maximum_rows_per_band)
            ]
            # The configured trajectory search currently permits at most twice
            # the final-band cap (10 versus 5).  Keep a deterministic recursive
            # fallback if those settings are changed later.
            selection_pool = capacity_candidates or list(assessed)
        if not selection_pool:
            prior_split_applied = bool(
                band.get("trajectory_v25_partition_split_applied")
            )
            prior_additional_split_applied = bool(
                band.get("trajectory_v25_additional_partition_split_applied")
            )
            compact_assessed = [
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"left", "right"}
                }
                for record in assessed
            ]
            band.update(
                {
                    "trajectory_v25_partition_split_applied": bool(
                        prior_split_applied
                    ),
                    "trajectory_v25_additional_partition_split_applied": (
                        prior_additional_split_applied
                    ),
                    "trajectory_v25_partition_candidate_diagnostics": compact_assessed,
                    "trajectory_v25_partition_multiple_boundaries_permitted": True,
                }
            )
            partition_audit.append(
                {
                    "provisional_band_id": str(
                        band.get("provisional_band_id")
                        or band.get("band_id")
                        or ""
                    ),
                    "accepted": False,
                    "trajectory_v25_partition_split_applied": prior_split_applied,
                    "trajectory_v25_additional_partition_split_applied": (
                        prior_additional_split_applied
                    ),
                    "trajectory_v25_partition_no_additional_candidate_passed": True,
                    "trajectory_v25_partition_candidate_diagnostics": compact_assessed,
                }
            )
            finalized.append(band)
            continue
        if capacity_forced:
            chosen = max(
                selection_pool,
                key=lambda record: (
                    bool(record.get("image_support_identity_exact")),
                    (
                        float(record["image_support"])
                        if _finite(record.get("image_support")) is not None
                        else -math.inf
                    ),
                    float(record["gap_factor"]),
                    (
                        float(record["gap_ratio"])
                        if _finite(record.get("gap_ratio")) is not None
                        else -math.inf
                    ),
                    float(record["stability"]),
                    -abs(len(record["left"]) - len(record["right"])),
                    -int(record["cut"]),
                ),
            )
        else:
            chosen = max(
                selection_pool,
                key=lambda record: (
                    float(record["gap_factor"]),
                    (
                        float(record["image_support"])
                        if _finite(record.get("image_support")) is not None
                        else -math.inf
                    ),
                    float(record["stability"]),
                    -int(record["cut"]),
                ),
            )
        parent_split_count = int(
            band.get("trajectory_v25_partition_split_count_in_lineage") or 0
        )
        child_split_count = parent_split_count + 1
        additional_split_applied = parent_split_count >= 1
        capacity_forced_in_lineage = bool(
            band.get(
                "trajectory_v25_capacity_forced_unvalidated_boundary_in_lineage"
            )
            or band.get("trajectory_v25_capacity_forced_unvalidated_boundary")
            or capacity_forced
        )
        split_reason = (
            "capacity_forced_unvalidated_boundary"
            if capacity_forced
            else chosen["decision_route"]
        )
        partition = v24._partition_template(
            str(chosen["provisional_band_id"]),
            int(chosen["cut"]),
            float(chosen["gap_um"]),
            float(chosen["gap_ratio"]),
            float(chosen["gap_factor"]),
            _finite(chosen.get("image_support")),
            chosen["left"],
            chosen["right"],
            stoma_traits,
        )
        rebuilt: list[dict[str, Any]] = []
        for group_index, group in enumerate([chosen["left"], chosen["right"]], start=1):
            rebuilt.append(
                v24._rebuild_band_v24(
                    band,
                    group,
                    stoma_traits,
                    median_detector_width_um,
                    partition,
                    group_index,
                    {
                        "trajectory_v25_partition_split_applied": True,
                        "trajectory_v25_additional_partition_split_applied": (
                            additional_split_applied
                        ),
                        "trajectory_v25_partition_split_depth": child_split_count,
                        "trajectory_v25_partition_split_count_in_lineage": (
                            child_split_count
                        ),
                        "trajectory_v25_partition_split_reason": split_reason,
                        "trajectory_v25_capacity_forced_unvalidated_boundary": (
                            capacity_forced
                        ),
                        "trajectory_v25_capacity_forced_unvalidated_boundary_in_lineage": (
                            capacity_forced_in_lineage
                        ),
                        "trajectory_v25_capacity_forced_boundary_failed_gates": (
                            chosen["failed_gates"] if capacity_forced else []
                        ),
                        "trajectory_v25_partition_gap_um": chosen["gap_um"],
                        "trajectory_v25_partition_gap_width_factor": chosen[
                            "gap_factor"
                        ],
                        "trajectory_v25_partition_gap_ratio": chosen["gap_ratio"],
                        "trajectory_v25_partition_image_support_ratio": chosen[
                            "image_support"
                        ],
                        "trajectory_v25_partition_image_support_identity_exact": chosen[
                            "image_support_identity_exact"
                        ],
                        "trajectory_v25_partition_image_support_evidence_source": chosen[
                            "image_support_evidence_source"
                        ],
                        "trajectory_v25_partition_raw_image_boundary_gradient": chosen[
                            "raw_image_boundary_gradient"
                        ],
                        "trajectory_v25_partition_raw_image_control_gradient_median": chosen[
                            "raw_image_control_gradient_median"
                        ],
                        "trajectory_v25_partition_raw_image_control_source": chosen[
                            "raw_image_control_source"
                        ],
                        "trajectory_v25_partition_raw_image_support_sample_count": chosen[
                            "raw_image_support_sample_count"
                        ],
                        "trajectory_v25_partition_raw_image_support_available": chosen[
                            "raw_image_support_available"
                        ],
                        "trajectory_v25_partition_raw_image_support_unavailable_reason": chosen[
                            "raw_image_support_unavailable_reason"
                        ],
                        "trajectory_v25_partition_image_support_control_boundary_count": chosen[
                            "image_support_control_boundary_count"
                        ],
                        "trajectory_v25_partition_stability_fraction": chosen[
                            "stability"
                        ],
                        "trajectory_v25_partition_multiple_boundaries_permitted": True,
                        "trajectory_v25_partition_parent_v24_reversal_overcome": bool(
                            band.get("trajectory_v24_partition_reversal_applied")
                        ),
                    },
                )
            )
        partition_audit.append(
            {
                **partition,
                "accepted": not capacity_forced,
                "trajectory_v25_partition_split_applied": True,
                "trajectory_v25_additional_partition_split_applied": (
                    additional_split_applied
                ),
                "trajectory_v25_partition_split_depth": child_split_count,
                "trajectory_v25_partition_split_count_in_lineage": child_split_count,
                "trajectory_v25_partition_split_reason": split_reason,
                "trajectory_v25_capacity_forced_unvalidated_boundary": (
                    capacity_forced
                ),
                "trajectory_v25_capacity_forced_unvalidated_boundary_in_lineage": (
                    capacity_forced_in_lineage
                ),
                "trajectory_v25_capacity_forced_boundary_failed_gates": (
                    chosen["failed_gates"] if capacity_forced else []
                ),
                "trajectory_v25_partition_stability_fraction": chosen["stability"],
                "trajectory_v25_partition_multiple_boundaries_permitted": True,
                "trajectory_v25_partition_candidate_diagnostics": [
                    {
                        key: value
                        for key, value in record.items()
                        if key not in {"left", "right"}
                    }
                    for record in assessed
                ],
            }
        )
        # Re-enter both pieces so a second supported boundary can be added.
        pending = rebuilt + pending
    rows, finalized, assignments = v24._renumber_final_bands(
        rows, finalized, assignments
    )
    final_row_lookup = {str(row["row_id"]): row for row in rows}
    for band in finalized:
        provenance = {
            key: value
            for key, value in band.items()
            if (
                key.startswith("trajectory_v25_partition_")
                or key.startswith("trajectory_v25_capacity_")
                or key.startswith("trajectory_v25_parent_v24_")
                or key.startswith("trajectory_v25_inherited_singleton_boundary_")
                or key.startswith("trajectory_v25_validated_inherited_singleton_")
            )
            and key
            != "trajectory_v25_parent_v24_active_band_state_audit_only"
        }
        for row_id in band.get("row_ids", []):
            row = final_row_lookup[str(row_id)]
            row.update(provenance)
            for point_index in row.get("_point_indices", []):
                assignments[int(point_index)].update(provenance)
    final_band_by_row = {
        str(row_id): str(band.get("band_id") or "")
        for band in finalized
        for row_id in band.get("row_ids", [])
    }
    for _, left_row_id, right_row_id in rejected_singleton_boundaries:
        if (
            left_row_id in final_band_by_row
            and right_row_id in final_band_by_row
            and final_band_by_row[left_row_id] != final_band_by_row[right_row_id]
        ):
            raise RuntimeError(
                "v2.5 partition invariant failed: a singleton boundary rejected "
                "by standardized revalidation survived between final bands."
            )
    return rows, finalized, assignments, partition_audit


def _row_x_mapping(
    row: dict[str, Any], axis: base.AxisTransform
) -> tuple[float, float]:
    slope = float(row.get("row_common_slope_lateral_per_longitudinal") or 0.0)
    reference = float(row.get("row_longitudinal_reference_um") or 0.0)
    intercept = float(row["row_intercept_lateral_um"]) - slope * reference
    sine = math.sin(axis.radians)
    cosine = math.cos(axis.radians)
    coefficient = cosine * slope + sine
    x_intercept = cosine * intercept + axis.origin_x_um
    return coefficient, x_intercept


def _missing_length_through_relevant_lateral_side(
    reference_interval: tuple[float, float],
    visible_interval: tuple[float, float] | None,
    x_coefficient: float,
    x_intercept: float,
    width_um: float,
    relevant_sides: set[str],
) -> float:
    low, high = reference_interval
    segments: list[tuple[float, float]] = []
    if visible_interval is None:
        segments.append((low, high))
    else:
        visible_low = max(low, float(visible_interval[0]))
        visible_high = min(high, float(visible_interval[1]))
        if visible_low > low:
            segments.append((low, min(visible_low, high)))
        if visible_high < high:
            segments.append((max(visible_high, low), high))
    missing = 0.0
    for start, stop in segments:
        if stop <= start:
            continue
        midpoint = 0.5 * (start + stop)
        x = x_coefficient * midpoint + x_intercept
        exits_left = x < -1e-9 and "left" in relevant_sides
        exits_right = x > width_um + 1e-9 and "right" in relevant_sides
        if exits_left or exits_right:
            missing += stop - start
    return missing


def _v25_multirow_edge_evidence(
    band: dict[str, Any],
    rows: list[dict[str, Any]],
    axis: base.AxisTransform | None,
    width_um: float | None,
    height_um: float | None,
    is_outermost: bool,
    settings: TraitSettingsV25,
) -> dict[str, Any]:
    if not is_outermost or len(rows) < 2:
        return {
            "assessable": False,
            "reason": "not_an_outermost_multirow_band",
            "triggered": False,
        }
    if axis is None or width_um is None or height_um is None:
        return {
            "assessable": False,
            "reason": "source_geometry_or_calibration_unavailable",
            "triggered": False,
        }
    if base.parse_bool(band.get("trajectory_row_count_stable")) is not True:
        return {
            "assessable": False,
            "reason": "row_count_not_explicitly_stable",
            "triggered": False,
        }
    if base.parse_bool(band.get("trajectory_band_structure_stable")) is not True:
        return {
            "assessable": False,
            "reason": "band_structure_not_explicitly_stable",
            "triggered": False,
        }
    if base.parse_bool(band.get("trajectory_global_crossing_detected")) is True:
        return {
            "assessable": False,
            "reason": "trajectory_crossing_detected",
            "triggered": False,
        }
    topology_tiers = {
        str(row.get("trajectory_v25_topology_confidence_tier") or "not_assessed")
        for row in rows
    }
    if not topology_tiers.issubset({"supported", "audited"}):
        return {
            "assessable": False,
            "reason": "v25_row_membership_or_assignment_not_explicitly_supported",
            "triggered": False,
            "row_topology_confidence_tiers": sorted(topology_tiers),
        }
    intervals: list[dict[str, Any]] = []
    for row in rows:
        visible = v24._visible_row_longitudinal_interval(
            row, axis, width_um, height_um
        )
        if visible is None or not np.isfinite(visible).all():
            return {
                "assessable": False,
                "reason": f"row_visible_interval_unavailable:{row.get('row_id')}",
                "triggered": False,
            }
        span = max(0.0, float(visible[1]) - float(visible[0]))
        intervals.append(
            {
                "row": row,
                "row_id": str(row.get("row_id") or ""),
                "visible_interval": (float(visible[0]), float(visible[1])),
                "visible_span_um": span,
                "member_count": int(row.get("stoma_count") or 0),
            }
        )
    reference = max(
        intervals,
        key=lambda item: (
            float(item["visible_span_um"]),
            int(item["member_count"]),
            str(item["row_id"]),
        ),
    )
    reference_interval = reference["visible_interval"]
    reference_span = float(reference["visible_span_um"])
    if reference_span <= 1e-9:
        return {
            "assessable": False,
            "reason": "reference_trajectory_has_no_visible_span",
            "triggered": False,
        }
    relevant_sides: set[str] = set()
    if bool(band.get("band_lateral_is_leftmost_band")):
        relevant_sides.add("left")
    if bool(band.get("band_lateral_is_rightmost_band")):
        relevant_sides.add("right")
    if not relevant_sides:
        return {
            "assessable": False,
            "reason": "outer_side_identity_unavailable",
            "triggered": False,
        }
    row_evidence: list[dict[str, Any]] = []
    for item in intervals:
        row = item["row"]
        x_coefficient, x_intercept = _row_x_mapping(row, axis)
        x_visible = v24._linear_interval(
            x_coefficient, x_intercept, 0.0, width_um
        )
        missing = _missing_length_through_relevant_lateral_side(
            reference_interval,
            x_visible,
            x_coefficient,
            x_intercept,
            width_um,
            relevant_sides,
        )
        loss = min(max(missing / reference_span, 0.0), 1.0)
        row_evidence.append(
            {
                "row_id": item["row_id"],
                "member_count": item["member_count"],
                "full_visible_interval_um": list(item["visible_interval"]),
                "x_visible_interval_um": list(x_visible) if x_visible is not None else None,
                "reference_interval_um": list(reference_interval),
                "relevant_outer_sides": sorted(relevant_sides),
                "lateral_missing_length_um": missing,
                "lateral_visibility_fraction": 1.0 - loss,
                "lateral_loss_fraction": loss,
            }
        )
    maximum_loss = max(float(item["lateral_loss_fraction"]) for item in row_evidence)
    threshold = float(settings.v25_multirow_lateral_loss_threshold_fraction)
    triggered = maximum_loss >= threshold - 1e-9
    return {
        "assessable": True,
        "reason": (
            "outer_multirow_lateral_loss_reaches_threshold"
            if triggered
            else "outer_multirow_lateral_loss_below_threshold"
        ),
        "triggered": triggered,
        "reference_row_id": reference["row_id"],
        "reference_interval_um": list(reference_interval),
        "reference_span_um": reference_span,
        "relevant_outer_sides": sorted(relevant_sides),
        "row_evidence": row_evidence,
        "minimum_visibility_fraction": 1.0 - maximum_loss,
        "maximum_loss_fraction": maximum_loss,
        "threshold_fraction": threshold,
        "loss_interpretation": (
            "differential_lateral_clipping_relative_to_the_longest_visible_"
            "fitted_row_in_the_same_band_not_absolute_shared_clipping"
        ),
    }


def _apply_v25_standardized_multirow_edge_rule(
    rows: list[dict[str, Any]],
    bands: list[dict[str, Any]],
    assignments: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettingsV25,
    parent_edge_audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    row_lookup = {str(row["row_id"]): row for row in rows}
    ordered = sorted(
        bands, key=lambda band: int(band.get("band_order_left_to_right") or 0)
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
    audit_by_band = {str(record.get("band_id")): record for record in parent_edge_audit}
    revised: list[dict[str, Any]] = []
    for band in bands:
        band_id = str(band["band_id"])
        linked = [row_lookup[str(row_id)] for row_id in band.get("row_ids", [])]
        single_row_snapshot = (
            bool(band.get("band_complete_topology_eligible")),
            str(band.get("band_edge_completeness_state") or ""),
            str(band.get("band_edge_completeness_reason") or ""),
            bool(band.get("band_lateral_half_cut_excluded")),
            bool(band.get("band_lateral_sparse_border_excluded")),
            bool(band.get("band_longitudinal_edge_fragment_excluded")),
            bool(band.get("band_spatial_eligible")),
            bool(band.get("band_fov_truncated")),
            str(band.get("band_spatial_exclusion_reason") or ""),
            bool(band.get("band_edge_topology_excluded")),
            bool(band.get("band_edge_fragment_excluded")),
            bool(band.get("band_edge_adequate_support_veto_applied")),
            str(band.get("band_lateral_edge_assessment_state") or ""),
            str(band.get("band_lateral_half_cut_decision_source") or ""),
            bool(band.get("band_row_structure_eligible")),
            bool(band.get("band_completeness_dependent_traits_eligible")),
        ) if len(linked) == 1 else None
        evidence = _v25_multirow_edge_evidence(
            band,
            linked,
            axis,
            width_um,
            height_um,
            band_id in outermost,
            settings,
        )
        triggered = bool(evidence.get("triggered"))
        previously_excluded = bool(band.get("band_edge_topology_excluded"))
        newly_excluded = bool(triggered and not previously_excluded)
        if newly_excluded:
            reason = (
                "outermost_multirow_band_has_row_with_geometric_lateral_"
                "visibility_loss_ge_20pct"
            )
            state = "multirow_lateral_visibility_loss"
            previous_veto = bool(band.get("band_edge_adequate_support_veto_applied"))
            previous_lateral_state = str(
                band.get("band_lateral_edge_assessment_state") or ""
            )
            previous_lateral_source = str(
                band.get("band_lateral_half_cut_decision_source") or ""
            )
            band.update(
                {
                    "partial_band_eligibility_version": SCRIPT_VERSION,
                    "band_complete_topology_eligible": False,
                    "band_edge_topology_excluded": True,
                    "band_edge_fragment_excluded": True,
                    "band_multirow_lateral_visibility_loss_excluded": True,
                    "band_edge_completeness_state": state,
                    "band_edge_completeness_reason": reason,
                    "band_edge_exclusion_candidate": True,
                    "band_edge_adequate_support_veto_applied": False,
                    "band_v24_edge_adequate_support_veto_applied": previous_veto,
                    "band_v25_overrode_v24_adequate_support_veto": previous_veto,
                    "band_v24_lateral_edge_assessment_state": previous_lateral_state,
                    "band_v24_lateral_half_cut_decision_source": previous_lateral_source,
                    "band_lateral_edge_assessment_state": state,
                    "band_lateral_half_cut_decision_source": (
                        "v25_geometric_multirow_lateral_visibility"
                    ),
                    "band_row_structure_eligible": False,
                    "band_spatial_eligible": False,
                    "band_fov_truncated": True,
                    "band_spatial_exclusion_reason": reason,
                    "band_completeness_dependent_traits_eligible": False,
                }
            )
            for row in linked:
                row_specific = next(
                    (
                        item
                        for item in evidence.get("row_evidence", [])
                        if str(item.get("row_id")) == str(row.get("row_id"))
                    ),
                    {},
                )
                row.update(
                    {
                        "partial_band_eligibility_version": SCRIPT_VERSION,
                        "row_complete_topology_eligible": False,
                        "row_edge_topology_excluded": True,
                        "row_edge_fragment_excluded": True,
                        "row_multirow_lateral_visibility_loss_excluded": True,
                        "row_edge_completeness_state": state,
                        "row_edge_completeness_reason": reason,
                        "row_edge_exclusion_candidate": True,
                        "row_edge_adequate_support_veto_applied": False,
                        "row_structure_eligible": False,
                        "row_band_spatial_eligible": False,
                        "row_band_spatial_exclusion_reason": reason,
                        "row_completeness_dependent_traits_eligible": False,
                        "row_v25_lateral_visibility_fraction": row_specific.get(
                            "lateral_visibility_fraction"
                        ),
                        "row_v25_lateral_loss_fraction": row_specific.get(
                            "lateral_loss_fraction"
                        ),
                    }
                )
                for point_index in row.get("_point_indices", []):
                    assignments[int(point_index)].update(
                        {
                            "partial_band_eligibility_version": SCRIPT_VERSION,
                            "row_complete_topology_eligible": False,
                            "row_edge_topology_excluded": True,
                            "row_edge_fragment_excluded": True,
                            "row_multirow_lateral_visibility_loss_excluded": True,
                            "row_edge_completeness_state": state,
                            "row_edge_completeness_reason": reason,
                            "row_edge_exclusion_candidate": True,
                            "row_edge_adequate_support_veto_applied": False,
                            "row_structure_eligible": False,
                            "row_band_spatial_eligible": False,
                            "row_band_spatial_exclusion_reason": reason,
                            "row_completeness_dependent_traits_eligible": False,
                            "row_v25_lateral_visibility_fraction": row_specific.get(
                                "lateral_visibility_fraction"
                            ),
                            "row_v25_lateral_loss_fraction": row_specific.get(
                                "lateral_loss_fraction"
                            ),
                        }
                    )
        else:
            band.setdefault("band_multirow_lateral_visibility_loss_excluded", False)
            band.setdefault("band_v25_overrode_v24_adequate_support_veto", False)
            for row in linked:
                row_specific = next(
                    (
                        item
                        for item in evidence.get("row_evidence", [])
                        if str(item.get("row_id")) == str(row.get("row_id"))
                    ),
                    {},
                )
                row.setdefault("row_multirow_lateral_visibility_loss_excluded", False)
                row["row_v25_lateral_visibility_fraction"] = row_specific.get(
                    "lateral_visibility_fraction"
                )
                row["row_v25_lateral_loss_fraction"] = row_specific.get(
                    "lateral_loss_fraction"
                )
                for point_index in row.get("_point_indices", []):
                    assignments[int(point_index)].setdefault(
                        "row_multirow_lateral_visibility_loss_excluded", False
                    )
                    assignments[int(point_index)][
                        "row_v25_lateral_visibility_fraction"
                    ] = row_specific.get("lateral_visibility_fraction")
                    assignments[int(point_index)]["row_v25_lateral_loss_fraction"] = (
                        row_specific.get("lateral_loss_fraction")
                    )
        band.update(
            {
                "band_v25_multirow_lateral_visibility_assessable": bool(
                    evidence.get("assessable")
                ),
                "band_v25_multirow_lateral_visibility_assessment_reason": evidence.get(
                    "reason"
                ),
                "band_v25_multirow_lateral_visibility_reference_row_id": evidence.get(
                    "reference_row_id"
                ),
                "band_v25_multirow_lateral_visibility_reference_interval_um": evidence.get(
                    "reference_interval_um"
                ),
                "band_v25_multirow_lateral_visibility_row_evidence": evidence.get(
                    "row_evidence", []
                ),
                "band_v25_multirow_lateral_visibility_minimum_fraction": evidence.get(
                    "minimum_visibility_fraction"
                ),
                "band_v25_multirow_lateral_visibility_maximum_loss_fraction": evidence.get(
                    "maximum_loss_fraction"
                ),
                "band_v25_multirow_lateral_visibility_threshold_fraction": float(
                    settings.v25_multirow_lateral_loss_threshold_fraction
                ),
                "band_v25_multirow_lateral_visibility_triggered": triggered,
                "band_v25_multirow_lateral_visibility_effective_exclusion_added": newly_excluded,
            }
        )
        if single_row_snapshot is not None:
            current = (
                bool(band.get("band_complete_topology_eligible")),
                str(band.get("band_edge_completeness_state") or ""),
                str(band.get("band_edge_completeness_reason") or ""),
                bool(band.get("band_lateral_half_cut_excluded")),
                bool(band.get("band_lateral_sparse_border_excluded")),
                bool(band.get("band_longitudinal_edge_fragment_excluded")),
                bool(band.get("band_spatial_eligible")),
                bool(band.get("band_fov_truncated")),
                str(band.get("band_spatial_exclusion_reason") or ""),
                bool(band.get("band_edge_topology_excluded")),
                bool(band.get("band_edge_fragment_excluded")),
                bool(band.get("band_edge_adequate_support_veto_applied")),
                str(band.get("band_lateral_edge_assessment_state") or ""),
                str(band.get("band_lateral_half_cut_decision_source") or ""),
                bool(band.get("band_row_structure_eligible")),
                bool(band.get("band_completeness_dependent_traits_eligible")),
            )
            if current != single_row_snapshot:
                raise RuntimeError(
                    "v2.5 edge rule changed a locked v2.4 single-row decision; "
                    f"band={band_id}."
                )
        audit_record = dict(audit_by_band.get(band_id, {}))
        audit_record.update(
            {
                "band_id": band_id,
                "partial_band_eligibility_version": SCRIPT_VERSION,
                "v25_multirow_lateral_visibility_evidence": evidence,
                "v25_multirow_lateral_visibility_triggered": triggered,
                "v25_multirow_lateral_visibility_effective_exclusion_added": newly_excluded,
                "spatial_eligible": bool(band.get("band_spatial_eligible")),
                "row_structure_eligible": bool(
                    band.get("band_row_structure_eligible")
                ),
                "complete_topology_eligible": bool(
                    band.get("band_complete_topology_eligible")
                ),
                "fov_truncated": bool(band.get("band_fov_truncated")),
                "edge_topology_excluded": bool(band.get("band_edge_topology_excluded")),
                "edge_fragment_excluded": bool(band.get("band_edge_fragment_excluded")),
                "lateral_half_cut_excluded": bool(
                    band.get("band_lateral_half_cut_excluded")
                ),
                "lateral_sparse_border_excluded": bool(
                    band.get("band_lateral_sparse_border_excluded")
                ),
                "longitudinal_edge_fragment_excluded": bool(
                    band.get("band_longitudinal_edge_fragment_excluded")
                ),
                "multirow_lateral_visibility_loss_excluded": bool(
                    band.get("band_multirow_lateral_visibility_loss_excluded")
                ),
                "edge_completeness_state": str(
                    band.get("band_edge_completeness_state") or ""
                ),
                "edge_completeness_reason": str(
                    band.get("band_edge_completeness_reason") or ""
                ),
                "edge_exclusion_candidate": bool(
                    band.get("band_edge_exclusion_candidate")
                ),
                "complete_support_available": bool(
                    band.get("band_edge_complete_support_available")
                ),
                "complete_row_level_support": bool(
                    band.get("band_edge_complete_row_level_support")
                ),
                "adequate_support_veto_applied": bool(
                    band.get("band_edge_adequate_support_veto_applied")
                ),
                "v24_adequate_support_veto_before_v25": bool(
                    band.get("band_v24_edge_adequate_support_veto_applied")
                ),
                "v25_overrode_v24_adequate_support_veto": bool(
                    band.get("band_v25_overrode_v24_adequate_support_veto")
                ),
                "lateral_edge_assessment_state": str(
                    band.get("band_lateral_edge_assessment_state") or ""
                ),
                "lateral_half_cut_decision_source": str(
                    band.get("band_lateral_half_cut_decision_source") or ""
                ),
                "completeness_dependent_traits_eligible": bool(
                    band.get("band_completeness_dependent_traits_eligible")
                ),
                "exclusion_reason": str(band.get("band_spatial_exclusion_reason") or ""),
            }
        )
        revised.append(audit_record)
    return revised


def _annotate_v25_singleton_rows(
    rows: list[dict[str, Any]],
    assignments: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
) -> None:
    for row in rows:
        members = [int(value) for value in row.get("_point_indices", [])]
        supported_ids = {
            str(value)
            for value in row.get("trajectory_v25_supported_singleton_stoma_ids", [])
        }
        ambiguous_ids = {
            str(value)
            for value in row.get("trajectory_v25_ambiguous_singleton_stoma_ids", [])
        }
        stoma_id = (
            str(stoma_traits[members[0]].get("stoma_id") or "")
            if len(members) == 1
            else ""
        )
        is_singleton = len(members) == 1 and stoma_id in supported_ids | ambiguous_ids
        if is_singleton and stoma_id in supported_ids:
            tier = "supported"
        elif is_singleton:
            tier = "ambiguous"
        else:
            tier = "not_singleton"
        row.update(
            {
                "trajectory_v25_singleton_row": is_singleton,
                "trajectory_v25_singleton_evidence_tier": tier,
                "trajectory_v25_singleton_orientation_inherited": is_singleton,
                "trajectory_v25_singleton_orientation_independently_estimable": False
                if is_singleton
                else None,
                "trajectory_v25_singleton_within_row_spacing_estimable": False
                if is_singleton
                else len(members) >= 2,
            }
        )
        for point_index in members:
            point_stoma_id = str(stoma_traits[point_index].get("stoma_id") or "")
            point_singleton = bool(is_singleton and point_stoma_id == stoma_id)
            assignments[int(point_index)].update(
                {
                    "trajectory_v25_singleton_stoma": point_singleton,
                    "trajectory_v25_singleton_evidence_tier": tier
                    if point_singleton
                    else "not_singleton",
                    "trajectory_v25_singleton_orientation_inherited": point_singleton,
                    "trajectory_v25_singleton_orientation_independently_estimable": False
                    if point_singleton
                    else None,
                }
            )


def _assert_v25_invariants(
    stoma_traits: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    bands: list[dict[str, Any]],
    assignments: dict[int, dict[str, Any]],
    eligibility_before: dict[str, list[bool | None]],
    edge_audit: list[dict[str, Any]],
    settings: TraitSettingsV25,
) -> None:
    source = (
        str(stoma_traits[0].get("source_image_name") or "unknown_image")
        if stoma_traits
        else "unknown_image"
    )
    if set(assignments) != set(range(len(stoma_traits))):
        raise RuntimeError(
            f"v2.5 invariant failed: incomplete point assignment; source={source}."
        )

    row_lookup = {str(row["row_id"]): row for row in rows}
    audit_by_band: dict[str, list[dict[str, Any]]] = {}
    for record in edge_audit:
        audit_by_band.setdefault(str(record.get("band_id") or ""), []).append(
            record
        )
    final_band_ids = {str(band.get("band_id") or "") for band in bands}
    if set(audit_by_band) != final_band_ids:
        raise RuntimeError(
            "v2.5 invariant failed: final edge-audit band IDs do not match "
            f"final bands; source={source}."
        )

    seen: list[int] = []
    for band in bands:
        band_id = str(band.get("band_id") or "")
        matching_audit = audit_by_band.get(band_id, [])
        if len(matching_audit) != 1:
            raise RuntimeError(
                "v2.5 invariant failed: expected exactly one final edge-audit "
                f"record; source={source}, band={band_id}, "
                f"records={len(matching_audit)}."
            )
        audit_record = matching_audit[0]
        linked = [row_lookup[str(row_id)] for row_id in band.get("row_ids", [])]
        if len(linked) > int(settings.maximum_rows_per_band):
            raise RuntimeError(
                "v2.5 invariant failed: final biological band exceeds maximum "
                f"rows; source={source}, band={band_id}."
            )
        members = sorted(
            {
                int(point)
                for row in linked
                for point in row.get("_point_indices", [])
            }
        )
        if members != sorted(
            int(value) for value in band.get("_point_indices", [])
        ):
            raise RuntimeError(
                "v2.5 invariant failed: band/row membership mismatch."
            )
        seen.extend(members)

        excluded = bool(band.get("band_edge_topology_excluded"))
        if excluded != bool(band.get("band_edge_fragment_excluded")):
            raise RuntimeError(
                "v2.5 invariant failed: generic band edge flags disagree."
            )
        if not (
            bool(band.get("band_complete_topology_eligible"))
            == bool(band.get("band_spatial_eligible"))
            == bool(band.get("band_row_structure_eligible"))
        ):
            raise RuntimeError(
                "v2.5 invariant failed: band eligibility fields disagree."
            )
        subtypes = (
            bool(band.get("band_lateral_half_cut_excluded")),
            bool(band.get("band_lateral_sparse_border_excluded")),
            bool(band.get("band_longitudinal_edge_fragment_excluded")),
            bool(band.get("band_multirow_lateral_visibility_loss_excluded")),
        )
        if excluded and sum(subtypes) != 1:
            raise RuntimeError(
                "v2.5 invariant failed: excluded band does not have exactly one "
                f"effective edge subtype; source={source}, band={band_id}."
            )
        if not excluded and any(subtypes):
            raise RuntimeError(
                "v2.5 invariant failed: retained band carries an edge subtype."
            )
        if bool(band.get("band_multirow_lateral_visibility_loss_excluded")):
            maximum_loss = _finite(
                band.get(
                    "band_v25_multirow_lateral_visibility_maximum_loss_fraction"
                )
            )
            valid_multirow_exclusion = bool(
                len(linked) >= 2
                and (
                    bool(band.get("band_lateral_is_leftmost_band"))
                    or bool(band.get("band_lateral_is_rightmost_band"))
                )
                and bool(
                    band.get("band_v25_multirow_lateral_visibility_assessable")
                )
                and bool(
                    band.get("band_v25_multirow_lateral_visibility_triggered")
                )
                and maximum_loss is not None
                and maximum_loss
                >= float(settings.v25_multirow_lateral_loss_threshold_fraction)
                - 1e-9
            )
            if not valid_multirow_exclusion:
                raise RuntimeError(
                    "v2.5 invariant failed: multi-row lateral-loss exclusion "
                    "lacks stable outer-band geometric evidence."
                )

        audit_checks = (
            (
                bool(audit_record.get("spatial_eligible")),
                bool(band.get("band_spatial_eligible")),
            ),
            (
                bool(audit_record.get("row_structure_eligible")),
                bool(band.get("band_row_structure_eligible")),
            ),
            (
                bool(audit_record.get("complete_topology_eligible")),
                bool(band.get("band_complete_topology_eligible")),
            ),
            (
                bool(audit_record.get("edge_topology_excluded")),
                excluded,
            ),
            (
                bool(audit_record.get("edge_fragment_excluded")),
                bool(band.get("band_edge_fragment_excluded")),
            ),
            (
                bool(audit_record.get("lateral_half_cut_excluded")),
                bool(band.get("band_lateral_half_cut_excluded")),
            ),
            (
                bool(audit_record.get("lateral_sparse_border_excluded")),
                bool(band.get("band_lateral_sparse_border_excluded")),
            ),
            (
                bool(audit_record.get("longitudinal_edge_fragment_excluded")),
                bool(band.get("band_longitudinal_edge_fragment_excluded")),
            ),
            (
                bool(
                    audit_record.get(
                        "multirow_lateral_visibility_loss_excluded"
                    )
                ),
                bool(
                    band.get("band_multirow_lateral_visibility_loss_excluded")
                ),
            ),
        )
        if any(observed != expected for observed, expected in audit_checks):
            raise RuntimeError(
                "v2.5 invariant failed: final edge audit contradicts band state."
            )
        if str(audit_record.get("exclusion_reason") or "") != str(
            band.get("band_spatial_exclusion_reason") or ""
        ):
            raise RuntimeError(
                "v2.5 invariant failed: final edge-audit reason contradicts band."
            )

        expected_eligible = bool(band.get("band_complete_topology_eligible"))
        expected_reason = str(band.get("band_spatial_exclusion_reason") or "")
        for row in linked:
            if bool(row.get("row_complete_topology_eligible")) != expected_eligible:
                raise RuntimeError(
                    "v2.5 invariant failed: row/band eligibility mismatch."
                )
            if str(row.get("row_band_spatial_exclusion_reason") or "") != expected_reason:
                raise RuntimeError(
                    "v2.5 invariant failed: row/band reason mismatch."
                )
            row_members = [
                int(value) for value in row.get("_point_indices", [])
            ]
            if bool(row.get("trajectory_v25_singleton_row")) and len(
                row_members
            ) != 1:
                raise RuntimeError(
                    "v2.5 invariant failed: singleton row has !=1 point."
                )
            for point_index in row_members:
                assignment = assignments[point_index]
                if str(assignment.get("row_id")) != str(row.get("row_id")):
                    raise RuntimeError(
                        "v2.5 invariant failed: point/row link mismatch."
                    )
                if bool(
                    assignment.get("row_complete_topology_eligible")
                ) != expected_eligible:
                    raise RuntimeError(
                        "v2.5 invariant failed: point/band eligibility mismatch."
                    )
                if str(
                    assignment.get("row_band_spatial_exclusion_reason") or ""
                ) != expected_reason:
                    raise RuntimeError(
                        "v2.5 invariant failed: point/band reason mismatch."
                    )

    if sorted(seen) != list(range(len(stoma_traits))):
        raise RuntimeError(
            f"v2.5 invariant failed: points duplicated or lost; source={source}."
        )
    for field, before in eligibility_before.items():
        after = [base.parse_bool(record.get(field)) for record in stoma_traits]
        if after != before:
            raise RuntimeError(
                "v2.5 invariant failed: density/morphometry eligibility changed; "
                f"source={source}, field={field}."
            )


def infer_rows_and_bands_v25(
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettingsV25,
    audited_row_counts: dict[str, int] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
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
    rows, bands, assignments, audit = v24.infer_rows_and_bands_v24(
        stoma_traits,
        settings,
        audited_row_counts=audited_row_counts,
    )
    parent_edge_audit = (
        audit.get("band_spatial_eligibility")
        if isinstance(audit.get("band_spatial_eligibility"), list)
        else []
    )
    edge_audit = _apply_v25_standardized_multirow_edge_rule(
        rows,
        bands,
        assignments,
        stoma_traits,
        settings,
        parent_edge_audit,
    )
    audit["band_spatial_eligibility"] = edge_audit
    _annotate_v25_singleton_rows(rows, assignments, stoma_traits)

    diagnostics = list(audit.get("band_selection_diagnostics", {}).values())
    changed_groups = sum(
        bool(record.get("trajectory_v25_joint_change_applied"))
        for record in diagnostics
    )
    supported_singletons = sum(
        int(record.get("trajectory_v25_supported_singleton_count") or 0)
        for record in diagnostics
    )
    ambiguous_singletons = sum(
        int(record.get("trajectory_v25_ambiguous_singleton_count") or 0)
        for record in diagnostics
    )
    possible_ids = {
        str(stoma_id)
        for record in diagnostics
        for stoma_id in record.get("trajectory_v25_possible_singleton_stoma_ids", [])
        if str(stoma_id)
    }
    angle_bound_hits = sum(
        bool(record.get("trajectory_v25_residual_angle_at_search_bound"))
        for record in diagnostics
    )
    lower_bound = sum(
        int(
            record.get("trajectory_v25_row_count_lower_bound")
            or record.get("row_count_used_for_traits")
            or 0
        )
        for record in diagnostics
    )
    upper_bound = sum(
        int(
            record.get("trajectory_v25_row_count_upper_bound")
            or record.get("row_count_used_for_traits")
            or 0
        )
        for record in diagnostics
    )
    partition_diagnostics = audit.get("band_partition_diagnostics", [])
    added_boundaries = sum(
        bool(record.get("trajectory_v25_partition_split_applied"))
        and bool(record.get("accepted"))
        for record in partition_diagnostics
    )
    capacity_forced_boundaries = sum(
        bool(record.get("trajectory_v25_capacity_forced_unvalidated_boundary"))
        for record in partition_diagnostics
    )
    unvalidated_inherited_singleton_boundaries = sum(
        bool(
            record.get(
                "trajectory_v25_inherited_singleton_boundary_revalidation_unavailable"
            )
        )
        for record in partition_diagnostics
    )
    outer_multirow = [
        band
        for band in bands
        if (
            bool(band.get("band_lateral_is_leftmost_band"))
            or bool(band.get("band_lateral_is_rightmost_band"))
        )
        and int(band.get("row_count") or band.get("band_row_count") or 0) >= 2
    ]
    edge_assessed = sum(
        bool(band.get("band_v25_multirow_lateral_visibility_assessable"))
        for band in outer_multirow
    )
    edge_triggered = sum(
        bool(band.get("band_v25_multirow_lateral_visibility_triggered"))
        for band in outer_multirow
    )
    edge_new = sum(
        bool(
            band.get(
                "band_v25_multirow_lateral_visibility_effective_exclusion_added"
            )
        )
        for band in outer_multirow
    )
    edge_unassessable = len(outer_multirow) - edge_assessed
    audit.update(
        {
            "row_model_version": SCRIPT_VERSION,
            "trajectory_singleton_policy_version_current": SCRIPT_VERSION,
            "trajectory_v25_policy_version": SCRIPT_VERSION,
            "trajectory_v25_joint_changed_group_count": changed_groups,
            "trajectory_v25_supported_singleton_row_count": supported_singletons,
            "trajectory_v25_ambiguous_singleton_row_count": ambiguous_singletons,
            "trajectory_v25_possible_singleton_candidate_count": len(possible_ids),
            "trajectory_v25_residual_angle_bound_hit_group_count": angle_bound_hits,
            "trajectory_v25_added_band_boundary_count": added_boundaries,
            "trajectory_v25_capacity_forced_unvalidated_boundary_count": (
                capacity_forced_boundaries
            ),
            "trajectory_v25_unvalidated_inherited_singleton_boundary_count": (
                unvalidated_inherited_singleton_boundaries
            ),
            "trajectory_v25_multirow_edge_assessed_band_count": edge_assessed,
            "trajectory_v25_multirow_edge_triggered_band_count": edge_triggered,
            "trajectory_v25_multirow_edge_newly_excluded_band_count": edge_new,
            "trajectory_v25_multirow_edge_unassessable_band_count": edge_unassessable,
            "detected_row_count_v25": len(rows),
            "detected_band_count_v25": len(bands),
            "detected_row_count_v25_lower_bound": lower_bound,
            "detected_row_count_v25_upper_bound": upper_bound,
            "partial_band_eligibility_version": SCRIPT_VERSION,
            "final_detected_row_count": len(rows),
            "final_detected_band_count": len(bands),
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
            "automatic_row_count_policy": (
                "v2.5 jointly compares band-local shared-angle row partitions, "
                "ordinary rows and up to two singleton identities; confidence "
                "is soft evidence and ambiguous best estimates remain flagged"
            ),
            "band_partition_policy": (
                "v2.5 reconstructs rows first, archives and supersedes active "
                "v2.4 band cuts, then globally reassesses each provisional "
                "group and permits multiple stable boundaries; direct exact-pair "
                "raw-image support is required for isolated-row and singleton-side "
                "cuts, while any capacity-forced cut remains explicitly unvalidated"
            ),
            "partial_band_policy": (
                "v2.5 preserves every v2.4 single-row decision and excludes an "
                "outer multi-row band from complete topology if any row loses "
                "at least 20 percent of comparable trajectory through the "
                "relevant lateral image edge relative to the longest visible "
                "row in that band; shared absolute clipping is not estimated"
            ),
            "trajectory_row_band_trait_policy": (
                "best values are retained with lower/upper row-count bounds; "
                "density and valid cell morphometry never depend on topology QC"
            ),
        }
    )
    for collection in (rows, bands, list(assignments.values())):
        for record in collection:
            record["row_model_version"] = SCRIPT_VERSION
            record["trajectory_v25_policy_version"] = SCRIPT_VERSION
            record["partial_band_eligibility_version"] = SCRIPT_VERSION
    _assert_v25_invariants(
        stoma_traits,
        rows,
        bands,
        assignments,
        eligibility_before,
        edge_audit,
        settings,
    )
    return rows, bands, assignments, audit


def image_trait_summary_v25(
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
    result = v24.image_trait_summary_v24(
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
    supported = int(
        row_band_audit.get("trajectory_v25_supported_singleton_row_count") or 0
    )
    ambiguous = int(
        row_band_audit.get("trajectory_v25_ambiguous_singleton_row_count") or 0
    )
    reasons: list[str] = []
    for key, label in (
        ("trajectory_v25_joint_changed_group_count", "joint_row_change"),
        ("trajectory_v25_ambiguous_singleton_row_count", "ambiguous_singleton"),
        (
            "trajectory_v25_residual_angle_bound_hit_group_count",
            "residual_angle_search_bound",
        ),
        ("trajectory_v25_added_band_boundary_count", "additional_band_boundary"),
        (
            "trajectory_v25_capacity_forced_unvalidated_boundary_count",
            "capacity_forced_unvalidated_band_boundary",
        ),
        (
            "trajectory_v25_unvalidated_inherited_singleton_boundary_count",
            "unvalidated_inherited_singleton_boundary",
        ),
        (
            "trajectory_v25_multirow_edge_newly_excluded_band_count",
            "multirow_edge_exclusion",
        ),
        (
            "trajectory_v25_multirow_edge_unassessable_band_count",
            "multirow_edge_unassessable",
        ),
    ):
        if int(row_band_audit.get(key) or 0):
            reasons.append(f"v25_{label}")
    values = {
        field: row_band_audit.get(field)
        for field in _V25_IMAGE_FIELDS
        if field.startswith("trajectory_v25_")
    }
    values.update(
        {
            "detected_row_count_v25": len(row_records),
            "detected_band_count_v25": len(band_records),
            "detected_row_count_v25_lower_bound": row_band_audit.get(
                "detected_row_count_v25_lower_bound"
            ),
            "detected_row_count_v25_upper_bound": row_band_audit.get(
                "detected_row_count_v25_upper_bound"
            ),
            "supported_singleton_row_fraction_v25": (
                supported / len(row_records) if row_records else None
            ),
            "ambiguous_singleton_row_fraction_v25": (
                ambiguous / len(row_records) if row_records else None
            ),
            "row_band_v25_qc_status": (
                "flagged_provisional_change_or_uncertainty"
                if reasons
                else "pass_no_v25_change"
            ),
            "row_band_v25_qc_flag": bool(reasons),
            "row_band_v25_qc_reason": ";".join(reasons),
            "row_band_v25_value_interpretation": (
                "v2.5 best estimates include joint local-angle row assignment; "
                "supported singletons contribute to the primary topology, "
                "selected ambiguous singletons remain in the best estimate but "
                "are explicitly represented by QC and bounds, and "
                "confirmed edge-censored bands are excluded only from complete "
                "row/band topology traits"
            ),
            "row_geometry_model_version": SCRIPT_VERSION,
            "partial_band_eligibility_version": SCRIPT_VERSION,
            "trajectory_row_band_trait_policy": row_band_audit.get(
                "trajectory_row_band_trait_policy"
            ),
        }
    )
    result.update(values)
    return result


def image_gwas_trait_summary_v25(image_traits: dict[str, Any]) -> dict[str, Any]:
    summary = v24.image_gwas_trait_summary_v24(image_traits)
    for field in _V25_IMAGE_FIELDS:
        summary[field] = image_traits.get(field)
    summary["ambiguous_singleton_row_fraction_v25"] = image_traits.get(
        "ambiguous_singleton_row_fraction_v25"
    )
    summary["row_geometry_model_version"] = SCRIPT_VERSION
    summary["partial_band_eligibility_version"] = SCRIPT_VERSION
    return summary


def add_row_neighbors_v25(
    stoma_traits: list[dict[str, Any]],
    row_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
) -> None:
    v24.add_row_neighbors_v24(stoma_traits, row_records, point_assignment)
    for point_index, assignment in point_assignment.items():
        if not 0 <= int(point_index) < len(stoma_traits):
            continue
        stoma_traits[int(point_index)].update(
            {
                key: value
                for key, value in assignment.items()
                if key.startswith("trajectory_v25_")
                or key.startswith("row_v25_")
                or key == "row_multirow_lateral_visibility_loss_excluded"
            }
        )


def make_spatial_audit_overlay_v25(
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
    v24.make_spatial_audit_overlay_v24(
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
    image = base.cv2.imread(str(output_path), base.cv2.IMREAD_COLOR)
    if image is None:
        return
    row_lookup = {str(row["row_id"]): row for row in row_records}
    for band in band_records:
        members = [int(value) for value in band.get("_point_indices", [])]
        if not members:
            continue
        x = int(
            round(
                float(np.mean([stoma_traits[index]["center_x_pixels"] for index in members]))
            )
        )
        y = int(
            round(
                float(np.mean([stoma_traits[index]["center_y_pixels"] for index in members]))
            )
        )
        labels: list[tuple[str, tuple[int, int, int]]] = []
        if bool(band.get("trajectory_v25_partition_split_applied")):
            labels.append((f"{band['band_id']} V2.5 BAND SPLIT", (255, 255, 255)))
        if bool(band.get("band_multirow_lateral_visibility_loss_excluded")):
            loss = _finite(
                band.get(
                    "band_v25_multirow_lateral_visibility_maximum_loss_fraction"
                )
            )
            suffix = f" {100.0 * loss:.1f}%" if loss is not None else ""
            labels.append(
                (f"{band['band_id']} MULTIROW LATERAL LOSS{suffix}", (0, 0, 255))
            )
        elif bool(band.get("band_v25_multirow_lateral_visibility_triggered")):
            labels.append((f"{band['band_id']} EDGE LOSS CONFIRMED", (0, 100, 255)))
        for offset, (label, color) in enumerate(labels):
            base.cv2.putText(
                image,
                label,
                (max(5, x - 125), max(22, y - 46 - 18 * offset)),
                base.cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                color,
                2,
                base.cv2.LINE_AA,
            )
    for row in row_records:
        if not bool(row.get("trajectory_v25_singleton_row")):
            continue
        members = [int(value) for value in row.get("_point_indices", [])]
        if len(members) != 1:
            continue
        point_index = members[0]
        x = int(round(float(stoma_traits[point_index]["center_x_pixels"])))
        y = int(round(float(stoma_traits[point_index]["center_y_pixels"])))
        tier = str(row.get("trajectory_v25_singleton_evidence_tier") or "ambiguous")
        color = (0, 255, 0) if tier == "supported" else (0, 165, 255)
        base.cv2.circle(image, (x, y), 29, (0, 0, 0), 6, base.cv2.LINE_AA)
        base.cv2.circle(image, (x, y), 25, color, 3, base.cv2.LINE_AA)
        base.cv2.putText(
            image,
            f"{row['row_id']} V2.5 {tier.upper()} SINGLETON",
            (max(5, x - 130), max(22, y - 32)),
            base.cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            2,
            base.cv2.LINE_AA,
        )
    selected_singleton_ids = {
        str(stoma_traits[index].get("stoma_id") or "")
        for row in row_records
        if bool(row.get("trajectory_v25_singleton_row"))
        for index in row.get("_point_indices", [])
    }
    possible_frequencies: dict[str, float | None] = {}
    for row in row_records:
        frequency_map = row.get(
            "trajectory_v25_singleton_frequency_by_stoma", {}
        )
        if not isinstance(frequency_map, dict):
            frequency_map = {}
        for stoma_id in row.get(
            "trajectory_v25_possible_singleton_stoma_ids", []
        ):
            key = str(stoma_id)
            if key and key not in selected_singleton_ids:
                possible_frequencies[key] = _finite(frequency_map.get(key))
    point_by_id = {
        str(record.get("stoma_id") or ""): record for record in stoma_traits
    }
    for stoma_id, frequency in sorted(possible_frequencies.items()):
        record = point_by_id.get(stoma_id)
        if record is None:
            continue
        x = int(round(float(record["center_x_pixels"])))
        y = int(round(float(record["center_y_pixels"])))
        color = (255, 0, 255)
        base.cv2.circle(image, (x, y), 19, color, 2, base.cv2.LINE_AA)
        suffix = f" {frequency:.2f}" if frequency is not None else ""
        base.cv2.putText(
            image,
            f"V2.5 POSSIBLE SINGLETON{suffix}",
            (max(5, x - 115), min(image.shape[0] - 8, y + 34)),
            base.cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            2,
            base.cv2.LINE_AA,
        )
    if not base.cv2.imwrite(
        str(output_path), image, [base.cv2.IMWRITE_PNG_COMPRESSION, 6]
    ):
        raise RuntimeError(f"Failed to update v2.5 audit overlay: {output_path}")


def _write_csv_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fieldnames = sorted({str(key) for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: base.csv_value(record.get(key)) for key in fieldnames})


def _write_v25_additional_qc_tables(output_dir: Path) -> None:
    """Flatten the versioned JSON diagnostics already present in band QC."""

    band_path = output_dir / "band_traits_qc.csv"
    if not band_path.is_file():
        return
    topology_records: list[dict[str, Any]] = []
    singleton_records: list[dict[str, Any]] = []
    final_boundary_records: list[dict[str, Any]] = []
    boundary_candidate_records: list[dict[str, Any]] = []
    seen_groups: set[tuple[str, str]] = set()
    with band_path.open(newline="", encoding="utf-8") as handle:
        for band in csv.DictReader(handle):
            source = str(band.get("source_image_name") or "")
            provisional = str(band.get("provisional_band_id") or band.get("band_id") or "")
            group_key = (source, provisional)
            if group_key not in seen_groups:
                seen_groups.add(group_key)
                for field, target in (
                    ("trajectory_v25_candidate_diagnostics", topology_records),
                    (
                        "trajectory_v25_singleton_prefilter_diagnostics",
                        singleton_records,
                    ),
                ):
                    raw = band.get(field)
                    try:
                        values = json.loads(raw) if raw else []
                    except json.JSONDecodeError:
                        values = []
                    if isinstance(values, list):
                        for value in values:
                            if isinstance(value, dict):
                                target.append(
                                    {
                                        "source_image_name": source,
                                        "provisional_band_id": provisional,
                                        **value,
                                    }
                                )
            final_boundary_records.append(
                {
                    "source_image_name": source,
                    "band_id": band.get("band_id"),
                    "provisional_band_id": provisional,
                    "row_count": band.get("row_count"),
                    "stoma_count": band.get("stoma_count"),
                    "v25_partition_split_applied": band.get(
                        "trajectory_v25_partition_split_applied"
                    ),
                    "v25_additional_partition_split_applied": band.get(
                        "trajectory_v25_additional_partition_split_applied"
                    ),
                    "v25_partition_split_depth": band.get(
                        "trajectory_v25_partition_split_depth"
                    ),
                    "v25_partition_split_count_in_lineage": band.get(
                        "trajectory_v25_partition_split_count_in_lineage"
                    ),
                    "v25_capacity_forced_unvalidated_boundary": band.get(
                        "trajectory_v25_capacity_forced_unvalidated_boundary"
                    ),
                    "v25_capacity_forced_unvalidated_boundary_in_lineage": band.get(
                        "trajectory_v25_capacity_forced_unvalidated_boundary_in_lineage"
                    ),
                    "v25_capacity_forced_boundary_failed_gates": band.get(
                        "trajectory_v25_capacity_forced_boundary_failed_gates"
                    ),
                    "v25_partition_split_reason": band.get(
                        "trajectory_v25_partition_split_reason"
                    ),
                    "v25_partition_gap_um": band.get("trajectory_v25_partition_gap_um"),
                    "v25_partition_gap_width_factor": band.get(
                        "trajectory_v25_partition_gap_width_factor"
                    ),
                    "v25_partition_gap_ratio": band.get(
                        "trajectory_v25_partition_gap_ratio"
                    ),
                    "v25_partition_image_support_ratio": band.get(
                        "trajectory_v25_partition_image_support_ratio"
                    ),
                    "v25_partition_image_support_identity_exact": band.get(
                        "trajectory_v25_partition_image_support_identity_exact"
                    ),
                    "v25_partition_image_support_evidence_source": band.get(
                        "trajectory_v25_partition_image_support_evidence_source"
                    ),
                    "v25_partition_stability_fraction": band.get(
                        "trajectory_v25_partition_stability_fraction"
                    ),
                    "v25_inherited_singleton_boundary_validated": band.get(
                        "trajectory_v25_inherited_singleton_boundary_validated"
                    ),
                    "v25_validated_inherited_singleton_boundaries": band.get(
                        "trajectory_v25_validated_inherited_singleton_boundaries"
                    ),
                    "v25_validated_inherited_singleton_boundary_count": band.get(
                        "trajectory_v25_validated_inherited_singleton_boundary_count"
                    ),
                    "v25_inherited_singleton_boundary_revalidation_required": band.get(
                        "trajectory_v25_inherited_singleton_boundary_revalidation_required"
                    ),
                    "v25_inherited_singleton_boundary_revalidation_unavailable": band.get(
                        "trajectory_v25_inherited_singleton_boundary_revalidation_unavailable"
                    ),
                    "v25_inherited_singleton_boundary_retained_due_capacity": band.get(
                        "trajectory_v25_inherited_singleton_boundary_retained_due_capacity"
                    ),
                    "v25_multirow_edge_assessable": band.get(
                        "band_v25_multirow_lateral_visibility_assessable"
                    ),
                    "v25_multirow_edge_triggered": band.get(
                        "band_v25_multirow_lateral_visibility_triggered"
                    ),
                    "v25_multirow_edge_maximum_loss_fraction": band.get(
                        "band_v25_multirow_lateral_visibility_maximum_loss_fraction"
                    ),
                    "v25_multirow_edge_excluded": band.get(
                        "band_multirow_lateral_visibility_loss_excluded"
                    ),
                    "complete_topology_eligible": band.get(
                        "band_complete_topology_eligible"
                    ),
                }
            )
    provenance_path = output_dir / "trait_extraction_provenance.json"
    if provenance_path.is_file():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            provenance = {}
        audits = provenance.get("row_band_model_audit", {})
        if isinstance(audits, dict):
            for source, audit in audits.items():
                if not isinstance(audit, dict):
                    continue
                partitions = audit.get("band_partition_diagnostics", [])
                if not isinstance(partitions, list):
                    continue
                for partition_index, partition in enumerate(partitions, start=1):
                    if not isinstance(partition, dict):
                        continue
                    candidates = partition.get(
                        "trajectory_v25_partition_candidate_diagnostics", []
                    )
                    if not isinstance(candidates, list):
                        continue
                    for candidate_index, candidate in enumerate(candidates, start=1):
                        if not isinstance(candidate, dict):
                            continue
                        boundary_candidate_records.append(
                            {
                                "source_image_name": source,
                                "partition_audit_index": partition_index,
                                "candidate_index": candidate_index,
                                "parent_partition_accepted": partition.get("accepted"),
                                "parent_v25_split_applied": partition.get(
                                    "trajectory_v25_partition_split_applied"
                                ),
                                **candidate,
                            }
                        )

    _write_csv_records(output_dir / "row_topology_candidate_qc.csv", topology_records)
    _write_csv_records(output_dir / "singleton_candidate_qc.csv", singleton_records)
    _write_csv_records(
        output_dir / "band_boundary_candidate_qc.csv", boundary_candidate_records
    )
    _write_csv_records(
        output_dir / "band_boundary_final_qc.csv", final_boundary_records
    )


def _update_v25_output_provenance(output_dir: Path) -> None:
    provenance_path = output_dir / "trait_extraction_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        outputs = provenance.setdefault("outputs", {})
        optional_outputs = {
            "v25_row_topology_candidates": "row_topology_candidate_qc.csv",
            "v25_singleton_prefilter_candidates": "singleton_candidate_qc.csv",
            "v25_band_boundary_candidates": "band_boundary_candidate_qc.csv",
            "v25_final_band_boundary_qc": "band_boundary_final_qc.csv",
        }
        outputs.update(
            {
                key: filename
                for key, filename in optional_outputs.items()
                if (output_dir / filename).is_file()
            }
        )
        provenance["v25_development_status"] = (
            "development rules; audit this 40-image set, freeze all thresholds, "
            "then confirm once on a different balanced 40-image set"
        )
        provenance["v25_multirow_edge_loss_interpretation"] = (
            "differential lateral clipping relative to the longest visible "
            "fitted row in the same band; absolute clipping shared by all rows "
            "is not estimated"
        )
        provenance["v25_raw_image_support_threshold_status"] = (
            "development thresholds 1.65 (isolated multipoint row) and 1.80 "
            "(singleton side); recheck against this manual development audit "
            "using exact-pair local-trajectory/parallel-flank evidence before "
            "freezing and independent confirmation"
        )
        provenance_path.write_text(
            json.dumps(base.json_safe(provenance), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    report_path = output_dir / "TRAIT_EXTRACTION_REPORT.txt"
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8").rstrip()
        report += (
            "\nAvailable v2.5 joint row/singleton and standardized edge QC tables were "
            "written.\nV2.5 remains a development version pending this audit "
            "and an independent balanced confirmation set. The 1.65/1.80 "
            "raw-image support thresholds must be checked before freezing.\n"
        )
        report_path.write_text(report, encoding="utf-8")


def install_v25_policy() -> None:
    """Install v2.4, then replace only the versioned v2.5 hooks."""

    v24.install_v24_policy()
    v14._partition_provisional_bands = _partition_trajectory_bands_v25
    v15._trajectory_diagnostic_fields = _trajectory_diagnostic_fields_v25

    base.build_settings = build_settings_v25
    base.select_row_model = select_row_model_v25
    base.infer_rows_and_bands = infer_rows_and_bands_v25
    base.image_trait_summary = image_trait_summary_v25
    base.image_gwas_trait_summary = image_gwas_trait_summary_v25
    base.add_row_neighbors = add_row_neighbors_v25
    base.make_spatial_audit_overlay = make_spatial_audit_overlay_v25
    for module in (v14, v15, v16, base):
        module.SCRIPT_VERSION = SCRIPT_VERSION
    base.__file__ = __file__


def run_trait_extraction() -> Path:
    install_v25_policy()
    output_dir = base.run_trait_extraction()
    _write_v25_additional_qc_tables(output_dir)
    _update_v25_output_provenance(output_dir)
    return output_dir


if __name__ == "__main__":
    run_trait_extraction()
