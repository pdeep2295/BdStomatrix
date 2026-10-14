#!/usr/bin/env python3
"""Experimental v2.0 trait extractor with guarded singleton rows and bands.

This is a separate successor to the audited v1.9 extractor.  It does not
modify v1.9 or any model/pipeline output.  The only scientific change is that
a reliable stoma that was present in v1.6/v1.9's rejected singleton-row
hypothesis may be restored as a provisional row.  A strongly separated
singleton row may also form a biologically real one-stoma band.

The anti-fragmentation rule is deliberately strict.  v2.0 starts from the
unchanged v1.9 non-singleton solution, preserves every established row
membership, borrows its shared image/band orientation, and adds at most one
singleton to a multi-stoma provisional group.  A point that can be explained
as the tilted, curved or terminal continuation of an established row is not
split into a new row.

Accepted singleton rows contribute to row count, band count, row density and
rows-per-band.  A single point cannot independently estimate orientation,
within-row spacing, continuity, alternation or phase; those values therefore
remain undefined or are excluded by the established sample-count logic.

All thresholds are fixed and recorded in provenance for the requested
40-image audit.  They are provisional and must not be tuned on an independent
test set.  Run this script with a new output directory.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import extract_traits_from_pipeline_output as base
import extract_traits_from_pipeline_output_v1_4_global_rows as v14
import extract_traits_from_pipeline_output_v1_5_trajectory_rows as v15
import extract_traits_from_pipeline_output_v1_6_local_trajectory_guarded as v16
import extract_traits_from_pipeline_output_v1_7_retain_flagged_row_band_values as v17
import extract_traits_from_pipeline_output_v1_9_row_aware_lateral_half_cut_eligibility as v19


SCRIPT_VERSION = "2.0.0-experimental-validated-singleton-rows-bands"
SINGLETON_POLICY_VERSION = SCRIPT_VERSION
ROW_GEOMETRY_MODEL_VERSION = SCRIPT_VERSION
PARTIAL_BAND_ELIGIBILITY_VERSION = v19.PARTIAL_BAND_ELIGIBILITY_VERSION
TRAIT_EXPORT_POLICY_VERSION = v17.SCRIPT_VERSION

# Candidate evidence.  The confidence threshold is inherited from v1.5's
# sparse-trajectory audit and is not interpreted as a calibrated probability.
SINGLETON_MIN_DETECTOR_CONFIDENCE = 0.75
SINGLETON_MIN_ADJUSTED_SCORE_IMPROVEMENT = 2.0
SINGLETON_REQUIRED_NON_SINGLETON_COMEMBERSHIP = 1.0
SINGLETON_MAX_ADDED_ROWS_PER_MULTIPOINT_GROUP = 1

# Anti-fragmentation geometry, normalized by the image median detector-box
# width W.  <=0.30 W is compatible with continuation; 0.30--0.40 W is
# unresolved; >=0.40 W is distinct enough to continue validation.  The
# continuation radius is capped at 45% of the nearest confirmed inter-row gap.
SINGLETON_CONTINUATION_ABSORB_WIDTH_FACTOR = 0.30
SINGLETON_CONTINUATION_UNRESOLVED_WIDTH_FACTOR = 0.40
SINGLETON_CONTINUATION_INTERROW_GAP_CAP_FRACTION = 0.45
SINGLETON_DISTINCT_MINIMUM_UM = 4.0
SINGLETON_LOCAL_DIRECTION_MAX_DEVIATION_DEG = 35.0

# A within-band singleton must overlap an established row longitudinally or
# extend only a biologically small distance beyond it.  Strong singleton-band
# candidates do not require overlap.
SINGLETON_TERMINAL_EXTENSION_MEDIAN_GAP_FRACTION = 0.50
SINGLETON_TERMINAL_EXTENSION_MAX_WIDTH_FACTOR = 1.00
SINGLETON_TERMINAL_EXTENSION_FALLBACK_WIDTH_FACTOR = 0.50

# Use the established v1.5 band geometry.  Intermediate evidence retains the
# singleton row in the nearest band but is explicitly ambiguous.
SINGLETON_SAME_BAND_MAX_GAP_WIDTH_FACTOR = 1.20
SINGLETON_STRONG_BAND_GAP_WIDTH_FACTOR = 1.50
SINGLETON_ORIGINAL_SLOPE_MAX_DIFFERENCE_DEG = 2.0

_PARENT_V19_SCRIPT_PATH = Path(v19.__file__).resolve()

# Save concrete parent callables before installation changes dynamic hooks.
_PARENT_V16_CHOOSE_ONCE = v16._choose_trajectory_model_once_v16
_PARENT_V15_SELECT = v15.select_row_model_v15
_PARENT_V15_PARTITION = v15._partition_trajectory_bands_v15
_PARENT_V16_DIAGNOSTIC_FIELDS = v16._trajectory_diagnostic_fields_v16


@dataclass(frozen=True)
class TraitSettingsV20(v19.TraitSettingsV19):
    """Append immutable v2.0 singleton policy to the v1.9 settings."""

    trajectory_exact_singleton_rows_permitted: bool = True
    v20_singleton_policy_version: str = SINGLETON_POLICY_VERSION
    v20_singleton_rows_provisional_until_audit: bool = True
    v20_singleton_bands_permitted: bool = True
    v20_singleton_uses_classifier_or_segmentation: bool = False
    v20_singleton_uses_sobel_as_acceptance_gate: bool = False
    v20_non_singleton_membership_must_be_unchanged: bool = True
    v20_singleton_orientation_is_borrowed: bool = True
    v20_singleton_min_detector_confidence: float = (
        SINGLETON_MIN_DETECTOR_CONFIDENCE
    )
    v20_singleton_min_adjusted_score_improvement: float = (
        SINGLETON_MIN_ADJUSTED_SCORE_IMPROVEMENT
    )
    v20_singleton_required_non_singleton_comembership: float = (
        SINGLETON_REQUIRED_NON_SINGLETON_COMEMBERSHIP
    )
    v20_singleton_max_added_rows_per_multipoint_group: int = (
        SINGLETON_MAX_ADDED_ROWS_PER_MULTIPOINT_GROUP
    )
    v20_singleton_continuation_absorb_width_factor: float = (
        SINGLETON_CONTINUATION_ABSORB_WIDTH_FACTOR
    )
    v20_singleton_continuation_unresolved_width_factor: float = (
        SINGLETON_CONTINUATION_UNRESOLVED_WIDTH_FACTOR
    )
    v20_singleton_continuation_interrow_gap_cap_fraction: float = (
        SINGLETON_CONTINUATION_INTERROW_GAP_CAP_FRACTION
    )
    v20_singleton_distinct_minimum_um: float = SINGLETON_DISTINCT_MINIMUM_UM
    v20_singleton_local_direction_max_deviation_deg: float = (
        SINGLETON_LOCAL_DIRECTION_MAX_DEVIATION_DEG
    )
    v20_singleton_terminal_extension_median_gap_fraction: float = (
        SINGLETON_TERMINAL_EXTENSION_MEDIAN_GAP_FRACTION
    )
    v20_singleton_terminal_extension_max_width_factor: float = (
        SINGLETON_TERMINAL_EXTENSION_MAX_WIDTH_FACTOR
    )
    v20_singleton_terminal_extension_fallback_width_factor: float = (
        SINGLETON_TERMINAL_EXTENSION_FALLBACK_WIDTH_FACTOR
    )
    v20_singleton_same_band_max_gap_width_factor: float = (
        SINGLETON_SAME_BAND_MAX_GAP_WIDTH_FACTOR
    )
    v20_singleton_strong_band_gap_width_factor: float = (
        SINGLETON_STRONG_BAND_GAP_WIDTH_FACTOR
    )
    v20_singleton_original_slope_max_difference_deg: float = (
        SINGLETON_ORIGINAL_SLOPE_MAX_DIFFERENCE_DEG
    )
    v20_parent_v19_script_sha256: str = ""


def build_settings_v20(args: Any) -> TraitSettingsV20:
    """Keep every v1.9 setting and record the fixed v2.0 audit policy."""

    inherited = asdict(v19.build_settings_v19(args))
    inherited["trajectory_exact_singleton_rows_permitted"] = True
    inherited.update(
        {
            "v20_singleton_policy_version": SINGLETON_POLICY_VERSION,
            "v20_parent_v19_script_sha256": hashlib.sha256(
                _PARENT_V19_SCRIPT_PATH.read_bytes()
            ).hexdigest(),
        }
    )
    return TraitSettingsV20(**inherited)


def _finite(value: Any) -> float | None:
    parsed = base.optional_float(value)
    if parsed is None or not np.isfinite(parsed):
        return None
    return float(parsed)


def _candidate_label_map(candidate: dict[str, Any]) -> dict[int, int]:
    return {
        int(index): int(label)
        for index, label in zip(
            candidate.get("point_indices", []),
            candidate.get("labels", []),
            strict=True,
        )
    }


def _singleton_point_indices(candidate: dict[str, Any]) -> list[int]:
    labels = np.asarray(candidate.get("labels", []), dtype=int)
    points = [int(value) for value in candidate.get("point_indices", [])]
    singleton_groups = {
        group
        for group, size in enumerate(candidate.get("cluster_sizes", []))
        if int(size) == 1
    }
    return [
        points[position]
        for position, label in enumerate(labels)
        if int(label) in singleton_groups
    ]


def _detector_confidence(point_index: int) -> float | None:
    if v15._ACTIVE_STOMA_TRAITS_V15 is None:
        return None
    if not 0 <= point_index < len(v15._ACTIVE_STOMA_TRAITS_V15):
        return None
    return _finite(
        v15._ACTIVE_STOMA_TRAITS_V15[point_index].get("detector_confidence")
    )


def _stoma_id(point_index: int) -> str:
    if v15._ACTIVE_STOMA_TRAITS_V15 is None:
        return ""
    if not 0 <= point_index < len(v15._ACTIVE_STOMA_TRAITS_V15):
        return ""
    return str(v15._ACTIVE_STOMA_TRAITS_V15[point_index].get("stoma_id") or "")


def _pairwise_comembership_agreement(
    first: dict[str, Any],
    second: dict[str, Any],
    excluded_points: set[int],
) -> float:
    """Compare partitions without depending on arbitrary numeric row labels."""

    first_map = _candidate_label_map(first)
    second_map = _candidate_label_map(second)
    common = sorted((set(first_map) & set(second_map)) - excluded_points)
    comparisons = 0
    agreements = 0
    for left_position, left in enumerate(common):
        for right in common[left_position + 1 :]:
            comparisons += 1
            agreements += int(
                (first_map[left] == first_map[right])
                == (second_map[left] == second_map[right])
            )
    return agreements / comparisons if comparisons else 1.0


def _local_continuation_geometry(
    point_index: int,
    fallback: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV20,
) -> dict[str, Any]:
    """Test whether a singleton is better explained by a curved existing row."""

    point_map = _candidate_label_map(fallback)
    if point_index not in point_map:
        return {"assessable": False, "reason": "point_missing_from_fallback"}

    point_lateral = float(lateral[point_index])
    point_longitudinal = float(longitudinal[point_index])
    slope = float(fallback["slope"])
    reference = float(fallback["long_reference_um"])
    intercepts = np.asarray(fallback["intercepts_um"], dtype=float)
    adjusted_point = point_lateral - slope * (point_longitudinal - reference)
    shared_distances = np.abs(intercepts - adjusted_point)
    nearest_group = int(np.argmin(shared_distances))
    shared_residual_um = float(shared_distances[nearest_group])

    fallback_labels = point_map
    members = sorted(
        index
        for index, label in fallback_labels.items()
        if label == nearest_group and index != point_index
    )
    if len(members) < 2:
        return {
            "assessable": False,
            "reason": "removing_singleton_leaves_fewer_than_two_supported_points",
            "nearest_fallback_group": nearest_group,
            "shared_line_residual_um": shared_residual_um,
        }

    member_longitudinal = np.asarray(
        [float(longitudinal[index]) for index in members], dtype=float
    )
    member_lateral = np.asarray(
        [float(lateral[index]) for index in members], dtype=float
    )
    order = np.argsort(member_longitudinal, kind="stable")
    member_longitudinal = member_longitudinal[order]
    member_lateral = member_lateral[order]

    bracketed = bool(
        float(member_longitudinal[0])
        <= point_longitudinal
        <= float(member_longitudinal[-1])
    )
    unique_longitudinal = np.unique(member_longitudinal)
    gaps = np.diff(unique_longitudinal)
    median_gap = float(np.median(gaps)) if len(gaps) else None

    if bracketed and len(unique_longitudinal) >= 2:
        # np.interp is a local piecewise prediction and is deliberately more
        # forgiving of gradual curvature than the shared straight trajectory.
        local_prediction = float(
            np.interp(point_longitudinal, member_longitudinal, member_lateral)
        )
    else:
        if point_longitudinal < float(member_longitudinal[0]):
            selected = np.asarray([0, 1], dtype=int)
        else:
            selected = np.asarray(
                [len(member_longitudinal) - 2, len(member_longitudinal) - 1],
                dtype=int,
            )
        y_pair = member_longitudinal[selected]
        x_pair = member_lateral[selected]
        delta_y = float(y_pair[1] - y_pair[0])
        if abs(delta_y) <= 1e-9:
            local_slope = slope
        else:
            local_slope = float((x_pair[1] - x_pair[0]) / delta_y)
        slope_limit = math.tan(
            math.radians(
                float(settings.v20_singleton_local_direction_max_deviation_deg)
            )
        )
        local_slope = float(np.clip(local_slope, -slope_limit, slope_limit))
        local_prediction = float(
            x_pair[-1] + local_slope * (point_longitudinal - y_pair[-1])
            if point_longitudinal >= float(member_longitudinal[-1])
            else x_pair[0] + local_slope * (point_longitudinal - y_pair[0])
        )

    local_residual_um = abs(point_lateral - local_prediction)
    conservative_continuation_residual_um = min(
        shared_residual_um, local_residual_um
    )

    extension_um = max(
        float(member_longitudinal[0]) - point_longitudinal,
        point_longitudinal - float(member_longitudinal[-1]),
        0.0,
    )
    if median_gap is not None and median_gap > 0:
        extension_limit_um = min(
            float(settings.v20_singleton_terminal_extension_median_gap_fraction)
            * median_gap,
            float(settings.v20_singleton_terminal_extension_max_width_factor)
            * median_detector_width_um,
        )
    else:
        extension_limit_um = (
            float(settings.v20_singleton_terminal_extension_fallback_width_factor)
            * median_detector_width_um
        )

    other_gaps = [
        abs(float(value) - float(intercepts[nearest_group]))
        for group, value in enumerate(intercepts)
        if group != nearest_group
    ]
    nearest_interrow_gap_um = min(other_gaps) if other_gaps else None
    cap_um = (
        float(settings.v20_singleton_continuation_interrow_gap_cap_fraction)
        * nearest_interrow_gap_um
        if nearest_interrow_gap_um is not None
        else math.inf
    )
    absorb_threshold_um = min(
        float(settings.v20_singleton_continuation_absorb_width_factor)
        * median_detector_width_um,
        cap_um,
    )
    distinct_threshold_um = max(
        float(settings.v20_singleton_distinct_minimum_um),
        float(settings.v20_singleton_continuation_unresolved_width_factor)
        * median_detector_width_um,
    )
    if distinct_threshold_um < absorb_threshold_um:
        distinct_threshold_um = absorb_threshold_um

    if conservative_continuation_residual_um <= absorb_threshold_um:
        continuation_state = "absorbed_existing_row"
    elif conservative_continuation_residual_um < distinct_threshold_um:
        continuation_state = "unresolved_keep_conservative"
    else:
        continuation_state = "geometrically_distinct"

    return {
        "assessable": True,
        "reason": "",
        "nearest_fallback_group": nearest_group,
        "fallback_members_after_removal": members,
        "shared_line_residual_um": shared_residual_um,
        "local_piecewise_residual_um": local_residual_um,
        "conservative_continuation_residual_um": (
            conservative_continuation_residual_um
        ),
        "conservative_continuation_residual_width_factor": (
            conservative_continuation_residual_um
            / max(median_detector_width_um, 1e-9)
        ),
        "shared_line_separation_width_factor": (
            shared_residual_um / max(median_detector_width_um, 1e-9)
        ),
        "continuation_absorb_threshold_um": absorb_threshold_um,
        "continuation_distinct_threshold_um": distinct_threshold_um,
        "nearest_confirmed_interrow_gap_um": nearest_interrow_gap_um,
        "continuation_state": continuation_state,
        "longitudinally_bracketed": bracketed,
        "terminal_extension_um": extension_um,
        "terminal_extension_limit_um": extension_limit_um,
        "terminal_extension_supported": bool(
            bracketed or extension_um <= extension_limit_um
        ),
        "nearest_row_median_longitudinal_gap_um": median_gap,
    }


def _build_hybrid_singleton_candidate(
    fallback: dict[str, Any],
    singleton_point: int,
    lateral: np.ndarray,
    longitudinal: np.ndarray,
) -> dict[str, Any]:
    """Add one singleton while preserving every other fallback membership."""

    point_indices = [int(value) for value in fallback["point_indices"]]
    labels_by_point = _candidate_label_map(fallback)
    old_group = labels_by_point[singleton_point]
    labels_by_point[singleton_point] = int(fallback["k"])
    slope = float(fallback["slope"])
    reference = float(fallback["long_reference_um"])

    intercepts: list[float] = []
    for group in range(int(fallback["k"])):
        members = [
            index
            for index in point_indices
            if labels_by_point[index] == group
        ]
        if group == old_group and len(members) < 2:
            raise RuntimeError(
                "v2.0 hybrid singleton removal would leave an unsupported row."
            )
        adjusted = [
            float(lateral[index])
            - slope * (float(longitudinal[index]) - reference)
            for index in members
        ]
        intercepts.append(float(np.mean(adjusted)))
    singleton_intercept = float(lateral[singleton_point]) - slope * (
        float(longitudinal[singleton_point]) - reference
    )
    intercepts.append(singleton_intercept)

    order = np.argsort(np.asarray(intercepts, dtype=float))
    remap = {int(old): int(new) for new, old in enumerate(order)}
    sorted_intercepts = np.asarray(intercepts, dtype=float)[order]
    labels = np.asarray(
        [remap[labels_by_point[index]] for index in point_indices], dtype=int
    )
    local_lateral = np.asarray([float(lateral[index]) for index in point_indices])
    local_longitudinal = np.asarray(
        [float(longitudinal[index]) for index in point_indices]
    )
    fitted = slope * (local_longitudinal - reference) + sorted_intercepts[labels]
    residuals = local_lateral - fitted
    rss = float(np.sum(residuals * residuals))
    raw_variance = rss / max(len(point_indices), 1)
    residual_floor_um = float(fallback.get("bic_residual_floor_um") or 0.0)
    bic_variance = max(raw_variance, residual_floor_um**2, 1e-9)
    k = int(fallback["k"]) + 1
    bic = float(
        len(point_indices) * math.log(bic_variance)
        + (2 * k) * math.log(max(len(point_indices), 2))
    )

    hybrid = dict(fallback)
    hybrid.update(
        {
            "k": k,
            "point_indices": point_indices,
            "labels": labels,
            "long_reference_um": reference,
            "slope": slope,
            "slope_fixed_for_candidate": True,
            "intercepts_um": sorted_intercepts,
            "rss": rss,
            "rms_residual_um": math.sqrt(max(raw_variance, 0.0)),
            "bic_residual_floor_um": residual_floor_um,
            "bic": bic,
            "cluster_sizes": [
                int(np.count_nonzero(labels == group)) for group in range(k)
            ],
        }
    )
    return hybrid


_V20_DEFAULTS: dict[str, Any] = {
    "trajectory_v20_singleton_policy_version": SINGLETON_POLICY_VERSION,
    "trajectory_v20_singleton_candidate": False,
    "trajectory_v20_singleton_rescue_applied": False,
    "trajectory_v20_singleton_validation_status": "not_applicable",
    "trajectory_v20_singleton_validation_reason": "",
    "trajectory_v20_singleton_candidate_point_indices": [],
    "trajectory_v20_singleton_candidate_stoma_ids": [],
    "trajectory_v20_singleton_candidate_detector_confidences": [],
    "trajectory_v20_conservative_row_count": None,
    "trajectory_v20_augmented_row_count": None,
    "trajectory_v20_added_row_count": 0,
    "trajectory_v20_candidate_row_count": None,
    "trajectory_v20_candidate_cluster_sizes": [],
    "trajectory_v20_adjusted_score_improvement": None,
    "trajectory_v20_hybrid_adjusted_score_improvement": None,
    "trajectory_v20_non_singleton_comembership_agreement": None,
    "trajectory_v20_original_slope_difference_deg": None,
    "trajectory_v20_shared_line_residual_um": None,
    "trajectory_v20_local_piecewise_residual_um": None,
    "trajectory_v20_continuation_residual_um": None,
    "trajectory_v20_continuation_residual_width_factor": None,
    "trajectory_v20_continuation_state": "",
    "trajectory_v20_longitudinally_bracketed": None,
    "trajectory_v20_terminal_extension_um": None,
    "trajectory_v20_terminal_extension_limit_um": None,
    "trajectory_v20_terminal_extension_supported": None,
    "trajectory_v20_nearest_confirmed_interrow_gap_um": None,
    "trajectory_v20_singleton_route": "",
    "trajectory_v20_singleton_band_boundary_ambiguous": False,
    "trajectory_v20_singleton_only_supergroup": False,
    "trajectory_v20_orientation_inherited": False,
    "trajectory_v20_orientation_independently_estimable": True,
    "trajectory_v20_traits_involving_singleton_provisional": False,
    "trajectory_v20_parent_v16_fallback_applied": False,
}


def _metadata_with_defaults(metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    for key, value in _V20_DEFAULTS.items():
        result.setdefault(key, value)
    return result


def _choose_trajectory_model_once_v20(
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV20,
    separation_scale: float = 1.0,
    allow_sparse_image_rescue: bool = True,
) -> tuple[
    dict[str, Any],
    dict[int, tuple[dict[str, Any], dict[str, Any]]],
    dict[str, Any],
]:
    """Restore only a singleton that passes the fixed v2.0 safeguards."""

    fallback, best_by_k, parent_metadata = _PARENT_V16_CHOOSE_ONCE(
        point_indices,
        lateral,
        longitudinal,
        median_detector_width_um,
        settings,
        separation_scale=separation_scale,
        allow_sparse_image_rescue=allow_sparse_image_rescue,
    )
    metadata = _metadata_with_defaults(parent_metadata)
    fallback_k = int(fallback["k"])
    metadata["trajectory_v20_conservative_row_count"] = fallback_k
    metadata["trajectory_v20_augmented_row_count"] = fallback_k
    metadata["trajectory_v20_parent_v16_fallback_applied"] = bool(
        parent_metadata.get("trajectory_validation_fallback_applied")
    )

    # An isolated one-stoma provisional group is biologically allowed.  There
    # is no existing row that could be fragmented and no simpler zero-row model.
    if len(point_indices) == 1 and any(
        int(size) == 1 for size in fallback.get("cluster_sizes", [])
    ):
        point_index = int(point_indices[0])
        confidence = _detector_confidence(point_index)
        accepted = bool(
            confidence is not None
            and confidence >= float(settings.v20_singleton_min_detector_confidence)
        )
        metadata.update(
            {
                "trajectory_v20_singleton_candidate": True,
                "trajectory_v20_singleton_rescue_applied": accepted,
                "trajectory_v20_singleton_validation_status": (
                    "accepted_singleton_only_band"
                    if accepted
                    else "unresolved_singleton_only_band_low_confidence"
                ),
                "trajectory_v20_singleton_validation_reason": (
                    "isolated_reliable_detection_no_row_can_be_fragmented"
                    if accepted
                    else "detector_confidence_below_provisional_singleton_limit"
                ),
                "trajectory_v20_singleton_candidate_point_indices": [point_index],
                "trajectory_v20_singleton_candidate_stoma_ids": [
                    _stoma_id(point_index)
                ],
                "trajectory_v20_singleton_candidate_detector_confidences": [
                    confidence
                ],
                "trajectory_v20_candidate_row_count": 1,
                "trajectory_v20_candidate_cluster_sizes": [1],
                "trajectory_v20_singleton_route": "singleton_only_band",
                "trajectory_v20_singleton_only_supergroup": True,
                "trajectory_v20_orientation_inherited": True,
                "trajectory_v20_orientation_independently_estimable": False,
                "trajectory_v20_traits_involving_singleton_provisional": True,
                "trajectory_singleton_rows_validated": accepted,
                "trajectory_singleton_validation_failure_reason": (
                    ""
                    if accepted
                    else "v20_singleton_detector_confidence_below_limit"
                ),
                "trajectory_exact_row_traits_eligible": False,
                "trajectory_validation_fallback_applied": (
                    False
                    if accepted
                    else bool(
                        parent_metadata.get(
                            "trajectory_validation_fallback_applied"
                        )
                    )
                ),
                "trajectory_validation_fallback_reason": (
                    ""
                    if accepted
                    else str(
                        parent_metadata.get(
                            "trajectory_validation_fallback_reason"
                        )
                        or ""
                    )
                ),
                "trajectory_row_count_min": 1,
                "trajectory_row_count_max": 1,
                "trajectory_possible_sparse_row": not accepted,
                "trajectory_possible_sparse_additional_row_count": 0,
            }
        )
        if accepted:
            metadata["automatic_selection_reason"] = (
                str(metadata.get("automatic_selection_reason") or "")
                + "__v20_validated_singleton_only_band"
            )
        return fallback, best_by_k, metadata

    rejected_k = base.optional_int(
        parent_metadata.get("trajectory_pre_validation_row_count")
    )
    if not bool(parent_metadata.get("trajectory_validation_fallback_applied")):
        return fallback, best_by_k, metadata
    if rejected_k is None or rejected_k not in best_by_k:
        metadata.update(
            {
                "trajectory_v20_singleton_candidate": True,
                "trajectory_v20_singleton_validation_status": "rejected",
                "trajectory_v20_singleton_validation_reason": (
                    "parent_rejected_candidate_not_recoverable"
                ),
            }
        )
        return fallback, best_by_k, metadata

    rejected_candidate, rejected_diagnostic = best_by_k[int(rejected_k)]
    singleton_points = _singleton_point_indices(rejected_candidate)
    confidences = [_detector_confidence(index) for index in singleton_points]
    stoma_ids = [_stoma_id(index) for index in singleton_points]
    metadata.update(
        {
            "trajectory_v20_singleton_candidate": True,
            "trajectory_v20_singleton_candidate_point_indices": singleton_points,
            "trajectory_v20_singleton_candidate_stoma_ids": stoma_ids,
            "trajectory_v20_singleton_candidate_detector_confidences": confidences,
            "trajectory_v20_candidate_row_count": int(rejected_candidate["k"]),
            "trajectory_v20_candidate_cluster_sizes": [
                int(value) for value in rejected_candidate.get("cluster_sizes", [])
            ],
        }
    )

    rejection_reasons: list[str] = []
    if int(rejected_candidate["k"]) != fallback_k + 1:
        rejection_reasons.append("candidate_does_not_add_exactly_one_row")
    if len(singleton_points) != int(
        settings.v20_singleton_max_added_rows_per_multipoint_group
    ):
        rejection_reasons.append("candidate_does_not_contain_exactly_one_singleton")

    singleton_point = singleton_points[0] if len(singleton_points) == 1 else None
    if singleton_point is None:
        metadata.update(
            {
                "trajectory_v20_singleton_validation_status": "rejected",
                "trajectory_v20_singleton_validation_reason": ";".join(
                    rejection_reasons
                ),
            }
        )
        return fallback, best_by_k, metadata

    confidence = confidences[0]
    if confidence is None or confidence < float(
        settings.v20_singleton_min_detector_confidence
    ):
        rejection_reasons.append("detector_confidence_below_provisional_limit")

    strict_k = base.optional_int(parent_metadata.get("strict_best_row_count"))
    if strict_k != int(rejected_candidate["k"]):
        rejection_reasons.append("singleton_candidate_is_not_strict_best_model")

    fallback_diagnostic = best_by_k.get(fallback_k, ({}, {}))[1]
    fallback_score = _finite(fallback_diagnostic.get("adjusted_score"))
    rejected_score = _finite(rejected_diagnostic.get("adjusted_score"))
    score_improvement = (
        fallback_score - rejected_score
        if fallback_score is not None and rejected_score is not None
        else None
    )
    metadata["trajectory_v20_adjusted_score_improvement"] = score_improvement
    if score_improvement is None or score_improvement < float(
        settings.v20_singleton_min_adjusted_score_improvement
    ):
        rejection_reasons.append("adjusted_score_improvement_below_tie_tolerance")

    agreement = _pairwise_comembership_agreement(
        rejected_candidate, fallback, {singleton_point}
    )
    metadata["trajectory_v20_non_singleton_comembership_agreement"] = agreement
    if not math.isclose(
        agreement,
        float(settings.v20_singleton_required_non_singleton_comembership),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        rejection_reasons.append("established_non_singleton_memberships_changed")

    fallback_label_map = _candidate_label_map(fallback)
    fallback_parent_group = fallback_label_map.get(singleton_point)
    fallback_parent_members_after_removal = [
        index
        for index, label in fallback_label_map.items()
        if label == fallback_parent_group and index != singleton_point
    ]
    if fallback_parent_group is None or len(
        fallback_parent_members_after_removal
    ) < 2:
        rejection_reasons.append(
            "singleton_removal_would_leave_an_existing_row_under_supported"
        )

    candidate_angle = math.degrees(
        math.atan(float(rejected_candidate.get("slope") or 0.0))
    )
    fallback_angle = math.degrees(math.atan(float(fallback.get("slope") or 0.0)))
    slope_difference = abs(candidate_angle - fallback_angle)
    metadata["trajectory_v20_original_slope_difference_deg"] = slope_difference
    if slope_difference > float(
        settings.v20_singleton_original_slope_max_difference_deg
    ):
        rejection_reasons.append("candidate_slope_differs_from_supported_rows")

    geometry = _local_continuation_geometry(
        singleton_point,
        fallback,
        lateral,
        longitudinal,
        median_detector_width_um,
        settings,
    )
    if not bool(geometry.get("assessable")):
        rejection_reasons.append(
            str(geometry.get("reason") or "continuation_geometry_unassessable")
        )
    else:
        metadata.update(
            {
                "trajectory_v20_shared_line_residual_um": geometry.get(
                    "shared_line_residual_um"
                ),
                "trajectory_v20_local_piecewise_residual_um": geometry.get(
                    "local_piecewise_residual_um"
                ),
                "trajectory_v20_continuation_residual_um": geometry.get(
                    "conservative_continuation_residual_um"
                ),
                "trajectory_v20_continuation_residual_width_factor": geometry.get(
                    "conservative_continuation_residual_width_factor"
                ),
                "trajectory_v20_continuation_state": geometry.get(
                    "continuation_state"
                ),
                "trajectory_v20_longitudinally_bracketed": geometry.get(
                    "longitudinally_bracketed"
                ),
                "trajectory_v20_terminal_extension_um": geometry.get(
                    "terminal_extension_um"
                ),
                "trajectory_v20_terminal_extension_limit_um": geometry.get(
                    "terminal_extension_limit_um"
                ),
                "trajectory_v20_terminal_extension_supported": geometry.get(
                    "terminal_extension_supported"
                ),
                "trajectory_v20_nearest_confirmed_interrow_gap_um": geometry.get(
                    "nearest_confirmed_interrow_gap_um"
                ),
            }
        )
        continuation_state = str(geometry.get("continuation_state") or "")
        if continuation_state == "absorbed_existing_row":
            rejection_reasons.append("compatible_with_local_curved_row_continuation")
        elif continuation_state != "geometrically_distinct":
            rejection_reasons.append("continuation_geometry_unresolved")

    separation_factor = _finite(
        geometry.get("shared_line_separation_width_factor")
    )
    if separation_factor is None:
        route = "unresolved"
        rejection_reasons.append("lateral_route_unassessable")
    elif separation_factor <= float(
        settings.v20_singleton_same_band_max_gap_width_factor
    ):
        route = "within_existing_band"
    elif separation_factor < float(
        settings.v20_singleton_strong_band_gap_width_factor
    ):
        route = "nearest_band_boundary_ambiguous"
        metadata["trajectory_v20_singleton_band_boundary_ambiguous"] = True
    else:
        route = "strong_gap_singleton_band_candidate"
    metadata["trajectory_v20_singleton_route"] = route

    if route in {"within_existing_band", "nearest_band_boundary_ambiguous"}:
        if not bool(geometry.get("terminal_extension_supported")):
            rejection_reasons.append(
                "within_band_terminal_point_not_longitudinally_supported"
            )

    if rejection_reasons:
        continuation_state = str(
            metadata.get("trajectory_v20_continuation_state") or ""
        )
        status = (
            "absorbed_existing_row"
            if "compatible_with_local_curved_row_continuation" in rejection_reasons
            else "unresolved_keep_conservative"
            if continuation_state == "unresolved_keep_conservative"
            else "rejected"
        )
        metadata.update(
            {
                "trajectory_v20_singleton_validation_status": status,
                "trajectory_v20_singleton_validation_reason": ";".join(
                    dict.fromkeys(rejection_reasons)
                ),
            }
        )
        return fallback, best_by_k, metadata

    hybrid = _build_hybrid_singleton_candidate(
        fallback, singleton_point, lateral, longitudinal
    )
    angle_deg = math.degrees(math.atan(float(hybrid["slope"])))
    hybrid_diagnostic = v15._score_trajectory_candidate(
        hybrid, median_detector_width_um, angle_deg, settings
    )
    hybrid_score = _finite(hybrid_diagnostic.get("adjusted_score"))
    hybrid_score_improvement = (
        fallback_score - hybrid_score
        if fallback_score is not None and hybrid_score is not None
        else None
    )
    metadata["trajectory_v20_hybrid_adjusted_score_improvement"] = (
        hybrid_score_improvement
    )
    if hybrid_score_improvement is None or hybrid_score_improvement < float(
        settings.v20_singleton_min_adjusted_score_improvement
    ):
        metadata.update(
            {
                "trajectory_v20_singleton_validation_status": "rejected",
                "trajectory_v20_singleton_validation_reason": (
                    "membership_preserving_hybrid_score_improvement_below_"
                    "tie_tolerance"
                ),
            }
        )
        return fallback, best_by_k, metadata
    best_by_k[int(hybrid["k"])] = (hybrid, hybrid_diagnostic)

    metadata.update(
        {
            "automatic_row_count": int(hybrid["k"]),
            "automatic_selection_reason": (
                str(parent_metadata.get("automatic_selection_reason") or "")
                + "__v20_validated_singleton_rescue_hybrid_membership_preserved"
            ),
            "strict_best_row_count": int(hybrid["k"]),
            "strict_best_score": hybrid_diagnostic.get("adjusted_score"),
            "trajectory_pre_validation_row_count": int(rejected_candidate["k"]),
            "trajectory_post_validation_row_count": int(hybrid["k"]),
            "trajectory_validation_fallback_applied": False,
            "trajectory_validation_fallback_reason": "",
            "trajectory_displayed_non_singleton_row_count": fallback_k,
            "trajectory_displayed_fallback_row_count": fallback_k,
            "trajectory_row_count_min": fallback_k,
            "trajectory_row_count_max": int(hybrid["k"]),
            "trajectory_row_count_range_reason": (
                "v20_validated_singleton_rescue_provisional"
            ),
            "trajectory_possible_sparse_row": False,
            "trajectory_possible_sparse_additional_row_count": 0,
            "trajectory_singleton_rows_validated": True,
            "trajectory_singleton_validation_failure_reason": "",
            "trajectory_exact_row_traits_eligible": False,
            "trajectory_v20_singleton_rescue_applied": True,
            "trajectory_v20_singleton_validation_status": "accepted_provisional",
            "trajectory_v20_singleton_validation_reason": (
                "detector_score_geometry_membership_and_bend_guards_passed"
            ),
            "trajectory_v20_augmented_row_count": int(hybrid["k"]),
            "trajectory_v20_added_row_count": 1,
            "trajectory_v20_orientation_inherited": True,
            "trajectory_v20_orientation_independently_estimable": False,
            "trajectory_v20_traits_involving_singleton_provisional": True,
        }
    )
    return hybrid, best_by_k, metadata


def select_row_model_v20(
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV20,
    audited_row_count: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Use v1.5 stability machinery with the guarded v2.0 chooser."""

    diagnostic_count_before = len(v14._ACTIVE_BAND_DIAGNOSTICS)
    chosen, candidates = _PARENT_V15_SELECT(
        point_indices,
        lateral,
        longitudinal,
        median_detector_width_um,
        settings,
        audited_row_count=audited_row_count,
    )
    if len(v14._ACTIVE_BAND_DIAGNOSTICS) != diagnostic_count_before + 1:
        raise RuntimeError(
            "v2.0 expected exactly one provisional-band diagnostic record."
        )
    diagnostic = v14._ACTIVE_BAND_DIAGNOSTICS[-1]

    rescue_applied = bool(chosen.get("trajectory_v20_singleton_rescue_applied"))
    final_has_singleton = any(
        int(size) == 1 for size in chosen.get("cluster_sizes", [])
    )
    unresolved = bool(
        chosen.get("trajectory_validation_fallback_applied")
        or chosen.get("trajectory_possible_sparse_row")
        or chosen.get("trajectory_competing_two_row_hypothesis_unresolved")
    )
    chosen["tilt_slope_source"] = (
        "image_wide_axis_prior_plus_band_local_shared_residual_tilt"
    )
    diagnostic["tilt_slope_source"] = chosen["tilt_slope_source"]

    if rescue_applied and final_has_singleton:
        chosen["trajectory_singleton_rows_validated"] = True
        chosen["trajectory_singleton_validation_failure_reason"] = ""
        chosen["trajectory_exact_row_traits_eligible"] = False
        chosen["model_confidence"] = "low"
        diagnostic["trajectory_singleton_rows_validated"] = True
        diagnostic["trajectory_singleton_validation_failure_reason"] = ""
        diagnostic["trajectory_exact_row_traits_eligible"] = False
        diagnostic["global_row_model_confidence"] = "low"
    elif unresolved or final_has_singleton:
        chosen["trajectory_exact_row_traits_eligible"] = False
        chosen["model_confidence"] = "low"
        diagnostic["trajectory_exact_row_traits_eligible"] = False
        diagnostic["global_row_model_confidence"] = "low"

    for key, default in _V20_DEFAULTS.items():
        value = chosen.get(key, diagnostic.get(key, default))
        chosen[key] = value
        diagnostic[key] = value
    return chosen, candidates


def _trajectory_diagnostic_fields_v20(
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """Propagate all v2.0 decisions to band, row and stoma-level tables."""

    fields = _PARENT_V16_DIAGNOSTIC_FIELDS(diagnostic)
    fields["row_model_version"] = SCRIPT_VERSION
    for key, default in _V20_DEFAULTS.items():
        fields[key] = diagnostic.get(key, default)
    return fields


def _strong_gap_stability(
    gap_um: float,
    median_detector_width_um: float,
    settings: TraitSettingsV20,
    seed_offset: int,
) -> float:
    rng = np.random.default_rng(
        int(settings.trajectory_stability_seed) + int(seed_offset)
    )
    votes = 0
    for _ in range(int(settings.trajectory_stability_iterations)):
        perturbed_gap = max(
            0.0,
            gap_um
            + float(
                rng.normal(
                    0.0,
                    math.sqrt(2.0)
                    * float(settings.trajectory_stability_jitter_width_fraction)
                    * median_detector_width_um,
                )
            ),
        )
        votes += int(
            perturbed_gap / max(median_detector_width_um, 1e-9)
            >= float(settings.v20_singleton_strong_band_gap_width_factor)
        )
    return votes / max(int(settings.trajectory_stability_iterations), 1)


def _is_v20_rescued_singleton_row(row: dict[str, Any]) -> bool:
    reason = str(row.get("automatic_row_count_selection_reason") or "")
    return bool(
        (base.optional_int(row.get("stoma_count")) or 0) == 1
        and (
            "__v20_validated_singleton_rescue" in reason
            or "__v20_validated_singleton_only_band" in reason
        )
    )


def _partition_trajectory_bands_v20(
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    settings: TraitSettingsV20,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Keep ordinary v1.5 bands, then allow only strong singleton splits."""

    rows, parent_bands, assignments, partition_audit = _PARENT_V15_PARTITION(
        row_records,
        band_records,
        point_assignment,
        stoma_traits,
        median_detector_width_um,
        settings,
    )
    row_lookup = {str(row["row_id"]): row for row in rows}
    rebuilt: list[dict[str, Any]] = []

    for parent_band in parent_bands:
        linked_rows = sorted(
            [row_lookup[str(row_id)] for row_id in parent_band.get("row_ids", [])],
            key=lambda row: base.row_lateral_at(row, 0.0),
        )
        initial_singleton_band = bool(
            len(linked_rows) == 1
            and (base.optional_int(linked_rows[0].get("stoma_count")) or 0) == 1
            and "__v20_validated_singleton_only_band"
            in str(
                linked_rows[0].get("automatic_row_count_selection_reason") or ""
            )
        )
        if len(linked_rows) < 2 or not any(
            _is_v20_rescued_singleton_row(row) for row in linked_rows
        ):
            parent_band.setdefault("v20_singleton_partition_applied", False)
            parent_band["v20_singleton_band"] = initial_singleton_band
            parent_band["v20_singleton_band_origin"] = (
                "initial_singleton_only_supergroup"
                if initial_singleton_band
                else str(parent_band.get("v20_singleton_band_origin") or "")
            )
            if initial_singleton_band:
                linked_rows[0]["v20_singleton_band"] = True
                linked_rows[0]["v20_singleton_partition_applied"] = False
                for point_index in linked_rows[0].get("_point_indices", []):
                    assignments[int(point_index)]["v20_singleton_band"] = True
                    assignments[int(point_index)][
                        "v20_singleton_partition_applied"
                    ] = False
            rebuilt.append(parent_band)
            continue

        gaps = [
            v15._matched_trajectory_gap(first, second)[0]
            for first, second in zip(linked_rows[:-1], linked_rows[1:])
        ]
        gap_factors = [
            gap / max(median_detector_width_um, 1e-9) for gap in gaps
        ]
        gap_stabilities = [
            _strong_gap_stability(
                gap,
                median_detector_width_um,
                settings,
                sum(ord(char) for char in str(parent_band.get("band_id")))
                + index,
            )
            for index, gap in enumerate(gaps, start=1)
        ]
        strong = [
            bool(
                factor
                >= float(settings.v20_singleton_strong_band_gap_width_factor)
                and stability
                >= float(settings.band_boundary_stability_min_fraction)
            )
            for factor, stability in zip(gap_factors, gap_stabilities, strict=True)
        ]

        accepted_cuts: set[int] = set()
        cut_records: dict[int, dict[str, Any]] = {}
        for position, row in enumerate(linked_rows):
            if not _is_v20_rescued_singleton_row(row):
                continue
            left_ok = position > 0 and strong[position - 1]
            right_ok = position < len(linked_rows) - 1 and strong[position]
            if position == 0 and right_ok:
                accepted_cuts.add(1)
            elif position == len(linked_rows) - 1 and left_ok:
                accepted_cuts.add(position)
            elif 0 < position < len(linked_rows) - 1 and left_ok and right_ok:
                accepted_cuts.update({position, position + 1})

        for cut in sorted(accepted_cuts):
            gap_um = gaps[cut - 1]
            other_gaps = [
                value for index, value in enumerate(gaps) if index != cut - 1
            ]
            local_reference = (
                float(np.median(other_gaps)) if other_gaps else None
            )
            record = {
                "provisional_band_id": parent_band.get("provisional_band_id"),
                "split": cut,
                "boundary_gap_um": gap_um,
                "gap_ratio": (
                    gap_um / max(local_reference, 1e-9)
                    if local_reference is not None
                    else math.inf
                ),
                "gap_width_factor": gap_factors[cut - 1],
                "longitudinal_overlap_fraction": 0.0,
                "left_alternation": v14._row_group_alternation(
                    linked_rows[:cut], stoma_traits
                ),
                "right_alternation": v14._row_group_alternation(
                    linked_rows[cut:], stoma_traits
                ),
                "image_support_ratio": None,
                "accepted": True,
                "v20_singleton_zero_overlap_exception": True,
                "v20_singleton_boundary_stability_fraction": gap_stabilities[
                    cut - 1
                ],
                "v20_singleton_boundary_reason": (
                    "strong_matched_lateral_gap_adjacent_to_validated_singleton"
                ),
            }
            cut_records[cut] = record
            partition_audit.append(record)

        if not accepted_cuts:
            parent_band["v20_singleton_partition_applied"] = False
            parent_band["v20_singleton_band"] = False
            rebuilt.append(parent_band)
            continue

        boundaries = [0, *sorted(accepted_cuts), len(linked_rows)]
        groups = [
            linked_rows[start:stop]
            for start, stop in zip(boundaries[:-1], boundaries[1:])
            if stop > start
        ]
        original_provisional = parent_band.get("provisional_band_id")
        for group_index, group in enumerate(groups, start=1):
            start = boundaries[group_index - 1]
            stop = boundaries[group_index]
            nearby_cut = stop if stop in cut_records else start
            if nearby_cut not in cut_records:
                nearby_cut = sorted(cut_records)[0]
            partition = cut_records[nearby_cut]
            record = v14._rebuild_band_record(
                parent_band,
                group,
                stoma_traits,
                median_detector_width_um,
                partition,
                group_index,
            )
            record["provisional_band_id"] = original_provisional
            matched_gaps = [
                v15._matched_trajectory_gap(first, second)[0]
                for first, second in zip(group[:-1], group[1:])
            ]
            record.update(
                base.sample_summary(
                    matched_gaps, "within_band_interrow_spacing_um"
                )
            )
            record.update(
                {
                    "within_band_interrow_spacing_mean_um": (
                        float(np.mean(matched_gaps)) if matched_gaps else None
                    ),
                    "within_band_interrow_spacing_median_um": (
                        float(np.median(matched_gaps)) if matched_gaps else None
                    ),
                    "within_band_spacing_basis": (
                        "matched_longitudinal_positions_between_fitted_trajectories"
                    ),
                    "trajectory_band_partition_method": (
                        "v1.5_matched_gap_plus_v2.0_validated_singleton_strong_gap"
                    ),
                    "trajectory_multiple_boundaries_permitted": True,
                    "trajectory_unresolved_boundary_candidate": bool(
                        parent_band.get("trajectory_unresolved_boundary_candidate")
                    ),
                    "trajectory_rows_within_expected_band_range": bool(
                        1 <= len(group) <= settings.maximum_rows_per_band
                    ),
                    "trajectory_band_structure_stable": bool(
                        parent_band.get("trajectory_band_structure_stable")
                    ),
                    "trajectory_exact_row_band_traits_eligible": False,
                    "v20_singleton_partition_applied": True,
                    "v20_singleton_band": bool(
                        len(group) == 1
                        and _is_v20_rescued_singleton_row(group[0])
                    ),
                    "v20_pre_singleton_partition_band_id": parent_band.get(
                        "band_id"
                    ),
                    "v20_singleton_band_origin": (
                        "v2_strong_lateral_gap_partition"
                        if len(group) == 1
                        and _is_v20_rescued_singleton_row(group[0])
                        else ""
                    ),
                }
            )
            rebuilt.append(record)

    rebuilt.sort(key=lambda band: float(band["band_center_lateral_um"]))
    for band_number, band in enumerate(rebuilt, start=1):
        final_band_id = f"B{band_number:03d}"
        band["band_id"] = final_band_id
        band["band_order_left_to_right"] = band_number
        linked_rows = sorted(
            [row_lookup[str(row_id)] for row_id in band.get("row_ids", [])],
            key=lambda row: base.row_lateral_at(row, 0.0),
        )
        for position, row in enumerate(linked_rows, start=1):
            row["band_id"] = final_band_id
            row["row_position_within_band"] = position
            row["rows_in_band"] = len(linked_rows)
            row["automatic_rows_in_band"] = len(linked_rows)
            row["band_row_count"] = len(linked_rows)
            row["band_partition_applied"] = bool(
                band.get("band_partition_applied")
            )
            row["band_partition_group_index"] = band.get(
                "band_partition_group_index"
            )
            row["trajectory_band_structure_stable"] = band.get(
                "trajectory_band_structure_stable"
            )
            row["trajectory_exact_row_band_traits_eligible"] = False if (
                (base.optional_int(row.get("stoma_count")) or 0) == 1
            ) else band.get("trajectory_exact_row_band_traits_eligible")
            row["v20_singleton_band"] = bool(band.get("v20_singleton_band"))
            row["v20_singleton_partition_applied"] = bool(
                band.get("v20_singleton_partition_applied")
            )
            for point_index in row.get("_point_indices", []):
                assignment = assignments[int(point_index)]
                assignment["band_id"] = final_band_id
                assignment["row_position_within_band"] = position
                assignment["rows_in_band"] = len(linked_rows)
                assignment["automatic_rows_in_band"] = len(linked_rows)
                assignment["band_partition_applied"] = bool(
                    band.get("band_partition_applied")
                )
                assignment["trajectory_band_structure_stable"] = band.get(
                    "trajectory_band_structure_stable"
                )
                assignment["trajectory_exact_row_band_traits_eligible"] = (
                    row.get("trajectory_exact_row_band_traits_eligible")
                )
                assignment["v20_singleton_band"] = bool(
                    band.get("v20_singleton_band")
                )
                assignment["v20_singleton_partition_applied"] = bool(
                    band.get("v20_singleton_partition_applied")
                )

    return rows, rebuilt, assignments, partition_audit


def _assert_v20_invariants(
    stoma_traits: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    bands: list[dict[str, Any]],
    assignments: dict[int, dict[str, Any]],
) -> None:
    """Fail loudly rather than exporting internally inconsistent phenotypes."""

    expected_points = set(range(len(stoma_traits)))
    if set(assignments) != expected_points:
        raise RuntimeError("v2.0 invariant failed: every stoma must be assigned once.")
    row_ids = [str(row.get("row_id")) for row in rows]
    band_ids = [str(band.get("band_id")) for band in bands]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("v2.0 invariant failed: duplicate row IDs.")
    if len(band_ids) != len(set(band_ids)):
        raise RuntimeError("v2.0 invariant failed: duplicate band IDs.")

    row_lookup = {str(row["row_id"]): row for row in rows}
    row_members: list[int] = []
    for row in rows:
        members = [int(value) for value in row.get("_point_indices", [])]
        if not members:
            raise RuntimeError("v2.0 invariant failed: empty row.")
        row_members.extend(members)
        if (base.optional_int(row.get("stoma_count")) or 0) != len(members):
            raise RuntimeError("v2.0 invariant failed: row stoma count mismatch.")
    if sorted(row_members) != sorted(expected_points):
        raise RuntimeError("v2.0 invariant failed: row memberships do not partition stomata.")

    linked_row_ids: list[str] = []
    for expected_order, band in enumerate(bands, start=1):
        if base.optional_int(band.get("band_order_left_to_right")) != expected_order:
            raise RuntimeError("v2.0 invariant failed: non-contiguous band order.")
        linked = [str(value) for value in band.get("row_ids", [])]
        if any(row_id not in row_lookup for row_id in linked):
            raise RuntimeError("v2.0 invariant failed: band links unknown row.")
        linked_row_ids.extend(linked)
        declared = base.optional_int(band.get("row_count_used_for_traits"))
        if declared is None:
            declared = base.optional_int(band.get("row_count"))
        if declared != len(linked):
            raise RuntimeError("v2.0 invariant failed: band row-count mismatch.")
        members = sorted(
            {
                int(index)
                for row_id in linked
                for index in row_lookup[row_id].get("_point_indices", [])
            }
        )
        if members != sorted(int(value) for value in band.get("_point_indices", [])):
            raise RuntimeError("v2.0 invariant failed: band member union mismatch.")
    if sorted(linked_row_ids) != sorted(row_ids):
        raise RuntimeError("v2.0 invariant failed: each row must belong to one band.")


def infer_rows_and_bands_v20(
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettingsV20,
    audited_row_counts: dict[str, int] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    """Run v1.9 eligibility after the guarded singleton geometry."""

    rows, bands, assignments, audit = v19.infer_rows_and_bands_v19(
        stoma_traits,
        settings,
        audited_row_counts=audited_row_counts,
    )
    # v1.6 deliberately stamps its own module version on detailed records.
    # Replace that inherited label so every v2.0 table has truthful provenance.
    for row in rows:
        row["row_model_version"] = SCRIPT_VERSION
    for band in bands:
        band["row_model_version"] = SCRIPT_VERSION
    for assignment in assignments.values():
        assignment["row_model_version"] = SCRIPT_VERSION

    row_lookup = {str(row["row_id"]): row for row in rows}
    for row in rows:
        accepted_singleton_row = bool(
            (base.optional_int(row.get("stoma_count")) or 0) == 1
            and bool(row.get("trajectory_v20_singleton_rescue_applied"))
        )
        row.update(
            {
                "v20_rescued_singleton_row": accepted_singleton_row,
                "v20_singleton_support_status": (
                    "accepted_provisional"
                    if accepted_singleton_row
                    else "not_singleton_row"
                ),
                "v20_orientation_inherited": accepted_singleton_row,
                "v20_orientation_independently_estimable": (
                    not accepted_singleton_row
                ),
                "v20_traits_involving_singleton_provisional": (
                    accepted_singleton_row
                ),
            }
        )
        for point_index in row.get("_point_indices", []):
            assignments[int(point_index)].update(
                {
                    "v20_rescued_singleton_row": accepted_singleton_row,
                    "v20_singleton_support_status": row[
                        "v20_singleton_support_status"
                    ],
                    "v20_orientation_inherited": accepted_singleton_row,
                    "v20_orientation_independently_estimable": (
                        not accepted_singleton_row
                    ),
                    "v20_traits_involving_singleton_provisional": (
                        accepted_singleton_row
                    ),
                }
            )
    for band in bands:
        linked_rows = [
            row_lookup[str(row_id)]
            for row_id in band.get("row_ids", [])
            if str(row_id) in row_lookup
        ]
        accepted_singleton_present = any(
            (base.optional_int(row.get("stoma_count")) or 0) == 1
            and bool(row.get("trajectory_v20_singleton_rescue_applied"))
            for row in linked_rows
        )
        band["v20_traits_involving_singleton_provisional"] = (
            accepted_singleton_present
        )
        band["v20_singleton_supported_row_count"] = sum(
            bool(row.get("v20_rescued_singleton_row")) for row in linked_rows
        )
        if accepted_singleton_present:
            # One observation cannot support alternation or phase.  Keep the
            # row/band counts and lateral geometry, but do not allow the one
            # point to manufacture a longitudinal pattern statistic.
            band["successive_stoma_row_change_fraction"] = None
            band["two_row_alternation_index"] = None
            band["two_row_longest_alternating_sequence"] = None
            band["adjacent_row_phase_offset_normalized_mean"] = None
    _assert_v20_invariants(stoma_traits, rows, bands, assignments)

    diagnostics = audit.get("band_selection_diagnostics", {})
    unique_diagnostics = list(diagnostics.values())
    candidate_count = sum(
        bool(item.get("trajectory_v20_singleton_candidate"))
        for item in unique_diagnostics
    )
    candidate_point_count = sum(
        len(item.get("trajectory_v20_singleton_candidate_point_indices") or [])
        for item in unique_diagnostics
        if bool(item.get("trajectory_v20_singleton_candidate"))
    )
    accepted_group_count = sum(
        bool(item.get("trajectory_v20_singleton_rescue_applied"))
        for item in unique_diagnostics
    )
    stable_accepted_group_count = sum(
        bool(item.get("trajectory_v20_singleton_rescue_applied"))
        and bool(item.get("trajectory_row_count_stable"))
        for item in unique_diagnostics
    )
    absorbed_count = sum(
        item.get("trajectory_v20_singleton_validation_status")
        == "absorbed_existing_row"
        for item in unique_diagnostics
    )
    unresolved_count = sum(
        item.get("trajectory_v20_singleton_validation_status")
        in {
            "unresolved_keep_conservative",
            "unresolved_singleton_only_band_low_confidence",
        }
        for item in unique_diagnostics
    )
    rejected_count = max(
        0, candidate_count - accepted_group_count - absorbed_count - unresolved_count
    )
    accepted_rows = [
        row
        for row in rows
        if (base.optional_int(row.get("stoma_count")) or 0) == 1
        and bool(row.get("trajectory_v20_singleton_rescue_applied"))
    ]
    partition_diagnostics = audit.get("band_partition_diagnostics", [])
    added_band_count = sum(
        bool(item.get("v20_singleton_zero_overlap_exception"))
        and bool(item.get("accepted"))
        for item in partition_diagnostics
    )
    audit.update(
        {
            "row_model_version": SCRIPT_VERSION,
            "trajectory_v20_singleton_policy_version": SINGLETON_POLICY_VERSION,
            "trajectory_v20_singleton_candidate_group_count": candidate_count,
            "trajectory_v20_singleton_candidate_point_count": (
                candidate_point_count
            ),
            "trajectory_v20_singleton_accepted_group_count": accepted_group_count,
            "trajectory_v20_singleton_stable_accepted_group_count": (
                stable_accepted_group_count
            ),
            "trajectory_v20_singleton_accepted_group_stability_fraction": (
                stable_accepted_group_count / accepted_group_count
                if accepted_group_count
                else None
            ),
            "trajectory_v20_singleton_absorbed_group_count": absorbed_count,
            "trajectory_v20_singleton_unresolved_group_count": unresolved_count,
            "trajectory_v20_singleton_rejected_group_count": rejected_count,
            "trajectory_v20_singleton_supported_row_count": len(accepted_rows),
            "trajectory_v20_singleton_partition_added_band_count": added_band_count,
            "trajectory_v20_singleton_only_band_count": sum(
                bool(band.get("v20_singleton_band")) for band in bands
            ),
            "trajectory_v20_singleton_trait_policy": (
                "accepted singleton rows contribute to counts/densities and "
                "rows-per-band; orientation is borrowed; one-point within-row "
                "spacing, continuity, alternation and phase are undefined"
            ),
            "trajectory_sobel_policy": (
                "generic raw-edge contrast is not an acceptance gate for a "
                "singleton row or band; v2.0 uses detector confidence, fitted "
                "trajectory geometry, local bend/continuation checks and "
                "membership-preserving model comparison"
            ),
            "automatic_row_count_policy": (
                "start from the unchanged v1.9 non-singleton solution; allow a "
                "biologically possible one-stoma row or one-stoma band only "
                "when the fixed v2.0 safeguards pass; preserve all established "
                "non-singleton memberships and add at most one singleton row "
                "per multi-point provisional group"
            ),
            "trajectory_stability_policy": (
                "repeat the complete guarded selector after deterministic "
                "lateral jitter and minimum-separation perturbation; report "
                "stability as QC, but never use stability alone to accept a "
                "singleton"
            ),
            "trajectory_stability_limitations": (
                "accepted singleton rows and bands remain provisional pending "
                "the planned 40-image AB/AD visual audit; detection-dropout and "
                "longitudinal-window perturbations are not implemented"
            ),
            "trajectory_row_band_trait_policy": (
                "v2.0 retains both conservative v1.9 and singleton-augmented "
                "topology counts; accepted singleton observations can support "
                "row/band counts but cannot independently estimate orientation, "
                "spacing, continuity, alternation or phase"
            ),
        }
    )
    return rows, bands, assignments, audit


_V20_IMAGE_FIELDS = (
    "trajectory_v20_singleton_policy_version",
    "trajectory_v20_singleton_trait_policy",
    "trajectory_v20_singleton_candidate_group_count",
    "trajectory_v20_singleton_candidate_point_count",
    "trajectory_v20_singleton_accepted_group_count",
    "trajectory_v20_singleton_stable_accepted_group_count",
    "trajectory_v20_singleton_accepted_group_stability_fraction",
    "trajectory_v20_singleton_absorbed_group_count",
    "trajectory_v20_singleton_unresolved_group_count",
    "trajectory_v20_singleton_rejected_group_count",
    "trajectory_v20_singleton_supported_row_count",
    "trajectory_v20_singleton_partition_added_band_count",
    "trajectory_v20_singleton_only_band_count",
    "stomatal_row_count_conservative_v19",
    "stomatal_row_count_augmented_v20",
    "stomatal_row_count_added_by_v20",
    "stomatal_band_count_conservative_v19",
    "stomatal_band_count_augmented_v20",
    "stomatal_band_count_added_by_v20",
    "row_band_v20_qc_status",
    "row_band_v20_qc_flag",
    "row_band_v20_qc_reason",
    "row_band_v20_value_interpretation",
)


def image_trait_summary_v20(
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
    """Append conservative-versus-augmented singleton audit fields."""

    result = v19.image_trait_summary_v19(
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
    added_rows = sum(
        base.optional_int(item.get("trajectory_v20_added_row_count")) or 0
        for item in row_band_audit.get("band_selection_diagnostics", {}).values()
    )
    added_bands = int(
        row_band_audit.get("trajectory_v20_singleton_partition_added_band_count")
        or 0
    )
    # These are raw detected topology counts, matching the established meaning
    # of stomatal_row_count/stomatal_band_count.  Spatial eligibility remains a
    # separate QC layer in the v1.9 fields and is not mixed into this comparison.
    augmented_rows = len(row_records)
    augmented_bands = len(band_records)
    accepted_groups = int(
        row_band_audit.get("trajectory_v20_singleton_accepted_group_count") or 0
    )
    qc_reasons: list[str] = []
    if accepted_groups:
        qc_reasons.append("singleton_rows_are_provisional_pending_40_image_audit")
    if int(
        row_band_audit.get("trajectory_v20_singleton_unresolved_group_count") or 0
    ):
        qc_reasons.append("unresolved_singleton_candidates_retained_conservatively")
    if int(
        row_band_audit.get("trajectory_v20_singleton_rejected_group_count") or 0
    ):
        qc_reasons.append("rejected_singleton_candidates_recorded")

    result.update(
        {
            key: row_band_audit.get(key)
            for key in _V20_IMAGE_FIELDS
            if key.startswith("trajectory_v20_")
        }
    )
    result.update(
        {
            "stomatal_row_count_conservative_v19": augmented_rows - added_rows,
            "stomatal_row_count_augmented_v20": augmented_rows,
            "stomatal_row_count_added_by_v20": added_rows,
            "stomatal_band_count_conservative_v19": augmented_bands - added_bands,
            "stomatal_band_count_augmented_v20": augmented_bands,
            "stomatal_band_count_added_by_v20": added_bands,
            "row_band_v20_qc_status": (
                "flagged_provisional" if qc_reasons else "pass_no_singleton_change"
            ),
            "row_band_v20_qc_flag": bool(qc_reasons),
            "row_band_v20_qc_reason": ";".join(qc_reasons),
            "row_band_v20_value_interpretation": (
                "raw detected topology counts before spatial-eligibility "
                "filtering: v1.9 conservative and v2.0 singleton-augmented "
                "values are retained side by side; singleton-derived values "
                "remain provisional until biological audit"
            ),
            "row_geometry_model_version": SCRIPT_VERSION,
            "partial_band_eligibility_version": (
                PARTIAL_BAND_ELIGIBILITY_VERSION
            ),
            "trait_export_policy_version": TRAIT_EXPORT_POLICY_VERSION,
            "trajectory_row_band_trait_policy": (
                "v2.0 retains conservative v1.9 and singleton-augmented raw "
                "topology counts side by side; accepted singleton observations "
                "can support row/band counts but cannot independently estimate "
                "orientation, spacing, continuity, alternation or phase"
            ),
        }
    )
    return result


def image_gwas_trait_summary_v20(
    image_traits: dict[str, Any],
) -> dict[str, Any]:
    """Carry v2.0 audit fields into the compact GWAS-oriented table."""

    summary = v19.image_gwas_trait_summary_v19(image_traits)
    for field in _V20_IMAGE_FIELDS:
        summary[field] = image_traits.get(field)
    summary["row_geometry_model_version"] = SCRIPT_VERSION
    summary["partial_band_eligibility_version"] = (
        PARTIAL_BAND_ELIGIBILITY_VERSION
    )
    summary["trait_export_policy_version"] = TRAIT_EXPORT_POLICY_VERSION
    return summary


def add_row_neighbors_v20(
    stoma_traits: list[dict[str, Any]],
    row_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
) -> None:
    """Retain normal neighbors and expose singleton QC per stoma."""

    v19.add_row_neighbors_v19(stoma_traits, row_records, point_assignment)
    for index, assignment in point_assignment.items():
        if not 0 <= int(index) < len(stoma_traits):
            continue
        stoma_traits[int(index)].update(
            {
                key: value
                for key, value in assignment.items()
                if key.startswith("trajectory_v20_") or key.startswith("v20_")
            }
        )


def make_spatial_audit_overlay_v20(
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
    """Mark accepted singleton rows cyan and singleton bands orange."""

    v19.make_spatial_audit_overlay_v19(
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
    singleton_rows = [
        row
        for row in row_records
        if (base.optional_int(row.get("stoma_count")) or 0) == 1
        and bool(row.get("trajectory_v20_singleton_rescue_applied"))
    ]
    if not singleton_rows:
        return
    image = base.cv2.imread(str(output_path), base.cv2.IMREAD_COLOR)
    if image is None:
        return
    for row in singleton_rows:
        members = [int(value) for value in row.get("_point_indices", [])]
        if not members:
            continue
        index = members[0]
        x = int(round(float(stoma_traits[index]["center_x_pixels"])))
        y = int(round(float(stoma_traits[index]["center_y_pixels"])))
        singleton_band = bool(row.get("v20_singleton_band"))
        color = (0, 165, 255) if singleton_band else (255, 255, 0)
        label = (
            f"{row['row_id']} SINGLETON BAND"
            if singleton_band
            else f"{row['row_id']} SINGLETON ROW"
        )
        base.cv2.circle(image, (x, y), 24, color, 3, base.cv2.LINE_AA)
        base.cv2.line(image, (x - 20, y), (x + 20, y), color, 2, base.cv2.LINE_AA)
        base.cv2.line(image, (x, y - 20), (x, y + 20), color, 2, base.cv2.LINE_AA)
        text_y = max(22, y - 30)
        base.cv2.putText(
            image,
            label,
            (max(5, x - 90), text_y),
            base.cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            base.cv2.LINE_AA,
        )
    if not base.cv2.imwrite(
        str(output_path), image, [base.cv2.IMWRITE_PNG_COMPRESSION, 6]
    ):
        raise RuntimeError(f"Failed to update v2.0 audit overlay: {output_path}")


def install_v20_singleton_policy() -> None:
    """Install v1.9 first, then replace only singleton-related hooks."""

    v19.install_v19_partial_band_policy()

    # Dynamic hooks used inside the saved v1.5/v1.4 inference functions.
    v15._choose_trajectory_model_once = _choose_trajectory_model_once_v20
    v15._trajectory_diagnostic_fields = _trajectory_diagnostic_fields_v20
    v14._partition_provisional_bands = _partition_trajectory_bands_v20
    v15.SCRIPT_VERSION = SCRIPT_VERSION
    v14.SCRIPT_VERSION = SCRIPT_VERSION
    v16.SCRIPT_VERSION = SCRIPT_VERSION

    base.build_settings = build_settings_v20
    base.select_row_model = select_row_model_v20
    base.infer_rows_and_bands = infer_rows_and_bands_v20
    base.image_trait_summary = image_trait_summary_v20
    base.image_gwas_trait_summary = image_gwas_trait_summary_v20
    base.add_row_neighbors = add_row_neighbors_v20
    base.make_spatial_audit_overlay = make_spatial_audit_overlay_v20
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base.__file__ = __file__


def run_trait_extraction() -> Path:
    """Run the separate v2.0 audit candidate through the established engine."""

    install_v20_singleton_policy()
    return base.run_trait_extraction()


if __name__ == "__main__":
    run_trait_extraction()
