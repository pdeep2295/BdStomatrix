#!/usr/bin/env python3
"""Experimental v2.2 trait extractor with leave-one-out singleton rescue.

This is a separate successor to v2.1.  It does not modify v2.1, earlier trait
extractors, model output or pipeline output.  The complete v2.1 policy is kept
unchanged.  v2.2 adds one narrow rescue for a biologically audited failure
mode: an unrestricted fit may jump from k rows to k+2 (or rearrange supported
rows) even though the biologically coherent solution is to remove one lateral
outlier from its current row and return that one stoma as a k+1 singleton row.

The new route is nomination-only: the rejected unrestricted model may identify
the candidate stoma, but none of its memberships are adopted.  v2.2 freezes the
conservative v2.1 memberships, removes the nominated stoma once, verifies that
the remaining row is smooth and that the complete n-1 row solution is stable,
then tests a membership-preserving k+1 hybrid.  All established pairwise row
relationships must remain identical.  The candidate must be longitudinally
bracketed, strongly separated from a locally interpolated continuation, lie
within normal within-band lateral spacing, improve the adjusted model score,
and pass deterministic perturbation checks.  A terminal point, a curved-row
continuation, a parent model nominating multiple singleton points, or a point
whose removal leaves fewer than three supported stomata is not rescued.

This is not an image-name override and does not use classifier class,
segmentation quality, intensity/Sobel evidence, or manual labels as acceptance
evidence.  Only one non-recursive v2.2 singleton can be added per provisional
group, and no new band may be created by this route.  These constraints are
intended to recover genuine sparse rows without subdividing correct tilted or
gently curved rows.

Accepted singleton rows contribute to row count, band count, row density and
rows-per-band.  A single point cannot independently estimate orientation,
within-row spacing, continuity, alternation or phase; those values therefore
remain undefined or are excluded by the established sample-count logic.

All thresholds are fixed and recorded in provenance for the representative
40-image audit.  They remain provisional until a separate biological audit and
must not be tuned on an independent test set.  Run this script with a new
output directory.
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


SCRIPT_VERSION = "2.2.0-experimental-leave-one-out-singleton-rescue"
V21_POLICY_VERSION = "2.1.0-experimental-membership-preserving-singleton-rescue"
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

# Narrow v2.1 relaxation learned from the representative 40-image biological
# audit, not from an independent test set.  The unconstrained candidate may
# rearrange some points, but a majority of pairwise row relationships must
# already agree and the exported hybrid must preserve every established
# non-singleton relationship exactly.
V21_RELAXED_ORIGINAL_COMEMBERSHIP_MINIMUM = 2.0 / 3.0
V21_RELAXED_STRICT_BEST_MAX_EXTRA_ROWS = 1
V21_RELAXED_STRICT_BEST_MUST_CONTAIN_SINGLETON = True

# v2.2 leave-one-out safeguards.  A candidate is never discovered by scanning
# arbitrary points: it must first have been the sole singleton nominated by the
# guarded parent pre-validation model.  The remaining source row must contain
# at least three stomata and be very coherent after the candidate is removed.
V22_LOO_MIN_SOURCE_ROW_MEMBERS = 4
V22_LOO_MIN_REMAINING_ROW_MEMBERS = 3
V22_LOO_MIN_LOCAL_RESIDUAL_WIDTH_FACTOR = 0.40
V22_LOO_MIN_SHARED_RESIDUAL_WIDTH_FACTOR = 0.35
V22_LOO_MAX_REMAINING_ROW_RMS_WIDTH_FACTOR = 0.08
V22_LOO_MIN_RESIDUAL_DOMINANCE_RATIO = 3.0
V22_LOO_MAX_BRACKETING_GAP_RATIO = 2.50
V22_LOO_MIN_LATERAL_GAP_WIDTH_FACTOR = 0.40
V22_LOO_MAX_LATERAL_GAP_WIDTH_FACTOR = 1.20
V22_LOO_MIN_LATERAL_GAP_UM = 4.0
V22_LOO_MIN_ADJUSTED_SCORE_IMPROVEMENT = 2.0
V22_LOO_REQUIRED_COMEMBERSHIP = 1.0
V22_LOO_MAX_ACCEPTED_CANDIDATES_PER_GROUP = 1
V22_LOO_BASELINE_SLOPE_MAX_DIFFERENCE_DEG = 2.0

# The stability cutoffs deliberately inherit the established v1.5 values for
# row count and membership.  Candidate geometry and within-band routing use the
# same 0.80 minimum fraction as the established row-count stability policy.
V22_LOO_MIN_ROW_COUNT_STABILITY_FRACTION = 0.80
V22_LOO_MIN_ASSIGNMENT_STABILITY_FRACTION = 0.85
V22_LOO_MIN_GEOMETRY_STABILITY_FRACTION = 0.80
V22_LOO_MIN_SCORE_STABILITY_FRACTION = 0.80
V22_LOO_MIN_ROUTE_STABILITY_FRACTION = 0.80
V22_LOO_MIN_JOINT_STABILITY_FRACTION = 0.80

_PARENT_V19_SCRIPT_PATH = Path(v19.__file__).resolve()
_PARENT_V20_SCRIPT_PATH = Path(__file__).with_name(
    "extract_traits_from_pipeline_output_v2_0_validated_singleton_rows_bands.py"
).resolve()
_PARENT_V21_SCRIPT_PATH = Path(__file__).with_name(
    "extract_traits_from_pipeline_output_v2_1_membership_preserving_singleton_rescue.py"
).resolve()

# Save concrete parent callables before installation changes dynamic hooks.
_PARENT_V16_CHOOSE_ONCE = v16._choose_trajectory_model_once_v16
_PARENT_V15_SELECT = v15.select_row_model_v15
_PARENT_V15_PARTITION = v15._partition_trajectory_bands_v15
_PARENT_V16_DIAGNOSTIC_FIELDS = v16._trajectory_diagnostic_fields_v16


@dataclass(frozen=True)
class TraitSettingsV20(v19.TraitSettingsV19):
    """Append the immutable v2.2 singleton policy to the v1.9 settings."""

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
    v21_membership_preserving_relaxation_enabled: bool = True
    v21_relaxed_original_comembership_minimum: float = (
        V21_RELAXED_ORIGINAL_COMEMBERSHIP_MINIMUM
    )
    v21_relaxed_strict_best_max_extra_rows: int = (
        V21_RELAXED_STRICT_BEST_MAX_EXTRA_ROWS
    )
    v21_relaxed_strict_best_must_contain_singleton: bool = (
        V21_RELAXED_STRICT_BEST_MUST_CONTAIN_SINGLETON
    )
    v21_parent_v20_script_sha256: str = ""
    v22_leave_one_out_singleton_rescue_enabled: bool = True
    v22_loo_min_source_row_members: int = V22_LOO_MIN_SOURCE_ROW_MEMBERS
    v22_loo_min_remaining_row_members: int = V22_LOO_MIN_REMAINING_ROW_MEMBERS
    v22_loo_min_local_residual_width_factor: float = (
        V22_LOO_MIN_LOCAL_RESIDUAL_WIDTH_FACTOR
    )
    v22_loo_min_shared_residual_width_factor: float = (
        V22_LOO_MIN_SHARED_RESIDUAL_WIDTH_FACTOR
    )
    v22_loo_max_remaining_row_rms_width_factor: float = (
        V22_LOO_MAX_REMAINING_ROW_RMS_WIDTH_FACTOR
    )
    v22_loo_min_residual_dominance_ratio: float = (
        V22_LOO_MIN_RESIDUAL_DOMINANCE_RATIO
    )
    v22_loo_max_bracketing_gap_ratio: float = (
        V22_LOO_MAX_BRACKETING_GAP_RATIO
    )
    v22_loo_min_lateral_gap_width_factor: float = (
        V22_LOO_MIN_LATERAL_GAP_WIDTH_FACTOR
    )
    v22_loo_max_lateral_gap_width_factor: float = (
        V22_LOO_MAX_LATERAL_GAP_WIDTH_FACTOR
    )
    v22_loo_min_lateral_gap_um: float = V22_LOO_MIN_LATERAL_GAP_UM
    v22_loo_min_adjusted_score_improvement: float = (
        V22_LOO_MIN_ADJUSTED_SCORE_IMPROVEMENT
    )
    v22_loo_required_comembership: float = V22_LOO_REQUIRED_COMEMBERSHIP
    v22_loo_max_accepted_candidates_per_group: int = (
        V22_LOO_MAX_ACCEPTED_CANDIDATES_PER_GROUP
    )
    v22_loo_baseline_slope_max_difference_deg: float = (
        V22_LOO_BASELINE_SLOPE_MAX_DIFFERENCE_DEG
    )
    v22_loo_min_row_count_stability_fraction: float = (
        V22_LOO_MIN_ROW_COUNT_STABILITY_FRACTION
    )
    v22_loo_min_assignment_stability_fraction: float = (
        V22_LOO_MIN_ASSIGNMENT_STABILITY_FRACTION
    )
    v22_loo_min_geometry_stability_fraction: float = (
        V22_LOO_MIN_GEOMETRY_STABILITY_FRACTION
    )
    v22_loo_min_score_stability_fraction: float = (
        V22_LOO_MIN_SCORE_STABILITY_FRACTION
    )
    v22_loo_min_route_stability_fraction: float = (
        V22_LOO_MIN_ROUTE_STABILITY_FRACTION
    )
    v22_loo_min_joint_stability_fraction: float = (
        V22_LOO_MIN_JOINT_STABILITY_FRACTION
    )
    v22_parent_v21_script_sha256: str = ""
    v20_parent_v19_script_sha256: str = ""


def build_settings_v20(args: Any) -> TraitSettingsV20:
    """Keep every v1.9 setting and record the fixed v2.2 audit policy."""

    inherited = asdict(v19.build_settings_v19(args))
    inherited["trajectory_exact_singleton_rows_permitted"] = True
    inherited.update(
        {
            "v20_singleton_policy_version": SINGLETON_POLICY_VERSION,
            "v21_parent_v20_script_sha256": hashlib.sha256(
                _PARENT_V20_SCRIPT_PATH.read_bytes()
            ).hexdigest(),
            "v22_parent_v21_script_sha256": hashlib.sha256(
                _PARENT_V21_SCRIPT_PATH.read_bytes()
            ).hexdigest(),
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


def _refit_fixed_membership_candidate(
    template: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
) -> dict[str, Any]:
    """Refit intercepts and residual score while preserving every partition."""

    point_indices = [int(value) for value in template["point_indices"]]
    label_map = _candidate_label_map(template)
    slope = float(template["slope"])
    reference = float(template["long_reference_um"])
    k = int(template["k"])
    intercepts: list[float] = []
    for group in range(k):
        members = [
            index for index in point_indices if int(label_map[index]) == group
        ]
        if not members:
            raise RuntimeError("Fixed-membership refit encountered an empty row.")
        intercepts.append(
            float(
                np.mean(
                    [
                        float(lateral[index])
                        - slope * (float(longitudinal[index]) - reference)
                        for index in members
                    ]
                )
            )
        )
    order = np.argsort(np.asarray(intercepts, dtype=float))
    remap = {int(old): int(new) for new, old in enumerate(order)}
    sorted_intercepts = np.asarray(intercepts, dtype=float)[order]
    labels = np.asarray(
        [remap[int(label_map[index])] for index in point_indices], dtype=int
    )
    local_lateral = np.asarray(
        [float(lateral[index]) for index in point_indices], dtype=float
    )
    local_longitudinal = np.asarray(
        [float(longitudinal[index]) for index in point_indices], dtype=float
    )
    fitted = slope * (local_longitudinal - reference) + sorted_intercepts[labels]
    residuals = local_lateral - fitted
    rss = float(np.sum(residuals * residuals))
    raw_variance = rss / max(len(point_indices), 1)
    residual_floor_um = float(template.get("bic_residual_floor_um") or 0.0)
    bic_variance = max(raw_variance, residual_floor_um**2, 1e-9)
    bic = float(
        len(point_indices) * math.log(bic_variance)
        + (2 * k) * math.log(max(len(point_indices), 2))
    )
    refitted = dict(template)
    refitted.update(
        {
            "point_indices": point_indices,
            "labels": labels,
            "intercepts_um": sorted_intercepts,
            "rss": rss,
            "rms_residual_um": math.sqrt(max(raw_variance, 0.0)),
            "bic": bic,
            "cluster_sizes": [
                int(np.count_nonzero(labels == group)) for group in range(k)
            ],
        }
    )
    return refitted


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
                "Membership-preserving singleton removal would leave an "
                "unsupported row."
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


def _v22_leave_one_out_geometry(
    point_index: int,
    fallback: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV20,
) -> dict[str, Any]:
    """Measure one nominated point against its row after removing that point.

    Established memberships and the shared band slope are frozen.  This makes
    the test deliberately insensitive to an unrestricted higher-k model that
    fragments or rearranges otherwise coherent rows.
    """

    label_map = _candidate_label_map(fallback)
    if point_index not in label_map:
        return {"assessable": False, "reason": "candidate_missing_from_fallback"}

    source_group = int(label_map[point_index])
    source_members = sorted(
        index for index, label in label_map.items() if int(label) == source_group
    )
    remaining_source = [index for index in source_members if index != point_index]
    if len(source_members) < int(settings.v22_loo_min_source_row_members):
        return {
            "assessable": False,
            "reason": "source_row_has_too_few_members_before_leave_one_out",
            "source_group": source_group,
            "source_row_member_count_before": len(source_members),
            "source_row_member_count_after": len(remaining_source),
        }
    if len(remaining_source) < int(settings.v22_loo_min_remaining_row_members):
        return {
            "assessable": False,
            "reason": "source_row_has_too_few_members_after_leave_one_out",
            "source_group": source_group,
            "source_row_member_count_before": len(source_members),
            "source_row_member_count_after": len(remaining_source),
        }

    width = max(float(median_detector_width_um), 1e-9)
    slope = float(fallback["slope"])
    reference = float(fallback["long_reference_um"])

    def adjusted(index: int) -> float:
        return float(lateral[index]) - slope * (
            float(longitudinal[index]) - reference
        )

    group_intercepts: dict[int, float] = {}
    for group in range(int(fallback["k"])):
        members = [
            index
            for index, label in label_map.items()
            if int(label) == group and index != point_index
        ]
        if not members:
            return {
                "assessable": False,
                "reason": "leave_one_out_emptied_an_established_row",
                "source_group": source_group,
            }
        group_intercepts[group] = float(
            np.mean([adjusted(index) for index in members])
        )

    point_adjusted = adjusted(point_index)
    distances_by_group = {
        group: abs(point_adjusted - intercept)
        for group, intercept in group_intercepts.items()
    }
    nearest_group = min(
        distances_by_group,
        key=lambda group: (distances_by_group[group], group),
    )
    source_shared_residual_um = float(distances_by_group[source_group])
    nearest_row_separation_um = float(distances_by_group[nearest_group])

    remaining_adjusted = np.asarray(
        [adjusted(index) for index in remaining_source], dtype=float
    )
    source_intercept = float(group_intercepts[source_group])
    remaining_residuals = np.abs(remaining_adjusted - source_intercept)
    remaining_row_rms_um = float(
        math.sqrt(float(np.mean(remaining_residuals * remaining_residuals)))
    )
    maximum_remaining_residual_um = float(
        np.max(remaining_residuals) if len(remaining_residuals) else 0.0
    )
    residual_dominance_ratio = source_shared_residual_um / max(
        maximum_remaining_residual_um,
        0.05 * width,
        1e-9,
    )

    ordered_remaining = sorted(
        remaining_source,
        key=lambda index: (float(longitudinal[index]), int(index)),
    )
    remaining_longitudinal = np.asarray(
        [float(longitudinal[index]) for index in ordered_remaining], dtype=float
    )
    remaining_lateral = np.asarray(
        [float(lateral[index]) for index in ordered_remaining], dtype=float
    )
    point_longitudinal = float(longitudinal[point_index])
    point_lateral = float(lateral[point_index])
    lower = remaining_longitudinal[remaining_longitudinal < point_longitudinal]
    upper = remaining_longitudinal[remaining_longitudinal > point_longitudinal]
    bracketed = bool(len(lower) and len(upper))
    local_prediction = (
        float(
            np.interp(
                point_longitudinal,
                remaining_longitudinal,
                remaining_lateral,
            )
        )
        if bracketed
        else None
    )
    local_residual_um = (
        abs(point_lateral - local_prediction)
        if local_prediction is not None
        else None
    )

    source_longitudinal = np.sort(
        np.asarray(
            [float(longitudinal[index]) for index in source_members], dtype=float
        )
    )
    positive_source_gaps = np.diff(np.unique(source_longitudinal))
    positive_source_gaps = positive_source_gaps[positive_source_gaps > 1e-9]
    typical_source_gap_um = (
        float(np.median(positive_source_gaps))
        if len(positive_source_gaps)
        else None
    )
    bracketing_interval_um = (
        float(np.min(upper) - np.max(lower)) if bracketed else None
    )
    bracketing_gap_ratio = (
        bracketing_interval_um / typical_source_gap_um
        if bracketing_interval_um is not None
        and typical_source_gap_um is not None
        and typical_source_gap_um > 0
        else None
    )

    other_group_separations = [
        float(distance)
        for group, distance in distances_by_group.items()
        if int(group) != source_group
    ]
    nearest_other_established_row_um = (
        min(other_group_separations) if other_group_separations else None
    )
    other_row_points = [
        index
        for index, label in label_map.items()
        if int(label) != source_group
    ]
    nearest_other_row_longitudinal_offset_um = (
        min(
            abs(float(longitudinal[index]) - point_longitudinal)
            for index in other_row_points
        )
        if other_row_points
        else None
    )
    matched_longitudinal_lane_support = bool(
        nearest_other_row_longitudinal_offset_um is not None
        and typical_source_gap_um is not None
        and nearest_other_row_longitudinal_offset_um
        <= max(0.50 * typical_source_gap_um, 0.50 * width)
    )

    local_width_factor = (
        local_residual_um / width if local_residual_um is not None else None
    )
    shared_width_factor = source_shared_residual_um / width
    remaining_rms_width_factor = remaining_row_rms_um / width
    nearest_row_width_factor = nearest_row_separation_um / width
    minimum_gap_um = max(
        float(settings.v22_loo_min_lateral_gap_um),
        float(settings.v22_loo_min_lateral_gap_width_factor) * width,
    )
    maximum_gap_um = float(settings.v22_loo_max_lateral_gap_width_factor) * width

    gate_results = {
        "source_group_is_nearest": nearest_group == source_group,
        "longitudinally_bracketed": bracketed,
        "local_residual_passed": bool(
            local_width_factor is not None
            and local_width_factor
            >= float(settings.v22_loo_min_local_residual_width_factor)
        ),
        "shared_residual_passed": bool(
            shared_width_factor
            >= float(settings.v22_loo_min_shared_residual_width_factor)
        ),
        "remaining_row_rms_passed": bool(
            remaining_rms_width_factor
            <= float(settings.v22_loo_max_remaining_row_rms_width_factor)
        ),
        "residual_dominance_passed": bool(
            residual_dominance_ratio
            >= float(settings.v22_loo_min_residual_dominance_ratio)
        ),
        "bracketing_gap_passed": bool(
            bracketing_gap_ratio is not None
            and bracketing_gap_ratio
            <= float(settings.v22_loo_max_bracketing_gap_ratio)
        ),
        "within_existing_band_route_passed": bool(
            minimum_gap_um <= nearest_row_separation_um <= maximum_gap_um
        ),
    }
    geometry_passed = all(gate_results.values())
    failed_gates = [key for key, passed in gate_results.items() if not passed]
    return {
        "assessable": True,
        "reason": "" if geometry_passed else ";".join(failed_gates),
        "geometry_passed": geometry_passed,
        "source_group": source_group,
        "nearest_group": int(nearest_group),
        "source_row_member_count_before": len(source_members),
        "source_row_member_count_after": len(remaining_source),
        "source_row_members_after_removal": remaining_source,
        "shared_line_residual_um": source_shared_residual_um,
        "shared_line_residual_width_factor": shared_width_factor,
        "local_piecewise_residual_um": local_residual_um,
        "local_piecewise_residual_width_factor": local_width_factor,
        "remaining_row_rms_um": remaining_row_rms_um,
        "remaining_row_rms_width_factor": remaining_rms_width_factor,
        "maximum_remaining_residual_um": maximum_remaining_residual_um,
        "residual_dominance_ratio": residual_dominance_ratio,
        "longitudinally_bracketed": bracketed,
        "bracketing_interval_um": bracketing_interval_um,
        "typical_source_row_gap_um": typical_source_gap_um,
        "bracketing_gap_ratio": bracketing_gap_ratio,
        "nearest_row_separation_um": nearest_row_separation_um,
        "nearest_row_separation_width_factor": nearest_row_width_factor,
        "nearest_other_established_row_um": nearest_other_established_row_um,
        "nearest_other_row_longitudinal_offset_um": (
            nearest_other_row_longitudinal_offset_um
        ),
        "matched_longitudinal_lane_support": matched_longitudinal_lane_support,
        "minimum_within_band_gap_um": minimum_gap_um,
        "maximum_within_band_gap_um": maximum_gap_um,
        **gate_results,
    }


def _v22_fit_leave_one_out_baseline(
    point_index: int,
    fallback: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV20,
    separation_scale: float = 1.0,
) -> dict[str, Any]:
    """Refit the complete n-1 group and compare it with frozen memberships."""

    remaining_points = [
        int(index)
        for index in fallback.get("point_indices", [])
        if int(index) != int(point_index)
    ]
    if len(remaining_points) < 2:
        return {"assessable": False, "reason": "too_few_n_minus_one_points"}
    refitted, _, _ = _PARENT_V16_CHOOSE_ONCE(
        remaining_points,
        lateral,
        longitudinal,
        median_detector_width_um,
        settings,
        separation_scale=separation_scale,
        allow_sparse_image_rescue=False,
    )
    fallback_k = int(fallback["k"])
    refitted_k = int(refitted["k"])
    agreement = _pairwise_comembership_agreement(
        refitted, fallback, {int(point_index)}
    )
    slope_difference_deg = abs(
        math.degrees(math.atan(float(refitted.get("slope") or 0.0)))
        - math.degrees(math.atan(float(fallback.get("slope") or 0.0)))
    )
    cluster_support_passed = bool(
        refitted.get("cluster_sizes")
        and all(int(size) >= 2 for size in refitted.get("cluster_sizes", []))
    )
    row_count_passed = refitted_k == fallback_k
    membership_passed = math.isclose(
        agreement,
        float(settings.v22_loo_required_comembership),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    slope_passed = bool(
        slope_difference_deg
        <= float(settings.v22_loo_baseline_slope_max_difference_deg)
    )
    return {
        "assessable": True,
        "reason": "",
        "model": refitted,
        "row_count": refitted_k,
        "row_count_passed": row_count_passed,
        "comembership_agreement": agreement,
        "membership_passed": membership_passed,
        "minimum_cluster_support_passed": cluster_support_passed,
        "slope_difference_deg": slope_difference_deg,
        "slope_passed": slope_passed,
        "passed": bool(
            row_count_passed
            and membership_passed
            and cluster_support_passed
            and slope_passed
        ),
    }


def _v22_score_frozen_hybrid(
    point_index: int,
    fallback: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV20,
) -> dict[str, Any]:
    """Score only the k-to-k+1 frozen-membership singleton hypothesis."""

    refitted_fallback = _refit_fixed_membership_candidate(
        fallback, lateral, longitudinal
    )
    hybrid = _build_hybrid_singleton_candidate(
        refitted_fallback, point_index, lateral, longitudinal
    )
    fallback_angle = math.degrees(
        math.atan(float(refitted_fallback["slope"]))
    )
    hybrid_angle = math.degrees(math.atan(float(hybrid["slope"])))
    fallback_diagnostic = v15._score_trajectory_candidate(
        refitted_fallback,
        median_detector_width_um,
        fallback_angle,
        settings,
    )
    hybrid_diagnostic = v15._score_trajectory_candidate(
        hybrid,
        median_detector_width_um,
        hybrid_angle,
        settings,
    )
    fallback_score = float(fallback_diagnostic["adjusted_score"])
    hybrid_score = float(hybrid_diagnostic["adjusted_score"])
    improvement = fallback_score - hybrid_score
    agreement = _pairwise_comembership_agreement(
        hybrid, refitted_fallback, {int(point_index)}
    )
    score_passed = bool(
        improvement
        >= float(settings.v22_loo_min_adjusted_score_improvement)
    )
    membership_passed = math.isclose(
        agreement,
        float(settings.v22_loo_required_comembership),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    return {
        "hybrid": hybrid,
        "refitted_fallback": refitted_fallback,
        "fallback_diagnostic": fallback_diagnostic,
        "hybrid_diagnostic": hybrid_diagnostic,
        "fallback_adjusted_score": fallback_score,
        "hybrid_adjusted_score": hybrid_score,
        "adjusted_score_improvement": improvement,
        "score_passed": score_passed,
        "final_comembership_agreement": agreement,
        "membership_passed": membership_passed,
        "passed": bool(score_passed and membership_passed),
    }


_V20_DEFAULTS: dict[str, Any] = {
    "trajectory_singleton_policy_version_current": SCRIPT_VERSION,
    "trajectory_v20_singleton_policy_version": SINGLETON_POLICY_VERSION,
    "trajectory_v20_singleton_policy_version_semantics": (
        "legacy_compatibility_field_carries_current_umbrella_singleton_policy"
    ),
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
    "trajectory_v21_policy_version": V21_POLICY_VERSION,
    "trajectory_v21_membership_relaxation_candidate": False,
    "trajectory_v21_membership_relaxation_applied": False,
    "trajectory_v21_membership_relaxation_status": "not_applicable",
    "trajectory_v21_membership_relaxation_reason": "",
    "trajectory_v21_original_comembership_minimum": (
        V21_RELAXED_ORIGINAL_COMEMBERSHIP_MINIMUM
    ),
    "trajectory_v21_original_comembership_passed": None,
    "trajectory_v21_final_hybrid_comembership_agreement": None,
    "trajectory_v21_parent_prevalidation_candidate_selected": None,
    "trajectory_v21_unconstrained_strict_best_row_count": None,
    "trajectory_v21_strict_best_bypass_required": False,
    "trajectory_v21_strict_best_bypass_applied": False,
    "trajectory_v21_strict_best_singleton_row_count": None,
    "trajectory_v21_hybrid_adjusted_score_improvement": None,
    "trajectory_v21_source_candidate_adjusted_score_improvement": None,
    "trajectory_v21_parent_v20_row_count_stability_fraction": None,
    "trajectory_v21_parent_v20_assignment_stability_fraction": None,
    "trajectory_v21_parent_v20_stability_gate_passed": None,
    "trajectory_v21_parent_v20_stability_count_distribution": {},
    "trajectory_v21_final_hybrid_stability_reestimated": False,
    "trajectory_v21_stability_role": (
        "parent_v20_gate_only_final_hybrid_not_independently_perturbed"
    ),
    "trajectory_v22_policy_version": SCRIPT_VERSION,
    "trajectory_v22_leave_one_out_candidate": False,
    "trajectory_v22_leave_one_out_applied": False,
    "trajectory_v22_leave_one_out_status": "not_applicable",
    "trajectory_v22_leave_one_out_reason": "",
    "trajectory_v22_candidate_point_index": None,
    "trajectory_v22_candidate_stoma_id": "",
    "trajectory_v22_candidate_detector_confidence": None,
    "trajectory_v22_candidate_density_eligible": None,
    "trajectory_v22_nominating_candidate_row_count": None,
    "trajectory_v22_nominating_candidate_cluster_sizes": [],
    "trajectory_v22_conservative_row_count": None,
    "trajectory_v22_augmented_row_count": None,
    "trajectory_v22_source_group": None,
    "trajectory_v22_source_row_member_count_before": None,
    "trajectory_v22_source_row_member_count_after": None,
    "trajectory_v22_nominal_loo_row_count": None,
    "trajectory_v22_nominal_loo_comembership_agreement": None,
    "trajectory_v22_nominal_loo_slope_difference_deg": None,
    "trajectory_v22_nominal_loo_baseline_passed": None,
    "trajectory_v22_local_piecewise_residual_um": None,
    "trajectory_v22_local_piecewise_residual_width_factor": None,
    "trajectory_v22_shared_line_residual_um": None,
    "trajectory_v22_shared_line_residual_width_factor": None,
    "trajectory_v22_remaining_row_rms_um": None,
    "trajectory_v22_remaining_row_rms_width_factor": None,
    "trajectory_v22_residual_dominance_ratio": None,
    "trajectory_v22_longitudinally_bracketed": None,
    "trajectory_v22_bracketing_interval_um": None,
    "trajectory_v22_typical_source_row_gap_um": None,
    "trajectory_v22_bracketing_gap_ratio": None,
    "trajectory_v22_nearest_row_separation_um": None,
    "trajectory_v22_nearest_row_separation_width_factor": None,
    "trajectory_v22_nearest_other_established_row_um": None,
    "trajectory_v22_nearest_other_row_longitudinal_offset_um": None,
    "trajectory_v22_matched_longitudinal_lane_support": None,
    "trajectory_v22_source_group_is_nearest": None,
    "trajectory_v22_within_existing_band_route_passed": None,
    "trajectory_v22_nominal_geometry_passed": None,
    "trajectory_v22_hybrid_fallback_adjusted_score": None,
    "trajectory_v22_hybrid_adjusted_score": None,
    "trajectory_v22_hybrid_adjusted_score_improvement": None,
    "trajectory_v22_final_hybrid_comembership_agreement": None,
    "trajectory_v22_nominal_score_and_membership_passed": None,
    "trajectory_v22_nominal_gate_passed": None,
    "trajectory_v22_stability_iterations": None,
    "trajectory_v22_loo_row_count_stability_fraction": None,
    "trajectory_v22_loo_assignment_stability_fraction": None,
    "trajectory_v22_geometry_stability_fraction": None,
    "trajectory_v22_score_stability_fraction": None,
    "trajectory_v22_route_stability_fraction": None,
    "trajectory_v22_joint_gate_stability_fraction": None,
    "trajectory_v22_stability_gate_passed": None,
    "trajectory_v22_stability_count_distribution": {},
    "trajectory_v22_stability_role": (
        "fixed_candidate_n_minus_one_baseline_and_frozen_membership_hybrid_"
        "perturbation_not_unconstrained_full_model_reperturbation"
    ),
    "trajectory_v22_candidate_uses_classifier_or_segmentation": False,
    "trajectory_v22_candidate_uses_raw_intensity_or_sobel": False,
    "trajectory_v22_adds_new_band": False,
    "trajectory_v22_nonrecursive_single_pass": True,
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
    metadata["trajectory_v21_unconstrained_strict_best_row_count"] = strict_k
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
    relaxed_agreement_passed = bool(
        agreement + 1e-12
        >= float(settings.v21_relaxed_original_comembership_minimum)
    )
    metadata["trajectory_v21_original_comembership_passed"] = (
        relaxed_agreement_passed
    )
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

    # v2.1 narrow relaxation: the unconstrained candidate is used only to
    # identify one well-supported singleton.  The exported hybrid below still
    # preserves the conservative memberships exactly.  This addresses the
    # biologically audited S043_AD_P1 B006 failure mode without hard-coding an
    # image name or weakening the geometry/score safeguards.
    candidate_k = int(rejected_candidate["k"])
    prevalidation_k = base.optional_int(
        parent_metadata.get("trajectory_pre_validation_row_count")
    )
    parent_preselected_candidate = bool(prevalidation_k == candidate_k)
    metadata["trajectory_v21_parent_prevalidation_candidate_selected"] = (
        parent_preselected_candidate
    )

    strict_best_bypass_needed = bool(strict_k != candidate_k)
    strict_best_singleton_count: int | None = None
    strict_best_bypass_allowed = not strict_best_bypass_needed
    if strict_best_bypass_needed and strict_k is not None and strict_k in best_by_k:
        strict_candidate = best_by_k[int(strict_k)][0]
        strict_best_singleton_count = sum(
            int(size) == 1
            for size in strict_candidate.get("cluster_sizes", [])
        )
        strict_extra_rows = int(strict_k) - candidate_k
        strict_best_bypass_allowed = bool(
            0 < strict_extra_rows
            <= int(settings.v21_relaxed_strict_best_max_extra_rows)
            and (
                not bool(
                    settings.v21_relaxed_strict_best_must_contain_singleton
                )
                or strict_best_singleton_count >= 1
            )
        )
    metadata["trajectory_v21_strict_best_singleton_row_count"] = (
        strict_best_singleton_count
    )

    unique_rejection_reasons = list(dict.fromkeys(rejection_reasons))
    relaxable_reasons = {
        "singleton_candidate_is_not_strict_best_model",
        "established_non_singleton_memberships_changed",
    }
    # Do not relax inside this one-pass chooser.  Parent v2.0 stability is
    # calculated only after all deterministic perturbations have completed.
    # The selection wrapper below applies the constrained hybrid, if eligible,
    # after that stability evidence is available.
    relaxation_candidate = False
    deferred_relaxation_candidate = bool(
        settings.v21_membership_preserving_relaxation_enabled
        and unique_rejection_reasons
        and set(unique_rejection_reasons).issubset(relaxable_reasons)
        and candidate_k == fallback_k + 1
        and len(singleton_points) == 1
        and parent_preselected_candidate
        and relaxed_agreement_passed
        and strict_best_bypass_allowed
    )
    metadata["trajectory_v21_membership_relaxation_candidate"] = (
        deferred_relaxation_candidate
    )
    if deferred_relaxation_candidate:
        metadata.update(
            {
                "trajectory_v21_membership_relaxation_status": (
                    "deferred_until_parent_v20_stability_available"
                ),
                "trajectory_v21_membership_relaxation_reason": (
                    "one_pass_safeguards_passed_pending_post_stability_gate"
                ),
                "trajectory_v21_strict_best_bypass_required": (
                    strict_best_bypass_needed
                ),
            }
        )
    elif unique_rejection_reasons:
        metadata.update(
            {
                "trajectory_v21_membership_relaxation_status": "not_eligible",
                "trajectory_v21_membership_relaxation_reason": (
                    "relaxed_route_safeguards_not_all_satisfied"
                ),
            }
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
    metadata["trajectory_v21_hybrid_adjusted_score_improvement"] = (
        hybrid_score_improvement if relaxation_candidate else None
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

    final_hybrid_agreement = _pairwise_comembership_agreement(
        hybrid, fallback, {singleton_point}
    )
    metadata["trajectory_v21_final_hybrid_comembership_agreement"] = (
        final_hybrid_agreement if relaxation_candidate else None
    )
    if relaxation_candidate and not math.isclose(
        final_hybrid_agreement,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        metadata.update(
            {
                "trajectory_v20_singleton_validation_status": "rejected",
                "trajectory_v20_singleton_validation_reason": (
                    "v21_constrained_hybrid_did_not_preserve_all_"
                    "established_memberships"
                ),
                "trajectory_v21_membership_relaxation_status": "rejected",
                "trajectory_v21_membership_relaxation_reason": (
                    "final_hybrid_comembership_below_one"
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
                + (
                    "__v21_relaxed_original_membership_but_final_hybrid_"
                    "membership_preserved"
                    if relaxation_candidate
                    else "__v20_validated_singleton_rescue_hybrid_"
                    "membership_preserved"
                )
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
                "v21_constrained_hybrid_preserved_established_memberships_"
                "after_biologically_audited_relaxation"
                if relaxation_candidate
                else "detector_score_geometry_membership_and_bend_guards_passed"
            ),
            "trajectory_v20_augmented_row_count": int(hybrid["k"]),
            "trajectory_v20_added_row_count": 1,
            "trajectory_v20_orientation_inherited": True,
            "trajectory_v20_orientation_independently_estimable": False,
            "trajectory_v20_traits_involving_singleton_provisional": True,
            "trajectory_v21_membership_relaxation_applied": (
                relaxation_candidate
            ),
            "trajectory_v21_membership_relaxation_status": (
                "accepted_provisional"
                if relaxation_candidate
                else "not_needed_v20_strict_path_passed"
            ),
            "trajectory_v21_membership_relaxation_reason": (
                "constrained_hybrid_passed_all_v21_safeguards"
                if relaxation_candidate
                else ""
            ),
        }
    )
    return hybrid, best_by_k, metadata


def _apply_v21_post_stability_rescue(
    chosen: dict[str, Any],
    candidates: list[dict[str, Any]],
    diagnostic: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV20,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Apply the narrow constrained rescue after parent v2.0 stability.

    The parent selector has already completed all perturbations at this point.
    Its stability values describe the conservative v2.0 solution and are used
    only as the gate for this audited failure mode.  The final hybrid is not
    falsely described as having undergone a second independent stability run.
    """

    if not bool(
        diagnostic.get("trajectory_v21_membership_relaxation_candidate")
    ):
        return chosen, candidates, diagnostic

    allowed_reasons = {
        "singleton_candidate_is_not_strict_best_model",
        "established_non_singleton_memberships_changed",
    }
    reasons = {
        value
        for value in str(
            diagnostic.get("trajectory_v20_singleton_validation_reason") or ""
        ).split(";")
        if value
    }
    parent_row_stability = _finite(
        diagnostic.get("trajectory_row_count_stability_fraction")
    )
    parent_assignment_stability = _finite(
        diagnostic.get("trajectory_assignment_stability_fraction")
    )
    parent_distribution = dict(
        diagnostic.get("trajectory_stability_count_distribution") or {}
    )
    stability_gate_passed = bool(
        reasons
        and reasons.issubset(allowed_reasons)
        and diagnostic.get("trajectory_row_count_stable") is True
    )
    shared_gate_fields = {
        "trajectory_v21_parent_v20_row_count_stability_fraction": (
            parent_row_stability
        ),
        "trajectory_v21_parent_v20_assignment_stability_fraction": (
            parent_assignment_stability
        ),
        "trajectory_v21_parent_v20_stability_gate_passed": (
            stability_gate_passed
        ),
        "trajectory_v21_parent_v20_stability_count_distribution": (
            parent_distribution
        ),
        "trajectory_v21_final_hybrid_stability_reestimated": False,
        "trajectory_v21_stability_role": (
            "parent_v20_gate_only_final_hybrid_not_independently_perturbed"
        ),
    }
    chosen.update(shared_gate_fields)
    diagnostic.update(shared_gate_fields)
    if not stability_gate_passed:
        for target in (chosen, diagnostic):
            target.update(
                {
                    "trajectory_v21_membership_relaxation_applied": False,
                    "trajectory_v21_membership_relaxation_status": (
                        "rejected_parent_v20_stability_gate"
                    ),
                    "trajectory_v21_membership_relaxation_reason": (
                        "parent_v20_row_or_assignment_stability_below_limit"
                    ),
                }
            )
        return chosen, candidates, diagnostic

    singleton_points = [
        int(value)
        for value in (
            diagnostic.get("trajectory_v20_singleton_candidate_point_indices")
            or []
        )
    ]
    fallback_k = int(chosen["k"])
    candidate_k = base.optional_int(
        diagnostic.get("trajectory_v20_candidate_row_count")
    )
    if len(singleton_points) != 1 or candidate_k != fallback_k + 1:
        for target in (chosen, diagnostic):
            target.update(
                {
                    "trajectory_v21_membership_relaxation_applied": False,
                    "trajectory_v21_membership_relaxation_status": (
                        "rejected_invalid_post_stability_candidate"
                    ),
                    "trajectory_v21_membership_relaxation_reason": (
                        "candidate_no_longer_adds_exactly_one_singleton_row"
                    ),
                }
            )
        return chosen, candidates, diagnostic

    singleton_point = singleton_points[0]
    hybrid = _build_hybrid_singleton_candidate(
        chosen, singleton_point, lateral, longitudinal
    )
    hybrid_angle_deg = math.degrees(math.atan(float(hybrid["slope"])))
    hybrid_diagnostic = v15._score_trajectory_candidate(
        hybrid,
        median_detector_width_um,
        hybrid_angle_deg,
        settings,
    )
    fallback_diagnostic = next(
        (
            item
            for item in diagnostic.get("candidate_diagnostics", [])
            if base.optional_int(item.get("candidate_row_count")) == fallback_k
        ),
        {},
    )
    fallback_score = _finite(fallback_diagnostic.get("adjusted_score"))
    hybrid_score = _finite(hybrid_diagnostic.get("adjusted_score"))
    hybrid_improvement = (
        fallback_score - hybrid_score
        if fallback_score is not None and hybrid_score is not None
        else None
    )
    final_agreement = _pairwise_comembership_agreement(
        hybrid, chosen, {singleton_point}
    )
    score_passed = bool(
        hybrid_improvement is not None
        and hybrid_improvement
        >= float(settings.v20_singleton_min_adjusted_score_improvement)
    )
    membership_passed = math.isclose(
        final_agreement,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    if not score_passed or not membership_passed:
        reason = (
            "constrained_hybrid_score_improvement_below_limit"
            if not score_passed
            else "constrained_hybrid_changed_established_memberships"
        )
        for target in (chosen, diagnostic):
            target.update(
                {
                    "trajectory_v21_hybrid_adjusted_score_improvement": (
                        hybrid_improvement
                    ),
                    "trajectory_v21_final_hybrid_comembership_agreement": (
                        final_agreement
                    ),
                    "trajectory_v21_membership_relaxation_applied": False,
                    "trajectory_v21_membership_relaxation_status": "rejected",
                    "trajectory_v21_membership_relaxation_reason": reason,
                }
            )
        return chosen, candidates, diagnostic

    source_candidate_score_improvement = diagnostic.get(
        "trajectory_v20_adjusted_score_improvement"
    )
    original_strict_best_k = diagnostic.get(
        "trajectory_v21_unconstrained_strict_best_row_count"
    )
    singleton_detector_confidences = list(
        diagnostic.get(
            "trajectory_v20_singleton_candidate_detector_confidences"
        )
        or []
    )
    delta_bic_vs_conservative = float(chosen["bic"]) - float(hybrid["bic"])
    trajectory_maximum_k_reached = bool(
        int(candidate_k) == int(settings.trajectory_max_rows_per_supergroup)
        and len(hybrid.get("point_indices", []))
        > int(settings.trajectory_max_rows_per_supergroup)
    )
    legacy_row_count = base.optional_int(
        diagnostic.get("legacy_sequential_row_count")
    )
    rescued = dict(chosen)
    rescued.update(hybrid)
    selection_reason = (
        str(chosen.get("automatic_selection_reason") or "")
        + "__v21_post_stability_membership_preserving_singleton_rescue"
    )
    rescue_fields = {
        "automatic_row_count": int(hybrid["k"]),
        "automatic_selection_reason": selection_reason,
        "delta_bic_vs_simpler": delta_bic_vs_conservative,
        "row_count_selection_reason": (
            "v21_post_stability_membership_preserving_singleton_rescue"
        ),
        "sparse_row_support": True,
        "model_confidence": "low",
        "trajectory_slope_angle_deg": hybrid_angle_deg,
        "trajectory_singleton_validation_required": True,
        "trajectory_singleton_rows_validated": True,
        "trajectory_singleton_detector_confidences": (
            singleton_detector_confidences
        ),
        "trajectory_singleton_local_edge_support_ratios": [],
        "trajectory_singleton_validation_failure_reason": "",
        "trajectory_exact_row_traits_eligible": False,
        "trajectory_maximum_k_reached": trajectory_maximum_k_reached,
        "trajectory_unresolved_sparse_third_trajectory": False,
        "trajectory_pre_validation_row_count": int(candidate_k),
        "trajectory_post_validation_row_count": int(candidate_k),
        "trajectory_validation_fallback_applied": False,
        "trajectory_validation_fallback_reason": "",
        "trajectory_displayed_non_singleton_row_count": fallback_k,
        "trajectory_displayed_fallback_row_count": fallback_k,
        "trajectory_row_count_min": fallback_k,
        "trajectory_row_count_max": int(candidate_k),
        "trajectory_row_count_range_reason": (
            "v21_membership_preserving_singleton_rescue_provisional"
        ),
        "trajectory_possible_sparse_row": False,
        "trajectory_possible_sparse_additional_row_count": 0,
        # Parent stability is retained only under explicit v21 names.  These
        # generic fields must not imply that the final hybrid was perturbed.
        "trajectory_row_count_stability_fraction": None,
        "trajectory_assignment_stability_fraction": None,
        "trajectory_conditional_row_count_sensitivity_fraction": None,
        "trajectory_conditional_assignment_sensitivity_fraction": None,
        "trajectory_row_count_stable": False,
        "trajectory_stability_count_distribution": {},
        "trajectory_v20_singleton_rescue_applied": True,
        "trajectory_v20_singleton_validation_status": "accepted_provisional",
        "trajectory_v20_singleton_validation_reason": (
            "v21_post_stability_constrained_hybrid_passed"
        ),
        "trajectory_v20_augmented_row_count": int(candidate_k),
        "trajectory_v20_added_row_count": 1,
        "trajectory_v20_hybrid_adjusted_score_improvement": hybrid_improvement,
        "trajectory_v20_orientation_inherited": True,
        "trajectory_v20_orientation_independently_estimable": False,
        "trajectory_v20_traits_involving_singleton_provisional": True,
        "trajectory_v21_membership_relaxation_candidate": True,
        "trajectory_v21_membership_relaxation_applied": True,
        "trajectory_v21_membership_relaxation_status": (
            "accepted_provisional_parent_v20_stability_gate_passed"
        ),
        "trajectory_v21_membership_relaxation_reason": (
            "constrained_hybrid_preserved_established_rows"
        ),
        "trajectory_v21_hybrid_adjusted_score_improvement": hybrid_improvement,
        "trajectory_v21_source_candidate_adjusted_score_improvement": (
            source_candidate_score_improvement
        ),
        "trajectory_v21_final_hybrid_comembership_agreement": final_agreement,
        "trajectory_v21_unconstrained_strict_best_row_count": (
            original_strict_best_k
        ),
        "trajectory_v21_strict_best_bypass_applied": bool(
            original_strict_best_k is not None
            and int(original_strict_best_k) != int(candidate_k)
        ),
        **shared_gate_fields,
    }
    rescued.update(rescue_fields)

    updated_candidates = [
        hybrid if int(candidate.get("k")) == int(candidate_k) else candidate
        for candidate in candidates
    ]
    updated_candidate_diagnostics = [
        hybrid_diagnostic
        if base.optional_int(item.get("candidate_row_count"))
        == int(candidate_k)
        else item
        for item in diagnostic.get("candidate_diagnostics", [])
    ]
    diagnostic.update(
        {
            **rescue_fields,
            "global_automatic_row_count": int(candidate_k),
            "row_count_used_for_traits": int(candidate_k),
            "row_count_used_for_traits_source": (
                "v21_post_stability_membership_preserving_rescue"
            ),
            "alternative_row_count": fallback_k,
            "global_row_selection_score": hybrid_score,
            "global_row_score_margin": 0.0,
            "global_row_raw_margin": None,
            "global_row_automatic_is_strict_best": False,
            "global_row_model_confidence": "low",
            "global_row_count_changed_from_legacy": bool(
                legacy_row_count is not None
                and int(candidate_k) != int(legacy_row_count)
            ),
            "potential_row_undersplit_in_legacy": bool(
                legacy_row_count is not None
                and int(candidate_k) > int(legacy_row_count)
            ),
            "potential_row_oversplit_in_legacy": bool(
                legacy_row_count is not None
                and int(candidate_k) < int(legacy_row_count)
            ),
            "singleton_supported_row_count": 1,
            "singleton_detector_confidence_mean": (
                float(np.mean(singleton_detector_confidences))
                if singleton_detector_confidences
                else None
            ),
            "candidate_diagnostics": updated_candidate_diagnostics,
        }
    )
    return rescued, updated_candidates, diagnostic


def _v22_nominal_leave_one_out_evaluation(
    chosen: dict[str, Any],
    diagnostic: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV20,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Evaluate the single point nominated by the parent rejected model."""

    nominated_points = [
        int(value)
        for value in (
            diagnostic.get("trajectory_v20_singleton_candidate_point_indices")
            or []
        )
    ]
    nominating_k = base.optional_int(
        diagnostic.get("trajectory_v20_candidate_row_count")
    )
    nominating_sizes = [
        int(value)
        for value in (
            diagnostic.get("trajectory_v20_candidate_cluster_sizes") or []
        )
    ]
    common_fields: dict[str, Any] = {
        "trajectory_v22_policy_version": SCRIPT_VERSION,
        "trajectory_v22_leave_one_out_candidate": bool(nominated_points),
        "trajectory_v22_leave_one_out_applied": False,
        "trajectory_v22_nominating_candidate_row_count": nominating_k,
        "trajectory_v22_nominating_candidate_cluster_sizes": nominating_sizes,
        "trajectory_v22_conservative_row_count": int(chosen["k"]),
        "trajectory_v22_augmented_row_count": int(chosen["k"]),
    }
    if (
        len(nominated_points) != 1
        or len(nominated_points)
        > int(settings.v22_loo_max_accepted_candidates_per_group)
    ):
        common_fields.update(
            {
                "trajectory_v22_leave_one_out_status": (
                    "not_applicable"
                    if not nominated_points
                    else "rejected_ambiguous_nomination"
                ),
                "trajectory_v22_leave_one_out_reason": (
                    "parent_model_did_not_nominate_a_singleton"
                    if not nominated_points
                    else "parent_model_nominated_more_than_one_singleton"
                ),
                "trajectory_v22_nominal_gate_passed": False,
            }
        )
        return common_fields, None, None

    point_index = nominated_points[0]
    confidence = _detector_confidence(point_index)
    stoma_id = _stoma_id(point_index)
    active_record = (
        v15._ACTIVE_STOMA_TRAITS_V15[point_index]
        if v15._ACTIVE_STOMA_TRAITS_V15 is not None
        and 0 <= point_index < len(v15._ACTIVE_STOMA_TRAITS_V15)
        else None
    )
    density_eligible = (
        base.parse_bool(active_record.get("density_eligible"))
        if active_record is not None
        else None
    )
    common_fields.update(
        {
            "trajectory_v22_candidate_point_index": point_index,
            "trajectory_v22_candidate_stoma_id": stoma_id,
            "trajectory_v22_candidate_detector_confidence": confidence,
            "trajectory_v22_candidate_density_eligible": density_eligible,
        }
    )

    rejection_reasons: list[str] = []
    fallback_k = int(chosen["k"])
    if nominating_k is None or nominating_k <= fallback_k:
        rejection_reasons.append("nominating_model_did_not_add_rows")
    if nominating_sizes.count(1) != 1:
        rejection_reasons.append("nominating_model_did_not_have_one_singleton")
    if any(int(size) == 1 for size in chosen.get("cluster_sizes", [])):
        rejection_reasons.append("conservative_model_already_contains_singleton")
    if fallback_k + 1 > int(settings.maximum_rows_per_band):
        rejection_reasons.append("augmented_count_exceeds_biological_row_limit")
    if confidence is None or confidence < float(
        settings.v20_singleton_min_detector_confidence
    ):
        rejection_reasons.append("detector_confidence_below_limit")
    if density_eligible is not True:
        rejection_reasons.append("candidate_not_density_eligible")

    geometry = _v22_leave_one_out_geometry(
        point_index,
        chosen,
        lateral,
        longitudinal,
        median_detector_width_um,
        settings,
    )
    if not bool(geometry.get("assessable")):
        rejection_reasons.append(
            str(geometry.get("reason") or "leave_one_out_geometry_unassessable")
        )
    elif not bool(geometry.get("geometry_passed")):
        rejection_reasons.extend(
            value
            for value in str(geometry.get("reason") or "").split(";")
            if value
        )

    try:
        loo_baseline = _v22_fit_leave_one_out_baseline(
            point_index,
            chosen,
            lateral,
            longitudinal,
            median_detector_width_um,
            settings,
        )
    except (RuntimeError, ValueError) as error:
        loo_baseline = {
            "assessable": False,
            "reason": f"n_minus_one_refit_failed:{type(error).__name__}",
        }
    if not bool(loo_baseline.get("assessable")):
        rejection_reasons.append(
            str(loo_baseline.get("reason") or "n_minus_one_baseline_unassessable")
        )
    elif not bool(loo_baseline.get("passed")):
        if not bool(loo_baseline.get("row_count_passed")):
            rejection_reasons.append("n_minus_one_row_count_changed")
        if not bool(loo_baseline.get("membership_passed")):
            rejection_reasons.append("n_minus_one_memberships_changed")
        if not bool(loo_baseline.get("minimum_cluster_support_passed")):
            rejection_reasons.append("n_minus_one_row_support_below_two")
        if not bool(loo_baseline.get("slope_passed")):
            rejection_reasons.append("n_minus_one_slope_changed_excessively")

    try:
        score = _v22_score_frozen_hybrid(
            point_index,
            chosen,
            lateral,
            longitudinal,
            median_detector_width_um,
            settings,
        )
    except (RuntimeError, ValueError) as error:
        score = {
            "score_passed": False,
            "membership_passed": False,
            "passed": False,
            "reason": f"frozen_hybrid_failed:{type(error).__name__}",
        }
    if not bool(score.get("score_passed")):
        rejection_reasons.append("frozen_hybrid_score_improvement_below_limit")
    if not bool(score.get("membership_passed")):
        rejection_reasons.append("frozen_hybrid_changed_established_memberships")
    if score.get("reason"):
        rejection_reasons.append(str(score["reason"]))

    geometry_fields = {
        "trajectory_v22_source_group": geometry.get("source_group"),
        "trajectory_v22_source_row_member_count_before": geometry.get(
            "source_row_member_count_before"
        ),
        "trajectory_v22_source_row_member_count_after": geometry.get(
            "source_row_member_count_after"
        ),
        "trajectory_v22_local_piecewise_residual_um": geometry.get(
            "local_piecewise_residual_um"
        ),
        "trajectory_v22_local_piecewise_residual_width_factor": geometry.get(
            "local_piecewise_residual_width_factor"
        ),
        "trajectory_v22_shared_line_residual_um": geometry.get(
            "shared_line_residual_um"
        ),
        "trajectory_v22_shared_line_residual_width_factor": geometry.get(
            "shared_line_residual_width_factor"
        ),
        "trajectory_v22_remaining_row_rms_um": geometry.get(
            "remaining_row_rms_um"
        ),
        "trajectory_v22_remaining_row_rms_width_factor": geometry.get(
            "remaining_row_rms_width_factor"
        ),
        "trajectory_v22_residual_dominance_ratio": geometry.get(
            "residual_dominance_ratio"
        ),
        "trajectory_v22_longitudinally_bracketed": geometry.get(
            "longitudinally_bracketed"
        ),
        "trajectory_v22_bracketing_interval_um": geometry.get(
            "bracketing_interval_um"
        ),
        "trajectory_v22_typical_source_row_gap_um": geometry.get(
            "typical_source_row_gap_um"
        ),
        "trajectory_v22_bracketing_gap_ratio": geometry.get(
            "bracketing_gap_ratio"
        ),
        "trajectory_v22_nearest_row_separation_um": geometry.get(
            "nearest_row_separation_um"
        ),
        "trajectory_v22_nearest_row_separation_width_factor": geometry.get(
            "nearest_row_separation_width_factor"
        ),
        "trajectory_v22_nearest_other_established_row_um": geometry.get(
            "nearest_other_established_row_um"
        ),
        "trajectory_v22_nearest_other_row_longitudinal_offset_um": geometry.get(
            "nearest_other_row_longitudinal_offset_um"
        ),
        "trajectory_v22_matched_longitudinal_lane_support": geometry.get(
            "matched_longitudinal_lane_support"
        ),
        "trajectory_v22_source_group_is_nearest": geometry.get(
            "source_group_is_nearest"
        ),
        "trajectory_v22_within_existing_band_route_passed": geometry.get(
            "within_existing_band_route_passed"
        ),
        "trajectory_v22_nominal_geometry_passed": geometry.get(
            "geometry_passed"
        ),
    }
    baseline_fields = {
        "trajectory_v22_nominal_loo_row_count": loo_baseline.get("row_count"),
        "trajectory_v22_nominal_loo_comembership_agreement": loo_baseline.get(
            "comembership_agreement"
        ),
        "trajectory_v22_nominal_loo_slope_difference_deg": loo_baseline.get(
            "slope_difference_deg"
        ),
        "trajectory_v22_nominal_loo_baseline_passed": loo_baseline.get(
            "passed"
        ),
    }
    score_fields = {
        "trajectory_v22_hybrid_fallback_adjusted_score": score.get(
            "fallback_adjusted_score"
        ),
        "trajectory_v22_hybrid_adjusted_score": score.get(
            "hybrid_adjusted_score"
        ),
        "trajectory_v22_hybrid_adjusted_score_improvement": score.get(
            "adjusted_score_improvement"
        ),
        "trajectory_v22_final_hybrid_comembership_agreement": score.get(
            "final_comembership_agreement"
        ),
        "trajectory_v22_nominal_score_and_membership_passed": score.get(
            "passed"
        ),
    }
    nominal_passed = not rejection_reasons
    common_fields.update(
        {
            **geometry_fields,
            **baseline_fields,
            **score_fields,
            "trajectory_v22_leave_one_out_status": (
                "nominal_candidate_passed_pending_stability"
                if nominal_passed
                else "rejected_nominal_safeguards"
            ),
            "trajectory_v22_leave_one_out_reason": ";".join(
                dict.fromkeys(rejection_reasons)
            ),
            "trajectory_v22_nominal_gate_passed": nominal_passed,
        }
    )
    return (
        common_fields,
        score.get("hybrid") if nominal_passed else None,
        score.get("hybrid_diagnostic") if nominal_passed else None,
    )


def _v22_leave_one_out_stability(
    point_index: int,
    fallback: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV20,
) -> dict[str, Any]:
    """Perturb the n-1 baseline and frozen hybrid without freeing memberships."""

    iterations = int(settings.trajectory_stability_iterations)
    point_indices = [int(value) for value in fallback.get("point_indices", [])]
    seed_offset = sum(
        (position + 1) * (index + 1)
        for position, index in enumerate(point_indices)
    ) + 1009 * (int(point_index) + 1) + 22003
    rng = np.random.default_rng(
        int(settings.trajectory_stability_seed) + seed_offset
    )
    jitter_um = (
        float(settings.trajectory_stability_jitter_width_fraction)
        * float(median_detector_width_um)
    )
    fallback_k = int(fallback["k"])
    row_count_votes = 0
    assignment_votes = 0
    geometry_votes = 0
    score_votes = 0
    route_votes = 0
    joint_votes = 0
    stability_counts: dict[int, int] = {}

    for _ in range(iterations):
        jittered = np.asarray(lateral, dtype=float).copy()
        jittered[point_indices] += rng.normal(0.0, jitter_um, len(point_indices))
        separation_scale = float(rng.uniform(0.90, 1.10))
        baseline_passed = False
        try:
            loo_baseline = _v22_fit_leave_one_out_baseline(
                point_index,
                fallback,
                jittered,
                longitudinal,
                median_detector_width_um,
                settings,
                separation_scale=separation_scale,
            )
            perturbed_k = base.optional_int(loo_baseline.get("row_count"))
            if perturbed_k is not None:
                stability_counts[int(perturbed_k)] = (
                    stability_counts.get(int(perturbed_k), 0) + 1
                )
            if perturbed_k == fallback_k:
                row_count_votes += 1
            if bool(loo_baseline.get("passed")):
                assignment_votes += 1
                baseline_passed = True
        except (RuntimeError, ValueError):
            pass

        geometry = _v22_leave_one_out_geometry(
            point_index,
            fallback,
            jittered,
            longitudinal,
            median_detector_width_um,
            settings,
        )
        geometry_passed = bool(geometry.get("geometry_passed"))
        route_passed = bool(
            geometry.get("within_existing_band_route_passed")
        )
        if geometry_passed:
            geometry_votes += 1
        if route_passed:
            route_votes += 1
        score_passed = False
        try:
            scored = _v22_score_frozen_hybrid(
                point_index,
                fallback,
                jittered,
                longitudinal,
                median_detector_width_um,
                settings,
            )
            if bool(scored.get("passed")):
                score_votes += 1
                score_passed = True
        except (RuntimeError, ValueError):
            pass
        if baseline_passed and geometry_passed and route_passed and score_passed:
            joint_votes += 1

    row_count_fraction = row_count_votes / iterations
    assignment_fraction = assignment_votes / iterations
    geometry_fraction = geometry_votes / iterations
    score_fraction = score_votes / iterations
    route_fraction = route_votes / iterations
    joint_fraction = joint_votes / iterations
    passed = bool(
        row_count_fraction
        >= float(settings.v22_loo_min_row_count_stability_fraction)
        and assignment_fraction
        >= float(settings.v22_loo_min_assignment_stability_fraction)
        and geometry_fraction
        >= float(settings.v22_loo_min_geometry_stability_fraction)
        and score_fraction
        >= float(settings.v22_loo_min_score_stability_fraction)
        and route_fraction
        >= float(settings.v22_loo_min_route_stability_fraction)
        and joint_fraction
        >= float(settings.v22_loo_min_joint_stability_fraction)
    )
    return {
        "trajectory_v22_stability_iterations": iterations,
        "trajectory_v22_loo_row_count_stability_fraction": row_count_fraction,
        "trajectory_v22_loo_assignment_stability_fraction": (
            assignment_fraction
        ),
        "trajectory_v22_geometry_stability_fraction": geometry_fraction,
        "trajectory_v22_score_stability_fraction": score_fraction,
        "trajectory_v22_route_stability_fraction": route_fraction,
        "trajectory_v22_joint_gate_stability_fraction": joint_fraction,
        "trajectory_v22_stability_gate_passed": passed,
        "trajectory_v22_stability_count_distribution": stability_counts,
        "trajectory_v22_stability_role": (
            "fixed_candidate_n_minus_one_baseline_and_frozen_membership_"
            "hybrid_perturbation_not_unconstrained_full_model_reperturbation"
        ),
    }


def _apply_v22_leave_one_out_rescue(
    chosen: dict[str, Any],
    candidates: list[dict[str, Any]],
    diagnostic: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV20,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Add one bracketed singleton without changing any established row."""

    if not bool(settings.v22_leave_one_out_singleton_rescue_enabled):
        fields = {
            "trajectory_v22_leave_one_out_status": "disabled",
            "trajectory_v22_leave_one_out_reason": "policy_disabled_in_settings",
        }
        chosen.update(fields)
        diagnostic.update(fields)
        return chosen, candidates, diagnostic
    if bool(chosen.get("trajectory_v20_singleton_rescue_applied")):
        fields = {
            "trajectory_v22_leave_one_out_status": (
                "not_applied_existing_singleton_rescue"
            ),
            "trajectory_v22_leave_one_out_reason": (
                "v20_or_v21_already_added_a_singleton"
            ),
        }
        chosen.update(fields)
        diagnostic.update(fields)
        return chosen, candidates, diagnostic
    if not bool(diagnostic.get("trajectory_v20_singleton_candidate")):
        return chosen, candidates, diagnostic

    nominal_fields, hybrid, hybrid_diagnostic = (
        _v22_nominal_leave_one_out_evaluation(
            chosen,
            diagnostic,
            lateral,
            longitudinal,
            median_detector_width_um,
            settings,
        )
    )
    chosen.update(nominal_fields)
    diagnostic.update(nominal_fields)
    if hybrid is None or hybrid_diagnostic is None:
        return chosen, candidates, diagnostic

    point_index = int(nominal_fields["trajectory_v22_candidate_point_index"])
    stability_fields = _v22_leave_one_out_stability(
        point_index,
        chosen,
        lateral,
        longitudinal,
        median_detector_width_um,
        settings,
    )
    chosen.update(stability_fields)
    diagnostic.update(stability_fields)
    if not bool(stability_fields["trajectory_v22_stability_gate_passed"]):
        rejection_fields = {
            "trajectory_v22_leave_one_out_applied": False,
            "trajectory_v22_leave_one_out_status": (
                "rejected_perturbation_stability"
            ),
            "trajectory_v22_leave_one_out_reason": (
                "n_minus_one_or_candidate_specific_stability_below_limit"
            ),
        }
        chosen.update(rejection_fields)
        diagnostic.update(rejection_fields)
        return chosen, candidates, diagnostic

    fallback_k = int(chosen["k"])
    hybrid_k = int(hybrid["k"])
    confidence = nominal_fields.get(
        "trajectory_v22_candidate_detector_confidence"
    )
    final_agreement = float(
        nominal_fields["trajectory_v22_final_hybrid_comembership_agreement"]
    )
    hybrid_improvement = float(
        nominal_fields["trajectory_v22_hybrid_adjusted_score_improvement"]
    )
    delta_bic_vs_conservative = float(chosen["bic"]) - float(hybrid["bic"])
    trajectory_maximum_k_reached = bool(
        hybrid_k == int(settings.trajectory_max_rows_per_supergroup)
        and len(hybrid.get("point_indices", []))
        > int(settings.trajectory_max_rows_per_supergroup)
    )
    legacy_row_count = base.optional_int(
        diagnostic.get("legacy_sequential_row_count")
    )
    selection_reason = (
        str(chosen.get("automatic_selection_reason") or "")
        + "__v22_leave_one_out_membership_preserving_singleton_rescue"
    )
    accepted_fields = {
        **nominal_fields,
        **stability_fields,
        "automatic_row_count": hybrid_k,
        "automatic_selection_reason": selection_reason,
        "delta_bic_vs_simpler": delta_bic_vs_conservative,
        "row_count_selection_reason": (
            "v22_leave_one_out_membership_preserving_singleton_rescue"
        ),
        "sparse_row_support": True,
        "model_confidence": "low",
        "trajectory_slope_angle_deg": math.degrees(
            math.atan(float(hybrid["slope"]))
        ),
        "trajectory_singleton_validation_required": True,
        "trajectory_singleton_rows_validated": True,
        "trajectory_singleton_detector_confidences": [confidence],
        "trajectory_singleton_local_edge_support_ratios": [],
        "trajectory_singleton_validation_failure_reason": "",
        "trajectory_exact_row_traits_eligible": False,
        "trajectory_maximum_k_reached": trajectory_maximum_k_reached,
        "trajectory_unresolved_sparse_third_trajectory": False,
        "trajectory_pre_validation_row_count": hybrid_k,
        "trajectory_post_validation_row_count": hybrid_k,
        "trajectory_validation_fallback_applied": False,
        "trajectory_validation_fallback_reason": "",
        "trajectory_displayed_non_singleton_row_count": fallback_k,
        "trajectory_displayed_fallback_row_count": fallback_k,
        "trajectory_row_count_min": fallback_k,
        "trajectory_row_count_max": hybrid_k,
        "trajectory_row_count_range_reason": (
            "v22_leave_one_out_singleton_rescue_provisional"
        ),
        "trajectory_possible_sparse_row": False,
        "trajectory_possible_sparse_additional_row_count": 0,
        # The explicit v2.2 fields describe the constrained perturbation.  The
        # generic fields stay empty so they cannot be mistaken for a complete
        # unrestricted-model re-perturbation.
        "trajectory_row_count_stability_fraction": None,
        "trajectory_assignment_stability_fraction": None,
        "trajectory_conditional_row_count_sensitivity_fraction": None,
        "trajectory_conditional_assignment_sensitivity_fraction": None,
        "trajectory_row_count_stable": False,
        "trajectory_stability_count_distribution": {},
        "trajectory_v20_singleton_rescue_applied": True,
        "trajectory_v20_singleton_validation_status": "accepted_provisional",
        "trajectory_v20_singleton_validation_reason": (
            "v22_leave_one_out_constrained_hybrid_passed"
        ),
        "trajectory_v20_conservative_row_count": fallback_k,
        "trajectory_v20_augmented_row_count": hybrid_k,
        "trajectory_v20_added_row_count": 1,
        "trajectory_v20_hybrid_adjusted_score_improvement": (
            hybrid_improvement
        ),
        "trajectory_v20_singleton_route": "within_existing_band",
        "trajectory_v20_singleton_band_boundary_ambiguous": False,
        "trajectory_v20_orientation_inherited": True,
        "trajectory_v20_orientation_independently_estimable": False,
        "trajectory_v20_traits_involving_singleton_provisional": True,
        "trajectory_v22_augmented_row_count": hybrid_k,
        "trajectory_v22_leave_one_out_applied": True,
        "trajectory_v22_leave_one_out_status": "accepted_provisional",
        "trajectory_v22_leave_one_out_reason": (
            "n_minus_one_baseline_stable_and_frozen_k_plus_one_hybrid_passed"
        ),
        "trajectory_v22_adds_new_band": False,
        "trajectory_v22_nonrecursive_single_pass": True,
    }

    rescued = dict(chosen)
    rescued.update(hybrid)
    rescued.update(accepted_fields)

    updated_candidates: list[dict[str, Any]] = []
    candidate_replaced = False
    for candidate in candidates:
        if int(candidate.get("k", -1)) == hybrid_k and not candidate_replaced:
            updated_candidates.append(hybrid)
            candidate_replaced = True
        else:
            updated_candidates.append(candidate)
    if not candidate_replaced:
        updated_candidates.append(hybrid)
        updated_candidates.sort(key=lambda item: int(item["k"]))

    updated_candidate_diagnostics: list[dict[str, Any]] = []
    diagnostic_replaced = False
    for item in diagnostic.get("candidate_diagnostics", []):
        if (
            base.optional_int(item.get("candidate_row_count")) == hybrid_k
            and not diagnostic_replaced
        ):
            updated_candidate_diagnostics.append(hybrid_diagnostic)
            diagnostic_replaced = True
        else:
            updated_candidate_diagnostics.append(item)
    if not diagnostic_replaced:
        updated_candidate_diagnostics.append(hybrid_diagnostic)
        updated_candidate_diagnostics.sort(
            key=lambda item: int(item["candidate_row_count"])
        )

    diagnostic.update(
        {
            **accepted_fields,
            "global_automatic_row_count": hybrid_k,
            "row_count_used_for_traits": hybrid_k,
            "row_count_used_for_traits_source": (
                "v22_leave_one_out_membership_preserving_rescue"
            ),
            "alternative_row_count": fallback_k,
            "global_row_selection_score": float(
                hybrid_diagnostic["adjusted_score"]
            ),
            "global_row_score_margin": max(0.0, hybrid_improvement),
            "global_row_raw_margin": hybrid_improvement,
            "global_row_automatic_is_strict_best": False,
            "global_row_model_confidence": "low",
            "global_row_count_changed_from_legacy": bool(
                legacy_row_count is not None and hybrid_k != legacy_row_count
            ),
            "potential_row_undersplit_in_legacy": bool(
                legacy_row_count is not None and hybrid_k > legacy_row_count
            ),
            "potential_row_oversplit_in_legacy": bool(
                legacy_row_count is not None and hybrid_k < legacy_row_count
            ),
            "singleton_supported_row_count": 1,
            "singleton_detector_confidence_mean": confidence,
            "candidate_diagnostics": updated_candidate_diagnostics,
        }
    )
    return rescued, updated_candidates, diagnostic


def select_row_model_v20(
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV20,
    audited_row_count: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Use v1.5 stability machinery with the guarded v2.2 chooser."""

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
            "v2.2 expected exactly one provisional-band diagnostic record."
        )
    diagnostic = v14._ACTIVE_BAND_DIAGNOSTICS[-1]

    # An explicit audited count remains authoritative.  Neither automatic
    # singleton route may replace a user-supplied row-count override.
    if audited_row_count is None:
        chosen, candidates, diagnostic = _apply_v21_post_stability_rescue(
            chosen,
            candidates,
            diagnostic,
            lateral,
            longitudinal,
            median_detector_width_um,
            settings,
        )
        chosen, candidates, diagnostic = _apply_v22_leave_one_out_rescue(
            chosen,
            candidates,
            diagnostic,
            lateral,
            longitudinal,
            median_detector_width_um,
            settings,
        )
    else:
        if bool(
            diagnostic.get("trajectory_v21_membership_relaxation_candidate")
        ):
            override_fields = {
                "trajectory_v21_membership_relaxation_applied": False,
                "trajectory_v21_membership_relaxation_status": (
                    "not_applied_audited_row_count_override"
                ),
                "trajectory_v21_membership_relaxation_reason": (
                    "explicit_audited_row_count_remains_authoritative"
                ),
            }
            chosen.update(override_fields)
            diagnostic.update(override_fields)
        if bool(diagnostic.get("trajectory_v20_singleton_candidate")):
            v22_override_fields = {
                "trajectory_v22_leave_one_out_applied": False,
                "trajectory_v22_leave_one_out_status": (
                    "not_applied_audited_row_count_override"
                ),
                "trajectory_v22_leave_one_out_reason": (
                    "explicit_audited_row_count_remains_authoritative"
                ),
            }
            chosen.update(v22_override_fields)
            diagnostic.update(v22_override_fields)

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
    """Propagate all v2.2 decisions to band, row and stoma-level tables."""

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
            bool(row.get("trajectory_v20_singleton_rescue_applied"))
            or "__v20_validated_singleton_rescue" in reason
            or "__v20_validated_singleton_only_band" in reason
            or "__v21_post_stability_membership_preserving_singleton_rescue"
            in reason
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
                        "v1.5_matched_gap_plus_v2.2_validated_singleton_strong_gap"
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
        raise RuntimeError("v2.2 invariant failed: every stoma must be assigned once.")
    row_ids = [str(row.get("row_id")) for row in rows]
    band_ids = [str(band.get("band_id")) for band in bands]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("v2.2 invariant failed: duplicate row IDs.")
    if len(band_ids) != len(set(band_ids)):
        raise RuntimeError("v2.2 invariant failed: duplicate band IDs.")

    row_lookup = {str(row["row_id"]): row for row in rows}
    row_members: list[int] = []
    for row in rows:
        members = [int(value) for value in row.get("_point_indices", [])]
        if not members:
            raise RuntimeError("v2.2 invariant failed: empty row.")
        row_members.extend(members)
        if (base.optional_int(row.get("stoma_count")) or 0) != len(members):
            raise RuntimeError("v2.2 invariant failed: row stoma count mismatch.")
    if sorted(row_members) != sorted(expected_points):
        raise RuntimeError("v2.2 invariant failed: row memberships do not partition stomata.")

    linked_row_ids: list[str] = []
    for expected_order, band in enumerate(bands, start=1):
        if base.optional_int(band.get("band_order_left_to_right")) != expected_order:
            raise RuntimeError("v2.2 invariant failed: non-contiguous band order.")
        linked = [str(value) for value in band.get("row_ids", [])]
        if any(row_id not in row_lookup for row_id in linked):
            raise RuntimeError("v2.2 invariant failed: band links unknown row.")
        linked_row_ids.extend(linked)
        declared = base.optional_int(band.get("row_count_used_for_traits"))
        if declared is None:
            declared = base.optional_int(band.get("row_count"))
        if declared != len(linked):
            raise RuntimeError("v2.2 invariant failed: band row-count mismatch.")
        members = sorted(
            {
                int(index)
                for row_id in linked
                for index in row_lookup[row_id].get("_point_indices", [])
            }
        )
        if members != sorted(int(value) for value in band.get("_point_indices", [])):
            raise RuntimeError("v2.2 invariant failed: band member union mismatch.")
    if sorted(linked_row_ids) != sorted(row_ids):
        raise RuntimeError("v2.2 invariant failed: each row must belong to one band.")

    applied_candidate_indices: set[int] = set()
    for row in rows:
        if not bool(row.get("trajectory_v22_leave_one_out_applied")):
            if bool(row.get("v22_rescued_singleton_row")):
                raise RuntimeError(
                    "v2.2 invariant failed: rescued row lacks applied provenance."
                )
            continue
        candidate_index = base.optional_int(
            row.get("trajectory_v22_candidate_point_index")
        )
        if candidate_index is None:
            raise RuntimeError(
                "v2.2 invariant failed: applied group lacks candidate index."
            )
        applied_candidate_indices.add(candidate_index)
        if bool(row.get("v22_rescued_singleton_row")):
            members = [int(value) for value in row.get("_point_indices", [])]
            if members != [candidate_index]:
                raise RuntimeError(
                    "v2.2 invariant failed: rescued singleton membership mismatch."
                )
            if bool(row.get("v20_singleton_band")):
                raise RuntimeError(
                    "v2.2 invariant failed: leave-one-out rescue created a band."
                )

    for candidate_index in applied_candidate_indices:
        if candidate_index not in assignments:
            raise RuntimeError(
                "v2.2 invariant failed: candidate lacks point assignment."
            )
        if not bool(assignments[candidate_index].get("v22_candidate_stoma")):
            raise RuntimeError(
                "v2.2 invariant failed: candidate-specific assignment flag missing."
            )
        if not bool(
            assignments[candidate_index].get("v22_rescued_singleton_row")
        ):
            raise RuntimeError(
                "v2.2 invariant failed: candidate not linked to rescued row."
            )
        matching_rows = [
            row
            for row in rows
            if bool(row.get("v22_rescued_singleton_row"))
            and [int(value) for value in row.get("_point_indices", [])]
            == [candidate_index]
        ]
        if len(matching_rows) != 1:
            raise RuntimeError(
                "v2.2 invariant failed: candidate must map to one singleton row."
            )


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
    # Replace that inherited label so every v2.2 table has truthful provenance.
    for row in rows:
        row["row_model_version"] = SCRIPT_VERSION
    for band in bands:
        band["row_model_version"] = SCRIPT_VERSION
    for assignment in assignments.values():
        assignment["row_model_version"] = SCRIPT_VERSION

    row_lookup = {str(row["row_id"]): row for row in rows}
    for row in rows:
        members = [int(value) for value in row.get("_point_indices", [])]
        v22_candidate_index = base.optional_int(
            row.get("trajectory_v22_candidate_point_index")
        )
        v22_candidate_in_row = bool(
            v22_candidate_index is not None
            and v22_candidate_index in members
        )
        accepted_singleton_row = bool(
            (base.optional_int(row.get("stoma_count")) or 0) == 1
            and bool(row.get("trajectory_v20_singleton_rescue_applied"))
        )
        v22_rescued_singleton_row = bool(
            accepted_singleton_row
            and v22_candidate_in_row
            and bool(row.get("trajectory_v22_leave_one_out_applied"))
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
                "v22_candidate_stoma_in_row": v22_candidate_in_row,
                "v22_rescued_singleton_row": v22_rescued_singleton_row,
            }
        )
        for point_index in members:
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
                    "v22_candidate_stoma": bool(
                        v22_candidate_index is not None
                        and int(point_index) == v22_candidate_index
                    ),
                    "v22_rescued_singleton_row": (
                        v22_rescued_singleton_row
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
        band["v22_leave_one_out_singleton_present"] = any(
            bool(row.get("v22_rescued_singleton_row")) for row in linked_rows
        )
        band["v22_leave_one_out_singleton_row_count"] = sum(
            bool(row.get("v22_rescued_singleton_row")) for row in linked_rows
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
        and bool(
            item.get("trajectory_row_count_stable")
            or (
                item.get("trajectory_v21_membership_relaxation_applied")
                and item.get("trajectory_v21_parent_v20_stability_gate_passed")
            )
            or (
                item.get("trajectory_v22_leave_one_out_applied")
                and item.get("trajectory_v22_stability_gate_passed")
            )
        )
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
    relaxation_candidate_group_count = sum(
        bool(item.get("trajectory_v21_membership_relaxation_candidate"))
        for item in unique_diagnostics
    )
    relaxed_group_count = sum(
        bool(item.get("trajectory_v21_membership_relaxation_applied"))
        for item in unique_diagnostics
    )
    relaxed_stability_passed_count = sum(
        bool(item.get("trajectory_v21_membership_relaxation_applied"))
        and bool(
            item.get("trajectory_v21_parent_v20_stability_gate_passed")
        )
        for item in unique_diagnostics
    )
    relaxed_stability_flagged_count = sum(
        bool(item.get("trajectory_v21_membership_relaxation_candidate"))
        and item.get("trajectory_v21_membership_relaxation_status")
        == "rejected_parent_v20_stability_gate"
        for item in unique_diagnostics
    )
    v22_candidate_group_count = sum(
        bool(item.get("trajectory_v22_leave_one_out_candidate"))
        for item in unique_diagnostics
    )
    v22_nominal_passed_group_count = sum(
        bool(item.get("trajectory_v22_nominal_gate_passed"))
        for item in unique_diagnostics
    )
    v22_stability_passed_group_count = sum(
        bool(item.get("trajectory_v22_stability_gate_passed"))
        for item in unique_diagnostics
    )
    v22_accepted_group_count = sum(
        bool(item.get("trajectory_v22_leave_one_out_applied"))
        for item in unique_diagnostics
    )
    v22_rejected_nominal_group_count = sum(
        item.get("trajectory_v22_leave_one_out_status")
        in {"rejected_nominal_safeguards", "rejected_ambiguous_nomination"}
        for item in unique_diagnostics
    )
    v22_rejected_stability_group_count = sum(
        item.get("trajectory_v22_leave_one_out_status")
        == "rejected_perturbation_stability"
        for item in unique_diagnostics
    )
    audit.update(
        {
            "row_model_version": SCRIPT_VERSION,
            "trajectory_singleton_policy_version_current": SCRIPT_VERSION,
            "trajectory_v20_singleton_policy_version": SINGLETON_POLICY_VERSION,
            "trajectory_v20_singleton_policy_version_semantics": (
                "legacy_compatibility_field_carries_current_umbrella_"
                "singleton_policy"
            ),
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
            "trajectory_v21_policy_version": V21_POLICY_VERSION,
            "trajectory_v21_membership_relaxation_candidate_group_count": (
                relaxation_candidate_group_count
            ),
            "trajectory_v21_membership_relaxed_group_count": (
                relaxed_group_count
            ),
            "trajectory_v21_membership_relaxed_stability_passed_group_count": (
                relaxed_stability_passed_count
            ),
            "trajectory_v21_membership_relaxed_stability_flagged_group_count": (
                relaxed_stability_flagged_count
            ),
            "trajectory_v21_membership_relaxation_rejected_by_parent_v20_stability_group_count": (
                relaxed_stability_flagged_count
            ),
            "trajectory_v21_membership_relaxation_policy": (
                "permit only a one-row one-singleton parent candidate whose "
                "unconstrained non-singleton comembership is at least two "
                "thirds; require the unchanged parent v2.0 solution to pass "
                "the established perturbation-stability thresholds; export "
                "only a constrained hybrid with complete preservation of "
                "established memberships; do not claim that the final hybrid "
                "was independently re-perturbed"
            ),
            "trajectory_v22_policy_version": SCRIPT_VERSION,
            "trajectory_v22_leave_one_out_candidate_group_count": (
                v22_candidate_group_count
            ),
            "trajectory_v22_leave_one_out_nominal_passed_group_count": (
                v22_nominal_passed_group_count
            ),
            "trajectory_v22_leave_one_out_stability_passed_group_count": (
                v22_stability_passed_group_count
            ),
            "trajectory_v22_leave_one_out_accepted_group_count": (
                v22_accepted_group_count
            ),
            "trajectory_v22_leave_one_out_rejected_nominal_group_count": (
                v22_rejected_nominal_group_count
            ),
            "trajectory_v22_leave_one_out_rejected_stability_group_count": (
                v22_rejected_stability_group_count
            ),
            "trajectory_v22_leave_one_out_policy": (
                "use only the sole singleton nominated by the guarded parent "
                "model; freeze all conservative memberships; remove that point "
                "once; require a smooth supported n-minus-one row, local and "
                "shared lateral separation, longitudinal bracketing, a normal "
                "within-band gap, adjusted-score improvement, exact final "
                "comembership and deterministic candidate-specific stability; "
                "add at most one row and never create a band"
            ),
            "trajectory_v20_singleton_trait_policy": (
                "accepted singleton rows contribute to counts/densities and "
                "rows-per-band; orientation is borrowed; one-point within-row "
                "spacing, continuity, alternation and phase are undefined"
            ),
            "trajectory_sobel_policy": (
                "generic raw-edge contrast is not an acceptance gate for a "
                "singleton row or band; v2.2 uses detector confidence, fitted "
                "trajectory geometry, local bend/continuation checks and "
                "membership-preserving model comparison"
            ),
            "automatic_row_count_policy": (
                "start from the unchanged v1.9 non-singleton solution; allow a "
                "biologically possible one-stoma row or one-stoma band only "
                "when the fixed v2.1 safeguards or the stricter v2.2 "
                "leave-one-out safeguards pass; preserve all established "
                "non-singleton memberships and add at most one singleton row "
                "per multi-point provisional group without recursion"
            ),
            "trajectory_stability_policy": (
                "repeat the complete guarded selector after deterministic "
                "lateral jitter and minimum-separation perturbation; report "
                "stability as QC for the strict v2.0 route; for the new v2.1 "
                "membership-relaxed route, require parent-v2.0 stability in "
                "addition to every detector, geometry, terminal, score and "
                "membership safeguard; v2.2 separately perturbs the n-minus-one "
                "baseline and frozen-membership hybrid and records those results "
                "without claiming unrestricted full-model re-perturbation; a "
                "joint per-iteration gate additionally requires the same "
                "perturbation to pass baseline, geometry, route and score checks"
            ),
            "trajectory_stability_limitations": (
                "accepted singleton rows and bands retain explicit stability "
                "QC after the representative 40-image biological audit; "
                "detection-dropout and longitudinal-window perturbations are "
                "not implemented"
            ),
            "trajectory_row_band_trait_policy": (
                "v2.2 retains both conservative v1.9 and singleton-augmented "
                "topology counts; accepted singleton observations can support "
                "row/band counts but cannot independently estimate orientation, "
                "spacing, continuity, alternation or phase"
            ),
        }
    )
    return rows, bands, assignments, audit


_V20_IMAGE_FIELDS = (
    "trajectory_singleton_policy_version_current",
    "trajectory_v20_singleton_policy_version",
    "trajectory_v20_singleton_policy_version_semantics",
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
    "trajectory_v21_policy_version",
    "trajectory_v21_membership_relaxation_candidate_group_count",
    "trajectory_v21_membership_relaxed_group_count",
    "trajectory_v21_membership_relaxed_stability_passed_group_count",
    "trajectory_v21_membership_relaxed_stability_flagged_group_count",
    "trajectory_v21_membership_relaxation_rejected_by_parent_v20_stability_group_count",
    "trajectory_v21_membership_relaxation_policy",
    "trajectory_v22_policy_version",
    "trajectory_v22_leave_one_out_candidate_group_count",
    "trajectory_v22_leave_one_out_nominal_passed_group_count",
    "trajectory_v22_leave_one_out_stability_passed_group_count",
    "trajectory_v22_leave_one_out_accepted_group_count",
    "trajectory_v22_leave_one_out_rejected_nominal_group_count",
    "trajectory_v22_leave_one_out_rejected_stability_group_count",
    "trajectory_v22_leave_one_out_policy",
    "stomatal_row_count_conservative_v19",
    "stomatal_row_count_augmented_v20",
    "stomatal_row_count_added_by_v20",
    "stomatal_band_count_conservative_v19",
    "stomatal_band_count_augmented_v20",
    "stomatal_band_count_added_by_v20",
    "detected_row_count_conservative_v19",
    "detected_row_count_augmented_v21",
    "detected_row_count_added_by_v21",
    "detected_row_count_augmented_through_v21",
    "detected_row_count_total_added_through_v21",
    "detected_row_count_augmented_v22",
    "detected_row_count_added_by_v22",
    "detected_band_count_conservative_v19",
    "detected_band_count_augmented_v21",
    "detected_band_count_added_by_v21",
    "detected_band_count_augmented_through_v21",
    "detected_band_count_total_added_through_v21",
    "detected_band_count_augmented_v22",
    "detected_band_count_added_by_v22",
    "row_band_v20_qc_status",
    "row_band_v20_qc_flag",
    "row_band_v20_qc_reason",
    "row_band_v20_value_interpretation",
    "row_band_v21_qc_status",
    "row_band_v21_qc_flag",
    "row_band_v21_qc_reason",
    "row_band_v21_value_interpretation",
    "row_band_v22_qc_status",
    "row_band_v22_qc_flag",
    "row_band_v22_qc_reason",
    "row_band_v22_value_interpretation",
    "legacy_added_by_field_semantics",
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
    # These are raw detected topology counts.  The established unqualified
    # stomatal_row_count/stomatal_band_count fields are spatially eligible
    # phenotype counts, so v2.2 uses explicit ``detected_*`` names here.
    augmented_rows = len(row_records)
    augmented_bands = len(band_records)
    added_rows_v22 = sum(
        bool(item.get("trajectory_v22_leave_one_out_applied"))
        for item in row_band_audit.get("band_selection_diagnostics", {}).values()
    )
    augmented_rows_before_v22 = augmented_rows - added_rows_v22
    accepted_groups = int(
        row_band_audit.get("trajectory_v20_singleton_accepted_group_count") or 0
    )
    qc_reasons: list[str] = []
    if accepted_groups:
        qc_reasons.append("singleton_rows_retain_explicit_provisional_qc")
    if added_rows_v22:
        qc_reasons.append("v22_leave_one_out_singleton_rows_are_provisional")
    if int(
        row_band_audit.get(
            "trajectory_v21_membership_relaxation_rejected_by_parent_v20_stability_group_count"
        )
        or 0
    ):
        qc_reasons.append(
            "v21_singleton_candidate_rejected_by_parent_v20_stability"
        )
    if int(
        row_band_audit.get("trajectory_v20_singleton_unresolved_group_count") or 0
    ):
        qc_reasons.append("unresolved_singleton_candidates_retained_conservatively")
    if int(
        row_band_audit.get("trajectory_v20_singleton_rejected_group_count") or 0
    ):
        qc_reasons.append("rejected_singleton_candidates_recorded")
    if int(
        row_band_audit.get(
            "trajectory_v22_leave_one_out_rejected_stability_group_count"
        )
        or 0
    ):
        qc_reasons.append("v22_candidate_rejected_by_specific_stability_gate")

    result.update(
        {
            key: row_band_audit.get(key)
            for key in _V20_IMAGE_FIELDS
            if key.startswith(
                (
                    "trajectory_singleton_",
                    "trajectory_v20_",
                    "trajectory_v21_",
                    "trajectory_v22_",
                )
            )
        }
    )
    result.update(
        {
            "stomatal_row_count_conservative_v19": augmented_rows - added_rows,
            "stomatal_row_count_augmented_v20": augmented_rows_before_v22,
            "stomatal_row_count_added_by_v20": added_rows - added_rows_v22,
            "stomatal_band_count_conservative_v19": augmented_bands - added_bands,
            "stomatal_band_count_augmented_v20": augmented_bands,
            "stomatal_band_count_added_by_v20": added_bands,
            "detected_row_count_conservative_v19": augmented_rows - added_rows,
            "detected_row_count_augmented_v21": augmented_rows_before_v22,
            "detected_row_count_added_by_v21": added_rows - added_rows_v22,
            "detected_row_count_augmented_through_v21": (
                augmented_rows_before_v22
            ),
            "detected_row_count_total_added_through_v21": (
                added_rows - added_rows_v22
            ),
            "detected_row_count_augmented_v22": augmented_rows,
            "detected_row_count_added_by_v22": added_rows_v22,
            "detected_band_count_conservative_v19": augmented_bands - added_bands,
            "detected_band_count_augmented_v21": augmented_bands,
            "detected_band_count_added_by_v21": added_bands,
            "detected_band_count_augmented_through_v21": augmented_bands,
            "detected_band_count_total_added_through_v21": added_bands,
            "detected_band_count_augmented_v22": augmented_bands,
            "detected_band_count_added_by_v22": 0,
            "row_band_v20_qc_status": (
                "flagged_provisional" if qc_reasons else "pass_no_singleton_change"
            ),
            "row_band_v20_qc_flag": bool(qc_reasons),
            "row_band_v20_qc_reason": ";".join(qc_reasons),
            "row_band_v20_value_interpretation": (
                "raw detected topology counts before spatial-eligibility "
                "filtering: v1.9 conservative and pre-v2.2 singleton-augmented "
                "values are retained side by side; singleton-derived values "
                "remain provisional until biological audit"
            ),
            "row_band_v21_qc_status": (
                "flagged_provisional" if qc_reasons else "pass_no_singleton_change"
            ),
            "row_band_v21_qc_flag": bool(qc_reasons),
            "row_band_v21_qc_reason": ";".join(qc_reasons),
            "row_band_v21_value_interpretation": (
                "explicit detected_* fields compare raw v1.9 topology with "
                "pre-v2.2 singleton-augmented topology; unqualified stomatal row "
                "and band counts remain spatially eligible image phenotypes"
            ),
            "row_band_v22_qc_status": (
                "flagged_provisional" if qc_reasons else "pass_no_singleton_change"
            ),
            "row_band_v22_qc_flag": bool(qc_reasons),
            "row_band_v22_qc_reason": ";".join(qc_reasons),
            "row_band_v22_value_interpretation": (
                "detected_row_count_augmented_v22 includes only v2.2 "
                "leave-one-out singleton rows that passed nominal and "
                "candidate-specific stability safeguards; unqualified row and "
                "band traits remain spatially eligible image phenotypes"
            ),
            "legacy_added_by_field_semantics": (
                "for backward table compatibility, fields named added_by_v20 "
                "or added_by_v21 are cumulative additions through that version, "
                "not additions uniquely attributable to that route; the "
                "explicit added_by_v22 field is route-specific"
            ),
            "row_geometry_model_version": SCRIPT_VERSION,
            "partial_band_eligibility_version": (
                PARTIAL_BAND_ELIGIBILITY_VERSION
            ),
            "trait_export_policy_version": TRAIT_EXPORT_POLICY_VERSION,
            "trajectory_row_band_trait_policy": (
                "v2.2 retains conservative v1.9, pre-v2.2 and final "
                "singleton-augmented raw "
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
    """Carry v2.2 audit fields into the compact GWAS-oriented table."""

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
                if key.startswith("trajectory_singleton_")
                or key.startswith("trajectory_v20_")
                or key.startswith("trajectory_v21_")
                or key.startswith("trajectory_v22_")
                or key.startswith("v20_")
                or key.startswith("v21_")
                or key.startswith("v22_")
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
    """Mark strict, v2.1 and v2.2 singleton routes distinctly."""

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
        v21_relaxed = bool(
            row.get("trajectory_v21_membership_relaxation_applied")
        )
        v22_leave_one_out = bool(
            row.get("trajectory_v22_leave_one_out_applied")
        )
        if v22_leave_one_out:
            color = (255, 255, 255)
            label = f"{row['row_id']} V2.2 LOO SINGLETON ROW"
        elif v21_relaxed:
            color = (255, 0, 255)
            label = (
                f"{row['row_id']} V2.1 RESCUED SINGLETON BAND"
                if singleton_band
                else f"{row['row_id']} V2.1 RESCUED SINGLETON ROW"
            )
        else:
            color = (0, 165, 255) if singleton_band else (255, 255, 0)
            label = (
                f"{row['row_id']} SINGLETON BAND"
                if singleton_band
                else f"{row['row_id']} SINGLETON ROW"
            )
        if v22_leave_one_out:
            # A black outer keyline keeps the white v2.2 marker visible over
            # both bright and dark DIC backgrounds.
            base.cv2.circle(
                image, (x, y), 27, (0, 0, 0), 6, base.cv2.LINE_AA
            )
        base.cv2.circle(image, (x, y), 24, color, 3, base.cv2.LINE_AA)
        base.cv2.line(image, (x - 20, y), (x + 20, y), color, 2, base.cv2.LINE_AA)
        base.cv2.line(image, (x, y - 20), (x, y + 20), color, 2, base.cv2.LINE_AA)
        text_y = max(22, y - 30)
        if v22_leave_one_out:
            base.cv2.putText(
                image,
                label,
                (max(5, x - 90), text_y),
                base.cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                5,
                base.cv2.LINE_AA,
            )
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
        raise RuntimeError(f"Failed to update v2.2 audit overlay: {output_path}")


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
    """Run the separate v2.2 audit candidate through the established engine."""

    install_v20_singleton_policy()
    return base.run_trait_extraction()


if __name__ == "__main__":
    run_trait_extraction()
