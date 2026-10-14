#!/usr/bin/env python3
"""Experimental v2.3 trait extractor with frozen-band singleton rescue.

This is a separate successor to v2.2.  It imports the complete v2.2 policy and
adds only the biologically audited corrections below.  It does not modify the
v2.2 script, pipeline inference results, masks, OME-TIFFs, or earlier trait
tables.

v2.3 retains the standard v2.2 bracketed leave-one-out route and adds:

1. A strict bracketed 3 -> 2 + 1 route.  It permits a nominated singleton to
   be removed from a three-stoma source row only when the remaining pair is
   exceptionally straight, the candidate is strongly separated and
   longitudinally bracketed, detector confidence is high, and the frozen
   membership score supports the additional row.
2. A strict terminal singleton route.  It permits a nominated terminal point
   from a row of at least four stomata only when a locally refitted remaining
   row is exceptionally smooth, the candidate is strongly separated, and its
   terminal longitudinal gap is consistent with ordinary row sampling.
3. A narrow two-row collapse.  A two-row solution is returned to one row only
   when both proposed rows have at least two stomata, their band-local lateral
   separation is strictly below 0.25 median detector-box widths, and the
   combined one-row fit remains smooth.
4. Frozen-band reconstruction.  A rescued singleton is kept with its nearest
   established source row, while any strong pre-existing external band
   boundary is restored.  The route cannot create an isolated singleton band.

All decisions are filename-independent and use detector coordinates and
confidence only.  Classifier, segmentor, raw intensity and Sobel evidence are
not acceptance gates.  The thresholds were developed on the representative
40-image biological audit and remain provisional until an independent audit.
Run this script only with a new output directory.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import extract_traits_from_pipeline_output_v2_2_leave_one_out_singleton_rescue as parent


base = parent.base
v14 = parent.v14
v15 = parent.v15
v16 = parent.v16
v19 = parent.v19

SCRIPT_VERSION = (
    "2.3.2-experimental-post-v22-frozen-bands-extended-singletons"
)
SINGLETON_POLICY_VERSION = SCRIPT_VERSION
ROW_GEOMETRY_MODEL_VERSION = SCRIPT_VERSION
PARTIAL_BAND_ELIGIBILITY_VERSION = parent.PARTIAL_BAND_ELIGIBILITY_VERSION
TRAIT_EXPORT_POLICY_VERSION = parent.TRAIT_EXPORT_POLICY_VERSION

_PARENT_V22_SCRIPT_PATH = Path(parent.__file__).resolve()

# Bracketed 3 -> 2 + 1 rescue.  The remaining pair defines a local segment;
# this route therefore uses substantially stronger separation and dominance
# requirements than the ordinary v2.2 route.
V23_BRACKETED_SMALL_MIN_SOURCE_MEMBERS = 3
V23_BRACKETED_SMALL_REMAINING_MEMBERS = 2
V23_BRACKETED_SMALL_MIN_CONFIDENCE = 0.80
V23_BRACKETED_SMALL_MIN_RESIDUAL_WIDTH_FACTOR = 0.65
V23_BRACKETED_SMALL_MAX_REMAINING_RMS_WIDTH_FACTOR = 0.05
V23_BRACKETED_SMALL_MIN_RESIDUAL_DOMINANCE = 10.0
V23_BRACKETED_SMALL_MIN_SCORE_IMPROVEMENT = 4.0

# Terminal route.  A locally refitted remaining row is used because a terminal
# outlier can bias the shared conservative slope.  The normal terminal-gap
# interval prevents a nearby continuation or a very remote isolated point from
# being promoted.
V23_TERMINAL_MIN_SOURCE_MEMBERS = 4
V23_TERMINAL_MIN_REMAINING_MEMBERS = 3
V23_TERMINAL_MIN_CONFIDENCE = 0.80
V23_TERMINAL_MIN_RESIDUAL_WIDTH_FACTOR = 0.60
V23_TERMINAL_MAX_RESIDUAL_WIDTH_FACTOR = 1.20
V23_TERMINAL_MAX_REMAINING_RMS_WIDTH_FACTOR = 0.05
V23_TERMINAL_MIN_RESIDUAL_DOMINANCE = 10.0
V23_TERMINAL_MIN_GAP_RATIO = 0.80
V23_TERMINAL_MAX_GAP_RATIO = 1.25
V23_TERMINAL_MAX_REFITTED_SLOPE_DEG = 7.0

# Candidate-specific perturbation stability.
V23_MIN_ROUTE_STABILITY_FRACTION = 0.80

# False two-row split protection.  The width is calculated from boxes in the
# provisional group, not from the complete image.
V23_NARROW_SPLIT_MAX_SEPARATION_WIDTH_FACTOR = 0.25
V23_NARROW_SPLIT_MAX_COMBINED_RMS_WIDTH_FACTOR = 0.15
V23_NARROW_SPLIT_MIN_MEMBERS_PER_PROPOSED_ROW = 2

# Frozen-band reconstruction uses the established strong-boundary definition.
V23_BAND_FREEZE_STRONG_GAP_WIDTH_FACTOR = 1.50
V23_BAND_FREEZE_MIN_STABILITY_FRACTION = 0.80


@dataclass(frozen=True)
class TraitSettingsV23(parent.TraitSettingsV20):
    """Append immutable v2.3 audit thresholds to the v2.2 settings."""

    v23_policy_version: str = SCRIPT_VERSION
    v23_parent_v22_script_sha256: str = ""
    v23_bracketed_small_min_source_members: int = (
        V23_BRACKETED_SMALL_MIN_SOURCE_MEMBERS
    )
    v23_bracketed_small_remaining_members: int = (
        V23_BRACKETED_SMALL_REMAINING_MEMBERS
    )
    v23_bracketed_small_min_confidence: float = (
        V23_BRACKETED_SMALL_MIN_CONFIDENCE
    )
    v23_bracketed_small_min_residual_width_factor: float = (
        V23_BRACKETED_SMALL_MIN_RESIDUAL_WIDTH_FACTOR
    )
    v23_bracketed_small_max_remaining_rms_width_factor: float = (
        V23_BRACKETED_SMALL_MAX_REMAINING_RMS_WIDTH_FACTOR
    )
    v23_bracketed_small_min_residual_dominance: float = (
        V23_BRACKETED_SMALL_MIN_RESIDUAL_DOMINANCE
    )
    v23_bracketed_small_min_score_improvement: float = (
        V23_BRACKETED_SMALL_MIN_SCORE_IMPROVEMENT
    )
    v23_terminal_min_source_members: int = V23_TERMINAL_MIN_SOURCE_MEMBERS
    v23_terminal_min_remaining_members: int = V23_TERMINAL_MIN_REMAINING_MEMBERS
    v23_terminal_min_confidence: float = V23_TERMINAL_MIN_CONFIDENCE
    v23_terminal_min_residual_width_factor: float = (
        V23_TERMINAL_MIN_RESIDUAL_WIDTH_FACTOR
    )
    v23_terminal_max_residual_width_factor: float = (
        V23_TERMINAL_MAX_RESIDUAL_WIDTH_FACTOR
    )
    v23_terminal_max_remaining_rms_width_factor: float = (
        V23_TERMINAL_MAX_REMAINING_RMS_WIDTH_FACTOR
    )
    v23_terminal_min_residual_dominance: float = (
        V23_TERMINAL_MIN_RESIDUAL_DOMINANCE
    )
    v23_terminal_min_gap_ratio: float = V23_TERMINAL_MIN_GAP_RATIO
    v23_terminal_max_gap_ratio: float = V23_TERMINAL_MAX_GAP_RATIO
    v23_terminal_max_refitted_slope_deg: float = (
        V23_TERMINAL_MAX_REFITTED_SLOPE_DEG
    )
    v23_min_route_stability_fraction: float = (
        V23_MIN_ROUTE_STABILITY_FRACTION
    )
    v23_narrow_split_max_separation_width_factor: float = (
        V23_NARROW_SPLIT_MAX_SEPARATION_WIDTH_FACTOR
    )
    v23_narrow_split_max_combined_rms_width_factor: float = (
        V23_NARROW_SPLIT_MAX_COMBINED_RMS_WIDTH_FACTOR
    )
    v23_narrow_split_min_members_per_proposed_row: int = (
        V23_NARROW_SPLIT_MIN_MEMBERS_PER_PROPOSED_ROW
    )
    v23_band_freeze_strong_gap_width_factor: float = (
        V23_BAND_FREEZE_STRONG_GAP_WIDTH_FACTOR
    )
    v23_band_freeze_min_stability_fraction: float = (
        V23_BAND_FREEZE_MIN_STABILITY_FRACTION
    )


_V23_DEFAULTS: dict[str, Any] = {
    "trajectory_v23_policy_version": SCRIPT_VERSION,
    "trajectory_v23_additional_singleton_candidate": False,
    "trajectory_v23_additional_singleton_applied": False,
    "trajectory_v23_rescue_route": "not_applicable",
    "trajectory_v23_rescue_status": "not_applicable",
    "trajectory_v23_rescue_reason": "",
    "trajectory_v23_candidate_point_index": None,
    "trajectory_v23_candidate_stoma_id": "",
    "trajectory_v23_source_row_member_count_before": None,
    "trajectory_v23_source_row_member_count_after": None,
    "trajectory_v23_local_detector_width_um": None,
    "trajectory_v23_candidate_residual_um": None,
    "trajectory_v23_candidate_residual_width_factor": None,
    "trajectory_v23_remaining_row_rms_um": None,
    "trajectory_v23_remaining_row_rms_width_factor": None,
    "trajectory_v23_residual_dominance_ratio": None,
    "trajectory_v23_longitudinally_bracketed": None,
    "trajectory_v23_terminal_gap_ratio": None,
    "trajectory_v23_refitted_remaining_slope_deg": None,
    "trajectory_v23_parent_frozen_score_improvement": None,
    "trajectory_v23_route_stability_fraction": None,
    "trajectory_v23_route_stability_passed": None,
    "trajectory_v23_candidate_uses_classifier_or_segmentation": False,
    "trajectory_v23_candidate_uses_raw_intensity_or_sobel": False,
    "trajectory_v23_narrow_two_row_collapse_candidate": False,
    "trajectory_v23_narrow_two_row_collapse_applied": False,
    "trajectory_v23_narrow_two_row_collapse_status": "not_applicable",
    "trajectory_v23_narrow_two_row_collapse_reason": "",
    "trajectory_v23_narrow_two_row_separation_um": None,
    "trajectory_v23_narrow_two_row_separation_width_factor": None,
    "trajectory_v23_narrow_two_row_combined_rms_um": None,
    "trajectory_v23_narrow_two_row_combined_rms_width_factor": None,
    "trajectory_v23_narrow_two_row_cluster_sizes": [],
    "trajectory_v23_band_membership_frozen": False,
    "trajectory_v23_band_boundary_restored": False,
    "trajectory_v23_band_boundary_restored_count": 0,
}


_V23_IMAGE_FIELDS = (
    "trajectory_v23_policy_version",
    "trajectory_v23_bracketed_small_rescue_count",
    "trajectory_v23_terminal_rescue_count",
    "trajectory_v23_additional_singleton_rescue_count",
    "trajectory_v23_narrow_two_row_collapse_count",
    "trajectory_v23_band_boundary_restored_count",
    "trajectory_v23_policy",
    "detected_row_count_augmented_v23",
    "detected_row_count_added_by_v23_additional_routes",
    "detected_row_count_removed_by_v23_narrow_collapse",
    "detected_row_count_net_change_by_v23",
    "detected_band_count_augmented_v23",
    "detected_band_count_boundaries_restored_by_v23",
    "row_band_v23_qc_status",
    "row_band_v23_qc_flag",
    "row_band_v23_qc_reason",
    "row_band_v23_value_interpretation",
)


def build_settings_v23(args: Any) -> TraitSettingsV23:
    """Build v2.3 settings while retaining every parent setting."""

    inherited = asdict(parent.build_settings_v20(args))
    inherited.update(
        {
            "v23_policy_version": SCRIPT_VERSION,
            "v23_parent_v22_script_sha256": hashlib.sha256(
                _PARENT_V22_SCRIPT_PATH.read_bytes()
            ).hexdigest(),
        }
    )
    return TraitSettingsV23(**inherited)


def _active_stoma_record(point_index: int) -> dict[str, Any] | None:
    records = v15._ACTIVE_STOMA_TRAITS_V15
    if records is None or not 0 <= int(point_index) < len(records):
        return None
    return records[int(point_index)]


def _local_detector_width_um(
    point_indices: list[int], fallback_width_um: float
) -> float:
    widths: list[float] = []
    for point_index in point_indices:
        record = _active_stoma_record(point_index)
        value = (
            base.optional_float(record.get("prediction_width_um"))
            if record is not None
            else None
        )
        if value is not None and np.isfinite(value) and value > 0:
            widths.append(float(value))
    return (
        float(np.median(np.asarray(widths, dtype=float)))
        if widths
        else float(fallback_width_um)
    )


def _fit_local_line(
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return intercept, slope, RMS and maximum absolute residual."""

    if len(point_indices) < 2:
        raise ValueError("A local line requires at least two stomata.")
    x = np.asarray([float(longitudinal[index]) for index in point_indices])
    y = np.asarray([float(lateral[index]) for index in point_indices])
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denominator = float(np.sum((x - x_mean) ** 2))
    slope = (
        float(np.sum((x - x_mean) * (y - y_mean)) / denominator)
        if denominator > 1e-12
        else 0.0
    )
    intercept = y_mean - slope * x_mean
    residuals = y - (intercept + slope * x)
    rms = float(math.sqrt(float(np.mean(residuals * residuals))))
    maximum = float(np.max(np.abs(residuals)))
    return intercept, slope, rms, maximum


def _v23_local_geometry(
    chosen: dict[str, Any],
    point_index: int,
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    fallback_width_um: float,
) -> dict[str, Any]:
    """Fit the candidate's source row after removing the candidate once."""

    label_map = parent._candidate_label_map(chosen)
    if point_index not in label_map:
        return {"assessable": False, "reason": "candidate_not_in_conservative_model"}
    source_group = int(label_map[point_index])
    source_members = sorted(
        index for index, group in label_map.items() if int(group) == source_group
    )
    remaining = [index for index in source_members if index != point_index]
    if len(remaining) < 2:
        return {"assessable": False, "reason": "fewer_than_two_remaining_members"}

    intercept, slope, remaining_rms, maximum_remaining = _fit_local_line(
        remaining, lateral, longitudinal
    )
    candidate_longitudinal = float(longitudinal[point_index])
    candidate_lateral = float(lateral[point_index])
    fitted_candidate = intercept + slope * candidate_longitudinal
    candidate_residual = abs(candidate_lateral - fitted_candidate)
    local_width = _local_detector_width_um(source_members, fallback_width_um)
    residual_factor = candidate_residual / max(local_width, 1e-9)
    rms_factor = remaining_rms / max(local_width, 1e-9)
    dominance = (
        candidate_residual / maximum_remaining
        if maximum_remaining > 1e-9
        else math.inf
    )

    ordered_remaining = sorted(float(longitudinal[index]) for index in remaining)
    bracketed = bool(
        ordered_remaining[0]
        < candidate_longitudinal
        < ordered_remaining[-1]
    )
    positive_gaps = [
        right - left
        for left, right in zip(
            ordered_remaining[:-1], ordered_remaining[1:], strict=True
        )
        if right > left
    ]
    typical_gap = (
        float(np.median(np.asarray(positive_gaps, dtype=float)))
        if positive_gaps
        else None
    )
    terminal_distance = min(
        abs(candidate_longitudinal - ordered_remaining[0]),
        abs(candidate_longitudinal - ordered_remaining[-1]),
    )
    terminal_gap_ratio = (
        terminal_distance / typical_gap
        if typical_gap is not None and typical_gap > 0
        else None
    )
    return {
        "assessable": True,
        "reason": "",
        "source_group": source_group,
        "source_members": source_members,
        "remaining_members": remaining,
        "source_member_count_before": len(source_members),
        "source_member_count_after": len(remaining),
        "local_detector_width_um": local_width,
        "candidate_residual_um": candidate_residual,
        "candidate_residual_width_factor": residual_factor,
        "remaining_row_rms_um": remaining_rms,
        "remaining_row_rms_width_factor": rms_factor,
        "residual_dominance_ratio": dominance,
        "longitudinally_bracketed": bracketed,
        "terminal_gap_ratio": terminal_gap_ratio,
        "refitted_slope": slope,
        "refitted_slope_deg": math.degrees(math.atan(slope)),
    }


def _v23_route_name_and_pass(
    geometry: dict[str, Any],
    confidence: float | None,
    density_eligible: bool | None,
    frozen_score_improvement: float | None,
    settings: TraitSettingsV23,
) -> tuple[str | None, list[str]]:
    """Evaluate the two narrow v2.3 routes and return failed safeguards."""

    if not bool(geometry.get("assessable")):
        return None, [str(geometry.get("reason") or "geometry_unassessable")]
    before = int(geometry["source_member_count_before"])
    after = int(geometry["source_member_count_after"])
    bracketed = bool(geometry["longitudinally_bracketed"])
    residual = float(geometry["candidate_residual_width_factor"])
    rms = float(geometry["remaining_row_rms_width_factor"])
    dominance = float(geometry["residual_dominance_ratio"])
    terminal_ratio = base.optional_float(geometry.get("terminal_gap_ratio"))
    slope_deg = abs(float(geometry["refitted_slope_deg"]))

    if (
        before == int(settings.v23_bracketed_small_min_source_members)
        and after == int(settings.v23_bracketed_small_remaining_members)
        and bracketed
    ):
        failures: list[str] = []
        if confidence is None or confidence < float(
            settings.v23_bracketed_small_min_confidence
        ):
            failures.append("failed_gate:detector_confidence")
        if density_eligible is not True:
            failures.append("failed_gate:density_eligible")
        if residual < float(
            settings.v23_bracketed_small_min_residual_width_factor
        ):
            failures.append("failed_gate:candidate_residual")
        if residual > float(settings.v22_loo_max_lateral_gap_width_factor):
            failures.append("failed_gate:within_existing_band_route")
        if rms > float(
            settings.v23_bracketed_small_max_remaining_rms_width_factor
        ):
            failures.append("failed_gate:remaining_pair_rms")
        if dominance < float(
            settings.v23_bracketed_small_min_residual_dominance
        ):
            failures.append("failed_gate:residual_dominance")
        if (
            frozen_score_improvement is None
            or frozen_score_improvement
            < float(settings.v23_bracketed_small_min_score_improvement)
        ):
            failures.append("failed_gate:frozen_score_improvement")
        return "bracketed_3_to_2_plus_1", failures

    if (
        before >= int(settings.v23_terminal_min_source_members)
        and after >= int(settings.v23_terminal_min_remaining_members)
        and not bracketed
    ):
        failures = []
        if confidence is None or confidence < float(
            settings.v23_terminal_min_confidence
        ):
            failures.append("failed_gate:detector_confidence")
        if density_eligible is not True:
            failures.append("failed_gate:density_eligible")
        if residual < float(settings.v23_terminal_min_residual_width_factor):
            failures.append("failed_gate:candidate_residual")
        if residual > float(settings.v23_terminal_max_residual_width_factor):
            failures.append("failed_gate:within_existing_band_route")
        if rms > float(settings.v23_terminal_max_remaining_rms_width_factor):
            failures.append("failed_gate:remaining_row_rms")
        if dominance < float(settings.v23_terminal_min_residual_dominance):
            failures.append("failed_gate:residual_dominance")
        if (
            terminal_ratio is None
            or terminal_ratio < float(settings.v23_terminal_min_gap_ratio)
            or terminal_ratio > float(settings.v23_terminal_max_gap_ratio)
        ):
            failures.append("failed_gate:terminal_gap_ratio")
        if slope_deg > float(settings.v23_terminal_max_refitted_slope_deg):
            failures.append("failed_gate:refitted_remaining_slope")
        return "terminal_refitted_remaining_row", failures

    return None, ["candidate_does_not_match_a_v23_route"]


def _v23_frozen_score(
    chosen: dict[str, Any],
    point_index: int,
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    width_um: float,
    settings: TraitSettingsV23,
) -> dict[str, Any]:
    try:
        return parent._v22_score_frozen_hybrid(
            point_index,
            chosen,
            lateral,
            longitudinal,
            width_um,
            settings,
        )
    except (RuntimeError, ValueError):
        return {
            "adjusted_score_improvement": None,
            "hybrid": None,
            "hybrid_diagnostic": None,
        }


def _v23_route_stability(
    route: str,
    chosen: dict[str, Any],
    point_index: int,
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    fallback_width_um: float,
    confidence: float | None,
    density_eligible: bool | None,
    settings: TraitSettingsV23,
) -> tuple[float, bool]:
    """Perturb lateral coordinates while keeping the candidate and memberships fixed."""

    iterations = int(settings.trajectory_stability_iterations)
    seed_offset = sum(
        (position + 1) * (int(index) + 1)
        for position, index in enumerate(chosen.get("point_indices", []))
    ) + 1009 * (int(point_index) + 1)
    rng = np.random.default_rng(
        int(settings.trajectory_stability_seed) + seed_offset
    )
    jitter_um = float(settings.trajectory_stability_jitter_width_fraction) * float(
        fallback_width_um
    )
    passes = 0
    for _ in range(iterations):
        jittered = np.asarray(lateral, dtype=float).copy()
        points = [int(value) for value in chosen.get("point_indices", [])]
        jittered[points] += rng.normal(0.0, jitter_um, len(points))
        geometry = _v23_local_geometry(
            chosen,
            point_index,
            jittered,
            longitudinal,
            fallback_width_um,
        )
        score_improvement: float | None = None
        if route == "bracketed_3_to_2_plus_1":
            score = _v23_frozen_score(
                chosen,
                point_index,
                jittered,
                longitudinal,
                fallback_width_um,
                settings,
            )
            score_improvement = base.optional_float(
                score.get("adjusted_score_improvement")
            )
        perturbed_route, failures = _v23_route_name_and_pass(
            geometry,
            confidence,
            density_eligible,
            score_improvement,
            settings,
        )
        passes += int(perturbed_route == route and not failures)
    fraction = passes / max(iterations, 1)
    return fraction, bool(
        fraction >= float(settings.v23_min_route_stability_fraction)
    )


def _replace_candidate_and_diagnostic(
    candidates: list[dict[str, Any]],
    diagnostic: dict[str, Any],
    hybrid: dict[str, Any],
    hybrid_diagnostic: dict[str, Any],
) -> list[dict[str, Any]]:
    hybrid_k = int(hybrid["k"])
    updated: list[dict[str, Any]] = []
    replaced = False
    for candidate in candidates:
        if int(candidate.get("k", -1)) == hybrid_k and not replaced:
            updated.append(hybrid)
            replaced = True
        else:
            updated.append(candidate)
    if not replaced:
        updated.append(hybrid)
        updated.sort(key=lambda item: int(item["k"]))

    diagnostics: list[dict[str, Any]] = []
    diagnostic_replaced = False
    for item in diagnostic.get("candidate_diagnostics", []):
        if (
            base.optional_int(item.get("candidate_row_count")) == hybrid_k
            and not diagnostic_replaced
        ):
            diagnostics.append(hybrid_diagnostic)
            diagnostic_replaced = True
        else:
            diagnostics.append(item)
    if not diagnostic_replaced:
        diagnostics.append(hybrid_diagnostic)
        diagnostics.sort(
            key=lambda item: int(item.get("candidate_row_count") or 0)
        )
    diagnostic["candidate_diagnostics"] = diagnostics
    return updated


def _apply_v23_additional_singleton_route(
    chosen: dict[str, Any],
    candidates: list[dict[str, Any]],
    diagnostic: dict[str, Any],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    fallback_width_um: float,
    settings: TraitSettingsV23,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Apply one of the two extra routes only after v2.2 stayed conservative."""

    if bool(chosen.get("trajectory_v20_singleton_rescue_applied")):
        return chosen, candidates, diagnostic
    nominated = [
        int(value)
        for value in (
            diagnostic.get("trajectory_v20_singleton_candidate_point_indices")
            or []
        )
    ]
    if len(nominated) != 1:
        return chosen, candidates, diagnostic
    point_index = nominated[0]
    confidence = parent._detector_confidence(point_index)
    active = _active_stoma_record(point_index)
    density_eligible = (
        base.parse_bool(active.get("density_eligible"))
        if active is not None
        else None
    )
    geometry = _v23_local_geometry(
        chosen,
        point_index,
        lateral,
        longitudinal,
        fallback_width_um,
    )
    score = _v23_frozen_score(
        chosen,
        point_index,
        lateral,
        longitudinal,
        fallback_width_um,
        settings,
    )
    score_improvement = base.optional_float(
        score.get("adjusted_score_improvement")
    )
    route, failures = _v23_route_name_and_pass(
        geometry,
        confidence,
        density_eligible,
        score_improvement,
        settings,
    )
    candidate_fields = {
        "trajectory_v23_additional_singleton_candidate": route is not None,
        "trajectory_v23_candidate_point_index": point_index,
        "trajectory_v23_candidate_stoma_id": parent._stoma_id(point_index),
        "trajectory_v23_source_row_member_count_before": geometry.get(
            "source_member_count_before"
        ),
        "trajectory_v23_source_row_member_count_after": geometry.get(
            "source_member_count_after"
        ),
        "trajectory_v23_local_detector_width_um": geometry.get(
            "local_detector_width_um"
        ),
        "trajectory_v23_candidate_residual_um": geometry.get(
            "candidate_residual_um"
        ),
        "trajectory_v23_candidate_residual_width_factor": geometry.get(
            "candidate_residual_width_factor"
        ),
        "trajectory_v23_remaining_row_rms_um": geometry.get(
            "remaining_row_rms_um"
        ),
        "trajectory_v23_remaining_row_rms_width_factor": geometry.get(
            "remaining_row_rms_width_factor"
        ),
        "trajectory_v23_residual_dominance_ratio": geometry.get(
            "residual_dominance_ratio"
        ),
        "trajectory_v23_longitudinally_bracketed": geometry.get(
            "longitudinally_bracketed"
        ),
        "trajectory_v23_terminal_gap_ratio": geometry.get("terminal_gap_ratio"),
        "trajectory_v23_refitted_remaining_slope_deg": geometry.get(
            "refitted_slope_deg"
        ),
        "trajectory_v23_parent_frozen_score_improvement": score_improvement,
        "trajectory_v23_rescue_route": route or "not_applicable",
    }
    chosen.update(candidate_fields)
    diagnostic.update(candidate_fields)
    if route is None or failures:
        rejection = {
            "trajectory_v23_additional_singleton_applied": False,
            "trajectory_v23_rescue_status": (
                "rejected_safeguards" if route is not None else "not_applicable"
            ),
            "trajectory_v23_rescue_reason": ";".join(failures),
        }
        chosen.update(rejection)
        diagnostic.update(rejection)
        return chosen, candidates, diagnostic

    stability_fraction, stability_passed = _v23_route_stability(
        route,
        chosen,
        point_index,
        lateral,
        longitudinal,
        fallback_width_um,
        confidence,
        density_eligible,
        settings,
    )
    stability_fields = {
        "trajectory_v23_route_stability_fraction": stability_fraction,
        "trajectory_v23_route_stability_passed": stability_passed,
    }
    chosen.update(stability_fields)
    diagnostic.update(stability_fields)
    if not stability_passed:
        rejection = {
            "trajectory_v23_additional_singleton_applied": False,
            "trajectory_v23_rescue_status": "rejected_perturbation_stability",
            "trajectory_v23_rescue_reason": "failed_gate:route_stability",
        }
        chosen.update(rejection)
        diagnostic.update(rejection)
        return chosen, candidates, diagnostic

    refitted_template = dict(chosen)
    refitted_template["slope"] = float(geometry["refitted_slope"])
    refitted_fallback = parent._refit_fixed_membership_candidate(
        refitted_template, lateral, longitudinal
    )
    hybrid = parent._build_hybrid_singleton_candidate(
        refitted_fallback, point_index, lateral, longitudinal
    )
    hybrid_angle = math.degrees(math.atan(float(hybrid["slope"])))
    hybrid_diagnostic = v15._score_trajectory_candidate(
        hybrid,
        fallback_width_um,
        hybrid_angle,
        settings,
    )
    fallback_k = int(chosen["k"])
    hybrid_k = int(hybrid["k"])
    selection_reason = (
        str(chosen.get("automatic_selection_reason") or "")
        + f"__v23_{route}_membership_preserving_singleton_rescue"
    )
    accepted_fields = {
        **candidate_fields,
        **stability_fields,
        "automatic_row_count": hybrid_k,
        "automatic_selection_reason": selection_reason,
        "row_count_selection_reason": f"v23_{route}_singleton_rescue",
        "sparse_row_support": True,
        "model_confidence": "low",
        "trajectory_slope_angle_deg": hybrid_angle,
        "trajectory_singleton_validation_required": True,
        "trajectory_singleton_rows_validated": True,
        "trajectory_singleton_detector_confidences": [confidence],
        "trajectory_singleton_local_edge_support_ratios": [],
        "trajectory_singleton_validation_failure_reason": "",
        "trajectory_exact_row_traits_eligible": False,
        "trajectory_pre_validation_row_count": hybrid_k,
        "trajectory_post_validation_row_count": hybrid_k,
        "trajectory_validation_fallback_applied": False,
        "trajectory_validation_fallback_reason": "",
        "trajectory_displayed_non_singleton_row_count": fallback_k,
        "trajectory_displayed_fallback_row_count": fallback_k,
        "trajectory_row_count_min": fallback_k,
        "trajectory_row_count_max": hybrid_k,
        "trajectory_row_count_range_reason": f"v23_{route}_provisional",
        "trajectory_row_count_stability_fraction": None,
        "trajectory_assignment_stability_fraction": None,
        "trajectory_conditional_row_count_sensitivity_fraction": None,
        "trajectory_conditional_assignment_sensitivity_fraction": None,
        "trajectory_row_count_stable": False,
        "trajectory_stability_count_distribution": {},
        "trajectory_v20_singleton_rescue_applied": True,
        "trajectory_v20_singleton_validation_status": "accepted_provisional",
        "trajectory_v20_singleton_validation_reason": f"v23_{route}_passed",
        "trajectory_v20_conservative_row_count": fallback_k,
        "trajectory_v20_augmented_row_count": hybrid_k,
        "trajectory_v20_added_row_count": 1,
        "trajectory_v20_singleton_route": "within_existing_band",
        "trajectory_v20_singleton_band_boundary_ambiguous": False,
        "trajectory_v20_orientation_inherited": True,
        "trajectory_v20_orientation_independently_estimable": False,
        "trajectory_v20_traits_involving_singleton_provisional": True,
        "trajectory_v22_augmented_row_count": hybrid_k,
        "trajectory_v22_leave_one_out_applied": True,
        "trajectory_v22_leave_one_out_status": "accepted_provisional_v23_route",
        "trajectory_v22_leave_one_out_reason": f"v23_{route}_passed",
        "trajectory_v22_stability_iterations": int(
            settings.trajectory_stability_iterations
        ),
        "trajectory_v22_loo_row_count_stability_fraction": 1.0,
        "trajectory_v22_loo_assignment_stability_fraction": 1.0,
        "trajectory_v22_geometry_stability_fraction": stability_fraction,
        "trajectory_v22_score_stability_fraction": (
            stability_fraction
            if route == "bracketed_3_to_2_plus_1"
            else None
        ),
        "trajectory_v22_route_stability_fraction": stability_fraction,
        "trajectory_v22_joint_gate_stability_fraction": stability_fraction,
        "trajectory_v22_stability_gate_passed": True,
        "trajectory_v22_adds_new_band": False,
        "trajectory_v22_nonrecursive_single_pass": True,
        "trajectory_v23_additional_singleton_applied": True,
        "trajectory_v23_rescue_status": "accepted_provisional",
        "trajectory_v23_rescue_reason": f"v23_{route}_all_safeguards_passed",
        "trajectory_v23_band_membership_frozen": True,
    }
    rescued = dict(chosen)
    rescued.update(hybrid)
    rescued.update(accepted_fields)
    candidates = _replace_candidate_and_diagnostic(
        candidates, diagnostic, hybrid, hybrid_diagnostic
    )
    diagnostic.update(
        {
            **accepted_fields,
            "global_automatic_row_count": hybrid_k,
            "row_count_used_for_traits": hybrid_k,
            "row_count_used_for_traits_source": f"v23_{route}",
            "alternative_row_count": fallback_k,
            "global_row_selection_score": float(
                hybrid_diagnostic["adjusted_score"]
            ),
            "global_row_score_margin": max(0.0, score_improvement or 0.0),
            "global_row_raw_margin": score_improvement,
            "global_row_automatic_is_strict_best": False,
            "global_row_model_confidence": "low",
            "singleton_supported_row_count": 1,
            "singleton_detector_confidence_mean": confidence,
        }
    )
    return rescued, candidates, diagnostic


def _clarify_parent_rejection_reason(value: Any) -> str:
    """Make v2.2 failed-gate names unambiguous in new v2.3 outputs."""

    gate_names = {
        "source_group_is_nearest",
        "longitudinally_bracketed",
        "local_residual_passed",
        "shared_residual_passed",
        "remaining_row_rms_passed",
        "residual_dominance_passed",
        "bracketing_gap_passed",
        "within_existing_band_route_passed",
    }
    clarified: list[str] = []
    for token in str(value or "").split(";"):
        if not token:
            continue
        if token in gate_names:
            name = token.removesuffix("_passed")
            clarified.append(f"failed_gate:{name}")
        else:
            clarified.append(token)
    return ";".join(dict.fromkeys(clarified))


def _apply_v23_narrow_two_row_collapse(
    chosen: dict[str, Any],
    candidates: list[dict[str, Any]],
    diagnostic: dict[str, Any],
    fallback_width_um: float,
    settings: TraitSettingsV23,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Collapse only a sub-0.25 local-width marginal two-row solution."""

    if bool(chosen.get("trajectory_v20_singleton_rescue_applied")):
        return chosen, candidates, diagnostic
    if int(chosen.get("k", 0)) != 2:
        return chosen, candidates, diagnostic
    sizes = [int(value) for value in chosen.get("cluster_sizes", [])]
    if len(sizes) != 2 or min(sizes) < int(
        settings.v23_narrow_split_min_members_per_proposed_row
    ):
        return chosen, candidates, diagnostic
    one = next((item for item in candidates if int(item.get("k", 0)) == 1), None)
    if one is None:
        return chosen, candidates, diagnostic
    points = [int(value) for value in chosen.get("point_indices", [])]
    local_width = _local_detector_width_um(points, fallback_width_um)
    intercepts = sorted(float(value) for value in chosen.get("intercepts_um", []))
    if len(intercepts) != 2:
        return chosen, candidates, diagnostic
    separation_um = abs(intercepts[1] - intercepts[0])
    separation_factor = separation_um / max(local_width, 1e-9)
    combined_rms = float(one.get("rms_residual_um") or math.inf)
    combined_rms_factor = combined_rms / max(local_width, 1e-9)
    candidate = bool(
        separation_factor
        < float(settings.v23_narrow_split_max_separation_width_factor)
    )
    failures: list[str] = []
    if not candidate:
        failures.append("failed_gate:band_local_row_separation")
    if combined_rms_factor > float(
        settings.v23_narrow_split_max_combined_rms_width_factor
    ):
        failures.append("failed_gate:combined_one_row_rms")
    fields = {
        "trajectory_v23_narrow_two_row_collapse_candidate": candidate,
        "trajectory_v23_narrow_two_row_collapse_applied": bool(
            candidate and not failures
        ),
        "trajectory_v23_narrow_two_row_collapse_status": (
            "accepted_provisional" if candidate and not failures else "not_applied"
        ),
        "trajectory_v23_narrow_two_row_collapse_reason": (
            "band_local_separation_below_strict_minimum_and_one_row_fit_smooth"
            if candidate and not failures
            else ";".join(failures)
        ),
        "trajectory_v23_narrow_two_row_separation_um": separation_um,
        "trajectory_v23_narrow_two_row_separation_width_factor": (
            separation_factor
        ),
        "trajectory_v23_narrow_two_row_combined_rms_um": combined_rms,
        "trajectory_v23_narrow_two_row_combined_rms_width_factor": (
            combined_rms_factor
        ),
        "trajectory_v23_narrow_two_row_cluster_sizes": sizes,
    }
    chosen.update(fields)
    diagnostic.update(fields)
    if not bool(fields["trajectory_v23_narrow_two_row_collapse_applied"]):
        return chosen, candidates, diagnostic

    collapsed = dict(chosen)
    collapsed.update(one)
    collapsed.update(
        {
            **fields,
            "automatic_row_count": 1,
            "automatic_selection_reason": (
                str(chosen.get("automatic_selection_reason") or "")
                + "__v23_band_local_narrow_two_row_collapse"
            ),
            "row_count_selection_reason": (
                "v23_band_local_narrow_two_row_collapse"
            ),
            "model_confidence": "low",
            "trajectory_exact_row_traits_eligible": False,
            "trajectory_pre_validation_row_count": 2,
            "trajectory_post_validation_row_count": 1,
            "trajectory_row_count_min": 1,
            "trajectory_row_count_max": 2,
            "trajectory_row_count_range_reason": (
                "v23_narrow_two_row_solution_collapsed_provisionally"
            ),
            "trajectory_row_count_stable": False,
            "trajectory_stability_count_distribution": {},
        }
    )
    one_diagnostic = next(
        (
            item
            for item in diagnostic.get("candidate_diagnostics", [])
            if base.optional_int(item.get("candidate_row_count")) == 1
        ),
        None,
    )
    diagnostic.update(
        {
            **fields,
            "global_automatic_row_count": 1,
            "row_count_used_for_traits": 1,
            "row_count_used_for_traits_source": (
                "v23_band_local_narrow_two_row_collapse"
            ),
            "alternative_row_count": 2,
            "global_row_selection_score": (
                one_diagnostic.get("adjusted_score")
                if one_diagnostic is not None
                else None
            ),
            "global_row_score_margin": 0.0,
            "global_row_automatic_is_strict_best": False,
            "global_row_model_confidence": "low",
            "singleton_supported_row_count": 0,
            "potential_row_undersplit_in_legacy": False,
            "potential_row_oversplit_in_legacy": True,
        }
    )
    return collapsed, candidates, diagnostic


def select_row_model_v23(
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV23,
    audited_row_count: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run v2.2, then apply only the fixed v2.3 post-selection rules."""

    diagnostic_count = len(v14._ACTIVE_BAND_DIAGNOSTICS)
    chosen, candidates = parent.select_row_model_v20(
        point_indices,
        lateral,
        longitudinal,
        median_detector_width_um,
        settings,
        audited_row_count=audited_row_count,
    )
    if len(v14._ACTIVE_BAND_DIAGNOSTICS) != diagnostic_count + 1:
        raise RuntimeError("v2.3 expected one provisional-band diagnostic record.")
    diagnostic = v14._ACTIVE_BAND_DIAGNOSTICS[-1]

    if str(chosen.get("trajectory_v22_leave_one_out_status") or "").startswith(
        "rejected"
    ):
        clarified = _clarify_parent_rejection_reason(
            chosen.get("trajectory_v22_leave_one_out_reason")
        )
        chosen["trajectory_v22_leave_one_out_reason"] = clarified
        diagnostic["trajectory_v22_leave_one_out_reason"] = clarified

    if audited_row_count is None:
        chosen, candidates, diagnostic = _apply_v23_additional_singleton_route(
            chosen,
            candidates,
            diagnostic,
            lateral,
            longitudinal,
            median_detector_width_um,
            settings,
        )
        chosen, candidates, diagnostic = _apply_v23_narrow_two_row_collapse(
            chosen,
            candidates,
            diagnostic,
            median_detector_width_um,
            settings,
        )

    for key, default in _V23_DEFAULTS.items():
        value = chosen.get(key, diagnostic.get(key, default))
        chosen[key] = value
        diagnostic[key] = value
    return chosen, candidates


def _trajectory_diagnostic_fields_v23(
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    fields = parent._trajectory_diagnostic_fields_v20(diagnostic)
    fields["row_model_version"] = SCRIPT_VERSION
    for key, default in _V23_DEFAULTS.items():
        fields[key] = diagnostic.get(key, default)
    return fields


def _is_v23_singleton_row(row: dict[str, Any]) -> bool:
    members = [int(value) for value in row.get("_point_indices", [])]
    candidate = base.optional_int(row.get("trajectory_v22_candidate_point_index"))
    return bool(
        len(members) == 1
        and candidate is not None
        and members == [candidate]
        and bool(row.get("trajectory_v22_leave_one_out_applied"))
    )


def _rebuild_v23_band_group(
    parent_band: dict[str, Any],
    group: list[dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    partition: dict[str, Any],
    group_index: int,
) -> dict[str, Any]:
    required_partition_fields = {
        "split",
        "boundary_gap_um",
        "gap_ratio",
        "gap_width_factor",
        "left_alternation",
        "right_alternation",
        "image_support_ratio",
    }
    missing_partition_fields = sorted(
        required_partition_fields.difference(partition)
    )
    if missing_partition_fields:
        raise RuntimeError(
            "v2.3 cannot rebuild an accepted band boundary because its "
            "audit record is incomplete; "
            f"missing_fields={missing_partition_fields}, "
            f"provisional_band={partition.get('provisional_band_id')}, "
            f"cut={partition.get('split')}."
        )
    record = v14._rebuild_band_record(
        parent_band,
        group,
        stoma_traits,
        median_detector_width_um,
        partition,
        group_index,
    )
    gaps = [
        v15._matched_trajectory_gap(first, second)[0]
        for first, second in zip(group[:-1], group[1:], strict=True)
    ]
    record.update(base.sample_summary(gaps, "within_band_interrow_spacing_um"))
    record.update(
        {
            # Keep the true parent provisional identity.  The shared rebuild
            # helper otherwise substitutes the current final band identifier.
            "provisional_band_id": parent_band.get("provisional_band_id"),
            "within_band_interrow_spacing_mean_um": (
                float(np.mean(gaps)) if gaps else None
            ),
            "within_band_interrow_spacing_median_um": (
                float(np.median(gaps)) if gaps else None
            ),
            "within_band_spacing_basis": (
                "matched_longitudinal_positions_between_fitted_trajectories"
            ),
            "trajectory_band_partition_method": (
                "v2.3_frozen_source_band_strong_external_boundary_restoration"
            ),
            "trajectory_exact_row_band_traits_eligible": False,
            "v20_singleton_partition_applied": False,
            "v20_singleton_band": False,
            "trajectory_v23_band_membership_frozen": True,
            "trajectory_v23_band_boundary_restored": True,
        }
    )
    return record


def _restore_v23_band_boundaries_after_v22(
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    settings: TraitSettingsV23,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Restore accepted external boundaries on the completed v2.2 result.

    This deliberately runs after ``infer_rows_and_bands_v20`` has finished.
    At that point the leave-one-out candidate identity and source membership
    are final, so v2.3 never has to infer a boundary in one coordinate context
    and verify it later in a different, rebuilt context.
    """

    rows = row_records
    parent_bands = band_records
    assignments = point_assignment
    partition_audit: list[dict[str, Any]] = []
    row_lookup = {str(row["row_id"]): row for row in rows}
    rebuilt: list[dict[str, Any]] = []
    accepted_separations: list[
        tuple[set[str], set[str], str, int, float, float]
    ] = []

    for band in parent_bands:
        linked = sorted(
            [row_lookup[str(row_id)] for row_id in band.get("row_ids", [])],
            key=lambda row: base.row_lateral_at(row, 0.0),
        )
        singleton_positions = [
            position
            for position, row in enumerate(linked)
            if _is_v23_singleton_row(row)
        ]
        if len(linked) < 3 or not singleton_positions:
            band.setdefault("trajectory_v23_band_membership_frozen", False)
            band.setdefault("trajectory_v23_band_boundary_restored", False)
            rebuilt.append(band)
            continue

        gaps = [
            v15._matched_trajectory_gap(first, second)[0]
            for first, second in zip(linked[:-1], linked[1:], strict=True)
        ]
        accepted_cuts: set[int] = set()
        cut_records: dict[int, dict[str, Any]] = {}
        for singleton_position in singleton_positions:
            singleton = linked[singleton_position]
            other_positions = [
                position
                for position in range(len(linked))
                if position != singleton_position
                and not _is_v23_singleton_row(linked[position])
            ]
            if not other_positions:
                continue
            source_position = min(
                other_positions,
                key=lambda position: v15._matched_trajectory_gap(
                    singleton, linked[position]
                )[0],
            )
            pair_start = min(singleton_position, source_position)
            pair_stop = max(singleton_position, source_position)
            candidate_cuts: list[int] = []
            if pair_start > 0:
                candidate_cuts.append(pair_start)
            if pair_stop < len(linked) - 1:
                candidate_cuts.append(pair_stop + 1)
            for cut in candidate_cuts:
                gap_um = float(gaps[cut - 1])
                factor = gap_um / max(float(median_detector_width_um), 1e-9)
                seed_token = (
                    band.get("provisional_band_id") or band.get("band_id")
                )
                stability = parent._strong_gap_stability(
                    gap_um,
                    median_detector_width_um,
                    settings,
                    sum(ord(char) for char in str(seed_token)) + cut + 2300,
                )
                if not (
                    factor
                    >= float(settings.v23_band_freeze_strong_gap_width_factor)
                    and stability
                    >= float(settings.v23_band_freeze_min_stability_fraction)
                ):
                    continue
                accepted_cuts.add(cut)
                other_gaps = [
                    float(value)
                    for gap_index, value in enumerate(gaps)
                    if gap_index != cut - 1
                ]
                local_gap_reference = (
                    float(np.median(np.asarray(other_gaps, dtype=float)))
                    if other_gaps
                    else None
                )
                cut_records[cut] = {
                    "provisional_band_id": band.get("provisional_band_id"),
                    "split": cut,
                    "boundary_gap_um": gap_um,
                    "gap_ratio": (
                        gap_um / max(local_gap_reference, 1e-9)
                        if local_gap_reference is not None
                        else math.inf
                    ),
                    "gap_width_factor": factor,
                    "left_alternation": v14._row_group_alternation(
                        linked[:cut], stoma_traits
                    ),
                    "right_alternation": v14._row_group_alternation(
                        linked[cut:], stoma_traits
                    ),
                    # This route is accepted from fixed detector geometry and
                    # perturbation stability; raw-image edge evidence is not
                    # an additional acceptance gate.
                    "image_support_ratio": None,
                    "boundary_stability_fraction": stability,
                    "boundary_stable": True,
                    "accepted": True,
                    "trajectory_v23_band_membership_frozen": True,
                    "trajectory_v23_band_boundary_restored": True,
                    "trajectory_v23_boundary_reason": (
                        "strong_external_boundary_restored_around_singleton_source_pair"
                    ),
                }
                partition_audit.append(cut_records[cut])

        if not accepted_cuts:
            band["trajectory_v23_band_membership_frozen"] = True
            band["trajectory_v23_band_boundary_restored"] = False
            rebuilt.append(band)
            continue

        for cut in sorted(accepted_cuts):
            accepted_separations.append(
                (
                    {str(row["row_id"]) for row in linked[:cut]},
                    {str(row["row_id"]) for row in linked[cut:]},
                    str(
                        band.get("provisional_band_id")
                        or band.get("band_id")
                    ),
                    cut,
                    float(cut_records[cut]["gap_width_factor"]),
                    float(
                        cut_records[cut]["boundary_stability_fraction"]
                    ),
                )
            )

        boundaries = [0, *sorted(accepted_cuts), len(linked)]
        groups = [
            linked[start:stop]
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
            if stop > start
        ]
        for group_index, group in enumerate(groups, start=1):
            start = boundaries[group_index - 1]
            stop = boundaries[group_index]
            nearby = stop if stop in cut_records else start
            if nearby not in cut_records:
                nearby = sorted(cut_records)[0]
            rebuilt.append(
                _rebuild_v23_band_group(
                    band,
                    group,
                    stoma_traits,
                    median_detector_width_um,
                    cut_records[nearby],
                    group_index,
                )
            )

    rebuilt.sort(key=lambda band: float(band["band_center_lateral_um"]))
    for band_number, band in enumerate(rebuilt, start=1):
        final_band_id = f"B{band_number:03d}"
        band["band_id"] = final_band_id
        band["band_order_left_to_right"] = band_number
        linked = sorted(
            [row_lookup[str(row_id)] for row_id in band.get("row_ids", [])],
            key=lambda row: base.row_lateral_at(row, 0.0),
        )
        for position, row in enumerate(linked, start=1):
            row.update(
                {
                    "band_id": final_band_id,
                    "row_position_within_band": position,
                    "rows_in_band": len(linked),
                    "automatic_rows_in_band": len(linked),
                    "band_row_count": len(linked),
                    "trajectory_v23_band_membership_frozen": bool(
                        band.get("trajectory_v23_band_membership_frozen")
                    ),
                    "trajectory_v23_band_boundary_restored": bool(
                        band.get("trajectory_v23_band_boundary_restored")
                    ),
                }
            )
            if _is_v23_singleton_row(row):
                row["trajectory_exact_row_band_traits_eligible"] = False
            for point_index in row.get("_point_indices", []):
                assignments[int(point_index)].update(
                    {
                        "band_id": final_band_id,
                        "row_position_within_band": position,
                        "rows_in_band": len(linked),
                        "automatic_rows_in_band": len(linked),
                        "trajectory_v23_band_membership_frozen": row[
                            "trajectory_v23_band_membership_frozen"
                        ],
                        "trajectory_v23_band_boundary_restored": row[
                            "trajectory_v23_band_boundary_restored"
                        ],
                        "trajectory_exact_row_band_traits_eligible": row.get(
                            "trajectory_exact_row_band_traits_eligible"
                        ),
                    }
                )

    # Validate the exact decisions that were accepted above.  Do not search
    # the rebuilt bands for new geometric boundaries: row context and cut
    # indices have changed by then, so that would be a second, non-equivalent
    # biological decision rather than an invariant.
    row_to_final_band = {
        str(row["row_id"]): str(row["band_id"])
        for row in rows
    }
    for (
        left_row_ids,
        right_row_ids,
        provisional_band_id,
        cut,
        gap_factor,
        stability,
    ) in accepted_separations:
        left_final_bands = {
            row_to_final_band[row_id]
            for row_id in left_row_ids
            if row_id in row_to_final_band
        }
        right_final_bands = {
            row_to_final_band[row_id]
            for row_id in right_row_ids
            if row_id in row_to_final_band
        }
        shared = sorted(left_final_bands & right_final_bands)
        if shared:
            raise RuntimeError(
                "v2.3 accepted-boundary invariant failed: an accepted cut "
                "did not separate its recorded row memberships; "
                f"provisional_band={provisional_band_id}, cut={cut}, "
                f"left_rows={sorted(left_row_ids)}, "
                f"right_rows={sorted(right_row_ids)}, "
                f"shared_final_bands={shared}, "
                f"gap_width_factor={gap_factor:.6f}, "
                f"stability={stability:.6f}."
            )
    return rows, rebuilt, assignments, partition_audit


def _assert_v23_invariants(
    stoma_traits: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    bands: list[dict[str, Any]],
    assignments: dict[int, dict[str, Any]],
    settings: TraitSettingsV23,
) -> None:
    """Check final memberships without making a second boundary decision."""

    parent._assert_v20_invariants(stoma_traits, rows, bands, assignments)
    source_image = (
        str(stoma_traits[0].get("source_image_name") or "unknown_image")
        if stoma_traits
        else "unknown_image"
    )
    row_lookup = {str(row["row_id"]): row for row in rows}
    for row in rows:
        if not bool(row.get("trajectory_v23_additional_singleton_applied")):
            continue
        candidate = base.optional_int(row.get("trajectory_v23_candidate_point_index"))
        if candidate is None:
            raise RuntimeError(
                "v2.3 invariant failed: accepted route lacks candidate; "
                f"source_image={source_image}, row_id={row.get('row_id')}."
            )
        matching = [
            item
            for item in rows
            if [int(value) for value in item.get("_point_indices", [])]
            == [candidate]
        ]
        if len(matching) != 1:
            raise RuntimeError(
                "v2.3 invariant failed: accepted candidate must form one "
                f"singleton row; source_image={source_image}, "
                f"candidate_index={candidate}."
            )
        if bool(matching[0].get("v20_singleton_band")):
            raise RuntimeError(
                "v2.3 invariant failed: additional route created a singleton "
                f"band; source_image={source_image}, candidate_index={candidate}."
            )

    for band in bands:
        linked = [
            row_lookup[str(row_id)]
            for row_id in band.get("row_ids", [])
            if str(row_id) in row_lookup
        ]
        singleton_rows = [row for row in linked if _is_v23_singleton_row(row)]
        if singleton_rows and len(linked) == 1:
            raise RuntimeError(
                "v2.3 invariant failed: leave-one-out singleton is isolated "
                f"as a band; source_image={source_image}, "
                f"band_id={band.get('band_id')}, "
                f"row_id={singleton_rows[0].get('row_id')}."
            )


def infer_rows_and_bands_v23(
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettingsV23,
    audited_row_counts: dict[str, int] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    rows, bands, assignments, audit = parent.infer_rows_and_bands_v20(
        stoma_traits,
        settings,
        audited_row_counts=audited_row_counts,
    )
    detector_widths_um = [
        float(value)
        for value in (
            base.optional_float(record.get("prediction_width_um"))
            for record in stoma_traits
        )
        if value is not None and value > 0
    ]
    median_detector_width_um = (
        float(np.median(detector_widths_um)) if detector_widths_um else 20.0
    )
    rows, bands, assignments, v23_partition_audit = (
        _restore_v23_band_boundaries_after_v22(
            rows,
            bands,
            assignments,
            stoma_traits,
            median_detector_width_um,
            settings,
        )
    )
    audit.setdefault("band_partition_diagnostics", []).extend(
        v23_partition_audit
    )
    diagnostics = list(audit.get("band_selection_diagnostics", {}).values())
    bracketed_count = sum(
        bool(item.get("trajectory_v23_additional_singleton_applied"))
        and item.get("trajectory_v23_rescue_route") == "bracketed_3_to_2_plus_1"
        for item in diagnostics
    )
    terminal_count = sum(
        bool(item.get("trajectory_v23_additional_singleton_applied"))
        and item.get("trajectory_v23_rescue_route")
        == "terminal_refitted_remaining_row"
        for item in diagnostics
    )
    collapse_count = sum(
        bool(item.get("trajectory_v23_narrow_two_row_collapse_applied"))
        for item in diagnostics
    )
    restored_count = sum(
        bool(item.get("trajectory_v23_band_boundary_restored"))
        and bool(item.get("accepted"))
        for item in audit.get("band_partition_diagnostics", [])
    )
    audit.update(
        {
            "row_model_version": SCRIPT_VERSION,
            "trajectory_singleton_policy_version_current": SCRIPT_VERSION,
            "trajectory_v23_policy_version": SCRIPT_VERSION,
            "trajectory_v23_bracketed_small_rescue_count": bracketed_count,
            "trajectory_v23_terminal_rescue_count": terminal_count,
            "trajectory_v23_additional_singleton_rescue_count": (
                bracketed_count + terminal_count
            ),
            "trajectory_v23_narrow_two_row_collapse_count": collapse_count,
            "trajectory_v23_band_boundary_restored_count": restored_count,
            "trajectory_v23_policy": (
                "retain standard v2.2 rescue; allow only strongly supported "
                "bracketed 3-to-2-plus-1 or terminal refitted singleton routes; "
                "collapse only a sub-0.25 band-local-width smooth two-row split; "
                "keep every rescued singleton with its nearest established row "
                "and restore strong external band boundaries"
            ),
            "automatic_row_count_policy": (
                "v2.3 keeps the v2.2 hierarchy and adds two fixed candidate-only "
                "singleton routes plus one strict band-local narrow-split collapse; "
                "no filename override or recursive candidate scan is permitted"
            ),
            "trajectory_row_band_trait_policy": (
                "v2.3 retains computable row and band phenotypes with explicit "
                "provisional QC; a singleton contributes to topology counts but "
                "cannot independently estimate within-row spacing, continuity, "
                "alternation, phase or orientation"
            ),
        }
    )
    for collection in (rows, bands, list(assignments.values())):
        for record in collection:
            record["row_model_version"] = SCRIPT_VERSION
            record.setdefault("trajectory_v23_policy_version", SCRIPT_VERSION)
    _assert_v23_invariants(stoma_traits, rows, bands, assignments, settings)
    return rows, bands, assignments, audit


def image_trait_summary_v23(
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
    result = parent.image_trait_summary_v20(
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
    bracketed = int(
        row_band_audit.get("trajectory_v23_bracketed_small_rescue_count") or 0
    )
    terminal = int(
        row_band_audit.get("trajectory_v23_terminal_rescue_count") or 0
    )
    collapse = int(
        row_band_audit.get("trajectory_v23_narrow_two_row_collapse_count") or 0
    )
    restored = int(
        row_band_audit.get("trajectory_v23_band_boundary_restored_count") or 0
    )
    reasons: list[str] = []
    if bracketed:
        reasons.append("v23_bracketed_3_to_2_plus_1_singleton_provisional")
    if terminal:
        reasons.append("v23_terminal_singleton_provisional")
    if collapse:
        reasons.append("v23_narrow_two_row_collapse_provisional")
    if restored:
        reasons.append("v23_strong_external_band_boundary_restored")
    result.update(
        {
            key: row_band_audit.get(key)
            for key in _V23_IMAGE_FIELDS
            if key.startswith("trajectory_v23_")
        }
    )
    result.update(
        {
            "detected_row_count_augmented_v23": len(row_records),
            "detected_row_count_added_by_v23_additional_routes": (
                bracketed + terminal
            ),
            "detected_row_count_removed_by_v23_narrow_collapse": collapse,
            "detected_row_count_net_change_by_v23": (
                bracketed + terminal - collapse
            ),
            "detected_band_count_augmented_v23": len(band_records),
            "detected_band_count_boundaries_restored_by_v23": restored,
            "row_band_v23_qc_status": (
                "flagged_provisional" if reasons else "pass_no_v23_change"
            ),
            "row_band_v23_qc_flag": bool(reasons),
            "row_band_v23_qc_reason": ";".join(reasons),
            "row_band_v23_value_interpretation": (
                "v2.3 topology includes standard v2.2 singleton rows, strictly "
                "validated small-source or terminal singleton rows, and the "
                "band-local narrow-split correction; all route-specific changes "
                "remain explicitly provisional"
            ),
            "row_geometry_model_version": SCRIPT_VERSION,
            "trajectory_row_band_trait_policy": row_band_audit.get(
                "trajectory_row_band_trait_policy"
            ),
        }
    )
    return result


def image_gwas_trait_summary_v23(
    image_traits: dict[str, Any],
) -> dict[str, Any]:
    summary = parent.image_gwas_trait_summary_v20(image_traits)
    for field in _V23_IMAGE_FIELDS:
        summary[field] = image_traits.get(field)
    summary["row_geometry_model_version"] = SCRIPT_VERSION
    return summary


def add_row_neighbors_v23(
    stoma_traits: list[dict[str, Any]],
    row_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
) -> None:
    parent.add_row_neighbors_v20(stoma_traits, row_records, point_assignment)
    for point_index, assignment in point_assignment.items():
        if not 0 <= int(point_index) < len(stoma_traits):
            continue
        stoma_traits[int(point_index)].update(
            {
                key: value
                for key, value in assignment.items()
                if key.startswith("trajectory_v23_") or key.startswith("v23_")
            }
        )


def make_spatial_audit_overlay_v23(
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
    """Draw the established overlay and label every singleton route explicitly."""

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
    image = base.cv2.imread(str(output_path), base.cv2.IMREAD_COLOR)
    if image is None:
        return
    singleton_rows = [
        row
        for row in row_records
        if (base.optional_int(row.get("stoma_count")) or 0) == 1
        and bool(row.get("trajectory_v20_singleton_rescue_applied"))
    ]
    for row in singleton_rows:
        members = [int(value) for value in row.get("_point_indices", [])]
        if not members:
            continue
        point_index = members[0]
        x = int(round(float(stoma_traits[point_index]["center_x_pixels"])))
        y = int(round(float(stoma_traits[point_index]["center_y_pixels"])))
        singleton_band = bool(row.get("v20_singleton_band"))
        v21_relaxed = bool(
            row.get("trajectory_v21_membership_relaxation_applied")
        )
        leave_one_out = bool(row.get("trajectory_v22_leave_one_out_applied"))
        if leave_one_out:
            route = str(row.get("trajectory_v23_rescue_route") or "")
            if route == "bracketed_3_to_2_plus_1":
                label = f"{row['row_id']} V2.3 BRACKETED SINGLETON ROW"
            elif route == "terminal_refitted_remaining_row":
                label = f"{row['row_id']} V2.3 TERMINAL SINGLETON ROW"
            else:
                label = f"{row['row_id']} V2.2 LOO SINGLETON ROW"
            color = (255, 255, 255)
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
        if leave_one_out:
            base.cv2.circle(image, (x, y), 27, (0, 0, 0), 6, base.cv2.LINE_AA)
        base.cv2.circle(image, (x, y), 24, color, 3, base.cv2.LINE_AA)
        base.cv2.line(image, (x - 20, y), (x + 20, y), color, 2, base.cv2.LINE_AA)
        base.cv2.line(image, (x, y - 20), (x, y + 20), color, 2, base.cv2.LINE_AA)
        text_y = max(22, y - 30)
        if leave_one_out:
            base.cv2.putText(
                image,
                label,
                (max(5, x - 110), text_y),
                base.cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                5,
                base.cv2.LINE_AA,
            )
        base.cv2.putText(
            image,
            label,
            (max(5, x - 110), text_y),
            base.cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            base.cv2.LINE_AA,
        )

    for row in row_records:
        members = [int(value) for value in row.get("_point_indices", [])]
        if members and bool(
            row.get("trajectory_v23_narrow_two_row_collapse_applied")
        ):
            point_index = members[0]
            x = int(round(float(stoma_traits[point_index]["center_x_pixels"])))
            y = int(round(float(stoma_traits[point_index]["center_y_pixels"])))
            base.cv2.putText(
                image,
                f"{row['band_id']} V2.3 NARROW 2-TO-1 COLLAPSE",
                (max(5, x - 80), max(22, y - 24)),
                base.cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                base.cv2.LINE_AA,
            )
    if not base.cv2.imwrite(
        str(output_path), image, [base.cv2.IMWRITE_PNG_COMPRESSION, 6]
    ):
        raise RuntimeError(f"Failed to update v2.3 audit overlay: {output_path}")


def install_v23_policy() -> None:
    """Install v2.2, then replace only the audited v2.3 hooks."""

    parent.install_v20_singleton_policy()

    # Keep all inherited v2.0-v2.2 version labels unchanged.  The wrappers
    # below add the separate v2.3 policy and final row-model identifiers.

    v15._trajectory_diagnostic_fields = _trajectory_diagnostic_fields_v23
    v15.SCRIPT_VERSION = SCRIPT_VERSION
    v14.SCRIPT_VERSION = SCRIPT_VERSION
    v16.SCRIPT_VERSION = SCRIPT_VERSION

    base.build_settings = build_settings_v23
    base.select_row_model = select_row_model_v23
    base.infer_rows_and_bands = infer_rows_and_bands_v23
    base.image_trait_summary = image_trait_summary_v23
    base.image_gwas_trait_summary = image_gwas_trait_summary_v23
    base.add_row_neighbors = add_row_neighbors_v23
    base.make_spatial_audit_overlay = make_spatial_audit_overlay_v23
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base.__file__ = __file__


def run_trait_extraction() -> Path:
    """Run the separate v2.3 audit candidate through the established engine."""

    install_v23_policy()
    return base.run_trait_extraction()


if __name__ == "__main__":
    run_trait_extraction()
