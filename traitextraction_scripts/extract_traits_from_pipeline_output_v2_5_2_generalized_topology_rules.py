#!/usr/bin/env python3
"""Experimental v2.5.2 generalized row/band topology extractor.

This is a separate successor to v2.5.  It does not modify v2.5, v2.5.1,
completed pipeline inference, microscopy files, masks, density eligibility, or
cell-morphometry eligibility.

The balanced development audit identified three general failure modes.  This
version addresses them without image names, stomatal identifiers, manual row
counts, or fixed per-image memberships:

1. A row newly introduced by the v2.5 joint search can be a longitudinal
   fragment of a neighbouring row.  V2.5.2 chooses the valid k-1 model only
   when the change is a single, geometrically continuous fragment merge (or a
   balanced four-row alternative to an unstable five-row capacity solution),
   and the simpler model is within a frozen score window.
2. A strong biological-band boundary can be rejected only because the row
   topology that supplied it was provisional.  After rule 1 has resolved that
   topology, the boundary may pass if every original geometric, stability and
   raw-image gate passed.  A separate narrow route permits a multipoint
   isolated-row boundary with slightly sub-threshold gap width only when its
   local gap ratio, exact-pair raw-image support and perturbation stability are
   all strong.
3. A one-stoma row whose detector bounding box touches the source-image edge
   cannot delimit a row.  It is absorbed into a valid neighbouring row when
   possible; otherwise it is omitted from row/band topology.  The detection
   remains available for density and any independently valid morphometry.

The numerical thresholds below are development-audit values.  They must be
reported as such and checked on an independent confirmation set before this
version is described as final.  Run this script only in a fresh Python process
and always use a new output directory.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import extract_traits_from_pipeline_output_v2_5_joint_rows_standardized_edges as v25


base = v25.base
v14 = v25.v14
v15 = v25.v15
v24 = v25.v24

SCRIPT_VERSION = "2.5.2-experimental-generalized-topology-rules"
SINGLETON_POLICY_VERSION = SCRIPT_VERSION
ROW_GEOMETRY_MODEL_VERSION = SCRIPT_VERSION
PARTIAL_BAND_ELIGIBILITY_VERSION = SCRIPT_VERSION
TRAIT_EXPORT_POLICY_VERSION = v25.TRAIT_EXPORT_POLICY_VERSION

_PARENT_V25_SCRIPT_PATH = Path(v25.__file__).resolve()
_PARENT_APPLY_V25_JOINT_SEARCH = v25._apply_v25_joint_search
_PARENT_PARTITION_CANDIDATE = v25._v25_partition_candidate
_PARENT_INFER_V25 = v25.infer_rows_and_bands_v25
_PARENT_IMAGE_SUMMARY_V25 = v25.image_trait_summary_v25
_PARENT_GWAS_SUMMARY_V25 = v25.image_gwas_trait_summary_v25
_PARENT_OVERLAY_V25 = v25.make_spatial_audit_overlay_v25

# General fragment-continuity rule.  The selected v2.5 solution must have
# added rows relative to its inherited baseline.  The alternative must remove
# exactly one row and remain valid under all v2.5 structural gates.
V252_FRAGMENT_MAX_SCORE_COST = 6.10
V252_FRAGMENT_MIN_SELECTED_ROWS = 3
V252_FRAGMENT_MIN_COMBINED_MEMBERS = 4
V252_FRAGMENT_MAX_LONGITUDINAL_OVERLAP_FRACTION = 0.10
V252_FRAGMENT_MAX_ORDINARY_RMS_WIDTH_FACTOR = 0.15

# At the configured five-row search ceiling, permit a balanced four-row model
# when it contains no sparse rows and is nearly equivalent in score.  This is
# a general protection against capacity-bound over-partitioning; it is not an
# image-specific row-count override.
V252_CAPACITY_SIMPLIFICATION_MAX_SCORE_COST = 5.50
V252_CAPACITY_SIMPLIFICATION_MIN_MEMBERS_PER_ROW = 3

# A narrowly relaxed band-boundary route for an isolated multipoint row.  All
# other v2.5 gates, including exact raw-image evidence, must already pass.
V252_MULTIPOINT_BOUNDARY_MIN_GAP_WIDTH_FACTOR = 1.65
V252_MULTIPOINT_BOUNDARY_MIN_LOCAL_GAP_RATIO = 3.00
V252_MULTIPOINT_BOUNDARY_MIN_IMAGE_SUPPORT_RATIO = 2.00
V252_MULTIPOINT_BOUNDARY_MIN_STABILITY_FRACTION = 0.80
V252_MULTIPOINT_BOUNDARY_MIN_STOMATA_PER_SIDE = 3

# "Touches" means that the detector box reaches the source-image boundary to
# within one pixel.  This is deliberately not a broad near-edge exclusion.
V252_EDGE_TOUCH_TOLERANCE_PIXELS = 1.0


@dataclass(frozen=True)
class TraitSettingsV252(v25.TraitSettingsV25):
    """Immutable v2.5.2 settings written into provenance."""

    v252_policy_version: str = SCRIPT_VERSION
    v252_parent_v25_script_sha256: str = ""
    v252_fragment_max_score_cost: float = V252_FRAGMENT_MAX_SCORE_COST
    v252_fragment_min_selected_rows: int = V252_FRAGMENT_MIN_SELECTED_ROWS
    v252_fragment_min_combined_members: int = V252_FRAGMENT_MIN_COMBINED_MEMBERS
    v252_fragment_max_longitudinal_overlap_fraction: float = (
        V252_FRAGMENT_MAX_LONGITUDINAL_OVERLAP_FRACTION
    )
    v252_fragment_max_ordinary_rms_width_factor: float = (
        V252_FRAGMENT_MAX_ORDINARY_RMS_WIDTH_FACTOR
    )
    v252_capacity_simplification_max_score_cost: float = (
        V252_CAPACITY_SIMPLIFICATION_MAX_SCORE_COST
    )
    v252_capacity_simplification_min_members_per_row: int = (
        V252_CAPACITY_SIMPLIFICATION_MIN_MEMBERS_PER_ROW
    )
    v252_multipoint_boundary_min_gap_width_factor: float = (
        V252_MULTIPOINT_BOUNDARY_MIN_GAP_WIDTH_FACTOR
    )
    v252_multipoint_boundary_min_local_gap_ratio: float = (
        V252_MULTIPOINT_BOUNDARY_MIN_LOCAL_GAP_RATIO
    )
    v252_multipoint_boundary_min_image_support_ratio: float = (
        V252_MULTIPOINT_BOUNDARY_MIN_IMAGE_SUPPORT_RATIO
    )
    v252_multipoint_boundary_min_stability_fraction: float = (
        V252_MULTIPOINT_BOUNDARY_MIN_STABILITY_FRACTION
    )
    v252_multipoint_boundary_min_stomata_per_side: int = (
        V252_MULTIPOINT_BOUNDARY_MIN_STOMATA_PER_SIDE
    )
    v252_edge_touch_tolerance_pixels: float = V252_EDGE_TOUCH_TOLERANCE_PIXELS


def build_settings_v252(args: Any) -> TraitSettingsV252:
    inherited = asdict(v25.build_settings_v25(args))
    inherited.update(
        {
            "v252_policy_version": SCRIPT_VERSION,
            "v252_parent_v25_script_sha256": hashlib.sha256(
                _PARENT_V25_SCRIPT_PATH.read_bytes()
            ).hexdigest(),
        }
    )
    return TraitSettingsV252(**inherited)


_V252_ROW_DEFAULTS: dict[str, Any] = {
    "trajectory_v252_policy_version": SCRIPT_VERSION,
    "trajectory_v252_general_row_simplification_applied": False,
    "trajectory_v252_general_row_simplification_route": "not_applied",
    "trajectory_v252_general_row_simplification_score_cost": None,
    "trajectory_v252_general_row_simplification_geometry": {},
    "trajectory_v252_edge_touching_singleton_rule_applied": False,
    "trajectory_v252_edge_touching_singleton_absorbed_count": 0,
    "trajectory_v252_edge_touching_singleton_absorbed": [],
    "trajectory_v252_edge_touching_singleton_pending_topology_removal_count": 0,
    "trajectory_v252_edge_touching_singleton_pending_topology_removal": [],
    "trajectory_v252_edge_touch_tolerance_pixels": (
        V252_EDGE_TOUCH_TOLERANCE_PIXELS
    ),
}


def _edge_touch_details(point_index: int, settings: TraitSettingsV252) -> dict[str, Any]:
    record = v25._active_record(int(point_index))
    metadata = v14._ACTIVE_IMAGE_METADATA or {}
    if record is None:
        return {"assessable": False, "touches": False, "sides": []}
    width = base.optional_float(metadata.get("image_width"))
    height = base.optional_float(metadata.get("image_height"))
    x1 = base.optional_float(record.get("prediction_x1_pixels"))
    y1 = base.optional_float(record.get("prediction_y1_pixels"))
    x2 = base.optional_float(record.get("prediction_x2_pixels"))
    y2 = base.optional_float(record.get("prediction_y2_pixels"))
    assessable = all(value is not None for value in (width, height, x1, y1, x2, y2))
    sides: list[str] = []
    if assessable:
        tolerance = float(settings.v252_edge_touch_tolerance_pixels)
        if float(x1) <= tolerance:
            sides.append("left")
        if float(x2) >= float(width) - tolerance:
            sides.append("right")
        if float(y1) <= tolerance:
            sides.append("top")
        if float(y2) >= float(height) - tolerance:
            sides.append("bottom")
    return {
        "assessable": assessable,
        "touches": bool(sides),
        "sides": sides,
        "tolerance_pixels": float(settings.v252_edge_touch_tolerance_pixels),
        "prediction_x1_pixels": x1,
        "prediction_y1_pixels": y1,
        "prediction_x2_pixels": x2,
        "prediction_y2_pixels": y2,
        "image_width_pixels": width,
        "image_height_pixels": height,
    }


def _candidate_records(
    candidates: list[dict[str, Any]],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV252,
) -> list[dict[str, Any]]:
    retained: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in candidates:
        record = v25._evaluate_joint_candidate(
            candidate, lateral, longitudinal, median_width_um, settings
        )
        key = (
            record["canonical_partition"],
            round(float(record["slope_angle_deg"]), 10),
        )
        previous = retained.get(key)
        if previous is None or float(record["adjusted_score"]) < float(
            previous["adjusted_score"]
        ):
            retained[key] = record
    return list(retained.values())


def _clean_single_fragment_merge(
    selected: dict[str, Any], simpler: dict[str, Any]
) -> tuple[int, int] | None:
    """Return the two selected groups joined by a pure k -> k-1 merge.

    Every selected group must remain intact, all unaffected groups must be
    identical, and exactly two adjacent lateral groups must map to one simpler
    group.  Any broad reassignment is rejected by this route.
    """

    selected_groups = [set(group) for group in v25._candidate_group_members(selected)]
    simpler_groups = [set(group) for group in v25._candidate_group_members(simpler)]
    owners: list[int] = []
    for group in selected_groups:
        matches = [index for index, target in enumerate(simpler_groups) if group <= target]
        if len(matches) != 1:
            return None
        owners.append(matches[0])
    counts = [owners.count(index) for index in range(len(simpler_groups))]
    if counts.count(2) != 1 or any(value not in {1, 2} for value in counts):
        return None
    merged_owner = counts.index(2)
    pair = [index for index, owner in enumerate(owners) if owner == merged_owner]
    if len(pair) != 2 or pair[1] != pair[0] + 1:
        return None
    for owner, target in enumerate(simpler_groups):
        components = [
            selected_groups[index]
            for index, assigned_owner in enumerate(owners)
            if assigned_owner == owner
        ]
        if not components or set().union(*components) != target:
            return None
    return pair[0], pair[1]


def _fragment_continuity_geometry(
    selected: dict[str, Any],
    simpler_record: dict[str, Any],
    pair: tuple[int, int],
    longitudinal: np.ndarray,
    settings: TraitSettingsV252,
) -> dict[str, Any]:
    groups = v25._candidate_group_members(selected)
    first = groups[pair[0]]
    second = groups[pair[1]]
    first_long = np.asarray([float(longitudinal[index]) for index in first], dtype=float)
    second_long = np.asarray([float(longitudinal[index]) for index in second], dtype=float)
    overlap = max(
        0.0,
        min(float(np.max(first_long)), float(np.max(second_long)))
        - max(float(np.min(first_long)), float(np.min(second_long))),
    )
    overlap_reference = max(
        min(float(np.ptp(first_long)), float(np.ptp(second_long))), 1e-9
    )
    overlap_fraction = overlap / overlap_reference
    combined_members = len(first) + len(second)
    maximum_rms = float(simpler_record["maximum_ordinary_row_rms_width_factor"])
    failures: list[str] = []
    if combined_members < int(settings.v252_fragment_min_combined_members):
        failures.append("too_few_combined_members")
    if overlap_fraction > float(
        settings.v252_fragment_max_longitudinal_overlap_fraction
    ):
        failures.append("fragments_overlap_longitudinally")
    if maximum_rms > float(settings.v252_fragment_max_ordinary_rms_width_factor):
        failures.append("simpler_model_row_rms")
    return {
        "selected_group_pair": list(pair),
        "selected_fragment_member_counts": [len(first), len(second)],
        "combined_member_count": combined_members,
        "longitudinal_overlap_fraction": overlap_fraction,
        "simpler_maximum_ordinary_row_rms_width_factor": maximum_rms,
        "passed": not failures,
        "failed_gates": failures,
    }


def _choose_general_simplification(
    chosen: dict[str, Any],
    records: list[dict[str, Any]],
    longitudinal: np.ndarray,
    settings: TraitSettingsV252,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]] | None:
    selected_record = next(
        (
            record
            for record in records
            if record["canonical_partition"] == v25._canonical_partition(chosen)
            and math.isclose(
                float(record["slope_angle_deg"]),
                math.degrees(math.atan(float(chosen.get("slope") or 0.0))),
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        ),
        None,
    )
    if selected_record is None:
        return None
    selected_k = int(chosen.get("k") or 0)
    baseline_k = base.optional_int(chosen.get("trajectory_v25_baseline_row_count"))
    if not bool(chosen.get("trajectory_v25_joint_change_applied")):
        return None
    if baseline_k is None or selected_k <= baseline_k:
        return None
    alternatives = [
        record
        for record in records
        if bool(record.get("valid"))
        and int(record["candidate_row_count"]) == selected_k - 1
    ]
    if not alternatives:
        return None
    simpler_record = min(
        alternatives,
        key=lambda record: (
            float(record["adjusted_score"]),
            int(record["singleton_row_count"]),
            abs(float(record["slope_angle_deg"])),
        ),
    )
    score_cost = float(simpler_record["adjusted_score"]) - float(
        selected_record["adjusted_score"]
    )

    if selected_k >= int(settings.v252_fragment_min_selected_rows):
        pair = _clean_single_fragment_merge(chosen, simpler_record["candidate"])
        if pair is not None:
            geometry = _fragment_continuity_geometry(
                chosen, simpler_record, pair, longitudinal, settings
            )
            if (
                bool(geometry["passed"])
                and score_cost <= float(settings.v252_fragment_max_score_cost)
            ):
                return (
                    simpler_record["candidate"],
                    simpler_record,
                    "longitudinal_fragment_continuity",
                    {**geometry, "score_cost": score_cost},
                )

    selected_sizes = [len(group) for group in v25._candidate_group_members(chosen)]
    simpler_sizes = [
        len(group) for group in v25._candidate_group_members(simpler_record["candidate"])
    ]
    capacity = int(settings.trajectory_max_rows_per_supergroup)
    capacity_passed = bool(
        selected_k == capacity
        and all(value >= 2 for value in selected_sizes)
        and all(
            value >= int(settings.v252_capacity_simplification_min_members_per_row)
            for value in simpler_sizes
        )
        and score_cost
        <= float(settings.v252_capacity_simplification_max_score_cost)
        and float(simpler_record["maximum_ordinary_row_rms_width_factor"])
        <= float(settings.v252_fragment_max_ordinary_rms_width_factor)
    )
    if capacity_passed:
        return (
            simpler_record["candidate"],
            simpler_record,
            "balanced_capacity_bound_simplification",
            {
                "selected_row_count": selected_k,
                "selected_cluster_sizes": selected_sizes,
                "simpler_cluster_sizes": simpler_sizes,
                "simpler_maximum_ordinary_row_rms_width_factor": float(
                    simpler_record["maximum_ordinary_row_rms_width_factor"]
                ),
                "score_cost": score_cost,
                "passed": True,
                "failed_gates": [],
            },
        )
    return None


def _refresh_selected_fields(
    chosen: dict[str, Any],
    diagnostic: dict[str, Any],
    records: list[dict[str, Any]],
    selected_record: dict[str, Any],
    reason: str,
    median_width_um: float,
    settings: TraitSettingsV252,
) -> None:
    groups = v25._candidate_group_members(chosen)
    singleton_points = [group[0] for group in groups if len(group) == 1]
    singleton_ids = [v25._point_stoma_id(point) for point in singleton_points]
    fields = {
        "trajectory_v25_selected_row_count": int(chosen["k"]),
        "trajectory_v25_selected_cluster_sizes": [len(group) for group in groups],
        "trajectory_v25_selected_singleton_count": len(singleton_points),
        "trajectory_v25_selected_singleton_point_indices": singleton_points,
        "trajectory_v25_selected_singleton_stoma_ids": singleton_ids,
        "trajectory_v25_supported_singleton_count": len(singleton_points),
        "trajectory_v25_supported_singleton_stoma_ids": singleton_ids,
        "trajectory_v25_ambiguous_singleton_count": 0,
        "trajectory_v25_ambiguous_singleton_stoma_ids": [],
        "trajectory_v25_row_count_lower_bound": int(chosen["k"]),
        "trajectory_v25_row_count_upper_bound": int(chosen["k"]),
        "trajectory_v25_topology_confidence_tier": (
            "supported_by_v252_general_geometry"
        ),
        "automatic_row_count": int(chosen["k"]),
        "automatic_selection_reason": reason,
        "row_count_selection_reason": reason,
        "audited_row_count_override": False,
        "model_confidence": "moderate",
        "trajectory_row_count_min": int(chosen["k"]),
        "trajectory_row_count_max": int(chosen["k"]),
        "trajectory_row_count_stable": True,
        "trajectory_exact_row_traits_eligible": True,
        "trajectory_singleton_validation_required": bool(singleton_points),
        "trajectory_singleton_rows_validated": True,
        "trajectory_singleton_detector_confidences": [
            v25._point_confidence(point) for point in singleton_points
        ],
        "trajectory_singleton_validation_failure_reason": "",
        "trajectory_v252_policy_version": SCRIPT_VERSION,
    }
    chosen.update(fields)
    diagnostic.update(fields)
    diagnostic.update(
        {
            "global_automatic_row_count": int(chosen["k"]),
            "row_count_used_for_traits": int(chosen["k"]),
            "row_count_used_for_traits_source": reason,
            "global_row_selection_score": float(selected_record["adjusted_score"]),
            "global_row_model_confidence": "moderate",
            "candidate_diagnostics": v25._parent_compatible_candidate_diagnostics(
                records, selected_record, median_width_um, settings
            ),
            "trajectory_v25_candidate_diagnostics": v25._top_candidate_diagnostics(
                records, selected_record
            ),
        }
    )
    chosen["trajectory_v25_candidate_diagnostics"] = diagnostic[
        "trajectory_v25_candidate_diagnostics"
    ]


def _apply_v25_joint_search_v252(
    parent_chosen: dict[str, Any],
    parent_candidates: list[dict[str, Any]],
    diagnostic: dict[str, Any],
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV252,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    chosen, candidates, diagnostic = _PARENT_APPLY_V25_JOINT_SEARCH(
        parent_chosen,
        parent_candidates,
        diagnostic,
        point_indices,
        lateral,
        longitudinal,
        median_width_um,
        settings,
    )
    records = _candidate_records(
        [*candidates, chosen], lateral, longitudinal, median_width_um, settings
    )

    absorbed: list[dict[str, Any]] = []
    while True:
        edge_singletons = [
            (point, _edge_touch_details(point, settings))
            for point in v25._singleton_points(chosen)
            if _edge_touch_details(point, settings)["touches"]
        ]
        if not edge_singletons:
            break
        changed = False
        for point, details in edge_singletons:
            revised = v25._absorb_singleton_into_nearest_supported_row(
                chosen, int(point), lateral, longitudinal
            )
            if revised is None:
                continue
            revised_record = v25._evaluate_joint_candidate(
                revised, lateral, longitudinal, median_width_um, settings
            )
            if not bool(revised_record["valid"]):
                continue
            chosen = dict(chosen)
            chosen.update(revised)
            records.append(revised_record)
            absorbed.append(
                {
                    "point_index": int(point),
                    "stoma_id": v25._point_stoma_id(int(point)),
                    "edge_touch": details,
                }
            )
            changed = True
        if not changed:
            break

    simplification = None
    if not absorbed:
        simplification = _choose_general_simplification(
            chosen, records, longitudinal, settings
        )
    if simplification is not None:
        replacement, selected_record, route, geometry = simplification
        chosen = dict(chosen)
        chosen.update(replacement)
        general_fields = {
            "trajectory_v252_policy_version": SCRIPT_VERSION,
            "trajectory_v252_general_row_simplification_applied": True,
            "trajectory_v252_general_row_simplification_route": route,
            "trajectory_v252_general_row_simplification_score_cost": float(
                geometry["score_cost"]
            ),
            "trajectory_v252_general_row_simplification_geometry": geometry,
        }
        chosen.update(general_fields)
        diagnostic.update(general_fields)
        _refresh_selected_fields(
            chosen,
            diagnostic,
            records,
            selected_record,
            f"v252_{route}",
            median_width_um,
            settings,
        )

    remaining = [
        {
            "point_index": int(point),
            "stoma_id": v25._point_stoma_id(int(point)),
            "edge_touch": _edge_touch_details(int(point), settings),
        }
        for point in v25._singleton_points(chosen)
        if _edge_touch_details(int(point), settings)["touches"]
    ]
    edge_fields = {
        "trajectory_v252_policy_version": SCRIPT_VERSION,
        "trajectory_v252_edge_touching_singleton_rule_applied": bool(
            absorbed or remaining
        ),
        "trajectory_v252_edge_touching_singleton_absorbed_count": len(absorbed),
        "trajectory_v252_edge_touching_singleton_absorbed": absorbed,
        "trajectory_v252_edge_touching_singleton_pending_topology_removal_count": len(
            remaining
        ),
        "trajectory_v252_edge_touching_singleton_pending_topology_removal": remaining,
        "trajectory_v252_edge_touch_tolerance_pixels": float(
            settings.v252_edge_touch_tolerance_pixels
        ),
    }
    chosen.update(edge_fields)
    diagnostic.update(edge_fields)
    if absorbed:
        selected_record = v25._evaluate_joint_candidate(
            chosen, lateral, longitudinal, median_width_um, settings
        )
        records.append(selected_record)
        _refresh_selected_fields(
            chosen,
            diagnostic,
            records,
            selected_record,
            "v252_edge_touching_singleton_absorbed_into_supported_row",
            median_width_um,
            settings,
        )
        chosen.update(edge_fields)
        diagnostic.update(edge_fields)

    best_by_k: dict[int, dict[str, Any]] = {
        int(candidate["k"]): candidate for candidate in candidates
    }
    best_by_k[int(chosen["k"])] = chosen
    return chosen, [best_by_k[key] for key in sorted(best_by_k)], diagnostic


def _v25_partition_candidate_v252(
    band: dict[str, Any],
    linked: list[dict[str, Any]],
    cut: int,
    gaps: list[float],
    boundary_evidence_by_rows: dict[tuple[str, str, str], dict[str, Any]],
    median_width_um: float,
    settings: TraitSettingsV252,
) -> dict[str, Any]:
    result = _PARENT_PARTITION_CANDIDATE(
        band,
        linked,
        cut,
        gaps,
        boundary_evidence_by_rows,
        median_width_um,
        settings,
    )
    if bool(result.get("accepted")):
        return result
    failures = set(str(value) for value in result.get("failed_gates", []))

    # Row topology was provisional only because v2.5 had introduced a
    # fragment.  Once the general continuity rule resolves that fragment, a
    # boundary that passes every other original v2.5 gate can be retained.
    rows_resolved_by_general_rule = bool(
        linked
        and all(
            bool(row.get("trajectory_v252_general_row_simplification_applied"))
            for row in linked
        )
    )
    topology_only = bool(
        failures == {"row_topology_not_supported"}
        and rows_resolved_by_general_rule
    )

    # One isolated multipoint row may have a moderately narrower physical gap
    # only when the local contrast and exact raw-image boundary evidence are
    # strong.  This route never applies to a singleton side.
    relaxed_multipoint_gap = bool(
        failures == {"gap_width"}
        and bool(result.get("isolated_one_row_side"))
        and min(
            int(result.get("left_stomata") or 0),
            int(result.get("right_stomata") or 0),
        )
        >= int(settings.v252_multipoint_boundary_min_stomata_per_side)
        and float(result.get("gap_factor") or 0.0)
        >= float(settings.v252_multipoint_boundary_min_gap_width_factor)
        and float(result.get("gap_ratio") or 0.0)
        >= float(settings.v252_multipoint_boundary_min_local_gap_ratio)
        and bool(result.get("image_support_identity_exact"))
        and result.get("image_support") is not None
        and float(result["image_support"])
        >= float(settings.v252_multipoint_boundary_min_image_support_ratio)
        and float(result.get("stability") or 0.0)
        >= float(settings.v252_multipoint_boundary_min_stability_fraction)
        and not bool(result.get("nonlinear_row_topology"))
    )
    if not (topology_only or relaxed_multipoint_gap):
        return result

    result.update(
        {
            "accepted": True,
            "failed_gates_before_v252_general_rule": sorted(failures),
            "failed_gates": [],
            "trajectory_v252_general_band_boundary_applied": True,
            "trajectory_v252_general_band_boundary_route": (
                "resolved_row_topology_all_other_v25_gates_passed"
                if topology_only
                else "strong_exact_image_supported_multipoint_gap"
            ),
            "decision_route": (
                "v252_resolved_row_topology"
                if topology_only
                else "v252_strong_exact_image_supported_multipoint_gap"
            ),
        }
    )
    return result


def _remaining_edge_singleton_rows(
    rows: list[dict[str, Any]], settings: TraitSettingsV252
) -> list[tuple[dict[str, Any], int, dict[str, Any]]]:
    result: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
    for row in rows:
        members = [int(value) for value in row.get("_point_indices", [])]
        if len(members) != 1:
            continue
        details = _edge_touch_details(members[0], settings)
        if details["touches"]:
            result.append((row, members[0], details))
    return result


def _remove_unabsorbed_edge_singletons_from_topology(
    rows: list[dict[str, Any]],
    bands: list[dict[str, Any]],
    assignments: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    audit: dict[str, Any],
    settings: TraitSettingsV252,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    candidates = _remaining_edge_singleton_rows(rows, settings)
    if not candidates:
        audit["trajectory_v252_edge_touching_singleton_removed_from_topology_count"] = 0
        audit["trajectory_v252_edge_touching_singleton_removed_from_topology"] = []
        return rows, bands, assignments, audit

    removed_ids = {str(row["row_id"]) for row, _, _ in candidates}
    removed_records: list[dict[str, Any]] = []
    for row, point_index, details in candidates:
        row_id = str(row["row_id"])
        former_band = str(row.get("band_id") or "")
        marker = {
            "trajectory_v252_policy_version": SCRIPT_VERSION,
            "trajectory_v252_edge_touching_singleton_ignored_for_row_delimitation": True,
            "trajectory_v252_edge_touching_singleton_edge_sides": details["sides"],
            "trajectory_v252_edge_touch_tolerance_pixels": float(
                settings.v252_edge_touch_tolerance_pixels
            ),
            "trajectory_v252_topology_exclusion_reason": (
                "singleton_detector_box_touches_image_edge"
            ),
            "row_id": "",
            "band_id": "",
            "row_band_spatial_eligible": False,
            "row_complete_topology_eligible": False,
            "row_structure_eligible": False,
            "row_band_spatial_exclusion_reason": (
                "singleton_detector_box_touches_image_edge"
            ),
        }
        assignments[int(point_index)].update(marker)
        stoma_traits[int(point_index)].update(marker)
        removed_records.append(
            {
                "stoma_id": str(stoma_traits[int(point_index)].get("stoma_id") or ""),
                "former_row_id": row_id,
                "former_band_id": former_band,
                "edge_sides": details["sides"],
            }
        )

    rows = [row for row in rows if str(row["row_id"]) not in removed_ids]
    row_lookup = {str(row["row_id"]): row for row in rows}
    lower_before = base.optional_int(audit.get("detected_row_count_v25_lower_bound"))
    upper_before = base.optional_int(audit.get("detected_row_count_v25_upper_bound"))
    median_widths = [
        base.optional_float(record.get("prediction_width_um"))
        for record in stoma_traits
    ]
    median_widths = [value for value in median_widths if value is not None and value > 0]
    median_width_um = float(np.median(np.asarray(median_widths, dtype=float))) if median_widths else 1.0
    rebuilt_bands: list[dict[str, Any]] = []
    for band in bands:
        linked = [
            row_lookup[str(row_id)]
            for row_id in band.get("row_ids", [])
            if str(row_id) in row_lookup
        ]
        if not linked:
            continue
        rebuilt_bands.append(
            v24._rebuild_band_v24(
                band,
                linked,
                stoma_traits,
                median_width_um,
                None,
                1,
                {
                    "trajectory_v252_edge_touching_singleton_removed_from_band": True,
                    "trajectory_v252_policy_version": SCRIPT_VERSION,
                },
            )
        )
    rows, rebuilt_bands, assignments = v24._renumber_final_bands(
        rows, rebuilt_bands, assignments
    )
    parent_edge_audit = (
        audit.get("band_spatial_eligibility")
        if isinstance(audit.get("band_spatial_eligibility"), list)
        else []
    )
    edge_audit = v25._apply_v25_standardized_multirow_edge_rule(
        rows,
        rebuilt_bands,
        assignments,
        stoma_traits,
        settings,
        parent_edge_audit,
    )
    removed_count = len(removed_records)
    supported_singletons = sum(
        str(row.get("trajectory_v25_singleton_evidence_tier") or "") == "supported"
        for row in rows
    )
    ambiguous_singletons = sum(
        str(row.get("trajectory_v25_singleton_evidence_tier") or "") == "ambiguous"
        for row in rows
    )
    audit.update(
        {
            "trajectory_v252_edge_touching_singleton_removed_from_topology_count": (
                removed_count
            ),
            "trajectory_v252_edge_touching_singleton_removed_from_topology": (
                removed_records
            ),
            "final_detected_row_count": len(rows),
            "final_detected_band_count": len(rebuilt_bands),
            "detected_row_count_v25": len(rows),
            "detected_band_count_v25": len(rebuilt_bands),
            "detected_row_count_v25_lower_bound": max(
                0,
                lower_before - removed_count
                if lower_before is not None
                else len(rows),
            ),
            "detected_row_count_v25_upper_bound": max(
                0,
                upper_before - removed_count
                if upper_before is not None
                else len(rows),
            ),
            "trajectory_v25_supported_singleton_row_count": supported_singletons,
            "trajectory_v25_ambiguous_singleton_row_count": ambiguous_singletons,
            "spatially_eligible_band_count": sum(
                base.parse_bool(band.get("band_spatial_eligible")) is True
                for band in rebuilt_bands
            ),
            "spatially_excluded_band_count": sum(
                base.parse_bool(band.get("band_spatial_eligible")) is not True
                for band in rebuilt_bands
            ),
            "spatially_eligible_row_count": sum(
                base.parse_bool(row.get("row_band_spatial_eligible")) is True
                for row in rows
            ),
            "spatially_excluded_row_count": sum(
                base.parse_bool(row.get("row_band_spatial_eligible")) is not True
                for row in rows
            ),
            "band_spatial_eligibility": edge_audit,
        }
    )
    return rows, rebuilt_bands, assignments, audit


def infer_rows_and_bands_v252(
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettingsV252,
    audited_row_counts: dict[str, int] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    rows, bands, assignments, audit = _PARENT_INFER_V25(
        stoma_traits,
        settings,
        audited_row_counts=audited_row_counts,
    )
    rows, bands, assignments, audit = _remove_unabsorbed_edge_singletons_from_topology(
        rows, bands, assignments, stoma_traits, audit, settings
    )
    diagnostics = list(audit.get("band_selection_diagnostics", {}).values())
    general_row_changes = sum(
        bool(record.get("trajectory_v252_general_row_simplification_applied"))
        for record in diagnostics
    )
    boundary_diagnostics = audit.get("band_partition_diagnostics", [])
    general_boundary_changes = sum(
        bool(record.get("trajectory_v252_general_band_boundary_applied"))
        for record in boundary_diagnostics
    )
    audit.update(
        {
            "row_model_version": SCRIPT_VERSION,
            "trajectory_v252_policy_version": SCRIPT_VERSION,
            "trajectory_v252_general_row_simplification_count": general_row_changes,
            "trajectory_v252_general_band_boundary_count": general_boundary_changes,
            "automatic_row_count_policy": (
                "v2.5 joint topology followed by general longitudinal-fragment "
                "and balanced-capacity simplification rules; no image-specific "
                "memberships or row-count overrides"
            ),
            "band_partition_policy": (
                "v2.5 boundary rules plus two general evidence routes: resolved "
                "fragment topology with every other v2.5 gate passed, or a "
                "multipoint isolated-row cut with strong gap ratio, stability "
                "and exact-pair raw-image support"
            ),
            "trajectory_row_band_trait_policy": (
                "edge-touching singleton detector boxes cannot delimit rows; "
                "their detections remain available for density and independently "
                "valid morphometry"
            ),
        }
    )
    for collection in (rows, bands, list(assignments.values())):
        for record in collection:
            record["row_model_version"] = SCRIPT_VERSION
            record["trajectory_v252_policy_version"] = SCRIPT_VERSION
            record["partial_band_eligibility_version"] = SCRIPT_VERSION
    return rows, bands, assignments, audit


def image_trait_summary_v252(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = _PARENT_IMAGE_SUMMARY_V25(*args, **kwargs)
    audit = args[7] if len(args) > 7 else kwargs.get("row_band_audit", {})
    result.update(
        {
            "row_geometry_model_version": SCRIPT_VERSION,
            "trajectory_v252_policy_version": SCRIPT_VERSION,
            "general_row_simplification_count_v252": int(
                audit.get("trajectory_v252_general_row_simplification_count", 0) or 0
            ),
            "general_band_boundary_count_v252": int(
                audit.get("trajectory_v252_general_band_boundary_count", 0) or 0
            ),
            "edge_touching_singletons_excluded_from_row_topology": int(
                audit.get(
                    "trajectory_v252_edge_touching_singleton_removed_from_topology_count",
                    0,
                )
                or 0
            ),
        }
    )
    return result


def image_gwas_trait_summary_v252(image_traits: dict[str, Any]) -> dict[str, Any]:
    result = _PARENT_GWAS_SUMMARY_V25(image_traits)
    for field in (
        "trajectory_v252_policy_version",
        "general_row_simplification_count_v252",
        "general_band_boundary_count_v252",
        "edge_touching_singletons_excluded_from_row_topology",
    ):
        result[field] = image_traits.get(field)
    result["row_geometry_model_version"] = SCRIPT_VERSION
    return result


def make_spatial_audit_overlay_v252(
    input_dir: Path,
    output_path: Path,
    image_metadata: dict[str, Any],
    stoma_traits: list[dict[str, Any]],
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    axis: Any,
    px_per_um_x: float,
    px_per_um_y: float,
) -> None:
    _PARENT_OVERLAY_V25(
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
    for record in stoma_traits:
        if not bool(
            record.get(
                "trajectory_v252_edge_touching_singleton_ignored_for_row_delimitation"
            )
        ):
            continue
        x = int(round(float(record["center_x_pixels"])))
        y = int(round(float(record["center_y_pixels"])))
        color = (0, 0, 255)
        base.cv2.line(image, (x - 22, y - 22), (x + 22, y + 22), color, 4)
        base.cv2.line(image, (x - 22, y + 22), (x + 22, y - 22), color, 4)
        base.cv2.putText(
            image,
            "EDGE SINGLETON: NOT A ROW",
            (max(5, x - 125), min(image.shape[0] - 8, y + 40)),
            base.cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            2,
            base.cv2.LINE_AA,
        )
    if not base.cv2.imwrite(
        str(output_path), image, [base.cv2.IMWRITE_PNG_COMPRESSION, 6]
    ):
        raise RuntimeError(f"Failed to update v2.5.2 audit overlay: {output_path}")


def _update_v252_output_provenance(output_dir: Path) -> None:
    provenance_path = output_dir / "trait_extraction_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance.update(
            {
                "script_version": SCRIPT_VERSION,
                "v252_parent_script": str(_PARENT_V25_SCRIPT_PATH),
                "v252_parent_script_sha256": hashlib.sha256(
                    _PARENT_V25_SCRIPT_PATH.read_bytes()
                ).hexdigest(),
                "v252_uses_v251": False,
                "v252_contains_image_specific_rules": False,
                "v252_policy": (
                    "general geometry-based row simplification, general "
                    "evidence-based band boundaries, and exclusion of an "
                    "edge-touching singleton box from row delimitation"
                ),
                "v252_development_status": (
                    "thresholds selected from the development audit; confirmation "
                    "on an independent balanced set is required before final lock"
                ),
            }
        )
        provenance_path.write_text(
            json.dumps(base.json_safe(provenance), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    report_path = output_dir / "TRAIT_EXTRACTION_REPORT.txt"
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8").rstrip()
        report += (
            "\nV2.5.2 uses v2.5 directly and does not use v2.5.1.\n"
            "No image name, stomatal identifier, fixed membership or manual row "
            "count is used by v2.5.2.\n"
            "A singleton detector box touching the source-image edge cannot "
            "delimit a row; density and independently valid morphometry are "
            "unchanged.\n"
        )
        report_path.write_text(report, encoding="utf-8")


def install_v252_policy() -> None:
    """Install v2.5, then replace only the generalized v2.5.2 hooks."""

    v25.install_v25_policy()
    v25._V25_DEFAULTS.update(_V252_ROW_DEFAULTS)
    v25._apply_v25_joint_search = _apply_v25_joint_search_v252
    v25._v25_partition_candidate = _v25_partition_candidate_v252

    base.build_settings = build_settings_v252
    base.infer_rows_and_bands = infer_rows_and_bands_v252
    base.image_trait_summary = image_trait_summary_v252
    base.image_gwas_trait_summary = image_gwas_trait_summary_v252
    base.make_spatial_audit_overlay = make_spatial_audit_overlay_v252
    for module in (v14, v15, base):
        module.SCRIPT_VERSION = SCRIPT_VERSION
    base.__file__ = __file__


def run_trait_extraction() -> Path:
    install_v252_policy()
    output_dir = base.run_trait_extraction()
    v25._write_v25_additional_qc_tables(output_dir)
    v25._update_v25_output_provenance(output_dir)
    _update_v252_output_provenance(output_dir)
    return output_dir


if __name__ == "__main__":
    run_trait_extraction()
