#!/usr/bin/env python3
"""Experimental v2.5.3 confirmation-audit generalized topology extractor.

This is a separate successor to v2.5.2.  It preserves the completed detector,
classifier and segmentor outputs and changes only row/band topology.  No image
name, stomatal identifier, fixed membership or manual row count is used.

The confirmation audit exposed four general failure modes:

1. A low-stability extra row can be a short pseudo-row whose members continue
   two already established rows.  A guarded k -> k-1 alternative is accepted
   only when exactly one small selected group redistributes into two intact
   established rows, all other memberships are preserved, perturbation
   stability is poor, and the valid simpler model is within a frozen score
   and residual window.
2. Two multipoint groups can be longitudinally disjoint fragments of one row.
   A higher score window is allowed only for a pure adjacent merge in which
   both fragments contain at least two stomata, all selected groups are
   nonsingletons, longitudinal overlap is negligible, and the simpler model
   passes the established residual gate.
3. A strong biological-band boundary can be vetoed solely because otherwise
   plausible row topology is labelled ambiguous.  Such a boundary is retained
   only when all original geometric and stability gates pass and exact-pair
   raw-image support is strong.  Separately, when v2.5.3 resolves the row
   topology itself, an isolated multipoint boundary may cross a narrowly
   missed parent raw-image-support cutoff only within its own stricter frozen
   geometry, stability, membership and exact-pair support gates.
4. Only a fully visible stoma may designate a one-stoma row.  After final row
   assignment, a row containing exactly one detector box that touches any
   source-image boundary is removed from row and band topology.  Boundary-
   touching detections that naturally belong to a multi-stoma row remain.
   Density and independently valid morphometry are never changed.

All thresholds below are confirmation-audit development values.  They require
validation on a new independent balanced image set before final scientific
lock.  Run this script in a fresh Python process and use a new output folder.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import extract_traits_from_pipeline_output_v2_5_2_generalized_topology_rules as v252


v25 = v252.v25
base = v252.base
v14 = v252.v14
v15 = v252.v15
v24 = v252.v24

SCRIPT_VERSION = "2.5.3-experimental-confirmation-generalized-topology"
SINGLETON_POLICY_VERSION = SCRIPT_VERSION
ROW_GEOMETRY_MODEL_VERSION = SCRIPT_VERSION
PARTIAL_BAND_ELIGIBILITY_VERSION = SCRIPT_VERSION
TRAIT_EXPORT_POLICY_VERSION = v252.TRAIT_EXPORT_POLICY_VERSION

_PARENT_V252_SCRIPT_PATH = Path(v252.__file__).resolve()
_PARENT_INFER_V252 = v252.infer_rows_and_bands_v252
_PARENT_IMAGE_SUMMARY_V252 = v252.image_trait_summary_v252
_PARENT_GWAS_SUMMARY_V252 = v252.image_gwas_trait_summary_v252
_PARENT_OVERLAY_V252 = v252.make_spatial_audit_overlay_v252

# Guarded redistribution of one short pseudo-row into two established rows.
V253_REDISTRIBUTION_MAX_SCORE_COST = 4.50
V253_REDISTRIBUTION_MAX_SPLIT_GROUP_MEMBERS = 3
V253_REDISTRIBUTION_MAX_ROW_COUNT_STABILITY = 0.60
V253_REDISTRIBUTION_MAX_ASSIGNMENT_STABILITY = 0.50
V253_REDISTRIBUTION_MAX_ORDINARY_RMS_WIDTH_FACTOR = 0.14

# High-cost fragment merge is permitted only for fully multipoint fragments.
V253_MULTIPOINT_FRAGMENT_MAX_SCORE_COST = 9.25
V253_MULTIPOINT_FRAGMENT_MIN_MEMBERS_EACH = 2
V253_MULTIPOINT_FRAGMENT_MAX_LONGITUDINAL_OVERLAP_FRACTION = 0.10
V253_MULTIPOINT_FRAGMENT_MAX_ORDINARY_RMS_WIDTH_FACTOR = 0.14

# Strong ambiguous-topology boundary rescue.  Every original non-topology gate
# must already have passed; these values add exact-pair raw-image corroboration.
V253_AMBIGUOUS_BOUNDARY_MIN_GAP_WIDTH_FACTOR = 3.00
V253_AMBIGUOUS_BOUNDARY_MIN_LOCAL_GAP_RATIO = 4.00
V253_AMBIGUOUS_BOUNDARY_MIN_IMAGE_SUPPORT_RATIO = 1.80
V253_AMBIGUOUS_BOUNDARY_MIN_STABILITY_FRACTION = 0.90
V253_AMBIGUOUS_BOUNDARY_MIN_STOMATA_PER_SIDE = 4
V253_AMBIGUOUS_BOUNDARY_MIN_ROW_COUNT_STABILITY = 0.90
V253_AMBIGUOUS_BOUNDARY_MIN_ASSIGNMENT_STABILITY = 0.75
V253_RESOLVED_BOUNDARY_MIN_IMAGE_SUPPORT_RATIO = 1.60

# A detector box at or within one source pixel of an image edge is not fully
# visible.  This is applied only when that detection is the sole row member.
V253_SINGLETON_FULL_VISIBILITY_TOLERANCE_PIXELS = 1.0


@dataclass(frozen=True)
class TraitSettingsV253(v252.TraitSettingsV252):
    """Immutable v2.5.3 settings written into provenance."""

    v253_policy_version: str = SCRIPT_VERSION
    v253_parent_v252_script_sha256: str = ""
    v253_redistribution_max_score_cost: float = (
        V253_REDISTRIBUTION_MAX_SCORE_COST
    )
    v253_redistribution_max_split_group_members: int = (
        V253_REDISTRIBUTION_MAX_SPLIT_GROUP_MEMBERS
    )
    v253_redistribution_max_row_count_stability: float = (
        V253_REDISTRIBUTION_MAX_ROW_COUNT_STABILITY
    )
    v253_redistribution_max_assignment_stability: float = (
        V253_REDISTRIBUTION_MAX_ASSIGNMENT_STABILITY
    )
    v253_redistribution_max_ordinary_rms_width_factor: float = (
        V253_REDISTRIBUTION_MAX_ORDINARY_RMS_WIDTH_FACTOR
    )
    v253_multipoint_fragment_max_score_cost: float = (
        V253_MULTIPOINT_FRAGMENT_MAX_SCORE_COST
    )
    v253_multipoint_fragment_min_members_each: int = (
        V253_MULTIPOINT_FRAGMENT_MIN_MEMBERS_EACH
    )
    v253_multipoint_fragment_max_longitudinal_overlap_fraction: float = (
        V253_MULTIPOINT_FRAGMENT_MAX_LONGITUDINAL_OVERLAP_FRACTION
    )
    v253_multipoint_fragment_max_ordinary_rms_width_factor: float = (
        V253_MULTIPOINT_FRAGMENT_MAX_ORDINARY_RMS_WIDTH_FACTOR
    )
    v253_ambiguous_boundary_min_gap_width_factor: float = (
        V253_AMBIGUOUS_BOUNDARY_MIN_GAP_WIDTH_FACTOR
    )
    v253_ambiguous_boundary_min_local_gap_ratio: float = (
        V253_AMBIGUOUS_BOUNDARY_MIN_LOCAL_GAP_RATIO
    )
    v253_ambiguous_boundary_min_image_support_ratio: float = (
        V253_AMBIGUOUS_BOUNDARY_MIN_IMAGE_SUPPORT_RATIO
    )
    v253_ambiguous_boundary_min_stability_fraction: float = (
        V253_AMBIGUOUS_BOUNDARY_MIN_STABILITY_FRACTION
    )
    v253_ambiguous_boundary_min_stomata_per_side: int = (
        V253_AMBIGUOUS_BOUNDARY_MIN_STOMATA_PER_SIDE
    )
    v253_ambiguous_boundary_min_row_count_stability: float = (
        V253_AMBIGUOUS_BOUNDARY_MIN_ROW_COUNT_STABILITY
    )
    v253_ambiguous_boundary_min_assignment_stability: float = (
        V253_AMBIGUOUS_BOUNDARY_MIN_ASSIGNMENT_STABILITY
    )
    v253_resolved_boundary_min_image_support_ratio: float = (
        V253_RESOLVED_BOUNDARY_MIN_IMAGE_SUPPORT_RATIO
    )
    v253_singleton_full_visibility_tolerance_pixels: float = (
        V253_SINGLETON_FULL_VISIBILITY_TOLERANCE_PIXELS
    )


def build_settings_v253(args: Any) -> TraitSettingsV253:
    inherited = asdict(v252.build_settings_v252(args))
    inherited.update(
        {
            "v253_policy_version": SCRIPT_VERSION,
            "v253_parent_v252_script_sha256": hashlib.sha256(
                _PARENT_V252_SCRIPT_PATH.read_bytes()
            ).hexdigest(),
        }
    )
    return TraitSettingsV253(**inherited)


_V253_ROW_DEFAULTS: dict[str, Any] = {
    "trajectory_v253_policy_version": SCRIPT_VERSION,
    "trajectory_v253_general_row_simplification_applied": False,
    "trajectory_v253_general_row_simplification_route": "not_applied",
    "trajectory_v253_general_row_simplification_score_cost": None,
    "trajectory_v253_general_row_simplification_geometry": {},
    "trajectory_v253_singleton_full_visibility_rule_applied": False,
    "trajectory_v253_non_fully_visible_singleton_pending_count": 0,
    "trajectory_v253_non_fully_visible_singleton_pending": [],
}


def _visibility_details(
    point_index: int,
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettingsV253,
) -> dict[str, Any]:
    """Assess full detector-box visibility using explicit records and metadata."""

    metadata = v14._ACTIVE_IMAGE_METADATA or {}
    record = (
        stoma_traits[int(point_index)]
        if 0 <= int(point_index) < len(stoma_traits)
        else None
    )
    width = base.optional_float(metadata.get("image_width"))
    height = base.optional_float(metadata.get("image_height"))
    x1 = base.optional_float(record.get("prediction_x1_pixels")) if record else None
    y1 = base.optional_float(record.get("prediction_y1_pixels")) if record else None
    x2 = base.optional_float(record.get("prediction_x2_pixels")) if record else None
    y2 = base.optional_float(record.get("prediction_y2_pixels")) if record else None
    values = (width, height, x1, y1, x2, y2)
    assessable = bool(
        all(value is not None and math.isfinite(float(value)) for value in values)
        and float(width) > 0.0
        and float(height) > 0.0
        and float(x1) <= float(x2)
        and float(y1) <= float(y2)
    )
    sides: list[str] = []
    if assessable:
        tolerance = float(
            settings.v253_singleton_full_visibility_tolerance_pixels
        )
        if float(x1) <= tolerance:
            sides.append("left")
        if float(x2) >= float(width) - tolerance:
            sides.append("right")
        if float(y1) <= tolerance:
            sides.append("top")
        if float(y2) >= float(height) - tolerance:
            sides.append("bottom")
    clipped = bool(assessable and sides)
    fully_visible = bool(assessable and not sides)
    return {
        "assessable": assessable,
        "fully_visible": fully_visible,
        "clipped": clipped,
        "visibility_status": (
            "fully_visible"
            if fully_visible
            else "detector_box_touches_image_edge"
            if clipped
            else "full_visibility_unassessable"
        ),
        "sides": sides,
        "tolerance_pixels": float(
            settings.v253_singleton_full_visibility_tolerance_pixels
        ),
        "prediction_x1_pixels": x1,
        "prediction_y1_pixels": y1,
        "prediction_x2_pixels": x2,
        "prediction_y2_pixels": y2,
        "image_width_pixels": width,
        "image_height_pixels": height,
    }


def _selected_record(
    chosen: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any] | None:
    canonical = v25._canonical_partition(chosen)
    angle = math.degrees(math.atan(float(chosen.get("slope") or 0.0)))
    return next(
        (
            record
            for record in records
            if record["canonical_partition"] == canonical
            and math.isclose(
                float(record["slope_angle_deg"]),
                angle,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        ),
        None,
    )


def _redistribution_geometry(
    selected: dict[str, Any],
    simpler_record: dict[str, Any],
    settings: TraitSettingsV253,
) -> dict[str, Any]:
    """Recognize one selected pseudo-row redistributed into two intact rows."""

    selected_groups = [
        set(group) for group in v25._candidate_group_members(selected)
    ]
    simpler_groups = [
        set(group)
        for group in v25._candidate_group_members(simpler_record["candidate"])
    ]
    failures: list[str] = []
    split_groups: list[tuple[int, list[int], list[int]]] = []
    intact_owners: dict[int, int] = {}
    for selected_index, group in enumerate(selected_groups):
        intersections = [
            target_index
            for target_index, target in enumerate(simpler_groups)
            if group & target
        ]
        if len(intersections) == 1 and group <= simpler_groups[intersections[0]]:
            intact_owners[selected_index] = intersections[0]
        elif len(intersections) == 2:
            split_groups.append(
                (
                    selected_index,
                    intersections,
                    [
                        len(group & simpler_groups[target_index])
                        for target_index in intersections
                    ],
                )
            )
        else:
            failures.append("broad_membership_reassignment")

    if len(split_groups) != 1:
        failures.append("not_exactly_one_redistributed_group")
        return {
            "passed": False,
            "failed_gates": sorted(set(failures)),
        }
    split_index, targets, piece_sizes = split_groups[0]
    split_group = selected_groups[split_index]
    if not 0 < split_index < len(selected_groups) - 1:
        failures.append("redistributed_group_not_interior")
    if set().union(*selected_groups) != set().union(*simpler_groups):
        failures.append("point_membership_changed")
    if len(split_group) > int(
        settings.v253_redistribution_max_split_group_members
    ):
        failures.append("redistributed_group_too_large")
    if any(value < 1 for value in piece_sizes):
        failures.append("empty_redistributed_piece")
    if abs(targets[1] - targets[0]) != 1:
        failures.append("target_rows_not_adjacent")
    owner_values = list(intact_owners.values())
    if (
        len(intact_owners) != len(selected_groups) - 1
        or len(set(owner_values)) != len(simpler_groups)
    ):
        failures.append("intact_rows_not_bijective")
    immediate_neighbor_targets = [
        intact_owners.get(split_index - 1),
        intact_owners.get(split_index + 1),
    ]
    if targets != immediate_neighbor_targets:
        failures.append("targets_not_immediate_neighbor_rows")
    for target_index in targets:
        has_intact_core = any(
            selected_index != split_index
            and owner == target_index
            and len(selected_groups[selected_index]) >= 2
            for selected_index, owner in intact_owners.items()
        )
        if not has_intact_core:
            failures.append("target_row_lacks_intact_multipoint_core")
    for target_index, target in enumerate(simpler_groups):
        components = [
            selected_groups[selected_index]
            for selected_index, owner in intact_owners.items()
            if owner == target_index
        ]
        split_piece = split_group & target
        reconstructed = set().union(*components, split_piece) if components else split_piece
        if reconstructed != target:
            failures.append("simpler_row_not_exactly_reconstructed")
    return {
        "selected_redistributed_group_index": split_index,
        "selected_redistributed_group_member_count": len(split_group),
        "simpler_target_group_indices": targets,
        "redistributed_piece_member_counts": piece_sizes,
        "selected_cluster_sizes": [len(group) for group in selected_groups],
        "simpler_cluster_sizes": [len(group) for group in simpler_groups],
        "passed": not failures,
        "failed_gates": sorted(set(failures)),
    }


def _choose_v253_additional_simplification(
    chosen: dict[str, Any],
    records: list[dict[str, Any]],
    longitudinal: np.ndarray,
    settings: TraitSettingsV253,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]] | None:
    selected_record = _selected_record(chosen, records)
    if selected_record is None:
        return None
    selected_k = int(chosen.get("k") or 0)
    alternatives = [
        record
        for record in records
        if bool(record.get("valid"))
        and int(record["candidate_row_count"]) == selected_k - 1
        and int(record.get("singleton_row_count") or 0) == 0
    ]
    if selected_k < 2 or not alternatives:
        return None
    simpler_record = min(
        alternatives,
        key=lambda record: (
            float(record["adjusted_score"]),
            abs(float(record["slope_angle_deg"])),
        ),
    )
    score_cost = float(simpler_record["adjusted_score"]) - float(
        selected_record["adjusted_score"]
    )
    selected_sizes = [
        len(group) for group in v25._candidate_group_members(chosen)
    ]
    simpler_sizes = [
        len(group)
        for group in v25._candidate_group_members(simpler_record["candidate"])
    ]

    row_stability = base.optional_float(
        chosen.get("trajectory_v25_row_count_stability_fraction")
    )
    assignment_stability = base.optional_float(
        chosen.get("trajectory_v25_assignment_stability_fraction")
    )
    redistribution = _redistribution_geometry(chosen, simpler_record, settings)
    redistribution.update(
        {
            "score_cost": score_cost,
            "row_count_stability_fraction": row_stability,
            "assignment_stability_fraction": assignment_stability,
            "simpler_maximum_ordinary_row_rms_width_factor": float(
                simpler_record["maximum_ordinary_row_rms_width_factor"]
            ),
        }
    )
    redistribution_passed = bool(
        redistribution.get("passed")
        and selected_k >= 3
        and all(value >= 2 for value in selected_sizes)
        and all(value >= 2 for value in simpler_sizes)
        and row_stability is not None
        and row_stability
        <= float(settings.v253_redistribution_max_row_count_stability) + 1e-12
        and assignment_stability is not None
        and assignment_stability
        < float(settings.v253_redistribution_max_assignment_stability)
        and score_cost <= float(settings.v253_redistribution_max_score_cost)
        and float(simpler_record["maximum_ordinary_row_rms_width_factor"])
        <= float(settings.v253_redistribution_max_ordinary_rms_width_factor)
    )
    if redistribution_passed:
        redistribution["passed"] = True
        redistribution["failed_gates"] = []
        return (
            simpler_record["candidate"],
            simpler_record,
            "unstable_pseudorow_redistribution",
            redistribution,
        )

    baseline_k = base.optional_int(chosen.get("trajectory_v25_baseline_row_count"))
    pair = v252._clean_single_fragment_merge(chosen, simpler_record["candidate"])
    if pair is not None:
        fragment_geometry = v252._fragment_continuity_geometry(
            chosen, simpler_record, pair, longitudinal, settings
        )
        pair_sizes = [selected_sizes[pair[0]], selected_sizes[pair[1]]]
        fragment_geometry.update(
            {
                "score_cost": score_cost,
                "selected_fragment_member_counts": pair_sizes,
                "selected_cluster_sizes": selected_sizes,
                "simpler_cluster_sizes": simpler_sizes,
            }
        )
        multipoint_fragment_passed = bool(
            fragment_geometry.get("passed")
            and base.parse_bool(chosen.get("trajectory_v25_joint_change_applied"))
            is True
            and baseline_k is not None
            and selected_k == baseline_k + 1
            and all(value >= 2 for value in selected_sizes)
            and all(value >= 2 for value in simpler_sizes)
            and min(pair_sizes)
            >= int(settings.v253_multipoint_fragment_min_members_each)
            and float(fragment_geometry["longitudinal_overlap_fraction"])
            <= float(
                settings.v253_multipoint_fragment_max_longitudinal_overlap_fraction
            )
            and float(
                fragment_geometry["simpler_maximum_ordinary_row_rms_width_factor"]
            )
            <= float(
                settings.v253_multipoint_fragment_max_ordinary_rms_width_factor
            )
            and score_cost
            <= float(settings.v253_multipoint_fragment_max_score_cost)
        )
        if multipoint_fragment_passed:
            fragment_geometry["passed"] = True
            fragment_geometry["failed_gates"] = []
            return (
                simpler_record["candidate"],
                simpler_record,
                "disjoint_multipoint_fragment_continuity",
                fragment_geometry,
            )
    return None


def _apply_simplification(
    chosen: dict[str, Any],
    diagnostic: dict[str, Any],
    records: list[dict[str, Any]],
    replacement: dict[str, Any],
    selected_record: dict[str, Any],
    route: str,
    geometry: dict[str, Any],
    median_width_um: float,
    settings: TraitSettingsV253,
    inherited_v252: bool,
) -> dict[str, Any]:
    revised = dict(chosen)
    revised.update(replacement)
    if inherited_v252:
        fields = {
            "trajectory_v252_policy_version": v252.SCRIPT_VERSION,
            "trajectory_v252_general_row_simplification_applied": True,
            "trajectory_v252_general_row_simplification_route": route,
            "trajectory_v252_general_row_simplification_score_cost": float(
                geometry["score_cost"]
            ),
            "trajectory_v252_general_row_simplification_geometry": geometry,
        }
        reason = f"v252_{route}"
    else:
        fields = {
            "trajectory_v253_policy_version": SCRIPT_VERSION,
            "trajectory_v253_general_row_simplification_applied": True,
            "trajectory_v253_general_row_simplification_route": route,
            "trajectory_v253_general_row_simplification_score_cost": float(
                geometry["score_cost"]
            ),
            "trajectory_v253_general_row_simplification_geometry": geometry,
        }
        reason = f"v253_{route}"
    revised.update(fields)
    diagnostic.update(fields)
    v252._refresh_selected_fields(
        revised,
        diagnostic,
        records,
        selected_record,
        reason,
        median_width_um,
        settings,
    )
    if not inherited_v252:
        confidence_fields = {
            "trajectory_v25_topology_confidence_tier": (
                "supported_by_v253_general_geometry"
            ),
            "trajectory_v253_policy_version": SCRIPT_VERSION,
            **fields,
        }
        revised.update(confidence_fields)
        diagnostic.update(confidence_fields)
    return revised


def _apply_v25_joint_search_v253(
    parent_chosen: dict[str, Any],
    parent_candidates: list[dict[str, Any]],
    diagnostic: dict[str, Any],
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_width_um: float,
    settings: TraitSettingsV253,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    # Start from the unmodified v2.5 joint search.  Unlike v2.5.2, this does
    # not force an edge singleton into a neighbouring row.  Final membership
    # is established normally and visibility is enforced afterward.
    chosen, candidates, diagnostic = v252._PARENT_APPLY_V25_JOINT_SEARCH(
        parent_chosen,
        parent_candidates,
        diagnostic,
        point_indices,
        lateral,
        longitudinal,
        median_width_um,
        settings,
    )
    records = v252._candidate_records(
        [*candidates, chosen], lateral, longitudinal, median_width_um, settings
    )

    inherited = v252._choose_general_simplification(
        chosen, records, longitudinal, settings
    )
    if inherited is not None:
        replacement, selected_record, route, geometry = inherited
        chosen = _apply_simplification(
            chosen,
            diagnostic,
            records,
            replacement,
            selected_record,
            route,
            geometry,
            median_width_um,
            settings,
            inherited_v252=True,
        )
    else:
        additional = _choose_v253_additional_simplification(
            chosen, records, longitudinal, settings
        )
        if additional is not None:
            replacement, selected_record, route, geometry = additional
            chosen = _apply_simplification(
                chosen,
                diagnostic,
                records,
                replacement,
                selected_record,
                route,
                geometry,
                median_width_um,
                settings,
                inherited_v252=False,
            )

    active_traits = v15._ACTIVE_STOMA_TRAITS_V15 or []
    pending: list[dict[str, Any]] = []
    if active_traits:
        for point in v25._singleton_points(chosen):
            details = _visibility_details(int(point), active_traits, settings)
            if not details["fully_visible"]:
                pending.append(
                    {
                        "point_index": int(point),
                        "stoma_id": v25._point_stoma_id(int(point)),
                        "visibility": details,
                    }
                )
    edge_fields = {
        "trajectory_v253_policy_version": SCRIPT_VERSION,
        "trajectory_v253_singleton_full_visibility_rule_applied": bool(pending),
        "trajectory_v253_non_fully_visible_singleton_pending_count": len(pending),
        "trajectory_v253_non_fully_visible_singleton_pending": pending,
        # Compatibility fields keep the inherited overlay/QC machinery useful.
        "trajectory_v252_edge_touching_singleton_rule_applied": bool(pending),
        "trajectory_v252_edge_touching_singleton_absorbed_count": 0,
        "trajectory_v252_edge_touching_singleton_absorbed": [],
        "trajectory_v252_edge_touching_singleton_pending_topology_removal_count": (
            len(pending)
        ),
        "trajectory_v252_edge_touching_singleton_pending_topology_removal": pending,
        "trajectory_v252_edge_touch_tolerance_pixels": float(
            settings.v253_singleton_full_visibility_tolerance_pixels
        ),
    }
    chosen.update(edge_fields)
    diagnostic.update(edge_fields)

    best_by_k: dict[int, dict[str, Any]] = {
        int(candidate["k"]): candidate for candidate in candidates
    }
    best_by_k[int(chosen["k"])] = chosen
    return chosen, [best_by_k[key] for key in sorted(best_by_k)], diagnostic


def _v25_partition_candidate_v253(
    band: dict[str, Any],
    linked: list[dict[str, Any]],
    cut: int,
    gaps: list[float],
    boundary_evidence_by_rows: dict[tuple[str, str, str], dict[str, Any]],
    median_width_um: float,
    settings: TraitSettingsV253,
) -> dict[str, Any]:
    result = v252._v25_partition_candidate_v252(
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
    failures = {str(value) for value in result.get("failed_gates", [])}
    topology_only = failures == {"row_topology_not_supported"}
    rows_resolved_by_v253 = bool(
        linked
        and all(
            bool(row.get("trajectory_v253_general_row_simplification_applied"))
            for row in linked
        )
    )
    resolved_exact_support = bool(
        result.get("image_support_identity_exact")
        and result.get("image_support") is not None
        and float(result["image_support"])
        >= float(settings.v253_resolved_boundary_min_image_support_ratio)
    )
    resolved_near_threshold_isolated_boundary = bool(
        failures
        == {"row_topology_not_supported", "isolated_row_raw_image_support"}
        and bool(result.get("isolated_one_row_side"))
        and float(result.get("gap_factor") or 0.0)
        >= float(settings.v253_ambiguous_boundary_min_gap_width_factor)
        and float(result.get("gap_ratio") or 0.0)
        >= float(settings.v253_ambiguous_boundary_min_local_gap_ratio)
        and float(result.get("stability") or 0.0)
        >= float(settings.v253_ambiguous_boundary_min_stability_fraction)
        and min(
            int(result.get("left_stomata") or 0),
            int(result.get("right_stomata") or 0),
        )
        >= int(settings.v253_ambiguous_boundary_min_stomata_per_side)
        and resolved_exact_support
        and not bool(result.get("nonlinear_row_topology"))
    )
    resolved_by_v253 = bool(
        rows_resolved_by_v253
        and (topology_only or resolved_near_threshold_isolated_boundary)
    )
    exact_support = bool(
        result.get("image_support_identity_exact")
        and result.get("image_support") is not None
        and float(result["image_support"])
        >= float(settings.v253_ambiguous_boundary_min_image_support_ratio)
    )
    topology_stability_supported = bool(
        linked
        and all(
            base.optional_float(
                row.get("trajectory_v25_row_count_stability_fraction")
            )
            is not None
            and float(row["trajectory_v25_row_count_stability_fraction"])
            >= float(
                settings.v253_ambiguous_boundary_min_row_count_stability
            )
            and base.optional_float(
                row.get("trajectory_v25_assignment_stability_fraction")
            )
            is not None
            and float(row["trajectory_v25_assignment_stability_fraction"])
            >= float(
                settings.v253_ambiguous_boundary_min_assignment_stability
            )
            and base.optional_int(row.get("trajectory_v25_row_count_lower_bound"))
            == base.optional_int(row.get("trajectory_v25_selected_row_count"))
            == base.optional_int(row.get("trajectory_v25_row_count_upper_bound"))
            for row in linked
        )
    )
    strong_ambiguous_boundary = bool(
        topology_only
        and not bool(result.get("isolated_one_row_side"))
        and float(result.get("gap_factor") or 0.0)
        >= float(settings.v253_ambiguous_boundary_min_gap_width_factor)
        and float(result.get("gap_ratio") or 0.0)
        >= float(settings.v253_ambiguous_boundary_min_local_gap_ratio)
        and float(result.get("stability") or 0.0)
        >= float(settings.v253_ambiguous_boundary_min_stability_fraction)
        and min(
            int(result.get("left_stomata") or 0),
            int(result.get("right_stomata") or 0),
        )
        >= int(settings.v253_ambiguous_boundary_min_stomata_per_side)
        and exact_support
        and topology_stability_supported
        and not bool(result.get("nonlinear_row_topology"))
    )
    if not (resolved_by_v253 or strong_ambiguous_boundary):
        return result
    route = (
        "resolved_v253_row_topology_near_threshold_exact_image_boundary"
        if resolved_near_threshold_isolated_boundary
        else "resolved_v253_row_topology_all_other_v25_gates_passed"
        if resolved_by_v253
        else "strong_exact_image_supported_ambiguous_topology_boundary"
    )
    result.update(
        {
            "accepted": True,
            "failed_gates_before_v253_general_rule": sorted(failures),
            "failed_gates": [],
            "trajectory_v253_general_band_boundary_applied": True,
            "trajectory_v253_general_band_boundary_route": route,
            "decision_route": f"v253_{route}",
        }
    )
    return result


def _non_fully_visible_singleton_rows(
    rows: list[dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettingsV253,
) -> list[tuple[dict[str, Any], int, dict[str, Any]]]:
    candidates: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
    for row in rows:
        members = [int(value) for value in row.get("_point_indices", [])]
        if len(members) != 1:
            continue
        details = _visibility_details(members[0], stoma_traits, settings)
        if not details["fully_visible"]:
            candidates.append((row, members[0], details))
    return candidates


def _image_wide_trajectory_slope(rows: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for row in rows:
        total = base.optional_float(
            row.get("row_common_slope_lateral_per_longitudinal")
        )
        residual = base.optional_float(
            row.get("trajectory_residual_slope_after_image_detrending")
        )
        if total is not None and residual is not None:
            values.append(float(total) - float(residual))
    return float(np.median(np.asarray(values, dtype=float))) if values else 0.0


def _remove_non_fully_visible_singletons_from_topology(
    rows: list[dict[str, Any]],
    bands: list[dict[str, Any]],
    assignments: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    audit: dict[str, Any],
    settings: TraitSettingsV253,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    candidates = _non_fully_visible_singleton_rows(rows, stoma_traits, settings)
    if not candidates:
        # Preserve the first cleanup audit if this function is called again on
        # an already-cleaned topology.
        if audit.get("trajectory_v253_non_fully_visible_singleton_removed"):
            return rows, bands, assignments, audit
        audit.update(
            {
                "trajectory_v253_non_fully_visible_singleton_removed_count": 0,
                "trajectory_v253_half_cut_singleton_removed_count": 0,
                "trajectory_v253_unassessable_singleton_removed_count": 0,
                "trajectory_v253_non_fully_visible_singleton_removed": [],
                "trajectory_v252_edge_touching_singleton_removed_from_topology_count": 0,
                "trajectory_v252_edge_touching_singleton_removed_from_topology": [],
                "final_detected_row_count": len(rows),
                "final_detected_band_count": len(bands),
                "detected_row_count_v253": len(rows),
                "detected_band_count_v253": len(bands),
                "trajectory_v253_removed_point_indices": [],
            }
        )
        return rows, bands, assignments, audit

    removed_ids = {str(row["row_id"]) for row, _, _ in candidates}
    removed_points = {int(point) for _, point, _ in candidates}
    removed_records: list[dict[str, Any]] = []
    for row, point_index, details in candidates:
        row_id = str(row.get("row_id") or "")
        band_id = str(row.get("band_id") or "")
        provisional = str(row.get("provisional_band_id") or "")
        status = str(details["visibility_status"])
        reason = (
            "singleton_detector_box_touches_image_edge"
            if details["clipped"]
            else "singleton_full_visibility_unassessable"
        )
        marker = {
            "trajectory_v253_policy_version": SCRIPT_VERSION,
            "trajectory_v253_non_fully_visible_singleton_excluded": True,
            "trajectory_v253_singleton_visibility_status": status,
            "trajectory_v253_singleton_visibility_details": details,
            "trajectory_v253_final_topology_status": "selected_then_excluded",
            "trajectory_v253_topology_exclusion_reason": reason,
            "trajectory_v253_former_row_id": row_id,
            "trajectory_v253_former_band_id": band_id,
            "trajectory_v253_former_provisional_band_id": provisional,
            "trajectory_v252_edge_touching_singleton_ignored_for_row_delimitation": bool(
                details["clipped"]
            ),
            "trajectory_v252_edge_touching_singleton_edge_sides": details["sides"],
            "row_id": "",
            "band_id": "",
            "provisional_band_id": "",
            "row_position_within_band": None,
            "rows_in_band": None,
            "automatic_rows_in_band": None,
            "band_row_count": None,
            "row_band_spatial_eligible": False,
            "row_complete_topology_eligible": False,
            "row_structure_eligible": False,
            "row_completeness_dependent_traits_eligible": False,
            "trajectory_exact_row_traits_eligible": False,
            "trajectory_exact_row_band_traits_eligible": False,
            "row_band_spatial_exclusion_reason": reason,
        }
        assignments[int(point_index)].update(marker)
        stoma_traits[int(point_index)].update(marker)
        removed_records.append(
            {
                "point_index": int(point_index),
                "stoma_id": str(
                    stoma_traits[int(point_index)].get("stoma_id") or ""
                ),
                "former_row_id": row_id,
                "former_band_id": band_id,
                "former_provisional_band_id": provisional,
                "visibility_status": status,
                "edge_sides": details["sides"],
            }
        )

    rows = [row for row in rows if str(row.get("row_id") or "") not in removed_ids]
    row_lookup = {str(row["row_id"]): row for row in rows}
    detector_widths = [
        base.optional_float(record.get("prediction_width_um"))
        for record in stoma_traits
    ]
    detector_widths = [
        value for value in detector_widths if value is not None and value > 0.0
    ]
    median_width_um = (
        float(np.median(np.asarray(detector_widths, dtype=float)))
        if detector_widths
        else 1.0
    )
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
                    "trajectory_v253_non_fully_visible_singleton_removed_from_band": True,
                    "trajectory_v253_policy_version": SCRIPT_VERSION,
                },
            )
        )
    rows, rebuilt_bands, assignments = v24._renumber_final_bands(
        rows, rebuilt_bands, assignments
    )

    # Recompute edge eligibility from current band identities.  Reusing the
    # pre-removal audit can attach stale evidence after a singleton-only band
    # disappears and all later bands are renumbered.
    trajectory_slope = _image_wide_trajectory_slope(rows)
    fresh_v24_edge_audit = v24._remark_partial_band_eligibility_v24(
        rows,
        rebuilt_bands,
        assignments,
        stoma_traits,
        median_width_um,
        trajectory_slope,
        settings,
        parent_audit=None,
    )
    edge_audit = v25._apply_v25_standardized_multirow_edge_rule(
        rows,
        rebuilt_bands,
        assignments,
        stoma_traits,
        settings,
        fresh_v24_edge_audit,
    )

    removed_count = len(removed_records)
    clipped_count = sum(record["visibility_status"] != "full_visibility_unassessable" for record in removed_records)
    unassessable_count = removed_count - clipped_count
    lower_before = base.optional_int(audit.get("detected_row_count_v25_lower_bound"))
    upper_before = base.optional_int(audit.get("detected_row_count_v25_upper_bound"))
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
            "trajectory_v253_non_fully_visible_singleton_removed_count": removed_count,
            "trajectory_v253_half_cut_singleton_removed_count": clipped_count,
            "trajectory_v253_unassessable_singleton_removed_count": unassessable_count,
            "trajectory_v253_non_fully_visible_singleton_removed": removed_records,
            # Compatibility field used by the inherited image summary.
            "trajectory_v252_edge_touching_singleton_removed_from_topology_count": clipped_count,
            "trajectory_v252_edge_touching_singleton_removed_from_topology": [
                record
                for record in removed_records
                if record["visibility_status"] != "full_visibility_unassessable"
            ],
            "final_detected_row_count": len(rows),
            "final_detected_band_count": len(rebuilt_bands),
            "detected_row_count_v25": len(rows),
            "detected_band_count_v25": len(rebuilt_bands),
            "detected_row_count_v253": len(rows),
            "detected_band_count_v253": len(rebuilt_bands),
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
            "trajectory_v253_removed_point_indices": sorted(removed_points),
        }
    )
    return rows, rebuilt_bands, assignments, audit


def _assert_v253_invariants(
    stoma_traits: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    bands: list[dict[str, Any]],
    assignments: dict[int, dict[str, Any]],
    audit: dict[str, Any],
    eligibility_before: dict[str, list[bool | None]],
    settings: TraitSettingsV253,
) -> None:
    source = str(stoma_traits[0].get("source_image_name") or "") if stoma_traits else ""
    final_points: list[int] = [
        int(point)
        for row in rows
        for point in row.get("_point_indices", [])
    ]
    removed_points = {
        int(record["point_index"])
        for record in audit.get(
            "trajectory_v253_non_fully_visible_singleton_removed", []
        )
    }
    if len(final_points) != len(set(final_points)):
        raise RuntimeError(
            f"v2.5.3 invariant failed: duplicated final topology point; source={source}."
        )
    if set(final_points) & removed_points:
        raise RuntimeError(
            f"v2.5.3 invariant failed: excluded point remains in topology; source={source}."
        )
    if set(final_points) | removed_points != set(range(len(stoma_traits))):
        raise RuntimeError(
            f"v2.5.3 invariant failed: topology/exclusion partition incomplete; source={source}."
        )

    row_lookup = {str(row["row_id"]): row for row in rows}
    for row in rows:
        members = [int(value) for value in row.get("_point_indices", [])]
        if len(members) == 1:
            details = _visibility_details(members[0], stoma_traits, settings)
            if not details["fully_visible"]:
                raise RuntimeError(
                    "v2.5.3 invariant failed: a non-fully-visible singleton "
                    f"survived final topology; source={source}, row={row['row_id']}."
                )
        for point in members:
            assignment = assignments[int(point)]
            if str(assignment.get("row_id") or "") != str(row["row_id"]):
                raise RuntimeError(
                    f"v2.5.3 invariant failed: point/row mismatch; source={source}."
                )
            if str(assignment.get("band_id") or "") != str(row.get("band_id") or ""):
                raise RuntimeError(
                    f"v2.5.3 invariant failed: point/band mismatch; source={source}."
                )
            if bool(
                stoma_traits[point].get(
                    "trajectory_v253_non_fully_visible_singleton_excluded"
                )
            ):
                raise RuntimeError(
                    f"v2.5.3 invariant failed: retained point marked excluded; source={source}."
                )
    for point in removed_points:
        assignment = assignments[int(point)]
        if assignment.get("row_id") or assignment.get("band_id"):
            raise RuntimeError(
                f"v2.5.3 invariant failed: excluded point retains topology IDs; source={source}."
            )
        details = _visibility_details(point, stoma_traits, settings)
        if details["fully_visible"]:
            raise RuntimeError(
                f"v2.5.3 invariant failed: fully visible singleton excluded; source={source}."
            )

    band_ids = {str(band["band_id"]) for band in bands}
    for band in bands:
        linked = [str(value) for value in band.get("row_ids", [])]
        if not linked or any(row_id not in row_lookup for row_id in linked):
            raise RuntimeError(
                f"v2.5.3 invariant failed: band/row link mismatch; source={source}."
            )
        if any(str(row_lookup[row_id].get("band_id") or "") != str(band["band_id"]) for row_id in linked):
            raise RuntimeError(
                f"v2.5.3 invariant failed: row/band identity mismatch; source={source}."
            )
    if {str(row.get("band_id") or "") for row in rows} - band_ids:
        raise RuntimeError(
            f"v2.5.3 invariant failed: row refers to missing band; source={source}."
        )

    marker_count = sum(
        bool(record.get("trajectory_v253_non_fully_visible_singleton_excluded"))
        for record in stoma_traits
    )
    if marker_count != len(removed_points):
        raise RuntimeError(
            f"v2.5.3 invariant failed: exclusion marker count mismatch; source={source}."
        )
    for field, before in eligibility_before.items():
        after = [base.parse_bool(record.get(field)) for record in stoma_traits]
        if after != before:
            raise RuntimeError(
                "v2.5.3 invariant failed: density/morphometry eligibility changed; "
                f"source={source}, field={field}."
            )


def _count_v253_general_boundaries(audit: dict[str, Any]) -> int:
    """Count unique accepted v2.5.3 boundaries, including nested candidates."""

    accepted: set[tuple[str, str, str, str]] = set()
    for partition in audit.get("band_partition_diagnostics", []):
        if not isinstance(partition, dict):
            continue
        records = [partition]
        nested = partition.get("trajectory_v25_partition_candidate_diagnostics", [])
        if isinstance(nested, list):
            records.extend(record for record in nested if isinstance(record, dict))
        for record in records:
            if not bool(record.get("trajectory_v253_general_band_boundary_applied")):
                continue
            if base.parse_bool(record.get("accepted")) is not True:
                continue
            accepted.add(
                (
                    str(record.get("provisional_band_id") or ""),
                    str(record.get("boundary_left_row_id") or ""),
                    str(record.get("boundary_right_row_id") or ""),
                    str(record.get("trajectory_v253_general_band_boundary_route") or ""),
                )
            )
    return len(accepted)


def infer_rows_and_bands_v253(
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettingsV253,
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
    rows, bands, assignments, audit = _PARENT_INFER_V252(
        stoma_traits,
        settings,
        audited_row_counts=audited_row_counts,
    )
    rows, bands, assignments, audit = (
        _remove_non_fully_visible_singletons_from_topology(
            rows, bands, assignments, stoma_traits, audit, settings
        )
    )

    diagnostics = list(audit.get("band_selection_diagnostics", {}).values())
    row_changes = sum(
        bool(record.get("trajectory_v253_general_row_simplification_applied"))
        for record in diagnostics
    )
    redistribution_changes = sum(
        str(record.get("trajectory_v253_general_row_simplification_route") or "")
        == "unstable_pseudorow_redistribution"
        for record in diagnostics
    )
    fragment_changes = sum(
        str(record.get("trajectory_v253_general_row_simplification_route") or "")
        == "disjoint_multipoint_fragment_continuity"
        for record in diagnostics
    )
    boundary_changes = _count_v253_general_boundaries(audit)
    audit.update(
        {
            "row_model_version": SCRIPT_VERSION,
            "trajectory_v253_policy_version": SCRIPT_VERSION,
            "trajectory_v253_general_row_simplification_count": row_changes,
            "trajectory_v253_pseudorow_redistribution_count": redistribution_changes,
            "trajectory_v253_multipoint_fragment_merge_count": fragment_changes,
            "trajectory_v253_general_band_boundary_count": boundary_changes,
            "automatic_row_count_policy": (
                "v2.5.2 generalized topology followed by guarded unstable-"
                "pseudorow redistribution and disjoint multipoint-fragment "
                "continuity; no image-specific memberships or row counts"
            ),
            "band_partition_policy": (
                "v2.5.2 boundaries plus exact-image-supported routes for a "
                "strong boundary rejected solely by row-topology confidence "
                "and for an isolated boundary whose v2.5.3-resolved rows "
                "narrowly miss the parent raw-image-support cutoff"
            ),
            "trajectory_row_band_trait_policy": (
                "only fully visible detections may designate singleton rows; "
                "non-fully-visible final singleton rows are removed from row/"
                "band topology while density and independently valid "
                "morphometry remain unchanged"
            ),
        }
    )
    for collection in (rows, bands, list(assignments.values()), stoma_traits):
        for record in collection:
            record["row_model_version"] = SCRIPT_VERSION
            record["trajectory_v253_policy_version"] = SCRIPT_VERSION
            record["partial_band_eligibility_version"] = SCRIPT_VERSION
            record["trajectory_singleton_policy_version_current"] = SCRIPT_VERSION
    _assert_v253_invariants(
        stoma_traits,
        rows,
        bands,
        assignments,
        audit,
        eligibility_before,
        settings,
    )
    return rows, bands, assignments, audit


def image_trait_summary_v253(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = _PARENT_IMAGE_SUMMARY_V252(*args, **kwargs)
    audit = args[7] if len(args) > 7 else kwargs.get("row_band_audit", {})
    result.update(
        {
            "row_model_version": SCRIPT_VERSION,
            "row_geometry_model_version": SCRIPT_VERSION,
            "partial_band_eligibility_version": SCRIPT_VERSION,
            "trajectory_singleton_policy_version_current": SCRIPT_VERSION,
            "trajectory_v253_policy_version": SCRIPT_VERSION,
            "general_row_simplification_count_v253": int(
                audit.get("trajectory_v253_general_row_simplification_count", 0)
                or 0
            ),
            "pseudorow_redistribution_count_v253": int(
                audit.get("trajectory_v253_pseudorow_redistribution_count", 0)
                or 0
            ),
            "multipoint_fragment_merge_count_v253": int(
                audit.get("trajectory_v253_multipoint_fragment_merge_count", 0)
                or 0
            ),
            "general_band_boundary_count_v253": int(
                audit.get("trajectory_v253_general_band_boundary_count", 0)
                or 0
            ),
            "half_cut_singletons_excluded_from_row_topology_v253": int(
                audit.get("trajectory_v253_half_cut_singleton_removed_count", 0)
                or 0
            ),
            "unassessable_singletons_excluded_from_row_topology_v253": int(
                audit.get("trajectory_v253_unassessable_singleton_removed_count", 0)
                or 0
            ),
            "detected_row_count_v253": len(args[3] if len(args) > 3 else kwargs["row_records"]),
            "detected_band_count_v253": len(args[4] if len(args) > 4 else kwargs["band_records"]),
        }
    )
    return result


def image_gwas_trait_summary_v253(image_traits: dict[str, Any]) -> dict[str, Any]:
    result = _PARENT_GWAS_SUMMARY_V252(image_traits)
    for field in (
        "trajectory_v253_policy_version",
        "general_row_simplification_count_v253",
        "pseudorow_redistribution_count_v253",
        "multipoint_fragment_merge_count_v253",
        "general_band_boundary_count_v253",
        "half_cut_singletons_excluded_from_row_topology_v253",
        "unassessable_singletons_excluded_from_row_topology_v253",
        "detected_row_count_v253",
        "detected_band_count_v253",
    ):
        result[field] = image_traits.get(field)
    result["row_model_version"] = SCRIPT_VERSION
    result["row_geometry_model_version"] = SCRIPT_VERSION
    result["partial_band_eligibility_version"] = SCRIPT_VERSION
    result["trajectory_singleton_policy_version_current"] = SCRIPT_VERSION
    return result


def _update_v253_output_provenance(output_dir: Path) -> None:
    provenance_path = output_dir / "trait_extraction_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance.update(
            {
                "script_version": SCRIPT_VERSION,
                "v253_parent_script": str(_PARENT_V252_SCRIPT_PATH),
                "v253_parent_script_sha256": hashlib.sha256(
                    _PARENT_V252_SCRIPT_PATH.read_bytes()
                ).hexdigest(),
                "v253_contains_image_specific_rules": False,
                "v253_contains_manual_row_count_overrides": False,
                "v253_policy": (
                    "guarded unstable-pseudorow redistribution, guarded "
                    "multipoint longitudinal-fragment merging, strong exact-"
                    "image-supported ambiguous-topology boundaries, narrowly "
                    "missed exact-image support for isolated boundaries whose "
                    "rows v2.5.3 resolves, and final full-visibility enforcement "
                    "for singleton row designation"
                ),
                "v253_development_status": (
                    "rules selected from the confirmation audit; validation on "
                    "a new independent balanced image set is required before "
                    "final scientific lock"
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
        report = report.replace(
            "Row/band settings are provisional until representative AB and AD "
            "overlays are audited.",
            "The representative AB and AD confirmation overlays were audited; "
            "v2.5.3 remains provisional pending an independent balanced image set.",
        )
        report = report.replace(
            "V2.5 remains a development version pending this audit and an "
            "independent balanced confirmation set. The 1.65/1.80 raw-image "
            "support thresholds must be checked before freezing.",
            "V2.5.3 is a confirmation-audit development version pending an "
            "independent balanced validation set. The inherited 1.65/1.80 "
            "raw-image-support thresholds and the guarded v2.5.3 1.60 resolved-"
            "topology route must be validated before freezing.",
        )
        addition = (
            "\nV2.5.3 is a separate successor to v2.5.2.\n"
            "No image name, stomatal identifier, fixed membership or manual "
            "row count is used by v2.5.3.\n"
            "Boundary routes distinguish strong topology-only ambiguity from "
            "an isolated boundary whose v2.5.3-resolved rows narrowly miss the "
            "parent raw-image-support cutoff; both retain frozen exact-image, "
            "geometry, stability and membership gates.\n"
            "Only fully visible stomata can designate singleton rows. A clipped "
            "detection in a multi-stoma row remains assigned; a clipped final "
            "singleton is excluded from row/band topology. Density and "
            "independently valid morphometry are unchanged.\n"
        )
        if "V2.5.3 is a separate successor to v2.5.2." not in report:
            report += addition
        report_path.write_text(report, encoding="utf-8")


def install_v253_policy() -> None:
    """Install v2.5.2, then replace only the v2.5.3 generalized hooks."""

    v252.install_v252_policy()
    v25._V25_DEFAULTS.update(_V253_ROW_DEFAULTS)
    v25._apply_v25_joint_search = _apply_v25_joint_search_v253
    v25._v25_partition_candidate = _v25_partition_candidate_v253

    base.build_settings = build_settings_v253
    base.infer_rows_and_bands = infer_rows_and_bands_v253
    base.image_trait_summary = image_trait_summary_v253
    base.image_gwas_trait_summary = image_gwas_trait_summary_v253
    base.make_spatial_audit_overlay = _PARENT_OVERLAY_V252
    for module in (v14, v15, base):
        module.SCRIPT_VERSION = SCRIPT_VERSION
    base.__file__ = __file__


def run_trait_extraction() -> Path:
    install_v253_policy()
    output_dir = base.run_trait_extraction()
    v25._write_v25_additional_qc_tables(output_dir)
    v25._update_v25_output_provenance(output_dir)
    _update_v253_output_provenance(output_dir)
    return output_dir


if __name__ == "__main__":
    run_trait_extraction()
