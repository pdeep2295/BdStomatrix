#!/usr/bin/env python3
"""Version 1.9 Brachypodium trait extractor with row-aware half-cut-band QC.

This is an eligibility-only successor to v1.8.  It deliberately preserves:

* the v1.6 band-local row geometry and guarded-singleton inference;
* the v1.7 policy that retains computable row/band values with QC flags;
* detector, classifier, segmentation, calibration, density and morphometry;
* every original OME-TIFF and every existing pipeline/trait output.

The revision separates two biological questions that earlier versions treated
as one:

1. Is the lateral cross-section of a stomatal band cut by the left or right
   image boundary?  Only this condition excludes the band from row-structure
   analysis.
2. Is the observed longitudinal support short relative to the available field
   and the other bands in the image?  This is retained as a completeness QC; it
   no longer removes an otherwise fully visible band from row counting.

The lateral decision has four transparent states: ``half_cut``,
``edge_adjacent_uncertain``, ``laterally_contained`` and ``unknown``.  Only an
outermost band can be excluded.  A one-row outermost band uses a 0.30
detector-box-width member-centre threshold because truncation can hide its
structure.  A multirow outermost band uses a 0.25 threshold independently for
each inferred row and is excluded only when every row meets the same-side
member-centre and fitted-trajectory criteria.  If only some rows approach the
edge, the whole band is retained with QC.  Detector bounding boxes remain
diagnostic because their context padding can touch an edge even when the
biological band remains usable.

Run with a new output directory.  This script reads an existing production
pipeline output and does not rerun or alter any trained model.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import extract_traits_from_pipeline_output as base
import extract_traits_from_pipeline_output_v1_4_global_rows as v14
import extract_traits_from_pipeline_output_v1_5_trajectory_rows as v15
import extract_traits_from_pipeline_output_v1_6_local_trajectory_guarded as v16
import extract_traits_from_pipeline_output_v1_7_retain_flagged_row_band_values as v17
import extract_traits_from_pipeline_output_v1_8_lateral_half_cut_eligibility as v18_parent


SCRIPT_VERSION = "1.9.0-row-aware-lateral-half-cut-eligibility"
PARTIAL_BAND_ELIGIBILITY_VERSION = SCRIPT_VERSION
ROW_GEOMETRY_MODEL_VERSION = v16.SCRIPT_VERSION
TRAIT_EXPORT_POLICY_VERSION = v17.SCRIPT_VERSION

# Single-row bands receive truncation-priority screening at 0.30 of a typical
# detector-box width.  For multirow bands the direct threshold is 0.25 and must
# be met independently by every inferred row.  Fitted row geometry must
# approach the same side within 0.70 box widths.  Detector-box contact is
# recorded as context but is never sufficient by itself.
SINGLE_ROW_CENTER_EDGE_CLEARANCE_BOX_WIDTH_FACTOR = 0.30
MULTIROW_CENTER_EDGE_CLEARANCE_BOX_WIDTH_FACTOR = 0.25
# Backward-compatible alias for downstream fields that previously stored one
# direct member-centre threshold.  In v1.9 it denotes the single-row threshold.
LATERAL_CENTER_EDGE_CLEARANCE_BOX_WIDTH_FACTOR = (
    SINGLE_ROW_CENTER_EDGE_CLEARANCE_BOX_WIDTH_FACTOR
)
FITTED_CENTER_EDGE_CORROBORATION_BOX_WIDTH_FACTOR = 0.70
LATERAL_EDGE_ADJACENT_BUFFER_BOX_WIDTH_FACTOR = 0.50
DETECTOR_BOX_EDGE_TOUCH_TOLERANCE_PIXELS = 1.0
FITTED_TRAJECTORY_EDGE_SAMPLES = 101

_PARENT_V18_SCRIPT_PATH = Path(v18_parent.__file__).resolve()

# Save the concrete parent implementation before installation changes the
# dynamic v1.5 hook.  Calling the patched v1.5 name from within v1.9 would
# recurse.
_V16_PARTIAL_BAND_ELIGIBILITY = v16._remark_partial_band_eligibility_v16


@dataclass(frozen=True)
class TraitSettingsV19(v18_parent.TraitSettingsV18):
    """Record the fixed v1.9 eligibility policy in provenance."""

    partial_band_eligibility_version: str = PARTIAL_BAND_ELIGIBILITY_VERSION
    partial_band_row_structure_uses_lateral_half_cut_only: bool = True
    partial_band_severe_short_observed_support_retained_for_rows: bool = True
    partial_band_detector_bbox_edge_not_decisive: bool = True
    partial_band_member_and_fitted_edge_evidence_required: bool = True
    partial_band_lateral_thresholds_provisional_audit_derived: bool = True
    partial_band_outermost_band_required_for_exclusion: bool = True
    partial_band_single_row_truncation_priority: bool = True
    partial_band_multirow_all_rows_must_meet_cutoff: bool = True
    partial_band_lateral_center_clearance_box_width_factor: float = (
        LATERAL_CENTER_EDGE_CLEARANCE_BOX_WIDTH_FACTOR
    )
    partial_band_single_row_center_clearance_box_width_factor: float = (
        SINGLE_ROW_CENTER_EDGE_CLEARANCE_BOX_WIDTH_FACTOR
    )
    partial_band_multirow_center_clearance_box_width_factor: float = (
        MULTIROW_CENTER_EDGE_CLEARANCE_BOX_WIDTH_FACTOR
    )
    partial_band_fitted_center_corroboration_box_width_factor: float = (
        FITTED_CENTER_EDGE_CORROBORATION_BOX_WIDTH_FACTOR
    )
    partial_band_lateral_edge_adjacent_buffer_box_width_factor: float = (
        LATERAL_EDGE_ADJACENT_BUFFER_BOX_WIDTH_FACTOR
    )
    partial_band_detector_box_edge_touch_tolerance_pixels: float = (
        DETECTOR_BOX_EDGE_TOUCH_TOLERANCE_PIXELS
    )
    partial_band_fitted_trajectory_edge_samples: int = (
        FITTED_TRAJECTORY_EDGE_SAMPLES
    )
    partial_band_parent_v18_script_sha256: str = ""


def build_settings_v19(args: Any) -> TraitSettingsV19:
    """Keep every v1.8 setting and append immutable v1.9 policy fields."""

    inherited = asdict(v18_parent.build_settings_v18(args))
    inherited.update(
        {
            "partial_band_eligibility_version": (
                PARTIAL_BAND_ELIGIBILITY_VERSION
            ),
            "partial_band_lateral_center_clearance_box_width_factor": (
                LATERAL_CENTER_EDGE_CLEARANCE_BOX_WIDTH_FACTOR
            ),
            "partial_band_parent_v18_script_sha256": hashlib.sha256(
                _PARENT_V18_SCRIPT_PATH.read_bytes()
            ).hexdigest(),
        }
    )
    return TraitSettingsV19(**inherited)


def _finite_number(value: Any) -> float | None:
    """Return a finite float or ``None`` without changing the input."""

    parsed = base.optional_float(value)
    if parsed is None or not np.isfinite(parsed):
        return None
    return float(parsed)


def _minimum_with_side(
    left_values: list[float], right_values: list[float]
) -> tuple[float | None, str]:
    """Return the minimum side-edge clearance and the corresponding side."""

    left = min(left_values) if left_values else None
    right = min(right_values) if right_values else None
    if left is None and right is None:
        return None, ""
    if right is None or (left is not None and left <= right):
        return left, "left"
    return right, "right"


def _band_source_edge_geometry(
    band: dict[str, Any],
    rows: list[dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    axis: base.AxisTransform | None,
    image_width_pixels: int | None,
    pixels_per_um_x: float | None,
    fallback_detector_width_um: float,
    sample_count: int,
) -> dict[str, Any]:
    """Measure direct and fitted distances to the source left/right edges.

    All distances are based on stoma or fitted-row *centres*.  Detector-box
    widths provide a scale but a box touching an edge is not itself evidence
    that the biological band is cut, because boxes contain annotation/model
    context around the complex.
    """

    indices = [
        int(index)
        for index in band.get("_point_indices", [])
        if 0 <= int(index) < len(stoma_traits)
    ]
    member_widths_pixels: list[float] = []
    member_left: list[float] = []
    member_right: list[float] = []
    detector_box_left: list[float] = []
    detector_box_right: list[float] = []
    longitudinal: list[float] = []

    for index in indices:
        stoma = stoma_traits[index]
        x = _finite_number(stoma.get("center_x_pixels"))
        x1 = _finite_number(stoma.get("prediction_x1_pixels"))
        x2 = _finite_number(stoma.get("prediction_x2_pixels"))
        long_value = _finite_number(stoma.get("longitudinal_coordinate_um"))
        if x1 is not None and x2 is not None and x2 > x1:
            member_widths_pixels.append(x2 - x1)
            if image_width_pixels is not None:
                detector_box_left.append(x1)
                detector_box_right.append(float(image_width_pixels) - x2)
        if x is not None and image_width_pixels is not None:
            member_left.append(x)
            member_right.append(float(image_width_pixels) - x)
        if long_value is not None:
            longitudinal.append(long_value)

    detector_width_pixels = None
    detector_width_source = ""
    if (
        pixels_per_um_x is not None
        and pixels_per_um_x > 0
        and fallback_detector_width_um > 0
    ):
        detector_width_pixels = fallback_detector_width_um * pixels_per_um_x
        detector_width_source = "image_median_detector_width"
    if detector_width_pixels is None and member_widths_pixels:
        detector_width_pixels = float(np.median(member_widths_pixels))
        detector_width_source = "band_member_median_detector_width_fallback"
    if detector_width_pixels is None:
        detector_width_pixels = _finite_number(
            band.get("band_median_detector_box_width_pixels")
        )
        if detector_width_pixels is not None:
            detector_width_source = "stored_band_median_detector_width_fallback"

    member_margin, member_side = _minimum_with_side(member_left, member_right)
    member_factor = base.safe_divide(member_margin, detector_width_pixels)
    detector_box_margin, detector_box_side = _minimum_with_side(
        detector_box_left,
        detector_box_right,
    )

    fitted_left: list[float] = []
    fitted_right: list[float] = []
    if (
        axis is not None
        and image_width_pixels is not None
        and pixels_per_um_x is not None
        and pixels_per_um_x > 0
        and rows
        and longitudinal
    ):
        long_min = min(longitudinal)
        long_max = max(longitudinal)
        if long_max > long_min:
            samples = np.linspace(
                long_min,
                long_max,
                max(3, int(sample_count)),
                dtype=float,
            ).tolist()
        else:
            samples = [long_min]
        samples.extend(longitudinal)
        for longitudinal_um in samples:
            for row in rows:
                lateral_um = base.row_lateral_at(row, float(longitudinal_um))
                x_um, _ = axis.inverse(lateral_um, float(longitudinal_um))
                x_pixels = float(x_um) * pixels_per_um_x
                fitted_left.append(x_pixels)
                fitted_right.append(float(image_width_pixels) - x_pixels)

    fitted_margin, fitted_side = _minimum_with_side(fitted_left, fitted_right)
    fitted_factor = base.safe_divide(fitted_margin, detector_width_pixels)

    return {
        "band_member_center_left_edge_distance_pixels": (
            min(member_left) if member_left else None
        ),
        "band_member_center_right_edge_distance_pixels": (
            min(member_right) if member_right else None
        ),
        "band_member_center_left_edge_distance_um": (
            base.safe_divide(min(member_left), pixels_per_um_x)
            if member_left and pixels_per_um_x is not None
            else None
        ),
        "band_member_center_right_edge_distance_um": (
            base.safe_divide(min(member_right), pixels_per_um_x)
            if member_right and pixels_per_um_x is not None
            else None
        ),
        "band_member_center_nearest_side": member_side,
        "band_member_center_side_edge_margin_pixels": member_margin,
        "band_member_center_side_edge_margin_box_width_factor": member_factor,
        "band_member_center_left_edge_distance_box_width_factor": (
            base.safe_divide(min(member_left), detector_width_pixels)
            if member_left
            else None
        ),
        "band_member_center_right_edge_distance_box_width_factor": (
            base.safe_divide(min(member_right), detector_width_pixels)
            if member_right
            else None
        ),
        "band_detector_box_left_edge_clearance_pixels": (
            min(detector_box_left) if detector_box_left else None
        ),
        "band_detector_box_right_edge_clearance_pixels": (
            min(detector_box_right) if detector_box_right else None
        ),
        "band_detector_box_left_edge_clearance_um": (
            base.safe_divide(min(detector_box_left), pixels_per_um_x)
            if detector_box_left and pixels_per_um_x is not None
            else None
        ),
        "band_detector_box_right_edge_clearance_um": (
            base.safe_divide(min(detector_box_right), pixels_per_um_x)
            if detector_box_right and pixels_per_um_x is not None
            else None
        ),
        "band_detector_box_nearest_side": detector_box_side,
        "band_detector_box_side_edge_clearance_pixels": detector_box_margin,
        "band_detector_box_side_edge_clearance_box_width_factor": (
            base.safe_divide(detector_box_margin, detector_width_pixels)
        ),
        "band_fitted_row_center_left_edge_distance_pixels": (
            min(fitted_left) if fitted_left else None
        ),
        "band_fitted_row_center_right_edge_distance_pixels": (
            min(fitted_right) if fitted_right else None
        ),
        "band_fitted_row_center_left_edge_distance_um": (
            base.safe_divide(min(fitted_left), pixels_per_um_x)
            if fitted_left and pixels_per_um_x is not None
            else None
        ),
        "band_fitted_row_center_right_edge_distance_um": (
            base.safe_divide(min(fitted_right), pixels_per_um_x)
            if fitted_right and pixels_per_um_x is not None
            else None
        ),
        "band_fitted_row_center_nearest_side": fitted_side,
        "band_fitted_row_center_side_edge_margin_pixels": fitted_margin,
        "band_fitted_row_center_side_edge_margin_box_width_factor": fitted_factor,
        "band_fitted_row_center_left_edge_distance_box_width_factor": (
            base.safe_divide(min(fitted_left), detector_width_pixels)
            if fitted_left
            else None
        ),
        "band_fitted_row_center_right_edge_distance_box_width_factor": (
            base.safe_divide(min(fitted_right), detector_width_pixels)
            if fitted_right
            else None
        ),
        "band_edge_scale_detector_box_width_pixels": detector_width_pixels,
        "band_edge_scale_detector_box_width_source": detector_width_source,
        "band_edge_member_stoma_count": len(indices),
        "band_edge_fitted_sample_count": len(fitted_left),
    }


def _row_source_edge_geometries(
    rows: list[dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    axis: base.AxisTransform | None,
    image_width_pixels: int | None,
    pixels_per_um_x: float | None,
    detector_width_pixels: float | None,
    sample_count: int,
) -> list[dict[str, Any]]:
    """Measure direct and fitted source-edge clearance for every row.

    Each fitted trajectory is evaluated only across that row's observed
    longitudinal support.  This avoids allowing one edgeward row to determine
    the eligibility of every other row in a multirow band.
    """

    row_geometries: list[dict[str, Any]] = []
    for row in rows:
        indices = [
            int(index)
            for index in row.get("_point_indices", [])
            if 0 <= int(index) < len(stoma_traits)
        ]
        member_left: list[float] = []
        member_right: list[float] = []
        longitudinal: list[float] = []
        for index in indices:
            stoma = stoma_traits[index]
            x = _finite_number(stoma.get("center_x_pixels"))
            long_value = _finite_number(
                stoma.get("longitudinal_coordinate_um")
            )
            if x is not None and image_width_pixels is not None:
                member_left.append(x)
                member_right.append(float(image_width_pixels) - x)
            if long_value is not None:
                longitudinal.append(long_value)

        fitted_left: list[float] = []
        fitted_right: list[float] = []
        if (
            axis is not None
            and image_width_pixels is not None
            and pixels_per_um_x is not None
            and pixels_per_um_x > 0
            and longitudinal
        ):
            long_min = min(longitudinal)
            long_max = max(longitudinal)
            if long_max > long_min:
                samples = np.linspace(
                    long_min,
                    long_max,
                    max(3, int(sample_count)),
                    dtype=float,
                ).tolist()
            else:
                samples = [long_min]
            samples.extend(longitudinal)
            for longitudinal_um in samples:
                lateral_um = base.row_lateral_at(row, float(longitudinal_um))
                x_um, _ = axis.inverse(lateral_um, float(longitudinal_um))
                x_pixels = float(x_um) * pixels_per_um_x
                fitted_left.append(x_pixels)
                fitted_right.append(float(image_width_pixels) - x_pixels)

        member_left_factor = (
            base.safe_divide(min(member_left), detector_width_pixels)
            if member_left
            else None
        )
        member_right_factor = (
            base.safe_divide(min(member_right), detector_width_pixels)
            if member_right
            else None
        )
        fitted_left_factor = (
            base.safe_divide(min(fitted_left), detector_width_pixels)
            if fitted_left
            else None
        )
        fitted_right_factor = (
            base.safe_divide(min(fitted_right), detector_width_pixels)
            if fitted_right
            else None
        )
        row_geometries.append(
            {
                "row_id": str(row.get("row_id") or ""),
                "row_member_center_left_edge_distance_pixels": (
                    min(member_left) if member_left else None
                ),
                "row_member_center_right_edge_distance_pixels": (
                    min(member_right) if member_right else None
                ),
                "row_member_center_left_edge_distance_box_width_factor": (
                    member_left_factor
                ),
                "row_member_center_right_edge_distance_box_width_factor": (
                    member_right_factor
                ),
                "row_fitted_center_left_edge_distance_pixels": (
                    min(fitted_left) if fitted_left else None
                ),
                "row_fitted_center_right_edge_distance_pixels": (
                    min(fitted_right) if fitted_right else None
                ),
                "row_fitted_center_left_edge_distance_box_width_factor": (
                    fitted_left_factor
                ),
                "row_fitted_center_right_edge_distance_box_width_factor": (
                    fitted_right_factor
                ),
                "row_edge_member_stoma_count": len(indices),
                "row_edge_fitted_sample_count": len(fitted_left),
                "row_edge_geometry_assessable": all(
                    value is not None
                    for value in (
                        member_left_factor,
                        member_right_factor,
                        fitted_left_factor,
                        fitted_right_factor,
                    )
                ),
            }
        )
    return row_geometries


def _outermost_assessed_side(
    geometry: dict[str, Any],
    is_leftmost: bool,
    is_rightmost: bool,
) -> str:
    """Choose the biologically relevant source edge for an outermost band."""

    if is_leftmost and not is_rightmost:
        return "left"
    if is_rightmost and not is_leftmost:
        return "right"
    if not is_leftmost and not is_rightmost:
        return "none"
    left = _finite_number(
        geometry.get("band_member_center_left_edge_distance_box_width_factor")
    )
    right = _finite_number(
        geometry.get("band_member_center_right_edge_distance_box_width_factor")
    )
    if left is None and right is None:
        return "unknown"
    if right is None or (left is not None and left <= right):
        return "left"
    return "right"


def _row_aware_half_cut_decision(
    geometry: dict[str, Any],
    row_geometries: list[dict[str, Any]],
    expected_row_count: int | None,
    is_leftmost: bool,
    is_rightmost: bool,
    single_row_clearance_limit: float,
    multirow_clearance_limit: float,
    fitted_clearance_limit: float,
    adjacent_buffer_limit: float,
    detector_touch_tolerance_pixels: float,
) -> dict[str, Any]:
    """Apply the v1.9 single-row and all-rows multirow exclusion rules."""

    # A sole detected band is both leftmost and rightmost.  Assess both sides
    # independently so a slightly nearer non-truncated side cannot mask clear
    # all-row truncation at the opposite source edge.
    if is_leftmost and is_rightmost:
        left_result = _row_aware_half_cut_decision(
            geometry,
            row_geometries,
            expected_row_count,
            True,
            False,
            single_row_clearance_limit,
            multirow_clearance_limit,
            fitted_clearance_limit,
            adjacent_buffer_limit,
            detector_touch_tolerance_pixels,
        )
        right_result = _row_aware_half_cut_decision(
            geometry,
            row_geometries,
            expected_row_count,
            False,
            True,
            single_row_clearance_limit,
            multirow_clearance_limit,
            fitted_clearance_limit,
            adjacent_buffer_limit,
            detector_touch_tolerance_pixels,
        )
        confirmed = [
            result
            for result in (left_result, right_result)
            if bool(result.get("half_cut"))
        ]
        nearest_side = _outermost_assessed_side(geometry, True, True)
        chosen = (
            left_result if nearest_side != "right" else right_result
        )
        if len(confirmed) == 1:
            chosen = confirmed[0]
        elif len(confirmed) == 2:
            chosen = dict(chosen)
            chosen["half_cut_side"] = "both"
            chosen["decision_source"] = (
                "sole_outermost_band_all_rows_meet_cutoff_at_both_edges"
            )
        chosen = dict(chosen)
        chosen["sole_band_both_source_edges_assessed"] = True
        chosen["sole_band_left_edge_half_cut_confirmed"] = bool(
            left_result.get("half_cut")
        )
        chosen["sole_band_right_edge_half_cut_confirmed"] = bool(
            right_result.get("half_cut")
        )
        return chosen

    assessed_side = _outermost_assessed_side(
        geometry,
        is_leftmost,
        is_rightmost,
    )
    evaluated_row_count = len(row_geometries)
    declared_row_count = (
        int(expected_row_count)
        if expected_row_count is not None and int(expected_row_count) >= 0
        else evaluated_row_count
    )
    row_linkage_complete = declared_row_count == evaluated_row_count
    single_row = declared_row_count == 1
    threshold = (
        single_row_clearance_limit
        if single_row
        else multirow_clearance_limit
    )
    rule_branch = (
        "single_row_edge_priority"
        if single_row
        else "multirow_all_rows_must_meet_cutoff"
    )

    if assessed_side == "none":
        return {
            "half_cut": False,
            "assessment_state": "laterally_contained",
            "assessment_assessable": True,
            "half_cut_side": "none",
            "same_side_half_cut_corroborated": False,
            "decision_source": "internal_band_not_eligible_for_edge_exclusion",
            "decision_rule_branch": "internal_band_not_assessed",
            "assessed_side": "none",
            "declared_row_count": declared_row_count,
            "evaluated_row_count": evaluated_row_count,
            "row_linkage_complete": row_linkage_complete,
            "assessable_row_count": 0,
            "rows_meeting_cutoff_count": 0,
            "all_rows_meet_cutoff": False,
            "member_center_threshold": threshold,
            "row_decisions": [],
            "sole_band_both_source_edges_assessed": False,
            "sole_band_left_edge_half_cut_confirmed": False,
            "sole_band_right_edge_half_cut_confirmed": False,
        }

    if assessed_side == "unknown" or declared_row_count == 0:
        return {
            "half_cut": False,
            "assessment_state": "unknown",
            "assessment_assessable": False,
            "half_cut_side": "unknown",
            "same_side_half_cut_corroborated": False,
            "decision_source": (
                "outermost_band_without_assessable_row_geometry_retained_with_qc"
            ),
            "decision_rule_branch": rule_branch,
            "assessed_side": assessed_side,
            "declared_row_count": declared_row_count,
            "evaluated_row_count": evaluated_row_count,
            "row_linkage_complete": row_linkage_complete,
            "assessable_row_count": 0,
            "rows_meeting_cutoff_count": 0,
            "all_rows_meet_cutoff": False,
            "member_center_threshold": threshold,
            "row_decisions": [],
            "sole_band_both_source_edges_assessed": False,
            "sole_band_left_edge_half_cut_confirmed": False,
            "sole_band_right_edge_half_cut_confirmed": False,
        }

    row_decisions: list[dict[str, Any]] = []
    for row_geometry in row_geometries:
        member_factor = _finite_number(
            row_geometry.get(
                f"row_member_center_{assessed_side}_edge_distance_"
                "box_width_factor"
            )
        )
        fitted_factor = _finite_number(
            row_geometry.get(
                f"row_fitted_center_{assessed_side}_edge_distance_"
                "box_width_factor"
            )
        )
        assessable = member_factor is not None and fitted_factor is not None
        criterion_met = bool(
            assessable
            and member_factor <= threshold
            and fitted_factor <= fitted_clearance_limit
        )
        row_decisions.append(
            {
                **row_geometry,
                "row_lateral_edge_assessed_side": assessed_side,
                "row_lateral_member_center_side_edge_margin_box_width_factor": (
                    member_factor
                ),
                "row_lateral_fitted_side_edge_margin_box_width_factor": (
                    fitted_factor
                ),
                "row_lateral_member_center_threshold_box_width_factor": (
                    threshold
                ),
                "row_lateral_fitted_threshold_box_width_factor": (
                    fitted_clearance_limit
                ),
                "row_lateral_edge_geometry_assessable": assessable,
                "row_lateral_half_cut_criterion_met": criterion_met,
            }
        )

    assessable_count = sum(
        bool(row["row_lateral_edge_geometry_assessable"])
        for row in row_decisions
    )
    rows_meeting = sum(
        bool(row["row_lateral_half_cut_criterion_met"])
        for row in row_decisions
    )
    all_rows_assessable = bool(
        row_linkage_complete and assessable_count == declared_row_count
    )
    all_rows_meet = bool(
        declared_row_count > 0
        and all_rows_assessable
        and rows_meeting == declared_row_count
    )

    detector_margin = _finite_number(
        geometry.get(
            f"band_detector_box_{assessed_side}_edge_clearance_pixels"
        )
    )
    detector_box_touches = bool(
        detector_margin is not None
        and detector_margin <= detector_touch_tolerance_pixels
    )
    any_member_near = any(
        _finite_number(
            row.get(
                "row_lateral_member_center_side_edge_margin_box_width_factor"
            )
        )
        is not None
        and float(
            row["row_lateral_member_center_side_edge_margin_box_width_factor"]
        )
        <= adjacent_buffer_limit
        for row in row_decisions
    )
    any_fitted_near = any(
        _finite_number(
            row.get("row_lateral_fitted_side_edge_margin_box_width_factor")
        )
        is not None
        and float(row["row_lateral_fitted_side_edge_margin_box_width_factor"])
        <= adjacent_buffer_limit
        for row in row_decisions
    )

    if all_rows_meet:
        state = "half_cut"
        source = (
            "outermost_single_row_same_side_member_and_fitted_criteria_met"
            if single_row
            else "outermost_multirow_all_rows_meet_same_side_cutoff"
        )
    elif not all_rows_assessable:
        state = "edge_adjacent_uncertain"
        source = "outermost_band_has_incomplete_row_edge_geometry_retained_with_qc"
    elif rows_meeting > 0:
        state = "edge_adjacent_uncertain"
        source = (
            "only_subset_of_multirow_band_meets_same_side_cutoff_retained_with_qc"
        )
    elif any_member_near or any_fitted_near or detector_box_touches:
        state = "edge_adjacent_uncertain"
        source = (
            "outermost_band_near_edge_without_confirmed_all_rows_truncation"
        )
    else:
        state = "laterally_contained"
        source = "outermost_band_rows_clear_of_rule_specific_half_cut_limit"

    return {
        "half_cut": all_rows_meet,
        "assessment_state": state,
        "assessment_assessable": all_rows_assessable,
        "half_cut_side": assessed_side if all_rows_meet else "none",
        "same_side_half_cut_corroborated": all_rows_meet,
        "decision_source": source,
        "decision_rule_branch": rule_branch,
        "assessed_side": assessed_side,
        "declared_row_count": declared_row_count,
        "evaluated_row_count": evaluated_row_count,
        "row_linkage_complete": row_linkage_complete,
        "assessable_row_count": assessable_count,
        "rows_meeting_cutoff_count": rows_meeting,
        "all_rows_meet_cutoff": all_rows_meet,
        "member_center_threshold": threshold,
        "row_decisions": row_decisions,
        "sole_band_both_source_edges_assessed": False,
        "sole_band_left_edge_half_cut_confirmed": False,
        "sole_band_right_edge_half_cut_confirmed": False,
    }


def _remark_partial_band_eligibility_v19(
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    trajectory_slope: float,
    settings: TraitSettingsV19,
) -> list[dict[str, Any]]:
    """Retain complete lateral bands and separately flag short support."""

    parent_audit = _V16_PARTIAL_BAND_ELIGIBILITY(
        row_records,
        band_records,
        point_assignment,
        stoma_traits,
        median_detector_width_um,
        trajectory_slope,
        settings,
    )
    parent_by_band = {
        str(record.get("band_id")): dict(record) for record in parent_audit
    }
    row_lookup = {str(row.get("row_id")): row for row in row_records}

    metadata = v14._ACTIVE_IMAGE_METADATA
    axis = v14._active_axis(stoma_traits)
    image_width_pixels = (
        base.optional_int(metadata.get("image_width"))
        if metadata is not None
        else None
    )
    pixels_per_um_x = (
        _finite_number(metadata.get("metadata_pixels_per_um_x"))
        if metadata is not None
        else None
    )
    single_row_clearance_limit = float(
        settings.partial_band_single_row_center_clearance_box_width_factor
    )
    multirow_clearance_limit = float(
        settings.partial_band_multirow_center_clearance_box_width_factor
    )
    fitted_clearance_limit = float(
        settings.partial_band_fitted_center_corroboration_box_width_factor
    )
    adjacent_buffer_limit = float(
        settings.partial_band_lateral_edge_adjacent_buffer_box_width_factor
    )
    detector_touch_tolerance = float(
        settings.partial_band_detector_box_edge_touch_tolerance_pixels
    )

    ordered_bands = sorted(
        enumerate(band_records),
        key=lambda item: (
            base.optional_int(item[1].get("band_order_left_to_right"))
            is None,
            base.optional_int(item[1].get("band_order_left_to_right"))
            if base.optional_int(item[1].get("band_order_left_to_right"))
            is not None
            else float("inf"),
            _finite_number(item[1].get("band_center_lateral_um"))
            if _finite_number(item[1].get("band_center_lateral_um"))
            is not None
            else float("inf"),
            item[0],
        ),
    )
    leftmost_band_id = (
        str(ordered_bands[0][1].get("band_id")) if ordered_bands else ""
    )
    rightmost_band_id = (
        str(ordered_bands[-1][1].get("band_id")) if ordered_bands else ""
    )

    revised_audit: list[dict[str, Any]] = []
    for band in band_records:
        band_id = str(band.get("band_id"))
        parent_record = parent_by_band.get(band_id, {})
        parent_eligible = bool(band.get("band_spatial_eligible"))
        parent_reason = str(band.get("band_spatial_exclusion_reason") or "")
        rows = [
            row_lookup[str(row_id)]
            for row_id in band.get("row_ids", [])
            if str(row_id) in row_lookup
        ]
        declared_row_count = base.optional_int(
            band.get("row_count_used_for_traits")
        )
        if declared_row_count is None:
            declared_row_count = base.optional_int(band.get("row_count"))
        if declared_row_count is None:
            declared_row_count = len(band.get("row_ids", []))
        geometry = _band_source_edge_geometry(
            band,
            rows,
            stoma_traits,
            axis,
            image_width_pixels,
            pixels_per_um_x,
            median_detector_width_um,
            int(settings.partial_band_fitted_trajectory_edge_samples),
        )
        row_geometries = _row_source_edge_geometries(
            rows,
            stoma_traits,
            axis,
            image_width_pixels,
            pixels_per_um_x,
            _finite_number(
                geometry.get("band_edge_scale_detector_box_width_pixels")
            ),
            int(settings.partial_band_fitted_trajectory_edge_samples),
        )
        lateral_decision = _row_aware_half_cut_decision(
            geometry,
            row_geometries,
            declared_row_count,
            band_id == leftmost_band_id,
            band_id == rightmost_band_id,
            single_row_clearance_limit,
            multirow_clearance_limit,
            fitted_clearance_limit,
            adjacent_buffer_limit,
            detector_touch_tolerance,
        )
        half_cut = bool(lateral_decision["half_cut"])
        lateral_state = str(lateral_decision["assessment_state"])
        lateral_assessable = bool(
            lateral_decision["assessment_assessable"]
        )
        half_cut_side = str(lateral_decision["half_cut_side"])
        same_side_corroborated = bool(
            lateral_decision["same_side_half_cut_corroborated"]
        )
        decision_source = str(lateral_decision["decision_source"])
        decision_rule_branch = str(
            lateral_decision["decision_rule_branch"]
        )
        assessed_side = str(lateral_decision["assessed_side"])
        declared_row_count = int(lateral_decision["declared_row_count"])
        evaluated_row_count = int(lateral_decision["evaluated_row_count"])
        row_linkage_complete = bool(
            lateral_decision["row_linkage_complete"]
        )
        assessable_row_count = int(lateral_decision["assessable_row_count"])
        rows_meeting_cutoff = int(
            lateral_decision["rows_meeting_cutoff_count"]
        )
        all_rows_meet_cutoff = bool(
            lateral_decision["all_rows_meet_cutoff"]
        )
        sole_band_both_edges_assessed = bool(
            lateral_decision["sole_band_both_source_edges_assessed"]
        )
        sole_band_left_half_cut = bool(
            lateral_decision["sole_band_left_edge_half_cut_confirmed"]
        )
        sole_band_right_half_cut = bool(
            lateral_decision["sole_band_right_edge_half_cut_confirmed"]
        )
        member_clearance_limit = float(
            lateral_decision["member_center_threshold"]
        )
        row_decision_by_id = {
            str(record.get("row_id")): record
            for record in lateral_decision["row_decisions"]
        }

        coverage = _finite_number(
            band.get("band_longitudinal_coverage_fraction")
        )
        relative_span = _finite_number(band.get("band_relative_observed_span"))
        observed_span = _finite_number(
            band.get("band_observed_longitudinal_span_um")
        )
        longitudinal_assessable = bool(
            coverage is not None
            and relative_span is not None
            and observed_span is not None
        )
        low_absolute: bool | None = None
        low_relative: bool | None = None
        limited_relative_support: bool | None = None
        short_longitudinal_support: bool | None = None
        longitudinal_complete: bool | None = None
        longitudinal_status = "unknown"
        longitudinal_reason = "longitudinal_sampling_not_assessable"
        if longitudinal_assessable:
            low_absolute = bool(
                coverage
                < float(
                    settings.partial_band_min_longitudinal_coverage_fraction
                )
            )
            low_relative = bool(
                relative_span
                < float(settings.partial_band_min_relative_observed_span)
            )
            short_longitudinal_support = bool(
                (low_absolute and low_relative) or observed_span <= 0.0
            )
            limited_relative_support = bool(
                observed_span > 0.0 and low_relative and not low_absolute
            )
            longitudinal_complete = bool(
                observed_span > 0.0 and not low_absolute and not low_relative
            )
            if observed_span <= 0.0:
                longitudinal_status = "zero_observed_span"
                longitudinal_reason = (
                    "zero_longitudinal_span_retained_for_row_structure"
                )
            elif low_absolute and low_relative:
                longitudinal_status = "severe_short_observed_support"
                longitudinal_reason = (
                    "severe_short_longitudinal_support_retained_for_row_structure"
                )
            elif low_relative:
                longitudinal_status = "limited_relative_observed_support"
                longitudinal_reason = (
                    "limited_relative_longitudinal_extent_retained_for_row_structure"
                )
            elif low_absolute:
                longitudinal_status = "low_accessible_span_coverage"
                longitudinal_reason = (
                    "low_accessible_span_coverage_retained_for_row_structure"
                )
            else:
                longitudinal_status = "adequate"
                longitudinal_reason = ""
        row_structure_eligible = not half_cut
        completeness_eligible = bool(
            row_structure_eligible
            and lateral_assessable
            and lateral_state == "laterally_contained"
            and longitudinal_complete is True
        )
        exclusion_reason = (
            "lateral_band_cross_section_half_cut_by_source_image_edge"
            if half_cut
            else ""
        )
        legacy_fov_truncated: bool | None = (
            True
            if half_cut
            else False
            if lateral_state == "laterally_contained"
            else None
        )
        band.update(
            {
                "partial_band_eligibility_version": (
                    PARTIAL_BAND_ELIGIBILITY_VERSION
                ),
                "band_previous_v1_7_spatial_eligible": parent_eligible,
                "band_previous_v1_7_spatial_exclusion_reason": parent_reason,
                "band_row_structure_eligible": row_structure_eligible,
                "band_lateral_half_cut": half_cut,
                "band_lateral_half_cut_side": half_cut_side,
                "band_lateral_same_side_half_cut_corroborated": (
                    same_side_corroborated
                ),
                "band_lateral_edge_assessment_state": lateral_state,
                "band_lateral_edge_assessment_assessable": lateral_assessable,
                "band_lateral_half_cut_decision_source": decision_source,
                "band_lateral_decision_rule_branch": decision_rule_branch,
                "band_lateral_assessed_side": assessed_side,
                "band_lateral_is_leftmost_band": (
                    band_id == leftmost_band_id
                ),
                "band_lateral_is_rightmost_band": (
                    band_id == rightmost_band_id
                ),
                "band_lateral_evaluated_row_count": evaluated_row_count,
                "band_lateral_declared_row_count": declared_row_count,
                "band_lateral_row_linkage_complete": row_linkage_complete,
                "band_lateral_assessable_row_count": assessable_row_count,
                "band_lateral_rows_meeting_cutoff_count": (
                    rows_meeting_cutoff
                ),
                "band_lateral_all_rows_meet_cutoff": (
                    all_rows_meet_cutoff
                ),
                "band_lateral_sole_band_both_source_edges_assessed": (
                    sole_band_both_edges_assessed
                ),
                "band_lateral_sole_band_left_edge_half_cut_confirmed": (
                    sole_band_left_half_cut
                ),
                "band_lateral_sole_band_right_edge_half_cut_confirmed": (
                    sole_band_right_half_cut
                ),
                "band_lateral_half_cut_clearance_threshold_box_width_factor": (
                    member_clearance_limit
                ),
                "band_lateral_single_row_clearance_threshold_box_width_factor": (
                    single_row_clearance_limit
                ),
                "band_lateral_multirow_clearance_threshold_box_width_factor": (
                    multirow_clearance_limit
                ),
                "band_lateral_fitted_corroboration_threshold_box_width_factor": (
                    fitted_clearance_limit
                ),
                "band_lateral_edge_adjacent_buffer_box_width_factor": (
                    adjacent_buffer_limit
                ),
                "band_short_longitudinal_support": short_longitudinal_support,
                "band_severe_short_observed_longitudinal_support": (
                    short_longitudinal_support
                ),
                "band_limited_relative_longitudinal_extent": (
                    limited_relative_support
                ),
                "band_low_relative_observed_support": low_relative,
                "band_low_accessible_span_coverage": low_absolute,
                "band_longitudinal_sampling_assessable": (
                    longitudinal_assessable
                ),
                "band_longitudinal_sampling_complete": longitudinal_complete,
                "band_longitudinal_sampling_status": longitudinal_status,
                "band_longitudinal_sampling_qc_reason": longitudinal_reason,
                "band_completeness_dependent_traits_eligible": (
                    completeness_eligible
                ),
                "band_spatial_eligible": row_structure_eligible,
                "band_fov_truncated": legacy_fov_truncated,
                "band_confirmed_lateral_half_cut": half_cut,
                "band_spatial_exclusion_reason": exclusion_reason,
                **geometry,
            }
        )

        for row in rows:
            row_decision = row_decision_by_id.get(
                str(row.get("row_id")),
                {},
            )
            row_specific_qc = {
                "row_band_lateral_decision_rule_branch": (
                    decision_rule_branch
                ),
                "row_band_lateral_assessed_side": assessed_side,
                "row_band_lateral_is_leftmost_band": (
                    band_id == leftmost_band_id
                ),
                "row_band_lateral_is_rightmost_band": (
                    band_id == rightmost_band_id
                ),
                "row_band_lateral_evaluated_row_count": evaluated_row_count,
                "row_band_lateral_declared_row_count": declared_row_count,
                "row_band_lateral_row_linkage_complete": (
                    row_linkage_complete
                ),
                "row_band_lateral_assessable_row_count": assessable_row_count,
                "row_band_lateral_rows_meeting_cutoff_count": (
                    rows_meeting_cutoff
                ),
                "row_band_lateral_all_rows_meet_cutoff": (
                    all_rows_meet_cutoff
                ),
                "row_band_lateral_sole_band_both_source_edges_assessed": (
                    sole_band_both_edges_assessed
                ),
                "row_band_lateral_sole_band_left_edge_half_cut_confirmed": (
                    sole_band_left_half_cut
                ),
                "row_band_lateral_sole_band_right_edge_half_cut_confirmed": (
                    sole_band_right_half_cut
                ),
                "row_band_lateral_row_geometry_assessable": (
                    row_decision.get("row_lateral_edge_geometry_assessable")
                ),
                "row_band_lateral_row_half_cut_criterion_met": (
                    row_decision.get("row_lateral_half_cut_criterion_met")
                ),
                "row_band_lateral_row_member_center_side_edge_margin_"
                "box_width_factor": row_decision.get(
                    "row_lateral_member_center_side_edge_margin_"
                    "box_width_factor"
                ),
                "row_band_lateral_row_fitted_side_edge_margin_"
                "box_width_factor": row_decision.get(
                    "row_lateral_fitted_side_edge_margin_box_width_factor"
                ),
                "row_band_lateral_row_member_center_threshold_"
                "box_width_factor": row_decision.get(
                    "row_lateral_member_center_threshold_box_width_factor"
                ),
                "row_band_lateral_row_fitted_threshold_box_width_factor": (
                    row_decision.get(
                        "row_lateral_fitted_threshold_box_width_factor"
                    )
                ),
            }
            row.update(
                {
                    "partial_band_eligibility_version": (
                        PARTIAL_BAND_ELIGIBILITY_VERSION
                    ),
                    "row_band_previous_v1_7_spatial_eligible": parent_eligible,
                    "row_band_previous_v1_7_spatial_exclusion_reason": (
                        parent_reason
                    ),
                    "row_structure_eligible": row_structure_eligible,
                    "row_band_lateral_half_cut": half_cut,
                    "row_band_lateral_half_cut_side": half_cut_side,
                    "row_band_lateral_same_side_half_cut_corroborated": (
                        same_side_corroborated
                    ),
                    "row_band_lateral_edge_assessment_state": lateral_state,
                    "row_band_lateral_edge_assessment_assessable": (
                        lateral_assessable
                    ),
                    "row_band_short_longitudinal_support": (
                        short_longitudinal_support
                    ),
                    "row_band_severe_short_observed_longitudinal_support": (
                        short_longitudinal_support
                    ),
                    "row_band_limited_relative_longitudinal_extent": (
                        limited_relative_support
                    ),
                    "row_band_low_relative_observed_support": low_relative,
                    "row_band_low_accessible_span_coverage": low_absolute,
                    "row_band_longitudinal_sampling_assessable": (
                        longitudinal_assessable
                    ),
                    "row_band_longitudinal_sampling_complete": (
                        longitudinal_complete
                    ),
                    "row_band_longitudinal_sampling_status": (
                        longitudinal_status
                    ),
                    "row_completeness_dependent_traits_eligible": (
                        completeness_eligible
                    ),
                    "row_band_spatial_eligible": row_structure_eligible,
                    "row_band_spatial_exclusion_reason": exclusion_reason,
                    **row_specific_qc,
                }
            )
            for index in row.get("_point_indices", []):
                if int(index) not in point_assignment:
                    continue
                point_assignment[int(index)].update(
                    {
                        "partial_band_eligibility_version": (
                            PARTIAL_BAND_ELIGIBILITY_VERSION
                        ),
                        "row_structure_eligible": row_structure_eligible,
                        "row_band_lateral_half_cut": half_cut,
                        "row_band_lateral_half_cut_side": half_cut_side,
                        "row_band_lateral_same_side_half_cut_corroborated": (
                            same_side_corroborated
                        ),
                        "row_band_lateral_edge_assessment_state": (
                            lateral_state
                        ),
                        "row_band_lateral_edge_assessment_assessable": (
                            lateral_assessable
                        ),
                        "row_band_short_longitudinal_support": (
                            short_longitudinal_support
                        ),
                        "row_band_severe_short_observed_longitudinal_support": (
                            short_longitudinal_support
                        ),
                        "row_band_limited_relative_longitudinal_extent": (
                            limited_relative_support
                        ),
                        "row_band_low_relative_observed_support": low_relative,
                        "row_band_low_accessible_span_coverage": low_absolute,
                        "row_band_longitudinal_sampling_assessable": (
                            longitudinal_assessable
                        ),
                        "row_band_longitudinal_sampling_complete": (
                            longitudinal_complete
                        ),
                        "row_band_longitudinal_sampling_status": (
                            longitudinal_status
                        ),
                        "row_completeness_dependent_traits_eligible": (
                            completeness_eligible
                        ),
                        "row_band_spatial_eligible": row_structure_eligible,
                        "row_band_spatial_exclusion_reason": exclusion_reason,
                        **row_specific_qc,
                    }
                )

        audit_record = dict(parent_record)
        audit_record.update(
            {
                "band_id": band_id,
                "provisional_band_id": band.get("provisional_band_id"),
                "partial_band_eligibility_version": (
                    PARTIAL_BAND_ELIGIBILITY_VERSION
                ),
                "previous_v1_7_spatial_eligible": parent_eligible,
                "previous_v1_7_exclusion_reason": parent_reason,
                "spatial_eligible": row_structure_eligible,
                "row_structure_eligible": row_structure_eligible,
                "lateral_half_cut": half_cut,
                "lateral_half_cut_side": half_cut_side,
                "lateral_same_side_half_cut_corroborated": (
                    same_side_corroborated
                ),
                "fov_truncated": legacy_fov_truncated,
                "confirmed_lateral_half_cut": half_cut,
                "lateral_edge_assessment_state": lateral_state,
                "lateral_edge_assessment_assessable": lateral_assessable,
                "lateral_half_cut_decision_source": decision_source,
                "lateral_decision_rule_branch": decision_rule_branch,
                "lateral_assessed_side": assessed_side,
                "lateral_is_leftmost_band": band_id == leftmost_band_id,
                "lateral_is_rightmost_band": band_id == rightmost_band_id,
                "lateral_evaluated_row_count": evaluated_row_count,
                "lateral_declared_row_count": declared_row_count,
                "lateral_row_linkage_complete": row_linkage_complete,
                "lateral_assessable_row_count": assessable_row_count,
                "lateral_rows_meeting_cutoff_count": rows_meeting_cutoff,
                "lateral_all_rows_meet_cutoff": all_rows_meet_cutoff,
                "lateral_sole_band_both_source_edges_assessed": (
                    sole_band_both_edges_assessed
                ),
                "lateral_sole_band_left_edge_half_cut_confirmed": (
                    sole_band_left_half_cut
                ),
                "lateral_sole_band_right_edge_half_cut_confirmed": (
                    sole_band_right_half_cut
                ),
                "lateral_half_cut_clearance_threshold_box_width_factor": (
                    member_clearance_limit
                ),
                "lateral_single_row_clearance_threshold_box_width_factor": (
                    single_row_clearance_limit
                ),
                "lateral_multirow_clearance_threshold_box_width_factor": (
                    multirow_clearance_limit
                ),
                "lateral_fitted_corroboration_threshold_box_width_factor": (
                    fitted_clearance_limit
                ),
                "lateral_edge_adjacent_buffer_box_width_factor": (
                    adjacent_buffer_limit
                ),
                "short_longitudinal_support": short_longitudinal_support,
                "severe_short_observed_longitudinal_support": (
                    short_longitudinal_support
                ),
                "limited_relative_longitudinal_extent": (
                    limited_relative_support
                ),
                "low_relative_observed_support": low_relative,
                "low_accessible_span_coverage": low_absolute,
                "longitudinal_sampling_assessable": longitudinal_assessable,
                "longitudinal_sampling_complete": longitudinal_complete,
                "longitudinal_sampling_status": longitudinal_status,
                "longitudinal_sampling_qc_reason": longitudinal_reason,
                "completeness_dependent_traits_eligible": completeness_eligible,
                "exclusion_reason": exclusion_reason,
                **geometry,
            }
        )
        revised_audit.append(audit_record)

    return revised_audit


def infer_rows_and_bands_v19(
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettingsV19,
    audited_row_counts: dict[str, int] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    """Run unchanged v1.6 geometry and append v1.9 eligibility provenance."""

    row_records, band_records, point_assignment, audit = (
        v16.infer_rows_and_bands_v16(
            stoma_traits,
            settings,
            audited_row_counts=audited_row_counts,
        )
    )
    audit.update(
        {
            "partial_band_eligibility_version": (
                PARTIAL_BAND_ELIGIBILITY_VERSION
            ),
            "partial_band_row_structure_policy": (
                "only an outermost band can be excluded; a one-row band uses "
                "the 0.30 member-centre threshold with same-side fitted "
                "corroboration; a multirow band uses 0.25 independently for "
                "each row and is excluded only when every inferred row meets "
                "the same-side criterion; partial-row edge evidence is retained "
                "with QC; detector-box contact alone is not decisive"
            ),
            "partial_band_longitudinal_support_policy": (
                "short longitudinal support is retained for row structure and "
                "reported separately as completeness QC"
            ),
            "partial_band_settings": {
                "single_row_member_center_half_cut_threshold_box_width_factor": (
                    settings.partial_band_single_row_center_clearance_box_width_factor
                ),
                "multirow_member_center_half_cut_threshold_box_width_factor": (
                    settings.partial_band_multirow_center_clearance_box_width_factor
                ),
                "fitted_center_corroboration_threshold_box_width_factor": (
                    settings.partial_band_fitted_center_corroboration_box_width_factor
                ),
                "edge_adjacent_buffer_box_width_factor": (
                    settings.partial_band_lateral_edge_adjacent_buffer_box_width_factor
                ),
                "detector_box_edge_touch_tolerance_pixels": (
                    settings.partial_band_detector_box_edge_touch_tolerance_pixels
                ),
                "fitted_trajectory_edge_samples": (
                    settings.partial_band_fitted_trajectory_edge_samples
                ),
                "same_side_member_and_fitted_evidence_required": True,
                "outermost_band_required_for_exclusion": True,
                "single_row_truncation_priority": True,
                "multirow_all_rows_must_meet_cutoff": True,
                "detector_bbox_contact_decisive": False,
                "thresholds_provisional_audit_derived": True,
                "short_observed_longitudinal_support_retained_for_rows": True,
            },
            "partial_band_lateral_center_clearance_box_width_factor": (
                settings.partial_band_lateral_center_clearance_box_width_factor
            ),
            "partial_band_single_row_center_clearance_box_width_factor": (
                settings.partial_band_single_row_center_clearance_box_width_factor
            ),
            "partial_band_multirow_center_clearance_box_width_factor": (
                settings.partial_band_multirow_center_clearance_box_width_factor
            ),
            "row_structure_eligible_band_count": sum(
                bool(band.get("band_row_structure_eligible"))
                for band in band_records
            ),
            "lateral_half_cut_band_count": sum(
                bool(band.get("band_lateral_half_cut"))
                for band in band_records
            ),
            "lateral_edge_adjacent_uncertain_band_count": sum(
                band.get("band_lateral_edge_assessment_state")
                == "edge_adjacent_uncertain"
                for band in band_records
            ),
            "lateral_edge_assessment_unknown_band_count": sum(
                band.get("band_lateral_edge_assessment_state") == "unknown"
                for band in band_records
            ),
            "laterally_contained_band_count": sum(
                band.get("band_lateral_edge_assessment_state")
                == "laterally_contained"
                for band in band_records
            ),
            "single_row_edge_priority_band_count": sum(
                band.get("band_lateral_decision_rule_branch")
                == "single_row_edge_priority"
                for band in band_records
            ),
            "multirow_all_rows_rule_band_count": sum(
                band.get("band_lateral_decision_rule_branch")
                == "multirow_all_rows_must_meet_cutoff"
                for band in band_records
            ),
            "retained_partial_multirow_edge_band_count": sum(
                band.get("band_lateral_decision_rule_branch")
                == "multirow_all_rows_must_meet_cutoff"
                and int(
                    band.get("band_lateral_rows_meeting_cutoff_count") or 0
                )
                > 0
                and not bool(
                    band.get("band_lateral_all_rows_meet_cutoff")
                )
                for band in band_records
            ),
            "severe_short_observed_longitudinal_support_band_count": sum(
                bool(
                    band.get(
                        "band_severe_short_observed_longitudinal_support"
                    )
                )
                for band in band_records
            ),
            "limited_relative_longitudinal_extent_band_count": sum(
                band.get("band_limited_relative_longitudinal_extent") is True
                for band in band_records
            ),
            "low_relative_observed_support_band_count": sum(
                band.get("band_low_relative_observed_support") is True
                for band in band_records
            ),
            "low_accessible_span_coverage_band_count": sum(
                band.get("band_low_accessible_span_coverage") is True
                for band in band_records
            ),
            "longitudinal_sampling_complete_band_count": sum(
                band.get("band_longitudinal_sampling_complete") is True
                for band in band_records
            ),
            "longitudinal_sampling_incomplete_band_count": sum(
                band.get("band_longitudinal_sampling_complete") is False
                for band in band_records
            ),
            "longitudinal_sampling_unknown_band_count": sum(
                band.get("band_longitudinal_sampling_complete") is None
                for band in band_records
            ),
            "longitudinal_sampling_assessable_band_count": sum(
                band.get("band_longitudinal_sampling_assessable") is True
                for band in band_records
            ),
            "completeness_dependent_traits_eligible_band_count": sum(
                bool(band.get("band_completeness_dependent_traits_eligible"))
                for band in band_records
            ),
            "partial_band_accessible_span_basis": (
                "v1.6 fitted-band longitudinal coverage retained as QC; v1.9 "
                "row eligibility uses direct source left/right edge clearance"
            ),
        }
    )
    return row_records, band_records, point_assignment, audit


def image_trait_summary_v19(
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
    """Keep v1.7 traits and add separate row/completeness summaries."""

    result = v17.image_trait_summary_v17(
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

    row_structure_bands = [
        band
        for band in band_records
        if bool(band.get("band_row_structure_eligible"))
    ]
    strict_complete_bands = [
        band
        for band in band_records
        if bool(band.get("band_completeness_dependent_traits_eligible"))
    ]
    longitudinally_complete_bands = [
        band
        for band in band_records
        if band.get("band_longitudinal_sampling_complete") is True
    ]
    longitudinally_incomplete_bands = [
        band
        for band in band_records
        if band.get("band_longitudinal_sampling_complete") is False
    ]
    longitudinally_unknown_bands = [
        band
        for band in band_records
        if band.get("band_longitudinal_sampling_complete") is None
    ]
    strict_complete_band_ids = {
        str(band.get("band_id")) for band in strict_complete_bands
    }
    complete_rows = [
        row
        for row in row_records
        if str(row.get("band_id")) in strict_complete_band_ids
        and bool(row.get("row_completeness_dependent_traits_eligible"))
    ]
    complete_stoma_indices = {
        int(index)
        for band in strict_complete_bands
        for index in band.get("_point_indices", [])
    }
    complete_row_accessible_length_um = sum(
        float(band.get("band_accessible_longitudinal_span_um") or 0.0)
        * int(band.get("row_count") or 0)
        for band in strict_complete_bands
    )
    complete_frequency = base.safe_divide(
        float(len(complete_stoma_indices)),
        complete_row_accessible_length_um / 1000.0
        if complete_row_accessible_length_um > 0.0
        else None,
    )

    result.update(
        base.sample_summary(
            (row.get("stoma_count") for row in complete_rows),
            "stomata_per_completeness_eligible_row",
        )
    )
    result.update(
        base.sample_summary(
            (band.get("stoma_count") for band in strict_complete_bands),
            "stomata_per_completeness_eligible_band",
        )
    )
    result.update(
        base.sample_summary(
            (row.get("row_continuity") for row in complete_rows),
            "completeness_eligible_row_continuity",
        )
    )
    result.update(
        {
            "partial_band_eligibility_version": (
                PARTIAL_BAND_ELIGIBILITY_VERSION
            ),
            "row_geometry_model_version": ROW_GEOMETRY_MODEL_VERSION,
            "trait_export_policy_version": TRAIT_EXPORT_POLICY_VERSION,
            "row_structure_eligible_band_count": len(row_structure_bands),
            "row_structure_eligible_row_count": sum(
                int(band.get("row_count") or 0) for band in row_structure_bands
            ),
            "lateral_half_cut_band_count": sum(
                bool(band.get("band_lateral_half_cut"))
                for band in band_records
            ),
            "lateral_edge_adjacent_uncertain_band_count": sum(
                band.get("band_lateral_edge_assessment_state")
                == "edge_adjacent_uncertain"
                for band in band_records
            ),
            "lateral_edge_assessment_unknown_band_count": sum(
                band.get("band_lateral_edge_assessment_state") == "unknown"
                for band in band_records
            ),
            "laterally_contained_band_count": sum(
                band.get("band_lateral_edge_assessment_state")
                == "laterally_contained"
                for band in band_records
            ),
            "single_row_edge_priority_band_count": sum(
                band.get("band_lateral_decision_rule_branch")
                == "single_row_edge_priority"
                for band in band_records
            ),
            "multirow_all_rows_rule_band_count": sum(
                band.get("band_lateral_decision_rule_branch")
                == "multirow_all_rows_must_meet_cutoff"
                for band in band_records
            ),
            "retained_partial_multirow_edge_band_count": sum(
                band.get("band_lateral_decision_rule_branch")
                == "multirow_all_rows_must_meet_cutoff"
                and int(
                    band.get("band_lateral_rows_meeting_cutoff_count") or 0
                )
                > 0
                and not bool(
                    band.get("band_lateral_all_rows_meet_cutoff")
                )
                for band in band_records
            ),
            "severe_short_observed_longitudinal_support_detected": any(
                bool(
                    band.get(
                        "band_severe_short_observed_longitudinal_support"
                    )
                )
                for band in band_records
            ),
            "severe_short_observed_longitudinal_support_band_count": sum(
                bool(
                    band.get(
                        "band_severe_short_observed_longitudinal_support"
                    )
                )
                for band in band_records
            ),
            "limited_relative_longitudinal_extent_band_count": sum(
                band.get("band_limited_relative_longitudinal_extent") is True
                for band in band_records
            ),
            "low_relative_observed_support_band_count": sum(
                band.get("band_low_relative_observed_support") is True
                for band in band_records
            ),
            "low_accessible_span_coverage_band_count": sum(
                band.get("band_low_accessible_span_coverage") is True
                for band in band_records
            ),
            "longitudinal_sampling_complete_band_count": len(
                longitudinally_complete_bands
            ),
            "longitudinal_sampling_incomplete_band_count": (
                len(longitudinally_incomplete_bands)
            ),
            "longitudinal_sampling_unknown_band_count": (
                len(longitudinally_unknown_bands)
            ),
            "longitudinal_sampling_assessable_band_count": sum(
                band.get("band_longitudinal_sampling_assessable") is True
                for band in band_records
            ),
            "completeness_dependent_traits_eligible_band_count": len(
                strict_complete_bands
            ),
            "completeness_dependent_traits_eligible_row_count": len(
                complete_rows
            ),
            "completeness_dependent_traits_eligible_stoma_count": len(
                complete_stoma_indices
            ),
            "completeness_eligible_total_row_accessible_length_um": (
                complete_row_accessible_length_um
            ),
            "within_row_stomatal_frequency_completeness_eligible_per_mm": (
                complete_frequency
            ),
            "partial_band_lateral_center_clearance_box_width_factor": (
                LATERAL_CENTER_EDGE_CLEARANCE_BOX_WIDTH_FACTOR
            ),
            "partial_band_single_row_center_clearance_box_width_factor": (
                SINGLE_ROW_CENTER_EDGE_CLEARANCE_BOX_WIDTH_FACTOR
            ),
            "partial_band_multirow_center_clearance_box_width_factor": (
                MULTIROW_CENTER_EDGE_CLEARANCE_BOX_WIDTH_FACTOR
            ),
            "partial_band_fitted_center_corroboration_box_width_factor": (
                FITTED_CENTER_EDGE_CORROBORATION_BOX_WIDTH_FACTOR
            ),
            "row_band_spatial_eligibility_policy": (
                "only outermost bands can be excluded; one-row bands use the "
                "0.30 same-side member/fitted truncation-priority rule; "
                "multirow bands use 0.25 per row and are excluded only when "
                "every inferred row meets the criterion; partial-row edge "
                "evidence is retained with QC; short longitudinal support is "
                "retained and flagged"
            ),
        }
    )
    v19_qc_reasons: list[str] = []
    for field, reason in (
        (
            "lateral_half_cut_band_count",
            "confirmed_lateral_half_cut_band_present",
        ),
        (
            "lateral_edge_adjacent_uncertain_band_count",
            "edge_adjacent_uncertain_band_present",
        ),
        (
            "lateral_edge_assessment_unknown_band_count",
            "lateral_edge_geometry_unknown_band_present",
        ),
        (
            "longitudinal_sampling_incomplete_band_count",
            "longitudinally_incomplete_band_present",
        ),
        (
            "longitudinal_sampling_unknown_band_count",
            "longitudinal_sampling_unknown_band_present",
        ),
    ):
        if int(result.get(field) or 0) > 0:
            v19_qc_reasons.append(reason)
    result.update(
        {
            "row_band_v19_qc_status": (
                "flagged_provisional" if v19_qc_reasons else "pass"
            ),
            "row_band_v19_qc_flag": bool(v19_qc_reasons),
            "row_band_v19_qc_reason": ";".join(v19_qc_reasons),
            "row_band_v19_value_interpretation": (
                "computed row/band values are retained; use the v1.9 lateral "
                "and longitudinal band-level fields plus contributing sample "
                "counts for trait-specific sensitivity analysis"
            ),
        }
    )
    return result


_V19_GWAS_FIELDS = (
    "partial_band_eligibility_version",
    "row_structure_eligible_band_count",
    "row_structure_eligible_row_count",
    "lateral_half_cut_band_count",
    "lateral_edge_adjacent_uncertain_band_count",
    "lateral_edge_assessment_unknown_band_count",
    "laterally_contained_band_count",
    "single_row_edge_priority_band_count",
    "multirow_all_rows_rule_band_count",
    "retained_partial_multirow_edge_band_count",
    "severe_short_observed_longitudinal_support_detected",
    "severe_short_observed_longitudinal_support_band_count",
    "limited_relative_longitudinal_extent_band_count",
    "low_relative_observed_support_band_count",
    "low_accessible_span_coverage_band_count",
    "longitudinal_sampling_complete_band_count",
    "longitudinal_sampling_incomplete_band_count",
    "longitudinal_sampling_unknown_band_count",
    "longitudinal_sampling_assessable_band_count",
    "completeness_dependent_traits_eligible_band_count",
    "completeness_dependent_traits_eligible_row_count",
    "completeness_dependent_traits_eligible_stoma_count",
    "completeness_eligible_total_row_accessible_length_um",
    "within_row_stomatal_frequency_completeness_eligible_per_mm",
    "stomata_per_completeness_eligible_row_n",
    "stomata_per_completeness_eligible_row_mean",
    "stomata_per_completeness_eligible_band_n",
    "stomata_per_completeness_eligible_band_mean",
    "completeness_eligible_row_continuity_n",
    "completeness_eligible_row_continuity_mean",
    "partial_band_lateral_center_clearance_box_width_factor",
    "partial_band_single_row_center_clearance_box_width_factor",
    "partial_band_multirow_center_clearance_box_width_factor",
    "partial_band_fitted_center_corroboration_box_width_factor",
    "row_band_spatial_eligibility_policy",
    "row_band_v19_qc_status",
    "row_band_v19_qc_flag",
    "row_band_v19_qc_reason",
    "row_band_v19_value_interpretation",
)


def image_gwas_trait_summary_v19(
    image_traits: dict[str, Any],
) -> dict[str, Any]:
    """Carry v1.9 eligibility traits/QC into the compact image table."""

    summary = v17.image_gwas_trait_summary_v17(image_traits)
    for field in _V19_GWAS_FIELDS:
        summary[field] = image_traits.get(field)
    summary["row_geometry_model_version"] = ROW_GEOMETRY_MODEL_VERSION
    summary["trait_export_policy_version"] = TRAIT_EXPORT_POLICY_VERSION
    return summary


def add_row_neighbors_v19(
    stoma_traits: list[dict[str, Any]],
    row_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
) -> None:
    """Retain normal neighbor traits and propagate every v1.9 QC field."""

    v15.add_row_neighbors_v15(stoma_traits, row_records, point_assignment)
    for index, assignment in point_assignment.items():
        if not 0 <= int(index) < len(stoma_traits):
            continue
        eligibility_qc = {
            key: value
            for key, value in assignment.items()
            if key == "partial_band_eligibility_version"
            or key.startswith("row_structure_")
            or key.startswith("row_band_lateral_")
            or key.startswith("row_band_short_")
            or key.startswith("row_band_severe_")
            or key.startswith("row_band_limited_")
            or key.startswith("row_band_low_")
            or key.startswith("row_band_longitudinal_")
            or key.startswith("row_completeness_")
            or key.startswith("row_band_spatial_")
        }
        stoma_traits[int(index)].update(eligibility_qc)


def make_spatial_audit_overlay_v19(
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
    """Add purple longitudinal-QC and gold lateral-edge-QC marks."""

    v15.make_spatial_audit_overlay_v15(
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
    retained_longitudinal_qc = [
        band
        for band in band_records
        if bool(band.get("band_row_structure_eligible"))
        and band.get("band_longitudinal_sampling_complete") is False
    ]
    retained_edge_uncertain = [
        band
        for band in band_records
        if bool(band.get("band_row_structure_eligible"))
        and band.get("band_lateral_edge_assessment_state")
        == "edge_adjacent_uncertain"
    ]
    if not retained_longitudinal_qc and not retained_edge_uncertain:
        return
    image = base.cv2.imread(str(output_path), base.cv2.IMREAD_COLOR)
    if image is None:
        return
    purple = (180, 0, 180)
    for band in retained_longitudinal_qc:
        members = [int(index) for index in band.get("_point_indices", [])]
        for index in members:
            x = int(round(stoma_traits[index]["center_x_pixels"]))
            y = int(round(stoma_traits[index]["center_y_pixels"]))
            base.cv2.rectangle(
                image,
                (x - 13, y - 13),
                (x + 13, y + 13),
                purple,
                2,
                base.cv2.LINE_AA,
            )
        if members:
            x = int(
                round(
                    np.mean(
                        [
                            stoma_traits[index]["center_x_pixels"]
                            for index in members
                        ]
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
                - 32,
            )
            base.cv2.putText(
                image,
                f"{band['band_id']} LONGITUDINAL QC",
                (max(5, x - 105), y),
                base.cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                purple,
                2,
                base.cv2.LINE_AA,
            )
    gold = (0, 190, 255)
    for band in retained_edge_uncertain:
        members = [int(index) for index in band.get("_point_indices", [])]
        for index in members:
            x = int(round(stoma_traits[index]["center_x_pixels"]))
            y = int(round(stoma_traits[index]["center_y_pixels"]))
            base.cv2.circle(image, (x, y), 18, gold, 2, base.cv2.LINE_AA)
        if members:
            x = int(
                round(
                    np.mean(
                        [
                            stoma_traits[index]["center_x_pixels"]
                            for index in members
                        ]
                    )
                )
            )
            y = min(
                image.shape[0] - 10,
                int(
                    round(
                        max(
                            stoma_traits[index]["center_y_pixels"]
                            for index in members
                        )
                    )
                )
                + 34,
            )
            base.cv2.putText(
                image,
                f"{band['band_id']} LATERAL EDGE QC",
                (max(5, x - 105), y),
                base.cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                gold,
                2,
                base.cv2.LINE_AA,
            )
    if not base.cv2.imwrite(
        str(output_path), image, [base.cv2.IMWRITE_PNG_COMPRESSION, 6]
    ):
        raise RuntimeError(f"Failed to update v1.9 audit overlay: {output_path}")


def install_v19_partial_band_policy() -> None:
    """Install v1.7 first and then replace only partial-band eligibility."""

    v17.install_v17_export_policy()

    # This is the dynamic hook called inside the saved v1.5 inference function.
    v15._remark_partial_band_eligibility_v15 = (
        _remark_partial_band_eligibility_v19
    )

    base.build_settings = build_settings_v19
    base.infer_rows_and_bands = infer_rows_and_bands_v19
    base.image_trait_summary = image_trait_summary_v19
    base.image_gwas_trait_summary = image_gwas_trait_summary_v19
    base.add_row_neighbors = add_row_neighbors_v19
    base.make_spatial_audit_overlay = make_spatial_audit_overlay_v19
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base.__file__ = __file__


def run_trait_extraction() -> Path:
    """Run v1.9 through the established read-only trait-extraction engine."""

    install_v19_partial_band_policy()
    return base.run_trait_extraction()


if __name__ == "__main__":
    run_trait_extraction()
