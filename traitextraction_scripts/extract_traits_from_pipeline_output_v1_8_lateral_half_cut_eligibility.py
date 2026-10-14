#!/usr/bin/env python3
"""Version 1.8 Brachypodium trait extractor with direct half-cut-band QC.

This is an eligibility-only successor to v1.7.  It deliberately preserves:

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
``edge_adjacent_uncertain``, ``laterally_contained`` and ``unknown``.  A
half-cut decision requires direct source-pixel member-centre evidence and the
fitted row trajectory to support the same image side.  Detector bounding boxes
are retained as diagnostics but do not decide eligibility, because their
annotation/prediction padding can touch an image edge even when the biological
band cross-section remains visible.

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


SCRIPT_VERSION = "1.8.0-direct-lateral-half-cut-eligibility"
PARTIAL_BAND_ELIGIBILITY_VERSION = SCRIPT_VERSION
ROW_GEOMETRY_MODEL_VERSION = v16.SCRIPT_VERSION
TRAIT_EXPORT_POLICY_VERSION = v17.SCRIPT_VERSION

# A directly observed member centre closer than 0.35 of a typical detector-box
# width to a source side edge is a candidate half-cut band.  The fitted row
# geometry must approach the *same* edge within 0.70 box widths.  The fitted
# tolerance is wider because a fit through a sparse/tilted row need not pass
# through the single most edgeward observed centre.  Detector-box contact is
# recorded as context but is not by itself an exclusion rule.
LATERAL_CENTER_EDGE_CLEARANCE_BOX_WIDTH_FACTOR = 0.35
FITTED_CENTER_EDGE_CORROBORATION_BOX_WIDTH_FACTOR = 0.70
LATERAL_EDGE_ADJACENT_BUFFER_BOX_WIDTH_FACTOR = 0.50
DETECTOR_BOX_EDGE_TOUCH_TOLERANCE_PIXELS = 1.0
FITTED_TRAJECTORY_EDGE_SAMPLES = 101

_PARENT_V17_SCRIPT_PATH = Path(v17.__file__).resolve()

# Save the concrete parent implementation before installation changes the
# dynamic v1.5 hook.  Calling the patched v1.5 name from within v1.8 would
# recurse.
_V16_PARTIAL_BAND_ELIGIBILITY = v16._remark_partial_band_eligibility_v16


@dataclass(frozen=True)
class TraitSettingsV18(v17.TraitSettingsV17):
    """Record the fixed v1.8 eligibility policy in provenance."""

    partial_band_eligibility_version: str = PARTIAL_BAND_ELIGIBILITY_VERSION
    partial_band_row_structure_uses_lateral_half_cut_only: bool = True
    partial_band_severe_short_observed_support_retained_for_rows: bool = True
    partial_band_detector_bbox_edge_not_decisive: bool = True
    partial_band_member_and_fitted_edge_evidence_required: bool = True
    partial_band_lateral_thresholds_provisional_audit_derived: bool = True
    partial_band_lateral_center_clearance_box_width_factor: float = (
        LATERAL_CENTER_EDGE_CLEARANCE_BOX_WIDTH_FACTOR
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
    partial_band_parent_v17_script_sha256: str = ""


def build_settings_v18(args: Any) -> TraitSettingsV18:
    """Keep every v1.7 setting and append immutable v1.8 policy fields."""

    inherited = v17.build_settings_v17(args)
    return TraitSettingsV18(
        **asdict(inherited),
        partial_band_parent_v17_script_sha256=hashlib.sha256(
            _PARENT_V17_SCRIPT_PATH.read_bytes()
        ).hexdigest(),
    )


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


def _half_cut_decision(
    geometry: dict[str, Any],
    member_clearance_limit: float,
    fitted_clearance_limit: float,
    adjacent_buffer_limit: float,
    detector_touch_tolerance_pixels: float,
) -> dict[str, Any]:
    """Classify lateral visibility using same-side corroboration.

    Missing geometry is never converted into a half-cut call.  The band is
    retained for row structure, marked ``unknown``, and omitted only from the
    stricter completeness-dependent summary.
    """

    member_left = _finite_number(
        geometry.get("band_member_center_left_edge_distance_box_width_factor")
    )
    member_right = _finite_number(
        geometry.get("band_member_center_right_edge_distance_box_width_factor")
    )
    fitted_left = _finite_number(
        geometry.get(
            "band_fitted_row_center_left_edge_distance_box_width_factor"
        )
    )
    fitted_right = _finite_number(
        geometry.get(
            "band_fitted_row_center_right_edge_distance_box_width_factor"
        )
    )
    assessable = all(
        value is not None
        for value in (member_left, member_right, fitted_left, fitted_right)
    )
    if not assessable:
        return {
            "half_cut": False,
            "assessment_state": "unknown",
            "assessment_assessable": False,
            "half_cut_side": "unknown",
            "same_side_half_cut_corroborated": False,
            "decision_source": (
                "insufficient_same_side_member_and_fitted_geometry_retained_with_qc"
            ),
        }

    left_cut = bool(
        member_left <= member_clearance_limit
        and fitted_left <= fitted_clearance_limit
    )
    right_cut = bool(
        member_right <= member_clearance_limit
        and fitted_right <= fitted_clearance_limit
    )
    half_cut = left_cut or right_cut
    if left_cut and right_cut:
        half_cut_side = "both"
    elif left_cut:
        half_cut_side = "left"
    elif right_cut:
        half_cut_side = "right"
    else:
        half_cut_side = "none"

    detector_box_margin = _finite_number(
        geometry.get("band_detector_box_side_edge_clearance_pixels")
    )
    member_near = min(member_left, member_right) <= adjacent_buffer_limit
    fitted_near = min(fitted_left, fitted_right) <= adjacent_buffer_limit
    detector_box_touches = bool(
        detector_box_margin is not None
        and detector_box_margin <= detector_touch_tolerance_pixels
    )
    if half_cut:
        state = "half_cut"
        source = "same_side_direct_member_and_fitted_row_centres"
    elif member_near or fitted_near or detector_box_touches:
        state = "edge_adjacent_uncertain"
        source = (
            "member_or_fitted_near_edge_or_detector_box_contact_without_same_side_"
            "half_cut_corroboration"
        )
    else:
        state = "laterally_contained"
        source = "same_side_member_and_fitted_geometry_clear_of_half_cut_limit"

    return {
        "half_cut": half_cut,
        "assessment_state": state,
        "assessment_assessable": True,
        "half_cut_side": half_cut_side,
        "same_side_half_cut_corroborated": half_cut,
        "decision_source": source,
    }


def _remark_partial_band_eligibility_v18(
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    trajectory_slope: float,
    settings: TraitSettingsV18,
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
    member_clearance_limit = float(
        settings.partial_band_lateral_center_clearance_box_width_factor
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
        lateral_decision = _half_cut_decision(
            geometry,
            member_clearance_limit,
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
                "band_lateral_half_cut_clearance_threshold_box_width_factor": (
                    member_clearance_limit
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
                "lateral_half_cut_clearance_threshold_box_width_factor": (
                    member_clearance_limit
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


def infer_rows_and_bands_v18(
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettingsV18,
    audited_row_counts: dict[str, int] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    """Run unchanged v1.6 geometry and append v1.8 eligibility provenance."""

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
                "exclude a band from row structure only when direct source-pixel "
                "member-centre and fitted-row-centre clearances corroborate that "
                "its lateral cross-section is half-cut by the left/right image "
                "edge; detector-box contact alone is diagnostic, not decisive"
            ),
            "partial_band_longitudinal_support_policy": (
                "short longitudinal support is retained for row structure and "
                "reported separately as completeness QC"
            ),
            "partial_band_settings": {
                "member_center_half_cut_threshold_box_width_factor": (
                    settings.partial_band_lateral_center_clearance_box_width_factor
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
                "detector_bbox_contact_decisive": False,
                "thresholds_provisional_audit_derived": True,
                "short_observed_longitudinal_support_retained_for_rows": True,
            },
            "partial_band_lateral_center_clearance_box_width_factor": (
                settings.partial_band_lateral_center_clearance_box_width_factor
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
                "v1.6 fitted-band longitudinal coverage retained as QC; v1.8 "
                "row eligibility uses direct source left/right edge clearance"
            ),
        }
    )
    return row_records, band_records, point_assignment, audit


def image_trait_summary_v18(
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
            "partial_band_fitted_center_corroboration_box_width_factor": (
                FITTED_CENTER_EDGE_CORROBORATION_BOX_WIDTH_FACTOR
            ),
            "row_band_spatial_eligibility_policy": (
                "row structure excludes only a same-side corroborated lateral "
                "half-cut; edge-adjacent uncertain and unknown cases are "
                "retained with QC; short observed longitudinal support is also "
                "retained and flagged; stricter completeness-dependent "
                "summaries require laterally-contained geometry and adequate "
                "longitudinal support"
            ),
        }
    )
    v18_qc_reasons: list[str] = []
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
            v18_qc_reasons.append(reason)
    result.update(
        {
            "row_band_v18_qc_status": (
                "flagged_provisional" if v18_qc_reasons else "pass"
            ),
            "row_band_v18_qc_flag": bool(v18_qc_reasons),
            "row_band_v18_qc_reason": ";".join(v18_qc_reasons),
            "row_band_v18_value_interpretation": (
                "computed row/band values are retained; use the v1.8 lateral "
                "and longitudinal band-level fields plus contributing sample "
                "counts for trait-specific sensitivity analysis"
            ),
        }
    )
    return result


_V18_GWAS_FIELDS = (
    "partial_band_eligibility_version",
    "row_structure_eligible_band_count",
    "row_structure_eligible_row_count",
    "lateral_half_cut_band_count",
    "lateral_edge_adjacent_uncertain_band_count",
    "lateral_edge_assessment_unknown_band_count",
    "laterally_contained_band_count",
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
    "partial_band_fitted_center_corroboration_box_width_factor",
    "row_band_spatial_eligibility_policy",
    "row_band_v18_qc_status",
    "row_band_v18_qc_flag",
    "row_band_v18_qc_reason",
    "row_band_v18_value_interpretation",
)


def image_gwas_trait_summary_v18(
    image_traits: dict[str, Any],
) -> dict[str, Any]:
    """Carry v1.8 eligibility traits/QC into the compact image table."""

    summary = v17.image_gwas_trait_summary_v17(image_traits)
    for field in _V18_GWAS_FIELDS:
        summary[field] = image_traits.get(field)
    summary["row_geometry_model_version"] = ROW_GEOMETRY_MODEL_VERSION
    summary["trait_export_policy_version"] = TRAIT_EXPORT_POLICY_VERSION
    return summary


def add_row_neighbors_v18(
    stoma_traits: list[dict[str, Any]],
    row_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
) -> None:
    """Retain normal neighbor traits and propagate every v1.8 QC field."""

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


def make_spatial_audit_overlay_v18(
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
        raise RuntimeError(f"Failed to update v1.8 audit overlay: {output_path}")


def install_v18_partial_band_policy() -> None:
    """Install v1.7 first and then replace only partial-band eligibility."""

    v17.install_v17_export_policy()

    # This is the dynamic hook called inside the saved v1.5 inference function.
    v15._remark_partial_band_eligibility_v15 = (
        _remark_partial_band_eligibility_v18
    )

    base.build_settings = build_settings_v18
    base.infer_rows_and_bands = infer_rows_and_bands_v18
    base.image_trait_summary = image_trait_summary_v18
    base.image_gwas_trait_summary = image_gwas_trait_summary_v18
    base.add_row_neighbors = add_row_neighbors_v18
    base.make_spatial_audit_overlay = make_spatial_audit_overlay_v18
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base.__file__ = __file__


def run_trait_extraction() -> Path:
    """Run v1.8 through the established read-only trait-extraction engine."""

    install_v18_partial_band_policy()
    return base.run_trait_extraction()


if __name__ == "__main__":
    run_trait_extraction()
