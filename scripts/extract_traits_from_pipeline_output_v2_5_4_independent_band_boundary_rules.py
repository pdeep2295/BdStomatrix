#!/usr/bin/env python3
"""Experimental v2.5.4 generalized band-boundary successor.

This is a separate successor to v2.5.3.  It preserves detector, classifier,
segmentor, row-assignment, singleton-visibility, density and morphometry
policies.  It changes only final band partitioning through three generalized
routes derived from the independent 40-image audit:

1. accept a stable, image-supported, multipoint boundary with a dominant
   local gap when the remaining failures are limited to the conservative gap,
   row-topology-confidence or nonlinear-diagnostic vetoes;
2. accept a supported near-threshold multipoint boundary when gap width is the
   sole failed gate and all frozen corroborating gates are strong; and
3. for an over-capacity provisional band, prefer a balanced, image-supported
   internal gap over an arbitrary raw-contrast-maximizing forced split.

No image name, stomatal identifier, row identifier, fixed membership or manual
row count is used.  Thresholds remain experimental until validated on a third,
untouched balanced image set.  Run in a fresh Python process and output folder.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import extract_traits_from_pipeline_output_v2_5_3_confirmation_generalized_rules as v253


v252 = v253.v252
v25 = v253.v25
base = v253.base
v14 = v253.v14
v15 = v253.v15

SCRIPT_VERSION = "2.5.4-experimental-independent-generalized-band-boundaries"
SINGLETON_POLICY_VERSION = v253.SINGLETON_POLICY_VERSION
ROW_GEOMETRY_MODEL_VERSION = v253.ROW_GEOMETRY_MODEL_VERSION
PARTIAL_BAND_ELIGIBILITY_VERSION = SCRIPT_VERSION
TRAIT_EXPORT_POLICY_VERSION = v253.TRAIT_EXPORT_POLICY_VERSION

_PARENT_V253_SCRIPT_PATH = Path(v253.__file__).resolve()
_PARENT_INFER_V253 = v253.infer_rows_and_bands_v253
_PARENT_IMAGE_SUMMARY_V253 = v253.image_trait_summary_v253
_PARENT_GWAS_SUMMARY_V253 = v253.image_gwas_trait_summary_v253
_PARENT_OVERLAY_V253 = v253._PARENT_OVERLAY_V252

# Stable dominant-gap route.
V254_DOMINANT_MIN_GAP_WIDTH_FACTOR = 2.25
V254_DOMINANT_MIN_LOCAL_GAP_RATIO = 2.50
V254_DOMINANT_MIN_STABILITY_FRACTION = 0.99
V254_DOMINANT_MIN_IMAGE_SUPPORT_RATIO = 1.40
V254_DOMINANT_MIN_STOMATA_PER_SIDE = 5

# Supported near-threshold route.
V254_NEAR_MIN_GAP_WIDTH_FACTOR = 1.50
V254_NEAR_MIN_LOCAL_GAP_RATIO = 2.50
V254_NEAR_MIN_STABILITY_FRACTION = 0.88
V254_NEAR_MIN_IMAGE_SUPPORT_RATIO = 1.80
V254_NEAR_MIN_STOMATA_PER_SIDE = 7

# Balanced over-capacity route.
V254_CAPACITY_MIN_GAP_WIDTH_FACTOR = 1.25
V254_CAPACITY_MIN_LOCAL_GAP_RATIO = 2.00
V254_CAPACITY_MIN_IMAGE_SUPPORT_RATIO = 1.70
V254_CAPACITY_MIN_ROWS_PER_SIDE = 2


@dataclass(frozen=True)
class TraitSettingsV254(v253.TraitSettingsV253):
    """Immutable v2.5.4 settings recorded in provenance."""

    v254_policy_version: str = SCRIPT_VERSION
    v254_parent_v253_script_sha256: str = ""
    v254_dominant_min_gap_width_factor: float = V254_DOMINANT_MIN_GAP_WIDTH_FACTOR
    v254_dominant_min_local_gap_ratio: float = V254_DOMINANT_MIN_LOCAL_GAP_RATIO
    v254_dominant_min_stability_fraction: float = V254_DOMINANT_MIN_STABILITY_FRACTION
    v254_dominant_min_image_support_ratio: float = V254_DOMINANT_MIN_IMAGE_SUPPORT_RATIO
    v254_dominant_min_stomata_per_side: int = V254_DOMINANT_MIN_STOMATA_PER_SIDE
    v254_near_min_gap_width_factor: float = V254_NEAR_MIN_GAP_WIDTH_FACTOR
    v254_near_min_local_gap_ratio: float = V254_NEAR_MIN_LOCAL_GAP_RATIO
    v254_near_min_stability_fraction: float = V254_NEAR_MIN_STABILITY_FRACTION
    v254_near_min_image_support_ratio: float = V254_NEAR_MIN_IMAGE_SUPPORT_RATIO
    v254_near_min_stomata_per_side: int = V254_NEAR_MIN_STOMATA_PER_SIDE
    v254_capacity_min_gap_width_factor: float = V254_CAPACITY_MIN_GAP_WIDTH_FACTOR
    v254_capacity_min_local_gap_ratio: float = V254_CAPACITY_MIN_LOCAL_GAP_RATIO
    v254_capacity_min_image_support_ratio: float = V254_CAPACITY_MIN_IMAGE_SUPPORT_RATIO
    v254_capacity_min_rows_per_side: int = V254_CAPACITY_MIN_ROWS_PER_SIDE


def build_settings_v254(args: Any) -> TraitSettingsV254:
    inherited = asdict(v253.build_settings_v253(args))
    inherited.update(
        {
            "v254_policy_version": SCRIPT_VERSION,
            "v254_parent_v253_script_sha256": hashlib.sha256(
                _PARENT_V253_SCRIPT_PATH.read_bytes()
            ).hexdigest(),
        }
    )
    return TraitSettingsV254(**inherited)


def _finite_number(value: Any) -> float | None:
    parsed = base.optional_float(value)
    return float(parsed) if parsed is not None else None


def _exact_image_support_at_least(
    result: dict[str, Any], threshold: float
) -> bool:
    support = _finite_number(result.get("image_support"))
    return bool(
        result.get("image_support_identity_exact")
        and support is not None
        and support >= float(threshold)
    )


def _v25_partition_candidate_v254(
    band: dict[str, Any],
    linked: list[dict[str, Any]],
    cut: int,
    gaps: list[float],
    boundary_evidence_by_rows: dict[tuple[str, str, str], dict[str, Any]],
    median_width_um: float,
    settings: TraitSettingsV254,
) -> dict[str, Any]:
    """Apply v2.5.3 first, then the three guarded v2.5.4 routes."""

    result = v253._v25_partition_candidate_v253(
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
    gap_factor = _finite_number(result.get("gap_factor")) or 0.0
    gap_ratio = _finite_number(result.get("gap_ratio")) or 0.0
    stability = _finite_number(result.get("stability")) or 0.0
    left_stomata = int(result.get("left_stomata") or 0)
    right_stomata = int(result.get("right_stomata") or 0)
    multipoint_sides = min(left_stomata, right_stomata)
    nonisolated = not bool(result.get("isolated_one_row_side"))
    supported_topology = bool(result.get("row_topology_supported"))
    nonlinear = bool(result.get("nonlinear_row_topology"))

    allowed_dominant_failures = {
        "gap_width",
        "row_topology_not_supported",
        "nonlinear_row_topology",
    }
    dominant_gap = bool(
        failures
        and failures <= allowed_dominant_failures
        and nonisolated
        and gap_factor
        >= float(settings.v254_dominant_min_gap_width_factor)
        and gap_ratio >= float(settings.v254_dominant_min_local_gap_ratio)
        and stability
        >= float(settings.v254_dominant_min_stability_fraction)
        and multipoint_sides
        >= int(settings.v254_dominant_min_stomata_per_side)
        and _exact_image_support_at_least(
            result, settings.v254_dominant_min_image_support_ratio
        )
    )

    supported_near_threshold = bool(
        failures == {"gap_width"}
        and nonisolated
        and supported_topology
        and not nonlinear
        and gap_factor >= float(settings.v254_near_min_gap_width_factor)
        and gap_ratio >= float(settings.v254_near_min_local_gap_ratio)
        and stability >= float(settings.v254_near_min_stability_fraction)
        and multipoint_sides >= int(settings.v254_near_min_stomata_per_side)
        and _exact_image_support_at_least(
            result, settings.v254_near_min_image_support_ratio
        )
    )

    left_rows = len(result.get("left", []))
    right_rows = len(result.get("right", []))
    balanced_over_capacity = bool(
        len(linked) > int(settings.maximum_rows_per_band)
        and failures == {"gap_width", "boundary_stability"}
        and nonisolated
        and supported_topology
        and not nonlinear
        and left_rows == right_rows
        and min(left_rows, right_rows)
        >= int(settings.v254_capacity_min_rows_per_side)
        and gap_factor
        >= float(settings.v254_capacity_min_gap_width_factor)
        and gap_ratio >= float(settings.v254_capacity_min_local_gap_ratio)
        and _exact_image_support_at_least(
            result, settings.v254_capacity_min_image_support_ratio
        )
    )

    if dominant_gap:
        route = "stable_exact_image_supported_dominant_multipoint_gap"
    elif supported_near_threshold:
        route = "supported_near_threshold_exact_image_multipoint_gap"
    elif balanced_over_capacity:
        route = "balanced_over_capacity_evidence_boundary"
    else:
        return result

    result.update(
        {
            "accepted": True,
            "failed_gates_before_v254_general_rule": sorted(failures),
            "failed_gates": [],
            "trajectory_v254_general_band_boundary_applied": True,
            "trajectory_v254_general_band_boundary_route": route,
            "decision_route": f"v254_{route}",
        }
    )
    return result


def _count_v254_general_boundaries(audit: dict[str, Any]) -> tuple[int, dict[str, int]]:
    accepted: set[tuple[str, str, str, str]] = set()
    for partition in audit.get("band_partition_diagnostics", []):
        if not isinstance(partition, dict):
            continue
        records = [partition]
        nested = partition.get("trajectory_v25_partition_candidate_diagnostics", [])
        if isinstance(nested, list):
            records.extend(value for value in nested if isinstance(value, dict))
        for record in records:
            if not bool(record.get("trajectory_v254_general_band_boundary_applied")):
                continue
            if base.parse_bool(record.get("accepted")) is not True:
                continue
            accepted.add(
                (
                    str(record.get("provisional_band_id") or ""),
                    str(record.get("boundary_left_row_id") or ""),
                    str(record.get("boundary_right_row_id") or ""),
                    str(record.get("trajectory_v254_general_band_boundary_route") or ""),
                )
            )
    by_route: dict[str, int] = {}
    for _, _, _, route in accepted:
        by_route[route] = by_route.get(route, 0) + 1
    return len(accepted), by_route


def infer_rows_and_bands_v254(
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettingsV254,
    audited_row_counts: dict[str, int] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    rows, bands, assignments, audit = _PARENT_INFER_V253(
        stoma_traits,
        settings,
        audited_row_counts=audited_row_counts,
    )
    boundary_count, route_counts = _count_v254_general_boundaries(audit)
    audit.update(
        {
            "row_model_version": SCRIPT_VERSION,
            "trajectory_v254_policy_version": SCRIPT_VERSION,
            "trajectory_v254_general_band_boundary_count": boundary_count,
            "trajectory_v254_general_band_boundary_route_counts": route_counts,
            "detected_row_count_v254": len(rows),
            "detected_band_count_v254": len(bands),
            "band_partition_policy": (
                "v2.5.3 boundaries plus guarded exact-image-supported dominant-"
                "gap, supported-near-threshold and balanced-over-capacity routes; "
                "no image-specific identifiers, memberships or row counts"
            ),
        }
    )
    for collection in (rows, bands, list(assignments.values()), stoma_traits):
        for record in collection:
            record["row_model_version"] = SCRIPT_VERSION
            record["trajectory_v254_policy_version"] = SCRIPT_VERSION
            record["partial_band_eligibility_version"] = SCRIPT_VERSION
    return rows, bands, assignments, audit


def image_trait_summary_v254(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = _PARENT_IMAGE_SUMMARY_V253(*args, **kwargs)
    audit = args[7] if len(args) > 7 else kwargs.get("row_band_audit", {})
    result.update(
        {
            "row_model_version": SCRIPT_VERSION,
            "row_geometry_model_version": SCRIPT_VERSION,
            "partial_band_eligibility_version": SCRIPT_VERSION,
            "trajectory_v254_policy_version": SCRIPT_VERSION,
            "general_band_boundary_count_v254": int(
                audit.get("trajectory_v254_general_band_boundary_count", 0) or 0
            ),
            "general_band_boundary_route_counts_v254": audit.get(
                "trajectory_v254_general_band_boundary_route_counts", {}
            ),
            "detected_row_count_v254": len(
                args[3] if len(args) > 3 else kwargs["row_records"]
            ),
            "detected_band_count_v254": len(
                args[4] if len(args) > 4 else kwargs["band_records"]
            ),
        }
    )
    return result


def image_gwas_trait_summary_v254(image_traits: dict[str, Any]) -> dict[str, Any]:
    result = _PARENT_GWAS_SUMMARY_V253(image_traits)
    for field in (
        "trajectory_v254_policy_version",
        "general_band_boundary_count_v254",
        "general_band_boundary_route_counts_v254",
        "detected_row_count_v254",
        "detected_band_count_v254",
    ):
        result[field] = image_traits.get(field)
    result["row_model_version"] = SCRIPT_VERSION
    result["row_geometry_model_version"] = SCRIPT_VERSION
    result["partial_band_eligibility_version"] = SCRIPT_VERSION
    return result


def _update_v254_output_provenance(output_dir: Path) -> None:
    provenance_path = output_dir / "trait_extraction_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance.update(
            {
                "script_version": SCRIPT_VERSION,
                "v254_parent_script": str(_PARENT_V253_SCRIPT_PATH),
                "v254_parent_script_sha256": hashlib.sha256(
                    _PARENT_V253_SCRIPT_PATH.read_bytes()
                ).hexdigest(),
                "v254_contains_image_specific_rules": False,
                "v254_contains_manual_row_count_overrides": False,
                "v254_policy": (
                    "guarded exact-image-supported dominant multipoint gaps, "
                    "supported near-threshold multipoint gaps, and balanced "
                    "evidence-based over-capacity partitioning"
                ),
                "v254_development_status": (
                    "rules derived from two audited 40-image sets; validation "
                    "on a third untouched balanced set is required before lock"
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
        addition = (
            "\nV2.5.4 is a separate band-boundary successor to v2.5.3.\n"
            "It uses no image name, stomatal identifier, row identifier, fixed "
            "membership or manual row count. Density, morphometry, row assignment "
            "and the v2.5.3 fully-visible-singleton rule are unchanged.\n"
            "V2.5.4 requires validation on a third untouched balanced set.\n"
        )
        if "V2.5.4 is a separate band-boundary successor" not in report:
            report += addition
        report_path.write_text(report, encoding="utf-8")


def install_v254_policy() -> None:
    """Install v2.5.3, then replace only generalized band-boundary hooks."""

    v253.install_v253_policy()
    v25._v25_partition_candidate = _v25_partition_candidate_v254
    base.build_settings = build_settings_v254
    base.infer_rows_and_bands = infer_rows_and_bands_v254
    base.image_trait_summary = image_trait_summary_v254
    base.image_gwas_trait_summary = image_gwas_trait_summary_v254
    base.make_spatial_audit_overlay = _PARENT_OVERLAY_V253
    for module in (v14, v15, base):
        module.SCRIPT_VERSION = SCRIPT_VERSION
    base.__file__ = __file__


def run_trait_extraction() -> Path:
    install_v254_policy()
    output_dir = base.run_trait_extraction()
    v25._write_v25_additional_qc_tables(output_dir)
    v25._update_v25_output_provenance(output_dir)
    _update_v254_output_provenance(output_dir)
    return output_dir


if __name__ == "__main__":
    run_trait_extraction()
