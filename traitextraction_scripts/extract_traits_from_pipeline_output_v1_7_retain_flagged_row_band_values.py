#!/usr/bin/env python3
"""Version 1.7 Brachypodium trait extractor with non-destructive row/band QC.

This is a versioned export-policy successor to the biologically audited v1.6
row/band extractor.  It deliberately leaves the v1.6 row and band inference,
partial-field rules, density, calibration, morphometry and nearest-neighbour
calculations unchanged.

The only scientific-policy change is how the compact image-level GWAS table is
written:

* every row/band phenotype that can be calculated is retained;
* a failed strict trajectory QC check adds explicit flags and reasons instead
  of replacing all row/band phenotypes with blank values; and
* a measurement that is mathematically undefined remains blank.  For example,
  one stoma can support a row count, but it cannot by itself define within-row
  spacing, row continuity or row orientation.

The strict v1.6 eligibility result is preserved as a QC field.  This permits a
later sensitivity analysis comparing all provisional measurements with the
strict high-confidence subset without discarding biological observations.

Run this script with a new output directory.  It does not modify the pipeline
output directory, original OME-TIFF files, trained models or earlier trait
extraction results.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import extract_traits_from_pipeline_output as base
import extract_traits_from_pipeline_output_v1_4_global_rows as v14
import extract_traits_from_pipeline_output_v1_5_trajectory_rows as v15
import extract_traits_from_pipeline_output_v1_6_local_trajectory_guarded as v16


SCRIPT_VERSION = "1.7.0-retain-flagged-row-band-gwas-values"
ROW_GEOMETRY_MODEL_VERSION = v16.SCRIPT_VERSION
_PARENT_V16_SCRIPT_PATH = Path(v16.__file__).resolve()


@dataclass(frozen=True)
class TraitSettingsV17(v16.TraitSettingsV16):
    """Record the immutable v1.7 output policy in the provenance JSON."""

    row_band_computable_gwas_values_retained_when_qc_flagged: bool = True
    row_band_qc_flags_required: bool = True
    row_band_undefined_measurements_remain_blank: bool = True
    trajectory_parent_v16_script_sha256: str = ""


def build_settings_v17(args: Any) -> TraitSettingsV17:
    """Retain all validated v1.6 settings and append the v1.7 policy."""

    inherited = v16.build_settings_v16(args)
    return TraitSettingsV17(
        **asdict(inherited),
        trajectory_parent_v16_script_sha256=hashlib.sha256(
            _PARENT_V16_SCRIPT_PATH.read_bytes()
        ).hexdigest(),
    )


def _row_band_qc_fields(image_traits: dict[str, Any]) -> dict[str, Any]:
    """Return short, analysis-friendly aliases for the strict v1.6 QC."""

    strict_eligible = bool(
        image_traits.get("trajectory_exact_row_band_traits_gwas_eligible")
    )
    strict_reason = str(
        image_traits.get("trajectory_row_band_gwas_exclusion_reason") or ""
    ).strip()
    if strict_eligible:
        status = "pass"
        strict_reason = ""
    else:
        status = "flagged_provisional"
        if not strict_reason:
            strict_reason = "strict_row_band_qc_not_passed_unspecified_reason"

    return {
        "row_band_computable_values_retained_in_gwas": True,
        "row_band_qc_status": status,
        "row_band_qc_flag": not strict_eligible,
        "row_band_qc_reason": strict_reason,
        "row_band_strict_gwas_eligible": strict_eligible,
        "row_band_value_interpretation": (
            "computed_values_retained; use row_band_qc_status and detailed "
            "row/band audit tables for sensitivity analysis"
        ),
    }


def image_trait_summary_v17(
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
    """Keep the established v1.6 measurements and add v1.7 QC aliases."""

    # Correct a legacy v1.4 label before the audit dictionary is serialized.
    # v1.6 uses the image-wide direction as a prior and permits a bounded,
    # shared residual tilt within each provisional band; it does not enforce a
    # single zero-residual direction on every row.
    row_band_audit["image_wide_row_orientation_enforced"] = False
    row_band_audit["image_wide_row_orientation_prior_used"] = True
    result = v16.image_trait_summary_v16(
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

    # The inherited image summary assigns the fitted shared slope to every row,
    # including a one-stoma row.  A singleton can contribute to row/band counts
    # but cannot independently estimate orientation.  Recalculate the complete
    # orientation summary using only spatially eligible rows supported by at
    # least two longitudinally separated stomata.
    orientation_supported_rows = [
        row
        for row in row_records
        if bool(row.get("row_band_spatial_eligible"))
        and (base.optional_int(row.get("stoma_count")) or 0) >= 2
        and (base.optional_float(row.get("observed_longitudinal_span_um")) or 0.0)
        > 0.0
    ]
    result.update(
        base.sample_summary(
            (
                row.get("row_orientation_relative_leaf_axis_deg")
                for row in orientation_supported_rows
            ),
            "row_orientation_relative_leaf_axis_deg",
        )
    )
    result.update(_row_band_qc_fields(result))
    result.update(
        {
            "row_geometry_model_version": ROW_GEOMETRY_MODEL_VERSION,
            "trait_export_policy_version": SCRIPT_VERSION,
            "row_orientation_supported_row_count": len(
                orientation_supported_rows
            ),
            "row_orientation_support_policy": (
                "spatially eligible rows with at least two stomata and positive "
                "observed longitudinal span"
            ),
            "image_wide_row_orientation_enforced": False,
            "image_wide_row_orientation_prior_used": True,
            "trajectory_row_band_trait_policy": (
                "v1.6 row/band estimates and detailed QC are retained; the "
                "compact GWAS CSV keeps every computable row/band phenotype "
                "and marks non-strict results as flagged_provisional instead "
                "of blanking them; measurements that are mathematically "
                "undefined remain blank"
            ),
        }
    )
    return result


def image_gwas_trait_summary_v17(
    image_traits: dict[str, Any],
) -> dict[str, Any]:
    """Restore QC-flagged row/band values after the inherited strict export."""

    # The inherited function carries every established identifier, phenotype
    # and QC field, but v1.5/v1.6 deliberately blank the fields below when the
    # strict trajectory criterion fails.  Restore the already-calculated image
    # values without estimating, imputing or modifying them.
    summary = v16.image_gwas_trait_summary_v16(image_traits)
    for field in v15._ROW_BAND_GWAS_TRAITS:
        summary[field] = image_traits.get(field)

    summary.update(_row_band_qc_fields(image_traits))
    summary.update(
        {
            "row_geometry_model_version": ROW_GEOMETRY_MODEL_VERSION,
            "trait_export_policy_version": SCRIPT_VERSION,
            "row_orientation_supported_row_count": image_traits.get(
                "row_orientation_supported_row_count"
            ),
            "row_orientation_support_policy": image_traits.get(
                "row_orientation_support_policy"
            ),
            "image_wide_row_orientation_enforced": False,
            "image_wide_row_orientation_prior_used": True,
            "trajectory_row_band_trait_policy": image_traits.get(
                "trajectory_row_band_trait_policy"
            ),
        }
    )
    return summary


def process_image_v17(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Reset legacy orientation wording after inherited v1.4 post-processing."""

    result = list(v15.process_image_v15(*args, **kwargs))
    if len(result) != 6 or not isinstance(result[1], dict):
        raise RuntimeError(
            "v1.7 expected the established six-part per-image result."
        )
    image_traits = result[1]
    image_traits["image_wide_row_orientation_enforced"] = False
    image_traits["image_wide_row_orientation_prior_used"] = True
    if isinstance(result[5], dict):
        result[5]["image_wide_row_orientation_enforced"] = False
        result[5]["image_wide_row_orientation_prior_used"] = True
    return tuple(result)


def install_v17_export_policy() -> None:
    """Install v1.6 inference first, then replace only the export policy."""

    v16.install_v16_trajectory_selector()

    # Keep v1.6 as the row-geometry model label.  Only the top-level extractor
    # and export-policy provenance are versioned as v1.7.
    base.build_settings = build_settings_v17
    base.image_trait_summary = image_trait_summary_v17
    base.image_gwas_trait_summary = image_gwas_trait_summary_v17
    base.process_image = process_image_v17
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base.__file__ = __file__


def run_trait_extraction() -> Path:
    """Run v1.7 through the established read-only trait-extraction engine."""

    install_v17_export_policy()
    return base.run_trait_extraction()


if __name__ == "__main__":
    run_trait_extraction()
