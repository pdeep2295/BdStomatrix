#!/usr/bin/env python3
"""Experimental v1.6 Brachypodium row/band trait extractor.

This is a separate successor to the already-run v1.5 extractor.  It preserves
the detector, density, calibration, segmentation morphometry, ordinary nearest
neighbours, file-safety checks and output structure inherited from v1.5.

The revision addresses two general failure modes found during biological audit:

1. A candidate row supported by only one detected stoma was flagged after model
   selection, so the displayed row count could remain too high.  In v1.6 every
   singleton-supported candidate is inadmissible as an *exact* row before the
   final count is selected.  The best non-singleton simpler model is displayed
   when one is available, and
   the rejected sparse hypothesis is retained as a lower/upper row-count range.
2. v1.5 estimated an image-wide row direction and then forced every provisional
   band to have exactly zero residual tilt.  v1.6 retains that image-wide axis as
   a prior but allows one bounded, shared residual tilt for all rows in each
   provisional band.  This can join tilted fragments without allowing every row
   to choose an unrelated direction.

For row-count selection, generic Sobel/edge contrast is not consulted.  It
cannot create or validate a singleton biological row, because a confirmed false
row in S005_AD_P2_EDF band 2 had a strong edge ratio.  The inherited v1.5 rule
still uses local raw-edge support when deciding an otherwise ambiguous boundary
between already-supported rows; that separate band-partition rule is unchanged.

This file contains no image-specific overrides.  It remains an experimental
inspection candidate until it is checked on a representative, biologically
audited AB+AD set.  Do not replace the locked production extractor solely from
the small row-abnormality development set.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import extract_traits_from_pipeline_output as base
import extract_traits_from_pipeline_output_v1_4_global_rows as v14
import extract_traits_from_pipeline_output_v1_5_trajectory_rows as v15


SCRIPT_VERSION = "1.6.0-experimental-band-local-tilt-guarded-singletons"
_PARENT_BASE_SCRIPT_PATH = Path(base.__file__).resolve()
_PARENT_V14_SCRIPT_PATH = Path(v14.__file__).resolve()
_PARENT_V15_SCRIPT_PATH = Path(v15.__file__).resolve()


@dataclass(frozen=True)
class TraitSettingsV16(v15.TraitSettingsV15):
    """Record the v1.6 scientific rules in provenance."""

    trajectory_band_local_residual_tilt_enabled: bool = True
    trajectory_exact_singleton_rows_permitted: bool = False
    trajectory_sobel_can_create_or_validate_rows: bool = False
    trajectory_rejected_sparse_candidate_marks_exact_traits_unresolved: bool = True
    trajectory_parent_base_script_sha256: str = ""
    trajectory_parent_v14_script_sha256: str = ""
    trajectory_parent_v15_script_sha256: str = ""


# Save the original callables before installation replaces module attributes.
_V15_CHOOSE_TRAJECTORY_MODEL_ONCE = v15._choose_trajectory_model_once
_V15_SELECT_ROW_MODEL = v15.select_row_model_v15
_V15_DIAGNOSTIC_FIELDS = v15._trajectory_diagnostic_fields
_V15_REMARK_PARTIAL_BAND_ELIGIBILITY = v15._remark_partial_band_eligibility_v15
_V15_INFER_ROWS_AND_BANDS = v15.infer_rows_and_bands_v15
_V15_IMAGE_TRAIT_SUMMARY = v15.image_trait_summary_v15
_V15_IMAGE_GWAS_TRAIT_SUMMARY = v15.image_gwas_trait_summary_v15


def parse_arguments_v16() -> argparse.Namespace:
    """Use the established v1.5 command-line interface unchanged."""

    if any(value in {"-h", "--help"} for value in sys.argv[1:]):
        print(
            "v1.6 uses the established v1.5 options; guarded singleton-row "
            "selection and band-local shared tilt are fixed, recorded policies.\n"
        )
    return v15.parse_arguments_v15()


def build_settings_v16(args: argparse.Namespace) -> TraitSettingsV16:
    """Validate inherited settings and add immutable v1.6 policy fields."""

    inherited = v15.build_settings_v15(args)
    return TraitSettingsV16(
        **asdict(inherited),
        trajectory_parent_base_script_sha256=hashlib.sha256(
            _PARENT_BASE_SCRIPT_PATH.read_bytes()
        ).hexdigest(),
        trajectory_parent_v14_script_sha256=hashlib.sha256(
            _PARENT_V14_SCRIPT_PATH.read_bytes()
        ).hexdigest(),
        trajectory_parent_v15_script_sha256=hashlib.sha256(
            _PARENT_V15_SCRIPT_PATH.read_bytes()
        ).hexdigest(),
    )


def _slope_angles_v16(settings: TraitSettingsV16) -> list[float]:
    """Return the residual tilt grid even after image-wide detrending.

    All rows fitted within one provisional band still share one angle.  The
    change therefore permits a small band-local tilt, not independent row
    directions.
    """

    limit = float(settings.trajectory_local_tilt_limit_deg)
    step = float(settings.trajectory_local_tilt_step_deg)
    angles = np.arange(-limit, limit + step * 0.5, step, dtype=float).tolist()
    angles.append(0.0)
    return sorted({round(float(angle), 10) for angle in angles})


def _has_singleton(candidate: dict[str, Any]) -> bool:
    """Return whether any fitted trajectory contains only one stoma."""

    return any(int(size) == 1 for size in candidate.get("cluster_sizes", []))


def _two_row_structural_evidence_v16(
    candidate: dict[str, Any],
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV16,
) -> dict[str, Any]:
    """Describe whether two non-singleton trajectories have row-like support.

    This is used before the generic preference for a simpler model.  It does
    not turn a singleton into a row.  The checks ask whether both trajectories
    contain longitudinally separated stomata and whether their pooled order has
    the staggered/alternating structure expected for adjacent Brachypodium rows.
    """

    evidence: dict[str, Any] = {
        "direct_two_row_alternation": None,
        "direct_two_row_phase_offset_normalized": None,
        "direct_two_row_longest_alternating_sequence": None,
        "direct_two_row_minimum_longitudinal_span_um": None,
        "direct_two_row_longitudinal_span_supported": False,
        "direct_two_row_longitudinal_overlap_fraction": None,
        "direct_two_row_structural_support": False,
    }
    if int(candidate.get("k", 0)) != 2 or _has_singleton(candidate):
        return evidence

    labels = np.asarray(candidate["labels"], dtype=int)
    point_indices = [int(index) for index in candidate["point_indices"]]
    local_longitudinal = np.asarray(
        [float(longitudinal[index]) for index in point_indices], dtype=float
    )
    if len(local_longitudinal) < 4:
        return evidence

    row_coordinates = [
        np.sort(local_longitudinal[labels == group]) for group in (0, 1)
    ]
    if any(len(values) < 2 for values in row_coordinates):
        return evidence

    spans = [float(np.ptp(values)) for values in row_coordinates]
    minimum_required_span = max(1.0, 0.50 * median_detector_width_um)
    span_supported = all(span >= minimum_required_span for span in spans)
    evidence["direct_two_row_minimum_longitudinal_span_um"] = min(spans)
    evidence["direct_two_row_longitudinal_span_supported"] = span_supported

    overlap = max(
        0.0,
        min(float(values[-1]) for values in row_coordinates)
        - max(float(values[0]) for values in row_coordinates),
    )
    union = max(float(values[-1]) for values in row_coordinates) - min(
        float(values[0]) for values in row_coordinates
    )
    evidence["direct_two_row_longitudinal_overlap_fraction"] = (
        overlap / union if union > 0 else 0.0
    )

    order = np.argsort(local_longitudinal, kind="stable")
    sequence = labels[order].tolist()
    alternation = float(
        sum(first != second for first, second in zip(sequence[:-1], sequence[1:]))
        / max(len(sequence) - 1, 1)
    )
    longest = current = 1
    for first, second in zip(sequence[:-1], sequence[1:]):
        current = current + 1 if first != second else 1
        longest = max(longest, current)

    pooled_within_row_gaps = np.concatenate(
        [np.diff(values) for values in row_coordinates if len(values) >= 2]
    )
    phase: float | None = None
    if len(pooled_within_row_gaps):
        typical_spacing = float(np.median(pooled_within_row_gaps))
        if typical_spacing > 0:
            first, second = row_coordinates
            offsets = [float(np.min(np.abs(second - value))) for value in first]
            offsets.extend(float(np.min(np.abs(first - value))) for value in second)
            phase = float(np.median(offsets) / typical_spacing)

    evidence.update(
        {
            "direct_two_row_alternation": alternation,
            "direct_two_row_phase_offset_normalized": phase,
            "direct_two_row_longest_alternating_sequence": int(longest),
        }
    )
    evidence["direct_two_row_structural_support"] = bool(
        span_supported
        and alternation >= settings.direct_two_row_min_alternation
        and phase is not None
        and phase <= settings.direct_two_row_max_phase_offset_normalized
        and longest >= settings.direct_two_row_min_longest_alternating_sequence
    )
    return evidence


def _choose_trajectory_model_once_v16(
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV16,
    separation_scale: float = 1.0,
    allow_sparse_image_rescue: bool = True,
) -> tuple[
    dict[str, Any],
    dict[int, tuple[dict[str, Any], dict[str, Any]]],
    dict[str, Any],
]:
    """Choose the final model only after rejecting singleton exact rows."""

    # Deliberately disable v1.5's Sobel-driven sparse-row rescue.  Raw edge
    # contrast remains available to the inherited ambiguous band-boundary rule,
    # but it cannot alter the v1.6 row count selected for display or traits.
    _ = allow_sparse_image_rescue
    automatic, best_by_k, metadata = _V15_CHOOSE_TRAJECTORY_MODEL_ONCE(
        point_indices,
        lateral,
        longitudinal,
        median_detector_width_um,
        settings,
        separation_scale=separation_scale,
        allow_sparse_image_rescue=False,
    )

    inherited_automatic_k = int(automatic["k"])
    strict_k = int(metadata.get("strict_best_row_count", inherited_automatic_k))
    strict_candidate = best_by_k.get(strict_k, (automatic, {}))[0]

    # Close the inherited k=2 -> k=1 control-flow gap.  When k=2 is the strict
    # non-singleton fit but v1.5 simplified it to one row, reconsider it using
    # the longitudinal support that remains available after band-local tilt.
    direct_two_reconsidered_v16 = False
    competing_two_row_hypothesis_unresolved = False
    reconsidered_two_evidence: dict[str, Any] = {}
    reconsidered_two_score_improvement: float | None = None
    reconsidered_two_strong_score_support = False
    reconsidered_two_structural_support = False
    if (
        inherited_automatic_k == 1
        and strict_k == 2
        and 1 in best_by_k
        and 2 in best_by_k
        and not _has_singleton(strict_candidate)
        and bool(metadata.get("direct_two_row_rejected_to_one_row"))
    ):
        direct_two_reconsidered_v16 = True
        reconsidered_two_evidence = _two_row_structural_evidence_v16(
            strict_candidate,
            longitudinal,
            median_detector_width_um,
            settings,
        )
        reconsidered_two_score_improvement = float(
            best_by_k[1][1]["adjusted_score"]
            - best_by_k[2][1]["adjusted_score"]
        )
        reconsidered_two_strong_score_support = bool(
            reconsidered_two_score_improvement
            >= settings.direct_two_row_min_score_improvement
        )
        reconsidered_two_structural_support = bool(
            reconsidered_two_evidence.get("direct_two_row_structural_support")
        )
        reconsidered_two_validated = bool(
            reconsidered_two_strong_score_support
            or reconsidered_two_structural_support
        )
        metadata.update(
            {
                "trajectory_direct_two_reconsidered_v16": True,
                "direct_two_row_candidate_evaluated": True,
                "direct_two_row_validated": reconsidered_two_validated,
                "direct_two_row_rejected_to_one_row": not reconsidered_two_validated,
                "direct_two_row_score_improvement_vs_one": (
                    reconsidered_two_score_improvement
                ),
                "direct_two_row_strong_score_support": (
                    reconsidered_two_strong_score_support
                ),
                "direct_two_row_structural_support": (
                    reconsidered_two_structural_support
                ),
                **reconsidered_two_evidence,
            }
        )
        if reconsidered_two_validated:
            automatic = strict_candidate
            inherited_automatic_k = 2
            metadata["automatic_row_count"] = 2
            metadata["automatic_selection_reason"] = (
                str(metadata.get("automatic_selection_reason", ""))
                + "__v16_two_row_reinstated_after_longitudinal_support"
            )
        else:
            competing_two_row_hypothesis_unresolved = True

    # Prefer the actual inherited selection as the candidate to validate.  If
    # it was already simplified, still reject a better-scoring strict candidate
    # when that candidate owes its extra row to a singleton.
    rejected_k: int | None = None
    rejected_candidate: dict[str, Any] | None = None
    rejected_diagnostic: dict[str, Any] | None = None
    if _has_singleton(automatic):
        rejected_k = inherited_automatic_k
        rejected_candidate, rejected_diagnostic = best_by_k[rejected_k]
    elif strict_k in best_by_k and _has_singleton(strict_candidate):
        rejected_k = strict_k
        rejected_candidate, rejected_diagnostic = best_by_k[rejected_k]

    fallback_applied = rejected_k is not None
    selected = automatic
    selected_k = inherited_automatic_k
    two_evidence: dict[str, Any] = dict(reconsidered_two_evidence)
    two_score_improvement: float | None = reconsidered_two_score_improvement
    two_strong_score_support = reconsidered_two_strong_score_support
    two_structural_support = reconsidered_two_structural_support
    fallback_internal_secondary_support_passed = False

    if fallback_applied and rejected_k is not None:
        admissible = [
            (int(k), candidate, diagnostic)
            for k, (candidate, diagnostic) in best_by_k.items()
            if int(k) < rejected_k and not _has_singleton(candidate)
        ]
        if not admissible:
            # This occurs only when a group itself contains one stoma.  Keep the
            # unavoidable one-row display but make exact traits unresolved.
            selected_k = 1
            selected = best_by_k[1][0]
        else:
            # Once an invalid complex model is removed, compare admissible models
            # by their fitted score before applying a generic simplicity tie.
            # This prevents the confirmed k=3 -> k=1 failure when k=2 is the
            # best non-singleton admissible model.
            selected_k, selected, _ = min(
                admissible,
                key=lambda item: (
                    float(item[2]["adjusted_score"]),
                    item[0],
                    abs(float(item[2]["trajectory_slope_angle_deg"])),
                ),
            )

            if 1 in best_by_k and 2 in best_by_k:
                two_candidate, two_diagnostic = best_by_k[2]
                if not _has_singleton(two_candidate):
                    two_evidence = _two_row_structural_evidence_v16(
                        two_candidate,
                        longitudinal,
                        median_detector_width_um,
                        settings,
                    )
                    two_score_improvement = float(
                        best_by_k[1][1]["adjusted_score"]
                        - two_diagnostic["adjusted_score"]
                    )
                    two_strong_score_support = bool(
                        two_score_improvement
                        >= settings.direct_two_row_min_score_improvement
                    )
                    two_structural_support = bool(
                        two_evidence.get("direct_two_row_structural_support")
                    )
                    two_is_score_compatible = bool(
                        float(two_diagnostic["adjusted_score"])
                        <= float(best_by_k[1][1]["adjusted_score"])
                        + settings.trajectory_simpler_tie_tolerance
                    )
                    if selected_k == 1 and two_structural_support and two_is_score_compatible:
                        selected_k = 2
                        selected = two_candidate

        fallback_internal_secondary_support_passed = bool(
            (
                selected_k == 1
                and not _has_singleton(selected)
            )
            or (
                selected_k == 2
                and (two_strong_score_support or two_structural_support)
            )
        )
        plausible_minimum_k = (
            selected_k if fallback_internal_secondary_support_passed else 1
        )

        reason = str(metadata.get("automatic_selection_reason", ""))
        reason += (
            f"__reject_unvalidated_singleton_k{rejected_k}"
            f"__fallback_to_non_singleton_k{selected_k}"
        )
        rejected_sizes = [
            int(value) for value in (rejected_candidate or {}).get("cluster_sizes", [])
        ]
        rejected_confidences = list(
            (rejected_diagnostic or {}).get("singleton_detector_confidences", [])
        )
        metadata.update(
            {
                "automatic_row_count": selected_k,
                "automatic_selection_reason": reason,
                "trajectory_pre_validation_row_count": rejected_k,
                "trajectory_post_validation_row_count": selected_k,
                "trajectory_validation_fallback_applied": True,
                "trajectory_validation_fallback_reason": (
                    "singleton_supported_candidate_rejected_before_exact_row_selection"
                ),
                "trajectory_rejected_unvalidated_singleton_candidate_row_count": (
                    rejected_k
                ),
                "trajectory_rejected_unvalidated_singleton_cluster_sizes": (
                    rejected_sizes
                ),
                "trajectory_rejected_unvalidated_singleton_detector_confidences": (
                    rejected_confidences
                ),
                "trajectory_displayed_non_singleton_row_count": (
                    None if _has_singleton(selected) else selected_k
                ),
                "trajectory_displayed_fallback_row_count": selected_k,
                "trajectory_fallback_internal_secondary_support_passed": (
                    fallback_internal_secondary_support_passed
                ),
                "trajectory_row_count_min": plausible_minimum_k,
                "trajectory_row_count_max": rejected_k,
                "trajectory_row_count_range_reason": (
                    "rejected_singleton_upper_hypothesis"
                    if fallback_internal_secondary_support_passed
                    else "fallback_model_internal_secondary_support_failed"
                ),
                "trajectory_possible_sparse_row": True,
                "trajectory_possible_sparse_additional_row_count": max(
                    0, rejected_k - selected_k
                ),
                "trajectory_direct_two_reconsidered_v16": (
                    direct_two_reconsidered_v16
                ),
                "trajectory_competing_two_row_hypothesis_unresolved": False,
                "trajectory_row_count_range_scope": (
                    "pre_partition_provisional_band"
                ),
                "trajectory_row_count_range_interpretation": (
                    "algorithmic_plausible_candidate_range_not_confidence_interval_"
                    "or_manual_truth"
                ),
                "trajectory_sobel_role": (
                    "not_consulted_for_row_selection;inherited_support_for_"
                    "ambiguous_band_partition"
                ),
                "trajectory_band_local_residual_tilt_enabled": True,
                # The old rescue may have proposed the rejected singleton.  It
                # is not considered an applied rescue in the final v1.6 model.
                "sparse_trajectory_rescue_applied_pre_validation": bool(
                    metadata.get("sparse_trajectory_rescue_applied")
                ),
                "sparse_trajectory_rescue_applied": False,
                "direct_two_row_candidate_evaluated": bool(two_evidence),
                "direct_two_row_validated": bool(
                    two_strong_score_support or two_structural_support
                ),
                "direct_two_row_rejected_to_one_row": bool(
                    selected_k == 1
                    and bool(two_evidence)
                    and not (two_strong_score_support or two_structural_support)
                ),
                "direct_two_row_score_improvement_vs_one": two_score_improvement,
                "direct_two_row_strong_score_support": two_strong_score_support,
                "direct_two_row_structural_support": two_structural_support,
                **two_evidence,
            }
        )
    else:
        metadata.update(
            {
                "trajectory_pre_validation_row_count": inherited_automatic_k,
                "trajectory_post_validation_row_count": inherited_automatic_k,
                "trajectory_validation_fallback_applied": False,
                "trajectory_validation_fallback_reason": "",
                "trajectory_rejected_unvalidated_singleton_candidate_row_count": None,
                "trajectory_rejected_unvalidated_singleton_cluster_sizes": [],
                "trajectory_rejected_unvalidated_singleton_detector_confidences": [],
                "trajectory_displayed_non_singleton_row_count": (
                    inherited_automatic_k
                ),
                "trajectory_displayed_fallback_row_count": None,
                "trajectory_fallback_internal_secondary_support_passed": None,
                "trajectory_row_count_min": inherited_automatic_k,
                "trajectory_row_count_max": (
                    2
                    if competing_two_row_hypothesis_unresolved
                    else inherited_automatic_k
                ),
                "trajectory_row_count_range_reason": (
                    "competing_two_row_hypothesis_internal_support_failed"
                    if competing_two_row_hypothesis_unresolved
                    else "single_selected_count"
                ),
                "trajectory_possible_sparse_row": _has_singleton(automatic),
                "trajectory_possible_sparse_additional_row_count": 0,
                "trajectory_direct_two_reconsidered_v16": (
                    direct_two_reconsidered_v16
                ),
                "trajectory_competing_two_row_hypothesis_unresolved": (
                    competing_two_row_hypothesis_unresolved
                ),
                "trajectory_row_count_range_scope": (
                    "pre_partition_provisional_band"
                ),
                "trajectory_row_count_range_interpretation": (
                    "algorithmic_plausible_candidate_range_not_confidence_interval_"
                    "or_manual_truth"
                ),
                "trajectory_sobel_role": (
                    "not_consulted_for_row_selection;inherited_support_for_"
                    "ambiguous_band_partition"
                ),
                "trajectory_band_local_residual_tilt_enabled": True,
                "sparse_trajectory_rescue_applied_pre_validation": False,
            }
        )

    return selected, best_by_k, metadata


def select_row_model_v16(
    point_indices: list[int],
    lateral: np.ndarray,
    longitudinal: np.ndarray,
    median_detector_width_um: float,
    settings: TraitSettingsV16,
    audited_row_count: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run v1.5 stability analysis using the guarded v1.6 chooser."""

    diagnostic_count_before = len(v14._ACTIVE_BAND_DIAGNOSTICS)
    chosen, candidates = _V15_SELECT_ROW_MODEL(
        point_indices,
        lateral,
        longitudinal,
        median_detector_width_um,
        settings,
        audited_row_count=audited_row_count,
    )
    if len(v14._ACTIVE_BAND_DIAGNOSTICS) != diagnostic_count_before + 1:
        raise RuntimeError(
            "v1.6 expected exactly one provisional-band diagnostic record."
        )
    diagnostic = v14._ACTIVE_BAND_DIAGNOSTICS[-1]

    fallback_applied = bool(chosen.get("trajectory_validation_fallback_applied"))
    possible_sparse = bool(chosen.get("trajectory_possible_sparse_row"))
    competing_two_unresolved = bool(
        chosen.get("trajectory_competing_two_row_hypothesis_unresolved")
    )
    final_candidate_has_singleton = _has_singleton(chosen)
    if final_candidate_has_singleton:
        inherited_reason = str(
            chosen.get("trajectory_singleton_validation_failure_reason") or ""
        )
        policy_reason = "v1_6_singleton_exact_rows_not_permitted"
        failure_reason = ";".join(
            value for value in (inherited_reason, policy_reason) if value
        )
        chosen["trajectory_singleton_rows_validated"] = False
        chosen["trajectory_singleton_validation_failure_reason"] = failure_reason
        diagnostic["trajectory_singleton_rows_validated"] = False
        diagnostic["trajectory_singleton_validation_failure_reason"] = failure_reason
    if fallback_applied or possible_sparse or competing_two_unresolved:
        # The displayed count is a provisional non-singleton model.  Exact
        # row-dependent GWAS traits remain withheld until competing structure is
        # resolved on a representative biological validation set.
        chosen["trajectory_exact_row_traits_eligible"] = False
        chosen["model_confidence"] = "low"
        diagnostic["trajectory_exact_row_traits_eligible"] = False
        diagnostic["global_row_model_confidence"] = "low"

    chosen["tilt_slope_source"] = (
        "image_wide_axis_prior_plus_band_local_shared_residual_tilt"
    )
    diagnostic["tilt_slope_source"] = chosen["tilt_slope_source"]

    copied_fields = {
        key: chosen.get(key)
        for key in (
            "trajectory_pre_validation_row_count",
            "trajectory_post_validation_row_count",
            "trajectory_validation_fallback_applied",
            "trajectory_validation_fallback_reason",
            "trajectory_rejected_unvalidated_singleton_candidate_row_count",
            "trajectory_rejected_unvalidated_singleton_cluster_sizes",
            "trajectory_rejected_unvalidated_singleton_detector_confidences",
            "trajectory_displayed_non_singleton_row_count",
            "trajectory_displayed_fallback_row_count",
            "trajectory_fallback_internal_secondary_support_passed",
            "trajectory_row_count_min",
            "trajectory_row_count_max",
            "trajectory_row_count_range_reason",
            "trajectory_possible_sparse_row",
            "trajectory_possible_sparse_additional_row_count",
            "trajectory_direct_two_reconsidered_v16",
            "trajectory_competing_two_row_hypothesis_unresolved",
            "trajectory_row_count_range_scope",
            "trajectory_row_count_range_interpretation",
            "trajectory_sobel_role",
            "trajectory_band_local_residual_tilt_enabled",
            "sparse_trajectory_rescue_applied_pre_validation",
        )
    }
    diagnostic.update(copied_fields)
    return chosen, candidates


def _trajectory_diagnostic_fields_v16(
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """Copy v1.6 uncertainty/provenance fields into detailed CSV records."""

    fields = _V15_DIAGNOSTIC_FIELDS(diagnostic)
    fields["row_model_version"] = SCRIPT_VERSION
    for key, default in (
        ("trajectory_pre_validation_row_count", None),
        ("trajectory_post_validation_row_count", None),
        ("trajectory_validation_fallback_applied", False),
        ("trajectory_validation_fallback_reason", ""),
        ("trajectory_rejected_unvalidated_singleton_candidate_row_count", None),
        ("trajectory_rejected_unvalidated_singleton_cluster_sizes", []),
        ("trajectory_rejected_unvalidated_singleton_detector_confidences", []),
        ("trajectory_displayed_non_singleton_row_count", None),
        ("trajectory_displayed_fallback_row_count", None),
        ("trajectory_fallback_internal_secondary_support_passed", None),
        ("trajectory_row_count_min", None),
        ("trajectory_row_count_max", None),
        ("trajectory_row_count_range_reason", ""),
        ("trajectory_possible_sparse_row", False),
        ("trajectory_possible_sparse_additional_row_count", 0),
        ("trajectory_direct_two_reconsidered_v16", False),
        ("trajectory_competing_two_row_hypothesis_unresolved", False),
        ("trajectory_row_count_range_scope", "pre_partition_provisional_band"),
        (
            "trajectory_row_count_range_interpretation",
            "algorithmic_plausible_candidate_range_not_confidence_interval_"
            "or_manual_truth",
        ),
        (
            "trajectory_sobel_role",
            "not_consulted_for_row_selection;inherited_support_for_"
            "ambiguous_band_partition",
        ),
        ("trajectory_band_local_residual_tilt_enabled", True),
        ("sparse_trajectory_rescue_applied_pre_validation", False),
    ):
        fields[key] = diagnostic.get(key, default)
    return fields


def _remark_partial_band_eligibility_v16(
    row_records: list[dict[str, Any]],
    band_records: list[dict[str, Any]],
    point_assignment: dict[int, dict[str, Any]],
    stoma_traits: list[dict[str, Any]],
    median_detector_width_um: float,
    trajectory_slope: float,
    settings: TraitSettingsV16,
) -> list[dict[str, Any]]:
    """Recalculate field coverage along each band's fitted row trajectories."""

    axis = v14._active_axis(stoma_traits)
    metadata = v14._ACTIVE_IMAGE_METADATA
    if axis is None or metadata is None:
        return _V15_REMARK_PARTIAL_BAND_ELIGIBILITY(
            row_records,
            band_records,
            point_assignment,
            stoma_traits,
            median_detector_width_um,
            trajectory_slope,
            settings,
        )
    px_x = base.optional_float(metadata.get("metadata_pixels_per_um_x"))
    px_y = base.optional_float(metadata.get("metadata_pixels_per_um_y"))
    width_pixels = base.optional_int(metadata.get("image_width"))
    height_pixels = base.optional_int(metadata.get("image_height"))
    if px_x is None or px_y is None or width_pixels is None or height_pixels is None:
        return _V15_REMARK_PARTIAL_BAND_ELIGIBILITY(
            row_records,
            band_records,
            point_assignment,
            stoma_traits,
            median_detector_width_um,
            trajectory_slope,
            settings,
        )

    width_um = width_pixels / px_x
    height_um = height_pixels / px_y
    corners = [
        axis.forward(x, y)
        for x, y in (
            (0.0, 0.0),
            (width_um, 0.0),
            (0.0, height_um),
            (width_um, height_um),
        )
    ]
    image_long_min = min(item[1] for item in corners)
    image_long_max = max(item[1] for item in corners)
    row_lookup = {str(row["row_id"]): row for row in row_records}

    measurements: list[dict[str, Any]] = []
    for band in band_records:
        observed_values = [
            float(stoma_traits[index]["longitudinal_coordinate_um"])
            for index in band["_point_indices"]
        ]
        observed = float(np.ptp(observed_values)) if observed_values else 0.0
        rows = [
            row_lookup[str(row_id)]
            for row_id in band.get("row_ids", [])
            if str(row_id) in row_lookup
        ]
        visible: list[float] = []
        for longitudinal_um in np.linspace(image_long_min, image_long_max, 1200):
            if rows:
                lateral_um = float(
                    np.median(
                        [
                            base.row_lateral_at(row, float(longitudinal_um))
                            for row in rows
                        ]
                    )
                )
            else:
                lateral_um = float(band["band_center_lateral_um"]) + (
                    trajectory_slope * float(longitudinal_um)
                )
            x_um, y_um = axis.inverse(lateral_um, float(longitudinal_um))
            if 0.0 <= x_um <= width_um and 0.0 <= y_um <= height_um:
                visible.append(float(longitudinal_um))
        accessible = float(max(visible) - min(visible)) if len(visible) >= 2 else None

        edge_factor = base.optional_float(
            band.get("band_detector_side_edge_margin_box_width_factor")
        )
        edge_source = "detector_bbox_source_pixels"
        if edge_factor is None:
            edge_factor = base.optional_float(
                band.get("band_rotated_lateral_edge_margin_box_width_factor")
            )
            edge_source = "inherited_rotated_center_fallback"
        measurements.append(
            {
                "band": band,
                "observed": observed,
                "accessible": accessible,
                "coverage": base.safe_divide(observed, accessible),
                "edge_factor": edge_factor,
                "edge_source": edge_source,
            }
        )

    reference_pool = [
        item["observed"]
        for item in measurements
        if item["observed"] > 0
        and (
            item["edge_factor"] is None
            or item["edge_factor"]
            > settings.partial_band_edge_margin_box_width_factor
        )
    ]
    if not reference_pool:
        reference_pool = [
            item["observed"] for item in measurements if item["observed"] > 0
        ]
    reference_span = float(np.median(reference_pool)) if reference_pool else None
    eligibility_audit: list[dict[str, Any]] = []
    basis = "median_of_fitted_band_row_trajectories_clipped_to_source_field"

    for item in measurements:
        band = item["band"]
        observed = float(item["observed"])
        coverage = item["coverage"]
        relative_span = base.safe_divide(observed, reference_span)
        edge_factor = item["edge_factor"]
        low_absolute = bool(
            coverage is not None
            and coverage < settings.partial_band_min_longitudinal_coverage_fraction
        )
        low_relative = bool(
            relative_span is not None
            and relative_span < settings.partial_band_min_relative_observed_span
        )
        edge_risk = bool(
            edge_factor is not None
            and edge_factor <= settings.partial_band_edge_margin_box_width_factor
        )
        excluded = bool(
            (low_absolute and (low_relative or observed == 0.0))
            or (edge_risk and low_relative)
        )
        reason = ""
        if excluded:
            reason = (
                "field_truncated_band_short_relative_span_at_lateral_edge"
                if edge_risk
                else "incomplete_band_low_absolute_and_relative_longitudinal_span"
            )
        band.update(
            {
                "band_spatial_eligible": not excluded,
                "band_fov_truncated": excluded,
                "band_spatial_exclusion_reason": reason,
                "band_observed_longitudinal_span_um": observed,
                "band_accessible_longitudinal_span_um": item["accessible"],
                "band_longitudinal_coverage_fraction": coverage,
                "band_relative_observed_span": relative_span,
                "partial_band_reference_span_um": reference_span,
                "partial_band_accessible_span_basis": basis,
                "band_lateral_edge_margin_source": item["edge_source"],
            }
        )
        for row_id in band.get("row_ids", []):
            row = row_lookup[str(row_id)]
            row["row_band_spatial_eligible"] = not excluded
            row["row_band_spatial_exclusion_reason"] = reason
            for index in row["_point_indices"]:
                point_assignment[index]["row_band_spatial_eligible"] = not excluded
                point_assignment[index]["row_band_spatial_exclusion_reason"] = reason
        eligibility_audit.append(
            {
                "band_id": band["band_id"],
                "provisional_band_id": band.get("provisional_band_id"),
                "spatial_eligible": not excluded,
                "exclusion_reason": reason,
                "observed_longitudinal_span_um": observed,
                "accessible_longitudinal_span_um": item["accessible"],
                "coverage_fraction": coverage,
                "relative_observed_span": relative_span,
                "lateral_edge_margin_box_width_factor": edge_factor,
                "lateral_edge_margin_source": item["edge_source"],
                "v16_recalculation_available": True,
                "accessible_span_basis": basis,
            }
        )
    return eligibility_audit


def infer_rows_and_bands_v16(
    stoma_traits: list[dict[str, Any]],
    settings: TraitSettingsV16,
    audited_row_counts: dict[str, int] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    """Run inherited inference and make the v1.6 policies explicit."""

    row_records, band_records, point_assignment, audit = _V15_INFER_ROWS_AND_BANDS(
        stoma_traits,
        settings,
        audited_row_counts=audited_row_counts,
    )
    for row in row_records:
        row["row_model_version"] = SCRIPT_VERSION
        row["row_tilt_slope_source"] = (
            "image_wide_axis_prior_plus_band_local_shared_residual_tilt"
        )
    for band in band_records:
        band["row_model_version"] = SCRIPT_VERSION
        band["tilt_slope_source"] = (
            "image_wide_axis_prior_plus_band_local_shared_residual_tilt"
        )
        band["trajectory_tilt_model"] = (
            "image_wide_axis_prior_plus_one_shared_residual_tilt_per_provisional_band"
        )
    for assignment in point_assignment.values():
        assignment["row_model_version"] = SCRIPT_VERSION
        assignment["tilt_slope_source"] = (
            "image_wide_axis_prior_plus_band_local_shared_residual_tilt"
        )

    diagnostics = audit.get("band_selection_diagnostics", {})
    audit.update(
        {
            "row_model_version": SCRIPT_VERSION,
            "trajectory_image_wide_orientation_prior_used": True,
            "trajectory_image_wide_orientation_enforced": False,
            "trajectory_noncrossing_constrained": False,
            "trajectory_noncrossing_checked_at_image_endpoints": True,
            "trajectory_band_local_residual_tilt_enabled": True,
            "trajectory_sobel_policy": (
                "generic raw-edge contrast is not consulted for row-count "
                "selection and cannot create or validate a singleton row; the "
                "inherited local edge rule remains active only for an ambiguous "
                "boundary between already-supported rows"
            ),
            "automatic_row_count_policy": (
                "estimate an image-wide orientation prior; fit every provisional "
                "band across a bounded shared residual-tilt grid; reject any exact "
                "candidate containing a singleton before final model selection; "
                "display the best non-singleton simpler count when available and "
                "retain the rejected sparse hypothesis as an algorithmic plausible "
                "row-count range, not a confidence interval or manual truth"
            ),
            "trajectory_stability_policy": (
                "repeat the complete post-validation selector after deterministic "
                "lateral jitter and minimum-separation perturbation; stability is "
                "a QC condition and cannot validate a singleton row"
            ),
            "trajectory_stability_limitations": (
                "experimental v1.6 retains v1.5 lateral-jitter and separation "
                "sensitivity; detection-dropout, longitudinal-window and angle-grid "
                "perturbations are not yet implemented"
            ),
            "partial_band_accessible_span_basis": (
                "median_of_fitted_band_row_trajectories_clipped_to_source_field"
            ),
            "trajectory_validation_fallback_group_count": sum(
                bool(item.get("trajectory_validation_fallback_applied"))
                for item in diagnostics.values()
            ),
            "trajectory_possible_sparse_row_group_count": sum(
                bool(item.get("trajectory_possible_sparse_row"))
                for item in diagnostics.values()
            ),
            "trajectory_settings": {
                key: value
                for key, value in asdict(settings).items()
                if key.startswith(("trajectory_", "sparse_trajectory_", "band_"))
            },
        }
    )
    return row_records, band_records, point_assignment, audit


def image_trait_summary_v16(
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
    """Correct image/GWAS provenance for the v1.6 soft-orientation model."""

    result = _V15_IMAGE_TRAIT_SUMMARY(
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
    eligible_bands = [
        band for band in band_records if bool(band.get("band_spatial_eligible"))
    ]
    provisional: dict[str, dict[str, Any]] = {}
    for band in band_records:
        provisional.setdefault(
            str(band.get("provisional_band_id") or band.get("band_id")), band
        )
    eligible_provisional: dict[str, dict[str, Any]] = {}
    for band in eligible_bands:
        eligible_provisional.setdefault(
            str(band.get("provisional_band_id") or band.get("band_id")), band
        )

    fallback_groups = [
        band
        for band in provisional.values()
        if bool(band.get("trajectory_validation_fallback_applied"))
    ]
    eligible_fallback_groups = [
        band
        for band in eligible_provisional.values()
        if bool(band.get("trajectory_validation_fallback_applied"))
    ]
    possible_sparse_groups = [
        band
        for band in provisional.values()
        if bool(band.get("trajectory_possible_sparse_row"))
    ]
    competing_two_groups = [
        band
        for band in eligible_provisional.values()
        if bool(band.get("trajectory_competing_two_row_hypothesis_unresolved"))
    ]

    reasons = [
        value
        for value in str(
            result.get("trajectory_row_band_gwas_exclusion_reason") or ""
        ).split(";")
        if value
    ]
    if eligible_fallback_groups:
        reasons.append("singleton_candidate_rejected_row_count_unresolved")
    if competing_two_groups:
        reasons.append("competing_one_vs_two_row_structure_unresolved")

    result.update(
        {
            "row_model_version": SCRIPT_VERSION,
            "trajectory_image_wide_orientation_prior_used": True,
            "trajectory_image_wide_orientation_enforced": False,
            "trajectory_noncrossing_constrained": False,
            "trajectory_noncrossing_checked_at_image_endpoints": True,
            "trajectory_band_local_residual_tilt_enabled": True,
            "trajectory_validation_fallback_group_count": len(fallback_groups),
            "trajectory_validation_fallback_spatially_eligible_group_count": len(
                eligible_fallback_groups
            ),
            "trajectory_possible_sparse_row_group_count": len(
                possible_sparse_groups
            ),
            "trajectory_competing_two_row_unresolved_group_count": len(
                competing_two_groups
            ),
            "trajectory_row_band_gwas_exclusion_reason": ";".join(
                dict.fromkeys(reasons)
            ),
            "trajectory_row_band_trait_policy": (
                "detailed CSV retains provisional v1.6 estimates and row-count "
                "ranges; compact GWAS CSV blanks exact row/band phenotypes for "
                "an unresolved singleton fallback, competing one-versus-two "
                "structure, unstable assignments, unresolved band boundaries or "
                "other inherited spatial QC failure; density, morphometry and "
                "ordinary nearest-neighbour traits remain available"
            ),
            "trajectory_stability_limitations": row_band_audit.get(
                "trajectory_stability_limitations"
            ),
        }
    )
    return result


def image_gwas_trait_summary_v16(
    image_traits: dict[str, Any],
) -> dict[str, Any]:
    """Carry v1.6 QC/provenance into the compact image-level GWAS CSV."""

    summary = _V15_IMAGE_GWAS_TRAIT_SUMMARY(image_traits)
    for field in (
        "trajectory_image_wide_orientation_prior_used",
        "trajectory_image_wide_orientation_enforced",
        "trajectory_noncrossing_constrained",
        "trajectory_noncrossing_checked_at_image_endpoints",
        "trajectory_band_local_residual_tilt_enabled",
        "trajectory_validation_fallback_group_count",
        "trajectory_validation_fallback_spatially_eligible_group_count",
        "trajectory_possible_sparse_row_group_count",
        "trajectory_competing_two_row_unresolved_group_count",
        "trajectory_stability_limitations",
    ):
        summary[field] = image_traits.get(field)
    summary["row_model_version"] = SCRIPT_VERSION
    summary["trajectory_row_band_gwas_exclusion_reason"] = image_traits.get(
        "trajectory_row_band_gwas_exclusion_reason"
    )
    summary["trajectory_row_band_trait_policy"] = image_traits.get(
        "trajectory_row_band_trait_policy"
    )
    return summary


def install_v16_trajectory_selector() -> None:
    """Install v1.6 extension points without changing parent scripts."""

    v15.install_v15_trajectory_selector()

    # Functions looked up dynamically inside the v1.5 implementation.
    v15._slope_angles = _slope_angles_v16
    v15._choose_trajectory_model_once = _choose_trajectory_model_once_v16
    v15._trajectory_diagnostic_fields = _trajectory_diagnostic_fields_v16
    v15._remark_partial_band_eligibility_v15 = _remark_partial_band_eligibility_v16
    v15.SCRIPT_VERSION = SCRIPT_VERSION
    v14.SCRIPT_VERSION = SCRIPT_VERSION

    base.parse_arguments = parse_arguments_v16
    base.build_settings = build_settings_v16
    base.select_row_model = select_row_model_v16
    base.infer_rows_and_bands = infer_rows_and_bands_v16
    base.image_trait_summary = image_trait_summary_v16
    base.image_gwas_trait_summary = image_gwas_trait_summary_v16
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base.__file__ = __file__


def run_trait_extraction() -> Path:
    """Run the separate v1.6 inspection candidate with inherited safeguards."""

    install_v16_trajectory_selector()
    return base.run_trait_extraction()


if __name__ == "__main__":
    run_trait_extraction()
