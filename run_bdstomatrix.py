#!/usr/bin/env python3
"""Run the BdStomatrix inference and trait-extraction pipeline.

This is the final production entry point for either one OME-TIFF or a
directory of OME-TIFFs. It uses the audited inference module and the
self-contained, hash-verified single-file v2.5.4 trait extractor.

Lean output policy
------------------
* The v2.5.4 trait directory is the only consolidated CSV output.  The
  redundant ``batch_image_traits.csv`` and ``batch_image_gwas_traits.csv``
  copies made by the older directory-only batch wrapper are not created.
* Spatial row/band audit overlays are always created.
* Detector overview JPEGs are removed after successful trait extraction by
  default because the spatial overlays are the retained full-image audit.
  Use ``--keep-inference-overviews`` only for a deliberate diagnostic run.
* Inference manifests, compressed masks, masked-crop references and
  provenance are retained.  They are not duplicate trait tables and are
  required to audit or reproduce mask-derived traits without rerunning the
  models.

The output root must not already exist.  A failed stage is retained for
inspection and original images are never modified.

Examples
--------
Single image::

    python run_bdstomatrix.py \
        --input /path/to/S001_AB_P1_EDF.ome.tif \
        --output-dir /path/to/S001_AB_P1_final_v254

Directory::

    python run_bdstomatrix.py \
        --input /path/to/ome_tiffs \
        --output-dir /path/to/cohort_final_v254

Add ``--recursive`` only when nested input directories should be searched.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import shutil
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import trait_extractor
import run_ome_to_final_segmentation_pipeline as inference_pipeline


SCRIPT_VERSION = "1.1.0"
INFERENCE_DIRECTORY_NAME = "inference_support"
TRAIT_DIRECTORY_NAME = "traits_v2_5_4"

# These hashes freeze the two runtime Python components. The single trait
# extractor independently verifies all 16 embedded audited source identities.
EXPECTED_COMPONENT_SHA256 = {
    "run_ome_to_final_segmentation_pipeline.py": (
        "a41a50a6e82d8f40698c9fadd2ff36c659b36e60dd8f5e0f871329232b7fc98c"
    ),
    "trait_extractor.py": (
        "87121bb33b4fab3de1dd66263b14b9b7462182a10afd42be178fa54cde47def3"
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one OME-TIFF or a directory through detection, "
            "classification, segmentation and trait extraction."
        )
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="One OME-TIFF file or a directory containing OME-TIFF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New final output root; it must not already exist.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search nested input subdirectories; off by default.",
    )
    parser.add_argument(
        "--device",
        choices=("mps", "cpu", "cuda"),
        default="mps",
        help="Inference device; MPS is default and never silently falls back.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="Zero-based NVIDIA GPU index; used only with --device cuda.",
    )
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=32,
        help="Classifier batch size; reduce to 16 or 8 if memory is limited.",
    )
    parser.add_argument(
        "--expected-image-width",
        type=int,
        default=inference_pipeline.EXPECTED_IMAGE_WIDTH,
        help=(
            "Expected input width in pixels; defaults to the validated "
            f"{inference_pipeline.EXPECTED_IMAGE_WIDTH}."
        ),
    )
    parser.add_argument(
        "--expected-image-height",
        type=int,
        default=inference_pipeline.EXPECTED_IMAGE_HEIGHT,
        help=(
            "Expected input height in pixels; defaults to the validated "
            f"{inference_pipeline.EXPECTED_IMAGE_HEIGHT}."
        ),
    )
    parser.add_argument(
        "--calibration-policy",
        choices=("strict-reference", "use-image-metadata"),
        default="strict-reference",
        help=(
            "Strict mode requires OME calibration to match 4.5848 px/um. "
            "Metadata mode uses valid per-image OME X/Y calibration and "
            "flags any disagreement."
        ),
    )
    parser.add_argument(
        "--allow-ultralytics-version-mismatch",
        action="store_true",
        help=(
            "Deliberately allow and record an Ultralytics version mismatch. "
            "Do not use for the locked production run."
        ),
    )
    parser.add_argument(
        "--keep-inference-overviews",
        action="store_true",
        help=(
            "Retain detector overview JPEGs. By default they are removed "
            "after successful extraction because spatial overlays are kept."
        ),
    )
    return parser.parse_args()


def validate_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    input_path = args.input.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    inference_output = output_root / INFERENCE_DIRECTORY_NAME
    trait_output = output_root / TRAIT_DIRECTORY_NAME

    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist:\n{input_path}")
    if not (input_path.is_file() or input_path.is_dir()):
        raise ValueError(f"Input is neither a file nor a directory:\n{input_path}")
    if output_root.exists():
        raise FileExistsError(f"Output root already exists:\n{output_root}")
    if output_root.with_name(output_root.name + ".incomplete").exists():
        raise FileExistsError(
            "A final-run staging directory already exists:\n"
            f"{output_root.with_name(output_root.name + '.incomplete')}"
        )
    if input_path.is_dir() and output_root.is_relative_to(input_path):
        raise ValueError(
            "Place the output outside the input directory so generated files "
            "cannot be rediscovered as source images."
        )
    if args.classifier_batch_size < 1:
        raise ValueError("--classifier-batch-size must be at least 1.")
    if args.cuda_device < 0:
        raise ValueError("--cuda-device cannot be negative.")
    if args.expected_image_width < 1 or args.expected_image_height < 1:
        raise ValueError("Expected image width and height must be positive.")
    return input_path, output_root, inference_output, trait_output


def verify_locked_python_components() -> dict[str, dict[str, str]]:
    component_root = Path(__file__).resolve().parent
    observed: dict[str, dict[str, str]] = {}
    mismatches: list[str] = []
    for filename, expected_hash in EXPECTED_COMPONENT_SHA256.items():
        path = component_root / filename
        if not path.is_file():
            mismatches.append(f"missing: {path}")
            continue
        actual_hash = sha256_file(path)
        observed[filename] = {
            "path": str(path),
            "sha256": actual_hash,
        }
        if actual_hash != expected_hash:
            mismatches.append(
                f"hash mismatch: {filename}\n"
                f"  expected: {expected_hash}\n"
                f"  observed: {actual_hash}"
            )
    if mismatches:
        raise RuntimeError(
            "The locked Python component set has changed:\n" + "\n".join(mismatches)
        )
    return observed


@contextmanager
def delegated_arguments(arguments: Sequence[str]) -> Iterator[None]:
    original_arguments = sys.argv[:]
    sys.argv = [str(Path(__file__).resolve()), *map(str, arguments)]
    try:
        yield
    finally:
        sys.argv = original_arguments


def inference_arguments(
    args: argparse.Namespace, input_path: Path, output_dir: Path
) -> list[str]:
    values = [
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--cuda-device",
        str(args.cuda_device),
        "--classifier-batch-size",
        str(args.classifier_batch_size),
        "--expected-image-width",
        str(args.expected_image_width),
        "--expected-image-height",
        str(args.expected_image_height),
        "--calibration-reference",
        str(inference_pipeline.REFERENCE_PIXELS_PER_MICROMETRE),
        "--calibration-absolute-tolerance",
        str(inference_pipeline.CALIBRATION_ABSOLUTE_TOLERANCE),
        "--calibration-policy",
        args.calibration_policy,
        "--stop-on-calibration-problem",
    ]
    if args.recursive:
        values.append("--recursive")
    if args.allow_ultralytics_version_mismatch:
        values.append("--allow-ultralytics-version-mismatch")
    return values


def trait_arguments(inference_output: Path, trait_output: Path) -> list[str]:
    # All scientific settings intentionally use the audited v2.5.4 defaults.
    # Spatial audit overlays are intentionally always enabled.
    return [
        "--pipeline-output-dir",
        str(inference_output),
        "--output-dir",
        str(trait_output),
    ]


def release_inference_memory() -> None:
    gc.collect()
    if inference_pipeline.torch.backends.mps.is_available():
        inference_pipeline.torch.mps.empty_cache()
    if inference_pipeline.torch.cuda.is_available():
        inference_pipeline.torch.cuda.empty_cache()
    gc.collect()


def remove_inference_overviews(inference_output: Path) -> int:
    overview_dir = inference_output / "overviews"
    if not overview_dir.exists():
        return 0
    if overview_dir.resolve().parent != inference_output.resolve():
        raise RuntimeError(f"Refusing to remove unexpected path: {overview_dir}")
    overview_count = sum(path.is_file() for path in overview_dir.rglob("*"))
    shutil.rmtree(overview_dir)
    return overview_count


def write_final_provenance(
    args: argparse.Namespace,
    input_path: Path,
    output_root: Path,
    inference_output: Path,
    trait_output: Path,
    component_record: dict[str, dict[str, str]],
    started_at: str,
    removed_overview_count: int,
) -> None:
    script_path = Path(__file__).resolve()
    payload = {
        "workflow": "Brachypodium_final_OME_to_traits_v2_5_4",
        "script_version": SCRIPT_VERSION,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "input_argument": str(input_path),
        "input_mode": "single_file" if input_path.is_file() else "directory",
        "recursive_directory_discovery": bool(args.recursive),
        "output_root": str(output_root),
        "inference_support_directory": str(inference_output),
        "trait_output_directory": str(trait_output),
        "source_images_modified": False,
        "calibration_policy": args.calibration_policy,
        "calibration_failure_policy": (
            "stop_on_missing_or_invalid_metadata_and_reference_mismatch"
            if args.calibration_policy == "strict-reference"
            else "use_valid_image_metadata_with_recorded_mismatch_warning"
        ),
        "configured_expected_image_width": args.expected_image_width,
        "configured_expected_image_height": args.expected_image_height,
        "configured_resolution_is_default": bool(
            args.expected_image_width == inference_pipeline.EXPECTED_IMAGE_WIDTH
            and args.expected_image_height
            == inference_pipeline.EXPECTED_IMAGE_HEIGHT
        ),
        "scientific_trait_settings": "locked_v2_5_4_defaults",
        "spatial_audit_overlays_created": True,
        "duplicate_batch_trait_tables_created": False,
        "inference_overviews_retained": bool(args.keep_inference_overviews),
        "inference_overview_files_removed": removed_overview_count,
        "retained_inference_support": [
            "input and prediction manifests",
            "compressed selected masks",
            "masked-crop QC references",
            "inference provenance and report",
        ],
        "entry_point": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
        },
        "locked_python_components": component_record,
        "model_checkpoints": {
            "detector": {
                "path": str(inference_pipeline.DETECTOR_CHECKPOINT),
                "expected_sha256": inference_pipeline.EXPECTED_DETECTOR_SHA256,
            },
            "classifier": {
                "path": str(inference_pipeline.CLASSIFIER_CHECKPOINT),
                "expected_sha256": inference_pipeline.EXPECTED_CLASSIFIER_SHA256,
            },
            "segmentor": {
                "path": str(inference_pipeline.SEGMENTOR_CHECKPOINT),
                "expected_sha256": inference_pipeline.EXPECTED_SEGMENTOR_SHA256,
            },
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": inference_pipeline.np.__version__,
            "opencv": inference_pipeline.cv2.__version__,
            "tifffile": inference_pipeline.tifffile.__version__,
            "torch": inference_pipeline.torch.__version__,
            "torchvision": inference_pipeline.torchvision.__version__,
            "ultralytics": inference_pipeline.ultralytics.__version__,
        },
        "operational_arguments": {
            "device": args.device,
            "cuda_device": args.cuda_device,
            "classifier_batch_size": args.classifier_batch_size,
            "expected_image_width": args.expected_image_width,
            "expected_image_height": args.expected_image_height,
            "calibration_policy": args.calibration_policy,
            "allow_ultralytics_version_mismatch": (
                args.allow_ultralytics_version_mismatch
            ),
            "keep_inference_overviews": args.keep_inference_overviews,
        },
        "principal_outputs": {
            "stoma_traits": str(trait_output / "stoma_traits.csv"),
            "image_traits": str(trait_output / "image_traits.csv"),
            "image_gwas_traits": str(trait_output / "image_gwas_traits.csv"),
            "row_traits_qc": str(trait_output / "row_traits_qc.csv"),
            "band_traits_qc": str(trait_output / "band_traits_qc.csv"),
            "spatial_audit_overlays": str(
                trait_output / "spatial_audit_overlays"
            ),
        },
    }
    (output_root / "final_run_provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_final_report(
    output_root: Path,
    inference_output: Path,
    trait_output: Path,
    removed_overview_count: int,
) -> None:
    lines = [
        "Brachypodium final inference + v2.5.4 trait pipeline completed",
        "",
        f"Completed: {utc_now()}",
        f"Inference support: {inference_output}",
        f"Final traits: {trait_output}",
        f"Spatial audits: {trait_output / 'spatial_audit_overlays'}",
        f"Inference overview JPEGs removed: {removed_overview_count}",
        "",
        "No duplicate top-level batch trait CSV files were created.",
        "Original OME-TIFF files were not modified.",
        "Inference manifests and masks were retained for reproducibility.",
        "AB and AD must remain separate in downstream aggregation and GWAS.",
    ]
    (output_root / "FINAL_RUN_REPORT.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_failure_report(output_root: Path, stage: str) -> None:
    if not output_root.exists():
        return
    failure_path = output_root / "FINAL_PIPELINE_FAILURE.txt"
    if not failure_path.exists():
        failure_path.write_text(
            f"Final pipeline stopped during: {stage}\n\n" + traceback.format_exc(),
            encoding="utf-8",
        )


def run_final_pipeline() -> tuple[Path, Path]:
    args = parse_arguments()
    input_path, output_root, inference_output, trait_output = validate_paths(args)
    component_record = verify_locked_python_components()
    started_at = utc_now()
    active_stage = "inference"

    try:
        print("Stage 1/2: locked detection, classification and segmentation", flush=True)
        with delegated_arguments(
            inference_arguments(args, input_path, inference_output)
        ):
            completed_inference = inference_pipeline.run_pipeline()
        if completed_inference.resolve() != inference_output.resolve():
            raise RuntimeError(
                "Inference returned an unexpected output directory:\n"
                f"Expected: {inference_output}\nObserved: {completed_inference}"
            )

        release_inference_memory()

        active_stage = "trait extraction"
        print("Stage 2/2: trait extraction", flush=True)
        with delegated_arguments(trait_arguments(inference_output, trait_output)):
            completed_traits = trait_extractor.run_trait_extraction()
        if completed_traits.resolve() != trait_output.resolve():
            raise RuntimeError(
                "Trait extraction returned an unexpected output directory:\n"
                f"Expected: {trait_output}\nObserved: {completed_traits}"
            )

        active_stage = "final output compaction"
        removed_overview_count = 0
        if not args.keep_inference_overviews:
            removed_overview_count = remove_inference_overviews(inference_output)

        write_final_provenance(
            args=args,
            input_path=input_path,
            output_root=output_root,
            inference_output=inference_output,
            trait_output=trait_output,
            component_record=component_record,
            started_at=started_at,
            removed_overview_count=removed_overview_count,
        )
        write_final_report(
            output_root,
            inference_output,
            trait_output,
            removed_overview_count,
        )
    except Exception:
        write_failure_report(output_root, active_stage)
        raise

    print("Final pipeline completed successfully.")
    print(f"Final output root: {output_root}")
    print(f"Image traits: {trait_output / 'image_traits.csv'}")
    print(f"GWAS traits: {trait_output / 'image_gwas_traits.csv'}")
    print(f"Spatial audit overlays: {trait_output / 'spatial_audit_overlays'}")
    return inference_output, trait_output


if __name__ == "__main__":
    run_final_pipeline()
