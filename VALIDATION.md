# Validation record

## Scientific development evidence

- The inherited v2.5.3 topology and clipped-singleton policy was evaluated on
  the original balanced 40-image confirmation set.
- The v2.5.4 generalized band-boundary rules were evaluated on an additional
  balanced 40-image audit set.
- Density and morphometry eligibility/count summaries were unchanged by the
  row/band-only v2.5.4 correction.
- The final modular BdStomatrix bundle completed both a single-image run and a
  directory run before this single-file distribution was built.

## Single-file extractor regression

The v1 single-file extractor was run independently on the completed
single-image inference output for `S006_AB_P3_EDF.ome.tif` and compared with a
fresh run of the modular v2.5.4 extractor.

Result:

- All CSV and QC tables were byte-identical.
- The spatial row/band audit overlay was byte-identical.
- Eleven non-provenance output artifacts were compared exactly.
- The only intentional differences were the report and provenance additions
  describing the single-file bundle.
- All 16 embedded source hashes passed before import.
- No private temporary source path remained in final provenance.

The same comparison was then run on the completed four-image directory
inference output (`S047_AB_P3`, `S047_AD_P5`, `S051_AB_P3` and `S051_AB_P4`).
All ten generated CSV/QC artifacts were byte-identical between the modular and
single-file extractors. Spatial overlay identity was tested separately in the
single-image regression.

## BdStomatrix final v1.1.0 full 40-image regression

The complete `run_bdstomatrix.py` workflow was rerun on the balanced
independent 40-image set using MPS, the default 3072 x 2048 dimensions and the
default strict-reference calibration policy. Results were compared with the
original modular inference plus v2.5.4 extraction.

Results:

- All ten trait and QC CSV files were byte-identical.
- All 40 spatial row/band audit overlays were byte-identical.
- All common fields in six inference manifests were identical.
- All 924 compressed mask archives contained identical arrays.
- All 924 masked-crop reference PNGs were byte-identical.
- Final counts were identical: 40 images, 1,439 stomata, 3,507 selected cell
  instances, 455 rows and 234 bands.
- All 40 images used the default resolution and passed strict reference
  calibration.
- New differences were limited to added resolution/calibration audit fields,
  script/provenance identities and final packaging metadata.

## New-option branch tests

- The default and user-specified dimension validation paths passed using
  controlled TIFF dimensions.
- Strict calibration accepted 4.585 pixels/µm and retained the metadata value
  for calculations.
- Strict calibration rejected an out-of-tolerance metadata value.
- The opt-in metadata policy permitted isotropic and anisotropic positive X/Y
  metadata, marked the result provisional, and emitted the appropriate status
  and warning.
- Missing metadata remained ineligible under both policies.
- Explicit CPU device selection passed.
- CUDA unavailability produces an informative error without CPU fallback.

## v1.1.0 release-identity cleanup

After the full scientific regression, release-only text and identity metadata
were cleaned without changing model inference or trait calculations. Private
development paths and the obsolete inference-stage trait message were removed,
the application and single-file distribution bundle were aligned to v1.1.0,
and the production entry point gained `--version`. The 16 embedded scientific
extractor sources and all three checkpoint files remained byte-identical.
Syntax parsing, embedded-source import, command help, version reporting,
component-integrity verification and the complete release checksum manifest
passed after this cleanup.


