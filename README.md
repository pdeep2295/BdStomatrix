# BdStomatrix (v1.1.0)

![BdStomatrix](logo.png)

BdStomatrix analyses OME-TIFF microscope images of the *Brachypodium* leaf
epidermis. It measures stomatal density and morphometry and describes stomatal
rows and bands.

The pipeline uses a detector, classifier and segmentation model to identify
stomata and obtain guard-cell and subsidiary-cell masks. The detected objects
and masks are then used to calculate image-level and individual-stoma traits.

The current models were trained and tested using 20× Differential Interference
Contrast (DIC) images. Images collected at other magnifications or with other
microscopy methods should be independently validated before biological
analysis.

The main scripts are:

- `run_bdstomatrix.py` runs the complete pipeline on one TIFF image or a
  directory of images.
- `trait_extractor.py` calculates traits from existing BdStomatrix detection
  and segmentation results. It contains the full effective v2.5.4 calculation
  chain as readable Python source in one file.

## Readable source code

The full trait extraction code is provided in `trait_extractor.py` as a single
readable Python file. The individual development scripts are included in
`traitextraction_scripts/` for reference and are not needed to run the pipeline.

## Model training scripts

The scripts used to generate the classifier crops and train the detector,
classifier and segmentation model are provided in the `training/` directory.
They document the training settings and are not required for analysing images.
The scripts retain the original local paths and Apple MPS settings, which must
be changed before training on another computer. See `training/README.md` for
details.

## Model evaluation

The models were evaluated once using held-out data that were not used for
training or model selection. The detector test set contained 58 images with
2,111 stomata and 2,765 hair cells. The classifier test set contained 657
crops, and the segmentation test set contained 69 annotated stomatal crops
from eight slides.

| Model and class | Precision | Recall | F1 | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| Detection: stoma | 0.9615 | 0.9725 | 0.9670 | 0.9916 | 0.7510 |
| Detection: hair cell | 0.8928 | 0.8864 | 0.8896 | 0.9517 | 0.6824 |
| Segmentation: guard cell mask | 0.9329 | 0.9068 | — | 0.9699 | 0.6305 |
| Segmentation: subsidiary cell mask | 0.8848 | 0.8768 | — | 0.9419 | 0.5698 |

The crop classifier had a balanced accuracy of 0.7208 and a macro-F1 score of
0.7123. Class-specific F1 scores were 0.8411 for full-good crops, 0.5744 for
guard-only crops and 0.7214 for rejected crops. Further details of dataset
construction and model evaluation will be provided in the accompanying
manuscript.

## Versions

- BdStomatrix software release: `1.1.0`
- Canonical readable single-file extractor: `1.0.0`
- Legacy embedded extractor bundle: `1.1.0`
- Trait-calculation method: `2.5.4`

The software version identifies the GitHub release. The trait-calculation
method has a separate version so that changes in biological measurements can
be distinguished from changes in installation or file organisation.

## What the pipeline does

- Accepts one TIFF file or one directory through the same command.
- Uses strict reference-calibration checks by default. Valid calibration from
  individual OME files can be used instead, with any mismatch recorded.
- Runs on CPU, Apple MPS or NVIDIA CUDA.
- Expects the tested image size of 3072 x 2048 pixels by default; other image
  dimensions can be supplied through command-line options.
- Produces annotated images for checking stomatal rows and bands.
- Removes detector overview JPEGs after successful trait extraction unless
  the user chooses to keep them.
- Keeps the detection records, compressed masks, masked crops and provenance
  files needed to check and reproduce the analysis.

## Reference for spatial traits

Some spatial stomatal-patterning traits were informed by Hodge and Leakey
(2026); the remaining traits were developed independently for BdStomatrix.

Hodge, J. G., & Leakey, A. D. B. (2026). Spatial analysis of cell patterning
to aid genetic and phenotypic understanding of grass stomatal density: A case
study in maize. *The Plant Phenome Journal*, 9, e70066.
https://doi.org/10.1002/ppj2.70066.

## Supported input

- File extension: `.ome.tif`, `.ome.tiff`, `.tif` or `.tiff`.
- Default image dimensions: 3072 x 2048 pixels. Other dimensions must be
  supplied explicitly and should be validated before analysis.
- Image type: unsigned 8-bit grayscale.
- Filename begins with `S001_AB_P1_...` or `S001_P1_AB_...`.
- OME metadata contains valid positive X/Y physical pixel dimensions.
- Strict mode also requires isotropic metadata matching the reference value of
  4.5848 pixels/µm within 0.001 pixels/µm.

## Run one image

```bash
python -B run_bdstomatrix.py \
--input "/full/path/to/S001_AB_P1_EDF.ome.tif" \
--output-dir "/full/path/to/S001_AB_P1_BdStomatrix_final" \
--device mps
```

## Run a directory

```bash
python -B run_bdstomatrix.py \
--input "/full/path/to/ome_tiff_directory" \
--output-dir "/full/path/to/cohort_BdStomatrix_final" \
--device mps
```

Add `--recursive` only when nested directories should be searched. Reduce
`--classifier-batch-size` from 32 to 16 or 8 if GPU memory is limited.
Use `--device cpu`, `--device mps`, or `--device cuda --cuda-device 0`.

`--keep-inference-overviews` retains detector overview JPEGs. These are
removed by default because the spatial row/band audit overlays are retained.

## Example images for a test run

The `examples_for_run/` directory contains one abaxial image
(`S047_AB_P3_EDF.ome.tif`) and one adaxial image
(`S047_AD_P5_EDF.ome.tif`). They are the source images used for the example
outputs shown below.

From inside the BdStomatrix directory, run both example images with:

```bash
python -B run_bdstomatrix.py \
--input "examples_for_run" \
--output-dir "example_run_output" \
--device cpu
```

The output directory must not already exist.

## Non-default resolution

Both dimensions must describe the actual input images:

```bash
python run_bdstomatrix.py \
--input "/full/path/to/images" \
--output-dir "/full/path/to/new_output" \
--expected-image-width 4096 \
--expected-image-height 3072 \
--device cpu
```

Changing these values does not retrain the models. Check detections,
segmentations and spatial overlays before using traits from images with a
different resolution.

## Calibration policies

The default setting is:

```text
--calibration-policy strict-reference
```

It requires valid isotropic OME metadata matching 4.5848 pixels/µm within the
0.001 tolerance. Trait calculations use the actual metadata values after the
image passes this check.

To use the calibration recorded in each image when it differs from the
reference:

```text
--calibration-policy use-image-metadata
```

This setting accepts positive X/Y calibration values and uses them separately
when calculating calibrated traits. The output records the calibration values,
their differences from the reference and either
`metadata_used_reference_mismatch` or
`metadata_used_anisotropic_calibration`. Missing or invalid calibration still
stops the pipeline.

## Run trait extraction alone

```bash
python -B trait_extractor.py \
--pipeline-output-dir "/full/path/to/inference_support" \
--output-dir "/full/path/to/new_traits_v2_5_4"
```

The trait output must be new and outside the inference directory. This command
uses the canonical readable single-file implementation.

## Output layout

```text
<output-root>/
├── inference_support/
│   ├── input_image_manifest.csv
│   ├── stoma_segmentation_manifest.csv
│   ├── hair_cell_detection_manifest.csv
│   ├── selected_mask_instances.csv
│   ├── image_pipeline_summary.csv
│   ├── masks/
│   ├── masked_crops/
│   └── pipeline_provenance.json
├── traits_v2_5_4/
│   ├── stoma_traits.csv
│   ├── image_traits.csv
│   ├── image_gwas_traits.csv
│   ├── cell_instance_traits_qc.csv
│   ├── row_traits_qc.csv
│   ├── band_traits_qc.csv
│   ├── spatial_audit_overlays/
│   └── trait_extraction_provenance.json
├── final_run_provenance.json
└── FINAL_RUN_REPORT.txt
```

## Example outputs

The following abaxial and adaxial examples are from the Conda test run. Their
filenames were changed for the repository, but the image content was not
altered.

### Spatial audit overlays

#### Abaxial surface

![Representative abaxial spatial audit overlay](examples/spatial_audit/example_AB_spatial_audit.png)

#### Adaxial surface

![Representative adaxial spatial audit overlay](examples/spatial_audit/example_AD_spatial_audit.png)

### Masked stomatal-complex crops

Each crop below passed the preliminary complete-complex checks and contains
two selected guard-cell masks and two selected subsidiary-cell masks.

#### Abaxial masked crops

| Stoma 0011 | Stoma 0019 |
| --- | --- |
| ![Abaxial stoma 0011 masked stomatal-complex crop](examples/masked_crops/example_AB_masked_crop_01.png) | ![Abaxial stoma 0019 masked stomatal-complex crop](examples/masked_crops/example_AB_masked_crop_02.png) |

#### Adaxial masked crops

| Stoma 0008 | Stoma 0017 |
| --- | --- |
| ![Adaxial stoma 0008 masked stomatal-complex crop](examples/masked_crops/example_AD_masked_crop_01.png) | ![Adaxial stoma 0017 masked stomatal-complex crop](examples/masked_crops/example_AD_masked_crop_02.png) |

### Spatial-overlay and mask colour key

The spatial audit overlay is placed on the detector output, so it shows both
detection boxes and row/band annotations. Colours should be interpreted with
the symbol and adjacent text because some colours are used for more than one
type of annotation. The key includes all annotations that v2.5.4 can produce,
including annotations not present in these examples.

#### Detection and segmentation boxes

| Colour and mark | Meaning |
| --- | --- |
| Yellow box with an `H`-prefixed label | Detected hair cell. |
| Green numbered box | Stoma with guard-cell and subsidiary-cell masks that passed the initial checks. |
| Purple numbered box | Two guard-cell and two subsidiary-cell masks were found, but the result failed a geometry, target-association or border check. |
| Blue numbered box | Two guard-cell masks were found, but subsidiary-cell segmentation was incomplete. |
| Orange numbered box | Stoma selected for segmentation but not assigned one of the preceding results. |
| Grey numbered box | Detected stoma not selected for segmentation. |

The number identifies the stoma within the source image; it is not a class
label.

#### Rows, bands and orientation

| Colour and mark | Meaning |
| --- | --- |
| Thin coloured longitudinal line, matching centre dot and `Bxxx/Rxxx` label | Inferred row and its band assignment. All rows and stomata in one band share a colour. The eight-colour palette cycles through green, blue, red, purple, yellow-gold, cyan, pink and olive; colour alone is not a QC state and repeats if more than eight bands are present. |
| Grey centre dot | A stoma without an assigned band in the rendered records. |
| Black arrow labelled `longitudinal axis` | Estimated longitudinal leaf axis used for row geometry. A white under-stroke may be visible for contrast. |

#### QC and topology annotations

| Colour and mark | Meaning |
| --- | --- |
| Red X or red exclusion/loss text | Excluded partial-field band, lateral half-cut, sparse lateral edge, longitudinal-edge fragment, multirow lateral visibility loss, or an edge-touching singleton explicitly excluded from row designation. The adjacent text gives the exact reason. |
| Orange circles with `TRAJECTORY QC` | A spatially retained band for which exact trajectory-based row/band traits are uncertain. |
| Purple rectangles with `LONGITUDINAL QC` | A retained row-structure band with incomplete longitudinal sampling. |
| Gold circles with `LATERAL EDGE QC` | A retained band with an uncertain lateral-edge assessment. |
| Cyan cross-in-circle singleton mark | Strict accepted singleton row. |
| Orange cross-in-circle singleton mark | Strict singleton band; orange also denotes an ambiguous v2.5 singleton when the label says so. |
| Magenta singleton mark or text | Membership-relaxed rescued singleton or a v2.5 possible-singleton candidate. Magenta `V2.4 REATTACHED` text denotes a band reattachment. |
| White singleton mark with black outline | Leave-one-out, bracketed or terminal singleton rescue. |
| Yellow singleton mark with black outline | v2.4 expanded singleton row. |
| Green singleton mark | v2.5 supported singleton. Green `COMPLETE-SUPPORT RETAIN` text indicates that adequate complete-band support overrode an edge veto. |
| White `BAND SPLIT` or `MERGED CONTINUATION` text | A topology split or continuation merge accepted by the inherited v2.4/v2.5 rules. |
| Orange-red `EDGE LOSS CONFIRMED` text | Multirow lateral edge loss was detected but the final exclusion condition was not met. |
| Black outlines or text shadows | Contrast aids only; they do not encode an additional biological or QC class. |

#### Masked-crop colours

| Colour and mark | Meaning |
| --- | --- |
| Cyan-blue translucent masks and contours | Selected guard cells. |
| Pink translucent masks and contours | Selected subsidiary cells. |
| Yellow cross | Target detector-box centre within the crop. |

Colours are display annotations only; measurements are calculated from the
stored masks and calibrated coordinates, not from the rendered PNG colours.

## Installation

Create the Conda environment from inside the BdStomatrix directory:

```bash
conda env create --file environment.yaml
conda activate bdstomatrix
```

Verify the environment before processing images:

```bash
python -B run_bdstomatrix.py --help
python -B run_bdstomatrix.py --version
python -B trait_extractor.py --help
```

To update an existing environment after a future YAML change:

```bash
conda env update --file environment.yaml --prune
conda activate bdstomatrix
```

The YAML file creates a Python 3.11 environment and installs the required
packages. Additional Python packages required by Ultralytics are installed
automatically.

Alternatively, install into an already activated Python 3.11 environment:

```bash
python -m pip install -r requirements.txt
```

See `DEPENDENCIES.md` for the tested package versions and checksums for the
pipeline components and model files.

The YAML file cannot select the correct CUDA-enabled PyTorch installation for
every NVIDIA driver and operating system. CUDA users should first create the
environment and then install the appropriate PyTorch and torchvision versions
using the command provided by the official PyTorch installation selector. See
`PLATFORM_GUIDE.md`.

## Verify the release

From inside the release directory:

```bash
shasum -a 256 -c SHA256SUMS.txt
python -B run_bdstomatrix.py --help
python -B trait_extractor.py --help
```

## Citation and repository metadata

`CITATION.cff` provides citation information for GitHub, `.zenodo.json`
provides the Zenodo metadata override, and `pyproject.toml` records the Python
project metadata and direct dependencies. The reproducible runtime environment
is still created from `environment.yaml`; BdStomatrix is distributed as a
source-and-model bundle rather than as a Python wheel.

## Licence

The detector, classifier and segmentation model files are included.

Copyright (C) 2026 University of Bern and contributors. BdStomatrix source code
and the included trained model checkpoints are distributed under the GNU
Affero General Public License, version 3 (`AGPL-3.0-only`), subject to the
ownership qualification in `LICENSE_NOTICE.md`. See `LICENSE` for the complete
terms and `LICENSE_NOTICE.md` for third-party dependency licences and
model-weight details.
