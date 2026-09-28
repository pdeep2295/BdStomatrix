# Dependencies

## Runtime Python files

Only these three Python files are required at runtime:

1. `run_bdstomatrix.py`
2. `run_ome_to_final_segmentation_pipeline.py`
3. `trait_extractor.py`

`trait_extractor.py` contains and verifies 16 embedded audited extractor
sources. The former versioned extractor files are not external runtime
dependencies in BdStomatrix final.

## Tested environment

| Component | Version |
|---|---:|
| Python | 3.11.14 |
| NumPy | 1.26.4 |
| OpenCV runtime (`cv2`) | 4.11.0 |
| Pillow | 12.1.0 |
| tifffile | 2026.1.14 |
| PyTorch | 2.10.0 |
| torchvision | 0.25.0 |
| Ultralytics | 8.4.6 |

CPU and MPS use the tested environment above. CUDA execution requires a
CUDA-enabled PyTorch/torchvision build compatible with the target NVIDIA
driver; consult `PLATFORM_GUIDE.md`.

`environment.yaml` is the recommended reproducible installation entry point.
It pins the direct runtime dependencies while allowing pip to resolve the
transitive packages required by Ultralytics. `requirements.txt` contains the
same direct Python dependency set for users who do not use Conda.

The validated source environment contains metadata for two OpenCV wheel
families, while its active `cv2` runtime is 4.11.0. For clean deployment,
`requirements.txt` specifies a single `opencv-python==4.11.0.86` package.
Avoid installing multiple OpenCV wheel families in a new environment.

## Model checkpoints

| Model | Relative path | SHA-256 |
|---|---|---|
| Detector | `detector/model_training_record/best.pt` | `6fa85a2f895f79777b188a69c9b384b730da3feb6956f4bbd2beb626ad4d2862` |
| Crop-quality classifier | `classifier/model_training_record/best.pt` | `e253dae9064c4a6ab1f84ac0aee204ae604c4dcf2afddd38f6818d3d238a249d` |
| Instance segmentor | `segmentor/model_training_record/best.pt` | `d52c91a2f7aab9341aae2d964ef492dbe2186ab0d09a354b5cc63347a058e9b4` |

The directory structure is part of the runtime interface. Checkpoint hashes
are verified before inference.

See `LICENSE_NOTICE.md` for the declared licences of these separately
installed dependencies and for the licensing scope of the included model
checkpoints.

## Locked Python identities

| File | SHA-256 |
|---|---|
| `run_ome_to_final_segmentation_pipeline.py` | `a41a50a6e82d8f40698c9fadd2ff36c659b36e60dd8f5e0f871329232b7fc98c` |
| `trait_extractor.py` | `87121bb33b4fab3de1dd66263b14b9b7462182a10afd42be178fa54cde47def3` |

The entry point also has a release checksum in `SHA256SUMS.txt` and records
its own identity in each completed run.
