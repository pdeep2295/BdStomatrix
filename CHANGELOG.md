# Changelog

## BdStomatrix v1.1.0

- Added configurable expected image width and height while preserving the
  validated 3072 x 2048 defaults.
- Added strict-reference and opt-in per-image metadata calibration policies.
- Added indexed NVIDIA CUDA device support for Linux and Windows.
- Preserved CPU and Apple MPS execution without silent device fallback.
- Added explicit calibration mismatch, resolution and runtime-device
  provenance.
- Added a 40-image compiled-versus-modular v2.5.4 regression requirement.
- Added a portable Conda `environment.yaml` and documented environment creation,
  activation, updating and CUDA-specific PyTorch installation.
- Licensed BdStomatrix under `AGPL-3.0-only` and documented third-party
  dependency and included-checkpoint licensing.
- Removed development-machine paths and aligned the application and
  single-file extractor bundle identities to v1.1.0.
- Simplified user-facing stage labels and removed the obsolete inference-stage
  message stating that trait extraction was not performed.

## BdStomatrix v1.0.0

- Added a single production command for file or directory input.
- Bundled the complete audited v2.5.4 extractor chain into one hash-verified
  `trait_extractor.py` runtime file.
- Preserved the locked inference module and three checkpoint identities.
- Removed duplicate batch-level copies of final image trait CSVs.
- Removed inference overview JPEGs after successful extraction by default.
- Retained spatial audit overlays, masks, manifests and provenance.
- Added release checksums and deployment/validation documentation.
