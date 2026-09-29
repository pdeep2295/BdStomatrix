#Version policy

BdStomatrix final v1.1.0 freezes the detector, classifier, segmentor and v2.5.4 trait
logic used for the audited production workflow.

- Do not edit `trait_extractor.py` or replace embedded source payloads in place.
- Do not replace checkpoint files under the existing names.
- Do not disable component, embedded-source or checkpoint hash verification.
- Scientific changes require a new version, new checksums and new validation.
- Preserve AB and AD as separate surface-level phenotypes.
- Retain provenance, QC tables and spatial overlays with downstream datasets.
- Treat non-default dimensions, non-reference calibration and a new compute
  backend as explicitly flagged validation domains.

Unusual future image domains can expose unobserved failures. Record and
audit such cases rather than silently modifying this locked release.
