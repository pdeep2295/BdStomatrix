# Model training scripts

This directory contains copies of the scripts used to train the models included
with BdStomatrix.

- `detector/final_model.py` trained the final YOLO26n stomatal and hair-cell
  detector.
- `detector/modelrunningfromlastcheckpoint.py` resumed the detector after the
  original run was interrupted at epoch 58.
- `classifier/generate_classifier_crops_padding30.py` generated the final
  detector-centred classifier crops with 30 pixels of requested padding.
- `classifier/train_efficientnet_b0_crop_quality.py` trained the three-class
  EfficientNet-B0 crop-quality classifier.
- `segmentor/train_yolo26n_seg_ab_ad_combined_candidate_v1.py` trained the
  combined abaxial and adaxial YOLO26n instance-segmentation model.

These scripts are supplied to document model training and are not required to
run BdStomatrix. They retain the dataset checks, hyperparameters, random seeds
and model-selection rules used during training.

The original local dataset and output paths, together with Apple MPS device
settings, are retained in the scripts. Before using them on another computer,
change the paths and device settings and review the expected dataset structure,
split counts and checksum checks. The training datasets are not included here.
