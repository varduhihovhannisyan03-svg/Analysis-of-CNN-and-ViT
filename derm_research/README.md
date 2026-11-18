# DERM12345 research v2 — Jupyter

1. Use Python `.venv` and install dependencies: `python -m pip install -r requirements.txt`.
2. Place extracted image directories and both CSVs under `data/DERM12345/`, or adjust `DATA_DIR` in `01_Preprocessing.ipynb` to your actual location.
3. Run `01_Preprocessing.ipynb` once to create `results/selected_four_classes.csv`. If this file already exists, it can be reused; verify its `path` entries resolve on your current machine.
4. Run each model in its **own** notebook: `02A_ResNet50`, `02B_DenseNet121`, `02C_EfficientNet_B7`, `02D_ViT_B16`.
5. Run `03_Comparison_and_Figures.ipynb` for figures and tables. Model-specific files go into `results_v2/<model_name>/`.

## Training method

- Training-only augmentation (flip, rotate, limited color jitter), fixed resize to 224x224, ImageNet normalization.
- Inverse square root class-weighted cross-entropy for all models.
- Head warmup (3 epochs by default) with backbone frozen; full-network fine-tuning (up to 8 epochs by default).
- AdamW; head LR 1e-3, fine-tune LR 2e-5; patience 4, checkpoint selected by validation **macro F1**.
- Test evaluated once per final run; per-class scores and predictions exported.
- All backbones pretrained; pretrained weight downloads require connectivity at least once.

## Notes

Do not interpret the short initial test runs (2 epochs) as publication-quality results. Test scores from the previous prototype are already known; avoid using them to repeatedly tune hyperparameters. For publication-grade comparison, predefine a fair training protocol and report uncertainty; only four Spitz/Reed test cases exist. This code uses the official patient-separated test data (subject to the preprocessing validation). CPU full fine-tuning of EfficientNet-B7 and ViT can be very slow; if it is impractical, be explicit about compute limitations, do not substitute partial runs as equivalent. **224px is a controlled equal-resolution experiment, not necessarily the optimal/native resolution of EfficientNet-B7**. A later sensitivity study can test a higher resolution.

For saving best checkpoints, note that the validation macro F1 may be noisy due to only four Spitz/Reed validation images. The code does not repeatedly inspect the test partition during the epochs.
