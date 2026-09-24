# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `team_share_25pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/no_lora/best_model_team_share_25pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `30.00%` (3/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00246_original_GT_obj251.jpg` | danger | danger | cut: 33.84% | danger: 61.92% | excluded: 4.24% | ✅ |
| 2 | `00100_original_GT_duplicate_obj103.jpg` | danger | danger | cut: 31.39% | danger: 49.71% | excluded: 18.90% | ✅ |
| 3 | `00327_xlsx_excluded.jpg` | excluded | cut | cut: 69.06% | danger: 20.60% | excluded: 10.35% | ❌ |
| 4 | `00001_original_GT_duplicate_obj2.jpg` | danger | cut | cut: 55.05% | danger: 33.05% | excluded: 11.89% | ❌ |
| 5 | `00110_original_GT_duplicate_obj109.jpg` | danger | cut | cut: 50.76% | danger: 44.34% | excluded: 4.90% | ❌ |
| 6 | `00319_xlsx_excluded.jpg` | excluded | danger | cut: 40.11% | danger: 52.94% | excluded: 6.95% | ❌ |
| 7 | `00178_original_GT_obj170.jpg` | danger | danger | cut: 24.92% | danger: 67.80% | excluded: 7.29% | ✅ |
| 8 | `00126_original_GT_obj128.jpg` | danger | cut | cut: 84.51% | danger: 13.59% | excluded: 1.90% | ❌ |
| 9 | `00102_original_GT_duplicate_obj104.jpg` | danger | cut | cut: 60.94% | danger: 17.01% | excluded: 22.05% | ❌ |
| 10 | `00042_original_GT_duplicate_obj49.jpg` | danger | cut | cut: 66.99% | danger: 26.11% | excluded: 6.90% | ❌ |
