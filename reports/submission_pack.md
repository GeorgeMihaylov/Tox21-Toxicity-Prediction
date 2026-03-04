# Tox21 NR-AR submission pack

## Model artifacts
- Scaler: models\nr_ar_scaler.joblib
- Model: models\nr_ar_xgb.json

## Threshold-free metrics (OOF)
- ROC-AUC: 0.779037
- Average Precision (AP): 0.473834

## Operating point (OOF-optimized)
- Optimize: f1
- Threshold: 0.669348
- Precision / Recall / F1: 0.8161 / 0.4610 / 0.5892
- MCC: 0.6015
- Balanced Acc: 0.7282
- Confusion (TP, FP, TN, FN): 142, 32, 6918, 166

## Predictions
- File: reports\step19_predictions_with_best_threshold.csv
- Predicted positives: 190 / 7258 (2.6178%)
