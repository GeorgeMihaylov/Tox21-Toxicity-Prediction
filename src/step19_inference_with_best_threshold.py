from pathlib import Path
import json
import numpy as np
import pandas as pd

import joblib
import xgboost as xgb


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    reports_dir = repo_root / "reports"
    models_dir = repo_root / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Артефакты модели (из step17) + operating point (из step18)
    scaler_path = models_dir / "nr_ar_scaler.joblib"
    model_path = models_dir / "nr_ar_xgb.json"
    op_path = reports_dir / "step18_submission_operating_point.json"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not op_path.exists():
        raise FileNotFoundError(f"Missing operating point: {op_path}")

    with open(op_path, "r", encoding="utf-8") as f:
        op = json.load(f)
    threshold = float(op["best_threshold"])

    scaler = joblib.load(scaler_path)

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    # Данные для инференса (те же фичи, что и в обучении)
    feat_path = data_dir / "tox21_nr_ar_features_adv.csv"
    df = pd.read_csv(feat_path)

    if "mol_id" not in df.columns:
        raise ValueError("Expected 'mol_id' column in features file")

    # Берём все признаки кроме mol_id и таргета (если вдруг есть)
    drop_cols = [c for c in ["mol_id", "NR-AR"] if c in df.columns]
    X = df.drop(columns=drop_cols).values
    Xs = scaler.transform(X)

    prob = model.predict_proba(Xs)[:, 1]
    pred = (prob >= threshold).astype(int)

    out = pd.DataFrame({
        "mol_id": df["mol_id"].values,
        "nr_ar_prob": prob.astype(float),
        "nr_ar_pred": pred.astype(int),
    })

    out_path = reports_dir / "step19_predictions_with_best_threshold.csv"
    out.to_csv(out_path, index=False)

    print("Loaded threshold:", threshold)
    print("Saved:", out_path)
    print("Predicted positives:", int(out["nr_ar_pred"].sum()), "of", len(out))


if __name__ == "__main__":
    main()
