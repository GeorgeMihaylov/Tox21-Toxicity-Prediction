from pathlib import Path
import json
import pandas as pd


def main():
    repo_root = Path(__file__).resolve().parents[1]
    reports_dir = repo_root / "reports"
    models_dir = repo_root / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)

    step17_path = reports_dir / "step17_final_operating_point.json"
    step18_path = reports_dir / "step18_submission_operating_point.json"
    preds_path = reports_dir / "step19_predictions_with_best_threshold.csv"

    scaler_path = models_dir / "nr_ar_scaler.joblib"
    model_path = models_dir / "nr_ar_xgb.json"

    for p in [step17_path, step18_path, preds_path, scaler_path, model_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    with open(step17_path, "r", encoding="utf-8") as f:
        s17 = json.load(f)
    with open(step18_path, "r", encoding="utf-8") as f:
        s18 = json.load(f)

    preds = pd.read_csv(preds_path)
    n_total = int(len(preds))
    n_pos = int((preds["nr_ar_pred"] == 1).sum())
    pos_rate_pred = float(n_pos / n_total) if n_total else 0.0

    pack = {
        "task": "NR-AR",
        "feature_file": s17.get("feature_file"),
        "model_artifacts": {
            "scaler": str(scaler_path.relative_to(repo_root)),
            "xgb_model": str(model_path.relative_to(repo_root)),
        },
        "operating_point_for_submission": {
            "opt_metric": s18["opt_metric"],
            "best_threshold": s18["best_threshold"],
            "oof_precision": s18["precision"],
            "oof_recall": s18["recall"],
            "oof_f1": s18["f1"],
            "oof_mcc": s18["mcc"],
            "oof_balanced_accuracy": s18["balanced_accuracy"],
            "oof_confusion": s18["confusion"],
        },
        "threshold_free_quality": {
            "oof_roc_auc": s18["oof_roc_auc"],
            "oof_average_precision": s18["oof_average_precision"],
        },
        "predictions_file": str(preds_path.relative_to(repo_root)),
        "predictions_summary": {
            "n_total": n_total,
            "n_predicted_positive": n_pos,
            "predicted_positive_rate": pos_rate_pred,
        },
        "notes": {
            "cv": "StratifiedGroupKFold by Murcko scaffold; OOF probs used for threshold selection",
            "why_oof": "Single global threshold selected on OOF to avoid per-fold thresholds",
        }
    }

    out_json = reports_dir / "submission_pack.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)

    out_md = reports_dir / "submission_pack.md"
    md = f"""# Tox21 NR-AR submission pack

## Model artifacts
- Scaler: {pack["model_artifacts"]["scaler"]}
- Model: {pack["model_artifacts"]["xgb_model"]}

## Threshold-free metrics (OOF)
- ROC-AUC: {pack["threshold_free_quality"]["oof_roc_auc"]:.6f}
- Average Precision (AP): {pack["threshold_free_quality"]["oof_average_precision"]:.6f}

## Operating point (OOF-optimized)
- Optimize: {pack["operating_point_for_submission"]["opt_metric"]}
- Threshold: {pack["operating_point_for_submission"]["best_threshold"]:.6f}
- Precision / Recall / F1: {pack["operating_point_for_submission"]["oof_precision"]:.4f} / {pack["operating_point_for_submission"]["oof_recall"]:.4f} / {pack["operating_point_for_submission"]["oof_f1"]:.4f}
- MCC: {pack["operating_point_for_submission"]["oof_mcc"]:.4f}
- Balanced Acc: {pack["operating_point_for_submission"]["oof_balanced_accuracy"]:.4f}
- Confusion (TP, FP, TN, FN): {pack["operating_point_for_submission"]["oof_confusion"]["tp"]}, {pack["operating_point_for_submission"]["oof_confusion"]["fp"]}, {pack["operating_point_for_submission"]["oof_confusion"]["tn"]}, {pack["operating_point_for_submission"]["oof_confusion"]["fn"]}

## Predictions
- File: {pack["predictions_file"]}
- Predicted positives: {n_pos} / {n_total} ({pos_rate_pred:.4%})
"""
    out_md.write_text(md, encoding="utf-8")

    print("Saved:", out_json)
    print("Saved:", out_md)


if __name__ == "__main__":
    main()
