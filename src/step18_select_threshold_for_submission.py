from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def fbeta(precision, recall, beta: float):
    beta2 = beta * beta
    num = (1 + beta2) * precision * recall
    den = (beta2 * precision) + recall
    return np.divide(num, den, out=np.zeros_like(den), where=(den != 0))


def metrics_from_cm(tn, fp, fn, tp):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_acc = 0.5 * (recall + specificity)

    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt(denom) if denom else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_acc),
        "mcc": float(mcc),
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    reports_dir = repo_root / "reports"
    models_dir = repo_root / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    oof_path = reports_dir / "step17_oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(
            f"Not found: {oof_path}\n"
            f"Run step17 with saving OOF predictions first."
        )

    df = pd.read_csv(oof_path)
    if not {"NR-AR", "oof_prob"}.issubset(df.columns):
        raise ValueError(f"Expected columns: NR-AR, oof_prob. Got: {list(df.columns)}")

    y = df["NR-AR"].astype(int).values
    prob = df["oof_prob"].astype(float).values

    # === what we optimize for submission ===
    OPT_METRIC = "f1"   # options: "f1", "f0.5", "f2"
    beta = 1.0
    if OPT_METRIC == "f0.5":
        beta = 0.5
    elif OPT_METRIC == "f2":
        beta = 2.0

    # PR curve arrays: precision/recall are len(thr)+1, so align with thr using [:-1]
    p, r, thr = precision_recall_curve(y, prob)
    p_thr, r_thr = p[:-1], r[:-1]

    scores = fbeta(p_thr, r_thr, beta=beta)
    best_i = int(np.argmax(scores))
    best_thr = float(thr[best_i])

    # Build sweep table (for report/debug)
    pred_all = (prob[:, None] >= thr[None, :]).astype(np.int8)  # shape [n_samples, n_thr]
    # Compute TP/FP/FN/TN for each threshold efficiently
    # Positive class is 1
    y_col = y[:, None].astype(np.int8)
    tp = (pred_all & (y_col == 1)).sum(axis=0)
    fp = (pred_all & (y_col == 0)).sum(axis=0)
    fn = ((1 - pred_all) & (y_col == 1)).sum(axis=0)
    tn = ((1 - pred_all) & (y_col == 0)).sum(axis=0)

    precision_s = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=((tp + fp) != 0))
    recall_s = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=((tp + fn) != 0))
    f1_s = np.divide(2 * precision_s * recall_s, precision_s + recall_s,
                     out=np.zeros_like(precision_s, dtype=float), where=((precision_s + recall_s) != 0))
    fbeta_s = fbeta(precision_s, recall_s, beta=beta)

    sweep = pd.DataFrame({
        "threshold": thr.astype(float),
        "precision": precision_s.astype(float),
        "recall": recall_s.astype(float),
        "f1": f1_s.astype(float),
        f"f{beta:g}": fbeta_s.astype(float),
        "tp": tp.astype(int),
        "fp": fp.astype(int),
        "tn": tn.astype(int),
        "fn": fn.astype(int),
    }).sort_values(f"f{beta:g}", ascending=False)

    sweep_path = reports_dir / "step18_oof_threshold_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    # Best threshold metrics
    y_pred_best = (prob >= best_thr).astype(int)
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y, y_pred_best, labels=[0, 1]).ravel()
    best_metrics = metrics_from_cm(tn_b, fp_b, fn_b, tp_b)

    # Also compute threshold-free ranking metrics on OOF
    roc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    ap = average_precision_score(y, prob) if len(np.unique(y)) > 1 else float("nan")

    out = {
        "task": "NR-AR",
        "source_oof_file": oof_path.name,
        "opt_metric": OPT_METRIC,
        "beta": float(beta),
        "best_threshold": float(best_thr),
        "oof_roc_auc": float(roc),
        "oof_average_precision": float(ap),
        "confusion": {"tp": int(tp_b), "fp": int(fp_b), "tn": int(tn_b), "fn": int(fn_b)},
        **best_metrics,
    }

    out_path = reports_dir / "step18_submission_operating_point.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved:", sweep_path)
    print("Saved:", out_path)
    print("Best threshold:", best_thr)
    print("Best metrics:", out)


if __name__ == "__main__":
    main()
