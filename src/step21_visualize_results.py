from pathlib import Path
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def plot_pr_curve(y, prob, threshold, out_path: Path):
    p, r, thr = precision_recall_curve(y, prob)
    ap = average_precision_score(y, prob) if len(np.unique(y)) > 1 else float("nan")

    # точка threshold на PR: найдём ближайший threshold
    thr = thr.astype(float)
    idx = int(np.argmin(np.abs(thr - threshold)))
    p_at = float(p[:-1][idx])
    r_at = float(r[:-1][idx])

    plt.figure(figsize=(6, 5))
    plt.plot(r, p, linewidth=2)
    plt.scatter([r_at], [p_at], s=70)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve (AP={ap:.4f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return {"ap": float(ap), "p_at": p_at, "r_at": r_at, "thr_nearest": float(thr[idx])}


def plot_roc_curve(y, prob, out_path: Path):
    fpr, tpr, _ = roc_curve(y, prob)
    roc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC={roc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return {"roc_auc": float(roc)}


def plot_confusion(y, prob, threshold, out_path: Path):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    cm = np.array([[tn, fp], [fn, tp]], dtype=int)

    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", fontsize=12)

    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title(f"Confusion matrix (t={threshold:.4f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision), "recall": float(recall), "f1": float(f1)
    }


def plot_prob_hist(y, prob, threshold, out_path: Path):
    prob0 = prob[y == 0]
    prob1 = prob[y == 1]

    plt.figure(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 40)
    plt.hist(prob0, bins=bins, alpha=0.6, label=f"Class 0 (n={len(prob0)})")
    plt.hist(prob1, bins=bins, alpha=0.6, label=f"Class 1 (n={len(prob1)})")
    plt.axvline(threshold, linestyle="--", linewidth=2, label=f"threshold={threshold:.4f}")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("OOF probability histogram")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    repo_root = Path(__file__).resolve().parents[1]
    reports_dir = repo_root / "reports"
    fig_dir = reports_dir / "figures"
    ensure_dir(fig_dir)

    oof_path = reports_dir / "step17_oof_predictions.csv"
    op_path = reports_dir / "step18_submission_operating_point.json"

    if not oof_path.exists():
        raise FileNotFoundError(f"Missing: {oof_path}")
    if not op_path.exists():
        raise FileNotFoundError(f"Missing: {op_path}")

    df = pd.read_csv(oof_path)
    with open(op_path, "r", encoding="utf-8") as f:
        op = json.load(f)

    y = df["NR-AR"].astype(int).values
    prob = df["oof_prob"].astype(float).values
    threshold = float(op["best_threshold"])

    pr_info = plot_pr_curve(y, prob, threshold, fig_dir / "step21_pr_curve.png")
    roc_info = plot_roc_curve(y, prob, fig_dir / "step21_roc_curve.png")
    cm_info = plot_confusion(y, prob, threshold, fig_dir / "step21_confusion_matrix.png")
    plot_prob_hist(y, prob, threshold, fig_dir / "step21_prob_hist.png")

    # Сохраним небольшой json-лог, чтобы удобно вставить в отчёт
    out = {
        "threshold": threshold,
        "pr_point_nearest": pr_info,
        "roc": roc_info,
        "confusion_at_threshold": cm_info,
        "files": {
            "pr_curve": str((fig_dir / "step21_pr_curve.png").relative_to(repo_root)),
            "roc_curve": str((fig_dir / "step21_roc_curve.png").relative_to(repo_root)),
            "confusion_matrix": str((fig_dir / "step21_confusion_matrix.png").relative_to(repo_root)),
            "prob_hist": str((fig_dir / "step21_prob_hist.png").relative_to(repo_root)),
        }
    }

    out_path = reports_dir / "step21_viz_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved figures to:", fig_dir)
    print("Saved:", out_path)
    print("Threshold:", threshold)
    print("OOF @ threshold:", cm_info)


if __name__ == "__main__":
    main()
