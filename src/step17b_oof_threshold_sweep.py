from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, confusion_matrix

def pick_threshold_max_recall_at_precision(y_true, y_prob, target_precision):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    p_thr, r_thr = p[:-1], r[:-1]
    ok = np.where(p_thr >= target_precision)[0]
    if len(ok) == 0:
        return None
    i = ok[np.argmax(r_thr[ok])]
    return float(thr[i]), float(p_thr[i]), float(r_thr[i])

def main():
    repo_root = Path(__file__).resolve().parents[1]
    reports_dir = repo_root / "reports"
    path = reports_dir / "step17_oof_predictions.csv"

    df = pd.read_csv(path)
    y = df["NR-AR"].astype(int).values
    prob = df["oof_prob"].astype(float).values

    targets = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
    rows = []
    for P in targets:
        res = pick_threshold_max_recall_at_precision(y, prob, P)
        if res is None:
            rows.append({"target_precision": P, "threshold": None, "precision": None, "recall": None,
                         "tp": None, "fp": None, "tn": None, "fn": None})
            continue

        t, prec, rec = res
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({"target_precision": P, "threshold": t, "precision": prec, "recall": rec,
                     "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})

    out = pd.DataFrame(rows)
    print(out)

if __name__ == "__main__":
    main()
