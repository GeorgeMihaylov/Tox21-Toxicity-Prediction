from pathlib import Path
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)

import xgboost as xgb


def scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return ""
    return Chem.MolToSmiles(scaf, isomericSmiles=False)


def best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1 = np.zeros_like(p)
    denom = p + r
    m = denom > 0
    f1[m] = 2 * p[m] * r[m] / denom[m]
    if len(thr) == 0:
        return 0.5
    return float(thr[int(np.argmax(f1[:-1]))])


def prf1_from_cm(tn, fp, fn, tp):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    reports_dir = repo_root / "reports"
    fig_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    feat_path = data_dir / "tox21_nr_ar_features_adv.csv"
    clean_path = data_dir / "tox21_nr_ar_cleaned.csv"
    best_path = reports_dir / "xgb_best_params_ap_scaffoldcv.json"

    feat = pd.read_csv(feat_path)
    clean = pd.read_csv(clean_path)[["mol_id", "smiles"]]
    with open(best_path, "r", encoding="utf-8") as f:
        best = json.load(f)
    tuned_params = best["params"]

    df = feat.merge(clean, on="mol_id", how="inner").dropna(subset=["smiles", "NR-AR"]).copy()
    df["NR-AR"] = df["NR-AR"].astype(int)
    df["scaffold"] = df["smiles"].astype(str).map(scaffold_smiles)

    X = df.drop(columns=["mol_id", "smiles", "scaffold", "NR-AR"]).values
    y = df["NR-AR"].values
    groups = df["scaffold"].values

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    thresholds = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        X_train, y_train = X[tr_idx], y[tr_idx]
        X_test, y_test = X[te_idx], y[te_idx]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        ratio = float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum()))

        model = xgb.XGBClassifier(
            eval_metric="aucpr",
            scale_pos_weight=ratio,
            random_state=42,
            n_jobs=-1,
            **tuned_params,
        )
        model.fit(X_tr_s, y_tr)

        val_prob = model.predict_proba(X_val_s)[:, 1]
        t = best_threshold_by_f1(y_val, val_prob)
        thresholds.append(t)

        model.fit(X_train_s, y_train)
        test_prob = model.predict_proba(X_test_s)[:, 1]

        roc = roc_auc_score(y_test, test_prob) if len(np.unique(y_test)) > 1 else float("nan")
        ap = average_precision_score(y_test, test_prob) if len(np.unique(y_test)) > 1 else float("nan")

        y_pred = (test_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        prec, rec, f1 = prf1_from_cm(tn, fp, fn, tp)

        row = {
            "fold": fold,
            "roc_auc": float(roc),
            "average_precision": float(ap),
            "threshold": float(t),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "pos_rate_test": float(y_test.mean()),
        }
        rows.append(row)
        print(row)

    out = pd.DataFrame(rows)
    out_path = reports_dir / "scaffold_cv_threshold_metrics_tuned.csv"
    out.to_csv(out_path, index=False)

    plt.figure(figsize=(7, 4))
    plt.hist(thresholds, bins=15)
    plt.xlabel("Selected threshold (per fold)")
    plt.ylabel("Count")
    plt.title("Threshold distribution (tuned XGBoost)")
    plt.tight_layout()
    plt.savefig(fig_dir / "threshold_distribution_scaffold_cv_tuned.png", dpi=300)
    plt.close()

    print("Saved", out_path)
    print("Mean ROC-AUC:", float(out["roc_auc"].mean()))
    print("Mean AP:", float(out["average_precision"].mean()))
    print("Mean F1:", float(out["f1"].mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
