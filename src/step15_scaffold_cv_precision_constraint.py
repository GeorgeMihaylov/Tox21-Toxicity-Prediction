from pathlib import Path
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


def threshold_max_recall_at_precision(y_true, y_prob, target_precision=0.90, fallback="max_f1"):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    p_thr = p[:-1]
    r_thr = r[:-1]

    ok = np.where(p_thr >= target_precision)[0]
    if len(ok) > 0:
        i = ok[np.argmax(r_thr[ok])]
        return float(thr[i]), float(p_thr[i]), float(r_thr[i]), "precision_constraint"

    if fallback == "high_precision":
        return 0.999, float(p[-1]), float(r[-1]), "fallback_high_precision"

    # fallback: max F1
    f1 = np.zeros_like(p_thr)
    denom = p_thr + r_thr
    m = denom > 0
    f1[m] = 2 * p_thr[m] * r_thr[m] / denom[m]
    j = int(np.argmax(f1)) if len(f1) else -1
    if j >= 0:
        return float(thr[j]), float(p_thr[j]), float(r_thr[j]), "fallback_max_f1"
    return 0.5, float(p[-1]), float(r[-1]), "fallback_default"


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

    # Вариант 1: битовые (как основной baseline)
    feat_path = data_dir / "tox21_nr_ar_features_adv.csv"
    # Вариант 2: count-фичи (если хочешь прогнать их):
    # feat_path = data_dir / "tox21_nr_ar_features_counts.csv"

    clean_path = data_dir / "tox21_nr_ar_cleaned.csv"

    feat = pd.read_csv(feat_path)
    clean = pd.read_csv(clean_path)[["mol_id", "smiles"]]

    df = feat.merge(clean, on="mol_id", how="inner").dropna(subset=["smiles", "NR-AR"]).copy()
    df["NR-AR"] = df["NR-AR"].astype(int)
    df["scaffold"] = df["smiles"].astype(str).map(scaffold_smiles)

    X = df.drop(columns=["mol_id", "smiles", "scaffold", "NR-AR"]).values
    y = df["NR-AR"].values
    groups = df["scaffold"].values

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    target_precision = 0.90

    rows = []
    thresholds = []
    chosen_modes = []

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
            early_stopping_rounds=100,
            scale_pos_weight=ratio,
            learning_rate=0.05,
            n_estimators=4000,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)

        val_prob = model.predict_proba(X_val_s)[:, 1]
        t, p_val, r_val, mode = threshold_max_recall_at_precision(
            y_val, val_prob, target_precision=target_precision, fallback="max_f1"
        )
        thresholds.append(t)
        chosen_modes.append(mode)

        model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)

        test_prob = model.predict_proba(X_test_s)[:, 1]
        roc = roc_auc_score(y_test, test_prob) if len(np.unique(y_test)) > 1 else float("nan")
        ap = average_precision_score(y_test, test_prob) if len(np.unique(y_test)) > 1 else float("nan")

        y_pred = (test_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        prec, rec, f1 = prf1_from_cm(tn, fp, fn, tp)

        row = {
            "fold": fold,
            "target_precision": target_precision,
            "threshold": float(t),
            "threshold_mode": mode,
            "val_precision_at_t": float(p_val),
            "val_recall_at_t": float(r_val),
            "roc_auc": float(roc),
            "average_precision": float(ap),
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
    out_path = reports_dir / "scaffold_cv_precision90_maxrecall.csv"
    out.to_csv(out_path, index=False)

    plt.figure(figsize=(7, 4))
    plt.hist(thresholds, bins=15)
    plt.xlabel("Selected threshold (per fold)")
    plt.ylabel("Count")
    plt.title("Threshold distribution (Precision>=0.90, max recall)")
    plt.tight_layout()
    plt.savefig(fig_dir / "threshold_distribution_precision90.png", dpi=300)
    plt.close()

    print("Saved", out_path)
    print("Mean ROC-AUC:", float(out["roc_auc"].mean()))
    print("Mean AP:", float(out["average_precision"].mean()))
    print("Mean Precision:", float(out["precision"].mean()))
    print("Mean Recall:", float(out["recall"].mean()))
    print("Mean F1:", float(out["f1"].mean()))
    print("Threshold modes:", dict(out["threshold_mode"].value_counts()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
