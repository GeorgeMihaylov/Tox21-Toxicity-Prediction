from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
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


def make_scaffold_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    scaffolds = df["scaffold"].fillna("").astype(str).values
    idx = np.arange(len(df))

    groups = {}
    for i, sc in zip(idx, scaffolds):
        groups.setdefault(sc, []).append(i)

    rng = np.random.RandomState(seed)
    unique_scaffolds = list(groups.keys())
    rng.shuffle(unique_scaffolds)
    unique_scaffolds.sort(key=lambda s: len(groups[s]), reverse=True)

    n_test_target = int(round(test_size * len(df)))
    test_idx = []
    for sc in unique_scaffolds:
        if len(test_idx) >= n_test_target:
            break
        test_idx.extend(groups[sc])

    test_idx = np.array(sorted(set(test_idx)), dtype=int)
    train_idx = np.array(sorted(set(idx) - set(test_idx)), dtype=int)
    return train_idx, test_idx


def threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1 = np.zeros_like(p)
    denom = p + r
    mask = denom > 0
    f1[mask] = 2 * p[mask] * r[mask] / denom[mask]
    if len(thr) == 0:
        return 0.5, 0.0
    best_i = int(np.argmax(f1[:-1]))
    return float(thr[best_i]), float(f1[best_i])


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    reports_dir = repo_root / "reports"
    fig_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    feat_path = data_dir / "tox21_nr_ar_features_adv.csv"
    clean_path = data_dir / "tox21_nr_ar_cleaned.csv"
    if not feat_path.exists():
        print(f"Missing: {feat_path}")
        return 2
    if not clean_path.exists():
        print(f"Missing: {clean_path}")
        return 3

    feat = pd.read_csv(feat_path)
    clean = pd.read_csv(clean_path)[["mol_id", "smiles"]]

    df = feat.merge(clean, on="mol_id", how="inner")
    df = df.dropna(subset=["smiles", "NR-AR"]).copy()
    df["NR-AR"] = df["NR-AR"].astype(int)

    df["scaffold"] = df["smiles"].astype(str).map(scaffold_smiles)

    train_idx, test_idx = make_scaffold_split(df, test_size=0.2, seed=42)
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    X_train = train_df.drop(columns=["mol_id", "smiles", "scaffold", "NR-AR"]).values
    y_train = train_df["NR-AR"].values
    X_test = test_df.drop(columns=["mol_id", "smiles", "scaffold", "NR-AR"]).values
    y_test = test_df["NR-AR"].values

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
        eval_metric="logloss",
        scale_pos_weight=ratio,
        learning_rate=0.05,
        n_estimators=600,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_tr_s, y_tr)

    val_prob = model.predict_proba(X_val_s)[:, 1]
    best_t, best_f1 = threshold_by_f1(y_val, val_prob)

    model.fit(X_train_s, y_train)
    test_prob = model.predict_proba(X_test_s)[:, 1]

    roc_auc = roc_auc_score(y_test, test_prob) if len(np.unique(y_test)) > 1 else float("nan")
    ap = average_precision_score(y_test, test_prob) if len(np.unique(y_test)) > 1 else float("nan")

    y_pred = (test_prob >= best_t).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "split": "scaffold",
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "pos_rate_total": float(df["NR-AR"].mean()),
        "pos_rate_train": float(train_df["NR-AR"].mean()),
        "pos_rate_test": float(test_df["NR-AR"].mean()),
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "threshold_selected_on_val": float(best_t),
        "val_best_f1": float(best_f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    with open(reports_dir / "scaffold_split_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    fpr, tpr, _ = roc_curve(y_test, test_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC (scaffold split): XGBoost")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve_scaffold_xgb.png", dpi=300)
    plt.close()

    prec, rec, _ = precision_recall_curve(y_test, test_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (scaffold split): XGBoost")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(fig_dir / "pr_curve_scaffold_xgb.png", dpi=300)
    plt.close()

    sizes = pd.Series(df["scaffold"].value_counts())
    plt.figure(figsize=(7, 4))
    plt.hist(np.clip(sizes.values, 1, 50), bins=50)
    plt.xlabel("Scaffold cluster size (clipped at 50)")
    plt.ylabel("Count")
    plt.title("Distribution of scaffold cluster sizes (NR-AR set)")
    plt.tight_layout()
    plt.savefig(fig_dir / "scaffold_cluster_sizes.png", dpi=300)
    plt.close()

    print("Saved reports/scaffold_split_metrics.json")
    print("Saved figures: roc_curve_scaffold_xgb.png, pr_curve_scaffold_xgb.png, scaffold_cluster_sizes.png")
    print(metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
