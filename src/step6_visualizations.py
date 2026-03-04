from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

import xgboost as xgb


TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"


def download_if_missing(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        f.write(r.read())


def compute_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
        "Rings": Descriptors.RingCount(mol),
    }


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray):
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    sns.set_style("whitegrid")

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    raw_dir = data_dir / "raw"
    reports_dir = repo_root / "reports"
    fig_dir = reports_dir / "figures"

    raw_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tox21_path = raw_dir / "tox21.csv.gz"
    download_if_missing(TOX21_URL, tox21_path)

    tox21 = pd.read_csv(tox21_path)
    task_cols = [c for c in tox21.columns if c.startswith("NR-") or c.startswith("SR-")]

    summary_rows = []
    n_total = len(tox21)
    for task in task_cols:
        s = tox21[task]
        n_labeled = int(s.notna().sum())
        n_missing = int(s.isna().sum())
        pos = int((s == 1).sum(skipna=True))
        neg = int((s == 0).sum(skipna=True))
        pos_rate = (pos / (pos + neg)) if (pos + neg) else np.nan
        missing_rate = n_missing / n_total
        summary_rows.append(
            {
                "task": task,
                "n_total": n_total,
                "n_labeled": n_labeled,
                "n_missing": n_missing,
                "missing_rate": float(missing_rate),
                "pos": pos,
                "neg": neg,
                "pos_rate": float(pos_rate),
            }
        )

    ds_summary = pd.DataFrame(summary_rows).sort_values("task")
    ds_summary_path = reports_dir / "tox21_dataset_summary.csv"
    ds_summary.to_csv(ds_summary_path, index=False)

    plt.figure(figsize=(10, 5))
    ds_sorted = ds_summary.sort_values("n_labeled", ascending=False)
    sns.barplot(data=ds_sorted, x="task", y="n_labeled", color="#4C72B0")
    plt.xticks(rotation=45, ha="right")
    plt.title("Tox21: количество размеченных молекул по задачам")
    plt.xlabel("Task")
    plt.ylabel("Labeled molecules")
    out = fig_dir / "tox21_labeled_counts_by_task.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    ds_sorted = ds_summary.sort_values("pos_rate", ascending=False)
    sns.barplot(data=ds_sorted, x="task", y="pos_rate", color="#DD8452")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, min(1.0, float(ds_summary["pos_rate"].max()) + 0.05))
    plt.title("Tox21: доля активных (pos_rate) среди размеченных по задачам")
    plt.xlabel("Task")
    plt.ylabel("Positive rate")
    out = fig_dir / "tox21_positive_rate_by_task.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    nr_clean_path = data_dir / "tox21_nr_ar_cleaned.csv"
    if not nr_clean_path.exists():
        print(f"Missing file: {nr_clean_path}. Run step2 first.")
        return 2

    nr = pd.read_csv(nr_clean_path)
    desc_list = []
    ok = []
    for smi in nr["smiles"].astype(str).tolist():
        d = compute_descriptors(smi)
        if d is None:
            ok.append(False)
            desc_list.append({k: np.nan for k in ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "Rings"]})
        else:
            ok.append(True)
            desc_list.append(d)

    desc_df = pd.DataFrame(desc_list)
    nr_desc = pd.concat([nr.reset_index(drop=True), desc_df], axis=1)
    nr_desc = nr_desc.loc[pd.Series(ok).values].copy()

    nr_desc_path = reports_dir / "nr_ar_descriptors.csv"
    nr_desc.to_csv(nr_desc_path, index=False)

    long = nr_desc.melt(
        id_vars=["NR-AR"],
        value_vars=["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "Rings"],
        var_name="descriptor",
        value_name="value",
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=long,
        x="descriptor",
        y="value",
        hue="NR-AR",
        showfliers=False,
        palette="Set2",
    )
    plt.title("NR-AR: распределения дескрипторов по классам")
    plt.xlabel("")
    plt.ylabel("Value")
    plt.legend(title="NR-AR", loc="upper right")
    out = fig_dir / "nr_ar_descriptors_boxplot.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    corr = nr_desc[["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "Rings"]].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("NR-AR: корреляции между дескрипторами")
    out = fig_dir / "nr_ar_descriptors_corr.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    feat_path = data_dir / "tox21_nr_ar_features_adv.csv"
    if not feat_path.exists():
        print(f"Missing file: {feat_path}. Run advanced step4 first.")
        return 3

    feat = pd.read_csv(feat_path)
    X = feat.drop(columns=["mol_id", "NR-AR"])
    y = feat["NR-AR"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    ratio = float((y_train == 0).sum()) / float((y_train == 1).sum())
    model = xgb.XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=ratio,
        learning_rate=0.05,
        n_estimators=400,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    y_prob = model.predict_proba(X_test_s)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc:.3f})", color="#DD8452")
    plt.plot([0, 1], [0, 1], "k--", label="Random guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC: NR-AR (XGBoost, advanced features)")
    plt.legend(loc="lower right")
    out = fig_dir / "roc_curve_xgb_adv.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    precision, recall, pr_thr = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="#4C72B0", label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall: NR-AR (XGBoost, advanced features)")
    plt.legend(loc="lower left")
    out = fig_dir / "pr_curve_xgb_adv.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    thresholds = np.linspace(0.01, 0.99, 99)
    sweep = threshold_sweep(y_test, y_prob, thresholds)
    sweep_path = reports_dir / "threshold_sweep_xgb_adv.csv"
    sweep.to_csv(sweep_path, index=False)

    best_row = sweep.iloc[sweep["f1"].values.argmax()]
    best_t = float(best_row["threshold"])

    plt.figure(figsize=(8, 6))
    plt.plot(sweep["threshold"], sweep["precision"], label="precision")
    plt.plot(sweep["threshold"], sweep["recall"], label="recall")
    plt.plot(sweep["threshold"], sweep["f1"], label="f1")
    plt.axvline(best_t, color="k", linestyle="--", linewidth=1, label=f"best_t={best_t:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold sweep: precision/recall/F1 (XGBoost)")
    plt.legend(loc="best")
    out = fig_dir / "threshold_sweep_xgb_adv.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    y_pred = (y_prob >= best_t).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix (threshold={best_t:.2f})")
    out = fig_dir / "confusion_matrix_xgb_adv.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    print("Saved:")
    print(ds_summary_path)
    print(nr_desc_path)
    print(sweep_path)
    print("Figures in:", fig_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
