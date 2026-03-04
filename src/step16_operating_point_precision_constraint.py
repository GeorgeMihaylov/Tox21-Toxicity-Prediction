from pathlib import Path
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

import xgboost as xgb


RDLogger.DisableLog("rdApp.warning")


def scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return ""
    return Chem.MolToSmiles(scaf, isomericSmiles=False)


def threshold_max_recall_at_precision(y_true, y_prob, target_precision=0.90):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    p_thr, r_thr = p[:-1], r[:-1]

    ok = np.where(p_thr >= target_precision)[0]
    if len(ok) == 0:
        return 0.999, float(p[-1]), float(r[-1]), "no_feasible_threshold"

    i = ok[np.argmax(r_thr[ok])]
    return float(thr[i]), float(p_thr[i]), float(r_thr[i]), "feasible"


def micro_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    feat_path = data_dir / "tox21_nr_ar_features_adv.csv"
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

    target_precisions = [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]

    summary_rows = []
    detailed_rows = []

    # ВАЖНО: scale_pos_weight тут НЕ задаём, чтобы не словить duplicate kwargs
    base_params = dict(
        eval_metric="aucpr",
        learning_rate=0.05,
        n_estimators=4000,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    for P in target_precisions:
        TP = FP = TN = FN = 0
        fold_rocs, fold_aps = [], []
        feasible_cnt = 0

        for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
            X_train, y_train = X[tr_idx], y[tr_idx]
            X_test, y_test = X[te_idx], y[te_idx]

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

            scaler_a = StandardScaler()
            X_tr_s = scaler_a.fit_transform(X_tr)
            X_val_s = scaler_a.transform(X_val)

            ratio_tr = float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum()))

            model_a = xgb.XGBClassifier(
                **base_params,
                scale_pos_weight=ratio_tr,
                early_stopping_rounds=100,
            )
            model_a.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)

            val_prob = model_a.predict_proba(X_val_s)[:, 1]
            t, p_val, r_val, mode = threshold_max_recall_at_precision(y_val, val_prob, target_precision=P)
            feasible_cnt += int(mode == "feasible")

            best_iter = getattr(model_a, "best_iteration", None)
            best_n_estimators = int(best_iter + 1) if best_iter is not None else int(base_params["n_estimators"])

            scaler_b = StandardScaler()
            X_train_s = scaler_b.fit_transform(X_train)
            X_test_s = scaler_b.transform(X_test)

            # Для финального fit лучше пересчитать ratio уже на всём train-фолде
            ratio_train = float((y_train == 0).sum()) / max(1.0, float((y_train == 1).sum()))

            model_b = xgb.XGBClassifier(
                **{k: v for k, v in base_params.items() if k != "n_estimators"},
                n_estimators=best_n_estimators,
                scale_pos_weight=ratio_train,
            )
            model_b.fit(X_train_s, y_train, verbose=False)

            test_prob = model_b.predict_proba(X_test_s)[:, 1]
            roc = roc_auc_score(y_test, test_prob) if len(np.unique(y_test)) > 1 else float("nan")
            ap = average_precision_score(y_test, test_prob) if len(np.unique(y_test)) > 1 else float("nan")
            fold_rocs.append(float(roc))
            fold_aps.append(float(ap))

            y_pred = (test_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

            TP += int(tp)
            FP += int(fp)
            TN += int(tn)
            FN += int(fn)

            detailed_rows.append({
                "target_precision": P,
                "fold": fold,
                "threshold": t,
                "threshold_mode": mode,
                "val_precision_at_t": p_val,
                "val_recall_at_t": r_val,
                "best_iteration": int(best_iter) if best_iter is not None else None,
                "final_n_estimators": int(best_n_estimators),
                "roc_auc": float(roc),
                "average_precision": float(ap),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            })

        prec_micro, rec_micro, f1_micro = micro_metrics(TP, FP, FN)
        summary_rows.append({
            "target_precision": P,
            "feasible_folds": feasible_cnt,
            "micro_precision_test": prec_micro,
            "micro_recall_test": rec_micro,
            "micro_f1_test": f1_micro,
            "mean_roc_auc": float(np.nanmean(fold_rocs)),
            "mean_ap": float(np.nanmean(fold_aps)),
            "TP": int(TP), "FP": int(FP), "TN": int(TN), "FN": int(FN),
        })

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detailed_rows)

    summary_path = reports_dir / "operating_point_sweep_precision_constraint_summary.csv"
    detail_path = reports_dir / "operating_point_sweep_precision_constraint_folds.csv"
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    print("Saved:", summary_path)
    print("Saved:", detail_path)

    ok = summary_df[summary_df["micro_precision_test"] >= 0.90].copy()
    if len(ok) == 0:
        print("\nNo setting reached micro_precision_test >= 0.90")
    else:
        best = ok.sort_values(["micro_recall_test", "micro_precision_test"], ascending=[False, False]).head(1)
        print("\nBest (micro_precision_test>=0.90, max micro_recall_test):")
        print(best)

    print("\nAll results:")
    print(summary_df.sort_values("target_precision"))


if __name__ == "__main__":
    main()
