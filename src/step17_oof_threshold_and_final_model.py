from pathlib import Path
import json
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score, average_precision_score

import xgboost as xgb
import joblib


RDLogger.DisableLog("rdApp.warning")


def scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return ""
    return Chem.MolToSmiles(scaf, isomericSmiles=False)


def pick_threshold_max_recall_at_precision(y_true, y_prob, target_precision=0.90):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    p_thr, r_thr = p[:-1], r[:-1]  # align with thresholds

    ok = np.where(p_thr >= target_precision)[0]
    if len(ok) == 0:
        return 0.999, float(p[-1]), float(r[-1]), "no_feasible_threshold"

    i = ok[np.argmax(r_thr[ok])]
    return float(thr[i]), float(p_thr[i]), float(r_thr[i]), "feasible"


def prf1_from_cm(tn, fp, fn, tp):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    reports_dir = repo_root / "reports"
    models_dir = repo_root / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    feat_path = data_dir / "tox21_nr_ar_features_adv.csv"
    clean_path = data_dir / "tox21_nr_ar_cleaned.csv"

    feat = pd.read_csv(feat_path)
    clean = pd.read_csv(clean_path)[["mol_id", "smiles"]]

    df = feat.merge(clean, on="mol_id", how="inner").dropna(subset=["smiles", "NR-AR"]).copy()
    df["NR-AR"] = df["NR-AR"].astype(int)
    df["scaffold"] = df["smiles"].astype(str).map(scaffold_smiles)

    X = df.drop(columns=["mol_id", "smiles", "scaffold", "NR-AR"]).values
    y = df["NR-AR"].values
    groups = df["scaffold"].values

    # === настройки operating point ===
    target_precision_val = 0.95   # по твоему sweep это best, чтобы на тесте было >=0.90
    target_precision_oof = 0.90   # то, что хотим гарантировать по OOF

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

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

    oof_prob = np.full(len(df), np.nan, dtype=float)
    best_iters = []
    fold_rows = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        X_train, y_train = X[tr_idx], y[tr_idx]
        X_test, y_test = X[te_idx], y[te_idx]

        # inner split (для early stopping + подбора порога)
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
        t_fold, p_val, r_val, mode = pick_threshold_max_recall_at_precision(
            y_val, val_prob, target_precision=target_precision_val
        )

        best_iter = getattr(model_a, "best_iteration", None)
        best_n_estimators = int(best_iter + 1) if best_iter is not None else int(base_params["n_estimators"])
        best_iters.append(best_n_estimators)

        # final fit на train-фолде фиксированным числом деревьев
        scaler_b = StandardScaler()
        X_train_s = scaler_b.fit_transform(X_train)
        X_test_s = scaler_b.transform(X_test)

        ratio_train = float((y_train == 0).sum()) / max(1.0, float((y_train == 1).sum()))
        model_b = xgb.XGBClassifier(
            **{k: v for k, v in base_params.items() if k != "n_estimators"},
            n_estimators=best_n_estimators,
            scale_pos_weight=ratio_train,
        )
        model_b.fit(X_train_s, y_train, verbose=False)

        test_prob = model_b.predict_proba(X_test_s)[:, 1]
        oof_prob[te_idx] = test_prob

        y_pred = (test_prob >= t_fold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        prec, rec, f1 = prf1_from_cm(tn, fp, fn, tp)

        roc = roc_auc_score(y_test, test_prob) if len(np.unique(y_test)) > 1 else float("nan")
        ap = average_precision_score(y_test, test_prob) if len(np.unique(y_test)) > 1 else float("nan")

        fold_rows.append({
            "fold": fold,
            "threshold_val_selected": float(t_fold),
            "threshold_mode": mode,
            "val_precision_at_t": float(p_val),
            "val_recall_at_t": float(r_val),
            "final_n_estimators": int(best_n_estimators),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_f1": float(f1),
            "test_roc_auc": float(roc),
            "test_ap": float(ap),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })

    assert np.isfinite(oof_prob).all(), "OOF probs contain NaN; check fold assignment."

    oof_out = df[["mol_id", "NR-AR"]].copy()
    oof_out["oof_prob"] = oof_prob
    oof_path = reports_dir / "step17_oof_predictions.csv"
    oof_out.to_csv(oof_path, index=False)
    print("Saved:", oof_path)

    # === OOF threshold selection: Precision>=0.90, maximize recall ===
    t_oof, p_oof, r_oof, mode_oof = pick_threshold_max_recall_at_precision(
        y, oof_prob, target_precision=target_precision_oof
    )

    y_oof_pred = (oof_prob >= t_oof).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_oof_pred, labels=[0, 1]).ravel()
    prec_oof, rec_oof, f1_oof = prf1_from_cm(tn, fp, fn, tp)
    roc_oof = roc_auc_score(y, oof_prob) if len(np.unique(y)) > 1 else float("nan")
    ap_oof = average_precision_score(y, oof_prob) if len(np.unique(y)) > 1 else float("nan")

    # === Train final model on ALL data ===
    final_n_estimators = int(np.median(best_iters)) if len(best_iters) else 400
    ratio_all = float((y == 0).sum()) / max(1.0, float((y == 1).sum()))

    final_scaler = StandardScaler()
    X_all_s = final_scaler.fit_transform(X)

    final_model = xgb.XGBClassifier(
        **{k: v for k, v in base_params.items() if k != "n_estimators"},
        n_estimators=final_n_estimators,
        scale_pos_weight=ratio_all,
    )
    final_model.fit(X_all_s, y, verbose=False)

    # === Save artifacts ===
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(reports_dir / "step17_folds_thresholds_and_metrics.csv", index=False)

    meta = {
        "task": "NR-AR",
        "feature_file": feat_path.name,
        "target_precision_val_for_threshold_selection": target_precision_val,
        "oof_target_precision_constraint": target_precision_oof,
        "oof_threshold": float(t_oof),
        "oof_precision_at_threshold": float(prec_oof),
        "oof_recall_at_threshold": float(rec_oof),
        "oof_f1_at_threshold": float(f1_oof),
        "oof_roc_auc": float(roc_oof),
        "oof_average_precision": float(ap_oof),
        "oof_confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "final_n_estimators": int(final_n_estimators),
        "best_iters_per_fold": [int(x) for x in best_iters],
        "scale_pos_weight_all": float(ratio_all),
    }
    with open(reports_dir / "step17_final_operating_point.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    joblib.dump(final_scaler, models_dir / "nr_ar_scaler.joblib")
    final_model.save_model(models_dir / "nr_ar_xgb.json")

    print("OOF threshold:", t_oof, "mode:", mode_oof)
    print("OOF precision/recall/f1:", prec_oof, rec_oof, f1_oof)
    print("Saved:",
          reports_dir / "step17_folds_thresholds_and_metrics.csv",
          reports_dir / "step17_final_operating_point.json",
          models_dir / "nr_ar_scaler.joblib",
          models_dir / "nr_ar_xgb.json"
          )


if __name__ == "__main__":
    main()
