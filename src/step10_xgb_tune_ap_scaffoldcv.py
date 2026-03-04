from pathlib import Path
import json
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

import xgboost as xgb


def scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return ""
    return Chem.MolToSmiles(scaf, isomericSmiles=False)


def sample_params(rng: np.random.RandomState):
    return {
        "max_depth": int(rng.choice([3, 4, 5, 6, 7, 8])),
        "min_child_weight": float(rng.choice([1, 2, 3, 5, 8, 13])),
        "gamma": float(rng.choice([0.0, 0.5, 1.0, 2.0, 5.0])),
        "subsample": float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
        "colsample_bytree": float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
        "reg_lambda": float(rng.choice([0.5, 1.0, 2.0, 5.0, 10.0])),
        "reg_alpha": float(rng.choice([0.0, 0.1, 0.5, 1.0])),
        "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.07])),
        "n_estimators": int(rng.choice([400, 600, 800, 1200])),
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

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

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    rng = np.random.RandomState(42)
    n_trials = 20

    results = []
    best = {"mean_ap": -1.0, "params": None}

    for trial in range(1, n_trials + 1):
        params = sample_params(rng)

        fold_aps = []
        for tr_idx, te_idx in cv.split(X, y, groups=groups):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            ratio = float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum()))

            model = xgb.XGBClassifier(
                eval_metric="aucpr",
                scale_pos_weight=ratio,
                random_state=42,
                n_jobs=-1,
                **params,
            )
            model.fit(X_tr_s, y_tr)
            prob = model.predict_proba(X_te_s)[:, 1]
            ap = average_precision_score(y_te, prob) if len(np.unique(y_te)) > 1 else float("nan")
            fold_aps.append(float(ap))

        mean_ap = float(np.nanmean(fold_aps))
        std_ap = float(np.nanstd(fold_aps))

        row = {"trial": trial, "mean_ap": mean_ap, "std_ap": std_ap, **params}
        results.append(row)

        if mean_ap > best["mean_ap"]:
            best["mean_ap"] = mean_ap
            best["params"] = params

        print(f"Trial {trial:02d}: mean AP={mean_ap:.4f} ± {std_ap:.4f} | params={params}")

    res_df = pd.DataFrame(results).sort_values("mean_ap", ascending=False)
    res_df.to_csv(reports_dir / "xgb_tuning_ap_scaffoldcv.csv", index=False)

    with open(reports_dir / "xgb_best_params_ap_scaffoldcv.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("Saved reports/xgb_tuning_ap_scaffoldcv.csv")
    print("Saved reports/xgb_best_params_ap_scaffoldcv.json")
    print("Best mean AP:", best["mean_ap"])
    print("Best params:", best["params"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
