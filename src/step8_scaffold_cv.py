from pathlib import Path
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

import xgboost as xgb


def scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return ""
    return Chem.MolToSmiles(scaf, isomericSmiles=False)


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

    rows = []
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

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

        prob = model.predict_proba(X_te_s)[:, 1]

        roc = roc_auc_score(y_te, prob) if len(np.unique(y_te)) > 1 else float("nan")
        ap = average_precision_score(y_te, prob) if len(np.unique(y_te)) > 1 else float("nan")

        rows.append(
            {
                "fold": fold,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "pos_rate_train": float(y_tr.mean()),
                "pos_rate_test": float(y_te.mean()),
                "roc_auc": float(roc),
                "average_precision": float(ap),
            }
        )
        print(rows[-1])

    out = pd.DataFrame(rows)
    out.to_csv(reports_dir / "scaffold_cv_metrics.csv", index=False)

    print("Saved reports/scaffold_cv_metrics.csv")
    print("Mean ROC-AUC:", float(out["roc_auc"].mean()))
    print("Mean AP:", float(out["average_precision"].mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
