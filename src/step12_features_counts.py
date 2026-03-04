from pathlib import Path
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator


def morgan_count_fp(mol, radius=2, nBits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    # GetCountFingerprintAsNumPy() возвращает numpy array (counts)
    return gen.GetCountFingerprintAsNumPy(mol)


def compute_desc(mol):
    # минимальный набор “как у вас”
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    return np.array([mw, logp, tpsa, hbd, hba, rot, rings], dtype=np.float32)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    inp = data_dir / "tox21_nr_ar_cleaned.csv"
    out = data_dir / "tox21_nr_ar_features_counts.csv"

    df = pd.read_csv(inp)
    df = df.dropna(subset=["smiles", "NR-AR"]).copy()
    df["NR-AR"] = df["NR-AR"].astype(int)

    X_rows = []
    ok_ids = []
    y = []

    for mol_id, smi, label in df[["mol_id", "smiles", "NR-AR"]].itertuples(index=False):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = morgan_count_fp(mol, radius=2, nBits=2048)
        desc = compute_desc(mol)
        feats = np.concatenate([fp.astype(np.float32), desc], axis=0)
        X_rows.append(feats)
        ok_ids.append(mol_id)
        y.append(int(label))

    X = np.vstack(X_rows)
    y = np.array(y, dtype=int)

    cols = [f"morgan_count_{i}" for i in range(2048)] + ["MW","LogP","TPSA","HBD","HBA","RotBonds","Rings"]
    out_df = pd.DataFrame(X, columns=cols)
    out_df.insert(0, "mol_id", ok_ids)
    out_df.insert(1, "NR-AR", y)

    out_df.to_csv(out, index=False)
    print("Saved:", out, "shape=", out_df.shape)


if __name__ == "__main__":
    main()
