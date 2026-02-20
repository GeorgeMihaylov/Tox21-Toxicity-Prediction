import sys
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    input_file = data_dir / "tox21_nr_ar_cleaned.csv"
    output_features = data_dir / "tox21_nr_ar_features_adv.csv"

    if not input_file.exists():
        print(f"Error: {input_file} not found. Run step 2 first.")
        return 1

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    print("Generating Morgan Fingerprints and 2D Descriptors...")
    fps = []
    descriptors = []

    total = len(df)

    for idx, row in df.iterrows():
        smi = str(row['smiles'])
        mol = Chem.MolFromSmiles(smi)

        if mol is not None:
            # 1. Фингерпринты (2048 бит)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros((0,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)

            # 2. Физико-химические дескрипторы (6 штук)
            desc = [
                Descriptors.MolWt(mol),  # Молекулярная масса
                Descriptors.MolLogP(mol),  # Липофильность
                Descriptors.TPSA(mol),  # Топологическая площадь полярной поверхности
                Descriptors.NumHDonors(mol),  # Доноры водородной связи
                Descriptors.NumHAcceptors(mol),  # Акцепторы водородной связи
                Descriptors.NumRotatableBonds(mol)  # Вращающиеся связи
            ]
            descriptors.append(desc)
        else:
            fps.append(np.zeros((2048,), dtype=np.int8))
            descriptors.append([0.0] * 6)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{total} molecules")

    print(f"Processed {total}/{total} molecules")

    print("Building features DataFrame...")
    # Собираем фингерпринты
    col_names_fp = [f"bit_{i}" for i in range(2048)]
    fps_df = pd.DataFrame(fps, columns=col_names_fp)

    # Собираем дескрипторы
    col_names_desc = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds']
    desc_df = pd.DataFrame(descriptors, columns=col_names_desc)

    # Объединяем всё вместе
    final_df = pd.concat([df[['mol_id', 'NR-AR']], desc_df, fps_df], axis=1)

    print(f"Final dataset shape: {final_df.shape}")

    print(f"Saving advanced features to {output_features}...")
    final_df.to_csv(output_features, index=False)
    print("Done!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
