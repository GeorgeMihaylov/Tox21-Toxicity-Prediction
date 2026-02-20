import sys
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    input_file = data_dir / "tox21_nr_ar_cleaned.csv"
    output_features = data_dir / "tox21_nr_ar_features.csv"

    if not input_file.exists():
        print(f"Error: {input_file} not found. Run step 2 first.")
        return 1

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    print("Generating Morgan Fingerprints (radius=2, nBits=2048)...")
    fps = []

    # Чтобы видеть прогресс в консоли, если молекул много
    total = len(df)

    for idx, row in df.iterrows():
        smi = str(row['smiles'])
        mol = Chem.MolFromSmiles(smi)

        if mol is not None:
            # radius=2 эквивалентно ECFP4
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            # Конвертируем в массив numpy
            arr = np.zeros((0,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        else:
            # На случай, если проскочила невалидная молекула
            fps.append(np.zeros((2048,), dtype=np.int8))

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{total} molecules")

    print(f"Processed {total}/{total} molecules")

    # Создаем DataFrame из фингерпринтов
    print("Building features DataFrame...")
    col_names = [f"bit_{i}" for i in range(2048)]
    fps_df = pd.DataFrame(fps, columns=col_names)

    # Объединяем с исходным датасетом (оставляем только ID и таргет)
    # SMILES убираем, так как для ML он больше не нужен в виде строки
    final_df = pd.concat([df[['mol_id', 'NR-AR']], fps_df], axis=1)

    print(f"Final dataset shape: {final_df.shape}")

    # Сохраняем в CSV
    print(f"Saving features to {output_features}...")
    final_df.to_csv(output_features, index=False)
    print("Done!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
