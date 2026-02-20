import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    input_file = data_dir / "tox21_nr_ar.csv"
    output_file = data_dir / "tox21_nr_ar_cleaned.csv"

    if not input_file.exists():
        print(f"Error: {input_file} not found. Run step 1 first.")
        return 1

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    initial_count = len(df)
    print(f"Initial row count: {initial_count}")

    # Валидация SMILES через RDKit
    valid_smiles = []
    canonical_smiles = []
    invalid_count = 0

    print("Validating SMILES strings with RDKit...")
    for idx, row in df.iterrows():
        smi = str(row['smiles'])
        # Пробуем создать объект молекулы
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            valid_smiles.append(False)
            canonical_smiles.append(None)
            invalid_count += 1
        else:
            valid_smiles.append(True)
            # Сохраняем канонический вид SMILES (чтобы точно найти дубликаты)
            canonical_smiles.append(Chem.MolToSmiles(mol))

    df['is_valid'] = valid_smiles
    df['canonical_smiles'] = canonical_smiles

    # Фильтруем только валидные
    df_valid = df[df['is_valid']].copy()
    print(f"Removed {invalid_count} invalid SMILES strings.")

    # Поиск и удаление дубликатов по каноническому SMILES
    before_dedup = len(df_valid)
    # Если есть дубликаты с разными метками, оставляем первое вхождение
    # (в идеале нужно проверять конфликтующие метки, но для простоты берем первый)
    df_cleaned = df_valid.drop_duplicates(subset=['canonical_smiles'], keep='first')
    duplicates_removed = before_dedup - len(df_cleaned)
    print(f"Removed {duplicates_removed} duplicate molecules.")

    # Оставляем нужные колонки
    final_df = df_cleaned[['mol_id', 'canonical_smiles', 'NR-AR']].copy()
    final_df.rename(columns={'canonical_smiles': 'smiles'}, inplace=True)

    print(f"\nFinal cleaned dataset size: {len(final_df)}")

    # Сохраняем результат
    final_df.to_csv(output_file, index=False)
    print(f"Saved cleaned dataset to: {output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
