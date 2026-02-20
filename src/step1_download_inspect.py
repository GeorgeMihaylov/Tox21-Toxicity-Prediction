from pathlib import Path
import sys
import pandas as pd
import urllib.request

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
FILENAME = "tox21.csv.gz"


def main() -> int:
    # Определяем пути (сохраняем в папку data/raw)
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    file_path = raw_dir / FILENAME

    # Скачиваем файл, если его нет
    if not file_path.exists() or file_path.stat().st_size == 0:
        print(f"Downloading {URL} ...")
        try:
            with urllib.request.urlopen(URL) as r, open(file_path, "wb") as f:
                f.write(r.read())
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download: {e}")
            return 1
    else:
        print("File already downloaded.")

    # Читаем данные
    df = pd.read_csv(file_path)
    print("\nDataset info:")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    # Выбираем нашу мишень (Андрогеновый рецептор)
    task = "NR-AR"

    # Отфильтруем строки, где есть SMILES и есть метка для NR-AR (не NaN)
    df_task = df.dropna(subset=[task, "smiles"]).copy()

    # Превращаем таргет в целое число (0 или 1)
    df_task[task] = df_task[task].astype(int)

    print(f"\nTask: {task}")
    print(f"Molecules with labels for {task}: {len(df_task)}")

    # Смотрим баланс классов
    counts = df_task[task].value_counts().sort_index()
    print("\nClass balance (0/1):")
    print(counts.to_string())

    # Сохраняем очищенный набор только с нужными колонками
    out_path = data_dir / "tox21_nr_ar.csv"
    df_task[["mol_id", "smiles", task]].to_csv(out_path, index=False)
    print(f"\nSaved filtered data to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
