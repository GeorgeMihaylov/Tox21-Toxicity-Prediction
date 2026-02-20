import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    fig_dir = repo_root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    input_file = data_dir / "tox21_nr_ar_cleaned.csv"

    if not input_file.exists():
        print(f"Error: {input_file} not found. Run step 2 first.")
        return 1

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    print("Calculating MW and LogP...")
    mw_list = []
    logp_list = []

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol:
            mw_list.append(Descriptors.MolWt(mol))
            logp_list.append(Descriptors.MolLogP(mol))
        else:
            mw_list.append(None)
            logp_list.append(None)

    df['MW'] = mw_list
    df['LogP'] = logp_list

    # 1. График баланса классов
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='NR-AR', palette='Set2')
    plt.title('Class Balance (NR-AR)')
    plt.xlabel('Toxicity Class (0=Inactive, 1=Active)')
    plt.ylabel('Count')
    class_fig = fig_dir / "class_balance.png"
    plt.savefig(class_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {class_fig}")

    # 2. График распределения молекулярной массы (MW)
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='MW', hue='NR-AR', bins=50, kde=True, palette='Set2', alpha=0.6)
    plt.title('Molecular Weight Distribution by Class')
    plt.xlabel('Molecular Weight (Da)')
    plt.ylabel('Count')
    plt.xlim(0, 1000)  # Ограничим ось X для лучшей читаемости
    mw_fig = fig_dir / "mw_distribution.png"
    plt.savefig(mw_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {mw_fig}")

    # 3. График распределения LogP (липофильность)
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='LogP', hue='NR-AR', bins=50, kde=True, palette='Set2', alpha=0.6)
    plt.title('LogP Distribution by Class')
    plt.xlabel('LogP')
    plt.ylabel('Count')
    logp_fig = fig_dir / "logp_distribution.png"
    plt.savefig(logp_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {logp_fig}")

    # 4. Scatter plot (Chemical Space): MW vs LogP
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='MW', y='LogP', hue='NR-AR', palette='Set2', alpha=0.5, s=20)
    plt.title('Chemical Space: MW vs LogP')
    plt.xlabel('Molecular Weight (Da)')
    plt.ylabel('LogP')
    plt.xlim(0, 1000)
    plt.ylim(-10, 15)
    scatter_fig = fig_dir / "chemical_space.png"
    plt.savefig(scatter_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {scatter_fig}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
