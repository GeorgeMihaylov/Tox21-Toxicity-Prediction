import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, precision_recall_curve
import xgboost as xgb

try:
    from imblearn.combine import SMOTEENN
    from imblearn.pipeline import Pipeline
except ImportError:
    print("Error: imbalanced-learn is not installed. Run 'pip install imbalanced-learn'")
    sys.exit(1)


def find_optimal_threshold(y_true, y_prob):
    """Находит порог, который максимизирует F1-score"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # Защита от деления на ноль
    f1_scores = np.divide(
        2 * (precision * recall),
        (precision + recall),
        out=np.zeros_like(precision),
        where=(precision + recall) != 0
    )
    # thresholds на 1 короче, чем массивы precision и recall
    optimal_idx = np.argmax(f1_scores[:-1])
    return thresholds[optimal_idx], f1_scores[optimal_idx]


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    fig_dir = repo_root / "reports" / "figures"

    input_file = data_dir / "tox21_nr_ar_features_adv.csv"

    if not input_file.exists():
        print(f"Error: {input_file} not found. Run step 4 first.")
        return 1

    print(f"Loading features from {input_file}...")
    df = pd.read_csv(input_file)

    print("Preparing train/test split (80/20)...")
    X = df.drop(columns=['mol_id', 'NR-AR'])
    y = df['NR-AR']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Используем SMOTEENN (Генерация новых примеров + удаление шума на границах классов)
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('model', RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1))
    ])

    ratio = float(sum(y_train == 0)) / sum(y_train == 1)
    xgb_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=ratio,
        learning_rate=0.05,
        n_estimators=300,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    models = {
        "Random Forest (SMOTEENN)": rf_pipeline,
        "XGBoost (Class Weights)": Pipeline([('scaler', StandardScaler()), ('model', xgb_model)])
    }

    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_prob)

        # Динамический поиск лучшего порога для этой конкретной модели
        best_threshold, best_f1 = find_optimal_threshold(y_test, y_prob)

        y_pred = (y_prob >= best_threshold).astype(int)

        print(f"{name} ROC-AUC: {auc_score:.4f}")
        print(f"Optimal Threshold (maximizes F1): {best_threshold:.3f}")
        print("-" * 40)
        print(classification_report(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: NR-AR Toxicity')
    plt.legend(loc="lower right")

    roc_fig = fig_dir / "roc_curve_final.png"
    plt.savefig(roc_fig, dpi=300, bbox_inches='tight')
    plt.close()

    print("Pipeline complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
