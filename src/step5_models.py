import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc
import xgboost as xgb


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    fig_dir = repo_root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    input_file = data_dir / "tox21_nr_ar_features.csv"

    if not input_file.exists():
        print(f"Error: {input_file} not found. Run step 4 first.")
        return 1

    print(f"Loading features from {input_file}...")
    df = pd.read_csv(input_file)

    # 1. Подготовка данных
    print("Preparing train/test split (80/20)...")
    X = df.drop(columns=['mol_id', 'NR-AR'])
    y = df['NR-AR']

    # Стратифицированное разбиение, чтобы в тесте было пропорциональное количество 1 и 0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 2. Инициализация моделей
    # Используем class_weight='balanced' и scale_pos_weight из-за сильного дисбаланса
    ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio,
                                     random_state=42, n_jobs=-1)
    }

    # 3. Обучение и оценка
    results = {}
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Предсказание вероятностей (нужно для ROC-AUC)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Предсказание классов (нужно для classification_report)
        y_pred = model.predict(X_test)

        auc_score = roc_auc_score(y_test, y_prob)
        results[name] = auc_score

        print(f"{name} ROC-AUC: {auc_score:.4f}")
        print("-" * 40)
        print(classification_report(y_test, y_pred))

        # Построение кривой для графика
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

    # 4. Сохранение графика ROC-кривой
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: NR-AR Toxicity Prediction')
    plt.legend(loc="lower right")

    roc_fig = fig_dir / "roc_curve.png"
    plt.savefig(roc_fig, dpi=300, bbox_inches='tight')
    plt.close()

    print("\nSummary of ROC-AUC Scores:")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")

    print(f"\nSaved ROC curve to: {roc_fig}")
    print("Pipeline complete!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
