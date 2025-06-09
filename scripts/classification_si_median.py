import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE  # Для балансировки классов
import optuna  # Для оптимизации гиперпараметров
from collections import Counter

# Настройка логирования для записи всех действий и ошибок
logging.basicConfig(
    filename='logs/classification_si_median.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Определение путей к данным, результатам и графикам
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_dir, 'data', 'processed_data.csv')
results_dir = os.path.join(project_dir, 'results')
figures_dir = os.path.join(project_dir, 'figures')

# Создание директорий, если они не существуют
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

def train_model(model_class, X_train, X_test, y_train, y_test, model_name):
    """
    Обучает модель с оптимизацией гиперпараметров через Optuna.
    
    Args:
        model_class: Класс модели (RandomForestClassifier и т.д.).
        X_train, X_test: Признаки для обучения и теста.
        y_train, y_test: Целевые переменные.
        model_name: Имя модели ('RF', 'XGB', 'LGB', 'GB', 'LR').
    
    Returns:
        tuple: (model, accuracy, f1, precision, recall, roc_auc, pr_auc, y_pred, y_proba)
               или (None, 0, 0, 0, 0, 0, 0, None, None) при ошибке.
    """
    # Оптимизация гиперпараметров
    def objective(trial):
        try:
            if model_name == 'RF':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_categorical('max_depth', [10, 20, 30, None]),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
                }
                model = RandomForestClassifier(**params, random_state=42, n_jobs=1)
            elif model_name == 'XGB':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0)
                }
                model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
            elif model_name == 'LGB':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 70),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 100)
                }
                model = LGBMClassifier(**params, random_state=42, verbose=-1, force_row_wise=True)
            elif model_name == 'GB':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
                }
                model = GradientBoostingClassifier(**params, random_state=42)
            else:  # LR
                params = {
                    'C': trial.suggest_float('C', 0.001, 10.0, log=True),
                    'max_iter': 2000
                }
                model = LogisticRegression(**params, random_state=42, n_jobs=1)
            
            # Оценка модели по F1 через кросс-валидацию
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            return scores.mean()
        except Exception as e:
            logging.error(f"Ошибка оптимизации {model_name}: {e}")
            return 0.0

    # Создание исследования Optuna
    study = optuna.create_study(direction='maximize')
    try:
        study.optimize(objective, n_trials=50, n_jobs=1)
        best_params = study.best_params
        logging.info(f"Лучшие параметры для {model_name}: {best_params}")
    except Exception as e:
        logging.error(f"Ошибка оптимизации Optuna для {model_name}: {e}")
        return None, 0, 0, 0, 0, 0, 0, None, None

    # Инициализация модели с лучшими параметрами
    if model_name == 'RF':
        model = RandomForestClassifier(**best_params, random_state=42, n_jobs=1)
    elif model_name == 'XGB':
        model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    elif model_name == 'LGB':
        model = LGBMClassifier(**best_params, random_state=42, verbose=-1, force_row_wise=True)
    elif model_name == 'GB':
        model = GradientBoostingClassifier(**best_params, random_state=42)
    else:
        model = LogisticRegression(**best_params, random_state=42, n_jobs=1)

    try:
        # Обучение модели
        model.fit(X_train, y_train)
        # Предсказания
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Проверка на недопустимые значения
        if np.any(np.isnan(y_pred)) or np.any(np.isnan(y_proba)):
            logging.error(f"Недопустимые предсказания для {model_name}")
            return None, 0, 0, 0, 0, 0, 0, None, None

        # Вычисление метрик
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        # Оценка на обучающей выборке
        y_train_pred = model.predict(X_train)
        train_f1 = f1_score(y_train, y_train_pred)
        logging.info(f"Обучающий F1 для {model_name}: {train_f1:.3f}")
        logging.info(f"Тестовые метрики {model_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, ROC_AUC={roc_auc:.3f}, PR_AUC={pr_auc:.3f}")
        return model, accuracy, f1, precision, recall, roc_auc, pr_auc, y_pred, y_proba
    except Exception as e:
        logging.error(f"Ошибка обучения {model_name}: {e}")
        return None, 0, 0, 0, 0, 0, 0, None, None

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Создает и сохраняет матрицу ошибок."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Матрица ошибок (SI_median, {model_name})')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        plt.savefig(os.path.join(figures_dir, f'confusion_matrix_si_median_{model_name.lower()}.png'))
        plt.close()
        logging.info(f"Матрица ошибок сохранена: figures/confusion_matrix_si_median_{model_name.lower()}.png")
    except Exception as e:
        logging.error(f"Ошибка при создании матрицы ошибок: {e}")

def plot_roc_curve(y_true, y_proba, model_name):
    """Создает и сохраняет ROC-кривую."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC кривая (AUC={roc_auc_score(y_true, y_proba):.3f})')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Ложно-положительный уровень')
        plt.ylabel('Истинно-положительный уровень')
        plt.title(f'ROC кривая (SI_median, {model_name})')
        plt.legend()
        plt.savefig(os.path.join(figures_dir, f'roc_curve_si_median_{model_name.lower()}.png'))
        plt.close()
        logging.info(f"ROC кривая сохранена: figures/roc_curve_si_median_{model_name.lower()}.png")
    except Exception as e:
        logging.error(f"Ошибка при создании ROC кривой: {e}")

def main():
    """Основная функция для классификации SI по медиане."""
    logging.info("Запуск classification_si_median.py")
    try:
        # Загрузка данных
        df = pd.read_csv(data_path)
        if 'SI' not in df.columns:
            logging.error("Колонка SI отсутствует")
            print("Ошибка: Колонка SI отсутствует!")
            exit(1)
        
        # Бинаризация SI по медиане
        median_si = df['SI'].median()
        df['SI_class'] = (df['SI'] > median_si).astype(int)
        logging.info(f"Медиана SI: {median_si}, классы: {Counter(df['SI_class'])}")

        # Проверка распределения классов
        class_counts = Counter(df['SI_class'])
        if len(class_counts) < 2 or min(class_counts.values()) == 0:
            logging.error("Один из классов пустой")
            print("Ошибка: Один из классов пустой!")
            exit(1)

        # Выбор признаков
        features = [col for col in df.columns if col not in ['IC50_mM', 'CC50_mM', 'SI', 'log_ic50', 'log_cc50', 'log_si', 'SI_class']]
        X = df[features]
        y = df['SI_class']

        # Проверка на NaN и бесконечные значения
        if X.isnull().values.any() or np.any(np.isinf(X.values)) or y.isnull().any():
            logging.error("NaN или бесконечные значения в данных")
            print("Ошибка: NaN или бесконечные значения в данных!")
            exit(1)

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        logging.info(f"Разделение данных: train={X_train.shape[0]}, test={X_test.shape[0]}")

        # Стандартизация признаков
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info("Признаки стандартизированы")

        # Применение SMOTE при дисбалансе классов
        class_counts = pd.Series(y_train).value_counts(normalize=True).to_dict()
        if abs(class_counts.get(0, 0) - class_counts.get(1, 0)) > 0.1:
            logging.info("Применение SMOTE")
            min_samples = min(pd.Series(y_train).value_counts().values)
            if min_samples < 5:
                logging.warning("Недостаточно образцов для SMOTE")
            else:
                smote = SMOTE(random_state=42, k_neighbors=min(5, min_samples-1))
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                logging.info(f"После SMOTE: train samples={X_train_scaled.shape[0]}")

        # Список моделей
        models = [
            (RandomForestClassifier, 'RF'),
            (XGBClassifier, 'XGB'),
            (LGBMClassifier, 'LGB'),
            (GradientBoostingClassifier, 'GB'),
            (LogisticRegression, 'LR')
        ]

        results = []
        best_model = None
        best_f1 = 0
        best_y_pred = None
        best_y_proba = None
        best_model_name = None

        # Обучение моделей
        for model_class, model_name in models:
            model, accuracy, f1, precision, recall, roc_auc, pr_auc, y_pred, y_proba = train_model(
                model_class, X_train_scaled, X_test_scaled, y_train, y_test, model_name
            )
            if model is None:
                continue
            results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'F1': f1,
                'Precision': precision,
                'Recall': recall,
                'ROC_AUC': roc_auc,
                'PR_AUC': pr_auc
            })

            # Визуализация важности признаков
            if hasattr(model, 'feature_importances_'):
                try:
                    importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
                    importance = importance.sort_values('Importance', ascending=False).head(10)
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=importance)
                    plt.title(f'Важность признаков (SI_median, {model_name})')
                    plt.savefig(os.path.join(figures_dir, f'feature_importance_si_median_{model_name.lower()}.png'))
                    plt.close()
                    logging.info(f"Важность признаков сохранена: figures/feature_importance_si_median_{model_name.lower()}.png")
                except Exception as e:
                    logging.error(f"Ошибка сохранения важности признаков для {model_name}: {e}")

            # Визуализация матрицы ошибок и ROC-кривой
            plot_confusion_matrix(y_test, y_pred, model_name)
            plot_roc_curve(y_test, y_proba, model_name)

            # Сохранение лучшей модели
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_y_pred = y_pred
                best_y_proba = y_proba
                best_model_name = model_name

        # Обучение ансамбля
        logging.info("Обучение ансамбля")
        estimators = [
            ('rf', RandomForestClassifier(random_state=42, n_jobs=1)),
            ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')),
            ('lgb', LGBMClassifier(random_state=42, verbose=-1, force_row_wise=True)),
            ('gb', GradientBoostingClassifier(random_state=42))
        ]
        voting_classifier = VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)
        try:
            voting_classifier.fit(X_train_scaled, y_train)
            y_pred_voting = voting_classifier.predict(X_test_scaled)
            y_proba_voting = voting_classifier.predict_proba(X_test_scaled)[:, 1]

            if np.any(np.isnan(y_pred_voting)) or np.any(np.isnan(y_proba_voting)):
                logging.error("Недопустимые предсказания для ансамбля")
            else:
                results.append({
                    'Model': 'Voting',
                    'Accuracy': accuracy_score(y_test, y_pred_voting),
                    'F1': f1_score(y_test, y_pred_voting),
                    'Precision': precision_score(y_test, y_pred_voting),
                    'Recall': recall_score(y_test, y_pred_voting),
                    'ROC_AUC': roc_auc_score(y_test, y_proba_voting),
                    'PR_AUC': average_precision_score(y_test, y_proba_voting)
                })
                plot_confusion_matrix(y_test, y_pred_voting, 'Voting')
                plot_roc_curve(y_test, y_proba_voting, 'Voting')
                logging.info(f"Метрики ансамбля: Accuracy={results[-1]['Accuracy']:.3f}, F1={results[-1]['F1']:.3f}")
        except Exception as e:
            logging.error(f"Ошибка обучения ансамбля: {e}")

        # Сохранение результатов
        results_df = pd.DataFrame(results)
        results_df = results_df.round(3)
        results_df.to_csv(os.path.join(results_dir, 'classification_si_median.csv'), index=False)
        logging.info("Результаты классификации SI_median:\n" + results_df.to_string())
        print("Результаты классификации SI_median:")
        print(results_df)
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")
        print(f"Критическая ошибка: {e}")
        exit(1)

if __name__ == "__main__":
    main()