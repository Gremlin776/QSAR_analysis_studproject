import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import importlib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
from collections import Counter

# Проверяем наличие всех необходимых библиотек, чтобы обеспечить выполнение классификации log_si, которая оценивает селективность соединений против вируса гриппа
required_libs = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'xgboost', 'lightgbm', 'optuna']
for lib in required_libs:
    if importlib.util.find_spec(lib) is None:
        print(f"Ошибка: Библиотека {lib} не найдена! Установите ее.")
        exit(1)

# Создаем или проверяем существование директорий для логов, графиков и результатов, чтобы организовать выходные данные и упростить их анализ
try:
    for directory in ['logs', 'figures', 'results']:
        os.makedirs(directory, exist_ok=True)
except PermissionError as e:
    print(f"Ошибка: Невозможно создать директории: {e}")
    exit(1)

# Настраиваем логирование для записи всех операций и ошибок, что позволяет отслеживать процесс выполнения скрипта и выявлять проблемы
logging.basicConfig(
    filename='logs/classification_si_8.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_model(model_class, X_train, X_test, y_train, y_test, model_name):
    """
    Обучает модель с оптимизацией гиперпараметров через Optuna.

    Args:
        model_class: Класс модели (RandomForestClassifier и т.д.).
        X_train, X_test: Признаки для обучения и теста.
        y_train, y_test: Целевые переменные.
        model_name: Имя модели ('RF', 'XGB', 'LGB', 'GB', 'LR').

    Returns:
        tuple: (model, accuracy, f1, precision, recall, y_pred) или (None, 0, 0, 0, 0, None) при ошибке.
    """
    # Определяем функцию оптимизации гиперпараметров с использованием Optuna, чтобы максимизировать точность и улучшить качество классификации log_si
    def objective(trial):
        try:
            if model_name == 'RF':
                # Настраиваем параметры RandomForest для классификации log_si, включая количество деревьев и ограничения на их сложность
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_categorical('max_depth', [10, 20, 30, None]),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
                }
                model = RandomForestClassifier(**params, random_state=42, n_jobs=1)
            elif model_name == 'XGB':
                # Оптимизируем XGBoost, добавляя параметры регуляризации для предотвращения переобучения
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'nthread': 1  # Ограничиваем многопоточность
                }
                model = XGBClassifier(**params, random_state=42, eval_metric='logloss')
            elif model_name == 'LGB':
                # Настраиваем LightGBM, уделяя внимание количеству листьев и минимальному числу образцов в листе
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 70),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                    'n_jobs': 1  # Ограничиваем многопоточность
                }
                model = LGBMClassifier(**params, random_state=42, verbose=-1, force_row_wise=True)
            elif model_name == 'GB':
                # Оптимизируем GradientBoosting, фокусируясь на глубине деревьев и скорости обучения
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
                }
                model = GradientBoostingClassifier(**params, random_state=42)
            else:  # LR
                # Настраиваем логистическую регрессию, оптимизируя параметр регуляризации C
                params = {
                    'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
                    'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
                }
                model = LogisticRegression(**params, random_state=42, max_iter=1000, n_jobs=1)
            
            # Используем 5-кратную кросс-валидацию для оценки модели по точности
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            return scores.mean()
        except Exception as e:
            # Логируем любые ошибки, возникшие во время оптимизации гиперпараметров
            logging.error(f"Ошибка оптимизации {model_name}: {e}")
            return float('-inf')

    # Создаем исследование Optuna для поиска оптимальных гиперпараметров, максимизирующих точность
    study = optuna.create_study(direction='maximize')
    try:
        # Выполняем 20 итераций оптимизации в однопоточном режиме
        study.optimize(objective, n_trials=20, n_jobs=1)
        best_params = study.best_params
        logging.info(f"Лучшие параметры для {model_name}: {best_params}")
    except Exception as e:
        # Фиксируем любые ошибки, возникшие в процессе оптимизации Optuna
        logging.error(f"Ошибка оптимизации Optuna для {model_name}: {e}")
        return None, 0, 0, 0, 0, None

    # Инициализируем модель с лучшими найденными параметрами
    if model_name == 'RF':
        model = RandomForestClassifier(**best_params, random_state=42, n_jobs=1)
    elif model_name == 'XGB':
        best_params['nthread'] = 1
        model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
    elif model_name == 'LGB':
        best_params['n_jobs'] = 1
        model = LGBMClassifier(**best_params, random_state=42, verbose=-1, force_row_wise=True)
    elif model_name == 'GB':
        model = GradientBoostingClassifier(**best_params, random_state=42)
    else:
        model = LogisticRegression(**best_params, random_state=42, max_iter=1000, n_jobs=1)

    try:
        # Обучаем модель на обучающей выборке
        model.fit(X_train, y_train)
        # Делаем предсказания на тестовой выборке
        y_pred = model.predict(X_test)

        # Проверяем предсказания на наличие NaN или бесконечных значений
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            logging.error(f"Недопустимые предсказания для {model_name}")
            return None, 0, 0, 0, 0, None

        # Вычисляем метрики качества: Accuracy, F1, Precision, Recall
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Оцениваем модель на обучающей выборке
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        logging.info(f"Обучающая точность для {model_name}: {train_accuracy:.3f}")

        # Логируем метрики для анализа
        logging.info(f"Тестовые метрики {model_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
        return model, accuracy, f1, precision, recall, y_pred
    except Exception as e:
        # Фиксируем любые ошибки, возникшие при обучении модели
        logging.error(f"Ошибка обучения {model_name}: {e}")
        return None, 0, 0, 0, 0, None

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Строим матрицу ошибок для оценки качества классификации."""
    try:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Предсказанный класс')
        plt.ylabel('Истинный класс')
        plt.title(f'Матрица ошибок (log_si, {model_name})')
        plt.savefig(f'figures/confusion_matrix_log_si_{model_name.lower()}.png')
        plt.close()
        logging.info(f"Матрица ошибок сохранена: figures/confusion_matrix_log_si_{model_name.lower()}.png")
    except Exception as e:
        logging.error(f"Ошибка при создании матрицы ошибок: {e}")

def main():
    """Основная функция для классификации log_si."""
    # Начинаем выполнение основной функции и логируем начало работы
    logging.info("Загрузка данных")
    try:
        # Загружаем предварительно обработанные данные
        data = pd.read_csv('data/processed_data.csv')
    except FileNotFoundError as e:
        # Обрабатываем случай, если файл данных отсутствует
        logging.error(f"Файл data/processed_data.csv не найден: {e}")
        print(f"Ошибка: Файл data/processed_data.csv не найден!")
        exit(1)

    # Преобразуем log_si в бинарные классы (0 и 1) с порогом SI > 8
    if 'log_si' in data.columns:
        threshold = np.log10(8)  # Порог SI > 8 в логарифмической шкале
        data['log_si'] = (data['log_si'] >= threshold).astype(int)
        logging.info(f"Целевая переменная log_si преобразована в бинарные классы с порогом {threshold}, распределение: {Counter(data['log_si'])}")
    else:
        logging.error("Колонка log_si отсутствует")
        print("Ошибка: Колонка log_si отсутствует!")
        exit(1)

    # Проверяем распределение классов
    class_counts = Counter(data['log_si'])
    if len(class_counts) < 2 or min(class_counts.values()) == 0:
        logging.error("Один из классов пустой или отсутствует")
        print("Ошибка: Один из классов пустой или отсутствует!")
        exit(1)

    # Выбираем признаки, исключая целевые и логарифмированные переменные
    features = [col for col in data.columns if col not in ['IC50_mM', 'CC50_mM', 'SI', 'log_ic50', 'log_cc50', 'log_si']]
    X = data[features]
    y = data['log_si']

    # Проверяем данные на наличие пропусков или бесконечных значений
    if X.isnull().values.any() or np.any(np.isinf(X.values)):
        logging.error("NaN или бесконечные значения в признаках")
        print("Ошибка: NaN или бесконечные значения в признаках!")
        exit(1)
    if y.isnull().any() or np.any(np.isinf(y.values)):
        logging.error("NaN или бесконечные значения в log_si")
        print("Ошибка: NaN или бесконечные значения в log_si!")
        exit(1)

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Разделение данных: train={X_train.shape[0]}, test={X_test.shape[0]}")

    # Стандартизируем признаки
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info("Признаки стандартизированы")
    except Exception as e:
        logging.error(f"Ошибка масштабирования признаков: {e}")
        print(f"Ошибка масштабирования признаков: {e}")
        exit(1)

    # Определяем список моделей для классификации
    models = [
        (RandomForestClassifier, 'RF'),
        (XGBClassifier, 'XGB'),
        (LGBMClassifier, 'LGB'),
        (GradientBoostingClassifier, 'GB'),
        (LogisticRegression, 'LR')
    ]

    results = []
    best_model = None
    best_accuracy = -float('inf')
    best_y_pred = None
    best_model_name = None

    # Обучаем каждую модель и сохраняем результаты
    for model_class, model_name in models:
        logging.info(f"Обучение {model_name}")
        model, accuracy, f1, precision, recall, y_pred = train_model(
            model_class, X_train_scaled, X_test_scaled, y_train, y_test, model_name
        )
        if model is None:
            continue
        # Сохраняем метрики модели (Accuracy, F1, Precision, Recall)
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'F1': f1,
            'Precision': precision,
            'Recall': recall
        })

        # Визуализируем важность признаков для моделей, поддерживающих эту функцию
        if hasattr(model, 'feature_importances_'):
            try:
                importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
                importance = importance.sort_values('Importance', ascending=False).head(10)
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance)
                plt.title(f'Важность признаков (log_si, {model_name})')
                plt.savefig(f'figures/feature_importance_log_si_{model_name.lower()}.png')
                plt.close()
                logging.info(f"Важность признаков сохранена: figures/feature_importance_log_si_{model_name.lower()}.png")
            except Exception as e:
                logging.error(f"Ошибка сохранения важности признаков для {model_name}: {e}")

        # Строим матрицу ошибок для оценки качества классификации
        if y_pred is not None:
            plot_confusion_matrix(y_test, y_pred, model_name)

        # Сохраняем лучшую модель по метрике Accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_y_pred = y_pred
            best_model_name = model_name

    # Обучаем ансамблевую модель (VotingClassifier)
    logging.info("Обучение ансамбля")
    estimators = [
        ('rf', RandomForestClassifier(random_state=42, n_jobs=1)),
        ('xgb', XGBClassifier(random_state=42, eval_metric='logloss', nthread=1)),
        ('lgb', LGBMClassifier(random_state=42, verbose=-1, force_row_wise=True, n_jobs=1)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ]
    voting_model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)
    try:
        # Обучаем ансамбль и делаем предсказания
        voting_model.fit(X_train_scaled, y_train)
        y_pred_voting = voting_model.predict(X_test_scaled)

        # Проверяем предсказания ансамбля
        if np.any(np.isnan(y_pred_voting)) or np.any(np.isinf(y_pred_voting)):
            logging.error("Недопустимые предсказания для ансамбля")
        else:
            # Вычисляем метрики ансамбля
            accuracy_voting = accuracy_score(y_test, y_pred_voting)
            f1_voting = f1_score(y_test, y_pred_voting)
            precision_voting = precision_score(y_test, y_pred_voting)
            recall_voting = recall_score(y_test, y_pred_voting)
            results.append({
                'Model': 'Voting',
                'Accuracy': accuracy_voting,
                'F1': f1_voting,
                'Precision': precision_voting,
                'Recall': recall_voting
            })
            logging.info(f"Метрики ансамбля: Accuracy={accuracy_voting:.3f}, F1={f1_voting:.3f}, Precision={precision_voting:.3f}, Recall={recall_voting:.3f}")
            plot_confusion_matrix(y_test, y_pred_voting, 'Voting')
    except Exception as e:
        # Фиксируем любые ошибки, связанные с обучением ансамбля
        logging.error(f"Ошибка обучения ансамбля: {e}")

    # Визуализируем матрицу ошибок для лучшей модели
    if best_y_pred is not None:
        try:
            plot_confusion_matrix(y_test, best_y_pred, best_model_name)
        except Exception as e:
            logging.error(f"Ошибка сохранения матрицы ошибок для {best_model_name}: {e}")

    # Сохраняем результаты классификации в CSV-файл
    try:
        results_df = pd.DataFrame(results)
        results_df = results_df.round(3)
        results_df.to_csv('results/classification_si_8.csv', index=False)
        logging.info("Результаты классификации log_si:\n" + results_df.to_string())
        print("Результаты классификации log_si:")
        print(results_df)
    except Exception as e:
        logging.error(f"Ошибка сохранения результатов: {e}")
        print(f"Ошибка сохранения результатов: {e}")

if __name__ == "__main__":
    # Запускаем основную функцию для выполнения классификации log_si
    main()