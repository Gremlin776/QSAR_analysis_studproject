import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import importlib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna

# Проверяем наличие всех необходимых библиотек, чтобы обеспечить корректное выполнение анализа и моделирования для регрессии log_cc50
required_libs = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'xgboost', 'lightgbm', 'optuna']
for lib in required_libs:
    if importlib.util.find_spec(lib) is None:
        print(f"Error: Library {lib} not found! Please install it.")
        exit(1)

# Создаем или проверяем существование директорий для хранения логов, графиков и результатов, чтобы организовать выходные данные
try:
    for directory in ['logs', 'figures', 'results']:
        os.makedirs(directory, exist_ok=True)
except PermissionError as e:
    print(f"Error: Cannot create directories: {e}")
    exit(1)

# Настраиваем логирование для записи всех действий и ошибок, чтобы отслеживать выполнение скрипта и сохранять информацию для анализа
logging.basicConfig(
    filename='logs/regression_log_cc50.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_model(model_class, X_train, X_test, y_train, y_test, model_name):
    """
    Обучает модель с оптимизацией гиперпараметров через Optuna и возвращает метрики.

    Args:
        model_class: Класс модели (например, RandomForestRegressor).
        X_train, X_test: Обучающие и тестовые признаки.
        y_train, y_test: Обучающие и тестовые целевые переменные.
        model_name (str): Имя модели ('RF', 'XGB', 'LGB', 'GB', 'LR').

    Returns:
        tuple: (model, mse, r2, mae, y_pred) или (None, 0, 0, 0, None) при ошибке.
    """
    # Определяем функцию для оптимизации гиперпараметров с использованием Optuna, чтобы минимизировать среднеквадратичную ошибку (MSE)
    def objective(trial):
        try:
            if model_name == 'RF':
                # Настраиваем параметры RandomForest для регрессии log_cc50, включая количество деревьев и ограничения на их сложность
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_categorical('max_depth', [10, 20, 30, None]),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
                }
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            elif model_name == 'XGB':
                # Оптимизируем XGBoost, добавляя параметры регуляризации для предотвращения переобучения
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0)
                }
                model = XGBRegressor(**params, random_state=42)
            elif model_name == 'LGB':
                # Настраиваем LightGBM, уделяя внимание количеству листьев и минимальному числу образцов
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 70),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 100)
                }
                model = LGBMRegressor(**params, random_state=42, verbose=-1, force_row_wise=True)
            elif model_name == 'GB':
                # Оптимизируем GradientBoosting, фокусируясь на глубине деревьев и скорости обучения
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
                }
                model = GradientBoostingRegressor(**params, random_state=42)
            else:  # LR
                # Настраиваем линейную регрессию, оптимизируя параметр перехвата
                params = {
                    'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
                }
                model = LinearRegression(**params)
            
            # Используем кросс-валидацию для оценки модели, чтобы обеспечить надежность результатов
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            return -scores.mean()
        except Exception as e:
            # Логируем любые ошибки, возникшие во время оптимизации гиперпараметров
            logging.error(f"Optimization error for {model_name}: {e}")
            return float('inf')

    # Создаем исследование Optuna для поиска оптимальных гиперпараметров
    study = optuna.create_study(direction='minimize')
    try:
        # Выполняем 50 итераций оптимизации в многопоточном режиме для ускорения
        study.optimize(objective, n_trials=50, n_jobs=-1)
        best_params = study.best_params
        logging.info(f"Best params for {model_name}: {best_params}")
    except Exception as e:
        # Фиксируем любые ошибки, возникшие в процессе оптимизации
        logging.error(f"Optuna optimization failed for {model_name}: {e}")
        return None, 0, 0, 0, None

    # Инициализируем модель с лучшими найденными параметрами
    if model_name == 'RF':
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    elif model_name == 'XGB':
        model = XGBRegressor(**best_params, random_state=42)
    elif model_name == 'LGB':
        model = LGBMRegressor(**best_params, random_state=42, verbose=-1, force_row_wise=True)
    elif model_name == 'GB':
        model = GradientBoostingRegressor(**best_params, random_state=42)
    else:
        model = LinearRegression(**best_params)

    try:
        # Обучаем модель на обучающей выборке
        model.fit(X_train, y_train)
        # Делаем предсказания на тестовой выборке
        y_pred = model.predict(X_test)

        # Проверяем предсказания на наличие недопустимых значений
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            logging.error(f"Invalid predictions for {model_name}")
            return None, 0, 0, 0, None

        # Вычисляем метрики качества модели: MSE, R2 и MAE
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Оцениваем модель на обучающей выборке для проверки переобучения
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        logging.info(f"Train R2 for {model_name}: {train_r2:.3f}")

        # Логируем метрики для последующего анализа
        logging.info(f"Test Metrics {model_name}: MSE={mse:.3f}, R2={r2:.3f}, MAE={mae:.3f}")
        return model, mse, r2, mae, y_pred
    except Exception as e:
        # Фиксируем любые ошибки, возникшие при обучении модели
        logging.error(f"Error training {model_name}: {e}")
        return None, 0, 0, 0, None

def main():
    """Основная функция для регрессии log_cc50."""
    # Начинаем выполнение основной функции и логируем начало работы
    logging.info("Loading data")
    try:
        # Загружаем предварительно обработанные данные из файла
        data = pd.read_csv('data/processed_data.csv')
    except FileNotFoundError as e:
        # Обрабатываем случай, если файл данных отсутствует
        logging.error(f"File data/processed_data.csv not found: {e}")
        print(f"Error: File data/processed_data.csv not found!")
        exit(1)

    # Проверяем наличие колонки log_cc50, которая критически важна для регрессии
    if 'log_cc50' not in data.columns:
        logging.error("Column log_cc50 missing")
        print("Error: Column log_cc50 missing!")
        exit(1)

    # Выбираем признаки, исключая целевые и логарифмированные переменные
    features = [col for col in data.columns if col not in ['IC50_mM', 'CC50_mM', 'SI', 'log_ic50', 'log_cc50', 'log_si']]
    X = data[features]
    y = data['log_cc50']

    # Проверяем данные на наличие пропусков или бесконечных значений
    if X.isnull().values.any() or np.any(np.isinf(X.values)):
        logging.error("NaN or infinite values in features")
        print("Error: NaN or infinite values in features!")
        exit(1)
    if y.isnull().any() or np.any(np.isinf(y.values)):
        logging.error("NaN or infinite values in log_cc50")
        print("Error: NaN or infinite values in log_cc50!")
        exit(1)

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Data split: train={X_train.shape[0]}, test={X_test.shape[0]}")

    # Стандартизируем признаки для улучшения работы моделей
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info("Features standardized")
    except Exception as e:
        # Фиксируем ошибки, связанные с масштабированием данных
        logging.error(f"Error in feature scaling: {e}")
        print(f"Error in feature scaling: {e}")
        exit(1)

    # Определяем список моделей для обучения
    models = [
        (RandomForestRegressor, 'RF'),
        (XGBRegressor, 'XGB'),
        (LGBMRegressor, 'LGB'),
        (GradientBoostingRegressor, 'GB'),
        (LinearRegression, 'LR')
    ]

    results = []
    best_model = None
    best_r2 = -float('inf')
    best_y_pred = None
    best_model_name = None

    # Обучаем каждую модель и сохраняем результаты
    for model_class, model_name in models:
        logging.info(f"Training {model_name}")
        model, mse, r2, mae, y_pred = train_model(
            model_class, X_train_scaled, X_test_scaled, y_train, y_test, model_name
        )
        if model is None:
            continue
        # Сохраняем метрики модели для последующего анализа
        results.append({
            'Model': model_name,
            'MSE': mse,
            'R2': r2,
            'MAE': mae
        })

        # Визуализируем важность признаков для моделей, поддерживающих эту функцию
        if hasattr(model, 'feature_importances_'):
            try:
                importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
                importance = importance.sort_values('Importance', ascending=False).head(10)
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance)
                plt.title(f'Feature Importance (log_cc50, {model_name})')
                plt.savefig(f'figures/feature_importance_log_cc50_{model_name.lower()}.png')
                plt.close()
                logging.info(f"Feature importance saved: figures/feature_importance_log_cc50_{model_name.lower()}.png")
            except Exception as e:
                logging.error(f"Error saving feature importance for {model_name}: {e}")

        # Сохраняем лучшую модель по R2
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_y_pred = y_pred
            best_model_name = model_name

    # Обучаем ансамблевую модель, комбинирующую несколько алгоритмов
    logging.info("Training voting regressor")
    estimators = [
        ('rf', RandomForestRegressor(random_state=42, n_jobs=-1)),
        ('xgb', XGBRegressor(random_state=42)),
        ('lgb', LGBMRegressor(random_state=42, verbose=-1, force_row_wise=True)),
        ('gb', GradientBoostingRegressor(random_state=42))
    ]
    voting_model = VotingRegressor(estimators=estimators, n_jobs=-1)
    try:
        # Обучаем ансамбль и делаем предсказания
        voting_model.fit(X_train_scaled, y_train)
        y_pred_voting = voting_model.predict(X_test_scaled)

        # Проверяем предсказания ансамбля на валидность
        if np.any(np.isnan(y_pred_voting)) or np.any(np.isinf(y_pred_voting)):
            logging.error("Invalid predictions for voting regressor")
        else:
            # Вычисляем и сохраняем метрики ансамбля
            mse_voting = mean_squared_error(y_test, y_pred_voting)
            r2_voting = r2_score(y_test, y_pred_voting)
            mae_voting = mean_absolute_error(y_test, y_pred_voting)
            results.append({
                'Model': 'Voting',
                'MSE': mse_voting,
                'R2': r2_voting,
                'MAE': mae_voting
            })
            logging.info(f"Voting metrics: MSE={mse_voting:.3f}, R2={r2_voting:.3f}, MAE={mae_voting:.3f}")
    except Exception as e:
        # Фиксируем любые ошибки, связанные с обучением ансамбля
        logging.error(f"Error training voting regressor: {e}")

    # Визуализируем результаты лучшей модели
    if best_y_pred is not None:
        try:
            # Создаем и сохраняем график предсказанных значений против истинных для анализа качества модели
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, best_y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel('True log_cc50')
            plt.ylabel('Predicted log_cc50')
            plt.title(f'Predictions vs True Values (log_cc50, {best_model_name})')
            plt.savefig(f'figures/pred_vs_true_log_cc50_{best_model_name.lower()}.png')
            plt.close()
            logging.info(f"Prediction plot saved: figures/pred_vs_true_log_cc50_{best_model_name.lower()}.png")
        except Exception as e:
            # Фиксируем любые ошибки, связанные с визуализацией
            logging.error(f"Error saving prediction plot for {best_model_name}: {e}")

    # Сохраняем результаты регрессии в CSV-файл для дальнейшего анализа
    try:
        results_df = pd.DataFrame(results)
        results_df = results_df.round(3)
        results_df.to_csv('results/regression_log_cc50.csv', index=False)
        logging.info("Regression results log_cc50:\n" + results_df.to_string())
        print("Regression results log_cc50:")
        print(results_df)
    except Exception as e:
        # Фиксируем любые ошибки, связанные с сохранением результатов
        logging.error(f"Error saving results: {e}")
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    # Запускаем основную функцию для выполнения регрессии log_cc50
    main()