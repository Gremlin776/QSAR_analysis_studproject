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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna

# Проверяем наличие всех необходимых библиотек, чтобы обеспечить выполнение регрессии log_si, которая оценивает селективность соединений против вируса гриппа
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
    filename='logs/regression_log_si.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_model(model_class, X_train, X_test, y_train, y_test, model_name):
    """
    Обучает модель с оптимизацией гиперпараметров через Optuna.

    Args:
        model_class: Класс модели (RandomForestRegressor и т.д.).
        X_train, X_test: Признаки для обучения и теста.
        y_train, y_test: Целевые переменные.
        model_name: Имя модели ('RF', 'XGB', 'LGB', 'GB', 'LR').

    Returns:
        tuple: (model, mse, r2, mae, y_pred) или (None, 0, 0, 0, None) при ошибке.
    """
    # Определяем функцию оптимизации гиперпараметров с использованием Optuna, чтобы минимизировать среднеквадратичную ошибку и улучшить качество предсказаний log_si
    def objective(trial):
        try:
            if model_name == 'RF':
                # Настраиваем параметры RandomForest для регрессии log_si, включая количество деревьев и ограничения на их сложность, чтобы сбалансировать точность и обобщающую способность
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_categorical('max_depth', [10, 20, 30, None]),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
                }
                # Используем однопоточный режим для стабильности в Optuna
                model = RandomForestRegressor(**params, random_state=42, n_jobs=1)  
            elif model_name == 'XGB':
                # Оптимизируем XGBoost, добавляя параметры регуляризации (reg_lambda, reg_alpha) для предотвращения переобучения и улучшения предсказаний
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0)
                }
                model = XGBRegressor(**params, random_state=42)
            elif model_name == 'LGB':
                # Настраиваем LightGBM, уделяя внимание количеству листьев и минимальному числу образцов в листе, чтобы оптимизировать производительность
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 70),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 100)
                }
                model = LGBMRegressor(**params, random_state=42, verbose=-1, force_row_wise=True)
            elif model_name == 'GB':
                # Оптимизируем GradientBoosting, фокусируясь на глубине деревьев и скорости обучения для достижения высокой точности
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
                }
                model = GradientBoostingRegressor(**params, random_state=42)
            else:  
                # Настраиваем линейную регрессию, оптимизируя только параметр перехвата, так как модель простая
                params = {
                    'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
                }
                model = LinearRegression(**params)
            
            # Используем 5-кратную кросс-валидацию для оценки модели, чтобы обеспечить надежность результатов и избежать переобучения
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            return -scores.mean()
        except Exception as e:
            # Логируем любые ошибки, возникшие во время оптимизации гиперпараметров, для последующего анализа
            logging.error(f"Ошибка оптимизации {model_name}: {e}")
            return float('inf')

    # Создаем исследование Optuna для поиска оптимальных гиперпараметров, минимизирующих ошибку
    study = optuna.create_study(direction='minimize')
    try:
        # Выполняем 50 итераций оптимизации в однопоточном режиме, чтобы избежать конфликтов при работе с некоторыми моделями
        study.optimize(objective, n_trials=50, n_jobs=1)
        best_params = study.best_params
        logging.info(f"Лучшие параметры для {model_name}: {best_params}")
    except Exception as e:
        # Фиксируем любые ошибки, возникшие в процессе оптимизации Optuna, чтобы предотвратить сбои
        logging.error(f"Ошибка оптимизации Optuna для {model_name}: {e}")
        return None, 0, 0, 0, None

    # Инициализируем модель с лучшими найденными параметрами для последующего обучения
    if model_name == 'RF':
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=1)
    elif model_name == 'XGB':
        model = XGBRegressor(**best_params, random_state=42)
    elif model_name == 'LGB':
        model = LGBMRegressor(**best_params, random_state=42, verbose=-1, force_row_wise=True)
    elif model_name == 'GB':
        model = GradientBoostingRegressor(**best_params, random_state=42)
    else:
        model = LinearRegression(**best_params)

    try:
        # Обучаем модель на обучающей выборке, чтобы она могла предсказывать значения log_si
        model.fit(X_train, y_train)
        # Делаем предсказания на тестовой выборке для оценки качества модели
        y_pred = model.predict(X_test)

        # Проверяем предсказания на наличие NaN или бесконечных значений, чтобы избежать некорректных результатов
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            logging.error(f"Недопустимые предсказания для {model_name}")
            return None, 0, 0, 0, None

        # Вычисляем метрики качества: MSE, R2 и MAE, чтобы оценить, насколько точно модель предсказывает log_si
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Оцениваем модель на обучающей выборке, чтобы проверить наличие переобучения
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        logging.info(f"Обучающий R2 для {model_name}: {train_r2:.3f}")

        # Логируем метрики для последующего анализа и сравнения моделей
        logging.info(f"Тестовые метрики {model_name}: MSE={mse:.3f}, R2={r2:.3f}, MAE={mae:.3f}")
        return model, mse, r2, mae, y_pred
    except Exception as e:
        # Фиксируем любые ошибки, возникшие при обучении модели, чтобы обеспечить надежность процесса
        logging.error(f"Ошибка обучения {model_name}: {e}")
        return None, 0, 0, 0, None

def main():
    """Основная функция для регрессии log_si."""
    # Начинаем выполнение основной функции и логируем начало работы для отслеживания
    logging.info("Загрузка данных")
    try:
        # Загружаем предварительно обработанные данные из файла, созданного в процессе EDA
        data = pd.read_csv('data/processed_data.csv')
    except FileNotFoundError as e:
        # Обрабатываем случай, если файл данных отсутствует, чтобы предотвратить сбой
        logging.error(f"Файл data/processed_data.csv не найден: {e}")
        print(f"Ошибка: Файл data/processed_data.csv не найден!")
        exit(1)

    # Проверяем наличие колонки log_si, которая является целевой переменной для регрессии
    if 'log_si' not in data.columns:
        logging.error("Колонка log_si отсутствует")
        print("Ошибка: Колонка log_si отсутствует!")
        exit(1)

    # Выбираем признаки, исключая целевые и логарифмированные переменные, чтобы использовать только релевантные данные
    features = [col for col in data.columns if col not in ['IC50_mM', 'CC50_mM', 'SI', 'log_ic50', 'log_cc50', 'log_si']]
    X = data[features]
    y = data['log_si']  

    # Проверяем данные на наличие пропусков или бесконечных значений, чтобы гарантировать качество входных данных
    if X.isnull().values.any() or np.any(np.isinf(X.values)):
        logging.error("NaN или бесконечные значения в признаках")
        print("Ошибка: NaN или бесконечные значения в признаках!")
        exit(1)
    if y.isnull().any() or np.any(np.isinf(y.values)):
        logging.error("NaN или бесконечные значения в log_si")
        print("Ошибка: NaN или бесконечные значения в log_si!")
        exit(1)

    # Разделяем данные на обучающую и тестовую выборки в соотношении 80/20 для последующего обучения и тестирования моделей
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Разделение данных: train={X_train.shape[0]}, test={X_test.shape[0]}")

    # Стандартизируем признаки, чтобы привести их к единому масштабу, что улучшает производительность моделей
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info("Признаки стандартизированы")
    except Exception as e:
        # Фиксируем ошибки, связанные с масштабированием данных, чтобы предотвратить сбои
        logging.error(f"Ошибка масштабирования признаков: {e}")
        print(f"Ошибка масштабирования признаков: {e}")
        exit(1)

    # Определяем список моделей, которые будут обучаться для сравнения их эффективности
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

    # Обучаем каждую модель и сохраняем результаты для последующего анализа
    for model_class, model_name in models:
        logging.info(f"Обучение {model_name}")
        model, mse, r2, mae, y_pred = train_model(
            model_class, X_train_scaled, X_test_scaled, y_train, y_test, model_name
        )
        if model is None:
            continue
        # Сохраняем метрики модели (MSE, R2, MAE) для сравнения производительности
        results.append({
            'Model': model_name,
            'MSE': mse,
            'R2': r2,
            'MAE': mae
        })

        # Визуализируем важность признаков для моделей, поддерживающих эту функцию, чтобы понять, какие молекулярные дескрипторы наиболее влияют на log_si
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

        # Сохраняем лучшую модель по метрике R2 для дальнейшей визуализации и анализа
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_y_pred = y_pred
            best_model_name = model_name

    # Обучаем ансамблевую модель, комбинирующую несколько алгоритмов, чтобы потенциально улучшить предсказания
    logging.info("Обучение ансамбля")
    estimators = [
        ('rf', RandomForestRegressor(random_state=42, n_jobs=1)),
        ('xgb', XGBRegressor(random_state=42)),
        ('lgb', LGBMRegressor(random_state=42, verbose=-1, force_row_wise=True)),
        ('gb', GradientBoostingRegressor(random_state=42))
    ]
    voting_model = VotingRegressor(estimators=estimators, n_jobs=1)
    try:
        # Обучаем ансамбль и делаем предсказания на тестовой выборке
        voting_model.fit(X_train_scaled, y_train)
        y_pred_voting = voting_model.predict(X_test_scaled)

        # Проверяем предсказания ансамбля на валидность
        if np.any(np.isnan(y_pred_voting)) or np.any(np.isinf(y_pred_voting)):
            logging.error("Недопустимые предсказания для ансамбля")
        else:
            # Вычисляем метрики ансамбля (MSE, R2, MAE) и добавляем их в результаты
            mse_voting = mean_squared_error(y_test, y_pred_voting)
            r2_voting = r2_score(y_test, y_pred_voting)
            mae_voting = mean_absolute_error(y_test, y_pred_voting)
            results.append({
                'Model': 'Voting',
                'MSE': mse_voting,
                'R2': r2_voting,
                'MAE': mae_voting
            })
            logging.info(f"Метрики ансамбля: MSE={mse_voting:.3f}, R2={r2_voting:.3f}, MAE={mae_voting:.3f}")
    except Exception as e:
        # Фиксируем любые ошибки, связанные с обучением ансамбля
        logging.error(f"Ошибка обучения ансамбля: {e}")

    # Визуализируем результаты лучшей модели, чтобы оценить качество предсказаний
    if best_y_pred is not None:
        try:
            # Создаем график рассеяния истинных и предсказанных значений log_si, чтобы визуально оценить точность модели
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, best_y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel('Истинный log_si')
            plt.ylabel('Предсказанный log_si')
            plt.title(f'Предсказания vs Истинные значения (log_si, {best_model_name})')
            plt.savefig(f'figures/pred_vs_true_log_si_{best_model_name.lower()}.png')
            plt.close()
            logging.info(f"График предсказаний сохранен: figures/pred_vs_true_log_si_{best_model_name.lower()}.png")
        except Exception as e:
            logging.error(f"Ошибка сохранения графика предсказаний для {best_model_name}: {e}")

    # Сохраняем результаты регрессии в CSV-файл для дальнейшего анализа и отчетности
    try:
        results_df = pd.DataFrame(results)
        results_df = results_df.round(3)
        results_df.to_csv('results/regression_log_si.csv', index=False)
        logging.info("Результаты регрессии log_si:\n" + results_df.to_string())
        print("Результаты регрессии log_si:")
        print(results_df)
    except Exception as e:
        logging.error(f"Ошибка сохранения результатов: {e}")
        print(f"Ошибка сохранения результатов: {e}")

if __name__ == "__main__":
    # Запускаем основную функцию для выполнения регрессии log_si
    main()