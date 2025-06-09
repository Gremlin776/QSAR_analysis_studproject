# QSAR-анализ соединений против вируса гриппа

Этот проект выполняет количественный анализ структуры и активности (QSAR) для 1000 химических соединений, чтобы предсказать их противовирусную активность, цитотоксичность и селективность против вируса гриппа. Проект включает регрессионные и классификационные модели машинного обучения, визуализации данных и автоматическую генерацию отчёта в формате Markdown и PDF.

## Описание проекта

Проект решает следующие задачи:
- **Предобработка данных**: Анализ и очистка данных из файла `data/coursework_data.xlsx`, создание обработанного файла `data/processed_data.csv`.
- **Регрессия**: Предсказание логарифмированных значений:
  - `log_ic50` (противовирусная активность, IC50_mM).
  - `log_cc50` (цитотоксичность, CC50_mM).
  - `log_si` (индекс селективности, SI).
- **Классификация**: Бинарная классификация:
  - `IC50_mM`, `CC50_mM`, `SI` по медианным значениям.
  - `SI > 8` (в логарифмической шкале: `log_si >= log10(8)`).
- **Модели**: Используются `RandomForest`, `XGBoost`, `LightGBM`, `GradientBoosting`, `LinearRegression`/`LogisticRegression` и ансамбли (`VotingRegressor`/`VotingClassifier`).
- **Оптимизация**: Гиперпараметры моделей оптимизируются с помощью `Optuna`.
- **Визуализации**: Гистограммы, боксплоты, корреляции (EDA); графики важности признаков, предсказания против истинных значений (регрессия); матрицы ошибок, ROC-кривые (классификация, кроме `SI > 8`).
- **Отчёт**: Генерируется в формате `results/report.md` и конвертируется в PDF (`results/report_[timestamp].pdf`).

## Требования

- **Python**: Версия 3.12.
- **Библиотеки Python**: Указаны в `requirements.txt`.
- **Системные зависимости** (для `weasyprint` на Ubuntu):
  ```bash
  sudo apt-get install libcairo2 libpango-1.0-0
  ```
- **Входной файл**: `data/coursework_data.xlsx` (должен быть в папке `data/`).

**Примечание**: Некоторые библиотеки (например, `weasyprint`) требуют системных пакетов. Если используется серверная среда, убедитесь, что `matplotlib` настроен на бэкенд `Agg`. Устаревший модуль `pkg_resources` может вызывать предупреждения; рекомендуется использовать `importlib.metadata` в будущем.

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone <URL_репозитория>
   cd qsar_an1
   ```
2. Создайте и активируйте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
4. Установите системные библиотеки (для Ubuntu):
   ```bash
   sudo apt-get install libcairo2 libpango-1.0-0
   ```
5. Убедитесь, что файл `data/coursework_data.xlsx` находится в папке `data/`.

## Структура проекта

- **data/**: Хранит входные и обработанные данные.
  - `coursework_data.xlsx`: Исходные данные.
  - `processed_data.csv`: Обработанные данные (создаётся `eda.py`).
- **scripts/**: Скрипты Python.
  - `eda.py`: Предобработка и анализ данных.
  - `regression_log_ic50.py`, `regression_log_cc50.py`, `regression_log_si.py`: Регрессия.
  - `classification_cc50_median.py`, `classification_ic50_median.py`, `classification_si_median.py`, `classification_si_8.py`: Классификация.
  - `generate_report.py`: Генерация отчёта в Markdown.
  - `convert_to_pdf.py`: Конвертация отчёта в PDF.
- **results/**: Результаты моделей и отчёты.
  - `regression_*.csv`, `classification_*.csv`: Метрики моделей.
  - `report.md`, `report_[timestamp].pdf`: Отчёт.
- **figures/**: Визуализации.
  - `histograms.png`, `boxplots.png`, `correlation.png`: Результаты EDA.
  - `feature_importance_*.png`, `pred_vs_true_*.png`: Регрессия.
  - `confusion_matrix_*.png`, `roc_curve_*.png`, `feature_importance_*.png`: Классификация.
- **logs/**: Лог-файлы для каждого скрипта (`eda.log`, `regression_*.log`, `classification_*.log`, и т.д.).
- **main.py**: Основной скрипт для запуска всех этапов.
- **requirements.txt**: Список зависимостей.
- **README_rus.md**, **README_eng.md**: Документация.

## Использование

1. Убедитесь, что `data/coursework_data.xlsx` находится в папке `data/`.
2. Запустите основной скрипт:
   ```bash
   python main.py
   ```
   - Для продолжения выполнения при ошибках используйте флаг:
     ```bash
     python main.py --continue-on-error
     ```
3. Последовательность выполнения:
   - `eda.py`: Обработка данных, создание `processed_data.csv`, визуализации (`figures/histograms.png`, `boxplots.png`, `correlation.png`).
   - Регрессия: Обучение моделей для `log_ic50`, `log_cc50`, `log_si`, сохранение метрик (`results/regression_*.csv`) и визуализаций (`figures/feature_importance_*.png`, `pred_vs_true_*.png`).
   - Классификация: Обучение моделей для `IC50_mM`, `CC50_mM`, `SI` (по медиане) и `SI > 8`, сохранение метрик (`results/classification_*.csv`) и визуализаций (`figures/confusion_matrix_*.png`, `roc_curve_*.png`, `feature_importance_*.png`).
   - `generate_report.py`: Создание отчёта `results/report.md`.
   - `convert_to_pdf.py`: Конвертация отчёта в `results/report_[timestamp].pdf`.

## Результаты

- **CSV-файлы** (`results/`):
  - Регрессия: `regression_log_ic50.csv`, `regression_log_cc50.csv`, `regression_log_si.csv` (метрики: `MSE`, `R2`, `MAE`).
  - Классификация: `classification_cc50_median.csv`, `classification_ic50_median.csv`, `classification_si_median.csv`, `classification_si_8.csv` (метрики: `Accuracy`, `F1`, `Precision`, `Recall`, `ROC_AUC`, `PR_AUC`, кроме `SI > 8`).
- **Визуализации** (`figures/`):
  - EDA: Гистограммы, боксплоты, корреляционная матрица.
  - Регрессия: Графики важности признаков, предсказания против истинных значений.
  - Классификация: Матрицы ошибок, ROC-кривые (кроме `SI > 8`), графики важности признаков.
- **Отчёт**:
  - `results/report.md`: Текстовый отчёт с результатами и визуализациями.
  - `results/report_[timestamp].pdf`: PDF-версия отчёта.
- **Логи** (`logs/`): Подробные записи выполнения каждого скрипта.

## Замечания

- Если `data/coursework_data.xlsx` отсутствует, скрипт завершится с ошибкой.
- Для серверных сред настройте `matplotlib` на бэкенд `Agg` (например, добавьте `matplotlib.use('Agg')` в `regression_log_cc50.py`).
- Скрипт `classification_si_8.py` использует `accuracy` вместо `F1` и не создаёт ROC-кривые, что ограничивает сравнение с другими задачами классификации.
- Для ускорения обучения измените `n_jobs=1` на `n_jobs=-1` в скриптах (например, `regression_log_ic50.py`), но это может снизить стабильность `Optuna`.
