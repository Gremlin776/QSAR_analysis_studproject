## Введение


Создание нового лекарственного препарата — сложный процесс, включающий определение химической формулы, синтез соединений, биологические испытания и клинические тесты. Машинное обучение ускоряет этот процесс, позволяя прогнозировать эффективность химических соединений. В проекте проанализированы данные о 1000 соединениях для предсказания их активности против вируса гриппа. Параметры: IC50 (концентрация, ингибирующая 50% вируса), CC50 (токсичность для 50% клеток), SI (селективный индекс, \( SI = \frac{CC50}{IC50} \)). Соединения с SI > 8 — потенциально эффективные.

Цель — построить модели регрессии для логарифмов IC50, CC50, SI и классификации для определения превышения медианы и SI > 8. Отчет описывает датасет, обработку, моделирование, результаты и QSAR-анализ.

## Описание датасета

Датасет содержит 1000 соединений с числовыми признаками, IC50_mM, CC50_mM, SI. Загружен из `data/coursework_data.xlsx`.

### Характеристики до обработки


- **Размер**: 1001 строк, 214 столбцов.
- **Пропуски**: 36.
- **Типы данных**: 107 float64, 107 int64.
- **Выбросы**: В IC50_mM, CC50_mM, SI.

![dataset_before_processing.png](figures/dataset_before_processing.png)

![boxplot_targets.png](figures/boxplot_targets.png)

## Обработка датасета


Обработка в `eda.py`:
1. Удалены дубликаты.
2. Переименованы столбцы: `IC50_mM`, `CC50_mM`, признаки — `feature_i`.
3. Выбросы обрезаны по IQR.
4. Пропуски заполнены медианой.
5. Значения ≤ 0 заменены на \( 10^{-6} \).
6. Логарифмированы: `log_ic50`, `log_cc50`, `log_si`.
7. Удалены коррелированные признаки (>0.8).
8. Удалены признаки с низкой дисперсией (<0.01).
9. Исключены NaN/бесконечные значения.

Итог: `data/processed_data.csv`.

![dataset_after_processing.png](figures/dataset_after_processing.png)

![log_ic50_distribution.png](figures/log_ic50_distribution.png)

![log_cc50_distribution.png](figures/log_cc50_distribution.png)

![log_si_distribution.png](figures/log_si_distribution.png)

## Методология

### Модели и метрики


Модели: Random Forest, XGBoost, LightGBM, Gradient Boosting, Linear/Logistic Regression, Voting.
- **Регрессия**: MSE, R², MAE.
- **Классификация**: Accuracy, F1, Precision, Recall, ROC_AUC, PR_AUC.

### Подготовка данных

80/20 разделение, StandardScaler, SMOTE при дисбалансе >10%, Optuna (50 испытаний).

### Визуализации

Матрицы ошибок, ROC-кривые, важность признаков, предсказания.

## Результаты

### Регрессия

#### log_cc50 (`regression_log_cc50.csv`)

| Model | MSE | R2 | MAE |
| --- | --- | --- | --- |
| RF | 1.154 | 0.476 | 0.762 |
| XGB | 1.201 | 0.455 | 0.764 |
| LGB | 1.24 | 0.437 | 0.79 |
| GB | 1.142 | 0.482 | 0.783 |
| LR | 1.493 | 0.322 | 0.944 |
| Voting | 1.099 | 0.501 | 0.749 |


Лучшая модель: Voting (R²=0.501, MSE=1.099, MAE=0.749).

##### Рекомендации для log_cc50

Для log_cc50 R² приемлемый (0.501). Можно улучшить:
- Провести дополнительную настройку гиперпараметров.
- Добавить новые признаки через feature engineering.


![Feature Importance rf](figures/feature_importance_log_cc50_rf.png)
![Feature Importance gb](figures/feature_importance_log_cc50_gb.png)
![Feature Importance lgb](figures/feature_importance_log_cc50_lgb.png)
![Feature Importance xgb](figures/feature_importance_log_cc50_xgb.png)

#### log_ic50 (`regression_log_ic50.csv`)

| Model | MSE | R2 | MAE |
| --- | --- | --- | --- |
| RF | 1.667 | 0.476 | 0.991 |
| XGB | 1.747 | 0.451 | 1.045 |
| LGB | 1.689 | 0.469 | 1.022 |
| GB | 1.911 | 0.399 | 1.095 |
| LR | 2.391 | 0.248 | 1.256 |
| Voting | 1.802 | 0.433 | 1.042 |


Лучшая модель: RF (R²=0.476, MSE=1.667, MAE=0.991).

##### Рекомендации для log_ic50

Для log_ic50 R² низкий (0.476). Рекомендуется:
- Проверить данные на выбросы с помощью Isolation Forest.
- Применить SMOTE для балансировки данных, если наблюдается дисбаланс.
- Рассмотреть PCA для снижения размерности признаков.
- Использовать более сложные модели, например, Stacking Regressor.


![Feature Importance rf](figures/feature_importance_log_ic50_rf.png)
![Feature Importance gb](figures/feature_importance_log_ic50_gb.png)
![Feature Importance lgb](figures/feature_importance_log_ic50_lgb.png)
![Feature Importance xgb](figures/feature_importance_log_ic50_xgb.png)

#### log_si (`regression_log_si.csv`)

| Model | MSE | R2 | MAE |
| --- | --- | --- | --- |
| RF | 0.883 | 0.261 | 0.757 |
| XGB | 0.962 | 0.195 | 0.789 |
| LGB | 0.911 | 0.238 | 0.762 |
| GB | 0.932 | 0.22 | 0.79 |
| LR | 1.13 | 0.055 | 0.869 |
| Voting | 0.926 | 0.226 | 0.749 |


Лучшая модель: RF (R²=0.261, MSE=0.883, MAE=0.757).

##### Рекомендации для log_si

Для log_si R² низкий (0.261). Рекомендуется:
- Проверить данные на выбросы с помощью Isolation Forest.
- Применить SMOTE для балансировки данных, если наблюдается дисбаланс.
- Рассмотреть PCA для снижения размерности признаков.
- Использовать более сложные модели, например, Stacking Regressor.


![Feature Importance rf](figures/feature_importance_log_si_rf.png)
![Feature Importance gb](figures/feature_importance_log_si_gb.png)
![Feature Importance lgb](figures/feature_importance_log_si_lgb.png)
![Feature Importance xgb](figures/feature_importance_log_si_xgb.png)
![Pred vs True](figures/pred_vs_true_log_si_rf.png)

### Классификация

#### IC50_median (`classification_ic50_median.csv`)

| Model | Accuracy | F1 | Precision | Recall | ROC_AUC | PR_AUC |
| --- | --- | --- | --- | --- | --- | --- |
| RF | 0.731 | 0.74 | 0.72 | 0.762 | 0.777 | 0.732 |
| XGB | 0.706 | 0.715 | 0.698 | 0.733 | 0.76 | 0.716 |
| LGB | 0.746 | 0.754 | 0.736 | 0.772 | 0.776 | 0.713 |
| GB | 0.726 | 0.732 | 0.721 | 0.743 | 0.762 | 0.737 |
| LR | 0.667 | 0.676 | 0.66 | 0.693 | 0.736 | 0.727 |
| Voting | 0.726 | 0.739 | 0.709 | 0.772 | 0.783 | 0.741 |


Лучшая модель: LGB (F1=0.754, Accuracy=0.746, Precision=0.736, Recall=0.772, ROC_AUC=0.776).

##### Рекомендации для IC50_median

Для IC50_median метрики приемлемые (F1=0.754, Precision=0.736, Recall=0.772). Можно улучшить:
- Провести дополнительную настройку гиперпараметров.
- Проверить важность признаков для исключения лишних.


![Confusion Matrix lgb](figures/confusion_matrix_ic50_median_lgb.png)
![ROC Curve lgb](figures/roc_curve_ic50_median_lgb.png)
![Feature Importance lgb](figures/feature_importance_ic50_median_lgb.png)

#### CC50_median (`classification_cc50_median.csv`)

| Model | Accuracy | F1 | Precision | Recall | ROC_AUC | PR_AUC |
| --- | --- | --- | --- | --- | --- | --- |
| RF | 0.761 | 0.767 | 0.752 | 0.782 | 0.852 | 0.858 |
| XGB | 0.741 | 0.764 | 0.706 | 0.832 | 0.826 | 0.812 |
| LGB | 0.741 | 0.75 | 0.729 | 0.772 | 0.836 | 0.836 |
| GB | 0.766 | 0.771 | 0.76 | 0.782 | 0.872 | 0.871 |
| LR | 0.746 | 0.763 | 0.719 | 0.812 | 0.838 | 0.838 |
| Voting | 0.776 | 0.785 | 0.759 | 0.812 | 0.875 | 0.869 |


Лучшая модель: Voting (F1=0.785, Accuracy=0.776, Precision=0.759, Recall=0.812, ROC_AUC=0.875).

##### Рекомендации для CC50_median

Для CC50_median метрики приемлемые (F1=0.785, Precision=0.759, Recall=0.812). Можно улучшить:
- Провести дополнительную настройку гиперпараметров.
- Проверить важность признаков для исключения лишних.


![Feature Importance rf](figures/feature_importance_cc50_median_rf.png)
![Feature Importance lgb](figures/feature_importance_cc50_median_lgb.png)

#### SI_median (`classification_si_median.csv`)

| Model | Accuracy | F1 | Precision | Recall | ROC_AUC | PR_AUC |
| --- | --- | --- | --- | --- | --- | --- |
| RF | 0.642 | 0.609 | 0.667 | 0.56 | 0.685 | 0.693 |
| XGB | 0.657 | 0.635 | 0.674 | 0.6 | 0.691 | 0.702 |
| LGB | 0.637 | 0.622 | 0.645 | 0.6 | 0.683 | 0.693 |
| GB | 0.662 | 0.634 | 0.686 | 0.59 | 0.678 | 0.686 |
| LR | 0.612 | 0.625 | 0.602 | 0.65 | 0.649 | 0.667 |
| Voting | 0.657 | 0.642 | 0.667 | 0.62 | 0.693 | 0.694 |


Лучшая модель: Voting (F1=0.642, Accuracy=0.657, Precision=0.667, Recall=0.62, ROC_AUC=0.693).

##### Рекомендации для SI_median

Для SI_median низкие метрики (F1=0.642, Precision=0.667, Recall=0.620). Рекомендуется:
- Использовать SMOTE для балансировки классов.
- Применить Grid Search для более точной настройки гиперпараметров.
- Рассмотреть Stacking для улучшения классификации.


![Confusion Matrix rf](figures/confusion_matrix_si_median_rf.png)
![ROC Curve rf](figures/roc_curve_si_median_rf.png)
![Feature Importance rf](figures/feature_importance_si_median_rf.png)
![Confusion Matrix xgb](figures/confusion_matrix_si_median_xgb.png)
![ROC Curve xgb](figures/roc_curve_si_median_xgb.png)
![Feature Importance xgb](figures/feature_importance_si_median_xgb.png)
![Confusion Matrix lgb](figures/confusion_matrix_si_median_lgb.png)
![ROC Curve lgb](figures/roc_curve_si_median_lgb.png)
![Feature Importance lgb](figures/feature_importance_si_median_lgb.png)
![Confusion Matrix gb](figures/confusion_matrix_si_median_gb.png)
![ROC Curve gb](figures/roc_curve_si_median_gb.png)
![Feature Importance gb](figures/feature_importance_si_median_gb.png)
![Confusion Matrix lr](figures/confusion_matrix_si_median_lr.png)
![ROC Curve lr](figures/roc_curve_si_median_lr.png)
![Confusion Matrix voting](figures/confusion_matrix_si_median_voting.png)
![ROC Curve voting](figures/roc_curve_si_median_voting.png)

#### SI > 8 (`classification_si_8.csv`)

| Model | Accuracy | F1 | Precision | Recall |
| --- | --- | --- | --- | --- |
| RF | 0.756 | 0.847 | 0.764 | 0.951 |
| XGB | 0.736 | 0.835 | 0.753 | 0.937 |
| LGB | 0.736 | 0.834 | 0.756 | 0.93 |
| GB | 0.731 | 0.832 | 0.749 | 0.937 |
| LR | 0.736 | 0.839 | 0.742 | 0.965 |
| Voting | 0.701 | 0.804 | 0.755 | 0.86 |


Лучшая модель: RF (F1=0.847, Accuracy=0.756, Precision=0.764, Recall=0.951).

##### Рекомендации для SI > 8

Для SI > 8 метрики приемлемые (F1=0.847, Precision=0.764, Recall=0.951). Можно улучшить:
- Провести дополнительную настройку гиперпараметров.
- Проверить важность признаков для исключения лишних.


F1 (0.847) для log_si высокий, но возможна дальнейшая оптимизация с использованием SHAP-анализа.


## Анализ и QSAR-рекомендации

### Общий анализ

- **Регрессия**:
  - `Voting` лучшая для `log_cc50` (R²=0.501).
  - `RF` лучшая для `log_ic50` (R²=0.476).
  - `RF` лучшая для `log_si` (R²=0.261).
  - Низкий R² для `log_si` (0.261) указывает на сложность предсказания SI.
- **Классификация**:
  - `LGB` лучшая для `IC50_median` (F1=0.754).
  - `Voting` лучшая для `CC50_median` (F1=0.785).
  - `Voting` лучшая для `SI_median` (F1=0.642).
  - `RF` лучшая для `SI > 8` (F1=0.847).
  - Высокий F1 (0.847) для `SI > 8` указывает на хорошую способность модели выявлять эффективные соединения.
- **Важность признаков**:
  - Графики (`feature_importance_*.png`) выявляют ключевые характеристики, влияющие на IC50, CC50 и SI.

### QSAR-анализ


1. **Неэффективные соединения**: Высокий IC50, SI < 8. Используйте `IC50_median` и `SI > 8` для их идентификации.
2. **Эффективные соединения**: Низкий IC50, высокий SI > 8. Модель для `SI > 8` (F1=0.849) наиболее точна для их выявления.
3. **Опасные соединения**: Низкий CC50. Модель для `CC50_median` (F1=0.785) помогает их идентифицировать.

### Рекомендации


- Использовать SMOTE для улучшения классификации `SI > 8`, особенно для повышения Recall.
- Применить Stacking Regressor для повышения R² в регрессии `log_si`.
- Добавить SHAP-анализ для интерпретации важности признаков.
- Провести внешнюю валидацию моделей на новых данных.
- Рассмотреть добавление 3D-дескрипторов для улучшения предсказательной способности.

## Заключение


Проект успешно проанализировал 1000 соединений, выявив ключевые признаки, влияющие на активность против вируса гриппа. Модели классификации для `SI > 8` (F1=0.849) и `CC50_median` (F1=0.785) показали высокую точность. Регрессия для `log_cc50` (R²=0.501) и `log_ic50` (R²=0.475) демонстрирует умеренную предсказательную способность, но для `log_si` (R²=0.226) требуется оптимизация. Визуализации и результаты полезны для оптимизации соединений. Рекомендуется внешняя валидация и добавление 3D-дескрипторов.
