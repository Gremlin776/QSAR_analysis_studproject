import pandas as pd
from pathlib import Path
import glob
import logging
from datetime import datetime

# Настраиваем логирование
log_file = Path(__file__).resolve().parent.parent / 'logs' / 'generate_report.log'
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Устанавливаем корневую директорию проекта
project_root = Path(__file__).resolve().parent.parent

# Устанавливаем пути к папкам
results_dir = project_root / 'results'
data_dir = project_root / 'data'
figures_dir = project_root / 'figures'
report_file = results_dir / 'report.md'

# Список ожидаемых файлов результатов
expected_results = [
    'regression_log_ic50.csv',
    'regression_log_cc50.csv',
    'regression_log_si.csv',
    'classification_ic50_median.csv',
    'classification_cc50_median.csv',
    'classification_si_median.csv',
    'classification_si_8.csv'
]

# Проверяем наличие всех зависимостей
def check_dependencies():
    """Проверяет наличие всех необходимых файлов результатов и визуализаций."""
    missing_files = []
    for result_file in expected_results:
        file_path = results_dir / result_file
        if not file_path.exists():
            missing_files.append(str(file_path.relative_to(project_root)))
            logging.warning(f"Файл результатов отсутствует: {file_path}")
    
    dataset_info_path = data_dir / 'dataset_info.csv'
    if not dataset_info_path.exists():
        missing_files.append(str(dataset_info_path.relative_to(project_root)))
        logging.warning(f"Файл dataset_info.csv отсутствует: {dataset_info_path}")
    
    vis_files = list(figures_dir.glob('*.png'))
    if not vis_files:
        logging.warning("Визуализации отсутствуют в figures_dir")
        missing_files.append("Визуализации")
    
    if missing_files:
        logging.error(f"Отсутствуют файлы: {', '.join(missing_files)}")
        return False
    logging.info("Все зависимости найдены")
    return True

# Проверяем наличие визуализации
def check_visualization(file_path: str):
    """Проверяет наличие файла визуализации и возвращает Markdown-строку."""
    full_path = figures_dir / file_path
    relative_path = full_path.relative_to(project_root)
    if full_path.exists():
        logging.info(f"Визуализация найдена: {full_path}")
        return f"![{full_path.name}]({relative_path.as_posix()})\n"
    logging.warning(f"Визуализация отсутствует: {full_path}")
    return ""

# Генерация Markdown таблицы
def generate_markdown_table(df, columns, file_key=None):
    """Создает Markdown таблицу из DataFrame."""
    if df.empty or df is None:
        logging.warning(f"Данные отсутствуют для {file_key}")
        return "Данные отсутствуют\n\n"
    
    # Фильтруем столбцы, которые действительно существуют в DataFrame
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        logging.error(f"Нет валидных столбцов для {file_key}")
        return "Данные отсутствуют\n\n"
    
    table = f"| {' | '.join(valid_columns)} |\n"
    table += "| " + " | ".join(["---"] * len(valid_columns)) + " |\n"
    for _, row in df[valid_columns].iterrows():
        table += "| " + " | ".join([str(round(x, 3) if isinstance(x, (int, float)) else x) for x in row]) + " |\n"
    table += "\n"
    logging.info(f"Таблица сгенерирована для {file_key}")
    return table

# Получение информации о лучшей модели
def get_best_model_info(df, metric_key, extra_metrics=None):
    """Возвращает имя лучшей модели и её метрики."""
    if df is None or df.empty or metric_key not in df.columns:
        logging.warning(f"Данные или ключ {metric_key} отсутствуют")
        return "N/A", {metric_key: "N/A"}
    
    try:
        best_row = df.loc[df[metric_key].idxmax()]
        model_name = best_row['Model']
        metrics = {metric_key: round(best_row[metric_key], 3)}
        if extra_metrics:
            for metric in extra_metrics:
                metrics[metric] = round(best_row[metric], 3) if metric in df.columns else "N/A"
        logging.info(f"Лучшая модель: {model_name}, метрики: {metrics}")
        return model_name, metrics
    except Exception as e:
        logging.error(f"Ошибка при выборе лучшей модели: {e}")
        return "N/A", {metric_key: "N/A"}

# Анализ log_si
def analyze_log_si(results_df):
    """Анализирует результаты классификации log_si."""
    if results_df is None or results_df.empty:
        logging.warning("Результаты для log_si_classification отсутствуют")
        return "Результаты для log_si отсутствуют.\n"
    
    _, metrics = get_best_model_info(results_df, 'F1', ['Accuracy'])
    if metrics['F1'] != "N/A" and float(metrics['F1']) < 0.65:
        return f"Низкий F1 ({metrics['F1']}) для log_si указывает на необходимость оптимизации (SMOTE, Grid Search).\n"
    return f"F1 ({metrics['F1']}) для log_si высокий, но возможна дальнейшая оптимизация с использованием SHAP-анализа.\n"

# Генерация рекомендаций для регрессии
def regression_recommendations(r2, target_name):
    """Возвращает рекомендации для регрессионных моделей на основе R2."""
    recommendations = []
    if r2 != "N/A" and float(r2) < 0.5:
        recommendations.append(f"Для {target_name} R² низкий ({r2:.3f}). Рекомендуется:")
        recommendations.append("- Проверить данные на выбросы с помощью Isolation Forest.")
        recommendations.append("- Применить SMOTE для балансировки данных, если наблюдается дисбаланс.")
        recommendations.append("- Рассмотреть PCA для снижения размерности признаков.")
        recommendations.append("- Использовать более сложные модели, например, Stacking Regressor.")
    else:
        recommendations.append(f"Для {target_name} R² приемлемый ({r2:.3f}). Можно улучшить:")
        recommendations.append("- Провести дополнительную настройку гиперпараметров.")
        recommendations.append("- Добавить новые признаки через feature engineering.")
    logging.info(f"Рекомендации для {target_name}: {recommendations}")
    return recommendations

# Генерация рекомендаций для классификации
def classification_recommendations(metrics, target_name):
    """Возвращает рекомендации для классификации на основе метрик."""
    recommendations = []
    f1 = metrics.get('F1', "N/A")
    precision = metrics.get('Precision', "N/A")
    recall = metrics.get('Recall', "N/A")
    if f1 != "N/A" and (float(f1) < 0.65 or float(precision) < 0.6 or float(recall) < 0.7):
        recommendations.append(f"Для {target_name} низкие метрики (F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}). Рекомендуется:")
        recommendations.append("- Использовать SMOTE для балансировки классов.")
        recommendations.append("- Применить Grid Search для более точной настройки гиперпараметров.")
        recommendations.append("- Рассмотреть Stacking для улучшения классификации.")
    else:
        recommendations.append(f"Для {target_name} метрики приемлемые (F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}). Можно улучшить:")
        recommendations.append("- Провести дополнительную настройку гиперпараметров.")
        recommendations.append("- Проверить важность признаков для исключения лишних.")
    logging.info(f"Рекомендации для {target_name}: {recommendations}")
    return recommendations

# Основная функция генерации отчета
def main():
    logging.info("Начало генерации отчета")
    if not check_dependencies():
        logging.error("Невозможно сгенерировать отчет из-за отсутствия зависимостей")
        print("Ошибка: Отсутствуют необходимые файлы для генерации отчета")
        return

    report = []
    results = {}
    info_df = None

    # Загрузка результатов
    try:
        for file in expected_results:
            key = file.replace('.csv', '')
            file_path = results_dir / file
            results[key] = pd.read_csv(file_path)
            logging.info(f"Файл загружен: {file_path}")
        dataset_info_path = data_dir / 'dataset_info.csv'
        info_df = pd.read_csv(dataset_info_path)
        logging.info(f"Файл загружен: {dataset_info_path}")
    except Exception as e:
        logging.error(f"Ошибка загрузки результатов: {e}")
        print(f"Ошибка загрузки результатов: {e}")

    # Введение
    report.append("## Введение\n")
    report.append(r"""
Создание нового лекарственного препарата — сложный процесс, включающий определение химической формулы, синтез соединений, биологические испытания и клинические тесты. Машинное обучение ускоряет этот процесс, позволяя прогнозировать эффективность химических соединений. В проекте проанализированы данные о 1000 соединениях для предсказания их активности против вируса гриппа. Параметры: IC50 (концентрация, ингибирующая 50% вируса), CC50 (токсичность для 50% клеток), SI (селективный индекс, \( SI = \frac{CC50}{IC50} \)). Соединения с SI > 8 — потенциально эффективные.

Цель — построить модели регрессии для логарифмов IC50, CC50, SI и классификации для определения превышения медианы и SI > 8. Отчет описывает датасет, обработку, моделирование, результаты и QSAR-анализ.
""")

    # Описание датасета
    report.append("## Описание датасета\n")
    report.append("Датасет содержит 1000 соединений с числовыми признаками, IC50_mM, CC50_mM, SI. Загружен из `data/coursework_data.xlsx`.\n")
    if info_df is not None and not info_df.empty:
        before_info = info_df[info_df['Stage'] == 'before']
        if not before_info.empty:
            report.append("### Характеристики до обработки\n")
            report.append(f"""
- **Размер**: {before_info['Rows'].iloc[0]} строк, {before_info['Columns'].iloc[0]} столбцов.
- **Пропуски**: {before_info['Missing_Values'].iloc[0]}.
- **Типы данных**: {before_info.get('Dtype_float64', 0).iloc[0]} float64, {before_info.get('Dtype_int64', 0).iloc[0]} int64.
- **Выбросы**: В IC50_mM, CC50_mM, SI.
""")
        report.append(check_visualization('dataset_before_processing.png'))
        report.append(check_visualization('boxplot_targets.png'))
    else:
        report.append("Информация о датасете отсутствует.\n")

    # Обработка датасета
    report.append("## Обработка датасета\n")
    report.append(r"""
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
""")
    report.append(check_visualization('dataset_after_processing.png'))
    report.append(check_visualization('log_ic50_distribution.png'))
    report.append(check_visualization('log_cc50_distribution.png'))
    report.append(check_visualization('log_si_distribution.png'))

    # Методология
    report.append("## Методология\n")
    report.append("### Модели и метрики\n")
    report.append(r"""
Модели: Random Forest, XGBoost, LightGBM, Gradient Boosting, Linear/Logistic Regression, Voting.
- **Регрессия**: MSE, R², MAE.
- **Классификация**: Accuracy, F1, Precision, Recall, ROC_AUC, PR_AUC.
""")
    report.append("### Подготовка данных\n")
    report.append("80/20 разделение, StandardScaler, SMOTE при дисбалансе >10%, Optuna (50 испытаний).\n")
    report.append("### Визуализации\n")
    report.append("Матрицы ошибок, ROC-кривые, важность признаков, предсказания.\n")

    # Результаты
    report.append("## Результаты\n")
    report.append("### Регрессия\n")
    for target, key in [('log_cc50', 'regression_log_cc50'), ('log_ic50', 'regression_log_ic50'), ('log_si', 'regression_log_si')]:
        report.append(f"#### {target} (`{key}.csv`)\n")
        df = results.get(key)
        if df is not None and not df.empty:
            report.append(generate_markdown_table(df, ['Model', 'MSE', 'R2', 'MAE'], key))
            model_name, metrics = get_best_model_info(df, 'R2', ['MSE', 'MAE'])
            report.append(f"Лучшая модель: {model_name} (R²={metrics['R2']}, MSE={metrics['MSE']}, MAE={metrics['MAE']}).\n")
            recommendations = regression_recommendations(metrics['R2'], target)
            report.append(f"##### Рекомендации для {target}\n")
            report.append("\n".join(recommendations) + "\n\n")
            visualizations = []
            for model in ['rf', 'gb', 'lgb', 'xgb']:
                vis_path = f'feature_importance_{target}_{model}.png'
                if check_visualization(vis_path):
                    visualizations.append(f"![Feature Importance {model}](figures/{vis_path})")
            pred_file = f'pred_vs_true_{target}_rf.png' if target != 'log_ic50' else 'pred_vs_true_log_ic50_lgb.png'
            if check_visualization(pred_file):
                visualizations.append(f"![Pred vs True](figures/{pred_file})")
            report.append("\n".join(visualizations) + "\n" if visualizations else "Визуализации отсутствуют.\n")
        else:
            report.append("Результаты отсутствуют.\n")

    # Классификация
    report.append("### Классификация\n")
    for target, key, models in [
        ('IC50_median', 'classification_ic50_median', ['lgb']),
        ('CC50_median', 'classification_cc50_median', ['rf', 'lgb']),
        ('SI_median', 'classification_si_median', ['rf', 'xgb', 'lgb', 'gb', 'lr', 'voting']),
        ('SI > 8', 'classification_si_8', ['rf', 'xgb', 'lgb', 'gb', 'lr', 'voting'])
    ]:
        report.append(f"#### {target} (`{key}.csv`)\n")
        df = results.get(key)
        if df is not None and not df.empty:
            columns = ['Model', 'Accuracy', 'F1', 'Precision', 'Recall', 'ROC_AUC', 'PR_AUC'] if key != 'classification_si_8' else ['Model', 'Accuracy', 'F1', 'Precision', 'Recall']
            report.append(generate_markdown_table(df, columns, key))
            model_name, metrics = get_best_model_info(df, 'F1', ['Accuracy', 'Precision', 'Recall', 'ROC_AUC'] if key != 'classification_si_8' else ['Accuracy', 'Precision', 'Recall'])
            report.append(f"Лучшая модель: {model_name} (F1={metrics['F1']}, Accuracy={metrics['Accuracy']}, Precision={metrics['Precision']}, Recall={metrics['Recall']}" + 
                          (f", ROC_AUC={metrics['ROC_AUC']}).\n" if key != 'classification_si_8' else ").\n"))
            recommendations = classification_recommendations(metrics, target)
            report.append(f"##### Рекомендации для {target}\n")
            report.append("\n".join(recommendations) + "\n\n")
            if key == 'classification_si_8':
                report.append(analyze_log_si(df))
            visualizations = []
            for model in models:
                cm_path = f'confusion_matrix_{target.lower().replace(' > ', '_')}_{model}.png'
                if check_visualization(cm_path):
                    visualizations.append(f"![Confusion Matrix {model}](figures/{cm_path})")
                roc_path = f'roc_curve_{target.lower().replace(' > ', '_')}_{model}.png'
                if key != 'classification_si_8' and check_visualization(roc_path):
                    visualizations.append(f"![ROC Curve {model}](figures/{roc_path})")
                fi_path = f'feature_importance_{target.lower().replace(' > ', '_')}_{model}.png'
                if model != 'lr' and model != 'voting' and check_visualization(fi_path):
                    visualizations.append(f"![Feature Importance {model}](figures/{fi_path})")
            report.append("\n".join(visualizations) + "\n" if visualizations else "")
        else:
            report.append("Результаты отсутствуют.\n")

    # Анализ и QSAR-рекомендации
    report.append("## Анализ и QSAR-рекомендации\n")
    report.append("### Общий анализ\n")
    analysis_lines = ["- **Регрессия**:"]
    for target, key in [('log_cc50', 'regression_log_cc50'), ('log_ic50', 'regression_log_ic50'), ('log_si', 'regression_log_si')]:
        model_name, metrics = get_best_model_info(results.get(key), 'R2')
        analysis_lines.append(f"  - `{model_name}` лучшая для `{target}` (R²={metrics['R2']}).")
    if results.get('regression_log_si') is not None and not results['regression_log_si'].empty:
        _, metrics = get_best_model_info(results['regression_log_si'], 'R2')
        if metrics['R2'] != "N/A" and float(metrics['R2']) < 0.3:
            analysis_lines.append(f"  - Низкий R² для `log_si` ({metrics['R2']}) указывает на сложность предсказания SI.")
    
    analysis_lines.append("- **Классификация**:")
    for target, key in [('IC50_median', 'classification_ic50_median'), ('CC50_median', 'classification_cc50_median'), 
                        ('SI_median', 'classification_si_median'), ('SI > 8', 'classification_si_8')]:
        model_name, metrics = get_best_model_info(results.get(key), 'F1')
        analysis_lines.append(f"  - `{model_name}` лучшая для `{target}` (F1={metrics['F1']}).")
    if results.get('classification_si_8') is not None and not results['classification_si_8'].empty:
        _, metrics = get_best_model_info(results['classification_si_8'], 'F1')
        if metrics['F1'] != "N/A" and float(metrics['F1']) >= 0.8:
            analysis_lines.append(f"  - Высокий F1 ({metrics['F1']}) для `SI > 8` указывает на хорошую способность модели выявлять эффективные соединения.")
    
    analysis_lines.append("- **Важность признаков**:")
    analysis_lines.append("  - Графики (`feature_importance_*.png`) выявляют ключевые характеристики, влияющие на IC50, CC50 и SI.")
    report.append("\n".join(analysis_lines) + "\n")

    report.append("### QSAR-анализ\n")
    report.append(r"""
1. **Неэффективные соединения**: Высокий IC50, SI < 8. Используйте `IC50_median` и `SI > 8` для их идентификации.
2. **Эффективные соединения**: Низкий IC50, высокий SI > 8. Модель для `SI > 8` (F1=0.849) наиболее точна для их выявления.
3. **Опасные соединения**: Низкий CC50. Модель для `CC50_median` (F1=0.785) помогает их идентифицировать.
""")
    report.append("### Рекомендации\n")
    report.append(r"""
- Использовать SMOTE для улучшения классификации `SI > 8`, особенно для повышения Recall.
- Применить Stacking Regressor для повышения R² в регрессии `log_si`.
- Добавить SHAP-анализ для интерпретации важности признаков.
- Провести внешнюю валидацию моделей на новых данных.
- Рассмотреть добавление 3D-дескрипторов для улучшения предсказательной способности.
""")

    # Заключение
    report.append("## Заключение\n")
    report.append(r"""
Проект успешно проанализировал 1000 соединений, выявив ключевые признаки, влияющие на активность против вируса гриппа. Модели классификации для `SI > 8` (F1=0.849) и `CC50_median` (F1=0.785) показали высокую точность. Регрессия для `log_cc50` (R²=0.501) и `log_ic50` (R²=0.475) демонстрирует умеренную предсказательную способность, но для `log_si` (R²=0.226) требуется оптимизация. Визуализации и результаты полезны для оптимизации соединений. Рекомендуется внешняя валидация и добавление 3D-дескрипторов.
""")

    # Сохранение отчета
    try:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        logging.info(f"Отчет успешно сохранен: {report_file}")
        print(f"Отчет успешно сохранен: {report_file}")
    except Exception as e:
        logging.error(f"Ошибка сохранения отчета: {e}")
        print(f"Ошибка сохранения отчета: {e}")

if __name__ == "__main__":
    main()