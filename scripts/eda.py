import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import importlib
from sklearn.feature_selection import VarianceThreshold
from uuid import uuid4

# Проверяем наличие всех необходимых библиотек, чтобы обеспечить выполнение разведочного анализа данных (EDA) в рамках проекта qsar_analisys
required_libs = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn']
for lib in required_libs:
    if importlib.util.find_spec(lib) is None:
        print(f"Error: Library {lib} not found! Please install it.")
        exit(1)

# Настраиваем логирование для записи всех операций и ошибок, чтобы отслеживать процесс EDA и упрощать отладку
try:
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/eda.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
except PermissionError as e:
    print(f"Error: Cannot create logs directory: {e}")
    exit(1)

# Создаем директории для хранения графиков, данных и логов, чтобы организовать результаты анализа
for directory in ['figures', 'data', 'logs']:
    try:
        os.makedirs(directory, exist_ok=True)
    except PermissionError as e:
        logging.error(f"Cannot create directory {directory}: {e}")
        print(f"Error: Cannot create directory {directory}: {e}")
        exit(1)

def collect_dataset_info(df, stage="before"):
    """
    Собирает информацию о датасете: размеры, пропущенные значения, типы данных, статистики и выбросы.
    
    Args:
        df (pd.DataFrame): Входной датасет.
        stage (str): Стадия обработки ("before" или "after").
    
    Returns:
        dict: Информация о датасете.
    """
    # Собираем информацию о датасете, чтобы понять его структуру, качество данных и выявить потенциальные проблемы
    info = {
        'stage': stage,
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_values': df.isnull().sum().sum(),
        'dtypes': {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()}
    }
    
    # Вычисляем описательные статистики для числовых признаков, чтобы оценить их распределение и диапазоны
    desc_stats = df.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    desc_stats = desc_stats.round(2)
    info['desc_stats'] = desc_stats.reset_index().to_dict(orient='records')
    
    # Идентифицируем выбросы в ключевых целевых переменных (IC50_mM, CC50_mM, SI) с использованием метода межквартильного размаха
    outliers = {}
    cols = ['IC50_mM', 'CC50_mM', 'SI'] if stage == 'before' else ['IC50_mM', 'CC50_mM', 'SI', 'log_ic50', 'log_cc50', 'log_si']
    for col in cols:
        if col in df.columns:
            series = df[col]
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = int(len(series[(series < lower_bound) | (series > upper_bound)]))
    info['outliers'] = outliers
    
    return info

def info_to_dataframe(info):
    """
    Преобразует информацию о датасете в два DataFrame: общую информацию и статистики.
    
    Args:
        info (dict): Информация о датасете для стадий "before" и "after".
    
    Returns:
        tuple: (info_df, stats_df) - DataFrame с общей информацией и статистиками.
    """
    # Структурируем информацию о датасете в два DataFrame для удобного представления и сохранения
    rows = []
    for stage in ['before', 'after']:
        stage_info = info[stage]
        row = {
            'Stage': stage,
            'Rows': stage_info['rows'],
            'Columns': stage_info['columns'],
            'Missing_Values': stage_info['missing_values']
        }
        for dtype, count in stage_info['dtypes'].items():
            row[f'Dtype_{dtype}'] = count
        for col, count in stage_info['outliers'].items():
            row[f'Outliers_{col}'] = count
        rows.append(row)
    
    stats_rows = []
    for stage in ['before', 'after']:
        for stat in info[stage]['desc_stats']:
            stats_row = {
                'Stage': stage,
                'Feature': stat['index'],
                'Count': stat['count'],
                'Mean': stat['mean'],
                'Std': stat['std'],
                'Min': stat['min'],
                '25%': stat['25%'],
                '50%': stat['50%'],
                '75%': stat['75%'],
                'Max': stat['max']
            }
            stats_rows.append(stats_row)
    
    return pd.DataFrame(rows), pd.DataFrame(stats_rows)

def describe_dataset(df, title, filename):
    """
    Создает текстовую визуализацию описательных статистик и сохраняет в PNG.
    
    Args:
        df (pd.DataFrame): Входной датасет.
        title (str): Заголовок визуализации.
        filename (str): Имя файла для сохранения.
    """
    # Создаем текстовую визуализацию статистик датасета, чтобы предоставить обзор данных в удобной форме
    try:
        num_rows, num_cols = df.shape
        missing_values = df.isnull().sum().sum()
        dtypes = {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()}
        desc_stats = df.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        desc_stats = desc_stats.round(2)
        
        plt.figure(figsize=(14, 10))
        plt.text(0.1, 0.95, f'{title}\nRows: {num_rows}, Columns: {num_cols}\nMissing Values: {missing_values}\nData Types: {dtypes}', 
                 fontsize=12, fontweight='bold')
        plt.text(0.1, 0.85, desc_stats.to_string(), fontsize=10, family='monospace')
        plt.axis('off')
        plt.savefig(f'figures/{filename}', bbox_inches='tight')
        plt.close()
        logging.info(f"Descriptive stats saved to figures/{filename}")
    except Exception as e:
        logging.error(f"Error saving descriptive stats {filename}: {e}")
        print(f"Error saving descriptive stats {filename}: {e}")

def main():
    """Основная функция для выполнения EDA."""
    # Начинаем выполнение EDA и логируем начало работы для отслеживания
    logging.info("Loading data")
    try:
        # Загружаем исходный датасет из Excel-файла, содержащего данные о молекулярных дескрипторах и биологических параметрах
        data = pd.read_excel('data/coursework_data.xlsx')
    except FileNotFoundError as e:
        logging.error(f"File data/coursework_data.xlsx not found: {e}")
        print(f"Error: File data/coursework_data.xlsx not found!")
        exit(1)
    except Exception as e:
        logging.error(f"Error reading data: {e}")
        print(f"Error reading data: {e}")
        exit(1)

    # Проверяем наличие дубликатов и удаляем их, чтобы обеспечить чистоту данных
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        logging.info(f"Found {duplicates} duplicate rows, removing them")
        data = data.drop_duplicates()
    
    # Собираем информацию о датасете до обработки, чтобы зафиксировать его исходное состояние
    logging.info("Collecting info before processing")
    dataset_info_before = collect_dataset_info(data, "before")

    # Логируем и выводим информацию о размере и столбцах датасета для анализа
    logging.info(f"Dataset shape: {data.shape}")
    logging.info(f"Dataset columns: {list(data.columns)}")
    print("Dataset shape:", data.shape)
    print("Dataset columns:", list(data.columns))

    # Визуализируем статистику исходного датасета для последующего сравнения
    describe_dataset(data, "Dataset Before Processing", "dataset_before_processing.png")

    # Переименовываем столбцы для унификации названий и упрощения дальнейшей обработки
    data.rename(columns={'IC50, mM': 'IC50_mM', 'CC50, mM': 'CC50_mM'}, inplace=True)
    logging.info("Renamed columns 'IC50, mM' and 'CC50, mM'")

    # Переименовываем признаки в формат feature_X для стандартизации
    rename_dict = {col: f"feature_{i}" for i, col in enumerate(data.columns) if col not in ['IC50_mM', 'CC50_mM', 'SI']}
    data.rename(columns=rename_dict, inplace=True)
    try:
        pd.DataFrame(list(rename_dict.items()), columns=['Original', 'New']).to_csv('data/rename_dict.csv', index=False)
        logging.info("Rename dictionary saved to data/rename_dict.csv")
    except Exception as e:
        logging.error(f"Error saving rename_dict.csv: {e}")
        print(f"Error saving rename_dict.csv: {e}")

    # Удаляем ненужные столбцы, если они присутствуют, чтобы очистить датасет
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
        logging.info("Dropped column 'Unnamed: 0'")

    # Проверяем наличие необходимых столбцов (IC50_mM, CC50_mM, SI) для анализа
    required_columns = ['IC50_mM', 'CC50_mM', 'SI']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns: {missing_columns}")
        print(f"Error: Missing columns {missing_columns}!")
        exit(1)

    # Обрабатываем выбросы в целевых переменных, обрезая их по методу IQR, чтобы минимизировать их влияние
    logging.info("Handling outliers")
    for col in required_columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = len(data[col][(data[col] < lower_bound) | (data[col] > upper_bound)])
        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        logging.info(f"Outliers in {col} clipped ({outliers_count}): lower={lower_bound:.2f}, upper={upper_bound:.2f}")

    # Заполняем пропущенные значения, чтобы обеспечить целостность данных
    logging.info(f"Missing values before: {data.isnull().sum().sum()}")
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if not numeric_cols.empty:
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    
    for col in data.select_dtypes(include=['object']).columns:
        if not data[col].empty and not data[col].mode().empty:
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            logging.warning(f"Column {col} is empty or has no mode, filling with 'Unknown'")
            data[col] = data[col].fillna('Unknown')
    
    logging.info(f"Missing values after: {data.isnull().sum().sum()}")

    # Обрабатываем нулевые или отрицательные значения в целевых переменных, чтобы избежать проблем с логарифмированием
    for col in required_columns:
        zero_negative_count = (data[col] <= 0).sum()
        if zero_negative_count > 0:
            logging.warning(f"Found {zero_negative_count} zero/negative values in {col}, clipping to 1e-6")
            data[col] = data[col].clip(lower=1e-6)

    # Создаем логарифмированные столбцы для регрессии, чтобы нормализовать распределение данных
    logging.info("Creating log-transformed columns")
    log_columns = pd.DataFrame({
        'log_ic50': np.log1p(data['IC50_mM']),
        'log_cc50': np.log1p(data['CC50_mM']),
        'log_si': np.log1p(data['SI'])
    })
    data = pd.concat([data, log_columns], axis=1)

    # Удаляем сильно коррелированные признаки, чтобы снизить мультиколлинеарность и улучшить качество моделей
    features = [col for col in data.columns if col not in ['IC50_mM', 'CC50_mM', 'SI', 'log_ic50', 'log_cc50', 'log_si']]
    if features:
        corr_matrix = data[features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
        if to_drop:
            pd.DataFrame(to_drop, columns=['Dropped_Feature']).to_csv('data/dropped_correlated_features.csv', index=False)
            logging.info(f"Dropped {len(to_drop)} correlated features: {to_drop}")
            data = data.drop(columns=to_drop)
    
    # Фильтруем признаки с низкой дисперсией, чтобы оставить только информативные для моделей
    features = [col for col in data.columns if col not in ['IC50_mM', 'CC50_mM', 'SI', 'log_ic50', 'log_cc50', 'log_si']]
    if features:
        try:
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(data[features])
            selected_features = data[features].columns[selector.get_support()].tolist()
            if selected_features:
                data = data[selected_features + ['IC50_mM', 'CC50_mM', 'SI', 'log_ic50', 'log_cc50', 'log_si']]
                logging.info(f"Kept {len(selected_features)} features after variance filtering")
            else:
                logging.error("No features left after variance filtering")
                print("Error: No features left after variance filtering!")
                exit(1)
        except Exception as e:
            logging.error(f"Error in variance filtering: {e}")
            print(f"Error in variance filtering: {e}")
            exit(1)
    else:
        logging.error("No features available for variance filtering")
        print("Error: No features available!")
        exit(1)

    # Проверяем данные на наличие NaN или бесконечных значений после обработки, чтобы гарантировать их пригодность
    if data.isnull().values.any():
        logging.error("NaN values detected after processing")
        print("Error: NaN values detected!")
        exit(1)
    if np.isinf(data.values).any():
        logging.error("Infinite values detected")
        print("Error: Infinite values detected!")
        exit(1)
    if not all(data.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        logging.error("Non-numeric columns detected")
        print("Error: Non-numeric columns detected!")
        exit(1)

    # Собираем информацию о датасете после обработки для сравнения с исходным состоянием
    logging.info("Collecting info after processing")
    dataset_info_after = collect_dataset_info(data, "after")

    # Сохраняем информацию о датасете в CSV-файлы для отчетности
    dataset_info = {"before": dataset_info_before, "after": dataset_info_after}
    info_df, stats_df = info_to_dataframe(dataset_info)
    try:
        info_df.to_csv('data/dataset_info.csv', index=False)
        stats_df.to_csv('data/dataset_stats.csv', index=False)
        logging.info("Dataset info saved to data/dataset_info.csv and data/dataset_stats.csv")
    except Exception as e:
        logging.error(f"Error saving dataset info: {e}")
        print(f"Error saving dataset info: {e}")

    # Визуализируем статистику обработанного датасета для анализа изменений
    describe_dataset(data, "Dataset After Processing", "dataset_after_processing.png")

    # Сохраняем обработанный датасет для использования в регрессионных моделях
    try:
        output_path = 'data/processed_data.csv'
        data.to_csv(output_path, index=False)
        logging.info(f"Processed dataset saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving processed data: {e}")
        print(f"Error saving processed data: {e}")

    # Создаем визуализации распределений логарифмированных переменных для оценки их свойств
    logging.info("Creating visualizations")
    for col, title, fname in [
        ('log_ic50', 'Distribution of log(IC50)', 'log_ic50_distribution.png'),
        ('log_cc50', 'Distribution of log(CC50)', 'log_cc50_distribution.png'),
        ('log_si', 'Distribution of log(SI)', 'log_si_distribution.png')
    ]:
        if col in data.columns:
            try:
                plt.figure(figsize=(10, 6))
                sns.histplot(data[col], kde=True)
                plt.title(title)
                plt.savefig(f'figures/{fname}')
                plt.close()
                logging.info(f"Saved {fname}")
            except Exception as e:
                logging.error(f"Error saving {fname}: {e}")
                print(f"Error saving {fname}: {e}")

    # Создаем ящик с усами для целевых переменных, чтобы визуально оценить выбросы и распределение
    if all(col in data.columns for col in ['IC50_mM', 'CC50_mM', 'SI']):
        try:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=data[['IC50_mM', 'CC50_mM', 'SI']])
            plt.title('Boxplot of Target Variables')
            plt.savefig('figures/boxplot_targets.png')
            plt.close()
            logging.info("Saved boxplot_targets.png")
        except Exception as e:
            logging.error(f"Error saving boxplot_targets.png: {e}")
            print(f"Error saving boxplot_targets.png: {e}")

if __name__ == "__main__":
    # Запускаем основную функцию для выполнения EDA
    main()