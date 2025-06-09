import subprocess
import os
import logging
import argparse
from datetime import datetime
import pkg_resources

# Настраиваем аргументы командной строки
parser = argparse.ArgumentParser(description="Запуск QSAR пайплайна")
parser.add_argument('--continue-on-error', action='store_true', help="Продолжать выполнение при ошибках скриптов")
args = parser.parse_args()

# Устанавливаем корневую директорию проекта
project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)  # Устанавливаем рабочую директорию

# Создаем директории для логов, данных и графиков
dirs_to_create = ['logs', 'figures', 'data', 'results']
for directory in dirs_to_create:
    try:
        os.makedirs(directory, exist_ok=True)
    except PermissionError as e:
        print(f"Ошибка: Невозможно создать директорию {directory}: {e}")
        exit(1)

# Настраиваем логирование
log_file = os.path.join('logs', f"main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
try:
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s'
    )
except Exception as e:
    print(f"Ошибка при настройке логирования: {e}")
    exit(1)

# Проверяем минимальные версии библиотек
required_versions = {
    'pandas': '1.0.0',
    'scikit-learn': '0.24.0',
    'optuna': '2.0.0'
}
for lib, min_version in required_versions.items():
    try:
        installed_version = pkg_resources.get_distribution(lib).version
        if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
            logging.error(f"Версия {lib} ({installed_version}) ниже минимальной ({min_version})")
            print(f"Ошибка: Версия {lib} ({installed_version}) ниже минимальной ({min_version})")
            exit(1)
    except pkg_resources.DistributionNotFound:
        logging.error(f"Библиотека {lib} не установлена")
        print(f"Ошибка: Библиотека {lib} не установлена")
        exit(1)

# Определяем зависимости для каждого скрипта
script_dependencies = {
    'eda.py': [],
    'regression_log_ic50.py': ['data/processed_data.csv'],
    'regression_log_cc50.py': ['data/processed_data.csv'],
    'regression_log_si.py': ['data/processed_data.csv'],
    'classification_ic50_median.py': ['data/processed_data.csv'],
    'classification_cc50_median.py': ['data/processed_data.csv'],
    'classification_si_median.py': ['data/processed_data.csv'],
    'classification_si_8.py': ['data/processed_data.csv'],
    'generate_report.py': ['results/classification_ic50_median.csv', 'results/classification_cc50_median.csv',
                           'results/classification_si_median.csv', 'results/classification_si_8.csv'],
    'convert_to_pdf.py': ['results/report.md']
}

# Список скриптов для выполнения
scripts_dir = os.path.join(project_dir, 'scripts')
scripts = [
    os.path.join(scripts_dir, script_name) for script_name in [
        'eda.py',
        'regression_log_ic50.py',
        'regression_log_cc50.py',
        'regression_log_si.py',
        'classification_ic50_median.py',
        'classification_cc50_median.py',
        'classification_si_median.py',
        'classification_si_8.py',
        'generate_report.py',
        'convert_to_pdf.py'
    ]
]

def check_dependencies(script_name):
    """Проверяет наличие зависимых файлов для скрипта."""
    dependencies = script_dependencies.get(os.path.basename(script_name), [])
    for dep in dependencies:
        if not os.path.exists(dep):
            logging.error(f"Зависимость {dep} для {script_name} не найдена")
            print(f"Ошибка: Зависимость {dep} для {script_name} не найдена")
            return False
    return True

def run_script(script_path):
    """Запускает скрипт и логирует результат."""
    script_name = os.path.basename(script_path)
    logging.info(f"Запуск {script_name}")
    print(f"Запуск: {script_name}")

    if not check_dependencies(script_name):
        logging.warning(f"Пропуск {script_name} из-за отсутствия зависимостей")
        print(f"Пропуск {script_name} из-за отсутствия зависимостей")
        return False

    try:
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Скрипт {script_path} не найден")
        
        result = subprocess.run(['python', script_path], capture_output=True, text=True, check=True)
        logging.info(f"{script_name} успешно завершён")
        logging.debug(f"Вывод: {result.stdout[:500]}")  # Ограничиваем длину вывода
        if result.stderr:
            logging.warning(f"Предупреждения в {script_name}: {result.stderr[:500]}")
        print(f"{script_name} завершён успешно")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Ошибка в {script_name}: {e}")
        logging.error(f"Вывод: {e.output[:500]}")
        logging.error(f"Стандартная ошибка: {e.stderr[:500]}")
        print(f"Ошибка в {script_name}: {e}")
        return False
    except FileNotFoundError as e:
        logging.error(f"Скрипт не найден: {e}")
        print(f"Ошибка: {e}")
        return False
    except Exception as e:
        logging.error(f"Непредвиденная ошибка в {script_name}: {e}")
        print(f"Непредвиденная ошибка в {script_name}: {e}")
        return False

def main():
    """Запускает все скрипты последовательно."""
    logging.info("Запуск QSAR пайплайна")
    print("Запуск QSAR пайплайна")

    failed_scripts = []
    for script in scripts:
        success = run_script(script)
        if not success and not args.continue_on_error:
            logging.error(f"Пайплайн прерван из-за ошибки в {os.path.basename(script)}")
            print(f"Пайплайн прерван из-за ошибки в {os.path.basename(script)}")
            exit(1)
        if not success:
            failed_scripts.append(os.path.basename(script))

    if failed_scripts:
        logging.warning(f"Следующие скрипты завершились с ошибками: {', '.join(failed_scripts)}")
        print(f"Следующие скрипты завершились с ошибками: {', '.join(failed_scripts)}")
    else:
        logging.info("QSAR пайплайн успешно завершён")
        print("QSAR пайплайн успешно завершён")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Критическая ошибка пайплайна: {e}")
        print(f"Критическая ошибка пайплайна: {e}")
        exit(1)