import markdown2
from weasyprint import HTML, CSS
import os
import logging
from datetime import datetime

# Создаём папку logs, если она не существует
os.makedirs('logs', exist_ok=True)

# Настраиваем логирование
logging.basicConfig(
    filename='logs/convert_to_pdf.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Устанавливаем пути
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
figures_dir = os.path.join(project_dir, 'figures')
base_dir = os.path.join(project_dir, 'results')
markdown_file = os.path.join(base_dir, 'report.md')
output_pdf = os.path.join(base_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
temp_html = os.path.join(base_dir, f"temp_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")

logging.info(f"Базовые пути: project_dir={project_dir}, figures_dir={figures_dir}, base_dir={base_dir}")

# Проверяем существование Markdown-файла
if not os.path.exists(markdown_file):
    logging.error(f"Файл {markdown_file} не найден")
    raise FileNotFoundError(f"Markdown file {markdown_file} not found")

# CSS для стилизации PDF
css = CSS(string='''
    @page { size: A4; margin: 1cm; }
    body { font-family: Arial, sans-serif; font-size: 12pt; }
    h1, h2, h3 { color: #2E4053; }
    img { max-width: 100%; height: auto; display: block; margin: 10px auto; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #f2f2f2; }
    pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; }
''')

def resolve_image_path(img_src):
    """Преобразуем относительный путь изображения в абсолютный."""
    # Удаляем начальные слэши или обратные слэши
    img_src = img_src.strip('/\\')
    # Проверяем, начинается ли путь с figures/
    if img_src.startswith('figures/'):
        img_path = os.path.join(project_dir, img_src).replace('/', os.sep)
    else:
        img_path = os.path.join(figures_dir, img_src).replace('/', os.sep)
    
    # Нормализуем путь
    img_path = os.path.normpath(img_path)
    
    # Проверяем существование файла
    if os.path.exists(img_path):
        logging.info(f"Изображение найдено: {img_path}")
        return img_path
    logging.warning(f"Изображение не найдено: {img_path}")
    return None

def convert_to_pdf():
    """Конвертируем Markdown в PDF с обработкой путей к изображениям."""
    logging.info(f"Начало конвертации {markdown_file} в PDF")
    
    try:
        # Читаем Markdown
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        
        # Конвертируем Markdown в HTML с помощью markdown2
        extras = ["tables", "fenced-code-blocks", "break-on-newline"]
        html_text = markdown2.markdown(markdown_text, extras=extras)
        
        # Исправляем пути к изображениям в HTML
        import re
        def replace_img_src(match):
            img_src = match.group(1)
            resolved_path = resolve_image_path(img_src)
            if resolved_path:
                # Преобразуем в file:// URL для weasyprint
                resolved_path = resolved_path.replace('\\', '/')
                return f'src="file:///{resolved_path}"'
            return f'src="{img_src}"'  # Оставляем как есть, если не найдено
        
        html_text = re.sub(r'src="([^"]+)"', replace_img_src, html_text)
        
        # Сохраняем временный HTML для отладки
        with open(temp_html, 'w', encoding='utf-8') as f:
            f.write(html_text)
        logging.info(f"Временный HTML сохранен: {temp_html}")
        
        # Конвертируем HTML в PDF
        HTML(string=html_text, base_url=project_dir).write_pdf(output_pdf, stylesheets=[css])
        logging.info(f"PDF успешно сгенерирован: {output_pdf}")
        
    except Exception as e:
        logging.error(f"Ошибка генерации PDF: {e}")
        raise
    
    finally:
        # Не удаляем HTML для отладки
        pass

def main():
    convert_to_pdf()

if __name__ == "__main__":
    main()