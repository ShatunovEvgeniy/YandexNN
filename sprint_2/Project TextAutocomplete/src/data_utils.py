import re
import os
from typing import List
from transformers import AutoTokenizer

from src.utils import load_config


def read_file(filename: str) -> str | None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, 'data', filename)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
        print(f"Файл успешно прочитан: {file_path}")
        print(f"Размер файла: {len(raw_text)} символов")
        return raw_text

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути: {file_path}")
        print("Проверьте структуру проекта:")
        print(f"Текущая директория: {current_dir}")
        print(f"Корень проекта: {project_root}")
        print(f"Ожидаемый путь к файлу: {file_path}")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Привести к нижнему регистру
    text = text.lower()

    # Удалить упоминания пользователей (@username)
    text = re.sub(r'@\w+', ' ', text)

    # Удалить хештеги (#hashtag)
    text = re.sub(r'#\w+', ' ', text)

    # Удалить ссылки (http://, https://, www.)
    text = re.sub(r'https?://\S+', ' ', text)  # http:// и https://
    text = re.sub(r'www\.\S+', ' ', text)  # www.example.com
    text = re.sub(r'\b[a-z0-9.-]+\.[a-z]{2,}\S*', ' ', text)  # другие домены

    # Удалить специальные символы, оставив буквы, цифры и основные знаки препинания
    text = re.sub(r"[^a-z0-9\s.,!?;:()'\"]+", " ", text)

    # Убрать дублирующиеся пробелы и лишние пробелы в начале/конце
    text = re.sub(r"\s+", " ", text).strip()

    return text


def save_tokenized_data(tokens: List[int], output_file_path: str, mode: str = 'a') -> None:
    """
    Сохраняет токенизированные данные в файл.
    mode: Режим записи ('w' для перезаписи, 'a' для добавления)
    """
    try:
        with open(output_file_path, mode, encoding='utf-8') as f:
            # Записываем токены как строку с разделителями-пробелами
            tokens_str = ' '.join(map(str, tokens))
            f.write(tokens_str + '\n')
        print(f"Успешно записано {len(tokens)} токенов")
    except Exception as e:
        print(f"Ошибка при записи в файл: {e}")


if __name__ == "__main__":
    # Загрузка config файла
    data_config = load_config("config.yaml")

    # Открытие сырого файла и очистка
    raw_file_name = data_config["raw_dataset"]
    raw_data = read_file(raw_file_name)

    # Открытие целевого файла
    processed_file_name = data_config["processed_dataset"]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    processed_file_path = os.path.join(data_dir, processed_file_name)
    with open(processed_file_path, 'w', encoding='utf-8') as f:
        f.write('')  # Очищаем файл
    print(f"Выходной файл подготовлен: {processed_file_path}")

    # Разделение текста на строки
    lines = raw_data.splitlines()
    total_lines = len(lines)
    print(f"Начало обработки файла с {total_lines} строками")

    # Токенизатор
    tokenizer_name = data_config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Обработка файла по строкам
    for i, line in enumerate(lines):
        print(f"\nОбработка строки {i + 1}/{total_lines}")

        # Очистка текста
        cleaned_line = clean_text(line)
        if not cleaned_line:
            print("Пустая строка после очистки, пропускаем")
            continue

        print(f"Размер после очистки: {len(cleaned_line)} символов")

        # Токенизация
        tokenized_line = tokenizer.encode(cleaned_line)
        tokenized_line.append(tokenizer.eos_token_id)  # Добавляем EOS в конец каждого предложения
        print(f"Количество токенов: {len(tokenized_line)}")

        # Сохранение в файл (режим 'a' для добавления)
        save_tokenized_data(tokenized_line, processed_file_path, mode='a')

        # Прогресс
        progress = ((i + 1) / total_lines) * 100
        print(f"Прогресс: {progress:.1f}%")

    print(f"\nОбработка завершена!")
    print(f"Все данные сохранены в: {processed_file_path}")
    print(f"Размер выходного файла: {os.path.getsize(processed_file_path)} байт")