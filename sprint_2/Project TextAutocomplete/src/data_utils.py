import re
import os
from typing import List
from transformers import AutoTokenizer

from utils import load_config


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
            f.write(tokens_str)
        print(f"Успешно записано {len(tokens)} токенов")
    except Exception as e:
        print(f"Ошибка при записи в файл: {e}")


if __name__ == "__main__":
    # Загрузка configs файла
    data_config = load_config("dataset_config.yaml")

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

    # Параметры обработки
    chunk_size = data_config.get("chunk_size", 10000)  # Размер кусочка в символах
    total_chars = len(raw_data)
    total_chunks = (total_chars + chunk_size - 1) // chunk_size  # Округление вверх
    print(f"Начало обработки файла размером {total_chars} символов")
    print(f"Размер кусочка: {chunk_size} символов")
    print(f"Всего кусочков: {total_chunks}")

    # Токенизатор
    tokenizer_name = data_config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Обработка файла по кусочкам
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_chars)

        # Извлечение кусочка
        chunk = raw_data[start_idx:end_idx]
        print(f"\nОбработка кусочка {i + 1}/{total_chunks} (символы {start_idx}-{end_idx})")

        # Очистка текста
        cleaned_chunk = clean_text(chunk)
        print(f"Размер после очистки: {len(cleaned_chunk)} символов")

        # Токенизация
        tokenized_chunk = tokenizer.encode(cleaned_chunk)
        print(f"Количество токенов: {len(tokenized_chunk)}")

        # Сохранение в файл (режим 'a' для добавления)
        save_tokenized_data(tokenized_chunk, processed_file_path, mode='a')

        # Прогресс
        progress = ((i + 1) / total_chunks) * 100
        print(f"Прогресс: {progress:.1f}%")

    print(f"\nОбработка завершена!")
    print(f"Все данные сохранены в: {processed_file_path}")
    print(f"Размер выходного файла: {os.path.getsize(processed_file_path)} байт")
