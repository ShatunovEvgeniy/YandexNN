import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
from typing import Tuple, List

from utils import load_config


class TextAutocompleteDataset(Dataset):
    def __init__(self, data: List[List[int]]):
        """
        Для каждой последовательности [w1, w2, w3, ..., wn] создаем:
        - X: [w1], [w1, w2], [w1, w2, w3], ..., [w1, w2, ..., w_{n-1}]
        - Y: [w2], [w2, w3], [w2, w3, w4], ..., [w2, w3, ..., wn]
        """
        self.x_sequences = []
        self.y_sequences = []

        # Используем numpy для эффективной обработки
        for message in data:
            if len(message) < 2:  # Нужно минимум 2 токена для создания пары
                continue

            message_array = np.array(message)

            # Создаем все возможные префиксы и соответствующие им продолжения
            for i in range(1, len(message)):
                # X: последовательность от начала до i-го элемента (включительно)
                x_seq = message_array[:i]
                # Y: последовательность от 1-го до i+1-го элемента
                y_seq = message_array[1:i + 1]

                self.x_sequences.append(x_seq.tolist())
                self.y_sequences.append(y_seq.tolist())

        print(f"Created {len(self.x_sequences)} training pairs from {len(data)} messages")

    def __len__(self):
        return len(self.x_sequences)

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.x_sequences[idx], dtype=torch.long),
            'y': torch.tensor(self.y_sequences[idx], dtype=torch.long)
        }


def load_and_preprocess_data(file_path: str) -> List:
    """
    Читает txt файл, где каждая строка содержит числа (токены), разделенные пробелами.
    Возвращает одномерный массив всех токенов.
    """
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                tokens = [int(token) for token in line.strip().split() if token.strip()]
                data.append(tokens)

        print(f"Successfully loaded {len(data)} data points from {file_path}")
        return data

    except Exception as e:
        print(f"Error loading data: {e}")
        raise


# кастомная функция collate_fn для формирования батчей
def collate_fn(batch):
    config = load_config("dataset_config.yaml")
    tokenizer_name = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    input_seq = [item['x'] for item in batch]
    target_seq = [item['y'] for item in batch]
    lengths = torch.tensor([len(seq) for seq in input_seq])

    padded_sequences = pad_sequence(input_seq, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_targets = pad_sequence(target_seq, batch_first=True, padding_value=tokenizer.pad_token_id)

    return {
        'input_seq': padded_sequences,
        'lengths': lengths,
        'target_seq': padded_targets
    }


def create_dataloader(debug:bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создаёт Dataloader для исходного датасета.
    :param debug: Нужна ли отладочная информация по датасету.
    """
    config = load_config("config.yaml")

    # Получение пути до файла с обработанными данными
    datafile_name = config["processed_dataset"]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    datafile_path = os.path.join(data_dir, datafile_name)

    # Загрузка данных в numpy массив
    data = load_and_preprocess_data(datafile_path)

    train_end_idx = int(config["train_size"] * len(data))
    val_end_idx = int((config["train_size"] + config["val_size"]) * len(data))

    train_data = data[:train_end_idx]
    val_data = data[train_end_idx:val_end_idx]
    test_data = data[val_end_idx:]

    # Создание Dataset
    train_dataset = TextAutocompleteDataset(train_data)
    val_dataset = TextAutocompleteDataset(val_data)
    test_dataset = TextAutocompleteDataset(test_data)

    # Создание DataLoader
    batch_size = config["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn = collate_fn,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn = collate_fn,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn = collate_fn,
        num_workers=4
    )

    if debug:
        print("\n=== TRAIN DATASET INFO ===")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Train batch example: {next(iter(train_loader))}")

        print("\n=== VALIDATION DATASET INFO ===")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Validation batch example: {next(iter(val_loader))}")

        print("\n=== TEST DATASET INFO ===")
        print(f"Test dataset size: {len(test_dataset)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Test batch example: {next(iter(test_loader))}")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloader(debug=True)