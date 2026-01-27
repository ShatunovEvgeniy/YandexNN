import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
from typing import Tuple, List

from utils import load_config


class TextAutocompleteDataset(Dataset):
    def __init__(self, data, window_size=3):
        """
        data: numpy array или torch tensor с временными рядами
        window_size: размер окна для последовательности (3 для [x(t-1), x(t), x(t+1)])
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size

        # Проверка, что данных достаточно для создания хотя бы одной последовательности
        if len(self.data) < window_size + 1:
            raise ValueError(f"Data length {len(self.data)} is too short for window size {window_size}")

    def __len__(self):
        # Количество возможных последовательностей
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        """
        Возвращает:
        x: [x(t-1), x(t), x(t+1)] для индекса idx
        y: [y(t), y(t+1), y(t+2)] для индекса idx
        """
        # Входная последовательность: [x(t-1), x(t), x(t+1)]
        x = self.data[idx:idx + self.window_size]

        # Целевая последовательность: [x(t), x(t+1), x(t+2)]
        y = self.data[idx + 1:idx + self.window_size + 1]

        return x, y


def load_and_preprocess_data(file_path: str) -> np.array:
    """
    Читает txt файл с числами, разделенными пробелами
    """
    try:
        # Чтение данных из файла
        data = np.loadtxt(file_path, delimiter=' ')
        print(f"Successfully loaded {len(data)} data points")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def create_dataloader(debug:bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создаёт Dataloader для исходного датасета.
    :param debug: Нужна ли отладочная информация по датасету.
    """
    config = load_config("dataset_config.yaml")

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
        drop_last=True  # Удалять последний неполный батч
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True  # Удалять последний неполный батч
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True  # Удалять последний неполный батч
    )

    if debug:
        print("\n=== TRAIN DATASET INFO ===")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Train batches: {len(train_loader)}")

        print("\n=== VALIDATION DATASET INFO ===")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Validation batches: {len(val_loader)}")

        print("\n=== TEST DATASET INFO ===")
        print(f"Test dataset size: {len(test_dataset)}")
        print(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloader(debug=True)