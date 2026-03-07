import torch
from pathlib import Path

from utils import train

BASE_DIR = Path.cwd().parent
dataset_path = BASE_DIR / "data" / "multimodal"
images_path = dataset_path / "images"
df_path = dataset_path / "items.csv"

class Config:
    # для воспроизводимости
    SEED = 42

    # Модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"

    # Какие слои размораживаем - совпадают с нэймингом в моделях
    TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE = "blocks.6|conv_head|bn2"
    
    # Гиперпараметры
    BATCH_SIZE = 256 
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    CLASSIFIER_LR = 1e-3
    EPOCHS = 30
    DROPOUT = 0.3
    HIDDEN_DIM = 256
    NUM_CLASSES = 4

    # Пути
    TRAIN_DF_PATH = dataset_path / "imdb_train.csv"
    VAL_DF_PATH = dataset_path / "imdb_val.csv"
    SAVE_PATH = BASE_DIR / "models" / "multimodal" / "best_model_img_mask.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

cfg = Config()
train(cfg, device, mask="image")