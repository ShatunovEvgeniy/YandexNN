import torch
import random
import os
import numpy as np
from torch.optim import AdamW
import torch.nn as nn
import timm
from tqdm import tqdm
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoTokenizer

from dataset import prepare_dataloaders


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class CaloriesModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)
        self.mass_proj = nn.Linear(1, config.HIDDEN_DIM)

        self.ingr_attention = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 4),
            nn.Tanh(),
            nn.Linear(config.HIDDEN_DIM // 4, 1)
        )

        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )

    def forward(self, ingrs, attention_mask, image, mass):
        """
        ingrs: [B, N, L] - batch, num_ingredients, tokens_per_ingredient
        attention_mask: [B, N, L] - маска токенов
        image: [B, C, H, W]
        mass: [B]
        """
        batch_size, num_ingredients, seq_len = ingrs.shape

        # === 1. Обработка каждого ингредиента ===
        ingrs_flat = ingrs.view(-1, seq_len)  # [B*N, L]
        mask_flat = attention_mask.view(-1, seq_len)  # [B*N, L]

        # [B*N, L] -> [B*N, hidden_size]
        text_outputs = self.text_model(ingrs_flat, attention_mask=mask_flat)
        ingredient_embs = text_outputs.last_hidden_state[:, 0, :]  # [CLS]

        # Восстанавливаем: [B*N, H] -> [B, N, H]
        ingredient_embs = ingredient_embs.view(batch_size, num_ingredients, -1)

        # === 2. Projection в общее пространство до attention ===
        ingredient_embs_proj = self.text_proj(ingredient_embs)  # [B, N, HIDDEN_DIM]

        # === 3. Attention-weighted pooling ===
        # Считаем "важность" каждого ингредиента
        attn_scores = self.ingr_attention(ingredient_embs_proj).squeeze(-1)  # [B, N]

        # Маскируем паддинг-ингредиенты: если все токены ингредиента — паддинг, вес = -inf
        ingr_valid_mask = attention_mask.any(dim=-1)  # [B, N], bool
        attn_scores = attn_scores.masked_fill(~ingr_valid_mask, -1e9)

        # Softmax по ингредиентам -> веса в [0, 1], сумма = 1
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, N]

        # Взвешенная сумма эмбеддингов
        text_features = torch.bmm(
            attn_weights.unsqueeze(1),  # [B, 1, N]
            ingredient_embs_proj  # [B, N, HIDDEN_DIM]
        ).squeeze(1)  # [B, HIDDEN_DIM]

        # === 4. Обработка изображения и массы ===
        image_features = self.image_model(image)  # [B, image_hidden_size]

        image_emb = self.image_proj(image_features)  # [B, HIDDEN_DIM]
        numeric_emb = self.mass_proj(mass.unsqueeze(1))  # [B, 1] -> [B, HIDDEN_DIM]

        # === 5. Fusion модальностей ===
        fused_emb = text_features + image_emb + numeric_emb  # [B, HIDDEN_DIM]

        # === 6. Предсказание ===
        calories = self.head(fused_emb).squeeze(-1)  # [B]
        return calories

    @torch.no_grad()
    def infer(
            self,
            ingrs: torch.Tensor,
            attention_mask: torch.Tensor,
            image: torch.Tensor,
            mass: torch.Tensor | float | int,
            device: torch.device | str | None = None,
            return_dict: bool = False,
            clip_negative: bool = True,
    ) -> torch.Tensor | dict:
        """
        Инференс модели для предсказания калорийности.

        Args:
            ingrs: Tensor [B, N, L] - токенизированные ингредиенты
            attention_mask: Tensor [B, N, L] - маска токенов
            image: Tensor [B, C, H, W] - предобработанное изображение
            mass: Tensor [B], float или int - масса блюда в граммах
            device: Устройство для инференса (None = auto)
            return_dict: Если True, возвращает dict с метаданными
            clip_negative: Если True, обрезает отрицательные предсказания до 0

        Returns:
            Tensor [B] с предсказанными калориями ИЛИ dict с результатами
        """
        # Сохраняем и переключаем режим
        training = self.training
        self.eval()

        try:
            # Авто-определение устройства
            if device is None:
                device = next(self.parameters()).device

            # Нормализация mass к тензору [B]
            if isinstance(mass, (int, float)):
                mass = torch.tensor([mass], device=device)
            elif isinstance(mass, torch.Tensor) and mass.dim() == 0:
                mass = mass.unsqueeze(0)

            # Перенос на device
            ingrs = ingrs.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True)
            mass = mass.to(device, non_blocking=True).float()

            # Forward pass
            calories_pred = self.forward(
                ingrs=ingrs,
                attention_mask=attention_mask,
                image=image,
                mass=mass
            )

            # Возврат на CPU для удобства
            calories_pred = calories_pred.cpu()
            mass_cpu = mass.cpu()

            if return_dict:
                result = {
                    'calories': calories_pred,
                    'input_mass': mass_cpu,
                }
                if clip_negative:
                    result['calories_clipped'] = torch.clamp(calories_pred, min=0)
                return result

            return torch.clamp(calories_pred, min=0) if clip_negative else calories_pred

        finally:
            # Восстанавливаем режим обучения
            if training:
                self.train()


def train(config, device, weights_path=None):
    seed_everything(config.SEED)

    # Инициализация модели
    model = CaloriesModel(config).to(device)
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    set_requires_grad(model.text_model, unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=False)
    set_requires_grad(model.image_model, unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=False)

    # Оптимизатор с разными LR
    optimizer = AdamW([{
        'params': model.text_model.parameters(),
        'lr': config.TEXT_LR
    }, {
        'params': model.image_model.parameters(),
        'lr': config.IMAGE_LR
    }, {
        'params': model.head.parameters(),
        'lr': config.MLP_LR
    }])

    # Функция потерь
    criterion = nn.MSELoss()

    # Загрузка данных
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(config)

    # Инициализация метрик
    mae_metric_train = MeanAbsoluteError().to(device)
    mape_metric_train = MeanAbsolutePercentageError().to(device)
    mae_metric_val = MeanAbsoluteError().to(device)
    mape_metric_val = MeanAbsolutePercentageError().to(device)

    best_mae_val = float('inf')

    print("Training started")

    # Обучение
    for epoch in tqdm(range(config.EPOCHS), desc="Epochs", leave=True):
        model.train()
        total_loss = 0.0

        # Прогресс-бар по батчам обучения
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} [Train]", leave=False)
        for batch in train_pbar:
            inputs = {
                'ingrs': batch['ingrs'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device),
            }
            target = batch['target'].to(device)

            optimizer.zero_grad()
            prediction = model(**inputs)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _ = mae_metric_train(preds=prediction, target=target)
            _ = mape_metric_train(preds=prediction, target=target)

            # 🔹 Обновление описания бара в реальном времени
            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (train_pbar.n + 1):.4f}"
            })

        # Вычисление метрик после эпохи
        train_mae = mae_metric_train.compute().cpu().numpy()
        train_mape = mape_metric_train.compute().cpu().numpy()
        val_mae, val_mape = validate(model, val_dataloader, device, mae_metric_val, mape_metric_val)

        mae_metric_train.reset()
        mape_metric_train.reset()
        mae_metric_val.reset()
        mape_metric_val.reset()

        # Вывод итогов эпохи
        epoch_msg = (f"Epoch {epoch + 1}/{config.EPOCHS} | "
                     f"Loss: {total_loss / len(train_dataloader):.4f} | "
                     f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
                     f"Train MAPE: {train_mape:.4f} | Val MAPE: {val_mape:.4f}")
        tqdm.write(epoch_msg)

        # Сохранение лучшей модели
        if val_mae < best_mae_val:
            best_mae_val = val_mae
            tqdm.write(f"✨ New best model at epoch {epoch + 1} (Val MAE: {val_mae:.4f})")
            torch.save(model.state_dict(), config.WEIGHTS_DIR / f"epoch_{epoch + 1}.pth")


def validate(model, val_loader, device, mae_metric, mape_metric):
    model.eval()

    # Прогресс-бар для валидации
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Val]", leave=False):
            inputs = {
                'ingrs': batch['ingrs'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device),
            }
            target = batch['target'].to(device)

            prediction = model(**inputs)
            _ = mae_metric(preds=prediction, target=target)
            _ = mape_metric(preds=prediction, target=target)

    return mae_metric.compute().cpu().numpy(), mape_metric.compute().cpu().numpy()


def evaluate(model, config, device):
    """
    Оценка модели на тестовом наборе данных.

    Args:
        model: CaloriesModel — обученная модель
        config: Configuration файл
        device: torch.device — устройство для вычислений

    Returns:
        tuple: (test_mae, test_mape) — значения метрик в виде numpy.float32
    """
    _, _, test_loader = prepare_dataloaders(config)
    # Инициализация метрик, если не переданы
    mae_metric = MeanAbsoluteError().to(device)
    mape_metric = MeanAbsolutePercentageError().to(device)
    model.eval()

    print(f"\nStarting evaluation on test set ({len(test_loader)} batches)...")

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="[Test]", leave=True)
        for batch in test_pbar:
            inputs = {
                'ingrs': batch['ingrs'].to(device, non_blocking=True),
                'attention_mask': batch['attention_mask'].to(device, non_blocking=True),
                'image': batch['image'].to(device, non_blocking=True),
                'mass': batch['mass'].to(device, non_blocking=True),
            }
            target = batch['target'].to(device, non_blocking=True)

            # Forward pass
            prediction = model(**inputs)

            # Обновление метрик
            _ = mae_metric(preds=prediction, target=target)
            _ = mape_metric(preds=prediction, target=target)

            # Обновление прогресс-бара с текущими значениями
            test_pbar.set_postfix({
                "MAE": f"{mae_metric.compute().item():.4f}",
                "MAPE": f"{mape_metric.compute().item():.4f}"
            })

    # Вычисление финальных значений метрик
    test_mae = mae_metric.compute().cpu().numpy()
    test_mape = mape_metric.compute().cpu().numpy()

    # Вывод результатов
    print(f"\n✅ Test Results:")
    print(f"   MAE:  {test_mae:.4f}")
    print(f"   MAPE: {test_mape:.4f}")

    # Сброс метрик для возможного повторного использования
    mae_metric.reset()
    mape_metric.reset()

    return test_mae, test_mape


@torch.no_grad()
def get_worst_predictions(model, config, device, top_k=5, display_images=True):
    """
    Находит блюда с наибольшими ошибками предсказания на тестовом наборе.
    Возвращает DataFrame с результатами и отображает изображения.

    Args:
        model: CaloriesModel — обученная модель
        config: Configuration — конфиг с путями и параметрами данных
        device: torch.device — устройство для вычислений
        top_k: int — количество худших предсказаний для возврата
        display_images: bool — отображать ли изображения блюд

    Returns:
        pd.DataFrame — таблица с данными о худших предсказаниях
            Columns: dish_name, mass_g, ingredients, calories_true,
                     calories_pred, abs_error, rel_error_pct
    """
    # Загружаем тестовый даталоадер
    _, _, test_loader = prepare_dataloaders(config)

    # Инициализация токенайзера для декодирования ингредиентов
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    model.eval()
    results = []

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="[Analyzing]", leave=True)):
        inputs = {
            'ingrs': batch['ingrs'].to(device, non_blocking=True),
            'attention_mask': batch['attention_mask'].to(device, non_blocking=True),
            'image': batch['image'].to(device, non_blocking=True),
            'mass': batch['mass'].to(device, non_blocking=True).float(),
        }
        target = batch['target'].to(device, non_blocking=True)

        prediction = model(**inputs)
        abs_error = torch.abs(prediction - target)  # [B]
        rel_error = abs_error / (target + 1e-6) * 100  # MAPE в процентах

        # Сбор результатов по батчу
        batch_size = target.shape[0]
        for i in range(batch_size):
            # === Декодирование ингредиентов из токенов в текст ===
            # ingrs имеет форму [B, N, L] -> берём [i, :, :] -> [N, L]
            ingr_tokens = batch['ingrs'][i]  # [N, L]
            ingr_attn_mask = batch['attention_mask'][i]  # [N, L]

            ingredients_list = []
            for j in range(ingr_tokens.shape[0]):  # по каждому ингредиенту
                token_ids = ingr_tokens[j]  # [L]
                attn_mask = ingr_attn_mask[j]  # [L]

                # Фильтруем паддинг токены
                valid_tokens = token_ids[attn_mask.bool()]

                # Декодим в текст
                ingr_text = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()

                if ingr_text:  # добавляем только непустые ингредиенты
                    ingredients_list.append(ingr_text)

            ingredients_str = ", ".join(ingredients_list)

            # === Сохранение изображения для отображения ===
            image_tensor = batch['image'][i].cpu().clone()

            dish_info = {
                'mass_g': batch['mass'][i].item(),
                'ingredients': ingredients_str,
                'calories_true': target[i].item(),
                'calories_pred': prediction[i].item(),
                'abs_error': abs_error[i].item(),
                'rel_error_pct': rel_error[i].item(),
                'image_tensor': image_tensor,  # сохраняем для последующего отображения
            }
            results.append(dish_info)

    # Сортировка по абсолютной ошибке и возврат топ-k
    results.sort(key=lambda x: x['abs_error'], reverse=True)
    worst_k = results[:top_k]

    # Создание DataFrame (без image_tensor, т.к. это не сериализуется в CSV)
    df_worst = pd.DataFrame([
        {k: v for k, v in item.items() if k != 'image_tensor'}
        for item in worst_k
    ])

    # === Отображение изображений ===
    if display_images:
        n_images = len(worst_k)
        fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))
        if n_images == 1:
            axes = [axes]

        for idx, (ax, item) in enumerate(zip(axes, worst_k)):
            # Денормализация изображения
            img = item['image_tensor'].clone()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)

            # Конвертация в PIL для отображения
            img = img.permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(f"#{idx + 1}\nΔ={item['abs_error']:.0f} kcal",
                         fontsize=9, pad=5)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    return df_worst

if __name__ == '__main__':
    from pathlib import Path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class Config:
        SEED = 42

        # Модели
        TEXT_MODEL_NAME = "bert-base-uncased"
        IMAGE_MODEL_NAME = "resnet50"

        # Какие слои размораживаем - совпадают с неймингом в моделях
        TEXT_MODEL_UNFREEZE = "encoder.layer.9|encoder.layer.10|encoder.layer.11|pooler"
        IMAGE_MODEL_UNFREEZE = "layer4.|conv_head|bn2"

        # Гиперпараметры
        BATCH_SIZE = 32
        TEXT_LR = 3e-5  # LR для текстовой модели
        IMAGE_LR = 1e-4  # LR для изображений
        MLP_LR = 5e-4  # LR для классификатора
        EPOCHS = 100
        DROPOUT = 0.15
        HIDDEN_DIM = 256  # размерность проекции признаков моделей

        # Пути
        BASE_DIR = Path.cwd().parent
        DATASET_DIR = BASE_DIR / "data" / "calories"
        IMAGES_DIR = DATASET_DIR / "images"
        DISH_CSV_PATH = DATASET_DIR / "dish.csv"
        INGREDIENTS_CSV_PATH = DATASET_DIR / "ingredients.csv"
        WEIGHTS_DIR = BASE_DIR / "models" / "calories_text_aug"

    config = Config()
    train(config, device)