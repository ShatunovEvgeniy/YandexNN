import os
import random
from functools import partial

import numpy as np

import timm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchmetrics

from transformers import AutoModel, AutoTokenizer

from dataset import MultimodalDataset, collate_fn, get_transforms


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


class MultimodalModel(nn.Module):
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

        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(config.HIDDEN_DIM // 2, config.NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, image):
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:,  0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        fused_emb = text_emb * image_emb

        logits = self.classifier(fused_emb)
        return logits


def train(config, device, mask=None):
    seed_everything(config.SEED)

    # Инициализация модели
    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    set_requires_grad(model.text_model,
                      unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    # Оптимизатор с разными LR
    optimizer = AdamW([{
        'params': model.text_model.parameters(),
        'lr': config.TEXT_LR
    }, {
        'params': model.image_model.parameters(),
        'lr': config.IMAGE_LR
    }, {
        'params': model.classifier.parameters(),
        'lr': config.CLASSIFIER_LR
    }])

    criterion = nn.CrossEntropyLoss()

    # Загрузка данных
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")
    train_dataset = MultimodalDataset(config, transforms, mask=mask)
    val_dataset = MultimodalDataset(config, val_transforms, ds_type="val", mask=mask)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn, tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn, tokenizer=tokenizer))

    # Инициализация метрик
    f1_metric_train = torchmetrics.F1Score(
        task="binary" if config.NUM_CLASSES == 2 else "multiclass",
        num_classes=config.NUM_CLASSES).to(device)
    f1_metric_val = torchmetrics.F1Score(
        task="binary" if config.NUM_CLASSES == 2 else "multiclass",
        num_classes=config.NUM_CLASSES).to(device)
    
    best_f1_val = 0.0

    print("Training started")
    
    # 🔹 Прогресс-бар по эпохам
    for epoch in tqdm(range(config.EPOCHS), desc="Epochs", leave=True):
        model.train()
        total_loss = 0.0

        # 🔹 Прогресс-бар по батчам обучения
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for batch in train_pbar:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device)
            }
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(**inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predicted = logits.argmax(dim=1)
            _ = f1_metric_train(preds=predicted, target=labels)
            
            # 🔹 Обновление описания бара в реальном времени
            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (train_pbar.n + 1):.4f}"
            })

        # Вычисление метрик после эпохи
        train_f1 = f1_metric_train.compute().cpu().numpy()
        val_f1 = validate(model, val_loader, device, f1_metric_val)
        
        f1_metric_train.reset()
        f1_metric_val.reset()

        # 🔹 Вывод итогов эпохи
        epoch_msg = (f"Epoch {epoch+1}/{config.EPOCHS} | "
                     f"Loss: {total_loss/len(train_loader):.4f} | "
                     f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
        tqdm.write(epoch_msg)  # tqdm.write предотвращает ломку прогресс-бара

        # Сохранение лучшей модели
        if val_f1 > best_f1_val:
            best_f1_val = val_f1
            tqdm.write(f"✨ New best model at epoch {epoch+1} (Val F1: {val_f1:.4f})")
            torch.save(model.state_dict(), config.SAVE_PATH)


def validate(model, val_loader, device, f1_metric):
    model.eval()
    
    # 🔹 Прогресс-бар для валидации
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Val]", leave=False):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device)
            }
            labels = batch['label'].to(device)

            logits = model(**inputs)
            predicted = logits.argmax(dim=1)
            _ = f1_metric(preds=predicted, target=labels)

    return f1_metric.compute().cpu().numpy()
