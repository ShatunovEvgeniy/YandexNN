from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from clearml import Task, Logger
import os.path

from src.utils import load_config
from src.lstm_model import TextAutocompleteLSTM
from src.next_token_dataset import create_dataloaders
from src.eval_lstm import calculate_rouge_metrics


def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion,
                device):
    model.train()
    total_loss = 0
    total_rouge1 = 0
    total_rouge2 = 0
    batch_count = 0

    for i, batch in enumerate(loader):
        input_seq = batch['input_seq'].to(device)
        lengths = batch['lengths']
        target_seq = batch['target_seq'].to(device)

        optimizer.zero_grad()
        logits = model(input_seq, lengths)
        loss = criterion(logits.reshape(-1, logits.size(-1)),
                         target_seq.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Вычисляем ROUGE метрики
        if i % 10 == 0:  # Вычислять метрики реже
            # Декодируем только небольшую часть для экономии памяти
            with torch.no_grad():
                sample_size = min(8, input_seq.size(0))  # Берем только 8 примеров
                sample_logits = logits[:sample_size]
                sample_target = target_seq[:sample_size]

                probs = torch.softmax(sample_logits, dim=-1)
                output_indices = torch.argmax(probs, dim=-1)
                output_text = model.tokenizer.batch_decode(output_indices, skip_special_tokens=True)
                target_text = model.tokenizer.batch_decode(sample_target, skip_special_tokens=True)

                metrics = calculate_rouge_metrics(output_text, target_text)
                total_rouge1 += metrics['rouge1']
                total_rouge2 += metrics['rouge2']
                batch_count += 1

        # Принудительная очистка памяти после каждого батча
        del input_seq, lengths, target_seq, logits, loss
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader)
    avg_rouge1 = total_rouge1 / batch_count if batch_count > 0 else 0
    avg_rouge2 = total_rouge2 / batch_count if batch_count > 0 else 0
    return avg_loss, avg_rouge1, avg_rouge2


def evaluate(model: nn.Module, criterion, loader: DataLoader, device) -> tuple:
    model.eval()
    total_rouge1 = 0
    total_rouge2 = 0
    total_loss = 0
    batch_count = 0

    # Использование смешанной точности для инференса
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, batch in enumerate(loader):
            input_seq = batch['input_seq'].to(device)
            lengths = batch['lengths']
            target_seq = batch['target_seq'].to(device)

            logits = model(input_seq, lengths)
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             target_seq.reshape(-1))
            total_loss += loss.item()

            # Вычисление метрик только для каждого 5-го батча
            if i % 5 == 0:
                sample_size = min(4, input_seq.size(0))  # Еще меньше примеров для валидации
                sample_logits = logits[:sample_size]
                sample_target = target_seq[:sample_size]

                probs = torch.softmax(sample_logits, dim=-1)
                output_indices = torch.argmax(probs, dim=-1)
                output_text = model.tokenizer.batch_decode(output_indices, skip_special_tokens=True)
                target_text = model.tokenizer.batch_decode(sample_target, skip_special_tokens=True)

                metrics = calculate_rouge_metrics(output_text, target_text)
                total_rouge1 += metrics['rouge1']
                total_rouge2 += metrics['rouge2']
                batch_count += 1

            # Очистка памяти
            del input_seq, lengths, target_seq, logits, loss
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader)
    avg_rouge1 = total_rouge1 / batch_count if batch_count > 0 else 0
    avg_rouge2 = total_rouge2 / batch_count if batch_count > 0 else 0
    return avg_loss, avg_rouge1, avg_rouge2


if __name__ == "__main__":
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

    task = Task.init(project_name='TextAutocomplete', task_name='Train')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("config.yaml")

    # Настройка токенизатора
    tokenizer_name = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': config['pad_token']})
    pad_token_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    # Инициализация модели
    hidden_dim = config["hidden_dim"]
    model = TextAutocompleteLSTM(vocab_size=vocab_size,
                                 hidden_dim=hidden_dim,
                                 padding_idx=pad_token_id)
    model = model.to(device)

    # Загрузка Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders()

    # Создание оптимизатора и функции потерь
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Подготовка к сохранению модели
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_model.pth")
    print("Path for saving a model:", best_model_path)
    best_val_loss = float('inf')

    num_epochs = config['epochs']
    for epoch in range(num_epochs):
        # TRAIN
        train_loss, train_rouge1, train_rouge2 = train_epoch(model, train_loader, optimizer, criterion, device)

        # Отправка метрик в ClearML
        Logger.current_logger().report_scalar("train", "loss", iteration=epoch, value=train_loss)
        Logger.current_logger().report_scalar("train", "ROUGE1", iteration=epoch, value=train_rouge1)
        Logger.current_logger().report_scalar("train", "ROUGE2", iteration=epoch, value=train_rouge2)
        print(f'Epoch {epoch + 1}/{num_epochs} — Loss: {train_loss:.4f}, ROUGE1: {train_rouge1:.2f}%, ROUGE2: {train_rouge2:.2f}%')

        # VALIDATION
        val_loss, val_rouge1, val_rouge2 = evaluate(model, criterion, val_loader, device)

        # Отправка метрик в ClearML
        Logger.current_logger().report_scalar("val", "loss", iteration=epoch, value=val_loss)
        Logger.current_logger().report_scalar("val", "ROUGE1", iteration=epoch, value=val_rouge1)
        Logger.current_logger().report_scalar("val", "ROUGE2", iteration=epoch, value=val_rouge2)
        print(f'Val Loss: {val_loss:.4f}, Val ROUGE1: {val_rouge1:.2f}%, Val ROUGE2: {val_rouge2:.2f}%')

        # Сохранение лучшей модели по валидационной потере
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Сохранение состояния модели
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_rouge1': val_rouge1,
                'val_rouge2': val_rouge2,
                'config': config
            }, best_model_path)
            print(f"Сохранена новая лучшая модель с валидационной потерей: {val_loss:.4f}")
            # Отправка информации о сохранении в ClearML
            Logger.current_logger().report_text(
                f"Сохранена лучшая модель на эпохе {epoch + 1} с val_loss = {val_loss:.4f}")
