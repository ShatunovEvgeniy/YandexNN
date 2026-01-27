from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from clearml import Task, Logger

from utils import load_config
from lstm_model import TextAutocompleteLSTM
from next_token_dataset import create_dataloaders
from eval_lstm import calculate_rouge_metrics


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

    for batch in loader:
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

        # Декодируем ПРЕДСКАЗАНИЯ в текст
        probs = torch.softmax(logits, dim=-1)
        output_indices = torch.argmax(probs, dim=-1)
        output_text = model.tokenizer.batch_decode(output_indices, skip_special_tokens=True)

        # Декодируем ЦЕЛЕВЫЕ ПОСЛЕДОВАТЕЛЬНОСТИ в текст
        target_text = model.tokenizer.batch_decode(target_seq, skip_special_tokens=True)

        # Вычисляем ROUGE метрики
        metrics = calculate_rouge_metrics(output_text, target_text)
        total_rouge1 += metrics['rouge1']
        total_rouge2 += metrics['rouge2']
        batch_count += 1

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

    with torch.no_grad():
        for batch in loader:
            input_seq = batch['input_seq'].to(device)
            lengths = batch['lengths']
            target_seq = batch['target_seq'].to(device)

            logits = model(input_seq, lengths)
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             target_seq.reshape(-1))
            total_loss += loss.item()

            # Декодируем предсказания и цели в текст
            probs = torch.softmax(logits, dim=-1)
            output_indices = torch.argmax(probs, dim=-1)
            output_text = model.tokenizer.batch_decode(output_indices, skip_special_tokens=True)
            target_text = model.tokenizer.batch_decode(target_seq, skip_special_tokens=True)

            metrics = calculate_rouge_metrics(output_text, target_text)
            total_rouge1 += metrics['rouge1']
            total_rouge2 += metrics['rouge2']
            batch_count += 1

    avg_loss = total_loss / len(loader)
    avg_rouge1 = total_rouge1 / batch_count if batch_count > 0 else 0
    avg_rouge2 = total_rouge2 / batch_count if batch_count > 0 else 0
    return avg_loss, avg_rouge1, avg_rouge2


if __name__ == "__main__":
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

    num_epochs = config['epochs']
    for epoch in range(num_epochs):
        # TRAIN
        train_loss, train_rouge1, train_rouge2 = train_epoch(model, train_loader, optimizer, criterion, device)

        # Отправка метрик в ClearML
        Logger.current_logger().report_scalar("train", "loss", iteration=epoch, value=train_loss)
        Logger.current_logger().report_scalar("train", "ROUGE1", iteration=epoch, value=train_rouge1)
        Logger.current_logger().report_scalar("train", "ROUGE2", iteration=epoch, value=train_rouge2)
        print(f'Epoch {epoch + 1}/{num_epochs} — Loss: {train_loss:.4f}, ROUGE1: {train_rouge1:.2f0}%, ROUGE2: {train_rouge2:.2f}%')

        # VALIDATION
        val_loss, val_rouge1, val_rouge2 = evaluate(model, criterion, val_loader, device)

        # Отправка метрик в ClearML
        Logger.current_logger().report_scalar("val", "loss", iteration=epoch, value=val_loss)
        Logger.current_logger().report_scalar("val", "ROUGE1", iteration=epoch, value=val_rouge1)
        Logger.current_logger().report_scalar("val", "ROUGE2", iteration=epoch, value=val_rouge2)
        print(f'Val Loss: {val_loss:.4f}, Val ROUGE1: {val_rouge1:.2f}%, , Val ROUGE2: {val_rouge2:.2f}%')
