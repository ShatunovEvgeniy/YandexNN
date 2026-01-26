import re
import numpy as np
import pandas as pd
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.nn as nn
from sklearn.metrics import accuracy_score
from transformers import BertTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


class AmazonRNNDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)  # Исправлено: добавлен dtype=torch.long

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label': self.labels[idx]
        }


class MeanPoolingRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, hidden_dim=256, output_dim=2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        for name, param in self.rnn.named_parameters():
            if "weight" in name:  # Исправлено: правильная проверка имени параметра
                nn.init.xavier_uniform_(param)

    def forward(self, input_ids, mask):
        x = self.embedding(input_ids)
        rnn_out, _ = self.rnn(x)

        rnn_out_normed = self.norm(rnn_out)

        # mean pooling по attention_mask
        mask = mask.unsqueeze(2).expand_as(rnn_out_normed)
        masked_out = mask * rnn_out_normed
        summed = masked_out.sum(dim=1)
        lengths = mask.sum(dim=1)  # Сначала суммируем по последнему измерению
        lengths = torch.clamp(lengths, min=1.0)  # Защита от деления на ноль
        lengths = lengths.sum(dim=1, keepdim=True)  # Суммируем по измерению признаков
        mean_pooled = summed / lengths

        out = self.dropout(mean_pooled)
        logits = self.fc(out)

        return logits


def clean_text(text):
    text = text.lower()  # к нижнему регистру
    text = re.sub(r"[^a-z0-9 ]+", " ", text)  # оставить только буквы и цифры
    text = re.sub(r"\s+", " ", text).strip()  # убрать дублирующиеся пробелы
    return text

def analyze_dataset(texts, labels):
    # Распределение классов
    label_counts = Counter(labels)
    print("Распределение классов:")
    for label, count in label_counts.items():
        print(f"Класс {label}: {count} примеров")

    # Статистика по количеству слов
    word_counts = [len(text.split()) for text in texts]  # количество слов в текстах
    print("\nСтатистика по количеству слов в тексте:")
    print("Среднее:", np.mean(word_counts))  # среднее кол-во слов в текстах
    print("Медиана:", np.median(word_counts))  # медианное кол-во слов в текстах
    print("5-й перцентиль:", np.percentile(word_counts, 5))  # 5й перцентиль кол-ва слов в текстах
    print("95-й перцентиль:", np.percentile(word_counts, 95))  # 95й перцентиль кол-ва слов в текстах

    # Гистограмма распределения длины
    plt.hist(word_counts, bins=50, edgecolor='black')
    plt.title("Распределение количества слов в текстах")
    plt.xlabel("Количество слов")
    plt.ylabel("Частота")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Загрузка датасета
    dataset = load_dataset("amazon_polarity", split="train[:20000]")
    texts, labels = dataset['content'], dataset['label']

    # Очистка текстов
    texts = [clean_text(text) for text in texts]

    # Разбиение на train и val
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Обучение на bag of words
    vectorizer = CountVectorizer(max_features=5000)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_val_bow = vectorizer.transform(X_val)

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train_bow, y_train)
    y_pred = clf.predict(X_val_bow)

    print(classification_report(y_val, y_pred))

    # Обучение RNN
    train_ds = AmazonRNNDataset(X_train, y_train, tokenizer)
    val_ds = AmazonRNNDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # создание модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeanPoolingRNN(vocab_size=tokenizer.vocab_size, pad_idx=tokenizer.pad_token_id).to(device)

    # создание оптимизатора и функции потерь
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # код обучения одной эпохи
    def train_epoch(model, loader):
        model.train()
        total_loss = 0
        for batch in loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)


    # код подсчёта accuracy на валидации
    def evaluate(model, loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['label'].cpu()
                logits = model(ids, mask)
                preds += torch.argmax(logits, dim=1).cpu().tolist()
                trues += labels.tolist()
        return accuracy_score(trues, preds)


    # обучение
    for epoch in range(3):
        loss = train_epoch(model, train_loader)
        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")