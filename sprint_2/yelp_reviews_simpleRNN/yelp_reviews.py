import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from torch.optim import Adam
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import classification_report


class YelpDataset(Dataset):
    def __init__(self, texts, labels, max_len=256):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': torch.tensor(self.texts[idx][:self.max_len], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# кастомная функция collate_fn для формирования батчей
def collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    lengths = torch.tensor([len(text) for text in texts])
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    return {
            'input_ids': padded_texts,
            'lengths': lengths,
            'labels': labels
        }


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed)

        # Используем последнее скрытое состояние для классификации
        out = self.fc(hidden[-1])
        return out


if __name__ == "__main__":
    raw = pd.read_csv('yelp_reviews.csv')
    texts = raw['text'].to_list()
    labels = raw['label'].to_list()

    # разделение выборки на train и test
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # создание токенизатора с помощью класса AutoTokenizer
    model_name = "bert-base-uncased"
    tokenizer =  AutoTokenizer.from_pretrained(model_name)

    # токенизация текстов
    train_texts_tokenized = tokenizer(train_texts, truncation=True)['input_ids']
    val_texts_tokenized = tokenizer(val_texts, truncation=True)['input_ids']

    # создаём датасеты
    train_dataset = YelpDataset(texts=train_texts_tokenized, labels=train_labels)
    val_dataset = YelpDataset(texts=val_texts_tokenized, labels=val_labels)

    batch_size = 64

    # создаём даталоадеры
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f'Количество батчей в train_dataloader: {len(train_dataloader)}')
    print(f'Количество батчей в val_dataloader: {len(val_dataloader)}')

    print('Размерности батчей:')
    for batch in train_dataloader:
        print('input_ids:', batch['input_ids'].shape)
        print('lengths:', batch['lengths'].shape)
        print('labels:', batch['labels'].shape)
        break

    # модель, оптимизатор, функция потерь
    vocab_size = tokenizer.vocab_size

    model = SimpleRNN(vocab_size, embedding_dim=128, hidden_size=128, output_size=5)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    train_losses = []

    n_epochs = 10

    # обучение по эпохам
    for epoch in range(n_epochs):
        model.train()
        total_train_loss, total_val_loss = 0., 0.
        for batch in tqdm(train_dataloader):
            inputs = batch["input_ids"]
            lengths = batch["lengths"]
            labels = batch["labels"]

            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # подсчёт метрик по валидационному датасету
        y_true, y_pred = [], []
        for batch in tqdm(val_dataloader):
            inputs = batch['input_ids']
            lengths = batch['lengths']
            labels = batch['labels']

            with torch.no_grad():
                outputs = model(inputs, lengths)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.tolist())
                y_pred.extend(preds.tolist())

            avg_val_loss = total_train_loss / len(train_dataloader)

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")
        print('Метрики классификации')
        print(classification_report(y_true, y_pred))