import torch
import torch.nn as nn
import re
import random
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizerFast
from tqdm import tqdm
from sklearn.model_selection import train_test_split


random.seed(42)
torch.manual_seed(42)

# Определяем устройство для вычислений
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def clean_string(text):
    # функция для "чистки" текстов
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # удаление всего, кроме латинских букв, цифр и пробелов
    text = re.sub(r'\s+', ' ', text).strip()  # удаление дублирующихся пробелов, удаление пробелов по краям
    return text


class MaskedBertDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=7):
        # self.samples - список пар (x, y)
        # x - токенизированный текст с пропущенным токеном
        # y - пропущенный токен
        self.samples = []

        for line in texts:
            token_ids = tokenizer.encode(line, add_special_tokens=False, max_length=512, truncation=True)

            # если строка слишком короткая, то пропускаем её
            if len(token_ids) < seq_len:
                continue

            # проходимся по всем токенам в последовательности
            for i in range(1, len(token_ids) - 1):
                '''
                context - список из seq_len // 2 токенов до i-го токена, токена tokenizer.mask_token_id, и seq_len // 2 токенов после i-го токена
                '''
                # берем ровно seq_len//2 токенов до и после позиции i
                start_idx = max(0, i - seq_len // 2)
                end_idx = i + 1 + seq_len // 2

                # создаем контекст с маской в середине
                context = token_ids[start_idx:i] + [tokenizer.mask_token_id] + token_ids[i + 1:end_idx]

                # если контекст слишком короткий, то пропускаем его
                if len(context) < seq_len:
                    continue

                target = token_ids[i]
                self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, rnn_type="GRU", combine="concat"):
        super().__init__()
        self.combine = combine
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)

        rnn_class = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[rnn_type]
        self.rnn = rnn_class(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)

        # out_dim может быть разным в зависимости от значения combine
        out_dim = hidden_dim * 2 if combine=="concat" else hidden_dim

        # выходной линейный слой
        self.fc = nn.Linear(out_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)

        center = out.size(1) // 2

        # скрытые состояния <MASK> токена
        # после двух проходов двунаправленной сети
        hidden_forward = out[:, center, :out.size(2) // 2]
        hidden_backward = out[:, center, out.size(2) // 2:]

        # агрегация скрытых состояний в зависимости от self.combine
        hidden_agg = hidden_forward + hidden_backward if self.combine=="sum" else torch.concat([hidden_forward, hidden_backward], dim=1)

        linear_out = self.fc(hidden_agg)
        return linear_out


# Подсчёт параметров
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# функция замера лосса и accuracy
def evaluate(model, criterion, loader):
    model.eval()
    correct, total = 0, 0
    sum_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            # Переносим тензоры на устройство
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_output = model(x_batch)
            loss = criterion(y_output, y_batch)
            preds = torch.argmax(y_output, dim=1)
            correct += (preds==y_batch).sum().item()
            total += y_batch.size(0)
            sum_loss += loss.item()

    # лосс и accuracy
    avg_loss = sum_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

if __name__ == "__main__":
    # загружаем датасет WikiText-2
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # длины последовательностей в датасете
    # seq_len = 7 => 3 токена до <MASK> + токен <MASK> + 3 токена после
    seq_len = 7

    # удаляем слишком короткие тексты
    texts = [line for line in dataset["text"] if len(line.split()) >= seq_len]

    # "чистим" тексты
    cleaned_texts = list(map(clean_string, texts))

    # для упрощения используем только max_texts_count текстов
    max_texts_count = 7000

    # разбиение на тренировочную и валидационную выборки
    val_size = 0.05
    train_texts, val_texts = train_test_split(cleaned_texts[:max_texts_count], test_size=val_size, random_state=42)
    print(f"Train texts: {len(train_texts)}, Val texts: {len(val_texts)}")

    # загружаем токенизатор
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # тренировочный и валидационный датасеты
    train_dataset = MaskedBertDataset(train_texts, tokenizer, seq_len=seq_len)
    val_dataset = MaskedBertDataset(val_texts, tokenizer, seq_len=seq_len)

    # даталоадеры
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # размер словаря
    vocab_size = tokenizer.vocab_size

    # размер скрытого состояния
    hidden_dim = 128

    # типы рекуррентных блоков
    rnn_types = ["RNN", "GRU", "LSTM"]

    # методы агрегаций скрытых состояний
    combine_methods = ["sum", "concat"]

    # Сравнение
    print(f"{'RNN Type':<6} | {'Combine':<6} | {'Params':>10}")
    print("-" * 35)
    for rnn_type in rnn_types:
        for combine in combine_methods:
            model = BiRNNClassifier(vocab_size, hidden_dim, rnn_type, combine)
            param_count = count_parameters(model)
            print(f"{rnn_type:<6} | {combine:<6} | {param_count:>10,}")

    # объекты модели, оптимизатора, функции потерь
    model = BiRNNClassifier(vocab_size, rnn_type="LSTM", combine="concat")
    # Переносим модель на устройство
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    # Основной цикл обучения
    n_epochs = 3

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # Переносим тензоры на устройство
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_output = model(x_batch)
            loss = criterion(y_output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_acc = evaluate(model, criterion, val_loader)
        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}")

    # Инференс
    model.eval()
    bad_cases, good_cases = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            # Переносим тензоры на устройство
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)
            for i in range(len(y_batch)):
                input_tokens = tokenizer.convert_ids_to_tokens(x_batch[i].cpu().tolist())
                true_tok = tokenizer.convert_ids_to_tokens([y_batch[i].cpu().item()])[0]
                pred_tok = tokenizer.convert_ids_to_tokens([preds[i].cpu().item()])[0]

                if preds[i] != y_batch[i]:
                    bad_cases.append((input_tokens, true_tok, pred_tok))
                else:
                    good_cases.append((input_tokens, true_tok, pred_tok))

    random.seed(42)
    bad_cases_sampled = random.sample(bad_cases, 5)
    good_cases_sampled = random.sample(good_cases, 5)

    print("\nSome incorrect predictions:")
    for context, true_tok, pred_tok in bad_cases_sampled:
        print(f"Input: {' '.join(context)} | True: {true_tok} | Predicted: {pred_tok}")

    print("\nSome correct predictions:")
    for context, true_tok, pred_tok in good_cases_sampled:
        if true_tok == pred_tok:
            print(f"Input: {' '.join(context)} | True: {true_tok} | Predicted: {pred_tok}")