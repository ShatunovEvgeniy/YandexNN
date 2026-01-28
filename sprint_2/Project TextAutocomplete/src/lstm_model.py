import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer
import math

from src.utils import load_config


class TextAutocompleteLSTM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, padding_idx: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax()
        self._init_weights()

        self.config = load_config('config.yaml')

        tokenizer_name = self.config['tokenizer']
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, input, lengths):
        embedded = self.emb(input)
        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.lstm(packed_input)
        lstm_output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        logits = self.fc(lstm_output)
        return logits

    def generate(self, input_seq, max_length: int = 50):
        """
        Упрощенная версия generate
        """
        self.eval()
        device = next(self.parameters()).device

        # Преобразуем input_seq в тензор, если нужно
        if not isinstance(input_seq, torch.Tensor):
            input_seq = torch.tensor(input_seq, dtype=torch.long)

        input_seq = input_seq.to(device)

        with torch.no_grad():
            batch_size = input_seq.size(0)
            generated_sequences = []

            for batch_idx in range(batch_size):
                # Берем один пример из батча
                seq = input_seq[batch_idx]

                # Удаляем pad токены (если pad_token_id=0)
                non_zero_mask = seq != 0
                if non_zero_mask.any():
                    seq = seq[non_zero_mask]

                # Если последовательность пустая, пропускаем
                if len(seq) == 0:
                    generated_sequences.append(torch.tensor([], dtype=torch.long))
                    continue

                # Готовим вход для LSTM
                current_input = seq.unsqueeze(0)  # [1, seq_len]

                # Получаем hidden state из контекста
                embedded = self.emb(current_input)
                _, (h_n, c_n) = self.lstm(embedded)

                # Генерация
                generated_tokens = []

                for _ in range(max_length):
                    # Берем последний токен
                    if len(generated_tokens) > 0:
                        last_token = torch.tensor([[generated_tokens[-1]]], device=device)
                    else:
                        last_token = current_input[:, -1:]

                    # Прямой проход
                    embedded = self.emb(last_token)
                    lstm_out, (h_n, c_n) = self.lstm(embedded, (h_n, c_n))
                    logits = self.fc(lstm_out.squeeze(1))

                    # Greedy выбор
                    next_token = torch.argmax(logits, dim=-1)

                    # Проверка на EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                    generated_tokens.append(next_token.item())

                # Сохраняем результат
                generated_sequences.append(torch.tensor(generated_tokens, dtype=torch.long))

            return generated_sequences