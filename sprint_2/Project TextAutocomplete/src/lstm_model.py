import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextAutocompleteLSTM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, padding_idx: int, eos_idx: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax()
        self._init_weights()

        self.pad_token_id = padding_idx
        self.eos_token_id = eos_idx

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, inputs, lengths=None, hidden=None):
        embedded = self.emb(inputs)  # [batch_size, seq_len, hidden_dim]

        if lengths is not None:
            # Используем pack_padded_sequence для эффективности
            packed_input = pack_padded_sequence(
                embedded, lengths,
                batch_first=True,
                enforce_sorted=False
            )
            packed_output, hidden = self.lstm(packed_input, hidden)
            lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            # Простой forward без packing
            lstm_output, hidden = self.lstm(embedded, hidden)

        logits = self.fc(lstm_output)  # [batch_size, seq_len, vocab_size]
        return logits, hidden

    def generate(self, input_seq: torch.Tensor, max_length: int = 50):
        """
        Генерация продолжения для входных последовательностей.
        """
        self.eval()
        device = next(self.parameters()).device

        # Добавляем batch dimension если нужно
        if input_seq.dim() == 1:
            input_seq = input_seq.unsqueeze(0)

        input_seq = input_seq.to(device)
        batch_size = input_seq.size(0)

        # Удаляем pad токены из каждой последовательности
        processed_inputs = []
        original_lengths = []

        for i in range(batch_size):
            seq = input_seq[i]
            # Ищем индексы не-pad токенов
            non_pad_mask = (seq != self.pad_token_id)
            valid_tokens = seq[non_pad_mask]

            # Если вся последовательность из pad токенов, используем пустую
            if len(valid_tokens) == 0:
                valid_tokens = torch.tensor([], dtype=torch.long, device=device)

            processed_inputs.append(valid_tokens)
            original_lengths.append(len(valid_tokens))

        # Находим максимальную длину без pad токенов
        max_len = max(original_lengths)

        # Создаем тензор без pad токенов
        clean_inputs = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
            device=device
        )

        for i, tokens in enumerate(processed_inputs):
            if len(tokens) > 0:
                clean_inputs[i, :len(tokens)] = tokens

        # Получаем initial hidden state через forward
        logits, hidden = self.forward(torch.tensor(clean_inputs))

        # Начинаем генерацию с последних токенов
        last_tokens = clean_inputs[:, -1].unsqueeze(1)  # [batch_size, 1]

        # Маска для отслеживания завершенных последовательностей
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Сохраняем сгенерированные токены
        generated = [[] for _ in range(batch_size)]

        with torch.no_grad():
            for step in range(max_length):
                # Пропускаем уже завершенные последовательности
                if finished.all():
                    break

                # Forward для одного токена
                embedded = self.emb(last_tokens)  # [batch_size, 1, hidden_dim]
                lstm_output, hidden = self.lstm(embedded, hidden)
                logits = self.fc(lstm_output.squeeze(1))  # [batch_size, vocab_size]

                # Greedy выбор следующего токена
                next_tokens = torch.argmax(logits, dim=-1)  # [batch_size]

                # Обновляем маску завершения
                eos_mask = (next_tokens == self.eos_token_id)
                finished = finished | eos_mask

                # Сохраняем сгенерированные токены
                for i in range(batch_size):
                    if not finished[i]:
                        generated[i].append(next_tokens[i].item())

                # Подготавливаем вход для следующего шага
                last_tokens = next_tokens.unsqueeze(1)  # [batch_size, 1]

        return generated

