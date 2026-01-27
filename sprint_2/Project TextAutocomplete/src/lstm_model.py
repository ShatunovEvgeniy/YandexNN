import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer
import math

from utils import load_config


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

    def generate(self, input: str, max_length: int = 50, beam_width: int = 1):
        """
        Генерация n вариантов продолжения с помощью beam search
        Возвращает: список кортежей (текст, вероятность)
        """
        self.eval()
        with torch.no_grad():
            input_ids = self.tokenizer.encode(input, return_tensors='pt')  # [1, seq_len]
            hidden = None

            # Beam search state: (последовательность, логарифм вероятности, hidden state)
            beams = [(input_ids.clone(), 0.0, hidden)]  # [(tensor, log_prob, hidden)]

            for step in range(max_length):
                new_beams = []
                all_candidates = []

                for beam_seq, beam_log_prob, beam_hidden in beams:
                    # Получаем длины последовательностей
                    current_lengths = torch.tensor([beam_seq.size(1)], dtype=torch.long)

                    # Forward pass
                    embedded = self.emb(beam_seq)
                    packed_input = pack_padded_sequence(embedded, current_lengths, batch_first=True,
                                                        enforce_sorted=True)
                    packed_output, (h_n, c_n) = self.lstm(packed_input, beam_hidden if beam_hidden else None)
                    output, _ = pad_packed_sequence(packed_output, batch_first=True)
                    logits = self.fc(output[:, -1, :])  # Только последний токен

                    # Получаем топ-k наиболее вероятных следующих токенов
                    topk_probs, topk_indices = torch.topk(torch.softmax(logits, dim=-1), k=beam_width, dim=-1)

                    for i in range(beam_width):
                        next_token_id = topk_indices[0, i].unsqueeze(0).unsqueeze(0)  # [1, 1]
                        token_prob = topk_probs[0, i].item()

                        # Создаем новую последовательность
                        new_seq = torch.cat([beam_seq, next_token_id], dim=1)
                        new_log_prob = beam_log_prob + math.log(token_prob + 1e-10)  # избегаем log(0)

                        # Проверяем, достигли ли EOS
                        if next_token_id.item() == self.tokenizer.eos_token_id:
                            all_candidates.append((new_seq, new_log_prob, None))
                        else:
                            new_beams.append((new_seq, new_log_prob, (h_n.clone(), c_n.clone())))

                # Сортируем все кандидаты по вероятности и выбираем лучшие beam_width
                all_candidates.extend(new_beams)
                all_candidates.sort(key=lambda x: x[1], reverse=True)  # сортировка по log_prob
                beams = all_candidates[:beam_width]

                # Проверяем, все ли последовательности завершились EOS
                if all(beam[0][0, -1].item() == self.tokenizer.eos_token_id for beam in beams):
                    break

            # Декодируем результаты
            results = []
            for beam_seq, log_prob, _ in beams:
                text = self.tokenizer.decode(beam_seq[0], skip_special_tokens=False)
                # Нормализуем вероятность по длине
                normalized_prob = math.exp(log_prob / beam_seq.size(1))
                results.append((text, normalized_prob))

            return results
