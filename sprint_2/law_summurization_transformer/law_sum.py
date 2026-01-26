from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datasets import load_dataset
import evaluate
from tqdm import tqdm

if __name__ == "__main__":
    # Определяем устройство (GPU если доступен, иначе CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    print()

    # Загрузка модели и токенизатора
    model_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Переносим модель на устройство
    model = model.to(device)
    print(f"Model moved to {device}")
    print()

    # 1) Общая конфигурация
    cfg = model.config
    print("Model config:")
    print(f"d_model={cfg.d_model}, num_layers={cfg.num_layers}, feed_forward_size={cfg.d_ff}")
    print()

    # 2) Блок энкодера
    first_encoder_block = model.encoder.block[0]
    print("Encoder block:")
    print(first_encoder_block)
    print()

    # 3) Декодер — блоки с self-attention и cross-attention
    first_decoder_block = model.decoder.block[0]
    print("Decoder block:")
    print(first_decoder_block)
    print()

    # 4) количество параметров
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()

    # Загружаем подмножество для быстрого теста
    dataset = load_dataset("billsum", split="ca_test[:50]")

    # Пример
    print("Example from dataset:")
    print(dataset[0])
    print()

    # тексты и их референсные суммаризации
    texts = [item['text'] for item in dataset]
    references = [item['summary'] for item in dataset]

    tokenized_texts = []

    for text in texts:
        input_text = "summarize: " + text.strip().replace("\n", " ")
        tokenized_input_text = tokenizer.encode(input_text,
                                                return_tensors='pt',
                                                truncation=True,
                                                max_length=512,
                                                padding='max_length')
        # Переносим тензор на устройство
        tokenized_input_text = tokenized_input_text.to(device)
        tokenized_texts.append(tokenized_input_text)

    print('Example of tokenized text (device):')
    print(f"Shape: {tokenized_texts[0].shape}, Device: {tokenized_texts[0].device}")
    print(tokenized_texts[0])
    print()

    generated_summaries = []

    for inputs in tqdm(tokenized_texts, desc="Generating summaries"):
        with torch.no_grad():
            # Генерация на том же устройстве, где находится модель
            summary_ids = model.generate(inputs,
                                         max_length=50,
                                         min_length=10,
                                         length_penalty=2.0,
                                         num_beams=4,
                                         early_stopping=True)

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        generated_summaries.append(summary)

    rouge = evaluate.load("rouge")

    # Подсчёт метрик
    results = rouge.compute(predictions=generated_summaries, references=references)

    print('\nMetrics')
    print("-" * 40)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # пример сгенерированного саммари
    print('\nSummary example:')
    print("-" * 40)
    print(f"Reference: {references[0][:200]}...")
    print(f"Generated: {generated_summaries[0][:200]}...")

    # Очистка памяти GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(
            f"\nMemory after cleanup: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB allocated, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB cached")