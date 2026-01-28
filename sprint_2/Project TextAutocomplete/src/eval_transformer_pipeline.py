from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.utils import load_config


def transformer_generate(model, tokenizer, input_ids: torch.Tensor, device, max_new_tokens=10):
    model = model.to(device)
    input_ids = input_ids.to(device)

    # Создаем attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    # Проверяем размеры
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Max input ID: {input_ids.max().item()}")
    print(f"Vocab size: {model.config.vocab_size}")

    # Убедимся, что все индексы в пределах vocab
    if input_ids.max().item() >= model.config.vocab_size:
        raise ValueError(f"Input ID {input_ids.max().item()} exceeds vocab size {model.config.vocab_size}")

    try:
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1,
        )

        # Декодируем только сгенерированную часть
        generated_texts = []
        for i in range(len(outputs)):
            generated_part = outputs[i, input_ids.shape[1]:]
            # Удаляем pad tokens в конце
            non_pad_mask = generated_part != tokenizer.pad_token_id
            if non_pad_mask.any():
                generated_part = generated_part[:non_pad_mask.sum()]
            generated_text = tokenizer.decode(generated_part, skip_special_tokens=True)
            generated_texts.append(generated_text)

        return generated_texts

    except Exception as e:
        print(f"Error during generation: {e}")
        # Fallback: вернем пустые строки
        return [""] * len(input_ids)


if __name__ == "__main__":
    config = load_config('config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Создаём модель и токенизатор
        model_name = config['transformer_model']
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)

        tokenizer_name = config['tokenizer']
        print(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side='left')

        # Настройка pad token
        if tokenizer.pad_token is None:
            print(f"Setting pad_token to: {config['pad_token']}")
            tokenizer.add_special_tokens({'pad_token': config['pad_token']})
            # Если у модели есть embedding слой, нужно его изменить
            model.resize_token_embeddings(len(tokenizer))

        # Проверим токены
        print(f"Pad token ID: {tokenizer.pad_token_id}")
        print(f"EOS token ID: {tokenizer.eos_token_id}")

        # Batch обработка
        texts = ["I love to eat", "Machine learning is"]
        print(f"\nProcessing {len(texts)} texts")

        # Токенизация
        batch_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Ограничим максимальную длину
        )
        print(f"Batch input IDs shape: {batch_inputs.input_ids.shape}")

        # Работа модели
        model = model.to(device)
        completions = transformer_generate(
            model,
            tokenizer,
            batch_inputs.input_ids,
            device
        )

        print("\nBatch results:")
        for text, completion in zip(texts, completions):
            print(f"Input: '{text}'")
            print(f"Completion: '{completion}э")
            print("-" * 30)

    except Exception as e:
        print(f"Error: {e}")
