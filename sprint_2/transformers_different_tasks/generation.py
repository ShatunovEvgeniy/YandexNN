from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1) Загрузка токенизатора и модели
model_name = "distilgpt2"          # лёгкая версия GPT-2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2) Создаём pipeline для генерации
generator = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # -1 = CPU; 0 = первый GPU (если есть)
)

# 3) Исходный промпт
prompt = "Когда искусственный интеллект станет по-настоящему творческим,"

# 4) Генерируем продолжение
out = generator(
    prompt,
    max_length=80,       # итоговая длина (включая prompt)
    num_return_sequences=1,
    do_sample=True,      # стохастическая генерация
    top_p=0.95,          # nucleus sampling
    temperature=0.8
)

print(out[0]["generated_text"])