from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="cointegrated/rubert-tiny-sentiment-balanced",
    tokenizer="cointegrated/rubert-tiny-sentiment-balanced",
    device=-1
)

text = "Этот фильм был неожиданно интересным и очень трогательным."

result = classifier(text)

print(f"Метка: {result[0]['label']}, Уверенность: {result[0]['score']:.4f}")