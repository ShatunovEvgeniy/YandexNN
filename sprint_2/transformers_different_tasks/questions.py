from transformers import pipeline

# 1) Загружаем pipeline для вопросно-ответной задачи
qa_pipeline = pipeline(
    task="question-answering",
    model="distilbert-base-uncased-distilled-squad",
    device=-1   # CPU
)

# 2) Пример входных данных
context = """
Transformers are neural network architectures introduced in 2017 by Vaswani et al. 
They rely entirely on self-attention mechanisms and have revolutionized natural language processing.
"""

question = "Who introduced the Transformer architecture?"

# 3) Получаем ответ
result = qa_pipeline({
    "context": context,
    "question": question
})

print(f"Answer: {result['answer']}")