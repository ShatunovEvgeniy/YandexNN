import evaluate
from typing import List, Union


def calculate_rouge_metrics(hypotheses: List[str], references: List[str]) -> dict:
    """
    Вычисляет метрики ROUGE-1 и ROUGE-2 между гипотезами и референсами.

    Args:
        hypotheses: Список сгенерированных текстов (гипотез)
        references: Список эталонных текстов (референсов)

    Returns:
        dict: Словарь с результатами ROUGE-1 и ROUGE-2 метрик (в процентах)
    """
    # Загружаем метрику ROUGE
    rouge = evaluate.load("rouge")

    # Вычисляем все метрики ROUGE
    results = rouge.compute(
        predictions=hypotheses,
        references=references,
        use_stemmer=True,
        rouge_types=["rouge1", "rouge2"]
    )

    # Форматируем результаты и переводим в проценты
    formatted_results = {
        'rouge1': results['rouge1'] * 100,  # ROUGE обычно возвращается как доля, умножаем на 100
        'rouge2': results['rouge2'] * 100
    }
    return formatted_results


if __name__ == "__main__":
    hypothesis = ["the cat sat on the mat"]
    reference = ["the cat is sitting on the mat"]

    # Вычисляем метрики
    rouge_scores = calculate_rouge_metrics(hypothesis, reference)

    # Выводим результаты
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"\nROUGE-2: {rouge_scores['rouge2']:.4f}")
