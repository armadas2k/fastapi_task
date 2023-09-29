from dataclasses import dataclass
from transformers import pipeline



@dataclass
class SentimentPrediction:
    """Класс - результат прогнозирования настроения"""

    label: str


def model_load():
    """Загружаем обученную модель

    модель принимает предложение и возвращает метку


    """
    # Создаем пайплайн пайплайн объединяет три этапа:
    # предварительную обработку, передачу входных данных через модель и последующую обработку
    # -1 ставим чтобы модель работала на CPU
    model_hf = pipeline("sentiment-analysis", model="seara/rubert-tiny2-russian-sentiment", device=-1)

    def pred_model(text: str) -> SentimentPrediction:
        # На вход пайплайна подается текст для оценки
        pred = model_hf(text)
        # Результат получается в виде списка со словарем [{метка : оценка}] извлекаем словарь
        pred_best_class = pred[0]
        return SentimentPrediction(
            label=pred_best_class["label"],
        )

    return pred_model