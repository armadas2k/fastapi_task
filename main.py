# Создаем приложение
from fastapi import FastAPI
from pydantic import BaseModel
from ML.model import model_load

model = None
app = FastAPI()


class SentimentResponse(BaseModel):
    text: str
    sentiment_label: str


# create a route
@app.get("/")
def root():
    return 'Привет. Проверить можно к локальному серверу добавить /docs'


# Делаем чтобы функция запускалась при запуске приложения
@app.on_event("startup")
def startup_event():
    global model
    model = model_load()


# Получаем результат
@app.get("/get_results")
def predict_sentiment(text: str):
    sentiment = model(text)

    response = SentimentResponse(
        text=text,
        sentiment_label=sentiment.label,
    )
    return response

