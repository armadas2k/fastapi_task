# Создаем приложение
from fastapi import FastAPI
from pydantic import BaseModel
from ML.model import model_load
import uvicorn

model = None
app = FastAPI()


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
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
@app.post("/get_results")
def predict_sentiment(request: SentimentRequest):
    sentiment = model(request.text)

    response = SentimentResponse(
        sentiment_label=sentiment.label,
    )

    return response

