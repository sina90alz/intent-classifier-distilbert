from fastapi import FastAPI
from app.schemas import PredictRequest, PredictResponse
from app.service import IntentClassifierService

app = FastAPI(title="Intent Classifier API")

classifier = IntentClassifierService(model_dir="model_dir")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return classifier.predict(req.text)
