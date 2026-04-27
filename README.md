# Intent Classifier with DistilBERT

FastAPI intent classification service using a fine-tuned `distilbert-base-uncased` model on SNIPS intents.

## Architecture

```text
SNIPS data -> CSV files -> DistilBERT training -> model_dir -> FastAPI /predict
```

Core flow:

- `scripts/prepare_data.py` prepares `data/processed/train.csv` and `valid.csv`.
- `src/train.py` trains and saves the model to `model_dir/`.
- `app/main.py` serves predictions through FastAPI.
- `scripts/run_eval.py` and report scripts evaluate model confidence and errors.

## Train

```bash
python scripts/prepare_data.py --src data/raw_snips/snips --dst data/processed

python -m src.train ^
  --data_dir data/processed ^
  --output_dir model_dir
```

## Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker:

```bash
docker build -t intent-classifier .
docker run -p 8000:8000 intent-classifier
```

`model_dir/` must exist before running the API.

## Predict

`POST /predict`

```json
{
  "text": "book a table for two tonight"
}
```

```json
{
  "intent": "BookRestaurant",
  "confidence": 0.9821
}
```

Docs:

```text
http://localhost:8000/docs
```

## Layout

```text
app/        FastAPI service
src/        training and inference code
scripts/    data prep and evaluation scripts
data/       processed and evaluation data
reports/    generated reports
artifacts/  prediction outputs
```
