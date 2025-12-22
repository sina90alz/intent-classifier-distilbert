# Intent Classification API with DistilBERT

A production-ready **Intent Classification service** built with **DistilBERT**, trained on the **SNIPS custom intent dataset**, and exposed as a **Dockerized FastAPI REST API**.

This project demonstrates the **full ML lifecycle**: data preparation → model training → evaluation → inference → API deployment.

---

## 🚀 Project Overview

The goal of this project is to classify user utterances into predefined intents (e.g. `AddToPlaylist`, `BookRestaurant`, `GetWeather`) using a modern transformer-based NLP model.

Key characteristics:

- Fine-tuned **DistilBERT** for intent classification
- Real-world dataset (**SNIPS custom intents**)
- Clean separation between **training** and **inference**
- **FastAPI** service for real-time predictions
- **Dockerized** for reproducible deployment

---

## Model & Dataset

### Model
- **Base model**: `distilbert-base-uncased`
- **Task**: Multi-class text classification (intent detection)
- **Framework**: Hugging Face Transformers + PyTorch
- **Label handling**: `label2id` / `id2label` stored in model config (no external encoders)

### Dataset
- **Source**: SNIPS Custom Intent Engines (2017-06)
- **Intents**: 7
- **Splits**:
  - Train: ~2100 samples
  - Validation: ~700 samples

---

## Training Results

- **Validation Accuracy**: ~99.5%
- **Macro F1 Score**: ~99.5%
- Stable metrics across epochs

---

## Project Structure

```

intent-classifier-distilbert/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   ├── raw_snips/                    
│   │   └── snips/ 
│   │
│   └── processed/              # ✅ canonical training data
│       ├── train.csv
│       ├── valid.csv
│
├── scripts/
│   └── prepare_data.py         # SNIPS JSON → CSV (data engineering step)
│
├── src/
│   ├── train.py                # training / fine-tuning
│   ├── inference.py            # inference logic
│   ├── dataset.py              # CSV dataset loader
│   └── utils.py                # metrics, helpers
│
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   └── schemas.py              # request / response schemas
│   └── service.py
│
└── model_dir/                  # generated after training

```

---

## API Usage

```bash
docker build -t intent-classifier .
docker run -p 8000:8000 intent-classifier
```

Swagger UI:

```
http://localhost:8000/docs
```

---