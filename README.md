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
Intent-Classifier-Distilbert/
├── app/
│   ├── main.py
│   ├── schemas.py
│   └── service.py
├── src/
│   ├── train.py
│   └── inference.py
├── model_dir/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
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