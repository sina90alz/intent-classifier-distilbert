from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
from typing import Tuple


MODEL_NAME = "distilbert-base-uncased"


def get_tokenizer(model_name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)

def get_model(num_labels: int, model_name: str = MODEL_NAME):
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model

def to_device(batch: dict, device: torch.device):
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}