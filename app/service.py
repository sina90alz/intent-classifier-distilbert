import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IntentClassifierService:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        confidence, predicted_id = torch.max(probs, dim=-1)

        return {
            "intent": self.model.config.id2label[predicted_id.item()],
            "confidence": round(confidence.item(), 4),
        }
