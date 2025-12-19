import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IntentClassifier:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        # Load label mapping from model config
        self.id2label = self.model.config.id2label

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
        intent = self.id2label[predicted_id.item()]

        return {
            "intent": intent,
            "confidence": round(confidence.item(), 4),
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference.py \"your text here\"")
        sys.exit(1)

    text = sys.argv[1]

    classifier = IntentClassifier(model_dir="model_dir")
    result = classifier.predict(text)

    print(result)
