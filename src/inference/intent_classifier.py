import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IntentClassifier:
    def __init__(self, model_dir: str, device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict_batch(self, texts: list[str], max_length: int = 128):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)  # (batch, num_labels)

        conf, pred_id = torch.max(probs, dim=-1)       
        pred_labels = [self.id2label[i.item()] for i in pred_id]

        return {
            "pred_label": pred_labels,
            "confidence": conf.detach().cpu().tolist(),
            "probs": probs.detach().cpu().tolist(),  # list[list[float]]
            "input_ids": inputs["input_ids"].detach().cpu().tolist(),
            "attention_mask": inputs["attention_mask"].detach().cpu().tolist(),
        }
