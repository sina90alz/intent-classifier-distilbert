"""
Train an intent classification model using DistilBERT.

Responsibilities of this file:
- Load and parse SNIPS JSON data
- Build label2id / id2label mappings
- Split data into train / validation
- Tokenize text
- Train the model with Hugging Face Trainer
- Save the trained model
"""

import argparse
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from src.dataset import load_csv_dataset
from src.utils import compute_metrics

MODEL_NAME = "distilbert-base-uncased"


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize text examples for DistilBERT."""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def main(args):

    datasets = load_csv_dataset(args.data_dir)

    train_ds = datasets["train"]
    valid_ds = datasets["validation"]

    labels = sorted(set(train_ds["label"]))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    def encode_labels(example):
        example["label"] = label2id[example["label"]]
        return example
    
    # 4. Train / validation split
    train_ds = train_ds.map(encode_labels)
    valid_ds = valid_ds.map(encode_labels)

    # 5. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = train_ds.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
    )
    valid_ds = valid_ds.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
    )

    # 6. Model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    # 7. Training configuration
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=50,
    )

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 9. Train & save
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="model_dir")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--validation_ratio", type=float, default=0.1)

    args = parser.parse_args()
    main(args)