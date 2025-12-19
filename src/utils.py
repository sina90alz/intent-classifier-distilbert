from sklearn.metrics import (
accuracy_score,
f1_score,
precision_score,
recall_score
)



def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for intent classification.


    Metrics:
    - accuracy: overall correctness
    - f1_weighted: handles class imbalance
    - f1_macro: treats all intents equally (important)
    - precision_macro
    - recall_macro
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)


    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "precision_macro": precision_score(labels, predictions, average="macro"),
        "recall_macro": recall_score(labels, predictions, average="macro"),
    }