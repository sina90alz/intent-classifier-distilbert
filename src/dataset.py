import os
import pandas as pd
from datasets import Dataset, DatasetDict


def load_csv_dataset(data_dir: str) -> DatasetDict:
    """
    Load train.csv and valid.csv from data_dir and return a Hugging Face DatasetDict.
    """
    train_path = os.path.join(data_dir, "train.csv")
    valid_path = os.path.join(data_dir, "valid.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError("train.csv not found")
    if not os.path.exists(valid_path):
        raise FileNotFoundError("valid.csv not found")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(valid_df),
    })
