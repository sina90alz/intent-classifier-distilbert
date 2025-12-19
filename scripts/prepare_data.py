import argparse
import json
import os
import glob
import pandas as pd


def extract_text_from_chunks(chunks):
    """
    Concatenate text from SNIPS tokenized format.
    Ignores entity information.
    """
    return "".join(chunk.get("text", "") for chunk in chunks)


def load_split(src_dir, split_prefix):
    """
    Loads files like:
      - train_AddToPlaylist.json
      - validate_AddToPlaylist.json

    Expected JSON format:
    {
      "IntentName": [
        { "data": [ { "text": "..." }, ... ] }
      ]
    }
    """
    rows = []

    pattern = os.path.join(src_dir, f"{split_prefix}_*.json")
    files = glob.glob(pattern)

    if not files:
        raise ValueError(f"No files found for pattern: {pattern}")

    for path in files:
        print(f"Loading {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)

        # content = { "IntentName": [ utterances ] }
        for intent, utterances in content.items():
            for utt in utterances:
                chunks = utt.get("data", [])
                text = extract_text_from_chunks(chunks)
                if text.strip():
                    rows.append({
                        "text": text.strip(),
                        "label": intent
                    })

    return pd.DataFrame(rows)


def main(src, dst):
    os.makedirs(dst, exist_ok=True)

    train_df = load_split(src, "train")
    valid_df = load_split(src, "validate")

    if train_df.empty or valid_df.empty:
        raise ValueError("Train or validation dataset is empty")

    print("\nSummary:")
    print(f"Train samples: {len(train_df)}")
    print(f"Valid samples: {len(valid_df)}")
    print(f"Number of intents: {train_df['label'].nunique()}")

    train_df.to_csv(os.path.join(dst, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(dst, "valid.csv"), index=False)

    print("\n✓ train.csv and valid.csv created successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    main(args.src, args.dst)
