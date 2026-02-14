import argparse
import json
from pathlib import Path

import pandas as pd

from src.inference.intent_classifier import IntentClassifier


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--eval_csv", default="data/eval/valid_eval.csv")
    parser.add_argument("--out_csv", default="artifacts/preds_eval.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    eval_path = Path(args.eval_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(eval_path)

    required = {"id", "text", "label", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {eval_path}: {sorted(missing)}")

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    df["source"] = df["source"].astype(str)

    clf = IntentClassifier(model_dir=args.model_dir)

    rows = []
    texts = df["text"].tolist()

    for start in range(0, len(df), args.batch_size):
        batch_df = df.iloc[start : start + args.batch_size]
        batch_texts = batch_df["text"].tolist()

        pred = clf.predict_batch(batch_texts, max_length=args.max_length)

        for i, (_, r) in enumerate(batch_df.iterrows()):
            probs = pred["probs"][i]
            pred_label = pred["pred_label"][i]
            confidence = float(pred["confidence"][i])

            input_ids = pred["input_ids"][i]
            attention_mask = pred["attention_mask"][i]
            token_len = int(sum(attention_mask)) if attention_mask is not None else len(input_ids)

            true_label = str(r["label"])
            is_correct = (pred_label == true_label)

            rows.append(
                {
                    "id": r["id"],
                    "text": r["text"],
                    "label_true": true_label,
                    "pred_label": pred_label,
                    "confidence": round(confidence, 6),
                    "is_correct": bool(is_correct),
                    "text_len_chars": len(str(r["text"])),
                    "text_len_tokens": token_len,
                    "source": str(r["source"]),
                    "probs_json": json.dumps(probs),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(out_df)} rows -> {out_path}")

    parquet_path = out_path.with_suffix(".parquet")
    try:
        out_df.to_parquet(parquet_path, index=False)
        print(f"[OK] Also wrote -> {parquet_path}")
    except Exception as e:
        print(f"[WARN] Parquet not written (install pyarrow). Reason: {e}")


if __name__ == "__main__":
    main()
