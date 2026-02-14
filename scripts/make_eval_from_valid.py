import argparse
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", default="data/processed/valid.csv")
    p.add_argument("--out_csv", default="data/eval/valid_eval.csv")
    p.add_argument("--source", default="valid")
    args = p.parse_args()

    inp = Path(args.in_csv)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    # Your schema is text,label
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected columns text,label. Found: {list(df.columns)}")

    out_df = pd.DataFrame(
        {
            "id": range(1, len(df) + 1),
            "text": df["text"].astype(str),
            "label": df["label"].astype(str),
            "source": args.source,
        }
    )

    out_df.to_csv(out, index=False)
    print(f"[OK] Wrote {len(out_df)} rows -> {out}")


if __name__ == "__main__":
    main()
