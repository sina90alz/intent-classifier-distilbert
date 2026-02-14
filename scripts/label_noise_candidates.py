import argparse
import json
from pathlib import Path

import pandas as pd


def top2_margin(probs_json: str) -> float:
    try:
        probs = json.loads(probs_json)
        if not probs:
            return float("nan")
        s = sorted(probs, reverse=True)
        if len(s) == 1:
            return float(s[0])
        return float(s[0] - s[1])
    except Exception:
        return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds_csv", default="artifacts/preds_eval.csv")
    p.add_argument("--out_csv", default="reports/label_noise_candidates.csv")
    p.add_argument("--top_k", type=int, default=50)
    args = p.parse_args()

    preds_path = Path(args.preds_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(preds_path)

    # add margin
    df["margin_top2"] = df["probs_json"].astype(str).apply(top2_margin)

    # 1) High-confidence wrong predictions
    wrong = df[df["is_correct"] == False].copy()
    wrong = wrong.sort_values(["confidence", "margin_top2"], ascending=[False, False])

    # 2) Ambiguous cases (low margin) - include both correct and wrong, but prioritize wrong
    amb = df.copy()
    amb = amb.sort_values(["margin_top2", "confidence"], ascending=[True, False])

    # Prepare outputs
    top_wrong = wrong.head(args.top_k).assign(reason="high_conf_wrong")
    top_amb = amb.head(args.top_k).assign(reason="low_margin_ambiguous")

    out = pd.concat([top_wrong, top_amb], ignore_index=True)

    cols = [
        "reason",
        "id",
        "source",
        "label_true",
        "pred_label",
        "confidence",
        "margin_top2",
        "text_len_tokens",
        "text",
    ]
    out = out[cols]

    out.to_csv(out_path, index=False)
    print(f"[OK] Wrote candidates -> {out_path}")
    print(f"[INFO] wrong count: {len(wrong)} / total: {len(df)}")


if __name__ == "__main__":
    main()
