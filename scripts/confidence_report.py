import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_csv", default="artifacts/preds_eval.csv")
    parser.add_argument("--out_dir", default="reports/img")
    args = parser.parse_args()

    preds_path = Path(args.preds_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not preds_path.exists():
        raise FileNotFoundError(f"{preds_path} not found")

    df = pd.read_csv(preds_path)

    df["is_correct"] = df["is_correct"].astype(bool)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    # Histogram
    plt.figure()
    correct = df[df["is_correct"]]["confidence"].dropna()
    wrong = df[~df["is_correct"]]["confidence"].dropna()

    plt.hist(correct, bins=20, alpha=0.7, label="correct")
    plt.hist(wrong, bins=20, alpha=0.7, label="wrong")
    plt.xlabel("confidence")
    plt.ylabel("count")
    plt.title("Confidence distribution")
    plt.legend()

    hist_path = out_dir / "confidence_hist.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Accuracy by bin
    bins = [i / 10 for i in range(11)]
    df["conf_bin"] = pd.cut(df["confidence"], bins=bins, include_lowest=True)

    agg = (
        df.groupby("conf_bin")
        .agg(n=("is_correct", "size"), acc=("is_correct", "mean"))
        .reset_index()
    )

    plt.figure()
    plt.plot(range(len(agg)), agg["acc"], marker="o")
    plt.xticks(range(len(agg)), agg["conf_bin"].astype(str), rotation=45)
    plt.ylim(0, 1)
    plt.title("Accuracy by confidence bin")

    acc_path = out_dir / "accuracy_by_conf_bin.png"
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Markdown report
    report_path = Path("reports/confidence_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    accuracy = df["is_correct"].mean()

    report_path.write_text(
        f"""
# Confidence Report

Accuracy: {accuracy:.4f}

## Histogram
![]({hist_path.as_posix()})

## Accuracy by bin
![]({acc_path.as_posix()})
"""
    )

    print(f"[OK] Report generated → {report_path}")


if __name__ == "__main__":
    main()
