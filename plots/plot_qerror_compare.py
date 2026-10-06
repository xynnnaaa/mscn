import struct
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

# ======== Configuration ========
QUERY_CSV = "/data2/xuyining/learnedcardinalities/data/imdb/test.csv"
CARDINALITY_CSV = "/data2/xuyining/learnedcardinalities/data/imdb/ori/results/bitmap-2.csv"
BITMAP_FILE = "/data2/xuyining/learnedcardinalities/data/imdb/test.bitmaps"
SAMPLE_SIZE = 1000

PLOT_FILENAME = "/data2/xuyining/learnedcardinalities/plots/imdb_qerror_bitmap_comparison.png"
# ===============================


def calculate_qerror(pred, true):
    """Compute standard Q-Error."""
    pred = max(float(pred), 1.0)
    true = max(float(true), 1.0)
    return max(pred / true, true / pred)


def main():
    bytes_per_table = int(np.ceil(SAMPLE_SIZE / 8))
    data_records = []

    with open(QUERY_CSV, "r") as f_csv, \
         open(BITMAP_FILE, "rb") as f_bit, \
         open(CARDINALITY_CSV, "r") as f_card:

        csv_reader = csv.reader(f_csv, delimiter="#")
        card_reader = csv.reader(f_card)

        for row_csv in csv_reader:
            if not row_csv:
                continue

            # Read bitmap header
            num_tables_bytes = f_bit.read(4)
            if not num_tables_bytes:
                break

            num_tables = struct.unpack("<I", num_tables_bytes)[0]

            # Check whether any table bitmap is empty
            has_empty_bitmap = False
            for _ in range(num_tables):
                bitmap_bytes = f_bit.read(bytes_per_table)
                if all(b == 0 for b in bitmap_bytes):
                    has_empty_bitmap = True

            # Read prediction and ground truth
            try:
                row_card = next(card_reader)
                pred_val = float(row_card[0])
                true_val = float(row_card[1])
            except (StopIteration, ValueError, IndexError):
                continue

            q_error = calculate_qerror(pred_val, true_val)

            data_records.append({
                "group": "All-Zero" if has_empty_bitmap else "Non-Zero",
                "error": q_error
            })

    df = pd.DataFrame(data_records)

    if df.empty:
        print("No valid data found.")
        return

    # Print summary statistics
    pd.set_option("display.float_format", lambda x: "%.2f" % x)
    stats = df.groupby("group")["error"].describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]
    )
    print(stats[["count", "mean", "50%", "90%", "95%", "99%", "max"]])

    counts = df["group"].value_counts()
    nz_count = counts.get("Non-Zero", 0)
    az_count = counts.get("All-Zero", 0)

    sns.set_theme(style="ticks")
    plt.figure(figsize=(7, 6))

    ax = sns.boxplot(
        x="group",
        y="error",
        hue="group",
        data=df,
        order=["Non-Zero", "All-Zero"],
        palette={
            "Non-Zero": "#4C72B0",
            "All-Zero": "#DD8452"
        },
        legend=False,
        width=0.4,
        linewidth=1.8,
        fliersize=3,
        flierprops={"alpha": 0.25},
    )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))

    plt.title(
        "Impact of Missing Bitmaps on Estimation Error",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel(
        "Bitmap Status in Query",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )
    plt.ylabel(
        "Q-Error (Log Scale)",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )

    plt.xticks(
        [0, 1],
        [
            f"Non-Zero\n(N={nz_count})",
            f"All-Zero\n(N={az_count})",
        ],
        fontsize=11,
    )
    plt.yticks(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    sns.despine()

    plt.savefig(PLOT_FILENAME, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {PLOT_FILENAME}")


if __name__ == "__main__":
    main()