"""Visualize GPT-2 metrics.csv with train/val/hella_subset logging.
Usage: python plot_training.py --metrics 'log/<run_id>/metrics.csv'
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot GPT-2 training metrics")
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output folder (default: beside metrics.csv)")
    args = parser.parse_args()
    output = args.output or args.metrics.parent
    output.mkdir(parents=True, exist_ok=True)

    metrics = defaultdict(list)
    with args.metrics.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                step = int(row["step"])
                value = float(row["value"])
                metrics[row["metric"].strip()].append((step, value))
            except (ValueError, KeyError, TypeError):
                print(f"Skipped malformed row: {row}")

    for series in metrics.values():
        series.sort(key=lambda item: item[0])

    if not metrics.get("train"):
        raise ValueError("No train metrics found; check file and column names")

    def make_plot(plot_train=True, plot_val=True, filename="loss_curve.png"):
        fig, ax = plt.subplots(figsize=(11, 6))
        for key, label in (("train", "Training loss"), ("val", "Validation loss")):
            if (key == "train" and not plot_train) or (key == "val" and not plot_val):
                continue
            if key in metrics:
                xs, ys = zip(*metrics[key])
                ax.plot(xs, ys, label=label, linewidth=1.3 if key == "train" else 2,
                        alpha=0.75 if key == "train" else 1, marker='o' if key == 'val' else None)
        ax.set(xlabel="Optimizer step", ylabel="Cross-entropy loss", title="GPT-2 124M training")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)
        print(f"Saved {output / filename}")

    make_plot()
    # Readable validation trend separately for quick convergence review.
    if metrics.get("val"):
        make_plot(plot_train=False, filename="validation_curve.png")

    print(f"Training metrics: {len(metrics['train'])} points")
    for key in ("train", "val"):
        if metrics.get(key):
            first = metrics[key][0]
            last = metrics[key][-1]
            print(f"{key}: step {first[0]} loss {first[1]:.4f} -> step {last[0]} loss {last[1]:.4f}")

    if metrics.get("hella_subset"):
        print("Note: In the original overnight run, in-script HellaSwag was affected by a string/int label bug.")
        print("Use the independently evaluated 37/128 (28.91%) result instead of the zero in metrics.csv.")

if __name__ == "__main__":
    main()
