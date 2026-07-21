import argparse
import csv
import os
from pathlib import Path


def read_best_history(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    best = max(rows, key=lambda row: float(row["val_dice"]))
    return rows, best


def read_best_threshold(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return max(rows, key=lambda row: float(row["dice"]))


def fmt(value):
    if value == "":
        return ""
    return f"{float(value):.4f}"


def save_summary_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Experiment",
        "Epochs",
        "Best Epoch",
        "Threshold",
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "Specificity",
        "Accuracy",
        "Notes",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_plot(rows, path):
    try:
        os.environ.setdefault("WINDIR", r"C:\Windows")
        os.environ.setdefault("MPLCONFIGDIR", str(Path(path).parent / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plot-output. Install matplotlib first.") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        row["Experiment"]
        .replace("UNet++ ECA, ", "")
        .replace(" epochs", "ep")
        .replace(" + threshold tuning", " + threshold")
        for row in rows
    ]
    dice_values = [float(row["Dice"]) for row in rows]
    colors = ["#7aa6c2", "#d5966c", "#b889c7", "#7cbf88", "#5f9ed1", "#d95f59", "#8c8c8c", "#c49a6c"]

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=220)
    bars = ax.bar(range(len(rows)), dice_values, color=colors[: len(rows)])
    ax.set_ylabel("Dice")
    ax.set_title("Validation Dice Comparison")
    ax.set_ylim(0, max(dice_values) * 1.18)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, dice_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def add_paper_comparison_rows(rows, data_dir):
    experiments = [
        ("U-Net", "paper_unet_256_50ep"),
        ("UNet++", "paper_unetpp_256_50ep"),
        ("UNet++ + Strong Augmentation", "paper_unetpp_256_50ep_strong_aug"),
        ("UNet++ 320", "paper_unetpp_320_50ep"),
        ("ECA-UNet++", "paper_eca_unetpp_256_50ep"),
        ("ECA-Dilated-UNet++", "paper_eca_dilated_unetpp_256_50ep"),
    ]
    for name, stem in experiments:
        history_path = data_dir / f"{stem}_history.csv"
        if not history_path.exists():
            continue
        history, best = read_best_history(history_path)
        threshold_rows = []
        for threshold_path in sorted(data_dir.glob(f"{stem}_thresholds*.csv")):
            threshold_rows.append(read_best_threshold(threshold_path))
        best_threshold = max(threshold_rows, key=lambda row: float(row["dice"])) if threshold_rows else None
        rows.append(
            {
                "Experiment": name,
                "Epochs": len(history),
                "Best Epoch": best["epoch"],
                "Threshold": fmt(best_threshold["threshold"]) if best_threshold else "default",
                "Dice": fmt(best_threshold["dice"]) if best_threshold else fmt(best["val_dice"]),
                "IoU": fmt(best_threshold["iou"]) if best_threshold else "",
                "Precision": fmt(best_threshold["precision"]) if best_threshold else "",
                "Recall": fmt(best_threshold["recall"]) if best_threshold else "",
                "Specificity": fmt(best_threshold["specificity"]) if best_threshold else "",
                "Accuracy": fmt(best_threshold["accuracy"]) if best_threshold else fmt(best["val_acc"]),
                "Notes": f"Best val_dice at training log: {fmt(best['val_dice'])}; validation threshold tuning",
            }
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment metrics for paper tables and figures.")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--output-csv", default="../data/paper_metrics_summary.csv")
    parser.add_argument("--plot-output", default="../data/paper_metrics_summary.png")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    experiments = [
        ("UNet++ ECA, 192, 50 epochs", "liver_eca_unetpp_gpu_50ep_history.csv", "Default threshold"),
        (
            "UNet++ ECA, 192, oversampling, 50 epochs",
            "liver_eca_unetpp_gpu_50ep_oversample_history.csv",
            "Default threshold",
        ),
        (
            "UNet++ ECA, 192, Focal Tversky, 50 epochs",
            "liver_eca_unetpp_gpu_50ep_focal_tversky_history.csv",
            "Default threshold",
        ),
        ("UNet++ ECA, 256, 50 epochs", "liver_eca_unetpp_gpu_50ep_256_history.csv", "Default threshold"),
        ("UNet++ ECA, 256, 100 epochs", "liver_eca_unetpp_gpu_100ep_256_history.csv", "Default threshold"),
    ]

    rows = []
    for name, file_name, notes in experiments:
        path = data_dir / file_name
        if not path.exists():
            continue
        history, best = read_best_history(path)
        rows.append(
            {
                "Experiment": name,
                "Epochs": len(history),
                "Best Epoch": best["epoch"],
                "Threshold": "default",
                "Dice": fmt(best["val_dice"]),
                "IoU": "",
                "Precision": "",
                "Recall": "",
                "Specificity": "",
                "Accuracy": fmt(best["val_acc"]),
                "Notes": notes,
            }
        )

    threshold_path = data_dir / "liver_eca_unetpp_gpu_100ep_256_thresholds_fine.csv"
    if threshold_path.exists():
        best_threshold = read_best_threshold(threshold_path)
        rows.append(
            {
                "Experiment": "UNet++ ECA, 256, 100 epochs + threshold tuning",
                "Epochs": 100,
                "Best Epoch": 40,
                "Threshold": fmt(best_threshold["threshold"]),
                "Dice": fmt(best_threshold["dice"]),
                "IoU": fmt(best_threshold["iou"]),
                "Precision": fmt(best_threshold["precision"]),
                "Recall": fmt(best_threshold["recall"]),
                "Specificity": fmt(best_threshold["specificity"]),
                "Accuracy": fmt(best_threshold["accuracy"]),
                "Notes": "Validation threshold tuning",
            }
        )

    add_paper_comparison_rows(rows, data_dir)

    save_summary_csv(rows, args.output_csv)
    save_summary_plot(rows, args.plot_output)

    print(f"saved_csv={args.output_csv}")
    print(f"saved_plot={args.plot_output}")
    for row in rows:
        print(f"{row['Experiment']}: Dice={row['Dice']} Threshold={row['Threshold']}")


if __name__ == "__main__":
    main()
