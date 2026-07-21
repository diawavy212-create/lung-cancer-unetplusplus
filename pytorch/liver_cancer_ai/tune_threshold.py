import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .inference import _load_image_as_array, load_model, preprocess_ct
from .train import ProgressBar


def load_mask(path, image_size):
    mask = np.squeeze(_load_image_as_array(path)).astype(np.float32)
    mask = (mask > 0).astype(np.float32)
    tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return tensor.squeeze(0).squeeze(0).bool()


@torch.no_grad()
def collect_predictions(image_dir, mask_dir, weights_path, device=None):
    checkpoint = torch.load(weights_path, map_location="cpu")
    image_size = int(checkpoint.get("image_size", 256)) if isinstance(checkpoint, dict) else 256
    model, device = load_model(weights_path=weights_path, device=device)

    image_paths = sorted([path for path in Path(image_dir).iterdir() if path.is_file()])
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    probabilities = []
    targets = []
    progress = ProgressBar(len(image_paths), "predict val")
    for step, image_path in enumerate(image_paths, start=1):
        mask_path = Path(mask_dir) / image_path.name
        if not mask_path.exists() and image_path.suffix.lower() != ".png":
            mask_path = Path(mask_dir) / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_path.name}")

        image = preprocess_ct(image_path, image_size=image_size).to(device)
        logits = model(image)
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]
        prob = torch.softmax(logits, dim=1)[0, 1].detach().cpu()
        target = load_mask(mask_path, image_size=image_size)
        probabilities.append(prob.flatten())
        targets.append(target.flatten())
        progress.update(step)

    return torch.cat(probabilities), torch.cat(targets), image_size, len(image_paths)


def metric_row(threshold, probability, target, eps=1e-6):
    pred = probability >= threshold
    target = target.bool()

    tp = torch.logical_and(pred, target).sum().item()
    fp = torch.logical_and(pred, ~target).sum().item()
    fn = torch.logical_and(~pred, target).sum().item()
    tn = torch.logical_and(~pred, ~target).sum().item()

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn + eps) / (tp + fp + fn + tn + eps)
    pred_ratio = (tp + fp) / max(tp + fp + fn + tn, 1)

    return {
        "threshold": threshold,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "pred_ratio": pred_ratio,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def save_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "threshold",
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "accuracy",
        "pred_ratio",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows, path):
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
    thresholds = [row["threshold"] for row in rows]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=220)
    ax.plot(thresholds, [row["dice"] for row in rows], label="Dice", linewidth=2)
    ax.plot(thresholds, [row["iou"] for row in rows], label="IoU", linewidth=2)
    ax.plot(thresholds, [row["precision"] for row in rows], label="Precision", linewidth=2)
    ax.plot(thresholds, [row["recall"] for row in rows], label="Recall", linewidth=2)
    best = max(rows, key=lambda row: row["dice"])
    ax.axvline(best["threshold"], color="black", linestyle="--", linewidth=1)
    ax.scatter([best["threshold"]], [best["dice"]], color="black", s=25, zorder=3)
    ax.set_title("Threshold Tuning on Validation Set")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Tune segmentation threshold on the validation set.")
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-masks", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--min-threshold", type=float, default=0.1)
    parser.add_argument("--max-threshold", type=float, default=0.9)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--plot-output", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    probability, target, image_size, sample_count = collect_predictions(
        args.val_images,
        args.val_masks,
        args.weights,
        device=args.device,
    )

    thresholds = np.arange(args.min_threshold, args.max_threshold + args.step * 0.5, args.step)
    rows = [metric_row(float(round(threshold, 4)), probability, target) for threshold in thresholds]
    best = max(rows, key=lambda row: row["dice"])

    save_csv(rows, args.output_csv)
    save_plot(rows, args.plot_output)

    print(
        f"samples={sample_count} image_size={image_size} "
        f"best_threshold={best['threshold']:.4f} best_dice={best['dice']:.4f} "
        f"iou={best['iou']:.4f} precision={best['precision']:.4f} recall={best['recall']:.4f} "
        f"specificity={best['specificity']:.4f} accuracy={best['accuracy']:.4f}"
    )
    print(f"output_csv={args.output_csv} plot_output={args.plot_output}")


if __name__ == "__main__":
    main()
