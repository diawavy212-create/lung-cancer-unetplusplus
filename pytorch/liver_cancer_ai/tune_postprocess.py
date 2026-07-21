import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from .inference import load_model, preprocess_ct
from .train import ProgressBar
from .tune_threshold import load_mask, metric_row


def remove_small_components(mask, min_area):
    if min_area <= 1:
        return mask

    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    cleaned = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue

        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))

        if len(component) >= min_area:
            ys, xs = zip(*component)
            cleaned[ys, xs] = True

    return cleaned


@torch.no_grad()
def collect_binary_predictions(image_dir, mask_dir, weights_path, threshold, device=None):
    checkpoint = torch.load(weights_path, map_location="cpu")
    image_size = int(checkpoint.get("image_size", 256)) if isinstance(checkpoint, dict) else 256
    model, device = load_model(weights_path=weights_path, device=device)

    image_paths = sorted([path for path in Path(image_dir).iterdir() if path.is_file()])
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    predictions = []
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
        probability = torch.softmax(logits, dim=1)[0, 1].detach().cpu().numpy()
        target = load_mask(mask_path, image_size=image_size).numpy().astype(bool)
        predictions.append(probability >= threshold)
        targets.append(target)
        progress.update(step)

    return predictions, targets, image_size, len(image_paths)


def save_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "threshold",
        "min_area",
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
    areas = [row["min_area"] for row in rows]
    best = max(rows, key=lambda row: row["dice"])

    fig, ax = plt.subplots(figsize=(7, 4), dpi=220)
    ax.plot(areas, [row["dice"] for row in rows], label="Dice", linewidth=2)
    ax.plot(areas, [row["iou"] for row in rows], label="IoU", linewidth=2)
    ax.plot(areas, [row["precision"] for row in rows], label="Precision", linewidth=2)
    ax.plot(areas, [row["recall"] for row in rows], label="Recall", linewidth=2)
    ax.axvline(best["min_area"], color="black", linestyle="--", linewidth=1)
    ax.scatter([best["min_area"]], [best["dice"]], color="black", s=25, zorder=3)
    ax.set_title("Connected Component Post-processing")
    ax.set_xlabel("Minimum Component Area (pixels)")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Tune connected component post-processing on validation predictions.")
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-masks", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--min-areas", nargs="+", type=int, default=[0, 5, 10, 20, 50, 100])
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--plot-output", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    predictions, targets, image_size, sample_count = collect_binary_predictions(
        args.val_images,
        args.val_masks,
        args.weights,
        threshold=args.threshold,
        device=args.device,
    )

    rows = []
    for min_area in args.min_areas:
        cleaned_predictions = []
        for prediction in predictions:
            cleaned_predictions.append(remove_small_components(prediction, min_area).reshape(-1))
        probability = torch.from_numpy(np.concatenate(cleaned_predictions).astype(np.float32))
        target = torch.from_numpy(np.concatenate([target.reshape(-1) for target in targets]).astype(bool))
        row = metric_row(args.threshold, probability, target)
        row["min_area"] = int(min_area)
        rows.append(row)

    best = max(rows, key=lambda row: row["dice"])
    save_csv(rows, args.output_csv)
    save_plot(rows, args.plot_output)

    print(
        f"samples={sample_count} image_size={image_size} threshold={args.threshold:.4f} "
        f"best_min_area={best['min_area']} best_dice={best['dice']:.4f} "
        f"iou={best['iou']:.4f} precision={best['precision']:.4f} recall={best['recall']:.4f} "
        f"specificity={best['specificity']:.4f} accuracy={best['accuracy']:.4f}"
    )
    print(f"output_csv={args.output_csv} plot_output={args.plot_output}")


if __name__ == "__main__":
    main()
