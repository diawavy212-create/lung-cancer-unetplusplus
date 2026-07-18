import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .inference import _load_image_as_array, load_model, preprocess_ct


def normalize_image(image):
    image = np.squeeze(image).astype(np.float32)
    low, high = np.percentile(image, (1, 99))
    image = np.clip(image, low, high)
    return (image - image.min()) / (image.max() - image.min() + 1e-6)


def prepare_mask(mask_path, image_size):
    mask = np.squeeze(_load_image_as_array(mask_path)).astype(np.float32)
    mask = (mask > 0).astype(np.float32)
    tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return tensor.squeeze().cpu().numpy()


def overlay_mask(image, mask, color):
    rgb = np.stack([image, image, image], axis=-1)
    color = np.asarray(color, dtype=np.float32)
    alpha = 0.45
    rgb[mask > 0] = (1 - alpha) * rgb[mask > 0] + alpha * color
    return np.clip(rgb, 0, 1)


@torch.no_grad()
def predict_mask(image_path, weights_path, device=None, threshold=0.5):
    checkpoint = torch.load(weights_path, map_location="cpu")
    image_size = int(checkpoint.get("image_size", 256)) if isinstance(checkpoint, dict) else 256
    model, device = load_model(weights_path=weights_path, device=device)
    tensor = preprocess_ct(image_path, image_size=image_size).to(device)
    logits = model(tensor)
    if isinstance(logits, (list, tuple)):
        logits = logits[-1]
    prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
    return prob, (prob >= threshold).astype(np.float32), image_size


def main():
    parser = argparse.ArgumentParser(description="Create a paper-ready CT segmentation visualization.")
    parser.add_argument("--image", required=True, help="Input CT slice PNG.")
    parser.add_argument("--mask", default=None, help="Optional ground-truth mask PNG.")
    parser.add_argument("--weights", required=True, help="Trained model checkpoint.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    try:
        os.environ.setdefault("WINDIR", r"C:\Windows")
        os.environ.setdefault("MPLCONFIGDIR", str(Path(args.output).parent / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required. Install matplotlib first.") from exc

    prob, pred_mask, image_size = predict_mask(args.image, args.weights, device=args.device, threshold=args.threshold)
    image = normalize_image(_load_image_as_array(args.image))
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    image = F.interpolate(image_tensor, size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze().numpy()

    gt_mask = prepare_mask(args.mask, image_size) if args.mask else None

    panels = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, panels, figsize=(4 * panels, 4), dpi=220)
    axes = np.atleast_1d(axes)

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("CT slice")
    axes[0].axis("off")

    panel_index = 1
    if gt_mask is not None:
        axes[panel_index].imshow(gt_mask, cmap="gray")
        axes[panel_index].set_title("Ground truth")
        axes[panel_index].axis("off")
        panel_index += 1

    axes[panel_index].imshow(pred_mask, cmap="gray")
    axes[panel_index].set_title("Prediction")
    axes[panel_index].axis("off")
    panel_index += 1

    overlay = overlay_mask(image, pred_mask, color=[1.0, 0.1, 0.1])
    if gt_mask is not None:
        overlay = overlay_mask(overlay.mean(axis=-1), gt_mask, color=[0.1, 0.8, 0.2])
    axes[panel_index].imshow(overlay)
    axes[panel_index].set_title("Overlay")
    axes[panel_index].axis("off")

    fig.suptitle(f"threshold={args.threshold} mean_prob={prob.mean():.4f}", fontsize=11)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
