import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import build_segmentation_model


def _load_image_as_array(path):
    suffix = Path(path).suffix.lower()
    if suffix in {".npy", ".npz"}:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            first_key = data.files[0]
            data = data[first_key]
        return np.asarray(data, dtype=np.float32)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for PNG/JPG inference. Install pillow or pass a .npy file.") from exc

    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32)


def preprocess_ct(path, image_size=256):
    image = _load_image_as_array(path)
    image = np.squeeze(image)
    if image.ndim == 3:
        image = image[image.shape[0] // 2]
    if image.ndim != 2:
        raise ValueError("Expected a 2D CT slice or a 3D volume that can be reduced to the middle slice.")

    low, high = np.percentile(image, (1, 99))
    image = np.clip(image, low, high)
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return tensor


def _load_checkpoint_metadata(weights_path, device):
    if not weights_path:
        return None, {}
    checkpoint = torch.load(weights_path, map_location=device)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    state_dict = metadata.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    return state_dict, metadata


def load_model(weights_path=None, device=None, base_channels=32, deep_supervision=False, model_name="eca-dilated-unetpp"):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state_dict, metadata = _load_checkpoint_metadata(weights_path, device)
    base_channels = int(metadata.get("base_channels", base_channels))
    deep_supervision = bool(metadata.get("deep_supervision", deep_supervision))
    model_name = metadata.get("model", metadata.get("model_name", model_name))
    model = build_segmentation_model(
        model_name=model_name,
        in_channels=1,
        num_classes=2,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
    )
    if state_dict is not None:
        clean_state = {key.replace("module.", ""): value for key, value in state_dict.items()}
        model.load_state_dict(clean_state, strict=False)
    model.to(device)
    model.eval()
    return model, device


@torch.no_grad()
def predict_ct(path, weights_path=None, device=None, threshold=0.5, image_size=256, base_channels=32):
    deep_supervision = False
    model_name = "eca-dilated-unetpp"
    if weights_path:
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            image_size = int(checkpoint.get("image_size", image_size))
            base_channels = int(checkpoint.get("base_channels", base_channels))
            deep_supervision = bool(checkpoint.get("deep_supervision", deep_supervision))
            model_name = checkpoint.get("model", checkpoint.get("model_name", "eca-dilated-unetpp"))
    model, device = load_model(
        weights_path=weights_path,
        device=device,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
        model_name=model_name,
    )
    tensor = preprocess_ct(path, image_size=image_size).to(device)

    started_at = time.perf_counter()
    logits = model(tensor)
    if isinstance(logits, (list, tuple)):
        logits = logits[-1]
    probability = torch.softmax(logits, dim=1)[:, 1]
    elapsed = time.perf_counter() - started_at

    mask = (probability >= threshold).float()
    lesion_ratio = float(mask.mean().item())
    confidence = float(probability.max().item())
    diagnosis = "suspicious liver lesion" if lesion_ratio > 0.005 else "no obvious liver lesion"

    return {
        "diagnosis": diagnosis,
        "is_suspicious": lesion_ratio > 0.005,
        "prediction_time": round(elapsed, 4),
        "confidence": round(confidence, 4),
        "lesion_ratio": round(lesion_ratio, 6),
        "accuracy": None,
        "dice": None,
        "image_size": image_size,
    }


def main():
    parser = argparse.ArgumentParser(description="Run liver CT inference with UNet++ + ECA + dilated convolutions.")
    parser.add_argument("image", help="Path to a CT image file. PNG/JPG and .npy are supported by default.")
    parser.add_argument("--weights", default=None, help="Optional trained model checkpoint.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or leave empty for auto.")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    result = predict_ct(args.image, weights_path=args.weights, device=args.device, image_size=args.image_size)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
