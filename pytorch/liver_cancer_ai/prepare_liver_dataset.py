import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_ct_slice(slice_array):
    slice_array = slice_array.astype(np.float32)
    low, high = np.percentile(slice_array, (1, 99))
    slice_array = np.clip(slice_array, low, high)
    slice_array = (slice_array - slice_array.min()) / (slice_array.max() - slice_array.min() + 1e-6)
    return (slice_array * 255).astype(np.uint8)


def save_png(array, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def load_volume(path):
    suffixes = "".join(Path(path).suffixes).lower()
    if suffixes.endswith(".npy"):
        return np.load(path)

    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required for .nii/.nii.gz volumes. Install nibabel first.") from exc

    return nib.load(str(path)).get_fdata()


def convert_case(image_path, label_path, image_out, mask_out, min_tumor_pixels=8):
    image = load_volume(image_path)
    label = load_volume(label_path)
    if image.shape != label.shape:
        raise ValueError(f"Image/label shape mismatch: {image_path.name} {image.shape} vs {label_path.name} {label.shape}")

    saved = 0
    case_name = image_path.name.replace(".nii.gz", "").replace(".nii", "").replace(".npy", "")
    for index in range(image.shape[2]):
        image_slice = image[:, :, index]
        mask_slice = (label[:, :, index] > 1).astype(np.uint8) * 255
        if int((mask_slice > 0).sum()) < min_tumor_pixels:
            continue

        stem = f"{case_name}_slice_{index:03d}.png"
        save_png(normalize_ct_slice(image_slice), image_out / stem)
        save_png(mask_slice, mask_out / stem)
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(description="Convert LiTS/MSD liver CT volumes into 2D tumor PNG pairs.")
    parser.add_argument("--images", required=True, help="Folder containing CT volumes.")
    parser.add_argument("--labels", required=True, help="Folder containing label volumes with matching file names.")
    parser.add_argument("--output", required=True, help="Output dataset folder.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-tumor-pixels", type=int, default=8)
    args = parser.parse_args()

    image_paths = sorted(
        [
            path
            for path in Path(args.images).iterdir()
            if path.is_file()
            and not path.name.startswith("._")
            and ("".join(path.suffixes).lower().endswith((".nii", ".nii.gz", ".npy")))
        ]
    )
    split_index = int(len(image_paths) * (1 - args.val_ratio))
    splits = {
        "train": image_paths[:split_index],
        "val": image_paths[split_index:],
    }

    total_saved = 0
    for split, paths in splits.items():
        for image_path in paths:
            label_path = Path(args.labels) / image_path.name
            if not label_path.exists():
                raise FileNotFoundError(f"Label not found for {image_path.name}")
            saved = convert_case(
                image_path,
                label_path,
                Path(args.output) / f"{split}_images",
                Path(args.output) / f"{split}_masks",
                min_tumor_pixels=args.min_tumor_pixels,
            )
            total_saved += saved
            print(f"{split}: {image_path.name} -> {saved} tumor slices")

    print(f"done: saved {total_saved} paired PNG slices")


if __name__ == "__main__":
    main()
