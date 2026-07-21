import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.nn.functional as F

from .inference import _load_image_as_array
from .model import build_segmentation_model


class ProgressBar:
    def __init__(self, total, prefix):
        self.total = max(int(total), 1)
        self.prefix = prefix

    def update(self, step, **metrics):
        width = 28
        filled = int(width * step / self.total)
        bar = "#" * filled + "." * (width - filled)
        metric_text = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"\r{self.prefix} [{bar}] {step}/{self.total} {metric_text}", end="", flush=True)
        if step >= self.total:
            print()


class LiverSliceDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256, augment=False, strong_augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.augment = augment
        self.strong_augment = strong_augment
        self.images = sorted([path for path in self.image_dir.iterdir() if path.is_file()])
        if not self.images:
            raise ValueError(f"No training images found in {self.image_dir}")
        self._positive_cache = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.mask_dir / image_path.name
        if not mask_path.exists() and image_path.suffix.lower() != ".png":
            mask_path = self.mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_path.name}")

        image = self._prepare_image(_load_image_as_array(image_path))
        mask = self._prepare_mask(_load_image_as_array(mask_path))
        if self.augment:
            image, mask = self._augment_pair(image, mask)
        return image, mask.long()

    def positive_flags(self):
        if self._positive_cache is not None:
            return self._positive_cache
        flags = []
        for image_path in self.images:
            mask_path = self.mask_dir / image_path.name
            if not mask_path.exists() and image_path.suffix.lower() != ".png":
                mask_path = self.mask_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask not found for {image_path.name}")
            mask = np.squeeze(_load_image_as_array(mask_path)).astype(np.float32)
            flags.append(bool((mask > 0).any()))
        self._positive_cache = flags
        return flags

    def _prepare_image(self, image):
        image = np.squeeze(image).astype(np.float32)
        if image.ndim == 3:
            image = image[image.shape[0] // 2]
        low, high = np.percentile(image, (1, 99))
        image = np.clip(image, low, high)
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return tensor.squeeze(0)

    def _prepare_mask(self, mask):
        mask = np.squeeze(mask).astype(np.float32)
        if mask.ndim == 3:
            mask = mask[mask.shape[0] // 2]
        mask = (mask > 0).astype(np.float32)
        tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode="nearest")
        return tensor.squeeze(0).squeeze(0)

    def _augment_pair(self, image, mask):
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[0])
        rotations = int(torch.randint(0, 4, (1,)).item())
        if rotations:
            image = torch.rot90(image, rotations, dims=[1, 2])
            mask = torch.rot90(mask, rotations, dims=[0, 1])
        if self.strong_augment:
            image, mask = self._random_affine_pair(image, mask)
            image = self._augment_intensity(image)
        return image.contiguous(), mask.contiguous()

    def _random_affine_pair(self, image, mask):
        if torch.rand(()) >= 0.6:
            return image, mask

        scale = float(torch.empty(1).uniform_(0.90, 1.10).item())
        translate_x = float(torch.empty(1).uniform_(-0.08, 0.08).item())
        translate_y = float(torch.empty(1).uniform_(-0.08, 0.08).item())
        theta = torch.tensor(
            [[[scale, 0.0, translate_x], [0.0, scale, translate_y]]],
            dtype=image.dtype,
            device=image.device,
        )
        image_batch = image.unsqueeze(0)
        mask_batch = mask.unsqueeze(0).unsqueeze(0).float()
        grid = F.affine_grid(theta, image_batch.size(), align_corners=False)
        image = F.grid_sample(image_batch, grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(0)
        mask = F.grid_sample(mask_batch, grid, mode="nearest", padding_mode="zeros", align_corners=False).squeeze(0).squeeze(0)
        return image, mask

    def _augment_intensity(self, image):
        if torch.rand(()) < 0.7:
            contrast = torch.empty(1, dtype=image.dtype, device=image.device).uniform_(0.80, 1.20)
            brightness = torch.empty(1, dtype=image.dtype, device=image.device).uniform_(-0.10, 0.10)
            image = image * contrast + brightness
        if torch.rand(()) < 0.5:
            gamma = torch.empty(1, dtype=image.dtype, device=image.device).uniform_(0.80, 1.30)
            image = torch.clamp(image, 0.0, 1.0).pow(gamma)
        if torch.rand(()) < 0.4:
            noise_std = torch.empty(1, dtype=image.dtype, device=image.device).uniform_(0.00, 0.03)
            image = image + torch.randn_like(image) * noise_std
        return torch.clamp(image, 0.0, 1.0)


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, tumor_weight=5.0, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.register_buffer("class_weights", torch.tensor([1.0, float(tumor_weight)], dtype=torch.float32))
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, target):
        if isinstance(logits, (list, tuple)):
            weights = [0.1, 0.2, 0.3, 0.4][-len(logits):]
            weights = [weight / sum(weights) for weight in weights]
            return sum(weight * self._single(output, target) for weight, output in zip(weights, logits))
        return self._single(logits, target)

    def _single(self, logits, target, eps=1e-6):
        class_weights = self.class_weights.to(logits.device)
        ce = F.cross_entropy(logits, target, weight=class_weights)
        prob = torch.softmax(logits, dim=1)[:, 1]
        target_float = (target == 1).float()
        intersection = (prob * target_float).sum(dim=(1, 2))
        union = prob.sum(dim=(1, 2)) + target_float.sum(dim=(1, 2))
        dice_loss = 1.0 - ((2.0 * intersection + eps) / (union + eps)).mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


class FocalTverskyCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        tumor_weight=5.0,
        alpha=0.3,
        beta=0.7,
        gamma=0.75,
        tversky_weight=1.0,
        ce_weight=0.5,
    ):
        super().__init__()
        self.register_buffer("class_weights", torch.tensor([1.0, float(tumor_weight)], dtype=torch.float32))
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tversky_weight = tversky_weight
        self.ce_weight = ce_weight

    def forward(self, logits, target):
        if isinstance(logits, (list, tuple)):
            weights = [0.1, 0.2, 0.3, 0.4][-len(logits):]
            weights = [weight / sum(weights) for weight in weights]
            return sum(weight * self._single(output, target) for weight, output in zip(weights, logits))
        return self._single(logits, target)

    def _single(self, logits, target, eps=1e-6):
        class_weights = self.class_weights.to(logits.device)
        ce = F.cross_entropy(logits, target, weight=class_weights)
        prob = torch.softmax(logits, dim=1)[:, 1]
        target_float = (target == 1).float()

        true_pos = (prob * target_float).sum(dim=(1, 2))
        false_pos = (prob * (1.0 - target_float)).sum(dim=(1, 2))
        false_neg = ((1.0 - prob) * target_float).sum(dim=(1, 2))
        tversky = (true_pos + eps) / (true_pos + self.alpha * false_pos + self.beta * false_neg + eps)
        focal_tversky = torch.pow(1.0 - tversky, self.gamma).mean()
        return self.ce_weight * ce + self.tversky_weight * focal_tversky


def dice_score(logits, target, eps=1e-6):
    if isinstance(logits, (list, tuple)):
        logits = logits[-1]
    pred = torch.argmax(logits, dim=1)
    pred = (pred == 1).float()
    target = (target == 1).float()
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return ((2 * intersection + eps) / (union + eps)).mean()


def pixel_accuracy(logits, target):
    if isinstance(logits, (list, tuple)):
        logits = logits[-1]
    pred = torch.argmax(logits, dim=1)
    return (pred == target).float().mean()


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, epoch=None, total_epochs=None):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_acc = 0.0
    progress = ProgressBar(len(loader), f"epoch {epoch}/{total_epochs} train") if epoch and total_epochs else None

    for step, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        metric_logits = [output.detach() for output in logits] if isinstance(logits, (list, tuple)) else logits.detach()
        total_dice += dice_score(metric_logits, masks).item()
        total_acc += pixel_accuracy(metric_logits, masks).item()
        if progress:
            progress.update(
                step,
                loss=total_loss / step,
                dice=total_dice / step,
                acc=total_acc / step,
            )

    batches = max(len(loader), 1)
    return total_loss / batches, total_dice / batches, total_acc / batches


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch=None, total_epochs=None):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_acc = 0.0
    progress = ProgressBar(len(loader), f"epoch {epoch}/{total_epochs} val  ") if epoch and total_epochs else None

    for step, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += loss.item()
        total_dice += dice_score(logits, masks).item()
        total_acc += pixel_accuracy(logits, masks).item()
        if progress:
            progress.update(
                step,
                loss=total_loss / step,
                dice=total_dice / step,
                acc=total_acc / step,
            )

    batches = max(len(loader), 1)
    return total_loss / batches, total_dice / batches, total_acc / batches


def default_metric_path(output_path, suffix):
    output_path = Path(output_path)
    if suffix.startswith("."):
        return output_path.with_suffix(suffix)
    return output_path.with_name(f"{output_path.stem}{suffix}")


def save_history_csv(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "train_dice", "train_acc", "val_loss", "val_dice", "val_acc", "lr"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_history_plot(history, path):
    if not history:
        return
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
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=200)
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train loss", linewidth=2)
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Val loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [row["train_dice"] for row in history], label="Train Dice", linewidth=2)
    axes[1].plot(epochs, [row["val_dice"] for row in history], label="Val Dice", linewidth=2)
    axes[1].set_title("Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(epochs, [row["train_acc"] for row in history], label="Train accuracy", linewidth=2)
    axes[2].plot(epochs, [row["val_acc"] for row in history], label="Val accuracy", linewidth=2)
    axes[2].set_title("Pixel Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train UNet++ + ECA + dilated convolution for liver CT diagnosis.")
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-masks", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-masks", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--model",
        choices=["unet", "unetpp", "eca-unetpp", "eca-dilated-unetpp"],
        default="eca-dilated-unetpp",
        help="Model variant for comparison and ablation experiments.",
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tumor-weight", type=float, default=5.0)
    parser.add_argument(
        "--loss",
        choices=["dice-ce", "focal-tversky-ce"],
        default="dice-ce",
        help="Training loss. focal-tversky-ce is usually better for small lesion segmentation.",
    )
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--tversky-gamma", type=float, default=0.75)
    parser.add_argument(
        "--positive-sample-weight",
        type=float,
        default=1.0,
        help="Oversample slices containing tumor pixels. Use 1.0 to disable, try 4.0 for imbalanced data.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deep-supervision", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--strong-augment",
        action="store_true",
        help="Use stronger CT-safe augmentation: mild affine transform, intensity jitter, gamma, and noise.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="liver_eca_unetpp_best.pth")
    parser.add_argument("--history-csv", default=None)
    parser.add_argument("--plot-output", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this Python environment has a CPU-only PyTorch build.")
    train_set = LiverSliceDataset(
        args.train_images,
        args.train_masks,
        image_size=args.image_size,
        augment=not args.no_augment,
        strong_augment=args.strong_augment,
    )
    val_set = LiverSliceDataset(args.val_images, args.val_masks, image_size=args.image_size, augment=False)
    pin_memory = device.type == "cuda"
    sampler = None
    shuffle = True
    positive_count = 0
    if args.positive_sample_weight > 1.0:
        positive_flags = train_set.positive_flags()
        positive_count = sum(positive_flags)
        if positive_count == 0:
            raise ValueError("positive_sample_weight was enabled, but no positive masks were found in the training set.")
        sample_weights = [args.positive_sample_weight if is_positive else 1.0 for is_positive in positive_flags]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_segmentation_model(
        model_name=args.model,
        base_channels=args.base_channels,
        deep_supervision=args.deep_supervision,
    ).to(device)
    if args.loss == "focal-tversky-ce":
        criterion = FocalTverskyCrossEntropyLoss(
            tumor_weight=args.tumor_weight,
            alpha=args.tversky_alpha,
            beta=args.tversky_beta,
            gamma=args.tversky_gamma,
        )
    else:
        criterion = DiceCrossEntropyLoss(tumor_weight=args.tumor_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    print(
        f"device={device} train_samples={len(train_set)} val_samples={len(val_set)} "
        f"epochs={args.epochs} batch_size={args.batch_size} image_size={args.image_size} "
        f"model={args.model} base_channels={args.base_channels} lr={args.lr} tumor_weight={args.tumor_weight} "
        f"loss={args.loss} "
        f"tversky_alpha={args.tversky_alpha} tversky_beta={args.tversky_beta} tversky_gamma={args.tversky_gamma} "
        f"positive_sample_weight={args.positive_sample_weight} positive_slices={positive_count or 'not_counted'} "
        f"deep_supervision={args.deep_supervision} amp={scaler.is_enabled()} augment={not args.no_augment} "
        f"strong_augment={args.strong_augment}",
        flush=True,
    )

    best_dice = -1.0
    history = []
    history_csv = args.history_csv or str(default_metric_path(args.output, ".csv"))
    plot_output = args.plot_output or str(default_metric_path(args.output, "_curves.png"))
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler, epoch=epoch, total_epochs=args.epochs
        )
        val_loss, val_dice, val_acc = evaluate(
            model, val_loader, criterion, device, epoch=epoch, total_epochs=args.epochs
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_acc": val_acc,
                "lr": current_lr,
            }
        )
        save_history_csv(history, history_csv)
        save_history_plot(history, plot_output)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_acc={val_acc:.4f} "
            f"lr={current_lr:.6f}"
        )
        print(f"history_csv={history_csv} plot_output={plot_output}", flush=True)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_dice": best_dice,
                    "image_size": args.image_size,
                    "model": args.model,
                    "base_channels": args.base_channels,
                    "deep_supervision": args.deep_supervision,
                    "loss": args.loss,
                    "strong_augment": args.strong_augment,
                },
                args.output,
            )
            print(f"saved_best={args.output} best_dice={best_dice:.4f}", flush=True)


if __name__ == "__main__":
    main()
