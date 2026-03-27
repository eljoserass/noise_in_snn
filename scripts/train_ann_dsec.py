"""
Train VGG11-SSD ANN on DSEC RGB frames.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dsec_dataset import DSECSSD_ANN, parse_class_ids
from src.models.ann import VGG11_SSD_ANN
from src.utils import SSDLoss, generate_anchors_for_model, match_anchors_to_targets


def parse_list(value: str) -> list[str] | None:
    if not value.strip():
        return None
    out = [x.strip() for x in value.split(",") if x.strip()]
    return out or None


def _load_split_list(split_config: Path, split_name: str) -> list[str]:
    if not split_config.exists():
        return []
    lines = split_config.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    in_section = False
    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" "):
            in_section = stripped == f"{split_name}:"
            continue
        if in_section and stripped.startswith("- "):
            out.append(stripped[2:].strip())
    return out


def resolve_val_split(args) -> None:
    val_dir = Path(args.dsec_root) / args.val_split
    if args.val_split != "val" or val_dir.exists():
        return
    print(f"[train_ann_dsec] val directory not found: {val_dir}")
    args.val_split = args.train_split
    if args.val_sequences.strip():
        print(
            "[train_ann_dsec] using provided --val-sequences with --val-split=train "
            "(DSEC val is typically a subsequence list of train)."
        )
        return
    seqs = _load_split_list(Path(args.split_config), "val")
    if seqs:
        args.val_sequences = ",".join(seqs)
        print(
            f"[train_ann_dsec] loaded {len(seqs)} val sequences from {args.split_config} "
            "and switched --val-split to train"
        )
    else:
        print(
            "[train_ann_dsec] split config missing/empty; "
            "falling back to full train split for validation."
        )


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def estimate_class_weights_from_dataset(
    dataset: DSECSSD_ANN,
    num_classes: int,
    bg_weight: float = 0.25,
    power: float = 0.5,
    min_weight: float = 0.25,
    max_weight: float = 8.0,
) -> torch.Tensor:
    """
    Estimate class weights from frame-aligned labels without loading images.
    Class index 0 is background.
    """
    counts = np.zeros((num_classes,), dtype=np.float64)
    for seq in dataset.sequences:
        for start, end in seq.frame_track_ranges:
            if end <= start:
                continue
            cls_ids = seq.tracks["class_id"][start:end].astype(np.int64)
            for cid in cls_ids:
                label = dataset.class_id_to_label.get(int(cid), 0)
                counts[label] += 1.0

    weights = np.ones((num_classes,), dtype=np.float32)
    weights[0] = float(bg_weight)
    pos = counts[1:]
    nz = pos[pos > 0]
    if nz.size == 0:
        return torch.from_numpy(weights)

    ref = float(np.median(nz))
    for cls in range(1, num_classes):
        c = max(float(counts[cls]), 1.0)
        w = (ref / c) ** float(power)
        w = max(float(min_weight), min(float(max_weight), float(w)))
        weights[cls] = float(w)

    return torch.from_numpy(weights)


def train_one_epoch(
    model: VGG11_SSD_ANN,
    dataloader: DataLoader,
    criterion: SSDLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    anchors: torch.Tensor,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for images, targets in pbar:
        images = images.to(device)
        cls_preds, loc_preds = model(images)

        batch_cls_targets = []
        batch_loc_targets = []
        for i in range(images.size(0)):
            gt_boxes = targets[i]["boxes"].to(device)
            gt_labels = targets[i]["labels"].to(device)
            cls_target, loc_target = match_anchors_to_targets(
                anchors, gt_boxes, gt_labels, iou_threshold=0.5
            )
            batch_cls_targets.append(cls_target)
            batch_loc_targets.append(loc_target)

        cls_targets = torch.stack(batch_cls_targets, dim=0)
        loc_targets = torch.stack(batch_loc_targets, dim=0)
        loss, (cls_loss, loc_loss) = criterion(cls_preds, loc_preds, cls_targets, loc_targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_loc_loss += loc_loss.item()
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "cls": f"{cls_loss.item():.4f}",
                "loc": f"{loc_loss.item():.4f}",
            }
        )

    denom = max(1, len(dataloader))
    return total_loss / denom, total_cls_loss / denom, total_loc_loss / denom


@torch.no_grad()
def validate(
    model: VGG11_SSD_ANN,
    dataloader: DataLoader,
    criterion: SSDLoss,
    device: torch.device,
    anchors: torch.Tensor,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0

    for images, targets in tqdm(dataloader, desc="Validation"):
        images = images.to(device)
        cls_preds, loc_preds = model(images)

        batch_cls_targets = []
        batch_loc_targets = []
        for i in range(images.size(0)):
            gt_boxes = targets[i]["boxes"].to(device)
            gt_labels = targets[i]["labels"].to(device)
            cls_target, loc_target = match_anchors_to_targets(
                anchors, gt_boxes, gt_labels, iou_threshold=0.5
            )
            batch_cls_targets.append(cls_target)
            batch_loc_targets.append(loc_target)

        cls_targets = torch.stack(batch_cls_targets, dim=0)
        loc_targets = torch.stack(batch_loc_targets, dim=0)
        loss, (cls_loss, loc_loss) = criterion(cls_preds, loc_preds, cls_targets, loc_targets)

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_loc_loss += loc_loss.item()

    denom = max(1, len(dataloader))
    return total_loss / denom, total_cls_loss / denom, total_loc_loss / denom


def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG11 SSD ANN on DSEC RGB")

    # Data
    parser.add_argument("--dsec-root", type=str, default="data/dsec", help="DSEC root containing train/val/test")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument(
        "--split-config",
        type=str,
        default="config/dsec_train_val_test_split.yaml",
        help="Optional split config used to resolve val sequences when val dir is absent.",
    )
    parser.add_argument("--train-sequences", type=str, default="", help="Optional comma-separated sequence names")
    parser.add_argument("--val-sequences", type=str, default="", help="Optional comma-separated sequence names")
    parser.add_argument("--image-relpath", type=str, default="images/left/distorted")
    parser.add_argument("--timestamps-relpath", type=str, default="images/timestamps.txt")
    parser.add_argument("--tracks-relpath", type=str, default="object_detections/left/tracks.npy")
    parser.add_argument("--class-ids", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--time-mode", type=str, default="nearest", choices=["nearest", "window"])
    parser.add_argument("--window-us", type=int, default=25_000)
    parser.add_argument(
        "--max-time-delta-us",
        type=int,
        default=50_000,
        help="Used only in nearest mode. Set <=0 to disable threshold.",
    )
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument(
        "--rgb-color",
        action="store_true",
        help="Use color RGB as-is. Default behavior converts input to grayscale then repeats to 3 channels.",
    )
    parser.add_argument("--crop-top-px", type=int, default=0)
    parser.add_argument("--crop-bottom-px", type=int, default=0)
    parser.add_argument("--crop-left-px", type=int, default=0)
    parser.add_argument("--crop-right-px", type=int, default=0)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--input-height", type=int, default=480)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--imagenet-norm", action="store_true", help="Apply ImageNet normalization.")
    parser.add_argument("--class-balance", action="store_true", help="Enable class-weighted classification loss.")
    parser.add_argument("--class-balance-bg-weight", type=float, default=0.25)
    parser.add_argument("--class-balance-power", type=float, default=0.5)
    parser.add_argument("--class-balance-min", type=float, default=0.25)
    parser.add_argument("--class-balance-max", type=float, default=8.0)

    # Runtime
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=15)

    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="neuromorph-vs-noise")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    resolve_val_split(args)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    class_ids = parse_class_ids(args.class_ids)
    num_classes = len(class_ids)
    max_delta = None if args.max_time_delta_us <= 0 else args.max_time_delta_us

    print(f"Using device: {device}")
    print(f"DSEC classes: {class_ids} (num_classes={num_classes})")

    wandb_run = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                tags=["ann", "dsec", "rgb"],
            )
            print(f"wandb: {wandb_run.url}")
        except ImportError:
            print("wandb not installed; continuing without wandb")
            args.wandb = False

    tfms = [
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((args.input_height, args.input_width), antialias=True),
    ]
    if args.imagenet_norm:
        tfms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(tfms)

    train_dataset = DSECSSD_ANN(
        dsec_root=Path(args.dsec_root),
        split=args.train_split,
        image_relpath=args.image_relpath,
        timestamps_relpath=args.timestamps_relpath,
        tracks_relpath=args.tracks_relpath,
        class_ids=class_ids,
        sequences=parse_list(args.train_sequences),
        time_mode=args.time_mode,
        window_us=args.window_us,
        max_time_delta_us=max_delta,
        max_frames_per_sequence=args.max_frames_per_sequence,
        force_grayscale=not args.rgb_color,
        crop_top_px=args.crop_top_px,
        crop_bottom_px=args.crop_bottom_px,
        crop_left_px=args.crop_left_px,
        crop_right_px=args.crop_right_px,
        transform=transform,
    )
    val_dataset = DSECSSD_ANN(
        dsec_root=Path(args.dsec_root),
        split=args.val_split,
        image_relpath=args.image_relpath,
        timestamps_relpath=args.timestamps_relpath,
        tracks_relpath=args.tracks_relpath,
        class_ids=class_ids,
        sequences=parse_list(args.val_sequences),
        time_mode=args.time_mode,
        window_us=args.window_us,
        max_time_delta_us=max_delta,
        max_frames_per_sequence=args.max_frames_per_sequence,
        force_grayscale=not args.rgb_color,
        crop_top_px=args.crop_top_px,
        crop_bottom_px=args.crop_bottom_px,
        crop_left_px=args.crop_left_px,
        crop_right_px=args.crop_right_px,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    class_weights = None
    if args.class_balance:
        class_weights = estimate_class_weights_from_dataset(
            train_dataset,
            num_classes=num_classes + 1,
            bg_weight=args.class_balance_bg_weight,
            power=args.class_balance_power,
            min_weight=args.class_balance_min,
            max_weight=args.class_balance_max,
        )
        class_weight_msg = ", ".join(
            f"{idx}:{class_weights[idx]:.3f}" for idx in range(class_weights.numel())
        )
        print(f"[train_ann_dsec] class weights (incl bg=0): {class_weight_msg}")
        if args.wandb and wandb_run:
            wandb.log({f"class_weight/{idx}": float(w) for idx, w in enumerate(class_weights.tolist())})

    model = VGG11_SSD_ANN(num_classes=num_classes + 1).to(device)
    anchors = generate_anchors_for_model(model, (3, args.input_height, args.input_width), device)
    criterion = SSDLoss(
        num_classes=num_classes + 1,
        class_weights=class_weights.to(device) if class_weights is not None else None,
    )
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.6), int(args.epochs * 0.8)],
        gamma=0.1,
    )

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        patience_counter = checkpoint.get("patience_counter", 0)
        print(f"Resumed from {args.resume} @ epoch {start_epoch}")

    print(f"Training ANN DSEC: train={len(train_dataset)} val={len(val_dataset)}")
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_cls_loss, train_loc_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, anchors
        )
        val_loss, val_cls_loss, val_loc_loss = validate(model, val_loader, criterion, device, anchors)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}: train={train_loss:.4f} (cls={train_cls_loss:.4f}, loc={train_loc_loss:.4f}) "
            f"val={val_loss:.4f} (cls={val_cls_loss:.4f}, loc={val_loc_loss:.4f}) lr={lr:.6f}"
        )

        if args.wandb and wandb_run:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/cls_loss": train_cls_loss,
                    "train/loc_loss": train_loc_loss,
                    "val/loss": val_loss,
                    "val/cls_loss": val_cls_loss,
                    "val/loc_loss": val_loc_loss,
                    "learning_rate": lr,
                }
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_ckpt = os.path.join(args.save_dir, "vgg11_ssd_ann_dsec_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                    "args": vars(args),
                },
                best_ckpt,
            )
            print(f"Saved best checkpoint: {best_ckpt}")
            if args.wandb and wandb_run:
                wandb.save(best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch} after {args.early_stop_patience} non-improving epochs. "
                    f"best_val={best_val_loss:.4f}"
                )
                break

        if (epoch + 1) % args.save_freq == 0:
            ckpt = os.path.join(args.save_dir, f"vgg11_ssd_ann_dsec_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                    "args": vars(args),
                },
                ckpt,
            )

    print(f"Training complete. best_val={best_val_loss:.4f}")
    if args.wandb and wandb_run:
        wandb.finish()


if __name__ == "__main__":
    main()
