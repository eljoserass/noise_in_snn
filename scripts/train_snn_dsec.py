"""
Train VGG11-SSD SNN on DSEC event data.

Important: this training path does NOT repeat each frame for extra timesteps.
Temporal integration is only through membrane state across chronological frames.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dsec_dataset import DSECSSD_SNN, parse_class_ids
from src.models.snn import VGG11_SSD_SNN
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
    print(f"[train_snn_dsec] val directory not found: {val_dir}")
    args.val_split = args.train_split
    if args.val_sequences.strip():
        print(
            "[train_snn_dsec] using provided --val-sequences with --val-split=train "
            "(DSEC val is typically a subsequence list of train)."
        )
        return
    seqs = _load_split_list(Path(args.split_config), "val")
    if seqs:
        args.val_sequences = ",".join(seqs)
        print(
            f"[train_snn_dsec] loaded {len(seqs)} val sequences from {args.split_config} "
            "and switched --val-split to train"
        )
    else:
        print(
            "[train_snn_dsec] split config missing/empty; "
            "falling back to full train split for validation."
        )


def collate_fn_snn(batch):
    images_list, targets_list = zip(*batch)
    return list(images_list), list(targets_list)


def _maybe_resize(frame: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if frame.shape[-2:] == (height, width):
        return frame
    return F.interpolate(frame, size=(height, width), mode="bilinear", align_corners=False)


def train_one_epoch(
    model: VGG11_SSD_SNN,
    dataloader: DataLoader,
    criterion: SSDLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    anchors: torch.Tensor,
    input_height: int,
    input_width: int,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for sequences, targets_sequences in pbar:
        optimizer.zero_grad()

        batch_loss_val = 0.0
        batch_cls_val = 0.0
        batch_loc_val = 0.0
        num_sequences = max(1, len(sequences))

        for seq_images, seq_targets in zip(sequences, targets_sequences):
            model.reset_states()
            num_frames = max(1, seq_images.shape[0])

            for frame_idx in range(num_frames):
                frame = seq_images[frame_idx : frame_idx + 1].to(device)
                frame = _maybe_resize(frame, input_height, input_width)
                target = seq_targets[frame_idx]

                gt_boxes = target["boxes"].to(device)
                gt_labels = target["labels"].to(device)
                cls_target, loc_target = match_anchors_to_targets(
                    anchors, gt_boxes, gt_labels, iou_threshold=0.5
                )
                cls_target = cls_target.unsqueeze(0)
                loc_target = loc_target.unsqueeze(0)

                cls_preds, loc_preds = model(frame)
                loss, (cls_loss, loc_loss) = criterion(cls_preds, loc_preds, cls_target, loc_target)

                scale = 1.0 / float(num_frames * num_sequences)
                (loss * scale).backward()

                batch_loss_val += loss.item() * scale
                batch_cls_val += cls_loss.item() * scale
                batch_loc_val += loc_loss.item() * scale

                model.detach_states()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += batch_loss_val
        total_cls_loss += batch_cls_val
        total_loc_loss += batch_loc_val
        pbar.set_postfix(
            {
                "loss": f"{batch_loss_val:.4f}",
                "cls": f"{batch_cls_val:.4f}",
                "loc": f"{batch_loc_val:.4f}",
            }
        )

    denom = max(1, len(dataloader))
    return total_loss / denom, total_cls_loss / denom, total_loc_loss / denom


@torch.no_grad()
def validate(
    model: VGG11_SSD_SNN,
    dataloader: DataLoader,
    criterion: SSDLoss,
    device: torch.device,
    anchors: torch.Tensor,
    input_height: int,
    input_width: int,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0

    for sequences, targets_sequences in tqdm(dataloader, desc="Validation"):
        batch_loss_val = 0.0
        batch_cls_val = 0.0
        batch_loc_val = 0.0
        num_sequences = max(1, len(sequences))

        for seq_images, seq_targets in zip(sequences, targets_sequences):
            model.reset_states()
            num_frames = max(1, seq_images.shape[0])

            for frame_idx in range(num_frames):
                frame = seq_images[frame_idx : frame_idx + 1].to(device)
                frame = _maybe_resize(frame, input_height, input_width)
                target = seq_targets[frame_idx]

                gt_boxes = target["boxes"].to(device)
                gt_labels = target["labels"].to(device)
                cls_target, loc_target = match_anchors_to_targets(
                    anchors, gt_boxes, gt_labels, iou_threshold=0.5
                )
                cls_target = cls_target.unsqueeze(0)
                loc_target = loc_target.unsqueeze(0)

                cls_preds, loc_preds = model(frame)
                loss, (cls_loss, loc_loss) = criterion(cls_preds, loc_preds, cls_target, loc_target)

                scale = 1.0 / float(num_frames * num_sequences)
                batch_loss_val += loss.item() * scale
                batch_cls_val += cls_loss.item() * scale
                batch_loc_val += loc_loss.item() * scale

        total_loss += batch_loss_val
        total_cls_loss += batch_cls_val
        total_loc_loss += batch_loc_val

    denom = max(1, len(dataloader))
    return total_loss / denom, total_cls_loss / denom, total_loc_loss / denom


def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG11 SSD SNN on DSEC event sequences")

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

    parser.add_argument("--event-source", type=str, default="auto", choices=["auto", "real", "simulated"])
    parser.add_argument("--event-relpath", type=str, default="events/left/events.h5")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=8)
    parser.add_argument("--event-window-mode", type=str, default="between_frames", choices=["between_frames", "fixed"])
    parser.add_argument("--event-window-us", type=int, default=50_000)
    parser.add_argument("--simulated-fps", type=float, default=20.0)
    parser.add_argument("--count-clip-value", type=float, default=5.0)
    parser.add_argument("--no-log-counts", action="store_true")

    # Model/training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--input-height", type=int, default=480)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--surrogate-slope", type=float, default=25.0)

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
    print("SNN temporal mode: one forward pass per frame (no repeated timestep accumulation)")

    wandb_run = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                tags=["snn", "dsec", "events"],
            )
            print(f"wandb: {wandb_run.url}")
        except ImportError:
            print("wandb not installed; continuing without wandb")
            args.wandb = False

    train_dataset = DSECSSD_SNN(
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
        event_source=args.event_source,
        event_relpath=args.event_relpath,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        event_window_mode=args.event_window_mode,
        event_window_us=args.event_window_us,
        simulated_fps=args.simulated_fps,
        count_clip_value=args.count_clip_value,
        log_counts=not args.no_log_counts,
    )
    val_dataset = DSECSSD_SNN(
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
        event_source=args.event_source,
        event_relpath=args.event_relpath,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        event_window_mode=args.event_window_mode,
        event_window_us=args.event_window_us,
        simulated_fps=args.simulated_fps,
        count_clip_value=args.count_clip_value,
        log_counts=not args.no_log_counts,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_snn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_snn,
        pin_memory=True,
    )

    from snntorch import surrogate

    model = VGG11_SSD_SNN(
        num_classes=num_classes + 1,
        beta=args.beta,
        threshold=args.threshold,
        spike_grad=surrogate.fast_sigmoid(slope=args.surrogate_slope),
    ).to(device)
    anchors = generate_anchors_for_model(model, (2, args.input_height, args.input_width), device)
    criterion = SSDLoss(num_classes=num_classes + 1)
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

    print(f"Training SNN DSEC: train={len(train_dataset)} val={len(val_dataset)}")
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_cls_loss, train_loc_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            anchors,
            args.input_height,
            args.input_width,
        )
        val_loss, val_cls_loss, val_loc_loss = validate(
            model, val_loader, criterion, device, anchors, args.input_height, args.input_width
        )
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
            best_ckpt = os.path.join(args.save_dir, "vgg11_ssd_snn_dsec_best.pth")
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
            ckpt = os.path.join(args.save_dir, f"vgg11_ssd_snn_dsec_epoch_{epoch}.pth")
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
