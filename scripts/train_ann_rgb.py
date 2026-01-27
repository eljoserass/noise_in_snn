"""
Training script for VGG11 SSD model on TUMTraf RGB dataset.
"""

import argparse
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from typing import Tuple

from src.data.dataset import TUMTrafSSD_ANN
from src.models.ann import VGG11_SSD_ANN
from src.utils import SSDLoss, match_anchors_to_targets, generate_anchors_for_model


# Global anchors (generated once at start)
ANCHORS = None


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def train_one_epoch(model: VGG11_SSD_ANN, dataloader: DataLoader, criterion: SSDLoss,
                    optimizer: optim.Optimizer, device: torch.device, epoch: int, 
                    anchors: torch.Tensor) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        
        # Forward pass
        cls_preds, loc_preds = model(images)
        
        # Match anchors to ground truth for each image in batch
        batch_size = images.size(0)
        batch_cls_targets = []
        batch_loc_targets = []
        
        for i in range(batch_size):
            gt_boxes = targets[i]['boxes'].to(device)
            gt_labels = targets[i]['labels'].to(device)
            
            # Match anchors to ground truth
            cls_target, loc_target = match_anchors_to_targets(
                anchors, gt_boxes, gt_labels, iou_threshold=0.5
            )
            batch_cls_targets.append(cls_target)
            batch_loc_targets.append(loc_target)
        
        # Stack targets
        cls_targets = torch.stack(batch_cls_targets, dim=0)  # (B, num_anchors)
        loc_targets = torch.stack(batch_loc_targets, dim=0)  # (B, num_anchors, 4)
        
        # Compute loss
        loss, (cls_loss, loc_loss) = criterion(cls_preds, loc_preds, cls_targets, loc_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_loc_loss += loc_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'loc': f'{loc_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_cls_loss = total_cls_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_loc_loss = total_loc_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    return avg_loss, avg_cls_loss, avg_loc_loss


@torch.no_grad()
def validate(model: VGG11_SSD_ANN, dataloader: DataLoader, criterion: SSDLoss, 
             device: torch.device, anchors: torch.Tensor) -> Tuple[float, float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0
    
    for images, targets in tqdm(dataloader, desc="Validation"):
        images = images.to(device)
        
        # Forward pass
        cls_preds, loc_preds = model(images)
        
        # Match anchors to ground truth for each image in batch
        batch_size = images.size(0)
        batch_cls_targets = []
        batch_loc_targets = []
        
        for i in range(batch_size):
            gt_boxes = targets[i]['boxes'].to(device)
            gt_labels = targets[i]['labels'].to(device)
            
            cls_target, loc_target = match_anchors_to_targets(
                anchors, gt_boxes, gt_labels, iou_threshold=0.5
            )
            batch_cls_targets.append(cls_target)
            batch_loc_targets.append(loc_target)
        
        cls_targets = torch.stack(batch_cls_targets, dim=0)
        loc_targets = torch.stack(batch_loc_targets, dim=0)
        
        # Compute loss
        loss, (cls_loss, loc_loss) = criterion(cls_preds, loc_preds, cls_targets, loc_targets)
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_loc_loss += loc_loss.item()
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_cls_loss = total_cls_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_loc_loss = total_loc_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    return avg_loss, avg_cls_loss, avg_loc_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG11 SSD model on TUMTraf RGB dataset")
    
    # Data paths
    parser.add_argument("--data-path", type=str, default="data/preprocessed",
                        help="Path to preprocessed data")
    parser.add_argument("--train-split", type=str, default="train",
                        help="Training split name")
    parser.add_argument("--val-split", type=str, default="val",
                        help="Validation split name")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    
    # Model parameters
    parser.add_argument("--num-classes", type=int, default=6, help="Number of classes")
    parser.add_argument("--input-height", type=int, default=480, help="Input image height")
    parser.add_argument("--input-width", type=int, default=640, help="Input image width")
    
    # Training options
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--save-freq", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # W&B options
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="neuromorph-vs-noise", 
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, 
                        help="W&B run name (default: auto-generated)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize W&B if enabled
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                tags=["ann", "rgb", "vgg11-ssd"]
            )
            print(f"✓ W&B initialized: {wandb_run.url}")
        except ImportError:
            print("⚠ W&B not installed. Run: pip install wandb")
            args.wandb = False
    
    # Image transforms
    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_img_dir = Path(args.data_path) / args.train_split / "images" / "rgb"
    train_label_dir = Path(args.data_path) / args.train_split / "OPENLabel_labels_rgb"
    
    val_img_dir = Path(args.data_path) / args.val_split / "images" / "rgb"
    val_label_dir = Path(args.data_path) / args.val_split / "OPENLabel_labels_rgb"
    
    train_dataset = TUMTrafSSD_ANN(
        img_dir=train_img_dir,
        label_dir=train_label_dir,
        transform=transform,
        target_size=(args.input_height, args.input_width)
    )
    
    val_dataset = TUMTrafSSD_ANN(
        img_dir=val_img_dir,
        label_dir=val_label_dir,
        transform=transform,
        target_size=(args.input_height, args.input_width)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = VGG11_SSD_ANN(num_classes=args.num_classes + 1)  # +1 for background class
    model = model.to(device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: VGG11_SSD_ANN")
    print(f"Parameters: {num_params:,}")
    
    # Generate anchors dynamically based on actual model feature maps
    # This ensures anchors match the model's output dimensions for RGB images (480x640)
    global ANCHORS
    input_channels = 3  # RGB
    ANCHORS = generate_anchors_for_model(
        model, 
        (input_channels, args.input_height, args.input_width),
        device
    )
    num_anchors = ANCHORS.size(0)
    print(f"Generated {num_anchors} anchors for RGB images ({args.input_height}x{args.input_width})")
    
    # Loss function
    criterion = SSDLoss(num_classes=args.num_classes + 1)  # +1 for background
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[int(args.epochs * 0.6), int(args.epochs * 0.8)],
        gamma=0.1
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_cls_loss, train_loc_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, ANCHORS
        )
        
        # Validate
        val_loss, val_cls_loss, val_loc_loss = validate(model, val_loader, criterion, device, ANCHORS)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}: "
              f"Train Loss = {train_loss:.4f} (cls: {train_cls_loss:.4f}, loc: {train_loc_loss:.4f}), "
              f"Val Loss = {val_loss:.4f} (cls: {val_cls_loss:.4f}, loc: {val_loc_loss:.4f}), "
              f"LR = {current_lr:.6f}")
        
        # Log to W&B
        if args.wandb and wandb_run:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/cls_loss": train_cls_loss,
                "train/loc_loss": train_loc_loss,
                "val/loss": val_loss,
                "val/cls_loss": val_cls_loss,
                "val/loc_loss": val_loc_loss,
                "learning_rate": current_lr,
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.save_dir, 'vgg11_ssd_ann_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"✓ Saved best model with val loss: {val_loss:.4f}")
            
            # Log best model to W&B
            if args.wandb and wandb_run:
                wandb.save(checkpoint_path)
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }, os.path.join(args.save_dir, f'vgg11_ssd_ann_epoch_{epoch}.pth'))
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Finish W&B run
    if args.wandb and wandb_run:
        wandb.finish()


if __name__ == "__main__":
    main()
