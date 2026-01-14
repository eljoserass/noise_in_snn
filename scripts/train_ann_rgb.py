"""
Training script for SSD with VGG-9 backbone on TUMTraf RGB dataset.
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

from src.data.dataset import TUMTrafSSD
from src.models.ann import SSD_VGG9
from src.utils import SSDLoss, match_anchors_to_targets


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def train_one_epoch(model: SSD_VGG9, dataloader: DataLoader, criterion: SSDLoss,
                    optimizer: optim.Optimizer, device: torch.device, epoch: int) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0
    
    anchors = model.get_anchors(device)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        
        # Prepare targets
        batch_cls_targets = []
        batch_loc_targets = []
        
        for target in targets:
            boxes = target['boxes'].to(device)
            labels = target['labels'].to(device)
            
            cls_targets, loc_targets = match_anchors_to_targets(anchors, boxes, labels)
            batch_cls_targets.append(cls_targets)
            batch_loc_targets.append(loc_targets)
        
        cls_targets = torch.stack(batch_cls_targets).to(device)
        loc_targets = torch.stack(batch_loc_targets).to(device)
        
        # Forward pass
        cls_preds, loc_preds = model(images)
        
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
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_loc_loss = total_loc_loss / len(dataloader)
    
    return avg_loss, avg_cls_loss, avg_loc_loss


@torch.no_grad()
def validate(model: SSD_VGG9, dataloader: DataLoader, criterion: SSDLoss, device: torch.device) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    anchors = model.get_anchors(device)
    
    for images, targets in tqdm(dataloader, desc="Validation"):
        images = images.to(device)
        
        # Prepare targets
        batch_cls_targets = []
        batch_loc_targets = []
        
        for target in targets:
            boxes = target['boxes'].to(device)
            labels = target['labels'].to(device)
            
            cls_targets, loc_targets = match_anchors_to_targets(anchors, boxes, labels)
            batch_cls_targets.append(cls_targets)
            batch_loc_targets.append(loc_targets)
        
        cls_targets = torch.stack(batch_cls_targets).to(device)
        loc_targets = torch.stack(batch_loc_targets).to(device)
        
        # Forward pass
        cls_preds, loc_preds = model(images)
        
        # Compute loss
        loss, _ = criterion(cls_preds, loc_preds, cls_targets, loc_targets)
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SSD with VGG-9 backbone on TUMTraf RGB dataset")
    
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
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
    
    train_dataset = TUMTrafSSD(
        img_dir=train_img_dir,
        label_dir=train_label_dir,
        transform=transform,
        target_size=(args.input_height, args.input_width)
    )
    
    val_dataset = TUMTrafSSD(
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
    model = SSD_VGG9(
        num_classes=args.num_classes,
        input_size=(args.input_height, args.input_width),
        in_channels=3
    )
    model = model.to(device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss function
    criterion = SSDLoss(num_classes=args.num_classes + 1)  # +1 for background
    
    # Optimizer with different learning rates for backbone and heads
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.extras.parameters()) + list(model.pred_heads.parameters())
    
    optimizer = optim.SGD([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': head_params, 'lr': args.lr}
    ], momentum=args.momentum, weight_decay=args.weight_decay)
    
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
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()

        print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f} (cls: {train_cls_loss:.4f}, loc: {train_loc_loss:.4f}), Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }, os.path.join(args.save_dir, 'ssd_vgg9_best.pth'))
            print(f"Saved best model with val loss: {val_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }, os.path.join(args.save_dir, f'ssd_vgg9_epoch_{epoch}.pth'))
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
 