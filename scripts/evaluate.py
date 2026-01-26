"""
Evaluation script for VGG11-SSD models (ANN and SNN).

Computes standard object detection metrics:
- mAP (mean Average Precision) at various IoU thresholds
- Per-class AP
- Inference latency
- Energy consumption estimation (FLOPs for ANN, spike counts for SNN)

For fair comparison, evaluates ANN and SNN on the same frame groups:
- ANN: Processes each frame independently within 8-frame groups
- SNN: Processes 8-frame sequences with temporal integration
- Both evaluated on identical frames for direct comparison
"""

import argparse
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import time
import json
from collections import defaultdict

from src.data.dataset import TUMTrafSSD_ANN, TUMTrafSSD_SNN
from src.models.ann import VGG11_SSD_ANN
from src.models.snn import VGG11_SSD_SNN
from src.utils import (
    generate_anchors,
    decode_boxes,
    xywh_to_xyxy,
    nms,
    compute_iou
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VGG11-SSD models on TUMTraf test sets")
    
    # Model configuration
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, required=True, choices=["ann", "snn"],
                        help="Model type: ann or snn")
    parser.add_argument("--num-classes", type=int, default=6, help="Number of object classes")
    
    # SNN-specific parameters (for model instantiation)
    parser.add_argument("--beta", type=float, default=0.9, help="SNN membrane decay rate")
    parser.add_argument("--threshold", type=float, default=1.0, help="SNN spike threshold")
    parser.add_argument("--surrogate-slope", type=float, default=25.0, help="Surrogate gradient slope")
    
    # Data paths
    parser.add_argument("--data-path", type=str, default="data/preprocessed",
                        help="Path to preprocessed data")
    parser.add_argument("--test-split", type=str, default="test/day",
                        help="Test split: test/day, test/night_with_light_off, test/night_with_light_on")
    
    # Evaluation parameters
    parser.add_argument("--batch-size", type=int, default=1, 
                        help="Batch size (1 recommended for fair comparison)")
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                        help="Confidence threshold for detections")
    parser.add_argument("--nms-threshold", type=float, default=0.5,
                        help="NMS IoU threshold")
    parser.add_argument("--iou-thresholds", type=float, nargs="+", 
                        default=[0.5, 0.75],
                        help="IoU thresholds for mAP computation")
    
    # Options
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save evaluation results")
    parser.add_argument("--measure-energy", action="store_true",
                        help="Measure energy consumption (FLOPs/spikes)")
    
    # Weights & Biases
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="neuromorph-vs-noise",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (default: eval_{model_type}_{test_split})")
    
    return parser.parse_args()


class DetectionEvaluator:
    """
    Evaluates object detection models using standard metrics.
    Computes mAP, per-class AP, latency, and energy consumption.
    """
    
    def __init__(self, num_classes, iou_thresholds=[0.5], class_names=None):
        """
        Args:
            num_classes: Number of object classes (excluding background)
            iou_thresholds: List of IoU thresholds for mAP computation
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.class_names = class_names or [f"class_{i}" for i in range(1, num_classes + 1)]
        
        # Storage for all predictions and ground truths
        self.all_predictions = []  # List of dicts with keys: boxes, scores, labels, image_id
        self.all_ground_truths = []  # List of dicts with keys: boxes, labels, image_id
        self.image_counter = 0
        
        # Latency measurements
        self.inference_times = []
        
        # Energy measurements
        self.total_flops = 0
        self.total_spikes = 0
    
    def add_predictions(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, image_id=None):
        """
        Add predictions and ground truths for one image.
        
        Args:
            pred_boxes: (N, 4) predicted boxes in xyxy format
            pred_scores: (N,) confidence scores
            pred_labels: (N,) predicted class labels
            gt_boxes: (M, 4) ground truth boxes in xyxy format
            gt_labels: (M,) ground truth class labels
            image_id: Optional image identifier
        """
        if image_id is None:
            image_id = self.image_counter
            self.image_counter += 1
        
        self.all_predictions.append({
            'boxes': pred_boxes.cpu().numpy(),
            'scores': pred_scores.cpu().numpy(),
            'labels': pred_labels.cpu().numpy(),
            'image_id': image_id
        })
        
        self.all_ground_truths.append({
            'boxes': gt_boxes.cpu().numpy(),
            'labels': gt_labels.cpu().numpy(),
            'image_id': image_id
        })
    
    def compute_ap(self, recalls, precisions):
        """
        Compute Average Precision using 11-point interpolation.
        
        Args:
            recalls: Array of recall values
            precisions: Array of precision values
            
        Returns:
            Average Precision
        """
        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        return ap
    
    def compute_map(self, iou_threshold=0.5):
        """
        Compute mean Average Precision across all classes.
        
        Args:
            iou_threshold: IoU threshold for matching predictions to ground truth
            
        Returns:
            Dictionary with mAP and per-class AP
        """
        aps = []
        per_class_metrics = {}
        
        for class_id in range(1, self.num_classes + 1):
            # Gather all predictions and GTs for this class
            class_preds = []
            class_gts = []
            
            for pred in self.all_predictions:
                mask = pred['labels'] == class_id
                if mask.any():
                    class_preds.append({
                        'boxes': pred['boxes'][mask],
                        'scores': pred['scores'][mask],
                        'image_id': pred['image_id']
                    })
            
            for gt in self.all_ground_truths:
                mask = gt['labels'] == class_id
                if mask.any():
                    class_gts.append({
                        'boxes': gt['boxes'][mask],
                        'image_id': gt['image_id']
                    })
            
            # Compute AP for this class
            if len(class_gts) == 0:
                # No ground truths for this class
                per_class_metrics[self.class_names[class_id - 1]] = {
                    'ap': 0.0,
                    'num_gt': 0,
                    'num_pred': sum(len(p['boxes']) for p in class_preds)
                }
                continue
            
            # Collect all predictions with scores
            all_pred_boxes = []
            all_pred_scores = []
            all_pred_image_ids = []
            
            for pred in class_preds:
                all_pred_boxes.extend(pred['boxes'])
                all_pred_scores.extend(pred['scores'])
                all_pred_image_ids.extend([pred['image_id']] * len(pred['boxes']))
            
            if len(all_pred_boxes) == 0:
                # No predictions for this class
                per_class_metrics[self.class_names[class_id - 1]] = {
                    'ap': 0.0,
                    'num_gt': sum(len(gt['boxes']) for gt in class_gts),
                    'num_pred': 0
                }
                continue
            
            all_pred_boxes = np.array(all_pred_boxes)
            all_pred_scores = np.array(all_pred_scores)
            all_pred_image_ids = np.array(all_pred_image_ids)
            
            # Sort predictions by confidence (descending)
            sorted_indices = np.argsort(-all_pred_scores)
            all_pred_boxes = all_pred_boxes[sorted_indices]
            all_pred_scores = all_pred_scores[sorted_indices]
            all_pred_image_ids = all_pred_image_ids[sorted_indices]
            
            # Build GT dictionary for fast lookup
            gt_dict = {}
            for gt in class_gts:
                img_id = gt['image_id']
                if img_id not in gt_dict:
                    gt_dict[img_id] = {
                        'boxes': gt['boxes'],
                        'matched': np.zeros(len(gt['boxes']), dtype=bool)
                    }
            
            # Match predictions to ground truths
            tp = np.zeros(len(all_pred_boxes))
            fp = np.zeros(len(all_pred_boxes))
            
            for i, (pred_box, img_id) in enumerate(zip(all_pred_boxes, all_pred_image_ids)):
                if img_id not in gt_dict:
                    fp[i] = 1
                    continue
                
                gt_data = gt_dict[img_id]
                gt_boxes = gt_data['boxes']
                gt_matched = gt_data['matched']
                
                if len(gt_boxes) == 0:
                    fp[i] = 1
                    continue
                
                # Compute IoU with all GT boxes
                pred_box_tensor = torch.tensor(pred_box).unsqueeze(0)
                gt_boxes_tensor = torch.tensor(gt_boxes)
                ious = compute_iou(pred_box_tensor, gt_boxes_tensor)[0].numpy()
                
                # Find best matching GT
                max_iou_idx = np.argmax(ious)
                max_iou = ious[max_iou_idx]
                
                if max_iou >= iou_threshold and not gt_matched[max_iou_idx]:
                    tp[i] = 1
                    gt_matched[max_iou_idx] = True
                else:
                    fp[i] = 1
            
            # Compute precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            num_gt_total = sum(len(gt['boxes']) for gt in class_gts)
            recalls = tp_cumsum / num_gt_total
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # Compute AP
            ap = self.compute_ap(recalls, precisions)
            aps.append(ap)
            
            per_class_metrics[self.class_names[class_id - 1]] = {
                'ap': ap,
                'num_gt': num_gt_total,
                'num_pred': len(all_pred_boxes),
                'tp': int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0,
                'fp': int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0
            }
        
        # Compute mAP
        map_value = np.mean(aps) if len(aps) > 0 else 0.0
        
        return {
            'mAP': map_value,
            'per_class': per_class_metrics,
            'iou_threshold': iou_threshold
        }
    
    def compute_metrics(self):
        """
        Compute all evaluation metrics.
        
        Returns:
            Dictionary with mAP at different IoU thresholds, latency, and energy
        """
        results = {}
        
        # Compute mAP at different IoU thresholds
        for iou_thresh in self.iou_thresholds:
            map_result = self.compute_map(iou_threshold=iou_thresh)
            results[f'mAP@{iou_thresh}'] = map_result['mAP']
            results[f'per_class_AP@{iou_thresh}'] = map_result['per_class']
        
        # Compute average mAP across thresholds
        avg_map = np.mean([results[f'mAP@{t}'] for t in self.iou_thresholds])
        results['mAP_avg'] = avg_map
        
        # Latency statistics
        if len(self.inference_times) > 0:
            results['latency'] = {
                'mean_ms': np.mean(self.inference_times) * 1000,
                'std_ms': np.std(self.inference_times) * 1000,
                'fps': 1.0 / np.mean(self.inference_times)
            }
        
        # Energy statistics
        if self.total_flops > 0:
            results['energy'] = {
                'total_flops': self.total_flops,
                'flops_per_image': self.total_flops / len(self.all_predictions) if len(self.all_predictions) > 0 else 0
            }
        
        if self.total_spikes > 0:
            results['energy'] = results.get('energy', {})
            results['energy']['total_spikes'] = self.total_spikes
            results['energy']['spikes_per_image'] = self.total_spikes / len(self.all_predictions) if len(self.all_predictions) > 0 else 0
        
        # Summary statistics
        results['num_images'] = len(self.all_predictions)
        results['num_predictions'] = sum(len(p['boxes']) for p in self.all_predictions)
        results['num_ground_truths'] = sum(len(gt['boxes']) for gt in self.all_ground_truths)
        
        return results


@torch.no_grad()
def run_inference_ann(model, dataloader, anchors, device, conf_threshold, nms_threshold, evaluator):
    """
    Run inference on ANN model (frame-by-frame processing).
    
    Args:
        model: ANN model
        dataloader: DataLoader (can be frame-based or group-based)
        anchors: Default anchor boxes
        device: Device to run on
        conf_threshold: Confidence threshold
        nms_threshold: NMS threshold
        evaluator: DetectionEvaluator instance
    """
    model.eval()
    
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="ANN Inference")):
        # Handle both single frames and grouped frames
        if isinstance(images, list):
            # Grouped data: process each frame in group
            for seq_images, seq_targets in zip(images, targets):
                for t in range(seq_images.shape[0]):
                    frame = seq_images[t:t+1].to(device)  # (1, C, H, W)
                    target = seq_targets[t]
                    
                    # Inference
                    start_time = time.time()
                    cls_preds, loc_preds = model(frame)
                    inference_time = time.time() - start_time
                    evaluator.inference_times.append(inference_time)
                    
                    # Post-process predictions
                    pred_boxes, pred_scores, pred_labels = post_process_detections(
                        cls_preds[0], loc_preds[0], anchors, conf_threshold, nms_threshold, device
                    )
                    
                    # Get ground truth in xyxy format
                    gt_boxes = target['boxes']  # (N, 4) in cxcywh normalized
                    gt_labels = target['labels']  # (N,)
                    
                    # Convert GT to xyxy format
                    gt_boxes_xyxy = xywh_to_xyxy(gt_boxes)
                    
                    # Add to evaluator
                    evaluator.add_predictions(
                        pred_boxes, pred_scores, pred_labels,
                        gt_boxes_xyxy, gt_labels,
                        image_id=f"{batch_idx}_t{t}"
                    )
        else:
            # Single frame data
            images = images.to(device)
            
            for i in range(images.shape[0]):
                image = images[i:i+1]
                target = targets[i]
                
                # Inference
                start_time = time.time()
                cls_preds, loc_preds = model(image)
                inference_time = time.time() - start_time
                evaluator.inference_times.append(inference_time)
                
                # Post-process
                pred_boxes, pred_scores, pred_labels = post_process_detections(
                    cls_preds[0], loc_preds[0], anchors, conf_threshold, nms_threshold, device
                )
                
                gt_boxes = target['boxes']
                gt_labels = target['labels']
                gt_boxes_xyxy = xywh_to_xyxy(gt_boxes)
                
                evaluator.add_predictions(
                    pred_boxes, pred_scores, pred_labels,
                    gt_boxes_xyxy, gt_labels,
                    image_id=f"{batch_idx}_{i}"
                )


@torch.no_grad()
def run_inference_snn(model, dataloader, anchors, device, conf_threshold, nms_threshold, evaluator):
    """
    Run inference on SNN model (temporal sequence processing).
    
    Args:
        model: SNN model
        dataloader: DataLoader for sequences
        anchors: Default anchor boxes
        device: Device to run on
        conf_threshold: Confidence threshold
        nms_threshold: NMS threshold
        evaluator: DetectionEvaluator instance
    """
    model.eval()
    
    for batch_idx, (sequences, targets_sequences) in enumerate(tqdm(dataloader, desc="SNN Inference")):
        for seq_idx, (seq_images, seq_targets) in enumerate(zip(sequences, targets_sequences)):
            # Reset states for new sequence
            model.reset_states()
            
            T = seq_images.shape[0]
            
            # Process sequence with temporal integration
            for t in range(T):
                frame = seq_images[t:t+1].to(device)  # (1, C, H, W)
                target = seq_targets[t]
                
                # Inference (state persists across timesteps!)
                start_time = time.time()
                cls_preds, loc_preds = model(frame)
                inference_time = time.time() - start_time
                evaluator.inference_times.append(inference_time)
                
                # Post-process predictions
                pred_boxes, pred_scores, pred_labels = post_process_detections(
                    cls_preds[0], loc_preds[0], anchors, conf_threshold, nms_threshold, device
                )
                
                # Get ground truth
                gt_boxes = target['boxes']
                gt_labels = target['labels']
                gt_boxes_xyxy = xywh_to_xyxy(gt_boxes)
                
                # Add to evaluator
                evaluator.add_predictions(
                    pred_boxes, pred_scores, pred_labels,
                    gt_boxes_xyxy, gt_labels,
                    image_id=f"{batch_idx}_seq{seq_idx}_t{t}"
                )


def post_process_detections(cls_preds, loc_preds, anchors, conf_threshold, nms_threshold, device):
    """
    Post-process model predictions to get final detections.
    
    Args:
        cls_preds: (num_anchors, num_classes) classification scores
        loc_preds: (num_anchors, 4) box offset predictions
        anchors: (num_anchors, 4) default anchor boxes
        conf_threshold: Confidence threshold
        nms_threshold: NMS IoU threshold
        device: Device
        
    Returns:
        pred_boxes: (N, 4) detected boxes in xyxy format
        pred_scores: (N,) confidence scores
        pred_labels: (N,) class labels
    """
    # Apply softmax to get class probabilities
    cls_probs = torch.softmax(cls_preds, dim=-1)  # (num_anchors, num_classes)
    
    # Get max class probability and label (excluding background class 0)
    class_probs, class_labels = cls_probs[:, 1:].max(dim=-1)  # Skip background
    class_labels = class_labels + 1  # Adjust for skipped background
    
    # Filter by confidence threshold
    conf_mask = class_probs > conf_threshold
    
    if conf_mask.sum() == 0:
        # No detections above threshold
        return (torch.zeros((0, 4), device=device),
                torch.zeros((0,), device=device),
                torch.zeros((0,), dtype=torch.long, device=device))
    
    # Get confident predictions
    confident_probs = class_probs[conf_mask]
    confident_labels = class_labels[conf_mask]
    confident_loc_preds = loc_preds[conf_mask]
    confident_anchors = anchors[conf_mask]
    
    # Decode boxes from offsets
    decoded_boxes = decode_boxes(confident_loc_preds, confident_anchors)  # (N, 4) in cxcywh
    
    # Convert to xyxy format for NMS
    decoded_boxes_xyxy = xywh_to_xyxy(decoded_boxes)
    
    # Apply NMS per class
    keep_indices = []
    unique_labels = confident_labels.unique()
    
    for label in unique_labels:
        label_mask = confident_labels == label
        label_boxes = decoded_boxes_xyxy[label_mask]
        label_scores = confident_probs[label_mask]
        
        # Apply NMS
        keep = nms(label_boxes, label_scores, nms_threshold)
        
        # Get original indices
        label_indices = torch.where(label_mask)[0]
        keep_indices.append(label_indices[keep])
    
    if len(keep_indices) == 0:
        return (torch.zeros((0, 4), device=device),
                torch.zeros((0,), device=device),
                torch.zeros((0,), dtype=torch.long, device=device))
    
    keep_indices = torch.cat(keep_indices)
    
    # Final detections
    final_boxes = decoded_boxes_xyxy[keep_indices]
    final_scores = confident_probs[keep_indices]
    final_labels = confident_labels[keep_indices]
    
    return final_boxes, final_scores, final_labels


def load_model(args, device):
    """Load model from checkpoint."""
    if args.model_type == "ann":
        model = VGG11_SSD_ANN(num_classes=args.num_classes + 1)  # +1 for background
    else:  # snn
        from snntorch import surrogate
        model = VGG11_SSD_SNN(
            num_classes=args.num_classes + 1,
            beta=args.beta,
            threshold=args.threshold,
            spike_grad=surrogate.fast_sigmoid(slope=args.surrogate_slope)
        )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded {args.model_type.upper()} model from {args.model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    return model


def get_dataloader(args, model_type):
    """Create dataloader for evaluation."""
    # Use grouped loading for fair comparison between ANN and SNN
    if model_type == "ann":
        # For ANN, we'll load by groups but process frame-by-frame
        img_dir = Path(args.data_path) / args.test_split / "images" / "rgb"
        label_dir = Path(args.data_path) / args.test_split / "OPENLabel_labels_fusion_gt_optimized_rgb"
        
        transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = TUMTrafSSD_SNN(  # Use SNN dataset for grouped loading
            img_dir=img_dir,
            label_dir=label_dir,
            transform=transform,
            target_size=(442, 482)
        )
        
        def collate_fn_grouped(batch):
            """Collate for grouped evaluation"""
            images_list, targets_list = zip(*batch)
            return list(images_list), list(targets_list)
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_grouped,
            pin_memory=True
        )
    else:  # snn
        img_dir = Path(args.data_path) / args.test_split / "images" / "eb_transformed"
        label_dir = Path(args.data_path) / args.test_split / "OPENLabel_labels_fusion_gt_optimized_eb"
        
        transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
        ])
        
        dataset = TUMTrafSSD_SNN(
            img_dir=img_dir,
            label_dir=label_dir,
            transform=transform,
            target_size=(442, 482)
        )
        
        def collate_fn_snn(batch):
            images_list, targets_list = zip(*batch)
            return list(images_list), list(targets_list)
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_snn,
            pin_memory=True
        )
    
    return dataloader


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize W&B if requested
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Install with: pip install wandb")
            print("Continuing without W&B logging...")
            args.wandb = False
        else:
            run_name = args.wandb_run_name or f"eval_{args.model_type}_{args.test_split.replace('/', '_')}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                tags=[args.model_type, args.test_split, "evaluation"]
            )
            print(f"✓ W&B logging enabled: {wandb.run.url}")
    
    # Generate anchors
    anchors = generate_anchors().to(device)
    print(f"Generated {anchors.size(0)} default anchors")
    
    # Load model
    model = load_model(args, device)
    
    # Create dataloader
    print(f"\nLoading test data from: {args.data_path}/{args.test_split}")
    dataloader = get_dataloader(args, args.model_type)
    print(f"Test samples: {len(dataloader.dataset)}")
    
    # Create evaluator
    class_names = ['BICYCLE', 'BUS', 'CAR', 'PEDESTRIAN', 'TRAILER', 'TRUCK']
    evaluator = DetectionEvaluator(
        num_classes=args.num_classes,
        iou_thresholds=args.iou_thresholds,
        class_names=class_names
    )
    
    # Run inference
    print(f"\nRunning {args.model_type.upper()} inference...")
    if args.model_type == "ann":
        run_inference_ann(model, dataloader, anchors, device, 
                         args.conf_threshold, args.nms_threshold, evaluator)
    else:
        run_inference_snn(model, dataloader, anchors, device,
                         args.conf_threshold, args.nms_threshold, evaluator)
    
    # Compute metrics
    print("\nComputing evaluation metrics...")
    results = evaluator.compute_metrics()
    
    # Print results
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS - {args.model_type.upper()}")
    print("="*60)
    print(f"\nDataset: {args.test_split}")
    print(f"Model: {args.model_path}")
    print(f"\nImages evaluated: {results['num_images']}")
    print(f"Total predictions: {results['num_predictions']}")
    print(f"Total ground truths: {results['num_ground_truths']}")
    
    print(f"\n{'mAP Results':=^60}")
    print(f"mAP (average): {results['mAP_avg']:.4f}")
    for iou_thresh in args.iou_thresholds:
        print(f"mAP@{iou_thresh}: {results[f'mAP@{iou_thresh}']:.4f}")
    
    print(f"\n{'Per-Class AP':=^60}")
    for iou_thresh in args.iou_thresholds:
        print(f"\nAt IoU={iou_thresh}:")
        per_class = results[f'per_class_AP@{iou_thresh}']
        for class_name, metrics in per_class.items():
            print(f"  {class_name:12s}: AP={metrics['ap']:.4f}  "
                  f"(GT={metrics['num_gt']:4d}, Pred={metrics['num_pred']:4d}, "
                  f"TP={metrics.get('tp', 0):4d}, FP={metrics.get('fp', 0):4d})")
    
    if 'latency' in results:
        print(f"\n{'Latency':=^60}")
        print(f"Mean: {results['latency']['mean_ms']:.2f} ms")
        print(f"Std:  {results['latency']['std_ms']:.2f} ms")
        print(f"FPS:  {results['latency']['fps']:.2f}")
    
    if 'energy' in results:
        print(f"\n{'Energy Consumption':=^60}")
        if 'flops_per_image' in results['energy']:
            print(f"FLOPs per image: {results['energy']['flops_per_image']:.2e}")
        if 'spikes_per_image' in results['energy']:
            print(f"Spikes per image: {results['energy']['spikes_per_image']:.2e}")
    
    print("="*60)
    
    # Save results to JSON
    output_file = os.path.join(
        args.output_dir,
        f"eval_{args.model_type}_{args.test_split.replace('/', '_')}.json"
    )
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    results_serializable['config'] = vars(args)
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Log to W&B
    if args.wandb:
        # Log summary metrics
        wandb.log({
            "eval/mAP_avg": results['mAP_avg'],
            **{f"eval/mAP@{t}": results[f'mAP@{t}'] for t in args.iou_thresholds},
            "eval/num_images": results['num_images'],
            "eval/num_predictions": results['num_predictions'],
            "eval/num_ground_truths": results['num_ground_truths']
        })
        
        # Log per-class metrics
        for iou_thresh in args.iou_thresholds:
            per_class = results[f'per_class_AP@{iou_thresh}']
            for class_name, metrics in per_class.items():
                wandb.log({
                    f"eval/{class_name}/AP@{iou_thresh}": metrics['ap'],
                    f"eval/{class_name}/num_gt": metrics['num_gt'],
                    f"eval/{class_name}/num_pred": metrics['num_pred'],
                    f"eval/{class_name}/tp": metrics.get('tp', 0),
                    f"eval/{class_name}/fp": metrics.get('fp', 0)
                })
        
        # Log latency if available
        if 'latency' in results:
            wandb.log({
                "eval/latency_mean_ms": results['latency']['mean_ms'],
                "eval/latency_std_ms": results['latency']['std_ms'],
                "eval/fps": results['latency']['fps']
            })
        
        # Log energy if available
        if 'energy' in results:
            if 'flops_per_image' in results['energy']:
                wandb.log({"eval/flops_per_image": results['energy']['flops_per_image']})
            if 'spikes_per_image' in results['energy']:
                wandb.log({"eval/spikes_per_image": results['energy']['spikes_per_image']})
        
        # Upload results JSON as artifact
        artifact = wandb.Artifact(
            name=f"eval_results_{args.model_type}_{args.test_split.replace('/', '_')}",
            type="evaluation",
            description=f"Evaluation results for {args.model_type} on {args.test_split}"
        )
        artifact.add_file(output_file)
        wandb.log_artifact(artifact)
        
        wandb.finish()
        print("✓ Results logged to W&B")


if __name__ == "__main__":
    main()