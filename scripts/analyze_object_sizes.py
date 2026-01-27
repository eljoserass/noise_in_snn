"""
Analyze object sizes in TUMTraf dataset to understand anchor configuration.
This helps validate/tune the anchor aspect ratios and scales.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Paths
label_dir_rgb = Path("data/preprocessed/train/OPENLabel_labels_rgb")
label_dir_eb = Path("data/preprocessed/train/OPENLabel_labels_eb")

# Classes
CLASSES = ['BICYCLE', 'BUS', 'CAR', 'PEDESTRIAN', 'TRAILER', 'TRUCK']

def analyze_labels(label_dir, image_width=640, image_height=480, name="RGB"):
    """Analyze object sizes from labels"""
    
    objects_by_class = defaultdict(list)
    
    # Collect all boxes
    for group_folder in sorted(label_dir.iterdir()):
        if not group_folder.is_dir():
            continue
        
        for label_file in group_folder.glob("*.json"):
            with open(label_file, 'r') as f:
                data = json.load(f)
            
            frames = data.get('openlabel', {}).get('frames', {})
            for frame_id, frame_content in frames.items():
                objects = frame_content.get('objects', {})
                for obj_id, obj_content in objects.items():
                    obj_data = obj_content.get('object_data', {})
                    obj_type = obj_data.get('type', '')
                    
                    if obj_type not in CLASSES:
                        continue
                    
                    bbox_list = obj_data.get('bbox', [])
                    for bbox in bbox_list:
                        if bbox.get('name') == 'full_bbox':
                            x, y, w, h = bbox.get('val', [0, 0, 0, 0])
                            
                            # Normalize
                            norm_w = w / image_width
                            norm_h = h / image_height
                            aspect_ratio = w / h if h > 0 else 0
                            
                            objects_by_class[obj_type].append({
                                'width': norm_w,
                                'height': norm_h,
                                'aspect_ratio': aspect_ratio,
                                'area': norm_w * norm_h
                            })
    
    # Compute statistics
    print(f"\n{'='*70}")
    print(f"{name} Dataset Analysis ({image_width}×{image_height})")
    print(f"{'='*70}")
    
    for cls in CLASSES:
        if cls not in objects_by_class or len(objects_by_class[cls]) == 0:
            print(f"\n{cls}: No samples")
            continue
        
        boxes = objects_by_class[cls]
        widths = [b['width'] for b in boxes]
        heights = [b['height'] for b in boxes]
        aspects = [b['aspect_ratio'] for b in boxes]
        areas = [b['area'] for b in boxes]
        
        print(f"\n{cls}: {len(boxes)} samples")
        print(f"  Width:  mean={np.mean(widths):.3f}, std={np.std(widths):.3f}, "
              f"min={np.min(widths):.3f}, max={np.max(widths):.3f}")
        print(f"  Height: mean={np.mean(heights):.3f}, std={np.std(heights):.3f}, "
              f"min={np.min(heights):.3f}, max={np.max(heights):.3f}")
        print(f"  Aspect: mean={np.mean(aspects):.2f}, std={np.std(aspects):.2f}, "
              f"min={np.min(aspects):.2f}, max={np.max(aspects):.2f}")
        print(f"  Area:   mean={np.mean(areas):.4f}, std={np.std(areas):.4f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{name} Object Size Analysis', fontsize=16)
    
    for idx, cls in enumerate(CLASSES):
        ax = axes[idx // 3, idx % 3]
        
        if cls not in objects_by_class or len(objects_by_class[cls]) == 0:
            ax.text(0.5, 0.5, f'{cls}\nNo samples', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue
        
        boxes = objects_by_class[cls]
        widths = [b['width'] for b in boxes]
        heights = [b['height'] for b in boxes]
        
        ax.scatter(widths, heights, alpha=0.5, s=10)
        ax.set_xlabel('Normalized Width')
        ax.set_ylabel('Normalized Height')
        ax.set_title(f'{cls} (n={len(boxes)})')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 0.5)
        
        # Add aspect ratio lines
        ax.plot([0, 0.5], [0, 0.5], 'r--', alpha=0.3, label='AR=1 (square)')
        ax.plot([0, 0.5], [0, 0.25], 'g--', alpha=0.3, label='AR=2 (wide)')
        ax.plot([0, 0.25], [0, 0.5], 'b--', alpha=0.3, label='AR=0.5 (tall)')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'object_size_analysis_{name.lower()}.png', dpi=150)
    print(f"\n✓ Saved visualization to object_size_analysis_{name.lower()}.png")
    
    return objects_by_class


def recommend_anchors(objects_by_class):
    """Recommend anchor scales based on object sizes"""
    print(f"\n{'='*70}")
    print("ANCHOR RECOMMENDATIONS")
    print(f"{'='*70}")
    
    # Collect all sizes
    all_areas = []
    all_aspects = []
    
    for cls, boxes in objects_by_class.items():
        for box in boxes:
            all_areas.append(box['area'])
            all_aspects.append(box['aspect_ratio'])
    
    # Compute scales (use sqrt of area)
    scales = [np.sqrt(area) for area in all_areas]
    
    percentiles = [10, 30, 50, 70, 90]
    scale_values = np.percentile(scales, percentiles)
    
    print("\nRecommended scales (based on object size percentiles):")
    for p, s in zip(percentiles, scale_values):
        print(f"  {p}th percentile: {s:.3f}")
    
    print("\nCurrent scales in DEFAULT_ANCHOR_CONFIG:")
    print("  feat1: 0.1   (small objects)")
    print("  feat2: 0.2   (medium objects)")
    print("  feat3: 0.375 (larger objects)")
    print("  feat4: 0.55  (large vehicles)")
    print("  feat5: 0.725 (very large objects)")
    
    # Aspect ratio analysis
    aspect_percentiles = [10, 25, 50, 75, 90]
    aspect_values = np.percentile(all_aspects, aspect_percentiles)
    
    print("\nAspect ratio distribution:")
    for p, a in zip(aspect_percentiles, aspect_values):
        print(f"  {p}th percentile: {a:.2f}")
    
    print("\nCurrent aspect ratios in DEFAULT_ANCHOR_CONFIG:")
    print("  [1, 2, 0.5] → square, 2× wide, 2× tall")
    print("  [1, 2, 0.5, 3, 1/3] → + 3× wide, 3× tall")


if __name__ == "__main__":
    print("Analyzing TUMTraf dataset object sizes...")
    print("This helps understand if anchor configurations are appropriate.")
    
    # Analyze RGB
    if label_dir_rgb.exists():
        rgb_objects = analyze_labels(label_dir_rgb, 640, 480, "RGB")
    else:
        print(f"\n⚠ RGB labels not found at {label_dir_rgb}")
        rgb_objects = {}
    
    # Analyze Event-based
    if label_dir_eb.exists():
        eb_objects = analyze_labels(label_dir_eb, 482, 442, "Event-Based")
    else:
        print(f"\n⚠ Event-based labels not found at {label_dir_eb}")
        eb_objects = {}
    
    # Recommendations
    if rgb_objects:
        recommend_anchors(rgb_objects)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
