"""Quick script to check actual image dimensions in preprocessed data"""
import torch
from pathlib import Path
from torchvision.io import decode_image

# Check RGB images
rgb_train_dir = Path("data/preprocessed/train/images/rgb")
eb_train_dir = Path("data/preprocessed/train/images/eb_transformed")

print("=" * 60)
print("CHECKING RGB IMAGES")
print("=" * 60)

rgb_samples = []
for group_folder in sorted(rgb_train_dir.iterdir())[:5]:  # Check first 5 groups
    if group_folder.is_dir():
        for img_file in sorted(group_folder.glob("*.jpg"))[:2]:  # 2 images per group
            img = decode_image(str(img_file))
            rgb_samples.append((str(img_file.name), img.shape))
            print(f"{group_folder.name}/{img_file.name}: {img.shape}")

print("\n" + "=" * 60)
print("CHECKING EVENT-BASED IMAGES")
print("=" * 60)

eb_samples = []
for group_folder in sorted(eb_train_dir.iterdir())[:5]:  # Check first 5 groups
    if group_folder.is_dir():
        for img_file in sorted(group_folder.glob("*.jpg"))[:2]:  # 2 images per group
            img = decode_image(str(img_file))
            eb_samples.append((str(img_file.name), img.shape))
            print(f"{group_folder.name}/{img_file.name}: {img.shape}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Check if all RGB same size
rgb_shapes = [s[1] for s in rgb_samples]
if len(set(rgb_shapes)) == 1:
    print(f"✓ All RGB images same size: {rgb_shapes[0]}")
else:
    print(f"✗ RGB images have different sizes: {set(rgb_shapes)}")

# Check if all EB same size
eb_shapes = [s[1] for s in eb_samples]
if len(set(eb_shapes)) == 1:
    print(f"✓ All EB images same size: {eb_shapes[0]}")
else:
    print(f"✗ EB images have different sizes: {set(eb_shapes)}")

print("\n" + "=" * 60)
print("SIMULATING MODEL FORWARD PASS")
print("=" * 60)

# Simulate what happens in VGG11_SSD with actual image sizes
if rgb_shapes:
    C, H, W = rgb_shapes[0]
    print(f"\nInput RGB: [{C}, {H}, {W}]")
    
    # VGG11 architecture from your models
    # Each block: conv -> relu -> (maybe conv) -> pool (stride 2)
    h, w = H, W
    
    # Initial layers before feat maps
    h, w = h // 2, w // 2  # block1 pool
    print(f"After block1 pool: [{h}, {w}]")
    h, w = h // 2, w // 2  # block2 pool
    print(f"After block2 pool: [{h}, {w}]")
    
    # Feature maps used for detection
    h, w = h // 2, w // 2  # block3 pool
    feat1_h, feat1_w = h, w
    print(f"feat1 (block3): [{feat1_h}, {feat1_w}]")
    
    h, w = h // 2, w // 2  # block4 pool
    feat2_h, feat2_w = h, w
    print(f"feat2 (block4): [{feat2_h}, {feat2_w}]")
    
    h, w = h // 2, w // 2  # block5 pool
    feat3_h, feat3_w = h, w
    print(f"feat3 (block5): [{feat3_h}, {feat3_w}]")
    
    # Extra layers - no pooling
    feat4_h, feat4_w = h, w
    print(f"feat4 (extra1): [{feat4_h}, {feat4_w}]")
    
    # Extra2 with stride 2
    h, w = h // 2, w // 2
    feat5_h, feat5_w = h, w
    print(f"feat5 (extra2): [{feat5_h}, {feat5_w}]")
    
    # Calculate total anchors
    anchors_per_cell = [4, 6, 6, 6, 4]  # From your config
    total_anchors = (
        feat1_h * feat1_w * anchors_per_cell[0] +
        feat2_h * feat2_w * anchors_per_cell[1] +
        feat3_h * feat3_w * anchors_per_cell[2] +
        feat4_h * feat4_w * anchors_per_cell[3] +
        feat5_h * feat5_w * anchors_per_cell[4]
    )
    
    print(f"\n{'='*60}")
    print(f"EXPECTED TOTAL ANCHORS FROM MODEL: {total_anchors}")
    print(f"{'='*60}")
    
    # What your DEFAULT_FEAT_SIZES produces
    default_feat_sizes = [(60, 55), (30, 27), (15, 13), (15, 13), (7, 6)]
    default_total = sum(h * w * n for (w, h), n in zip(default_feat_sizes, anchors_per_cell))
    print(f"DEFAULT_FEAT_SIZES produces: {default_total} anchors")
    print(f"\nMISMATCH: {total_anchors} != {default_total}")
