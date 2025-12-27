
# TODO  might delete later, keeping hidden here for now

from src.data.dataset import TUMTraf
from pathlib import Path
import os
from collections import Counter
from tqdm import tqdm

project_root = Path(os.path.dirname(os.path.abspath('__file__')))

def analyze_dataset(dataset, name):
    print(f"\n--- Analyzing {name} ---")
    print(f"Classes: {dataset.classes}")
    
    class_counts = Counter()
    empty_samples = 0
    total_samples = len(dataset)
    
    # Iterate through dataset
    # Note: This might be slow for large datasets as it loads every file
    # We access the internal structure directly to avoid loading images
    
    if dataset.by_group:
        # dataset.img_labels is dict of dicts: {group_id: {file_id: label_data}}
        iterator = dataset.img_labels.items()
    else:
        # dataset.img_labels is dict: {file_id: {'group': g, 'data': d}}
        iterator = dataset.img_labels.items()

    for key, item in tqdm(iterator, total=total_samples, desc=f"Scanning {name}"):
        
        # Extract label data depending on structure
        if dataset.by_group:
            # item is {file_id: label_data}
            frames_data = item.values()
        else:
            # item is {'group': g, 'data': d}
            frames_data = [item['data']]
            
        sample_has_objects = False
        
        for label_data in frames_data:
            # Navigate JSON structure
            # Structure: openlabel -> frames -> {frame_id} -> objects -> {obj_id} -> object_data -> name
            frames = label_data.get("openlabel", {}).get("frames", {})
            
            for frame_content in frames.values():
                objects = frame_content.get("objects", {})
                
                frame_has_valid_object = False
                
                for obj in objects.values():
                    object_data = obj.get("object_data", {})
                    obj_name = object_data.get("type") or obj.get("type")
                    
                    # Check if this object is in our allowed classes
                    if obj_name in dataset.classes:
                        class_counts[obj_name] += 1
                        frame_has_valid_object = True
                    # else: ignored class (e.g. MOTORCYCLE)
                
                if frame_has_valid_object:
                    sample_has_objects = True

        if not sample_has_objects:
            empty_samples += 1

    print(f"Total Samples: {total_samples}")
    print(f"Empty Samples (after filtering): {empty_samples} ({empty_samples/total_samples*100:.2f}%)")
    print("Class Distribution:")
    for cls, count in class_counts.most_common():
        print(f"  {cls}: {count}")
    
    return class_counts

data_dir_train = project_root / "data" / "preprocessed" / "train"
data_dir_val = project_root / "data" / "preprocessed" / "val"
data_dir_test_day = project_root / "data" / "preprocessed" / "test" / "day"
data_dir_test_night_with_light_off = project_root / "data" / "preprocessed" / "test" / "night_with_light_off"
data_dir_test_night_with_light_on = project_root / "data" / "preprocessed" / "test" / "night_with_light_on"

# Train datasets
rgb_train_data = str(data_dir_train / "images" / "rgb")
eb_train_data = str(data_dir_train / "images" / "eb_transformed")
rgb_train_label = str(data_dir_train / "OPENLabel_labels_rgb")
eb_train_label = str(data_dir_train / "OPENLabel_labels_eb")

rgb_train_dataset = TUMTraf(
    img_dir=rgb_train_data,
    label_dir=rgb_train_label,
    by_group=False,
)

eb_train_dataset = TUMTraf(
    img_dir=eb_train_data,
    label_dir=eb_train_label,
    by_group=False,
)

# Validation datasets
rgb_val_data = str(data_dir_val / "images" / "rgb")
eb_val_data = str(data_dir_val / "images" / "eb_transformed")
rgb_val_label = str(data_dir_val / "OPENLabel_labels_rgb")
eb_val_label = str(data_dir_val / "OPENLabel_labels_eb")

rgb_val_dataset = TUMTraf(
    img_dir=rgb_val_data,
    label_dir=rgb_val_label,
    by_group=False,
)

eb_val_dataset = TUMTraf(
    img_dir=eb_val_data,
    label_dir=eb_val_label,
    by_group=True,
)

# Test day datasets
rgb_test_day_data = str(data_dir_test_day / "images" / "rgb")
eb_test_day_data = str(data_dir_test_day / "images" / "eb_transformed")
rgb_test_day_label = str(data_dir_test_day / "OPENLabel_labels_fusion_gt_optimized_rgb")
eb_test_day_label = str(data_dir_test_day / "OPENLabel_labels_fusion_gt_optimized_eb")

rgb_test_day_dataset = TUMTraf(
    img_dir=rgb_test_day_data,
    label_dir=rgb_test_day_label,
    by_group=False,
)

eb_test_day_dataset = TUMTraf(
    img_dir=eb_test_day_data,
    label_dir=eb_test_day_label,
    by_group=True,
)

# Test night_with_light_off datasets
rgb_test_night_off_data = str(data_dir_test_night_with_light_off / "images" / "rgb")
eb_test_night_off_data = str(data_dir_test_night_with_light_off / "images" / "eb_transformed")
rgb_test_night_off_label = str(data_dir_test_night_with_light_off / "OPENLabel_labels_fusion_gt_optimized_rgb")
eb_test_night_off_label = str(data_dir_test_night_with_light_off / "OPENLabel_labels_fusion_gt_optimized_eb")

rgb_test_night_off_dataset = TUMTraf(
    img_dir=rgb_test_night_off_data,
    label_dir=rgb_test_night_off_label,
    by_group=False,
)

eb_test_night_off_dataset = TUMTraf(
    img_dir=eb_test_night_off_data,
    label_dir=eb_test_night_off_label,
    by_group=True,
)

# Test night_with_light_on datasets
rgb_test_night_on_data = str(data_dir_test_night_with_light_on / "images" / "rgb")
eb_test_night_on_data = str(data_dir_test_night_with_light_on / "images" / "eb_transformed")
rgb_test_night_on_label = str(data_dir_test_night_with_light_on / "OPENLabel_labels_fusion_gt_optimized_rgb")
eb_test_night_on_label = str(data_dir_test_night_with_light_on / "OPENLabel_labels_fusion_gt_optimized_eb")

rgb_test_night_on_dataset = TUMTraf(
    img_dir=rgb_test_night_on_data,
    label_dir=rgb_test_night_on_label,
    by_group=False,
)

eb_test_night_on_dataset = TUMTraf(
    img_dir=eb_test_night_on_data,
    label_dir=eb_test_night_on_label,
    by_group=True,
)

print("RGB train classes:", rgb_train_dataset.classes)
print("EB train classes:", eb_train_dataset.classes)
print("RGB val classes:", rgb_val_dataset.classes)
print("EB val classes:", eb_val_dataset.classes)
print("RGB test day classes:", rgb_test_day_dataset.classes)
print("EB test day classes:", eb_test_day_dataset.classes)
print("RGB test night_with_light_off classes:", rgb_test_night_off_dataset.classes)
print("EB test night_with_light_off classes:", eb_test_night_off_dataset.classes)
print("RGB test night_with_light_on classes:", rgb_test_night_on_dataset.classes)
print("EB test night_with_light_on classes:", eb_test_night_on_dataset.classes)

print("len rgb train dataset:", len(rgb_train_dataset))
print("len eb train dataset:", len(eb_train_dataset))
print("len rgb val dataset:", len(rgb_val_dataset))
print("len eb val dataset:", len(eb_val_dataset))
print("len rgb test day dataset:", len(rgb_test_day_dataset))
print("len eb test day dataset:", len(eb_test_day_dataset))
print("len rgb test night_with_light_off dataset:", len(rgb_test_night_off_dataset))
print("len eb test night_with_light_off dataset:", len(eb_test_night_off_dataset))
print("len rgb test night_with_light_on dataset:", len(rgb_test_night_on_dataset))
print("len eb test night_with_light_on dataset:", len(eb_test_night_on_dataset))





analyze_dataset(rgb_train_dataset, "RGB Train")
analyze_dataset(eb_train_dataset, "EB Train")
analyze_dataset(rgb_val_dataset, "RGB Val")
analyze_dataset(eb_val_dataset, "EB Val")
analyze_dataset(rgb_test_day_dataset, "RGB Test Day")
analyze_dataset(eb_test_day_dataset, "EB Test Day")
analyze_dataset(rgb_test_night_off_dataset, "RGB Test Night with Light Off")
analyze_dataset(eb_test_night_off_dataset, "EB Test Night with Light Off")
analyze_dataset(rgb_test_night_on_dataset, "RGB Test Night with Light On")
analyze_dataset(eb_test_night_on_dataset, "EB Test Night with Light On")