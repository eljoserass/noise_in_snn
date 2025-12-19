import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from shutil import copy2
from datetime import datetime
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data frm TUMTraf Event dataset")
    parser.add_argument("--data-path", type=str, default="data/TUMTraf_Event_Dataset", help="Root path of the data.")
    parser.add_argument("--out-path", type=str, default="data/preprocessed", help="Output path for preprocessed data.")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite existing content in output path.")
    parser.add_argument("--rgb", action="store_true", help="Process only RGB split.")
    parser.add_argument("--eb", action="store_true", help="Process only EB transformed split.")
    parser.add_argument("--all", action="store_true", help="Process both RGB and EB transformed splits.")
    parser.add_argument("--split", type=str, default="train,val,test/day,test/night_with_light_off,test/night_with_light_on", help="Comma-separated list of splits to process.") # check that this cannot be empty
    parser.add_argument("--eb_roi_path", type=str, default="data/TUMTraf_Event_Dataset/calibration/intrinsic/eb_8mm_roi.txt", help="Path to EB ROI JSON file.")
    parser.add_argument("--n_frames", type=int, default=8, help="Number of frames per video group.")
    parser.add_argument("--max_time_diff", type=int, default=1000, help="Maximum time difference (ms) between frames in a group.")
    return parser.parse_args()



def group_frames(src_image_path:Path, dest_image_path:Path, 
                 src_label_path:Path, dest_label_path:Path, 
                 n_frames:int, max_time_diff:int=1000) -> None:
    """
    Group images and labels from src_path into dest_path in n_frames chunks, ensuring that the time difference between consecutive frames does not exceed max_time_diff.
    :param src_image_path: Path to the directory containing frame images
    :param dest_image_path: Path to the directory where grouped frames will be written
    :param src_label_path: Path to the directory containing frame labels
    :param out_label_path: Path to the directory where grouped labels will be written
    :param n_frames: Number of frames per group
    :param max_time_diff: Maximum time difference (ms) between frames in a group

    :return: None, it writes direclty all the data
    """
    
    frame_files = sorted(src_image_path.glob("*.jpg"))
    print(f"Found {len(frame_files)} frames in {src_image_path}")
    grouped_frames = []
    current_group = []
    last_timestamp = None
    for frame_file in frame_files:
        timestamp = int(datetime.strptime(frame_file.stem, "%Y%m%d-%H%M%S.%f").timestamp() * 1000)
        # ValueError: invalid literal for int() with base 10: '20231114-084328.739529
        if last_timestamp is None:
            current_group.append(frame_file)
        else:
            time_diff = timestamp - last_timestamp
            if time_diff <= max_time_diff and len(current_group) < n_frames:
                current_group.append(frame_file)
            else:
                if len(current_group) == n_frames:
                    grouped_frames.append(current_group)
                current_group = [frame_file]
        last_timestamp = timestamp
    if len(current_group) == n_frames:
        grouped_frames.append(current_group)
    # now write the grouped frames and labels
    for group_id, group in enumerate(grouped_frames):
        group_dir = dest_image_path / f"{group_id:04d}"
        os.makedirs(group_dir, exist_ok=True)

        label_group_dir = dest_label_path / f"{group_id:04d}"
        os.makedirs(label_group_dir, exist_ok=True)

        for frame in group:
            copy2(frame, group_dir / frame.name)
            # copy the corresponding label
            label_file = src_label_path / f"{frame.stem}.json"
            if label_file.exists():
                copy2(label_file, label_group_dir / label_file.name)



def load_eb_roi(eb_roi_path):


    """Load EB ROI from .txt, returns dict with x,y,width,height"""
    # with open(eb_roi_path, 'r') as f:
    #     lines = f.readlines()
    #     x = int(float(lines[0].strip()))
    #     y = int(float(lines[1].strip()))
    #     width = int(float(lines[2].strip()))
    #     height = int(float(lines[3].strip()))
    # return {"x": x, "y": y, "width": width, "height": height}

    # TODO to check later, the roi in the text is suspiciously off by one 0
    return {"x": 130, "width": 612, "y": 9, "height": 451 }


import time
def apply_roi(image_path:Path, label_path:Path, roi:dict) -> None:
    """
    Applies the ROI transformation to labels and grouped images
    :param image_path: path to gruped frames
    :param label_path: path to labels
    :param roi: ROI coordinates (x, y, width, height)

    :return: None, writes directly to disk
    """

    frame_files = sorted(image_path.glob("*/*.jpg"))
    print (f"Found {len(frame_files)} frame files to process.")
    label_data = None
    for frame_file in frame_files:
        # load image
        img = cv2.imread(str(frame_file))
        # crop
        x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
        cropped_img = img[y:h, x:w]
        # overwrite
        # remove the file before writing
        if frame_file.exists():
            frame_file.unlink()
        status = cv2.imwrite(str(frame_file), cropped_img)
        print (f"Processed frame file: {str(frame_file)}, write status: {status}")

    # process labels
    label_files = sorted(label_path.glob("*/*.json"))
    print (f"Found {len(label_files)} label files to process.")    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            label_data = json.load(f)

        # example dict {'openlabel': {'metadata': {'schema_version': '1.0.0'}, 'coordinate_systems': 169, 'frames': {'169': {'objects': {'0': {'object_data': {'name': 'PEDESTRIAN_0', 'type': 'PEDESTRIAN', 'bbox': [{'name': 'full_bbox', 'val': [568, 167, 38, 58], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '1': {'object_data': {'name': 'PEDESTRIAN_1', 'type': 'PEDESTRIAN', 'bbox': [{'name': 'full_bbox', 'val': [574, 241, 46, 87], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '2': {'object_data': {'name': 'CAR_2', 'type': 'CAR', 'bbox': [{'name': 'full_bbox', 'val': [258, 24, 46, 39], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '3': {'object_data': {'name': 'TRUCK_3', 'type': 'TRUCK', 'bbox': [{'name': 'full_bbox', 'val': [309, 137, 81, 135], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '4': {'object_data': {'name': 'CAR_4', 'type': 'CAR', 'bbox': [{'name': 'full_bbox', 'val': [392, 22, 34, 36], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '5': {'object_data': {'name': 'TRAILER_5', 'type': 'TRAILER', 'bbox': [{'name': 'full_bbox', 'val': [347, 28, 48, 56], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '6': {'object_data': {'name': 'CAR_6', 'type': 'CAR', 'bbox': [{'name': 'full_bbox', 'val': [262, 33, 45, 43], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '7': {'object_data': {'name': 'CAR_7', 'type': 'CAR', 'bbox': [{'name': 'full_bbox', 'val': [375, 40, 41, 43], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '8': {'object_data': {'name': 'TRAILER_8', 'type': 'TRAILER', 'bbox': [{'name': 'full_bbox', 'val': [126, 107, 27, 166], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '9': {'object_data': {'name': 'CAR_9', 'type': 'CAR', 'bbox': [{'name': 'full_bbox', 'val': [195, 49, 56, 49], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}}}}}}
        for frame_id, frame_content in label_data.get("openlabel", {}).get("frames", {}).items():
            for obj_id, obj_content in frame_content.get("objects", {}).items():
                bbox_list = obj_content.get("object_data", {}).get("bbox", [])
                for bbox in bbox_list:
                    if bbox.get("name") == "full_bbox":
                        x_val, y_val, box_w, box_h = bbox.get("val", [0,0,0,0])
                        # adjust coordinates
                        x_val -= roi['x']
                        y_val -= roi['y']
                        # update bbox
                        bbox["val"] = [x_val, y_val, box_w, box_h]

        with open(label_file, 'w') as f:
            json.dump(label_data, f)


def preprocess_data(args):
    splits = args.split.split(',')

    if args.eb or args.all:
        # load roi
        eb_roi = load_eb_roi(args.eb_roi_path)
        print(f"Loaded EB ROI: {eb_roi}")

    for split in splits:
        print (f"Processing split: {split}")
        out_split_path = Path(args.out_path) / split
        os.makedirs(out_split_path, exist_ok=True)

        if args.rgb or args.all:
            # get images path
            src_rgb_image_path = Path(args.data_path) / split / "images" / "rgb"
            out_rgb_image_path = Path(args.out_path) / split / "images" / "rgb"
            # get labels path
            if split.startswith("test/"):
                src_rgb_labels_path = Path(args.data_path) / split / "OPENLabel_labels_fusion_gt_optimized_rgb"
                dest_rgb_labels_path = Path(args.out_path) / split / "OPENLabel_labels_fusion_gt_optimized_rgb"
            else:
                src_rgb_labels_path = Path(args.data_path) / split / "OPENLabel_labels_rgb"
                dest_rgb_labels_path = Path(args.out_path) / split / "OPENLabel_labels_rgb"
            
            group_frames(src_rgb_image_path, out_rgb_image_path, 
                         src_rgb_labels_path, dest_rgb_labels_path,
                         args.n_frames, args.max_time_diff)

    
        if args.eb or args.all:
    

            # get images path
            src_eb_image_path = Path(args.data_path) / split / "images" / "eb_transformed"
            out_eb_image_path = Path(args.out_path) / split / "images" / "eb_transformed"

            # get labels path
            if split.startswith("test/"):
                src_eb_labels_path = Path(args.data_path) / split / "OPENLabel_labels_fusion_gt_optimized_eb"
                dest_eb_labels_path = Path(args.out_path) / split / "OPENLabel_labels_fusion_gt_optimized_eb"
            else:
                src_eb_labels_path = Path(args.data_path) / split / "OPENLabel_labels_eb"
                dest_eb_labels_path = Path(args.out_path) / split / "OPENLabel_labels_eb"


            group_frames(src_eb_image_path, out_eb_image_path, 
                         src_eb_labels_path, dest_eb_labels_path,
                         args.n_frames, args.max_time_diff)
            
            apply_roi(out_eb_image_path, dest_eb_labels_path, eb_roi)
            

def main():
    args = parse_args()
    if not args.rewrite and Path(args.out_path).exists():
        print(f"Output path {args.out_path} already exists. Use --rewrite to overwrite.")
        return
    preprocess_data(args)


if __name__ == "__main__":
    main()