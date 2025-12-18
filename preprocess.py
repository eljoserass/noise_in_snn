# this will prepare the rgb and eb data. 
# it will use read data from train/test/val
# it will strip the lines that are empty from eb_transformed
# it will write write the labels for eb_transformed with the labels adjusted is probably easier
# it will group the frames into little videos of 1-2 s.
#   5-7 fps = 8-16 frames per video.
#   checking ms difference between frames, if its to large and we couldnt group. drop and try to group again
#   we will write it in a json or in folders idk


#params
# --data-path <path>- where the root is - has default
# --out-path <path> - where to write the data output, if it exists something dont rewrite - has default
# --rewrite - rewrite content ignore if it exists - defaults to false
# --rgb - do it only on rgb split - defaults to false
# --eb - do it only on eb_transformed split - defaults to false
# --split <split,splita,...> - do it only in the specified split - defaults to train/val/test
# --eb_roi_path - where the json is - defaults to roi_eb_transformed.json
# --n_frames - number of frames per video -  defaults to 16

# returns
# if main
# will dump a folder with similar structure,
#  but each grup of frames in a different folder, and eb_transformed with new labels and image dimensions

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from shutil import copy2
from datetime import datetime
import cv2


# data/TUMTraf_Event_Dataset/train/images/eb_transformed/20231114-085428.179411.jpg
# data/TUMTraf_Event_Dataset/train/images/eb_transformed/20231114-085428.375407.jpg
# data/TUMTraf_Event_Dataset/train/images/eb_transformed/20231114-085428.584928.jpg


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data frm TUMTraf Event dataset")
    parser.add_argument("--data-path", type=str, default="data/TUMTraf_Event_Dataset", help="Root path of the data.")
    parser.add_argument("--out-path", type=str, default="data/preprocessed", help="Output path for preprocessed data.")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite existing content in output path.")
    parser.add_argument("--rgb", action="store_true", help="Process only RGB split.")
    parser.add_argument("--eb", action="store_true", help="Process only EB transformed split.")
    parser.add_argument("--all", action="store_true", help="Process both RGB and EB transformed splits.")
    parser.add_argument("--split", type=str, default="train,val,test/day,test/night_with_light_off,test/night_with_light_on", help="Comma-separated list of splits to process.") # check that this cannot be empty
    parser.add_argument("--eb_roi_path", type=str, default="data/roi_eb_transformed.json", help="Path to EB ROI JSON file.")
    parser.add_argument("--n_frames", type=int, default=8, help="Number of frames per video group.")
    parser.add_argument("--max_time_diff", type=int, default=1000, help="Maximum time difference (ms) between frames in a group.")
    return parser.parse_args()



def group_frames(src_image_path:Path, out_image_path:Path, 
                 src_label_path:Path, out_label_path:Path, 
                 n_frames:int, max_time_diff:int=1000) -> None:
    """
    Group images and labels from src_path into out_path in n_frames chunks, ensuring that the time difference between consecutive frames does not exceed max_time_diff.

    :param src_image_path: Path to the directory containing frame images
    :param out_image_path: Path to the directory where grouped frames will be written
    :param src_label_path: Path to the directory containing frame labels
    :param out_label_path: Path to the directory where grouped labels will be written
    :param n_frames: Number of frames per group
    :param max_time_diff: Maximum time difference (ms) between frames in a group

    :return: None, it writes direclty all the data
    """

    # group the frames on the decided set
    # split/images/<n_video>/<frameid>.jpg
    # spllit/labels/<n_video>/<frameid>.json
    # frameid is the filename, which is also the timestamp
    # n_video is simply an integer id, kinda autoincrement which basically is where we found the video to use

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

    # now write the grouped frames and labels
    for group_id, group in enumerate(grouped_frames):
        group_dir = out_image_path / f"{group_id:04d}"
        os.makedirs(group_dir, exist_ok=True)

        label_group_dir = out_label_path / f"{group_id:04d}"
        os.makedirs(label_group_dir, exist_ok=True)

        for frame in group:
            copy2(frame, group_dir / frame.name)
            # copy the corresponding label
            label_file = src_label_path / f"{frame.stem}.json"
            if label_file.exists():
                copy2(label_file, label_group_dir / label_file.name)



def load_eb_roi(eb_roi_path):
    """Load EB ROI from .txt, returns dict with x,y,width,height"""
    with open(eb_roi_path, 'r') as f:
        lines = f.readlines()
        y = int(float(lines[0].strip()))
        x = int(float(lines[1].strip()))
        width = int(float(lines[2].strip()))
        height = int(float(lines[3].strip()))
    return {"x": x, "y": y, "width": width, "height": height}


def apply_roi(image_path:Path, label_path:Path, roi:dict):
    """
    Applies the ROI transformation to labels and grouped images
    :param image_path: path to gruped frames
    :param label_path: path to labels
    :param roi: ROI coordinates (x, y, width, height)

    """

#     # for the labels, create a new path, which subtitues the "images" with "lables"
#     # in the same loop, create the directory as same as the image but in label.
#     # copy the labels to the new directory, by mathinc {frame.stem}.json from the original labels path


#     for group_id, group in enumerate(grouped_frames):
#             group_dir = out_path / f"{group_id:04d}"

#             os.makedirs(group_dir, exist_ok=True)
#             for frame in group:
#                 copy2(frame, group_dir / frame.name)



def preprocess_data(args):
    # this will filter args, invoke pipeline steps with said args.
    # filter args. main decision points are wether to do it only on eb, rgb or both
    # another point is wether to do in train,test,val or all of them
    # 
    # group the frames on the decided set
    # split/images/<n_video>/<frameid>.png
    # spllit/labels/<n_video>/<frameid>.json
    # frameid is the filename, which is also the timestamp
    # n_video is simply an integer id, kinda autoincrement which basically is where we found the video to use
    # remember test has this split 
    # day/  night_with_light_off/  night_with_light_on/
    # 
    # adjust to roi
    # go to the preprocessed path we just written,

    splits = args.split.split(',')

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
                out_src_rgb_labels_path = Path(args.out_path) / split / "OPENLabel_labels_fusion_gt_optimized_rgb"
            else:
                src_rgb_labels_path = Path(args.data_path) / split / "OPENLabel_labels_rgb"
                out_src_rgb_labels_path = Path(args.out_path) / split / "OPENLabel_labels_rgb"
            
            group_frames(src_rgb_image_path, out_rgb_image_path, 
                         src_rgb_labels_path, out_src_rgb_labels_path,
                         args.n_frames, args.max_time_diff)

    
        if args.eb or args.all:
            # eb_roi = load_eb_roi(args.eb_roi_path)

            # get images path
            src_eb_image_path = Path(args.data_path) / split / "images" / "eb_transformed"
            out_eb_image_path = Path(args.out_path) / split / "images" / "eb_transformed"

            # get labels path
            if split.startswith("test/"):
                src_eb_labels_path = Path(args.data_path) / split / "OPENLabel_labels_fusion_gt_optimized_eb"
                out_src_eb_labels_path = Path(args.out_path) / split / "OPENLabel_labels_fusion_gt_optimized_eb"
            else:
                src_eb_labels_path = Path(args.data_path) / split / "OPENLabel_labels_eb"
                out_src_eb_labels_path = Path(args.out_path) / split / "OPENLabel_labels_eb"


            group_frames(src_eb_image_path, out_eb_image_path, 
                         src_eb_labels_path, out_src_eb_labels_path,
                         args.n_frames, args.max_time_diff)

def main():
    args = parse_args()
    if not args.rewrite and Path(args.out_path).exists():
        print(f"Output path {args.out_path} already exists. Use --rewrite to overwrite.")
        return
    preprocess_data(args)


if __name__ == "__main__":
    main()