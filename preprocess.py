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


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data frm TUMTraf Event dataset")
    parser.add_argument("--data-path", type=str, default="data/TUMTraf_Event_Dataset", help="Root path of the data.")
    parser.add_argument("--out-path", type=str, default="data/preprocessed", help="Output path for preprocessed data.")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite existing content in output path.")
    parser.add_argument("--rgb", action="store_true", help="Process only RGB split.")
    parser.add_argument("--eb", action="store_true", help="Process only EB transformed split.")
    parser.add_argument("--all", action="store_true", help="Process both RGB and EB transformed splits.")
    parser.add_argument("--split", type=str, default="train,val,test", help="Comma-separated list of splits to process.") # check that this cannot be empty
    parser.add_argument("--eb_roi_path", type=str, default="roi_eb_transformed.json", help="Path to EB ROI JSON file.")
    parser.add_argument("--n_frames", type=int, default=16, help="Number of frames per video group.")
    parser.add_argument("--max_time_diff", type=int, default=200, help="Maximum time difference (ms) between frames in a group.")
    return parser.parse_args()




def group_frames(frames, n_frames, max_time_diff=200) -> list[list[Path]]:
    """
    Group list of images in n_frames chunks, ensuring that the time difference between consecutive frames does not exceed max_time_diff.

    :param frames: List of frame paths
    :param n_frames: Number of frames per group
    :param max_time_diff: Maximum time difference (ms) between frames in a group

    :return: List of grouped frames
    """

    # group the frames on the decided set
    # split/images/<n_video>/<frameid>.png
    # spllit/labels/<n_video>/<frameid>.json
    # frameid is the filename, which is also the timestamp
    # n_video is simply an integer id, kinda autoincrement which basically is where we found the video to use

    # return 

    pass

def load_eb_roi(eb_roi_path):
    with open(eb_roi_path, 'r') as f:
        eb_roi = json.load(f)
    return eb_roi


def adjust_to_roi(preprocessed_path, roi):
    # go to path, maybe check it exists
    # go fo each group
    # adjust the eb_transformed images and labels to the roi
    # write back the adjusted images and labels

    pass


def write_grouped_data(grouped_frames, out_path):

    # write the grouped frames to the out_path

    # for the labels, create a new path, which subtitues the "images" with "lables"
    # in the same loop, create the directory as same as the image but in label.
    # copy the labels to the new directory, by mathinc {frame.stem}.json from the original labels path

    labels_out_path = out_path.parent / "labels" / out_path.name
    os.makedirs(labels_out_path, exist_ok=True)

    for group_id, group in enumerate(grouped_frames):
            group_dir = out_path / f"{group_id:04d}"
            labels_group_dir = labels_out_path / f"{group_id:04d}"

            os.makedirs(group_dir, exist_ok=True)
            os.makedirs(labels_group_dir, exist_ok=True)

            for frame in group:
                copy2(frame, group_dir / frame.name)

                label_src = frame.parent.parent / "labels" / (frame.stem + ".json")
                label_dst = labels_group_dir / label_src.name
                copy2(label_src, label_dst)
  
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
    # 
    # adjust to roi
    # go to the preprocessed path we just written,

    splits = args.split.split(',')
    if not splits or splits == ['']:
        splits = ['train', 'val', 'test']

    eb_roi = load_eb_roi(args.eb_roi_path)
    if args.rgb or args.all:
        for split in splits:
            print(f"Processing RGB split: {split}")
            # process rgb data
            rgb_data_path = Path(args.data_path) / split / "images" / "rgb"
            out_rgb_path = Path(args.out_path) / split / "images" / "rgb"
            os.makedirs(out_rgb_path, exist_ok=True)
            # load frames
            frames = sorted(rgb_data_path.glob("*.jpg"))
            grouped_frames = group_frames(frames, args.n_frames, args.max_time_diff)
            # copy grouped frames to out_rgb_path
            write_grouped_data(grouped_frames, out_rgb_path)
    
    if args.eb or args.all:
        for split in splits:
            print(f"Processing EB transformed split: {split}")
            # process eb_transformed data
            eb_data_path = Path(args.data_path) / split / "images" / "eb_transformed"
            out_eb_path = Path(args.out_path) / split / "images" / "eb_transformed"
            os.makedirs(out_eb_path, exist_ok=True)
            # load frames
            frames = sorted(eb_data_path.glob("*.jpg"))
            grouped_frames = group_frames(frames, args.n_frames, args.max_time_diff)
            # copy grouped frames to out_eb_path
            write_grouped_data(grouped_frames, out_eb_path)

            # adjust to roi
            adjust_to_roi(out_eb_path, eb_roi)

def main():
    args = parse_args()
    if not args.rewrite and Path(args.out_path).exists():
        print(f"Output path {args.out_path} already exists. Use --rewrite to overwrite.")
        return
    preprocess_data(args)


if __name__ == "__main__":
    main()