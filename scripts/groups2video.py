import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

TYPE_COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    "CAR": (0, 255, 0),
    "TRUCK": (0, 0, 255),
    "TRAILER": (255, 0, 0),
    "BUS": (255, 255, 0),
    "MOTORCYCLE": (255, 0, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create videos from grouped frame folders.")
    parser.add_argument("--input-path", type=str, default="data/preprocessed", help="Root directory containing grouped frames.")
    parser.add_argument("--output-path", type=str, default="data/videos", help="Root directory to write generated videos.")
    parser.add_argument("--rewrite", action="store_true", help="Overwrite existing videos.")
    parser.add_argument("--rgb", action="store_true", help="Process RGB groups only.")
    parser.add_argument("--eb", action="store_true", help="Process event-based groups only.")
    parser.add_argument("--all", action="store_true", help="Process both RGB and event-based groups.")
    parser.add_argument("--split", type=str, default="train,val,test/day,test/night_with_light_off,test/night_with_light_on", help="Comma-separated list of splits to include.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the generated videos.")
    parser.add_argument("--draw-bboxes", action="store_true", help="Overlay bounding boxes on top of each frame.")
    parser.add_argument("--bbox-color", type=str, default="0,255,0", help="Comma-separated BGR color to use as fallback when drawing boxes.")
    parser.add_argument("--bbox-thickness", type=int, default=2, help="Rectangle thickness when drawing boxes.")
    return parser.parse_args()


def parse_color(color_str: str) -> Tuple[int, int, int]:
    parts = [segment.strip() for segment in color_str.split(",")]
    if len(parts) != 3:
        raise ValueError("bbox-color must contain three comma-separated integers.")
    values: List[int] = []
    for part in parts:
        value = int(part)
        values.append(max(0, min(255, value)))
    return values[0], values[1], values[2]


def split_path(root: Path, split: str) -> Path:
    target = root
    for chunk in split.split("/"):
        if chunk:
            target = target / chunk
    return target


def camera_image_dir(split_root: Path, camera: str) -> Path:
    folder = "rgb" if camera == "rgb" else "eb_transformed"
    return split_root / "images" / folder


def camera_label_dir(split_root: Path, split: str, camera: str) -> Path:
    if camera == "rgb":
        label_folder = "OPENLabel_labels_fusion_gt_optimized_rgb" if split.startswith("test/") else "OPENLabel_labels_rgb"
    else:
        label_folder = "OPENLabel_labels_fusion_gt_optimized_eb" if split.startswith("test/") else "OPENLabel_labels_eb"
    return split_root / label_folder


def ensure_output_dir(output_root: Path, split: str, camera: str) -> Path:
    base = split_path(output_root, split)
    camera_dir = "rgb" if camera == "rgb" else "eb"
    target = base / camera_dir
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_openlabel_root(payload: Dict) -> Optional[Dict]:
    if "openlabel" in payload:
        return payload.get("openlabel")
    data_block = payload.get("data")
    if isinstance(data_block, dict):
        return data_block.get("openlabel")
    return None


def iter_frame_objects(openlabel_root: Dict) -> Iterable[Dict]:
    frames = openlabel_root.get("frames", {})
    for frame_key in sorted(frames.keys(), key=lambda item: int(item) if item.isdigit() else item):
        frame = frames[frame_key]
        objects = frame.get("objects", {})
        for obj in objects.values():
            yield obj


BoundingBox = Tuple[int, int, int, int, str, Optional[str]]


def extract_bboxes(label_file: Path) -> List[BoundingBox]:
    if not label_file.exists():
        return []
    with label_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    openlabel_root = load_openlabel_root(payload)
    if not openlabel_root:
        return []
    boxes: List[BoundingBox] = []
    for obj in iter_frame_objects(openlabel_root):
        object_data = obj.get("object_data", {})
        obj_name = object_data.get("name") or obj.get("name") or "object"
        obj_type = object_data.get("type") or obj.get("type")
        bbox_entries = object_data.get("bbox", [])
        if not bbox_entries:
            continue
        full_bbox = next((entry for entry in bbox_entries if entry.get("name") == "full_bbox"), bbox_entries[0])
        vals = full_bbox.get("val", [])
        if len(vals) != 4:
            continue
        x_center, y_center, width, height = [float(val) for val in vals]
        x_min = int(round(x_center - width / 2.0))
        y_min = int(round(y_center - height / 2.0))
        x_max = int(round(x_center + width / 2.0))
        y_max = int(round(y_center + height / 2.0))
        label_text = f"{obj_name} ({obj_type})" if obj_type else obj_name
        boxes.append((x_min, y_min, x_max, y_max, label_text, obj_type))
    return boxes


def draw_bboxes(frame, boxes: List[BoundingBox], default_color: Tuple[int, int, int], thickness: int) -> None:
    height, width = frame.shape[:2]
    for x1, y1, x2, y2, label_text, obj_type in boxes:
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        color = TYPE_COLOR_MAP.get(obj_type, default_color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        text = label_text
        text_scale = 0.5
        text_thickness = max(1, thickness - 1)
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        text_height = text_size[1] + baseline
        text_y = y1 - 4
        if text_y < text_height:
            text_y = y1 + text_height + 4
            if text_y > height:
                text_y = height - 4
        text_x = max(0, min(width - text_size[0], x1))
        cv2.putText(frame, text, (text_x, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, text_thickness, cv2.LINE_AA)


def build_video(
    group_dir: Path,
    label_group_dir: Path,
    video_path: Path,
    fps: int,
    draw_boxes: bool,
    default_color: Tuple[int, int, int],
    thickness: int,
) -> None:
    frames = sorted(group_dir.glob("*.jpg"))
    if not frames:
        print(f"No frames found in {group_dir}, skipping.")
        return
    sample = cv2.imread(str(frames[0]))
    if sample is None:
        print(f"Cannot read sample frame {frames[0]}, skipping.")
        return
    height, width = sample.shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print(f"Failed to open VideoWriter for {video_path}.")
        return
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Failed to read frame {frame_path}, skipping frame.")
            continue
        if draw_boxes:
            label_file = label_group_dir / f"{frame_path.stem}.json"
            boxes = extract_bboxes(label_file)
            if boxes:
                draw_bboxes(frame, boxes, default_color, thickness)
        writer.write(frame)
    writer.release()


def select_cameras(args: argparse.Namespace) -> List[str]:
    if args.all or (not args.rgb and not args.eb):
        return ["rgb", "eb"]
    selected: List[str] = []
    if args.rgb:
        selected.append("rgb")
    if args.eb:
        selected.append("eb")
    return selected


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_path)
    output_root = Path(args.output_path)
    if not input_root.exists():
        raise FileNotFoundError(f"Input path {input_root} does not exist.")
    cameras = select_cameras(args)
    fallback_color: Optional[Tuple[int, int, int]] = parse_color(args.bbox_color) if args.draw_bboxes else None
    splits = [entry.strip() for entry in args.split.split(",") if entry.strip()]
    for split in splits:
        split_root = split_path(input_root, split)
        if not split_root.exists():
            print(f"Split {split} not found at {split_root}, skipping.")
            continue
        for camera in cameras:
            image_dir = camera_image_dir(split_root, camera)
            if not image_dir.exists():
                print(f"No image directory at {image_dir}, skipping.")
                continue
            label_dir = camera_label_dir(split_root, split, camera)
            if not label_dir.exists() and args.draw_bboxes:
                print(f"Label directory {label_dir} missing; videos will be generated without boxes.")
            output_dir = ensure_output_dir(output_root, split, camera)
            group_dirs = [entry for entry in sorted(image_dir.iterdir()) if entry.is_dir()]
            for group_dir in group_dirs:
                video_path = output_dir / f"{group_dir.name}.mp4"
                if video_path.exists() and not args.rewrite:
                    print(f"Video {video_path} already exists, skipping.")
                    continue
                label_group_dir = label_dir / group_dir.name if label_dir.exists() else Path()
                build_video(
                    group_dir,
                    label_group_dir,
                    video_path,
                    args.fps,
                    args.draw_bboxes and label_dir.exists(),
                    fallback_color if fallback_color else (0, 255, 0),
                    args.bbox_thickness,
                )
                print(f"Wrote {video_path}")


if __name__ == "__main__":
    main()