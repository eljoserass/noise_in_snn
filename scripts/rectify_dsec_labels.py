#!/usr/bin/env python3
"""
Convert DSEC left-camera detection tracks from distorted 640x480 space
to rectified left-frame space, and render montage validation images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml


CLASS_NAMES = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "bus",
    4: "truck",
    5: "bicycle",
    6: "motorcycle",
    7: "train",
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 165, 0),
    2: (255, 0, 0),
    3: (0, 0, 255),
    4: (128, 0, 128),
    5: (255, 255, 0),
    6: (0, 255, 255),
    7: (128, 128, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rectify DSEC distorted-space bounding boxes using calibration."
    )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=Path("data/dsec_sample/zurich_city_02_b"),
        help="DSEC sequence directory.",
    )
    parser.add_argument(
        "--output-tracks",
        type=Path,
        default=None,
        help="Output .npy path. Defaults to object_detections/left/tracks_rectified.npy",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help="Directory to store montage images. Defaults to object_detections/left/rectified_validation",
    )
    parser.add_argument(
        "--transform-method",
        type=str,
        default="auto",
        choices=["auto", "remap", "scale"],
        help=(
            "BBox transform from distorted->rectified. "
            "remap uses DSEC per-pixel mapping (recommended); "
            "auto chooses by feature-match reprojection error."
        ),
    )
    parser.add_argument(
        "--window-us",
        type=int,
        default=25_000,
        help="Temporal matching window (+/-) around each frame timestamp (used only with --time-mode window).",
    )
    parser.add_argument(
        "--time-mode",
        type=str,
        default="nearest",
        choices=["nearest", "window"],
        help="How to select boxes for validation rendering.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="How many rectified frames to render for validation.",
    )
    parser.add_argument(
        "--frames-per-montage",
        type=int,
        default=6,
        help="How many frames per montage image.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=640,
        help="Validation tile width.",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=360,
        help="Validation tile height.",
    )
    return parser.parse_args()


def _camera_matrix_3x3(camera_matrix_flat: list[float]) -> np.ndarray:
    fx, fy, cx, cy = camera_matrix_flat
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _compute_event_to_frame_remap(sequence_dir: Path) -> np.ndarray:
    """
    Compute mapping from left event/distorted pixels (640x480) to
    left frame/rectified pixels (1440x1080).

    This mirrors dsec-det's compute_remapping() logic and is the correct
    mapping space for DSEC object_detections/left/tracks.npy.
    """
    calib_path = sequence_dir / "calibration" / "cam_to_cam.yaml"
    rectify_map_path = sequence_dir / "events" / "left" / "rectify_map.h5"

    with open(calib_path, "r", encoding="utf-8") as f:
        calibration = yaml.safe_load(f)
    with h5py.File(rectify_map_path, "r") as f:
        rectify_map = f["rectify_map"][:]

    k_r0 = _camera_matrix_3x3(calibration["intrinsics"]["camRect0"]["camera_matrix"])
    k_r1 = _camera_matrix_3x3(calibration["intrinsics"]["camRect1"]["camera_matrix"])
    r_r0_0 = np.asarray(calibration["extrinsics"]["R_rect0"], dtype=np.float64)
    r_r1_1 = np.asarray(calibration["extrinsics"]["R_rect1"], dtype=np.float64)
    r_1_0 = np.asarray(calibration["extrinsics"]["T_10"], dtype=np.float64)[:3, :3]

    # rect. frame-left -> rect. event-left
    p_r0_r1 = k_r0 @ r_r0_0 @ r_1_0.T @ r_r1_1.T @ np.linalg.inv(k_r1)

    h, w = rectify_map.shape[:2]
    coords_hom = np.concatenate((rectify_map, np.ones((h, w, 1), dtype=np.float32)), axis=-1)
    remap = (np.linalg.inv(p_r0_r1) @ coords_hom[..., None]).squeeze()
    remap = remap[..., :2] / remap[..., -1:]
    return remap.astype(np.float32)


def _load_calibration(sequence_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calib_path = sequence_dir / "calibration" / "cam_to_cam.yaml"
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = yaml.safe_load(f)

    intr = calib["intrinsics"]
    extr = calib["extrinsics"]
    cam1 = intr["cam1"]
    cam_rect1 = intr["camRect1"]

    k_cam1 = _camera_matrix_3x3(cam1["camera_matrix"])
    d_cam1 = np.array(cam1["distortion_coeffs"], dtype=np.float64)
    k_rect1 = _camera_matrix_3x3(cam_rect1["camera_matrix"])
    r_rect1 = np.array(extr["R_rect1"], dtype=np.float64)
    return k_cam1, d_cam1, r_rect1, k_rect1


def _compute_scale_factors(sequence_dir: Path) -> tuple[float, float, tuple[int, int], tuple[int, int]]:
    distorted_img = cv2.imread(str(sequence_dir / "images" / "left" / "distorted" / "000000.png"))
    rectified_img = cv2.imread(str(sequence_dir / "images" / "left" / "rectified" / "000000.png"))
    if distorted_img is None or rectified_img is None:
        raise FileNotFoundError("Could not load sample distorted/rectified image pair.")

    dist_h, dist_w = distorted_img.shape[:2]
    rect_h, rect_w = rectified_img.shape[:2]
    sx = rect_w / dist_w
    sy = rect_h / dist_h
    return sx, sy, (dist_w, dist_h), (rect_w, rect_h)


def rectify_tracks(
    tracks: np.ndarray,
    remap_event_to_frame: np.ndarray,
    sx: float,
    sy: float,
) -> np.ndarray:
    x = tracks["x"].astype(np.float64)
    y = tracks["y"].astype(np.float64)
    w = tracks["w"].astype(np.float64)
    h = tracks["h"].astype(np.float64)

    corners_dist = np.stack(
        [
            np.stack([x, y], axis=1),
            np.stack([x + w, y], axis=1),
            np.stack([x, y + h], axis=1),
            np.stack([x + w, y + h], axis=1),
        ],
        axis=1,
    )  # (N, 4, 2)

    h_map, w_map = remap_event_to_frame.shape[:2]
    map_x = remap_event_to_frame[..., 0]
    map_y = remap_event_to_frame[..., 1]

    corners = corners_dist.reshape(-1, 2).astype(np.float32)
    xs = corners[:, 0]
    ys = corners[:, 1]
    valid = (xs >= 0.0) & (xs <= (w_map - 1)) & (ys >= 0.0) & (ys <= (h_map - 1))

    rect_x = np.full(xs.shape, np.nan, dtype=np.float32)
    rect_y = np.full(ys.shape, np.nan, dtype=np.float32)
    if np.any(valid):
        vx = xs[valid].reshape(-1, 1)
        vy = ys[valid].reshape(-1, 1)
        rect_x[valid] = cv2.remap(
            map_x,
            vx,
            vy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        ).reshape(-1)
        rect_y[valid] = cv2.remap(
            map_y,
            vx,
            vy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        ).reshape(-1)

    corners_rect = np.stack([rect_x, rect_y], axis=1).reshape(-1, 4, 2)

    # Fallback for any invalid samples: simple scaling avoids NaN boxes.
    if np.any(~np.isfinite(corners_rect)):
        fallback = corners_dist * np.array([sx, sy], dtype=np.float64)
        corners_rect = np.where(np.isfinite(corners_rect), corners_rect, fallback.astype(np.float32))

    min_xy = corners_rect.min(axis=1)
    max_xy = corners_rect.max(axis=1)

    out = tracks.copy()
    out["x"] = min_xy[:, 0].astype(np.float32)
    out["y"] = min_xy[:, 1].astype(np.float32)
    out["w"] = (max_xy[:, 0] - min_xy[:, 0]).astype(np.float32)
    out["h"] = (max_xy[:, 1] - min_xy[:, 1]).astype(np.float32)
    return out


def scale_tracks(tracks: np.ndarray, sx: float, sy: float) -> np.ndarray:
    out = tracks.copy()
    out["x"] = (tracks["x"].astype(np.float64) * sx).astype(np.float32)
    out["y"] = (tracks["y"].astype(np.float64) * sy).astype(np.float32)
    out["w"] = (tracks["w"].astype(np.float64) * sx).astype(np.float32)
    out["h"] = (tracks["h"].astype(np.float64) * sy).astype(np.float32)
    return out


def _map_points_remap(
    pts_dist: np.ndarray,
    remap_event_to_frame: np.ndarray,
) -> np.ndarray:
    h_map, w_map = remap_event_to_frame.shape[:2]
    map_x = remap_event_to_frame[..., 0]
    map_y = remap_event_to_frame[..., 1]

    pts = pts_dist.astype(np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]
    valid = (xs >= 0.0) & (xs <= (w_map - 1)) & (ys >= 0.0) & (ys <= (h_map - 1))

    out = np.full((pts.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        vx = xs[valid].reshape(-1, 1)
        vy = ys[valid].reshape(-1, 1)
        out[valid, 0] = cv2.remap(
            map_x,
            vx,
            vy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        ).reshape(-1)
        out[valid, 1] = cv2.remap(
            map_y,
            vx,
            vy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        ).reshape(-1)
    return out


def _estimate_transform_method(
    sequence_dir: Path,
    sx: float,
    sy: float,
    remap_event_to_frame: np.ndarray,
    num_frames: int = 6,
) -> tuple[str, dict[str, float]]:
    distorted_paths = sorted((sequence_dir / "images" / "left" / "distorted").glob("*.png"))[:num_frames]
    rectified_paths = sorted((sequence_dir / "images" / "left" / "rectified").glob("*.png"))[:num_frames]
    if not distorted_paths or not rectified_paths:
        return "remap", {"median_scale_err": float("inf"), "median_remap_err": float("inf"), "matches": 0}

    orb = cv2.ORB_create(nfeatures=2500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    all_scale_err = []
    all_remap_err = []
    match_count = 0

    for d_path, r_path in zip(distorted_paths, rectified_paths):
        img_d = cv2.imread(str(d_path), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(str(r_path), cv2.IMREAD_GRAYSCALE)
        if img_d is None or img_r is None:
            continue

        kp_d, des_d = orb.detectAndCompute(img_d, None)
        kp_r, des_r = orb.detectAndCompute(img_r, None)
        if des_d is None or des_r is None or len(kp_d) < 8 or len(kp_r) < 8:
            continue

        matches = matcher.match(des_d, des_r)
        if not matches:
            continue
        matches = sorted(matches, key=lambda m: m.distance)[:300]

        pts_d = np.array([kp_d[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts_r = np.array([kp_r[m.trainIdx].pt for m in matches], dtype=np.float32)

        pred_scale = pts_d * np.array([sx, sy], dtype=np.float32)
        pred_remap = _map_points_remap(pts_d, remap_event_to_frame).astype(np.float32)

        scale_err = np.linalg.norm(pred_scale - pts_r, axis=1)
        finite = np.isfinite(pred_remap).all(axis=1)
        if not np.any(finite):
            continue
        remap_err = np.linalg.norm(pred_remap[finite] - pts_r[finite], axis=1)
        all_scale_err.append(scale_err)
        all_remap_err.append(remap_err)
        match_count += len(matches)

    if match_count == 0 or not all_remap_err:
        return "remap", {"median_scale_err": float("inf"), "median_remap_err": float("inf"), "matches": 0}

    scale_all = np.concatenate(all_scale_err)
    remap_all = np.concatenate(all_remap_err)
    med_scale = float(np.median(scale_all))
    med_remap = float(np.median(remap_all))
    method = "scale" if med_scale < med_remap else "remap"
    return method, {"median_scale_err": med_scale, "median_remap_err": med_remap, "matches": match_count}


def draw_frame_boxes(
    image_bgr: np.ndarray,
    frame_tracks: np.ndarray,
    rect_w: int,
    rect_h: int,
    frame_idx: int,
) -> np.ndarray:
    canvas = image_bgr.copy()
    for tr in frame_tracks:
        x1 = int(np.floor(float(tr["x"])))
        y1 = int(np.floor(float(tr["y"])))
        x2 = int(np.ceil(float(tr["x"] + tr["w"])))
        y2 = int(np.ceil(float(tr["y"] + tr["h"])))

        # Skip fully out-of-frame boxes
        if x2 < 0 or y2 < 0 or x1 >= rect_w or y1 >= rect_h:
            continue

        x1 = max(0, min(x1, rect_w - 1))
        y1 = max(0, min(y1, rect_h - 1))
        x2 = max(0, min(x2, rect_w - 1))
        y2 = max(0, min(y2, rect_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cls = int(tr["class_id"])
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        label = CLASS_NAMES.get(cls, str(cls))
        cv2.putText(
            canvas,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"frame {frame_idx:06d} | boxes={len(frame_tracks)}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def make_montages(
    sequence_dir: Path,
    tracks_rectified: np.ndarray,
    out_dir: Path,
    window_us: int,
    time_mode: str,
    max_frames: int,
    frames_per_montage: int,
    tile_w: int,
    tile_h: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_path = sequence_dir / "images" / "timestamps.txt"
    timestamps = np.loadtxt(ts_path, dtype=np.int64)

    rectified_imgs = sorted((sequence_dir / "images" / "left" / "rectified").glob("*.png"))
    if not rectified_imgs:
        raise FileNotFoundError("No rectified images found.")

    frame_ids = [int(p.stem) for p in rectified_imgs[:max_frames]]
    rendered_tiles = []
    for frame_id in frame_ids:
        img_path = sequence_dir / "images" / "left" / "rectified" / f"{frame_id:06d}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        ts = timestamps[frame_id]
        if time_mode == "nearest":
            dt = np.abs(tracks_rectified["t"].astype(np.int64) - np.int64(ts))
            if len(dt) == 0:
                frame_tracks = tracks_rectified[:0]
            else:
                min_dt = dt.min()
                frame_tracks = tracks_rectified[dt == min_dt]
        else:
            mask = (tracks_rectified["t"] >= ts - window_us) & (tracks_rectified["t"] < ts + window_us)
            frame_tracks = tracks_rectified[mask]
        painted = draw_frame_boxes(img, frame_tracks, w, h, frame_id)
        tile = cv2.resize(painted, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        rendered_tiles.append(tile)

    if not rendered_tiles:
        return []

    montage_paths = []
    for chunk_idx in range(0, len(rendered_tiles), frames_per_montage):
        chunk = rendered_tiles[chunk_idx : chunk_idx + frames_per_montage]
        cols = min(3, len(chunk))
        rows = int(np.ceil(len(chunk) / cols))
        montage = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
        for i, tile in enumerate(chunk):
            r = i // cols
            c = i % cols
            montage[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile

        out_path = out_dir / f"rectified_bbox_montage_{chunk_idx // frames_per_montage:02d}.png"
        cv2.imwrite(str(out_path), montage)
        montage_paths.append(out_path)

    return montage_paths


def main() -> None:
    args = parse_args()
    seq_dir = args.sequence_dir
    tracks_path = seq_dir / "object_detections" / "left" / "tracks.npy"

    output_tracks = args.output_tracks
    if output_tracks is None:
        output_tracks = seq_dir / "object_detections" / "left" / "tracks_rectified.npy"

    validation_dir = args.validation_dir
    if validation_dir is None:
        validation_dir = seq_dir / "object_detections" / "left" / "rectified_validation"

    tracks = np.load(tracks_path)
    _ = _load_calibration(seq_dir)  # kept for sanity checks / future extensions
    remap_event_to_frame = _compute_event_to_frame_remap(seq_dir)
    sx, sy, dist_size, rect_size = _compute_scale_factors(seq_dir)

    method = args.transform_method
    method_metrics = None
    if method == "auto":
        method, method_metrics = _estimate_transform_method(
            sequence_dir=seq_dir,
            sx=sx,
            sy=sy,
            remap_event_to_frame=remap_event_to_frame,
        )

    if method == "scale":
        tracks_rectified = scale_tracks(tracks=tracks, sx=sx, sy=sy)
    else:
        tracks_rectified = rectify_tracks(
            tracks=tracks,
            remap_event_to_frame=remap_event_to_frame,
            sx=sx,
            sy=sy,
        )
    np.save(output_tracks, tracks_rectified)

    montage_paths = make_montages(
        sequence_dir=seq_dir,
        tracks_rectified=tracks_rectified,
        out_dir=validation_dir,
        window_us=args.window_us,
        time_mode=args.time_mode,
        max_frames=args.max_frames,
        frames_per_montage=args.frames_per_montage,
        tile_w=args.tile_width,
        tile_h=args.tile_height,
    )

    print("Rectification complete")
    print(f"  tracks in:   {tracks_path}")
    print(f"  tracks out:  {output_tracks}")
    print(f"  distorted size: {dist_size[0]}x{dist_size[1]}")
    print(f"  rectified size: {rect_size[0]}x{rect_size[1]}")
    print(f"  scale factors: sx={sx:.6f}, sy={sy:.6f}")
    print(f"  transform method: {method}")
    if method_metrics is not None:
        print(
            "  auto metrics: "
            f"median_scale_err={method_metrics['median_scale_err']:.2f}px, "
            f"median_remap_err={method_metrics['median_remap_err']:.2f}px, "
            f"matches={method_metrics['matches']}"
        )
    print(f"  validation time mode: {args.time_mode}")
    print(f"  validation dir: {validation_dir}")
    if montage_paths:
        print("  montage images:")
        for p in montage_paths:
            print(f"    - {p}")
    else:
        print("  montage images: none")


if __name__ == "__main__":
    main()
