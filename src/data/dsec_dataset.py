"""
DSEC dataset loaders for ANN and SNN SSD training.

Expected directory layout (per sequence):
  <dsec_root>/<split>/<sequence>/
    images/timestamps.txt
    images/left/distorted/*.png
    object_detections/left/tracks.npy
    events/left/events.h5                         (real events)
    v2e_output_*/dvs_events.txt                  (simulated events, optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency for real events
    h5py = None


DSEC_CLASS_NAMES = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "bus",
    4: "truck",
    5: "bicycle",
    6: "motorcycle",
    7: "train",
}


def parse_class_ids(class_ids: str | list[int] | None) -> list[int]:
    if class_ids is None:
        return sorted(DSEC_CLASS_NAMES.keys())
    if isinstance(class_ids, str):
        parsed = [int(x.strip()) for x in class_ids.split(",") if x.strip()]
    else:
        parsed = [int(x) for x in class_ids]
    unique = sorted(set(parsed))
    if not unique:
        raise ValueError("class_ids resolved to empty list")
    return unique


def _to_int_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 0


def _sorted_frame_paths(frame_dir: Path) -> list[Path]:
    pngs = sorted(frame_dir.glob("*.png"), key=_to_int_stem)
    if pngs:
        return pngs
    jpgs = sorted(frame_dir.glob("*.jpg"), key=_to_int_stem)
    return jpgs


def _load_timestamps(ts_path: Path, count: int) -> np.ndarray:
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing timestamps file: {ts_path}")
    ts = np.loadtxt(ts_path, dtype=np.int64)
    if ts.ndim == 0:
        ts = np.asarray([np.int64(ts)], dtype=np.int64)
    if ts.shape[0] < count:
        raise ValueError(f"Timestamps shorter than frames: {ts.shape[0]} < {count} ({ts_path})")
    return ts[:count]


def _resolve_tracks_path(sequence_dir: Path, preferred_relpath: str) -> Path:
    preferred = sequence_dir / preferred_relpath
    if preferred.exists():
        return preferred

    candidates = [
        sequence_dir / "object_detections/left/tracks_rectified.npy",
        sequence_dir / "object_detections/left/tracks_rectified_remap_check.npy",
        sequence_dir / "object_detections/left/tracks.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No tracks file found in {sequence_dir}. Tried: {preferred_relpath}, tracks_rectified.npy, "
        "tracks_rectified_remap_check.npy, tracks.npy"
    )


def _load_tracks_for_classes(tracks_path: Path, class_ids: list[int]) -> np.ndarray:
    tracks = np.load(tracks_path)
    if tracks.dtype.names is None:
        raise ValueError(f"Tracks file is not structured array: {tracks_path}")
    required_fields = {"t", "x", "y", "w", "h", "class_id"}
    if not required_fields.issubset(set(tracks.dtype.names)):
        raise ValueError(f"Tracks file missing required fields {required_fields}: {tracks_path}")

    order = np.argsort(tracks["t"].astype(np.int64))
    tracks = tracks[order]

    class_mask = np.isin(tracks["class_id"].astype(np.int64), np.asarray(class_ids, dtype=np.int64))
    return tracks[class_mask]


def _build_frame_track_ranges_nearest(
    track_ts: np.ndarray,
    frame_ts: np.ndarray,
    max_time_delta_us: int | None,
) -> list[tuple[int, int]]:
    if track_ts.size == 0:
        return [(0, 0) for _ in range(frame_ts.shape[0])]

    unique_ts, first_idx, counts = np.unique(track_ts, return_index=True, return_counts=True)
    ranges: list[tuple[int, int]] = []

    for ts in frame_ts:
        pos = int(np.searchsorted(unique_ts, ts))
        candidates: list[int] = []
        if pos < unique_ts.shape[0]:
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        if not candidates:
            ranges.append((0, 0))
            continue

        best = min(candidates, key=lambda idx: abs(int(unique_ts[idx]) - int(ts)))
        best_dt = abs(int(unique_ts[best]) - int(ts))
        if max_time_delta_us is not None and best_dt > max_time_delta_us:
            ranges.append((0, 0))
            continue

        start = int(first_idx[best])
        end = start + int(counts[best])
        ranges.append((start, end))
    return ranges


def _build_frame_track_ranges_window(
    track_ts: np.ndarray,
    frame_ts: np.ndarray,
    window_us: int,
) -> list[tuple[int, int]]:
    if track_ts.size == 0:
        return [(0, 0) for _ in range(frame_ts.shape[0])]

    ranges: list[tuple[int, int]] = []
    for ts in frame_ts:
        left = int(np.searchsorted(track_ts, ts - window_us, side="left"))
        right = int(np.searchsorted(track_ts, ts + window_us, side="right"))
        ranges.append((left, right))
    return ranges


def _tracks_to_target(
    tracks_slice: np.ndarray,
    image_height: int,
    image_width: int,
    class_id_to_label: dict[int, int],
    crop_top_px: int = 0,
    crop_bottom_px: int = 0,
    crop_left_px: int = 0,
    crop_right_px: int = 0,
) -> dict[str, torch.Tensor]:
    eff_h = int(image_height - crop_top_px - crop_bottom_px)
    eff_w = int(image_width - crop_left_px - crop_right_px)
    if eff_h <= 0 or eff_w <= 0:
        raise ValueError(
            f"Invalid crop resulting in non-positive shape: "
            f"({image_height},{image_width}) with crop "
            f"top={crop_top_px}, bottom={crop_bottom_px}, left={crop_left_px}, right={crop_right_px}"
        )

    boxes: list[list[float]] = []
    labels: list[int] = []

    for tr in tracks_slice:
        class_id = int(tr["class_id"])
        if class_id not in class_id_to_label:
            continue

        x = float(tr["x"])
        y = float(tr["y"])
        w = float(tr["w"])
        h = float(tr["h"])

        if w <= 0 or h <= 0:
            continue

        # Clip box to cropped image bounds before normalization.
        x1 = x - float(crop_left_px)
        y1 = y - float(crop_top_px)
        x2 = x1 + w
        y2 = y1 + h

        x1 = max(0.0, min(x1, float(eff_w)))
        y1 = max(0.0, min(y1, float(eff_h)))
        x2 = max(0.0, min(x2, float(eff_w)))
        y2 = max(0.0, min(y2, float(eff_h)))
        cw = x2 - x1
        ch = y2 - y1
        if cw <= 0.0 or ch <= 0.0:
            continue

        cx = (x1 + 0.5 * cw) / float(eff_w)
        cy = (y1 + 0.5 * ch) / float(eff_h)
        nw = cw / float(eff_w)
        nh = ch / float(eff_h)

        if nw <= 0 or nh <= 0:
            continue

        boxes.append([cx, cy, nw, nh])
        labels.append(class_id_to_label[class_id])

    if boxes:
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
    else:
        boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.zeros((0,), dtype=torch.long)

    return {"boxes": boxes_t, "labels": labels_t}


@dataclass
class _SequenceMeta:
    sequence_name: str
    frame_paths: list[Path]
    frame_timestamps_us: np.ndarray
    tracks: np.ndarray
    frame_track_ranges: list[tuple[int, int]]
    image_height: int
    image_width: int


def _discover_sequence_dirs(dsec_root: Path, split: str, sequences: list[str] | None) -> list[Path]:
    split_dir = dsec_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir does not exist: {split_dir}")

    if sequences:
        seq_dirs = [split_dir / name for name in sequences if (split_dir / name).is_dir()]
    else:
        seq_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    if not seq_dirs:
        raise RuntimeError(f"No sequence folders found under: {split_dir}")
    return seq_dirs


def _build_sequence_meta(
    sequence_dir: Path,
    image_relpath: str,
    timestamps_relpath: str,
    tracks_relpath: str,
    class_ids: list[int],
    time_mode: str,
    window_us: int,
    max_time_delta_us: int | None,
    max_frames_per_sequence: int | None,
) -> _SequenceMeta:
    frame_dir = sequence_dir / image_relpath
    frame_paths = _sorted_frame_paths(frame_dir)
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frame_dir}")
    if max_frames_per_sequence is not None:
        frame_paths = frame_paths[:max_frames_per_sequence]

    frame_timestamps_us = _load_timestamps(sequence_dir / timestamps_relpath, len(frame_paths))

    first_img = decode_image(str(frame_paths[0]))
    if first_img.ndim == 2:
        first_img = first_img.unsqueeze(0)
    _, image_height, image_width = first_img.shape

    tracks_path = _resolve_tracks_path(sequence_dir, tracks_relpath)
    tracks = _load_tracks_for_classes(tracks_path, class_ids)
    track_ts = tracks["t"].astype(np.int64)

    if time_mode == "nearest":
        frame_track_ranges = _build_frame_track_ranges_nearest(track_ts, frame_timestamps_us, max_time_delta_us)
    elif time_mode == "window":
        frame_track_ranges = _build_frame_track_ranges_window(track_ts, frame_timestamps_us, window_us)
    else:
        raise ValueError(f"Unsupported time_mode: {time_mode}")

    return _SequenceMeta(
        sequence_name=sequence_dir.name,
        frame_paths=frame_paths,
        frame_timestamps_us=frame_timestamps_us,
        tracks=tracks,
        frame_track_ranges=frame_track_ranges,
        image_height=image_height,
        image_width=image_width,
    )


class DSECSSD_ANN(Dataset):
    """
    Frame-wise DSEC loader for ANN training on RGB (or grayscale-expanded) images.
    """

    def __init__(
        self,
        dsec_root: Path | str,
        split: str,
        image_relpath: str = "images/left/distorted",
        timestamps_relpath: str = "images/timestamps.txt",
        tracks_relpath: str = "object_detections/left/tracks.npy",
        class_ids: str | list[int] | None = None,
        sequences: list[str] | None = None,
        time_mode: str = "nearest",
        window_us: int = 25_000,
        max_time_delta_us: int | None = 50_000,
        max_frames_per_sequence: int | None = None,
        force_grayscale: bool = True,
        crop_top_px: int = 0,
        crop_bottom_px: int = 0,
        crop_left_px: int = 0,
        crop_right_px: int = 0,
        transform=None,
    ):
        self.dsec_root = Path(dsec_root)
        self.split = split
        self.image_relpath = image_relpath
        self.timestamps_relpath = timestamps_relpath
        self.tracks_relpath = tracks_relpath
        self.class_ids = parse_class_ids(class_ids)
        self.class_id_to_label = {class_id: idx + 1 for idx, class_id in enumerate(self.class_ids)}
        self.class_names = [DSEC_CLASS_NAMES.get(c, f"class_{c}") for c in self.class_ids]
        self.time_mode = time_mode
        self.window_us = int(window_us)
        self.max_time_delta_us = None if max_time_delta_us is None else int(max_time_delta_us)
        self.max_frames_per_sequence = max_frames_per_sequence
        self.force_grayscale = force_grayscale
        self.crop_top_px = int(crop_top_px)
        self.crop_bottom_px = int(crop_bottom_px)
        self.crop_left_px = int(crop_left_px)
        self.crop_right_px = int(crop_right_px)
        self.transform = transform

        sequence_dirs = _discover_sequence_dirs(self.dsec_root, self.split, sequences)

        self.sequences: list[_SequenceMeta] = []
        self.samples: list[tuple[int, int]] = []

        for seq_dir in sequence_dirs:
            try:
                meta = _build_sequence_meta(
                    sequence_dir=seq_dir,
                    image_relpath=self.image_relpath,
                    timestamps_relpath=self.timestamps_relpath,
                    tracks_relpath=self.tracks_relpath,
                    class_ids=self.class_ids,
                    time_mode=self.time_mode,
                    window_us=self.window_us,
                    max_time_delta_us=self.max_time_delta_us,
                    max_frames_per_sequence=self.max_frames_per_sequence,
                )
            except Exception as exc:
                print(f"[DSECSSD_ANN] skipping sequence {seq_dir.name}: {exc}")
                continue

            seq_idx = len(self.sequences)
            self.sequences.append(meta)
            for frame_idx in range(len(meta.frame_paths)):
                self.samples.append((seq_idx, frame_idx))

        if not self.samples:
            raise RuntimeError(f"No DSEC ANN samples found under {self.dsec_root / self.split}")

        print(
            f"DSECSSD_ANN: split={self.split} sequences={len(self.sequences)} "
            f"frames={len(self.samples)} classes={self.class_ids}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frame(self, frame_path: Path) -> torch.Tensor:
        image = decode_image(str(frame_path))
        if image.ndim == 2:
            image = image.unsqueeze(0)

        h = int(image.shape[1])
        w = int(image.shape[2])
        y0 = self.crop_top_px
        y1 = h - self.crop_bottom_px
        x0 = self.crop_left_px
        x1 = w - self.crop_right_px
        if y1 <= y0 or x1 <= x0:
            raise ValueError(
                f"Invalid crop for {frame_path}: image=({h},{w}) "
                f"crop top/bottom/left/right={self.crop_top_px}/{self.crop_bottom_px}/{self.crop_left_px}/{self.crop_right_px}"
            )
        image = image[:, y0:y1, x0:x1]

        if self.force_grayscale:
            gray = image.float().mean(dim=0, keepdim=True)
            image = gray.repeat(3, 1, 1).to(dtype=image.dtype)
        elif image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] > 3:
            image = image[:3]

        if self.transform:
            image = self.transform(image)
        else:
            image = image.float() / 255.0
        return image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        seq_idx, frame_idx = self.samples[idx]
        seq = self.sequences[seq_idx]

        image = self._load_frame(seq.frame_paths[frame_idx])
        start, end = seq.frame_track_ranges[frame_idx]
        tracks_slice = seq.tracks[start:end]
        target = _tracks_to_target(
            tracks_slice=tracks_slice,
            image_height=seq.image_height,
            image_width=seq.image_width,
            class_id_to_label=self.class_id_to_label,
            crop_top_px=self.crop_top_px,
            crop_bottom_px=self.crop_bottom_px,
            crop_left_px=self.crop_left_px,
            crop_right_px=self.crop_right_px,
        )
        return image, target


class _H5EventStore:
    """
    Lazy HDF5 event reader (supports DSEC compressed H5 files via hdf5plugin).
    """

    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        self._file = None
        self._events_t = None
        self._events_x = None
        self._events_y = None
        self._events_p = None
        self._ms_to_idx = None
        self._t_offset = 0
        self._num_events = 0

    def __getstate__(self) -> dict[str, Any]:
        # Avoid pickling open HDF5 handles when DataLoader forks workers.
        state = dict(self.__dict__)
        state["_file"] = None
        state["_events_t"] = None
        state["_events_x"] = None
        state["_events_y"] = None
        state["_events_p"] = None
        state["_ms_to_idx"] = None
        return state

    def _ensure_open(self) -> None:
        if self._file is not None:
            return
        if h5py is None:
            raise ImportError("h5py is required for real DSEC event loading. Install h5py and hdf5plugin.")
        # Needed for compressed DSEC events.h5 (registers compression filters).
        import hdf5plugin  # noqa: F401

        self._file = h5py.File(self.h5_path, "r")
        self._events_t = self._file["events"]["t"]
        self._events_x = self._file["events"]["x"]
        self._events_y = self._file["events"]["y"]
        self._events_p = self._file["events"]["p"]
        self._ms_to_idx = self._file["ms_to_idx"]
        self._t_offset = int(self._file["t_offset"][()])
        self._num_events = int(self._events_t.shape[0])

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_events(self, t_start_us: int, t_end_us: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_open()
        if t_end_us <= t_start_us:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.uint8),
            )

        local_start = max(0, int(t_start_us) - self._t_offset)
        local_end = max(local_start, int(t_end_us) - self._t_offset)

        ms_start = local_start // 1000
        ms_end = local_end // 1000

        if ms_start >= len(self._ms_to_idx):
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.uint8),
            )

        idx_lo = int(self._ms_to_idx[ms_start])
        if ms_end + 1 < len(self._ms_to_idx):
            idx_hi = int(self._ms_to_idx[ms_end + 1])
        else:
            idx_hi = self._num_events

        idx_lo = max(0, min(idx_lo, self._num_events))
        idx_hi = max(idx_lo, min(idx_hi, self._num_events))

        if idx_hi <= idx_lo:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.uint8),
            )

        t_chunk = self._events_t[idx_lo:idx_hi]
        rel_lo = int(np.searchsorted(t_chunk, local_start, side="left"))
        rel_hi = int(np.searchsorted(t_chunk, local_end, side="left"))
        start = idx_lo + rel_lo
        end = idx_lo + rel_hi
        if end <= start:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.uint8),
            )

        x = self._events_x[start:end].astype(np.int64, copy=False)
        y = self._events_y[start:end].astype(np.int64, copy=False)
        p = self._events_p[start:end].astype(np.uint8, copy=False)
        return x, y, p


class _TextEventStore:
    """
    v2e text event reader.
    Expected columns: time[s], x, y, polarity.
    """

    def __init__(self, txt_path: Path):
        self.txt_path = txt_path
        data = np.loadtxt(txt_path, comments="#", dtype=np.float64)
        if data.ndim == 1 and data.size == 4:
            data = data.reshape(1, 4)
        if data.ndim != 2 or data.shape[1] < 4:
            raise ValueError(f"Unexpected v2e text format at {txt_path}")

        self.t_us = np.round(data[:, 0] * 1_000_000.0).astype(np.int64)
        self.x = np.round(data[:, 1]).astype(np.int64)
        self.y = np.round(data[:, 2]).astype(np.int64)
        self.p = np.round(data[:, 3]).astype(np.uint8)

    def get_events(self, t_start_us: int, t_end_us: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if t_end_us <= t_start_us:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.uint8),
            )
        left = int(np.searchsorted(self.t_us, int(t_start_us), side="left"))
        right = int(np.searchsorted(self.t_us, int(t_end_us), side="left"))
        return self.x[left:right], self.y[left:right], self.p[left:right]


@dataclass
class _EventSequenceMeta(_SequenceMeta):
    event_store: Any
    event_frame_timestamps_us: np.ndarray


def _resolve_event_file(sequence_dir: Path, event_source: str, event_relpath: str) -> Path:
    if event_source not in {"real", "simulated", "auto"}:
        raise ValueError(f"Unsupported event_source: {event_source}")

    if event_source in {"real", "auto"}:
        real_candidates = [
            sequence_dir / event_relpath,
            sequence_dir / "events/left/events.h5",
            sequence_dir / "events/right/events.h5",
        ]
        for cand in real_candidates:
            if cand.exists() and cand.suffix.lower() == ".h5":
                return cand
        if event_source == "real":
            raise FileNotFoundError(f"No real events.h5 found in {sequence_dir}")

    sim_candidates = [
        sequence_dir / event_relpath,
        sequence_dir / "v2e_output_distorted_gray_clean/dvs_events.txt",
        sequence_dir / "v2e_output/dvs_events.txt",
    ]
    for cand in sim_candidates:
        if cand.exists() and cand.suffix.lower() == ".txt":
            return cand

    raise FileNotFoundError(
        f"No event file found for sequence {sequence_dir.name}. "
        f"Tried real (events.h5) and simulated (dvs_events.txt) candidates."
    )


def _event_timestamps_for_frames(
    frame_timestamps_us: np.ndarray,
    source_path: Path,
    simulated_fps: float,
) -> np.ndarray:
    if source_path.suffix.lower() == ".txt":
        # v2e text events use relative timestamps (seconds from 0).
        frame_dt_us = int(round(1_000_000.0 / simulated_fps))
        return np.arange(frame_timestamps_us.shape[0], dtype=np.int64) * frame_dt_us
    return frame_timestamps_us


class DSECSSD_SNN(Dataset):
    """
    Sequence-wise DSEC loader for SNN training.

    Each item is:
      - images: (T, 2, H, W) event count tensor
      - targets: list[T] of {'boxes', 'labels'}

    Temporal integration is performed by SNN membrane state across frames,
    with one forward pass per frame (no repeated timesteps per frame).
    """

    def __init__(
        self,
        dsec_root: Path | str,
        split: str,
        image_relpath: str = "images/left/distorted",
        timestamps_relpath: str = "images/timestamps.txt",
        tracks_relpath: str = "object_detections/left/tracks.npy",
        class_ids: str | list[int] | None = None,
        sequences: list[str] | None = None,
        time_mode: str = "nearest",
        window_us: int = 25_000,
        max_time_delta_us: int | None = 50_000,
        max_frames_per_sequence: int | None = None,
        event_source: str = "auto",
        event_relpath: str = "events/left/events.h5",
        sequence_length: int = 8,
        sequence_stride: int = 8,
        event_window_mode: str = "between_frames",
        event_window_us: int = 50_000,
        simulated_fps: float = 20.0,
        count_clip_value: float = 5.0,
        log_counts: bool = True,
    ):
        self.dsec_root = Path(dsec_root)
        self.split = split
        self.image_relpath = image_relpath
        self.timestamps_relpath = timestamps_relpath
        self.tracks_relpath = tracks_relpath
        self.class_ids = parse_class_ids(class_ids)
        self.class_id_to_label = {class_id: idx + 1 for idx, class_id in enumerate(self.class_ids)}
        self.class_names = [DSEC_CLASS_NAMES.get(c, f"class_{c}") for c in self.class_ids]
        self.time_mode = time_mode
        self.window_us = int(window_us)
        self.max_time_delta_us = None if max_time_delta_us is None else int(max_time_delta_us)
        self.max_frames_per_sequence = max_frames_per_sequence
        self.event_source = event_source
        self.event_relpath = event_relpath
        self.sequence_length = int(sequence_length)
        self.sequence_stride = int(sequence_stride)
        self.event_window_mode = event_window_mode
        self.event_window_us = int(event_window_us)
        self.simulated_fps = float(simulated_fps)
        self.count_clip_value = float(count_clip_value)
        self.log_counts = bool(log_counts)

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if self.sequence_stride <= 0:
            raise ValueError("sequence_stride must be > 0")
        if self.event_window_mode not in {"between_frames", "fixed"}:
            raise ValueError("event_window_mode must be one of: between_frames, fixed")

        sequence_dirs = _discover_sequence_dirs(self.dsec_root, self.split, sequences)

        self.sequences: list[_EventSequenceMeta] = []
        self.samples: list[tuple[int, int]] = []

        for seq_dir in sequence_dirs:
            try:
                base_meta = _build_sequence_meta(
                    sequence_dir=seq_dir,
                    image_relpath=self.image_relpath,
                    timestamps_relpath=self.timestamps_relpath,
                    tracks_relpath=self.tracks_relpath,
                    class_ids=self.class_ids,
                    time_mode=self.time_mode,
                    window_us=self.window_us,
                    max_time_delta_us=self.max_time_delta_us,
                    max_frames_per_sequence=self.max_frames_per_sequence,
                )
                event_path = _resolve_event_file(
                    sequence_dir=seq_dir,
                    event_source=self.event_source,
                    event_relpath=self.event_relpath,
                )
                if event_path.suffix.lower() == ".h5":
                    event_store = _H5EventStore(event_path)
                else:
                    event_store = _TextEventStore(event_path)
                event_frame_ts = _event_timestamps_for_frames(
                    frame_timestamps_us=base_meta.frame_timestamps_us,
                    source_path=event_path,
                    simulated_fps=self.simulated_fps,
                )
            except Exception as exc:
                print(f"[DSECSSD_SNN] skipping sequence {seq_dir.name}: {exc}")
                continue

            seq_meta = _EventSequenceMeta(
                sequence_name=base_meta.sequence_name,
                frame_paths=base_meta.frame_paths,
                frame_timestamps_us=base_meta.frame_timestamps_us,
                tracks=base_meta.tracks,
                frame_track_ranges=base_meta.frame_track_ranges,
                image_height=base_meta.image_height,
                image_width=base_meta.image_width,
                event_store=event_store,
                event_frame_timestamps_us=event_frame_ts,
            )

            seq_idx = len(self.sequences)
            self.sequences.append(seq_meta)

            num_frames = len(seq_meta.frame_paths)
            if num_frames < self.sequence_length:
                continue
            for start_idx in range(0, num_frames - self.sequence_length + 1, self.sequence_stride):
                self.samples.append((seq_idx, start_idx))

        if not self.samples:
            raise RuntimeError(f"No DSEC SNN samples found under {self.dsec_root / self.split}")

        print(
            f"DSECSSD_SNN: split={self.split} sequences={len(self.sequences)} "
            f"samples={len(self.samples)} seq_len={self.sequence_length} classes={self.class_ids}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _build_event_frame(
        self,
        seq: _EventSequenceMeta,
        frame_idx: int,
    ) -> torch.Tensor:
        frame_ts = int(seq.event_frame_timestamps_us[frame_idx])
        if self.event_window_mode == "between_frames" and frame_idx > 0:
            start_ts = int(seq.event_frame_timestamps_us[frame_idx - 1])
        else:
            start_ts = frame_ts - self.event_window_us
        end_ts = frame_ts

        x, y, p = seq.event_store.get_events(start_ts, end_ts)
        frame = np.zeros((2, seq.image_height, seq.image_width), dtype=np.float32)

        if x.size > 0:
            valid = (
                (x >= 0)
                & (x < seq.image_width)
                & (y >= 0)
                & (y < seq.image_height)
            )
            if np.any(valid):
                xv = x[valid].astype(np.int64, copy=False)
                yv = y[valid].astype(np.int64, copy=False)
                pv = (p[valid] > 0).astype(np.int64, copy=False)
                np.add.at(frame, (pv, yv, xv), 1.0)

        if self.log_counts:
            frame = np.log1p(frame)
        if self.count_clip_value > 0:
            frame = np.clip(frame, 0.0, self.count_clip_value) / self.count_clip_value

        return torch.from_numpy(frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        seq_idx, start_idx = self.samples[idx]
        seq = self.sequences[seq_idx]

        images: list[torch.Tensor] = []
        targets: list[dict[str, torch.Tensor]] = []

        for offset in range(self.sequence_length):
            frame_idx = start_idx + offset
            images.append(self._build_event_frame(seq, frame_idx))

            start, end = seq.frame_track_ranges[frame_idx]
            tracks_slice = seq.tracks[start:end]
            target = _tracks_to_target(
                tracks_slice=tracks_slice,
                image_height=seq.image_height,
                image_width=seq.image_width,
                class_id_to_label=self.class_id_to_label,
            )
            targets.append(target)

        return torch.stack(images, dim=0), targets
