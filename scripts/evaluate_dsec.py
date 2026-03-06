"""
Evaluate ANN/SNN SSD checkpoints on DSEC.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dsec_dataset import DSECSSD_ANN, DSECSSD_SNN, DSEC_CLASS_NAMES, parse_class_ids
from src.models.ann import VGG11_SSD_ANN
from src.models.snn import VGG11_SSD_SNN
from src.utils import compute_iou, decode_boxes, generate_anchors_for_model, nms, xywh_to_xyxy


def parse_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


class DetectionEvaluator:
    def __init__(self, num_classes, iou_thresholds=None, class_names=None):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5]
        self.class_names = class_names or [f"class_{i}" for i in range(1, num_classes + 1)]
        self.all_predictions = []
        self.all_ground_truths = []
        self.inference_times = []
        self.image_counter = 0

    def add_predictions(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, image_id=None):
        if image_id is None:
            image_id = self.image_counter
            self.image_counter += 1
        self.all_predictions.append(
            {
                "boxes": pred_boxes.cpu().numpy(),
                "scores": pred_scores.cpu().numpy(),
                "labels": pred_labels.cpu().numpy(),
                "image_id": image_id,
            }
        )
        self.all_ground_truths.append(
            {
                "boxes": gt_boxes.cpu().numpy(),
                "labels": gt_labels.cpu().numpy(),
                "image_id": image_id,
            }
        )

    @staticmethod
    def _compute_ap(recalls, precisions):
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        return ap

    def _compute_map_single(self, iou_threshold):
        aps = []
        per_class = {}

        for class_id in range(1, self.num_classes + 1):
            class_preds = []
            class_gts = []
            for pred in self.all_predictions:
                mask = pred["labels"] == class_id
                if mask.any():
                    class_preds.append(
                        {
                            "boxes": pred["boxes"][mask],
                            "scores": pred["scores"][mask],
                            "image_id": pred["image_id"],
                        }
                    )
            for gt in self.all_ground_truths:
                mask = gt["labels"] == class_id
                if mask.any():
                    class_gts.append({"boxes": gt["boxes"][mask], "image_id": gt["image_id"]})

            if len(class_gts) == 0:
                per_class[self.class_names[class_id - 1]] = {
                    "ap": 0.0,
                    "num_gt": 0,
                    "num_pred": sum(len(p["boxes"]) for p in class_preds),
                    "tp": 0,
                    "fp": sum(len(p["boxes"]) for p in class_preds),
                }
                continue

            all_pred_boxes = []
            all_pred_scores = []
            all_pred_image_ids = []
            for pred in class_preds:
                all_pred_boxes.extend(pred["boxes"])
                all_pred_scores.extend(pred["scores"])
                all_pred_image_ids.extend([pred["image_id"]] * len(pred["boxes"]))

            if len(all_pred_boxes) == 0:
                per_class[self.class_names[class_id - 1]] = {
                    "ap": 0.0,
                    "num_gt": sum(len(gt["boxes"]) for gt in class_gts),
                    "num_pred": 0,
                    "tp": 0,
                    "fp": 0,
                }
                continue

            all_pred_boxes = np.array(all_pred_boxes)
            all_pred_scores = np.array(all_pred_scores)
            all_pred_image_ids = np.array(all_pred_image_ids)
            order = np.argsort(-all_pred_scores)
            all_pred_boxes = all_pred_boxes[order]
            all_pred_image_ids = all_pred_image_ids[order]

            gt_dict = {}
            for gt in class_gts:
                image_id = gt["image_id"]
                if image_id not in gt_dict:
                    gt_dict[image_id] = {"boxes": gt["boxes"], "matched": np.zeros(len(gt["boxes"]), dtype=bool)}

            tp = np.zeros(len(all_pred_boxes))
            fp = np.zeros(len(all_pred_boxes))
            for i, (pred_box, image_id) in enumerate(zip(all_pred_boxes, all_pred_image_ids)):
                if image_id not in gt_dict:
                    fp[i] = 1
                    continue
                gt_info = gt_dict[image_id]
                if len(gt_info["boxes"]) == 0:
                    fp[i] = 1
                    continue

                pred_box_t = torch.tensor(pred_box).unsqueeze(0)
                gt_boxes_t = torch.tensor(gt_info["boxes"])
                ious = compute_iou(pred_box_t, gt_boxes_t)[0].numpy()
                best_idx = int(np.argmax(ious))
                best_iou = float(ious[best_idx])

                if best_iou >= iou_threshold and not gt_info["matched"][best_idx]:
                    tp[i] = 1
                    gt_info["matched"][best_idx] = True
                else:
                    fp[i] = 1

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            num_gt = sum(len(gt["boxes"]) for gt in class_gts)
            recalls = tp_cum / max(1, num_gt)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            ap = self._compute_ap(recalls, precisions)
            aps.append(ap)

            per_class[self.class_names[class_id - 1]] = {
                "ap": float(ap),
                "num_gt": int(num_gt),
                "num_pred": int(len(all_pred_boxes)),
                "tp": int(tp_cum[-1]) if len(tp_cum) else 0,
                "fp": int(fp_cum[-1]) if len(fp_cum) else 0,
            }

        return {"mAP": float(np.mean(aps) if aps else 0.0), "per_class": per_class}

    def compute_metrics(self):
        results = {}
        for t in self.iou_thresholds:
            r = self._compute_map_single(t)
            results[f"mAP@{t}"] = r["mAP"]
            results[f"per_class_AP@{t}"] = r["per_class"]
        results["mAP_avg"] = float(np.mean([results[f"mAP@{t}"] for t in self.iou_thresholds]))
        if self.inference_times:
            results["latency"] = {
                "mean_ms": float(np.mean(self.inference_times) * 1000.0),
                "std_ms": float(np.std(self.inference_times) * 1000.0),
                "fps": float(1.0 / np.mean(self.inference_times)),
            }
        results["num_images"] = len(self.all_predictions)
        results["num_predictions"] = int(sum(len(p["boxes"]) for p in self.all_predictions))
        results["num_ground_truths"] = int(sum(len(g["boxes"]) for g in self.all_ground_truths))
        return results


def post_process_detections(cls_preds, loc_preds, anchors, conf_threshold, nms_threshold, device):
    cls_probs = torch.softmax(cls_preds, dim=-1)
    class_probs, class_labels = cls_probs[:, 1:].max(dim=-1)
    class_labels = class_labels + 1
    conf_mask = class_probs > conf_threshold

    if conf_mask.sum() == 0:
        return (
            torch.zeros((0, 4), device=device),
            torch.zeros((0,), device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
        )

    confident_probs = class_probs[conf_mask]
    confident_labels = class_labels[conf_mask]
    confident_loc_preds = loc_preds[conf_mask]
    confident_anchors = anchors[conf_mask]
    decoded_boxes = decode_boxes(confident_loc_preds, confident_anchors)
    decoded_boxes_xyxy = xywh_to_xyxy(decoded_boxes)

    keep_indices = []
    for label in confident_labels.unique():
        label_mask = confident_labels == label
        keep = nms(decoded_boxes_xyxy[label_mask], confident_probs[label_mask], nms_threshold)
        label_indices = torch.where(label_mask)[0]
        keep_indices.append(label_indices[keep])

    if not keep_indices:
        return (
            torch.zeros((0, 4), device=device),
            torch.zeros((0,), device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
        )

    keep_indices = torch.cat(keep_indices)
    return (
        decoded_boxes_xyxy[keep_indices],
        confident_probs[keep_indices],
        confident_labels[keep_indices],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DSEC ANN/SNN checkpoint")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True, choices=["ann", "snn"])
    parser.add_argument("--dsec-root", type=str, default="data/dsec")
    parser.add_argument("--test-splits", type=str, default="test", help="Comma-separated split names")
    parser.add_argument("--sequences", type=str, default="", help="Optional comma-separated sequence names")
    parser.add_argument("--class-ids", type=str, default="0,1,2,3,4,5,6,7")

    parser.add_argument("--image-relpath", type=str, default="images/left/distorted")
    parser.add_argument("--timestamps-relpath", type=str, default="images/timestamps.txt")
    parser.add_argument("--tracks-relpath", type=str, default="object_detections/left/tracks.npy")
    parser.add_argument("--time-mode", type=str, default="nearest", choices=["nearest", "window"])
    parser.add_argument("--window-us", type=int, default=25_000)
    parser.add_argument("--max-time-delta-us", type=int, default=50_000)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--rgb-color", action="store_true")
    parser.add_argument("--crop-top-px", type=int, default=0)
    parser.add_argument("--crop-bottom-px", type=int, default=0)
    parser.add_argument("--crop-left-px", type=int, default=0)
    parser.add_argument("--crop-right-px", type=int, default=0)

    parser.add_argument("--event-source", type=str, default="auto", choices=["auto", "real", "simulated"])
    parser.add_argument("--event-relpath", type=str, default="events/left/events.h5")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=1)
    parser.add_argument("--event-window-mode", type=str, default="between_frames", choices=["between_frames", "fixed"])
    parser.add_argument("--event-window-us", type=int, default=50_000)
    parser.add_argument("--simulated-fps", type=float, default=20.0)
    parser.add_argument("--count-clip-value", type=float, default=5.0)
    parser.add_argument("--no-log-counts", action="store_true")

    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--surrogate-slope", type=float, default=25.0)
    parser.add_argument("--input-height", type=int, default=480)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="neuromorph-vs-noise")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def build_model(args, num_classes, device):
    if args.model_type == "ann":
        model = VGG11_SSD_ANN(num_classes=num_classes + 1)
    else:
        from snntorch import surrogate

        model = VGG11_SSD_SNN(
            num_classes=num_classes + 1,
            beta=args.beta,
            threshold=args.threshold,
            spike_grad=surrogate.fast_sigmoid(slope=args.surrogate_slope),
        )
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def get_dataloader(args, split, class_ids):
    sequences = parse_list(args.sequences) if args.sequences.strip() else None
    max_delta = None if args.max_time_delta_us <= 0 else args.max_time_delta_us

    if args.model_type == "ann":
        transform = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize((args.input_height, args.input_width), antialias=True),
            ]
        )
        dataset = DSECSSD_ANN(
            dsec_root=args.dsec_root,
            split=split,
            image_relpath=args.image_relpath,
            timestamps_relpath=args.timestamps_relpath,
            tracks_relpath=args.tracks_relpath,
            class_ids=class_ids,
            sequences=sequences,
            time_mode=args.time_mode,
            window_us=args.window_us,
            max_time_delta_us=max_delta,
            max_frames_per_sequence=args.max_frames_per_sequence,
            force_grayscale=not args.rgb_color,
            crop_top_px=args.crop_top_px,
            crop_bottom_px=args.crop_bottom_px,
            crop_left_px=args.crop_left_px,
            crop_right_px=args.crop_right_px,
            transform=transform,
        )

        def collate_fn_ann(batch):
            images, targets = zip(*batch)
            return torch.stack(images, dim=0), list(targets)

        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_ann,
            pin_memory=True,
        )

    dataset = DSECSSD_SNN(
        dsec_root=args.dsec_root,
        split=split,
        image_relpath=args.image_relpath,
        timestamps_relpath=args.timestamps_relpath,
        tracks_relpath=args.tracks_relpath,
        class_ids=class_ids,
        sequences=sequences,
        time_mode=args.time_mode,
        window_us=args.window_us,
        max_time_delta_us=max_delta,
        max_frames_per_sequence=args.max_frames_per_sequence,
        event_source=args.event_source,
        event_relpath=args.event_relpath,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        event_window_mode=args.event_window_mode,
        event_window_us=args.event_window_us,
        simulated_fps=args.simulated_fps,
        count_clip_value=args.count_clip_value,
        log_counts=not args.no_log_counts,
    )

    def collate_fn_snn(batch):
        images_list, targets_list = zip(*batch)
        return list(images_list), list(targets_list)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_snn,
        pin_memory=True,
    )


@torch.no_grad()
def run_ann_eval(model, dataloader, anchors, device, conf_threshold, nms_threshold, evaluator):
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="ANN inference")):
        images = images.to(device)
        for i in range(images.shape[0]):
            image = images[i : i + 1]
            target = targets[i]

            start = time.time()
            cls_preds, loc_preds = model(image)
            evaluator.inference_times.append(time.time() - start)

            pred_boxes, pred_scores, pred_labels = post_process_detections(
                cls_preds[0], loc_preds[0], anchors, conf_threshold, nms_threshold, device
            )
            gt_boxes_xyxy = xywh_to_xyxy(target["boxes"])
            evaluator.add_predictions(
                pred_boxes, pred_scores, pred_labels, gt_boxes_xyxy, target["labels"], image_id=f"{batch_idx}_{i}"
            )


@torch.no_grad()
def run_snn_eval(model, dataloader, anchors, device, conf_threshold, nms_threshold, evaluator, input_h, input_w):
    for batch_idx, (sequences, targets_sequences) in enumerate(tqdm(dataloader, desc="SNN inference")):
        for seq_idx, (seq_images, seq_targets) in enumerate(zip(sequences, targets_sequences)):
            model.reset_states()
            for frame_idx in range(seq_images.shape[0]):
                frame = seq_images[frame_idx : frame_idx + 1].to(device)
                if frame.shape[-2:] != (input_h, input_w):
                    frame = F.interpolate(frame, size=(input_h, input_w), mode="bilinear", align_corners=False)
                target = seq_targets[frame_idx]

                start = time.time()
                cls_preds, loc_preds = model(frame)
                evaluator.inference_times.append(time.time() - start)

                pred_boxes, pred_scores, pred_labels = post_process_detections(
                    cls_preds[0], loc_preds[0], anchors, conf_threshold, nms_threshold, device
                )
                gt_boxes_xyxy = xywh_to_xyxy(target["boxes"])
                evaluator.add_predictions(
                    pred_boxes,
                    pred_scores,
                    pred_labels,
                    gt_boxes_xyxy,
                    target["labels"],
                    image_id=f"{batch_idx}_seq{seq_idx}_t{frame_idx}",
                )


def print_results(split, results, iou_thresholds):
    print(f"\n=== DSEC eval: {split} ===")
    print(f"images={results['num_images']} preds={results['num_predictions']} gts={results['num_ground_truths']}")
    print(f"mAP_avg={results['mAP_avg']:.4f}")
    for t in iou_thresholds:
        print(f"mAP@{t}={results[f'mAP@{t}']:.4f}")
    if "latency" in results:
        print(
            f"latency mean={results['latency']['mean_ms']:.2f}ms std={results['latency']['std_ms']:.2f}ms "
            f"fps={results['latency']['fps']:.2f}"
        )


def to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    class_ids = parse_class_ids(args.class_ids)
    num_classes = len(class_ids)
    class_names = [DSEC_CLASS_NAMES.get(c, f"class_{c}") for c in class_ids]

    model = build_model(args, num_classes, device)
    if args.model_type == "ann":
        anchors = generate_anchors_for_model(model, (3, args.input_height, args.input_width), device)
    else:
        anchors = generate_anchors_for_model(model, (2, args.input_height, args.input_width), device)

    all_results = {}
    splits = parse_list(args.test_splits)
    for split in splits:
        dataloader = get_dataloader(args, split, class_ids)
        evaluator = DetectionEvaluator(num_classes, iou_thresholds=args.iou_thresholds, class_names=class_names)

        if args.model_type == "ann":
            run_ann_eval(
                model, dataloader, anchors, device, args.conf_threshold, args.nms_threshold, evaluator
            )
        else:
            run_snn_eval(
                model,
                dataloader,
                anchors,
                device,
                args.conf_threshold,
                args.nms_threshold,
                evaluator,
                args.input_height,
                args.input_width,
            )

        results = evaluator.compute_metrics()
        all_results[split] = results
        print_results(split, results, args.iou_thresholds)

        out_path = os.path.join(args.output_dir, f"eval_dsec_{args.model_type}_{split.replace('/', '_')}.json")
        payload = to_serializable(results)
        payload["config"] = vars(args)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"saved: {out_path}")

        if args.wandb:
            try:
                import wandb

                run_name = args.wandb_run_name or f"eval_dsec_{args.model_type}_{split.replace('/', '_')}"
                wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    config=vars(args),
                    tags=["evaluation", "dsec", args.model_type, split],
                    reinit=True,
                )
                wandb.log(
                    {
                        "eval/mAP_avg": results["mAP_avg"],
                        **{f"eval/mAP@{t}": results[f"mAP@{t}"] for t in args.iou_thresholds},
                        "eval/num_images": results["num_images"],
                    }
                )
                artifact = wandb.Artifact(
                    name=f"eval_dsec_{args.model_type}_{split.replace('/', '_')}", type="evaluation"
                )
                artifact.add_file(out_path)
                wandb.log_artifact(artifact)
                wandb.finish()
            except ImportError:
                print("wandb not installed; skipping wandb logging")

    print("\n=== summary ===")
    for split, results in all_results.items():
        print(f"{split}: mAP@0.5={results.get('mAP@0.5', 0.0):.4f}")
    if all_results:
        avg_map50 = np.mean([r.get("mAP@0.5", 0.0) for r in all_results.values()])
        print(f"overall mAP@0.5={avg_map50:.4f}")


if __name__ == "__main__":
    main()
