import os
import re
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from converter.core.engineeringGraph import EngGraph
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

__author__ = "Vojtěch Sýkora"


# turn the warning off everywhere
# MeanAveragePrecision.warn_on_many_detections = False

# 542 is the max number of bboxes in any image over both our datasets
MAX_DET = 542


def compute_map_single(pred, target):
    """
    computes mAP for a single prediction-target pair.
    This can be used for the acc_fn parameter in the training code.
    """
    if len(pred["boxes"]) == 0:
        return 0

    preds_list = [
        {
            "boxes": pred["boxes"].cpu(),
            "scores": pred["scores"].cpu(),
            "labels": pred["labels"].cpu(),
        }
    ]

    targets_list = [
        {
            "boxes": target["boxes"].cpu(),
            "labels": target["labels"].cpu(),
        }
    ]

    # Use a new metric instance to avoid interfering with ongoing calculations
    # we set it to 10,100,MAX_DET to get an average over how good it is both at the main detections and also the details over 100
    # default is [1,10,100]
    temp_map_metric = MeanAveragePrecision(
        iou_type="bbox", max_detection_thresholds=[10, 100, MAX_DET]
    )
    temp_map_metric.update(preds_list, targets_list)
    results = temp_map_metric.compute()

    return results["map"].item() * 100


def compute_iou(boxes1, boxes2):
    """Compute Intersection over Union (IoU) between two sets of bounding boxes."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros((len(boxes1), len(boxes2)))

    # Compute intersection
    x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))

    inter_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Compute union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area

    return inter_area / union_area  # IoU matrix


def compute_accuracy(
    pred: Dict[str, Tensor], target: Dict[str, Tensor], iou_threshold: float = 0.5
) -> float:
    """
    Compute accuracy for object detection based on IoU matching.
    Accuracy = Correct Detections / Total Ground Truth Objects.

    pred = {'boxes':tensor(float), 'labels':tensor(int), 'scores':tensor()}
    target = {'boxes':tensor(float), 'labels':tensor(int)}
    """
    total_gt = len(target["boxes"])
    if total_gt == 0:
        return (
            1.0 if len(pred["boxes"]) == 0 else 0.0
        )  # If no Ground Truth, accuracy is 1 only if no predictions

    if len(pred["boxes"]) == 0:
        return 0.0  # No predictions, accuracy is 0

    # Compute IoU between predicted and ground truth boxes
    iou_matrix = compute_iou(pred["boxes"], target["boxes"])

    # Get best matches (IoU > threshold)
    matches = (iou_matrix > iou_threshold).any(dim=1)
    correct_detections = matches.sum().item()

    return correct_detections / total_gt


def compute_accuracy_graphs(pred: EngGraph, target: EngGraph, processor):
    """
    Compute the accuracy of predicted graph bounding boxes against target graph bounding boxes using a processor and a fixed IoU threshold.
    """
    # transform position to bboxes
    pred_bl = pred.get_boxes_labels(processor.get_class_id)
    target_bl = target.get_boxes_labels(processor.get_class_id)
    # print(f"{pred_bl=}")

    return compute_accuracy(pred_bl, target_bl, iou_threshold=0.5)


def find_avg_accuracy_from_dir(dir_path: str) -> float:
    accuracies = []
    pattern = re.compile(r"accuracy_\d+(?:\.\d+)?\.txt")
    for sub in os.listdir(dir_path):
        subdir = os.path.join(dir_path, sub)
        if os.path.isdir(subdir):
            for file in os.listdir(subdir):
                if pattern.fullmatch(file):
                    file_path = os.path.join(subdir, file)
                    with open(file_path, "r") as f:
                        try:
                            acc = float(f.read().strip())
                            accuracies.append(acc)
                        except ValueError:
                            pass  # Skip files that can't be converted to float
    if accuracies:
        return sum(accuracies) / len(accuracies)
    return 0.0


def print_accuracy_summary(acc_arr: Union[List[float], np.ndarray]) -> List[float]:
    """Print and return mean, min, max of accuracies.
    Accepts both list and numpy array as input.
    """
    acc_arr_np = np.array(acc_arr, dtype=float)

    mean_acc = acc_arr_np.mean()
    min_acc = acc_arr_np.min()
    max_acc = acc_arr_np.max()

    print(
        "\n=== ACCURACY SUMMARY ===\n"
        f"Avg mAP: {mean_acc:.4f}\n"
        f"Min mAP: {min_acc:.4f}\n"
        f"Max mAP: {max_acc:.4f}\n"
        f"Total  : {acc_arr_np.shape[0]} images"
    )
    # Print average (min, max) format
    print(f"{mean_acc:.4f} ({min_acc:.4f}, {max_acc:.4f})")

    return [mean_acc, min_acc, max_acc]


def save_accuracy_summary(
    total_accuracies: Dict[str, List[float]],
    config: Dict,
    base_dir: str,
    output_file: str,
    note: str = "",
) -> Tuple[float, float, float]:
    """
    Save accuracy summary to a text file with per-drafter and overall statistics.
    """
    with open(output_file, "w") as f:
        # Write per-drafter statistics
        f.write(f"=== mAP STATISTICS for {base_dir} ===\n")

        f.write(f"{note}\n")

        overall_accs = []

        for drafter, accs in total_accuracies.items():
            accs_np = np.array(accs, dtype=float)
            avg = accs_np.mean()
            min_val = accs_np.min()
            max_val = accs_np.max()
            count = len(accs)

            f.write(f"== {drafter} ==\n")
            f.write(f"Avg mAP: {avg:.4f}\n")
            f.write(f"Min mAP: {min_val:.4f}\n")
            f.write(f"Max mAP: {max_val:.4f}\n")
            f.write(f"Images : {count}\n\n")

            # Collect all accuracies for overall statistics
            overall_accs.extend(accs)

        # Calculate and write overall statistics
        if overall_accs:
            overall_np = np.array(overall_accs, dtype=float)
            overall_avg = overall_np.mean()
            overall_min = overall_np.min()
            overall_max = overall_np.max()
            overall_count = len(overall_accs)

            f.write("=== OVERALL STATISTICS ===\n\n")
            f.write(f"Overall Avg mAP: {overall_avg:.4f}\n")
            f.write(f"Overall Min mAP: {overall_min:.4f}\n")
            f.write(f"Overall Max mAP: {overall_max:.4f}\n")
            f.write(f"Total Images   : {overall_count}\n")

            f.write(f"{overall_avg:.4f} ({overall_min:.4f}, {overall_max:.4f})\n")

        print(f"Accuracy summary saved to {output_file}")
    return (overall_avg, overall_min, overall_max)


def print_num_pred_bboxes(pred_dict: Dict, target_dict: Dict) -> float:
    num_pred_boxes = len(pred_dict["boxes"])
    num_gt_boxes = len(target_dict["boxes"])
    predicted_percentage = num_pred_boxes / num_gt_boxes
    print(
        f"Number of predicted boxes: {num_pred_boxes}/{num_gt_boxes} Ground Truth Boxes ({predicted_percentage*100:.2f}%)"
    )


if __name__ == "__main__":
    dir_path = os.path.join("extraction", "output")
    avg_acc = find_avg_accuracy_from_dir(dir_path)
    print(f"=== AVERAGE ACCURACY {dir_path} is {avg_acc} aka {avg_acc*100:.2f}%===")
