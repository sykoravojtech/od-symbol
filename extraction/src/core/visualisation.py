"""visualisation.py: Metric and Data Visualisation and Calculation Tools"""

# System Imports
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Third-Party Imports
import torch
from extraction.src.utils.position_utils import get_xymin_xymax_pos
from networkx.classes.reportviews import NodeDataView
from tueplots.constants.color import palettes, rgb

__author__ = "Johannes Bayer, Vojtěch Sýkora"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"


def debug_dump(batch_data: torch.Tensor, name: str, debug_path: str):
    """Dumps the First Sample of a batch tensor as a List of 2D Plots"""

    sample = batch_data[0].detach().cpu().numpy()

    if len(sample.shape) == 1:
        sample = sample[np.newaxis, np.newaxis]

    if len(sample.shape) == 2:
        sample = sample[np.newaxis]

    for counter, debug_slice in enumerate(sample):
        plt.clf()
        plt.imshow(debug_slice, cmap="hot", interpolation="nearest", vmin=-1, vmax=1)
        plt.savefig(os.path.join(debug_path, f"{name}_{counter}.png"))


def plot_curve(
    metrics: dict,
    max_epochs: int,
    exp_path: str = None,
    show_max: bool = True,
    loss_ymin: float = 0.0,
    loss_ymax: float = 1.4,
    percent: bool = True,
    fasterrcnn: bool = False,
):
    """Plots and Saves the Learning Curves"""

    # plt.clf()
    fig, axis_acc = plt.subplots()
    axis_loss = axis_acc.twinx()

    if percent:
        axis_acc.axis(xmin=0, xmax=max_epochs, ymin=0, ymax=100)
    else:
        axis_acc.axis(xmin=0, xmax=max_epochs, ymin=0, ymax=1.0)
    axis_loss.axis(xmin=0, xmax=max_epochs, ymin=loss_ymin, ymax=loss_ymax)
    axis_acc.grid()

    # plt.title("Learning Curve")
    axis_acc.set_xlabel("Epoch")
    axis_acc.set_ylabel("Accuracy (%)" if percent else "Accuracy")
    axis_loss.set_ylabel("Loss")

    if show_max:
        max_val_acc = max(metrics["Val Acc"])
        max_val_acc_epoch = metrics["Val Acc"].index(max_val_acc)
        axis_acc.axhline(y=max_val_acc, color="grey", linewidth=1)
        plt.axvline(x=max_val_acc_epoch, color="grey", linewidth=1)

        axis_acc.text(0.5, max_val_acc + 1, f"{max_val_acc:.4f}", color="grey")
        axis_acc.text(max_val_acc_epoch + 0.5, 1, f"{max_val_acc_epoch}", color="grey")

    for metric_name, metric_curve in metrics.items():
        axis = axis_loss if "Loss" in metric_name else axis_acc

        # For FasterRCNN, only show Val Acc and Train Loss in legend
        if fasterrcnn:
            show_in_legend = (metric_name == "Val Acc") or (metric_name == "Train Loss")
            label = metric_name if show_in_legend else None
        else:
            label = metric_name

        axis.plot(
            metric_curve,
            label=label,
            color=rgb.tue_blue if "Train" in metric_name else rgb.tue_red,
            linestyle="solid" if "Acc" in metric_name else "dotted",
        )

    axis_acc.legend(loc="upper left")
    axis_loss.legend(loc="upper right")
    fig.tight_layout()

    if exp_path:
        plt.savefig(os.path.join(exp_path, "learning_curve.png"), dpi=400)

        with open(os.path.join(exp_path, "learning_curve.json"), "w") as logfile:
            logfile.write(json.dumps(metrics, indent=4))

    else:
        plt.show()

    plt.close()


def save_comparison_image(
    img1: np.ndarray,
    img2: np.ndarray,
    save_path: str,
    note1: str = "ground truth",
    note2: str = "prediction",
) -> None:
    """Combine images side by side and save.
    also add notes on top left to distinguish the images"""
    # annotate copies
    a1 = img1.copy()
    a2 = img2.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    color = (0, 0, 255)  # red
    y = 20
    x = 5
    cv2.putText(a1, note1, (x, y), font, scale, color, thickness)
    cv2.putText(a2, note2, (x, y), font, scale, color, thickness)

    # stitch and save
    comp = cv2.hconcat([a1, a2])
    cv2.imwrite(save_path, comp)
    print(f"Saved comparison image to {save_path}")


def draw_bboxes(image: np.ndarray, graph: Any, verbose: int = 0) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    img = image.copy()
    # Extract annotations from the graph
    node_data: List[Tuple[Any, Dict]] = [
        (nid, ann) for nid, ann in graph.nodes(data=True)
    ]
    # for x, y in node_data:
    #     print(f"{x}: {y}")

    for id, ann in node_data:
        # print(f"{id}: {ann['name']}, {ann['type']}")
        ann_type = ann["type"]
        ann_pos = ann["position"]
        x = int(ann_pos["x"])
        y = int(ann_pos["y"])
        w = int(ann_pos["width"])
        h = int(ann_pos["height"])
        xmin, ymin, xmax, ymax = get_xymin_xymax_pos(x, y, w, h)

        # dd bbox
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Add label above bbox
        cv2.putText(
            img=img,
            text=ann_type,
            org=(xmin, ymin - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(255, 0, 0),
            thickness=1,
        )

    # cv2.imshow("Labeled image", img)
    # cv2.waitKey(0)
    return img


def draw_and_save_bboxes(
    save_path: str, image: np.ndarray, graph: Any, verbose: int = 0
) -> np.ndarray:
    image_with_bboxes = draw_bboxes(image, graph)
    cv2.imwrite(save_path, image_with_bboxes)
    if verbose > 0:
        print(f"Saved image with bboxes to {save_path}")
    return image_with_bboxes


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: One metrics JSON file needs to be provided")

    else:
        with open(sys.argv[1]) as json_file:
            metrics_vis = json.loads(json_file.read())

            plot_curve(
                metrics_vis,
                max_epochs=max([len(metric) for metric in metrics_vis.values()]),
            )

"""
poetry run python -m extraction.src.core.visualisation
"""
