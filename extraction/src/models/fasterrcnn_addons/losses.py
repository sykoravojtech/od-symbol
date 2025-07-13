"""losses.py: Custom loss functions and RoI heads with Focal and GIoU losses for object detection"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops.boxes import _box_inter_union

__author__ = "Vojtěch Sýkora"


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Calculates Focal Loss"""

    cross_entropy_loss = F.cross_entropy(inputs, targets, reduction="none")
    p_t = torch.exp(-cross_entropy_loss)
    focal_weight = alpha * (1 - p_t) ** gamma
    focal_loss = focal_weight * cross_entropy_loss

    return focal_loss.mean()


def giou_loss(
    input_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """Generalized IoU Loss"""
    # if empty tensors
    if input_boxes.numel() == 0 or target_boxes.numel() == 0:
        return torch.tensor(0.0, device=input_boxes.device, requires_grad=True)

    inter, union = _box_inter_union(input_boxes, target_boxes)

    # prevent division by zero
    union = torch.clamp(union, min=eps)
    iou = inter / union

    # area of the smallest enclosing box
    min_box = torch.min(input_boxes, target_boxes)
    max_box = torch.max(input_boxes, target_boxes)
    area_c = (max_box[:, 2] - min_box[:, 0]) * (max_box[:, 3] - min_box[:, 1])
    # area_c = torch.clamp(area_c, min=eps)

    giou = iou - ((area_c - union) / area_c)
    giou = torch.clamp(giou, min=-1.0, max=1.0)

    loss = 1 - giou

    return loss.sum()


class CustomRoIHeads(RoIHeads):
    """RoI heads with GIoU (box regression) and Focal (classification) losses"""

    def __init__(
        self,
        focal_loss_fn=None,
        giou_loss_fn=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.focal_loss_fn = focal_loss_fn
        self.giou_loss_fn = giou_loss_fn
        if self.focal_loss_fn is not None:
            print("\tUsing focal loss for classification")
        if self.giou_loss_fn is not None:
            print("\tUsing GIoU loss for box regression")

    def fastrcnn_loss(
        self,
        class_logits: Tensor,
        box_regression: Tensor,
        labels: Tensor,
        regression_targets: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Computes the loss for Faster R-CNN."""
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # classification loss
        if self.focal_loss_fn is None:
            clsf_loss = F.cross_entropy(class_logits, labels)
        else:
            clsf_loss = self.focal_loss_fn(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_idxs = torch.where(labels > 0)[0]

        # Handle case where there are no positive samples
        if len(sampled_pos_idxs) == 0:
            # Return zero box loss if no positive samples
            box_loss = torch.tensor(0.0, device=class_logits.device, requires_grad=True)
        else:
            labels_pos = labels[sampled_pos_idxs]
            N, num_classes = class_logits.shape
            box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

            # Box regression loss
            if self.giou_loss_fn is None:
                box_loss = F.smooth_l1_loss(
                    box_regression[sampled_pos_idxs, labels_pos],
                    regression_targets[sampled_pos_idxs],
                    beta=1 / 9,
                    reduction="sum",
                )
            else:
                box_loss = self.giou_loss_fn(
                    box_regression[sampled_pos_idxs, labels_pos],
                    regression_targets[sampled_pos_idxs],
                )

            # Normalize box loss
            box_loss = box_loss / max(labels.numel(), 1)

        return clsf_loss, box_loss
