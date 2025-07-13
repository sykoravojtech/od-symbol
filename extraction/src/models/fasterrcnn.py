"""fasterrcnn.py: Wrapped torchvision Implementation of Faster RCNN"""

# Third-Party Imports
import warnings

import torch
import torch.nn.functional as F
from extraction.src.models.fasterrcnn_addons.losses import (
    CustomRoIHeads,
    focal_loss,
    giou_loss,
)
from torch import Tensor, nn
from torchvision.models.detection import roi_heads
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN as TVFasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

__author__ = "Johannes Bayer, Vojtěch Sýkora"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"


class FasterRCNN(TVFasterRCNN):
    """Wrapped torchvision Implementation of Faster RCNN"""

    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool,
        backbone: str,
        box_detections: int,
        trainable_backbone_layers: int,
        use_focal_loss: bool = False,
        use_giou_loss: bool = False,
        **kwargs,
    ):
        if use_focal_loss or use_giou_loss:
            print(
                f"\n\t** Enhancement settings: \n\tFocal Loss={use_focal_loss}, \n\tGIoU Loss={use_giou_loss}**\n"
            )

        super().__init__(
            resnet_fpn_backbone(
                backbone,
                pretrained_backbone,
                trainable_layers=_validate_trainable_layers(
                    pretrained_backbone, trainable_backbone_layers, 5, 3
                ),
            ),
            num_classes,
            box_detections_per_img=box_detections,  # Set the maximum number of detections per image
        )

        self.roi_heads.box_predictor = FastRCNNPredictor(
            self.roi_heads.box_predictor.cls_score.in_features, num_classes
        )

        self.use_focal_loss = use_focal_loss
        self.use_giou_loss = use_giou_loss
        if use_giou_loss or use_focal_loss:
            print("USING SPECIAL LOSSES")
            focal_loss_fn = focal_loss if use_focal_loss else None
            giou_loss_fn = giou_loss if use_giou_loss else None

            self.roi_heads = CustomRoIHeads(
                box_roi_pool=self.roi_heads.box_roi_pool,
                box_head=self.roi_heads.box_head,
                box_predictor=self.roi_heads.box_predictor,
                fg_iou_thresh=self.roi_heads.proposal_matcher.high_threshold,
                bg_iou_thresh=self.roi_heads.proposal_matcher.low_threshold,
                batch_size_per_image=self.roi_heads.fg_bg_sampler.batch_size_per_image,
                positive_fraction=self.roi_heads.fg_bg_sampler.positive_fraction,
                bbox_reg_weights=self.roi_heads.box_coder.weights,
                score_thresh=self.roi_heads.score_thresh,
                nms_thresh=self.roi_heads.nms_thresh,
                detections_per_img=self.roi_heads.detections_per_img,
                focal_loss_fn=focal_loss_fn,
                giou_loss_fn=giou_loss_fn,
            )


def _validate_trainable_layers(
    pretrained, trainable_backbone_layers, max_value, default_value
):
    """
    copied from latest torchvision.models.detection.backbone_utils as backport solution,
    due to missing function in current version on cluster!
    """
    # dont freeze any layers if pretrained model or backbone is not used
    if not pretrained:
        if trainable_backbone_layers is not None:
            warnings.warn(
                "Changing trainable_backbone_layers has no effect if "
                "neither pretrained nor pretrained_backbone have been set to True, "
                "falling back to trainable_backbone_layers={} so that all layers are trainable".format(
                    max_value
                )
            )
        trainable_backbone_layers = max_value

    # by default freeze first blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = default_value
    assert 0 <= trainable_backbone_layers <= max_value
    return trainable_backbone_layers
