"""rotation.py: Helper Functions for """

# System Imports
import json
from os.path import join
from math import sin, cos, pi, atan2

# Project Imports
import matplotlib.pyplot as plt

# Third-Party Imports
from torch import Tensor

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"



def encode_rotation(rotation: int, symmetric: bool = False) -> Tensor:
    """Transforms a Rotation given as 360 degree Integer into a sin/cos Representation"""

    if symmetric and 90 < rotation < 270:
        rotation -= 180

    return Tensor((sin(rotation*2*pi/360), cos(rotation*2*pi/360)))


def decode_rotation(rotation_encoded: Tensor) -> int:
    """Transforms a sin/cos Representation to a 360 degree integer"""

    angle_cos, angle_sin = rotation_encoded.tolist()

    return round(360*(atan2(angle_cos, angle_sin))/(2*pi)) % 360


def angle_distance(pred: Tensor, target: Tensor, margin=5):
    """Rotation Prediction Accuracy Metric"""

    rot_pred = decode_rotation(pred)
    rot_target = decode_rotation(target)

    if min(abs(rot_pred-rot_target) % 360, -abs(rot_pred-rot_target) % 360) < margin:
        return 1

    return 0


def rotation_visualisation(inputs, targets, preds, infos, exp_path, set_name, individual=True, margin=5, cls_types=None):
    """"""

    angles_target = [decode_rotation(target) for target in targets]
    angles_pred = [decode_rotation(pred) for pred in preds]
    classes = [info['type'] for info in infos]
    samples = list(zip(angles_target, angles_pred, classes))

    cls_types = cls_types if cls_types else list(set(classes))

    rotation_plot(samples, cls_types, exp_path, set_name, margin, True)

    if individual:
        for cls in set(classes):
            rotation_plot(samples, [cls], exp_path, set_name, margin, False)

    acc = {}

    for cls in cls_types:
        samples_cls = [sample for sample in samples if sample[2] == cls]
        if samples_cls:
            acc[cls] = sum([1.0*(min(abs(sample[1]-sample[0]) % 360, -abs(sample[1]-sample[0]) % 360) < margin)
                            for sample in samples_cls])/len(samples_cls)
        # TODO use original acc metric here

    with open(join(exp_path, f"accuracy_{set_name}.json"), "w") as metric_file:
        metric_file.write(json.dumps(acc, indent=4))


def rotation_plot(samples: list, cls_types: list, exp_path: str, set_name: str, margin: float, legend: bool) -> None:
    """Generates and Saves ONE Rotation Prediction vs. GT Plot"""

    plt.clf()

    if legend:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)

    else:
        plt.figure(figsize=(5, 5))

    plt.title("Rotation Evaluation")
    plt.xlabel("Target Angle")
    plt.ylabel("Predicted Angle")
    plt.xlim((0, 360))
    plt.ylim((0, 360))

    for cls in cls_types:
        plt.scatter(x=[sample[0] for sample in samples if sample[2] == cls],
                    y=[sample[1] for sample in samples if sample[2] == cls],
                    label=cls)

    plt.plot([0, 360], [0, 360], color="black", linewidth=1)  # GT==Pred -> Identity
    plt.plot([0, 360], [margin, 360+margin], "--", color="black", linewidth=1)  # Margin of Acceptance
    plt.plot([0, 360], [-margin, 360-margin], "--", color="black", linewidth=1)  # Margin of Acceptance

    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=3)

    file_name = "eval_" + set_name + (f"_{cls_types[0]}" if len(cls_types) == 1 else "") + ".png"
    plt.savefig(join(exp_path, file_name), dpi=400)
    plt.close()
