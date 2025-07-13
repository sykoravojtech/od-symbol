"""inference_OD.py: Performs Model Inference given an Image and optional Graph Structure"""

# System Imports
import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from shutil import copy2
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

# Third-Party Imports
import cv2
import numpy as np
import torch

# Project Imports
from converter.converter.jsonConverter import JSONConverter
from converter.converter.pascalVocConverter import PascalVocConverter
from converter.converter.pngConverter import PngConverter
from converter.converter.svgConverter import SvgConverter
from converter.core.boundingbox import BoundingBox
from converter.core.engineeringGraph import EngGraph
from extraction.src.core.package_loader import build_processor_model
from extraction.src.core.processor import Processor
from extraction.src.core.visualisation import (
    draw_and_save_bboxes,
    save_comparison_image,
)
from extraction.src.utils.accuracy import (
    compute_map_single,
    print_accuracy_summary,
    print_num_pred_bboxes,
    save_accuracy_summary,
)
from extraction.src.utils.config_utils import load_config, save_config
from extraction.src.utils.paths import (
    create_subdir_timestamp,
    get_ann_path_from_img_path,
    get_subdir_basenames,
)
from networkx.classes.reportviews import NodeDataView

__author__ = "Vojtěch Sýkora"


def suppress_torchvision_warnings() -> None:
    """Silence specific torchvision UserWarnings about deprecated parameters."""
    # ignore all UserWarnings coming from torchvision.models._utils
    warnings.filterwarnings(
        "ignore", category=UserWarning, module=r"torchvision\.models\._utils"
    )


suppress_torchvision_warnings()


__author__ = "Vojtěch Sýkora"


def get_pred_dict(pred_list: List) -> Dict:
    """Extract prediction dictionary from prediction list containing boxes, labels, and scores."""
    boxes = pred_list[0][0]["boxes"]
    labels = pred_list[0][0]["labels"]
    scores = pred_list[0][0]["scores"]
    pred_dict = {"boxes": boxes, "labels": labels, "scores": scores}
    return pred_dict


def extract_vals(predictions: List[Dict], key: str) -> List:
    """Used for extracting accuracies from list of predicted dicts"""
    return [d[key] for d in predictions]


def inference(
    config: Dict,
    path_image: str,
    path_graph: str,
    base_output_dir: str,
    verbose: int = 0,
    augs_names: List[str] = None,
    processor: Processor = None,
) -> Dict:
    """Loads an Image, an EngGraph in Pascal VOC Format, and a Model,
    Performs Preprocessing, Applies the Model to all Tensors, Postprocesses them and Saves the Results

    verbose 0 prints just the crucial parts
    verbose 1 prints all
    verbose 2 also saves files in the experiment folder (extraction/output)

    augs_names: Optional augmentation to apply to the image before inference
    """
    name = config["training_parameter"]["name"]
    if verbose > 0:
        print(f"Performing {name} inference")

    # Load the graph
    if verbose > 0:
        print(f"Loading Graph... (from {path_graph})")
    true_graph: EngGraph = PascalVocConverter().load(path_graph)
    pred_graph = true_graph.create_empty_copy()

    if verbose > 0:
        print(f"Loading Image... (from {path_image})")
    image = cv2.imread(path_image)

    # Store original image for visualization
    original_image = image.copy()

    if verbose > 1:
        # create output subfolder with timestamp
        output_dir = create_subdir_timestamp(base_output_dir, "infI_", verbose)

        # render and save true bboxes over image
        true_img_with_bboxes = draw_and_save_bboxes(
            save_path=os.path.join(output_dir, f"bboxes_true.png"),
            image=original_image,
            graph=true_graph,
            verbose=verbose,
        )

    # build the processor if we dont have it
    if processor == None:
        processor, _ = build_processor_model(config=config, device="cpu")

    # run the actual inference
    pred_list = processor.inference(image=image, graph=pred_graph)
    pred_dict = get_pred_dict(pred_list)
    pred_dict = processor.filter_pred_dict_using_score(pred_dict)

    if verbose > 0:
        print("EVALUATING...")

    # get true bboxes and labels and compute accuracy (mAP)
    target_dict = true_graph.get_boxes_labels(processor.get_class_id)
    acc = compute_map_single(pred_dict, target_dict)
    print(f"--> mAP Accuracy = {acc} (T={augs_names}) <--")
    pred_dict["accuracy"] = acc
    if verbose > 1:
        with open(os.path.join(output_dir, f"accuracy_{acc:.2f}.txt"), "w") as f:
            f.write(f"{acc}\n")

    if verbose > 0:
        print_num_pred_bboxes(pred_dict, target_dict)

    if verbose > 1:
        # copy original image
        copy2(path_image, os.path.join(output_dir, os.path.basename(path_image)))
        print(f"Copied original image to {output_dir}")

        # render and save predicted bboxes over image
        pred_img_with_bboxes = draw_and_save_bboxes(
            os.path.join(output_dir, f"bboxes_pred.png"),
            image=image,
            graph=pred_graph,
            verbose=verbose,
        )

        # combine true vs. predicted and save comparison
        save_comparison_image(
            true_img_with_bboxes,
            pred_img_with_bboxes,
            os.path.join(output_dir, f"comparison_{name}.png"),
            note1="ground truth",
            note2=f"pred_{augs_names}" if augs_names else "prediction",
        )

        print("Saving Graph...")
        PascalVocConverter().store(
            pred_graph, os.path.join(output_dir, f"output_{name}.xml")
        )
        # JSONConverter().store(
        #     pred_graph, os.path.join(output_dir, f"output_{name}.json")
        # )
        # SvgConverter().store(pred_graph, os.path.join(output_dir, f"output_{name}.svg"))
        print(f"Saved XML and other to {output_dir}")
        # png converter doesnt work

    if verbose > 0:
        print("Done.")

    return pred_dict


def inference_drafter(
    config: Dict,
    base_drafter_dir: str,
    base_output_dir: str,
    verbose: int = 0,
    processor: Processor = None,
    augs_names: List[str] = None,
    ensemble_fn_name: str = "wbf",
    iou_threshold: float = 0.5,
) -> List[Dict]:
    """Performs inference on all images in a drafter directory, applying either standard
    inference or test-time augmentation based on the provided augmentation names.

    Returns a list of prediction dictionaries containing results for each processed image.
    """
    if processor is None:
        processor, _ = build_processor_model(config)

    images_dir: str = os.path.join(base_drafter_dir, "images")
    num_images: int = len(os.listdir(images_dir))
    i = 0

    predictions: List[float] = []
    # for each image in the directory
    for img_name in os.listdir(images_dir):
        i += 1
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path_image = os.path.join(images_dir, img_name)
        path_ann = get_ann_path_from_img_path(path_image)
        if not os.path.exists(path_ann):
            print(f"Warning: no annotation for {path_image}, skipping")
            continue

        print(f"\n---- ({i}/{num_images}) Inference on {path_image} ----")
        preds = inference(
            config,
            path_image,
            path_ann,
            verbose=verbose,
            base_output_dir=base_output_dir,
            processor=processor,
        )

        predictions.append(preds)

    if not predictions:
        print("No valid images found, cannot compute accuracies.")
        sys.exit(1)

    return predictions


def get_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Perform model inference on images with optional test-time augmentations"
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to configuration JSON file"
    )
    parser.add_argument(
        "-i", "--image", help="Path to a single image file", default=None
    )
    parser.add_argument(
        "--dir",
        help="Directory containing images and annotations folders",
        default=None,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Verbosity level: 0=minimal print, 1=all print, 2=detailed with file saving",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Optional note to include in the output summary",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=os.path.join("extraction", "output"),
        help="Base output directory for saving inference results",
    )

    parser.add_argument(
        "-iou",
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for ensembling",
    )
    parser.add_argument(
        "--all_drafters",
        action="store_true",
        default=False,
        help="Process all drafters found in the directory (rather than just the test set)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Scale factor for custom scale augmentation (when 'scale' is in --augmentations)",
    )
    parser.add_argument(
        "--gauss-kernel",
        "-k",
        type=int,
        default=7,
        help="Kernel size for gaussian blur augmentation (must be odd, when 'gaussian_blur' is in --augmentations)",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        default=False,
        help="Enable grid search mode to try multiple combinations of augmentations",
    )

    return parser


if __name__ == "__main__":
    import argparse

    # Create argument parser
    parser: argparse.ArgumentParser = get_parser()
    args = parser.parse_args()
    verbose = args.verbose
    augs_names = args.augmentations

    # Load config
    config = load_config(load_path=args.config)

    # ==== Process single image ====
    if args.image:
        # Get annotation path
        path_ann = get_ann_path_from_img_path(args.image)

        pred_dict = inference(
            config=config,
            path_image=args.image,
            path_graph=path_ann,
            base_output_dir=args.output_dir,
            verbose=verbose,
            augs_names=augs_names,
            # processor will be automatically loaded from config
        )

    # ==== Process directory of images ====
    elif args.dir:
        # Find all subdirectories in dir_path:
        subdirs = get_subdir_basenames(args.dir)

        # Filter to only include directories that start with "drafter_"
        drafters_basenames = [d for d in subdirs if d.startswith("drafter_")]

        # ==== Process drafter ====
        if "images" in subdirs and "annotations" in subdirs:
            base_output_dir = create_subdir_timestamp(args.output_dir, "infD_", verbose)

            # Save config JSON to the output directory
            save_config(
                config=config, save_path=os.path.join(base_output_dir, "config.json")
            )

            predictions = inference_drafter(
                config=config,
                base_drafter_dir=args.dir,
                base_output_dir=base_output_dir,
                augs_names=augs_names,
                verbose=verbose,
                # processor will be automatically loaded from config
            )

            print(f"==> {args.dir} <==")
            print_accuracy_summary(acc_arr=extract_vals(predictions, "accuracy"))

        # ==== Process dataset (all drafters) ====
        elif drafters_basenames:
            # Create experiment folder
            base_output_dir = create_subdir_timestamp(args.output_dir, "infF_", verbose)

            # Save config JSON to the output directory
            save_config(
                config=config, save_path=os.path.join(base_output_dir, "config.json")
            )

            processor, _ = build_processor_model(config)

            test_drafters = config["training_parameter"]["drafter_set_test"]

            total_accuracies: Dict = {}
            for drafter_basename in drafters_basenames:
                drafter_number = int(drafter_basename.split("_")[1])
                if not args.all_drafters and drafter_number not in test_drafters:
                    if verbose > 0:
                        print(f"Skipping {drafter_basename} (not in test_drafters)")
                    continue

                predictions = inference_drafter(
                    config=config,
                    base_drafter_dir=os.path.join(args.dir, drafter_basename),
                    base_output_dir=(
                        create_subdir_timestamp(base_output_dir, "infD_", verbose)
                        if verbose > 1
                        else None
                    ),
                    augs_names=augs_names,
                    processor=processor,
                    verbose=verbose,
                )
                total_accuracies[drafter_basename] = extract_vals(
                    predictions, "accuracy"
                )

                print(f"\n====> {args.dir}/{drafter_basename} <====")
                print_accuracy_summary(acc_arr=total_accuracies[drafter_basename])

            # Calculate overall stats and save summary
            output_file = os.path.join(base_output_dir, f"accuracy_summary.txt")
            note = f"augmentations={augs_names}\n{args.note}\n"
            save_accuracy_summary(
                total_accuracies, config, args.dir, output_file=output_file, note=note
            )

            # Print overall statistics to terminal
            all_accuracies = [acc for accs in total_accuracies.values() for acc in accs]
            print("\n=== OVERALL STATISTICS ===")
            print_accuracy_summary(all_accuracies)

        else:
            print("Wrong input")

    else:
        print("Error: Must provide either --dir or --image")
        parser.print_help()
        sys.exit(1)

"""
SINGLE CGHD IMAGE
poetry run python -m extraction.src.core.inference_OD -c extraction/config/object_detection_inf_cghd.json -i data/cghd_sample/drafter_1/images/C1_D2_P1.jpg -v 2

SINGLE RPi IMAGE
poetry run python -m extraction.src.core.inference_OD -c extraction/config/object_detection_inf_rpi.json -i data/rpi_pico_sample/drafter_31/images/48c9152c-0-1.jpg -v 1

DIFFICULT BIGGEST RPi
poetry run python -m extraction.src.core.inference_OD -c extraction/config/object_detection_inf_rpi.json -i data/rpi_pico_sample/drafter_31/images/af51898d-18-7.jpg -v 2

WHOLE FOLDER
poetry run python -m extraction.src.core.inference_OD -c extraction/config/object_detection_inf_rpi.json --dir data/rpi_pico_sample/drafter_31/ -v 2

WHOLE DATASET
poetry run python -m extraction.src.core.inference_OD -c extraction/config/object_detection_inf_cghd.json --dir data/cghd_raw -v 0

poetry run python -m extraction.src.core.inference_OD -c extraction/config/object_detection_inf_rpi.json --dir data/rpi_pico_sample/ -v 0
"""
