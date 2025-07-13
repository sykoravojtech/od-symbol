import argparse
import os
from statistics import median
from typing import Dict, List, Tuple

import numpy as np
from converter.converter.pascalVocConverter import PascalVocConverter
from converter.core.engineeringGraph import EngGraph

__author__ = "Vojtěch Sýkora"


def get_bbox_stats_from_dir(ann_dir: str, verbose: int = 0) -> Dict:
    # Check if it is even a directory
    if not os.path.isdir(ann_dir):
        raise ValueError(f"Direcotry not found {ann_dir}!")

    # Get all xml files from ann_dir
    ann_files = [a for a in os.listdir(ann_dir) if a.lower().endswith(".xml")]

    if len(ann_files) == 0:
        print("No XML files found")
        return {}

    # Extract bbox numbers from all files
    converter = PascalVocConverter()
    bbox_counts = []

    for file in ann_files:
        # Load the xml graph using the converter
        ann_path = os.path.join(ann_dir, file)
        xml_graph: EngGraph = converter.load(ann_path)
        if verbose > 0:
            print(f"File {file} bboxes {len(xml_graph.nodes)}")
        # Count number of bboxes in the file
        bbox_counts.append((file, len(xml_graph.nodes)))

    # Calculate the min and max
    min_bboxes = (None, float("inf"))  # file, number
    max_bboxes = (None, 0)

    for file, bbox_count in bbox_counts:
        if bbox_count > max_bboxes[1]:
            max_bboxes = (file, bbox_count)
        if 0 < bbox_count < min_bboxes[1]:
            min_bboxes = (file, bbox_count)

    # Calculate the avg and median and total
    just_counts = [count for _, count in bbox_counts]
    total_bboxes = sum(just_counts)
    avg_bboxes = total_bboxes / len(just_counts)
    median_bboxes = median(just_counts)

    stats = {
        "min_bboxes": min_bboxes,
        "max_bboxes": max_bboxes,
        "total_bboxes": total_bboxes,
        "avg_bboxes": avg_bboxes,
        "median_bboxes": median_bboxes,
        "file_count": len(bbox_counts),
        "bbox_counts": bbox_counts,
    }

    return stats


def process_drafter(drafter_path: str, verbose: int) -> Dict:
    return get_bbox_stats_from_dir(os.path.join(drafter_path, "annotations"), verbose)


def print_bbox_stats(dir_name: str, stats: dict, verbose: int) -> None:
    print("-----")
    print(f"{dir_name} bbox stats:")
    if verbose > 0:
        print(
            f"Minimum bounding boxes: {stats['min_bboxes'][1]} in file {stats['min_bboxes'][0]}"
        )
        print(
            f"Maximum bounding boxes: {stats['max_bboxes'][1]} in file {stats['max_bboxes'][0]}"
        )
        print(f"Total bounding boxes: {stats['total_bboxes']}")
        print(f"Average bounding boxes: {stats['avg_bboxes']:.1f}")
        print(f"Median bounding boxes: {stats['median_bboxes']}")
        print(f"Total files: {stats['file_count']}")
    else:
        print(
            f"min: {stats['min_bboxes'][1]} | max: {stats['max_bboxes'][1]} | tot: {stats['total_bboxes']} | avg: {stats['avg_bboxes']:.1f} | med: {stats['median_bboxes']} | files: {stats['file_count']}"
        )
    print("-----")


def main(dir_path: str, verbose: int) -> None:
    # Check if it is even a directory
    if not os.path.isdir(dir_path):
        raise ValueError(f"Direcotry not found {dir_path}!")

    # Check if directory is an annotations directory
    if os.path.basename(dir_path) == "annotations":
        stats = get_bbox_stats_from_dir(dir_path, verbose)
        print_bbox_stats(dir_path, stats, verbose)

    # Check if directory is a drafter directory
    elif os.path.basename(dir_path).startswith("drafter_"):
        stats = process_drafter(dir_path, verbose)
        print_bbox_stats(dir_path, stats, verbose)

    # It is the whole dataset folder
    else:
        # Find all subdirectories in dir_path:
        subdirs = [
            os.path.join(dir_path, d)
            for d in os.listdir(dir_path)
            if os.path.isdir(os.path.join(dir_path, d))
        ]

        # Filter to only include directories that start with "drafter_"
        drafter_dirs = [
            d for d in subdirs if os.path.basename(d).startswith("drafter_")
        ]

        if not drafter_dirs:
            print(f"No drafter directories found in {dir_path}")
            return

        if verbose > 0:
            print(f"Found {len(drafter_dirs)} drafter directories")

        # Process each drafter directory
        all_stats = []
        for drafter_dir in drafter_dirs:
            drafter_name = os.path.basename(drafter_dir)
            stats = process_drafter(drafter_dir, verbose)
            all_stats.append((drafter_name, stats))
            if verbose > 0:
                print_bbox_stats(drafter_dir, stats, verbose)

        # Calculate global statistics
        all_bbox_counts = []
        global_min_bboxes = (None, None, float("inf"))  # drafter, file, number
        global_max_bboxes = (None, None, 0)  # drafter, file, number
        global_total_bboxes = 0
        global_file_count = 0

        # Process each drafter's statistics
        for drafter_name, stats in all_stats:
            # Update global min/max
            drafter_min = stats["min_bboxes"]
            drafter_max = stats["max_bboxes"]

            if drafter_min[1] < global_min_bboxes[2]:
                global_min_bboxes = (drafter_name, drafter_min[0], drafter_min[1])

            if drafter_max[1] > global_max_bboxes[2]:
                global_max_bboxes = (drafter_name, drafter_max[0], drafter_max[1])

            # Update global counts
            global_total_bboxes += stats["total_bboxes"]
            global_file_count += stats["file_count"]

            # Collect all bbox counts for median/avg calculation
            all_bbox_counts.extend([count for _, count in stats["bbox_counts"]])

        # Calculate global average and median
        global_avg_bboxes = (
            global_total_bboxes / len(all_bbox_counts) if all_bbox_counts else 0
        )
        global_median_bboxes = median(all_bbox_counts) if all_bbox_counts else 0

        # Format global statistics
        global_stats = {
            "min_bboxes": (
                f"{global_min_bboxes[0]}:{global_min_bboxes[1]}",
                global_min_bboxes[2],
            ),
            "max_bboxes": (
                f"{global_max_bboxes[0]}:{global_max_bboxes[1]}",
                global_max_bboxes[2],
            ),
            "total_bboxes": global_total_bboxes,
            "avg_bboxes": global_avg_bboxes,
            "median_bboxes": global_median_bboxes,
            "file_count": global_file_count,
        }

        # Print global statistics
        print("\n --- Global Statistics ---")
        print_bbox_stats(dir_path, global_stats, verbose)


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(description="Process bounding box statistics.")
        parser.add_argument(
            "--dir", type=str, required=True, help="Directory path to process"
        )
        parser.add_argument(
            "--verbose",
            "-v",
            type=int,
            default=0,
            choices=[0, 1, 2],
            help="Verbosity level: 0=minimal, 1=standard, 2=detailed",
        )
        return parser.parse_args()

    args = parse_args()

    if args.verbose > 0:
        print(f"Processing directory: {args.dir}")

    main(dir_path=args.dir, verbose=args.verbose)

"""
Examples:
# Process a single annotations directory with minimal output
poetry run python -m extraction.src.utils.bbox_stats --dir data/rpi_pico_sample/drafter_31/annotations -v 0

# Process a single drafter directory with standard verbosity
poetry run python -m extraction.src.utils.bbox_stats --dir data/rpi_pico_sample/drafter_31 -v 0

# Process a complete dataset with detailed verbosity
poetry run python -m extraction.src.utils.bbox_stats --dir data/cghd_raw -v 0

"""

"""
data/cghd_raw bbox stats:
min: 6 | max: 542 | tot: 245962 | avg: 77.5 | med: 50 | files: 3173

data/rpi_pico_sample bbox stats:
min: 16 | max: 436 | tot: 1675 | avg: 76.1 | med: 41.5 | files: 22

"""

"""
data/cghd_raw bbox stats:
Minimum bounding boxes: 6 in file drafter_-1:C-22_D1_P7.xml
Maximum bounding boxes: 542 in file drafter_21:C248_D1_P3.xml
Total bounding boxes: 245962
Average bounding boxes: 77.5
Median bounding boxes: 50
Total files: 3173

data/rpi_pico_sample bbox stats:
Minimum bounding boxes: 16 in file drafter_31:3233d615-24-6.xml
Maximum bounding boxes: 436 in file drafter_31:af51898d-18-7.xml
Total bounding boxes: 1675
Average bounding boxes: 76.1
Median bounding boxes: 41.5
Total files: 22
"""
