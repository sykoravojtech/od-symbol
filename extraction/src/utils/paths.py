import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

__author__ = "Vojtěch Sýkora"


def is_folder(path: str) -> bool:
    """Check if path is an existing directory."""
    return Path(path).is_dir()


def is_image_file(path: str) -> bool:
    """Check if path is a file with a valid image extension."""
    p = Path(path)
    if not p.is_file():
        return False
    return p.suffix.lower() in {".jpg", ".jpeg", ".png"}


def classify_path(path: str) -> str:
    """Classify path as folder, image file, or invalid."""
    if is_folder(path):
        return "folder"
    if is_image_file(path):
        return "image"
    return "invalid"


def parent_basename(path: str) -> str:
    """Basename of parent directory."""
    return os.path.basename(os.path.dirname(path))


def grandparent_basename(path: str) -> str:
    """Basename of parent directory."""
    return os.path.basename(os.path.dirname(os.path.dirname(path)))


def parent_path(path: str) -> str:
    return os.path.dirname(path)


def grandparent_path(path: str) -> str:
    return os.path.dirname(os.path.dirname(path))


def base_name_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def get_ann_path_from_img_path(img_path: str) -> str:
    img_name = base_name_no_ext(img_path)
    ann_dir = os.path.join(grandparent_path(img_path), "annotations")
    path_ann = os.path.join(ann_dir, f"{img_name}.xml")

    if not os.path.exists(path_ann):
        print(f"Error: annotation file {path_ann} not found")
        sys.exit(1)

    return path_ann


def get_subdir_basenames(dir_path: str) -> List[str]:
    subdirs = [p.name for p in Path(dir_path).iterdir() if p.is_dir()]
    return subdirs


def create_subdir_timestamp(
    base_output_dir: str, dir_name_start: str, verbose: int = 0
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
        :19
    ]  # Include milliseconds and truncate to avoid too long names
    output_dir = os.path.join(base_output_dir, f"{dir_name_start}{ts}")
    os.makedirs(output_dir, exist_ok=True)
    if verbose > 0:
        print(f"Saving all outputs in {output_dir}")
    return output_dir
