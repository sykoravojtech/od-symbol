import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

__author__ = "Vojtěch Sýkora"

# I had issues with closing the window and this solved it
# Global flag for mouse click
clicked_flag: bool = False


def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
    """Mouse callback to update clicked flag."""
    global clicked_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_flag = True


def parse_voc_annotation(xml_file: Path) -> List[Tuple[str, int, int, int, int]]:
    """Parse VOC XML file."""
    objects: List[Tuple[str, int, int, int, int]] = []
    tree = ET.parse(str(xml_file))
    root = tree.getroot()
    for obj in root.findall("object"):
        name = obj.find("name").text if obj.find("name") is not None else "object"
        bndbox = obj.find("bndbox")
        if bndbox is not None:
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            objects.append((name, xmin, ymin, xmax, ymax))
    return objects


def find_image_file(image_dir: Path, base_filename: str) -> Optional[Path]:
    """Find corresponding image file."""
    valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for ext in valid_exts:
        candidate = image_dir / f"{base_filename}{ext}"
        if candidate.exists():
            return candidate
    return None


def visualize_annotations(
    image_path: Path, annotations: List[Tuple[str, int, int, int, int]]
) -> None:
    """Visualize image with annotations and wait for key or click."""
    global clicked_flag
    clicked_flag = False
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image {image_path}")
        return
    for name, xmin, ymin, xmax, ymax in annotations:
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            image,
            name,
            (xmin, max(ymin - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )
    window_name = f"Annotations - {image_path.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1800, 1000)
    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, on_mouse)

    # Wait until a key is pressed, the mouse is clicked, or the window is closed.
    while True:
        key = cv2.waitKey(100)
        if key != -1 or clicked_flag:
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(window_name)


def main() -> None:
    """Main execution."""
    base_dir = Path("data/cghd_sample")  # Replace with your actual base directory path
    drafter_folders = list(base_dir.glob("drafter_*"))
    if not drafter_folders:
        print("No drafter folders found.")
        return
    for folder in drafter_folders:
        annotations_dir = folder / "annotations"
        images_dir = folder / "images"
        if not annotations_dir.exists() or not images_dir.exists():
            print(f"Skipping {folder}, missing annotations or images folder.")
            continue
        for xml_file in annotations_dir.glob("*.xml"):
            base_filename = xml_file.stem
            image_file = find_image_file(images_dir, base_filename)
            if image_file is None:
                print(f"No image found for annotation {xml_file}")
                continue
            annotations = parse_voc_annotation(xml_file)
            visualize_annotations(image_file, annotations)


if __name__ == "__main__":
    main()
    main()
