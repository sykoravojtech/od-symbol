import os
import shutil
import sys
import xml.etree.ElementTree as ET


def organize_dataset(root_dir: str) -> None:
    """Organize files into images/ and annotations/, renaming based on prefix."""
    base_folder = os.path.basename(os.path.normpath(root_dir))
    images_dir = os.path.join(root_dir, "images")
    annotations_dir = os.path.join(root_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    for entry in os.listdir(root_dir):
        src_path = os.path.join(root_dir, entry)
        if not os.path.isfile(src_path):
            continue
        name, ext = os.path.splitext(entry)
        prefix = name.split("_")[0]

        if ext.lower() == ".jpg":
            new_name = prefix + ".jpg"
            dst = os.path.join(images_dir, new_name)
            shutil.move(src_path, dst)

        elif ext.lower() == ".xml":
            # parse and update XML
            tree = ET.parse(src_path)
            root = tree.getroot()

            fname_tag = root.find("filename")
            if fname_tag is not None:
                fname_tag.text = prefix + ".jpg"

            path_tag = root.find("path")
            if path_tag is not None:
                path_tag.text = f"./{base_folder}/images/{prefix}.jpg"

            # write updated XML to annotations/
            new_xml = prefix + ".xml"
            dst = os.path.join(annotations_dir, new_xml)
            tree.write(dst)

            os.remove(src_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dataset_directory>")
        sys.exit(1)
    organize_dataset(sys.argv[1])

"""
poetry run python -m data.utils.rpi.roboflow_preprocess /home/vojta/Documents/cg/cg25/data/rpi_pico_sample/drafter_31
"""
