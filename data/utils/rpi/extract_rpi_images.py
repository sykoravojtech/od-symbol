import base64
import hashlib
import io
import json
import os
import shutil
from typing import Dict, List, Optional

import cv2
import numpy as np
from data.utils.utils import generate_short_hash
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import BoundingBox
from PIL import Image

__author__ = "Vojtěch Sýkora"


class ImageMetadata:
    def __init__(
        self,
        caption=None,
        rel_path="",
        id="",
        page=-1,
        bbox: BoundingBox = None,
        size: dict = None,
        dpi: int = None,
    ):
        self.caption = caption if caption else []
        self.rel_path = (rel_path,)  # relative path to the image
        self.id = id
        self.page = page
        self.size = size
        self.dpi = dpi
        self.bbox = (
            bbox
            if isinstance(bbox, BoundingBox)
            else BoundingBox(**bbox) if isinstance(bbox, dict) else None
        )

    def to_dict(self):
        return {
            "caption": self.caption,
            "rel_path": self.rel_path,
            "id": self.id,
            "page": self.page,
            "size": self.size,
            "bbox": self.bbox.model_dump() if self.bbox else None,
        }


class DatasheetsModifications:
    def __init__(self, data_mods: dict):
        self.data_mods = data_mods

    def get(self, document_name: str, default=None):
        return self.data_mods.get(document_name, default)

    def includes(self, document_name: str):
        doc = self.data_mods.get(document_name)
        return doc != {"images": []}

    def get_ids_of_good_imgs(self, document_name: str):
        imgs = self.data_mods.get(document_name, {}).get("images", [])
        ids = [img.get("id", -1) for img in imgs]
        return ids

    def get_img_data(self, document_name: str, img_id: int):
        imgs = self.data_mods.get(document_name, {}).get("images", [])
        for img in imgs:
            if img.get("id", -1) == img_id:
                return img
        return None

    def get_rotation_of_img(self, document_name: str, img_id: int):
        img_data = self.get_img_data(document_name, img_id)
        return img_data.get("rotation", 0) if img_data else 0

    def get_subschematics_of_img(self, document_name: str, img_id: int):
        img_data = self.get_img_data(document_name, img_id)
        return img_data.get("subschematics", []) if img_data else []


def decode_image_from_uri(uri: str, rotation: float = 0.0) -> Image:
    """
    Decodes an image from a base64 URI and applies an optional rotation.
    """
    # check if the URI is in abse64 format
    prefix = "data:image/png;base64,"
    if not uri.startswith(prefix):
        raise ValueError("Unsupported image URI format.")

    # remove the prefix
    base64_data = uri[len(prefix) :]

    image_data = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_data))

    if rotation != 0:
        image = image.rotate(rotation, expand=True)

    return image


def save_img_from_uri(uri: str, out_file: str, rotation: float = 0.0) -> None:
    """
    Saves an image from a URI to a file.
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    try:
        image = decode_image_from_uri(uri, rotation)
    except ValueError as e:
        print(f"Skipping picture ({out_file}): {e}")
        return

    image.save(out_file)
    print(f"Saved image: {out_file}, rotated by {rotation}°.")


def save_subschematics_from_uri(
    uri: str,
    out_dir: str,
    base_filename: str,
    subschematics: list,
    rotation: float = 0.0,
    bg_color: np.ndarray = np.array([255, 255, 255], dtype=np.uint8),
) -> None:
    """
    Decodes an image from its URI and saves each subschematic (cropped via bounding box coordinates)
    as a separate file.

    1. extract region defined by polygon
    2. apply mask so only the region inside the polygon is visible and the rest is made white
    3. crop the image to the bounding rectangle of the polygon
    """
    os.makedirs(out_dir, exist_ok=True)

    try:
        image = decode_image_from_uri(uri, rotation)
    except ValueError as e:
        print(f"Skipping subschematics for {base_filename}: {e}")
        return

    # Convert to a NumPy array for cropping with OpenCV.
    image_np = np.array(image)

    for subs in subschematics:
        subs_id = subs.get("id", "")
        pts = subs.get("points")
        if not pts:
            print(
                f"Skipping subschematic {subs_id} in {base_filename}: no points provided"
            )
            continue

        # Convert the points list to a NumPy array.
        points = np.array(pts, dtype=np.int32)
        # If only 2 points are provided, assume a rectangle and convert to 4 points.
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            points = np.array(
                [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                dtype=np.int32,
            )

        # Create a mask from the polygon defined by points.
        mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        # Create a white background.
        filler_bg = (
            np.ones_like(image_np) * bg_color
        )  # make a background of the same size as our image

        # Keep the image only inside the polygon and fill outside with white.
        masked = cv2.bitwise_and(image_np, image_np, mask=mask)
        inverse_mask = cv2.bitwise_not(mask)
        filler_bg = cv2.bitwise_and(filler_bg, filler_bg, mask=inverse_mask)
        final = cv2.add(masked, filler_bg)

        # Crop to the bounding rectangle of the polygon.
        x, y, w, h = cv2.boundingRect(points)
        crop = final[y : y + h, x : x + w]

        # Convert from RGB to BGR
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        out_file = os.path.join(out_dir, f"{subs_id}.jpg")
        cv2.imwrite(out_file, crop_bgr)

        print(f"Saved subschematic: {out_file} rotated by {rotation}°. & bg {bg_color}")


def extract_imgs_info_from_pdf(
    pdf_path: str, target_dir: str, data_mods: Optional[DatasheetsModifications] = None
) -> List[ImageMetadata]:
    pipeline_options = PdfPipelineOptions()  # this takes a long time
    pipeline_options.generate_picture_images = True
    pipeline_options.do_ocr = False
    pipeline_options.images_scale = 6.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    converted = converter.convert(source=pdf_path)
    document = converted.document.export_to_dict()

    image_info_list = []
    document_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pictures = document.get("pictures", [])

    # save the docling dict in a txt file
    # with open(f"data/rpi_tobias/{document_name}.txt", "w", encoding="utf-8") as f:
    #     f.write(json.dumps(pictures, indent=4))
    # exit()

    print(f"Extracting {len(pictures)} images")
    for pic in pictures:
        captions = [
            c.get("$ref").replace("#", document_name) for c in pic.get("captions", [])
        ]
        page = pic.get("prov", [{}])[0].get("page_no", -1)
        bbox = pic.get("prov", [{}])[0].get("bbox", None)
        rel_path = pic.get("self_ref", "").replace("#", document_name)
        img_id_str = rel_path.rpartition("/")[-1]
        img_id = int(img_id_str) if img_id_str.isdigit() else None
        size = pic.get("image", {}).get("size", None)
        dpi = pic.get("image", {}).get("dpi", None)
        uri = pic.get("image", {}).get("uri", "")

        # Apply filtering based on img_filter
        if data_mods is not None:
            allowed_imgs = data_mods.get_ids_of_good_imgs(document_name)
            if not allowed_imgs or (img_id is not None and img_id not in allowed_imgs):
                continue  # Skip images that are not in the filter

        image_info_list.append(
            ImageMetadata(
                caption=captions,
                rel_path=rel_path,
                id=img_id,
                page=page,
                bbox=bbox,
                size=size,
                dpi=dpi,
            )
        )

        # Determine the rotation (if any) from the modifications.
        rotation = (
            data_mods.get_rotation_of_img(document_name, img_id) if data_mods else 0
        )
        # Compute output file path for the original image.
        out_file_path = f"{os.path.join(target_dir, rel_path)}.jpg"

        # Check if this image has subschematics defined.
        img_data = data_mods.get_img_data(document_name, img_id) if data_mods else None
        if img_data and "subschematics" in img_data:
            # Instead of saving the original image, decode the URI and extract subschematics.
            # We use the base filename for example "18" for image id 18
            base_filename = os.path.splitext(os.path.basename(out_file_path))[0]
            # Save each subschematic crop to the same folder as the original image.
            subs_out_dir = os.path.dirname(out_file_path)

            bg_color = img_data.get(
                "bg_color", [255, 255, 255]
            )  # datasheet_modification.json has optional bg_color
            bg_color = np.array(bg_color, dtype=np.uint8)

            save_subschematics_from_uri(
                uri,
                subs_out_dir,
                base_filename,
                img_data["subschematics"],
                rotation,
                bg_color,
            )
        else:
            # Save the original image if no subschematics are provided.
            save_img_from_uri(uri, out_file_path, rotation)

    return image_info_list


def process_pdf_to_json(
    pdf_path: str, target_dir: str, data_mods: Optional[DatasheetsModifications] = None
):
    if data_mods is not None:
        print("Using filter")
    else:
        print("No filter")

    # Extract all images from the PDF and save the png files
    images = extract_imgs_info_from_pdf(pdf_path, target_dir, data_mods)
    image_data = [img.to_dict() for img in images]

    # Save the image data to a JSON file
    datasheet_name = pdf_path.split("/")[-1].split(".")[0]
    json_path = os.path.join(target_dir, datasheet_name, datasheet_name + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"pictures": image_data}, f, indent=4)

    # copy pdf file to the target dir
    pdf_target_path = os.path.join(
        target_dir, datasheet_name, os.path.basename(pdf_path)
    )
    with open(pdf_path, "rb") as src_pdf, open(pdf_target_path, "wb") as dst_pdf:
        dst_pdf.write(src_pdf.read())

    print(
        f"Processed {pdf_path} and saved results to {os.path.join(target_dir, datasheet_name)}"
    )


def process_all_pdfs_in_dir(
    input_dir: str, target_dir: str, data_mods: Optional[DatasheetsModifications] = None
):
    print(os.listdir(input_dir))
    for pdf_file in os.listdir(input_dir):
        # if pdf_file has no good images, skip it
        if data_mods is not None:
            filename = pdf_file.split(".")[0]
            if not data_mods.includes(filename):
                print(f"\n==> Skipping {pdf_file}...")
                continue

        print(f"\n==> Processing {pdf_file}...")
        if pdf_file.endswith(".pdf"):
            process_pdf_to_json(
                os.path.join(input_dir, pdf_file), target_dir, data_mods
            )


def move_dirs_to_drafter(
    input_dir: str = "data/rpi_images_filt", target_dir: str = "data/rpi_pico_sample"
):
    # Hardcoded list of allowed image filenames (without extensions)
    allowed_images = {
        "3233d615-21",
        "3233d615-24-0",
        "3233d615-24-5",
        "3233d615-24-6",
        "48c9152c-0-1",
        "48c9152c-0-2",
        "48c9152c-0-4",
        "48c9152c-0-6",
        "48c9152c-0-7",
        "8a23460a-1",
        "af51898d-10",
        "af51898d-15",
        "af51898d-18-0",
        "af51898d-18-4",
        "af51898d-18-5",
        "af51898d-18-7",
        "afe737fb-11",
        "afe737fb-16-2",
        "afe737fb-16-5",
        "afe737fb-16-6",
        "afe737fb-16-7",
        "afe737fb-17-0",
    }

    # Make the target directories
    target_imgs_dir: str = os.path.join(target_dir, "drafter_31", "images")
    os.makedirs(target_imgs_dir, exist_ok=True)

    hash_map: Dict[str, str] = {}  # datasheet name hash: datasheet name

    # copy all images to one target directory with hashed datasheet names
    for dir in os.listdir(input_dir):
        imgs_dir: str = os.path.join(input_dir, dir, "pictures")
        hash_map[generate_short_hash(dir)] = dir

        for img in os.listdir(imgs_dir):
            # Filter: only process images in the allowed list
            img_name_without_ext = os.path.splitext(img)[0]
            if img_name_without_ext not in allowed_images:
                continue

            img_src_path: str = os.path.join(imgs_dir, img)
            datasheet_name_hash: str = generate_short_hash(dir)
            img_target_path: str = os.path.join(
                target_imgs_dir, f"{datasheet_name_hash}-{img}"
            )
            shutil.copy2(img_src_path, img_target_path)
            print(f"{img_src_path} -> {img_target_path}")

    print("-- Making RPi hash_map")
    # make a guide to the hashed datasheet names
    mapping_path = os.path.join(target_dir, "drafter_31", "hash_map.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(hash_map, f, indent=2)


if __name__ == "__main__":
    # Process one pdf
    # input_pdf = "data/rpi_tobias/pico_pico-2-datasheet.pdf"
    # process_pdf_to_json(input_pdf, target_dir="data/rpi_images")

    # Process all pdfs in a directory
    # process_all_pdfs_in_dir(
    #     input_dir="data/rpi_datasheets/pdf_pico", target_dir="data/rpi_images", data_mods=None
    # )

    # Process all pdfs in a directory with datasheet modifications
    with open(
        "data/utils/rpi/datasheets_modifications.json", "r", encoding="utf-8"
    ) as f:
        data_mods = DatasheetsModifications(json.load(f))

    process_all_pdfs_in_dir(
        input_dir="data/rpi_datasheets/pdf_pico",
        target_dir="data/rpi_images_filt",
        data_mods=data_mods,
    )

    move_dirs_to_drafter(
        input_dir="data/rpi_images_filt", target_dir="data/rpi_pico_sample"
    )


"""
Show sizef of images in the current folder
identify -format "%wx%h %f\n" *.jpg
"""
