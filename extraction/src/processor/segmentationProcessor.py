"""segmentationProcessor.py: Processor for Semantic Segmentation on Patches"""

# System Imports
from typing import Tuple, List, Union
from random import randint
from math import ceil

# Project Imports
from converter.core.utils import pos_to_bbox
from converter.core.engineeringGraph import EngGraph
from extraction.src.core.processor import Processor

# Third-Party Imports
import cv2
import torch
import numpy as np
from torch import Tensor
from torchvision import transforms

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2023, DFKI. 2024-2025, Johannes Bayer"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"



class GaussianNoise(object):  # TODO CAREFULLY move this an similar definitions to utils

    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            return img

        return img + torch.randn(img.shape) * 0.15


class SegmentationProcessor(Processor):
    """Processor for Semantic Segmentation on Patches"""

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    transform_augment = transforms.Compose([GaussianNoise(0.5),
                                            transforms.RandomGrayscale(),
                                            transforms.RandomInvert()])

    def __init__(self, augment: bool, train: bool, debug: bool, patch_size: int, augment_samples_per_image: int, context: bool):
        super().__init__(augment, train, debug)

        self.context = context
        self.patch_size = patch_size
        self.augment_samples_per_image = augment_samples_per_image


    def pre_process(self, img_raw: np.ndarray, img_map: np.ndarray) -> List[Tuple[Tensor, Tensor, any]]:
        """Turns ONE Raw Image and ONE Graph Structure into a List of Tuples of Input and Target/Info Tensors"""

        samples = [self.sample_likewise(img_raw, img_map, offset_x=offset_x, offset_y=offset_y)
                   for offset_x, offset_y in self.tile_image(img_raw)]

        if self.train and self.augment:
            img_height, img_width, _ = img_raw.shape
            samples += [self.sample_likewise(img_raw, img_map,
                                             offset_x=randint(0, img_width - self.patch_size),
                                             offset_y=randint(0, img_height - self.patch_size),
                                             rotation=randint(0, 3) * 90,
                                             augment_transform=True)
                        for _ in range(self.augment_samples_per_image)]

        return samples


    def post_process(self, pred_list: List[Tuple[Tensor, any]], graph: EngGraph) -> None:
        """Integrates a List of Prediction and Info Tensors into a Graph Structure"""

        segmentation_map = self.ensemble_image(pred_list)
        self.extract_edges(graph, segmentation_map)


    @staticmethod
    def crop_patch(img: np.ndarray, patch_size: int, offset_x: int, offset_y: int, rotation: int, scale: int) -> Tensor:
        """Safely Crops a Patch from an Image (adds padding if patch adrea exceeds image dimensions)
        Warning: Only have powers of 2 as scale and patch_size

        Args:
            img: Original Image
            patch_size: Size (==width==height) of the final Square to be returned
            offset_x: first X to be used (when scale==1)
            offset_y: first Y to be used (when scale==1)
            rotation: Rotation to be applied AFTER cropping
            scale: Extends the Area of the Original image to be cropped"""

        patch = np.ones(shape=tuple((dim if dim_nbr > 1 else int(patch_size*scale))
                                    for dim_nbr, dim in enumerate(img.shape)), dtype=img.dtype)*255

        # Desired Patch Area
        top = int(offset_y + patch_size*((1-scale)/2))
        bottom = int(offset_y + patch_size*((1+scale)/2))
        left = int(offset_x + patch_size*((1-scale)/2))
        right = int(offset_x + patch_size*((1+scale)/2))

        # Area that can be Actually Cropped
        img_height, img_width = img.shape[:2]
        crop_top, crop_bottom = (max(0, min(value, img_height)) for value in [top, bottom])
        crop_left, crop_right = (max(0, min(value, img_width)) for value in [left, right])

        patch[abs(top-crop_top):abs(patch_size*scale-abs(bottom-crop_bottom)),
              abs(left-crop_left):abs(patch_size*scale-abs(right-crop_right))] = img[crop_top:crop_bottom, crop_left:crop_right]

        if not scale == 1:
            patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)

        if rotation == 90:
            patch = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if rotation == 180:
            patch = cv2.rotate(patch, cv2.ROTATE_180)

        if rotation == 270:
            patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)

        return SegmentationProcessor.transform(patch)


    def sample_likewise(self, img_raw: np.ndarray, img_map: np.ndarray, offset_x: int, offset_y: int,
                        rotation: int = 0, zoom: int = 1, augment_transform: bool = False) -> Tuple[Tensor, Tensor, any]:
        """Crops and Processes Patches from two images similarly"""

        patch_raw = SegmentationProcessor.crop_patch(img_raw, self.patch_size, offset_x, offset_y, rotation, zoom)
        patch_map = None

        if type(img_map) == np.ndarray:
            patch_map = SegmentationProcessor.crop_patch(img_map, self.patch_size, offset_x, offset_y, rotation, zoom)

        patch_context = SegmentationProcessor.crop_patch(img_raw, self.patch_size, offset_x, offset_y, rotation, zoom * 2)

        if augment_transform and randint(0, 1):
            patch_raw = SegmentationProcessor.transform_augment(patch_raw)
            patch_context = SegmentationProcessor.transform_augment(patch_context)

        if self.context:
            patch_raw = torch.cat([patch_raw, patch_context], dim=0)

        img_height, img_width, _ = img_raw.shape
        info = (offset_x, offset_y, offset_x+self.patch_size, offset_y+self.patch_size, img_width, img_height)

        return patch_raw, patch_map, info


    def tile_image(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """Tiles the Image in Patches of Internal Size

        Returns:
             a List of offset pairs with full image coverage (Lower/Right patches might exceed image dimensions)"""

        img_height, img_width, _ = img.shape

        return [(x * self.patch_size, y * self.patch_size)
                for x in range(ceil(img_width/self.patch_size))
                for y in range(ceil(img_height/self.patch_size))]


    def ensemble_image(self, pred_list: List[Tuple[Tensor, any]]) -> np.ndarray:
        """Ensembles and Thresholds an Image from a List of Enriched Patches"""

        _, _, _, _, img_width, img_height = pred_list[0][1]
        seg_map = np.ones((self.patch_size*(1+(img_height//self.patch_size)),
                           self.patch_size*(1+(img_width//self.patch_size))), dtype=np.float32)

        for tensor, (start_x, start_y, end_x, end_y, _, _) in pred_list:
            seg_map_patch = tensor[0].detach().numpy()
            seg_map[start_y:end_y, start_x:end_x] = np.minimum(seg_map_patch,seg_map[start_y:end_y, start_x:end_x])

        seg_map_thres = (255 * (seg_map > 0)).astype(np.uint8)

        if self.debug:
            cv2.imwrite("0_seg_map.png", (255 * seg_map).astype(np.uint8))
            cv2.imwrite("1_seg_map_thes.png", seg_map_thres)

        return seg_map_thres


    def extract_edges(self, graph: EngGraph, seg_map_thres: np.ndarray,
                      cnt_area_min: int = 50, cnt_area_max: int = 100000, margin: int = 5, add_ports: bool = True):

        node_shapes = []

        for node_id in graph.nodes:
            if graph.nodes[node_id].get("shape"):
                shape = np.array([graph.nodes[node_id]["shape"]], dtype=np.int32)

            else:
                left, right, top, bottom, _, _, _ = pos_to_bbox(graph.nodes[node_id]["position"])
                shape = np.array([[[left, top], [right, top], [right, bottom], [left, bottom]]], dtype=np.int32)

            node_shapes += [(node_id, shape)]
            seg_map_thres = cv2.fillPoly(seg_map_thres, pts=shape, color=255)

        if self.debug:
            cv2.imwrite("2_seg_map_node_free.png", seg_map_thres)

        contours, hierarchy = cv2.findContours(seg_map_thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = [contour for contour in contours if cnt_area_min < cv2.contourArea(contour) < cnt_area_max]

        img_height, img_width = seg_map_thres.shape
        contour_img = cv2.drawContours(np.zeros((img_height, img_width, 3), dtype=np.uint8),
                                       contours, -1, (0, 255, 0), 3)

        if self.debug:
            cv2.imwrite("3_seg_map_contours.png", contour_img)

        for contour in contours:
            nodes_id = []
            ports_x = []
            ports_y = []

            def polygon_distance(poly_a: np.ndarray, poly_b: np.ndarray) -> Tuple[float, float, float]:
                """Calculates the Minimum Distance between two Polygons"""

                dist = float('inf')
                x_min = 0
                y_min = 0

                # Ensure poly_a is a NumPy array with shape (N, 1, 2)
                poly_a = np.array(poly_a, dtype=np.float32)
                if len(poly_a.shape) == 2 and poly_a.shape[1] == 2:
                    poly_a = poly_a.reshape((-1, 1, 2))  # Reshape to (N, 1, 2)

                for point in poly_b:
                    x, y = map(float, point[0])  # Convert point to float
                    point_dist = -cv2.pointPolygonTest(poly_a, (x, y), True)
                    
                    if point_dist < dist:
                        dist = point_dist
                        x_min = x
                        y_min = y

                return dist, x_min, y_min


            for node_id, node_shape in node_shapes:
                dist, x, y = polygon_distance(node_shape, contour)
                if dist < margin:
                    nodes_id += [node_id]
                    ports_x += [x]
                    ports_y += [y]

            if len(nodes_id) == 2:
                source_node_id, target_node_id = nodes_id
                source_port_x, target_port_x = ports_x
                source_port_y, target_port_y = ports_y
                edge_args = {'shape': [[int(x), int(y)] for [[x, y]] in contour]}

                if add_ports and graph.nodes[source_node_id]['type'] not in ("junction", "crossover"):
                    source_port = graph.add_port(source_node_id, {'x': int(source_port_x), 'y': int(source_port_y)})
                    edge_args['sourcePort'] = source_port

                if add_ports and graph.nodes[target_node_id]['type'] not in ("junction", "crossover"):
                    target_port = graph.add_port(target_node_id, {'x': int(target_port_x), 'y': int(target_port_y)})
                    edge_args['targetPort'] = target_port

                graph.add_edge(source=source_node_id, target=target_node_id, **edge_args)
