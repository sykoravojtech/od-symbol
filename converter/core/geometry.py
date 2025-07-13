"""geometry.py: Base Definitions of Geometric Objects and Transformations"""

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2022-2024, DFKI and others"
__status__ = "Development"

# System Imports
from collections import namedtuple
from math import sin, cos, radians, sqrt
from typing import List, Dict




Point = namedtuple("Point", ["x", "y"])
Line = namedtuple("Line", ["a", "b", "color", "stroke", "nodeId"])
Rectangle = namedtuple("Rectangle", ["left", "top", "right", "bottom", "color", "stroke", "fillColor", "nodeId"])  # TODO (x-y)-w-h
Circle = namedtuple("Circle", ["x", "y", "radius", "color", "stroke", "fillColor", "nodeId"])  # p
Polygon = namedtuple("Polygon", ["points", "color", "stroke", "fillColor", "nodeId"])
Text = namedtuple("Text", ["text", "x", "y", "rotation", "color", "anchor", "nodeId"])  # p



def shift(p: Point, q: Point) -> Point:
    """Shifts a Point by another point"""

    return Point(p.x+q.x, p.y+q.y)


def length(p: Point) -> float:
    """Euclidean Distance to Origin"""

    return sqrt(p.x**2 + p.y**2)


def scale(p: Point, scale_x: float, scale_y: float) -> Point:
    """Scales a Point in two Dimensions"""

    return Point(p.x*scale_x, p.y*scale_y)


def rotate(p, angle):
    """Rotates a Point by an Angle"""

    return Point(p.x * cos(angle) - p.y * sin(angle), p.x * sin(angle) + p.y * cos(angle))


def transform(point: Point, bb: dict) -> Point:
    """Transforms a Point from Unit Space (classes ports) to Global Bounding Box (image)"""

    p = shift(point, Point(-.5, -0.5))      # Normalize: [0.0, 1.0]^2 -> [-0.5, 0-5]^2
    p = scale(p, 1.0, -1.0)          # Flip

    if bb.get('mirror_horizontal', False):
        p = scale(p, -1.0, 1.0)      # Mirror Horizontally

    if bb.get('mirror_vertical', False):
        p = scale(p, 1.0, -1.0)      # Mirror Vertically

    p = rotate(p, -radians(bb['rotation']))
    p = scale(p, bb['width'], bb['height'])
    p = shift(p, Point(bb['x'], bb['y']))

    return p

def transform_to_local(point: Point, bb: dict) -> Point:
    """Transforms a Point from the """

    p = shift(point, Point(-bb['x'], -bb['y']))
    p = scale(p, 1/bb['width'], 1/bb['height'])
    p = rotate(p, radians(bb['rotation']))
    p = scale(p, 1.0, -1.0)
    p = shift(p, Point(0.5, 0.5))

    return p


def map_points(source: List[Point], target: List[Point]) -> Dict[Point, Point]:
    """Cheap Implementation of Point Cloud Matching"""

    distances = sorted([(p, q, length(shift(p, scale(q, -1, -1)))) for q in target for p in source],
                       key=lambda x: x[2])
    map_len = min(len(source), len(target))
    point_map = {}

    for p,q, dist in distances:

        if p not in point_map.keys() and q not in point_map.values():
            point_map[p] = q

        if len(point_map) == map_len:
            break

    return point_map


if __name__ == "__main__":
    print(map_points([Point(999, 999), Point(0, 0), Point(5, 0), Point(0, 5)],
                     [Point(5.1, 0), Point(0, 0.1), Point(0.1, 4.99)]))
