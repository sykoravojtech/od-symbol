__author__ = "Vojtěch Sýkora"


def get_xymin_xymax_pos(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    """Compute bounding box corners from x,y,width,height"""
    xmin = x - w // 2
    ymin = y - h // 2
    xmax = x + w // 2
    ymax = y + h // 2
    return xmin, ymin, xmax, ymax
