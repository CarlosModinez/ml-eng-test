import numpy as np
import cv2

from core.postprocessing.extract_walls import (
    detect_edges,
    detect_lines,
    extract_line_features,
    filter_lines,
)


def find_walls(raw_edges: np.ndarray) -> list[tuple]:
    edges_closed = cv2.dilate(
        raw_edges, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    )
    edges_closed = cv2.erode(
        edges_closed, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    )
    lines = detect_lines(edges_closed)
    features = extract_line_features(lines)
    return filter_lines(lines, features)