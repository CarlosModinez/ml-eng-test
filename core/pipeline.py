import cv2
import numpy as np

from core.preprocessing.pdf_to_image import pdf_to_image, remove_border, binarize
from core.postprocessing.extract_walls import (
    extract_walls,
    normalize_thickness,
    detect_edges,
    draw_lines,
)
from core.model.wall_detector import find_walls


class WallDetectionPipeline:

    def run(self, pdf_path: str) -> np.ndarray:
        img = pdf_to_image(pdf_path)
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        img = remove_border(img)

        binary = binarize(img)
        walls = extract_walls(binary)
        walls = normalize_thickness(walls)
        edges = detect_edges(walls)

        detected_walls = find_walls(edges)
        result = draw_lines(img, detected_walls)
        return result