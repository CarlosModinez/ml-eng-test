import numpy as np
import cv2


def extract_walls(binary: np.ndarray) -> np.ndarray:
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

    kernel_45 = np.eye(25, dtype=np.uint8)
    diag_45 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_45)

    kernel_135 = np.fliplr(kernel_45)
    diag_135 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_135)

    walls = cv2.bitwise_or(horizontal, vertical)
    walls = cv2.bitwise_or(walls, diag_45)
    walls = cv2.bitwise_or(walls, diag_135)
    return walls


def normalize_thickness(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(binary, kernel)


def detect_edges(img: np.ndarray) -> np.ndarray:
    return cv2.Canny(img, 50, 150)


def detect_lines(edges: np.ndarray) -> np.ndarray:
    return cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=80, maxLineGap=10
    )


def extract_line_features(lines) -> list[dict]:
    features = []
    if lines is None:
        return features
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        features.append({"line": (x1, y1, x2, y2), "length": length, "angle": angle})
    return features


def filter_lines(
    lines, features, min_length_threshold=50, max_length_threshold=1000, angle_tolerance=10
) -> list[tuple]:
    filtered = []
    if lines is None:
        return filtered
    for line, feat in zip(lines, features):
        x1, y1, x2, y2 = line[0]
        length = feat["length"]
        angle = feat["angle"]
        valid_angle = (
            abs(angle) < angle_tolerance
            or abs(angle - 45) < angle_tolerance
            or abs(angle - 60) < angle_tolerance
            or abs(angle - 90) < angle_tolerance
            or abs(angle - 135) < angle_tolerance
        )
        if valid_angle and min_length_threshold <= length <= max_length_threshold:
            filtered.append((x1, y1, x2, y2))
    return filtered


def draw_lines(img: np.ndarray, lines: list[tuple]) -> np.ndarray:
    output = img.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return output