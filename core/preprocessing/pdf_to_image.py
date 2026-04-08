import numpy as np
import cv2
from pdf2image import convert_from_path


def pdf_to_image(path: str) -> np.ndarray:
    images = convert_from_path(path, dpi=300)
    return cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)


def remove_border(img: np.ndarray, margin: int = 150) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    x = max(x + margin, 0)
    y = max(y + margin, 0)
    w = min(w - 2 * margin, img.shape[1] - x)
    h = min(h - 2 * margin, img.shape[0] - y)
    return img[y : y + h, x : x + w]


def binarize(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )