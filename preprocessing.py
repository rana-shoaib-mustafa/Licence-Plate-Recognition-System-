"""
preprocessing.py
Image preprocessing utilities for License Plate Recognition.
Steps: Grayscale → Noise Reduction → Edge Detection → CLAHE Enhancement
"""

import cv2
import numpy as np
from typing import Tuple, Dict


def preprocess_image(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Full preprocessing pipeline for a raw input image.

    Args:
        image (np.ndarray): BGR input image read by OpenCV.

    Returns:
        dict: Dictionary containing each preprocessing stage result.
              Keys: 'original', 'gray', 'blurred', 'edged', 'clahe'
    """
    results = {}

    # ── 1. Store the original BGR image ──────────────────────────────────────
    results["original"] = image.copy()

    # ── 2. Grayscale ──────────────────────────────────────────────────────────
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results["gray"] = gray

    # ── 3. Noise Reduction with Gaussian Blur ─────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    results["blurred"] = blurred

    # ── 4. Canny Edge Detection ───────────────────────────────────────────────
    edged = cv2.Canny(blurred, 50, 150)
    results["edged"] = edged

    # ── 5. CLAHE Enhancement (for better OCR on low-light plates) ─────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    results["clahe"] = enhanced

    return results


def enhance_plate_image(plate_bgr: np.ndarray) -> np.ndarray:
    """
    Apply targeted enhancements to a cropped license plate image to
    improve OCR accuracy.

    Pipeline:
        1. Upscale (2×) for better character resolution
        2. CLAHE for contrast normalisation
        3. Gaussian denoise
        4. Adaptive thresholding for clean black/white characters

    Args:
        plate_bgr (np.ndarray): Cropped BGR image of the license plate.

    Returns:
        np.ndarray: Enhanced grayscale (or thresholded) plate image.
    """
    if plate_bgr is None or plate_bgr.size == 0:
        raise ValueError("Empty plate image supplied to enhance_plate_image().")

    # ── Upscale ───────────────────────────────────────────────────────────────
    h, w = plate_bgr.shape[:2]
    upscaled = cv2.resize(plate_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # ── Grayscale ─────────────────────────────────────────────────────────────
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # ── CLAHE contrast enhancement ────────────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # ── Denoise ───────────────────────────────────────────────────────────────
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # ── Adaptive threshold ────────────────────────────────────────────────────
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    return thresh


def deskew_plate(plate_bgr: np.ndarray) -> np.ndarray:
    """
    Attempt to correct minor rotation/skew in a cropped plate image
    using the minimum-area rectangle of the largest contour.

    Args:
        plate_bgr (np.ndarray): Cropped BGR image of the license plate.

    Returns:
        np.ndarray: Deskewed BGR plate image.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))

    if len(coords) < 5:
        return plate_bgr  # Not enough points to compute angle

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle

    h, w = plate_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        plate_bgr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed
