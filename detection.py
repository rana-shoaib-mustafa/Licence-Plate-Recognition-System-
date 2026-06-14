"""
detection.py
License plate detection utilities using contour analysis.
Falls back to the largest rectangular region when the primary method fails.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


# ─── Aspect-ratio & area guards ───────────────────────────────────────────────
# Standard plates have an approximate aspect ratio between 2:1 and 6:1
PLATE_ASPECT_MIN = 1.5
PLATE_ASPECT_MAX = 7.0
PLATE_AREA_MIN   = 500          # pixels²  – ignore tiny noise contours
PLATE_AREA_MAX_RATIO = 0.40     # ignore regions > 40 % of total image area


def detect_plate(
    image: np.ndarray,
    edged: np.ndarray,
    gray: np.ndarray,
    top_k: int = 30,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Detect the most likely license plate region in an image by scoring candidates.

    Scoring criteria:
      - Aspect ratio (closer to standard plate aspect is better)
      - Edge density (plates have high edge density from characters)
      - Size (prefer moderately sized contours over tiny noise)
      - Vertical position (plates are usually in the lower part of the image)
      - Shape structure (reward 4-sided polygons)

    Args:
        image  : Original BGR image.
        edged  : Canny edge map (single-channel).
        gray   : Grayscale image.
        top_k  : How many top contours (by area) to examine.

    Returns:
        Tuple of:
          - plate_contour : 4-point contour of the plate.
          - plate_cropped : Cropped BGR plate image.
          - bbox          : (x, y, w, h) bounding box.
    """
    img_area = image.shape[0] * image.shape[1]
    img_height = image.shape[0]

    # ── Find & sort contours ──────────────────────────────────────────────────
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_k]

    best_score = -1.0
    best_contour = None
    best_bbox = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < PLATE_AREA_MIN or area > img_area * PLATE_AREA_MAX_RATIO:
            continue

        # Get bounding box and aspect ratio
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect = w / float(h)
        if not (PLATE_ASPECT_MIN <= aspect <= PLATE_ASPECT_MAX):
            continue

        # Check if it is a 4-sided polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
        is_quad = (len(approx) == 4)

        # Crop boundaries
        x_c = max(0, x)
        y_c = max(0, y)
        w_c = min(w, image.shape[1] - x_c)
        h_c = min(h, image.shape[0] - y_c)
        if w_c <= 0 or h_c <= 0:
            continue

        # Calculate edge density inside the candidate bounding box
        crop_edged = edged[y_c: y_c + h_c, x_c: x_c + w_c]
        edge_density = np.sum(crop_edged > 0) / float(w_c * h_c)

        # 1. Aspect score (target aspect: 3.5)
        aspect_diff = abs(aspect - 3.5)
        aspect_score = max(0.1, 1.0 - (aspect_diff / 3.5))

        # 2. Edge density score (prefer 0.10 to 0.40)
        if 0.10 <= edge_density <= 0.40:
            density_score = 1.0
        elif edge_density < 0.10:
            density_score = max(0.01, edge_density / 0.10)
        else:
            density_score = max(0.2, 1.0 - (edge_density - 0.40) * 2.0)

        # 3. Size score (normalised area)
        size_score = min(1.0, area / 2000.0)

        # 4. Vertical position (plates are rarely in the very top of the frame)
        y_center = y_c + h_c / 2.0
        y_rel = y_center / float(img_height)
        if y_rel < 0.20:
            y_score = 0.5 + (y_rel / 0.20) * 0.5
        else:
            y_score = 1.0

        # Calculate base score
        score = aspect_score * density_score * size_score * y_score

        # Boost if it forms a 4-sided polygon
        if is_quad:
            score *= 1.2

        if score > best_score:
            best_score = score
            best_contour = approx if is_quad else cnt
            best_bbox = (x_c, y_c, w_c, h_c)

    if best_contour is None:
        return None, None, None

    x, y, w, h = best_bbox
    plate_cropped = image[y: y + h, x: x + w]

    # Convert non-quad contours to a 4-point bounding box for visualization compatibility
    if len(best_contour) != 4:
        best_contour = np.array([
            [[x,     y    ]],
            [[x + w, y    ]],
            [[x + w, y + h]],
            [[x,     y + h]],
        ], dtype=np.int32)

    return best_contour, plate_cropped, best_bbox


def draw_detection(
    image: np.ndarray,
    plate_contour: np.ndarray,
    bbox: Tuple[int, int, int, int],
    text: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    font_scale: float = 0.9,
    thickness: int = 3,
) -> np.ndarray:
    """
    Overlay the detection contour and recognised text on a copy of the image.

    Args:
        image          : BGR image (will NOT be mutated).
        plate_contour  : 4-point contour array.
        bbox           : (x, y, w, h) of the bounding rectangle.
        text           : OCR result string to display.
        color          : BGR colour for the contour/text box.
        font_scale     : cv2.putText font scale.
        thickness      : Line thickness.

    Returns:
        np.ndarray: Annotated BGR image.
    """
    annotated = image.copy()
    x, y, w, h = bbox

    # Draw contour
    cv2.drawContours(annotated, [plate_contour], -1, color, thickness)

    # Draw text background
    if text:
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y = max(y - 10, text_h + 10)
        cv2.rectangle(
            annotated,
            (x, label_y - text_h - baseline),
            (x + text_w + 4, label_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (x + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


def get_all_candidates(
    image: np.ndarray,
    edged: np.ndarray,
    top_k: int = 20,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Return ALL plate-like rectangular candidates for debugging.

    Args:
        image  : BGR image.
        edged  : Canny edge map.
        top_k  : Number of largest contours to examine.

    Returns:
        List of (contour, (x, y, w, h)) tuples sorted by contour area desc.
    """
    img_area = image.shape[0] * image.shape[1]
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_k]

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < PLATE_AREA_MIN or area > img_area * PLATE_AREA_MAX_RATIO:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect = w / float(h)
        if PLATE_ASPECT_MIN <= aspect <= PLATE_ASPECT_MAX:
            candidates.append((cnt, (x, y, w, h)))

    return candidates
