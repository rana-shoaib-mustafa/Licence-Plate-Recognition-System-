"""
recognition.py
OCR utilities for License Plate Recognition.
Uses EasyOCR as the primary engine with configurable post-processing.
"""

import re
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

# ─── EasyOCR is lazily initialised to avoid a slow import at module load ──────
_reader_cache: Dict[str, object] = {}


def get_reader(languages: List[str] = None, use_gpu: bool = False):
    """
    Return (and cache) an EasyOCR Reader instance for the given language list.

    The Reader object is expensive to create (it loads a neural-network model),
    so we cache it in a module-level dict and reuse it across calls.

    Args:
        languages (list[str]): ISO language codes, e.g. ['en'].
        use_gpu   (bool)     : Use CUDA GPU if available.

    Returns:
        easyocr.Reader
    """
    import easyocr  # deferred import

    if languages is None:
        languages = ["en"]

    cache_key = "_".join(sorted(languages)) + f"_gpu={use_gpu}"
    if cache_key not in _reader_cache:
        _reader_cache[cache_key] = easyocr.Reader(languages, gpu=use_gpu)

    return _reader_cache[cache_key]


# ─── Character correction maps ────────────────────────────────────────────────
# OCR often confuses visually similar characters; these maps correct them.
_DIGIT_CORRECTIONS  = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
_LETTER_CORRECTIONS = {"0": "O", "1": "I", "5": "S", "8": "B"}


def postprocess_text(raw: str) -> str:
    """
    Clean up raw OCR output to make it look like a valid license plate string.

    Steps:
      1. Strip whitespace and convert to upper-case.
      2. Remove characters that are unlikely to appear on a plate.
      3. Remove internal spaces.

    Args:
        raw (str): Raw string from the OCR engine.

    Returns:
        str: Cleaned plate string.
    """
    if not raw:
        return ""

    # Upper-case and strip
    text = raw.upper().strip()

    # Keep only alphanumeric characters and hyphens/spaces
    text = re.sub(r"[^A-Z0-9\- ]", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def recognize_text(
    plate_image: np.ndarray,
    languages: List[str] = None,
    use_gpu: bool = False,
    detail_level: int = 1,
    min_confidence: float = 0.1,
) -> Tuple[str, float, List[dict]]:
    """
    Run OCR on a cropped plate image and return the extracted text.

    Args:
        plate_image    (np.ndarray) : BGR or grayscale plate crop.
        languages      (list[str])  : Language codes for EasyOCR.
        use_gpu        (bool)       : Use GPU acceleration if available.
        detail_level   (int)        : EasyOCR detail level (0 or 1).
        min_confidence (float)      : Minimum confidence to accept a result.

    Returns:
        Tuple of:
          - plate_text  (str)   : Best recognised plate string (post-processed).
          - confidence  (float) : Confidence of the best result (0.0–1.0).
          - raw_results (list)  : All raw EasyOCR result dicts for inspection.
    """
    if plate_image is None or plate_image.size == 0:
        return "", 0.0, []

    if languages is None:
        languages = ["en"]

    reader = get_reader(languages, use_gpu)

    # EasyOCR accepts BGR / grayscale / RGB numpy arrays
    raw_results = reader.readtext(plate_image, detail=detail_level)

    if not raw_results:
        return "", 0.0, []

    # ── Filter by confidence ──────────────────────────────────────────────────
    accepted = [r for r in raw_results if r[2] >= min_confidence]

    if not accepted:
        # Relax threshold and take the best guess
        accepted = raw_results

    # ── Sort by confidence descending ─────────────────────────────────────────
    accepted.sort(key=lambda r: r[2], reverse=True)

    # ── Concatenate all accepted texts (handles multi-line plates) ────────────
    combined_text = " ".join(r[1] for r in accepted)
    avg_confidence = float(np.mean([r[2] for r in accepted]))

    plate_text = postprocess_text(combined_text)

    return plate_text, avg_confidence, raw_results


def format_raw_results(raw_results: List) -> List[Dict]:
    """
    Convert EasyOCR raw output into a list of clean dicts for display.

    Args:
        raw_results: List of (bbox, text, confidence) tuples from EasyOCR.

    Returns:
        List of dicts with keys: 'text', 'confidence', 'bbox'.
    """
    formatted = []
    for item in raw_results:
        if len(item) == 3:
            bbox, text, conf = item
            formatted.append({
                "text": text,
                "confidence": round(float(conf) * 100, 1),
                "bbox": bbox,
            })
    return formatted
