"""
lpr_pipeline.py
================
Main License Plate Recognition (LPR) pipeline.

Orchestrates the full two-stage workflow:
  Stage 1 – Detection  : Locate the license plate in the image.
  Stage 2 – Recognition: Extract the alphanumeric text from the plate.

Usage (script):
    python lpr_pipeline.py --image path/to/car.jpg

Usage (import):
    from lpr_pipeline import run_pipeline
    result = run_pipeline(image_path="car.jpg")
"""

import os
import sys
import argparse

# Set standard streams to use UTF-8 to prevent Windows cp1252/UnicodeEncodeError with libraries (like EasyOCR) that print UTF-8 characters.
if sys.platform.startswith('win'):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
import time
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")       # Use non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from utils.preprocessing import preprocess_image, enhance_plate_image, deskew_plate
from utils.detection     import detect_plate, draw_detection
from utils.recognition   import recognize_text, format_raw_results


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    use_gpu: bool = False,
    languages: list = None,
    save_output: bool = True,
    output_dir: str = "outputs",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Execute the full LPR pipeline on a single image.

    Args:
        image_path  : Path to the input car image.
        use_gpu     : Pass True to use CUDA acceleration in EasyOCR.
        languages   : EasyOCR language codes (default: ['en']).
        save_output : Whether to save the annotated result image.
        output_dir  : Directory to save outputs.
        verbose     : Print progress messages.

    Returns:
        dict with keys:
          'plate_text'   – recognised plate string
          'confidence'   – OCR confidence (0-1)
          'bbox'         – (x, y, w, h) bounding box or None
          'plate_image'  – cropped BGR plate array or None
          'annotated'    – annotated BGR image
          'raw_results'  – raw EasyOCR output
          'output_path'  – path to saved result or None
          'timing'       – dict of stage timings in seconds
    """
    if languages is None:
        languages = ["en"]

    timing: Dict[str, float] = {}
    result: Dict[str, Any] = {
        "plate_text": "",
        "confidence": 0.0,
        "bbox": None,
        "plate_image": None,
        "annotated": None,
        "raw_results": [],
        "output_path": None,
        "timing": timing,
    }

    # ── Load image ────────────────────────────────────────────────────────────
    t0 = time.time()
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")

    if verbose:
        print(f"[LPR] Loaded image: {image_path}  ({image.shape[1]}x{image.shape[0]})")
    timing["load"] = round(time.time() - t0, 4)

    # ── Stage 1 – Preprocessing ───────────────────────────────────────────────
    t0 = time.time()
    stages = preprocess_image(image)
    timing["preprocess"] = round(time.time() - t0, 4)
    if verbose:
        print(f"[LPR] Preprocessing done in {timing['preprocess']:.3f}s")

    # ── Stage 2 – Detection ───────────────────────────────────────────────────
    t0 = time.time()
    plate_contour, plate_cropped, bbox = detect_plate(
        image, stages["edged"], stages["gray"]
    )
    timing["detection"] = round(time.time() - t0, 4)

    if plate_contour is None:
        if verbose:
            print("[LPR] WARNING: No license plate detected. Returning empty result.")
        result["annotated"] = image.copy()
        return result

    if verbose:
        x, y, w, h = bbox
        print(f"[LPR] Plate detected at ({x},{y})  size {w}x{h}  in {timing['detection']:.3f}s")

    result["bbox"]        = bbox
    result["plate_image"] = plate_cropped

    # ── Stage 3 – Plate Enhancement ───────────────────────────────────────────
    t0 = time.time()
    try:
        plate_deskewed = deskew_plate(plate_cropped)
        plate_enhanced = enhance_plate_image(plate_deskewed)
    except Exception as exc:
        if verbose:
            print(f"[LPR] Enhancement skipped ({exc}). Using raw crop.")
        plate_enhanced = plate_cropped
    timing["enhancement"] = round(time.time() - t0, 4)

    # ── Stage 4 – OCR ─────────────────────────────────────────────────────────
    t0 = time.time()
    plate_text, confidence, raw_results = recognize_text(
        plate_enhanced, languages=languages, use_gpu=use_gpu
    )
    timing["ocr"] = round(time.time() - t0, 4)

    result["plate_text"]  = plate_text
    result["confidence"]  = confidence
    result["raw_results"] = raw_results

    if verbose:
        print(f"[LPR] OCR result  : '{plate_text}'  (confidence: {confidence:.1%})")
        print(f"[LPR] OCR timing  : {timing['ocr']:.3f}s")

    # ── Annotate the original image ───────────────────────────────────────────
    annotated = draw_detection(image, plate_contour, bbox, text=plate_text)
    result["annotated"] = annotated

    # ── Save output ───────────────────────────────────────────────────────────
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        base_name  = os.path.splitext(os.path.basename(image_path))[0]
        out_path   = os.path.join(output_dir, f"{base_name}_result.jpg")
        cv2.imwrite(out_path, annotated)
        result["output_path"] = out_path
        if verbose:
            print(f"[LPR] Result saved : {out_path}")

    total = sum(timing.values())
    if verbose:
        print(f"[LPR] Total time   : {total:.3f}s")

    return result


# ─── Visualisation helper ──────────────────────────────────────────────────────

def visualize_pipeline(image_path: str, result: Dict[str, Any], save_path: str = None):
    """
    Generate a multi-panel figure showing every stage of the pipeline.

    Panels:
        1. Original image
        2. Grayscale
        3. Blurred
        4. Edge map
        5. Detected plate (annotated)
        6. Cropped plate

    Args:
        image_path : Input image path (re-read for preprocessing stages).
        result     : Dict returned by run_pipeline().
        save_path  : If provided, save the figure here; else display it.
    """
    image  = cv2.imread(image_path)
    stages = preprocess_image(image)

    fig = plt.figure(figsize=(18, 10), facecolor="#0f1117")
    fig.suptitle(
        f"🔍 LPR Pipeline  |  Plate: {result['plate_text'] or 'Not Detected'}  "
        f"(conf: {result['confidence']:.1%})",
        fontsize=16, fontweight="bold", color="white", y=0.98
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3)

    panels = [
        (cv2.cvtColor(stages["original"], cv2.COLOR_BGR2RGB), "① Original Image",    False),
        (stages["gray"],                                        "② Grayscale",          True),
        (stages["blurred"],                                     "③ Gaussian Blur",      True),
        (stages["edged"],                                       "④ Canny Edges",        True),
        (cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB), "⑤ Plate Detection",   False),
    ]

    if result["plate_image"] is not None:
        panels.append(
            (cv2.cvtColor(result["plate_image"], cv2.COLOR_BGR2RGB), "⑥ Cropped Plate", False)
        )

    for idx, (img_data, title, is_gray) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        cmap = "gray" if is_gray else None
        ax.imshow(img_data, cmap=cmap)
        ax.set_title(title, color="white", fontsize=11, pad=6)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[LPR] Pipeline visualisation saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ─── CLI entry point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="License Plate Recognition – command-line tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lpr_pipeline.py --image car.jpg
  python lpr_pipeline.py --image car.jpg --gpu
  python lpr_pipeline.py --image car.jpg --output-dir results --visualize
        """,
    )
    p.add_argument("--image",      required=True,            help="Path to input image")
    p.add_argument("--gpu",        action="store_true",      help="Use GPU for OCR")
    p.add_argument("--languages",  nargs="+", default=["en"], help="EasyOCR languages")
    p.add_argument("--output-dir", default="outputs",        help="Directory for results")
    p.add_argument("--visualize",  action="store_true",      help="Save pipeline visualisation")
    p.add_argument("--quiet",      action="store_true",      help="Suppress console output")
    return p


def main():
    args   = _build_parser().parse_args()
    result = run_pipeline(
        image_path = args.image,
        use_gpu    = args.gpu,
        languages  = args.languages,
        save_output= True,
        output_dir = args.output_dir,
        verbose    = not args.quiet,
    )

    if args.visualize:
        viz_path = os.path.join(
            args.output_dir,
            os.path.splitext(os.path.basename(args.image))[0] + "_pipeline.jpg",
        )
        visualize_pipeline(args.image, result, save_path=viz_path)

    # Exit code 0 = plate found, 1 = not found
    sys.exit(0 if result["plate_text"] else 1)


if __name__ == "__main__":
    main()
