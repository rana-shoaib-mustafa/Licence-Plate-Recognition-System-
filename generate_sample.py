"""
generate_sample.py
==================
Generates a synthetic car-with-license-plate test image for demo purposes.
Run this script if you don't have a real car image handy.

Usage:
    python generate_sample.py
"""

import os
import sys
import numpy as np
import cv2

# Set standard streams to use UTF-8 to prevent Windows cp1252/UnicodeEncodeError with libraries (like EasyOCR) that print UTF-8 characters.
if sys.platform.startswith('win'):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')


def draw_rounded_rect(img, x, y, w, h, radius, color, thickness=-1):
    """Draw a filled rectangle with rounded corners."""
    # Corners
    cv2.ellipse(img, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, thickness)
    # Rectangles to fill body
    cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), color, thickness)
    cv2.rectangle(img, (x, y + radius), (x + w, y + h - radius), color, thickness)
    return img


def create_sample_image(output_path: str = "sample_images/test_car.jpg"):
    """Create a synthetic vehicle image with a visible license plate."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    H, W = 480, 720
    img = np.ones((H, W, 3), dtype=np.uint8)

    # ── Sky gradient ─────────────────────────────────────────────────────────
    for row in range(H // 2):
        t = row / (H // 2)
        b = int(50  + t * 100)
        g = int(100 + t * 80)
        r = int(180 + t * 40)
        img[row, :] = (b, g, r)

    # ── Ground ────────────────────────────────────────────────────────────────
    img[H // 2:, :] = (55, 60, 55)

    # ── Road markings ─────────────────────────────────────────────────────────
    for i in range(0, W, 80):
        cv2.rectangle(img, (i, H * 3 // 4 - 3), (i + 40, H * 3 // 4 + 3), (220, 220, 50), -1)

    # ── Car body (dark blue) ──────────────────────────────────────────────────
    cx, cy = W // 2, H // 2 + 20
    # Main body
    draw_rounded_rect(img, cx - 200, cy + 10, 400, 100, 12, (80, 60, 40), -1)
    # Cabin
    draw_rounded_rect(img, cx - 130, cy - 60, 260, 80, 16, (100, 80, 55), -1)
    # Windshields
    cv2.fillPoly(img, [np.array([[cx - 110, cy - 55],
                                  [cx - 70, cy + 10],
                                  [cx + 70, cy + 10],
                                  [cx + 110, cy - 55]], dtype=np.int32)],
                 (180, 160, 100))
    # Window tint
    cv2.fillPoly(img, [np.array([[cx - 100, cy - 52],
                                  [cx - 65, cy + 8],
                                  [cx + 65, cy + 8],
                                  [cx + 100, cy - 52]], dtype=np.int32)],
                 (140, 120, 80))

    # ── Wheels ────────────────────────────────────────────────────────────────
    for wx in [cx - 130, cx + 130]:
        cv2.circle(img, (wx, cy + 110), 42, (30, 30, 30), -1)
        cv2.circle(img, (wx, cy + 110), 28, (70, 70, 70), -1)
        cv2.circle(img, (wx, cy + 110), 12, (45, 45, 45), -1)

    # ── Headlights & taillights ───────────────────────────────────────────────
    for lx in [cx - 195, cx + 165]:
        cv2.ellipse(img, (lx, cy + 55), (18, 10), 0, 0, 360, (220, 220, 180), -1)
    for tx in [cx - 170, cx + 148]:
        cv2.ellipse(img, (tx, cy + 55), (10, 6), 0, 0, 360, (50, 50, 220), -1)

    # ── Door lines ────────────────────────────────────────────────────────────
    cv2.line(img, (cx, cy + 12), (cx, cy + 105), (60, 45, 30), 2)
    cv2.line(img, (cx - 200, cy + 45), (cx + 200, cy + 45), (60, 45, 30), 1)

    # ── LICENSE PLATE ─────────────────────────────────────────────────────────
    # Plate dimensions and position (centred on front bumper)
    pw, ph = 130, 38
    px = cx - pw // 2
    py = cy + 118

    # Plate border (black frame)
    cv2.rectangle(img, (px - 4, py - 4), (px + pw + 4, py + ph + 4), (20, 20, 20), -1)
    # Plate background (white/cream)
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (240, 242, 245), -1)

    # Blue strip (EU style)
    cv2.rectangle(img, (px, py), (px + 18, py + ph), (30, 80, 200), -1)
    cv2.putText(img, "EU", (px + 1, py + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "**", (px + 2, py + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (240, 220, 30), 1, cv2.LINE_AA)

    # Plate text
    plate_number = "ABC 1234"
    cv2.putText(img, plate_number,
                (px + 22, py + 27),
                cv2.FONT_HERSHEY_DUPLEX,
                0.72,
                (10, 10, 10),
                2,
                cv2.LINE_AA)

    # ── Bolt holes on plate ───────────────────────────────────────────────────
    for bx in [px + 6, px + pw - 6]:
        for by in [py + 6, py + ph - 6]:
            cv2.circle(img, (bx, by), 2, (80, 80, 80), -1)

    # ── Shadow under car ─────────────────────────────────────────────────────
    shadow = np.zeros_like(img)
    cv2.ellipse(shadow, (cx, H // 2 + 155), (210, 18), 0, 0, 360, (30, 30, 30), -1)
    img = cv2.addWeighted(img, 1.0, shadow, 0.5, 0)

    # ── Save ──────────────────────────────────────────────────────────────────
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 97])
    print(f"[OK] Sample image saved: {output_path}  ({W}x{H})")
    return output_path


if __name__ == "__main__":
    create_sample_image()
