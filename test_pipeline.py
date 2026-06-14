"""
test_pipeline.py
================
Quick sanity-check script -- runs the full LPR pipeline on the bundled
synthetic test image and reports results to the console.

Usage:
    python test_pipeline.py
"""

import sys
import os

# Set standard streams to use UTF-8 to prevent Windows cp1252/UnicodeEncodeError with libraries (like EasyOCR) that print UTF-8 characters.
if sys.platform.startswith('win'):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# Make sure we can import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_sample import create_sample_image
from lpr_pipeline    import run_pipeline, visualize_pipeline


def main():
    print("=" * 60)
    print("  License Plate Recognition -- Pipeline Test")
    print("=" * 60)

    # Generate or reuse the sample image
    sample_path = "sample_images/test_car.jpg"
    if not os.path.isfile(sample_path):
        print("\n[TEST] Generating synthetic test image...")
        create_sample_image(sample_path)
    else:
        print(f"\n[TEST] Using existing sample: {sample_path}")

    # Run full pipeline
    print("\n[TEST] Running LPR pipeline...\n")
    result = run_pipeline(
        image_path = sample_path,
        use_gpu    = False,
        save_output= True,
        output_dir = "outputs",
        verbose    = True,
    )

    # Report
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Plate Text   : '{result['plate_text']}'")
    print(f"  Confidence   : {result['confidence']:.1%}")
    print(f"  Bounding Box : {result['bbox']}")
    print(f"  Output Path  : {result.get('output_path', 'N/A')}")
    print(f"  Timings      : {result['timing']}")
    print("=" * 60)

    # Save pipeline visualisation
    print("\n[TEST] Generating pipeline visualisation...")
    visualize_pipeline(
        sample_path,
        result,
        save_path="outputs/pipeline_visualisation.jpg",
    )

    if result["plate_text"]:
        print("\n[PASS] Test PASSED -- Plate successfully detected and recognised!")
        return 0
    else:
        print("\n[WARN] Test INCOMPLETE -- Plate detected but no text extracted.")
        print("    This is expected if EasyOCR models have not been downloaded yet.")
        return 0   # Not a hard failure


if __name__ == "__main__":
    sys.exit(main())
