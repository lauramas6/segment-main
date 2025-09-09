#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

# Folders to scan (relative to dataset root)
LABEL_DIRS = ["train_labels", "val_labels", "test_labels"]
# File types to read
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# OpenCV uses BGR channel order when writing images.
GREEN_BGR = (0, 255, 0)      # Sunlit Leaves
BROWN_BGR = (42, 42, 165)    # Noise (BGR that corresponds to RGB 165,42,42)


def process_image_cv(path: Path, thresh: int, kernel_size: int, dilate_iters: int, erode_iters: int) -> Path:
    """
    Read mask as grayscale, binarize by threshold, apply morphology (dilate then erode),
    then write a color PNG with two classes: GREEN (mask==255) and BROWN (mask==0).
    Saves next to the original as .png (may overwrite if original is already .png).
    """
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("cv.imread returned None (unreadable image)")

    # 1) Threshold to binary (0 or 255)
    _, binary = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)

    # 2) Morphology: dilate to fill tiny holes, then erode back to approximate original size
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv.dilate(binary, kernel, iterations=max(0, dilate_iters))
    mask = cv.erode(mask, kernel, iterations=max(0, erode_iters))

    # 3) Recolor into BGR classes for training output
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[mask == 255] = GREEN_BGR
    color[mask == 0]   = BROWN_BGR

    # 4) Save as PNG (same folder, same stem)
    out_path = path.with_suffix(".png")
    ok = cv.imwrite(str(out_path), color)
    if not ok:
        raise RuntimeError("cv.imwrite failed")
    return out_path


def process_dataset(root: Path, thresh: int, kernel: int, dilate: int, erode: int) -> dict:
    summary = {"images_total": 0, "images_cleaned": 0, "files": []}

    for sub in LABEL_DIRS:
        d = root / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if not (p.is_file() and p.suffix.lower() in IMAGE_EXTS):
                continue
            summary["images_total"] += 1
            try:
                out_path = process_image_cv(p, thresh=thresh, kernel_size=kernel, dilate_iters=dilate, erode_iters=erode)
                summary["images_cleaned"] += 1
                # Store relative path from dataset root for nice printing
                try:
                    rel = str(out_path.relative_to(root))
                except ValueError:
                    rel = str(out_path)
                summary["files"].append(rel)
            except Exception as e:
                print(f"WARNING: Failed to process {p}: {e}")
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Quick fix (OpenCV): binarize + morphology + recolor to two classes (green vs brown). Saves PNGs in-place."
    )
    ap.add_argument(
        "dataset_dir",
        type=str,
        help="Path to dataset folder containing train_labels/, val_labels/, test_labels/ (any that exist will be processed)."
    )
    ap.add_argument(
        "--thresh", type=int, default=127,
        help="Grayscale threshold for binary mask (0-255). Default: 127"
    )
    ap.add_argument(
        "--kernel", type=int, default=2,
        help="Square kernel size for morphology (>=1). Default: 2"
    )
    ap.add_argument(
        "--dilate", type=int, default=1,
        help="Number of dilation iterations (>=0). Default: 1"
    )
    ap.add_argument(
        "--erode", type=int, default=1,
        help="Number of erosion iterations (>=0). Default: 1"
    )
    args = ap.parse_args()

    root = Path(args.dataset_dir).resolve()
    if not root.exists():
        sys.exit(f"ERROR: Path does not exist: {root}")
    if args.kernel < 1:
        sys.exit("ERROR: --kernel must be >= 1")
    if not (0 <= args.thresh <= 255):
        sys.exit("ERROR: --thresh must be in [0, 255]")
    if args.dilate < 0 or args.erode < 0:
        sys.exit("ERROR: --dilate/--erode must be >= 0")

    summary = process_dataset(root, thresh=args.thresh, kernel=args.kernel, dilate=args.dilate, erode=args.erode)

    print("\nDone.")
    print(f"Images scanned:  {summary['images_total']}")
    print(f"Images cleaned:  {summary['images_cleaned']}")
    if summary["files"]:
        print("\nSaved PNG files (relative to dataset dir):")
        for f in summary["files"]:
            print(f" - {f}")


if __name__ == "__main__":
    main()
