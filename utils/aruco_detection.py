#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images_sorted(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: p.name)  # filename sort
    return files


def get_aruco_dict(dict_name: str):
    # Map common names to cv2.aruco constants
    name = dict_name.upper()
    if not name.startswith("DICT_"):
        name = "DICT_" + name

    if not hasattr(cv2.aruco, name):
        available = [a for a in dir(cv2.aruco) if a.startswith("DICT_")]
        raise ValueError(f"Unknown dictionary '{dict_name}'. Try one of: {', '.join(available)}")

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_detector(dict_obj):
    """
    Supports both newer OpenCV API (ArucoDetector) and older (detectMarkers).
    Returns a callable detect(gray) -> (corners_list, ids_array)
    """
    # Newer OpenCV (4.7+ typically)
    if hasattr(cv2.aruco, "ArucoDetector") and hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dict_obj, params)

        def detect(gray):
            corners, ids, _rej = detector.detectMarkers(gray)
            return corners, ids

        return detect

    # Older OpenCV
    params = cv2.aruco.DetectorParameters_create()

    def detect(gray):
        corners, ids, _rej = cv2.aruco.detectMarkers(gray, dict_obj, parameters=params)
        return corners, ids

    return detect


def marker_center_from_corners(corners: np.ndarray) -> Tuple[np.float64, np.float64]:
    """
    corners: shape (4,2) float-like (pixel coords)
    Returns (cx, cy) as float64.
    """
    c = np.asarray(corners, dtype=np.float64)
    center = c.mean(axis=0)  # (2,)
    return np.float64(center[0]), np.float64(center[1])


def ensure_dir(p: Optional[Path]):
    if p is None:
        return
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(
        description="Run ArUco detection on images from a folder and save center pixel coords + frame number."
    )
    ap.add_argument("--input", required=True, help="Input folder containing images.")
    ap.add_argument("--output", default="aruco_detections.csv", help="Output CSV file (single file).")
    ap.add_argument("--dict", default="DICT_ARUCO_ORIGINAL", help="Aruco dictionary, e.g. DICT_4X4_50, DICT_ARUCO_ORIGINAL, ...")
    ap.add_argument("--start-index", type=int, default=0, help="Frame number for the first image (default 0).")
    ap.add_argument(
        "--per-frame-dir",
        default=None,
        help="If set, also write one CSV per frame to this folder (e.g. detections_per_frame/).",
    )
    ap.add_argument(
        "--write-empty",
        action="store_true",
        help="If set, write a row for frames with no detections (marker_id=-1, coords=nan).",
    )
    ap.add_argument(
        "--annotate-dir",
        default=None,
        help="If set, saves annotated images (markers + centers drawn) to this folder.",
    )

    args = ap.parse_args()

    in_dir = Path(args.input)
    out_csv = Path(args.output)
    per_frame_dir = Path(args.per_frame_dir) if args.per_frame_dir else None
    annotate_dir = Path(args.annotate_dir) if args.annotate_dir else None

    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")

    ensure_dir(per_frame_dir)
    ensure_dir(annotate_dir)

    dict_obj = get_aruco_dict(args.dict)
    detect = make_detector(dict_obj)

    images = list_images_sorted(in_dir)
    if not images:
        raise RuntimeError(f"No images found in {in_dir} with extensions: {sorted(IMG_EXTS)}")

    # Write main CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "filename", "marker_id", "center_x", "center_y"])

        for i, img_path in enumerate(images):
            frame_num = args.start_index + i

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners_list, ids = detect(gray)

            # Normalize ids to a simple list of ints
            ids_list = []
            if ids is not None and len(ids) > 0:
                ids_list = [int(x) for x in ids.flatten().tolist()]

            detections_this_frame = []

            if ids_list:
                for corners, mid in zip(corners_list, ids_list):
                    # corners may be (1,4,2) in older API
                    c = np.asarray(corners)
                    if c.ndim == 3 and c.shape[0] == 1:
                        c = c[0]
                    cx, cy = marker_center_from_corners(c)
                    detections_this_frame.append((mid, cx, cy))
                    writer.writerow([frame_num, img_path.name, mid, float(cx), float(cy)])
            else:
                if args.write_empty:
                    writer.writerow([frame_num, img_path.name, -1, float("nan"), float("nan")])

            # Optional: per-frame CSV
            if per_frame_dir is not None:
                per_file = per_frame_dir / f"frame_{frame_num:06d}.csv"
                with per_file.open("w", newline="") as pf:
                    pw = csv.writer(pf)
                    pw.writerow(["frame", "filename", "marker_id", "center_x", "center_y"])
                    if detections_this_frame:
                        for mid, cx, cy in detections_this_frame:
                            pw.writerow([frame_num, img_path.name, mid, float(cx), float(cy)])
                    elif args.write_empty:
                        pw.writerow([frame_num, img_path.name, -1, float("nan"), float("nan")])

            # Optional: annotated images
            if annotate_dir is not None:
                ann = img.copy()
                if ids_list:
                    # Draw markers
                    try:
                        cv2.aruco.drawDetectedMarkers(ann, corners_list, ids)
                    except Exception:
                        pass
                    # Draw centers
                    for mid, cx, cy in detections_this_frame:
                        cv2.circle(ann, (int(round(cx)), int(round(cy))), 4, (0, 255, 0), -1)
                        cv2.putText(
                            ann,
                            str(mid),
                            (int(round(cx)) + 6, int(round(cy)) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                out_img = annotate_dir / img_path.name
                cv2.imwrite(str(out_img), ann)

    print(f"Done. Wrote detections to: {out_csv}")
    if per_frame_dir is not None:
        print(f"Per-frame CSVs in: {per_frame_dir}")
    if annotate_dir is not None:
        print(f"Annotated images in: {annotate_dir}")


if __name__ == "__main__":
    main()
