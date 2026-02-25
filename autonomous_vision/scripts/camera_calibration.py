"""
camera_calibration.py — Camera intrinsic parameter calibration.

Uses a checkerboard pattern to determine camera intrinsics
(focal length, principal point, distortion coefficients).

Usage:
    python scripts/camera_calibration.py --images calibration_images/ --board 9x6
    python scripts/camera_calibration.py --camera 0 --board 9x6

Output:
    Writes calibrated parameters to configs/camera_params.yaml
"""

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def find_checkerboard_corners(
    image_paths: list[str],
    board_size: tuple[int, int],
    square_size: float = 30.0,
) -> tuple[list, list, tuple[int, int]]:
    """
    Find checkerboard corners in calibration images.

    Args:
        image_paths: List of calibration image paths.
        board_size: (columns, rows) of inner corners.
        square_size: Size of each square in mm.

    Returns:
        (object_points, image_points, image_size)
    """
    # Prepare object points (3D) for the checkerboard
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []  # 3D points in world space
    img_points = []  # 2D points in image space
    image_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    total = len(image_paths)
    found = 0

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"  ⚠️ Cannot read: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)

        # Find corners
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            found += 1
            obj_points.append(objp)

            # Refine corners
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            img_points.append(corners_refined)

            print(f"  ✅ [{i+1}/{total}] Found corners in: {Path(path).name}")
        else:
            print(f"  ❌ [{i+1}/{total}] No corners in: {Path(path).name}")

    print(f"\n  Found corners in {found}/{total} images")
    return obj_points, img_points, image_size


def calibrate_camera(
    obj_points: list,
    img_points: list,
    image_size: tuple[int, int],
) -> dict:
    """
    Calibrate camera using detected checkerboard corners.

    Returns:
        Dictionary with calibration results.
    """
    print("\n🔧 Calibrating camera...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    if not ret:
        raise RuntimeError("Camera calibration failed!")

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # Compute reprojection error
    total_error = 0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        total_error += error
    avg_error = total_error / len(obj_points)

    print(f"\n  📐 Calibration Results:")
    print(f"  ├── Reprojection Error : {avg_error:.4f} pixels")
    print(f"  ├── Focal Length (fx)  : {fx:.2f} px")
    print(f"  ├── Focal Length (fy)  : {fy:.2f} px")
    print(f"  ├── Principal Point    : ({cx:.2f}, {cy:.2f})")
    print(f"  └── Distortion         : {dist_coeffs.flatten()[:5]}")

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "image_size": image_size,
        "reprojection_error": float(avg_error),
    }


def save_calibration(calibration: dict, output_path: str):
    """Save calibration results to camera_params.yaml."""
    output = Path(output_path)

    # Load existing config if it exists
    existing = {}
    if output.exists():
        with open(output, "r") as f:
            existing = yaml.safe_load(f) or {}

    # Update camera section
    if "camera" not in existing:
        existing["camera"] = {}

    w, h = calibration["image_size"]
    existing["camera"]["width"] = w
    existing["camera"]["height"] = h
    existing["camera"]["fx"] = round(calibration["fx"], 2)
    existing["camera"]["fy"] = round(calibration["fy"], 2)
    existing["camera"]["cx"] = round(calibration["cx"], 2)
    existing["camera"]["cy"] = round(calibration["cy"], 2)
    existing["camera"]["distortion"] = [
        round(float(d), 6) for d in calibration["dist_coeffs"].flatten()[:5]
    ]

    with open(output, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

    print(f"\n  💾 Saved calibration to: {output}")


def calibrate_from_camera(camera_id: int, board_size: tuple[int, int], num_images: int = 20):
    """
    Interactive calibration from a live camera feed.

    Press SPACE to capture a frame, ESC to finish.
    """
    print(f"\n📷 Live calibration from camera {camera_id}")
    print(f"   Board size: {board_size[0]}x{board_size[1]}")
    print(f"   Press SPACE to capture, ESC when done (need {num_images}+ images)")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= 30.0  # 30mm squares

    obj_points = []
    img_points = []
    image_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    captured = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size, None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, board_size, corners, found)

        # Status text
        status = f"Captured: {captured}/{num_images}  |  Press SPACE to capture, ESC to finish"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Camera Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 32 and found:  # SPACE
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            captured += 1
            print(f"  ✅ Captured frame {captured}")

        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured < 5:
        print(f"  ❌ Not enough images ({captured}). Need at least 5.")
        sys.exit(1)

    return obj_points, img_points, image_size


def main():
    parser = argparse.ArgumentParser(
        description="Camera calibration using checkerboard pattern"
    )
    parser.add_argument(
        "--images", "-i",
        help="Directory containing calibration images (*.jpg, *.png)",
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=None,
        help="Camera index for live calibration",
    )
    parser.add_argument(
        "--board", "-b",
        default="9x6",
        help="Checkerboard inner corners, e.g. '9x6'",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=30.0,
        help="Square size in mm (default: 30)",
    )
    parser.add_argument(
        "--output", "-o",
        default="configs/camera_params.yaml",
        help="Output YAML file path",
    )

    args = parser.parse_args()

    # Parse board size
    cols, rows = map(int, args.board.split("x"))
    board_size = (cols, rows)

    print("=" * 60)
    print("📷 Camera Calibration Tool")
    print("=" * 60)

    if args.camera is not None:
        # Live calibration
        obj_pts, img_pts, img_size = calibrate_from_camera(args.camera, board_size)
    elif args.images:
        # From saved images
        image_dir = Path(args.images)
        if not image_dir.exists():
            print(f"❌ Directory not found: {image_dir}")
            sys.exit(1)

        image_paths = sorted(
            glob.glob(str(image_dir / "*.jpg"))
            + glob.glob(str(image_dir / "*.png"))
            + glob.glob(str(image_dir / "*.bmp"))
        )

        if not image_paths:
            print(f"❌ No images found in {image_dir}")
            sys.exit(1)

        print(f"  Found {len(image_paths)} calibration images")
        obj_pts, img_pts, img_size = find_checkerboard_corners(
            image_paths, board_size, args.square_size
        )
    else:
        print("❌ Specify --images or --camera")
        sys.exit(1)

    # Calibrate
    calibration = calibrate_camera(obj_pts, img_pts, img_size)

    # Save
    save_calibration(calibration, args.output)

    print("\n✅ Calibration complete!")


if __name__ == "__main__":
    main()
