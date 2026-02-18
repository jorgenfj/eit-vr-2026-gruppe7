# webcam_seg.py
import argparse
import cv2
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="last.pt", help="segmentation model, e.g. yolo26n-seg.pt")
    ap.add_argument("--cam", type=int, default=0, help="webcam index")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size")
    ap.add_argument("--conf", type=float, default=0.90, help="confidence threshold")
    ap.add_argument("--device", default=None, help="e.g. 0 for CUDA, 'cpu' for CPU (leave empty for auto)")
    args = ap.parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.cam}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # frame is BGR uint8 (OpenCV). Ultralytics accepts this directly.
        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )

        annotated = results[0].plot()  # BGR numpy array (good for cv2.imshow)
        cv2.imshow("YOLO Seg (press q to quit)", annotated)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
