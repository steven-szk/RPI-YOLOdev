"""Run the trained YOLO model on Pi Camera frames, real-time on the robot.

Tuned for speed on the Pi: low capture resolution + small inference size,
and no per-frame disk writes in the loop.

Continuous (the robot's main mode):
    python detect.py                 # detect until Ctrl+C
    python detect.py --save          # also save annotated frames (slower)

Single shot (debug / aiming):
    python detect.py --once          # one frame -> data/detection.jpg

Reuse from robot logic:
    from detect import load_model, detect
    from capture import take_photo
    model = load_model()
    dets, _ = detect(model, take_photo())
    for label, conf, (x1, y1, x2, y2) in dets:
        cx = (x1 + x2) // 2          # object center -> steering
"""

import argparse
import os
import time

import psutil
from ultralytics import YOLO

from capture import get_camera, take_photo, close_camera

_TEMP_FILE = "/sys/class/thermal/thermal_zone0/temp"


def system_stats():
    """Return a short 'CPU/RAM/temp' string for the Pi."""
    cpu = psutil.cpu_percent()           # % since last call
    ram = psutil.virtual_memory().percent
    try:
        with open(_TEMP_FILE) as f:      # millidegrees C
            temp = int(f.read()) / 1000
        temp_str = f"{temp:.1f}C"
    except OSError:
        temp_str = "n/a"                 # not a Pi / file missing
    return f"CPU {cpu:4.0f}% | RAM {ram:4.0f}% | temp {temp_str}"

PT_PATH = "custom_model_5.pt"
NCNN_PATH = "custom_model_5_ncnn_model"   # folder made by `yolo export ... format=ncnn`
CONF = 0.5            # min confidence to report
IMGSZ = 640
    # inference size; smaller = faster, less accurate.
    # axtual size is 640*480 in a 640*640 square, other area is wasted

CAP_W, CAP_H = 640, 480   # capture resolution; small = faster pipeline


def load_model(path=None):
    """Load the YOLO model once and reuse it for every frame.

    Prefers the NCNN export (2-3x faster on the Pi CPU) when present,
    otherwise falls back to the .pt model.
    """
    if path is None:
        path = NCNN_PATH if os.path.isdir(NCNN_PATH) else PT_PATH
    print(f"Using model: {path}")
    return YOLO(path)


def detect(model, frame, conf=CONF, imgsz=IMGSZ):
    """Run the model on one BGR frame.

    Returns (detections, results) where detections is a list of
    (label, confidence, (x1, y1, x2, y2)) tuples.
    """
    results = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        out.append((label, confidence, (x1, y1, x2, y2)))
    return out, results


def main():
    p = argparse.ArgumentParser(description="Run YOLO on Pi Camera frames.")
    p.add_argument("--once", action="store_true", help="single shot instead of loop")
    p.add_argument("--save", action="store_true", help="save annotated frames")
    p.add_argument("--conf", type=float, default=CONF, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=IMGSZ, help="inference size")
    args = p.parse_args()

    model = load_model()
    get_camera(width=CAP_W, height=CAP_H)   # start camera at low res for speed
    if args.save:
        os.makedirs("data", exist_ok=True)

    def run_once():
        frame = take_photo()
        dets, results = detect(model, frame, conf=args.conf, imgsz=args.imgsz)
        for label, conf, (x1, y1, x2, y2) in dets:
            print(f"  {label:<15} {conf:.2f}  ({x1},{y1})-({x2},{y2})")
        if not dets:
            print("  (nothing detected)")
        if args.save:
            results.save(filename="data/detection.jpg")
        return dets

    if args.once:
        run_once()
        if args.save:
            print("saved data/detection.jpg")
        close_camera()
        return

    print("Detecting... Ctrl+C to stop.")
    try:
        while True:
            t0 = time.time()
            run_once()
            fps = 1 / (time.time() - t0)
            print(f"  [{fps:4.1f} FPS | {system_stats()}]")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        close_camera()


if __name__ == "__main__":
    main()
