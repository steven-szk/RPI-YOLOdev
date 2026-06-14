"""Run the trained YOLO model on Pi Camera frames, real-time on the robot.

    python detect.py            # detect in a loop until Ctrl+C
    python detect.py --once     # single shot
    python detect.py --save     # also save annotated frames

Reuse from robot logic:
    from detect import load_model, detect
    from capture import take_photo
    model = load_model()
    dets, _ = detect(model, take_photo())      # sorted nearest first
    for d in dets:
        d["type"], d["angle"], d["distance"]   # steer toward d["angle"]
"""

import argparse
import math
import os
import time

import psutil
from ultralytics import YOLO #type: ignore

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
NCNN_PATH = "custom_model_5_ncnn_model"

CONF = 0.5          # min confidence to report
IMGSZ = 640         # inference size; smaller = faster, less accurate
CAP_W, CAP_H = 1920, 1080   # capture resolution
FOV_DEG = 120.0     # lens horizontal field of view

# Real-world diameter (cm) of each class, used for the rough distance
# estimate. Fill in with YOUR model's class names exactly. Any class not
# listed here gets distance=None.
DIAMETERS_CM = {
    "pingpong": 4.0,
    "bearing": 2.0,
}


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
    """Run the model on one BGR frame and locate each object.

    Returns (detections, results). Each detection is a dict:
        type        class name
        confidence  0.0 - 1.0
        x, y        box centre in pixels
        angle       horizontal angle in degrees, negative = left, positive = right
        distance    rough distance in cm from apparent size, or None if the
                    box touches a frame edge (object partly out of view) or the
                    class has no known diameter
    Sorted nearest first (detections without a distance go last).
    """
    results = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    h, w = frame.shape[:2]
    focal_px = (w / 2.0) / math.tan(math.radians(FOV_DEG / 2.0))

    dets = []
    for box in results.boxes:
        label = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Pixel offset from centre -> horizontal angle.
        angle = ((cx - w / 2.0) / w) * FOV_DEG

        # Distance from apparent size; skip if the box is cut off at an edge
        # (its true size would be wrong) or the class has no known diameter.
        pixel_diameter = min(x2 - x1, y2 - y1)
        on_edge = x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2
        if label in DIAMETERS_CM and pixel_diameter > 0 and not on_edge:
            distance = (DIAMETERS_CM[label] * focal_px) / pixel_diameter
        else:
            distance = None

        dets.append({
            "type": label,
            "confidence": float(box.conf[0]),
            "x": cx,
            "y": cy,
            "angle": angle,
            "distance": distance,
        })

    dets.sort(key=lambda d: d["distance"] if d["distance"] is not None else float("inf"))
    return dets, results


def main():
    p = argparse.ArgumentParser(description="Run YOLO on Pi Camera frames.")
    p.add_argument("--once", action="store_true", help="single shot instead of loop")
    p.add_argument("--save", action="store_true", help="save annotated frames")
    args = p.parse_args()

    model = load_model()
    get_camera(width=CAP_W, height=CAP_H)   # start camera at res 
    if args.save:
        os.makedirs("data", exist_ok=True)

    def run_once():
        frame = take_photo()
        dets, results = detect(model, frame)   # uses CONF / IMGSZ defaults
        for d in dets:
            dist = f"{d['distance']:.1f}cm" if d["distance"] is not None else "edge"
            print(f"  {d['type']:<12} {d['confidence']:.2f}  "
                  f"angle {d['angle']:+6.1f} deg  dist {dist}")
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
