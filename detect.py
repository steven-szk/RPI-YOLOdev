"""Run the trained YOLO model on Pi Camera frames, real-time on the robot.

    python detect.py            # detect in a loop until Ctrl+C

Reuse from robot logic:
    from detect import load_model, detect
    from capture import take_photo
    model = load_model()
    dets, _ = detect(model, take_photo())      # sorted nearest first
    for d in dets:
        d["type"], d["angle"], d["distance"]   # steer toward d["angle"]
"""

import math
import os
import time

import psutil
from ultralytics import YOLO #type: ignore

from capture import take_photo, close_camera   # importing capture starts the camera

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

# different confidence interval
CONF_BY_CLASS = {
    "ping pong ball": 0.60,
    "steel ball": 0.25,
}
CONF = min(CONF_BY_CLASS.values())   # run YOLO at the lowest, then filter per class
IMGSZ = 640         # inference size; smaller = faster, less accurate
FOV_DEG = 120.0     # lens horizontal field of view (capture res lives in capture.py)

# Real-world diameter (cm) of each class, used for the rough distance estimate
DIAMETERS_CM = {
    "ping pong ball": 4.0,
    "steel ball": 2.0,
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

    Returns a list of (detections, results). 
    
    Each result is the raw data set from YOLO and may be used further
    
    Each detection is a dict:
        type        class name
        confidence  0.0 - 1.0
        x, y        box centre in pixelss
        angle       horizontal angle in degrees, negative = left, positive = right
        distance    rough distance in cm from apparent size, or None if the
                    box touches a frame edge (object partly out of view) or the
                    class has no known diameter
    Sorted nearest first (detections without a distance go last).
    """
    results = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0] 
        #results of the yolo model, only one frame at a time so only take the list index 0
    
    h, w = frame.shape[:2] #find the capture resolution as this is a separante function
    focal_px = (w / 2.0) / math.tan(math.radians(FOV_DEG / 2.0))

    dets = []
    for box in results.boxes:
        label = model.names[int(box.cls[0])] #box.cls[0] gives the id of the model.names
        confidence = float(box.conf[0])
        if confidence < CONF_BY_CLASS.get(label, conf): #drop if below this class's threshold
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Pixel offset from centre -> horizontal angle.
        angle = ((cx - w / 2.0) / w) * FOV_DEG

        # Distance from apparent size; skip if the box is cut off at an edge
        # (its true size would be wrong) or the class has no known diameter.
        pixel_diameter = min(x2 - x1, y2 - y1)
        on_edge = x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2
        if (label in DIAMETERS_CM) and (pixel_diameter > 0) and (not on_edge):
            distance = (DIAMETERS_CM[label] * focal_px) / pixel_diameter
        else:
            distance = None

        dets.append({ #dets is a list of dics
            "type": label,
            "confidence": confidence,
            "x": cx,
            "y": cy,
            "box": (x1, y1, x2, y2),   # corners, for drawing the filtered boxes
            "angle": angle,
            "distance": distance,
        })

    dets.sort(key=lambda d: d["distance"] if d["distance"] is not None else float("inf")) #sort list from closest to furthest
    return dets, results   # dets is useful data, results is raw result

def main():
    model = load_model()   # camera already started by importing capture

    print("Detecting... Ctrl+C to stop.")
    try:
        while True:
            t0 = time.time()
            dets, _ = detect(model, take_photo())
            for d in dets:
                dist = f"{d['distance']:.1f}cm" if d["distance"] is not None else "edge"
                print(f"  {d['type']:<12} {d['confidence']:.2f}  "
                      f"angle {d['angle']:+6.1f} deg  dist {dist}")
            if not dets:
                print("  (nothing detected)")
            fps = 1 / (time.time() - t0)
            print(f"  [{fps:4.1f} FPS | {system_stats()}]")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        close_camera()


if __name__ == "__main__":
    main()
