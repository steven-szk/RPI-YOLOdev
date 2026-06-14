"""Reusable Pi Camera capture API.

The camera is started once on first use and kept open, so calling take_photo()
in a loop is fast. Import this from YOLO train/run scripts:

    from capture import take_photo, close_camera
    try:
        while True:
            frame = take_photo()      # BGR numpy array, ready for YOLO
            ...
    finally:
        close_camera()

To serve photos over the LAN, run server.py (which uses this module).
"""

import atexit
import io
import threading
import time

from picamera2 import Picamera2 #type: ignore

_cam = None                      # Camera object, initialised once
_lock = threading.Lock()         # one capture at a time across threads


def get_camera(width=1920, height=1080, exposure_us=7000, gain=8.0):
    
    """Return the shared Pi Camera, starting it on first call.

    Defaults to a fast shutter to freeze motion while the robot moves.
    Shorter exposure = less blur but darker, so gain is raised to compensate.
    Pass exposure_us=None to use auto-exposure instead.
    """
    global _cam
    if _cam is None:
        cam = Picamera2()
        # BGR888 -> capture_array() returns BGR directly (what YOLO/cv2 want)
        cam.configure(cam.create_preview_configuration(
            main={"size": (width, height), "format": "BGR888"}))
        cam.start()
        if exposure_us is not None:
            # Fix the shutter manually so it can't drift slow and blur.
            cam.set_controls({"AeEnable": False,
                              "ExposureTime": exposure_us,
                              "AnalogueGain": gain})
        time.sleep(1)            # let settings/white-balance settle (first start)
        _cam = cam
    return _cam


def take_photo():
    """Capture a single BGR frame from the Pi Camera (numpy array, for YOLO)."""
    with _lock:
        return get_camera().capture_array()


def capture_jpeg():
    """Capture a single frame encoded as JPEG bytes."""
    stream = io.BytesIO()
    with _lock:
        get_camera().capture_file(stream, format="jpeg")
    return stream.getvalue()


def close_camera():
    """Stop and release the camera. Called automatically at exit."""
    global _cam
    if _cam is not None:
        _cam.stop()
        _cam.close()
        _cam = None


atexit.register(close_camera)


if __name__ == "__main__":
    # Quick self-test: grab two frames 2s apart, overwriting the same file.
    import os
    os.makedirs("data", exist_ok=True)
    path = "data/test.jpg"
    for i in range(2):
        with open(path, "wb") as f:
            f.write(capture_jpeg())
        print(f"saved {path}")
        time.sleep(1)
