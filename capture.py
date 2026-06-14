"""Reusable Pi Camera capture API.

The camera starts automatically when this module is imported and is kept
open, so take_photo() / capture_jpeg() are ready immediately:

    from capture import take_photo, close_camera   # camera starts here
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

# Camera config (edit here, since the camera is set up at import time).
WIDTH, HEIGHT = 1920, 1080       # capture resolution
EXPOSURE_US = 10000              # shutter speed in microsections, None = auto
'''VERY IMPORTANT, in UK, 50Hz mains, so use multiples of 10ms'''
GAIN = 6.5                       # exposure conpensation

_cam = None                      # Camera object, initialised once
_lock = threading.Lock()         # one capture at a time across threads


def get_camera(width=WIDTH, height=HEIGHT, exposure_us=EXPOSURE_US, gain=GAIN):
    """Return the shared Pi Camera, starting it on first call.

    Defaults to a fast shutter to freeze motion while the robot moves.
    Shorter exposure = less blur but darker, so gain is raised to compensate.
    Pass exposure_us=None to use auto-exposure instead.
    """
    global _cam
    if _cam is None:
        cam = Picamera2()
        # picamera2 quirk: "RGB888" actually gives a BGR-ordered array, which
        # is what YOLO/cv2 expect.   ("BGR888" would give RGB and swap R<->B )
        cam.configure(cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}))
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

# start the camera as soon as this module is imported, or it could be auto started with first picture taken, but this is better
get_camera()          


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
