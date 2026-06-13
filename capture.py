"""Reusable Pi Camera capture API.

The camera is started once on first use and kept open, so calling take_photo()
in a loop is fast. Import this from your YOLO train/run scripts:

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

from picamera2 import Picamera2

_cam = None                      # Camera object, initialised once
_lock = threading.Lock()         # one capture at a time across threads


def get_camera(width=1920, height=1080):
    """Return the shared Pi Camera, starting it on first call."""
    global _cam
    if _cam is None:
        cam = Picamera2()
        # BGR888 -> capture_array() returns BGR directly (what YOLO/cv2 want)
        cam.configure(cam.create_preview_configuration(
            main={"size": (width, height), "format": "BGR888"}))
        cam.start()
        time.sleep(1)            # let exposure settle (first start only)
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
    # Quick self-test: grab one frame and save it.
    with open("~data/test.jpg", "wb") as f:
        f.write(capture_jpeg())
    print("saved test.jpg")
