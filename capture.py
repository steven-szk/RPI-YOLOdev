"""Take a photo with the Pi Camera, save it, and upload it over the LAN.

Run on the Raspberry Pi:
    python capture.py

Other scripts (YOLO train/run) can reuse the camera. The camera is started
once on first use and kept open, so calling take_photo() in a loop is fast:

    from capture import take_photo, close_camera
    try:
        while True:
            frame = take_photo()      # BGR numpy array, ready for YOLO
            ...
    finally:
        close_camera()
"""

import atexit
import time

import cv2
import requests
from picamera2 import Picamera2

PHOTO = "photo.jpg"
SERVER = "http://192.168.0.33:8000/upload"   # MY PC's IP

_cam = None   # Camera OBJECT to be initialised once


def get_camera(width=1920, height=1080):
    """Return the shared Pi Camera, starting it on first call."""
    global _cam
    if _cam is None:
        cam = Picamera2()
        cam.configure(cam.create_still_configuration(main={"size": (width, height)}))
        cam.start()
        time.sleep(1)             # let exposure settle (first start only)
        _cam = cam
    return _cam

#_cam=get_camera()

def take_photo():
    """Capture a single BGR frame from the Pi Camera (numpy array)."""
    rgb = get_camera().capture_array()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def close_camera():
    """Stop and release the camera. Called automatically at exit."""
    global _cam
    if _cam is not None:
        _cam.stop()
        _cam.close()
        _cam = None


atexit.register(close_camera)     # cleanup even if caller forgets


def main():
    frame = take_photo()
    cv2.imwrite(PHOTO, frame)
    print(f"saved {PHOTO}")
    with open(PHOTO, "rb") as f:
        r = requests.post(SERVER, files={"image": (PHOTO, f, "image/jpeg")}, timeout=10)
    print(f"uploaded -> {r.json()}")


if __name__ == "__main__":
    main()
