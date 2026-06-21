"""
Both cameras start automatically when this module is imported
Pass the CSI port number to pick a camera (0 = default, 1 = the second port):

    from capture import take_photo, close_camera   # cameras start here
    try:
        while True:
            front = take_photo()      # port 0, BGR numpy array, ready for YOLO
            rear  = take_photo(1)     # port 1
            ...
    finally:
        close_camera()                # closes both

To serve photos over the LAN, run server.py (which uses this module).
"""

import atexit
import io
import threading
import time

from picamera2 import Picamera2 #type: ignore

# Camera config (edit here, since the cameras are set up at import time).
WIDTH, HEIGHT = 1920, 1080       # capture resolution
EXPOSURE_US = 20000              # shutter speed in microsections, None = auto
'''VERY IMPORTANT, in UK, 50Hz mains, so use multiples of 10ms'''
GAIN = 6.5                       # exposure conpensation
# Fixed white-balance gains (red, blue). With AWB off these lock the colour
COLOUR_GAINS = (1.4, 1.4)  

CAMERA_PORTS = [0]          # CSI ports to open (camera_num values)

_cams = {}                       # {port: Picamera2}, each initialised once
# one capture at a time per camera (the two cameras can capture in parallel)
_locks = {port: threading.Lock() for port in CAMERA_PORTS}


def get_camera(num=0, width=WIDTH, height=HEIGHT, exposure_us=EXPOSURE_US,
               gain=GAIN, colour_gains=COLOUR_GAINS):
    """Return the shared Pi Camera on CSI port `num`, starting it on first call.

    Defaults to a fast shutter to freeze motion while the robot moves.
    Shorter exposure = less blur but darker, so gain is raised to compensate.
    Pass exposure_us=None to use auto-exposure instead.
    """
    if num not in _cams:
        cam = Picamera2(num)         # camera_num selects the CSI port
        # picamera2 quirk: "RGB888" actually gives a BGR-ordered array, which
        # is what YOLO/cv2 expect.   ("BGR888" would give RGB and swap R<->B )
        cam.configure(cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}))
        cam.start()
        if exposure_us is not None:
            # Fix the shutter manually so it can't drift slow and blur.
            # AwbEnable: False + ColourGains locks white-balance to fixed gains
            # so colours stay consistent frame-to-frame for detection.
            cam.set_controls({"AeEnable": False,
                              "ExposureTime": exposure_us,
                              "AnalogueGain": gain,
                              "AwbEnable": False,
                              "ColourGains": colour_gains})
        time.sleep(0.2)            # let settings/white-balance settle (first start)
        _cams[num] = cam
    return _cams[num]


def take_photo(num=0):
    """Capture a single BGR frame from CSI port `num` (numpy array, for YOLO)."""
    with _locks[num]:
        return get_camera(num).capture_array()


def capture_jpeg(num=0):
    """Capture a single frame from CSI port `num` encoded as JPEG bytes."""
    stream = io.BytesIO()
    with _locks[num]:
        get_camera(num).capture_file(stream, format="jpeg")
    return stream.getvalue()


def close_camera():
    """Stop and release camera(s). num=None closes both. Called automatically at exit."""
    for port in CAMERA_PORTS:
        cam = _cams.pop(port, None)
        if cam is not None:
            cam.stop()
            cam.close()


atexit.register(close_camera)

# start both cameras as soon as this module is imported, or they could be auto
# started with the first picture taken, but this is better
for _port in CAMERA_PORTS:
    get_camera(_port)


if __name__ == "__main__":
    # Quick self-test: grab one frame from each camera into its own file.
    import os
    os.makedirs("data", exist_ok=True)
    for _port in CAMERA_PORTS:
        path = f"data/test_cam{_port}.jpg"
        with open(path, "wb") as f:
            f.write(capture_jpeg(_port))
        print(f"saved {path}")
