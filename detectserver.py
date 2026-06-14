"""Pi Camera detection server (Python stdlib + cv2, no Flask).

Like server.py, but each frame is run through the YOLO model and the
detections (type, distance, angle) are drawn on the live stream.

PC Web Links:
    http://<pi-ip>:1234/            -> info page (live detection view)
    http://<pi-ip>:1234/stream.mjpg -> live MJPEG with detections drawn

Run on the Raspberry Pi:
    python detectserver.py
"""

import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2  # type: ignore

from capture import take_photo, close_camera   # importing capture starts the camera
from detect import load_model, detect

PORT = 1234
model = load_model()

INFO_PAGE = b"""<!DOCTYPE html>
<html><head><title>Pi Detection</title></head>
<body style="text-align:center;background:#1e1e1e;color:#fff;font-family:sans-serif;">
  <h1>Pi Detection</h1>
  <img src="/stream.mjpg" style="max-width:90%;border:2px solid #555;border-radius:8px;">
</body></html>
"""


def annotated_jpeg():
    """Capture a frame, run detection, draw the FILTERED dets, return JPEG bytes."""
    frame = take_photo()
    dets, _ = detect(model, frame)               # dets already per-class filtered
    for d in dets:                               # draw only what detect() kept OR, use frame = result.plot() and use take_photo() for the input of model
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        dist = f"{d['distance']:.0f}cm" if d["distance"] is not None else "edge"
        label = f"{d['type']} {d['confidence']:.2f} {dist} {d['angle']:+.0f}deg"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    ok, buf = cv2.imencode(".jpg", frame) #converts a BGR numpy array into jpeg
    return buf.tobytes()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html", INFO_PAGE)

        elif self.path.startswith("/stream.mjpg"):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpeg = annotated_jpeg()
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except Exception:
                pass                         # browser tab closed
        else:
            self.send_error(404)

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass                                 # quiet logging


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Serving detections on http://<pi-ip>:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()
        close_camera()


if __name__ == "__main__":
    main()
