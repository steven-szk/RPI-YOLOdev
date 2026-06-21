"""Pi Camera HTTP server (Python stdlib, no Flask).

PC Web Links:
    http://<pi-ip>:1234/            -> info page (live preview + snapshot)
    http://<pi-ip>:1234/stream.mjpg -> live MJPEG preview (~1 FPS)

Run on the Raspberry Pi:
    python server.py
"""

import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2 #type: ignore

from capture import take_photo, close_camera   # importing capture starts the camera
from apriltaghoming import draw_tags


PORT = 1234

INFO_PAGE = b"""<!DOCTYPE html>
<html><head><title>Pi Camera</title></head>
<body style="text-align:center;background:#1e1e1e;color:#fff;font-family:sans-serif;">
  <h1>Pi Camera</h1>
  <h2>Live preview</h2>
  <img src="/stream.mjpg" style="max-width:90%;border:2px solid #555;border-radius:8px;">
</body></html>
"""

def _frame_jpeg(cam):
    """JPEG bytes for one frame of camera `cam`. The rear camera gets the
    detected AprilTags drawn on first; the front camera is served as-is.
    """
    frame = draw_tags(take_photo(cam))     # detect + annotate in place
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else b""


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
                    jpeg = _frame_jpeg(0)
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.15)            # preview frame
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
    print(f"Serving on http://<pi-ip>:{PORT}/  (photo: /photo.jpg)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()
        close_camera()


if __name__ == "__main__":
    main()
