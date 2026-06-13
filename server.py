"""Pi Camera HTTP server (Python stdlib, no Flask).

Runs ON THE PI. The PC runs NOTHING - it just opens a URL in a browser or
grabs it with curl:

    http://<pi-ip>:1234/            -> info page (live preview + snapshot)
    http://<pi-ip>:1234/photo.jpg   -> capture ONE fresh still, returns JPEG
    http://<pi-ip>:1234/stream.mjpg -> live MJPEG preview (~1 FPS)

Save a photo to the PC with no Pi-side interaction:
    curl http://<pi-ip>:1234/photo.jpg -o photo.jpg

Run on the Raspberry Pi:
    python server.py
"""

import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from capture import get_camera, capture_jpeg, close_camera

PORT = 1234

INFO_PAGE = b"""<!DOCTYPE html>
<html><head><title>Pi Camera</title></head>
<body style="text-align:center;background:#1e1e1e;color:#fff;font-family:sans-serif;">
  <h1>Pi Camera</h1>
  <p><a href="/photo.jpg" style="color:#6cf;" download>Download a fresh snapshot</a></p>
  <h2>Live preview</h2>
  <img src="/stream.mjpg" style="max-width:90%;border:2px solid #555;border-radius:8px;">
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html", INFO_PAGE)

        elif self.path.startswith("/photo.jpg"):
            # One fresh still -> JPEG. Browser/curl saves it; PC runs nothing.
            self._send(200, "image/jpeg", capture_jpeg())

        elif self.path.startswith("/stream.mjpg"):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpeg = capture_jpeg()
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.2)            # preview frame
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
    get_camera()                             # warm up before serving
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
