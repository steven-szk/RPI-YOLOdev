"""LAN image-collection server.

Run this on the machine that gathers photos (your PC / training box).
The Raspberry Pi (or any client) posts images to it with uploader.py.

Images are saved under ./dataset/images/ (and grouped into per-label
subfolders when a label is provided), ready to be picked up by a YOLO
training pipeline.

Run:
    pip install flask
    python server.py            # listens on 0.0.0.0:8000 for the whole LAN

Then point clients at  http://<this-machine-ip>:8000
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename


SAVE_ROOT = Path(os.environ.get("DATASET_DIR", "dataset/images")).resolve()
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB per upload

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok", save_root=str(SAVE_ROOT))


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify(error="no 'image' field in form-data"), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify(error="empty filename"), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify(error=f"unsupported extension {ext!r}"), 400

    # Optional label -> store in a per-class subfolder.
    label = request.form.get("label", "").strip()
    target_dir = SAVE_ROOT / secure_filename(label) if label else SAVE_ROOT
    target_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped, collision-free name. Keep the original stem for traceability.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = secure_filename(Path(file.filename).stem) or "img"
    out_path = target_dir / f"{stem}_{stamp}{ext}"
    file.save(out_path)

    rel = out_path.relative_to(SAVE_ROOT)
    print(f"[{datetime.now():%H:%M:%S}] saved {rel}  (label={label or '-'})")
    return jsonify(status="saved", path=str(rel), label=label or None)


if __name__ == "__main__":
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Saving uploads under: {SAVE_ROOT}")
    # 0.0.0.0 so other devices on the LAN can reach it.
    app.run(host="0.0.0.0", port=8000, threaded=True)
