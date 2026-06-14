"""Stable target finding on top of detect.py.

detect() gives raw per-frame detections, whose confidence can flicker (the
shiny steel bearing especially). This module adds temporal voting: only
objects seen across several recent frames are reported, with their angle /
distance averaged - giving the robot a steady target to home on.

    python YOLOpathfind.py      # print smoothed detections until Ctrl+C

Reuse from robot logic:
    from YOLOpathfind import Smoother
    from detect import load_model, detect
    from capture import take_photo

    model = load_model()
    smoother = Smoother()
    while True:
        dets, _ = detect(model, take_photo())
        dets = smoother.update(dets)        # stable, averaged, nearest-first
        if dets:
            steer_toward(dets[0]["angle"])  # nearest confirmed target
"""

import time
from collections import deque

from detect import load_model, detect, system_stats
from capture import take_photo, close_camera

SMOOTH_WINDOW = 5   # frames of history for temporal voting
SMOOTH_MIN = 3      # an object must appear in this many of them to be confirmed


def _by_distance(d):
    """Sort key: nearer first, unknown distance (None) goes last."""
    return d["distance"] if d["distance"] is not None else float("inf")


class Smoother:
    """Temporal voting to steady flickery detections (e.g. the shiny bearing
    whose confidence jumps around).

    Feed each frame's detections to update(); it returns only objects seen in
    at least `min_count` of the last `window` frames, with position / angle /
    distance averaged over those frames. Works per class on the NEAREST object
    of that class - enough for homing on the closest ball, but it does not
    track several objects of the same class separately.
    """

    def __init__(self, window=SMOOTH_WINDOW, min_count=SMOOTH_MIN):
        self.window = window
        self.min_count = min_count
        self.history = deque(maxlen=window)

    def update(self, dets):
        # Keep the nearest detection of each class for this frame.
        nearest = {}
        for d in dets:
            t = d["type"]
            if t not in nearest or _by_distance(d) < _by_distance(nearest[t]):
                nearest[t] = d
        self.history.append(nearest)

        # Confirm a class only if it shows up in enough recent frames.
        out = []
        types = {t for frame in self.history for t in frame}
        for t in types:
            seen = [frame[t] for frame in self.history if t in frame]
            if len(seen) >= self.min_count:
                out.append(self._average(t, seen))
        out.sort(key=_by_distance)
        return out

    @staticmethod
    def _average(t, seen):
        n = len(seen)
        dists = [s["distance"] for s in seen if s["distance"] is not None]
        return {
            "type": t,
            "confidence": sum(s["confidence"] for s in seen) / n,
            "x": sum(s["x"] for s in seen) / n,
            "y": sum(s["y"] for s in seen) / n,
            "angle": sum(s["angle"] for s in seen) / n,
            "distance": sum(dists) / len(dists) if dists else None,
            "count": n,                       # how many of the window saw it
        }


def main():
    model = load_model()   # camera already started by importing capture
    smoother = Smoother()

    print("Pathfinding... Ctrl+C to stop.")
    try:
        while True:
            t0 = time.time()
            dets, _ = detect(model, take_photo())
            dets = smoother.update(dets)          # temporal voting across frames
            for d in dets:
                dist = f"{d['distance']:.1f}cm" if d["distance"] is not None else "edge"
                print(f"  {d['type']:<12} {d['confidence']:.2f}  "
                      f"angle {d['angle']:+6.1f} deg  dist {dist}  "
                      f"seen {d['count']}/{smoother.window}")
            if not dets:
                print("  (nothing confirmed)")
            fps = 1 / (time.time() - t0)
            print(f"  [{fps:4.1f} FPS | {system_stats()}]")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        close_camera()


if __name__ == "__main__":
    main()
