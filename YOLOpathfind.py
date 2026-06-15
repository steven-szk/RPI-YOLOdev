"""Ball-collection planner: decide where to drive to maximise points/time.

Scoring: steel ball = 2 pts, ping pong ball = 1 pt. The robot collects one
ball at a time and the camera only sees part of the arena, so a full route
(TSP) is pointless - the world changes after every move. Instead we GREEDILY
re-plan every frame and pick the target with the best *expected points per
second*:

    value = points * confidence / time_to_reach
    time_to_reach ~= turn_time(angle) + drive_time(distance) + pickup overhead

So a 2-pt steel ball beats a 1-pt pingpong unless it's much farther / off to
the side. A small bonus is added for balls roughly along the same heading,
since they can be scooped on the way (a cheap "collect a line" heuristic).

Pipeline each frame:  detect() -> Smoother (steady detections) -> plan() -> command

    python YOLOpathfind.py          # preview the planned commands (no driving)

Drive it for real from main.py:
    from YOLOpathfind import Smoother, plan, execute
    smoother = Smoother()
    while playing:
        dets, _ = detect(model, take_photo())
        cmd = plan(smoother.update(dets))
        execute(robot, cmd)          # robot = UnibotDrive()
"""

import math
import time
from collections import deque

from detect import load_model, detect, system_stats
from capture import take_photo, close_camera

# ---- scoring -----------------------------------------------------------
POINTS = {
    "steel ball": 2,
    "ping pong ball": 1,
}

# Rough motion model for the time estimate (tune to your robot).
TURN_SEC_PER_DEG = 0.02     # ~1.8 s to turn 90 deg
DRIVE_SEC_PER_CM = 0.03     # ~3 s to drive 1 m
PICKUP_SEC = 1.0            # fixed overhead to scoop + settle per ball
UNKNOWN_DIST_CM = 200.0     # assume far when distance is None (edge box)

CORRIDOR_DEG = 12.0         # balls within this heading count as "on the way"
CORRIDOR_BONUS = 0.5        # fraction of their points credited to the candidate

# ---- approach behaviour ------------------------------------------------
MAX_STEP_CM = 40.0          # cap each forward drive, then re-detect
PICKUP_DIST_CM = 15.0       # within this, do the final scoop
SCOOP_EXTRA_CM = 12.0       # drive a bit past the centre to capture it
SEARCH_TURN_DEG = 45.0      # scan turn when nothing is in view
DRIVE_RPM = 100             # forward speed for execute()

# ---- temporal smoothing (steady the flickery detections) ---------------
SMOOTH_WINDOW = 5
SMOOTH_MIN = 3


def _by_distance(d):
    """Sort key: nearer first, unknown distance (None) goes last."""
    return d["distance"] if d["distance"] is not None else float("inf")


class Smoother:
    """Temporal voting so the planner acts on steady targets, not one-frame
    flickers. Returns objects seen in >= min_count of the last `window`
    frames, per class on the nearest of that class, with position averaged.
    """

    def __init__(self, window=SMOOTH_WINDOW, min_count=SMOOTH_MIN):
        self.window = window
        self.min_count = min_count
        self.history = deque(maxlen=window)

    def update(self, dets):
        nearest = {}
        for d in dets:
            t = d["type"]
            if t not in nearest or _by_distance(d) < _by_distance(nearest[t]):
                nearest[t] = d
        self.history.append(nearest)

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
            "count": n,
        }


# ---- planning ----------------------------------------------------------
def _dist(d):
    return d["distance"] if d["distance"] is not None else UNKNOWN_DIST_CM


def _time_to(d):
    """Estimated seconds to turn to, drive to, and scoop this ball."""
    return (abs(d["angle"]) * TURN_SEC_PER_DEG
            + _dist(d) * DRIVE_SEC_PER_CM
            + PICKUP_SEC)


def _value(d, dets):
    """Expected points per second for going after detection d."""
    pts = POINTS.get(d["type"], 0) * d["confidence"]
    if pts <= 0:
        return 0.0
    # Bonus for same-heading balls further down the line (scoop on the way).
    for o in dets:
        if o is d or POINTS.get(o["type"], 0) == 0:
            continue
        if abs(o["angle"] - d["angle"]) <= CORRIDOR_DEG and _dist(o) >= _dist(d):
            pts += POINTS[o["type"]] * o["confidence"] * CORRIDOR_BONUS
    return pts / _time_to(d)


def choose_target(dets):
    """Pick the ball with the best expected points-per-second, or None."""
    scorable = [d for d in dets if POINTS.get(d["type"], 0) > 0]
    return max(scorable, key=lambda d: _value(d, dets)) if scorable else None


def plan(dets):
    """Decide the next move. Returns a command dict:

        action    "collect" | "approach" | "search"
        turn_deg  turn this much first (right +, left -)
        drive_cm  then drive forward this far
        target    the chosen detection dict (or None)
        target_xy (forward_cm, lateral_cm) of the target, or None
        reason    short human-readable explanation
    """
    target = choose_target(dets)
    if target is None:
        return {"action": "search", "turn_deg": SEARCH_TURN_DEG, "drive_cm": 0.0,
                "target": None, "target_xy": None, "reason": "no balls in view"}

    angle = target["angle"]
    dist = target["distance"]
    pts = POINTS[target["type"]]

    # Position relative to the robot (for logging / higher-level logic).
    if dist is not None:
        fwd = dist * math.cos(math.radians(angle))
        lat = dist * math.sin(math.radians(angle))
        target_xy = (fwd, lat)
    else:
        target_xy = None

    # Close enough -> turn onto it and scoop right through it.
    if dist is not None and dist <= PICKUP_DIST_CM:
        return {"action": "collect", "turn_deg": angle, "drive_cm": dist + SCOOP_EXTRA_CM,
                "target": target, "target_xy": target_xy,
                "reason": f"scoop {pts}pt {target['type']}"}

    # Otherwise face it and drive a capped step, then re-detect next frame.
    step = MAX_STEP_CM if dist is None else min(MAX_STEP_CM, dist - PICKUP_DIST_CM)
    step = max(5.0, step)
    return {"action": "approach", "turn_deg": angle, "drive_cm": step,
            "target": target, "target_xy": target_xy,
            "reason": f"go for {pts}pt {target['type']} @ "
                      f"{'?' if dist is None else f'{dist:.0f}cm'} {angle:+.0f}deg"}


def execute(robot, cmd):
    """Carry out a command on a UnibotDrive: turn first, then drive forward."""
    if abs(cmd["turn_deg"]) > 1.0:
        robot.turn_angle(cmd["turn_deg"])
    if cmd["drive_cm"] > 0.0:
        robot.drive_distance(speed_rpm=DRIVE_RPM, distance_cm=cmd["drive_cm"], accel=2)


def main():
    model = load_model()
    smoother = Smoother()

    print("Pathfinding preview... Ctrl+C to stop.")
    try:
        while True:
            t0 = time.time()
            dets, _ = detect(model, take_photo())
            dets = smoother.update(dets)
            cmd = plan(dets)
            print(f"  {cmd['action'].upper():8} turn {cmd['turn_deg']:+6.1f} deg  "
                  f"drive {cmd['drive_cm']:5.1f} cm  | {cmd['reason']}")
            fps = 1 / (time.time() - t0)
            print(f"  [{fps:4.1f} FPS | {system_stats()}]")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        close_camera()


if __name__ == "__main__":
    main()
