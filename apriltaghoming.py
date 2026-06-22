import math
import time

import cv2  # type: ignore
from pupil_apriltags import Detector  # type: ignore

# importing capture starts both cameras (idempotent: motion.py already did this)
from capture import take_photo, close_camera, WIDTH, HEIGHT, FOV_DEG

TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.100           # all arena tags are 100 mm

#cam param
FOCAL_PX = (WIDTH / 2.0) / math.tan(math.radians(FOV_DEG / 2.0))
CAMERA_PARAMS = (FOCAL_PX, FOCAL_PX, WIDTH / 2.0, HEIGHT / 2.0) # pupil_apriltags wants (fx, fy, cx, cy)

_detector = Detector(families=TAG_FAMILY)

# detects all tags in frame
def detect_tags(frame):
    """Detect tag36h11 tags in a BGR rear-camera frame, with pose.

    Returns a list of dicts, one per tag:
        id          tag id
        cx, cy      pixel of the tag centre
        corners     4x2 array of the tag's corner pixels (for drawing)
        depth_cm    distance along the camera's optical axis (cm)
        lateral_cm  sideways offset, +right of the optical axis (cm)
        bearing_deg angle off the optical axis, +right (atan2(lateral, depth))
    """
    if frame is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = _detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=CAMERA_PARAMS,
        tag_size=TAG_SIZE_M,
    )

    dets = []
    for r in results:
        # pose_t is a (3,1) column vector, so index [row, 0] for a scalar.
        lateral_m = float(r.pose_t[0, 0])   # +ve = tag to camera's right
        depth_m = float(r.pose_t[2, 0])     # +ve = in front of the camera
        if depth_m <= 0:                    # behind the lens / bad pose, ignore
            continue
        dets.append({
            "id": r.tag_id,
            "cx": float(r.center[0]),
            "cy": float(r.center[1]),
            "corners": r.corners,           # 4x2 pixel corners, for overlays
            "depth_cm": depth_m * 100.0,
            "lateral_cm": lateral_m * 100.0,
            "bearing_deg": math.degrees(math.atan2(lateral_m, depth_m)),
        })

    #print detections
    for d in dets:
        print("id:", d["id"], d["depth_cm"], d["bearing_deg"])
    if dets != []: print("")
    
    return dets

# called in server only to draw tags
def draw_tags(frame, dets=None, teamtags=()):
    """Draw detected tag outlines, ids and pose onto a BGR frame, in place.

    Pass `dets` from detect_tags(frame) to reuse a detection, or leave it None
    to detect here. `teamtags` ids are drawn in red, all others in green.
    Returns the same frame (handy for one-liner use). Used by server.py to
    overlay the rear-camera preview.
    """
    if frame is None:
        return frame
    if dets is None:
        dets = detect_tags(frame)
    if isinstance(teamtags, int):
        teamtags = (teamtags,)

    for d in dets:
        is_team = d["id"] in teamtags
        colour = (0, 0, 255) if is_team else (0, 255, 0)   # BGR: team red, else green
        pts = d["corners"].astype(int)
        cv2.polylines(frame, [pts.reshape((-1, 1, 2))], True, colour, 3)
        cv2.circle(frame, (int(d["cx"]), int(d["cy"])), 6, colour, -1)
        label = f"id{d['id']} {d['depth_cm']:.0f}cm {d['bearing_deg']:+.0f}deg"
        x = int(pts[:, 0].min())
        y = max(int(pts[:, 1].min()) - 12, 24)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, colour, 3, cv2.LINE_AA)
    return frame

# ======================================================================
# Homing planner
# ----------------------------------------------------------------------
# The two HOMETAGS sit on the home wall a known distance apart. Seen from
# the camera each gives depth_cm (along the optical axis) and lateral_cm
# (+right), i.e. a top-down position in the camera frame !!!!origin at camera!!!!:
#
#       x = lateral (+right)        y = depth (+forward)
#
# Two tags => full 2-D pose: the dock centre T (their midpoint) AND the
# wall normal, so we know not just *where* the dock is but whether we are
# *square* to it. That matters because the 65 deg FOV loses both tags as we
# close in - up close we are blind. So homing splits at the commit point:
#
#   FUNNEL (vision) : while both tags are in view, steer toward a standoff
#                     point on the wall normal, in front of the dock, squaring
#                     up as we go.
#   COMMIT (blind)  : the instant a tag is about to leave the frame (or we
#                     reach the standoff), face the dock and drive straight in.
#                     When the tags finally leave the frame we keep that
#                     straight velocity for a moment longer and brake - no
#                     feedback is needed because we entered square and centred.
#
# This is a CONTINUOUS controller, not discrete moves: plan() is called every
# frame and returns a *velocity* (forward_rpm, turn_rpm) for the non-blocking
# UnibotDrive.set_velocity(). The robot keeps moving while we perceive, and the
# velocity is re-adjusted every frame.
#
# Graceful degradation: 2 tags -> 1 tag (creep, hold heading) -> 0 (search).
# ======================================================================

# ---- homing geometry / commit (tune to your robot) ---------------------
STANDOFF_CM      = 30.0   # aim point: distance in front of the dock, on the normal
WAYPOINT_TOL_CM  = 8.0    # within this of the standoff point = "arrived", commit
EDGE_MARGIN_PX   = 140    # a tag centre this near the frame edge is about to exit
COMMIT_RANGE_CM  = 20.0   # range to dock at/under which we commit regardless
SINGLE_COMMIT_DEPTH_CM = 35.0  # one tag only and this close -> assume squared, commit

# ---- velocity controller (rpm) -----------------------------------------
FWD_CRUISE_RPM   = 70.0   # funnel approach speed when pointed at the target
COMMIT_RPM       = 45.0   # steadier push once committed / driving into the dock
CREEP_RPM        = 30.0   # one-tag re-acquire speed
SEARCH_TURN_RPM  = 30.0   # in-place scan when nothing is in view
KP_TURN_RPM      = 1.6    # turn rpm per degree of bearing error
MAX_TURN_RPM     = 50.0   # clamp on the turn component
ALIGN_FULL_DEG   = 8.0    # |bearing| <= this -> full forward speed
TURN_ONLY_DEG    = 35.0   # |bearing| >= this -> turn in place (no forward)
BLIND_SEAT_SEC   = 0.8    # after the tags vanish, push straight this long, then brake

# Map camera-frame commands to robot motion. If the homing camera faces the
# REAR of the robot, set these to -1 (commanding motion "toward the tags" then
# drives/turns the robot the other way). Forward-facing camera: +1.
TURN_SIGN  = 1
FWD_SIGN   = 1


def _cmd(action, forward_rpm, turn_rpm, **extra):
    """Build a velocity command dict with consistent keys (extras fill diag)."""
    cmd = {
        "action": action,
        "forward_rpm": float(forward_rpm),
        "turn_rpm": float(turn_rpm),
        "range_cm": None,
        "bearing_deg": None,
        "skew_deg": None,
        "n_tags": 0,
        "reason": "",
    }
    cmd.update(extra)
    return cmd


def _turn_rpm(bearing_err):
    """Turn rate (+right) proportional to bearing error, clamped."""
    return max(-MAX_TURN_RPM, min(MAX_TURN_RPM, KP_TURN_RPM * bearing_err))


def _fwd_rpm(bearing_err, cruise):
    """Forward rpm: full when pointed at the target, ramped to 0 as the heading
    error grows, so a large error turns (nearly) in place before driving on."""
    e = abs(bearing_err)
    if e >= TURN_ONLY_DEG:
        return 0.0
    if e <= ALIGN_FULL_DEG:
        return cruise
    return cruise * (TURN_ONLY_DEG - e) / (TURN_ONLY_DEG - ALIGN_FULL_DEG)


def _select_home(dets, hometags):
    """Home-tag detections, nearest first (so [0] and [1] are the two best)."""
    home = [d for d in dets if d["id"] in hometags]
    home.sort(key=lambda d: d["depth_cm"])
    return home


def _pose_from_pair(a, b):
    """Top-down geometry from two home tags (camera frame, x=right, y=fwd).

    Returns the dock centre T and range/bearing to it, the standoff aim point
    P on the wall normal in front of the dock (range/bearing to P), the wall
    skew (0 = square to the wall) and the tags' min/max centre-x in pixels
    (for the frame-edge commit test).
    """
    ax, ay = a["lateral_cm"], a["depth_cm"]
    bx, by = b["lateral_cm"], b["depth_cm"]

    tx, ty = (ax + bx) / 2.0, (ay + by) / 2.0           # dock centre
    rng = math.hypot(tx, ty)
    bearing = math.degrees(math.atan2(tx, ty))

    # Wall vector and its two perpendiculars; pick the normal that points from
    # the dock back toward the robot (the origin), i.e. n . (robot - T) > 0.
    wx, wy = bx - ax, by - ay
    n = (-wy, wx)
    if n[0] * (-tx) + n[1] * (-ty) < 0:
        n = (wy, -wx)
    nmag = math.hypot(*n) or 1.0
    nhx, nhy = n[0] / nmag, n[1] / nmag

    px, py = tx + STANDOFF_CM * nhx, ty + STANDOFF_CM * nhy
    range_p = math.hypot(px, py)
    bearing_p = math.degrees(math.atan2(px, py))

    # Skew: angle of the wall off the camera's lateral axis (left tag -> right
    # tag). 0 when both tags are at equal depth, i.e. we are facing it square.
    if ax <= bx:
        lx, ly, rx, ry = ax, ay, bx, by
    else:
        lx, ly, rx, ry = bx, by, ax, ay
    skew = math.degrees(math.atan2(ry - ly, rx - lx))

    return {
        "range_cm": rng, "bearing_deg": bearing,
        "range_p": range_p, "bearing_p": bearing_p,
        "skew_deg": skew,
        "cx_min": min(a["cx"], b["cx"]),
        "cx_max": max(a["cx"], b["cx"]),
    }


def plan(dets, hometags):
    """Decide the next homing velocity. Called every frame; returns a command:

        action    "search" : no home tags in view -> rotate to find them
                  "funnel" : both tags seen -> steer onto the wall normal,
                             squaring up while we still have vision
                  "creep"  : one tag only -> hold heading, edge closer to
                             try to bring the pair back into view
                  "commit" : aligned, or a tag is about to leave the frame ->
                             face the dock and drive straight in
        forward_rpm  continuous forward speed for set_velocity (+ = forward)
        turn_rpm     continuous turn speed for set_velocity (+ = right)
        range_cm, bearing_deg, skew_deg, n_tags, reason   (diagnostics)

    Steering is proportional to a bearing error and forward speed ramps down as
    that error grows (_fwd_rpm), so a badly-aimed robot turns toward the target
    before driving on, and a well-aimed one cruises.
    """
    if isinstance(hometags, int):
        hometags = (hometags,)
    home = _select_home(dets, hometags)
    n = len(home)

    # --- nothing in view: scan in place ---------------------------------
    if n == 0:
        return _cmd("search", 0.0, SEARCH_TURN_RPM, n_tags=0,
                    reason="no home tags in view")

    # --- one tag: lost squareness info ----------------------------------
    if n == 1:
        d = home[0]
        bearing, depth = d["bearing_deg"], d["depth_cm"]
        # Close in we presumably funnelled in square already -> commit straight.
        # Otherwise creep forward holding the tag's bearing, to bring its
        # partner back into the frame (re-acquire the pair).
        if depth <= SINGLE_COMMIT_DEPTH_CM:
            return _cmd("commit", _fwd_rpm(bearing, COMMIT_RPM), _turn_rpm(bearing),
                        n_tags=1, range_cm=depth, bearing_deg=bearing,
                        reason=f"1 tag @ {depth:.0f}cm -> commit straight in")
        return _cmd("creep", _fwd_rpm(bearing, CREEP_RPM), _turn_rpm(bearing),
                    n_tags=1, range_cm=depth, bearing_deg=bearing,
                    reason=f"1 tag @ {depth:.0f}cm -> creep to re-acquire pair")

    # --- two tags: full pose --------------------------------------------
    g = _pose_from_pair(home[0], home[1])
    edge = g["cx_min"] < EDGE_MARGIN_PX or g["cx_max"] > (WIDTH - EDGE_MARGIN_PX)
    arrived = g["range_p"] <= WAYPOINT_TOL_CM        # on the normal at standoff

    # Commit: square on the normal, OR a tag is about to leave the frame, OR
    # already inside the close range. Steer onto the dock centre and push in;
    # when the tags drop out the home loop holds this straight velocity briefly.
    if edge or arrived or g["range_cm"] <= COMMIT_RANGE_CM:
        why = ("tag at frame edge" if edge else
               "squared at standoff" if arrived else "within commit range")
        return _cmd("commit", _fwd_rpm(g["bearing_deg"], COMMIT_RPM),
                    _turn_rpm(g["bearing_deg"]), n_tags=2,
                    range_cm=g["range_cm"], bearing_deg=g["bearing_deg"],
                    skew_deg=g["skew_deg"],
                    reason=f"{why} -> commit {g['range_cm']:.0f}cm")

    # Otherwise funnel: steer toward the standoff point on the normal.
    return _cmd("funnel", _fwd_rpm(g["bearing_p"], FWD_CRUISE_RPM),
                _turn_rpm(g["bearing_p"]), n_tags=2,
                range_cm=g["range_cm"], bearing_deg=g["bearing_deg"],
                skew_deg=g["skew_deg"],
                reason=f"funnel to standoff {g['range_p']:.0f}cm @ "
                       f"{g['bearing_p']:+.0f}deg, skew {g['skew_deg']:+.0f}deg")


def execute(robot, cmd):
    """Apply a homing velocity to a UnibotDrive (non-blocking set_velocity).

    Returns immediately so the perception loop keeps running; the velocity is
    re-issued every frame by the next plan()/execute(). TURN_SIGN / FWD_SIGN
    adapt the camera-frame command when the homing camera is rear-mounted.
    """
    robot.set_velocity(forward_rpm=cmd["forward_rpm"] * FWD_SIGN,
                       turn_rpm=cmd["turn_rpm"] * TURN_SIGN)


def home(robot, hometags, on_frame=None):
    """Drive the robot into the dock between `hometags`. Blocks until seated.

    Runs the continuous loop: detect -> plan -> set_velocity, re-adjusting the
    velocity every frame. Because set_velocity is non-blocking the wheels keep
    turning between frames, so the robot is always moving on the latest plan.

    Termination (the blind seat): once we have committed and the tags have left
    the frame (we are right on the dock), we hold a straight forward velocity
    for BLIND_SEAT_SEC and then brake - no vision is needed for that last bit
    because we entered square and centred. `on_frame(cmd)` is an optional hook
    for logging/preview.
    """
    committed = False
    blind_start = None
    try:
        while True:
            cmd = plan(detect_tags(take_photo()), hometags)
            if on_frame is not None:
                on_frame(cmd)
            committed = committed or cmd["action"] == "commit"

            # Tags gone after committing -> we are at the dock: push straight a
            # moment longer, then brake and finish.
            if committed and cmd["n_tags"] == 0:
                if blind_start is None:
                    blind_start = time.time()
                robot.set_velocity(forward_rpm=COMMIT_RPM * FWD_SIGN, turn_rpm=0.0)
                if time.time() - blind_start >= BLIND_SEAT_SEC:
                    robot.stop()
                    print("Seated. Homing complete.")
                    return
                continue

            blind_start = None          # tags back in view -> cancel blind seat
            execute(robot, cmd)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        robot.stop()


def main():
    """Preview the planned homing velocities (no driving).

        python apriltaghoming.py 14 15      # HOMETAGS = ids 14 and 15
    """
    import sys
    hometags = tuple(int(a) for a in sys.argv[1:]) or (14, 15)

    print(f"Homing preview for HOMETAGS {hometags}... Ctrl+C to stop.")
    try:
        while True:
            t0 = time.time()
            cmd = plan(detect_tags(take_photo()), hometags)
            rng = "  --" if cmd["range_cm"] is None else f"{cmd['range_cm']:5.0f}"
            print(f"  {cmd['action'].upper():7} fwd {cmd['forward_rpm']:5.1f}rpm  "
                  f"turn {cmd['turn_rpm']:+6.1f}rpm  range {rng}cm  | {cmd['reason']}")
            fps = 1.0 / max(time.time() - t0, 1e-3)
            print(f"  [{fps:4.1f} FPS]")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        close_camera()


if __name__ == "__main__":
    main()
