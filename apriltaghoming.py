import math
import time

import cv2  # type: ignore
from pupil_apriltags import Detector  # type: ignore

# importing capture starts both cameras (idempotent: motion.py already did this)
from capture import take_photo, WIDTH, HEIGHT, FOV_DEG

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

# plan the next move with a cmd rturn
def plan(dets, teamtags):
    """Decide the next action based on tag detections.
    
    Returns a command dict:
        action    "home" : Final stage of homing, called when (phi > 10 deg)
                  "approach" : Found any tag, adjusting position if (phi > 10 deg)
                  "search" : No target tag/tag found.
                  
        bearing   steer toward this angle
        depth_cm  estimated depth of the target (or None)
    """
    team_dets = [d for d in dets if d["id"] in teamtags]
    if not team_dets:
        return {
            "action": "search",
            "bearing": 0.0,
            "depth_cm": None,
        }

    # Calculate target position (midpoint of both if 2, or the single tag if 1)
    if len(team_dets) == 2:
        d1, d2 = team_dets[0], team_dets[1]
        l_1, l_2 = d1["depth_cm"], d2["depth_cm"]
        depth_cm = (l_1 + l_2) / 2.0
        lateral_cm = (d1["lateral_cm"] + d2["lateral_cm"]) / 2.0
        bearing = math.degrees(math.atan2(lateral_cm, depth_cm))
        phi = math.acos((l_1**2+l_2**2-50*50)/2*l_1*l_2)
        ids = [d1["id"], d2["id"]]
    else:
        d = team_dets[0]
        depth_cm = d["depth_cm"]
        lateral_cm = d["lateral_cm"]
        bearing = d["bearing_deg"]
        ids = [d["id"]]

    # Centred and close -> commit to final seat
    if depth_cm <= FINAL_SEAT_CM and abs(bearing) <= CENTER_DEADBAND_DEG:
        return {
            "action": "seat",
            "bearing": bearing,
            "depth_cm": depth_cm,     
        }

    # Not centred or not close enough yet
    if abs(bearing) < FACING_DEG:
        return {
            "action": "home",
            "bearing": bearing,
            "depth_cm": depth_cm,
        }
    else:
        return {
            "action": "align",
            "bearing": bearing,
            "depth_cm": depth_cm,
        }
