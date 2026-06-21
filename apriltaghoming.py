import math
import time

import cv2  # type: ignore
from pupil_apriltags import Detector  # type: ignore

# importing capture starts both cameras (idempotent: motion.py already did this)
from capture import take_photo, WIDTH, HEIGHT

TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.100           # all arena tags are 100 mm

#cam param
FOV_DEG = 60.0
FOCAL_PX = (WIDTH / 2.0) / math.tan(math.radians(FOV_DEG / 2.0))
CAMERA_PARAMS = (FOCAL_PX, FOCAL_PX, WIDTH / 2.0, HEIGHT / 2.0) # pupil_apriltags wants (fx, fy, cx, cy)

_detector = Detector(families=TAG_FAMILY)

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
    print(dets)
    return dets


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