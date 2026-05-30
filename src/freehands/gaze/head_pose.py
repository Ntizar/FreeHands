"""6DoF head-pose estimation from MediaPipe FaceMesh landmarks.

Extracts yaw, pitch and roll from the 468-face-landmark set and exposes a
compact ``HeadPose`` dataclass that the fusion layer can consume as a
*coarse-displacement* channel.

Head-pose is used for large, fast cursor movements (pan/tilt) — the user
simply turns their head left/right or up/down and the cursor sweeps across
the screen proportionally.  This complements the fine-grained gaze pointer
which handles precise targeting.

Design decisions
----------------
* Uses a minimal set of stable landmarks (nose, eyes, chin, forehead) — no
  full PnP solver needed, keeping latency low at 30 fps.
* Returns normalised angles in radians: yaw ∈ [-π/2, π/2], pitch ∈ [-π/2, π/2].
* A dead-zone around zero prevents jitter from firing coarse movements.
* Coarse movement is clamped to a configurable ``max_pan`` / ``max_tilt``
  (screen-fraction) so a full head turn never teleports the cursor off-screen.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ── Landmark indices (MediaPipe FaceMesh, 468 points) ──────────────────────
NOSE_TIP = 1
NOSE_MID = 2
NOSE_BASE = 4
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 362
CHIN = 152
FOREHEAD = 10

# ── Thresholds ─────────────────────────────────────────────────────────────
HEAD_POSE_DEADZONE_RAD = 0.03  # ~1.7° dead zone around centre
HEAD_POSE_SENSITIVITY = 3.0    # screen-fraction per radian of head rotation
MAX_PAN_FRACTION = 0.5         # max ±50% of screen width per frame
MAX_TILT_FRACTION = 0.4        # max ±40% of screen height per frame


@dataclass
class HeadPose:
    """Normalised head-pose angles and derived coarse displacement."""
    yaw: float = 0.0       # radians, positive = looking right
    pitch: float = 0.0     # radians, positive = looking down
    roll: float = 0.0      # radians, positive = head tilted right
    confidence: float = 0.0  # 0..1, how reliable the estimate is
    coarse_dx: float = 0.0  # horizontal displacement (screen fraction)
    coarse_dy: float = 0.0  # vertical displacement (screen fraction)
    coarse_active: bool = False  # True when dead-zone exceeded


def estimate_head_pose(
    landmarks: list[object],
    frame_width: int,
    frame_height: int,
) -> HeadPose:
    """Estimate head pose from MediaPipe FaceMesh landmarks.

    Uses geometric relationships between key landmarks to estimate the
    orientation of the face relative to the camera.  The algorithm is
    heuristic and works best for moderate head rotations (±30°).

    Parameters
    ----------
    landmarks :
        List of MediaPipe normalized landmarks (468 points, each with x, y, z).
    frame_width :
        Frame width in pixels (used for aspect-ratio correction).
    frame_height :
        Frame height in pixels.

    Returns
    -------
    HeadPose
        Normalised angles and coarse displacement.
    """
    def pt(idx: int) -> np.ndarray:
        lm = landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    # ── Check landmark availability ──────────────────────────────────────
    required = [NOSE_TIP, NOSE_MID, NOSE_BASE, LEFT_EYE_OUTER,
                RIGHT_EYE_OUTER, CHIN, FOREHEAD]
    if not all(idx < len(landmarks) for idx in required):
        return HeadPose()

    nose_tip = pt(NOSE_TIP)
    left_outer = pt(LEFT_EYE_OUTER)
    right_outer = pt(RIGHT_EYE_OUTER)
    chin = pt(CHIN)
    forehead = pt(FOREHEAD)

    # ── Aspect-ratio correction ──────────────────────────────────────────
    aspect = frame_width / frame_height

    # ── Eye-line (horizontal reference) ─────────────────────────────────
    eye_center = (left_outer + right_outer) / 2.0
    eye_line = right_outer - left_outer  # left→right vector
    eye_line_ar = np.array([eye_line[0] * aspect, eye_line[1], eye_line[2]])
    eye_width = float(np.linalg.norm(eye_line))

    # ── Face axis (vertical reference) ──────────────────────────────────
    # Forehead → Chin defines the face's vertical axis.
    face_axis = chin - forehead
    face_axis_ar = np.array([face_axis[0] * aspect, face_axis[1], face_axis[2]])

    # ── Confidence ───────────────────────────────────────────────────────
    face_height = float(np.linalg.norm(face_axis))
    if eye_width < 0.02 or face_height < 0.05:
        confidence = 0.0
    else:
        confidence = min(1.0, eye_width * 5.0 + face_height * 2.0)
        confidence = max(0.0, confidence)

    # ── Yaw: lateral head rotation ──────────────────────────────────────
    # Nose tip shifts left/right relative to the eye midpoint when the
    # head turns.  The Z (depth) component of the nose provides additional
    # information: when the head turns, the nose moves toward one eye.
    nose_to_eye = nose_tip - eye_center
    nose_to_eye_ar = np.array([nose_to_eye[0] * aspect, nose_to_eye[1], nose_to_eye[2]])

    # Lateral offset in the XZ plane
    lateral = nose_to_eye_ar[0]
    depth = nose_to_eye_ar[2]
    yaw = np.arctan2(lateral, abs(depth) + 1e-8)
    yaw = np.clip(yaw, -np.pi / 2, np.pi / 2)

    # ── Pitch: vertical head rotation ───────────────────────────────────
    # When the head tilts up/down, the face axis rotates relative to the
    # horizontal eye-line.  We use the cross product of the face axis
    # with the eye-line to extract the pitch component.
    #
    # cross(face_axis, eye_line) gives a vector perpendicular to both.
    # The Y component of this cross product is proportional to pitch.
    cross = np.cross(face_axis_ar, eye_line_ar)
    # The magnitude of the cross product in the Y direction indicates
    # how much the face axis deviates from being perpendicular to the
    # eye-line (which is the neutral pose).
    face_axis_len = np.linalg.norm(face_axis_ar) + 1e-8
    eye_line_len = np.linalg.norm(eye_line_ar) + 1e-8
    # Normalized cross product Y component
    cross_y_norm = cross[1] / (face_axis_len * eye_line_len)
    # Map to pitch angle: cross_y_norm ≈ 0 means neutral, ±1 means extreme
    pitch = np.arcsin(np.clip(cross_y_norm, -1.0, 1.0))
    # Invert sign: positive pitch = looking down
    pitch = -pitch
    pitch = np.clip(pitch, -np.pi / 2, np.pi / 2)

    # ── Roll: sideways head tilt ────────────────────────────────────────
    # When the head tilts sideways, the eye-line rotates.
    eye_angle = np.arctan2(eye_line_ar[1], eye_line_ar[0] + 1e-8)
    roll = eye_angle
    roll = np.clip(roll, -np.pi / 4, np.pi / 4)

    # ── Coarse displacement ─────────────────────────────────────────────
    coarse_active = (abs(yaw) > HEAD_POSE_DEADZONE_RAD or
                     abs(pitch) > HEAD_POSE_DEADZONE_RAD)

    coarse_dx = yaw * HEAD_POSE_SENSITIVITY
    coarse_dy = pitch * HEAD_POSE_SENSITIVITY

    coarse_dx = np.clip(coarse_dx, -MAX_PAN_FRACTION, MAX_PAN_FRACTION)
    coarse_dy = np.clip(coarse_dy, -MAX_TILT_FRACTION, MAX_TILT_FRACTION)

    return HeadPose(
        yaw=float(yaw),
        pitch=float(pitch),
        roll=float(roll),
        confidence=float(confidence),
        coarse_dx=float(coarse_dx),
        coarse_dy=float(coarse_dy),
        coarse_active=bool(coarse_active),
    )


def head_pose_to_screen_delta(
    head_pose: HeadPose,
    screen_width: int,
    screen_height: int,
) -> tuple[int, int]:
    """Convert a HeadPose to pixel displacement for the cursor.

    Parameters
    ----------
    head_pose :
        The current head pose estimate.
    screen_width :
        Screen width in pixels.
    screen_height :
        Screen height in pixels.

    Returns
    -------
    tuple[int, int]
        (dx, dy) pixel displacement to apply to the cursor position.
        Returns (0, 0) when head pose is within dead zone.
    """
    if not head_pose.coarse_active:
        return (0, 0)

    dx = int(head_pose.coarse_dx * screen_width)
    dy = int(head_pose.coarse_dy * screen_height)

    return (dx, dy)
