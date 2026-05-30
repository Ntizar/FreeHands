"""Unit tests for 6DoF head-pose estimation."""
from __future__ import annotations

import numpy as np
import pytest

from freehands.gaze.head_pose import (
    HeadPose,
    estimate_head_pose,
    head_pose_to_screen_delta,
    MAX_PAN_FRACTION,
    MAX_TILT_FRACTION,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _lm(x: float, y: float, z: float = 0.0) -> object:
    """Create a fake MediaPipe landmark with x, y, z attributes."""
    return type("Landmark", (), {"x": x, "y": y, "z": z})()


def _face(
    nose_x: float = 0.5,
    nose_y: float = 0.45,
    nose_z: float = -0.05,
    left_outer_x: float = 0.42,
    left_outer_y: float = 0.40,
    right_outer_x: float = 0.58,
    right_outer_y: float = 0.40,
    chin_y: float = 0.65,
    chin_z: float = 0.0,
    forehead_y: float = 0.25,
    forehead_z: float = 0.0,
    count: int = 468,
) -> list[object]:
    """Build a minimal 468-landmark list with key points set."""
    landmarks: list[object] = []
    for i in range(count):
        landmarks.append(_lm(0.5, 0.5, 0.0))
    landmarks[1] = _lm(nose_x, nose_y, nose_z)          # NOSE_TIP
    landmarks[2] = _lm(0.5, 0.48, 0.0)                  # NOSE_MID
    landmarks[4] = _lm(0.5, 0.52, 0.0)                  # NOSE_BASE
    landmarks[33] = _lm(left_outer_x, left_outer_y, 0.0)  # LEFT_EYE_OUTER
    landmarks[362] = _lm(right_outer_x, right_outer_y, 0.0)  # RIGHT_EYE_OUTER
    landmarks[152] = _lm(0.5, chin_y, chin_z)           # CHIN
    landmarks[10] = _lm(0.5, forehead_y, forehead_z)    # FOREHEAD
    return landmarks


# ── Tests ────────────────────────────────────────────────────────────────


class TestEstimateHeadPose:
    """Tests for estimate_head_pose()."""

    def test_neutral_face_returns_zero_angles(self) -> None:
        """A face looking straight at the camera should have near-zero angles."""
        hp = estimate_head_pose(_face(), 640, 480)
        assert abs(hp.yaw) < 0.01, f"yaw={hp.yaw}"
        assert abs(hp.pitch) < 0.01, f"pitch={hp.pitch}"
        assert abs(hp.roll) < 0.01, f"roll={hp.roll}"
        assert hp.confidence > 0.5
        assert not hp.coarse_active

    def test_head_turned_right_positive_yaw(self) -> None:
        """Nose shifted right relative to eyes → positive yaw."""
        hp = estimate_head_pose(_face(nose_x=0.55), 640, 480)
        assert hp.yaw > 0.1, f"Expected positive yaw, got {hp.yaw}"
        assert hp.coarse_active

    def test_head_turned_left_negative_yaw(self) -> None:
        """Nose shifted left relative to eyes → negative yaw."""
        hp = estimate_head_pose(_face(nose_x=0.45), 640, 480)
        assert hp.yaw < -0.1, f"Expected negative yaw, got {hp.yaw}"
        assert hp.coarse_active

    def test_head_tilted_down_negative_pitch(self) -> None:
        """Chin toward camera → negative pitch (looking down)."""
        hp = estimate_head_pose(_face(chin_z=0.1, forehead_z=-0.1), 640, 480)
        assert hp.pitch < -0.1, f"Expected negative pitch, got {hp.pitch}"
        assert hp.coarse_active

    def test_head_tilted_up_positive_pitch(self) -> None:
        """Chin away from camera → positive pitch (looking up)."""
        hp = estimate_head_pose(_face(chin_z=-0.1, forehead_z=0.1), 640, 480)
        assert hp.pitch > 0.1, f"Expected positive pitch, got {hp.pitch}"
        assert hp.coarse_active

    def test_confidence_zero_with_insufficient_landmarks(self) -> None:
        """If landmarks are too close together, confidence should be near zero."""
        hp = estimate_head_pose(
            _face(
                left_outer_x=0.499, left_outer_y=0.499,
                right_outer_x=0.501, right_outer_y=0.501,
                chin_y=0.501, forehead_y=0.499,
            ),
            640,
            480,
        )
        assert hp.confidence < 0.1

    def test_angles_clamped_to_range(self) -> None:
        """Yaw should be clamped to [-π/2, π/2]."""
        hp = estimate_head_pose(_face(nose_x=0.95, nose_y=0.45), 640, 480)
        assert -np.pi / 2 <= hp.yaw <= np.pi / 2
        assert -np.pi / 2 <= hp.pitch <= np.pi / 2

    def test_short_landmark_list_returns_empty(self) -> None:
        """If fewer landmarks than required, return empty HeadPose."""
        short = [_lm(0.5, 0.5)] * 10
        hp = estimate_head_pose(short, 640, 480)
        assert hp.yaw == 0.0
        assert hp.pitch == 0.0
        assert hp.confidence == 0.0

    def test_yaw_symmetry(self) -> None:
        """Equal but opposite nose offsets should give symmetric yaw values."""
        hp_right = estimate_head_pose(_face(nose_x=0.55), 640, 480)
        hp_left = estimate_head_pose(_face(nose_x=0.45), 640, 480)
        np.testing.assert_allclose(hp_right.yaw, -hp_left.yaw, rtol=0.01)


class TestHeadPoseToScreenDelta:
    """Tests for head_pose_to_screen_delta()."""

    def test_dead_zone_returns_zero(self) -> None:
        """Head within dead zone → no displacement."""
        hp = HeadPose(yaw=0.01, pitch=0.01, confidence=0.9)
        dx, dy = head_pose_to_screen_delta(hp, 1920, 1080)
        assert dx == 0
        assert dy == 0

    def test_active_head_returns_scaled_delta(self) -> None:
        """Head outside dead zone → proportional pixel displacement."""
        hp = HeadPose(
            yaw=0.5, pitch=0.3, confidence=0.9, coarse_active=True,
            coarse_dx=0.15, coarse_dy=0.1,
        )
        dx, dy = head_pose_to_screen_delta(hp, 1920, 1080)
        assert dx != 0
        assert dy != 0

    def test_delta_clamped_to_max_screen_fraction(self) -> None:
        """The estimate_head_pose function clamps coarse values."""
        # Extreme head pose → coarse_dx/coarse_dy should be clamped
        hp = estimate_head_pose(_face(nose_x=0.95, nose_z=0.5, chin_z=-0.5), 1920, 1080)
        assert hp.coarse_active
        assert abs(hp.coarse_dx) <= MAX_PAN_FRACTION
        assert abs(hp.coarse_dy) <= MAX_TILT_FRACTION

    def test_inactive_head_returns_zero(self) -> None:
        """head_pose_active=False → no displacement."""
        hp = HeadPose(
            yaw=1.0, pitch=0.5, confidence=0.9, coarse_active=False,
        )
        dx, dy = head_pose_to_screen_delta(hp, 1920, 1080)
        assert dx == 0
        assert dy == 0

    def test_different_screen_sizes_scale_proportionally(self) -> None:
        """Larger screens should get proportionally larger deltas."""
        hp = HeadPose(
            yaw=0.1, pitch=0.05, confidence=0.9, coarse_active=True,
            coarse_dx=0.1, coarse_dy=0.05,
        )
        dx_1080, dy_1080 = head_pose_to_screen_delta(hp, 1920, 1080)
        dx_4k, dy_4k = head_pose_to_screen_delta(hp, 3840, 2160)
        assert abs(dx_4k) > abs(dx_1080)
        assert abs(dy_4k) > abs(dy_1080)

    def test_delta_direction_matches_pose(self) -> None:
        """Positive yaw → positive dx; positive pitch → positive dy."""
        hp_right = HeadPose(
            yaw=0.5, pitch=0.0, confidence=0.9, coarse_active=True,
            coarse_dx=0.2, coarse_dy=0.0,
        )
        dx_r, _ = head_pose_to_screen_delta(hp_right, 1920, 1080)
        assert dx_r > 0

        hp_up = HeadPose(
            yaw=0.0, pitch=0.5, confidence=0.9, coarse_active=True,
            coarse_dx=0.0, coarse_dy=0.2,
        )
        _, dy_u = head_pose_to_screen_delta(hp_up, 1920, 1080)
        assert dy_u > 0
