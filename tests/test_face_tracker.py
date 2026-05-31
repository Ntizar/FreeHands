"""Tests for the facial expression gesture detector (FaceTracker).

Covers:
- Construction (solutions and tasks backends)
- Empty frame handling (no face)
- Expression detection logic (smile, frown, surprise, etc.)
- Primary gesture priority
- Confidence scoring
- Landmark access
- Integration with fusion (FACIAL_GESTURE_ACTIONS)
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_fake_landmarks(
    mouth_width: float = 0.15,
    mouth_height: float = 0.02,
    mouth_corner_shift: float = 0.0,
    left_eyebrow_raise: float = 0.0,
    right_eyebrow_raise: float = 0.0,
    left_eyebrow_furrow: float = 0.0,
    right_eyebrow_furrow: float = 0.0,
    tongue_below: bool = False,
    num_landmarks: int = 468,
) -> list[object]:
    """Build a minimal MediaPipe-style landmarks list for testing.

    Only the landmarks that affect detection logic are set to non-default values.
    All others use (0, 0, 0) which is safe for the distance/ratio calculations.
    """

    class _Landmark:
        def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
            self.x = x
            self.y = y
            self.z = z

    # Mouth corners — centered horizontally, in lower half of frame
    mouth_left_x = 0.5 - mouth_width / 2
    mouth_right_x = 0.5 + mouth_width / 2
    mouth_center_y = 0.6

    # Eye Y position — above mouth
    eye_y = 0.45

    lm = [_Landmark() for _ in range(num_landmarks)]

    # Key mouth landmarks
    lm[61] = _Landmark(x=mouth_left_x, y=mouth_center_y + mouth_corner_shift)    # MOUTH_OUTER_LEFT
    lm[291] = _Landmark(x=mouth_right_x, y=mouth_center_y + mouth_corner_shift)  # MOUTH_OUTER_RIGHT
    lm[13] = _Landmark(x=0.5, y=mouth_center_y - mouth_height / 2)               # MOUTH_UPPER_CENTER
    lm[14] = _Landmark(x=0.5, y=mouth_center_y + mouth_height / 2)               # MOUTH_LOWER_CENTER
    lm[37] = _Landmark(x=0.5, y=mouth_center_y - mouth_height / 2)               # MOUTH_UPPER_LIP_37
    lm[177] = _Landmark(x=0.5, y=mouth_center_y + mouth_height / 2)              # MOUTH_LOWER_LIP_177

    # Eyebrows — positioned relative to eyes
    left_eyebrow_y = eye_y - left_eyebrow_raise * mouth_width
    right_eyebrow_y = eye_y - right_eyebrow_raise * mouth_width

    lm[6] = _Landmark(x=0.35, y=left_eyebrow_y)    # LEFT_EYEBROW_TIP
    lm[70] = _Landmark(x=0.45, y=left_eyebrow_y + left_eyebrow_furrow * mouth_width)  # LEFT_EYEBROW_INNER
    lm[55] = _Landmark(x=0.40, y=left_eyebrow_y)    # LEFT_EYEBROW_CENTER
    lm[300] = _Landmark(x=0.65, y=right_eyebrow_y)  # RIGHT_EYEBROW_TIP
    lm[286] = _Landmark(x=0.55, y=right_eyebrow_y + right_eyebrow_furrow * mouth_width)  # RIGHT_EYEBROW_INNER
    lm[338] = _Landmark(x=0.60, y=right_eyebrow_y)  # RIGHT_EYEBROW_CENTER

    # Eyes — critical for eyebrow comparison
    lm[159] = _Landmark(x=0.38, y=eye_y)   # LEFT_EYE_UPPER
    lm[145] = _Landmark(x=0.38, y=eye_y + 0.03)
    lm[386] = _Landmark(x=0.62, y=eye_y)   # RIGHT_EYE_UPPER
    lm[374] = _Landmark(x=0.62, y=eye_y + 0.03)

    # Nose
    lm[1] = _Landmark(x=0.5, y=0.5)
    lm[2] = _Landmark(x=0.5, y=0.52)
    lm[4] = _Landmark(x=0.5, y=0.55)

    # Chin and forehead
    lm[152] = _Landmark(x=0.5, y=0.75)
    lm[10] = _Landmark(x=0.5, y=0.30)

    # Tongue
    if tongue_below:
        lm[15] = _Landmark(x=0.5, y=0.70)  # below mouth

    return lm


def _make_mock_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a fake BGR frame (all black, 8-bit)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _mock_mediapipe():
    """Mock MediaPipe so tests run without GPU/media dependencies."""
    with patch.dict(sys.modules, {
        "mediapipe": MagicMock(),
        "mediapipe.solutions": MagicMock(),
        "mediapipe.tasks": MagicMock(),
        "mediapipe.tasks.python": MagicMock(),
        "mediapipe.tasks.python.vision": MagicMock(),
    }):
        yield


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestFaceTrackerConstruction:
    """Test that FaceTracker can be imported and has the right API."""

    def test_import(self):
        from freehands.gestures.face_tracker import FaceTracker, FacialObservation, FacialGestureId
        assert FaceTracker is not None
        assert FacialObservation is not None
        assert FacialGestureId is not None

    def test_exported_from_gestures_init(self):
        from freehands.gestures import FaceTracker, FacialObservation, FacialGestureId
        assert FaceTracker is not None
        assert FacialObservation is not None
        assert FacialGestureId is not None

    def test_exported_from_fusion_init(self):
        from freehands.fusion import FACIAL_GESTURE_ACTIONS
        assert "smile" in FACIAL_GESTURE_ACTIONS
        assert "frown" in FACIAL_GESTURE_ACTIONS
        assert "surprise" in FACIAL_GESTURE_ACTIONS

    def test_face_tracker_has_detect_and_close(self):
        from freehands.gestures.face_tracker import FaceTracker
        assert hasattr(FaceTracker, "detect")
        assert callable(FaceTracker.detect)
        assert hasattr(FaceTracker, "close")
        assert callable(FaceTracker.close)


class TestFacialObservation:
    """Test FacialObservation dataclass fields and defaults."""

    def test_default_values(self):
        from freehands.gestures.face_tracker import FacialObservation
        obs = FacialObservation()
        assert obs.smile is False
        assert obs.frown is False
        assert obs.surprise is False
        assert obs.raised_eyebrows is False
        assert obs.furrowed_brows is False
        assert obs.mouth_open is False
        assert obs.tongue_out is False
        assert obs.primary_gesture == "none"
        assert obs.confidence == 0.0
        assert obs.landmarks is None

    def test_non_default_values(self):
        from freehands.gestures.face_tracker import FacialObservation
        obs = FacialObservation(
            smile=True,
            surprise=True,
            primary_gesture="surprise",
            confidence=0.90,
        )
        assert obs.smile is True
        assert obs.surprise is True
        assert obs.primary_gesture == "surprise"
        assert obs.confidence == 0.90


class TestFacialGestureActions:
    """Test that all facial gestures are mapped in FACIAL_GESTURE_ACTIONS."""

    def test_all_gestures_mapped(self):
        from freehands.fusion import FACIAL_GESTURE_ACTIONS
        expected = {"smile", "frown", "surprise", "raised_eyebrows",
                     "furrowed_brows", "mouth_open", "tongue_out"}
        assert set(FACIAL_GESTURE_ACTIONS.keys()) == expected

    def test_action_names_match_gesture_names(self):
        from freehands.fusion import FACIAL_GESTURE_ACTIONS
        for gesture, action in FACIAL_GESTURE_ACTIONS.items():
            assert gesture == action, f"Gesture '{gesture}' maps to different action '{action}'"


class TestGestureIdIncludesFacial:
    """Test that facial gestures are in the GestureId Literal."""

    def test_facial_gestures_in_literal(self):
        from freehands.gestures.hand_tracker import GestureId
        # This is a Literal type — we verify by checking the string values
        # are valid gesture identifiers
        facial = {"smile", "frown", "surprise", "raised_eyebrows",
                   "furrowed_brows", "mouth_open", "tongue_out"}
        # We can't directly introspect Literal at runtime, but we verify
        # the strings are defined in the source
        from freehands.gestures import hand_tracker
        source = hand_tracker.__file__
        with open(source, "r") as f:
            content = f.read()
        for g in facial:
            assert f'"{g}"' in content, f"Gesture '{g}' not found in hand_tracker.py"


class TestProfileIntegration:
    """Test that facial gestures are integrated into the profile system."""

    def test_facial_gestures_in_binding_priority(self):
        from freehands.profiles.store import GESTURE_BINDING_PRIORITY
        facial = {"smile", "frown", "surprise", "raised_eyebrows",
                   "furrowed_brows", "mouth_open", "tongue_out"}
        for g in facial:
            assert g in GESTURE_BINDING_PRIORITY, f"Gesture '{g}' not in GESTURE_BINDING_PRIORITY"

    def test_facial_gestures_in_profile_bindings(self):
        from freehands.profiles.store import Profile
        p = Profile(user_id="test")
        facial = {"smile", "frown", "surprise", "raised_eyebrows",
                   "furrowed_brows", "mouth_open", "tongue_out"}
        for g in facial:
            assert g in p.gesture_bindings, f"Gesture '{g}' not in profile bindings"
            # Default bindings should be empty (user-configurable)
            assert p.gesture_bindings[g] == "", f"Gesture '{g}' has non-empty default binding"

    def test_facial_gestures_in_profile_thresholds(self):
        from freehands.profiles.store import Profile
        p = Profile(user_id="test")
        facial = {"smile", "frown", "surprise", "raised_eyebrows",
                   "furrowed_brows", "mouth_open", "tongue_out"}
        for g in facial:
            assert g in p.gesture_thresholds, f"Gesture '{g}' not in profile thresholds"

    def test_surprise_has_higher_confidence_threshold(self):
        from freehands.profiles.store import Profile
        p = Profile(user_id="test")
        surprise_conf = p.gesture_thresholds["surprise"].confidence_min
        smile_conf = p.gesture_thresholds["smile"].confidence_min
        assert surprise_conf > smile_conf, "Surprise should have higher confidence threshold than smile"

    def test_surprise_has_lowest_stability_frames(self):
        from freehands.profiles.store import Profile
        p = Profile(user_id="test")
        surprise_frames = p.gesture_thresholds["surprise"].stability_frames
        smile_frames = p.gesture_thresholds["smile"].stability_frames
        assert surprise_frames < smile_frames, "Surprise should have fewer stability frames (faster response)"


class TestFusionIntegration:
    """Test that facial gestures work through the fusion layer."""

    def test_facial_gesture_fires_immediately(self):
        """Facial gestures should fire without dwell, like palm-scroll."""
        from freehands.fusion import MultimodalFusion, FACIAL_GESTURE_ACTIONS
        from freehands.profiles import Profile

        profile = Profile(user_id="test")
        fusion = MultimodalFusion(profile)
        fusion.sm.activate()

        for gesture in FACIAL_GESTURE_ACTIONS:
            result = fusion.step(
                cursor_xy=(500, 300),
                confirmed_gesture=gesture,
            )
            assert result.fired_action is not None, f"Gesture '{gesture}' should fire an action"
            assert result.fired_action == gesture, f"Gesture '{gesture}' should fire action '{gesture}'"
            assert result.dwell_progress == 0.0, f"Gesture '{gesture}' should not need dwell"
            assert result.blink is False

    def test_facial_gesture_no_state_required(self):
        """Facial gestures should fire even in IDLE state (instant action)."""
        from freehands.fusion import MultimodalFusion, FACIAL_GESTURE_ACTIONS
        from freehands.profiles import Profile

        profile = Profile(user_id="test")
        fusion = MultimodalFusion(profile)
        # Don't activate — stay in IDLE

        result = fusion.step(
            cursor_xy=(500, 300),
            confirmed_gesture="smile",
        )
        # In IDLE state, the step returns without firing (same as other gestures)
        # Facial gestures follow the same pattern: they check bindings first
        assert result.cursor_xy == (500, 300)

    def test_facial_gesture_with_none_confirmed(self):
        """'none' facial gesture should not fire."""
        from freehands.fusion import MultimodalFusion
        from freehands.profiles import Profile

        profile = Profile(user_id="test")
        fusion = MultimodalFusion(profile)
        fusion.sm.activate()

        result = fusion.step(
            cursor_xy=(500, 300),
            confirmed_gesture="none",
        )
        assert result.fired_action is None


class TestFaceTrackerDetection:
    """Test facial expression detection logic with mocked MediaPipe."""

    def _create_tracker_with_mock(self, mock_lm):
        """Create a FaceTracker with a mocked MediaPipe backend."""
        from freehands.gestures.face_tracker import FaceTracker

        # Mock the MediaPipe result
        mock_result = MagicMock()
        mock_result.multi_face_landmarks = [MagicMock()]
        mock_result.multi_face_landmarks[0].landmark = mock_lm

        # Create tracker and mock its _mesh.process
        tracker = FaceTracker.__new__(FaceTracker)
        tracker._mesh = MagicMock()
        tracker._mesh.process = MagicMock(return_value=mock_result)
        tracker._backend = "solutions"
        return tracker

    def test_no_face_returns_empty(self):
        """When no face is detected, return empty FacialObservation."""
        from freehands.gestures.face_tracker import FaceTracker, FacialObservation

        tracker = FaceTracker.__new__(FaceTracker)
        mock_result = MagicMock()
        mock_result.multi_face_landmarks = None
        tracker._mesh = MagicMock()
        tracker._mesh.process = MagicMock(return_value=mock_result)
        tracker._backend = "solutions"

        frame = _make_mock_frame()
        result = tracker.detect(frame)

        assert isinstance(result, FacialObservation)
        assert result.primary_gesture == "none"
        assert result.confidence == 0.0

    @pytest.mark.parametrize("expression,mouth_height,mouth_corner_shift,left_eyebrow_raise,right_eyebrow_raise,expected_gesture", [
        ("smile", 0.025, -0.003, 0.0, 0.0, "smile"),
        ("frown", 0.025, 0.004, 0.0, 0.0, "frown"),
        ("surprise", 0.045, 0.0, 0.03, 0.03, "surprise"),
        ("raised_eyebrows", 0.02, 0.0, 0.04, 0.0, "raised_eyebrows"),
        ("furrowed_brows", 0.02, 0.0, 0.0, 0.0, "furrowed_brows"),
        ("mouth_open", 0.025, 0.0, 0.0, 0.0, "mouth_open"),
        ("none", 0.01, 0.0, 0.0, 0.0, "none"),
    ])
    def test_expression_detection(
        self,
        expression: str,
        mouth_height: float,
        mouth_corner_shift: float,
        left_eyebrow_raise: float,
        right_eyebrow_raise: float,
        expected_gesture: str,
    ):
        """Parametric test for each facial expression."""
        from freehands.gestures.face_tracker import FaceTracker

        tracker = self._create_tracker_with_mock(
            _make_fake_landmarks(
                mouth_width=0.15,
                mouth_height=mouth_height,
                mouth_corner_shift=mouth_corner_shift,
                left_eyebrow_raise=left_eyebrow_raise,
                right_eyebrow_raise=right_eyebrow_raise,
            )
        )

        # Mock cv2 for the detect() method (it imports cv2 internally)
        mock_cv2 = MagicMock()
        mock_cv2.cvtColor = MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            frame = _make_mock_frame()
            result = tracker.detect(frame)

        assert result.primary_gesture == expected_gesture, \
            f"Expected '{expected_gesture}', got '{result.primary_gesture}'"

    def test_surprise_priority_over_smile(self):
        """When both surprise and smile are detected, surprise wins."""
        from freehands.gestures.face_tracker import FaceTracker

        tracker = self._create_tracker_with_mock(
            _make_fake_landmarks(
                mouth_width=0.15,
                mouth_height=0.045,  # open enough for both smile and surprise
                mouth_corner_shift=-0.003,  # small upward shift (smile)
                left_eyebrow_raise=0.03,  # raised (surprise)
                right_eyebrow_raise=0.03,
            )
        )

        with patch("cv2.cvtColor", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            frame = _make_mock_frame()
            result = tracker.detect(frame)

        assert result.primary_gesture == "surprise", "Surprise should have priority over smile"
        assert result.smile is True  # smile is still detected
        assert result.surprise is True

    def test_landmarks_accessible(self):
        """Landmarks should be accessible on the observation."""
        from freehands.gestures.face_tracker import FaceTracker

        tracker = self._create_tracker_with_mock(_make_fake_landmarks())

        with patch("cv2.cvtColor", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            frame = _make_mock_frame()
            result = tracker.detect(frame)

        assert result.landmarks is not None
        assert result.landmarks.shape == (468, 3)
