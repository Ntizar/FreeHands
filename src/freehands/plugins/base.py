"""Base classes and types for FreeHands plugins.

Each plugin implements hooks into the FreeHands pipeline. The pipeline
runs in this order every frame:

    Camera → [on_frame] → Gaze → [on_gaze] → Filter → [on_filter]
    → Gesture → [on_gesture] → Fusion → [on_fusion] → Action → [on_action]
    → Overlay → [on_overlay]

Plugins can return early from any hook to short-circuit the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class PluginPhase(Enum):
    """Stages of the FreeHands pipeline where plugins can hook in."""
    CAMERA = auto()       # Before camera frame processing
    GAZE = auto()         # After gaze prediction
    FILTER = auto()       # After Kalman filter
    GESTURE = auto()      # After gesture detection
    FUSION = auto()       # After fusion step
    ACTION = auto()       # Before action dispatch
    OVERLAY = auto()      # After overlay update


@dataclass
class PluginContext:
    """Shared context passed to all plugin hooks in a single frame.

    Attributes:
        frame: Raw camera frame (numpy array) or None if unavailable.
        cursor: Current cursor position (x, y) or None.
        gesture: Confirmed gesture name or None.
        action: Action to execute or None.
        blink: Whether a blink was detected this frame.
        blink_event: Blink event type or None.
        voice_action: Voice command action or None.
        state: Current fusion state machine state.
        metadata: Free-form dict for plugins to share data across hooks.
    """
    frame: Any = None                    # numpy.ndarray | None
    cursor: tuple[int, int] | None = None
    gesture: str | None = None
    action: str | None = None
    blink: bool = False
    blink_event: str | None = None
    voice_action: str | None = None
    state: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)


class FreeHandsPlugin(ABC):
    """Base class for FreeHands plugins.

    Subclass and override any hook(s) you need. Only the hooks you
    override will be called — unimplemented hooks are silently skipped.

    Attributes:
        name: Unique plugin identifier (used in logs and config).
        version: Semantic version string.
        description: Human-readable description shown in UI.
        enabled_by_default: Whether the plugin is active without config.
        priority: Execution order (lower = earlier in pipeline).
    """

    name: str = "unnamed_plugin"
    version: str = "0.0.0"
    description: str = ""
    enabled_by_default: bool = True
    priority: int = 100

    # ── Lifecycle ────────────────────────────────────────────────────────

    def on_load(self) -> None:
        """Called when the plugin is loaded. Use for one-time setup."""
        pass

    def on_unload(self) -> None:
        """Called when the plugin is unloaded. Use for cleanup."""
        pass

    # ── Pipeline hooks ───────────────────────────────────────────────────
    # Each hook receives the input value and the shared context.
    # Return the modified value, or the original to pass through unchanged.

    def on_frame(self, frame: Any, ctx: PluginContext) -> Any:
        """Process camera frame before trackers run.

        Args:
            frame: Raw camera frame (numpy BGR array).
            ctx: Shared plugin context for this frame.

        Returns:
            Modified frame or the original frame.
        """
        return frame

    def on_gaze(self, cursor: tuple[int, int] | None, ctx: PluginContext) -> tuple[int, int] | None:
        """Process gaze cursor after prediction.

        Args:
            cursor: Predicted cursor position or None.
            ctx: Shared plugin context for this frame.

        Returns:
            Modified cursor or the original.
        """
        return cursor

    def on_filter(self, cursor: tuple[int, int] | None, ctx: PluginContext) -> tuple[int, int] | None:
        """Process cursor after filtering (Kalman, EMA, etc.).

        Args:
            cursor: Filtered cursor position or None.
            ctx: Shared plugin context for this frame.

        Returns:
            Modified cursor or the original.
        """
        return cursor

    def on_gesture(self, gesture: str | None, confidence: float, ctx: PluginContext) -> tuple[str | None, float]:
        """Process gesture detection result.

        Args:
            gesture: Detected gesture name or None.
            confidence: Detection confidence (0.0–1.0).
            ctx: Shared plugin context for this frame.

        Returns:
            (modified_gesture, modified_confidence) tuple.
        """
        return gesture, confidence

    def on_fusion(
        self,
        cursor: tuple[int, int] | None,
        gesture: str | None,
        action: str | None,
        ctx: PluginContext,
    ) -> tuple[tuple[int, int] | None, str | None, str | None]:
        """Process fusion output before action dispatch.

        Args:
            cursor: Current cursor position.
            gesture: Confirmed gesture name.
            action: Action proposed by fusion.
            ctx: Shared plugin context for this frame.

        Returns:
            (cursor, gesture, action) — possibly modified.
        """
        return cursor, gesture, action

    def on_action(self, action: str | None, ctx: PluginContext) -> str | None:
        """Intercept action before dispatch.

        Args:
            action: Action to execute or None.
            ctx: Shared plugin context for this frame.

        Returns:
            Modified action or None to suppress.
        """
        return action

    def on_overlay(self, ctx: PluginContext) -> None:
        """Update overlay after action execution.

        Args:
            ctx: Shared plugin context for this frame.
        """
        pass

    # ── Registration ─────────────────────────────────────────────────────

    @classmethod
    def get_plugin_info(cls) -> dict[str, str]:
        """Return plugin metadata for UI display.

        Returns:
            Dict with name, version, description keys.
        """
        return {
            "name": cls.name,
            "version": cls.version,
            "description": cls.description,
        }
