"""Tests for the FreeHands plugin system (improvement #16)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# ── Import strategy for sandbox ──────────────────────────────────────────────
# The sandbox lacks heavy deps (mediapipe, PyQt6, cv2) that transitively
# load through __init__.py files. We load plugin modules directly via
# importlib.util to bypass the __init__.py import chain.

def _load_module(name: str, path: str) -> ModuleType:
    """Load a .py module directly, bypassing __init__.py chains."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load base module directly (no heavy deps)
_base_mod = _load_module(
    "freehands.plugins.base",
    str(Path(__file__).parent.parent / "src" / "freehands" / "plugins" / "base.py"),
)
FreeHandsPlugin = _base_mod.FreeHandsPlugin
PluginContext = _base_mod.PluginContext
PluginPhase = _base_mod.PluginPhase

# Load loader module (depends on base, already in sys.modules)
_loader_mod = _load_module(
    "freehands.plugins.loader",
    str(Path(__file__).parent.parent / "src" / "freehands" / "plugins" / "loader.py"),
)
PluginLoader = _loader_mod.PluginLoader


# ── Test plugin fixtures ─────────────────────────────────────────────────────

class TracingPlugin(FreeHandsPlugin):
    """Plugin that records which hooks were called."""

    name = "tracing"
    version = "1.0.0"
    description = "Records hook calls"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def on_load(self) -> None:
        self.calls.append("on_load")

    def on_unload(self) -> None:
        self.calls.append("on_unload")

    def on_frame(self, frame: Any, ctx: PluginContext) -> Any:
        self.calls.append("on_frame")
        return frame

    def on_gaze(self, cursor, ctx: PluginContext):
        self.calls.append("on_gaze")
        return cursor

    def on_filter(self, cursor, ctx: PluginContext):
        self.calls.append("on_filter")
        return cursor

    def on_gesture(self, gesture, confidence, ctx: PluginContext):
        self.calls.append("on_gesture")
        return gesture, confidence

    def on_fusion(self, cursor, gesture, action, ctx: PluginContext):
        self.calls.append("on_fusion")
        return cursor, gesture, action

    def on_action(self, action, ctx: PluginContext):
        self.calls.append("on_action")
        return action

    def on_overlay(self, ctx: PluginContext):
        self.calls.append("on_overlay")


class CursorModifyingPlugin(FreeHandsPlugin):
    """Plugin that modifies the cursor position."""

    name = "cursor_modifier"
    version = "1.0.0"
    description = "Shifts cursor by offset"
    priority = 10

    def __init__(self, dx: int = 10, dy: int = 20) -> None:
        super().__init__()
        self.dx = dx
        self.dy = dy

    def on_gaze(self, cursor, ctx: PluginContext):
        if cursor is not None:
            return (cursor[0] + self.dx, cursor[1] + self.dy)
        return cursor


class ActionSuppressingPlugin(FreeHandsPlugin):
    """Plugin that suppresses specific actions."""

    name = "action_suppressor"
    version = "1.0.0"
    description = "Suppresses 'zoom_in' action"

    def on_action(self, action, ctx: PluginContext):
        if action == "zoom_in":
            return None
        return action


class MetadataPlugin(FreeHandsPlugin):
    """Plugin that writes to context metadata."""

    name = "metadata_writer"
    version = "1.0.0"
    description = "Writes to plugin context metadata"

    def on_gaze(self, cursor, ctx: PluginContext):
        ctx.metadata["plugin_ran"] = True
        ctx.metadata["cursor_seen"] = cursor
        return cursor


# ── PluginContext tests ──────────────────────────────────────────────────────

def test_plugin_context_defaults() -> None:
    """PluginContext should have sensible defaults."""
    ctx = PluginContext()
    assert ctx.frame is None
    assert ctx.cursor is None
    assert ctx.gesture is None
    assert ctx.action is None
    assert ctx.blink is False
    assert ctx.blink_event is None
    assert ctx.voice_action is None
    assert ctx.state == "active"
    assert ctx.metadata == {}


def test_plugin_context_with_values() -> None:
    """PluginContext should accept custom values."""
    ctx = PluginContext(
        frame="test_frame",
        cursor=(100, 200),
        gesture="pointing_up",
        action="click",
        blink=True,
        state="confirming",
    )
    assert ctx.frame == "test_frame"
    assert ctx.cursor == (100, 200)
    assert ctx.gesture == "pointing_up"
    assert ctx.action == "click"
    assert ctx.blink is True
    assert ctx.state == "confirming"
    assert ctx.metadata == {}


# ── Plugin base class tests ──────────────────────────────────────────────────

def test_plugin_info_class_method() -> None:
    """get_plugin_info should return plugin metadata."""
    info = TracingPlugin.get_plugin_info()
    assert info["name"] == "tracing"
    assert info["version"] == "1.0.0"
    assert info["description"] == "Records hook calls"


def test_base_hooks_return_passthrough() -> None:
    """Base class hooks should return their inputs unchanged."""
    plugin = FreeHandsPlugin()
    ctx = PluginContext()

    assert plugin.on_frame("frame", ctx) == "frame"
    assert plugin.on_gaze((1, 2), ctx) == (1, 2)
    assert plugin.on_filter(None, ctx) is None
    assert plugin.on_gesture("test", 0.5, ctx) == ("test", 0.5)
    assert plugin.on_fusion((1, 2), "g", "a", ctx) == ((1, 2), "g", "a")
    assert plugin.on_action("click", ctx) == "click"
    # on_overlay returns None
    assert plugin.on_overlay(ctx) is None


# ── PluginLoader tests ───────────────────────────────────────────────────────

def test_loader_empty() -> None:
    """Loader with no plugins should have zero active plugins."""
    loader = PluginLoader()
    assert len(loader) == 0
    assert loader.active_plugins == []
    assert loader.get_plugin("any") is None


def test_loader_register_instance() -> None:
    """register() should add a plugin instance."""
    loader = PluginLoader()
    plugin = TracingPlugin()
    loader.register(plugin)
    assert len(loader) == 1
    assert loader.get_plugin("tracing") is plugin


def test_loader_register_class() -> None:
    """register_class() should store the class for later instantiation."""
    loader = PluginLoader()
    loader.register_class(TracingPlugin)
    assert len(loader) == 0  # not yet instantiated
    loaded = loader.load()
    assert loaded == 1
    assert loader.get_plugin("tracing") is not None


def test_loader_priority_ordering() -> None:
    """Plugins should be sorted by priority after load()."""
    loader = PluginLoader()
    loader.register(CursorModifyingPlugin(dx=10, dy=20))  # priority 10
    loader.register(ActionSuppressingPlugin())  # priority 100 (default)
    loader.load()
    assert loader.active_plugins[0].name == "cursor_modifier"
    assert loader.active_plugins[1].name == "action_suppressor"


def test_loader_on_load_on_load() -> None:
    """on_load should be called during load()."""
    loader = PluginLoader()
    loader.register(TracingPlugin())
    loader.load()
    plugin = loader.get_plugin("tracing")
    assert "on_load" in plugin.calls


def test_loader_on_unload_on_unload() -> None:
    """on_unload should be called during unload()."""
    loader = PluginLoader()
    plugin = TracingPlugin()
    loader.register(plugin)
    loader.load()
    loader.unload()
    # unload() clears plugins, so save reference before unload
    assert "on_unload" in plugin.calls


def test_loader_unload_clears_plugins() -> None:
    """unload() should clear all plugins."""
    loader = PluginLoader()
    loader.register(TracingPlugin())
    loader.load()
    assert len(loader) == 1
    loader.unload()
    assert len(loader) == 0


def test_loader_repr() -> None:
    """__repr__ should show plugin names and versions."""
    loader = PluginLoader()
    loader.register(CursorModifyingPlugin())
    loader.register(ActionSuppressingPlugin())
    loader.load()
    repr_str = repr(loader)
    assert "cursor_modifier" in repr_str
    assert "action_suppressor" in repr_str


# ── Pipeline execution tests ─────────────────────────────────────────────────

def test_run_all_calls_all_hooks() -> None:
    """run_all() should call all implemented hooks on each plugin."""
    loader = PluginLoader()
    plugin = TracingPlugin()
    loader.register(plugin)
    loader.load()

    ctx = PluginContext(
        frame="test_frame",
        cursor=(100, 200),
        gesture="pointing_up",
        action="click",
        blink=True,
    )
    loader.run_all(ctx)

    # All hooks should have been called
    expected = [
        "on_frame", "on_gaze", "on_filter", "on_gesture",
        "on_fusion", "on_action", "on_overlay",
    ]
    for hook in expected:
        assert hook in plugin.calls, f"{hook} was not called"


def test_run_all_cursor_modification() -> None:
    """Plugins should be able to modify cursor position."""
    loader = PluginLoader()
    loader.register(CursorModifyingPlugin(dx=10, dy=20))
    loader.load()

    ctx = PluginContext(cursor=(100, 200))
    loader.run_all(ctx)

    assert ctx.cursor == (110, 220)


def test_run_all_action_suppression() -> None:
    """Plugins should be able to suppress actions."""
    loader = PluginLoader()
    loader.register(ActionSuppressingPlugin())
    loader.load()

    ctx = PluginContext(action="zoom_in")
    loader.run_all(ctx)

    assert ctx.action is None


def test_run_all_metadata_propagation() -> None:
    """Plugins should be able to write to context metadata."""
    loader = PluginLoader()
    loader.register(MetadataPlugin())
    loader.load()

    ctx = PluginContext(cursor=(50, 50))
    loader.run_all(ctx)

    assert ctx.metadata.get("plugin_ran") is True
    assert ctx.metadata.get("cursor_seen") == (50, 50)


def test_run_all_preserves_none_cursor() -> None:
    """Plugins should not modify None cursor if they don't override."""
    loader = PluginLoader()
    # Register a plugin that doesn't override on_gaze
    loader.register(FreeHandsPlugin())
    loader.load()

    ctx = PluginContext(cursor=None)
    loader.run_all(ctx)

    assert ctx.cursor is None


def test_run_all_exception_isolation() -> None:
    """A failing plugin hook should not crash other plugins."""
    class FailingPlugin(FreeHandsPlugin):
        name = "failing"
        version = "1.0.0"

        def on_action(self, action, ctx):
            raise ValueError("intentional failure")

    loader = PluginLoader()
    loader.register(FailingPlugin())
    loader.register(ActionSuppressingPlugin())  # should still run
    loader.load()

    ctx = PluginContext(action="zoom_in")
    # Should not raise — failures are caught and logged
    loader.run_all(ctx)

    # The action_suppressor should have run, suppressing zoom_in
    assert ctx.action is None


def test_run_all_empty_loader() -> None:
    """run_all() on an empty loader should return context unchanged."""
    loader = PluginLoader()
    loader.load()

    ctx = PluginContext(cursor=(10, 20), action="click")
    result = loader.run_all(ctx)

    assert result.cursor == (10, 20)
    assert result.action == "click"


# ── Plugin discovery tests ───────────────────────────────────────────────────

def test_discover_from_nonexistent_dir() -> None:
    """discover_from_directory() on a non-existent dir should return 0."""
    loader = PluginLoader()
    count = loader.discover_from_directory("/nonexistent/path")
    assert count == 0


def test_discover_from_empty_dir(tmp_path: Path) -> None:
    """discover_from_directory() on an empty dir should return 0."""
    loader = PluginLoader()
    count = loader.discover_from_directory(str(tmp_path))
    assert count == 0


def test_discover_from_dir_with_plugin_file(tmp_path: Path) -> None:
    """discover_from_directory() should find and register plugins."""
    # Create a plugin file
    plugin_file = tmp_path / "my_plugin.py"
    plugin_file.write_text("""
from freehands.plugins.base import FreeHandsPlugin

class MyDiscoveredPlugin(FreeHandsPlugin):
    name = "discovered_plugin"
    version = "2.0.0"
    description = "A discovered plugin"
""")

    loader = PluginLoader()
    count = loader.discover_from_directory(str(tmp_path))
    assert count == 1

    loaded = loader.load()
    assert loaded == 1
    assert loader.get_plugin("discovered_plugin") is not None


def test_discover_skips_private_files(tmp_path: Path) -> None:
    """discover_from_directory() should skip files starting with underscore."""
    private_file = tmp_path / "_private.py"
    private_file.write_text("""
from freehands.plugins.base import FreeHandsPlugin

class PrivatePlugin(FreeHandsPlugin):
    name = "private"
    version = "1.0.0"
""")

    loader = PluginLoader()
    count = loader.discover_from_directory(str(tmp_path))
    assert count == 0


# ── Integration: loader + context data flow ──────────────────────────────────

def test_full_pipeline_data_flow() -> None:
    """Test the full plugin pipeline: load → run → verify modifications."""
    loader = PluginLoader()

    # Register a cursor modifier (priority 10 — runs first)
    loader.register(CursorModifyingPlugin(dx=5, dy=5))

    # Register an action suppressor (priority 100 — runs second)
    loader.register(ActionSuppressingPlugin())

    # Register a metadata writer (priority 100 — runs third)
    loader.register(MetadataPlugin())

    loader.load()

    # Simulate a full frame
    ctx = PluginContext(
        frame="bgr_frame_data",
        cursor=(100, 100),
        gesture="pointing_up",
        action="zoom_in",  # Will be suppressed by ActionSuppressingPlugin
        blink=False,
        state="confirming",
    )

    result = loader.run_all(ctx)

    # Cursor should be modified by CursorModifyingPlugin
    assert result.cursor == (105, 105)

    # Action should be suppressed by ActionSuppressingPlugin
    assert result.action is None

    # Metadata should be written by MetadataPlugin
    # Note: cursor_seen reflects the cursor at the time MetadataPlugin ran,
    # which is after CursorModifyingPlugin already modified it to (105, 105).
    assert result.metadata.get("plugin_ran") is True
    assert result.metadata.get("cursor_seen") == (105, 105)

    # All plugins should have run
    assert len(result.metadata) > 0
