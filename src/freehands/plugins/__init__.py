"""Plugin system for FreeHands — extensible pipeline architecture.

Plugins can hook into each stage of the FreeHands pipeline:
  Camera → Gaze → Filter → Gesture → Fusion → Action → Overlay

Usage:
    from freehands.plugins import PluginLoader, FreeHandsPlugin

    class MyPlugin(FreeHandsPlugin):
        name = \"my_plugin\"
        version = \"1.0.0\"

        def on_frame(self, frame, context):
            # Process every camera frame before trackers run
            return frame  # return modified frame or None to skip

        def on_gaze(self, cursor, context):
            # Modify gaze cursor before it reaches fusion
            return cursor

        def on_action(self, action, context):
            # Intercept actions before they are dispatched
            pass

    loader = PluginLoader(\"/path/to/plugins\")
    loader.register(MyPlugin())
"""

from .base import FreeHandsPlugin, PluginContext, PluginPhase
from .loader import PluginLoader

__all__ = [
    "FreeHandsPlugin",
    "PluginContext",
    "PluginPhase",
    "PluginLoader",
]
