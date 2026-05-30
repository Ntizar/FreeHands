"""Plugin loader — discovers and manages FreeHands plugins.

Supports two discovery modes:
  1. Directory scan: loads all .py files from a plugins directory.
  2. Manual registration: register plugin instances directly.

Plugins are sorted by priority (lower = earlier execution).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Type

from .base import FreeHandsPlugin, PluginContext

logger = logging.getLogger(__name__)


class PluginLoader:
    """Discovers, loads and runs FreeHands plugins.

    Args:
        plugins_dir: Directory to scan for .py plugin files.
            If None, only manually registered plugins are used.

    Example:
        loader = PluginLoader("/etc/freehands/plugins")

        # Register a plugin class (will be instantiated on load)
        loader.register_class(MyPlugin)

        # Or register an instance directly
        loader.register(MyPlugin())

        # Run the full plugin pipeline for one frame
        ctx = PluginContext(frame=frame, cursor=cursor, gesture=gesture)
        loader.run_all(ctx)
    """

    def __init__(self, plugins_dir: str | Path | None = None) -> None:
        self._plugins: list[FreeHandsPlugin] = []
        self._plugin_classes: list[Type[FreeHandsPlugin]] = []
        self._plugins_dir = Path(plugins_dir) if plugins_dir else None

    # ── Registration ─────────────────────────────────────────────────────

    def register(self, plugin: FreeHandsPlugin) -> None:
        """Register a plugin instance directly."""
        self._plugins.append(plugin)
        logger.info("Registered plugin: %s v%s", plugin.name, plugin.version)

    def register_class(self, plugin_cls: Type[FreeHandsPlugin]) -> None:
        """Register a plugin class (instantiated on load)."""
        self._plugin_classes.append(plugin_cls)
        logger.info("Registered plugin class: %s", plugin_cls.name)

    # ── Discovery ────────────────────────────────────────────────────────

    def discover_from_directory(self, directory: str | Path | None = None) -> int:
        """Scan a directory for .py files and auto-discover plugins.

        Each .py file is imported and all FreeHandsPlugin subclasses
        found in the module are registered.

        Args:
            directory: Path to scan. Defaults to the loader's plugins_dir.

        Returns:
            Number of plugins discovered.
        """
        target = Path(directory) if directory else self._plugins_dir
        if target is None or not target.is_dir():
            return 0

        count = 0
        for py_file in sorted(target.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    py_file.stem, py_file
                )
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[py_file.stem] = module
                spec.loader.exec_module(module)

                # Find all FreeHandsPlugin subclasses in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, FreeHandsPlugin)
                        and attr is not FreeHandsPlugin
                    ):
                        self.register_class(attr)
                        count += 1
                        logger.info("Discovered plugin from %s: %s",
                                    py_file.name, attr.name)
            except Exception as exc:
                logger.warning("Failed to load plugin from %s: %s",
                               py_file, exc)

        return count

    # ── Load / Unload ────────────────────────────────────────────────────

    def load(self) -> int:
        """Instantiate all registered plugin classes and call on_load.

        Returns:
            Total number of active plugins.
        """
        # Instantiate classes
        for cls in self._plugin_classes:
            instance = cls()
            self._plugins.append(instance)
        self._plugin_classes.clear()

        # Call on_load for each plugin
        for plugin in self._plugins:
            try:
                plugin.on_load()
            except Exception as exc:
                logger.error("Plugin %s on_load failed: %s",
                             plugin.name, exc)

        # Sort by priority (lower = earlier)
        self._plugins.sort(key=lambda p: p.priority)
        return len(self._plugins)

    def unload(self) -> None:
        """Call on_unload for all plugins and clear registry."""
        for plugin in self._plugins:
            try:
                plugin.on_unload()
            except Exception as exc:
                logger.error("Plugin %s on_unload failed: %s",
                             plugin.name, exc)
        self._plugins.clear()

    # ── Pipeline execution ───────────────────────────────────────────────

    def run_all(self, ctx: PluginContext) -> PluginContext:
        """Run all plugin hooks for the current frame.

        Hooks are executed in priority order. Each hook receives the
        current state and can modify it.

        Args:
            ctx: Plugin context with current frame data.

        Returns:
            Updated context with any modifications applied by plugins.
        """
        for plugin in self._plugins:
            try:
                # on_frame: process camera frame
                if hasattr(plugin, 'on_frame') and plugin.on_frame.__func__ is not FreeHandsPlugin.on_frame:
                    ctx.frame = plugin.on_frame(ctx.frame, ctx)

                # on_gaze: process gaze cursor
                if hasattr(plugin, 'on_gaze') and plugin.on_gaze.__func__ is not FreeHandsPlugin.on_gaze:
                    ctx.cursor = plugin.on_gaze(ctx.cursor, ctx)

                # on_filter: process filtered cursor
                if hasattr(plugin, 'on_filter') and plugin.on_filter.__func__ is not FreeHandsPlugin.on_filter:
                    ctx.cursor = plugin.on_filter(ctx.cursor, ctx)

                # on_gesture: process gesture detection
                if hasattr(plugin, 'on_gesture') and plugin.on_gesture.__func__ is not FreeHandsPlugin.on_gesture:
                    gesture, confidence = plugin.on_gesture(ctx.gesture, 0.0, ctx)
                    ctx.gesture = gesture

                # on_fusion: process fusion output
                if hasattr(plugin, 'on_fusion') and plugin.on_fusion.__func__ is not FreeHandsPlugin.on_fusion:
                    ctx.cursor, ctx.gesture, ctx.action = plugin.on_fusion(
                        ctx.cursor, ctx.gesture, ctx.action, ctx
                    )

                # on_action: intercept before dispatch
                if hasattr(plugin, 'on_action') and plugin.on_action.__func__ is not FreeHandsPlugin.on_action:
                    ctx.action = plugin.on_action(ctx.action, ctx)

                # on_overlay: post-action overlay update
                if hasattr(plugin, 'on_overlay') and plugin.on_overlay.__func__ is not FreeHandsPlugin.on_overlay:
                    plugin.on_overlay(ctx)

            except Exception as exc:
                logger.error("Plugin %s hook failed: %s", plugin.name, exc)

        return ctx

    # ── Status ───────────────────────────────────────────────────────────

    @property
    def active_plugins(self) -> list[FreeHandsPlugin]:
        """List of currently loaded plugin instances."""
        return list(self._plugins)

    def get_plugin(self, name: str) -> FreeHandsPlugin | None:
        """Find a plugin by name."""
        for plugin in self._plugins:
            if plugin.name == name:
                return plugin
        return None

    def __len__(self) -> int:
        return len(self._plugins)

    def __repr__(self) -> str:
        names = [f"{p.name} v{p.version}" for p in self._plugins]
        return f"PluginLoader(plugins=[{', '.join(names)}])"
