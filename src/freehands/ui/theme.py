"""Ntizar design system — light-mode liquid glass with blue & orange accents.

Used by every PyQt6 widget so the brand stays consistent.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NtizarPalette:
    # Brand
    blue:        str = "#1E5BFF"
    blue_soft:   str = "#5B85FF"
    blue_dark:   str = "#0E3FCC"
    orange:      str = "#FF7A1A"
    orange_soft: str = "#FFA365"
    orange_dark: str = "#CC5C0E"

    # Liquid-glass (light mode) surfaces
    glass_base:    str = "rgba(255, 255, 255, 0.55)"   # main panel
    glass_strong:  str = "rgba(255, 255, 255, 0.78)"   # popovers / tooltips
    glass_border:  str = "rgba(255, 255, 255, 0.85)"
    glass_shadow:  str = "rgba(30,  91, 255, 0.18)"

    # Text on light glass
    text_primary:   str = "#0B1F4D"
    text_secondary: str = "#3D4E78"
    text_muted:     str = "#7B8AAE"

    # Background gradient (page)
    bg_grad_top:    str = "#F4F7FF"
    bg_grad_bottom: str = "#FFEFE3"

    # Semantic
    success: str = "#1FBF75"
    warning: str = "#FFB020"
    danger:  str = "#E54848"


PALETTE = NtizarPalette()


# ── Reusable QSS (Qt Style Sheets) ────────────────────────────────────────
GLOBAL_STYLESHEET = f"""
* {{
    font-family: "Segoe UI Variable", "Segoe UI", "Inter", system-ui, sans-serif;
    color: {PALETTE.text_primary};
}}

QWidget#NtizarPage {{
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 {PALETTE.bg_grad_top},
        stop:1 {PALETTE.bg_grad_bottom}
    );
}}

/* Liquid-glass card */
QFrame.NtizarCard {{
    background: {PALETTE.glass_base};
    border: 1px solid {PALETTE.glass_border};
    border-radius: 22px;
    padding: 20px;
}}

QLabel.NtizarTitle {{
    font-size: 24px;
    font-weight: 600;
    color: {PALETTE.text_primary};
}}

QLabel.NtizarSubtitle {{
    font-size: 14px;
    color: {PALETTE.text_secondary};
}}

QLabel.NtizarBrand {{
    font-size: 28px;
    font-weight: 700;
    color: {PALETTE.blue};
    letter-spacing: -0.5px;
}}

/* Primary action — Ntizar blue */
QPushButton.NtizarPrimary {{
    background: {PALETTE.blue};
    color: white;
    border: none;
    border-radius: 14px;
    padding: 10px 22px;
    font-weight: 600;
    font-size: 14px;
}}
QPushButton.NtizarPrimary:hover  {{ background: {PALETTE.blue_soft}; }}
QPushButton.NtizarPrimary:pressed{{ background: {PALETTE.blue_dark}; }}

/* Secondary action — Ntizar orange */
QPushButton.NtizarAccent {{
    background: {PALETTE.orange};
    color: white;
    border: none;
    border-radius: 14px;
    padding: 10px 22px;
    font-weight: 600;
    font-size: 14px;
}}
QPushButton.NtizarAccent:hover  {{ background: {PALETTE.orange_soft}; }}
QPushButton.NtizarAccent:pressed{{ background: {PALETTE.orange_dark}; }}

/* Ghost button on glass */
QPushButton.NtizarGhost {{
    background: {PALETTE.glass_strong};
    color: {PALETTE.text_primary};
    border: 1px solid {PALETTE.glass_border};
    border-radius: 14px;
    padding: 10px 22px;
    font-weight: 500;
}}
QPushButton.NtizarGhost:hover {{
    background: white;
}}
"""
