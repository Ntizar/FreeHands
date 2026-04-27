# FreeHands

> Hands-free PC control with gaze, hand gestures and voice. FreeHands turns your webcam into a local control loop: look where you want to act, confirm with a gesture, and keep a fully open palm available as the always-on pause switch.

![status](https://img.shields.io/badge/status-MVP%20desktop%20%2B%20web-orange)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![platform](https://img.shields.io/badge/platform-Windows%20first-blue)
![license](https://img.shields.io/badge/license-MIT-blue)

Local Duck test: [ntizar.github.io/FreeHands](https://ntizar.github.io/FreeHands/)

FreeHands is built for accessibility experiments, productivity workflows and playful hands-free interaction. The desktop app can move the real Windows pointer from gaze, then execute actions through confirmed gestures. GitHub Pages is only the Duck test, so success there validates the local desktop app instead of a separate browser recognizer.

## What It Does

FreeHands combines three signals:

| Signal | Role |
| --- | --- |
| Gaze | Estimates the screen point you are looking at. The current model mixes eyes with nose/head cues for more stable desktop aiming. |
| Hand gestures | Confirms clicks, right clicks, double clicks, zoom and pause actions. |
| Voice | Optional local commands through faster-whisper, with English and Spanish phrases. |

The central design rule is simple: visible feedback first, action second. FreeHands is intentionally conservative because a missed action is easier to recover from than an accidental click.

## Quick Start

### Windows Launcher

Double-click [FreeHands.bat](FreeHands.bat), or run one of these commands from the repo:

```bat
FreeHands.bat
FreeHands.bat run
FreeHands.bat calibrate
FreeHands.bat gaze
FreeHands.bat gestures
FreeHands.bat camera
FreeHands.bat doctor
FreeHands.bat repair
FreeHands.bat run MyProfile
```

The launcher creates `.venv`, installs dependencies, opens calibration when a profile is missing, and writes logs to `logs/FreeHands-last.log`.

### Manual Setup

```powershell
git clone https://github.com/Ntizar/FreeHands.git
cd FreeHands
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

python -m freehands calibrate --user Ntizar
python -m freehands run --user Ntizar
```

Useful maintenance commands:

```powershell
python -m freehands camera --user Ntizar
python -m freehands calibrate-gaze --user Ntizar
python -m freehands calibrate-gestures --user Ntizar
python -m freehands doctor
python -m freehands repair
python -m freehands run --user Ntizar --no-voice
```

Profiles are stored locally under `%LOCALAPPDATA%\Ntizar\FreeHands\profiles` on Windows.

## Desktop Control

When the local app is active, gaze moves the real Windows pointer at a throttled rate. If your gaze stays inside the same screen zone for about one second, FreeHands switches into fine aim and damps the pointer around that target to make precise clicks easier. Click gestures are direct and fast: raising the index finger emits a normal OS click without waiting for dwell or a long cooldown. Side jitter is folded back to the generic click gesture, so a momentary left/right hand flip should not make clicks disappear. Safety gestures stay slower.

The control panel stays in the top-right corner and shows the live mirrored camera preview, detected hand landmarks, detected hand side, last fired action and a pause-hold progress bar. It can be minimized with the `-` button. If your camera reports left and right backwards, press `Swap L/R`; the setting is saved in your local profile. The same panel includes a visual gesture-action editor with icons: press `Edit actions`, choose an action or `No action` for each gesture, then press `Save`.

Index, middle and two-finger gestures use one-frame detection and one-frame release rearming. In practice, lowering and raising the finger again should produce repeated OS clicks with timing close to a normal mouse tap.

| Gesture | Default action |
| --- | --- |
| Left or right index up | Click |
| Left or right middle finger up | Right click |
| Left or right index + middle | Double click |
| Hands together | Zoom in |
| Hands apart | Zoom out |
| Left open palm | Undo / go back one action |
| Right open palm held about 2 seconds | Toggle active / paused |

The small control panel shows the current state, gaze source, confidence, cursor position and detected gesture. The transparent overlay shows the gaze cursor and dwell ring. Move the mouse to a screen corner to trigger the PyAutoGUI failsafe if you need to abort quickly.

## Pages And Local Duck Test

Open [ntizar.github.io/FreeHands](https://ntizar.github.io/FreeHands/). The Pages root redirects directly to the Duck test and keeps the selected user in the URL or browser storage.

The intended flow:

1. Run `FreeHands.bat run Ntizar` locally.
2. Complete calibration if the profile asks for it.
3. Activate FreeHands Desktop.
4. Open the Pages Duck test.
5. Look at a duck and use the index-click gesture.

The Duck test deliberately does not use browser gaze, browser gestures or the camera. It is a normal pointer-and-click web game: start FreeHands Desktop locally, activate it, move the Windows pointer with gaze, then use the index-click gesture to shoot. It fires on `pointerdown` for faster feedback, accepts rapid repeated shots, and adds a small aim assist when the pointer stays near a duck. That way the score still measures the local system, not a second browser-only recognizer.

## Playing Games

FreeHands is best for games or interactive tests that accept the normal OS pointer and clicks. Start with browser games, aim trainers, puzzle games, or windowed PC games before trying anything fast.

Recommended setup:

1. Recalibrate gaze in the same lighting you will use for the game.
2. Use the camera selector if the wrong webcam is active.
3. Run the game in windowed or borderless mode.
4. Keep the FreeHands panel visible at first.
5. Hold the right open palm for about 2 seconds to pause before menus, alt-tab, or risky clicks.

## Calibration Tips

Good calibration matters more than any single model setting.

| Problem | Fix |
| --- | --- |
| Gaze pulls to corners | Re-run `FreeHands.bat gaze`; look at each point before confirming. |
| Camera freezes or wrong camera opens | Use `FreeHands.bat camera` locally, or `Switch camera` in the browser. |
| Left and right hand are reversed | Press `Swap L/R` in the top-right FreeHands panel and check the live camera preview. |
| Eyes are not detected | Add frontal light, reduce backlight, clean the webcam frame, and keep your face centered. |
| Gestures feel unreliable | Re-run `FreeHands.bat gestures` and hold each gesture until the ring completes. |
| Clicks happen at the wrong place | Re-run gaze calibration and check the panel cursor readout before confirming actions. |
| Browser says camera is unavailable | Use HTTPS or localhost, allow site camera permission, and close other apps using the webcam. |

During local gaze calibration, press `C` to rotate cameras. The first points are screen corners, then the calibration moves through denser points to improve aiming.

## Voice Commands

Voice is optional and local. Action commands normally require a wake word such as `FreeHands` or `Ntizar`; pause and resume can be spoken directly for safety.

Examples:

```text
FreeHands click
Ntizar right click
FreeHands double click
Ntizar zoom in
Ntizar zoom out
FreeHands scroll down
pause
resume
```

Spanish phrases are still recognized for compatibility, including `clic`, `clic derecho`, `zoom mas`, `pausa` and `reanudar`.

## Architecture

```text
Camera frames
  |
  +-- GazeTracker   -> personal GazeRegressor -> screen x/y
  +-- HandTracker   -> GestureStabilizer      -> confirmed gesture
  +-- VoiceListener -> command parser         -> optional action
  |
  v
MultimodalFusion
  IDLE -> ACTIVE -> CONFIRMING -> COOLDOWN
  |
  v
ActionDispatcher (PyAutoGUI)
  |
  v
Real pointer, clicks, zoom, scroll, escape
```

Main components:

| Path | Purpose |
| --- | --- |
| [src/freehands/main.py](src/freehands/main.py) | Runtime orchestration, pointer movement, overlay updates and action dispatch. |
| [src/freehands/gaze](src/freehands/gaze) | Feature extraction, personal regression model and calibration helpers. |
| [src/freehands/gestures](src/freehands/gestures) | MediaPipe hand tracking and gesture stabilization. |
| [src/freehands/fusion](src/freehands/fusion) | State machine and multimodal safety logic. |
| [src/freehands/ui](src/freehands/ui) | PyQt6 calibration, camera selector and always-on overlay. |
| [docs](docs) | GitHub Pages Duck test and optional browser diagnostic files. |

## Privacy

The desktop app processes camera frames locally. The Pages Duck test does not use the camera. Profiles and calibration data stay on the user's machine unless they intentionally share files.

External downloads can happen for dependencies and ML models:

| Dependency | Why |
| --- | --- |
| MediaPipe models | Hand and face/gaze tracking assets. |
| WebGazer.js | Optional browser diagnostic page. |
| faster-whisper model | Optional local voice commands. |

## Development

```powershell
python -m pytest -q
python -m ruff check src tests
node --check docs/assets/demo.js
node --check docs/assets/duck-hunt.js
node --check docs/assets/gestures-v3.js
```

Run the local app from source:

```powershell
python -m freehands run --user Ntizar
```

## Roadmap

- Better game presets for click-heavy and pointer-heavy workflows.
- Configurable gesture bindings from the desktop UI.
- Optional spoken feedback for state changes and calibration results.
- Packaging for non-developer Windows installs.
- More robust browser game tests and mobile viewport checks.
- Optional advanced voice backend experiments.

## Design System

FreeHands uses the Ntizar light-mode liquid-glass look: Ntizar blue `#1E5BFF`, Ntizar orange `#FF7A1A`, soft translucent surfaces and compact control panels. Desktop theme details live in [src/freehands/ui/theme.py](src/freehands/ui/theme.py); web styling lives in [docs/assets/style.css](docs/assets/style.css).

## References

- [WebGazer.js](https://webgazer.cs.brown.edu/) for browser gaze calibration.
- [MediaPipe Tasks Vision](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer) for gesture recognition.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for optional local speech recognition.
- [PyAutoGUI](https://pyautogui.readthedocs.io/) for desktop pointer and action dispatch.

## License

MIT (c) Ntizar