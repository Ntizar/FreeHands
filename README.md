# FreeHands

> Hands-free PC control with gaze, hand gestures and voice. FreeHands turns your webcam into a local control loop: look where you want to act, confirm with a gesture, and keep a fist available as the always-on pause switch.

![status](https://img.shields.io/badge/status-MVP%20desktop%20%2B%20web-orange)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![platform](https://img.shields.io/badge/platform-Windows%20first-blue)
![license](https://img.shields.io/badge/license-MIT-blue)

Browser demo: [ntizar.github.io/FreeHands](https://ntizar.github.io/FreeHands/)

FreeHands is built for accessibility experiments, productivity workflows and playful hands-free interaction. The desktop app can move the real Windows pointer from gaze, then execute actions through confirmed gestures. The web demo lets anyone calibrate gaze in the browser and try the Duck test without installing anything.

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

When the local app is active, gaze moves the real Windows pointer at a throttled rate. Gestures confirm actions:

| Gesture | Default action |
| --- | --- |
| Index up | Click |
| Middle finger up | Right click |
| Index + middle | Double click |
| Hands together | Zoom in |
| Hands apart | Zoom out |
| Closed fist | Toggle active / paused |

The small control panel shows the current state, gaze source, confidence, cursor position and detected gesture. The transparent overlay shows the gaze cursor and dwell ring. Move the mouse to a screen corner to trigger the PyAutoGUI failsafe if you need to abort quickly.

## Browser Demo And Duck Test

Open [ntizar.github.io/FreeHands](https://ntizar.github.io/FreeHands/) and choose the browser demo.

The browser flow:

1. Allow camera access.
2. Click `Switch camera` if the preview is frozen or the wrong camera is selected.
3. Look at each orange target and click it.
4. Calibration is saved in browser `localStorage`.
5. Open the Duck test and shoot by looking at a duck, then raising your index finger.

The web version uses WebGazer.js for gaze and MediaPipe Tasks Vision for gestures. It runs on HTTPS and does not upload video frames.

## Playing Games

FreeHands is best for games or interactive tests that accept the normal OS pointer and clicks. Start with browser games, aim trainers, puzzle games, or windowed PC games before trying anything fast.

Recommended setup:

1. Recalibrate gaze in the same lighting you will use for the game.
2. Use the camera selector if the wrong webcam is active.
3. Run the game in windowed or borderless mode.
4. Keep the FreeHands panel visible at first.
5. Use the closed fist to pause before menus, alt-tab, or risky clicks.

## Calibration Tips

Good calibration matters more than any single model setting.

| Problem | Fix |
| --- | --- |
| Gaze pulls to corners | Re-run `FreeHands.bat gaze`; look at each point before confirming. |
| Camera freezes or wrong camera opens | Use `FreeHands.bat camera` locally, or `Switch camera` in the browser. |
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
| [docs](docs) | GitHub Pages demo, browser calibration and Duck test. |

## Privacy

The desktop app processes camera frames locally. The browser demo processes frames in the browser. Profiles and calibration data stay on the user's machine unless they intentionally share files.

External downloads can happen for dependencies and ML models:

| Dependency | Why |
| --- | --- |
| MediaPipe models | Hand and face/gaze tracking assets. |
| WebGazer.js | Browser gaze demo. |
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