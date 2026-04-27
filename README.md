<div align="center">

# FreeHands

**Control your PC with your eyes, your hands and your voice.**
No headset. No special hardware. Just a webcam.

[![status](https://img.shields.io/badge/status-MVP%20desktop%20%2B%20web-orange)](https://github.com/Ntizar/FreeHands)
[![python](https://img.shields.io/badge/python-3.11%2B-1E5BFF)](https://www.python.org/)
[![platform](https://img.shields.io/badge/platform-Windows%20first-1E5BFF)](#quick-start)
[![license](https://img.shields.io/badge/license-MIT-1E5BFF)](#license)

[**Try the Duck test**](https://ntizar.github.io/FreeHands/) · [Quick start](#quick-start) · [How it feels](#how-it-feels) · [Architecture](#architecture)

</div>

---

## Why FreeHands

Most hands-free PC tools force you to choose between accessibility, gaming or productivity. FreeHands is one local pipeline that does all three from a normal webcam.

- **Look to aim, gesture to act.** Your gaze moves the real Windows pointer, your fingers click. No dwell timers in the way of normal use.
- **Always-on safety.** A right open palm held for two seconds is the only thing that can pause or resume the system. You always have a kill switch.
- **No cloud.** Everything runs on your machine. The GitHub Pages Duck test does not even open the camera.
- **Plays well with games.** Direct OS clicks, fine aim assist, and a Duck Hunt style test page that talks to the local app, not a second browser recognizer.
- **Designed in your face.** Light mode liquid glass UI in Ntizar blue and orange, with a control panel that minimizes to a small bottom bar.

## How It Feels

| Moment | What FreeHands does |
| --- | --- |
| Look at a button | Pointer follows your gaze, then snaps into fine aim if you stay near a target. |
| Raise your index finger | Instant left click at the pointer. |
| Raise your middle finger | Right click. |
| Raise index + middle | Double click. |
| Bring both hands together / apart | Zoom in / zoom out. |
| Raise your left open palm | Unassigned by default; map it from the control panel if you want Undo or another action. |
| Hold your right open palm 2s | Toggle ACTIVE / PAUSED with a visible progress ring. |

Side jitter from MediaPipe (left hand reported as right and vice versa) is folded back to the same click so a flicker never eats your action. Gesture-action mappings are inline editable from the control panel: pick a new action and it saves on the spot, with duplicates removed automatically.

## Quick Start

### Windows launcher

```bat
git clone https://github.com/Ntizar/FreeHands.git
cd FreeHands
FreeHands.bat
```

The launcher creates `.venv`, installs dependencies, runs calibration if there is no profile yet, and writes logs to `logs\FreeHands-last.log`.

Useful shortcuts:

```bat
FreeHands.bat run           :: run with the default profile
FreeHands.bat run Ntizar    :: run a specific user
FreeHands.bat calibrate     :: full calibration wizard
FreeHands.bat gaze          :: re-run only gaze calibration
FreeHands.bat gestures      :: re-run only gesture calibration
FreeHands.bat camera        :: pick a different webcam
FreeHands.bat doctor        :: environment + dependency check
FreeHands.bat repair        :: rebuild the venv
```

### Manual setup

```powershell
git clone https://github.com/Ntizar/FreeHands.git
cd FreeHands
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

python -m freehands calibrate --user Ntizar
python -m freehands run --user Ntizar
```

Profiles live under `%LOCALAPPDATA%\Ntizar\FreeHands\profiles`.

## The Control Panel

The panel sits in the top-right corner and gives you everything at a glance:

- **Live mirrored camera preview** with hand landmarks and detected side labels.
- **Gaze readout**: source (pupil / iris), confidence, cursor position.
- **Hand readout**: gesture id, side and current mapped action.
- **Pause hold meter** for the right open palm.
- **Last action** in big orange text.
- **Inline gesture-action editor**. Click any dropdown, pick a new action, done. Setting an action on one gesture clears it from any other gesture, so you never end up with two gestures fighting for the same click.
- **Swap L/R** if your camera reports handedness backwards. Saved in your profile.
- **Activate / Pause / Close** buttons for keyboard-free control.
- **Minimize button**: collapses the panel into a small status bar pinned to the bottom-right of your screen so it does not block your work.

## The Duck Test

The repo ships a public Duck test that **does not** use the browser camera:

[ntizar.github.io/FreeHands](https://ntizar.github.io/FreeHands/duck-hunt.html?user=ntizar)

Workflow:

1. Run `FreeHands.bat run Ntizar` locally.
2. Activate FreeHands.
3. Open the Pages link.
4. Look at a duck and raise your index finger.

If you score, the local pipeline is working end to end: gaze, gesture, click, OS pointer event reaching the page. If you do not score, you know the issue is calibration or webcam, not the browser.

## Voice (Optional)

Local speech via `faster-whisper`. Action commands need a wake word (`FreeHands` or `Ntizar`); pause and resume are direct for safety.

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

Spanish phrases such as `clic`, `clic derecho`, `pausa`, `reanudar` are also recognized.

## Calibration Tips

| Symptom | Fix |
| --- | --- |
| Gaze pulls to corners | Re-run `FreeHands.bat gaze` and stare at each point until the ring confirms. |
| Wrong webcam opens | `FreeHands.bat camera` and pick the right one. |
| Left and right hand swapped | Press `Swap L/R` in the panel. |
| Eyes not detected | Front lighting, no backlight, clean lens, face centered. |
| Clicks land off target | Recalibrate gaze in the same lighting you will use. |
| Click feels missed when changing gesture | Already handled: stabilizer rearms in one frame on release, side-specific bindings fall back to the generic click. |

## Architecture

```text
Camera frame
  ├─ GazeTracker   → personal GazeRegressor → screen x, y
  ├─ HandTracker   → GestureStabilizer      → confirmed gesture
  └─ VoiceListener → command parser         → optional action
                       │
                       ▼
              MultimodalFusion
        IDLE → ACTIVE → CONFIRMING → COOLDOWN
                       │
                       ▼
        ActionDispatcher (PyAutoGUI)
                       │
                       ▼
   Real Windows pointer, clicks, zoom, scroll, undo
```

| Path | Purpose |
| --- | --- |
| [src/freehands/main.py](src/freehands/main.py) | Runtime loop, pointer movement, overlay, action dispatch. |
| [src/freehands/gaze](src/freehands/gaze) | Eye/face features and personal regression model. |
| [src/freehands/gestures](src/freehands/gestures) | MediaPipe hands, side-aware ids, multi-frame stabilizer. |
| [src/freehands/fusion](src/freehands/fusion) | State machine, side fallback bindings, safety logic. |
| [src/freehands/profiles](src/freehands/profiles) | Pydantic profile, dedupe + migration of bindings. |
| [src/freehands/ui](src/freehands/ui) | PyQt6 overlay, control panel and calibration. |
| [docs](docs) | GitHub Pages Duck test. |

## Privacy

- Camera frames are processed locally.
- Profiles, calibration data and gesture bindings stay on disk.
- The Pages site does not request the webcam.
- Optional model downloads: MediaPipe (hand/face), faster-whisper (voice).

## Development

```powershell
python -m pytest -q
python -m ruff check src tests
node --check docs/assets/demo.js
node --check docs/assets/duck-hunt.js
node --check docs/assets/gestures-v3.js
```

Run from source:

```powershell
python -m freehands run --user Ntizar
```

## Roadmap

- One-click installer for non-developer Windows users.
- Game presets (FPS, point-and-click, browser).
- Spoken feedback for state transitions.
- Better head-pose compensation for off-axis cameras.
- Cross-platform (macOS / Linux) gesture and pointer paths.

## Design System

Light-mode liquid glass with Ntizar blue `#1E5BFF` and Ntizar orange `#FF7A1A`. Soft translucent surfaces, compact panels and high-contrast status text. Desktop theme: [src/freehands/ui/theme.py](src/freehands/ui/theme.py). Web theme: [docs/assets/style.css](docs/assets/style.css).

## Credits

- [MediaPipe Tasks Vision](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer)
- [WebGazer.js](https://webgazer.cs.brown.edu/) (browser diagnostics)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [PyAutoGUI](https://pyautogui.readthedocs.io/)

## License

MIT © Ntizar
