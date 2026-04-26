# FreeHands

> Sistema multimodal hands-free de control de pantalla por **mirada + gestos + voz**, configurable mediante un minijuego tipo *aim trainer* y diseñado con prevención de falsos positivos como prioridad nº1.

![status](https://img.shields.io/badge/status-MVP%20Phase%201-orange)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-blue)

🌐 **Demo en navegador:** [ntizar.github.io/FreeHands](https://ntizar.github.io/FreeHands/)

---

## ✨ Visión

FreeHands permite controlar el ordenador sin teclado ni ratón combinando:

| Modalidad | Para qué |
|-----------|----------|
| 👁️ **Mirada (gaze)** | Posición aproximada del cursor |
| ✋ **Gestos de mano** | Confirmación, zoom, scroll |
| 👅 **Gestos faciales / lengua** | Comandos rápidos secundarios |
| 🎙️ **Voz (Whisper local)** | Dictado y comandos largos |

El usuario calibra el sistema **una sola vez** jugando un minijuego de puntería, y a partir de ahí cada perfil se guarda en `~/.freehands/profiles/{user}.json`.

> **Principio rector:** prevención de falsos positivos > suavidad de interacción.
> Es preferible que el sistema se sienta "rígido" antes de activar acciones no pedidas.

---

## 🧱 Arquitectura

```
                 CAPTURA (frame único @30fps)
                            │
        ┌───────────┬───────┴───────┬───────────┐
        ▼           ▼               ▼           ▼
   Gaze Tracker  Hand Tracker  Face/Tongue   Voice
   (ridge reg.) (MediaPipe)   (FaceMesh)   (Whisper)
        │           │               │           │
        └───────────┴───────┬───────┴───────────┘
                            ▼
              ┌────────────────────────────┐
              │ MULTIMODAL FUSION LAYER    │
              │ Máq. de estados + buffers  │
              │ idle → active → confirming │
              │      → cooldown            │
              └────────────┬───────────────┘
                           ▼
              ┌────────────────────────────┐
              │ ACTION DISPATCHER (pyautogui) │
              └────────────┬───────────────┘
                           ▼
              ┌────────────────────────────┐
              │ UI OVERLAY (PyQt6)         │
              │ Cursor · Dwell · Magnify   │
              └────────────────────────────┘
```

Stack: `Python 3.11+` · `OpenCV` · `MediaPipe` · `faster-whisper` · `PyQt6` · `pyautogui` · `pynput`.

---

## 🚀 Quick start

### Windows · doble-click (recomendado)

| Archivo | Qué hace |
|---------|----------|
| [calibrate.bat](calibrate.bat) | Crea el venv si no existe, instala deps y lanza la calibración del usuario `luis` |
| [run.bat](run.bat) | Arranca el sistema con el perfil `luis` |
| [freehands.bat](freehands.bat) `[calibrate\|run\|doctor]` `[usuario]` | Launcher genérico |

### Manual (cualquier SO)

```bash
git clone https://github.com/Ntizar/FreeHands.git
cd FreeHands
python -m venv .venv
.\.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python -m freehands calibrate --user luis
python -m freehands run --user luis
```

### Demo web (sin instalar nada)

Abre **[ntizar.github.io/FreeHands](https://ntizar.github.io/FreeHands/)** → *Probar demo*.
Usa **WebGazer.js** y todo el procesamiento ocurre en tu navegador.

---

## 🎮 Minijuego de calibración

Inspirado en `aim_botz` de Counter-Strike. Cuatro fases:

1. **Calibración de mirada** — 9-13 puntos, 3 muestras cada uno → entrena un modelo de **ridge regression** personalizado.
2. **Tiro al plato con gestos** — rondas separadas para `click`, `cancel`, `zoom_in`, `zoom_out`, `tongue_out`. Mide tasa de éxito y latencia.
3. **Combos** — ajusta el `dwell_time_ms` óptimo del usuario (rango 250-1000 ms).
4. **Validación final** — tareas reales sin ratón. Si el éxito < 80 %, sugiere recalibrar.

---

## 🛡️ Capa anti-falsos-positivos

| Capa | Mecanismo |
|------|-----------|
| 1 | Máquina de estados explícita: `IDLE → ACTIVE → CONFIRMING → COOLDOWN` |
| 2 | Estabilidad temporal: gesto válido sólo tras N frames consecutivos coincidentes |
| 3 | Confirmación multimodal obligatoria (mirada + gesto) |
| 4 | Zonas espaciales válidas (mano por encima del torso) |
| 5 | Confianza dinámica (cooldown largo si hay gestos contradictorios) |
| 6 | Gesto de pausa **siempre** disponible (puño 1 s) |
| 7 | Feedback visual obligatorio antes de cada acción |

---

## 🗺️ Roadmap

- [x] **Phase 1 — MVP** · gaze + 2 gestos (👍/👎) + cursor con dwell
- [ ] **Phase 2** · pinch zoom + minijuego completo + perfiles + capa anti-FP
- [ ] **Phase 3** · dictado por voz con Whisper local
- [ ] **Phase 4** · detección de lengua + bindings configurables
- [ ] **Phase 5** · empaquetado, GPU opcional, vídeo demo

---

## 🎨 Sistema de diseño Ntizar

Colores: **azul Ntizar `#1E5BFF`** + **naranja Ntizar `#FF7A1A`** sobre superficies *liquid glass* en *light mode*. Detalles en [src/freehands/ui/theme.py](src/freehands/ui/theme.py).

---

## 📚 Referencias

- [antoinelame/GazeTracking](https://github.com/antoinelame/GazeTracking) — detección de pupila
- [WebGazer.js](https://webgazer.cs.brown.edu/) — autocalibración por interacción
- [alighazi288/Multimodal-System-Control](https://github.com/alighazi288/Multimodal-System-Control) — arquitectura concurrente
- [Viral-Doshi/Gesture-Controlled-Virtual-Mouse](https://github.com/Viral-Doshi/Gesture-Controlled-Virtual-Mouse) — pinch zoom
- [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) — pipeline de gestos custom

---

## 📄 Licencia

MIT © Ntizar
