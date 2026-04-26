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

Sólo hay un launcher: **[FreeHands.bat](FreeHands.bat)**.

```
FreeHands.bat                       (menú interactivo)
FreeHands.bat run                   (arranca el sistema, usuario por defecto: Ntizar)
FreeHands.bat calibrate             (mirada + ronda de gestos)
FreeHands.bat run otro_usuario      (otro perfil)
```

Crea el `.venv`, instala dependencias la primera vez y, si el usuario no tiene
perfil, abre la calibración automáticamente.

### Manual (cualquier SO)

```bash
git clone https://github.com/Ntizar/FreeHands.git
cd FreeHands
python -m venv .venv
.\.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python -m freehands calibrate --user Ntizar
python -m freehands run --user Ntizar
```

### Demo web (sin instalar nada)

Abre **[ntizar.github.io/FreeHands](https://ntizar.github.io/FreeHands/)** → *Probar demo*.
Usa **WebGazer.js** y todo el procesamiento ocurre en tu navegador.

### Voz (Phase 3 local)

La voz se activa por defecto al ejecutar el sistema local. Para evitar falsos positivos,
los comandos de acción usan palabra de activación: `FreeHands` o `Ntizar`.

Ejemplos:

```
FreeHands clic
Ntizar doble clic
FreeHands botón derecho
Ntizar zoom más
Ntizar zoom menos
FreeHands scroll abajo
pausa
reanudar
```

Puedes desactivarla con:

```bash
python -m freehands run --user Ntizar --no-voice
```

#### VibeVoice

[microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) encaja como integración avanzada de voz:

- **ASR**: posible backend alternativo para transcripciones largas, diarización y contexto personalizado.
- **Realtime TTS**: candidato para feedback hablado del asistente (`pausado`, `calibración lista`, `acción cancelada`).

Por defecto FreeHands usa `faster_whisper` porque es más ligero para comandos cortos en tiempo real. El perfil ya deja preparado el punto de extensión:

```json
{
     "voice_asr_backend": "faster_whisper",
     "voice_tts_backend": "none",
     "voice_wake_words": ["freehands", "free hands", "ntizar"]
}
```

`vibevoice_asr` queda marcado como backend experimental hasta integrar pesos/modelos, requisitos GPU y una ruta de inferencia suficientemente rápida.

---

## 🎮 Minijuego de calibración

Inspirado en `aim_botz` de Counter-Strike. Cuatro fases:

1. **Calibración de mirada** — 9-13 puntos, 3 muestras cada uno → entrena un modelo de **ridge regression** personalizado.
2. **Ronda de gestos** — `thumb_up`, `thumb_down`, `pinch_close`, `fist_pause`; ajusta umbrales por perfil.
3. **Voz local** — comandos en español con wake word (`FreeHands` / `Ntizar`) usando faster-whisper.
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
- [x] **Phase 2** · gestos web/escritorio + perfiles + capa anti-FP
- [x] **Phase 3** · comandos de voz locales con faster-whisper
- [ ] **Phase 3.5** · VibeVoice opcional: ASR avanzado + feedback hablado con Realtime TTS
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
- [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) — ASR/TTS avanzado, posible backend opcional

---

## 📄 Licencia

MIT © Ntizar
