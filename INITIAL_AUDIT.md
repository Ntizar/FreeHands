# FreeHands — Auditoría Inicial

**Fecha:** 2026-05-29
**Versión del repo:** `b435354` (commit #20 desde inicio)
**Estado:** MVP funcional, 1 mejora aplicada vía pipeline 9009

---

## ¿Qué es FreeHands?

Aplicación de escritorio que permite **controlar el PC sin usar las manos**, usando únicamente una webcam normal. Combina tres canales de entrada:

- **Mirada (gaze)** → mueve el cursor del sistema real
- **Gestos de mano** → clic, clic derecho, doble clic, zoom
- **Voz** → comandos por wake word (faster-whisper local)

**Sin hardware especial. Sin nube. Todo local.**

---

## Para qué sirve

| Uso | Ejemplo |
|---|---|
| **Accesibilidad** | Personas con movilidad reducida que no pueden usar ratón |
| **Productividad** | Control manos libres (cocina, taller, reparaciones) |
| **Gaming** | Compatible con juegos que aceptan clicks del SO |
| **Experimentación** | Demo de pipeline multimodal en tiempo real |

---

## Arquitectura

```
Camera Frame
  ├─ GazeTracker   → GazeRegressor (Ridge propio) → screen x, y
  ├─ HandTracker   → GestureStabilizer → gesto confirmado
  └─ VoiceListener → faster-whisper → comando
       │
       ▼
  MultimodalFusion (State Machine)
  IDLE → ACTIVE → CONFIRMING → COOLDOWN
       │
       ▼
  ActionDispatcher (PyAutoGUI)
  → click, right_click, double_click, zoom_in, zoom_out,
    scroll_up, scroll_down, toggle_pause, undo
```

**Stack técnico:**

| Capa | Tecnología |
|---|---|
| Lenguaje | Python 3.11+ |
| UI Desktop | PyQt6 (overlay + control panel) |
| Visión | MediaPipe Tasks Vision (hands + face mesh) |
| Gaze | Regresión Ridge propia (scikit-learn) |
| Voz | faster-whisper (local, GPU opcional) |
| Acciones SO | PyAutoGUI + pynput |
| Perfiles | Pydantic v2 |
| Web | GitHub Pages (Duck test, vanilla JS) |

---

## Estado actual del código

| Métrica | Valor |
|---|---|
| **Archivos Python** | 29 módulos |
| **Líneas de código** | ~3,672 (src) + ~1,000 (tests) |
| **Tests pytest** | 11 archivos, ~1,000 líneas |
| **Dependencias** | 11 (requirements.txt) |
| **Commits** | 20 desde inicio |
| **Mejoras 9009 aplicadas** | 1 de 20 (scroll por palma abierta) |

### Estructura de módulos

```
src/freehands/
├── main.py              ← Orquestador runtime (430 líneas)
├── config.py            ← Constantes globales
├── doctor.py            ← Diagnóstico de entorno
├── capture/camera.py    ← Wrapper webcam OpenCV
├── gaze/
│   ├── tracker.py       ← GazeTracker (MediaPipe face mesh)
│   └── calibration.py   ← GazeRegressor + calibración 5 puntos
├── gestures/
│   ├── hand_tracker.py  ← HandTracker (MediaPipe hands)
│   ├── stabilizer.py    ← GestureStabilizer (multiframe)
│   └── face_tracker.py  ← FaceTracker (huérfano, no se usa)
├── fusion/
│   ├── fusion.py        ← MultimodalFusion
│   └── state_machine.py ← StateMachine (IDLE/ACTIVE/CONFIRMING/COOLDOWN)
├── voice/
│   └── whisper_listener.py ← faster-whisper listener
├── actions/
│   └── dispatcher.py    ← PyAutoGUI actions
├── profiles/
│   ├── __init__.py      ← Profile Pydantic model
│   └── store.py         ← Carga/guarda perfiles en disco
└── ui/
    ├── overlay.py       ← ControlPanel + GazeOverlay (PyQt6)
    ├── calibration_game.py ← Calibración visual interactiva
    ├── camera_selector.py  ← Selección de cámara
    └── theme.py         ← Colores Ntizar
```

---

## Lo que funciona HOY

- ✅ Cursor sigue la mirada con precisión personalizable
- ✅ Fine aim (snapping a elementos cercanos tras 300ms)
- ✅ Clic con índice, clic derecho con dedo medio, doble clic ambos
- ✅ Zoom con ambas manos juntas/separadas
- ✅ Pausa con palma abierta derecha (2s)
- ✅ Scroll con mouse (no con gesto aún — eso era la mejora #1)
- ✅ Comandos de voz con wake word
- ✅ Panel de control con preview de cámara, readouts, editor inline de gestos
- ✅ Panel minimizable a barra de estado
- ✅ Calibración visual de 5 puntos con regresión Ridge
- ✅ Duck Hunt test en GitHub Pages (validación end-to-end sin cámara en navegador)
- ✅ Perfiles persistentes por usuario
- ✅ Scroll por gesto con palma abierta (mejora #1 aplicada)

---

## Lo que NO funciona aún

- ❌ Sin dead zones en bordes de pantalla (el cursor va a coordenadas extremas)
- ❌ Sin feedback auditivo de confirmación
- ❌ Sin comandos de sistema por voz (show desktop, volumen, screenshot)
- ❌ Sin clic por guiño (blink detection)
- ❌ Sin configuración de gestos vía JSON externo
- ❌ Vosk no implementado como alternativa ligera a faster-whisper
- ❌ Priorización dinámica de canales (gesto vs voz)
- ❌ Snap-to-grid UI mejorado
- ❌ Overlay transparente full-screen (el panel actual bloquea contenido)
- ❌ Doble parpadeo = clic, prolongado = drag
- ❌ Fusión multimodal AND (voz + mirada simultánea)
- ❌ Filtro Kalman (actualmente usa filtro exponencial simple)
- ❌ Sistema de plugins extensible
- ❌ 6DoF head pose para desplazamiento grueso
- ❌ Expresiones faciales (sonrisa, ceño, sorpresa)
- ❌ Teclado virtual con selección por mirada
- ❌ Modo dictado multimodal

---

## Deuda técnica observada

| Problema | Impacto |
|---|---|
| `face_tracker.py` existe pero no se usa en `main.py` | Código huérfano, confusión para futuros desarrolladores |
| Scroll por gesto de mano no implementado | Limita interacciones naturales |
| Sin dead zones en bordes | El cursor puede ir a coordenadas inválidas |
| Sin feedback auditivo | El usuario no sabe cuándo se confirma una acción |
| Sin configuración JSON externa | Los gestos se editan solo en UI, no hay forma programática |
| Voz solo con faster-whisper | Requiere GPU para rendimiento, no hay alternativa ligera |

---

## Plan de mejora (pipeline 9009)

**20 mejoras planificadas**, priorizadas por dificultad:

- **Baja (7):** scroll palma, dead zones, feedback auditivo, comandos sistema, clic guiño, config JSON, Vosk
- **Media (7):** priorización canales, snap-to-grid, menú OSD radial, calibración 9 puntos, overlay transparente, doble parpadeo, fusión AND
- **Alta (6):** filtro Kalman, sistema plugins, 6DoF head pose, expresiones faciales, teclado virtual, modo dictado

**Progreso actual:** 1/20 completadas (5%)

---

## Métricas de referencia (baseline)

Estas métricas se usarán para comparar dentro de 1-3 meses:

### Código
- **Líneas de código (src):** ~3,672
- **Líneas de tests:** ~1,000
- **Ratio test/src:** 27%
- **Archivos Python:** 29
- **Commits totales:** 20

### Funcionalidad
- **Gestos implementados:** 5 (index click, middle right-click, both double-click, hands zoom, palm pause)
- **Comandos de voz:** 6 (click, right click, double click, zoom in, zoom out, scroll down) + pause/resume directos
- **Canales de entrada:** 3 (gaze, gesture, voice)
- **Acciones SO soportadas:** 8

### Calidad
- **Tests pytest:** 11 archivos
- **Lint:** ruff configurado (line-length=100)
- **CI/CD:** Ninguno (no hay GitHub Actions)
- **Cobertura de tests:** No medida (no hay tool de coverage configurado)

### Rendimiento
- **FPS objetivo:** TARGET_FPS en config.py (valor exacto en código)
- **Pipeline de fusión:** State machine síncrono
- **Modelo gaze:** Ridge regression personal (entrenado en calibración)

---

## Posibles evoluciones futuras

1. **Producto SaaS** — Versión comercial con installer, presets de juegos, soporte multi-usuario
2. **Accesibilidad certificada** — Cumplir estándares WCAG, certificaciones de accesibilidad
3. **API REST** — Exponer FreeHands como servicio para integrar con otros sistemas
4. **Extensión de plugins** — Comunidad que cree nuevos gestos, comandos, modos
5. **Versión móvil** — Control de tablet/teléfono por cámara frontal
6. **Integración con IA** — Usar LLM local para interpretar gestos ambiguos o contexto

---

*Esta auditoría es el punto de partida. Se re-evaluará dentro de 1-3 meses para medir el impacto del pipeline 9009 de mejora continua.*
