/* ──────────────────────────────────────────────────────────────────────
   FreeHands · web demo · Phase 2 gesture round
   Uses MediaPipe Tasks Vision GestureRecognizer (built-in model).
   Categories: None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down,
               Thumb_Up, Victory, ILoveYou.
   ────────────────────────────────────────────────────────────────────── */
import {
  GestureRecognizer,
  FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const $ = (id) => document.getElementById(id);

/** Sequence of gestures we want the user to perform. */
const ROUND = [
  { id: 'Thumb_Up',    label: 'Pulgar arriba 👍',  binding: 'click' },
  { id: 'Thumb_Down',  label: 'Pulgar abajo 👎',   binding: 'escape' },
  { id: 'Open_Palm',   label: 'Mano abierta 🖐️',   binding: 'pause' },
  { id: 'Closed_Fist', label: 'Puño cerrado ✊',    binding: 'reset' },
];
const HOLD_FRAMES = 12;        // ~0.4 s @ 30 fps
const HOLD_CONFIDENCE = 0.65;
const TIMEOUT_MS = 8000;       // per gesture

let recognizer = null;
let active = false;
let video = null;
let cancelRequested = false;

function showDiag(html, kind = 'info') {
  const el = $('g-diag');
  el.classList.remove('hidden');
  el.className = `diag diag-${kind}`;
  el.innerHTML = html;
}
function hideDiag() { $('g-diag').classList.add('hidden'); }

async function ensureRecognizer() {
  if (recognizer) return recognizer;
  showDiag('⏳ Descargando modelo de gestos (~5 MB, sólo la primera vez)…', 'info');
  const fileset = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
  );
  recognizer = await GestureRecognizer.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: 1,
  });
  hideDiag();
  return recognizer;
}

async function ensureVideo() {
  // Reuse the preview <video> from the gaze demo if it has a stream
  let v = $('webcam-preview');
  if (!v || !v.srcObject) {
    // Need our own stream
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    if (!v) {
      v = document.createElement('video');
      v.id = 'webcam-preview';
      v.autoplay = true;
      v.muted = true;
      v.playsInline = true;
      v.className = 'webcam-preview';
      document.body.appendChild(v);
    }
    v.srcObject = stream;
  }
  if (v.readyState < 2) {
    await new Promise((res) => v.addEventListener('loadeddata', res, { once: true }));
  }
  video = v;
  return v;
}

function paintProgress(ratio) {
  $('g-bar').style.width = `${Math.max(0, Math.min(1, ratio)) * 100}%`;
}

function paintList(results) {
  const ul = $('g-list');
  ul.innerHTML = '';
  for (const r of results) {
    const li = document.createElement('li');
    li.className = `gesture-item ${r.status}`;
    li.innerHTML = `<span>${r.label}</span><span class="badge">${
      r.status === 'pending'    ? '…'   :
      r.status === 'in-progress' ? '🔵' :
      r.status === 'ok'          ? '✅' :
      r.status === 'skip'        ? '⏭️' : '❌'
    }</span>`;
    ul.appendChild(li);
  }
}

async function detectOne(target, results, idx) {
  results[idx].status = 'in-progress';
  paintList(results);
  $('g-prompt').textContent = target.label;
  $('g-sub').textContent = 'Mantén el gesto frente a la cámara…';

  let held = 0;
  const t0 = performance.now();
  let lastTs = -1;
  let skipped = false;

  const skipHandler = () => { skipped = true; };
  $('g-skip').addEventListener('click', skipHandler, { once: true });

  while (!cancelRequested && !skipped) {
    const elapsed = performance.now() - t0;
    paintProgress(elapsed / TIMEOUT_MS);
    if (elapsed > TIMEOUT_MS) {
      results[idx].status = 'fail';
      $('g-skip').removeEventListener('click', skipHandler);
      return;
    }

    const ts = performance.now();
    if (ts === lastTs) { await new Promise(r => requestAnimationFrame(r)); continue; }
    lastTs = ts;

    const out = recognizer.recognizeForVideo(video, ts);
    const top = out.gestures && out.gestures[0] && out.gestures[0][0];
    if (top && top.categoryName === target.id && top.score >= HOLD_CONFIDENCE) {
      held++;
      $('g-sub').textContent = `Detectado (${held}/${HOLD_FRAMES})`;
      if (held >= HOLD_FRAMES) {
        results[idx].status = 'ok';
        $('g-skip').removeEventListener('click', skipHandler);
        return;
      }
    } else {
      held = Math.max(0, held - 1);
      if (top) {
        $('g-sub').textContent = `Detectado: ${top.categoryName} (${(top.score*100|0)}%)`;
      } else {
        $('g-sub').textContent = 'No veo tu mano…';
      }
    }
    await new Promise(r => requestAnimationFrame(r));
  }

  $('g-skip').removeEventListener('click', skipHandler);
  if (skipped) results[idx].status = 'skip';
  if (cancelRequested) results[idx].status = results[idx].status === 'in-progress' ? 'fail' : results[idx].status;
}

async function startGestureRound() {
  if (active) return;
  active = true;
  cancelRequested = false;
  $('result').classList.add('hidden');
  $('gesture-result').classList.add('hidden');
  $('gesture-arena').classList.remove('hidden');
  hideDiag();

  // pause WebGazer if running so the camera frames are crisp
  if (window.webgazer) { try { window.webgazer.pause(); } catch (_) {} }

  const results = ROUND.map((g) => ({ ...g, status: 'pending' }));
  paintList(results);
  paintProgress(0);

  try {
    await ensureVideo();
    await ensureRecognizer();
  } catch (err) {
    showDiag('❌ No se ha podido iniciar la detección de gestos: <code>' +
             (err && err.message || err) + '</code>', 'error');
    active = false;
    return;
  }

  for (let i = 0; i < ROUND.length; i++) {
    if (cancelRequested) break;
    await detectOne(ROUND[i], results, i);
    paintList(results);
  }

  $('gesture-arena').classList.add('hidden');
  $('gesture-result').classList.remove('hidden');
  const ul = $('g-summary');
  ul.innerHTML = '';
  for (const r of results) {
    const li = document.createElement('li');
    li.className = `gesture-item ${r.status}`;
    const icon =
      r.status === 'ok'    ? '✅' :
      r.status === 'skip'  ? '⏭️' :
      r.status === 'fail'  ? '❌' : '·';
    li.innerHTML = `<span>${r.label}</span><span class="badge">${icon} <i>→ ${r.binding}</i></span>`;
    ul.appendChild(li);
  }
  active = false;
}

function cancel() {
  cancelRequested = true;
  $('gesture-arena').classList.add('hidden');
  $('result').classList.remove('hidden');
}

document.addEventListener('DOMContentLoaded', () => {
  $('start-gestures')?.addEventListener('click', startGestureRound);
  $('g-cancel')?.addEventListener('click', cancel);
  $('g-restart')?.addEventListener('click', startGestureRound);
});
