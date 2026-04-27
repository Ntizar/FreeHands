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
  { id: 'Pointing_Up', label: 'Index up', binding: 'click' },
  { id: 'Thumb_Up',    label: 'Thumb up', binding: 'click' },
  { id: 'Thumb_Down',  label: 'Thumb down', binding: 'escape' },
  { id: 'Open_Palm',   label: 'Open palm', binding: 'pause' },
  { id: 'Closed_Fist', label: 'Closed fist', binding: 'reset' },
];
const HOLD_FRAMES = 8;         // test mode: easier lock
const HOLD_CONFIDENCE = 0.45;
const TIMEOUT_MS = 12000;      // per gesture

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

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
  showDiag('Downloading gesture model (~5 MB, first run only)...', 'info');
  const fileset = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
  );
  const modelAssetPath =
    'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task';
  async function create(delegate) {
    return GestureRecognizer.createFromOptions(fileset, {
      baseOptions: { modelAssetPath, delegate },
      runningMode: 'VIDEO',
      numHands: 1,
    });
  }
  try {
    recognizer = await create('GPU');
  } catch (gpuErr) {
    console.warn('[FreeHands] MediaPipe GPU delegate failed, retrying CPU.', gpuErr);
    showDiag('GPU is unavailable for gestures. Retrying with CPU...', 'warn');
    recognizer = await create('CPU');
  }
  hideDiag();
  return recognizer;
}

async function ensureVideo() {
  const sourcePreview = $('webcam-preview');
  let stream = sourcePreview && sourcePreview.srcObject;
  if (!stream) {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    if (sourcePreview) sourcePreview.srcObject = stream;
  }
  const v = $('gesture-video') || sourcePreview;
  v.srcObject = stream || sourcePreview.srcObject;
  v.muted = true;
  v.playsInline = true;
  await v.play().catch(() => {});
  if (v.readyState < 2) {
    await new Promise((res) => v.addEventListener('loadeddata', res, { once: true }));
  }
  video = v;
  return v;
}

function drawGestureFrame(out, top) {
  const canvas = $('gesture-canvas');
  const live = $('gesture-live');
  if (!canvas || !video) return;
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  const landmarks = out.landmarks && out.landmarks[0];
  if (landmarks && landmarks.length) {
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#FF7A1A';
    ctx.fillStyle = '#1E5BFF';
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
      ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
      ctx.stroke();
    }
    for (const lm of landmarks) {
      ctx.beginPath();
      ctx.arc(lm.x * w, lm.y * h, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (live) {
    if (top) live.textContent = `${top.categoryName} · ${(top.score * 100 | 0)}%`;
    else if (landmarks && landmarks.length) live.textContent = 'Hand detected · no clear gesture';
    else live.textContent = 'I cannot see your hand';
  }
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
  $('g-sub').textContent = 'Hold the gesture in front of the camera...';

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
    drawGestureFrame(out, top);
    if (top && top.categoryName === target.id && top.score >= HOLD_CONFIDENCE) {
      held++;
      $('g-sub').textContent = `Detected (${held}/${HOLD_FRAMES})`;
      if (held >= HOLD_FRAMES) {
        results[idx].status = 'ok';
        $('g-skip').removeEventListener('click', skipHandler);
        return;
      }
    } else {
      held = Math.max(0, held - 1);
      if (top) {
        $('g-sub').textContent = `Detected: ${top.categoryName} (${(top.score*100|0)}%)`;
      } else {
        $('g-sub').textContent = 'I cannot see your hand...';
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
    showDiag('Could not start gesture detection: <code>' +
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
    li.innerHTML = `<span>${r.label}</span><span class="badge">${icon} <i>-> ${r.binding}</i></span>`;
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
