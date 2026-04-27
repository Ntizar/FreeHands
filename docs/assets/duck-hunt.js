import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.163.0/build/three.module.js';
import { FilesetResolver, GestureRecognizer } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const $ = (id) => document.getElementById(id);

const TOTAL_DUCKS = 30;
const PASS_SCORE = 22;
const HIT_RADIUS = 132;
const POINTING_CONFIDENCE = 0.50;
const SHOT_COOLDOWN_MS = 520;
const GAZE_CALIBRATION_KEY = 'freehands:gazeCalibration:v1';
const CAMERA_DEVICE_KEY = 'freehands:cameraDeviceId';

let renderer, scene, camera;
let ducks = [];
let spawned = 0;
let score = 0;
let running = false;
let gaze = { x: innerWidth / 2, y: innerHeight / 2 };
let lastShotAt = 0;
let gestureRecognizer = null;
let video = null;
let gazeActive = false;
let gestureActive = false;
let gazeCalibrationModel = null;
let audioCtx = null;

function setStatus(text) { $('duck-status').textContent = text; }
function setScore() { $('duck-score').textContent = `${score} / ${TOTAL_DUCKS}`; }

function currentUser() {
  const params = new URLSearchParams(location.search);
  return params.get('user') || localStorage.getItem('freehands:user') || 'Ntizar';
}

function setGestureLabel(text) {
  const el = $('duck-gesture');
  if (el) el.textContent = text;
}

function cameraConstraints(deviceId = localStorage.getItem(CAMERA_DEVICE_KEY)) {
  const videoConstraints = { width: { ideal: 640 }, height: { ideal: 480 } };
  if (deviceId) videoConstraints.deviceId = { exact: deviceId };
  else videoConstraints.facingMode = 'user';
  return videoConstraints;
}

async function enumerateCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((device) => device.kind === 'videoinput');
}

async function chooseNextCamera() {
  const cameras = await enumerateCameras();
  if (!cameras.length) return null;
  const current = localStorage.getItem(CAMERA_DEVICE_KEY);
  const index = cameras.findIndex((cameraDevice) => cameraDevice.deviceId === current);
  const next = cameras[(index + 1 + cameras.length) % cameras.length];
  localStorage.setItem(CAMERA_DEVICE_KEY, next.deviceId);
  return next.deviceId;
}

function rememberCamera(stream) {
  const track = stream.getVideoTracks()[0];
  const settings = track?.getSettings?.() || {};
  if (settings.deviceId) localStorage.setItem(CAMERA_DEVICE_KEY, settings.deviceId);
  return track?.label || settings.deviceId || 'camera';
}

function applyWebGazerCameraConstraints() {
  if (!window.webgazer) return;
  const constraints = { video: cameraConstraints(), audio: false };
  for (const name of ['setCameraConstraints', 'setVideoConstraints']) {
    if (typeof window.webgazer[name] === 'function') {
      try { window.webgazer[name](constraints); } catch (_) {}
    }
  }
}

function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === 'suspended') audioCtx.resume().catch(() => {});
}

function playShot(hit) {
  if (!audioCtx) return;
  const now = audioCtx.currentTime;
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = 'square';
  osc.frequency.setValueAtTime(hit ? 135 : 95, now);
  osc.frequency.exponentialRampToValueAtTime(hit ? 46 : 38, now + 0.12);
  gain.gain.setValueAtTime(hit ? 0.22 : 0.14, now);
  gain.gain.exponentialRampToValueAtTime(0.001, now + 0.16);
  osc.connect(gain).connect(audioCtx.destination);
  osc.start(now);
  osc.stop(now + 0.17);
}

function showShot(x, y, hit) {
  const layer = $('duck-shot-layer');
  if (!layer) return;
  const mark = document.createElement('div');
  mark.className = `duck-shot ${hit ? 'hit' : 'miss'}`;
  mark.style.left = `${x}px`;
  mark.style.top = `${y}px`;
  layer.appendChild(mark);
  setTimeout(() => mark.remove(), 520);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function updateAim(x, y, smooth = false) {
  const nextX = clamp(Number.isFinite(x) ? x : gaze.x, 0, innerWidth - 1);
  const nextY = clamp(Number.isFinite(y) ? y : gaze.y, 0, innerHeight - 1);
  if (smooth) {
    gaze.x = gaze.x * 0.55 + nextX * 0.45;
    gaze.y = gaze.y * 0.55 + nextY * 0.45;
  } else {
    gaze.x = nextX;
    gaze.y = nextY;
  }
  const crosshair = $('duck-crosshair');
  crosshair.style.left = `${gaze.x}px`;
  crosshair.style.top = `${gaze.y}px`;
}

function solveLinear3(matrix, vector) {
  const a = matrix.map((row, idx) => [...row, vector[idx]]);
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let row = col + 1; row < 3; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-8) return null;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const div = a[col][col];
    for (let k = col; k < 4; k++) a[col][k] /= div;
    for (let row = 0; row < 3; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let k = col; k < 4; k++) a[row][k] -= factor * a[col][k];
    }
  }
  return [a[0][3], a[1][3], a[2][3]];
}

function buildAffineGazeModel(samples) {
  const valid = samples.filter((s) => (
    Number.isFinite(s.rawX) && Number.isFinite(s.rawY) &&
    Number.isFinite(s.x) && Number.isFinite(s.y)
  ));
  if (valid.length < 6) return null;
  const rawXs = valid.map((s) => s.rawX), rawYs = valid.map((s) => s.rawY);
  const rawRange = Math.max(Math.max(...rawXs) - Math.min(...rawXs), Math.max(...rawYs) - Math.min(...rawYs));
  if (rawRange < 24) return null;

  const normal = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const targetX = [0, 0, 0], targetY = [0, 0, 0];
  for (const sample of valid) {
    const row = [sample.rawX, sample.rawY, 1];
    for (let r = 0; r < 3; r++) {
      targetX[r] += row[r] * sample.x;
      targetY[r] += row[r] * sample.y;
      for (let c = 0; c < 3; c++) normal[r][c] += row[r] * row[c];
    }
  }
  const wx = solveLinear3(normal, targetX);
  const wy = solveLinear3(normal, targetY);
  if (!wx || !wy) return null;
  return { wx, wy, count: valid.length };
}

function loadGazeCalibration() {
  try {
    const payload = JSON.parse(localStorage.getItem(GAZE_CALIBRATION_KEY) || 'null');
    if (!payload || payload.user !== currentUser() || !Array.isArray(payload.samples)) return null;
    return buildAffineGazeModel(payload.samples);
  } catch (_) {
    return null;
  }
}

function mapGazePrediction(data) {
  if (!data || !gazeCalibrationModel) return null;
  const { wx, wy } = gazeCalibrationModel;
  const x = data.x * wx[0] + data.y * wx[1] + wx[2];
  const y = data.x * wy[0] + data.y * wy[1] + wy[2];
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y };
}

function setupThree() {
  const canvas = $('duck-canvas');
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  scene = new THREE.Scene();
  camera = new THREE.OrthographicCamera(
    -innerWidth / 2, innerWidth / 2,
    innerHeight / 2, -innerHeight / 2,
    -1000, 1000,
  );
  camera.position.z = 400;

  const light = new THREE.DirectionalLight(0xffffff, 1.3);
  light.position.set(0, 200, 300);
  scene.add(light);
  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
}

function makeDuck() {
  const group = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0xffcc33, roughness: 0.55 });
  const wingMat = new THREE.MeshStandardMaterial({ color: 0xff8a1a, roughness: 0.6 });
  const beakMat = new THREE.MeshStandardMaterial({ color: 0xff7a1a, roughness: 0.45 });
  const eyeMat = new THREE.MeshStandardMaterial({ color: 0x07122d });

  const body = new THREE.Mesh(new THREE.SphereGeometry(34, 24, 16), bodyMat);
  body.scale.set(1.25, 0.85, 0.8);
  group.add(body);

  const head = new THREE.Mesh(new THREE.SphereGeometry(20, 20, 12), bodyMat);
  head.position.set(36, 22, 0);
  group.add(head);

  const beak = new THREE.Mesh(new THREE.ConeGeometry(10, 24, 16), beakMat);
  beak.rotation.z = -Math.PI / 2;
  beak.position.set(60, 20, 0);
  group.add(beak);

  const wing = new THREE.Mesh(new THREE.SphereGeometry(18, 16, 10), wingMat);
  wing.scale.set(1.2, 0.35, 0.25);
  wing.position.set(-4, 0, 20);
  group.add(wing);

  const eye = new THREE.Mesh(new THREE.SphereGeometry(3.2, 10, 8), eyeMat);
  eye.position.set(48, 29, 15);
  group.add(eye);

  group.userData = {
    speedY: 0.65 + Math.random() * 1.55,
    speedX: (Math.random() - 0.5) * 0.9,
    amp: 10 + Math.random() * 18,
    phase: Math.random() * Math.PI * 2,
  };
  return group;
}

function spawnDuck() {
  if (spawned >= TOTAL_DUCKS) return;
  const duck = makeDuck();
  const startX = -innerWidth / 2 + 100 + Math.random() * Math.max(120, innerWidth - 200);
  duck.position.x = startX;
  duck.position.y = -innerHeight / 2 - 90 - Math.random() * 90;
  duck.scale.x = duck.userData.speedX >= 0 ? 1 : -1;
  scene.add(duck);
  ducks.push(duck);
  spawned++;
}

function stopVideoStream() {
  if (video?.srcObject) {
    video.srcObject.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
  }
}

async function setupCameraAndGestures(options = {}) {
  video = $('duck-video');
  if (options.next) await chooseNextCamera();
  stopVideoStream();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: cameraConstraints(), audio: false });
  } catch (err) {
    if (localStorage.getItem(CAMERA_DEVICE_KEY)) {
      localStorage.removeItem(CAMERA_DEVICE_KEY);
      stream = await navigator.mediaDevices.getUserMedia({ video: cameraConstraints(null), audio: false });
    } else {
      throw err;
    }
  }
  video.srcObject = stream;
  await video.play().catch(() => {});
  const label = rememberCamera(stream);

  if (!gestureRecognizer) {
    const fileset = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
    const modelAssetPath = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task';
    gestureRecognizer = await GestureRecognizer.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 1,
    }).catch(async () => GestureRecognizer.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath,
        delegate: 'CPU',
      },
      runningMode: 'VIDEO',
      numHands: 1,
    }));
  }
  return label;
}

async function setupGaze() {
  if (location.protocol !== 'https:') throw new Error('HTTPS is required for gaze tracking');
  if (!window.webgazer) throw new Error('WebGazer did not load');
  gazeCalibrationModel = loadGazeCalibration();
  if (!gazeCalibrationModel) throw new Error('Browser gaze calibration was not found');
  applyWebGazerCameraConstraints();
  await window.webgazer
    .setRegression('ridge')
    .setGazeListener((data) => {
      const mapped = mapGazePrediction(data);
      if (!mapped) return;
      updateAim(mapped.x, mapped.y, true);
    })
    .saveDataAcrossSessions(true)
    .begin();
  try { window.webgazer.showVideoPreview(false); } catch (_) {}
  try { window.webgazer.showFaceOverlay(false); } catch (_) {}
  try { window.webgazer.showPredictionPoints(false); } catch (_) {}
}

function maybeGestureShoot(now) {
  if (!gestureRecognizer || !video || video.readyState < 2) return;
  const out = gestureRecognizer.recognizeForVideo(video, now);
  const top = out.gestures && out.gestures[0] && out.gestures[0][0];
  if (!top) {
    setGestureLabel('Hand: not detected');
    return;
  }
  setGestureLabel(`${top.categoryName} · ${(top.score * 100 | 0)}%`);
  if (top.categoryName === 'Pointing_Up' && top.score >= POINTING_CONFIDENCE) {
    shoot();
  }
}

function shoot() {
  if (performance.now() - lastShotAt < SHOT_COOLDOWN_MS) return;
  lastShotAt = performance.now();
  let best = null;
  let bestD = Infinity;
  for (const duck of ducks) {
    const sx = duck.position.x + innerWidth / 2;
    const sy = innerHeight / 2 - duck.position.y;
    const d = Math.hypot(sx - gaze.x, sy - gaze.y);
    if (d < bestD) { bestD = d; best = duck; }
  }
  const hit = !!(best && bestD <= HIT_RADIUS);
  showShot(gaze.x, gaze.y, hit);
  playShot(hit);
  if (hit) {
    score++;
    setScore();
    scene.remove(best);
    ducks = ducks.filter((duck) => duck !== best);
  }
}

function finishIfNeeded() {
  if (spawned >= TOTAL_DUCKS && ducks.length === 0) {
    running = false;
    const passed = score >= PASS_SCORE;
    $('duck-title').textContent = passed ? 'Test passed' : 'Needs tuning';
    $('duck-copy').textContent = passed
      ? `${currentUser()}, you hit ${score} of ${TOTAL_DUCKS}. Gaze + index are ready for faster targets.`
      : `${currentUser()}, you hit ${score} of ${TOTAL_DUCKS}. Recalibrate gaze or improve lighting before increasing speed.`;
    $('duck-result').classList.remove('hidden');
  }
}

function animate(now = 0) {
  requestAnimationFrame(animate);
  if (!running) {
    renderer?.render(scene, camera);
    return;
  }

  if (spawned < TOTAL_DUCKS && (ducks.length < 2 || Math.random() < 0.008)) spawnDuck();

  for (const duck of [...ducks]) {
    duck.position.y += duck.userData.speedY;
    duck.position.x += duck.userData.speedX + Math.sin(now * 0.004 + duck.userData.phase) * 0.22;
    duck.rotation.z = Math.sin(now * 0.008 + duck.userData.phase) * 0.08;
    if (duck.position.y > innerHeight / 2 + 150 || duck.position.x < -innerWidth / 2 - 160 || duck.position.x > innerWidth / 2 + 160) {
      scene.remove(duck);
      ducks = ducks.filter((item) => item !== duck);
    }
  }

  maybeGestureShoot(now);
  finishIfNeeded();
  renderer.render(scene, camera);
}

async function start() {
  $('duck-result').classList.add('hidden');
  $('duck-start').disabled = true;
  localStorage.setItem('freehands:user', currentUser());
  setStatus(`Loading · ${currentUser()}`);
  if (!renderer) setupThree();
  score = 0; spawned = 0; ducks.forEach((d) => scene.remove(d)); ducks = [];
  gazeActive = false;
  gestureActive = false;
  setScore();
  ensureAudio();
  try {
    const label = await setupCameraAndGestures();
    gestureActive = true;
    setGestureLabel(`Camera: ${label}`);
  } catch (_) {
    gestureActive = false;
    setGestureLabel('Hand: no camera');
  }
  try {
    await setupGaze();
    gazeActive = true;
  } catch (_) {
    gazeActive = false;
  }

  if (gazeActive && gestureActive) setStatus(`Gaze + index (${gazeCalibrationModel.count} pts)`);
  else if (gazeActive) setStatus('Gaze + click');
  else if (gestureActive) setStatus('Pointer + index · calibrate gaze first');
  else setStatus('Pointer + click · calibrate gaze first');

  running = true;
  $('duck-start').disabled = false;
}

async function switchCamera() {
  const wasRunning = running;
  running = false;
  setStatus('Switching camera...');
  try {
    const label = await setupCameraAndGestures({ next: true });
    setGestureLabel(`Camera: ${label}`);
    gestureActive = true;
    if (window.webgazer) {
      try { window.webgazer.end(); } catch (_) {}
      if (gazeActive) await setupGaze();
    }
    setStatus(wasRunning ? 'Camera switched' : 'Ready');
  } catch (_) {
    setStatus('Camera switch failed');
  } finally {
    running = wasRunning;
  }
}

addEventListener('resize', () => {
  if (!renderer) return;
  renderer.setSize(innerWidth, innerHeight);
  camera.left = -innerWidth / 2;
  camera.right = innerWidth / 2;
  camera.top = innerHeight / 2;
  camera.bottom = -innerHeight / 2;
  camera.updateProjectionMatrix();
});

addEventListener('click', () => { if (running) shoot(); });
addEventListener('keydown', (ev) => { if (running && ev.code === 'Space') shoot(); });
addEventListener('pointermove', (ev) => { if (!gazeActive) updateAim(ev.clientX, ev.clientY, false); });

document.addEventListener('DOMContentLoaded', () => {
  const userLabel = $('duck-user');
  if (userLabel) userLabel.textContent = `FreeHands · ${currentUser()}`;
  setupThree();
  animate();
  $('duck-start').addEventListener('click', start);
  $('duck-again').addEventListener('click', start);
  $('duck-camera').addEventListener('click', switchCamera);
});
