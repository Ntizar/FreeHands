import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.163.0/build/three.module.js';
import { FilesetResolver, GestureRecognizer } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const $ = (id) => document.getElementById(id);

const TOTAL_DUCKS = 50;
const PASS_SCORE = 45;
const HIT_RADIUS = 118;
const POINTING_CONFIDENCE = 0.50;
const SHOT_COOLDOWN_MS = 520;

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
let audioCtx = null;

function setStatus(text) { $('duck-status').textContent = text; }
function setScore() { $('duck-score').textContent = `${score} / ${TOTAL_DUCKS}`; }

function setGestureLabel(text) {
  const el = $('duck-gesture');
  if (el) el.textContent = text;
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

function updateAim(x, y) {
  gaze.x = x;
  gaze.y = y;
  const crosshair = $('duck-crosshair');
  crosshair.style.left = `${x}px`;
  crosshair.style.top = `${y}px`;
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
    speed: 1.8 + Math.random() * 1.8,
    amp: 10 + Math.random() * 16,
    phase: Math.random() * Math.PI * 2,
  };
  return group;
}

function spawnDuck() {
  if (spawned >= TOTAL_DUCKS) return;
  const duck = makeDuck();
  const direction = Math.random() > 0.5 ? 1 : -1;
  duck.position.x = direction > 0 ? -innerWidth / 2 - 80 : innerWidth / 2 + 80;
  duck.position.y = -innerHeight * 0.30 + Math.random() * innerHeight * 0.60;
  duck.scale.x = direction;
  duck.userData.direction = direction;
  scene.add(duck);
  ducks.push(duck);
  spawned++;
}

async function setupCameraAndGestures() {
  video = $('duck-video');
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
  video.srcObject = stream;
  await video.play().catch(() => {});

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

async function setupGaze() {
  if (location.protocol !== 'https:') throw new Error('HTTPS requerido para mirada');
  if (!window.webgazer) throw new Error('WebGazer no cargó');
  await window.webgazer
    .setRegression('ridge')
    .setGazeListener((data) => {
      if (!data) return;
      updateAim(data.x, data.y);
    })
    .saveDataAcrossSessions(false)
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
    setGestureLabel('Mano: no detectada');
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
    $('duck-title').textContent = passed ? '45/50 desbloqueado' : 'Ajuste pendiente';
    $('duck-copy').textContent = passed
      ? `Has acertado ${score} de ${TOTAL_DUCKS}. Este perfil está listo para subir precisión.`
      : `Has acertado ${score} de ${TOTAL_DUCKS}. Repite una ronda corta y ajustamos sensibilidad.`;
    $('duck-result').classList.remove('hidden');
  }
}

function animate(now = 0) {
  requestAnimationFrame(animate);
  if (!running) {
    renderer?.render(scene, camera);
    return;
  }

  if (spawned < TOTAL_DUCKS && (ducks.length < 4 || Math.random() < 0.018)) spawnDuck();

  for (const duck of [...ducks]) {
    duck.position.x += duck.userData.direction * duck.userData.speed;
    duck.position.y += Math.sin(now * 0.006 + duck.userData.phase) * 0.45;
    duck.rotation.z = Math.sin(now * 0.008 + duck.userData.phase) * 0.08;
    if (duck.position.x < -innerWidth / 2 - 140 || duck.position.x > innerWidth / 2 + 140) {
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
  setStatus('Cargando');
  if (!renderer) setupThree();
  score = 0; spawned = 0; ducks.forEach((d) => scene.remove(d)); ducks = [];
  gazeActive = false;
  gestureActive = false;
  setScore();
  ensureAudio();
  try {
    await setupCameraAndGestures();
    gestureActive = true;
  } catch (_) {
    gestureActive = false;
    setGestureLabel('Mano: sin camara');
  }
  try {
    await setupGaze();
    gazeActive = true;
  } catch (_) {
    gazeActive = false;
  }

  if (gazeActive && gestureActive) setStatus('Mirada + indice');
  else if (gazeActive) setStatus('Mirada + click');
  else if (gestureActive) setStatus('Puntero + indice');
  else setStatus('Puntero + click');

  running = true;
  $('duck-start').disabled = false;
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
addEventListener('pointermove', (ev) => { if (!gazeActive) updateAim(ev.clientX, ev.clientY); });

document.addEventListener('DOMContentLoaded', () => {
  setupThree();
  animate();
  $('duck-start').addEventListener('click', start);
  $('duck-again').addEventListener('click', start);
});
