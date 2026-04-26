import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.163.0/build/three.module.js';
import { FaceLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const $ = (id) => document.getElementById(id);

const TOTAL_DUCKS = 50;
const PASS_SCORE = 45;
const HIT_RADIUS = 72;
const WINK_THRESHOLD = 0.55;
const OTHER_EYE_MAX = 0.35;

let renderer, scene, camera;
let ducks = [];
let spawned = 0;
let score = 0;
let running = false;
let gaze = { x: innerWidth / 2, y: innerHeight / 2 };
let lastShotAt = 0;
let faceLandmarker = null;
let video = null;
let gazeActive = false;
let winkActive = false;

function setStatus(text) { $('duck-status').textContent = text; }
function setScore() { $('duck-score').textContent = `${score} / ${TOTAL_DUCKS}`; }

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
    speed: 2.8 + Math.random() * 2.2,
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

async function setupCameraAndFace() {
  video = $('duck-video');
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
  video.srcObject = stream;
  await video.play().catch(() => {});

  const fileset = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
  }).catch(async () => FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
      delegate: 'CPU',
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
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

function maybeWinkShoot(now) {
  if (!faceLandmarker || !video || video.readyState < 2) return;
  const out = faceLandmarker.detectForVideo(video, now);
  const shapes = out.faceBlendshapes && out.faceBlendshapes[0] && out.faceBlendshapes[0].categories;
  if (!shapes) return;
  const get = (name) => shapes.find((c) => c.categoryName === name)?.score || 0;
  const left = get('eyeBlinkLeft');
  const right = get('eyeBlinkRight');
  const wink = (left > WINK_THRESHOLD && right < OTHER_EYE_MAX) || (right > WINK_THRESHOLD && left < OTHER_EYE_MAX);
  if (wink && now - lastShotAt > 650) shoot();
}

function shoot() {
  lastShotAt = performance.now();
  let best = null;
  let bestD = Infinity;
  for (const duck of ducks) {
    const sx = duck.position.x + innerWidth / 2;
    const sy = innerHeight / 2 - duck.position.y;
    const d = Math.hypot(sx - gaze.x, sy - gaze.y);
    if (d < bestD) { bestD = d; best = duck; }
  }
  if (best && bestD <= HIT_RADIUS) {
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

  maybeWinkShoot(now);
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
  winkActive = false;
  setScore();
  try {
    await setupCameraAndFace();
    winkActive = true;
  } catch (_) {
    winkActive = false;
  }
  try {
    await setupGaze();
    gazeActive = true;
  } catch (_) {
    gazeActive = false;
  }

  if (gazeActive && winkActive) setStatus('Mirada + guiño');
  else if (gazeActive) setStatus('Mirada + click');
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
