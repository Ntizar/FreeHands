import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.163.0/build/three.module.js';

const $ = (id) => document.getElementById(id);

const TOTAL_DUCKS = 30;
const PASS_SCORE = 22;
const HIT_RADIUS = 148;
const SHOT_COOLDOWN_MS = 120;
const AIM_ASSIST_HOLD_MS = 520;
const AIM_ASSIST_RADIUS = 190;
const AIM_ASSIST_STABLE_RADIUS = 64;

let renderer;
let scene;
let camera;
let ducks = [];
let spawned = 0;
let score = 0;
let running = false;
let aim = { x: innerWidth / 2, y: innerHeight / 2 };
let aimAnchor = { x: innerWidth / 2, y: innerHeight / 2, since: 0 };
let lastShotAt = 0;
let audioCtx = null;

function setStatus(text) { $('duck-status').textContent = text; }
function setScore() { $('duck-score').textContent = `${score} / ${TOTAL_DUCKS}`; }

function currentUser() {
  const params = new URLSearchParams(location.search);
  return params.get('user') || localStorage.getItem('freehands:user') || 'Ntizar';
}

function setLocalLabel(text) {
  const label = $('duck-gesture');
  if (label) label.textContent = text;
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

function updateAim(x, y) {
  const rawX = clamp(Number.isFinite(x) ? x : aim.x, 0, innerWidth - 1);
  const rawY = clamp(Number.isFinite(y) ? y : aim.y, 0, innerHeight - 1);
  const now = performance.now();
  if (Math.hypot(rawX - aimAnchor.x, rawY - aimAnchor.y) > AIM_ASSIST_STABLE_RADIUS) {
    aimAnchor = { x: rawX, y: rawY, since: now };
  }
  const assisted = now - aimAnchor.since >= AIM_ASSIST_HOLD_MS ? assistedAim(rawX, rawY) : null;
  aim.x = assisted?.x ?? rawX;
  aim.y = assisted?.y ?? rawY;
  const crosshair = $('duck-crosshair');
  crosshair.classList.toggle('snap', !!assisted);
  crosshair.style.left = `${aim.x}px`;
  crosshair.style.top = `${aim.y}px`;
}

function assistedAim(x, y) {
  let best = null;
  let bestDistance = Infinity;
  for (const duck of ducks) {
    const screenX = duck.position.x + innerWidth / 2;
    const screenY = innerHeight / 2 - duck.position.y;
    const distance = Math.hypot(screenX - x, screenY - y);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = { x: screenX, y: screenY };
    }
  }
  return best && bestDistance <= AIM_ASSIST_RADIUS ? best : null;
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

function shoot() {
  if (performance.now() - lastShotAt < SHOT_COOLDOWN_MS) return;
  lastShotAt = performance.now();
  let best = null;
  let bestDistance = Infinity;
  for (const duck of ducks) {
    const screenX = duck.position.x + innerWidth / 2;
    const screenY = innerHeight / 2 - duck.position.y;
    const distance = Math.hypot(screenX - aim.x, screenY - aim.y);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = duck;
    }
  }
  const hit = !!(best && bestDistance <= HIT_RADIUS);
  showShot(aim.x, aim.y, hit);
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
    $('duck-title').textContent = passed ? 'Local test passed' : 'Needs tuning';
    $('duck-copy').textContent = passed
      ? `${currentUser()}, you hit ${score} of ${TOTAL_DUCKS}. The desktop pointer and click gestures are working.`
      : `${currentUser()}, you hit ${score} of ${TOTAL_DUCKS}. Recalibrate local gaze or gestures, then try again.`;
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

  finishIfNeeded();
  renderer.render(scene, camera);
}

function start() {
  $('duck-result').classList.add('hidden');
  localStorage.setItem('freehands:user', currentUser());
  setStatus(`Local app test · ${currentUser()}`);
  setLocalLabel('Local app: OS pointer + clicks only');
  if (!renderer) setupThree();
  score = 0;
  spawned = 0;
  ducks.forEach((duck) => scene.remove(duck));
  ducks = [];
  lastShotAt = 0;
  setScore();
  ensureAudio();
  running = true;
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

addEventListener('click', (event) => {
  if (!running) return;
  const target = event.target;
  if (target instanceof Element && target.closest('button,a,.duck-help,.duck-hud,.duck-result,.nav')) return;
  shoot();
});
addEventListener('pointerdown', (event) => {
  if (!running || event.button !== 0) return;
  const target = event.target;
  if (target instanceof Element && target.closest('button,a,.duck-help,.duck-hud,.duck-result,.nav')) return;
  updateAim(event.clientX, event.clientY);
  shoot();
});
addEventListener('keydown', (event) => { if (running && event.code === 'Space') shoot(); });
addEventListener('pointermove', (event) => updateAim(event.clientX, event.clientY));

document.addEventListener('DOMContentLoaded', () => {
  const userLabel = $('duck-user');
  if (userLabel) userLabel.textContent = `FreeHands · ${currentUser()}`;
  setupThree();
  updateAim(innerWidth / 2, innerHeight / 2);
  animate();
  $('duck-start').addEventListener('click', (event) => {
    event.stopPropagation();
    start();
  });
  $('duck-again').addEventListener('click', (event) => {
    event.stopPropagation();
    start();
  });
});