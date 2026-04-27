/* ──────────────────────────────────────────────────────────────────────
   FreeHands · web demo
   Aim-trainer calibration on top of WebGazer.js
   100 % client-side, no data leaves the browser.
   ────────────────────────────────────────────────────────────────────── */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const ui = {
    welcome:  $('welcome'),
    arena:    $('arena'),
    cursor:   $('cursor'),
    target:   $('target'),
    hudState: $('hud-state'),
    hudProg:  $('hud-progress'),
    result:   $('result'),
    rPoints:  $('r-points'),
    rRms:     $('r-rms'),
    diag:     $('diag'),
    preview:  $('webcam-preview'),
    userName: $('user-name'),
    testDucks: $('test-ducks'),
    switchCam: $('switch-cam'),
    arenaSwitchCam: $('arena-switch-cam'),
    cameraName: $('camera-name'),
  };

  const NORMALIZED_POINTS = [
    [0.00, 0.00], [1.00, 0.00], [1.00, 1.00], [0.00, 1.00],
    [0.50, 0.50],
    [0.50, 0.08], [0.92, 0.50], [0.50, 0.92], [0.08, 0.50],
    [0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75],
    [0.50, 0.30], [0.70, 0.50], [0.50, 0.70], [0.30, 0.50],
  ];
  const SAMPLES_PER_POINT = 1;
  const HIT_RADIUS_PX = 75;
  const GAZE_CALIBRATION_KEY = 'freehands:gazeCalibration:v1';
  const CAMERA_DEVICE_KEY = 'freehands:cameraDeviceId';

  let plan = [];
  let idx = 0;
  let errors = [];
  let calibrationSamples = [];
  let webgazerReady = false;
  let cameraStream = null;

  function getUserName() {
    const raw = (ui.userName && ui.userName.value || localStorage.getItem('freehands:user') || 'Ntizar').trim();
    return raw || 'Ntizar';
  }

  function saveUserName() {
    const user = getUserName();
    localStorage.setItem('freehands:user', user);
    if (ui.userName) ui.userName.value = user;
    if (ui.testDucks) ui.testDucks.href = `duck-hunt.html?user=${encodeURIComponent(user)}`;
    return user;
  }

  function saveGazeCalibration() {
    if (calibrationSamples.length < 4) return;
    const payload = {
      user: getUserName(),
      createdAt: new Date().toISOString(),
      width: window.innerWidth,
      height: window.innerHeight,
      samples: calibrationSamples.slice(-40),
    };
    localStorage.setItem(GAZE_CALIBRATION_KEY, JSON.stringify(payload));
  }

  function recordGazeSample(pred, targetX, targetY) {
    if (!pred || typeof pred.x !== 'number' || typeof pred.y !== 'number') return;
    if (!Number.isFinite(pred.x) || !Number.isFinite(pred.y)) return;
    calibrationSamples.push({ rawX: pred.x, rawY: pred.y, x: targetX, y: targetY });
    saveGazeCalibration();
    const ex = pred.x - targetX, ey = pred.y - targetY;
    errors.push(Math.hypot(ex, ey));
  }

  // ── Diagnostics box ────────────────────────────────────────────────
  function showDiag(html, kind = 'info') {
    ui.diag.classList.remove('hidden');
    ui.diag.className = `diag diag-${kind}`;
    ui.diag.innerHTML = html;
  }
  function clearDiag() { ui.diag.classList.add('hidden'); ui.diag.innerHTML = ''; }

  // ── Environment checks ─────────────────────────────────────────────
  function envOk() {
    const issues = [];
    if (!('mediaDevices' in navigator) || !navigator.mediaDevices.getUserMedia) {
      issues.push('Tu navegador no soporta <code>getUserMedia</code>. Usa Chrome / Edge / Firefox / Safari recientes.');
    }
    if (location.protocol !== 'https:' &&
        location.hostname !== 'localhost' &&
        location.hostname !== '127.0.0.1') {
      issues.push('This page must run on <b>HTTPS</b> or <b>localhost</b> to access the webcam. ' +
          'Current URL: <code>' + location.href + '</code>');
    }
    return issues;
  }

  function describeCameraError(err) {
    const name = err && err.name;
    if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
      return 'Camera permission was <b>denied</b>. Click the lock icon in the address bar, open <i>Site permissions</i>, allow <b>Camera</b>, then reload.';
    }
    if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
      return 'No connected <b>webcam</b> was found.';
    }
    if (name === 'NotReadableError' || name === 'TrackStartError') {
      return 'The webcam is busy in another app or tab. Close it and try again.';
    }
    if (name === 'OverconstrainedError') {
      return 'This camera does not support the requested configuration.';
    }
    if (name === 'SecurityError') {
      return 'Browser security blocked camera access. Load the page from <b>https://</b>.';
    }
    return 'Unknown camera error: <code>' + (err && err.message || name || err) + '</code>';
  }

  function cameraConstraints(deviceId = localStorage.getItem(CAMERA_DEVICE_KEY)) {
    const video = { width: { ideal: 640 }, height: { ideal: 480 } };
    if (deviceId) video.deviceId = { exact: deviceId };
    else video.facingMode = 'user';
    return video;
  }

  async function enumerateCameras() {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((device) => device.kind === 'videoinput');
  }

  function setCameraName(label) {
    if (ui.cameraName) ui.cameraName.textContent = `Camera: ${label || 'auto'}`;
  }

  async function rememberStreamCamera(stream) {
    const track = stream.getVideoTracks()[0];
    const settings = track?.getSettings?.() || {};
    if (settings.deviceId) localStorage.setItem(CAMERA_DEVICE_KEY, settings.deviceId);
    setCameraName(track?.label || settings.deviceId || 'auto');
    return track;
  }

  async function chooseNextCamera() {
    const cameras = await enumerateCameras();
    if (!cameras.length) return null;
    const current = localStorage.getItem(CAMERA_DEVICE_KEY);
    const index = cameras.findIndex((camera) => camera.deviceId === current);
    const next = cameras[(index + 1 + cameras.length) % cameras.length];
    localStorage.setItem(CAMERA_DEVICE_KEY, next.deviceId);
    setCameraName(next.label || `camera ${cameras.indexOf(next) + 1}`);
    return next.deviceId;
  }

  async function requestCamera(options = {}) {
    const issues = envOk();
    if (issues.length) {
      const e = new Error(issues.map(i => '• ' + i).join('<br/>'));
      e.name = 'EnvError';
      throw e;
    }
    if (options.next) await chooseNextCamera();
    stopCamera();
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: cameraConstraints(),
        audio: false,
      });
    } catch (err) {
      if (localStorage.getItem(CAMERA_DEVICE_KEY)) {
        localStorage.removeItem(CAMERA_DEVICE_KEY);
        cameraStream = await navigator.mediaDevices.getUserMedia({
          video: cameraConstraints(null),
          audio: false,
        });
      } else {
        throw err;
      }
    }
    await rememberStreamCamera(cameraStream);
    if (ui.preview) {
      ui.preview.srcObject = cameraStream;
      await ui.preview.play().catch(() => {});
    }
    return cameraStream;
  }

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
      cameraStream = null;
    }
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

  async function switchCamera() {
    clearDiag();
    try {
      const stream = await requestCamera({ next: true });
      const track = stream.getVideoTracks()[0];
      showDiag(`Camera selected: <b>${track.label || 'next camera'}</b>`, 'success');
      if (webgazerReady && window.webgazer) {
        try { window.webgazer.end(); } catch (_) {}
        webgazerReady = false;
        await startWebGazer();
      }
    } catch (err) {
      showDiag('Camera switch failed: ' + describeCameraError(err), 'error');
    }
  }

  // ── Aim-trainer plan ───────────────────────────────────────────────
  function buildPlan() {
    const w = window.innerWidth, h = window.innerHeight;
    const margin = 42;
    const pts = [];
    for (const [nx, ny] of NORMALIZED_POINTS) {
      const px = Math.round(margin + nx * (w - 2 * margin));
      const py = Math.round(margin + ny * (h - 2 * margin));
      for (let i = 0; i < SAMPLES_PER_POINT; i++) pts.push([px, py]);
    }
    return pts;
  }

  function updateHUD(state) {
    ui.hudState.textContent = state;
    ui.hudProg.textContent = `${idx} / ${plan.length}`;
  }

  function placeTarget() {
    if (idx >= plan.length) { return finish(); }
    const [x, y] = plan[idx];
    ui.target.style.left = `${x}px`;
    ui.target.style.top  = `${y}px`;
    ui.target.classList.remove('hidden');
    updateHUD('CLICK THE TARGET');
  }

  function onTargetClick(ev) {
    const [tx, ty] = plan[idx];
    const dx = ev.clientX - tx, dy = ev.clientY - ty;
    if (dx * dx + dy * dy > HIT_RADIUS_PX * HIT_RADIUS_PX) return;

    if (window.webgazer && webgazerReady) {
      try {
        const got = window.webgazer.getCurrentPrediction();
        Promise.resolve(got).then((pred) => {
          if (pred && typeof pred.x === 'number') {
            recordGazeSample(pred, tx, ty);
          }
        }).catch(() => {});
      } catch (_) {}
    }

    idx++;
    placeTarget();
  }

  function finish() {
    ui.target.classList.add('hidden');
    updateHUD('DONE');
    if (window.webgazer) { try { window.webgazer.pause(); } catch (_) {} }
    setTimeout(() => {
      ui.arena.classList.add('hidden');
      ui.result.classList.remove('hidden');
      ui.rPoints.textContent = String(idx);
      const user = saveUserName();
      saveGazeCalibration();
      ui.testDucks.href = `duck-hunt.html?user=${encodeURIComponent(user)}`;
      if (errors.length) {
        const rms = Math.sqrt(errors.reduce((a, b) => a + b * b, 0) / errors.length);
        ui.rRms.textContent = `${Math.round(rms)} px`;
      } else {
        ui.rRms.textContent = 'no disponible';
      }
    }, 250);
  }

  // ── Wait for webgazer.js script to load ────────────────────────────
  function waitForWebGazer(timeoutMs = 10000) {
    return new Promise((resolve, reject) => {
      if (window.webgazer) return resolve();
      const t0 = Date.now();
      const timer = setInterval(() => {
        if (window.webgazer) { clearInterval(timer); resolve(); }
        else if (window.__webgazerLoadError) {
          clearInterval(timer);
          reject(new Error(window.__webgazerLoadError));
        }
        else if (Date.now() - t0 > timeoutMs) {
          clearInterval(timer);
          reject(new Error('WebGazer.js could not be loaded. Check ad blockers, network access, or the browser console.'));
        }
      }, 80);
    });
  }

  async function startWebGazer() {
    await waitForWebGazer();
    applyWebGazerCameraConstraints();
    return window.webgazer
      .setRegression('ridge')
      .setGazeListener((data) => {
        if (!data) return;
        ui.cursor.style.left = `${data.x}px`;
        ui.cursor.style.top  = `${data.y}px`;
      })
      .saveDataAcrossSessions(true)
      .begin()
      .then(() => {
        webgazerReady = true;
        try { window.webgazer.showVideoPreview(false); } catch (_) {}
        try { window.webgazer.showFaceOverlay(false); } catch (_) {}
        try { window.webgazer.showFaceFeedbackBox(false); } catch (_) {}
        try { window.webgazer.showPredictionPoints(false); } catch (_) {}
      });
  }

  // ── Public actions ─────────────────────────────────────────────────
  async function start() {
    clearDiag();
    saveUserName();
    ui.welcome.classList.add('hidden');
    ui.arena.classList.remove('hidden');
    updateHUD('REQUESTING CAMERA...');

    try {
      await requestCamera();
    } catch (err) {
      ui.arena.classList.add('hidden');
      ui.welcome.classList.remove('hidden');
      const msg = err && err.name === 'EnvError' ? err.message : describeCameraError(err);
      showDiag('<b>Camera could not start.</b><br/>' + msg, 'error');
      return;
    }

    updateHUD('LOADING MODEL...');
    try {
      await startWebGazer();
    } catch (err) {
      ui.arena.classList.add('hidden');
      ui.welcome.classList.remove('hidden');
      showDiag((err && err.message || err), 'error');
      stopCamera();
      return;
    }

    plan = buildPlan();
    idx = 0;
    errors = [];
    calibrationSamples = [];
    placeTarget();
  }

  async function checkCamera() {
    clearDiag();
    showDiag('Checking camera...', 'info');
    try {
      const stream = await requestCamera();
      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings();
      stream.getTracks().forEach(t => t.stop());
      showDiag(
        '<b>Camera is accessible.</b><br/>' +
        `Device: <code>${track.label || '(unnamed)'}</code><br/>` +
        `Resolution: <code>${settings.width || '?'}x${settings.height || '?'}</code> @ ${settings.frameRate || '?'} fps<br/>` +
        'Press <b>Start</b> to begin calibration, or <b>Switch camera</b> if the preview is frozen.',
        'success',
      );
    } catch (err) {
      const msg = err && err.name === 'EnvError' ? err.message : describeCameraError(err);
      showDiag('<b>Camera unavailable.</b><br/>' + msg, 'error');
    }
  }

  function restart() {
    ui.result.classList.add('hidden');
    ui.welcome.classList.remove('hidden');
    if (window.webgazer) { try { window.webgazer.end(); } catch (_) {} }
    stopCamera();
    webgazerReady = false;
  }

  // ── wire up ─────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    if (ui.userName) ui.userName.value = localStorage.getItem('freehands:user') || ui.userName.value || 'Ntizar';
    saveUserName();
    ui.userName?.addEventListener('input', saveUserName);
    $('start').addEventListener('click', start);
    $('check-cam').addEventListener('click', checkCamera);
    ui.switchCam?.addEventListener('click', switchCamera);
    ui.arenaSwitchCam?.addEventListener('click', switchCamera);
    $('finish').addEventListener('click', finish);
    $('restart').addEventListener('click', restart);
    ui.target.addEventListener('click', onTargetClick);
    window.addEventListener('beforeunload', () => {
      if (window.webgazer) { try { window.webgazer.end(); } catch (_) {} }
      stopCamera();
    });

    const issues = envOk();
    if (issues.length) {
      showDiag('⚠️ ' + issues.map(i => '• ' + i).join('<br/>'), 'warn');
    }
  });
})();
