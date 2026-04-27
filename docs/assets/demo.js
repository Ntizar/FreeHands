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

  let plan = [];
  let idx = 0;
  let errors = [];
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
      issues.push('La página debe servirse por <b>HTTPS</b> o <b>localhost</b> para acceder a la webcam. ' +
                  'URL actual: <code>' + location.href + '</code>');
    }
    return issues;
  }

  function describeCameraError(err) {
    const name = err && err.name;
    if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
      return 'Has <b>denegado el permiso</b> de cámara. Pulsa el icono del candado en la barra de direcciones → <i>Permisos del sitio</i> → permite la <b>Cámara</b> y recarga.';
    }
    if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
      return 'No se ha encontrado ninguna <b>webcam</b> conectada.';
    }
    if (name === 'NotReadableError' || name === 'TrackStartError') {
      return 'La webcam la está usando otra app (Zoom, Teams, OBS, otra pestaña…). Ciérrala y vuelve a probar.';
    }
    if (name === 'OverconstrainedError') {
      return 'Tu cámara no soporta la configuración pedida.';
    }
    if (name === 'SecurityError') {
      return 'Bloqueo de seguridad del navegador. Asegúrate de cargar la página desde <b>https://</b>.';
    }
    return 'Error desconocido al acceder a la cámara: <code>' + (err && err.message || name || err) + '</code>';
  }

  async function requestCamera() {
    const issues = envOk();
    if (issues.length) {
      const e = new Error(issues.map(i => '• ' + i).join('<br/>'));
      e.name = 'EnvError';
      throw e;
    }
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    return cameraStream;
  }

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
      cameraStream = null;
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
    updateHUD('CLIC EN EL PUNTO');
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
            const ex = pred.x - tx, ey = pred.y - ty;
            errors.push(Math.hypot(ex, ey));
          }
        }).catch(() => {});
      } catch (_) {}
    }

    idx++;
    placeTarget();
  }

  function finish() {
    ui.target.classList.add('hidden');
    updateHUD('FINALIZADO');
    if (window.webgazer) { try { window.webgazer.pause(); } catch (_) {} }
    setTimeout(() => {
      ui.arena.classList.add('hidden');
      ui.result.classList.remove('hidden');
      ui.rPoints.textContent = String(idx);
      const user = saveUserName();
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
          reject(new Error('No se ha podido cargar WebGazer.js (¿adblocker o sin red?). ' +
            'Mira la consola del navegador (F12) para ver qué CDN ha fallado.'));
        }
      }, 80);
    });
  }

  async function startWebGazer() {
    await waitForWebGazer();
    return window.webgazer
      .setRegression('ridge')
      .setGazeListener((data) => {
        if (!data) return;
        ui.cursor.style.left = `${data.x}px`;
        ui.cursor.style.top  = `${data.y}px`;
      })
      .saveDataAcrossSessions(false)
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
    updateHUD('PIDIENDO PERMISO DE CÁMARA…');

    try {
      const stream = await requestCamera();
      ui.preview.srcObject = stream;
    } catch (err) {
      ui.arena.classList.add('hidden');
      ui.welcome.classList.remove('hidden');
      const msg = err && err.name === 'EnvError' ? err.message : describeCameraError(err);
      showDiag('❌ <b>No se ha podido activar la cámara.</b><br/>' + msg, 'error');
      return;
    }

    updateHUD('CARGANDO MODELO…');
    try {
      await startWebGazer();
    } catch (err) {
      ui.arena.classList.add('hidden');
      ui.welcome.classList.remove('hidden');
      showDiag('❌ ' + (err && err.message || err), 'error');
      stopCamera();
      return;
    }

    plan = buildPlan();
    idx = 0;
    errors = [];
    placeTarget();
  }

  async function checkCamera() {
    clearDiag();
    showDiag('⏳ Comprobando cámara…', 'info');
    try {
      const stream = await requestCamera();
      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings();
      stream.getTracks().forEach(t => t.stop());
      showDiag(
        '✅ <b>Cámara accesible.</b><br/>' +
        `Dispositivo: <code>${track.label || '(sin nombre)'}</code><br/>` +
        `Resolución: <code>${settings.width || '?'}×${settings.height || '?'}</code> @ ${settings.frameRate || '?'} fps<br/>` +
        '👉 Pulsa <b>Empezar</b> para iniciar la calibración.',
        'success',
      );
    } catch (err) {
      const msg = err && err.name === 'EnvError' ? err.message : describeCameraError(err);
      showDiag('❌ <b>Cámara no disponible.</b><br/>' + msg, 'error');
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
