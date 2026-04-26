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
  };

  const NORMALIZED_POINTS = [
    [0.08, 0.10], [0.50, 0.10], [0.92, 0.10],
    [0.08, 0.50], [0.50, 0.50], [0.92, 0.50],
    [0.08, 0.90], [0.50, 0.90], [0.92, 0.90],
    [0.25, 0.30], [0.75, 0.30], [0.25, 0.70], [0.75, 0.70],
  ];
  const SAMPLES_PER_POINT = 3;
  const HIT_RADIUS_PX = 75;

  let plan = [];
  let idx = 0;
  let errors = []; // distances (px) between predicted gaze and clicked target
  let webgazerReady = false;

  // Build a randomized target plan in screen pixels.
  function buildPlan() {
    const w = window.innerWidth, h = window.innerHeight;
    const margin = 90;
    const pts = [];
    for (const [nx, ny] of NORMALIZED_POINTS) {
      const px = Math.round(margin + nx * (w - 2 * margin));
      const py = Math.round(margin + ny * (h - 2 * margin));
      for (let i = 0; i < SAMPLES_PER_POINT; i++) pts.push([px, py]);
    }
    // Fisher-Yates shuffle
    for (let i = pts.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pts[i], pts[j]] = [pts[j], pts[i]];
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
    updateHUD('CLICK EN EL PUNTO');
  }

  function onTargetClick(ev) {
    const [tx, ty] = plan[idx];
    const dx = ev.clientX - tx, dy = ev.clientY - ty;
    if (dx * dx + dy * dy > HIT_RADIUS_PX * HIT_RADIUS_PX) return;

    // Use the latest webgazer prediction to measure error
    if (window.webgazer && webgazerReady) {
      window.webgazer.getCurrentPrediction().then((pred) => {
        if (pred && typeof pred.x === 'number') {
          const ex = pred.x - tx, ey = pred.y - ty;
          errors.push(Math.hypot(ex, ey));
        }
      }).catch(() => {});
    }

    idx++;
    placeTarget();
  }

  function finish() {
    ui.target.classList.add('hidden');
    updateHUD('FINALIZADO');
    if (window.webgazer) {
      try { window.webgazer.showVideoPreview(false); } catch (_) {}
      try { window.webgazer.pause(); } catch (_) {}
    }
    // Hide arena, show result
    setTimeout(() => {
      ui.arena.classList.add('hidden');
      ui.result.classList.remove('hidden');
      ui.rPoints.textContent = String(idx);
      if (errors.length) {
        const rms = Math.sqrt(errors.reduce((a, b) => a + b * b, 0) / errors.length);
        ui.rRms.textContent = `${Math.round(rms)} px`;
      } else {
        ui.rRms.textContent = 'no disponible';
      }
    }, 250);
  }

  function startWebGazer() {
    if (typeof window.webgazer === 'undefined') {
      alert('No se ha podido cargar WebGazer.js (¿bloqueado por la red?). Revisa la consola.');
      return Promise.reject('webgazer missing');
    }
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
        // Hide WebGazer's default video feed for a cleaner demo
        try { window.webgazer.showVideoPreview(false); } catch (_) {}
        try { window.webgazer.showFaceOverlay(false); } catch (_) {}
        try { window.webgazer.showFaceFeedbackBox(false); } catch (_) {}
        try { window.webgazer.showPredictionPoints(false); } catch (_) {}
      });
  }

  function start() {
    ui.welcome.classList.add('hidden');
    ui.arena.classList.remove('hidden');
    updateHUD('INICIANDO WEBCAM…');

    startWebGazer().then(() => {
      plan = buildPlan();
      idx = 0;
      errors = [];
      placeTarget();
    }).catch((e) => {
      console.error(e);
      ui.hudState.textContent = 'ERROR DE CÁMARA';
    });
  }

  function restart() {
    ui.result.classList.add('hidden');
    ui.welcome.classList.remove('hidden');
    if (window.webgazer) {
      try { window.webgazer.end(); } catch (_) {}
    }
    webgazerReady = false;
  }

  // ── wire up ─────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    $('start').addEventListener('click', start);
    $('finish').addEventListener('click', finish);
    $('restart').addEventListener('click', restart);
    ui.target.addEventListener('click', onTargetClick);
    window.addEventListener('beforeunload', () => {
      if (window.webgazer) { try { window.webgazer.end(); } catch (_) {} }
    });
  });
})();
