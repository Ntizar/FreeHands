/* ──────────────────────────────────────────────────────────────────────
   FreeHands · WebGazer loader with CDN fallback chain.
   Tries several mirrors so the demo keeps working even if one CDN
   is blocked / down (e.g. webgazer.cs.brown.edu sometimes is).
   On full failure, sets window.__webgazerLoadError so demo.js can
   show a nicer message than "no script".
   ────────────────────────────────────────────────────────────────────── */
(function () {
  'use strict';
  if (window.webgazer) return;

  const SOURCES = [
    'https://cdn.jsdelivr.net/npm/webgazer@3.3.0/dist/webgazer.min.js',
    'https://webgazer.cs.brown.edu/webgazer.js',
  ];

  function tryLoad(idx) {
    if (idx >= SOURCES.length) {
      window.__webgazerLoadError =
        'No se ha podido descargar WebGazer.js desde ningún CDN (jsDelivr, Brown). ' +
        'Causas habituales: bloqueador de anuncios, red corporativa filtrando CDN, o sin conexión.';
      console.error('[FreeHands] WebGazer load failed from all sources.');
      return;
    }
    const src = SOURCES[idx];
    const s = document.createElement('script');
    s.src = src;
    s.async = false;
    s.defer = true;
    s.onload = () => { console.log('[FreeHands] WebGazer loaded from', src); };
    s.onerror = () => {
      console.warn('[FreeHands] WebGazer source failed:', src);
      s.remove();
      tryLoad(idx + 1);
    };
    document.head.appendChild(s);
  }

  tryLoad(0);
})();
