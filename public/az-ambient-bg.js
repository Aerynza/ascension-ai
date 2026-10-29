/*
 * Aerynza ambient background filler ("royal ether").
 * A single fixed <canvas> behind all content: slowly drifting orb-like glows
 * (AP's floating orb), faint gold dust, soft constellation lines and a very
 * faint watermark of the real gold Aerynza logo artwork
 * (public/brand/aerynza-gold-logo-watermark.webp, extracted from the official PNG).
 * Works as a standalone drop-in on static sites (see data-* config below). Separate tuning for dark and light themes.
 *
 * - Pure canvas 2D, no libs. One requestAnimationFrame loop, throttled to ~30fps.
 * - prefers-reduced-motion: renders one static frame, no loop.
 * - Pauses when the tab is hidden. Particle count capped (lower on phones).
 * - pointer-events: none, z-index: -1, aria-hidden: never covers text.
 * - Off switches: <html data-az-ambient-bg="off">, localStorage
 *   'az-ambient-bg' = 'off', or window.AzAmbientBg.disable().
 */
(function () {
  'use strict';
  if (typeof window === 'undefined' || window.AzAmbientBg) return;
  var doc = document, root = doc.documentElement, KEY = 'az-ambient-bg';
  // Standalone drop-in config (static sites): attributes on the <script> tag.
  //   data-logo-src="/path/aerynza-gold-logo-watermark.webp"  (artwork URL)
  //   data-theme="dark|light|auto"  (auto = follow <html data-theme>, then prefers-color-scheme)
  var cfgEl = doc.currentScript || null;
  function cfg(n, d) { var v = cfgEl && cfgEl.getAttribute('data-' + n); return v || d; }
  var LOGO_SRC = cfg('logo-src', '/brand/aerynza-gold-logo-watermark.webp');
  var THEME_MODE = cfg('theme', 'auto');

  var THEMES = {
    dark: {
      glows: [
        { h: 258, s: 80, l: 62, a: 0.16 }, { h: 42, s: 75, l: 62, a: 0.10 },
        { h: 278, s: 70, l: 58, a: 0.12 }, { h: 228, s: 80, l: 60, a: 0.10 }
      ],
      dust: 'rgba(236, 206, 140, A)', dustAlpha: 0.55, line: 'rgba(226, 200, 143, A)', lineAlpha: 0.10,
      markAlpha: 0.26, markGlow: 0.14, markComp: 'lighter', comp: 'lighter'
    },
    light: {
      // Richer champagne + soft violet so empty light screens feel warm, not washed out.
      glows: [
        { h: 40, s: 78, l: 66, a: 0.30 }, { h: 268, s: 60, l: 76, a: 0.26 },
        { h: 32, s: 70, l: 72, a: 0.24 }, { h: 250, s: 55, l: 80, a: 0.20 }
      ],
      dust: 'rgba(176, 128, 46, A)', dustAlpha: 0.50, line: 'rgba(169, 116, 37, A)', lineAlpha: 0.12,
      markAlpha: 0.30, markGlow: 0.16, markComp: 'source-over', comp: 'source-over'
    }
  };

  var canvas, ctx, raf = 0, last = 0, w = 0, h = 0, dpr = 1, enabled = false;
  var dust = [], glows = [], mark = null, t0 = performance.now();
  var mqReduce = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };

  function isOff() {
    if (root.getAttribute('data-az-ambient-bg') === 'off') return true;
    try { return localStorage.getItem(KEY) === 'off'; } catch (_) { return false; }
  }
  var mqLight = window.matchMedia ? window.matchMedia('(prefers-color-scheme: light)') : { matches: false };
  function theme() {
    if (THEME_MODE === 'light' || THEME_MODE === 'dark') return THEMES[THEME_MODE];
    var a = root.getAttribute('data-theme');
    if (a === 'light' || a === 'dark') return THEMES[a];
    return THEMES[mqLight.matches ? 'light' : 'dark'];
  }
  function reduced() { return !!mqReduce.matches; }
  function rnd(a, b) { return a + Math.random() * (b - a); }

  function seed() {
    var small = Math.min(window.innerWidth, window.innerHeight) < 700;
    var n = Math.min(small ? 28 : 60, Math.round((w * h) / 22000));
    dust = [];
    for (var i = 0; i < n; i++) {
      dust.push({ x: rnd(0, w), y: rnd(0, h), r: rnd(0.5, 1.8), vx: rnd(-4, 4), vy: rnd(-7, -1.5), p: rnd(0, 6.28), tw: rnd(0.4, 1.2) });
    }
    glows = [];
    for (var j = 0; j < 4; j++) {
      glows.push({ cx: rnd(0.1, 0.9), cy: rnd(0.1, 0.9), rx: rnd(0.04, 0.10), ry: rnd(0.03, 0.08), sp: rnd(0.012, 0.03), ph: rnd(0, 6.28), r: rnd(0.28, 0.45) });
    }
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    w = window.innerWidth; h = window.innerHeight;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seed();
    if (reduced()) draw(performance.now(), 0);
  }

  function draw(now, dt) {
    var T = theme(), t = (now - t0) / 1000, m = Math.max(w, h);
    ctx.clearRect(0, 0, w, h);
    ctx.globalCompositeOperation = T.comp;
    // Orb glows (slow lissajous drift)
    for (var i = 0; i < glows.length; i++) {
      var g = glows[i], c = T.glows[i % T.glows.length];
      var x = (g.cx + Math.sin(t * g.sp + g.ph) * g.rx) * w;
      var y = (g.cy + Math.cos(t * g.sp * 0.8 + g.ph) * g.ry) * h;
      var R = g.r * m, grd = ctx.createRadialGradient(x, y, 0, x, y, R);
      grd.addColorStop(0, 'hsla(' + c.h + ',' + c.s + '%,' + c.l + '%,' + c.a + ')');
      grd.addColorStop(0.5, 'hsla(' + c.h + ',' + c.s + '%,' + c.l + '%,' + (c.a * 0.35) + ')');
      grd.addColorStop(1, 'hsla(' + c.h + ',' + c.s + '%,' + c.l + '%,0)');
      ctx.fillStyle = grd; ctx.fillRect(x - R, y - R, R * 2, R * 2);
    }
    ctx.globalCompositeOperation = 'source-over';
    // Faint mark watermark, gently breathing
    if (mark && mark.complete && mark.naturalWidth) {
      var s = Math.min(w, h) * 0.55, mx = w * 0.5, my = h * 0.46, br = 0.85 + 0.15 * Math.sin(t * 0.25);
      var tm = mark;
      ctx.globalCompositeOperation = T.markComp;
      // Warm halo behind the mark
      var hg = ctx.createRadialGradient(mx, my, 0, mx, my, s * 0.6);
      hg.addColorStop(0, 'rgba(240, 180, 70,' + (T.markGlow * br) + ')');
      hg.addColorStop(0.6, 'rgba(214, 166, 75,' + (T.markGlow * 0.35 * br) + ')');
      hg.addColorStop(1, 'rgba(214, 166, 75,0)');
      ctx.fillStyle = hg; ctx.fillRect(mx - s, my - s, s * 2, s * 2);
      ctx.globalAlpha = T.markAlpha * br;
      ctx.drawImage(tm, mx - s / 2, my - s / 2, s, s);
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = 'source-over';
    }
    // Dust + constellation lines
    var k = dt / 1000, LD = 120 * 120;
    for (var a = 0; a < dust.length; a++) {
      var d = dust[a];
      if (k) { d.x += d.vx * k; d.y += d.vy * k; if (d.y < -4) { d.y = h + 4; d.x = rnd(0, w); } if (d.x < -4) d.x = w + 4; else if (d.x > w + 4) d.x = -4; }
    }
    ctx.lineWidth = 0.6;
    for (var p = 0; p < dust.length; p++) {
      for (var q = p + 1; q < dust.length; q++) {
        var dx = dust[p].x - dust[q].x, dy = dust[p].y - dust[q].y, dd = dx * dx + dy * dy;
        if (dd < LD) {
          ctx.strokeStyle = T.line.replace('A', (T.lineAlpha * (1 - dd / LD)).toFixed(3));
          ctx.beginPath(); ctx.moveTo(dust[p].x, dust[p].y); ctx.lineTo(dust[q].x, dust[q].y); ctx.stroke();
        }
      }
    }
    for (var b = 0; b < dust.length; b++) {
      var e = dust[b], al = T.dustAlpha * (0.55 + 0.45 * Math.sin(t * e.tw + e.p));
      ctx.fillStyle = T.dust.replace('A', al.toFixed(3));
      ctx.beginPath(); ctx.arc(e.x, e.y, e.r, 0, 6.2832); ctx.fill();
    }
  }

  function loop(now) {
    raf = 0;
    if (!enabled || doc.hidden || reduced()) return;
    var dt = now - last;
    if (dt >= 33) { draw(now, Math.min(dt, 100)); last = now; }
    raf = requestAnimationFrame(loop);
  }
  function start() {
    if (!enabled) return;
    if (reduced()) { cancelAnimationFrame(raf); raf = 0; draw(performance.now(), 0); return; }
    if (!raf && !doc.hidden) { last = performance.now(); raf = requestAnimationFrame(loop); }
  }

  function mount() {
    if (enabled || isOff() || !doc.body) return;
    canvas = doc.createElement('canvas');
    canvas.className = 'az-ambient-bg';
    canvas.setAttribute('aria-hidden', 'true');
    canvas.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;z-index:-1;pointer-events:none;display:block;';
    ctx = canvas.getContext('2d');
    if (!ctx) return;
    doc.body.insertBefore(canvas, doc.body.firstChild);
    mark = new Image(); mark.src = LOGO_SRC;
    mark.onload = function () { if (reduced()) draw(performance.now(), 0); };
    enabled = true;
    resize();
    start();
  }
  function unmount() {
    enabled = false; cancelAnimationFrame(raf); raf = 0;
    if (canvas && canvas.parentNode) canvas.parentNode.removeChild(canvas);
    canvas = null;
  }

  var rt = 0;
  window.addEventListener('resize', function () { if (!enabled) return; clearTimeout(rt); rt = setTimeout(resize, 150); }, { passive: true });
  doc.addEventListener('visibilitychange', function () { if (doc.hidden) { cancelAnimationFrame(raf); raf = 0; } else start(); });
  if (mqReduce.addEventListener) mqReduce.addEventListener('change', start);
  if (window.MutationObserver) {
    new MutationObserver(function () {
      if (isOff()) unmount(); else if (!enabled) mount(); else if (reduced()) draw(performance.now(), 0);
    }).observe(root, { attributes: true, attributeFilter: ['data-theme', 'data-az-ambient-bg'] });
  }

  window.AzAmbientBg = {
    enable: function () { try { localStorage.removeItem(KEY); } catch (_) {} root.removeAttribute('data-az-ambient-bg'); mount(); },
    disable: function () { try { localStorage.setItem(KEY, 'off'); } catch (_) {} unmount(); },
    isEnabled: function () { return enabled; }
  };
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', mount); else mount();
})();
