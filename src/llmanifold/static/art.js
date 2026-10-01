// Decorative background: a Penrose tribar built from voxels. It holds, bursts apart, drifts and
// reassembles while schematic marks come and go. Canvas 2D, no dependencies.
(() => {
  const root = document.getElementById('manifold-art');
  const toggle = document.getElementById('art-toggle');
  const canvas = root && root.querySelector('canvas');
  const ctx = canvas && canvas.getContext && canvas.getContext('2d');
  if (!ctx || !toggle) {
    if (root) root.hidden = true;
    if (toggle) toggle.hidden = true;
    return;
  }

  const TAU = Math.PI * 2;
  const GREEN = '164,188,146', AMBER = '216,173,109';
  const FRAME_MS = 1000 / 30;
  const KEY = 'llmanifold.art.motion';
  // Amber bands travelling round the loop: two when idle, one more per request in flight.
  const BANDS = [0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875], IDLE_BANDS = 2;
  const band = BANDS.map((_, b) => (b < IDLE_BANDS ? 1 : 0));
  let load = 0;

  const voxel = (() => {
    const L = 5, H = 0.5, CYCLE = 18;
    // Three bars at right angles: along x, then y, then z. From the isometric view the top of the
    // z bar lands exactly under the first cube of the x bar, so the open path looks closed.
    const groups = [];
    for (let i = 0; i <= L; i++) groups.push([i, 0, 0]);
    for (let j = 1; j <= L; j++) groups.push([L, j, 0]);
    for (let k = 1; k < L; k++) groups.push([L, L, k]);
    let seed = 11;
    const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    const vox = [];
    groups.forEach((g, gi) => {
      for (let n = 0; n < 8; n++) {
        const th = rnd() * TAU, zz = rnd() * 2 - 1, rr = Math.sqrt(1 - zz * zz);
        vox.push({ g: gi, u: gi / groups.length, p: [g[0] + (n & 1) * H, g[1] + ((n >> 1) & 1) * H, g[2] + ((n >> 2) & 1) * H],
          dir: [rr * Math.cos(th), rr * Math.sin(th), zz], D: 2.5 + rnd() * 5.5, r: rnd() });
      }
    });
    const mid = [0, 1, 2].map((k) => groups.reduce((s, g) => s + g[k] + 0.5, 0) / groups.length);
    const FACES = [[[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]], [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]]];
    const BASE = [[78, 91, 66], [50, 60, 43], [30, 38, 27]], HOT = [[226, 186, 124], [170, 132, 80], [112, 90, 56]];
    const HEX = [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]];
    const clamp = (v) => Math.max(0, Math.min(1, v));
    return {
      draw(ctx, w, h, t) {
        const sc = Math.min(w, h) / 10.5, tau = t % CYCLE, cx = w * 0.47, cy = h / 2;
        const P = (x, y, z) => [(x - y) * 0.7071 * sc, -(2 * z - x - y) * 0.40825 * sc];
        const c0 = P(mid[0], mid[1], mid[2]);
        const head = t * 0.1;
        let assembled = true;
        const items = vox.map((v, vi) => {
          const out = clamp((tau - (8 + v.u * 1.2 + v.r * 0.3)) / 1.6), back = clamp((tau - (12 + v.u * 3 + v.r * 0.4)) / 1.8);
          const eo = 1 - (1 - out) ** 3, eb = back < 0.5 ? 4 * back ** 3 : 1 - ((-2 * back + 2) ** 3) / 2;
          const e = eo * (1 - eb);
          if (e > 0.001) assembled = false;
          const k = v.D * e * (1 + 0.07 * Math.max(0, tau - 8)), size = H * (1 - 0.3 * e), off = (H - size) / 2;
          const x = v.p[0] + v.dir[0] * k + off, y = v.p[1] + v.dir[1] * k + off, z = v.p[2] + v.dir[2] * k + off;
          let glow = 0;
          for (let b = 0; b < BANDS.length; b++) {
            if (band[b] < 0.01) continue;
            const d = (((head + BANDS[b] - v.u) % 1) + 1) % 1;
            glow = Math.max(glow, band[b] * Math.exp(-d * d / 0.006));
          }
          return { i: vi, g: v.g, x, y, z, size, e, glow, depth: x + y + z };
        });
        const drawVox = (it) => {
          ctx.globalAlpha = 1 - 0.4 * it.e;
          for (let f = 0; f < 3; f++) {
            ctx.beginPath();
            FACES[f].forEach((o, i) => {
              const q = P(it.x + o[0] * it.size, it.y + o[1] * it.size, it.z + o[2] * it.size);
              if (i) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]);
            });
            ctx.closePath();
            const c = BASE[f].map((b, i) => Math.round(b + (HOT[f][i] - b) * it.glow));
            ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
            ctx.fill();
            ctx.strokeStyle = `rgba(${GREEN},.32)`; ctx.lineWidth = 0.8; ctx.stroke();
          }
        };
        ctx.save();
        ctx.translate(cx, cy); ctx.rotate(t * 0.06); ctx.translate(-c0[0], -c0[1]);
        ctx.lineJoin = 'round';
        // drawing-board notes: fade in for the reassembly, linger a moment after it completes
        const note = tau < 1.5 ? 1 - tau / 1.5 : clamp((tau - 11.6) / 0.8), asm = tau >= 11.6 ? tau : CYCLE + tau;
        const n = groups.length, due = (g) => 12 + (g / n) * 3;
        const hex = (g, grow) => {
          ctx.beginPath();
          HEX.forEach((o, i) => {
            const q = P(g[0] + 0.5 + (o[0] - 0.5) * grow, g[1] + 0.5 + (o[1] - 0.5) * grow, g[2] + 0.5 + (o[2] - 0.5) * grow);
            if (i) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]);
          });
          ctx.closePath();
        };
        if (note > 0.01) {
          // dashed outline of where each block will go
          ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
          groups.forEach((g, gi) => {
            const a = note * 0.55 * (1 - clamp((asm - due(gi) - 1) / 1.2));
            if (a < 0.01) return;
            hex(g, 1); ctx.strokeStyle = `rgba(129,139,117,${a.toFixed(3)})`; ctx.stroke();
          });
          ctx.setLineDash([]);
        }
        items.sort((a, b) => a.depth - b.depth).forEach(drawVox);
        ctx.globalAlpha = 1;
        if (note > 0.01) {
          // a block seating: its outline flashes outward
          groups.forEach((g, gi) => {
            const f = (asm - due(gi) - 2) / 0.6;
            if (f <= 0 || f >= 1) return;
            hex(g, 1 + f * 0.7); ctx.strokeStyle = `rgba(${AMBER},${(note * (1 - f) * 0.9).toFixed(3)})`; ctx.lineWidth = 1.2; ctx.stroke();
          });
        }
        if (assembled) {
          // close the loop: the first cube of the x bar sits over the top of the z bar,
          // and the second is redrawn except where the third really hides it
          items.filter((it) => it.g === 0).forEach(drawVox);
          ctx.save();
          ctx.beginPath(); ctx.rect(-1e4, -1e4, 2e4, 2e4);
          HEX.forEach((o, i) => { const q = P(groups[2][0] + o[0], groups[2][1] + o[1], groups[2][2] + o[2]); if (i) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); });
          ctx.closePath(); ctx.clip('evenodd');
          items.filter((it) => it.g === 1).forEach(drawVox);
          ctx.restore();
        }
        ctx.restore();
        ctx.globalAlpha = 1;
        const ang = t * 0.06, ca = Math.cos(ang), sa = Math.sin(ang);
        const S = (x, y, z) => {
          const q = P(x, y, z), dx = q[0] - c0[0], dy = q[1] - c0[1];
          return [cx + dx * ca - dy * sa, cy + dx * sa + dy * ca];
        };
        const seg = (a, b) => { ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); };
        ctx.font = "9px 'IBM Plex Mono', ui-monospace, monospace"; ctx.textBaseline = 'middle'; ctx.lineWidth = 1; ctx.lineCap = 'butt';

        // ---- stray marks: one is born about once a second and stays 6 to 10 seconds, pinned to a voxel or to empty space,
        // whether the figure is whole or in pieces. Everything comes from a hash of its slot number.
        const hash = (k, j) => { const x = Math.sin(k * 127.1 + j * 311.7) * 43758.5453; return x - Math.floor(x); };
        const SYM = ['∂', '∇', 'λ', 'φ', 'ψ', 'Ω', 'ξ', '∴', '⊥', '∥', '≈', '◊', 'Δ', 'τ', 'η'];
        const code = (k) => {
          const s = SYM[Math.floor(hash(k, 1) * SYM.length)], hex = Math.floor(hash(k, 3) * 4096).toString(16).toUpperCase().padStart(3, '0');
          switch (Math.floor(hash(k, 2) * 4)) {
            case 0: return `${s}·${hex}`;
            case 1: return `${s} ${(hash(k, 4) * 9.99).toFixed(3)}`;
            case 2: return `${hex.slice(0, 2)}:${s}:${Math.floor(hash(k, 5) * 90 + 10)}`;
            default: return `${s}${s} ${hex}`;
          }
        };
        const scr = [];
        for (const it of items) scr[it.i] = S(it.x + it.size / 2, it.y + it.size / 2, it.z + it.size / 2);
        const SLOT = 1.1, now = Math.floor(t / SLOT);
        for (let k = now - 10; k <= now; k++) {
          const life = 6 + hash(k, 6) * 4, age = t - k * SLOT;
          if (age < 0 || age >= life) continue;
          const al = Math.min(1, age / 0.5) * Math.min(1, (life - age) / 1.5), grow = Math.min(1, age / 0.8);
          const v = scr[Math.floor(hash(k, 8) * scr.length)], rgb = hash(k, 9) < 0.3 ? AMBER : '146,153,140';
          const col = (a) => `rgba(${rgb},${(a * al).toFixed(3)})`;
          const free = [cx + (hash(k, 11) - 0.5) * sc * 9, cy + (hash(k, 12) - 0.5) * sc * 7];
          ctx.strokeStyle = col(0.8); ctx.fillStyle = col(0.9); ctx.textAlign = 'left';
          switch (Math.floor(hash(k, 7) * 6)) {
            case 0: {                                   // tag on a leader
              const th = hash(k, 10) * TAU, r = (26 + hash(k, 13) * 48) * grow, side = Math.cos(th) >= 0 ? 1 : -1;
              const e1 = [v[0] + Math.cos(th) * (6 + r), v[1] + Math.sin(th) * (6 + r)], e2 = [e1[0] + side * 16 * grow, e1[1]];
              ctx.fillRect(v[0] - 1.5, v[1] - 1.5, 3, 3);
              seg([v[0] + Math.cos(th) * 6, v[1] + Math.sin(th) * 6], e1); seg(e1, e2);
              if (grow >= 1) { ctx.textAlign = side > 0 ? 'left' : 'right'; ctx.fillText(code(k), e2[0] + side * 5, e2[1]); }
              break;
            }
            case 1: {                                   // broken ring with graduations
              const r = 13 + hash(k, 13) * 15, a0 = hash(k, 10) * TAU + t * 0.7;
              ctx.beginPath(); ctx.arc(v[0], v[1], r, a0, a0 + 4.4 * grow); ctx.stroke();
              ctx.beginPath();
              for (let n = 0; n < 9 * grow; n++) {
                const a = a0 + n * 0.5, r1 = r + 3, r2 = r + (n % 3 ? 5 : 8);
                ctx.moveTo(v[0] + Math.cos(a) * r1, v[1] + Math.sin(a) * r1); ctx.lineTo(v[0] + Math.cos(a) * r2, v[1] + Math.sin(a) * r2);
              }
              ctx.stroke();
              break;
            }
            case 2: {                                   // corner brackets closing on a voxel
              const b = (11 + hash(k, 13) * 8) * (1.7 - 0.7 * grow);
              ctx.beginPath();
              for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
                ctx.moveTo(v[0] + sx * b, v[1] + sy * (b - 5)); ctx.lineTo(v[0] + sx * b, v[1] + sy * b); ctx.lineTo(v[0] + sx * (b - 5), v[1] + sy * b);
              }
              ctx.stroke();
              if (grow >= 1) ctx.fillText(code(k), v[0] + b + 6, v[1] + b);
              break;
            }
            case 3: {                                   // dotted tie between two voxels
              const o = scr[Math.floor(hash(k, 14) * scr.length)], e = [v[0] + (o[0] - v[0]) * grow, v[1] + (o[1] - v[1]) * grow];
              ctx.setLineDash([1, 4]); seg(v, e); ctx.setLineDash([]);
              const m = [(v[0] + o[0]) / 2, (v[1] + o[1]) / 2];
              if (grow >= 1) {
                ctx.beginPath(); ctx.moveTo(m[0], m[1] - 4); ctx.lineTo(m[0] + 4, m[1]); ctx.lineTo(m[0], m[1] + 4); ctx.lineTo(m[0] - 4, m[1]); ctx.closePath(); ctx.stroke();
                ctx.fillText(code(k), m[0] + 9, m[1]);
              }
              break;
            }
            case 4: {                                   // a length of ruler adrift
              const a = [0, Math.PI / 6, -Math.PI / 6, Math.PI / 2][Math.floor(hash(k, 10) * 4)], len = (60 + hash(k, 13) * 90) * grow;
              const ux = Math.cos(a), uy = Math.sin(a);
              seg(free, [free[0] + ux * len, free[1] + uy * len]);
              ctx.beginPath();
              for (let d = 0, n = 0; d <= len; d += 8, n++) {
                const tk = n % 5 ? 3 : 7;
                ctx.moveTo(free[0] + ux * d, free[1] + uy * d); ctx.lineTo(free[0] + ux * d - uy * tk, free[1] + uy * d + ux * tk);
              }
              ctx.stroke();
              if (grow >= 1) ctx.fillText(code(k), free[0] + ux * len + 8, free[1] + uy * len);
              break;
            }
            default: {                                  // registration cross
              const r = 5 + 6 * grow;
              seg([free[0] - r, free[1]], [free[0] + r, free[1]]); seg([free[0], free[1] - r], [free[0], free[1] + r]);
              ctx.beginPath(); ctx.arc(free[0], free[1], 4, 0, TAU); ctx.stroke();
              if (grow >= 1) ctx.fillText(code(k), free[0] + 14, free[1] + 10);
            }
          }
        }
        if (note <= 0.01) return;

        // ---- marks that belong to the reassembly
        const ink = (a) => `rgba(146,153,140,${(a * note).toFixed(3)})`, hot = (a) => `rgba(${AMBER},${(a * note).toFixed(3)})`;
        // a graduated rule along each bar, drawn out as the bar goes together
        const dims = [
          { a: [0, -0.9, 1], b: [L + 1, -0.9, 1], g0: 0, g1: L + 1, units: L + 1, label: '∥ α' },
          { a: [L + 1, 0, -0.9], b: [L + 1, L + 1, -0.9], g0: L + 1, g1: 2 * L + 1, units: L + 1, label: '∥ β' },
          { a: [L, L + 1.9, 0], b: [L, L + 1.9, L], g0: 2 * L + 1, g1: n, units: L, label: '∥ γ′' },
        ];
        for (const d of dims) {
          const f = clamp((asm - due(d.g0)) / (due(d.g1) - due(d.g0) + 2.2));
          if (f <= 0) continue;
          const a = S(...d.a), full = S(...d.b), b = [a[0] + (full[0] - a[0]) * f, a[1] + (full[1] - a[1]) * f];
          const len = Math.hypot(full[0] - a[0], full[1] - a[1]) || 1;
          let nx = -(full[1] - a[1]) / len, ny = (full[0] - a[0]) / len;
          const mx = (a[0] + full[0]) / 2, my = (a[1] + full[1]) / 2;
          if (nx * (mx - cx) + ny * (my - cy) < 0) { nx = -nx; ny = -ny; }
          ctx.strokeStyle = ink(0.75);
          seg(a, b);
          ctx.beginPath();
          for (let u = 0; u <= d.units * 2 * f + 0.001; u++) {
            const px = a[0] + (full[0] - a[0]) * (u / (d.units * 2)), py = a[1] + (full[1] - a[1]) * (u / (d.units * 2)), tk = u % 2 ? 3 : 6;
            ctx.moveTo(px, py); ctx.lineTo(px + nx * tk, py + ny * tk);
          }
          ctx.stroke();
          if (f > 0.5) {
            ctx.fillStyle = ink(Math.min(1, (f - 0.5) * 3)); ctx.textAlign = 'center';
            ctx.fillText(d.label, mx + nx * 18, my + ny * 18);
          }
        }
        // each corner gets an arc and a leader that ends in a small mark
        [{ p: [0.5, 0.5, 1], at: 12.3 }, { p: [L + 0.5, 0.5, 1], at: due(L) + 2 }, { p: [L + 0.5, L + 0.5, 1], at: due(2 * L) + 2 }].forEach((c, ci) => {
          const f = clamp((asm - c.at) / 0.6);
          if (f <= 0) return;
          const q = S(...c.p), dl = Math.hypot(q[0] - cx, q[1] - cy) || 1, ux = (q[0] - cx) / dl, uy = (q[1] - cy) / dl;
          const base = Math.atan2(uy, ux), reach = sc * 1.7 * f, e1 = [q[0] + ux * (12 + reach), q[1] + uy * (12 + reach)];
          ctx.strokeStyle = hot(0.85);
          ctx.beginPath(); ctx.arc(q[0], q[1], 12, base - 0.8 * f, base + 0.8 * f); ctx.stroke();
          ctx.beginPath(); ctx.arc(q[0], q[1], 17, base + Math.PI - 0.5 * f, base + Math.PI + 0.5 * f); ctx.stroke();
          seg([q[0] + ux * 12, q[1] + uy * 12], e1);
          if (f >= 1) {
            ctx.strokeRect(e1[0] - 3, e1[1] - 3, 6, 6);
            ctx.fillStyle = hot(0.95);
            for (let n = 0; n <= ci; n++) ctx.fillRect(e1[0] + ux * (9 + n * 5) - 1, e1[1] + uy * (9 + n * 5) - 1, 2.5, 2.5);
          }
        });
        // tally: one cell per block, filled as it seats; then the verdict
        const rx = cx - sc * 4.6, ry = cy + sc * 4.3;
        for (let g = 0; g < n; g++) {
          const seated = asm > due(g) + 2.2;
          ctx.strokeStyle = ink(0.7); ctx.strokeRect(rx + g * 8 + 0.5, ry - 3.5, 5, 5);
          if (seated) { ctx.fillStyle = ink(0.85); ctx.fillRect(rx + g * 8 + 0.5, ry - 3.5, 5, 5); }
        }
        const closed = clamp((asm - 17.2) / 0.4);
        if (closed > 0) {
          ctx.textAlign = 'left'; ctx.fillStyle = hot(0.95 * closed);
          ctx.fillText('Σ ≠ △  ∴  ε ∥ (1,1,1)', rx, ry + 15);
        }
      },
    };
  })();

  let cssW = 0, cssH = 0, dpr = 1;
  function resize() {
    cssW = root.clientWidth; cssH = root.clientHeight;
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.max(1, Math.round(cssW * dpr)), h = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }
  function render() {
    if (!cssW || !cssH) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    voxel.draw(ctx, cssW, cssH, clock);
  }

  // ---- loop and motion preference
  let running = false, raf = 0, last = 0, clock = 4;
  function frame(now) {
    raf = 0;
    if (!running) return;
    raf = requestAnimationFrame(frame);
    if (now - last < FRAME_MS) return;
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    clock += dt;
    for (let b = 0; b < band.length; b++) band[b] += ((b < IDLE_BANDS + load ? 1 : 0) - band[b]) * Math.min(1, dt * 1.5);
    render();
  }
  function start() {
    if (running || document.hidden) return;
    running = true; last = performance.now();
    raf = requestAnimationFrame(frame);
  }
  function stop() {
    running = false;
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
  }

  let stored = null;
  try { stored = localStorage.getItem(KEY); } catch (e) { /* storage unavailable */ }
  const small = () => window.matchMedia('(max-width: 700px)').matches;
  const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let wantMotion = stored === 'on' ? true : stored === 'off' ? false : !(reduced() || small());

  function apply() {
    toggle.textContent = wantMotion ? 'Pause artwork' : 'Resume artwork';
    toggle.setAttribute('aria-pressed', String(!wantMotion));
    if (wantMotion) start();
    else { stop(); resize(); render(); }
  }

  toggle.addEventListener('click', () => {
    wantMotion = !wantMotion;
    try { localStorage.setItem(KEY, wantMotion ? 'on' : 'off'); } catch (e) { /* ignore */ }
    apply();
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stop();
    else if (wantMotion) start();
  });
  if (window.ResizeObserver) {
    new ResizeObserver(() => { resize(); render(); }).observe(root);
  } else {
    window.addEventListener('resize', () => { resize(); render(); });
  }

  window.manifoldArt = {
    setLoad(n) { load = Math.max(0, Math.min(BANDS.length - IDLE_BANDS, Math.round(Number(n) || 0))); },
  };

  resize();
  toggle.hidden = false;
  apply();
})();
