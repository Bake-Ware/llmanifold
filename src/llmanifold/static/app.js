// llmanifold admin dashboard. Plain JS, polls the admin API.
// Words used on screen: a "model" is something llmanifold can send requests to (an
// endpoint in the config); an "alias flow" is a name clients ask for (a model in the config).
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const num = (n) => (n == null ? '–' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e4 ? (n / 1e3).toFixed(1) + 'k' : n.toLocaleString());
  const secs = (ms) => (ms == null ? '–' : ms < 1000 ? ms + ' ms' : (ms / 1000).toFixed(ms < 10000 ? 2 : 1) + ' s');
  const ago = (ts) => {
    if (!ts) return 'never';
    const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 60) return Math.round(s) + 's ago';
    if (s < 3600) return Math.round(s / 60) + 'm ago';
    if (s < 86400) return Math.round(s / 3600) + 'h ago';
    return Math.round(s / 86400) + 'd ago';
  };
  const clock = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const when = (ts) => {
    const d = new Date(ts * 1000);
    return Date.now() / 1000 - ts > 86400
      ? d.toLocaleString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' }) : clock(ts);
  };
  const enc = encodeURIComponent;

  let status = null, who = null, cfg = null, view = 'overview', lastOk = 0;
  const canEdit = () => !!(who && who.human && status && status.editable);

  async function api(path, opts = {}) {
    const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
    let body = null;
    try { body = await r.json(); } catch (_) { /* empty */ }
    if (!r.ok) throw new Error((body && (body.error || body.message)) || `HTTP ${r.status}`);
    return body;
  }
  const post = (path, body, method = 'POST') => api(path, { method, body: JSON.stringify(body || {}) });

  function banner(msg) {
    const b = $('#banner');
    b.hidden = !msg;
    b.textContent = msg || '';
  }

  function confirmAction(title, text, okLabel = 'Confirm', danger = false) {
    const d = $('#confirm');
    $('#confirm-title').textContent = title;
    $('#confirm-text').textContent = text;
    $('#confirm-ok').textContent = okLabel;
    $('#confirm-ok').className = danger ? 'danger' : '';
    return new Promise((res) => {
      d.addEventListener('close', () => res(d.returnValue === 'ok'), { once: true });
      d.returnValue = '';
      d.showModal();
    });
  }
  function formError(form, msg) {
    const p = $('.form-error', form);
    p.hidden = !msg;
    p.textContent = msg || '';
  }
  document.addEventListener('click', (ev) => {
    const c = ev.target.closest('[data-close]');
    if (c) c.closest('dialog').close();
  });

  // ------------------------------------------------------------- navigation
  const VIEWS = {
    overview: ['Overview', 'Alias flows, their models, and what each lane is doing'],
    requests: ['Requests', 'How much traffic, who served it, and how fast'],
    access: ['Models and tokens', 'What llmanifold can send requests to, and who may use it'],
  };
  function show(v) {
    view = VIEWS[v] ? v : 'overview';
    $$('.view').forEach((el) => (el.hidden = el.id !== 'view-' + view));
    $$('.side nav a').forEach((a) => a.classList.toggle('active', a.dataset.view === view));
    $('#title').textContent = VIEWS[view][0];
    $('#subtitle').textContent = VIEWS[view][1];
    hideTip();
    refreshView(true);
  }
  window.addEventListener('hashchange', () => show(location.hash.slice(1)));

  // ------------------------------------------------------------- overview
  function laneState(e) {
    if (e.draining) return 'paused';
    if (!e.healthy) return 'down';
    return e.load > 0 ? 'busy' : 'idle';
  }
  function laneNow(e, st) {
    if (st === 'paused') return e.inflight ? 'paused: finishing what it already has' : 'paused: gets no new requests';
    if (st === 'down') return e.last_error || 'not answering';
    if (e.current && e.current.length) return e.current.join(', ');
    if (e.probe_busy > e.inflight) return 'busy: serving a client that bypassed llmanifold';
    return 'idle';
  }
  function laneHtml(e, flow, fallback) {
    const st = laneState(e);
    const tags = [e.metered ? 'paid' : '', fallback ? 'fallback' : '', e.dialect === 'anthropic' ? 'anthropic api' : '']
      .filter(Boolean).join(' · ');
    const pause = e.draining
      ? `<button class="small" data-resume="${esc(e.name)}">Resume</button>`
      : `<button class="small secondary" data-pause="${esc(e.name)}">Pause</button>`;
    const remove = `<button class="small icon edit-only" data-remove-member="${esc(e.name)}" data-flow="${esc(flow)}"
      aria-label="Remove ${esc(e.name)} from ${esc(flow)}" title="Remove from this flow">✕</button>`;
    return `<div class="lane ${st} ${fallback ? 'fallback' : ''}" data-lane="${esc(e.name)}">
      <span class="st" title="${st}"></span>
      <span class="ln">${esc(e.name)}<small>${esc(tags || `${e.inflight}/${e.max_concurrency} in use`)}</small></span>
      <span class="now" title="${esc(laneNow(e, st))}">${esc(laneNow(e, st))}</span>
      <span class="speed">${e.tps ? e.tps + ' t/s' : '–'}<small>${e.ttft ? 'first ' + e.ttft + 's' : num(e.requests) + ' served'}</small></span>
      <span class="lane-act">${pause}${remove}</span></div>`;
  }
  function drawPipes(row) {
    const cell = $('.pipes-cell', row), svg = $('svg.pipes', row), lanes = $('.lanes', row), intake = $('.intake', row);
    if (!cell || !svg || !lanes || !intake) return;
    const box = cell.getBoundingClientRect(), h = box.height, w = box.width;
    if (!h || !w) return;
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    const ib = $('.name', intake).getBoundingClientRect(), y0 = ib.top - box.top + ib.height / 2;
    svg.innerHTML = $$('.lane', lanes).map((ln) => {
      const r = ln.getBoundingClientRect(), y = r.top - box.top + r.height / 2;
      const cls = ['busy', 'fallback', 'down', 'paused'].filter((c) => ln.classList.contains(c))
        .map((c) => (c === 'fallback' ? 'fb' : c)).join(' ');
      return `<path class="${cls}" d="M0 ${y0} C ${w / 2} ${y0}, ${w / 2} ${y}, ${w} ${y}"/>`;
    }).join('');
  }
  function renderOverview() {
    const s = status, c = s.counters;
    const done = c.ok + c.errors;
    const tiles = [
      [num(c.requests), 'requests since start', ''],
      [done ? Math.round((100 * c.ok) / done) + '%' : '–', 'succeeded', 'good'],
      [num(c.fallbacks), 'fell back', ''],
      [num(c.metered), 'served by paid APIs', ''],
      [String(s.queue.length), 'waiting now', s.queue.length ? '' : 'good'],
    ];
    $('#tiles').innerHTML = tiles.map(([k, l, cls]) => `<div class="tile"><div class="k ${cls}">${k}</div><div class="l">${l}</div></div>`).join('');
    const byName = Object.fromEntries(s.endpoints.map((e) => [e.name, e]));
    $('#manifold').innerHTML = s.models.map((m) => {
      const pool = m.pool.map((n) => byName[n]).filter(Boolean);
      const fb = m.fallback.map((n) => byName[n]).filter(Boolean);
      const chips = [
        m.default ? '<span class="chip">default</span>' : '',
        `<span class="chip ${m.auth === 'token' ? 'amber' : ''}">${m.auth === 'token' ? 'token required' : 'open'}</span>`,
        m.context ? `<span class="chip">${num(m.context)} ctx</span>` : '',
        m.queued ? `<span class="chip amber">${m.queued} waiting</span>` : '',
        m.warning ? `<span class="chip warn" title="${esc(m.warning)}">paid for anyone</span>` : '',
      ].join('');
      return `<div class="model-row" data-flow="${esc(m.name)}">
        <div class="intake"><span class="name">${esc(m.name)}</span>
          ${m.aliases.length ? `<span class="aliases">also ${esc(m.aliases.join(', '))}</span>` : ''}
          <div class="meta">${chips}</div>
          <div class="flow-actions edit-only">
            <button class="small secondary" data-add-member="${esc(m.name)}">Add model</button>
            <button class="small ghost" data-delete-flow="${esc(m.name)}">Delete flow</button>
          </div></div>
        <div class="pipes-cell"><svg class="pipes" aria-hidden="true"></svg></div>
        <div class="lanes">${pool.map((e) => laneHtml(e, m.name, false)).join('')}${fb.map((e) => laneHtml(e, m.name, true)).join('')}
          ${!pool.length && !fb.length ? '<div class="empty">No models in this flow.</div>' : ''}</div>
      </div>`;
    }).join('') || `<div class="panel empty">No alias flows yet. ${canEdit() ? 'Create one with “New alias flow”.' : 'A person signed in through the admin site can create one.'}</div>`;
    $$('.model-row').forEach(drawPipes);
    $('#queue').innerHTML = s.queue.length
      ? s.queue.map((q) => `<div class="q"><span>${esc(q.model)}</span><span>${esc(q.priority)}</span><span>${q.waiting_s}s</span></div>`).join('')
      : '<div class="empty">Nothing is waiting. Every request found a free lane.</div>';
  }

  // ------------------------------------------------------------- charts
  // Categorical slots validated (dataviz validator, dark, surface #151914): fixed order, never cycled.
  const SLOTS = ['#c08936', '#269c84', '#5a7fd9', '#d8657a', '#9a74dc', '#8aa02f'];
  const OTHER = '#6f7766';
  const FAIL = 'var(--bad)';
  const tip = $('#tooltip');
  function hideTip() { tip.hidden = true; }
  function showTip(x, y, html) {
    tip.innerHTML = html;
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let left = x + 14, top = y + 14;
    if (left + r.width > innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > innerHeight - 8) top = y - r.height - 14;
    tip.style.left = Math.max(8, left) + 'px';
    tip.style.top = Math.max(8, top) + 'px';
  }
  // clean axis ticks: a 1/2/5 step, about four intervals, whole numbers for counts
  function niceScale(v, integer = false) {
    v = v > 0 ? v : 1;
    let step = v / 4;
    const p = Math.pow(10, Math.floor(Math.log10(step))), f = step / p;
    step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
    if (integer) step = Math.max(1, Math.round(step));
    const max = Math.ceil(v / step) * step;
    const ticks = [];
    for (let t = 0; t <= max + step / 2; t += step) ticks.push(t);
    return { max, ticks };
  }
  const tickFmt = (v) => (v >= 1000 ? (v / 1000).toLocaleString() + 'k' : (Math.round(v * 100) / 100).toLocaleString());

  function seriesColors(names) {
    // color follows the model (its position in the config), never its rank in this range
    const order = hist.endpoints;
    return Object.fromEntries(names.map((n) => {
      const i = order.indexOf(n);
      return [n, i >= 0 && i < SLOTS.length ? SLOTS[i] : OTHER];
    }));
  }
  function legendHtml(series, kind) {
    if (series.length < 2) return '';
    return `<ul class="legend">${series.map((s) => `<li><span class="key ${kind}" style="--c:${s.color}"></span>${esc(s.name)}</li>`).join('')}</ul>`;
  }
  function xLabels(buckets, plotW, left) {
    const n = buckets.length, want = Math.max(2, Math.floor(plotW / 110)), step = Math.max(1, Math.ceil(n / want));
    const band = plotW / n, out = [];
    const fmt = hist.bucket >= 3600
      ? (t) => new Date(t * 1000).toLocaleString([], { weekday: 'short', hour: '2-digit' })
      : (t) => new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    for (let i = 0; i < n; i += step) out.push(`<text x="${left + band * i + band / 2}" y="0" text-anchor="middle">${esc(fmt(buckets[i].t))}</text>`);
    return out.join('');
  }
  function bucketTitle(b) {
    const a = new Date(b.t * 1000), z = new Date((b.t + hist.bucket) * 1000);
    const o = hist.bucket >= 3600 ? { weekday: 'short', hour: '2-digit', minute: '2-digit' } : { hour: '2-digit', minute: '2-digit' };
    return `${a.toLocaleString([], o)} – ${z.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
  }
  function roundTop(x, y, w, h, r) {
    r = Math.min(r, h, w / 2);
    return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
  }

  // stacked columns of successful requests per model, with a failures strip under the axis
  function drawColumns(host, card) {
    const W = host.clientWidth || 600, H = 210, L = 44, R = 10, T = 8, B = 22, STRIP = 24;
    const plotW = W - L - R, plotH = H - T - B, n = hist.buckets.length, band = plotW / n;
    const bw = Math.max(1, Math.min(24, band - 2));
    const names = card.series.map((s) => s.name);
    const totals = hist.buckets.map((b) => names.reduce((a, nm) => a + (b.requests[nm] || 0), 0));
    const { max, ticks } = niceScale(Math.max(...totals, 1), true);
    const y = (v) => T + plotH - (v / max) * plotH;
    const grid = ticks.map((t) => `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}"/>
      <text class="tick" x="${L - 6}" y="${y(t) + 3}" text-anchor="end">${tickFmt(t)}</text>`).join('');
    let marks = '';
    hist.buckets.forEach((b, i) => {
      const x = L + band * i + (band - bw) / 2;
      let base = y(0);
      const segs = card.series.map((s) => [s, b.requests[s.name] || 0]).filter(([, v]) => v > 0);
      segs.forEach(([s, v], k) => {
        const h = (v / max) * plotH, top = base - h;
        const gap = k > 0 ? 2 : 0;                              // 2px surface gap between stacked segments
        const hh = Math.max(0.5, h - gap);
        marks += k === segs.length - 1
          ? `<path d="${roundTop(x, top, bw, hh, 4)}" fill="${s.color}"/>`
          : `<rect x="${x}" y="${top}" width="${bw}" height="${hh}" fill="${s.color}"/>`;
        base = top;
      });
    });
    const maxErr = Math.max(...hist.buckets.map((b) => b.errors), 0);
    const sy = H + 6;
    let strip = '';
    if (maxErr) {
      hist.buckets.forEach((b, i) => {
        if (!b.errors) return;
        const h = Math.max(3, (b.errors / maxErr) * (STRIP - 8));
        strip += `<path d="${roundTop(L + band * i + (band - bw) / 2, sy + STRIP - 4 - h, bw, h, 2)}" fill="${FAIL}"/>`;
      });
    }
    const failLabel = `<text class="tick" x="${L - 6}" y="${sy + STRIP - 6}" text-anchor="end">failed</text>
      <line class="axis" x1="${L}" x2="${W - R}" y1="${sy + STRIP - 4}" y2="${sy + STRIP - 4}"/>
      ${maxErr ? '' : `<text class="tick" x="${L + 4}" y="${sy + STRIP - 8}">none</text>`}`;
    host.innerHTML = `<svg viewBox="0 0 ${W} ${H + STRIP + 10}" width="${W}" height="${H + STRIP + 10}" role="img" aria-label="${esc(card.title)}">
      ${grid}<line class="axis" x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}"/>
      <g class="marks">${marks}</g><g class="xl" transform="translate(0 ${H - 6})">${xLabels(hist.buckets, plotW, L)}</g>
      ${failLabel}${strip}<rect class="hover-band" x="0" y="${T}" width="0" height="${H + STRIP - T}" visibility="hidden"/>
      <rect class="hit" x="${L}" y="${T}" width="${plotW}" height="${H + STRIP - T}"/></svg>`;
    hover(host, L, band, (i, svg) => {
      const hb = $('.hover-band', svg);
      hb.setAttribute('x', L + band * i);
      hb.setAttribute('width', band);
      hb.setAttribute('visibility', 'visible');
      const b = hist.buckets[i];
      const rows = card.series.filter((s) => b.requests[s.name]).map((s) =>
        `<div class="tr"><span class="key bar" style="--c:${s.color}"></span><b>${num(b.requests[s.name])}</b><span>${esc(s.name)}</span></div>`);
      if (b.errors) rows.push(`<div class="tr"><span class="key bar" style="--c:${FAIL}"></span><b>${num(b.errors)}</b><span>failed</span></div>`);
      return `<div class="th">${esc(bucketTitle(b))}</div>${rows.join('') || '<div class="tr muted">no requests</div>'}`;
    }, (svg) => $('.hover-band', svg).setAttribute('visibility', 'hidden'));
  }

  // one line per series over the buckets; null values leave gaps
  function drawLines(host, card) {
    const W = host.clientWidth || 600, H = 190, L = 44, R = 10, T = 8, B = 22;
    const plotW = W - L - R, plotH = H - T - B, n = hist.buckets.length, band = plotW / n;
    const all = card.series.flatMap((s) => s.values.filter((v) => v != null));
    const { max, ticks } = niceScale(Math.max(...all, 0) * 1.05);
    const x = (i) => L + band * i + band / 2, y = (v) => T + plotH - (v / max) * plotH;
    const grid = ticks.map((t) => `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}"/>
      <text class="tick" x="${L - 6}" y="${y(t) + 3}" text-anchor="end">${card.tick(t)}</text>`).join('');
    const lines = card.series.map((s) => {
      // intervals with no traffic are skipped, not drawn as zero: quiet isn't slow
      let d = '';
      const pts = s.values.map((v, i) => [v, i]).filter(([v]) => v != null);
      pts.forEach(([v, i], k) => { d += `${k ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`; });
      const dotAt = new Set(pts.length ? [pts[pts.length - 1][1]] : []);     // end dot (and the only dot if one point)
      const dots = [...dotAt].map((i) => `<circle cx="${x(i)}" cy="${y(s.values[i])}" r="4" fill="${s.color}" class="ring"/>`).join('');
      return `<path d="${d}" stroke="${s.color}" class="line"/>${dots}`;
    }).join('');
    host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(card.title)}">
      ${grid}<line class="axis" x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}"/>
      <g class="marks">${lines}</g><g class="xl" transform="translate(0 ${H - 6})">${xLabels(hist.buckets, plotW, L)}</g>
      <line class="crosshair" x1="0" x2="0" y1="${T}" y2="${T + plotH}" visibility="hidden"/><g class="hover-dots"></g>
      <rect class="hit" x="${L}" y="${T}" width="${plotW}" height="${plotH}"/></svg>`;
    hover(host, L, band, (i, svg) => {
      const ch = $('.crosshair', svg);
      ch.setAttribute('x1', x(i)); ch.setAttribute('x2', x(i)); ch.setAttribute('visibility', 'visible');
      $('.hover-dots', svg).innerHTML = card.series.filter((s) => s.values[i] != null)
        .map((s) => `<circle cx="${x(i)}" cy="${y(s.values[i])}" r="4" fill="${s.color}" class="ring"/>`).join('');
      const rows = card.series.map((s) => `<div class="tr"><span class="key line" style="--c:${s.color}"></span>
        <b>${s.values[i] == null ? '–' : card.fmt(s.values[i])}</b><span>${esc(s.name)}</span></div>`);
      return `<div class="th">${esc(bucketTitle(hist.buckets[i]))}</div>${rows.join('')}`;
    }, (svg) => { $('.crosshair', svg).setAttribute('visibility', 'hidden'); $('.hover-dots', svg).innerHTML = ''; });
  }

  function hover(host, L, band, onMove, onLeave) {
    const svg = $('svg', host), hit = $('.hit', svg);
    const at = (ev) => {
      const r = svg.getBoundingClientRect();
      const i = Math.floor((ev.clientX - r.left - L) / band);
      return Math.max(0, Math.min(hist.buckets.length - 1, i));
    };
    hit.addEventListener('pointermove', (ev) => showTip(ev.clientX, ev.clientY, onMove(at(ev), svg)));
    hit.addEventListener('pointerleave', () => { hideTip(); onLeave(svg); });
  }

  function tableHtml(card) {
    const cols = card.series.map((s) => `<th>${esc(s.name)}</th>`).join('') + (card.kind === 'columns' ? '<th>failed</th>' : '');
    const rows = hist.buckets.map((b, i) => `<tr><td class="mono">${esc(bucketTitle(b))}</td>${card.series.map((s) =>
      `<td class="mono">${card.kind === 'columns' ? num(b.requests[s.name] || 0) : s.values[i] == null ? '–' : card.fmt(s.values[i])}</td>`).join('')}
      ${card.kind === 'columns' ? `<td class="mono">${num(b.errors)}</td>` : ''}</tr>`).reverse().join('');
    return `<div class="table-wrap"><table class="grid"><thead><tr><th>Time</th>${cols}</tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  // ------------------------------------------------------------- requests
  let range = '1h', hist = null, histAt = 0, histKey = '';
  const openTables = new Set();
  function chartCards() {
    const used = hist.endpoints.filter((n) => hist.buckets.some((b) => b.requests[n] || b.tps[n] != null));
    const colors = seriesColors(used);
    const series = (vals) => used.map((n) => ({ name: n, color: colors[n], values: hist.buckets.map((b) => vals(b, n)) }))
      .filter((s) => s.values.some((v) => v != null));
    return [
      { id: 'volume', kind: 'columns', title: 'Requests', sub: 'Successful requests per interval, by the model that answered',
        series: used.map((n) => ({ name: n, color: colors[n] })) },
      { id: 'speed', kind: 'lines', title: 'Output speed', sub: 'Average tokens per second while generating',
        series: series((b, n) => b.tps[n] ?? null), fmt: (v) => v.toFixed(1) + ' t/s', tick: (v) => tickFmt(v) },
      { id: 'ttft', kind: 'lines', title: 'Time to first token', sub: 'Across all models: typical (median) and slow (95th percentile)',
        series: [
          { name: 'median', color: SLOTS[2], values: hist.buckets.map((b) => b.ttft_p50) },
          { name: '95th percentile', color: SLOTS[3], values: hist.buckets.map((b) => b.ttft_p95) },
        ], fmt: (v) => secs(v), tick: (v) => (v >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 0 : 1) + 's' : Math.round(v) + 'ms') },
    ];
  }
  function drawCharts() {
    const holder = $('#charts');
    const cards = chartCards();
    holder.innerHTML = cards.map((c) => `<section class="panel chart-card" data-chart="${c.id}">
      <div class="panel-head"><div><h2>${esc(c.title)}</h2><p class="muted small">${esc(c.sub)}</p></div>
        <button class="small ghost" data-table="${c.id}" aria-pressed="${openTables.has(c.id)}">${openTables.has(c.id) ? 'Hide table' : 'Show as table'}</button></div>
      ${legendHtml(c.series, c.kind === 'columns' ? 'bar' : 'line')}
      <div class="plot"></div><div class="data-table" ${openTables.has(c.id) ? '' : 'hidden'}></div></section>`).join('');
    cards.forEach((c) => {
      const el = $(`[data-chart="${c.id}"]`, holder);
      const plot = $('.plot', el);
      if (c.kind === 'lines' && !c.series.length) plot.innerHTML = '<div class="empty">No data in this range yet.</div>';
      else if (c.kind === 'columns') drawColumns(plot, c);
      else drawLines(plot, c);
      if (openTables.has(c.id)) $('.data-table', el).innerHTML = tableHtml(c);
    });
    holder.dataset.cards = JSON.stringify(cards.map((c) => c.id));
  }
  async function renderRequests(force = false) {
    if (!force && hist && Date.now() - histAt < 15000) return;
    $('#charts').classList.add('loading');
    const h = await api('/api/history?range=' + range);
    $('#charts').classList.remove('loading');
    histAt = Date.now();
    const key = JSON.stringify([h.start, h.totals, h.buckets.length, h.problems.length && h.problems[0].id]);
    if (!force && key === histKey) return;
    histKey = key;
    hist = h;
    const t = h.totals;
    const tiles = [
      [num(t.requests), 'requests'],
      [t.requests ? Math.round((100 * t.ok) / t.requests) + '%' : '–', 'succeeded'],
      [secs(t.ttft_p50), 'typical first token'],
      [secs(t.ttft_p95), 'slow first token (p95)'],
      [num(t.tokens_out), 'tokens generated'],
      [`${num(t.fallbacks)} / ${num(t.metered)}`, 'fell back / paid'],
    ];
    $('#req-tiles').innerHTML = tiles.map(([k, l]) => `<div class="tile"><div class="k">${k}</div><div class="l">${l}</div></div>`).join('');
    drawCharts();
    $('#problems tbody').innerHTML = h.problems.slice(0, 10).map((r) => {
      const trail = (r.attempts || []).filter((a) => a.trigger || a.skipped).map((a) => `${a.endpoint}: ${a.skipped || a.trigger}`).join('; ');
      const what = r.ok ? `<span class="ok">answered</span> after falling back${trail ? ` (${esc(trail)})` : ''}` : `<span class="bad">${esc(r.error || 'failed')}</span>`;
      return `<tr><td class="mono">${esc(when(r.ts))}</td><td>${esc(r.model)}</td><td>${esc(r.endpoint || '–')}</td>
        <td>${esc(r.client)}</td><td class="wrap">${what}</td></tr>`;
    }).join('') || '<tr><td colspan="5" class="empty">Nothing failed or fell back in this range.</td></tr>';
  }
  $('#range').addEventListener('click', (ev) => {
    const b = ev.target.closest('button[data-range]');
    if (!b) return;
    range = b.dataset.range;
    $$('#range button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    renderRequests(true).catch((e) => banner(e.message));
  });
  let resizeT = 0;
  window.addEventListener('resize', () => {
    $$('.model-row').forEach(drawPipes);
    clearTimeout(resizeT);
    resizeT = setTimeout(() => { if (view === 'requests' && hist) drawCharts(); }, 150);
  });

  // ------------------------------------------------------------- models and tokens
  function epKind(spec) {
    const out = [];
    out.push(spec.metered ? '<span class="chip warn">paid</span>' : '<span class="chip">local</span>');
    if (spec.fallback) out.push('<span class="chip">fallback only</span>');
    if (spec.dialect === 'anthropic') out.push('<span class="chip">anthropic api</span>');
    return out.join(' ');
  }
  async function renderAccess() {
    const human = who && who.human;
    cfg = await api('/api/config');
    const live = Object.fromEntries(status.endpoints.map((e) => [e.name, e]));
    $('#endpoints tbody').innerHTML = Object.entries(cfg.endpoints).map(([n, sp]) => {
      const e = live[n] || {};
      const st = e.name ? laneState(e) : 'unknown';
      return `<tr>
        <td class="wrap"><b>${esc(n)}</b><div class="chain">${esc(sp.url)}</div></td>
        <td class="mono">${esc(sp.model || 'client’s name')}</td>
        <td>${epKind(sp)}</td>
        <td><span class="state ${st}"><span class="st"></span>${st}</span></td>
        <td class="chain wrap">${sp.flows.length ? esc(sp.flows.join(', ')) : 'no flows'}</td>
        <td class="mono">${sp.key_set ? 'set' : sp.metered ? '<span class="bad">missing</span>' : '–'}</td>
        <td class="actions">
          ${e.draining ? `<button class="small" data-resume="${esc(n)}">Resume</button>` : `<button class="small secondary" data-pause="${esc(n)}">Pause</button>`}
          <button class="small secondary edit-only" data-edit-model="${esc(n)}">Edit</button>
          <button class="small ghost edit-only" data-delete-model="${esc(n)}">Remove</button></td></tr>`;
    }).join('') || '<tr><td colspan="7" class="empty">No models yet. Add one to start routing.</td></tr>';
    $('#models tbody').innerHTML = status.models.map((m) => `<tr>
      <td class="wrap"><b>${esc(m.name)}</b>${m.aliases.length ? `<div class="chain">also ${esc(m.aliases.join(', '))}</div>` : ''}</td>
      <td class="chain wrap"><b>${esc(m.pool.join(', ') || '–')}</b>${m.fallback.length ? ` then ${esc(m.fallback.join(', '))}` : ''}</td>
      <td class="chain wrap">${m.fallback.length ? esc(m.fallback_on.join(', ')) : '–'}</td>
      <td><span class="seg" role="group" aria-label="Access for ${esc(m.name)}">
        <button data-auth="${esc(m.name)}" data-mode="open" aria-pressed="${m.auth === 'open'}" ${human ? '' : 'disabled'}>Open</button>
        <button data-auth="${esc(m.name)}" data-mode="token" aria-pressed="${m.auth === 'token'}" ${human ? '' : 'disabled'}>Token</button>
      </span>${m.warning ? `<div class="chain bad">${esc(m.warning)}</div>` : ''}</td></tr>`).join('');
    const form = $('#token-form');
    form.querySelectorAll('input, button').forEach((el) => (el.disabled = !human));
    if (!human) {
      $('#tokens tbody').innerHTML = '<tr><td colspan="7" class="empty">Sign in through the admin site to see and manage tokens.</td></tr>';
      return;
    }
    const toks = await api('/api/tokens');
    $('#tokens tbody').innerHTML = toks.map((t) => `<tr>
      <td><b>${esc(t.label)}</b></td><td class="mono">${esc(t.prefix)}…</td><td class="mono">${esc(t.models.join(', '))}</td>
      <td>${t.background ? 'background' : 'interactive'}</td><td class="mono">${num(t.uses)}</td><td class="mono">${ago(t.last_used)}</td>
      <td><button class="small danger" data-revoke="${t.id}" data-label="${esc(t.label)}">Revoke</button></td></tr>`).join('')
      || '<tr><td colspan="7" class="empty">No tokens yet. Create one for each client that should reach token-only flows.</td></tr>';
  }

  // ------------------------------------------------------------- model dialog (add / edit)
  const PRESETS = {
    local: { name: '', url: 'http://127.0.0.1:8080', dialect: 'openai', model: '', max_concurrency: 1, probe: 'llamacpp', metered: false, fallback: false },
    deepseek: { name: 'deepseek', url: 'https://api.deepseek.com', dialect: 'openai', model: 'deepseek-chat', max_concurrency: 8, context: 131072, probe: 'models', metered: true, fallback: true },
    anthropic: { name: 'anthropic', url: 'https://api.anthropic.com', dialect: 'anthropic', model: '', max_concurrency: 4, context: 200000, probe: 'none', metered: true, fallback: false },
    openai: { name: 'openai', url: 'https://api.openai.com', dialect: 'openai', model: '', max_concurrency: 8, probe: 'models', metered: true, fallback: false },
    openrouter: { name: 'openrouter', url: 'https://openrouter.ai/api', dialect: 'openai', model: '', max_concurrency: 8, probe: 'models', metered: true, fallback: false },
    other: { name: '', url: 'https://', dialect: 'openai', model: '', max_concurrency: 4, probe: 'models', metered: true, fallback: false },
  };
  let editing = null;
  function fillModelForm(v) {
    const f = $('#model-form');
    for (const k of ['name', 'url', 'dialect', 'model', 'max_concurrency', 'context', 'probe']) {
      if (k in v) f.elements[k].value = v[k] ?? '';
    }
    for (const k of ['metered', 'fallback']) if (k in v) f.elements[k].checked = !!v[k];
  }
  async function openModelDialog(name) {
    const f = $('#model-form');
    f.reset();
    formError(f, '');
    $('#test-result').textContent = '';
    $('#upstream-models').innerHTML = '';
    editing = name || null;
    $('#model-title').textContent = name ? `Edit ${name}` : 'Add a model';
    $('#model-save').textContent = name ? 'Save changes' : 'Add model';
    $('#preset-field').hidden = !!name;
    f.elements.name.readOnly = !!name;
    $('.clear-key', f).hidden = true;
    if (name) {
      cfg = await api('/api/config');
      const sp = cfg.endpoints[name];
      fillModelForm({ name, ...sp, context: sp.context || '' });
      f.elements.key.placeholder = sp.key_set ? 'saved; leave blank to keep it' : 'sk-…';
      $('.clear-key', f).hidden = sp.key !== 'file';
    } else {
      fillModelForm(PRESETS.local);
      f.elements.key.placeholder = 'sk-…';
    }
    $('#model-dialog').showModal();
  }
  $('#model-form').elements.preset.addEventListener('change', (ev) => {
    if (!editing) fillModelForm({ context: '', ...PRESETS[ev.target.value] });
  });
  $('#test-model').addEventListener('click', async () => {
    const f = $('#model-form'), out = $('#test-result');
    out.className = 'muted small';
    out.textContent = 'Testing…';
    try {
      const r = await post('/api/test-endpoint', { url: f.elements.url.value, dialect: f.elements.dialect.value,
        key: f.elements.key.value, name: editing });
      if (r.ok) {
        out.className = 'small ok';
        out.textContent = `Connected in ${r.ms} ms. ${r.models.length ? `It offers ${r.models.length} model${r.models.length === 1 ? '' : 's'}; pick one below.` : 'It didn’t list any models.'}`;
        $('#upstream-models').innerHTML = r.models.map((m) => `<option value="${esc(m)}"></option>`).join('');
        if (!f.elements.model.value && r.models.length === 1) f.elements.model.value = r.models[0];
      } else {
        out.className = 'small bad';
        out.textContent = `Couldn’t connect: ${r.error}`;
      }
    } catch (e) { out.className = 'small bad'; out.textContent = e.message; }
  });
  $('#model-form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const f = ev.target, el = f.elements;
    const body = {
      url: el.url.value.trim(), dialect: el.dialect.value, model: el.model.value.trim() || null,
      max_concurrency: Number(el.max_concurrency.value) || 1, context: el.context.value ? Number(el.context.value) : null,
      probe: el.probe.value, metered: el.metered.checked, fallback: el.fallback.checked,
    };
    if (el.key.value.trim()) body.key = el.key.value.trim();
    if (editing && el.clear_key.checked) body.clear_key = true;
    try {
      if (editing) await post(`/api/endpoints/${enc(editing)}`, body, 'PUT');
      else await post('/api/endpoints', { name: el.name.value.trim(), ...body });
      $('#model-dialog').close();
      await tick();
    } catch (e) { formError(f, e.message); }
  });

  // ------------------------------------------------------------- flow dialogs
  function openMemberDialog(flow) {
    const f = $('#member-form');
    f.reset();
    formError(f, '');
    $('#member-flow').textContent = flow;
    f.dataset.flow = flow;
    const m = status.models.find((x) => x.name === flow);
    const have = new Set([...m.pool, ...m.fallback]);
    const opts = status.endpoints.filter((e) => !have.has(e.name));
    f.elements.endpoint.innerHTML = opts.map((e) => `<option value="${esc(e.name)}" data-fb="${e.fallback ? 1 : 0}" data-paid="${e.metered ? 1 : 0}">${esc(e.name)}${e.metered ? ' (paid)' : ''}${e.fallback ? ' (fallback only)' : ''}</option>`).join('');
    if (!opts.length) { formError(f, 'Every model is already in this flow. Add a new one under Models and tokens.'); }
    syncRole();
    $('#member-dialog').showModal();
  }
  function syncRole() {
    const f = $('#member-form'), o = f.elements.endpoint.selectedOptions[0];
    const fbOnly = o && o.dataset.fb === '1';
    f.elements.role[0].disabled = fbOnly;
    f.elements.role.value = fbOnly || (o && o.dataset.paid === '1') ? 'fallback' : 'pool';
  }
  $('#member-form').elements.endpoint.addEventListener('change', syncRole);
  $('#member-form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const f = ev.target;
    try {
      await post(`/api/flows/${enc(f.dataset.flow)}/members`, { endpoint: f.elements.endpoint.value, role: f.elements.role.value });
      $('#member-dialog').close();
      await tick();
    } catch (e) { formError(f, e.message); }
  });

  function openFlowDialog() {
    const f = $('#flow-form');
    f.reset();
    formError(f, '');
    $('#flow-members').innerHTML = status.endpoints.map((e) => `<div class="pick-row" data-ep="${esc(e.name)}">
      <span>${esc(e.name)}<small>${e.metered ? 'paid' : 'local'}${e.fallback ? ' · fallback only' : ''}</small></span>
      <span class="seg" role="group" aria-label="Use ${esc(e.name)}">
        <button type="button" data-use="" aria-pressed="true">Not used</button>
        <button type="button" data-use="pool" aria-pressed="false" ${e.fallback ? 'disabled' : ''}>Shares load</button>
        <button type="button" data-use="fallback" aria-pressed="false">Fallback</button>
      </span></div>`).join('') || '<div class="empty">Add a model under Models and tokens first.</div>';
    $('#flow-dialog').showModal();
  }
  $('#flow-members').addEventListener('click', (ev) => {
    const b = ev.target.closest('button[data-use]');
    if (!b) return;
    $$('button', b.parentElement).forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  });
  $('#flow-form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const f = ev.target, pool = [], fallback = [];
    $$('.pick-row', f).forEach((row) => {
      const use = $('button[aria-pressed="true"]', row).dataset.use;
      if (use === 'pool') pool.push(row.dataset.ep);
      if (use === 'fallback') fallback.push(row.dataset.ep);
    });
    if (!pool.length && !fallback.length) { formError(f, 'Pick at least one model for this flow.'); return; }
    try {
      await post('/api/flows', { name: f.elements.name.value.trim(), aliases: f.elements.aliases.value, pool, fallback });
      $('#flow-dialog').close();
      await tick();
    } catch (e) { formError(f, e.message); }
  });
  $('#new-flow').addEventListener('click', openFlowDialog);
  $('#add-model').addEventListener('click', () => openModelDialog(null).catch((e) => banner(e.message)));

  // ------------------------------------------------------------- actions
  document.addEventListener('click', async (ev) => {
    const b = ev.target.closest('button');
    if (!b || b.closest('dialog')) return;
    const d = b.dataset;
    try {
      if (d.pause) {
        if (!(await confirmAction(`Pause ${d.pause}?`, `No new requests go to ${d.pause} from any flow; requests already running finish. It stays paused, even across restarts, until you resume it.`, 'Pause'))) return;
        await post(`/api/endpoints/${enc(d.pause)}/pause`);
      } else if (d.resume) {
        await post(`/api/endpoints/${enc(d.resume)}/resume`);
      } else if (d.addMember) {
        openMemberDialog(d.addMember);
        return;
      } else if (d.removeMember) {
        if (!(await confirmAction(`Remove ${d.removeMember} from ${d.flow}?`, `${d.flow} stops sending requests to it. The model itself stays, along with any other flows that use it.`, 'Remove', true))) return;
        await api(`/api/flows/${enc(d.flow)}/members/${enc(d.removeMember)}`, { method: 'DELETE' });
      } else if (d.deleteFlow) {
        if (!(await confirmAction(`Delete the ${d.deleteFlow} flow?`, `Clients asking for ${d.deleteFlow} or its other names will get “unknown model”. Its models stay.`, 'Delete flow', true))) return;
        await api(`/api/flows/${enc(d.deleteFlow)}`, { method: 'DELETE' });
      } else if (d.editModel) {
        await openModelDialog(d.editModel);
        return;
      } else if (d.deleteModel) {
        const sp = cfg && cfg.endpoints[d.deleteModel];
        const used = sp && sp.flows.length ? ` It's also taken out of ${sp.flows.join(', ')}.` : '';
        if (!(await confirmAction(`Remove ${d.deleteModel}?`, `llmanifold forgets this model and its saved key.${used}`, 'Remove', true))) return;
        await api(`/api/endpoints/${enc(d.deleteModel)}?force=1`, { method: 'DELETE' });
      } else if (d.auth) {
        const mode = d.mode;
        const text = mode === 'open' ? 'Anyone who can reach the API will be able to use this flow without a token.'
          : 'Requests without a valid token for this flow will be refused.';
        if (!(await confirmAction(`Make ${d.auth} ${mode === 'open' ? 'open' : 'token-only'}?`, text, mode === 'open' ? 'Make open' : 'Require tokens'))) return;
        await post(`/api/models/${enc(d.auth)}/auth`, { mode });
      } else if (d.revoke) {
        if (!(await confirmAction(`Revoke ${d.label}?`, 'Clients using this token will be refused on token-only flows straight away.', 'Revoke', true))) return;
        await api(`/api/tokens/${d.revoke}`, { method: 'DELETE' });
        $('#new-token').hidden = true;
      } else if (d.table) {
        openTables.has(d.table) ? openTables.delete(d.table) : openTables.add(d.table);
        drawCharts();
        return;
      } else return;
      banner('');
      await tick();
    } catch (e) { banner(e.message); }
  });

  $('#token-form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const fd = new FormData(ev.target);
    try {
      const t = await post('/api/tokens', { label: fd.get('label'), models: (fd.get('models') || '*'), background: fd.get('background') === 'on' });
      const box = $('#new-token');
      box.hidden = false;
      box.innerHTML = `Token for <b>${esc(t.label)}</b> created. Copy it now: it won't be shown again.<code>${esc(t.token)}</code>`;
      ev.target.reset();
      banner('');
      await renderAccess();
    } catch (e) { banner(e.message); }
  });

  // ------------------------------------------------------------- polling
  async function refreshView(force = false) {
    if (!status) return;
    try {
      if (view === 'overview') renderOverview();
      else if (view === 'requests') await renderRequests(force);
      else if (view === 'access') await renderAccess();
    } catch (e) { banner(e.message); }
  }
  async function tick() {
    try {
      status = await api('/api/status');
      who = status.who;
      lastOk = Date.now();
      document.body.classList.toggle('can-edit', canEdit());
      $('#who').textContent = who.human ? (who.email || 'signed in') : `view, pause and resume only (${who.ip})`;
      $('#ver').textContent = 'v' + status.version;
      if (status.reload_error) banner('The config file has an error, so the previous config is still in use: ' + status.reload_error);
      else if (who.human && !status.editable) banner('llmanifold can’t write its config file, so models and flows can only be changed by editing it.');
      if (!$$('dialog').some((d) => d.open)) await refreshView();
    } catch (e) {
      banner(e.message === 'forbidden' ? 'This address may not use the admin site.' : 'Lost contact with llmanifold: ' + e.message);
    }
    const stale = Date.now() - lastOk > 6000;
    $('#live-dot').classList.toggle('stale', stale);
    $('#live').textContent = stale ? 'not updating' : 'live';
  }
  show(location.hash.slice(1) || 'overview');
  tick();
  setInterval(() => { if (!document.hidden) tick(); }, 2000);
})();
