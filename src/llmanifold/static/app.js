// llmanifold admin dashboard. Plain JS, polls the admin API.
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const num = (n) => (n == null ? '–' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e4 ? (n / 1e3).toFixed(1) + 'k' : String(n));
  const ago = (ts) => {
    if (!ts) return 'never';
    const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 60) return Math.round(s) + 's ago';
    if (s < 3600) return Math.round(s / 60) + 'm ago';
    if (s < 86400) return Math.round(s / 3600) + 'h ago';
    return Math.round(s / 86400) + 'd ago';
  };
  const clock = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  let status = null, who = null, view = 'overview', lastOk = 0;

  async function api(path, opts = {}) {
    const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
    let body = null;
    try { body = await r.json(); } catch (_) { /* empty */ }
    if (!r.ok) throw new Error((body && (body.error || body.message)) || `HTTP ${r.status}`);
    return body;
  }

  function banner(msg) {
    const b = $('#banner');
    b.hidden = !msg;
    b.textContent = msg || '';
  }

  function confirmAction(title, text, okLabel = 'Confirm') {
    const d = $('#confirm');
    $('#confirm-title').textContent = title;
    $('#confirm-text').textContent = text;
    $('#confirm-ok').textContent = okLabel;
    return new Promise((res) => {
      d.addEventListener('close', () => res(d.returnValue === 'ok'), { once: true });
      d.showModal();
    });
  }

  // ------------------------------------------------------------- navigation
  const VIEWS = {
    overview: ['Overview', 'Live lanes, queue and recent traffic'],
    requests: ['Requests', 'What was asked, who served it, and how fast'],
    access: ['Models and tokens', 'Who may use which model'],
  };
  function show(v) {
    view = VIEWS[v] ? v : 'overview';
    document.querySelectorAll('.view').forEach((el) => (el.hidden = el.id !== 'view-' + view));
    document.querySelectorAll('.side nav a').forEach((a) => a.classList.toggle('active', a.dataset.view === view));
    $('#title').textContent = VIEWS[view][0];
    $('#subtitle').textContent = VIEWS[view][1];
    refreshView();
  }
  window.addEventListener('hashchange', () => show(location.hash.slice(1)));

  // ------------------------------------------------------------- overview
  function laneState(e) {
    if (!e.healthy) return 'down';
    if (e.draining) return 'draining';
    return e.load > 0 ? 'busy' : 'idle';
  }
  function laneNow(e, st) {
    if (st === 'down') return e.last_error || 'not answering';
    if (e.current && e.current.length) return e.current.join(', ');
    if (st === 'draining') return e.inflight ? 'draining: finishing in-flight work' : 'drained: no new requests';
    if (e.probe_busy > e.inflight) return 'busy: the engine is serving a client that bypassed llmanifold';
    return 'idle';
  }
  function laneHtml(e, fallback) {
    const st = laneState(e);
    const can = !!who;
    const act = e.draining
      ? `<button class="small secondary" data-undrain="${esc(e.name)}" ${can ? '' : 'disabled'}>Resume</button>`
      : `<button class="small secondary" data-drain="${esc(e.name)}" ${can ? '' : 'disabled'}>Drain</button>`;
    const tags = [e.metered ? 'metered' : '', fallback ? 'fallback' : '', e.dialect === 'anthropic' ? 'anthropic api' : '']
      .filter(Boolean).join(' · ');
    return `<div class="lane ${st} ${fallback ? 'fallback' : ''}" data-lane="${esc(e.name)}">
      <span class="st" title="${st}"></span>
      <span class="ln">${esc(e.name)}<small>${esc(tags || `${e.inflight}/${e.max_concurrency} in use`)}</small></span>
      <span class="now" title="${esc(laneNow(e, st))}">${esc(laneNow(e, st))}</span>
      <span class="speed">${e.tps ? e.tps + ' t/s' : '–'}<small>${e.ttft ? 'first ' + e.ttft + 's' : num(e.requests) + ' served'}</small></span>
      <span class="lane-act">${act}</span></div>`;
  }
  function drawPipes(row) {
    const cell = $('.pipes-cell', row), svg = $('svg.pipes', row), lanes = $('.lanes', row), intake = $('.intake', row);
    if (!cell || !svg || !lanes || !intake) return;
    const box = cell.getBoundingClientRect(), h = box.height, w = box.width;
    if (!h || !w) return;
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    const ib = intake.getBoundingClientRect(), y0 = ib.top - box.top + ib.height / 2;
    const paths = [...lanes.querySelectorAll('.lane')].map((ln) => {
      const r = ln.getBoundingClientRect(), y = r.top - box.top + r.height / 2;
      const cls = [ln.classList.contains('busy') ? 'busy' : '', ln.classList.contains('fallback') ? 'fb' : '',
        ln.classList.contains('down') ? 'down' : ''].join(' ');
      return `<path class="${cls}" d="M0 ${y0} C ${w / 2} ${y0}, ${w / 2} ${y}, ${w} ${y}"/>`;
    });
    svg.innerHTML = paths.join('');
  }
  function renderOverview() {
    const s = status, c = s.counters;
    const done = c.ok + c.errors;
    const tiles = [
      [num(c.requests), 'requests', ''],
      [done ? Math.round((100 * c.ok) / done) + '%' : '–', 'succeeded', 'good'],
      [num(c.fallbacks), 'fell back', ''],
      [num(c.metered), 'served by metered APIs', ''],
      [String(s.queue.length), 'waiting now', s.queue.length ? '' : 'good'],
    ];
    $('#tiles').innerHTML = tiles.map(([k, l, cls]) => `<div class="tile"><div class="k ${cls}">${k}</div><div class="l">${l}</div></div>`).join('');
    const byName = Object.fromEntries(s.endpoints.map((e) => [e.name, e]));
    $('#manifold').innerHTML = s.models.map((m) => {
      const pool = m.pool.map((n) => byName[n]).filter(Boolean);
      const fb = m.fallback.map((n) => byName[n]).filter(Boolean);
      const chips = [
        `<span class="chip ${m.auth === 'token' ? 'amber' : ''}">${m.auth === 'token' ? 'token required' : 'open'}</span>`,
        m.context ? `<span class="chip">${num(m.context)} ctx</span>` : '',
        m.queued ? `<span class="chip amber">${m.queued} waiting</span>` : '',
        m.warning ? `<span class="chip warn" title="${esc(m.warning)}">metered for anyone</span>` : '',
      ].join('');
      return `<div class="model-row">
        <div class="intake"><span class="name">${esc(m.name)}</span>
          ${m.aliases.length ? `<span class="aliases">also ${esc(m.aliases.join(', '))}</span>` : ''}
          <div class="meta">${chips}</div></div>
        <div class="pipes-cell"><svg class="pipes" aria-hidden="true"></svg></div>
        <div class="lanes">${pool.map((e) => laneHtml(e, false)).join('')}${fb.map((e) => laneHtml(e, true)).join('')}</div>
      </div>`;
    }).join('') || '<div class="panel empty">No models configured. Add some to the config file; it reloads by itself.</div>';
    document.querySelectorAll('.model-row').forEach(drawPipes);
    $('#queue').innerHTML = s.queue.length
      ? s.queue.map((q) => `<div class="q"><span>${esc(q.model)}</span><span>${esc(q.priority)}</span><span>${q.waiting_s}s</span></div>`).join('')
      : '<div class="empty">Nothing is waiting. Every request found a free lane.</div>';
  }

  // ------------------------------------------------------------- requests
  async function renderRequests() {
    const rows = await api('/api/requests?limit=200');
    const f = $('#req-filter').value;
    const keep = rows.filter((r) => f === 'all' || (f === 'errors' && !r.ok) || (f === 'fallback' && r.fallback) || (f === 'metered' && r.metered));
    $('#requests tbody').innerHTML = keep.map((r) => {
      const trail = (r.attempts || []).filter((a) => a.trigger || a.skipped)
        .map((a) => `${a.endpoint}: ${a.skipped || a.trigger}`).join('; ');
      const result = r.ok ? `<span class="ok">ok</span>${r.fallback ? ' after fallback' : ''}` : `<span class="bad">${esc(r.error || 'failed')}</span>`;
      return `<tr title="${esc(trail)}">
        <td class="mono">${clock(r.ts)}</td><td>${esc(r.model)}</td>
        <td>${esc(r.endpoint || '–')}${r.metered ? ' <span class="chip warn">metered</span>' : ''}</td>
        <td>${esc(r.client)}${r.priority === 'background' ? ' <span class="chip">background</span>' : ''}</td>
        <td class="mono">${num(r.tokens_in)} / ${num(r.tokens_out)}</td>
        <td class="mono">${r.ttft_ms != null ? (r.ttft_ms / 1000).toFixed(2) + 's' : '–'}</td>
        <td class="mono">${r.tps ? r.tps + ' t/s' : '–'}</td><td class="wrap">${result}</td></tr>`;
    }).join('') || `<tr><td colspan="8" class="empty">No requests yet${f !== 'all' ? ' match this filter' : ''}.</td></tr>`;
  }

  // ------------------------------------------------------------- access
  async function renderAccess() {
    const human = who && who.human;
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
      || '<tr><td colspan="7" class="empty">No tokens yet. Create one above for each client that should reach token-only models.</td></tr>';
  }

  // ------------------------------------------------------------- actions
  document.addEventListener('click', async (ev) => {
    const b = ev.target.closest('button');
    if (!b) return;
    try {
      if (b.dataset.drain) {
        if (!(await confirmAction(`Drain ${b.dataset.drain}?`, 'No new requests go to this endpoint; requests already running finish normally.', 'Drain'))) return;
        await api(`/api/endpoints/${encodeURIComponent(b.dataset.drain)}/drain`, { method: 'POST', body: '{}' });
      } else if (b.dataset.undrain) {
        await api(`/api/endpoints/${encodeURIComponent(b.dataset.undrain)}/undrain`, { method: 'POST', body: '{}' });
      } else if (b.dataset.auth) {
        const mode = b.dataset.mode;
        const text = mode === 'open' ? 'Anyone who can reach the API will be able to use this model without a token.'
          : 'Requests without a valid token for this model will be refused.';
        if (!(await confirmAction(`Make ${b.dataset.auth} ${mode === 'open' ? 'open' : 'token-only'}?`, text, mode === 'open' ? 'Make open' : 'Require tokens'))) return;
        await api(`/api/models/${encodeURIComponent(b.dataset.auth)}/auth`, { method: 'POST', body: JSON.stringify({ mode }) });
      } else if (b.dataset.revoke) {
        if (!(await confirmAction(`Revoke ${b.dataset.label}?`, 'Clients using this token will be refused on token-only models straight away.', 'Revoke'))) return;
        await api(`/api/tokens/${b.dataset.revoke}`, { method: 'DELETE' });
        $('#new-token').hidden = true;
      } else return;
      banner('');
      await tick();
    } catch (e) { banner(e.message); }
  });

  $('#token-form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const fd = new FormData(ev.target);
    try {
      const t = await api('/api/tokens', { method: 'POST', body: JSON.stringify({
        label: fd.get('label'), models: (fd.get('models') || '*'), background: fd.get('background') === 'on' }) });
      const box = $('#new-token');
      box.hidden = false;
      box.innerHTML = `Token for <b>${esc(t.label)}</b> created. Copy it now: it won't be shown again.<code>${esc(t.token)}</code>`;
      ev.target.reset();
      banner('');
      await renderAccess();
    } catch (e) { banner(e.message); }
  });
  $('#req-filter').addEventListener('change', () => renderRequests().catch((e) => banner(e.message)));

  // ------------------------------------------------------------- polling
  async function refreshView() {
    if (!status) return;
    try {
      if (view === 'overview') renderOverview();
      else if (view === 'requests') await renderRequests();
      else if (view === 'access') await renderAccess();
    } catch (e) { banner(e.message); }
  }
  async function tick() {
    try {
      status = await api('/api/status');
      who = status.who;
      lastOk = Date.now();
      $('#who').textContent = who.human ? (who.email || 'signed in') : `read and drain only (${who.ip})`;
      $('#ver').textContent = 'v' + status.version;
      if (status.reload_error) banner('The config file has an error, so the previous config is still in use: ' + status.reload_error);
      await refreshView();
    } catch (e) {
      banner(e.message === 'forbidden' ? 'This address may not use the admin site.' : 'Lost contact with llmanifold: ' + e.message);
    }
    const stale = Date.now() - lastOk > 6000;
    $('#live-dot').classList.toggle('stale', stale);
    $('#live').textContent = stale ? 'not updating' : 'live';
  }
  window.addEventListener('resize', () => document.querySelectorAll('.model-row').forEach(drawPipes));
  show(location.hash.slice(1) || 'overview');
  tick();
  setInterval(() => { if (view === 'overview' || !document.hidden) tick(); }, 2000);
})();
