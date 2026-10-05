"""Regenerate the README images in docs/img/ from the real dashboard code and
entirely fictional data (no real hosts, people, tokens or traffic).

    pip install playwright      # uses the Chrome already on this machine
    python docs/screenshots/make_screenshots.py [--out docs/img] [--no-gif]

Starts a real llmanifold on loopback in front of a few fake engines, seeds a day
of made-up request history and some tokens, holds a handful of streaming requests
open so lanes are busy and a queue forms, then drives each view with Playwright.

tribar.gif is the dashboard's background artwork on its own. It is rendered from
the real art.js; the copy served for the capture only gains a way to set the
clock and has its speeds rounded so that one 18 s cycle loops seamlessly. Needs
ffmpeg.
"""
import argparse
import asyncio
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src")]
import aiohttp
import yaml
from aiohttp import web
from playwright.async_api import async_playwright

from llmanifold.apps import build_admin_app, build_api_app
from llmanifold.config import load
from llmanifold.proxy import Core
from llmanifold.store import Store

STATIC = ROOT / "src/llmanifold/static"
WORDS = ("the lane is clear and the queue is short so the answer comes back at once and "
         "the next request finds a warm cache on the same engine ").split()


# ---------------------------------------------------------------- fake engines
def fake_engine(speed: float) -> web.Application:
    """An OpenAI-compatible engine that streams words for as long as the client listens."""
    async def chat(req: web.Request):
        body = await req.json()
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)

        def chunk(delta, finish=None):
            return ("data: " + json.dumps({"id": "x", "object": "chat.completion.chunk", "model": body.get("model"),
                                           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
                    + "\n\n").encode()
        await asyncio.sleep(0.25 + 4 / speed)
        await resp.write(chunk({"role": "assistant", "content": ""}))
        short = body.get("max_tokens")          # the warm-up requests finish; the rest stream until hung up on
        for i in range(short or 100000):
            await asyncio.sleep(1 / speed)
            await resp.write(chunk({"content": WORDS[i % len(WORDS)] + " "}))
        await resp.write(chunk({}, "stop"))
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def models(_):
        return web.json_response({"object": "list", "data": [{"id": "fake"}]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/v1/models", models)
    return app


async def start(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, "http://127.0.0.1:%d" % runner.addresses[0][1]


# ---------------------------------------------------------------- made-up history
def seed_history(store: Store, now: float) -> None:
    rnd = random.Random(7)
    clients = ["home-agent", "voice", "notebook", "nightly-digest", "ide", "10.0.0.24"]
    for minute in range(24 * 60):
        ts = now - minute * 60
        hour = (time.localtime(ts).tm_hour + time.localtime(ts).tm_min / 60)
        busy = 0.25 + 0.75 * max(0.0, 1 - abs(hour - 14.5) / 9) ** 1.5      # quiet nights, busy afternoons
        if minute < 50:
            busy = max(busy, 0.8)
        for _ in range(rnd.choices([0, 1, 2, 3, 5], [3, 4, 3, 2, 1])[0] if rnd.random() < busy else 0):
            model = rnd.choices(["local-large", "small-fast", "sonnet"], [6, 3, 1])[0]
            pressure = rnd.random() < 0.12 * busy
            attempts, fallback, error, ok = [], 0, None, 1
            if model == "local-large":
                endpoint = rnd.choice(["gpu0", "gpu1"])
                tps, ttft = rnd.gauss(46, 5), rnd.gauss(900, 250)
                if pressure:
                    endpoint, fallback = rnd.choice(["deepseek", "deepseek", "chatgpt"]), 1
                    tps, ttft = rnd.gauss(34, 6), rnd.gauss(2300, 600)
                    attempts = [{"endpoint": "pool", "trigger": "overflow"}, {"endpoint": endpoint, "ok": True, "fallback": True}]
            elif model == "small-fast":
                endpoint, tps, ttft = "gpu2", rnd.gauss(120, 12), rnd.gauss(260, 60)
            else:
                endpoint, tps, ttft = "claude", rnd.gauss(70, 8), rnd.gauss(1400, 300)
            if rnd.random() < 0.012:
                ok, error, endpoint = 0, "all pool endpoints stayed busy (pool: queue)", None
                attempts = [{"endpoint": "pool", "trigger": "queue"}]
            tin, tout = int(rnd.lognormvariate(7.2, 0.9)), int(rnd.lognormvariate(5.6, 0.8))
            store.record({
                "id": uuid.uuid4().hex, "ts": ts - rnd.random() * 60, "model": model, "endpoint": endpoint,
                "client": rnd.choice(clients), "priority": "interactive", "dialect_in": "openai", "dialect_out": "openai",
                "stream": 1, "status": 200 if ok else 503, "ok": ok, "attempts": attempts or [{"endpoint": endpoint, "ok": True}],
                "fallback": fallback, "metered": int(endpoint in ("deepseek", "chatgpt", "claude")),
                "tokens_in": tin if ok else 0, "tokens_out": tout if ok else 0, "queued_ms": int(rnd.random() * 900),
                "ttft_ms": max(80, int(ttft)) if ok else None, "duration_ms": int(max(200, ttft) + tout / max(tps, 5) * 1000),
                "tps": round(max(tps, 5), 1) if ok else None, "error": error})


# ---------------------------------------------------------------- the stack
async def build(tmp: Path):
    runners, urls = [], {}
    for name, speed in (("gpu0", 46), ("gpu1", 44), ("gpu2", 120), ("deepseek", 34), ("chatgpt", 38), ("claude", 70)):
        r, urls[name] = await start(fake_engine(speed))
        runners.append(r)
    (tmp / "k.key").write_text("sk-example")
    raw = {
        "listen": {"api": "127.0.0.1:0", "admin": "127.0.0.1:0"}, "data_dir": str(tmp / "data"), "keepalive_after": 0,
        "default_model": "local-large",
        "endpoints": {
            "gpu0": {"url": urls["gpu0"], "max_concurrency": 1, "context": 262144, "probe": "models"},
            "gpu1": {"url": urls["gpu1"], "max_concurrency": 1, "context": 262144, "probe": "models"},
            "gpu2": {"url": urls["gpu2"], "max_concurrency": 4, "context": 32768, "probe": "models"},
            "deepseek": {"url": urls["deepseek"], "model": "deepseek-chat", "key_file": str(tmp / "k.key"), "max_concurrency": 8,
                         "context": 65536, "metered": True, "fallback": True, "overflow_at": 3, "probe": "none"},
            "chatgpt": {"url": urls["chatgpt"], "model": "gpt-5.5", "key_file": str(tmp / "k.key"), "max_concurrency": 4,
                        "metered": True, "fallback": True, "probe": "none"},
            "claude": {"url": urls["claude"], "model": "claude-sonnet-4-5", "key_file": str(tmp / "k.key"),
                       "max_concurrency": 4, "metered": True, "probe": "none"},
        },
        "models": {
            "local-large": {"aliases": ["default", "gpt-4o"], "pool": ["gpu0", "gpu1"], "fallback": ["deepseek", "chatgpt"],
                            "queue_timeout": 120, "background_max_lanes": 1, "auth": "open",
                            "input_modalities": ["text", "image"], "allow_metered_unauthenticated": True},
            "small-fast": {"aliases": ["bulk"], "pool": ["gpu2"], "auth": "open"},
            "sonnet": {"pool": ["claude"], "auth": "token"},
        },
        "admin": {"allow_from": ["127.0.0.0/8"]},
    }
    (tmp / "data").mkdir()
    path = tmp / "config.yaml"
    path.write_text("# example\n" + yaml.safe_dump(raw, sort_keys=False))
    store = Store(tmp / "data" / "llmanifold.db")
    seed_history(store, time.time())
    tokens = {label: store.create_token(label, models, bg, "alex@example.com")["token"]
              for label, models, bg in (("home-agent", ["*"], False), ("voice", ["local-large", "small-fast"], False),
                                        ("nightly-digest", ["small-fast"], True), ("notebook", ["*"], False),
                                        ("ide", ["local-large", "sonnet"], False))}
    core = Core(load(path), store)
    await core.start()
    r1, api = await start(build_api_app(core))
    r2, admin = await start(build_admin_app(core))
    return core, store, runners + [r1, r2], api, admin, tokens


async def hold(session: aiohttp.ClientSession, api: str, token: str, model: str, max_tokens: int | None = None) -> None:
    """A client that keeps reading a streamed answer."""
    body = {"model": model, "stream": True, "messages": [{"role": "user", "content": "hello"}]}
    if max_tokens:
        body["max_tokens"] = max_tokens
    try:
        async with session.post(api + "/v1/chat/completions", headers={"Authorization": f"Bearer {token}"}, json=body) as r:
            async for _ in r.content.iter_any():
                pass
    except (aiohttp.ClientError, asyncio.CancelledError):
        pass


# ---------------------------------------------------------------- screenshots
# The fake engines listen on loopback ports; show the addresses a reader would expect instead.
NICE = {"gpu0": "http://gpu-01:8080", "gpu1": "http://gpu-01:8081", "gpu2": "http://gpu-02:8080",
        "deepseek": "https://api.deepseek.com", "chatgpt": "https://chatgpt.com/backend-api/codex",
        "claude": "https://api.anthropic.com"}
TIDY = """(() => {
  const nice = %s;
  for (const tr of document.querySelectorAll('#endpoints tbody tr')) {
    const name = tr.querySelector('td b, td strong, td')?.firstChild?.textContent?.trim();
    for (const el of tr.querySelectorAll('small, .mono, td *')) {
      if (el.children.length === 0 && /^http:\\/\\/127\\.0\\.0\\.1:\\d+/.test(el.textContent.trim()) && nice[name]) el.textContent = nice[name];
    }
  }
  const t = document.getElementById('art-toggle');
  if (t) t.textContent = 'Pause artwork';
})()""" % json.dumps(NICE)

async def screenshots(browser, admin: str, out: Path) -> None:
    async def page_at(width, height, scale=1):
        ctx = await browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=scale,
                                        reduced_motion="no-preference")
        page = await ctx.new_page()
        # a still of the artwork while it is whole, so it never lands mid-burst behind a chart
        await page.add_init_script("try { localStorage.setItem('llmanifold.art.motion', 'off') } catch (e) {}")
        await page.goto(admin + "/")
        await page.wait_for_selector(".model-row")
        await page.evaluate("document.fonts.ready")
        return ctx, page

    async def shot(page, name, width=1440, fit=False):
        if fit:     # grow the window to the page, so the fixed sidebar runs the full height
            await page.set_viewport_size({"width": width, "height": 600})
            height = await page.evaluate("document.documentElement.scrollHeight")
            await page.set_viewport_size({"width": width, "height": height})
            await page.wait_for_timeout(400)
        await page.evaluate(TIDY)
        await page.screenshot(path=out / name)

    ctx, page = await page_at(1440, 960)
    await page.wait_for_timeout(3500)
    await shot(page, "dashboard.png")

    await page.click('a[data-view="requests"]')
    await page.click('#range button[data-range="24h"]')
    await page.wait_for_selector(".chart-card svg")
    await page.wait_for_timeout(1200)
    await shot(page, "requests.png", fit=True)

    await page.click('a[data-view="access"]')
    await page.wait_for_selector("#tokens tbody tr")
    await page.wait_for_timeout(600)
    await shot(page, "models.png", fit=True)

    await page.click("#add-model")
    await page.select_option('#model-form select[name="preset"]', "deepseek")
    await page.wait_for_timeout(500)
    await page.set_viewport_size({"width": 1440, "height": 960})
    await page.wait_for_timeout(300)
    await shot(page, "add-model.png")
    await page.keyboard.press("Escape")

    await page.click('a[data-view="overview"]')
    await page.click('button[data-flow-settings="local-large"]')
    await page.wait_for_selector("#settings-dialog[open]")
    await page.fill('#settings-form input[name="name"]', "local-xl")     # mid-rename, so the keep-old-name option shows
    await page.wait_for_timeout(400)
    await shot(page, "flow-settings.png")
    await ctx.close()

    ctx, page = await page_at(390, 800, scale=2)
    await page.wait_for_timeout(1500)
    await shot(page, "dashboard-mobile.png")
    await ctx.close()


# ---------------------------------------------------------------- the artwork as a looping GIF
CYCLE, FPS, GIF_W, GIF_H = 18, 12, 400, 420
HOOKS = [   # (in art.js, replaced by): expose the clock, and round every speed to a divisor of one cycle
    ("window.manifoldArt = {", "window.manifoldArt = { frame(t) { clock = t; resize(); render(); },"),
    ("ctx.rotate(t * 0.06)", "ctx.rotate(t * Math.PI * 2 / 18)"),
    ("const ang = t * 0.06,", "const ang = t * Math.PI * 2 / 18,"),
    ("const head = t * 0.1;", "const head = t / 9;"),
    ("const SLOT = 1.1,", "const SLOT = 1.125,"),
    ("const hash = (k, j) => { const x", "const hash = (k, j) => { k = ((k % 16) + 16) % 16; const x"),
    ("cx = w * 0.47,", "cx = w * 0.5,"),
]
GIF_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>html,body{margin:0;background:#101310;overflow:hidden}
#manifold-art{position:relative;width:%dpx;height:%dpx;background:#101310}
canvas{position:absolute;inset:0;width:100%%;height:100%%}</style></head><body>
<div id="manifold-art"><canvas></canvas></div><button id="art-toggle" hidden></button>
<script>try{localStorage.setItem('llmanifold.art.motion','off')}catch(e){}</script>
<script>%s</script></body></html>"""


async def gif(browser, out: Path, tmp: Path) -> None:
    js = (STATIC / "art.js").read_text()
    for a, b in HOOKS:
        if js.count(a) != 1:
            raise SystemExit(f"art.js changed: expected exactly one {a!r}")
        js = js.replace(a, b)
    html = tmp / "art.html"
    html.write_text(GIF_PAGE % (GIF_W, GIF_H, js))
    ctx = await browser.new_context(viewport={"width": GIF_W, "height": GIF_H}, device_scale_factor=1)
    page = await ctx.new_page()
    await page.goto(html.as_uri())
    await page.evaluate("document.fonts.ready")
    frames = tmp / "frames"
    frames.mkdir()
    for k in range(CYCLE * FPS):
        await page.evaluate("t => window.manifoldArt.frame(t)", CYCLE * 2 + k / FPS)
        await page.screenshot(path=frames / f"{k:04d}.png")
    await ctx.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "%04d.png"),
                    "-vf", "split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                    "-loop", "0", str(out / "tribar.gif")], check=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/img"))
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="llmanifold-shots-"))
    core, store, runners, api, admin, tokens = await build(tmp)
    session = aiohttp.ClientSession()
    load_ = [("home-agent", "local-large"), ("voice", "local-large"), ("notebook", "local-large"), ("ide", "local-large"),
             ("home-agent", "local-large"), ("voice", "local-large"), ("nightly-digest", "small-fast"),
             ("notebook", "small-fast"), ("ide", "sonnet")]
    rnd = random.Random(3)

    async def warm(model, who, count):           # traffic that already finished, so the counters aren't empty
        for n in range(count):
            await hold(session, api, tokens[who[n % len(who)]], model, max_tokens=rnd.randint(20, 90))
    await asyncio.gather(warm("local-large", ["home-agent", "ide"], 14), warm("local-large", ["voice", "notebook"], 14),
                         warm("small-fast", ["nightly-digest", "notebook"], 22), warm("sonnet", ["ide"], 9))
    tasks = []
    for who, model in load_:
        tasks.append(asyncio.create_task(hold(session, api, tokens[who], model)))
        await asyncio.sleep(0.25)
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(channel="chrome")
            await screenshots(browser, admin, out)
            if not args.no_gif:
                if not shutil.which("ffmpeg"):
                    raise SystemExit("ffmpeg is needed for tribar.gif (or pass --no-gif)")
                await gif(browser, out, tmp)
            await browser.close()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await session.close()
        for r in runners:
            await r.cleanup()
        await core.stop()
        store.close()
        shutil.rmtree(tmp, ignore_errors=True)
    for f in sorted(out.iterdir()):
        print(f"{f.resolve().relative_to(ROOT) if f.resolve().is_relative_to(ROOT) else f}  {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    asyncio.run(main())
