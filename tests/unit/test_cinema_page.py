from html.parser import HTMLParser
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[2] / "services/api/recsys_api"


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tags = []
        self.copy = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.copy.append(data)


@pytest.fixture
def page():
    env = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=select_autoescape())
    html = env.get_template("index.html").render(
        min_ratings=5, url_for=lambda name, path: "/static/" + path
    )
    return Page(html)


def test_cinema_identity_and_onboarding(page):
    text = " ".join(page.copy)
    assert "Кино на вечер" in text
    assert "Найдём кино на вечер" in text
    assert "Оцените хотя бы 5 фильмов, которые смотрели" in text
    assert any(tag == "title" for tag, _ in page.tags)


def test_three_destination_navigation(page):
    links = [a.get("data-view") for tag, a in page.tags if tag == "button"]
    assert set(links) >= {"selection", "watchlist", "history"}
    text = " ".join(page.copy)
    for label in ("Подборка", "Хочу посмотреть", "Мои оценки"):
        assert label in text


def test_about_dialog_attribution(page):
    assert any(tag == "dialog" for tag, _ in page.tags)
    text = " ".join(page.copy)
    assert "О сервисе" in text
    assert "This product uses the TMDB API but is not endorsed or certified by TMDB." in text
    assert any(a.get("src") == "/static/tmdb-logo.svg" for _, a in page.tags)


def test_no_technical_product_copy(page):
    text = (" ".join(page.copy) + str(page.tags)).lower()
    for term in ("als", "lightgbm", "ndcg", "movielens", "model version", "кэш", "модель"):
        assert term not in text


def test_accessible_initial_state(page):
    by_id = {a["id"]: (tag, a) for tag, a in page.tags if "id" in a}
    assert "disabled" in by_id["start-selection"][1]
    assert by_id["search"][1]["maxlength"] == "100"
    assert by_id["deck"][1]["tabindex"] == "0"
    assert by_id["status"][1]["role"] == "status"


# Exercise real DOM/events and fetch in installed Chromium, without extra project dependencies.
# Backend responses are the fixed public HTTP contract; no owning API files are imported.
BROWSER_CHECK = r"""const fs = require("node:fs");
const http = require("node:http");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");
const assert = require("node:assert/strict");
const root = process.argv[1],
  chrome = process.argv[2],
  scenario = process.argv[3] || "contract";
const profile = fs.mkdtempSync(path.join(os.tmpdir(), "cinema-c-"));
let browser, ws, server;
let slowHistory = false,
  feedbackStatus = 503;
const movies = Array.from({ length: 80 }, (_, i) => ({
  movie_id: i + 1,
  title: i === 0 ? "История <img src=x onerror=alert(1)>" : "Фильм " + (i + 1),
  year: 1995,
  genres: ["Drama"],
  user_rating: null,
  poster_url: null,
}));
let saved = new Map(),
  feedback = new Map(),
  calls = [],
  failFeedback = false,
  slowWrite = false,
  failRating = false,
  failHistory = false,
  unavailable = false;
if (scenario.startsWith("stale-") || scenario === "keyboard-history")
  saved = new Map([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]]);
if (scenario === "stale-watchlist")
  feedback = new Map([[7, "watchlist"], [31, "watchlist"]]);
const heldReads = new Map();
function holdRead(route) {
  let started, release;
  const received = new Promise((resolve) => { started = resolve; });
  const gate = new Promise((resolve) => { release = resolve; });
  heldReads.set(route, async (res) => {
    started();
    await gate;
    res.statusCode = 503;
    res.setHeader("X-Cinema-Regression", scenario);
    res.end("{}");
  });
  return { received, release };
}
movies[0].poster_url = "https://image.tmdb.org/t/p/w500/fixture.jpg";
const delay = (ms) => new Promise((r) => setTimeout(r, ms));
async function main() {
  server = http.createServer(async (req, res) => {
    const url = new URL(req.url, "http://localhost");
    if (url.pathname === "/") {
      let html = fs
        .readFileSync(path.join(root, "templates/index.html"), "utf8")
        .replaceAll("{{ min_ratings }}", "5")
        .replace(/{{ url_for\('static', path='([^']+)'\) }}/g, "/static/$1");
      res.setHeader("Content-Type", "text/html;charset=utf-8");
      res.end(html);
      return;
    }
    if (url.pathname.startsWith("/static/")) {
      const file = path.join(root, "static", path.basename(url.pathname));
      res.setHeader(
        "Content-Type",
        file.endsWith(".js")
          ? "application/javascript"
          : file.endsWith(".svg")
            ? "image/svg+xml"
            : "text/css",
      );
      res.end(fs.readFileSync(file));
      return;
    }
    calls.push([req.method, url.pathname]);
    res.setHeader("Content-Type", "application/json");
    let body = "";
    for await (const chunk of req) body += chunk;
    const data = body ? JSON.parse(body) : {};
    if (req.method === "GET" && heldReads.has(url.pathname)) {
      const respond = heldReads.get(url.pathname);
      heldReads.delete(url.pathname);
      await respond(res);
      return;
    }
    if (req.method === "POST" && slowWrite) await delay(250);
    if (req.method === "POST" && url.pathname === "/ratings") {
      if (failRating) {
        failRating = false;
        res.statusCode = 503;
        res.end("{}");
        return;
      }
      saved.set(data.movie_id, data.rating);
      feedback.delete(data.movie_id);
      res.end(JSON.stringify({ ...data, n_ratings: saved.size }));
      return;
    }
    if (req.method === "POST" && url.pathname === "/feedback") {
      if (failFeedback) {
        failFeedback = false;
        res.statusCode = feedbackStatus;
        res.end("{}");
        return;
      }
      feedback.set(data.movie_id, data.state);
      res.end(JSON.stringify(data));
      return;
    }
    if (req.method === "DELETE") {
      feedback.delete(Number(url.pathname.split("/").pop()));
      res.statusCode = 204;
      res.end();
      return;
    }
    if (url.pathname === "/ratings") {
      if (failHistory) {
        res.statusCode = 503;
        res.end("{}");
        return;
      }
      const snapshot = {
        items: movies
          .filter((m) => saved.has(m.movie_id))
          .map((m) => ({ ...m, user_rating: saved.get(m.movie_id) })),
        n_ratings: saved.size,
        min_ratings: 5,
      };
      if (slowHistory) await delay(600);
      res.end(JSON.stringify(snapshot));
      return;
    }
    if (url.pathname === "/watchlist") {
      res.end(
        JSON.stringify({
          items: movies.filter((m) => feedback.get(m.movie_id) === "watchlist"),
        }),
      );
      return;
    }
    if (url.pathname === "/recommendations") {
      res.end(
        JSON.stringify({
          items: movies
            .filter((m) => !saved.has(m.movie_id) && !feedback.has(m.movie_id))
            .slice(0, 50),
          n_ratings: unavailable ? null : saved.size,
        }),
      );
      return;
    }
    if (url.pathname === "/movies/search") {
      await delay(300);
      res.end(JSON.stringify({ items: movies.slice(30, 32) }));
      return;
    }
    if (url.pathname === "/movies/onboarding") {
      res.end(JSON.stringify({ items: movies.slice(0, 20) }));
      return;
    }
    res.statusCode = 404;
    res.end("{}");
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  browser = spawn(
    chrome,
    [
      "--headless=new",
      "--no-sandbox",
      "--disable-gpu",
      "--no-first-run",
      "--remote-debugging-port=0",
      "--remote-allow-origins=*",
      "--user-data-dir=" + profile,
      "about:blank",
    ],
    { stdio: "ignore", windowsHide: true },
  );
  let port;
  for (let i = 0; i < 100; i++) {
    try {
      port = fs
        .readFileSync(path.join(profile, "DevToolsActivePort"), "utf8")
        .split("\n")[0];
      break;
    } catch {}
    await delay(100);
  }
  assert.ok(port, "Chrome debug port");
  const targets = await (
    await fetch("http://127.0.0.1:" + port + "/json/list")
  ).json();
  ws = new WebSocket(targets[0].webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  let seq = 0;
  const pending = new Map();
  const responses = new Map(), completed = new Set();
  ws.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    if (data.method === "Network.responseReceived")
      responses.set(data.params.requestId, data.params.response);
    if (data.method === "Network.loadingFinished")
      completed.add(data.params.requestId);
    if (data.id) {
      const p = pending.get(data.id);
      pending.delete(data.id);
      if (data.error) p.reject(new Error(JSON.stringify(data.error)));
      else p.resolve(data.result);
    }
  });
  const command = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const id = ++seq;
      pending.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params }));
    });
  const ev = async (expression) => {
    const r = await command("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  };
  const wait = async (expression) => {
    for (let i = 0; i < 150; i++) {
      if (await ev(expression)) return;
      await delay(30);
    }
    throw new Error("Timed out: " + expression);
  };
  const click = (selector) =>
    ev(`document.querySelector(${JSON.stringify(selector)}).click()`);
  const key = async (key, code) => {
    const event = { key, code, windowsVirtualKeyCode: { Enter: 13, ArrowRight: 39 }[key] };
    await command("Input.dispatchKeyEvent", { type: "keyDown", ...event, text: key === "Enter" ? "\r" : "" });
    await command("Input.dispatchKeyEvent", { type: "keyUp", ...event });
  };
  const settleResponses = async (matches, count = 1) => {
    for (let i = 0; i < 150; i++) {
      if ([...responses].filter(([id, response]) => matches(response) && completed.has(id)).length >= count) {
        await ev("new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))");
        return;
      }
      await delay(30);
    }
    throw new Error("Expected HTTP responses did not finish: " + scenario);
  };
  const finishHeldRead = async (held) => {
    held.release();
    await settleResponses(response => response.headers["X-Cinema-Regression"] === scenario);
  };
  await command("Page.enable");
  await command("Network.enable");
  await command("Network.setCacheDisabled", { cacheDisabled: true });
  await command("Network.setBlockedURLs", {
    urls: ["https://image.tmdb.org/*"],
  });
  await command("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false,
  });
  const origin = "http://127.0.0.1:" + server.address().port;
  await command("Page.navigate", { url: origin });
  await wait(
    'document.querySelectorAll("#onboarding-grid .movie-card").length === 10',
  );
  if (scenario.startsWith("stale-")) {
    await wait('document.querySelector(".deck-card") !== null');
    if (scenario === "stale-watchlist") {
      await click('[data-view="watchlist"]');
      await wait('document.querySelectorAll("#watchlist-grid .movie-card").length === 2');
    }
    await ev(`document.querySelector('#search').value='Фильм';document.querySelector('#search').dispatchEvent(new Event('input'));`);
    await wait('document.querySelectorAll("#search-grid .movie-card").length === 2');
    const history = scenario === "stale-history";
    const held = holdRead(history ? "/ratings" : "/watchlist");
    slowWrite = true;
    // Start the read while the write is pending: confirmation must invalidate it too.
    await click('#search-grid [data-movie-id="31"] .star');
    await click(`[data-view="${history ? "history" : "watchlist"}"]`);
    await held.received;
    await wait('document.querySelector("#status").textContent === "Оценка сохранена"');
    if (!history) {
      // This confirmed rating removes movie 31; keep another saved film visible.
      await settleResponses(response => response.url.endsWith("/watchlist") && response.status === 200, 2);
      await wait('document.querySelector("#watchlist-grid .movie-card") !== null');
    }
    const grid = history ? "history-grid" : "watchlist-grid";
    const expectedIds = history ? [1, 2, 3, 4, 5, 31] : [7];
    const ids = () => ev(`Array.from(document.querySelectorAll('#${grid} .movie-card'), card => Number(card.dataset.movieId))`);
    assert.deepEqual(await ids(), expectedIds, "confirmed mutation renders the current list");
    await finishHeldRead(held);
    assert.deepEqual(await ids(), expectedIds, `stale rejected ${history ? "history" : "watchlist"} GET cannot replace the confirmed list`);
    assert.equal(await ev(`document.querySelector('#${grid} .empty') === null`), true);
    console.log("PASS browser: " + scenario);
    return;
  }
  if (scenario.startsWith("keyboard-")) {
    const kind = scenario.slice("keyboard-".length);
    if (kind === "history") {
      await click('[data-view="history"]');
      await wait('document.querySelectorAll("#history-grid .movie-card").length === 5');
    } else if (kind === "search") {
      await ev(`document.querySelector('#search').value='Фильм';document.querySelector('#search').dispatchEvent(new Event('input'));`);
      await wait('document.querySelectorAll("#search-grid .movie-card").length === 2');
    }
    const grid = kind + "-grid", movieId = kind === "search" ? 31 : 1;
    const card = `#${grid} [data-movie-id="${movieId}"]`;
    failRating = true;
    slowWrite = true;
    await ev(`document.querySelector('${card} [data-rating="1"]').focus()`);
    await key("ArrowRight", "ArrowRight");
    await wait(`document.querySelector('${card} .card-error button') !== null && !document.querySelector('${card} .star').disabled`);
    const focus = () => ev(`({grid: document.activeElement.closest('.grid')?.id, movie: Number(document.activeElement.closest('[data-movie-id]')?.dataset.movieId), value: document.activeElement.dataset.rating})`);
    assert.deepEqual(await focus(), {grid, movie: movieId, value: "2"}, "failed rating restores the original keyboard star context");
    assert.equal(saved.get(movieId) || 0, kind === "history" ? 1 : 0, "failed rating is not applied");
    await ev(`document.querySelector('${card} .card-error button').focus()`);
    await key("Enter", "Enter");
    await wait('document.querySelector("#status").textContent === "Оценка сохранена"');
    assert.equal(saved.get(movieId), 2, "retry saves the originally chosen rating");
    assert.deepEqual(await focus(), {grid, movie: movieId, value: "2"}, "successful retry restores movie, grid and value instead of retry-button context");
    await key("ArrowRight", "ArrowRight");
    await wait(`document.querySelector('${card} [data-rating="3"]').getAttribute('aria-checked') === 'true' && !document.querySelector('${card} .star').disabled`);
    assert.equal(saved.get(movieId), 3, "keyboard rating continues after successful retry");
    console.log("PASS browser: " + scenario);
    return;
  }
  assert.equal(
    await ev('document.querySelector("#start-selection").disabled'),
    true,
  );
  await wait(
    'document.querySelector("#onboarding-grid .poster-placeholder") !== null',
  );
  assert.equal(
    await ev(
      'document.querySelector("#onboarding-grid .movie-card .poster img")',
    ),
    null,
    "image failure becomes local fallback",
  );
  assert.equal(
    await ev('document.querySelector("#onboarding-grid .movie-title img")'),
    null,
    "title is safe text",
  );
  await click("#onboarding-grid .text-button");
  assert.equal(saved.size, 0, "unfamiliar is not a rating");
  slowWrite = true;
  await ev("document.querySelector('#onboarding-grid .star').focus()");
  await click("#onboarding-grid .star");
  await click("#onboarding-grid .star");
  await wait(
    'document.querySelector("#progress-text").textContent.startsWith("1 из")',
  );
  assert.equal(
    calls.filter((c) => c[0] === "POST" && c[1] === "/ratings").length,
    1,
    "double click one mutation",
  );
  assert.equal(
    await ev("document.activeElement.classList.contains('star')"),
    true,
    "keyboard star focus survives save",
  );
  slowWrite = false;
  for (let id = 3; id <= 6; id++) {
    await click(`#onboarding-grid [data-movie-id="${id}"] .star`);
    await wait(
      `document.querySelector('#progress-text').textContent.startsWith('${id - 1} из')`,
    );
  }
  assert.equal(saved.size, 5);
  await ev(
    `document.querySelector('#onboarding-grid [data-movie-id="6"] .star').focus();document.activeElement.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));`,
  );
  await wait(
    'document.activeElement.dataset.rating === "2" && document.querySelector("#status").textContent === "Оценка сохранена"',
  );
  assert.equal(saved.size, 5, "repeat rating stays distinct count");
  assert.equal(saved.get(6), 2, "keyboard changes star value");
  assert.equal(
    await ev('document.querySelector("#start-selection").disabled'),
    false,
  );
  await click("#start-selection");
  await wait('document.querySelector(".deck-card") !== null');
  const current = await ev(
    'document.querySelector(".deck-card").dataset.movieId',
  );
  failFeedback = true;
  await click(".deck-actions .primary");
  await wait('document.querySelector("#deck .card-error button") !== null');
  assert.equal(
    await ev('document.querySelector(".deck-card").dataset.movieId'),
    current,
    "failed save retains card",
  );
  for (const status of [429, 409]) {
    failFeedback = true;
    feedbackStatus = status;
    await click("#deck .card-error button");
    await wait(
      `document.querySelector('#deck .card-error button') !== null && !document.querySelector('#deck .card-error button').disabled`,
    );
    assert.equal(
      await ev('document.querySelector(".deck-card").dataset.movieId'),
      current,
      "HTTP error retains card",
    );
  }
  await command("Network.enable");
  await command("Network.emulateNetworkConditions", {
    offline: true,
    latency: 0,
    downloadThroughput: 0,
    uploadThroughput: 0,
  });
  await click("#deck .card-error button");
  await wait(
    `document.querySelector('#deck .card-error button') !== null && !document.querySelector('#deck .card-error button').disabled`,
  );
  assert.equal(
    await ev('document.querySelector(".deck-card").dataset.movieId'),
    current,
    "network error retains card",
  );
  await command("Network.emulateNetworkConditions", {
    offline: false,
    latency: 0,
    downloadThroughput: -1,
    uploadThroughput: -1,
  });
  await click("#deck .card-error button");
  await wait(
    `document.querySelector('.deck-card').dataset.movieId !== '${current}'`,
  );
  assert.equal(feedback.get(Number(current)), "watchlist");
  await click('[data-view="watchlist"]');
  await wait(
    'document.querySelectorAll("#watchlist-grid .movie-card").length === 1',
  );
  await click("#watchlist-grid .muted");
  await wait(
    'document.querySelectorAll("#watchlist-grid .movie-card").length === 0',
  );
  assert.equal(feedback.size, 0);
  await click('[data-view="selection"]');
  const postCount = () =>
    calls.filter((c) => c[0] === "POST" && c[1] === "/feedback").length;
  const before = postCount();
  await ev(
    `(() => { const card=document.querySelector('.deck-card');card.setPointerCapture=()=>{}; for(const [type,x,y] of [['pointerdown',100,100],['pointerup',150,200]]) card.dispatchEvent(new PointerEvent(type,{pointerId:1,isPrimary:true,button:0,clientX:x,clientY:y})); })()`,
  );
  assert.equal(postCount(), before, "vertical movement does not commit");
  // Synthetic pointer capture is stubbed only because these events lack a hardware pointer.
  await ev(
    `(() => { const card=document.querySelector('.deck-card');card.setPointerCapture=()=>{}; for(const [type,x,y] of [['pointerdown',100,100],['pointerup',179,100]]) card.dispatchEvent(new PointerEvent(type,{pointerId:1,isPrimary:true,button:0,clientX:x,clientY:y})); })()`,
  );
  assert.equal(postCount(), before, "79px does not commit");
  await ev(
    `(() => { const card=document.querySelector('.deck-card');card.setPointerCapture=()=>{}; for(const [type,x,y] of [['pointerdown',100,100],['pointerup',180,160]]) card.dispatchEvent(new PointerEvent(type,{pointerId:1,isPrimary:true,button:0,clientX:x,clientY:y})); })()`,
  );
  assert.equal(
    postCount(),
    before,
    "insufficient horizontal dominance does not commit",
  );
  await ev(
    `(() => { const card=document.querySelector('.deck-card');card.setPointerCapture=()=>{}; for(const [type,x,y] of [['pointerdown',100,100],['pointercancel',200,100],['pointerup',200,100]]) card.dispatchEvent(new PointerEvent(type,{pointerId:1,isPrimary:true,button:0,clientX:x,clientY:y})); })()`,
  );
  assert.equal(postCount(), before, "pointer cancellation does not commit");
  await ev(
    `(() => { const card=document.querySelector('.deck-card');card.setPointerCapture=()=>{}; for(const [type,x,y] of [['pointerdown',100,100],['pointerup',180,100]]) card.dispatchEvent(new PointerEvent(type,{pointerId:1,isPrimary:true,button:0,clientX:x,clientY:y})); })()`,
  );
  await wait(
    `document.querySelector('#status').textContent === 'Фильм сохранён на вечер'`,
  );
  assert.equal(postCount(), before + 1, "80px horizontal commits");
  await ev(
    `document.querySelector('#search').value='Фильм';document.querySelector('#search').dispatchEvent(new Event('input'));`,
  );
  await delay(300);
  await ev(
    `document.querySelector('#search').value='';document.querySelector('#search').dispatchEvent(new Event('input'));`,
  );
  await delay(400);
  assert.equal(
    await ev('document.querySelector("#search-grid").children.length'),
    0,
    "cleared search stays empty",
  );
  await command("Emulation.setDeviceMetricsOverride", {
    width: 390,
    height: 844,
    deviceScaleFactor: 1,
    mobile: true,
  });
  assert.equal(
    await ev("document.documentElement.scrollWidth <= window.innerWidth"),
    true,
    "mobile no horizontal overflow",
  );
  assert.equal(
    await ev(
      'document.querySelector(".deck-actions .primary").getBoundingClientRect().bottom <= document.querySelector(".navigation").getBoundingClientRect().top',
    ),
    true,
    "mobile choice stays above bottom navigation",
  );
  await command("Page.reload");
  await wait(
    'document.querySelector("#recommendations").hidden === false && document.querySelector(".deck-card") !== null',
  );
  assert.equal(saved.size, 5, "history restored after reload");
  failHistory = true;
  await command("Page.reload");
  await wait('document.querySelector("#startup button") !== null');
  assert.equal(
    await ev('document.querySelector("#selection").hidden'),
    true,
    "failed restore does not start from zero",
  );
  failHistory = false;
  await click("#startup button");
  await wait('document.querySelector(".deck-card") !== null');
  // Poster failure is a one-shot local fallback, including hostile markup-like titles.
  await ev(
    `document.querySelector('#search').value='Фильм';document.querySelector('#search').dispatchEvent(new Event('input'));`,
  );
  await wait(
    'document.querySelectorAll("#search-grid .movie-card").length === 2',
  );
  slowHistory = true;
  slowWrite = true;
  await click("#search-grid .star");
  await click('[data-view="history"]');
  await wait(
    'document.querySelector("#progress-text").textContent.startsWith("6 из")',
  );
  await delay(700);
  assert.equal(
    await ev(
      'document.querySelector("#progress-text").textContent.startsWith("6 из")',
    ),
    true,
    "stale history cannot erase confirmed new rating",
  );
  slowHistory = false;
  slowWrite = false;
  await command("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-reduced-motion", value: "reduce" }],
  });
  assert.equal(
    await ev(
      'getComputedStyle(document.querySelector(".primary")).transitionDuration',
    ),
    "0s",
    "reduced motion respected",
  );
  unavailable = true;
  await command("Page.reload");
  await wait('document.querySelector("#deck button") !== null');
  assert.equal(
    await ev(
      'document.querySelector("#deck").textContent.includes("временно недоступна")',
    ),
    true,
    "unavailable is not exhaustion",
  );
  unavailable = false;
  await click("#deck button");
  await wait('document.querySelector(".deck-card") !== null');
  const skipFirst = await ev(
    'document.querySelector(".deck-card").dataset.movieId',
  );
  const skipPosts = calls.filter((c) => c[0] === "POST").length;
  const skipReads = calls.filter((c) => c[1] === "/recommendations").length;
  await ev(
    `document.querySelector('#deck').focus();document.querySelector('#deck').dispatchEvent(new KeyboardEvent('keydown',{key:' ',bubbles:true}));`,
  );
  assert.notEqual(
    await ev('document.querySelector(".deck-card").dataset.movieId'),
    skipFirst,
    "focused deck Space skips",
  );
  await ev(
    `for(let i=0;i<50;i++){const skip=document.querySelector('.deck-actions .secondary:nth-child(2)');if(skip)skip.click();}`,
  );
  assert.equal(
    await ev(
      'document.querySelector("#deck").textContent.includes("На сегодня всё")',
    ),
    true,
    "skips exhaust locally",
  );
  assert.equal(
    calls.filter((c) => c[0] === "POST").length,
    skipPosts,
    "skip has no mutation",
  );
  assert.equal(
    calls.filter((c) => c[1] === "/recommendations").length,
    skipReads,
    "skip exhaustion has no refetch loop",
  );
  await command("Page.reload");
  await wait('document.querySelector(".deck-card") !== null');
  assert.equal(
    await ev('document.querySelector(".deck-card").dataset.movieId'),
    skipFirst,
    "reload resets only local skips",
  );
  const watchedId = Number(
    await ev('document.querySelector(".deck-card").dataset.movieId'),
  );
  await click(".deck-actions .primary");
  await wait(
    `document.querySelector('.deck-card').dataset.movieId !== '${watchedId}'`,
  );
  await click('[data-view="watchlist"]');
  await wait(
    `document.querySelector('#watchlist-grid [data-movie-id="${watchedId}"]') !== null`,
  );
  const beforeWatched = saved.size;
  await click(`#watchlist-grid [data-movie-id="${watchedId}"] .text-button`);
  await click(`#watchlist-grid [data-movie-id="${watchedId}"] .star`);
  await wait(
    `document.querySelector('#watchlist-grid [data-movie-id="${watchedId}"]') === null`,
  );
  assert.equal(
    saved.size,
    beforeWatched + 1,
    "watched watchlist item becomes rating",
  );
  assert.equal(
    feedback.has(watchedId),
    false,
    "rating clears watchlist intention",
  );
  console.log(
    "PASS browser: onboarding 0->5; duplicate click; safe title; write failure/retry; watchlist/remove; 79/80px and vertical gestures; stale search; mobile overflow; reload history; failed history retry; stale history/write race; star keyboard/focus; reduced motion; local exhaustion/reload; watchlist watched",
  );
}
main()
  .catch((e) => {
    console.error(e);
    process.exitCode = 1;
  })
  .finally(async () => {
    ws?.close();
    browser?.kill();
    server?.close();
    // Temporary isolated Chrome profile is outside the repository; Chrome may still be closing.
    await delay(300);
    if (
      path.resolve(profile).startsWith(path.resolve(os.tmpdir()) + path.sep) &&
      path.basename(profile).startsWith("cinema-c-")
    ) {
      try {
        fs.rmSync(profile, { recursive: true, force: true });
      } catch {}
    }
  });
"""  # noqa: E501 - embedded JavaScript is formatted independently.


@pytest.mark.parametrize(
    "scenario",
    [
        "contract",
        "stale-history",
        "stale-watchlist",
        "keyboard-onboarding",
        "keyboard-history",
        "keyboard-search",
    ],
)
def test_cinema_browser_contract(scenario):
    import os
    import shutil
    import subprocess

    chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
    if not chrome.exists() or not shutil.which("node"):
        pytest.skip("Installed Chrome and Node are required for the local browser contract check")
    result = subprocess.run(
        ["node", "-e", BROWSER_CHECK, str(ROOT), str(chrome), scenario],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
        env={**os.environ, "NO_PROXY": "127.0.0.1,localhost"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS browser:" in result.stdout
    print(result.stdout.strip())
