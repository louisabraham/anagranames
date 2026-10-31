# End-to-end tests of the website: a server for web/ and a browser that types like a visitor. Each test compares
# the page with app/anagram.py (the Python version of the search) on the same pack and query, and fails on page
# errors, console errors and failed requests.
# Run: node web/build.mjs, then
#   uv run --no-project --with playwright --with numpy python web/e2e.py [--browser chromium webkit] [--headed]
#     [--devices "iPhone SE" "Pixel 7"] [-k globe]
# Without --devices, the phone and tablet tests run on the profiles of MOBILE; with "--devices" alone, not at all.
# The first time: uv run --no-project --with playwright playwright install chromium-headless-shell webkit firefox
import argparse
import http.server
import json
import math
import os
import sys
import tempfile
import threading
import time
from collections import Counter
from functools import partial
from urllib.parse import quote_plus

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/app")
import anagram  # noqa: E402

WAIT = 30_000  # ms
SHOTS = tempfile.gettempdir()  # the screenshots (main makes a new folder)
models = {}


def n(x):
    return f"{x:,}"


def people(x):
    return f"{n(x)} {'person' if x == 1 else 'people'}"


def model(pack):
    if pack not in models:
        models[pack] = anagram.Model(f"{ROOT}/app/data/{pack}.anag.gz")
    return models[pack]


def population(source):
    """The people of a source, as the page counts them: the world, its country, or the one country of a pack."""
    pack = source if source in ("world", "FR-insee") else "world"
    meta = anagram.read_pack(f"{ROOT}/app/data/{pack}.anag.gz")[0]
    return sum(meta["population"]) if source == "world" else meta["population"][0 if pack == source else meta["countries"].index(source)]


def close(text, want):
    """A number of the page with 2 significant digits (1,200; 0.47; < 0.01), and its value."""
    text = text.replace("≈", "").strip()
    if text.startswith("<"):
        return want < float(text[1:]) * 1.0001
    return abs(float(text.replace(",", "")) - want) <= 0.06 * want + 0.006


def expected(source, query, limit=50):
    """The search of app/anagram.py in a source: world, FR-insee or a country code (in the world pack)."""
    pack = source if source in ("world", "FR-insee") else "world"
    return model(pack).search(query, None if pack == source else source, limit)


class Server(http.server.ThreadingHTTPServer):
    request_queue_size = 128  # the page asks for up to 100 files at once: with the 5 of Python, the system drops some


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):  # the browser closed the page during a download
            pass


def wait_status(page, text):
    try:
        page.wait_for_function("t => document.getElementById('status').textContent === t", arg=text, timeout=WAIT)
    except PlaywrightTimeout:
        raise AssertionError(f"status is {page.text_content('#status')!r} after {WAIT // 1000} s, expected {text!r}")


def check_list(page, source, query, synthetic=False, limit=50):
    """The page shows the people of app/anagram.py for the query: the real people (most people first, with their
    people), or the synthetic people (most probable first). Returns these results of app/anagram.py."""
    real_names = expected(source, query, 0)["real_names"]
    res = expected(source, query, real_names + limit)
    want = [r for r in res["results"] if bool(r["real"]) != synthetic][:limit]
    mode = "synthetic" if synthetic else "real"
    try:
        page.wait_for_function("([k, m]) => { const h = document.getElementById('headline'); return h.dataset.key === k && h.dataset.mode === m; }",
                               arg=[f"{res['found']}|{res['real_names']}", mode], timeout=WAIT)
    except PlaywrightTimeout:
        raise AssertionError(f"headline {page.text_content('#headline')!r}, status {page.text_content('#status')!r} after"
                             f" {WAIT // 1000} s, expected {res['found']} names, {res['real_names']} real, {mode}")
    headline = page.text_content("#headline")
    if not synthetic and res["real_people"]:
        label = f"{n(res['real_people'])} real {'person' if res['real_people'] == 1 else 'people'}"
        assert label in headline, f"headline {headline!r}, expected {label}"
    pop = population(source)
    if synthetic and res["found"] - res["real_names"] and res["found"] < 60_000:  # all the names: their people expected
        everyone = expected(source, query, res["found"])["results"]
        total = pop * sum(math.exp(r["log_p"]) for r in everyone if not r["real"])
        shown = page.text_content("#headline .big").split(" synthetic")[0]
        assert close(shown, total), f"headline {headline!r}, expected about {total:.3g} synthetic people"
    shown = page.eval_on_selector_all("#results li", """lis => lis.map((li) => ({
        name: li.querySelector('.first').textContent + ' ' + li.querySelector('.last').textContent,
        value: li.querySelector('.value').textContent}))""")
    assert len(shown) == len(want), f"{len(shown)} people shown, expected {len(want)}"
    for i, (row, r) in enumerate(zip(shown, want), 1):
        assert row["name"] == f"{r['first']} {r['last']}", f"{i}: {row['name']!r}, expected {r['first']} {r['last']}"
        if r["real"]:
            assert row["value"] == n(r["real"]), f"{i} ({row['name']}): {row['value']}, expected {n(r['real'])}"
        else:
            x = pop * math.exp(r["log_p"])
            assert close(row["value"], x), f"{i} ({row['name']}): {row['value']}, expected about {x:.3g} people"
    return want


def link(base, query="", pack="world", synthetic=False):
    """The address of a view: ?Yuki+Tanaka, with &in=JP for a source other than the world, &synthetic for the
    synthetic people."""
    parts = [quote_plus(query)] + ([f"in={pack}"] if pack != "world" else []) + (["synthetic"] if synthetic else [])
    return f"{base}/?{'&'.join(parts)}" if query else f"{base}/"


def start(page, base, query="", pack="world", synthetic=False):
    page.goto(link(base, query, pack, synthetic))
    page.wait_for_function("document.body.dataset.ready", timeout=WAIT)  # app.js has run


def wait_hist(page):
    page.wait_for_function("document.querySelector('#hist .hbar')", timeout=WAIT)


def bars(page):
    return page.eval_on_selector_all("#hist .hbar", "bs => bs.map((b) => [b.dataset.country, b.querySelector('.tip').textContent])")


# ---------- tests: each gets a new page (empty cache) and the address of the server ----------

def test_type_world(page, base):
    """Type a name letter by letter in the world: the real full names, of 1 person too."""
    start(page, base)
    page.locator("#q").press_sequentially("Yuki Tanaka", delay=40)
    res = check_list(page, "world", "Yuki Tanaka")
    assert any(r["real"] == 1 for r in res), "no full name of 1 person in the reference"


def test_short_then_longer(page, base):
    """Fewer than 4 letters, then the 4th letter: the status changes from the hint to the people."""
    start(page, base)
    page.locator("#q").press_sequentially("Yuk", delay=40)
    wait_status(page, "Type at least 4 letters")
    page.locator("#q").press_sequentially("i", delay=40)
    check_list(page, "world", "Yuki")


def test_query_in_address(page, base):
    """A shared address with a query (and the address of the old pages): the people show without typing, and the
    address bar is clean."""
    start(page, base, "Branka Balevic")
    check_list(page, "world", "Branka Balevic")
    assert page.url == f"{base}/", f"the address is not cleaned: {page.url}"
    page.goto(f"{base}/?anagrams=Branka+Balevic")
    check_list(page, "world", "Branka Balevic")


def test_switch(page, base):
    """The switch of the synthetic people: the list, its number and the countries follow; the back and forward
    buttons go between the two."""
    q = "Yuki Tanaka"
    res = expected("world", q, 0)
    start(page, base, q)
    check_list(page, "world", q)
    assert page.get_attribute("#synthetic", "aria-checked") == "false"
    assert page.text_content("#names") == n(res["real_names"]), f"names: {page.text_content('#names')}"
    page.click("#synthetic")
    check_list(page, "world", q, synthetic=True)
    assert page.get_attribute("#synthetic", "aria-checked") == "true"
    assert page.text_content("#names") == n(res["found"] - res["real_names"]), f"names: {page.text_content('#names')}"
    page.wait_for_function("document.getElementById('hist').dataset.mode === 'synthetic'", timeout=WAIT)
    page.go_back()
    check_list(page, "world", q)
    page.go_forward()
    check_list(page, "world", q, synthetic=True)


def test_country(page, base):
    """A country of the world by its address; its button gives the world again."""
    start(page, base, "Marie Inuzuka", "JP")
    check_list(page, "JP", "Marie Inuzuka")
    page.wait_for_selector("#to-world", timeout=WAIT)
    assert page.text_content("#to-world") == "Japan ×", page.text_content("#to-world")
    page.click("#to-world")
    check_list(page, "world", "Marie Inuzuka")
    assert page.locator("#to-world").count() == 0, "the button of the country stays"


def test_insee(page, base):
    """France from INSEE, by its address: one pack file; INSEE has no real full names, the synthetic people only."""
    start(page, base, "Elena Ourgaud", "FR-insee")
    check_list(page, "FR-insee", "Elena Ourgaud")
    page.click("#synthetic")
    check_list(page, "FR-insee", "Elena Ourgaud", synthetic=True)


def test_country_in_world(page, base):
    """A name of a country is in the world too: the list shows it as your name (in the list of its kind)."""
    q = "Marie Inuzuka"
    for pack in ("JP", "world"):
        res = expected(pack, q, 10_000)["results"]
        mine = next(r for r in res if f"{r['first']} {r['last']}" == q)
        start(page, base, q, pack, synthetic=not mine["real"])
        check_list(page, pack, q, synthetic=not mine["real"])
        tags = page.eval_on_selector_all("#results .tag.you", "ts => ts.length")
        assert tags == 1, f"{pack}: the query is not in the list"


def test_capitals(page, base):
    """A capital letter at the start of each word of a name, as the visitor types, and in a shared address."""
    start(page, base)
    page.locator("#q").press_sequentially("jean-paul sartre", delay=20)
    assert page.input_value("#q") == "Jean-Paul Sartre", page.input_value("#q")
    page.goto(f"{base}/?first=yuki&last=tanaka")
    page.wait_for_function("document.getElementById('q').value !== ''", timeout=WAIT)
    assert page.input_value("#q") == "Yuki Tanaka", page.input_value("#q")


def test_you(page, base):
    """The tag "you" is on the query only: Louis Abraham, not Louisa Braham (its letters in the same order)."""
    start(page, base, "Louis Abraham", synthetic=True)
    check_list(page, "world", "Louis Abraham", synthetic=True)
    page.click("#synthetic")
    check_list(page, "world", "Louis Abraham")
    page.locator("#more").scroll_into_view_if_needed()
    page.wait_for_function("document.querySelectorAll('#results li').length === 85", timeout=WAIT)
    tags = page.eval_on_selector_all(".tag.you", "ts => ts.map((t) => t.closest('li').querySelector('.first').textContent + ' ' + t.closest('li').querySelector('.last').textContent)")
    assert tags == ["Louis Abraham"], f"the tag is on {tags}"


def test_histogram(page, base):
    """The list of the countries, with the numbers of app/anagram.py: the real people per country, then the synthetic
    people expected per country (as in the view of the country: the population of the country times the probability
    of each name). A click on a country shows its people, and the button of the country the world again."""
    q = "Yuki Tanaka"
    start(page, base, q)
    check_list(page, "world", q)
    wait_hist(page)
    real_names = expected("world", q, 0)["real_names"]
    count = Counter()
    for r in expected("world", q, real_names)["results"]:
        for c, x in r["real_countries"]:
            count[c] += x
    shown = bars(page)
    want = [[c, n(x)] for c, x in sorted(count.items(), key=lambda t: (-t[1], t[0]))][:len(shown)]
    assert shown == want, f"real people per country: {shown[:4]}, expected {want[:4]}"
    page.click("#hist .hbar[data-country='JP']")
    check_list(page, "JP", q)
    page.wait_for_function("document.querySelector('#hist .hbar.chosen')?.dataset.country === 'JP'", timeout=WAIT)
    page.click("#to-world")
    check_list(page, "world", q)
    assert page.locator(".hbar.chosen").count() == 0, "a country stays chosen"
    page.click("#synthetic")
    check_list(page, "world", q, synthetic=True)
    page.wait_for_function("document.getElementById('hist').dataset.mode === 'synthetic'", timeout=WAIT)
    for c, text in bars(page)[:3]:  # the people expected with the synthetic names of the country
        res = expected(c, q, expected(c, q, 0)["found"])["results"]
        want = population(c) * sum(math.exp(r["log_p"]) for r in res if not r["real"])
        assert close(text, want), f"{c}: {text}, expected about {want:.3g} people"


def test_rows(page, base):
    """The rows of the two lists: their bars start on one line, whatever the width of their numbers (0.083 and 2.9);
    the number of the chosen country can be read on its cyan row; the names of the countries are short (Hong Kong,
    not Hong Kong SAR China)."""
    start(page, base, "Sayuli Drouard", synthetic=True)
    check_list(page, "world", "Sayuli Drouard", synthetic=True)
    wait_hist(page)
    for box in ("#hist", "#results"):
        lefts = page.eval_on_selector_all(f"{box} .htrack", "ts => ts.map((t) => Math.round(t.getBoundingClientRect().left))")
        assert len(set(lefts)) == 1, f"{box}: the bars start at {sorted(set(lefts))}"
    first = page.get_attribute("#hist .hbar >> nth=0", "data-country")
    page.click(f"#hist .hbar[data-country='{first}']")
    page.wait_for_function("document.querySelector('#hist .hbar.chosen')", timeout=WAIT)
    colors = page.evaluate("""(() => { const b = document.querySelector('#hist .hbar.chosen');
        return [getComputedStyle(b.querySelector('.tip')).color, getComputedStyle(b).backgroundColor, b.querySelector('.tip').textContent]; })()""")
    assert colors[0] != colors[1] and colors[2], f"the number of the chosen country: {colors[2]!r} in {colors[0]} on {colors[1]}"
    start(page, base, "Hong Kong")
    page.wait_for_function("document.getElementById('headline').dataset.key", timeout=WAIT)
    page.wait_for_function("document.querySelector('#hist .hbar') || document.getElementById('hist').dataset.mode", timeout=WAIT)
    page.click("#synthetic")
    wait_hist(page)
    page.wait_for_function("document.getElementById('hist').dataset.mode === 'synthetic'", timeout=WAIT)
    names = page.eval_on_selector_all("#hist .hname", "ns => ns.map((n) => n.textContent)")
    assert "Hong Kong" in names and not [x for x in names if "SAR" in x], f"names: {[x for x in names if 'Hong' in x or 'SAR' in x]}"


def test_globe(page, base):
    """The globe: a country chosen in the list of the countries is chosen on the globe, its box has the numbers of
    app/anagram.py; the pointer on a country shows its box; a click on a country of the globe chooses it, a click on
    it again the world; a drag turns the globe (the earth follows the pointer). A screenshot, path in the output."""
    q = "Yuki Tanaka"
    page.emulate_media(reduced_motion="reduce")  # still: the places on the screen do not move
    start(page, base, q)
    check_list(page, "world", q)
    page.wait_for_function("document.getElementById('globe').globe", timeout=WAIT)
    try:
        page.wait_for_function("""(() => { const c = document.querySelector('#globe canvas');
            if (!c.width || !c.height) return false;
            const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            let n = 0; for (let i = 3; i < d.length; i += 16) n += d[i] > 0; return n > 1000; })()""", timeout=WAIT)
    except PlaywrightTimeout:
        raise AssertionError("the globe is not drawn")
    page.click("#hist .hbar[data-country='JP']")
    check_list(page, "JP", q)
    page.wait_for_function("document.getElementById('globe').dataset.chosen === 'JP'", timeout=WAIT)
    page.wait_for_timeout(300)
    tip = page.text_content(".globe-tip")
    for want in ("Japan", "Real people", n(expected("JP", q, 0)["real_people"]), "Synthetic, expected"):
        assert want in tip, f"the box of Japan {tip!r} lacks {want}"

    def place(code):
        page.evaluate("document.getElementById('globe').scrollIntoView({ block: 'center' })")
        canvas = page.locator("#globe canvas").bounding_box()
        w = page.evaluate(f"document.getElementById('globe').globe.where('{code}')")
        assert w, f"{code} is not on the screen"
        return canvas["x"] + w["x"], canvas["y"] + w["y"]

    page.mouse.move(*place("CN"))
    page.wait_for_function("document.getElementById('globe').dataset.hover === 'CN'", timeout=WAIT)
    page.wait_for_function("document.querySelector('.globe-tip').textContent.includes('China')", timeout=WAIT)
    path = f"{SHOTS}/{page.context.browser.browser_type.name}-globe.png"
    page.locator("#globe").screenshot(path=path)
    page.mouse.click(*place("CN"))
    check_list(page, "CN", q)
    page.wait_for_function("document.getElementById('globe').dataset.chosen === 'CN'", timeout=WAIT)
    page.wait_for_timeout(300)
    page.mouse.click(*place("CN"))
    check_list(page, "world", q)
    page.evaluate("document.getElementById('globe').scrollIntoView({ block: 'center' })")
    canvas = page.locator("#globe canvas").bounding_box()
    lon, r = page.evaluate("[document.getElementById('globe').globe.lon, document.getElementById('globe').globe.R]")
    x, y = canvas["x"] + canvas["width"] / 2, canvas["y"] + canvas["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    for i in range(1, 11):
        page.mouse.move(x - 10 * i, y)
    page.wait_for_timeout(200)  # no throw: the turn stops
    page.mouse.up()
    turned = page.evaluate("document.getElementById('globe').globe.lon") - lon
    want = 100 / (r * 3.141592653589793 / 180)
    assert abs(turned - want) < 0.15 * want, f"a drag of 100 px turns the globe {turned:.1f} degrees, expected {want:.1f}"
    return f"screenshot of the globe: {path}"


GLOBE = """() => { const box = document.getElementById('globe'), g = box.globe, c = box.querySelector('canvas'), dpr = devicePixelRatio;
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let lit = 0; for (let i = 3; i < d.length; i += 64) lit += d[i] > 0;
  return { w: box.clientWidth, h: box.clientHeight, cw: c.width / dpr, ch: c.height / dpr, R: g.R, cx: g.cx, cy: g.cy, lit }; }"""


def check_globe(page, label):
    """The globe fills its box at the size of the screen: the canvas has the size of the box (in pixels of the
    screen), the globe is a circle in the middle of it, and it is drawn (on the screen: else it draws no frames)."""
    try:
        page.wait_for_function("""(() => { const b = document.getElementById('globe'), c = b.querySelector('canvas');
            return b.globe && Math.abs(c.width / devicePixelRatio - b.clientWidth) < 1.5 && Math.abs(c.height / devicePixelRatio - b.clientHeight) < 1.5; })()""",
                               timeout=WAIT)
    except PlaywrightTimeout:
        raise AssertionError(f"{label}: the canvas does not follow its box: {page.evaluate(GLOBE)}")
    page.evaluate("document.getElementById('globe').scrollIntoView({ block: 'center' })")  # off the screen: no frames
    try:
        page.wait_for_function(f"({GLOBE})().lit > 200", timeout=WAIT)
    except PlaywrightTimeout:
        pass  # the assertion below gives the numbers
    m = page.evaluate(GLOBE)
    small = min(m["w"], m["h"])
    assert m["w"] >= 200 and m["h"] >= 200, f"{label}: the globe is {m['w']}x{m['h']} px"
    assert small / 2 - 30 <= m["R"] <= small / 2, f"{label}: radius {m['R']:.0f} px in a box of {m['w']}x{m['h']} px"
    assert abs(m["cx"] - m["w"] / 2) < 1 and abs(m["cy"] - m["h"] / 2) < 1, f"{label}: the globe is not in the middle: {m}"
    assert m["lit"] > 200, f"{label}: the globe is not drawn ({m['lit']} points)"


def test_globe_resize(page, base):
    """The globe at each size of the window, from a wide screen to a phone (the window changes size, the page stays):
    it fills its box, and the box of a country under the pointer stays in the globe."""
    page.emulate_media(reduced_motion="reduce")
    start(page, base, "Yuki Tanaka")
    check_list(page, "world", "Yuki Tanaka")
    for w, h in [(1600, 1000), (1180, 760), (960, 900), (600, 900), (320, 640)]:
        page.set_viewport_size({"width": w, "height": h})
        check_globe(page, f"{w}x{h}")
        page.evaluate("document.getElementById('globe').scrollIntoView({ block: 'center' })")
        box = page.locator("#globe").bounding_box()
        where = page.evaluate("document.getElementById('globe').globe.where('FR') || document.getElementById('globe').globe.where('DE')")
        if where:
            page.mouse.move(box["x"] + where["x"], box["y"] + where["y"])
            page.wait_for_function("!document.querySelector('.globe-tip').hidden", timeout=WAIT)
            tip = page.locator(".globe-tip").bounding_box()
            inside = tip["x"] >= box["x"] - 1 and tip["y"] >= box["y"] - 1 and tip["x"] + tip["width"] <= box["x"] + box["width"] + 1 \
                and tip["y"] + tip["height"] <= box["y"] + box["height"] + 1
            assert inside, f"{w}x{h}: the box of the country {tip} is out of the globe {box}"
            page.mouse.move(0, 0)


def test_address_of_a_view(page, base):
    """A shared address opens its view: a name, a country, the synthetic people; the address bar is cleaned."""
    page.goto(f"{base}/?Yuki+Tanaka&in=JP&synthetic")
    check_list(page, "JP", "Yuki Tanaka", synthetic=True)
    assert page.url == f"{base}/", f"the address is not cleaned: {page.url}"


def test_scroll_more(page, base):
    """Scrolling to the end of the list loads 50 more people."""
    start(page, base, "Yuki Tanaka", synthetic=True)
    check_list(page, "world", "Yuki Tanaka", synthetic=True)
    page.locator("#more").scroll_into_view_if_needed()
    page.wait_for_function("document.querySelectorAll('#results li').length === 100", timeout=WAIT)
    check_list(page, "world", "Yuki Tanaka", synthetic=True, limit=100)


def test_fast_typing_and_clear(page, base):
    """Type fast (one search per letter, most of them stopped by the next letter), then clear the query."""
    start(page, base)
    page.locator("#q").press_sequentially("Wolfgang Amadeus Mozart")
    check_list(page, "world", "Wolfgang Amadeus Mozart")
    page.fill("#q", "")
    wait_status(page, "")
    assert page.locator("#results li").count() == 0, "people stay after the query is cleared"
    assert page.locator("#view").is_hidden(), "the globe and the list stay after the query is cleared"


def test_reload_resets(page, base):
    """A reload starts again: no name, no people, the top of the page."""
    start(page, base, "Yuki Tanaka", "JP", synthetic=True)
    check_list(page, "JP", "Yuki Tanaka", synthetic=True)
    page.evaluate("scrollTo(0, 500)")
    page.reload()
    page.wait_for_function("document.body.dataset.ready", timeout=WAIT)
    state = page.evaluate("[document.getElementById('q').value, document.getElementById('view').hidden, scrollY, document.getElementById('synthetic').getAttribute('aria-checked')]")
    assert state == ["", True, 0, "false"], f"after a reload: field {state[0]!r}, view hidden {state[1]}, scroll {state[2]}, switch {state[3]}"


def test_new_build(page, base):
    """After a new build, the page asks for each file with the new version: no mix of two builds."""
    start(page, base, "Yuki Tanaka")
    check_list(page, "world", "Yuki Tanaka")
    page.route("**/data/version.json", lambda route: route.fulfill(json={"version": "newbuild"}))
    urls = []
    page.on("request", lambda r: urls.append(r.url))
    page.goto(link(base, "Yuki Tanaka"))
    check_list(page, "world", "Yuki Tanaka")
    files = [u for u in urls if u.endswith(".js") or ".js?" in u or "/data/" in u and "version.json" not in u]
    old = [u for u in files if not u.endswith("?v=newbuild")]
    assert files and not old, f"files without the new version: {old}"


def test_error_shown(page, base):
    """A file that fails to load gives an error in the status, not a status that stays."""
    page.route("**/data/world.real/**", lambda route: route.fulfill(status=503))
    start(page, base, "Yuki Tanaka")
    page.wait_for_function("document.getElementById('status').textContent.startsWith('Error: 503')", timeout=WAIT)


TESTS = [test_type_world, test_short_then_longer, test_query_in_address, test_switch, test_country, test_insee,
         test_country_in_world, test_capitals, test_you, test_histogram, test_rows, test_globe, test_globe_resize,
         test_address_of_a_view, test_scroll_more,
         test_fast_typing_and_clear, test_reload_resets, test_new_build, test_error_shown]
EXPECT_ERRORS = {test_error_shown}


# ---------- phones and tablets: Playwright profiles (screen size, touch, user agent) ----------
MOBILE = ["iPhone SE", "iPhone 13", "iPhone 13 landscape", "Pixel 7", "iPad Mini"]
MEASURE = """() => {
  const vw = innerWidth, shown = (el) => el.getClientRects().length > 0, box = (el) => el.getBoundingClientRect();
  const label = (el) => `${el.tagName.toLowerCase()} ${(el.id || el.textContent).trim().slice(0, 24)} (${Math.round(box(el).width)}x${Math.round(box(el).height)})`;
  const out = { vw, width: document.scrollingElement.scrollWidth, wide: [], inputs: [], controls: [], links: [] };
  const scrolls = (el) => { // in a box of the screen that scrolls sideways (a formula), or that ends its line with
    for (let a = el.parentElement; a; a = a.parentElement) { // "...": the reader sees that there is more
      const css = getComputedStyle(a);
      if (['auto', 'scroll'].includes(css.overflowX) || (css.overflowX === 'hidden' && css.textOverflow === 'ellipsis'))
        return box(a).right <= vw + 1;
    }
    return false;
  };
  for (const el of document.querySelectorAll('header *, main *'))
    if (shown(el) && box(el).right > vw + 1 && !el.closest('.katex-mathml, details:not([open])') && !scrolls(el)) out.wide.push(label(el));
  for (const el of document.querySelectorAll('input'))
    if (shown(el) && parseFloat(getComputedStyle(el).fontSize) < 16) out.inputs.push(label(el));
  for (const el of document.querySelectorAll('.top a, button, input'))
    if (shown(el) && box(el).height < 44) out.controls.push(label(el));
  for (const el of document.querySelectorAll('.how a'))
    if (shown(el) && box(el).height < 32) out.links.push(label(el));
  return out;
}"""
READY = "document.getElementById('headline').dataset.key && document.querySelector('#hist .hbar')"
LAYOUT = [("empty", "/", "document.body.dataset.ready"),
          ("real people", "/?Louis+Abraham", READY),
          ("synthetic people", "/?Louis+Abraham&synthetic", READY),
          ("a country", "/?Yuki+Tanaka&in=JP", READY),
          ("how it works", "/how.html", "document.readyState === 'complete'")]


def test_mobile_layout(page, base):
    """For each view: no part wider than the screen, text of 16 px at least in the fields (else iOS zooms in),
    controls of 44 px and links of 32 px at least for a finger, the globe in its box, the top bar at the top after a
    scroll."""
    for name, url, ready in LAYOUT:
        page.goto(base + url)
        page.wait_for_function(ready, timeout=WAIT)
        page.wait_for_timeout(300)
        if name == "how it works":
            page.evaluate("document.querySelectorAll('details').forEach((d) => (d.open = true))")
        m = page.evaluate(MEASURE)
        assert m["width"] <= m["vw"], f"{name}: the page is {m['width']} px wide on a screen of {m['vw']} px: {m['wide'][:4]}"
        assert not m["wide"], f"{name}: wider than the screen: {m['wide'][:4]}"
        assert not m["inputs"], f"{name}: text smaller than 16 px in {m['inputs']}"
        assert not m["controls"], f"{name}: controls smaller than 44 px: {m['controls'][:6]}"
        assert not m["links"], f"{name}: links smaller than 32 px: {m['links'][:6]}"
        if ready == READY:
            check_globe(page, name)
        page.evaluate("scrollTo(0, document.scrollingElement.scrollHeight)")
        page.wait_for_timeout(300)
        top = page.evaluate("document.querySelector('.top').getBoundingClientRect().top")
        assert top == 0, f"{name}: the top bar is not at the top of the screen after a scroll ({top} px)"


def test_mobile_flow(page, base):
    """With a touch screen: type a name, the switch, a country of the list, more people on scroll, the back button."""
    q = "Yuki Tanaka"
    start(page, base)
    page.tap("#q")
    page.keyboard.type(q)
    check_list(page, "world", q)
    wait_hist(page)
    page.tap("#hist .hbar[data-country='JP']")  # Japan: in the first countries of the real people
    check_list(page, "JP", q)
    page.tap("#synthetic")
    check_list(page, "JP", q, synthetic=True)
    page.locator("#more").scroll_into_view_if_needed()
    page.wait_for_function("document.querySelectorAll('#results li').length === 100", timeout=WAIT)
    page.go_back()
    check_list(page, "JP", q)


MOBILE_TESTS = [test_mobile_layout, test_mobile_flow]


def run(browser, base, test, shots, device=None):
    context = browser.new_context(**(device or {}))  # no route here: in Firefox, a route can stop the fonts (load)
    errors, pending = [], {}  # pending: the requests without an answer yet (shown if the test fails)

    def watch(page):  # every page of the context, new tabs included
        page.on("request", lambda r: pending.__setitem__(r, time.time()))
        page.on("requestfinished", lambda r: pending.pop(r, None))
        page.on("requestfailed", lambda r: pending.pop(r, None))
        page.on("pageerror", lambda e: errors.append(f"page error: {e}"))
        page.on("console", lambda m: m.type == "error" and errors.append(f"console: {m.text}"))
        page.on("response", lambda r: r.status >= 400 and errors.append(f"HTTP {r.status} for {r.url}"))

    context.on("page", watch)
    page = context.new_page()
    t = time.time()
    try:
        note = test(page, base)
        if test not in EXPECT_ERRORS:
            assert not errors, "; ".join(errors)
        return True, f"{time.time() - t:.1f}s" + (f"  {note}" if note else "")
    except (AssertionError, PlaywrightError) as e:
        path = f"{shots}/{browser.browser_type.name}-{test.__name__}{'-' + str(device['viewport']['width']) if device else ''}.png"
        page.screenshot(path=path, full_page=True)
        waits = [f"{r.url.replace(base, '')} ({time.time() - t0:.0f} s)" for r, t0 in pending.items()]
        return False, (f"{e}\n      screenshot: {path}" + (f"\n      also: {'; '.join(errors)}" if errors else "")
                       + (f"\n      no answer yet: {', '.join(waits[:6])}" if waits else ""))
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", nargs="*", default=["chromium", "webkit", "firefox"], help="chromium, webkit, firefox")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--devices", nargs="*", default=MOBILE, help="phone and tablet profiles of Playwright")
    parser.add_argument("-k", help="run the tests whose name contains this text")
    args = parser.parse_args()
    version = json.load(open(f"{ROOT}/web/data/version.json"))["version"]
    server = Server(("127.0.0.1", 0), partial(Quiet, directory=f"{ROOT}/web"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    global SHOTS
    shots = SHOTS = tempfile.mkdtemp(prefix="anagraname-e2e-")
    tests = [t for t in TESTS if not args.k or args.k in t.__name__]
    mobile = [t for t in MOBILE_TESTS if not args.k or args.k in t.__name__]
    print(f"web/ build {version} at {base}, {len(tests)} tests x {len(args.browser)} browsers,"
          f" {len(mobile)} tests x {len(args.devices)} devices")
    failed = 0
    with sync_playwright() as p:
        for name in args.browser:
            browser = getattr(p, name).launch(headless=not args.headed)
            for test in tests:
                ok, info = run(browser, base, test, shots)
                failed += not ok
                print(f"{'PASS' if ok else 'FAIL'}  {name:8s} {test.__name__}  {info}", flush=True)
            browser.close()
        for dev in args.devices if mobile else []:
            d = p.devices[dev]
            browser = getattr(p, d["default_browser_type"]).launch(headless=not args.headed)
            for test in mobile:
                ok, info = run(browser, base, test, shots, {k: v for k, v in d.items() if k != "default_browser_type"})
                failed += not ok
                print(f"{'PASS' if ok else 'FAIL'}  {dev:20s} {test.__name__}  {info}", flush=True)
            browser.close()
    server.shutdown()
    print(f"{failed} failed" if failed else "all passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
