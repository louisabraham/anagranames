# End-to-end tests of the website: a server for web/ and a browser that types like a visitor. Each test compares
# the page with app/anagram.py (the Python version of the search) on the same pack and query, and fails on page
# errors, console errors and failed requests.
# Run: node web/build.mjs, then
#   uv run --no-project --with playwright --with numpy python web/e2e.py [--browser chromium webkit] [--headed]
#     [--devices "iPhone SE" "Pixel 7"] [-k story]
# Without --devices, the phone and tablet tests run on the profiles of MOBILE; with "--devices" alone, not at all.
# The first time: uv run --no-project --with playwright playwright install chromium-headless-shell webkit
import argparse
import http.server
import json
import os
import sys
import tempfile
import threading
import time
from functools import partial
from urllib.parse import quote_plus

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/app")
import anagram  # noqa: E402

WAIT = 30_000  # ms
models = {}


def n(x):
    return f"{x:,}"


def people(x):
    return f"{n(x)} {'person' if x == 1 else 'people'}"


def model(pack):
    if pack not in models:
        models[pack] = anagram.Model(f"{ROOT}/app/data/{pack}.anag.gz")
    return models[pack]


def expected(source, query, limit=50):
    """The search of app/anagram.py in a source: world, FR-insee or a country code (in the world pack)."""
    pack = source if source in ("world", "FR-insee") else "world"
    return model(pack).search(query, None if pack == source else source, limit)


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


def check_results(page, pack, query, limit=50, real_limit=50):
    """The page shows the results of app/anagram.py for the query, in two panels: the real names (most people first,
    with their people) and the other names (most probable first). Returns the results of app/anagram.py: the real
    names, then the others."""
    real_names = expected(pack, query, 0)["real_names"]
    res = expected(pack, query, real_names + limit)
    real = [r for r in res["results"] if r["real"]][:real_limit]
    others = [r for r in res["results"] if not r["real"]][:limit]
    try:
        page.wait_for_function("k => document.getElementById('headline').dataset.key === k",
                               arg=f"{res['found']}|{res['real_names']}", timeout=WAIT)
    except PlaywrightTimeout:
        raise AssertionError(f"headline {page.text_content('#headline')!r}, status {page.text_content('#status')!r} after"
                             f" {WAIT // 1000} s, expected {res['found']} names and {res['real_names']} real")
    if res["real_people"]:
        headline = page.text_content("#headline")
        assert people(res["real_people"]) in headline, f"headline {headline!r}, expected {people(res['real_people'])}"
    for selector, want_rows in (("#results li", real), ("#possible li", others)):
        shown = page.eval_on_selector_all(selector, """lis => lis.map((li) => ({
            name: [...li.querySelectorAll('.name a')].map((a) => a.textContent).join(' '),
            value: li.querySelector('.detail a').textContent}))""")
        assert len(shown) == len(want_rows), f"{selector}: {len(shown)} names shown, expected {len(want_rows)}"
        for i, (row, r) in enumerate(zip(shown, want_rows), 1):
            assert row["name"] == f"{r['first']} {r['last']}", f"{selector} {i}: {row['name']!r}, expected {r['first']} {r['last']}"
            if r["real"]:
                assert row["value"] == people(r["real"]), f"{selector} {i} ({row['name']}): {row['value']}, expected {people(r['real'])}"
    return real + others


def link(base, query="", pack="world"):
    """The address of the anagrams of a query: ?anagrams=Yuki+Tanaka, with &in=JP for a source other than the
    world. A shared address without in= is in the world."""
    return f"{base}/?anagrams={quote_plus(query)}" + (f"&in={pack}" if pack != "world" else "")


def start(page, base, query="", pack="world"):
    page.goto(link(base, query, pack))
    page.wait_for_function("document.getElementById('from').value !== ''", timeout=WAIT)


# ---------- tests: each gets a new page (empty cache) and the address of the server ----------

def test_type_world(page, base):
    """Type a name letter by letter in the world: the real full names of 1 person are shown."""
    start(page, base)
    page.locator("#q").press_sequentially("Yuki Tanaka", delay=40)
    res = check_results(page, "world", "Yuki Tanaka")
    assert any(r["real"] == 1 for r in res), "no full name of 1 person in the reference"


def test_short_then_longer(page, base):
    """Fewer than 4 letters, then the 4th letter: the status changes from the hint to the results."""
    start(page, base)
    page.locator("#q").press_sequentially("Yuk", delay=40)
    wait_status(page, "Type at least 4 letters")
    page.locator("#q").press_sequentially("i", delay=40)
    check_results(page, "world", "Yuki")


def test_query_in_address(page, base):
    """A shared address with a query: the results show without typing."""
    start(page, base, "Branka Balevic")
    check_results(page, "world", "Branka Balevic")
    assert page.locator("#hero").is_hidden(), "the title shows with a query"


def test_country(page, base):
    """A country: the world data, the names seen in it."""
    start(page, base, "Marie Inuzuka", "JP")
    check_results(page, "JP", "Marie Inuzuka")


def test_insee(page, base):
    """France from INSEE: one pack file, with its real full names inside."""
    start(page, base, "Elena Ourgaud", "FR-insee")
    check_results(page, "FR-insee", "Elena Ourgaud")


def test_country_in_world(page, base):
    """A name of a country file is in the world file: the world results show it as your name."""
    for pack in ("JP", "world"):
        start(page, base, "Marie Inuzuka", pack)
        check_results(page, pack, "Marie Inuzuka")
        tags = page.eval_on_selector_all("#out-anagrams .tag.you", "ts => ts.length")
        assert tags == 1, f"{pack}: the query is not in the results"


def wait_names(page):
    """The summary of the name tab, when it is ready."""
    page.wait_for_function("""() => { const t = document.getElementById('summary').textContent;
        return t && !t.startsWith('Type') && !document.getElementById('status').textContent.startsWith('Loading'); }""",
                           timeout=WAIT)
    return page.text_content("#summary")


def check_names(page, label, total, countries, world):
    try:
        page.wait_for_function("t => document.getElementById('summary').textContent.includes(t)", arg=people(total),
                               timeout=WAIT)
    except PlaywrightTimeout:
        raise AssertionError(f"{label}: summary {page.text_content('#summary')!r} lacks {people(total)}")
    summary = wait_names(page)
    if page.locator("#all").is_visible():  # the first countries, then all of them
        page.click("#all")
        page.wait_for_function("k => document.querySelectorAll('#countries tbody tr').length === k", arg=len(countries),
                               timeout=WAIT)
    rows = page.eval_on_selector_all("#countries:not([hidden]) tbody tr", "trs => trs.map((tr) => tr.cells[1].textContent)")
    want = [n(x) for _, x in countries] if world else []
    assert rows == want, f"{label}: table {rows[:10]}, expected {want[:10]}"
    if not world:
        assert n(countries[0][1]) in summary, f"{label}: {summary!r} lacks the people in the country"


def test_tabs(page, base):
    """The 3 links of a result (first name, last name, probability) open the tabs of its first name, last name and
    full name in the page, with the people of app/anagram.py; the back button returns to the results, and the
    forward button to the name. The address bar does not change."""
    for pack in ("world", "JP"):
        start(page, base, "Yuki Tanaka", pack)
        r = check_results(page, pack, "Yuki Tanaka")[0]
        for i, (tab, total, countries) in enumerate([("first", r["first_people"], r["first_countries"]),
                                                     ("last", r["last_people"], r["last_countries"]),
                                                     ("full", r["real"], r["real_countries"])]):
            page.locator("#results li >> nth=0").locator("a").nth(i).click()
            assert page.get_attribute(f"#tab-{tab}", "aria-current") == "page", f"{pack}: the {tab} tab is not current"
            check_names(page, f"{pack} {tab}", total, countries, pack == "world")
            if tab == "full":
                assert page.locator("#model-panel").is_visible(), f"{pack}: no model estimate"
                assert page.text_content("#estimate").startswith("≈"), page.text_content("#estimate")
            page.go_back()
            check_results(page, pack, "Yuki Tanaka")
            assert page.url == f"{base}/", f"{pack}: the address is {page.url}"
            if tab == "last":
                page.go_forward()
                check_names(page, f"{pack} {tab} (forward)", total, countries, pack == "world")
                page.go_back()
                check_results(page, pack, "Yuki Tanaka")


def test_name_fields(page, base):
    """Type a first name in its tab, then a last name in the full name tab."""
    r = expected("world", "Yuki Tanaka")["results"][0]
    page.goto(link(base))
    page.click("#tab-first")
    page.wait_for_function("document.getElementById('summary').textContent.startsWith('Type')", timeout=WAIT)
    page.locator("#first").press_sequentially("Yuki", delay=40)
    check_names(page, "first", r["first_people"], r["first_countries"], True)
    page.click("#tab-full")
    page.locator("#last").press_sequentially("Tanaka", delay=40)
    check_names(page, "full", r["real"], r["real_countries"], True)
    assert page.locator("#first").input_value() == "Yuki", "the first name is lost between the tabs"


def test_choose_source(page, base):
    """Change the source with the list: type a country name, press Enter."""
    start(page, base, "Yuki Tanaka")
    check_results(page, "world", "Yuki Tanaka")
    page.click("#from")
    page.locator("#from").press_sequentially("Japan", delay=40)
    page.keyboard.press("Enter")
    check_results(page, "JP", "Yuki Tanaka")
    assert page.get_attribute("#share", "data-url") == link(base, "Yuki Tanaka", "JP"), page.get_attribute("#share", "data-url")


def test_copy_link(page, base):
    """The address bar stays clean; Copy link gives the short address of the view."""
    start(page, base)
    page.locator("#q").press_sequentially("Yuki Tanaka", delay=20)
    check_results(page, "world", "Yuki Tanaka")
    assert page.url == f"{base}/", f"the address changes while typing: {page.url}"
    assert page.get_attribute("#share", "data-url") == link(base, "Yuki Tanaka"), page.get_attribute("#share", "data-url")
    page.click("#share")
    page.wait_for_function("document.getElementById('share').textContent.startsWith('Copied')", timeout=WAIT)
    page.locator("#results li >> nth=0").locator("a").nth(2).click()
    wait_names(page)
    assert page.get_attribute("#share", "data-url") == f"{base}/?first=Yuki&last=Tanaka", page.get_attribute("#share", "data-url")
    page.goto(f"{base}/?first=Yuki&last=Tanaka")  # the copied address opens the same view
    try:
        page.wait_for_function("document.getElementById('tab-full').getAttribute('aria-current') === 'page'", timeout=WAIT)
    except PlaywrightTimeout:
        raise AssertionError("the address does not open the full name tab")
    wait_names(page)
    assert page.url == f"{base}/", f"the address is not cleaned: {page.url}"


def test_scroll_more(page, base):
    """Scrolling to the end of the lists loads 50 more names in each list."""
    start(page, base, "Yuki Tanaka")
    check_results(page, "world", "Yuki Tanaka")
    page.locator("#more").scroll_into_view_if_needed()
    page.wait_for_function("document.querySelectorAll('#possible li').length === 100", timeout=WAIT)
    check_results(page, "world", "Yuki Tanaka", 100, 100)


def test_fast_typing_and_clear(page, base):
    """Type fast (one search per letter, most of them stopped by the next letter), then clear the query."""
    start(page, base)
    page.locator("#q").press_sequentially("Wolfgang Amadeus Mozart")
    check_results(page, "world", "Wolfgang Amadeus Mozart")
    page.fill("#q", "")
    wait_status(page, "")
    assert page.locator("#results li, #possible li").count() == 0, "results stay after the query is cleared"


def test_new_build(page, base):
    """After a new build, the page asks for each file with the new version: no mix of two builds."""
    start(page, base, "Yuki Tanaka")
    check_results(page, "world", "Yuki Tanaka")
    page.route("**/data/version.json", lambda route: route.fulfill(json={"version": "newbuild"}))
    urls = []
    page.on("request", lambda r: urls.append(r.url))
    page.goto(link(base, "Yuki Tanaka"))
    check_results(page, "world", "Yuki Tanaka")
    files = [u for u in urls if u.endswith(".js") or ".js?" in u or "/data/" in u and "version.json" not in u]
    old = [u for u in files if not u.endswith("?v=newbuild")]
    assert files and not old, f"files without the new version: {old}"


def test_error_shown(page, base):
    """A file that fails to load gives an error in the status, not a status that stays."""
    page.route("**/data/world.real/**", lambda route: route.fulfill(status=503))
    start(page, base, "Yuki Tanaka")
    page.wait_for_function("document.getElementById('status').textContent.startsWith('Error: 503')", timeout=WAIT)


# ---------- the story of the home tab ----------

def story_numbers(query):
    """The numbers of the story of a query, from app/anagram.py in the world: the split into a first and a last
    name of the data with the largest probability, and the anagrams."""
    M, words, best = model("world"), query.split(), None
    for i in range(1, len(words)):
        f, l = M.lookup("first", M.key(" ".join(words[:i]))), M.lookup("last", M.key(" ".join(words[i:])))
        if f >= 0 and l >= 0:
            lp = float(M.log_p_mixture(anagram.np.array([f]), anagram.np.array([l]))[0][0])
            if best is None or lp > best[2]:
                best = (f, l, lp)
    res = expected("world", query, 0)
    out = {"real_people": res["real_people"], "others": res["found"] - res["real_names"], "names": None}
    if best:
        f, l, _ = best
        out["names"] = {"first": M.first.names[f], "last": M.last.names[l], "first_people": int(M.first.total[f]),
                        "last_people": int(M.last.total[l]), "real": sum(x for _, x in M.real_countries(f, l))}
    return out


def wait_step(page, k, count):
    """The claim of step k of a story of count steps, when it shows."""
    try:
        page.wait_for_function("k => document.querySelector('.slide .kicker')?.textContent.startsWith(k)",
                               arg=f"{k} / {count}", timeout=WAIT)
    except PlaywrightTimeout:
        raise AssertionError(f"step {k} / {count} does not show: {page.text_content('#story')!r}")
    return page.text_content(".slide .claim")


def test_story(page, base):
    """The story of a name: the question, one step per result with the numbers of app/anagram.py, the back and
    forward buttons and the arrow keys between the steps, then the tabs open."""
    page.goto(f"{base}/")
    page.wait_for_function("document.getElementById('from').value !== ''", timeout=WAIT)
    assert page.locator(".tabs").is_hidden(), "the tabs show before the story"
    page.fill("#q", "Yuki Tanaka")
    page.click("#start")
    s = story_numbers("Yuki Tanaka")
    wants = [n(s["names"]["first_people"]), n(s["names"]["last_people"]), n(s["names"]["real"]), people(s["real_people"]),
             n(s["others"])]
    for k, want in enumerate(wants, 1):
        claim = wait_step(page, k, 6)
        assert want in claim, f"step {k}: {claim!r} lacks {want}"
        if k == 2:
            page.go_back()
            wait_step(page, 1, 6)
            page.go_forward()
            wait_step(page, 2, 6)
        if k == 3:
            assert "≈" in page.text_content(".slide"), "no model estimate at the full name step"
        if k == 4:
            page.keyboard.press("ArrowRight")
        else:
            page.click("#next")
    wait_step(page, 6, 6)
    assert page.locator(".tabs").is_visible(), "the tabs do not show after the story"
    page.locator(".explore-links a").nth(0).click()
    check_results(page, "world", "Yuki Tanaka")


def test_story_split(page, base):
    """A name of three words: the story splits it as app/anagram.py does. A name that is not in the data has the
    steps of the anagrams only."""
    s = story_numbers("Jean Paul Sartre")
    page.goto(f"{base}/?Jean+Paul+Sartre")
    claim = wait_step(page, 1, 6)
    assert s["names"]["first"] in claim, f"{claim!r}: the first name is not {s['names']['first']}"
    page.click("#next")
    claim = wait_step(page, 2, 6)
    assert s["names"]["last"] in claim, f"{claim!r}: the last name is not {s['names']['last']}"
    page.goto(f"{base}/?Qwxz+Vbnjk")
    wait_step(page, 1, 3)


def test_story_link(page, base):
    """A shared address of a story opens its first step, with the tabs closed and a clean address bar."""
    page.goto(f"{base}/?Yuki+Tanaka")
    wait_step(page, 1, 6)
    assert page.locator(".tabs").is_hidden(), "the tabs show during the story"
    assert page.url == f"{base}/", f"the address is not cleaned: {page.url}"


def test_story_skip(page, base):
    """Skip the story: the anagrams of the name, and the tabs open; after a reload, the home tab shows the tabs."""
    page.goto(f"{base}/?Yuki+Tanaka")
    wait_step(page, 1, 6)
    page.click("text=Skip the story")
    check_results(page, "world", "Yuki Tanaka")
    assert page.locator(".tabs").is_visible(), "the tabs do not show after a skip"
    page.goto(f"{base}/")
    page.wait_for_function("document.getElementById('from').value !== ''", timeout=WAIT)
    assert page.locator(".tabs").is_visible(), "the tabs close again after a reload"


TESTS = [test_type_world, test_short_then_longer, test_query_in_address, test_country, test_insee,
         test_country_in_world, test_tabs, test_name_fields, test_choose_source, test_copy_link, test_scroll_more,
         test_fast_typing_and_clear, test_story, test_story_split, test_story_link, test_story_skip,
         test_new_build, test_error_shown]
EXPECT_ERRORS = {test_error_shown}


# ---------- phones and tablets: Playwright profiles (screen size, touch, user agent) ----------
MOBILE = ["iPhone SE", "iPhone 13", "iPhone 13 landscape", "Pixel 7", "iPad Mini"]
MEASURE = """() => {
  const vw = innerWidth, shown = (el) => el.getClientRects().length > 0, box = (el) => el.getBoundingClientRect();
  const label = (el) => `${el.tagName.toLowerCase()} ${(el.id || el.textContent).trim().slice(0, 24)} (${Math.round(box(el).width)}x${Math.round(box(el).height)})`;
  const out = { vw, width: document.scrollingElement.scrollWidth, wide: [], inputs: [], controls: [], links: [] };
  const scrolls = (el) => { // in a box of the screen that scrolls sideways (a formula): the reader can see it all
    for (let a = el.parentElement; a; a = a.parentElement)
      if (['auto', 'scroll'].includes(getComputedStyle(a).overflowX)) return box(a).right <= vw + 1;
    return false;
  };
  for (const el of document.querySelectorAll('header *, main *'))
    if (shown(el) && box(el).right > vw + 1 && !el.closest('.katex-mathml, details:not([open]), #from-list') && !scrolls(el)) out.wide.push(label(el));
  for (const el of document.querySelectorAll('input'))
    if (shown(el) && parseFloat(getComputedStyle(el).fontSize) < 16) out.inputs.push(label(el));
  for (const el of document.querySelectorAll('.tabs a, .top a, button, input'))
    if (shown(el) && box(el).height < 44) out.controls.push(label(el));
  for (const el of document.querySelectorAll('.name a, .detail a, .links a, .explore-links a, .story-nav a, .how a'))
    if (shown(el) && box(el).height < 32) out.links.push(label(el));
  const tabs = document.querySelector('.tabs');
  out.bar = tabs && shown(tabs) && getComputedStyle(tabs).position === 'fixed' ? box(tabs).height : 0;
  out.pad = parseFloat(getComputedStyle(document.body).paddingBottom);
  return out;
}"""
LAYOUT = [("question", "/", "document.getElementById('start').getClientRects().length"),
          ("story", "/?Louis+Abraham", "document.querySelector('.slide')"),
          ("anagrams", "/?anagrams=Louis+Abraham", "document.getElementById('headline').dataset.key"),
          ("first name", "/?first=Yuki", "!document.getElementById('countries').hidden"),
          ("full name", "/?first=Louis&last=Abraham", "!document.getElementById('model-countries').hidden"),
          ("how it works", "/how.html", "document.readyState === 'complete'")]


def test_mobile_layout(page, base):
    """For each view: no part wider than the screen, text of 16 px at least in the fields (else iOS zooms in),
    controls of 44 px and links of 32 px at least for a finger, and the tab bar at the bottom does not hide the end
    of the page."""
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
        assert m["pad"] >= m["bar"], f"{name}: the tab bar ({m['bar']} px) hides the end of the page ({m['pad']} px)"


def test_mobile_flow(page, base):
    """With a touch screen: the story with taps, then a name, the back button, more names on scroll, and a country
    of the list."""
    page.goto(f"{base}/")
    page.wait_for_function("document.getElementById('from').value !== ''", timeout=WAIT)
    page.tap("#q")
    page.keyboard.type("Yuki Tanaka")
    page.tap("#start")
    for k in range(1, 6):
        wait_step(page, k, 6)
        page.tap("#next")
    wait_step(page, 6, 6)
    assert page.locator(".tabs").is_visible(), "the tabs do not show after the story"
    page.tap(".explore-links a >> nth=0")
    r = check_results(page, "world", "Yuki Tanaka")[0]
    page.locator("#results li >> nth=0").locator("a").nth(0).tap()
    check_names(page, "first (tap)", r["first_people"], r["first_countries"], True)
    page.go_back()
    check_results(page, "world", "Yuki Tanaka")
    page.locator("#more").scroll_into_view_if_needed()
    page.wait_for_function("document.querySelectorAll('#possible li').length === 100", timeout=WAIT)
    page.tap("#from")
    page.keyboard.type("Japan")
    page.tap("#from-list li >> nth=0")
    check_results(page, "JP", "Yuki Tanaka")


MOBILE_TESTS = [test_mobile_layout, test_mobile_flow]


def run(browser, base, test, shots, device=None):
    context = browser.new_context(**(device or {}))
    context.route("**/favicon.ico", lambda route: route.fulfill(status=204))  # the site has no icon
    errors = []

    def watch(page):  # every page of the context, new tabs included
        page.on("pageerror", lambda e: errors.append(f"page error: {e}"))
        page.on("console", lambda m: m.type == "error" and errors.append(f"console: {m.text}"))
        page.on("response", lambda r: r.status >= 400 and errors.append(f"HTTP {r.status} for {r.url}"))

    context.on("page", watch)
    page = context.new_page()
    t = time.time()
    try:
        test(page, base)
        if test not in EXPECT_ERRORS:
            assert not errors, "; ".join(errors)
        return True, f"{time.time() - t:.1f}s"
    except (AssertionError, PlaywrightError) as e:
        path = f"{shots}/{browser.browser_type.name}-{test.__name__}{'-' + str(device['viewport']['width']) if device else ''}.png"
        page.screenshot(path=path, full_page=True)
        return False, f"{e}\n      screenshot: {path}" + (f"\n      also: {'; '.join(errors)}" if errors else "")
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", nargs="*", default=["chromium", "webkit"], help="chromium, webkit, firefox")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--devices", nargs="*", default=MOBILE, help="phone and tablet profiles of Playwright")
    parser.add_argument("-k", help="run the tests whose name contains this text")
    args = parser.parse_args()
    version = json.load(open(f"{ROOT}/web/data/version.json"))["version"]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=f"{ROOT}/web"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    shots = tempfile.mkdtemp(prefix="anagraname-e2e-")
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
