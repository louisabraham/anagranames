# End-to-end tests of the website: a server for web/ and a browser that types like a visitor. Each test compares
# the page with app/anagram.py (the Python version of the search) on the same pack and query, and fails on page
# errors, console errors and failed requests.
# Run: node web/build.mjs, then
#   uv run --no-project --with playwright --with numpy python web/e2e.py [--browser chromium webkit] [--headed]
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


def expected(source, query, limit=50):
    """The search of app/anagram.py in a source: world, FR-insee or a country code (in the world pack)."""
    pack = source if source in ("world", "FR-insee") else "world"
    if pack not in models:
        models[pack] = anagram.Model(f"{ROOT}/app/data/{pack}.anag.gz")
    return models[pack].search(query, None if pack == source else source, limit)


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


def check_results(page, pack, query, limit=50):
    """The page shows the results of app/anagram.py for the query: status, headline, names and real people."""
    res = expected(pack, query, limit)
    wait_status(page, f"{n(res['found'])} name{'' if res['found'] == 1 else 's'} · {n(res['real_names'])} real")
    if res["real_people"]:
        headline = page.text_content("#headline")
        assert people(res["real_people"]) in headline, f"headline {headline!r}, expected {people(res['real_people'])}"
    shown = page.eval_on_selector_all("#results li", """lis => lis.map((li) => ({
        name: li.querySelector('.name').firstChild.textContent,
        tags: [...li.querySelectorAll('.tag')].map((t) => t.textContent)}))""")
    assert len(shown) == len(res["results"]), f"{len(shown)} results shown, expected {len(res['results'])}"
    for i, (row, r) in enumerate(zip(shown, res["results"]), 1):
        assert row["name"] == f"{r['first']} {r['last']}", f"result {i}: {row['name']!r}, expected {r['first']} {r['last']}"
        real = [t for t in row["tags"] if t.startswith("real")]
        want = [f"real · {people(r['real'])}"] if r["real"] else []
        assert real == want, f"result {i} ({row['name']}): tags {real}, expected {want}"
    order = [r["real"] for r in res["results"]]
    assert order == sorted(order, reverse=True), f"the real full names are not first, most people first: {order}"
    return res


def start(page, base, query="", pack="world"):
    page.goto(f"{base}/?{'q=' + query.replace(' ', '+') + '&' if query else ''}from={pack}")
    page.wait_for_function("document.getElementById('from').value !== ''", timeout=WAIT)


# ---------- tests: each gets a new page (empty cache) and the address of the server ----------

def test_type_world(page, base):
    """Type a name letter by letter in the world: the real full names of 1 person are shown."""
    start(page, base)
    page.locator("#q").press_sequentially("Yuki Tanaka", delay=40)
    res = check_results(page, "world", "Yuki Tanaka")
    assert any(r["real"] == 1 for r in res["results"]), "no full name of 1 person in the reference"


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
    origin = page.text_content("#origin")
    assert origin, "no origin line"


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
        tags = page.eval_on_selector_all("#results li .tag.you", "ts => ts.length")
        assert tags == 1, f"{pack}: the query is not in the results"


def test_explore(page, base):
    """Click a result: 3 links open the pages of its first name, last name and full name in new tabs, with the
    people in the world and per country of app/anagram.py. Enter closes the links."""
    for pack in ("world", "JP"):
        start(page, base, "Yuki Tanaka", pack)
        r = check_results(page, pack, "Yuki Tanaka")["results"][0]
        page.click("#results li >> nth=0")
        links = page.locator("#results li >> nth=0").locator(".explore a")
        assert links.count() == 3, f"{pack}: {links.count()} links"
        world = pack == "world"
        for i, (total, countries) in enumerate([(r["first_people"], r["first_countries"]),
                                                (r["last_people"], r["last_countries"]),
                                                (r["real"], r["real_countries"])]):
            with page.context.expect_page() as info:
                links.nth(i).click()
            tab = info.value
            try:
                tab.wait_for_function("!['', 'Loading...'].includes(document.getElementById('summary').textContent)",
                                      timeout=WAIT)
                summary = tab.text_content("#summary")
                assert not summary.startswith("Error"), summary
                assert people(total) in summary, f"{pack} link {i + 1}: summary {summary!r} lacks {people(total)}"
                rows = tab.eval_on_selector_all("#countries tbody tr", "trs => trs.map((tr) => tr.cells[1].textContent)")
                want = [n(x) for _, x in countries] if world else []
                assert rows == want, f"{pack} link {i + 1}: table {rows[:10]}, expected {want[:10]}"
                if not world:
                    assert n(countries[0][1]) in summary, f"{pack} link {i + 1}: {summary!r} lacks the people in {pack}"
            finally:
                tab.close()
        page.locator("#results li >> nth=0").press("Enter")
        assert page.locator("#results .explore").count() == 0, f"{pack}: Enter does not close the links"


def test_choose_source(page, base):
    """Change the source with the list: type a country name, press Enter."""
    start(page, base, "Yuki Tanaka")
    check_results(page, "world", "Yuki Tanaka")
    page.click("#from")
    page.locator("#from").press_sequentially("Japan", delay=40)
    page.keyboard.press("Enter")
    check_results(page, "JP", "Yuki Tanaka")
    assert "from=JP" in page.url, page.url


def test_more(page, base):
    """The More button adds 50 names."""
    start(page, base, "Yuki Tanaka")
    check_results(page, "world", "Yuki Tanaka")
    page.click("#more")
    page.wait_for_function("document.querySelectorAll('#results li').length === 100", timeout=WAIT)
    check_results(page, "world", "Yuki Tanaka", 100)


def test_fast_typing_and_clear(page, base):
    """Type fast (one search per letter, most of them stopped by the next letter), then clear the query."""
    start(page, base)
    page.locator("#q").press_sequentially("Wolfgang Amadeus Mozart")
    check_results(page, "world", "Wolfgang Amadeus Mozart")
    page.fill("#q", "")
    wait_status(page, "")
    assert page.locator("#results li").count() == 0, "results stay after the query is cleared"


def test_new_build(page, base):
    """After a new build, the page asks for each file with the new version: no mix of two builds."""
    start(page, base, "Yuki Tanaka")
    check_results(page, "world", "Yuki Tanaka")
    page.route("**/data/version.json", lambda route: route.fulfill(json={"version": "newbuild"}))
    urls = []
    page.on("request", lambda r: urls.append(r.url))
    page.reload()
    check_results(page, "world", "Yuki Tanaka")
    files = [u for u in urls if u.endswith(".js") or ".js?" in u or "/data/" in u and "version.json" not in u]
    old = [u for u in files if not u.endswith("?v=newbuild")]
    assert files and not old, f"files without the new version: {old}"


def test_error_shown(page, base):
    """A file that fails to load gives an error in the status, not a status that stays."""
    page.route("**/data/world.real/**", lambda route: route.fulfill(status=503))
    start(page, base, "Yuki Tanaka")
    page.wait_for_function("document.getElementById('status').textContent.startsWith('Error: 503')", timeout=WAIT)


TESTS = [test_type_world, test_short_then_longer, test_query_in_address, test_country, test_insee,
         test_country_in_world, test_explore, test_choose_source, test_more, test_fast_typing_and_clear,
         test_new_build, test_error_shown]
EXPECT_ERRORS = {test_error_shown}


def run(browser, base, test, shots):
    context = browser.new_context()
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
    except AssertionError as e:
        path = f"{shots}/{browser.browser_type.name}-{test.__name__}.png"
        page.screenshot(path=path, full_page=True)
        return False, f"{e}\n      screenshot: {path}" + (f"\n      also: {'; '.join(errors)}" if errors else "")
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", nargs="*", default=["chromium", "webkit"], help="chromium, webkit, firefox")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("-k", help="run the tests whose name contains this text")
    args = parser.parse_args()
    version = json.load(open(f"{ROOT}/web/data/version.json"))["version"]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=f"{ROOT}/web"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    shots = tempfile.mkdtemp(prefix="anagraname-e2e-")
    tests = [t for t in TESTS if not args.k or args.k in t.__name__]
    print(f"web/ build {version} at {base}, {len(tests)} tests")
    failed = 0
    with sync_playwright() as p:
        for name in args.browser:
            browser = getattr(p, name).launch(headless=not args.headed)
            for test in tests:
                ok, info = run(browser, base, test, shots)
                failed += not ok
                print(f"{'PASS' if ok else 'FAIL'}  {name:8s} {test.__name__}  {info}", flush=True)
            browser.close()
    server.shutdown()
    print(f"{failed} failed" if failed else "all passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
