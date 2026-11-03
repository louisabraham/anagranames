# End-to-end tests of the website: a server for web/ and a browser that types like a visitor. Each test compares
# the page with app/anagram.py (the Python version of the search) on the same pack and query, and fails on page
# errors, console errors and failed requests.
# Run: node web/build.mjs, then
#   uv run --no-project --with playwright --with numpy python web/e2e.py [-k globe,keys] [--all] [-j 6]
# By default: Chromium only, no phones (while working, with -k: the tests of the change). --all (CI, and before a
# commit): also WebKit and Firefox (the tests of ENGINE only, unless -k) and the phone and tablet profiles of MOBILE.
# --browser and --devices choose them one by one. The tests run in parallel processes (-j).
# The first time: uv run --no-project --with playwright playwright install chromium-headless-shell webkit firefox
import argparse
import functools
import http.server
import json
import math
import os
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    """The people of a source, as the page counts them: the world or a country."""
    meta = anagram.read_pack(f"{ROOT}/app/data/world.anag.gz")[0]
    return sum(meta["population"]) if source == "world" else meta["population"][meta["countries"].index(source)]


def close(text, want):
    """A number of the page with 2 significant digits (1,200; 0.47; < 0.01), and its value."""
    text = text.replace("≈", "").strip()
    if text.startswith("<"):
        return want < float(text[1:]) * 1.0001
    return abs(float(text.replace(",", "")) - want) <= 0.06 * want + 0.006


@functools.cache  # a test asks for the same search several times (the list, the message, the countries)
def expected(source, query, limit=50, by_model=False):
    """The search of app/anagram.py in a source: world or a country code; by_model: all the names, the most probable
    first. The callers only read the result."""
    return model("world").search(query, None if source == "world" else source, limit, by_model)


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


def check_message(page, query, synthetic):
    """The message at the top of the globe: about the world (whatever the country chosen or with the focus), the real
    names; with the synthetic names, the names generated too."""
    w = expected("world", query, 0)
    rn, rp, z = w["real_names"], w["real_people"], w["found"] - w["real_names"]
    real = [
        f"Found {n(rn)} real anagraname{'' if rn == 1 else 's'} on {n(rp)} Facebook account{'' if rp == 1 else 's'}"
    ]
    gen = f"{n(z)} synthetic anagraname{'s' if z > 1 else ''} generated" if z else "No synthetic anagraname generated"
    want = real + [gen] if synthetic else real
    try:
        page.wait_for_function(
            "w => JSON.stringify([...document.querySelectorAll('#tally p')].map((p) => p.textContent)) === w",
            arg=json.dumps(want, separators=(",", ":"), ensure_ascii=False),
            timeout=WAIT,
        )
    except PlaywrightTimeout:
        lines = page.eval_on_selector_all("#tally p", "ps => ps.map((p) => p.textContent)")
        raise AssertionError(f"message {lines}, expected {want}")


def check_list(page, source, query, synthetic=False, limit=50):
    """The page shows the people of app/anagram.py for the query: the real people (most people first, with their
    people), or with the synthetic people all the names (most probable first, the people that the model expects with
    them, and a label of its real people on a real name). Returns these results of app/anagram.py."""
    real_names = expected(source, query, 0)["real_names"]
    res = expected(source, query, limit, by_model=True) if synthetic else expected(source, query, real_names + limit)
    want = res["results"][:limit] if synthetic else [r for r in res["results"] if r["real"]][:limit]
    mode = "synthetic" if synthetic else "real"
    try:
        page.wait_for_function(
            "([k, m]) => { const h = document.getElementById('headline'); return h.dataset.key === k && h.dataset.mode === m; }",
            arg=[f"{res['found']}|{res['real_names']}", mode],
            timeout=WAIT,
        )
    except PlaywrightTimeout:
        raise AssertionError(
            f"headline {page.text_content('#headline')!r}, status {page.text_content('#status')!r} after"
            f" {WAIT // 1000} s, expected {res['found']} names, {res['real_names']} real, {mode}"
        )
    headline = page.text_content("#headline")
    if not synthetic and res["real_people"]:
        label = f"{n(res['real_people'])} real {'person' if res['real_people'] == 1 else 'people'}"
        assert label in headline, f"headline {headline!r}, expected {label}"
    check_message(page, query, synthetic)
    pop = population(source)
    if synthetic and res["found"] and res["found"] < 60_000:  # all the names: their people expected
        everyone = expected(source, query, res["found"])["results"]
        total = pop * sum(math.exp(r["log_p"]) for r in everyone)
        shown = page.text_content("#headline .big").split(" people")[0]
        assert close(shown, total), f"headline {headline!r}, expected about {total:.3g} people"
    shown = page.eval_on_selector_all(
        "#results li",
        """lis => lis.map((li) => ({
        name: li.querySelector('.first').textContent + ' ' + li.querySelector('.last').textContent,
        value: li.querySelector('.value').textContent, tag: li.querySelector('.tag.real')?.textContent ?? ''}))""",
    )
    assert len(shown) == len(want), f"{len(shown)} people shown, expected {len(want)}"
    for i, (row, r) in enumerate(zip(shown, want), 1):
        assert row["name"] == f"{r['first']} {r['last']}", f"{i}: {row['name']!r}, expected {r['first']} {r['last']}"
        if not synthetic:
            assert row["value"] == n(r["real"]), f"{i} ({row['name']}): {row['value']}, expected {n(r['real'])}"
            continue
        x = pop * math.exp(r["log_p"])
        assert close(row["value"], x), f"{i} ({row['name']}): {row['value']}, expected about {x:.3g} people"
        label = f"{n(r['real'])} real" if r["real"] else ""
        assert row["tag"] == label, f"{i} ({row['name']}): label {row['tag']!r}, expected {label!r}"
    return want


def link(base, query="", pack="world", synthetic=False):
    """The address of a view: ?Yuki+Tanaka, with &in=JP for a source other than the world, &real for the real people
    only (the synthetic people are the default of the page)."""
    parts = (
        ([quote_plus(query)] if query else [])
        + ([f"in={pack}"] if pack != "world" else [])
        + ([] if synthetic else ["real"])
    )
    return f"{base}/?{'&'.join(parts)}" if parts else f"{base}/"


def start(page, base, query="", pack="world", synthetic=False):
    # the HTML only: app.js says when it is ready (in Firefox, the load event can wait long when a test has a route)
    page.goto(link(base, query, pack, synthetic), wait_until="domcontentloaded")
    page.wait_for_function("document.body.dataset.ready", timeout=WAIT)  # app.js has run


def wait_hist(page):
    page.wait_for_function("document.querySelector('#hist .hbar')", timeout=WAIT)


def bars(page):
    return page.eval_on_selector_all(
        "#hist .hbar", "bs => bs.map((b) => [b.dataset.country, b.querySelector('.tip').textContent])"
    )


# ---------- tests: each gets a new page (empty cache) and the address of the server ----------


def test_about(page, base):
    """A first visit: the synthetic people (the default), and a text under the question that says what the page does;
    a name to try searches it and the text goes away; a cleared field shows it again. The headers of the two lists
    give the unit of their numbers."""
    page.goto(f"{base}/")
    page.wait_for_function("document.body.dataset.ready", timeout=WAIT)
    assert page.locator("#about").is_visible(), "no text for a first visit"
    assert page.get_attribute("#synthetic", "aria-checked") == "true", "the synthetic people are not the default"
    page.click("#about .try button >> nth=0")
    check_list(page, "world", "Arsène Lupin", synthetic=True)
    assert page.input_value("#q") == "Arsène Lupin", page.input_value("#q")
    assert page.locator("#about").is_hidden(), "the text for a first visit stays with the people"
    wait_hist(page)
    units = page.evaluate("[...document.querySelectorAll('.pane-head .unit')].map((u) => u.textContent)")
    assert units == ["People expected"] * 2, f"units: {units}"
    page.click("#synthetic")
    check_list(page, "world", "Arsène Lupin")
    page.wait_for_function("document.getElementById('hist').dataset.mode === 'real'", timeout=WAIT)
    units = page.evaluate("[...document.querySelectorAll('.pane-head .unit')].map((u) => u.textContent)")
    assert units == ["Accounts"] * 2, f"units: {units}"
    page.click("#clear")
    page.wait_for_selector("#about", state="visible", timeout=WAIT)


def test_type_world(page, base):
    """Type a name letter by letter in the world: fewer than 4 letters give a hint; then the real full names, of 1
    person too; a capital letter at the start of each word."""
    start(page, base)
    page.locator("#q").press_sequentially("yuk", delay=40)
    wait_status(page, "Type at least 4 letters")
    page.locator("#q").press_sequentially("i tanaka", delay=40)
    assert page.input_value("#q") == "Yuki Tanaka", page.input_value("#q")
    res = check_list(page, "world", "Yuki Tanaka")
    assert any(r["real"] == 1 for r in res), "no full name of 1 person in the reference"


def test_capitals(page, base):
    """A capital letter at the start of each word, only on the text that the visitor adds: delete the A of Abc, then
    type it again: Abc (not ABc); a space typed in a word gives a capital letter to the next word; a paste too."""
    start(page, base)
    field = page.locator("#q")
    value = lambda: page.input_value("#q")  # noqa: E731
    caret = lambda i: page.evaluate(f"document.getElementById('q').setSelectionRange({i}, {i})")  # noqa: E731
    field.press_sequentially("abc", delay=20)
    assert value() == "Abc", value()
    caret(1)
    page.keyboard.press("Backspace")
    assert value() == "bc", f"a deletion changes the other letters: {value()!r}"
    caret(0)
    page.keyboard.type("a")
    assert value() == "Abc", f"the A again: {value()!r}"
    page.fill("#q", "")
    field.press_sequentially("louisabraham", delay=20)
    caret(5)
    page.keyboard.type(" ")
    assert value() == "Louis Abraham", f"a space in a word: {value()!r}"
    page.fill("#q", "")
    page.keyboard.insert_text("jean-paul sartre")  # as a paste
    assert value() == "Jean-Paul Sartre", f"a paste: {value()!r}"


def test_address(page, base):
    """A shared address opens its view without typing, and the address bar is cleaned: a name, a country, the real
    people only (&real). Without &real: the synthetic people (the default). The addresses of the old pages
    (?anagrams=..., ?first=...&last=..., &synthetic) too, with a capital letter at the start of each word."""
    page.goto(f"{base}/?Yuki+Tanaka&in=JP&real")
    check_list(page, "JP", "Yuki Tanaka")
    assert page.url == f"{base}/", f"the address is not cleaned: {page.url}"
    page.goto(f"{base}/?anagrams=Branka+Balevic")
    check_list(page, "world", "Branka Balevic", synthetic=True)
    assert page.get_attribute("#synthetic", "aria-checked") == "true"
    page.goto(f"{base}/?first=jean-paul&last=sartre&synthetic")
    page.wait_for_function("document.getElementById('q').value !== ''", timeout=WAIT)
    assert page.input_value("#q") == "Jean-Paul Sartre", page.input_value("#q")
    assert page.get_attribute("#synthetic", "aria-checked") == "true"


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
    assert page.text_content("#names") == n(res["found"]), f"names: {page.text_content('#names')}"
    page.wait_for_function("document.getElementById('hist').dataset.mode === 'synthetic'", timeout=WAIT)
    page.go_back()
    check_list(page, "world", q)
    page.go_forward()
    check_list(page, "world", q, synthetic=True)


def test_country(page, base):
    """A country of the world by its address, chosen in the list; a click on it gives the world again."""
    start(page, base, "Marie Inuzuka", "JP")
    check_list(page, "JP", "Marie Inuzuka")
    page.wait_for_function("document.querySelector('#hist .hbar.chosen')?.dataset.country === 'JP'", timeout=WAIT)
    page.click("#hist .hbar.chosen")  # the chosen country again: the world
    check_list(page, "world", "Marie Inuzuka")
    assert page.locator("#hist .hbar.chosen").count() == 0, "a country stays chosen"


def test_country_in_world(page, base):
    """A real name of a country is real in the world too: both lists show it as your name. A name real elsewhere
    only is in no list of the country (Marie Inuzuka: not real in Japan, not synthetic either)."""
    q = "Yuki Tanaka"
    jp = expected("JP", "Marie Inuzuka", 10_000)["results"]
    assert not [r for r in jp if f"{r['first']} {r['last']}" == "Marie Inuzuka"], "Marie Inuzuka in a list of Japan"
    for pack in ("JP", "world"):
        res = expected(pack, q, 10_000)["results"]
        mine = next(r for r in res if f"{r['first']} {r['last']}" == q)
        start(page, base, q, pack, synthetic=not mine["real"])
        check_list(page, pack, q, synthetic=not mine["real"])
        tags = page.eval_on_selector_all("#results li.you", "ts => ts.length")
        assert tags == 1, f"{pack}: the query is not in the list"


def test_you(page, base):
    """Your name has its own color, on the query only: Louis Abraham, not Louisa Braham (its letters in the same
    order)."""
    start(page, base, "Louis Abraham", synthetic=True)
    check_list(page, "world", "Louis Abraham", synthetic=True)
    page.click("#synthetic")
    check_list(page, "world", "Louis Abraham")
    page.locator("#more").scroll_into_view_if_needed()
    page.wait_for_function("document.querySelectorAll('#results li').length === 85", timeout=WAIT)
    tags = page.eval_on_selector_all(
        "#results li.you",
        "ls => ls.map((l) => l.querySelector('.first').textContent + ' ' + l.querySelector('.last').textContent)",
    )
    assert tags == ["Louis Abraham"], f"the color of your name is on {tags}"
    colors = page.evaluate(
        """[...document.querySelectorAll('#results li')].slice(0, 3).map((l) => getComputedStyle(l.querySelector('.name')).color)"""
    )
    assert len(set(colors)) == 2, f"your name has the color of the others: {colors}"


def test_histogram(page, base):
    """The list of the countries, with the numbers of app/anagram.py: the real people per country, then the synthetic
    people expected per country (as in the view of the country: the population of the country times the probability
    of each name). A click on a country shows its people, and a click on it again the world."""
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
    want = [[c, n(x)] for c, x in sorted(count.items(), key=lambda t: (-t[1], t[0]))][: len(shown)]
    assert shown == want, f"real people per country: {shown[:4]}, expected {want[:4]}"
    page.click("#hist .hbar[data-country='JP']")
    check_list(page, "JP", q)
    page.wait_for_function("document.querySelector('#hist .hbar.chosen')?.dataset.country === 'JP'", timeout=WAIT)
    page.click("#hist .hbar.chosen")
    check_list(page, "world", q)
    assert page.locator(".hbar.chosen").count() == 0, "a country stays chosen"
    page.click("#synthetic")
    check_list(page, "world", q, synthetic=True)
    page.wait_for_function("document.getElementById('hist').dataset.mode === 'synthetic'", timeout=WAIT)
    for c, text in bars(page)[:3]:  # the people expected with all the names of the view of the country
        res = expected(c, q, expected(c, q, 0)["found"])["results"]
        want = population(c) * sum(math.exp(r["log_p"]) for r in res)
        assert close(text, want), f"{c}: {text}, expected about {want:.3g} people"


def test_synthetic_nowhere_real(page, base):
    """A synthetic person has a name that nobody in the data has, in any country: Daniel Huynh (23 real people, none
    in South Korea) is not a synthetic person of South Korea."""
    start(page, base, "Daniel Huynh", "KR", synthetic=True)
    check_list(page, "KR", "Daniel Huynh", synthetic=True)
    names = page.eval_on_selector_all(
        "#results li",
        "ls => ls.map((l) => l.querySelector('.first').textContent + ' ' + l.querySelector('.last').textContent)",
    )
    assert "Daniel Huynh" not in names, f"a real name in the synthetic people of South Korea: {names[:5]}"


def test_real_among_synthetic(page, base):
    """With the synthetic people, the real names are in the list too, at their place for the model, with a label of
    their real people: Yuki Tanaka (41 real people)."""
    start(page, base, "Yuki Tanaka", synthetic=True)
    rows = check_list(page, "world", "Yuki Tanaka", synthetic=True)
    assert any(r["real"] for r in rows) and any(not r["real"] for r in rows), "no real and synthetic names together"
    page.wait_for_function("document.querySelector('#results li.you .tag.real')", timeout=WAIT)
    assert page.text_content("#results li.you .tag.real") == "41 real", page.text_content("#results li.you .tag.real")


def test_names_whole(page, base):
    """The names of people are whole (not cut), with their labels: the bars give the room. On a computer, in both
    kinds of people (Louis Abraham). The countries keep a fixed part for their names (a long one is cut)."""
    for synthetic in (False, True):
        start(page, base, "Louis Abraham", synthetic=synthetic)
        check_list(page, "world", "Louis Abraham", synthetic=synthetic)
        wait_hist(page)
        cut = page.evaluate("""[...document.querySelectorAll('#results .who')]
            .filter((e) => e.scrollWidth > e.clientWidth + 1).map((e) => e.textContent)""")
        assert not cut, f"cut names{' (synthetic)' if synthetic else ''}: {cut[:5]}"


def test_keyboard(page, base):
    """The arrows in the lists, without Tab (no row in the order of Tab): Arrow Down in the field goes to the countries;
    with the world, up and down only move; Enter chooses the country, then up and down choose the next one at once;
    Space gives the world again (the arrows only move again); right and left go from a list to the other; End to the
    last name (the next names load, the focus stays)."""
    q = "Yuki Tanaka"
    start(page, base, q, synthetic=True)
    check_list(page, "world", q, synthetic=True)
    wait_hist(page)
    stops = page.evaluate("[...document.querySelectorAll('.hbar, #results li')].filter((r) => r.tabIndex >= 0).length")
    assert stops == 0, f"{stops} rows in the order of Tab"
    countries = page.eval_on_selector_all("#hist .hbar", "bs => bs.map((b) => b.dataset.country)")
    focus = "c => document.activeElement?.dataset?.country === c"

    def none_chosen(label):
        page.wait_for_timeout(400)
        assert page.locator("#hist .hbar.chosen").count() == 0, f"{label}: a country is chosen"

    page.focus("#q")
    page.keyboard.press("ArrowDown")
    page.wait_for_function(focus, arg=countries[0], timeout=WAIT)
    page.keyboard.press("ArrowDown")
    page.wait_for_function(focus, arg=countries[1], timeout=WAIT)
    none_chosen("the world, Arrow Down")
    check_message(page, q, True)  # a country with the focus: the message of the world
    page.keyboard.press("Enter")
    check_list(page, countries[1], q, synthetic=True)
    page.wait_for_function(focus, arg=countries[1], timeout=WAIT)
    page.keyboard.press("ArrowDown")
    check_list(page, countries[2], q, synthetic=True)
    page.wait_for_function(focus, arg=countries[2], timeout=WAIT)
    page.keyboard.press("Space")
    check_list(page, "world", q, synthetic=True)
    page.wait_for_function(focus, arg=countries[2], timeout=WAIT)
    page.keyboard.press("ArrowDown")
    page.wait_for_function(focus, arg=countries[3], timeout=WAIT)
    none_chosen("the world again, Arrow Down")
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowDown")
    index = page.evaluate("[...document.querySelectorAll('#results li')].indexOf(document.activeElement)")
    assert index == 2, f"the focus is on the name {index}, expected 2"
    page.keyboard.press("ArrowLeft")
    page.wait_for_function(focus, arg=countries[3], timeout=WAIT)
    page.keyboard.press("Escape")  # no focus: the message of the view
    page.go_back()  # before Space: the country of the arrows
    check_list(page, countries[2], q, synthetic=True)
    page.go_back()  # before Enter: the world
    check_list(page, "world", q, synthetic=True)
    page.click("#results li >> nth=0")
    page.keyboard.press("End")
    page.wait_for_function("document.querySelectorAll('#results li').length > 50", timeout=WAIT)
    page.wait_for_timeout(300)
    assert page.evaluate("document.activeElement.closest('#results') !== null"), (
        "the focus leaves the list when the next names load"
    )


def test_keys(page, base):
    """The key S: the switch, but not in the field (a letter of a name); Enter and Escape: the field lets the focus go;
    then the arrows go to the lists (down: the countries, right: the names, also after a click on the globe); Escape: a
    country lets the focus go, and a chosen country gives the world again."""
    q = "Yuki Tanaka"
    start(page, base)
    page.focus("#q")
    page.keyboard.type(q)
    check_list(page, "world", q)
    page.keyboard.press("Enter")
    assert page.evaluate("document.activeElement.id") != "q", "Enter: the field keeps the focus"
    page.keyboard.press("s")
    check_list(page, "world", q, synthetic=True)
    page.keyboard.press("S")
    check_list(page, "world", q)
    page.focus("#q")
    page.keyboard.press("End")
    page.keyboard.type("s")  # a letter of the name: no switch
    page.wait_for_timeout(400)
    assert page.get_attribute("#synthetic", "aria-checked") == "false", "S in the field turns the switch"
    page.keyboard.press("Backspace")
    page.keyboard.press("Escape")
    assert page.evaluate("document.activeElement.id") != "q", "Escape: the field keeps the focus"
    check_list(page, "world", q)
    wait_hist(page)
    first = page.get_attribute("#hist .hbar >> nth=0", "data-country")
    page.keyboard.press("ArrowDown")  # out of the field: the arrows go to the countries
    page.wait_for_function("c => document.activeElement?.dataset?.country === c", arg=first, timeout=WAIT)
    page.keyboard.press("Escape")  # a country lets the focus go
    assert page.evaluate("document.activeElement === document.body"), "Escape: the country keeps the focus"
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")  # a country chosen, then Escape: the world again, and no focus
    check_list(page, first, q)
    page.keyboard.press("Escape")
    check_list(page, "world", q)
    assert page.evaluate("document.activeElement === document.body"), "Escape: the chosen country keeps the focus"
    box = page.locator("#globe canvas").bounding_box()
    page.mouse.click(box["x"] + 6, box["y"] + 6)  # a click on the globe (out of the sphere), then the arrows
    page.keyboard.press("ArrowRight")
    assert page.evaluate("document.activeElement.closest('#results') !== null"), (
        "Arrow Right after a click on the globe: not the names"
    )


def test_switch_country(page, base):
    """The switch with a chosen country: a country with people of both kinds stays chosen; a country without people
    of the new kind is not chosen any more (the world again). Yuki Tanaka: Japan has both, Afghanistan synthetic
    people only."""
    q = "Yuki Tanaka"
    start(page, base, q, "JP", synthetic=True)
    check_list(page, "JP", q, synthetic=True)
    wait_hist(page)
    page.click("#synthetic")
    check_list(page, "JP", q)
    page.wait_for_function("document.querySelector('#hist .hbar.chosen')?.dataset.country === 'JP'", timeout=WAIT)
    start(page, base, q, "AF", synthetic=True)
    check_list(page, "AF", q, synthetic=True)
    wait_hist(page)
    page.click("#synthetic")
    check_list(page, "world", q)
    page.wait_for_function("document.getElementById('hist').dataset.mode === 'real'", timeout=WAIT)
    assert page.locator("#hist .hbar.chosen").count() == 0, "a country without real people stays chosen"
    page.go_back()  # the back button: Afghanistan and its synthetic people again
    check_list(page, "AF", q, synthetic=True)


def test_rows(page, base):
    """The rows of the two lists: their bars start on one line, whatever the width of their numbers (0.083 and 2.9);
    the number of the chosen country can be read on its cyan row; the names of the countries are short (Hong Kong,
    not Hong Kong SAR China)."""
    start(page, base, "Sayuli Drouard", synthetic=True)
    check_list(page, "world", "Sayuli Drouard", synthetic=True)
    wait_hist(page)
    for box in ("#hist", "#results"):
        lefts = page.eval_on_selector_all(
            f"{box} .htrack", "ts => ts.map((t) => Math.round(t.getBoundingClientRect().left))"
        )
        assert len(set(lefts)) == 1, f"{box}: the bars start at {sorted(set(lefts))}"
    first = page.get_attribute("#hist .hbar >> nth=0", "data-country")
    page.click(f"#hist .hbar[data-country='{first}']")
    page.wait_for_function("document.querySelector('#hist .hbar.chosen')", timeout=WAIT)
    colors = page.evaluate("""(() => { const b = document.querySelector('#hist .hbar.chosen');
        return [getComputedStyle(b.querySelector('.tip')).color, getComputedStyle(b).backgroundColor, b.querySelector('.tip').textContent]; })()""")
    assert colors[0] != colors[1] and colors[2], (
        f"the number of the chosen country: {colors[2]!r} in {colors[0]} on {colors[1]}"
    )
    start(page, base, "Hong Kong")
    page.wait_for_function("document.getElementById('headline').dataset.key", timeout=WAIT)
    page.wait_for_function(
        "document.querySelector('#hist .hbar') || document.getElementById('hist').dataset.mode", timeout=WAIT
    )
    page.click("#synthetic")
    wait_hist(page)
    page.wait_for_function("document.getElementById('hist').dataset.mode === 'synthetic'", timeout=WAIT)
    names = page.eval_on_selector_all("#hist .hname", "ns => ns.map((n) => n.textContent)")
    assert "Hong Kong" in names and not [x for x in names if "SAR" in x], (
        f"names: {[x for x in names if 'Hong' in x or 'SAR' in x]}"
    )


def test_headers_stay(page, base):
    """The headers of the two lists stay in place while their lists scroll; the scroll bars: a line and a rectangle."""
    start(page, base, "Yuki Tanaka", synthetic=True)
    check_list(page, "world", "Yuki Tanaka", synthetic=True)
    wait_hist(page)
    for box in ("#hist", "#people"):
        head = page.locator(f"{box} .pane-head").bounding_box()
        page.evaluate(f"document.querySelector('{box} .pane-body').scrollTop = 300")
        page.wait_for_timeout(200)
        moved = page.evaluate(f"document.querySelector('{box} .pane-body').scrollTop")
        assert moved > 0, f"{box}: the list does not scroll"
        after = page.locator(f"{box} .pane-head").bounding_box()
        assert abs(after["y"] - head["y"]) < 1, f"{box}: the header moves with the list ({head['y']} -> {after['y']})"
    # the scroll bar of the page: a line and a rectangle; the bar of the browser hidden; the rectangle follows the list,
    # and a drag of it scrolls the list
    for box in ("#hist", "#people"):
        state = page.evaluate(f"""(() => {{ const b = document.querySelector('{box} .pane-body'), t = document.querySelector('{box} .sthumb');
            return [b.offsetWidth - b.clientWidth, !t.closest('.sbar').hidden, t.getBoundingClientRect().top - t.closest('.sbar').getBoundingClientRect().top]; }})()""")
        assert state[0] == 0, f"{box}: the scroll bar of the browser shows ({state[0]} px)"
        assert state[1] and state[2] > 1, f"{box}: the rectangle of the scroll bar does not follow the list: {state}"
        page.evaluate(f"document.querySelector('{box} .pane-body').scrollTop = 0")
        thumb = page.locator(f"{box} .sthumb").bounding_box()
        page.mouse.move(thumb["x"] + thumb["width"] / 2, thumb["y"] + 5)
        page.mouse.down()
        page.mouse.move(thumb["x"] + thumb["width"] / 2, thumb["y"] + 105, steps=5)
        page.mouse.up()
        moved = page.evaluate(f"document.querySelector('{box} .pane-body').scrollTop")
        assert moved > 100, f"{box}: a drag of the rectangle scrolls the list {moved} px"
    widths = page.evaluate(
        "['#hist', '#people'].map((q) => Math.round(document.querySelector(q).getBoundingClientRect().width))"
    )
    assert abs(widths[0] - widths[1]) <= 1, f"the two lists are {widths} px wide"


def test_definition(page, base):
    """The definition of the word: hidden, then open while the mouse is over the word, closed when it leaves; a click
    keeps it open, Escape closes it."""
    start(page, base, "Arsene Lupin")
    check_list(page, "world", "Arsene Lupin")
    assert page.locator("#entry").is_hidden(), "the definition shows before a hover"
    page.hover("#term")
    page.wait_for_selector("#entry", state="visible", timeout=WAIT)
    assert "neologism" in page.text_content("#entry") and "Paul Sernine" in page.text_content("#entry")
    page.mouse.move(2, page.viewport_size["height"] - 2)
    page.wait_for_selector("#entry", state="hidden", timeout=WAIT)
    page.click("#term")
    page.mouse.move(2, page.viewport_size["height"] - 2)
    page.wait_for_timeout(400)
    assert page.locator("#entry").is_visible(), "a click does not keep the definition open"
    page.keyboard.press("Escape")
    page.wait_for_selector("#entry", state="hidden", timeout=WAIT)


def test_fits_the_window(page, base):
    """On a computer, the page is the height of the window and no more (no scroll of the page), with the real or the
    synthetic people, and with a country."""
    start(page, base, "Arsene Lupin")
    check_list(page, "world", "Arsene Lupin")
    for w, h in [(1280, 650), (1920, 1080)]:  # the smallest computer screen, and a large one
        page.set_viewport_size({"width": w, "height": h})
        for step in ("real", "synthetic", "country"):
            if step == "synthetic":
                page.click("#synthetic")
                check_list(page, "world", "Arsene Lupin", synthetic=True)
            if step == "country":
                page.click("#hist .hbar >> nth=1")
                page.wait_for_function("document.querySelector('#hist .hbar.chosen')", timeout=WAIT)
            try:  # the page fits after its resize code (a slow browser, as WebKit on CI: more than 200 ms)
                page.wait_for_function("document.scrollingElement.scrollHeight <= innerHeight", timeout=5000)
            except PlaywrightTimeout:
                size = page.evaluate("[document.scrollingElement.scrollHeight, innerHeight]")
                raise AssertionError(f"{w}x{h}, {step}: the page is {size[0]} px high in a window of {size[1]} px")
        page.click("#synthetic")  # back to the real people, in the world
        page.wait_for_function("document.getElementById('headline').dataset.mode === 'real'", timeout=WAIT)
        if page.locator("#hist .hbar.chosen").count():
            page.click("#hist .hbar.chosen")
        page.wait_for_function("!document.querySelector('#hist .hbar.chosen')", timeout=WAIT)


def test_globe(page, base):
    """The globe: a country chosen in the list of the countries is chosen on the globe, its box has the numbers of
    app/anagram.py; the pointer on a country shows its box; a click on a country of the globe chooses it, a click on
    it again the world, a click on the sea the world too; a drag turns the globe (the earth follows the pointer). A screenshot, path in the output."""
    q = "Yuki Tanaka"
    page.emulate_media(reduced_motion="reduce")  # still: the places on the screen do not move
    start(page, base, q)
    check_list(page, "world", q)
    page.wait_for_function("document.getElementById('globe').globe", timeout=WAIT)
    try:
        page.wait_for_function(
            """(() => { const c = document.querySelector('#globe canvas');
            if (!c.width || !c.height) return false;
            const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            let n = 0; for (let i = 3; i < d.length; i += 16) n += d[i] > 0; return n > 1000; })()""",
            timeout=WAIT,
        )
    except PlaywrightTimeout:
        raise AssertionError("the globe is not drawn")
    page.click("#hist .hbar[data-country='JP']")
    check_list(page, "JP", q)
    page.wait_for_function("document.getElementById('globe').dataset.chosen === 'JP'", timeout=WAIT)
    page.wait_for_timeout(300)
    tip = page.text_content(".globe-tip")
    for want in ("Japan", "Real", n(expected("JP", q, 0)["real_people"]), "Synthetic"):
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
    dot = page.evaluate("document.getElementById('globe').globe.where('CN')")
    tip = page.evaluate(
        "(() => { const t = document.querySelector('.globe-tip'); return [t.offsetLeft, t.offsetTop, t.offsetWidth, t.offsetHeight]; })()"
    )
    assert not (tip[0] <= dot["x"] <= tip[0] + tip[2] and tip[1] <= dot["y"] <= tip[1] + tip[3]), (
        f"the box {tip} covers the country {dot}"
    )
    path = f"{SHOTS}/{page.context.browser.browser_type.name}-globe.png"
    page.locator("#globe").screenshot(path=path)
    page.mouse.click(*place("CN"))
    check_list(page, "CN", q)
    page.wait_for_function("document.getElementById('globe').dataset.chosen === 'CN'", timeout=WAIT)
    page.wait_for_timeout(300)
    page.mouse.click(*place("CN"))
    check_list(page, "world", q)
    page.mouse.click(*place("CN"))  # China again; then a click on the sea: the world
    check_list(page, "CN", q)
    page.wait_for_function("document.getElementById('globe').dataset.chosen === 'CN'", timeout=WAIT)
    page.wait_for_timeout(300)
    page.evaluate("document.getElementById('globe').scrollIntoView({ block: 'center' })")
    canvas = page.locator("#globe canvas").bounding_box()
    sea = page.evaluate("""(() => { const g = document.getElementById('globe').globe;
        for (let k = 0; k < 400; k++) { const a = k * 2.4, r = g.R * Math.sqrt((k + 0.5) / 400) * 0.95;
            const x = g.cx + r * Math.cos(a), y = g.cy + r * Math.sin(a);
            if (!g.countryAt(g.toEarth(x, y))) return { x, y }; } return null; })()""")
    assert sea, "no sea on the globe"
    page.mouse.click(canvas["x"] + sea["x"], canvas["y"] + sea["y"])
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
    assert abs(turned - want) < 0.15 * want, (
        f"a drag of 100 px turns the globe {turned:.1f} degrees, expected {want:.1f}"
    )
    return f"screenshot of the globe: {path}"


GLOBE = """() => { const box = document.getElementById('globe'), g = box.globe, c = box.querySelector('canvas'), dpr = devicePixelRatio;
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let lit = 0; for (let i = 3; i < d.length; i += 64) lit += d[i] > 0;
  return { w: box.clientWidth, h: box.clientHeight, cw: c.width / dpr, ch: c.height / dpr, R: g.R, cx: g.cx, cy: g.cy, lit }; }"""


def check_globe(page, label):
    """The globe fills its box at the size of the screen: the canvas has the size of the box (in pixels of the
    screen), the globe is a circle in the middle of it, and it is drawn (on the screen: else it draws no frames)."""
    try:
        page.wait_for_function(
            """(() => { const b = document.getElementById('globe'), c = b.querySelector('canvas');
            return b.globe && Math.abs(c.width / devicePixelRatio - b.clientWidth) < 1.5 && Math.abs(c.height / devicePixelRatio - b.clientHeight) < 1.5; })()""",
            timeout=WAIT,
        )
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
    assert abs(m["cx"] - m["w"] / 2) < 1 and abs(m["cy"] - m["h"] / 2) < 1, (
        f"{label}: the globe is not in the middle: {m}"
    )
    assert m["lit"] > 200, f"{label}: the globe is not drawn ({m['lit']} points)"


def test_globe_resize(page, base):
    """The globe at each size of the window, from a wide screen to a phone (the window changes size, the page stays):
    it fills its box, and the box of a country under the pointer stays in the globe."""
    page.emulate_media(reduced_motion="reduce")
    start(page, base, "Yuki Tanaka")
    check_list(page, "world", "Yuki Tanaka")
    for w, h in [(1600, 1000), (960, 900), (320, 640)]:  # the three layouts: 3 panels, 2 + 1, 1
        page.set_viewport_size({"width": w, "height": h})
        check_globe(page, f"{w}x{h}")
        page.evaluate("document.getElementById('globe').scrollIntoView({ block: 'center' })")
        box = page.locator("#globe").bounding_box()
        where = page.evaluate(
            "document.getElementById('globe').globe.where('FR') || document.getElementById('globe').globe.where('DE')"
        )
        if where:
            page.mouse.move(box["x"] + where["x"], box["y"] + where["y"])
            try:
                page.wait_for_function("!document.querySelector('.globe-tip').hidden", timeout=WAIT)
            except PlaywrightTimeout:
                raise AssertionError(
                    f"{w}x{h}: no box of the country under the pointer (hover {page.get_attribute('#globe', 'data-hover')!r})"
                )
            tip = page.locator(".globe-tip").bounding_box()
            inside = (
                tip["x"] >= box["x"] - 1
                and tip["y"] >= box["y"] - 1
                and tip["x"] + tip["width"] <= box["x"] + box["width"] + 1
                and tip["y"] + tip["height"] <= box["y"] + box["height"] + 1
            )
            assert inside, f"{w}x{h}: the box of the country {tip} is out of the globe {box}"
            page.mouse.move(0, 0)


def test_scroll_more(page, base):
    """Scrolling to the end of the list loads 50 more people."""
    start(page, base, "Yuki Tanaka", synthetic=True)
    check_list(page, "world", "Yuki Tanaka", synthetic=True)
    page.locator("#more").scroll_into_view_if_needed()
    page.wait_for_function("document.querySelectorAll('#results li').length === 100", timeout=WAIT)
    check_list(page, "world", "Yuki Tanaka", synthetic=True, limit=100)


def test_fast_typing_and_clear(page, base):
    """Type fast (one search per letter, most of them stopped by the next letter); the cross clears the name: no
    people, the field empty and ready to type."""
    start(page, base)
    page.locator("#q").press_sequentially("Wolfgang Amadeus Mozart")
    check_list(page, "world", "Wolfgang Amadeus Mozart")
    page.click("#clear")
    wait_status(page, "")
    page.wait_for_function("document.getElementById('view').hidden", timeout=WAIT)
    assert page.locator("#results li").count() == 0, "people stay after the query is cleared"
    assert page.input_value("#q") == "" and page.locator("#clear").is_hidden(), "the field is not cleared"
    assert page.evaluate("document.activeElement.id") == "q", "the field does not take the focus"


def test_reload_resets(page, base):
    """A reload starts again: no name, no people, the top of the page, the synthetic people (the default)."""
    start(page, base, "Yuki Tanaka", "JP")
    check_list(page, "JP", "Yuki Tanaka")
    page.evaluate("scrollTo(0, 500)")
    page.reload()
    page.wait_for_function("document.body.dataset.ready", timeout=WAIT)
    state = page.evaluate(
        "[document.getElementById('q').value, document.getElementById('view').hidden, scrollY, document.getElementById('synthetic').getAttribute('aria-checked')]"
    )
    assert state == ["", True, 0, "true"], (
        f"after a reload: field {state[0]!r}, view hidden {state[1]}, scroll {state[2]}, switch {state[3]}"
    )


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


TESTS = [
    test_about,
    test_type_world,
    test_capitals,
    test_address,
    test_switch,
    test_country,
    test_country_in_world,
    test_you,
    test_histogram,
    test_synthetic_nowhere_real,
    test_real_among_synthetic,
    test_names_whole,
    test_keyboard,
    test_keys,
    test_switch_country,
    test_rows,
    test_headers_stay,
    test_definition,
    test_fits_the_window,
    test_globe,
    test_globe_resize,
    test_scroll_more,
    test_fast_typing_and_clear,
    test_reload_resets,
    test_new_build,
    test_error_shown,
]
# WebKit and Firefox run only the tests of what the engines do in different ways (layout, focus and keys, pointer,
# scroll, canvas, Intl): the search and the numbers come from the same code in all browsers.
ENGINE = {
    test_about,
    test_capitals,
    test_names_whole,
    test_keyboard,
    test_keys,
    test_rows,
    test_headers_stay,
    test_definition,
    test_fits_the_window,
    test_globe,
    test_scroll_more,
}
EXPECT_ERRORS = {test_error_shown}


# ---------- phones and tablets: Playwright profiles (screen size, touch, user agent) ----------
MOBILE = ["iPhone SE", "Pixel 7", "iPad Mini"]  # the smallest phone (WebKit), Android (Chromium), a tablet
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
LAYOUT = [
    ("empty", "/", "document.body.dataset.ready"),
    ("real people", "/?Louis+Abraham&real", READY),
    ("synthetic people", "/?Louis+Abraham", READY),
    ("a country", "/?Yuki+Tanaka&in=JP", READY),
    ("how it works", "/how.html", "document.readyState === 'complete'"),
]


def test_mobile_layout(page, base):
    """For each view: no part wider than the screen, text of 16 px at least in the fields (else iOS zooms in),
    controls of 44 px and links of 32 px at least for a finger, on a phone the countries folded at first (the globe open),
    the globe in its box (opened), the top bar at the top after a scroll."""
    for name, url, ready in LAYOUT:
        page.goto(base + url)
        page.wait_for_function(ready, timeout=WAIT)
        page.wait_for_timeout(300)
        if name == "how it works":
            page.evaluate("document.querySelectorAll('details').forEach((d) => (d.open = true))")
        m = page.evaluate(MEASURE)
        assert m["width"] <= m["vw"], (
            f"{name}: the page is {m['width']} px wide on a screen of {m['vw']} px: {m['wide'][:4]}"
        )
        assert not m["wide"], f"{name}: wider than the screen: {m['wide'][:4]}"
        assert not m["inputs"], f"{name}: text smaller than 16 px in {m['inputs']}"
        assert not m["controls"], f"{name}: controls smaller than 44 px: {m['controls'][:6]}"
        assert not m["links"], f"{name}: links smaller than 32 px: {m['links'][:6]}"
        if ready == READY:
            if m["vw"] < 700:  # a phone: the list of the countries starts folded, the globe open
                folded = page.evaluate(
                    "['.globe-pane', '#hist'].map((s) => document.querySelector(s).classList.contains('folded'))"
                )
                assert folded == [False, True], f"{name}: on a phone, folded at first (globe, countries): {folded}"
            check_globe(page, name)
        page.evaluate("scrollTo(0, document.scrollingElement.scrollHeight)")
        page.wait_for_timeout(300)
        top = page.evaluate("document.querySelector('.top').getBoundingClientRect().top")
        assert top == 0, f"{name}: the top bar is not at the top of the screen after a scroll ({top} px)"


def test_mobile_flow(page, base):
    """With a touch screen: the definition on a tap, type a name, the switch, a country of the list, more people on
    scroll, the back button."""
    q = "Yuki Tanaka"
    start(page, base)
    page.tap("#term")  # the definition: a tap opens it, in the screen; a tap elsewhere closes it
    page.wait_for_selector("#entry", state="visible", timeout=WAIT)
    box = page.locator("#entry").bounding_box()
    assert box["x"] >= 0 and box["x"] + box["width"] <= page.viewport_size["width"] + 1, (
        f"the definition is out of the screen: {box}"
    )
    page.tap("h1", position={"x": 4, "y": 4})  # a tap outside the card (it covers the field)
    page.wait_for_selector("#entry", state="hidden", timeout=WAIT)
    page.tap("#q")
    page.keyboard.type(q)
    check_list(page, "world", q)
    wait_hist(page)
    if page.viewport_size["width"] < 700:
        page.tap("#hist .pane-head")  # a phone: the list of the countries starts folded
    page.tap("#hist .hbar[data-country='JP']")  # Japan: in the first countries of the real people
    check_list(page, "JP", q)
    page.tap("#synthetic")
    check_list(page, "JP", q, synthetic=True)
    page.locator("#more").scroll_into_view_if_needed()
    page.wait_for_function("document.querySelectorAll('#results li').length === 100", timeout=WAIT)
    page.go_back()
    check_list(page, "JP", q)


def test_mobile_globe(page, base):
    """With a finger, the globe turns in all directions and the page does not scroll; a finger on the box outside the
    globe scrolls the page (touch-action: manipulation, and globe.js stops the scroll of a touch on the globe). Swipes
    on Chromium (its touch events); the style on the others. On a phone, the list of the countries is short (about 5
    countries, then it scrolls)."""
    start(page, base, "Yuki Tanaka")
    check_list(page, "world", "Yuki Tanaka")
    wait_hist(page)
    if page.viewport_size["width"] < 700:  # a phone: open the list of the countries (folded at first)
        page.tap("#hist .pane-head")
        height = page.evaluate("document.getElementById('hist').getBoundingClientRect().height")
        assert height <= 300, f"the list of the countries is {height:.0f} px high on a phone"
    page.wait_for_function("document.getElementById('globe').globe", timeout=WAIT)
    action = page.evaluate("getComputedStyle(document.querySelector('#globe canvas')).touchAction")
    assert action == "manipulation", f"touch-action of the globe: {action}"
    if page.context.browser.browser_type.name != "chromium":
        return
    page.evaluate("document.getElementById('globe').scrollIntoView({ block: 'center' })")
    page.wait_for_timeout(300)
    cdp = page.context.new_cdp_session(page)

    def swipe(x, y, dx, dy):  # 10 steps of (dx, dy)
        cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
        for i in range(1, 11):
            cdp.send(
                "Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x + dx * i, "y": y + dy * i}]}
            )
            page.wait_for_timeout(16)
        cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        page.wait_for_timeout(300)
        return page.evaluate(
            "[document.getElementById('globe').globe.lat, document.getElementById('globe').globe.lon, scrollY]"
        )

    box = page.locator("#globe canvas").bounding_box()
    before = page.evaluate(
        "[document.getElementById('globe').globe.lat, document.getElementById('globe').globe.lon, scrollY]"
    )
    after = swipe(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, -4, 6)
    assert after[0] > before[0] + 2, f"a swipe down does not turn the globe: latitude {before[0]:.1f} -> {after[0]:.1f}"
    assert after[1] > before[1] + 1, (
        f"a swipe across does not turn the globe: longitude {before[1]:.1f} -> {after[1]:.1f}"
    )
    assert after[2] == before[2], f"the page scrolls under the globe: {before[2]} -> {after[2]}"
    before = after  # a corner of the box, outside the globe: the finger goes up, the page scrolls down
    after = swipe(box["x"] + 6, box["y"] + box["height"] - 6, 0, -12)
    assert after[2] > before[2] + 20, f"a swipe outside the globe does not scroll the page: {before[2]} -> {after[2]}"
    assert after[0] == before[0], f"a swipe outside the globe turns it: latitude {before[0]:.1f} -> {after[0]:.1f}"


def test_mobile_fold(page, base):
    """On a phone, the list of the countries starts folded; the globe is open (drawn), and a tap on the message does
    not fold it. A tap on the header of the countries opens their list, and a tap again folds it; the list stays folded
    when it changes (the switch). On a tablet, the list is open and a tap does nothing."""
    start(page, base, "Yuki Tanaka")
    check_list(page, "world", "Yuki Tanaka")
    wait_hist(page)
    phone = page.viewport_size["width"] < 700
    folded = lambda: page.locator("#hist .pane-body").is_hidden()  # noqa: E731
    assert folded() == phone, f"at first, the countries folded: {folded()}, phone: {phone}"
    check_globe(page, "at first")
    page.tap("#tally")
    page.wait_for_timeout(300)
    assert page.locator("#globe").is_visible(), "a tap on the message folds the globe"
    assert page.get_attribute("#tally", "role") is None, "the message is a button"
    page.tap("#hist .pane-head")
    page.wait_for_timeout(300)
    assert not folded(), f"after a tap on the header, the countries are folded (phone: {phone})"
    if not phone:
        return
    assert page.get_attribute("#hist .pane-head", "aria-expanded") == "true", (
        "the header does not say that the list is open"
    )
    page.tap("#hist .pane-head")
    page.wait_for_timeout(300)
    assert folded(), "a tap again does not fold the countries"
    assert page.get_attribute("#hist .pane-head", "aria-expanded") == "false", (
        "the header does not say that the list is folded"
    )
    page.tap("#synthetic")
    check_list(page, "world", "Yuki Tanaka", synthetic=True)
    assert folded(), "the list of the countries opens again when it changes"


MOBILE_TESTS = [test_mobile_layout, test_mobile_flow, test_mobile_globe, test_mobile_fold]


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
        return False, (
            f"{e}\n      screenshot: {path}"
            + (f"\n      also: {'; '.join(errors)}" if errors else "")
            + (f"\n      no answer yet: {', '.join(waits[:6])}" if waits else "")
        )
    finally:
        context.close()


def work(job):
    """A worker process: one browser (or device), some tests. Returns their results, in order."""
    kind, name, test_names, base, shots, headed = job
    global SHOTS
    SHOTS = shots
    out = []
    with sync_playwright() as p:
        if kind == "device":
            d = p.devices[name]
            browser = getattr(p, d["default_browser_type"]).launch(headless=not headed)
            device = {k: v for k, v in d.items() if k != "default_browser_type"}
        else:
            browser, device = getattr(p, name).launch(headless=not headed), None
        for test_name in test_names:
            ok, info = run(browser, base, globals()[test_name], shots, device)
            out.append((name, test_name, ok, info))
        browser.close()
    return out


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end tests of web/. By default: Chromium only, no phones; --all "
        "before a commit: the 3 browsers and the phone and tablet profiles."
    )
    parser.add_argument("--all", action="store_true", help="chromium, webkit and firefox, and the devices of MOBILE")
    parser.add_argument("--browser", nargs="*", help="chromium, webkit, firefox (default: chromium)")
    parser.add_argument("--devices", nargs="*", help="phone and tablet profiles of Playwright (default: none)")
    parser.add_argument("-k", help="run the tests whose name contains one of these words (comma or space separated)")
    parser.add_argument("-j", "--jobs", type=int, default=min(6, os.cpu_count() or 2), help="worker processes")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    browsers = (
        args.browser if args.browser is not None else (["chromium", "webkit", "firefox"] if args.all else ["chromium"])
    )
    devices = args.devices if args.devices is not None else (MOBILE if args.all else [])
    words = (args.k or "").replace(",", " ").split()
    pick = lambda tests: [t.__name__ for t in tests if not words or any(w in t.__name__ for w in words)]  # noqa: E731
    tests, mobile = pick(TESTS), pick(MOBILE_TESTS)
    # Chromium: all the tests; WebKit and Firefox: the tests of ENGINE (all the tests chosen with -k)
    of = lambda b: tests if b == "chromium" or words else [t for t in tests if globals()[t] in ENGINE]  # noqa: E731
    version = json.load(open(f"{ROOT}/web/data/version.json"))["version"]
    server = Server(("127.0.0.1", 0), partial(Quiet, directory=f"{ROOT}/web"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    shots = tempfile.mkdtemp(prefix="anagranames-e2e-")
    # the jobs: each browser and each device, its tests in chunks (a chunk: one launch of the browser)
    groups = [("browser", b, of(b)) for b in browsers if of(b)] + [("device", d, mobile) for d in devices if mobile]
    size = max(1, math.ceil(sum(len(g[2]) for g in groups) / max(1, args.jobs)))
    jobs = [
        (kind, name, names[i : i + size], base, shots, args.headed)
        for kind, name, names in groups
        for i in range(0, len(names), size)
    ]
    print(
        f"web/ build {version} at {base}: {', '.join(f'{len(of(b))} tests x {b}' for b in browsers)},"
        f" {len(mobile)} tests x {len(devices)} devices,"
        f" {len(jobs)} jobs on {args.jobs} processes",
        flush=True,
    )
    failed, t0 = 0, time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for done in as_completed([pool.submit(work, job) for job in jobs]):
            for name, test_name, ok, info in done.result():
                failed += not ok
                print(f"{'PASS' if ok else 'FAIL'}  {name:20s} {test_name}  {info}", flush=True)
    server.shutdown()
    print(f"{failed} failed" if failed else "all passed", f"in {time.time() - t0:.0f} s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
