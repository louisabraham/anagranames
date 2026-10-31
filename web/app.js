// The page: type a full name, see its anagranames, the full names with exactly its letters. A switch chooses the
// people: real (the people of the data, counted) or synthetic (the names that nobody in the data has, with the
// probability of the model). A globe and a list of the countries show them per country, and choose a country.
// index.html loads this file with the build version (?v=..., web/build.mjs), and each file below has it too: a
// browser that keeps files in its cache never mixes the files of two builds.
const version = new URL(import.meta.url).search;
const { Model } = await import(`./anagram.js${version}`);

const STEP = 50; // names shown, and added by the More button (and by the scroll)
const PAUSE = 200; // ms without typing before the page reads files
const $ = (id) => document.getElementById(id);
const [input, status, headline, list, more] = ["q", "status", "headline", "results", "more"].map($);
const n = (x) => x.toLocaleString("en-US");
const people = (x) => `${n(x)} ${x === 1 ? "person" : "people"}`;
const round2 = (x) => Number(x.toPrecision(2)); // 2 significant digits: the model gives an order of magnitude
/** An expected number of people: 1,200; 57; 2.3; 0.47; < 0.01. */
const about = (x) => (x < 0.01 ? "< 0.01" : x < 100 ? String(round2(x)) : n(round2(x)));
/** The chance that at least one person has a name (Poisson): 1 - exp(-expected people). */
const chance = (x) => { const p = 1 - Math.exp(-x); return p > 0.995 ? "> 99%" : p < 0.005 ? "< 1%" : `${Math.round(100 * p)}%`; };
function span(className, text) {
  const s = document.createElement("span");
  s.className = className;
  s.textContent = text;
  return s;
}
function el(tag, className, ...children) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  e.append(...children);
  return e;
}

// ---------- sources: the world, France from INSEE (by its address only), the countries of the world ----------
const english = new Intl.DisplayNames(["en"], { type: "region" });
/** The name of a country, short: Hong Kong, not Hong Kong SAR China. */
const countryName = (c) => english.of(c).replace(/ SAR China$/, "");
const sources = (await (await fetch(`data/packs.json${version}`)).json()).map((p) => {
  const text = p.world ? "World" : p.source === "INSEE" ? `${countryName(p.country)}, INSEE` : countryName(p.country);
  return { ...p, text };
});
const byId = new Map(sources.map((s) => [s.id, s]));
/** The country of a source for Model.search (null: the world, or the one country of the pack). */
const countryOf = (source) => (source.world || source.source === "INSEE" ? null : source.country);
/** The people of a source: the world, or its country (the population of the pack). */
function populationOf(model, source) {
  const c = countryOf(source);
  return c === null ? model.population.reduce((s, x) => s + x, 0) : model.population[model.countries.indexOf(c)];
}

// ---------- the view ----------
// The address stays the address of the page: the view is in the history entries (their state), so the back and
// forward buttons go between the views (a country, the switch). A shared address is read once, then removed from the
// address bar: ?Yuki+Tanaka, &in=JP for a country, &synthetic for the synthetic people. The addresses of the old pages
// (?anagrams=..., ?first=...&last=..., ?full=...) give the same view. A reload starts again.
const KNOWN = ["anagrams", "full", "first", "last", "in", "synthetic"];
function capital(text) { // a capital letter at the start of each word of a name (and after a hyphen: Jean-Paul)
  return text.replace(/(^|[\s-])(\p{Ll})/gu, (_, before, letter) => before + letter.toUpperCase());
}
function readView() {
  const p = new URLSearchParams(location.search);
  const bare = [...p].find(([k, v]) => !KNOWN.includes(k) && v === "")?.[0];
  const names = [p.get("first"), p.get("last")].filter(Boolean).join(" ");
  return {
    q: capital(p.get("anagrams") ?? p.get("full") ?? (names || bare) ?? ""),
    from: byId.has(p.get("in")) ? p.get("in") : "world",
    synthetic: p.has("synthetic"),
  };
}
const reloaded = performance.getEntriesByType("navigation")[0]?.type === "reload";
history.scrollRestoration = "manual";
let view = (!reloaded && history.state?.view?.synthetic !== undefined && history.state.view) || readView();
history.replaceState({ view }, "", location.pathname);

/** Change the view; push: a new history entry (a country, the switch), else the same entry (to type). */
function go(change, push = false) {
  if (("from" in change && change.from !== view.from) || ("synthetic" in change && change.synthetic !== view.synthetic)) limit = STEP;
  view = { ...view, ...change };
  history[push ? "pushState" : "replaceState"]({ view }, "", location.pathname);
  show();
  update();
}
addEventListener("popstate", (e) => {
  if (!e.state?.view) return;
  view = e.state.view;
  limit = STEP;
  show();
  update();
});

const synthetic = $("synthetic");
function show() {
  if (input.value !== view.q) input.value = view.q;
  $("clear").hidden = !view.q;
  synthetic.setAttribute("aria-checked", String(view.synthetic));
  document.body.classList.toggle("synthetic", view.synthetic);
}
// The switch; a chosen country without people of the other kind: the world again.
synthetic.addEventListener("click", () => {
  const next = !view.synthetic, change = { synthetic: next };
  if (shown.chosen && shown.q === view.q && !(next ? shown.counts.possible : shown.counts.real).some(([c, x]) => c === shown.chosen && x > 0)) {
    change.from = "world";
  }
  go(change, true);
});
$("clear").addEventListener("click", () => {
  limit = STEP;
  go({ q: "" });
  input.focus();
});
input.addEventListener("input", (e) => {
  if (!e.isComposing && input.value !== capital(input.value)) { // not a word in composition (Japanese, Chinese)
    const [a, b] = [input.selectionStart, input.selectionEnd];
    input.value = capital(input.value);
    input.setSelectionRange(a, b);
  }
  limit = STEP;
  go({ q: input.value });
});

// ---------- files ----------
const models = new Map(); // pack -> Promise<Model>
function load(pack) {
  if (!models.has(pack)) {
    const model = fetch(`data/${pack}.anag.gz${version}`)
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`${r.status} for ${r.url}`))))
      .then((b) => Model.load(new Uint8Array(b)));
    model.catch(() => models.delete(pack));
    models.set(pack, model);
  }
  return models.get(pack);
}

/** Read a file of data/ once and give its bytes to use: cache key -> Promise. */
const files = new Map();
function once(key, path, use) {
  if (!files.has(key)) {
    const done = fetch(`data/${path}${version}`)
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`${r.status} for ${r.url}`))))
      .then((b) => use(new Uint8Array(b)));
    done.catch(() => files.delete(key));
    files.set(key, done);
  }
  return files.get(key);
}
// The world pack has the data of its names in chunk files, and its real full names in 4096 parts.
const loadChunk = (model, pack, side, c) =>
  once(`${pack}/${side}/${c}`, `${pack}.names/${model.chunkFile(side, c)}`, (b) => model.addChunk(side, c, b));
const loadPart = (model, pack, part) =>
  once(`${pack}/${part}`, `${pack}.real/${model.partFile(part)}`, (b) => model.addPart(part, b));

/** Read the chunks and the part that a view needs, after a pause in typing; false if a newer update started. */
async function fetchAll(id, model, pack, need, part) {
  if (!need.length && (part === null || model.hasPart(part))) return true;
  await new Promise((resolve) => setTimeout(resolve, PAUSE));
  if (id !== run) return false; // the visitor typed again
  await Promise.all([...need.map(([side, c]) => loadChunk(model, pack, side, c)), part !== null && loadPart(model, pack, part)]);
  return id === run;
}

// The last searches, ranked: the back button and the next names of a list need no new computation.
const searches = new Map(); // source|letters -> ranking (the anagrams depend only on the letters)
const letters = (model, query) => [...model.key(query)].sort().join("");
async function search(id, model, source, query) {
  if (!(await fetchAll(id, model, source.pack, model.need(query), model.partOf(query)))) return null;
  const ranking = model.ranking(query, countryOf(source));
  searches.set(`${source.id}|${letters(model, query)}`, ranking);
  if (searches.size > 20) searches.delete(searches.keys().next().value);
  return ranking;
}

// ---------- the list of the people ----------
/** The full name of a result is the query: the same words (Louisa Braham has the letters of Louis Abraham in the same
 * order, but not its words). */
function sameName(model, r, query) {
  const words = (t) => t.trim().split(/\s+/).map((w) => model.key(w)).filter(Boolean).join(" ");
  return words(`${r.first} ${r.last}`) === words(query);
}

/** A person of the list: the full name, and its real people, or the people that the model expects with it (its
 * probability times the people of the view; the chance that one exists in the hover text). */
function row(r, isQuery, pop, max) {
  const model = view.synthetic, e = pop * Math.exp(r.logP), x = model ? e : r.real; // the model: people expected
  const name = el("span", "name", el("span", "who", span("first", r.first), " ", span("last", r.last)));
  if (model && r.real) name.append(span("tag real", `${n(r.real)} real`)); // a real name among the synthetic ones
  const meter = span("meter", "");
  meter.style.width = `${(100 * x) / max}%`;
  const value = span("value", model ? about(e) : n(r.real));
  const li = el("li", `${r.real ? "real" : "synthetic"}${isQuery ? " you" : ""}`, name, el("span", "htrack", meter), value);
  const where = r.real ? `${people(r.real)} in the data: ${r.realCountries.map(([c]) => countryName(c)).join(", ")}` : "";
  li.title = model ? [`${about(e)} people expected: a chance of ${chance(e)} that one exists`, where].filter(Boolean).join(". ") : where;
  return li;
}

let run = 0;
let limit = STEP; // the people shown; More (and the scroll) adds STEP
async function update() {
  const id = ++run;
  try {
    const source = byId.get(view.from);
    const model = await load(source.pack);
    if (id !== run) return;
    await render(id, model, source);
  } catch (e) {
    if (id === run) status.textContent = `Error: ${e.message}`;
    throw e;
  }
}

async function render(id, model, source) {
  const query = view.q, key = model.key(query);
  delete headline.dataset.key;
  if (key.length < 4) {
    list.replaceChildren();
    $("view").hidden = more.hidden = true;
    status.textContent = key ? "Type at least 4 letters" : "";
    return;
  }
  const ranking = searches.get(`${source.id}|${letters(model, query)}`) ?? (await search(id, model, source, query));
  if (!ranking) return; // a newer update started
  const sum = ranking.summary;
  const where = source.world ? "" : `in ${source.text}`, pop = populationOf(model, source); // the world: no words
  status.textContent = "";
  if (!view.synthetic) {
    stat(sum.realPeople ? `${n(sum.realPeople)} real ${sum.realPeople === 1 ? "person" : "people"}` : "No real person", where);
  } else stat(sum.found ? `≈ ${about(pop * ranking.mass)} people expected` : "No name", where);
  $("names").textContent = n(view.synthetic ? sum.found : sum.realNames);
  headline.dataset.key = `${sum.found}|${sum.realNames}`; // for the tests
  headline.dataset.mode = view.synthetic ? "synthetic" : "real";
  const rows = view.synthetic ? ranking.model(limit) : ranking.real(limit); // synthetic: all the names, as the model sees them
  const top = rows.length ? (view.synthetic ? pop * Math.exp(rows[0].logP) : rows[0].real) : 1; // the first: the most people
  const focused = [...list.children].indexOf(document.activeElement); // the keyboard: the same row after a new list
  list.replaceChildren(...rows.map((r) => row(r, sameName(model, r, query), pop, top)));
  rove($("people"), focused);
  more.hidden = rows.length >= (view.synthetic ? sum.found : sum.realNames);
  $("view").hidden = false;
  if (!more.hidden && near(more)) requestAnimationFrame(() => more.click()); // the screen is not full yet
  await countries(id, query, source);
}

/** The message at the top of the globe, always about the world (not the chosen country): the real names; with the
 * synthetic names, the names generated too. sum: the summary of the search in the world. */
function tally(sum) {
  const count = (x, one, many) => [el("b", "", n(x)), ` ${x === 1 ? one : many}`], synthetic = sum.found - sum.realNames;
  const lines = [el("p", "real", ...(sum.realNames ? [...count(sum.realNames, "anagraname has", "anagranames have"), " Facebook accounts"]
    : ["No anagraname has Facebook accounts"]))];
  if (view.synthetic) {
    lines.push(el("p", "generated", ...(synthetic ? [...count(synthetic, "synthetic anagraname", "synthetic anagranames"), " generated"]
      : ["No synthetic anagraname generated"])));
  }
  $("tally").replaceChildren(...lines);
}

/** The number of the view, and its country (the world: no words). */
function stat(value, where) {
  headline.replaceChildren(span("big", where ? `${value} ${where}` : value));
}

// The next people load when the end of the list comes near the screen (or near the end of its panel, when the panel
// scrolls); the More button stays for the keyboard.
const pane = $("people-body"); // the list scrolls in its panel (a computer), else with the page
const scrolls = () => pane.scrollHeight > pane.clientHeight + 1 && getComputedStyle(pane).overflowY !== "visible";
const near = (el) => (scrolls() ? el.offsetTop - pane.scrollTop < pane.clientHeight + 600 : el.getBoundingClientRect().top < innerHeight + 800);
pane.addEventListener("scroll", () => {
  if (!more.hidden && near(more)) more.click();
}, { passive: true });
more.addEventListener("click", () => {
  limit += STEP;
  update();
});
new IntersectionObserver((entries) => {
  if (entries.some((e) => e.isIntersecting) && !more.hidden) more.click();
}, { rootMargin: "800px" }).observe(more);

// ---------- the countries: a list of bars and a globe; a click on a country chooses it ----------
let shown = { values: new Map(), chosen: null }; // what the globe shows (its boxes read it)
/** The countries of the world (all, the chosen one too, from the search in the world): the real people per country,
 * or the synthetic people (the names of the view of the country: its first and last name seen in it, and no real
 * person with it there). A click on a country shows its people; on the chosen country, the world again. */
async function countries(id, query, source) {
  const wm = await load("world");
  if (id !== run) return;
  const ranking = searches.get(`world|${letters(wm, query)}`) ?? (await search(id, wm, byId.get("world"), query));
  if (!ranking || id !== run) return;
  const counts = ranking.byCountry(), chosen = source.pack === "world" && !source.world ? source.country : null;
  tally(ranking.summary); // the world
  const rows = [...(view.synthetic ? counts.possible : counts.real)]; // a copy: the counts are kept with the search
  if (chosen && !rows.some(([c]) => c === chosen)) rows.push([chosen, 0]); // the chosen country is always there: a click on it, the world
  const unit = (x) => (view.synthetic ? about(x) : n(x));
  const box = $("hist"), max = rows[0]?.[1] ?? 1;
  const bars = rows.map(([c, x]) => { // all the countries: the panel scrolls
    const b = el("button", `hbar${c === chosen ? " chosen" : ""}`);
    b.type = "button";
    b.dataset.country = c;
    b.setAttribute("aria-pressed", String(c === chosen));
    b.title = c === chosen ? "All the countries again" : `The ${view.synthetic ? "synthetic" : "real"} people in ${countryName(c)}`;
    const meter = span("meter", "");
    meter.style.width = `${(100 * x) / max}%`;
    b.append(span("hname", countryName(c)), el("span", "htrack", meter), span("tip", unit(x)));
    b.addEventListener("click", () => {
      b.focus({ preventScroll: true }); // Safari gives no focus to a button on a click: the arrows then go from here
      choose(c);
    });
    return b;
  });
  const body = el("div", "pane-body", ...bars), at = $("hist").querySelector(".pane-body")?.scrollTop ?? 0;
  const focused = document.activeElement?.closest?.("#hist .hbar")?.dataset.country; // the keyboard: the same country
  box.replaceChildren(el("p", "pane-head", span("", "Countries"), span("count", n(rows.length))), body, ...box.querySelectorAll(".sbar"));
  fold.mark();
  rove(box, focused ? bars.findIndex((b) => b.dataset.country === focused) : -1);
  body.scrollTop = at; // the same place in the list after a choice, with the chosen country in view
  const row = body.querySelector(".chosen");
  if (row && (row.offsetTop < body.scrollTop || row.offsetTop + row.offsetHeight > body.scrollTop + body.clientHeight)) {
    body.scrollTop = row.offsetTop - body.clientHeight / 2;
  }
  box.dataset.mode = view.synthetic ? "synthetic" : "real"; // for the tests
  // the globe: the same countries and the same choice
  const values = new Map(counts.possible.map(([c, x]) => [c, { real: 0, synthetic: x }]));
  for (const [c, x] of counts.real) values.set(c, { real: x, synthetic: values.get(c)?.synthetic ?? 0 });
  shown = { values, chosen, counts, q: query };
  const g = await globe();
  if (id !== run) return;
  g.setData(values, view.synthetic ? "synthetic" : "real");
  g.select(chosen);
}

/** Choose a country (from the list or the globe); the chosen country again, or none: the world. */
function choose(c) {
  go({ from: !c || c === shown.chosen ? "world" : c }, true); // null: the globe outside a country of the data
}

// The globe (globe.js and the countries of globe.json, loaded with the first people).
let globeLoad = null;
function globe() {
  globeLoad ??= Promise.all([import(`./globe.js${version}`), fetch(`globe.json${version}`).then((r) => r.json())])
    .then(([{ Globe }, data]) => new Globe($("globe"), data, {
      pick: choose,
      info: (c, name) => {
        const v = shown.values.get(c);
        if (!c || !byId.has(c)) return { title: name, rows: [["Not in the data", ""]] };
        return { title: countryName(c), rows: [["Real", n(v?.real ?? 0)], ["Synthetic", about(v?.synthetic ?? 0)]] };
      },
    }));
  return globeLoad;
}

// The scroll bars of the two lists: a thin line and a rectangle on it, drawn by the page (the same in all browsers; the
// bar of the browser is hidden). Drag the rectangle, or click the line.
function scrollbar(pane) {
  const bar = el("div", "sbar"), thumb = el("i", "sthumb");
  bar.append(thumb);
  bar.setAttribute("aria-hidden", "true");
  pane.append(bar);
  const body = () => pane.querySelector(".pane-body"); // the list of the countries makes a new one at each view
  const draw = () => {
    const b = body(), room = b?.clientHeight ?? 0, can = b ? b.scrollHeight - room : 0;
    bar.hidden = !b || can <= 1 || getComputedStyle(b).overflowY === "visible";
    if (bar.hidden) return;
    const h = Math.max(28, (room * room) / b.scrollHeight);
    bar.style.top = `${b.offsetTop}px`;
    bar.style.height = `${room}px`;
    thumb.style.height = `${h}px`;
    thumb.style.transform = `translateY(${(b.scrollTop / can) * (room - h)}px)`;
  };
  const sizes = new ResizeObserver(draw);
  new MutationObserver(() => {
    if (body()) sizes.observe(body());
    draw();
  }).observe(pane, { childList: true, subtree: true });
  pane.addEventListener("scroll", draw, { capture: true, passive: true });
  bar.addEventListener("pointerdown", (e) => {
    const b = body(), room = b.clientHeight, h = thumb.offsetHeight, per = (b.scrollHeight - room) / (room - h);
    e.preventDefault();
    if (e.target !== thumb) b.scrollTop = (e.clientY - bar.getBoundingClientRect().top - h / 2) * per; // the line: there
    const [y0, s0] = [e.clientY, b.scrollTop];
    bar.setPointerCapture(e.pointerId);
    bar.classList.add("held");
    const move = (ev) => (b.scrollTop = s0 + (ev.clientY - y0) * per);
    bar.addEventListener("pointermove", move);
    bar.addEventListener("pointerup", () => {
      bar.removeEventListener("pointermove", move);
      bar.classList.remove("held");
    }, { once: true });
  });
}
scrollbar($("hist"));
scrollbar($("people"));

// The keyboard in the two lists (not in the order of Tab): Arrow Down in the field, or a click on a row, goes to a list;
// the arrows up and down go from row to row (Home, End, Page Up, Page Down); left and right go to the other list. Enter
// or Space on a country: this country, or the world again. With a country chosen, the arrows choose the next country at
// once; with the world, they only move.
const items = (pane) => [...pane.querySelectorAll(".hbar, #results li")];
let nameAt = 0, countryAt = 0; // the last rows with the focus: the places of Arrow Right and Arrow Left
/** After a new list: no row in the order of Tab; focus: the index of the row that had the focus (it keeps it). */
function rove(pane, focus = -1) {
  const all = items(pane);
  for (const r of all) r.tabIndex = -1;
  if (focus >= 0 && all[focus]) all[focus].focus({ preventScroll: true });
}
function goTo(pane, i) {
  const all = items(pane), row = all[Math.max(0, Math.min(all.length - 1, i))];
  if (!row) return null;
  row.focus({ preventScroll: true });
  row.scrollIntoView({ block: "nearest" });
  if (pane.id === "people") nameAt = all.indexOf(row);
  else countryAt = all.indexOf(row);
  return row;
}
/** The country of a row, chosen by the arrows when a country is chosen (with the world: no choice); the same entry of
 * the history (the back button: the view before the arrows). */
function pick(row) {
  const c = row?.dataset.country;
  if (c && shown.chosen && c !== shown.chosen) go({ from: c });
}
/** The row of the countries to go to: the chosen country, else the last one with the focus. */
const firstCountry = () => {
  const i = items($("hist")).findIndex((r) => r.classList.contains("chosen"));
  return i >= 0 ? i : countryAt;
};
for (const pane of [$("hist"), $("people")]) {
  pane.addEventListener("keydown", (e) => {
    const all = items(pane), i = all.indexOf(document.activeElement);
    if (i < 0 || e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      return pane.id === "hist" ? goTo($("people"), nameAt) : goTo($("hist"), countryAt);
    }
    const page = Math.max(1, Math.floor(pane.querySelector(".pane-body").clientHeight / all[i].offsetHeight) - 1);
    const to = { ArrowDown: i + 1, ArrowUp: i - 1, Home: 0, End: all.length - 1, PageDown: i + page, PageUp: i - page }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    const row = goTo(pane, to);
    if (pane.id === "hist") pick(row);
  });
}
$("people").addEventListener("focusin", (e) => {
  const i = items($("people")).indexOf(e.target);
  if (i >= 0) nameAt = i;
});
$("hist").addEventListener("focusin", (e) => {
  const i = items($("hist")).indexOf(e.target);
  if (i >= 0) countryAt = i;
});
// The arrows with the focus out of the field and out of the lists (on the page, the globe, a button): up, down and left
// go to the countries (the chosen one, else the last one with the focus), right to the names.
document.addEventListener("keydown", (e) => {
  if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || $("view").hidden) return;
  if (e.target.closest?.("input, textarea, .hbar, #results li") || !items($("hist")).length) return;
  const to = { ArrowDown: "hist", ArrowUp: "hist", ArrowLeft: "hist", ArrowRight: "people" }[e.key];
  if (!to) return;
  e.preventDefault();
  if (to === "hist") goTo($("hist"), firstCountry());
  else goTo($("people"), nameAt);
});
// The key S: the switch of the synthetic people (not in the field: it is a letter of a name).
document.addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() !== "s" || e.altKey || e.ctrlKey || e.metaKey || e.target.closest?.("input, textarea")) return;
  e.preventDefault();
  synthetic.click();
});
// Enter or Escape in the field: the field lets the focus go (on a phone, the keyboard closes).
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === "Escape") input.blur();
});
input.addEventListener("keydown", (e) => {
  if (e.key !== "ArrowDown" || $("view").hidden || !items($("hist")).length) return;
  e.preventDefault();
  goTo($("hist"), firstCountry());
});

// On a phone, a tap on the message folds the globe, a tap on the header of the countries folds their list (a tap again:
// open). The headers say it to screen readers (a button, open or not); on a computer, they do nothing.
const phone = matchMedia("(max-width: 699px)");
const fold = {
  panes: { tally: () => document.querySelector(".globe-pane"), hist: () => $("hist") },
  heads: { tally: () => $("tally"), hist: () => document.querySelector("#hist .pane-head") },
  /** The state of the headers: a button on a phone, open or not. */
  mark() {
    for (const k of ["tally", "hist"]) {
      const head = fold.heads[k]();
      if (!head) continue;
      if (phone.matches) {
        head.setAttribute("role", "button");
        head.setAttribute("aria-expanded", String(!fold.panes[k]().classList.contains("folded")));
      } else ["role", "aria-expanded"].forEach((a) => head.removeAttribute(a));
    }
  },
};
for (const k of ["tally", "hist"]) {
  fold.panes[k]().addEventListener("click", (e) => {
    if (!phone.matches || !e.target.closest(k === "tally" ? "#tally" : "#hist .pane-head")) return;
    fold.panes[k]().classList.toggle("folded");
    fold.mark();
  });
}
phone.addEventListener("change", fold.mark);
fold.mark();

// The three panels (globe, countries, people) fill the window under the question, and no more (the page does not
// scroll on a computer): at least 320 px.
function fit() {
  const panes = document.querySelector(".panes");
  const credit = document.querySelector(".credit").offsetHeight; // the banner under the panels
  if (!$("view").hidden) document.documentElement.style.setProperty("--rest", `${Math.max(320, innerHeight - panes.offsetTop - 16 - credit)}px`);
}
addEventListener("resize", fit);
new ResizeObserver(fit).observe(document.querySelector(".intro"));
new MutationObserver(fit).observe($("view"), { attributes: true, attributeFilter: ["hidden"] });

// The definition of the word: open while a mouse is over the word or the card; a click keeps it open (a finger, the
// keyboard); a click outside, Escape or a second click closes it. Its card starts under the word, in the page.
const term = $("term"), entry = $("entry");
let pinned = false, leaving = 0;
function place() {
  const room = entry.parentElement.clientWidth - entry.offsetWidth;
  entry.style.setProperty("--entry-left", `${Math.max(0, Math.min(term.offsetLeft, room))}px`);
  const title = term.closest("h1"); // under the title (the word can be on its second line)
  entry.style.top = `${title.offsetTop + title.offsetHeight + 4}px`;
}
function define(open) {
  clearTimeout(leaving);
  if (open === !entry.hidden) return;
  entry.hidden = !open;
  term.setAttribute("aria-expanded", String(open));
  if (open) place();
}
function close() {
  pinned = false;
  define(false);
}
for (const el of [term, entry]) {
  el.addEventListener("pointerenter", (e) => e.pointerType === "mouse" && define(true));
  el.addEventListener("pointerleave", (e) => {
    if (e.pointerType === "mouse" && !pinned) leaving = setTimeout(() => define(false), 180);
  });
}
term.addEventListener("click", () => {
  pinned = entry.hidden || !pinned; // closed, or open on hover: keep it open; kept open: close it
  define(pinned);
});
document.addEventListener("pointerdown", (e) => !entry.hidden && !e.target.closest("#term, #entry") && close());
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape" || entry.hidden) return;
  e.preventDefault(); // Escape closes the definition only
  close();
});
// Escape out of the field: no country chosen (the world again), and no row with the focus (the arrows go to the lists
// again). The definition open: it closes first.
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape" || e.defaultPrevented || !entry.hidden || e.target.closest?.("input, textarea")) return;
  if (document.activeElement?.closest?.(".hbar, #results li")) document.activeElement.blur();
  if (shown.chosen) go({ from: "world" }, true);
});
addEventListener("resize", () => entry.hidden || place());

show();
update();
document.body.dataset.ready = "1"; // for the tests
if (matchMedia("(pointer: fine)").matches) input.focus(); // with a mouse only: on a phone, the focus opens the keyboard
