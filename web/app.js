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
  return { ...p, text, size: `${(p.bytes / 1e6).toFixed(1)} MB` }; // size: shown while the file loads
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
  synthetic.setAttribute("aria-checked", String(view.synthetic));
  synthetic.querySelector(".switch-state").textContent = view.synthetic ? "ON" : "OFF";
  document.body.classList.toggle("synthetic", view.synthetic);
}
synthetic.addEventListener("click", () => go({ synthetic: !view.synthetic }, true));
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
const ready = new Set();
function load(pack) {
  if (!models.has(pack)) {
    const model = fetch(`data/${pack}.anag.gz${version}`)
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`${r.status} for ${r.url}`))))
      .then((b) => Model.load(new Uint8Array(b)));
    model.then(() => ready.add(pack), () => models.delete(pack));
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
  status.textContent = need.length ? `Loading ${need.length} files of names...` : "Loading the real names...";
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
  const name = el("span", "name", span("first", r.first), " ", span("last", r.last));
  if (isQuery) name.append(" ", span("tag you", "you"));
  const x = r.real ? r.real : pop * Math.exp(r.logP);
  const meter = span("meter", "");
  meter.style.width = `${(100 * x) / max}%`;
  const value = span("value", r.real ? n(r.real) : about(x));
  const li = el("li", r.real ? "real" : "synthetic", name, el("span", "htrack", meter), value);
  li.title = r.real ? `${people(r.real)}: ${r.realCountries.map(([c]) => countryName(c)).join(", ")}`
    : `${about(x)} people expected: a chance of ${chance(x)} that one exists`;
  return li;
}

let run = 0;
let limit = STEP; // the people shown; More (and the scroll) adds STEP
async function update() {
  const id = ++run;
  try {
    const source = byId.get(view.from);
    if (!ready.has(source.pack)) status.textContent = `Loading ${source.pack === "world" ? `the world file (${source.size})` : source.text}...`;
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
  const sum = ranking.summary, others = sum.found - sum.realNames, world = source.world;
  const where = world ? "" : `in ${source.text}`, pop = populationOf(model, source); // the world: no words
  status.textContent = "";
  if (!view.synthetic) {
    stat(sum.realPeople ? `${n(sum.realPeople)} real ${sum.realPeople === 1 ? "person" : "people"}` : "No real person", where);
  } else stat(others ? `≈ ${about(pop * ranking.mass)} synthetic people` : "No synthetic person", where);
  $("names").textContent = n(view.synthetic ? others : sum.realNames);
  headline.dataset.key = `${sum.found}|${sum.realNames}`; // for the tests
  headline.dataset.mode = view.synthetic ? "synthetic" : "real";
  const rows = view.synthetic ? ranking.possible(limit) : ranking.real(limit);
  const top = rows.length ? (rows[0].real || pop * Math.exp(rows[0].logP)) : 1; // the first: the most people
  list.replaceChildren(...rows.map((r) => row(r, sameName(model, r, query), pop, top)));
  more.hidden = rows.length >= (view.synthetic ? others : sum.realNames);
  $("view").hidden = false;
  if (!more.hidden && near(more)) requestAnimationFrame(() => more.click()); // the screen is not full yet
  await countries(id, query, source);
}

/** The number of the view, and its country (the world: no words). */
function stat(value, where) {
  headline.replaceChildren(span("big", where ? `${value} ${where}` : value));
}

// The next people load when the end of the list comes near the screen (or near the end of its panel, when the panel
// scrolls); the More button stays for the keyboard.
const pane = $("people");
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
  const rows = view.synthetic ? counts.possible : counts.real;
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
    b.addEventListener("click", () => choose(c));
    return b;
  });
  const title = el("p", "pane-head", span("", "Countries"));
  if (view.from !== "world") { // the chosen country, or France from INSEE: a button for the world again
    const back = el("button", "chip", `${byId.get(view.from).text} ×`);
    back.id = "to-world";
    back.type = "button";
    back.title = "All the countries again";
    back.addEventListener("click", () => go({ from: "world" }, true));
    title.append(back);
  }
  title.append(span("count", n(rows.length)));
  box.replaceChildren(title, ...bars);
  box.dataset.mode = view.synthetic ? "synthetic" : "real"; // for the tests
  // the globe: the same countries and the same choice
  const values = new Map(counts.possible.map(([c, x]) => [c, { real: 0, synthetic: x }]));
  for (const [c, x] of counts.real) values.set(c, { real: x, synthetic: values.get(c)?.synthetic ?? 0 });
  shown = { values, chosen };
  const g = await globe();
  if (id !== run) return;
  g.setData(values, view.synthetic ? "synthetic" : "real");
  g.select(chosen);
}

/** Choose a country (from the list or the globe); the chosen country again: the world. */
function choose(c) {
  go({ from: c === shown.chosen ? "world" : c }, true);
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
        return { title: countryName(c), rows: [["Real people", n(v?.real ?? 0)], ["Synthetic, expected", about(v?.synthetic ?? 0)]],
          hint: c === shown.chosen ? "Click: all the countries" : "Click to choose" };
      },
    }));
  return globeLoad;
}

// The three panels (globe, countries, people) fill the screen under the question: at least 400 px.
function fit() {
  const panes = document.querySelector(".panes");
  if (!$("view").hidden) document.documentElement.style.setProperty("--rest", `${Math.max(400, innerHeight - panes.offsetTop - 24)}px`);
}
addEventListener("resize", fit);
new ResizeObserver(fit).observe(document.querySelector(".intro"));
new MutationObserver(fit).observe($("view"), { attributes: true, attributeFilter: ["hidden"] });

show();
update();
document.body.dataset.ready = "1"; // for the tests
if (matchMedia("(pointer: fine)").matches) input.focus(); // with a mouse only: on a phone, the focus opens the keyboard
