// The page: 5 tabs on the same data. Home tells the story of a name in steps (its first name, last name, full name
// and anagrams); at the end, the visitor can use the other tabs freely. Anagrams lists the full names with the
// letters of the query; First name, Last name and Full name give the people with a name, per country. Copy link
// gives the address of the view.
// index.html loads this file with the build version (?v=..., web/build.mjs), and each file below has it too: a
// browser that keeps files in its cache never mixes the files of two builds.
const version = new URL(import.meta.url).search;
const { Model } = await import(`./anagram.js${version}`);

const STEP = 50; // names shown, and added by the More button
const PAUSE = 200; // ms without typing before a view reads files
const TABS = ["home", "anagrams", "first", "last", "full"];
const FREE = "anagraname.free"; // localStorage: the visitor has seen the story once, the tabs show
const $ = (id) => document.getElementById(id);
const [input, from, choices, status, headline, list, more] =
  ["q", "from", "from-list", "status", "headline", "results", "more"].map($);
const n = (x) => x.toLocaleString("en-US");
const people = (x) => `${n(x)} ${x === 1 ? "person" : "people"}`;
const round2 = (x) => Number(x.toPrecision(2)); // 2 significant digits: the model gives an order of magnitude
/** A probability as "1 in N", N rounded (1 in 700,000; 1 in 650 million). */
function odds(logP) {
  const x = Math.exp(-logP);
  for (const [unit, word] of [[1e12, "trillion"], [1e9, "billion"], [1e6, "million"]]) {
    if (x >= unit) return `1 in ${round2(x / unit)} ${word}`;
  }
  return `1 in ${n(round2(x))}`;
}

// ---------- sources: World, France from INSEE, then the countries; chosen with a fuzzy search ----------
const english = new Intl.DisplayNames(["en"], { type: "region" });
const local = new Intl.DisplayNames([...navigator.languages, "en"], { type: "region" });
const ALIASES = {
  world: "world earth global international all",
  "FR-insee": "insee",
  US: "usa america united states",
  GB: "uk britain great britain england scotland wales",
  AE: "uae emirates",
  KR: "korea south korea",
  CZ: "czech republic czechia",
  NL: "holland netherlands",
};
const fold = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
/** Name of a country in its own main language (DE: Deutschland). */
const own = (country) => {
  const language = new Intl.Locale("und", { region: country }).maximize().language;
  return new Intl.DisplayNames([language], { type: "region" }).of(country);
};
// A source: id (world, FR-insee or a country code), the pack file that it reads, its country (null: the world).
const sources = (await (await fetch(`data/packs.json${version}`)).json()).map((p) => {
  const label = p.world ? "World" : p.source === "INSEE" ? `${english.of(p.country)}, INSEE` : english.of(p.country);
  const terms = [label, ...(p.country ? [local.of(p.country), own(p.country), p.country] : []), ALIASES[p.id] ?? ""];
  const size = `${(p.bytes / 1e6).toFixed(1)} MB`; // shown while the file loads
  return { ...p, label, text: label, size, terms: terms.filter(Boolean).map(fold) };
});
const rank = (s) => (s.world ? 0 : s.source === "INSEE" ? 1 : 2);
sources.sort((a, b) => rank(a) - rank(b) || a.label.localeCompare(b.label));
const byId = new Map(sources.map((s) => [s.id, s]));
const packOf = (country) => (country === "FR" ? "FR-insee" : country); // the default source of a country
/** The country of a source for Model.search (null: the world, or the one country of the pack). */
const countryOf = (source) => (source.world || source.source === "INSEE" ? null : source.country);

/** Optimal string alignment distance: insertions, deletions, substitutions and swaps of two letters. */
function distance(a, b) {
  const d = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
  for (let i = 0; i <= a.length; i++) d[i][0] = i;
  for (let j = 0; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
    }
  }
  return d[a.length][b.length];
}

/** Score of a query in a text: its letters in order in the text, with a bonus for consecutive letters and word
 * starts; else 1 or 2 typing errors in the start of the text; -Infinity if there is no match. */
function score(query, text) {
  let s = 0, at = 0, prev = -2;
  for (const ch of query) {
    const i = text.indexOf(ch, at);
    if (i < 0) {
      const max = query.length >= 6 ? 2 : query.length >= 3 ? 1 : 0;
      let best = Infinity;
      for (let k = Math.max(1, query.length - max); k <= query.length + max; k++) best = Math.min(best, distance(query, text.slice(0, k)));
      return best <= max ? query.length / 2 - 2 * best : -Infinity;
    }
    s += 1 + (i === prev + 1 ? 2 : 0) + (i === 0 || text[i - 1] === " " ? 2 : 0) - 0.1 * (i - at);
    prev = i;
    at = i + 1;
  }
  return s + (text === query ? 10 : 0);
}

let matches = [];
let active = 0;

function openChoices(query) {
  const q = fold(query).replace(/[^a-z]/g, "");
  matches = q
    ? sources
        .map((s) => ({ s, v: Math.max(...s.terms.map((t) => score(q, t))) }))
        .filter((m) => m.v > -Infinity)
        .sort((a, b) => b.v - a.v)
        .map((m) => m.s)
    : sources;
  active = Math.max(0, q ? 0 : matches.findIndex((s) => s.id === view.from));
  choices.replaceChildren(...matches.map((s, i) => {
    const li = document.createElement("li");
    li.id = `from-${i}`;
    li.setAttribute("role", "option");
    li.textContent = s.text;
    li.addEventListener("mousedown", (e) => {
      e.preventDefault(); // keep the focus until the choice is made
      choose(s.id);
    });
    return li;
  }));
  choices.hidden = false;
  from.setAttribute("aria-expanded", "true");
  highlight();
}

function highlight() {
  [...choices.children].forEach((li, i) => li.setAttribute("aria-selected", String(i === active)));
  const li = choices.children[active];
  if (li) {
    from.setAttribute("aria-activedescendant", li.id);
    li.scrollIntoView({ block: "nearest" });
  }
}

function closeChoices() {
  choices.hidden = true;
  from.setAttribute("aria-expanded", "false");
  from.removeAttribute("aria-activedescendant");
  from.value = byId.get(view.from).text;
}

function choose(id) {
  limits = [STEP, STEP];
  go({ from: id });
  closeChoices();
  (view.tab === "anagrams" ? input : view.tab === "last" ? $("last") : $("first")).focus();
}

from.addEventListener("focus", () => {
  // An empty field, not a selected text: after a click, Safari moves the caret and the selection is lost.
  from.placeholder = from.value;
  from.value = "";
  openChoices("");
});
from.addEventListener("input", () => openChoices(from.value));
from.addEventListener("blur", closeChoices);
from.addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    if (choices.hidden) openChoices("");
    active = Math.min(Math.max(active + (e.key === "ArrowDown" ? 1 : -1), 0), matches.length - 1);
    highlight();
  } else if (e.key === "Enter" && matches[active]) {
    e.preventDefault();
    choose(matches[active].id);
  } else if (e.key === "Escape") {
    from.blur();
  }
});

// Source: the address, else the first browser language with a country (France: INSEE), else the world.
function detect() {
  for (const lang of navigator.languages) {
    try {
      const id = packOf(new Intl.Locale(lang).maximize().region);
      if (byId.has(id)) return id;
    } catch {
      // not a language tag
    }
  }
  return "world";
}

// ---------- the view ----------
// The address stays the address of the page: the view is in the history entries (their state), so the back and
// forward buttons go between the views. A shared address is read once, then removed from the address bar. Its
// form: ?Yuki+Tanaka (the story of the name), ?anagrams=Yuki+Tanaka, ?first=Yuki, ?last=Tanaka,
// ?first=Yuki&last=Tanaka, with &in=JP for a source other than the world.
const KNOWN = ["anagrams", "first", "last", "in"];
function readView() {
  const p = new URLSearchParams(location.search);
  const first = p.get("first") ?? "", last = p.get("last") ?? "", anagrams = p.get("anagrams");
  const story = [...p].find(([k, v]) => !KNOWN.includes(k) && v === "")?.[0] ?? "";
  const tab = first && last ? "full" : first ? "first" : last ? "last" : anagrams !== null ? "anagrams" : "home";
  return {
    tab,
    q: anagrams ?? story,
    step: story ? 1 : 0, // the step of the story (0: the question)
    free: tab !== "home" || localStorage.getItem(FREE) === "1",
    from: byId.has(p.get("in")) ? p.get("in") : location.search ? "world" : detect(), // a shared address: the world
    first,
    last,
  };
}
let view = history.state?.view ?? readView(); // after a reload, the view of the history entry
history.replaceState({ view }, "", location.pathname);

/** The address of a view, to share it. */
function shareUrl(v) {
  const words = (t) => encodeURIComponent(t.trim()).replace(/%20/g, "+");
  const parts = v.tab === "home" ? (v.q.trim() ? [words(v.q)] : [])
    : v.tab === "anagrams" ? (v.q.trim() ? [`anagrams=${words(v.q)}`] : [])
    : [v.tab !== "last" && v.first.trim() && `first=${words(v.first)}`, v.tab !== "first" && v.last.trim() && `last=${words(v.last)}`].filter(Boolean);
  if (v.from !== "world" && v.tab !== "home") parts.push(`in=${v.from}`);
  return `${location.origin}${location.pathname}${parts.length ? `?${parts.join("&")}` : ""}`;
}

/** Change the view; push: a new history entry (to open a tab or a name), else the same entry (to type). */
function go(change, push = false) {
  if (view.tab === "home" && change.tab && change.tab !== "home" && !("from" in change)) change = { ...change, from: "world" }; // the story is in the world
  view = { ...view, ...change };
  if (view.tab !== "home" && !view.free) { // a tab other than the story: the tabs are open now
    view.free = true;
    localStorage.setItem(FREE, "1");
  }
  history[push ? "pushState" : "replaceState"]({ view }, "", location.pathname);
  show();
  update();
}
window.addEventListener("popstate", (e) => {
  if (!e.state?.view) return;
  view = e.state.view;
  from.value = byId.get(view.from).text;
  show();
  update();
});

const plainClick = (e) => !(e.button || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey);

/** A link to a view: a normal link (open in a new tab works), and a click changes the view in this page. */
function viewLink(text, change, title = "") {
  const a = document.createElement("a");
  a.href = shareUrl({ ...view, ...change });
  a.textContent = text;
  if (title) a.title = title;
  a.addEventListener("click", (e) => {
    if (!plainClick(e)) return;
    e.preventDefault();
    go(change, true);
    window.scrollTo(0, 0);
  });
  return a;
}

const TAGLINE = {
  home: "people whose name has your letters.",
  anagrams: "people whose name has your letters.",
  first: "who has this first name, and where.",
  last: "who has this last name, and where.",
  full: "who has this full name, and where.",
};

/** Show the tab of the view: tab links, fields, title, the address to copy. */
function show() {
  const home = view.tab === "home", names = !home && view.tab !== "anagrams";
  document.body.classList.toggle("onboarding", !view.free);
  document.body.classList.toggle("story-step", home && view.step > 0);
  for (const t of TABS) {
    const a = $(`tab-${t}`);
    a.href = shareUrl({ ...view, tab: t });
    if (t === view.tab) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  $("fields-anagrams").hidden = names || (home && view.step > 0);
  $("start").hidden = !home;
  $("q-label").className = home ? "prompt" : "visually-hidden";
  $("source-bar").hidden = home;
  $("out-home").hidden = !home || view.step === 0;
  $("out-anagrams").hidden = view.tab !== "anagrams";
  $("fields-names").hidden = $("out-names").hidden = !names;
  $("first-field").hidden = view.tab === "last";
  $("last-field").hidden = view.tab === "first";
  if (input.value !== view.q) input.value = view.q;
  if ($("first").value !== view.first) $("first").value = view.first;
  if ($("last").value !== view.last) $("last").value = view.last;
  $("tagline").textContent = TAGLINE[view.tab];
  const typed = view.tab === "anagrams" ? view.q.trim() : [view.tab !== "last" && view.first.trim(), view.tab !== "first" && view.last.trim()].some(Boolean);
  $("hero").hidden = home ? view.step > 0 : Boolean(typed); // the title only before a search
  if (!names) $("title").textContent = "Anagra\u00adname";
  if (document.activeElement !== from) from.value = byId.get(view.from).text;
  share.dataset.url = shareUrl(view);
  share.hidden = !typed;
  share.textContent = "Copy link";
}

for (const t of TABS) {
  $(`tab-${t}`).addEventListener("click", (e) => {
    if (!plainClick(e)) return;
    e.preventDefault();
    if (t !== view.tab) go({ tab: t }, true);
  });
}

// Copy the address of the view; without the clipboard API (old browsers, no permission), a hidden text field.
const share = $("share");
share.addEventListener("click", async () => {
  const url = share.dataset.url;
  try {
    await navigator.clipboard.writeText(url);
  } catch {
    const field = Object.assign(document.createElement("textarea"), { value: url });
    document.body.append(field);
    field.select();
    document.execCommand("copy");
    field.remove();
  }
  share.textContent = "Copied ✓";
  setTimeout(() => (share.textContent = "Copy link"), 2000);
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

function span(className, text) {
  const s = document.createElement("span");
  s.className = className;
  s.textContent = text;
  return s;
}

// ---------- anagrams ----------
/** The hero number of a view: a big value and a sentence. */
function stat(el, value, text) {
  el.replaceChildren(span("big", value), " ", span("say", text));
  el.hidden = false;
}

/** A result: its first name, its last name and its number link to the tabs of the names. Under the name: the
 * people and their countries (a real name), or the probability and the most probable country (another name). */
function row(r, isQuery, world) {
  const li = document.createElement("li");
  const name = span("name", "");
  name.append(viewLink(r.first, { tab: "first", first: r.first }, `The first name ${r.first}: people per country`), " ",
    viewLink(r.last, { tab: "last", last: r.last }, `The last name ${r.last}: people per country`));
  if (isQuery) name.append(" ", span("tag you", "you"));
  const detail = span("detail", "");
  detail.append(viewLink(r.real ? people(r.real) : odds(r.logP), { tab: "full", first: r.first, last: r.last },
    `${r.first} ${r.last}: people per country`));
  const where = r.real ? r.realCountries.map(([c]) => english.of(c)).join(", ") : world ? english.of(r.country) : "";
  if (where) detail.append(span("where", where));
  li.className = r.real ? "real" : "possible";
  li.append(name, detail);
  return li;
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

async function renderAnagrams(id, model, source) {
  const query = view.q;
  const key = model.key(query);
  const panels = $("anagram-panels"), possible = $("possible");
  delete headline.dataset.key;
  if (key.length < 4) {
    list.replaceChildren();
    possible.replaceChildren();
    panels.hidden = more.hidden = true;
    status.textContent = key ? "Type at least 4 letters" : "";
    return;
  }
  const ranking = searches.get(`${source.id}|${letters(model, query)}`) ?? (await search(id, model, source, query));
  if (!ranking) return; // a newer update started
  const res = { ...ranking.summary, results: ranking.real(limits[0]), possible: ranking.possible(limits[1]) };
  const world = source.world;
  status.textContent = "";
  const where = world ? "in the world" : `in ${english.of(source.country)}`;
  const others = res.found - res.realNames;
  if (res.realPeople) stat(headline, people(res.realPeople), `${where} ${res.realPeople === 1 ? "has" : "have"} a name made of your letters, with ${n(res.realNames)} full name${res.realNames === 1 ? "" : "s"}.`);
  else stat(headline, "Nobody", `${where} in the data has a name made of your letters.`);
  if (others) stat($("headline-model"), `${n(others)} name${others === 1 ? "" : "s"}`, `${others === 1 ? "is" : "are"} made of your letters, but nobody in the data has ${others === 1 ? "it" : "them"}. The most probable first.`);
  else stat($("headline-model"), "No other name", res.found ? "is made of your letters." : `${where} is made of your letters.`);
  headline.dataset.key = `${res.found}|${res.realNames}`; // for the tests
  const isQuery = (r) => model.key(r.first) + model.key(r.last) === key;
  list.replaceChildren(...res.results.map((r) => row(r, isQuery(r), world)));
  possible.replaceChildren(...res.possible.map((r) => row(r, isQuery(r), world)));
  panels.style.setProperty("--rows", Math.max(res.results.length, res.possible.length, 1)); // rows of the two lists
  more.hidden = res.realNames <= limits[0] && others <= limits[1];
  panels.hidden = false;
  if (!more.hidden && near(more)) requestAnimationFrame(() => more.click()); // the screen is not full yet
}

// The next names load when the end of the lists comes near the screen; the More button stays for the keyboard.
const near = (el) => el.getBoundingClientRect().top < innerHeight + 800;
new IntersectionObserver((entries) => {
  if (entries.some((e) => e.isIntersecting) && !more.hidden) more.click();
}, { rootMargin: "800px" }).observe(more);

// ---------- first name, last name, full name ----------
function cell(content, className = "") {
  const td = document.createElement("td");
  td.className = className;
  td.append(content);
  return td;
}

const TOP = 10; // countries shown before "Show all"
let allCountries = false;

/** Bars of the countries, most first, with the value at the end of each bar (the details in the hover text). */
function bars(tableEl, rows, value, hover) {
  const max = rows[0][1];
  tableEl.tBodies[0].replaceChildren(...rows.map(([c, x]) => {
    const bar = span("meter", "");
    bar.style.width = `calc((100% - 8ch) * ${x / max})`;
    const tr = document.createElement("tr");
    tr.title = hover(c, x);
    const td = cell(bar, "share");
    td.append(span("tip", value(x)));
    tr.append(cell(english.of(c)), td);
    return tr;
  }));
  tableEl.hidden = false;
}

/** The countries of the data: the first TOP, then all with the button. */
function table(countries, total) {
  bars($("countries"), allCountries ? countries : countries.slice(0, TOP), n, (c, x) => `${english.of(c)}: ${people(x)}, ${((100 * x) / total).toFixed(1)}%`);
  $("all").hidden = allCountries || countries.length <= TOP;
  $("all").textContent = `Show all ${countries.length} countries`;
}
$("all").addEventListener("click", () => {
  allCountries = true;
  update();
});

/** Expected people: about 170, 2.6, fewer than 1. */
function expected(x) {
  if (x < 1) return "< 1";
  return x < 10 ? String(round2(x)) : n(round2(x));
}

const HINT = { first: "Type a first name.", last: "Type a last name.", full: "Type a first name and a last name." };
let shown = ""; // the names of the last name view, to wait for a pause in typing when they change

async function renderName(id, model, source) {
  const summary = $("summary"), links = $("links"), modelPanel = $("model-panel"), countriesTable = $("countries");
  const want = { first: view.tab !== "last" ? view.first.trim() : "", last: view.tab !== "first" ? view.last.trim() : "" };
  const clear = (text) => {
    summary.replaceChildren(span("say", text));
    links.replaceChildren();
    countriesTable.hidden = $("all").hidden = modelPanel.hidden = true;
    status.textContent = "";
  };
  $("title").textContent = { first: "First name", last: "Last name", full: "Full name" }[view.tab];
  if ((view.tab !== "last" && !want.first) || (view.tab !== "first" && !want.last)) return clear(HINT[view.tab]);
  const key = `${source.id}|${view.tab}|${want.first}|${want.last}`;
  if (key !== shown) {
    await new Promise((resolve) => setTimeout(resolve, PAUSE));
    if (id !== run) return; // the visitor typed again
    shown = key;
    allCountries = false;
  }
  const need = [...(want.first ? model.needName("first", want.first) : []), ...(want.last ? model.needName("last", want.last) : [])];
  if (!(await fetchAll(id, model, source.pack, need, null))) return;
  const country = countryOf(source);
  const c = country === null ? -1 : model.countries.indexOf(country);
  const world = source.world;
  const where = world ? "in the world" : `in ${english.of(source.country)}`;
  const idOf = (side, text) => { // -1 when the name is not in the data (or not seen in the country)
    const i = text ? model.nameId(side, text) : -1;
    return i >= 0 && (c < 0 || model.nameCountries(side, i, c).length > 0) ? i : -1;
  };
  const f = idOf("first", want.first), l = idOf("last", want.last);
  const missing = [want.first && f < 0 && `The first name ${want.first}`, want.last && l < 0 && `The last name ${want.last}`].filter(Boolean);
  if (missing.length) return clear(`${missing.join(" and ")} ${missing.length > 1 ? "are" : "is"} not in the data ${where}.`);
  const names = { ...(f >= 0 && { first: model.first.name(f) }), ...(l >= 0 && { last: model.last.name(l) }) };
  const inCountries = (k) => (world ? `, in ${k} countr${k === 1 ? "y" : "ies"}` : "");
  let countries;
  if (f >= 0 && l >= 0) {
    const part = model.parts ? model.pairPart(f, l) : null;
    if (!(await fetchAll(id, model, source.pack, [], part))) return;
    countries = model.realCountries(f, l, c);
    const total = countries.reduce((s, [, x]) => s + x, 0);
    const full = `${names.first} ${names.last}`;
    stat(summary, total ? people(total) : "Nobody", total
      ? `${total === 1 ? "has" : "have"} the full name ${full} ${where}${inCountries(countries.length)}.`
      : `has the full name ${full} ${where}.`);
    // the model: p(f, l) per person; expected people = population x p
    const fe = model.first.entry(f), le = model.last.entry(l);
    const k = Math.max(c, 0);
    const lp = world ? model.logPMixture(fe, le)[0] : model.logP(fe, le, k);
    const pop = world ? model.population.reduce((s, x) => s + x, 0) : model.population[k];
    const mean = pop * Math.exp(lp);
    stat($("estimate"), `≈ ${expected(mean)} ${mean >= 1 && round2(mean) === 1 ? "person" : "people"}`,
      `expected ${world ? `in the ${model.countries.length} countries` : where}: ${odds(lp)} people.`);
    if (world) {
      const top = model.shares(f, l, 5).filter((o, j) => j === 0 || o.p >= 0.01);
      bars($("model-countries"), top.map((o) => [o.country, o.p * mean]), expected, (cc, x) => `${english.of(cc)}: about ${expected(x)} people expected`);
    } else $("model-countries").hidden = true;
    modelPanel.hidden = false;
    links.replaceChildren("Explore: ", viewLink(`first name ${names.first}`, { tab: "first", first: names.first }), " · ",
      viewLink(`last name ${names.last}`, { tab: "last", last: names.last }), " · ",
      viewLink("anagrams", { tab: "anagrams", q: full }));
  } else {
    const [side, i] = f >= 0 ? ["first", f] : ["last", l];
    countries = model.nameCountries(side, i, c);
    const label = `the ${side} name ${names[side]}`;
    const inWorld = model[side].total[i];
    if (world) stat(summary, people(inWorld), `${inWorld === 1 ? "has" : "have"} ${label}${inCountries(countries.length)}.`);
    else stat(summary, people(countries[0][1]), `${where} ${countries[0][1] === 1 ? "has" : "have"} ${label} · ${people(inWorld)} in the world.`);
    modelPanel.hidden = true;
    links.replaceChildren();
  }
  const total = countries.reduce((s, [, x]) => s + x, 0);
  if (world && total) table(countries, total);
  else countriesTable.hidden = $("all").hidden = true;
  status.textContent = "";
}

// ---------- home: the story of a name, step by step ----------
const KICKER = { first: "Your first name", last: "Your last name", full: "Your full name", real: "Your anagrams",
  possible: "Names that could exist", end: "Now, explore" };
const stories = new Map(); // query -> the numbers of its story

/** The split of a full name into a first name and a last name of the data: the most probable full name of the
 * splits of its words; null if no split gives two names of the data. */
function splitName(model, query) {
  const words = query.trim().split(/\s+/);
  let best = null, bestLp = -Infinity;
  for (let i = 1; i < words.length; i++) {
    const f = model.nameId("first", words.slice(0, i).join(" ")), l = model.nameId("last", words.slice(i).join(" "));
    if (f < 0 || l < 0) continue;
    const lp = model.logPMixture(model.first.entry(f), model.last.entry(l))[0];
    if (lp > bestLp) [best, bestLp] = [{ f, l, lp }, lp];
  }
  return best;
}

/** The numbers of the story of a query: its anagrams, and its first, last and full name (null if no split). */
async function storyOf(id, model, source, query) {
  const key = query.trim().replace(/\s+/g, " ").toLowerCase();
  if (stories.has(key)) return stories.get(key);
  const words = query.trim().split(/\s+/), need = [];
  for (let i = 1; i < words.length; i++) {
    need.push(...model.needName("first", words.slice(0, i).join(" ")), ...model.needName("last", words.slice(i).join(" ")));
  }
  if (!(await fetchAll(id, model, source.pack, need, null))) return null;
  const ranking = searches.get(`${source.id}|${letters(model, query)}`) ?? (await search(id, model, source, query));
  if (!ranking) return null;
  const data = { ranking, names: null };
  const best = splitName(model, query);
  if (best) {
    const { f, l, lp } = best;
    if (!(await fetchAll(id, model, source.pack, [], model.pairPart(f, l)))) return null;
    const pop = model.population.reduce((a, b) => a + b, 0);
    data.names = {
      first: model.first.name(f), last: model.last.name(l), firstPeople: model.first.total[f],
      firstCountries: model.nameCountries("first", f), lastPeople: model.last.total[l],
      lastCountries: model.nameCountries("last", l), realCountries: model.realCountries(f, l),
      expected: pop * Math.exp(lp), lp,
    };
  }
  stories.set(key, data);
  return data;
}

const stepsOf = (data) => [...(data.names ? ["first", "last", "full"] : []), "real", "possible", "end"];

function el(tag, className, ...children) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  e.append(...children);
  return e;
}
const big = (text) => el("b", "", text);
const nameOf = (text) => el("i", "", text);

/** Bars of the first countries, people at the tip. */
function miniBars(countries, top = 5) {
  const rows = countries.slice(0, top), max = rows[0]?.[1] ?? 1;
  return el("table", "countries mini", el("tbody", "", ...rows.map(([c, x]) => {
    const bar = span("meter", "");
    bar.style.width = `calc((100% - 8ch) * ${x / max})`;
    const td = cell(bar, "share");
    td.append(span("tip", n(x)));
    return el("tr", "", cell(english.of(c)), td);
  })));
}

/** A short list of full names, with their people (real names) or their probability (the others). */
function miniList(results, query, model) {
  return el("ol", "results mini", ...results.slice(0, 5).map((r) => row(r, model.key(r.first) + model.key(r.last) === model.key(query), true)));
}

function free() {
  localStorage.setItem(FREE, "1");
  view.free = true;
  history.replaceState({ view }, "", location.pathname);
  document.body.classList.remove("onboarding");
}

/** One step of the story. */
function slide(kind, data, k, count, model) {
  const { ranking, names } = data, sum = ranking.summary, query = view.q;
  const progress = el("ol", "progress", ...Array.from({ length: count }, (_, i) => el("li", i < k ? "done" : i === k ? "now" : "")));
  const claim = el("h2", "claim");
  const body = [];
  const where = (countries) => `in ${countries.length} countr${countries.length === 1 ? "y" : "ies"}`;
  if (kind === "first" || kind === "last") {
    const total = kind === "first" ? names.firstPeople : names.lastPeople;
    const countries = kind === "first" ? names.firstCountries : names.lastCountries;
    const top = countries[0];
    claim.append(big(n(total)), ` ${total === 1 ? "person has" : "people have"} the ${kind} name `, nameOf(names[kind]), ".");
    body.push(el("p", "say", `Most of them live in ${english.of(top[0])} (${Math.round((100 * top[1]) / total)}%). They live ${where(countries)}:`),
      miniBars(countries), el("p", "note", "Counted in the data: the accounts of name_dataset."));
  } else if (kind === "full") {
    const real = names.realCountries.reduce((a, [, x]) => a + x, 0), full = `${names.first} ${names.last}`;
    if (real) claim.append(big(n(real)), ` ${real === 1 ? "person is" : "people are"} called `, nameOf(full), ".");
    else claim.append(big("Nobody"), " in the data is called ", nameOf(full), ".");
    const counted = el("div", "panel", el("h3", "panel-title", "In the data ", span("", "counted")),
      el("p", "say", real ? `${people(real)}, ${where(names.realCountries)}.` : "No account has this full name."));
    if (real) counted.append(miniBars(names.realCountries, 3));
    const computed = el("div", "panel", el("h3", "panel-title", "Model estimate ", span("", "computed")),
      el("p", "say", `≈ ${expected(names.expected)} people expected in the 105 countries: ${odds(names.lp)} people.`));
    body.push(el("div", "panels", counted, computed));
  } else if (kind === "real") {
    if (sum.realPeople) {
      claim.append(big(people(sum.realPeople)), ` ${sum.realPeople === 1 ? "has" : "have"} a name made of your letters.`);
      body.push(el("p", "say", `${n(sum.realNames)} full name${sum.realNames === 1 ? "" : "s"} of the data use exactly the letters of ${query.trim()}. The most frequent:`),
        miniList(ranking.real(5), query, model));
    } else claim.append(big("Nobody"), " in the data has a name made of your letters.");
  } else if (kind === "possible") {
    const others = sum.found - sum.realNames;
    claim.append(big(n(others)), ` other name${others === 1 ? "" : "s"} could be made of your letters.`);
    if (others) body.push(el("p", "say", "Nobody in the data has them: the model gives their probability. The most probable:"),
      miniList(ranking.possible(5), query, model));
  } else {
    free();
    claim.append("Now, explore on your own.");
    const world = { from: "world" }; // the story is in the world: its links too
    const links = [viewLink("All your anagrams →", { ...world, tab: "anagrams" })];
    if (names) links.push(viewLink(`The first name ${names.first} →`, { ...world, tab: "first", first: names.first }),
      viewLink(`The last name ${names.last} →`, { ...world, tab: "last", last: names.last }),
      viewLink(`The full name ${names.first} ${names.last} →`, { ...world, tab: "full", first: names.first, last: names.last }));
    body.push(el("p", "say", "The tabs at the top are open now."), el("div", "explore-links", ...links));
  }
  const nav = el("div", "story-nav");
  const back = el("button", "quiet", "← Back");
  back.type = "button";
  back.addEventListener("click", () => go({ step: view.step - 1 }));
  nav.append(back);
  if (kind !== "end") {
    const next = el("button", "", "Next →");
    next.type = "button";
    next.id = "next";
    next.addEventListener("click", () => go({ step: view.step + 1 }, true));
    nav.append(next, viewLink("Skip the story", { tab: "anagrams", from: "world" }));
  }
  return el("article", `slide slide-${kind}`, progress, el("p", "kicker", `${k + 1} / ${count} · ${KICKER[kind]}`), claim, ...body, nav);
}

async function renderHome(id, model, source) {
  const story = $("story");
  status.textContent = "";
  if (view.step === 0) return; // the question
  if (model.key(view.q).length < 4) return go({ step: 0 });
  if (!story.firstChild) story.replaceChildren(el("p", "say", "Looking for your name..."));
  const data = await storyOf(id, model, source, view.q);
  if (!data || id !== run) return;
  const steps = stepsOf(data);
  const k = Math.min(view.step, steps.length) - 1;
  story.replaceChildren(slide(steps[k], data, k, steps.length, model));
  story.querySelector("#next, .explore-links a")?.focus({ preventScroll: true });
}

/** Start the story of the name of the field. */
function start() {
  if ($("q").value.trim().split(/\s+/).join("").length < 4) {
    $("q-label").textContent = "Your first and last name, please:";
    return;
  }
  $("story").replaceChildren();
  go({ step: 1 }, true);
}
$("start").addEventListener("click", start);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && view.tab === "home") start();
});
document.addEventListener("keydown", (e) => { // the arrows go between the steps
  if (view.tab !== "home" || view.step === 0 || e.target.closest?.("input")) return;
  if (e.key === "ArrowRight" && $("next")) $("next").click();
  if (e.key === "ArrowLeft") go({ step: view.step - 1 });
});

// ---------- updates ----------
let run = 0;
let limits = [STEP, STEP]; // the real names and the other names shown; More adds STEP to both
async function update() {
  const id = ++run;
  try {
    const source = byId.get(view.tab === "home" ? "world" : view.from); // the story is in the world
    if (!ready.has(source.pack)) status.textContent = `Loading ${source.pack === "world" ? `the world file (${source.size})` : source.text}...`;
    const model = await load(source.pack);
    if (id !== run) return;
    await ({ home: renderHome, anagrams: renderAnagrams }[view.tab] ?? renderName)(id, model, source);
  } catch (e) {
    if (id === run) status.textContent = `Error: ${e.message}`;
    throw e;
  }
}

input.addEventListener("input", () => {
  limits = [STEP, STEP];
  go({ q: input.value });
});
for (const side of ["first", "last"]) $(side).addEventListener("input", () => go({ [side]: $(side).value }));
more.addEventListener("click", () => {
  limits = limits.map((x) => x + STEP);
  update();
});
from.value = byId.get(view.from).text;
show();
update();
// the field of the view, with a mouse only: on a phone, the focus opens the keyboard over the page
if (matchMedia("(pointer: fine)").matches && !(view.tab === "home" && view.step > 0)) (["home", "anagrams"].includes(view.tab) ? input : view.tab === "last" ? $("last") : $("first")).focus();
