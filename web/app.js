// Search page: loads the pack of the selected source once, then lists the anagram names while the visitor types.
// The query and the source are kept in the address (?q=...&from=...), so a search can be shared.
// index.html loads this file with the build version (?v=..., web/build.mjs), and each file below has it too: a
// browser that keeps files in its cache never mixes the files of two builds.
const version = new URL(import.meta.url).search;
const { Model, oneIn } = await import(`./anagram.js${version}`);

const STEP = 50; // names shown, and added by the More button
const $ = (id) => document.getElementById(id);
const [input, from, choices, status, headline, origin, list, more] =
  ["q", "from", "from-list", "status", "headline", "origin", "results", "more"].map($);
const n = (x) => x.toLocaleString("en-US");
const people = (x) => `${n(x)} ${x === 1 ? "person" : "people"}`;

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
  const size = `${(p.bytes / 1e6).toFixed(1)} MB`;
  const text = p.world || p.source === "INSEE" ? `${label} (${size})` : label; // the countries share the world file
  return { ...p, label, text, size, terms: terms.filter(Boolean).map(fold) };
});
const rank = (s) => (s.world ? 0 : s.source === "INSEE" ? 1 : 2);
sources.sort((a, b) => rank(a) - rank(b) || a.label.localeCompare(b.label));
const byId = new Map(sources.map((s) => [s.id, s]));
const packOf = (country) => (country === "FR" ? "FR-insee" : country); // the default source of a country

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

let current = null; // id of the selected source
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
  active = Math.max(0, q ? 0 : matches.findIndex((s) => s.id === current));
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
  from.value = byId.get(current).text;
}

function choose(id) {
  current = id;
  closeChoices();
  limit = STEP;
  update();
  input.focus();
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
      const pack = packOf(new Intl.Locale(lang).maximize().region);
      if (byId.has(pack)) return pack;
    } catch {
      // not a language tag
    }
  }
  return "world";
}
const params = new URLSearchParams(location.search);
current = byId.has(params.get("from")) ? params.get("from") : detect();
from.value = byId.get(current).text;
input.value = params.get("q") ?? "";

// ---------- search ----------
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

// The world pack has the data of its names in chunk files: a search reads the chunks of the names that it finds.
const PAUSE = 200; // ms without typing before a search reads files
const chunks = new Map(); // pack/side/chunk -> Promise
function loadChunk(model, pack, side, c) {
  const id = `${pack}/${side}/${c}`;
  if (!chunks.has(id)) {
    const done = fetch(`data/${pack}.names/${model.chunkFile(side, c)}${version}`)
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`${r.status} for ${r.url}`))))
      .then((b) => model.addChunk(side, c, new Uint8Array(b)));
    done.catch(() => chunks.delete(id));
    chunks.set(id, done);
  }
  return chunks.get(id);
}

// World packs: the real full names are in 4096 parts; a search reads the part of the letters of its query.
const parts = new Map(); // pack/part -> Promise
function loadPart(model, pack, part) {
  const id = `${pack}/${part}`;
  if (!parts.has(id)) {
    const done = fetch(`data/${pack}.real/${model.partFile(part)}${version}`)
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`${r.status} for ${r.url}`))))
      .then((b) => model.addPart(part, new Uint8Array(b)));
    done.catch(() => parts.delete(id));
    parts.set(id, done);
  }
  return parts.get(id);
}

function span(className, text) {
  const s = document.createElement("span");
  s.className = className;
  s.textContent = text;
  return s;
}

/** Links to the pages of the first name, the last name and the full name of a result (name.html, new tab). */
function explore(r) {
  const p = document.createElement("p");
  p.className = "explore";
  const a = (text, q) => {
    const link = document.createElement("a");
    link.href = `name.html?${new URLSearchParams({ from: current, ...q })}`;
    link.target = "_blank";
    link.textContent = `${text} ↗`;
    link.addEventListener("click", (e) => e.stopPropagation()); // the click opens the page, it does not close the row
    return link;
  };
  p.append("Explore: ", a(`first name ${r.first}`, { first: r.first }), " · ", a(`last name ${r.last}`, { last: r.last }),
    " · ", a(`full name ${r.first} ${r.last}`, { first: r.first, last: r.last }));
  return p;
}

function row(r, i, isQuery, world) {
  const li = document.createElement("li");
  li.tabIndex = 0;
  li.setAttribute("aria-expanded", "false");
  const toggle = () => {
    const open = li.getAttribute("aria-expanded") === "true";
    li.setAttribute("aria-expanded", String(!open));
    if (open) li.querySelector(".explore").remove();
    else li.append(explore(r));
  };
  li.addEventListener("click", toggle);
  li.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    e.preventDefault();
    toggle();
  });
  const name = span("name", `${r.first} ${r.last}`);
  if (isQuery) name.append(span("tag you", "your name"));
  if (r.real) name.append(span("tag real", `real · ${people(r.real)}`));
  const right = span("odds", oneIn(r.logP));
  if (world) right.prepend(span("where", english.of(r.country)));
  li.append(span("rank", String(i + 1).padStart(3, "0")), name, right);
  return li;
}

function link(text, pack) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "link";
  b.textContent = text;
  b.addEventListener("click", () => choose(pack));
  return b;
}

function showOrigin(res, world, words, model) {
  origin.hidden = false;
  if (!world) {
    origin.replaceChildren("Where do people with your name live? ", link("See the world →", "world"));
  } else if (res.origin) {
    const top = res.origin.filter((o, i) => i === 0 || o.p >= 0.01).slice(0, 3);
    const where = top.map((o) => `${english.of(o.country)} ${Math.round(o.p * 100)}%`).join(" · ");
    const pack = packOf(top[0].country);
    origin.replaceChildren(`Where people with your name live (model estimate): ${where}. `);
    if (byId.has(pack)) origin.append(link(`Search in ${byId.get(pack).label} →`, pack));
  } else {
    origin.textContent = words < 2
      ? "Type a first name and a last name to see where people with your name live."
      : `Your name is not in the data: it keeps the first names of at least ${model.minFirst} people and the last names of at least ${model.minLast} people in the world.`;
  }
}

let run = 0;
let limit = STEP;
async function update() {
  const id = ++run;
  try {
    await render(id);
  } catch (e) {
    if (id === run) status.textContent = `Error: ${e.message}`;
    throw e;
  }
}

/** Show the results of the query; stops when a newer update (id) starts while a file is loading. */
async function render(id) {
  const query = input.value;
  const source = byId.get(current), pack = source.pack;
  history.replaceState(null, "", `?${new URLSearchParams(query ? { q: query, from: current } : { from: current })}`);
  if (!ready.has(pack)) status.textContent = `Loading ${pack === "world" ? `the world file (${source.size})` : source.text}...`;
  const model = await load(pack);
  if (id !== run) return;
  const key = model.key(query);
  if (key.length < 4) {
    list.replaceChildren();
    more.hidden = headline.hidden = origin.hidden = true;
    status.textContent = key ? "Type at least 4 letters" : "";
    return;
  }
  const need = model.need(query);
  const part = model.partOf(query);
  if (need.length || (part !== null && !model.hasPart(part))) {
    await new Promise((resolve) => setTimeout(resolve, PAUSE));
    if (id !== run) return; // the visitor typed again
    status.textContent = need.length ? `Loading ${need.length} files of names...` : "Loading the real names...";
    await Promise.all([...need.map(([side, c]) => loadChunk(model, pack, side, c)), part !== null && loadPart(model, pack, part)]);
    if (id !== run) return;
  }
  const world = source.world;
  const res = model.search(query, source.world || source.source === "INSEE" ? null : source.country, limit);
  status.textContent = `${n(res.found)} name${res.found === 1 ? "" : "s"} · ${n(res.realNames)} real`;
  const where = world ? "in the world" : `in ${english.of(source.country)}`;
  headline.hidden = false;
  headline.replaceChildren(!res.found
    ? `No name ${where} is made of your letters.`
    : res.realPeople
      ? `At least ${people(res.realPeople)} ${where} ${res.realPeople === 1 ? "has" : "have"} a name made of your letters.`
      : `No person ${where} in the data has a name made of your letters. These names could exist:`);
  showOrigin(res, world, query.trim().split(/\s+/).length, model);
  list.replaceChildren(...res.results.map((r, i) => row(r, i, model.key(r.first) + model.key(r.last) === key, world)));
  more.hidden = res.found <= limit;
}

input.addEventListener("input", () => {
  limit = STEP;
  update();
});
more.addEventListener("click", () => {
  limit += STEP;
  update();
});
update();
