// Page of a first name (?first=...), a last name (?last=...) or a full name (both), in a source (?from=: world,
// FR-insee or a country code; default world): its people in the world and in each country, or in the country,
// and for a full name the estimate of the model.
const version = new URL(import.meta.url).search;
const { Model, oneIn } = await import(`./anagram.js${version}`);

const $ = (id) => document.getElementById(id);
const english = new Intl.DisplayNames(["en"], { type: "region" });
const n = (x) => x.toLocaleString("en-US");
const people = (x) => `${n(x)} ${x === 1 ? "person" : "people"}`;
const params = new URLSearchParams(location.search);
const from = params.get("from") ?? "world";
const pack = from === "FR-insee" ? from : "world";
const country = from === "FR-insee" || from === "world" ? null : from; // a country of the world pack
const first = params.get("first"), last = params.get("last");

async function bytes(path) {
  const r = await fetch(`data/${path}${version}`);
  if (!r.ok) throw new Error(`${r.status} for ${r.url}`);
  return new Uint8Array(await r.arrayBuffer());
}

function link(text, names, source = from) {
  const a = document.createElement("a");
  a.href = `name.html?${new URLSearchParams({ from: source, ...names })}`;
  a.textContent = text;
  return a;
}

function cell(content, className = "") {
  const td = document.createElement("td");
  td.className = className;
  td.append(content);
  return td;
}

/** The table of the countries, most people first, with a bar for each. */
function table(countries, total) {
  const max = countries[0][1];
  $("countries").tBodies[0].replaceChildren(...countries.map(([c, x]) => {
    const bar = document.createElement("span");
    bar.className = "meter";
    bar.style.width = `${(100 * x) / max}%`;
    const tr = document.createElement("tr");
    tr.append(cell(english.of(c)), cell(n(x), "num"), cell(`${((100 * x) / total).toFixed(1)}%`, "num"), cell(bar, "share"));
    return tr;
  }));
  $("countries").hidden = false;
}

async function main() {
  $("title").textContent = [first, last].filter(Boolean).join(" ");
  const model = await Model.load(await bytes(`${pack}.anag.gz`));
  const need = [...(first ? model.needName("first", first) : []), ...(last ? model.needName("last", last) : [])];
  await Promise.all(need.map(async ([side, c]) => model.addChunk(side, c, await bytes(`${pack}.names/${model.chunkFile(side, c)}`))));
  const world = country === null && model.countries.length > 1;
  const c = country === null ? -1 : model.countries.indexOf(country);
  const where = world ? "in the world" : `in ${english.of(country ?? model.countries[0])}`;
  const nameId = (side, text) => { // -1 when the name is not in the data (or not seen in the country)
    const i = text ? model.nameId(side, text) : -1;
    return i >= 0 && (c < 0 || model.nameCountries(side, i, c).length > 0) ? i : -1;
  };
  const f = nameId("first", first), l = nameId("last", last);
  const missing = [first && f < 0 && `The first name ${first}`, last && l < 0 && `The last name ${last}`].filter(Boolean);
  if (missing.length) {
    $("summary").textContent = `${missing.join(" and ")} ${missing.length > 1 ? "are" : "is"} not in the data ${where}.`;
    return;
  }
  const names = { ...(f >= 0 && { first: model.first.name(f) }), ...(l >= 0 && { last: model.last.name(l) }) };
  const title = Object.values(names).join(" ");
  $("title").textContent = title;
  document.title = `ANAGRANAME: ${title}`;

  let countries, summary;
  if (f >= 0 && l >= 0) {
    const j = model.parts ? model.pairPart(f, l) : null;
    if (j !== null && !model.hasPart(j)) await model.addPart(j, await bytes(`${pack}.real/${model.partFile(j)}`));
    countries = model.realCountries(f, l, c);
    const total = countries.reduce((s, [, x]) => s + x, 0);
    const fe = model.first.entry(f), le = model.last.entry(l);
    const lp = world ? model.logPMixture(fe, le)[0] : model.logP(fe, le, Math.max(c, 0));
    summary = `Full name · ${total ? people(total) : "nobody in the data"} ${where} · model estimate: ${oneIn(lp)}`;
    $("links").append("Explore: ", link(`first name ${names.first}`, { first: names.first }), " · ",
      link(`last name ${names.last}`, { last: names.last }));
  } else {
    const [side, id] = f >= 0 ? ["first", f] : ["last", l];
    countries = model.nameCountries(side, id, c);
    const label = side === "first" ? "First name" : "Last name";
    const inWorld = people(model[side].total[id]);
    summary = world ? `${label} · ${inWorld} in the world` : `${label} · ${people(countries[0][1])} ${where}, ${inWorld} in the world`;
  }
  $("summary").textContent = summary;
  if (!world) $("links").append(" ", link("See in the world →", names, "world"));
  const total = countries.reduce((s, [, x]) => s + x, 0);
  if (world && total) table(countries, total);
}

main().catch((e) => {
  $("summary").textContent = `Error: ${e.message}`;
  throw e;
});
