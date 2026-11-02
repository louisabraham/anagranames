// Command line for app/anagram.ts, with the options of app/anagram.py. Node 22.18+ runs TypeScript directly.
//   node app/cli.ts "Elena Ourgaud"                        world: world mixture
//   node app/cli.ts "Elena Ourgaud" --country FR           world, France: the names seen in France
//   node app/cli.ts --check                                compare with the values of pairmodel.py
// Without a query, the queries are read from stdin, one per line.
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { basename, join } from "node:path";
import process from "node:process";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { Model, oneIn, type Search } from "./anagram.ts";

const { values: opt, positionals: queries } = parseArgs({
  allowPositionals: true,
  options: {
    country: { type: "string" },
    pack: { type: "string" },
    limit: { type: "string", short: "n", default: "20" },
    json: { type: "boolean", default: false },
    data: { type: "string", default: fileURLToPath(new URL("data/", import.meta.url)) },
    check: { type: "boolean", default: false },
  },
});

/** The pack file of the options: --pack, else the world pack. */
function packPath(): string {
  const name = opt.pack ?? "world";
  return existsSync(name) ? name : join(opt.data, `${name}.anag.gz`);
}

function show(res: Search, source: string): void {
  const n = (x: number) => x.toLocaleString("en-US");
  console.log(`${res.query} (${res.country ?? "world"}, ${source}): ${n(res.found)} names, ${n(res.realNames)} real with ${n(res.realPeople)} people`);
  if (res.origin && !res.country) console.log(`origin: ${res.origin.map((o) => `${o.country} ${Math.round(o.p * 100)}%`).join(", ")}`);
  const width = Math.max(0, ...res.results.map((r) => r.first.length + r.last.length)) + 1;
  res.results.forEach((r, i) => {
    const name = `${r.first} ${r.last}`.padEnd(width);
    const real = r.real ? `  real: ${n(r.real)} ${r.real === 1 ? "person" : "people"}` : "";
    console.log(`${String(i + 1).padStart(4)}. ${name}  ${r.country}  ${oneIn(r.logP)}${real}`);
  });
}

/** Read part j of the real full names (world packs), from the directory <pack>.real/ next to the pack file. */
async function readPart(model: Model, path: string, j: number): Promise<void> {
  if (model.hasPart(j)) return;
  const file = join(path.replace(/(\.anag)?(\.gz)?$/, ".real"), model.partFile(j));
  await model.addPart(j, new Uint8Array(await readFile(file)));
}

/** Read the chunk files of the names (world pack), from the directory <pack>.names/ next to the pack file. */
async function readChunks(model: Model, path: string, chunks: [string, number][]): Promise<void> {
  const dir = path.replace(/(\.anag)?(\.gz)?$/, ".names");
  await Promise.all(chunks.map(async ([side, c]) =>
    model.addChunk(side as "first" | "last", c, new Uint8Array(await readFile(join(dir, model.chunkFile(side, c)))))));
}

/** Compare log p with the values that export_app.py computed with pairmodel.py. */
async function check(model: Model, path: string): Promise<boolean> {
  const ref = JSON.parse(await readFile(path.replace(/\.anag\.gz$/, ".check.json"), "utf8"));
  const rows = [...ref.country, ...(ref.world ?? []), ...ref.real];
  await readChunks(model, path, model.needIds("last", rows.map((r) => r.last), model.needIds("first", rows.map((r) => r.first))));
  const errors: [string, number][] = [["country", 0]];
  for (const r of ref.country) {
    const got = model.logP(model.first.entry(r.first), model.last.entry(r.last), model.countries.indexOf(r.country));
    errors[0][1] = Math.max(errors[0][1], Math.abs(got - r.log_p));
  }
  if (ref.world) {
    errors.push(["world", 0]);
    for (const r of ref.world) {
      const [got] = model.logPMixture(model.first.entry(r.first), model.last.entry(r.last));
      errors[1][1] = Math.max(errors[1][1], Math.abs(got - r.log_p));
    }
  }
  console.log(`largest difference with pairmodel.py, in nats: ${errors.map(([k, v]) => `${v.toExponential(2)} (${k})`).join(", ")}`);
  if (model.parts) for (const r of ref.real) await readPart(model, path, model.pairPart(r.first, r.last));
  const byCode = (pairs: [string, number][]) => JSON.stringify([...pairs].sort());
  const wrong = ref.real.filter((r: { first: number; last: number; real: number; countries?: [string, number][] }) =>
    model.real(r.first, r.last) !== r.real || (r.countries && byCode(model.realCountries(r.first, r.last)) !== byCode(r.countries)));
  console.log(`real full names: ${wrong.length} wrong of ${ref.real.length} (${ref.real.filter((r: { real: number }) => r.real > 0).length} listed)`);
  return Math.max(...errors.map(([, v]) => v)) < 1e-4 && wrong.length === 0;
}

const t = performance.now();
const path = packPath();
const model = await Model.load(new Uint8Array(await readFile(path)));
console.error(`loaded ${basename(path)} in ${((performance.now() - t) / 1000).toFixed(1)}s`);
if (opt.check) process.exit((await check(model, path)) ? 0 : 1);

const lines = queries.length ? queries : createInterface({ input: process.stdin });
for await (const query of lines) {
  if (!query.trim()) continue;
  await readChunks(model, path, model.need(query));
  const part = model.partOf(query);
  if (part !== null) await readPart(model, path, part);
  const res = model.search(query.trim(), opt.country ?? null, Number(opt.limit));
  if (opt.json) {
    const results = res.results.map((r) => ({
      first: r.first, last: r.last, country: r.country, log_p: r.logP, real: r.real,
      first_people: r.firstPeople, first_countries: r.firstCountries, last_people: r.lastPeople,
      last_countries: r.lastCountries, real_countries: r.realCountries,
    }));
    const { query, country, found, realNames: real_names, realPeople: real_people, origin } = res;
    console.log(JSON.stringify({ query, country, found, real_names, real_people, origin, results }));
  } else show(res, model.source);
}
