// Build the website in web/: anagram.js (app/anagram.ts without its types), the packs in data/ with the list of
// the sources data/packs.json (the world and each country of the world pack).
// The world pack has its real full names in data/world.real/ and the data of its names in data/world.names/.
// data/version.json gives the build version: a hash of the code and of the sizes and dates of the packs.
// Run: node web/build.mjs   then serve web/, for example: python3 -m http.server -d web 8000
import { createHash } from "node:crypto";
import { constants, copyFileSync, cpSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { gunzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const web = `${root}web/`;
const ts = readFileSync(`${root}app/anagram.ts`, "utf8");
writeFileSync(`${web}anagram.js`, `// Built by web/build.mjs from app/anagram.ts: do not edit.\n${stripTypeScriptTypes(ts)}`);
const hash = createHash("sha256");
for (const file of ["anagram.js", "app.js", "globe.js", "globe.json", "style.css"]) hash.update(readFileSync(`${web}${file}`));

mkdirSync(`${web}data`, { recursive: true });
const packs = [];
for (const file of readdirSync(`${root}app/data`).sort()) {
  const pack = file.replace(/\.anag\.gz$/, "");
  if (pack === file) continue;
  copyFileSync(`${root}app/data/${file}`, `${web}data/${file}`, constants.COPYFILE_FICLONE);
  const { size, mtimeMs } = statSync(`${root}app/data/${file}`); // the export writes the parts before the pack
  hash.update(`${file} ${size} ${mtimeMs}\n`);
  for (const dir of [`${pack}.real`, `${pack}.names`]) { // world pack: parts of the real full names, chunks of names
    if (existsSync(`${root}app/data/${dir}`)) cpSync(`${root}app/data/${dir}`, `${web}data/${dir}`, { recursive: true });
  }
  const bytes = statSync(`${web}data/${file}`).size;
  if (pack !== "world") continue;
  packs.push({ id: pack, pack, world: true, country: null, source: "name_dataset", bytes });
  const raw = gunzipSync(readFileSync(`${web}data/${file}`));
  const meta = JSON.parse(raw.subarray(12, 12 + raw.readUInt32LE(8)).toString());
  for (const country of meta.countries) packs.push({ id: country, pack, world: false, country, source: "name_dataset", bytes });
}
writeFileSync(`${web}data/packs.json`, JSON.stringify(packs));
const version = hash.digest("hex").slice(0, 12);
writeFileSync(`${web}data/version.json`, JSON.stringify({ version }));
console.log(`web/: version ${version}, anagram.js, ${packs.length} sources`);
