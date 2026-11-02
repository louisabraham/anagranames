// Anagram search for full names, ranked by a pair model. TypeScript version of app/anagram.py: same packs
// (app/data/, written by export_app.py), same algorithm, same results. The comments in app/anagram.py give the
// search and the formulas; export_app.py gives the pack format. No dependencies: runs in Node 22.18+, Deno, Bun
// and browsers. Model.load() takes the bytes of a pack file (gzip or not): node:fs in app/cli.ts, fetch in a browser.

export interface Result {
  first: string;
  last: string;
  country: string; // the given country, or the country with the largest term of the mixture
  logP: number; // natural logarithm of the probability of the full name
  real: number; // people with this full name in the pack countries (0: none)
  firstPeople: number; // people with the first name in the world
  firstCountries: [string, number][]; // [country, people] of the first name in the pack countries, most people first
  lastPeople: number;
  lastCountries: [string, number][];
  realCountries: [string, number][]; // [country, people] of the full name in the pack countries
}

interface RealName {
  people: number;
  countries: [number, number][]; // [pack country, people]
}

export interface Origin {
  country: string;
  p: number;
}

export interface Search {
  query: string;
  country: string | null;
  found: number;
  realNames: number; // results that are real full names
  realPeople: number; // people with these real full names
  origin: Origin[] | null; // most probable countries of the query name
  results: Result[];
}

interface SideMeta {
  names: number;
  nodes: number;
  people: number[];
  prior_norm: number;
  scale?: number[]; // per country: the model uses count * scale (1, or for a country with a base, see export_app.py)
  groups: number;
  group_size: number;
  chunk: number; // names per chunk file
}

interface Meta {
  version: number;
  source: string;
  countries: string[];
  people: number[];
  population: number[];
  alpha: number;
  log_z: number[];
  gates: number[][];
  offsets: number[];
  letters: Record<string, string>;
  alphabet: string; // the 26 letters in the order of the signatures, the rarest first
  min_first?: number; // the count filter of the data (prepare_world.py): people with a name in the world
  min_last?: number;
  parts: number; // the real full names are in parts (export_app.py)
  part_hash: number[];
  sections: Record<string, [number, number]>;
  first: SideMeta;
  last: SideMeta;
}

interface Entry {
  x: Float64Array; // X[id, c] for all pack countries
  seen: Uint8Array; // 1 if the name is seen in country c
  inS: Uint8Array; // 1 if the pair term applies in country c: the name has people there (count > 0)
  countries: Uint8Array; // the countries where the name is seen, increasing
  a: number; // log(alpha pi): X[id, c] + log(N_c + alpha) where the name is not seen
  vec: Float64Array | null; // null for names without a vector (the null score)
}

type Section = (name: string) => Uint8Array;

/** Values of concatenated LEB128 varints, written to out. */
function varints<T extends Uint32Array | Float64Array>(b: Uint8Array, out: T): T {
  let v = 0, scale = 1, k = 0;
  for (let i = 0; i < b.length; i++) {
    v += (b[i] & 0x7f) * scale;
    if (b[i] < 0x80) {
      out[k++] = v;
      v = 0;
      scale = 1;
    } else scale *= 128;
  }
  return out;
}

/** A probability as "1 in N" (N in millions, billions or trillions when large). */
export function oneIn(logP: number): string {
  const x = Math.exp(-logP);
  for (const [unit, word] of [[1e12, "trillion"], [1e9, "billion"], [1e6, "million"]] as const) {
    if (x >= unit) return `1 in ${(x / unit).toFixed(1)} ${word}`;
  }
  return `1 in ${Math.round(x).toLocaleString("en-US")}`;
}

/** The ids 0 to n - 1 sorted by value, the largest first, then by id: a stable radix sort, 8 bits per pass. */
function mostFirst(values: Float64Array): Uint32Array {
  const n = values.length; // indexed loops: 20 to 40 times faster than from(..., fn) and for...of on millions
  let max = 0;
  for (let i = 0; i < n; i++) if (values[i] > max) max = values[i];
  const key = new Uint32Array(n); // increasing key: decreasing value
  let ids = new Uint32Array(n), next = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    key[i] = max - values[i];
    ids[i] = i;
  }
  const count = new Uint32Array(257);
  for (let shift = 0; shift < 32 && max >>> shift > 0; shift += 8) {
    count.fill(0);
    for (let i = 0; i < n; i++) count[((key[ids[i]] >>> shift) & 255) + 1]++;
    for (let d = 0; d < 256; d++) count[d + 1] += count[d];
    for (let i = 0; i < n; i++) next[count[(key[ids[i]] >>> shift) & 255]++] = ids[i];
    [ids, next] = [next, ids];
  }
  return ids;
}

async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  const stream = new Blob([bytes as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function logSumExp(t: Float64Array): number {
  let max = -Infinity, sum = 0;
  for (const x of t) max = Math.max(max, x);
  for (const x of t) sum += Math.exp(x - max);
  return max + Math.log(sum);
}

function f32(b: Uint8Array): Float32Array {
  if (b.byteOffset % 4) b = b.slice();
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}

/** Key letters (0 to 25) of each code point below 0x2000: ASCII letters, then the pack rules. */
function letterTable(letters: Record<string, string>): number[][] {
  const table: number[][] = Array.from({ length: 0x2000 }, () => []);
  for (let k = 0; k < 26; k++) table[65 + k] = table[97 + k] = [k];
  for (const [cp, key] of Object.entries(letters)) table[Number(cp)] = [...key].map((ch) => ch.charCodeAt(0) - 65);
  return table;
}

export const CHUNK_SECTIONS = ["names", "entries", "country", "count", "vec", "codes"];

/** The data of names start to start + n - 1 of a side: display forms, people per country, vectors. */
class Chunk {
  readonly bytes: Uint8Array;
  readonly start: Uint32Array; // name i is bytes[start[i], start[i + 1] - 1)
  readonly ptr: Uint32Array;
  readonly country: Uint8Array;
  readonly count: Uint32Array;
  readonly row: Int32Array; // row of name i in codes, -1 without a vector
  readonly codes: Uint8Array;

  /** The n names and their sections CHUNK_SECTIONS. */
  constructor(n: number, [names, entries, country, count, vec, codes]: Uint8Array[]) {
    this.bytes = names;
    this.start = new Uint32Array(n + 1);
    for (let i = 0, id = 0; id < n; i++) if (names[i] === 10) this.start[++id] = i + 1;
    const lens = varints(entries, new Uint32Array(n));
    this.ptr = new Uint32Array(n + 1);
    for (let i = 0; i < n; i++) this.ptr[i + 1] = this.ptr[i] + lens[i];
    this.country = country;
    this.count = varints(count, new Uint32Array(this.ptr[n]));
    this.row = new Int32Array(n);
    for (let i = 0, r = 0; i < n; i++) this.row[i] = (vec[i >> 3] >> (7 - (i & 7))) & 1 ? r++ : -1;
    this.codes = codes;
  }
}

/** The first or the last names of a pack: signature trie, people, and the data of the names in chunk files (read
 * with Model.addChunk). */
class Side {
  readonly n: number;
  readonly letters: Uint8Array; // index of the letter of each node in the pack alphabet
  readonly end: Uint32Array; // first node after the subtree
  readonly nameStart: Uint32Array; // names of node i: nameStart[i] to nameStart[i + 1] - 1
  readonly total: Float64Array; // people in the model data
  readonly part: Uint16Array; // the sum of part_hash over the letters, modulo parts
  readonly chunkSize: number;
  readonly chunks: (Chunk | undefined)[];
  private readonly alphaPrior: Float64Array; // alpha pi
  readonly logNorm: Float64Array; // log(N_c + alpha)
  private readonly scale: Float64Array; // per country: people of the model per count
  private readonly books: Float32Array;
  private readonly groups: number;
  private readonly groupSize: number;
  private readonly decoder = new TextDecoder();

  constructor(meta: Meta, side: "first" | "last", section: Section) {
    const m = meta[side];
    const get = (name: string) => section(`${side}.${name}`);
    const n = (this.n = m.names);
    this.letters = get("letters");
    this.end = varints(get("sizes"), new Uint32Array(m.nodes));
    for (let i = 0; i < m.nodes; i++) this.end[i] += i;
    const ends = varints(get("ends"), new Uint32Array(m.nodes));
    this.nameStart = new Uint32Array(m.nodes + 1);
    for (let i = 0; i < m.nodes; i++) this.nameStart[i + 1] = this.nameStart[i] + ends[i];

    this.chunkSize = m.chunk;
    this.chunks = new Array(Math.ceil(n / m.chunk));
    this.total = varints(get("total"), new Float64Array(n));
    this.alphaPrior = this.total.map((t) => (meta.alpha * (t + 0.5)) / m.prior_norm);
    this.part = new Uint16Array(n);
    const pathEnd: number[] = [], pathSum: number[] = []; // the nodes from the root to node i
    for (let i = 0; i < m.nodes; i++) {
      while (pathEnd.length && pathEnd[pathEnd.length - 1] <= i) {
        pathEnd.pop();
        pathSum.pop();
      }
      const sum = (pathSum.length ? pathSum[pathSum.length - 1] : 0) + meta.part_hash[meta.alphabet.charCodeAt(this.letters[i]) - 65];
      pathEnd.push(this.end[i]);
      pathSum.push(sum);
      this.part.fill(sum % meta.parts, this.nameStart[i], this.nameStart[i + 1]);
    }
    this.logNorm = Float64Array.from(m.people, (p) => Math.log(p + meta.alpha));
    this.scale = Float64Array.from(m.scale ?? m.people.map(() => 1));
    this.books = f32(get("books"));
    this.groups = m.groups;
    this.groupSize = m.group_size;
  }

  /** The chunk of name id and the index of the name in it; the chunk must be read. */
  private at(id: number): [Chunk, number] {
    const c = Math.floor(id / this.chunkSize);
    const chunk = this.chunks[c];
    if (!chunk) throw new Error(`chunk ${c} of the names is not read`);
    return [chunk, id - c * this.chunkSize];
  }

  /** [pack country, people] of name id. */
  countries(id: number): [number, number][] {
    const [k, i] = this.at(id);
    const out: [number, number][] = [];
    for (let e = k.ptr[i]; e < k.ptr[i + 1]; e++) out.push([k.country[e], k.count[e]]);
    return out;
  }

  name(id: number): string {
    const [k, i] = this.at(id);
    return this.decoder.decode(k.bytes.subarray(k.start[i], k.start[i + 1] - 1));
  }

  /** The trie node of the signature with the letter counts rest (in the order of the alphabet), or -1. */
  find(rest: Int32Array): number {
    let node = -1, i = 0, end = this.letters.length; // the children of node are the nodes i, this.end[i], ... before end
    for (let x = 0; x < 26; x++) {
      for (let t = 0; t < rest[x]; t++) {
        while (i < end && this.letters[i] < x) i = this.end[i];
        if (i === end || this.letters[i] !== x) return -1;
        node = i;
        end = this.end[i];
        i++;
      }
    }
    return node;
  }

  entry(id: number): Entry {
    const [k, i] = this.at(id);
    const C = this.logNorm.length;
    const a = this.alphaPrior[id];
    const x = new Float64Array(C);
    const seen = new Uint8Array(C), inS = new Uint8Array(C);
    for (let c = 0; c < C; c++) x[c] = Math.log(a) - this.logNorm[c]; // -Infinity when alpha = 0
    for (let e = k.ptr[i]; e < k.ptr[i + 1]; e++) {
      const c = k.country[e];
      x[c] = Math.log(k.count[e] * this.scale[c] + a) - this.logNorm[c];
      seen[c] = 1;
      inS[c] = k.count[e] > 0 ? 1 : 0;
    }
    const countries = k.country.subarray(k.ptr[i], k.ptr[i + 1]);
    const r = k.row[i];
    if (r < 0) return { x, seen, inS, countries, a: Math.log(a), vec: null };
    const g = this.groupSize;
    const vec = new Float64Array(this.groups * g);
    for (let j = 0; j < this.groups; j++) {
      const at = (j * 256 + k.codes[r * this.groups + j]) * g;
      for (let t = 0; t < g; t++) vec[j * g + t] = this.books[at + t];
    }
    return { x, seen, inS, countries, a: Math.log(a), vec };
  }
}

export class Model {
  readonly source: string;
  readonly countries: string[];
  readonly first: Side;
  readonly last: Side;
  readonly minFirst: number; // the data keeps the first names of at least minFirst people in the world
  readonly minLast: number;
  readonly parts: number; // parts of the real full names
  private readonly partHash: number[];
  private readonly partCache = new Map<number, Map<number, RealName>>(); // part -> key -> people
  private classes: { byRank: Uint32Array; byClass: Uint32Array; classStart: Uint32Array } | null = null;
  readonly population: Float64Array; // of each pack country
  private readonly logW: Float64Array;
  private base!: Float64Array; // log w_c - log Z_c, then - log(N_c + alpha) of both sides (after the sides)
  private baseOrder!: Uint8Array; // the countries, largest base first
  private baseMax = 0;
  private baseScaled!: Float64Array; // exp(base - baseMax)
  private baseSum = 0;
  private readonly terms = new Float64Array(256); // the terms of the countries where a name is seen
  private readonly logZ: Float64Array;
  private readonly gates: Float64Array; // (C, k)
  private readonly offsets: Float64Array;
  private readonly k: number;
  private readonly table: number[][];
  private readonly rank = new Uint8Array(26); // index of each letter A-Z in the pack alphabet

  private constructor(meta: Meta, section: Section) {
    this.source = meta.source;
    this.countries = meta.countries;
    this.population = Float64Array.from(meta.population);
    const total = meta.population.reduce((s, p) => s + p, 0);
    this.logW = Float64Array.from(meta.population, (p) => Math.log(p / total));
    this.logZ = Float64Array.from(meta.log_z);
    this.gates = Float64Array.from(meta.gates.flat());
    this.offsets = Float64Array.from(meta.offsets);
    // where neither name of a pair is seen, its term is a + base[c], with a the sum of the a of the two names
    const C = meta.countries.length;
    this.base = Float64Array.from({ length: C }, (_, c) => this.logW[c] - this.logZ[c]);
    this.baseOrder = Uint8Array.from(this.base.keys()).sort((p, q) => this.base[q] - this.base[p]);
    this.k = meta.gates[0].length;
    this.table = letterTable(meta.letters);
    for (let i = 0; i < 26; i++) this.rank[meta.alphabet.charCodeAt(i) - 65] = i;
    this.first = new Side(meta, "first", section);
    this.last = new Side(meta, "last", section);
    for (let c = 0; c < this.base.length; c++) this.base[c] -= this.first.logNorm[c] + this.last.logNorm[c];
    this.baseOrder.sort((p, q) => this.base[q] - this.base[p]);
    this.baseMax = this.base[this.baseOrder[0]];
    this.baseScaled = this.base.map((b) => Math.exp(b - this.baseMax));
    this.baseSum = this.baseScaled.reduce((s, x) => s + x, 0);
    this.minFirst = meta.min_first ?? 0;
    this.minLast = meta.min_last ?? 0;
    this.parts = meta.parts;
    this.partHash = meta.part_hash;
  }

  /** The model of a pack file. */
  static async load(bytes: Uint8Array): Promise<Model> {
    bytes = await gunzip(bytes);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const n = view.getUint32(8, true);
    if (new TextDecoder().decode(bytes.subarray(0, 4)) !== "ANAG" || view.getUint32(4, true) !== 1) {
      throw new Error("not a pack of version 1");
    }
    const meta = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + n))) as Meta;
    const base = 12 + n + ((4 - ((12 + n) % 4)) % 4);
    const section = (name: string) => {
      const [at, size] = meta.sections[name];
      return bytes.subarray(base + at, base + at + size);
    };
    return new Model(meta, section);
  }

  /** log p(f, l | c) */
  logP(f: Entry, l: Entry, c: number): number {
    let inter = 0;
    if (f.inS[c] && l.inS[c]) {
      let s = 0;
      if (f.vec && l.vec) for (let j = 0; j < this.k; j++) s += f.vec[j] * l.vec[j] * this.gates[c * this.k + j];
      inter = s + this.offsets[c];
    }
    return f.x[c] + l.x[c] + inter - this.logZ[c];
  }

  /** The part of the real full names with the letters of the query. */
  partOf(query: string): number {
    let sum = 0;
    for (const ch of this.key(query)) sum += this.partHash[ch.charCodeAt(0) - 65];
    return sum % this.parts;
  }

  /** The part of the full name (f, l). */
  pairPart(f: number, l: number): number {
    return (this.first.part[f] + this.last.part[l]) % this.parts;
  }

  /** The file of part j in the directory <pack>.real/ */
  partFile(j: number): string {
    return `${j.toString(16).padStart(3, "0")}.gz`;
  }

  hasPart(j: number): boolean {
    return this.partCache.has(j);
  }

  /** Read the bytes of the file of part j. */
  async addPart(j: number, bytes: Uint8Array): Promise<void> {
    bytes = await gunzip(bytes);
    let n = 0;
    for (const b of bytes) if (b < 0x80) n++;
    const v = varints(bytes, new Float64Array(n));
    if (!this.classes) {
      const byRank = mostFirst(this.first.total);
      const part = this.last.part;
      const classStart = new Uint32Array(this.parts + 1);
      for (let l = 0; l < part.length; l++) classStart[part[l] + 1]++;
      for (let c = 0; c < this.parts; c++) classStart[c + 1] += classStart[c];
      const byClass = new Uint32Array(part.length), at = classStart.slice(0, this.parts);
      const rankL = mostFirst(this.last.total);
      for (let r = 0; r < rankL.length; r++) byClass[at[part[rankL[r]]]++] = rankL[r]; // each class in the order of the ranks
      this.classes = { byRank, byClass, classStart };
    }
    const { byRank, byClass, classStart } = this.classes;
    const out = new Map<number, RealName>();
    const m = v[0];
    for (let i = 1, e = 1 + 3 * m, row = 0, col = 0; i < 1 + 3 * m; i += 3) {
      if (i === 1 || v[i] > 0) [row, col] = [row + v[i], v[i + 1]];
      else col += 1 + v[i + 1];
      const f = byRank[row];
      const l = byClass[classStart[(j - this.first.part[f] + this.parts) % this.parts] + col];
      const name: RealName = { people: 0, countries: [] };
      for (let k = 0, c = -1; k <= v[i + 2]; k++, e += 2) {
        c += v[e] + 1;
        name.countries.push([c, v[e + 1] + 1]);
        name.people += v[e + 1] + 1;
      }
      out.set(f * this.last.n + l, name);
    }
    this.partCache.set(j, out);
  }

  /** The real full name (f, l) of its part, which must be read first (addPart); undefined if nobody has it. */
  private realName(f: number, l: number): RealName | undefined {
    const part = this.partCache.get(this.pairPart(f, l));
    if (!part) throw new Error(`part ${this.pairPart(f, l)} of the real full names is not read`);
    return part.get(f * this.last.n + l);
  }

  /** [country, people] pairs, most people first, then by country code. */
  private byPeople(pairs: [number, number][]): [string, number][] {
    return pairs.map(([c, n]): [string, number] => [this.countries[c], n])
      .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  }

  /** [country, people] of a name in the pack countries (or in pack country c), most people first. */
  nameCountries(side: "first" | "last", id: number, c = -1): [string, number][] {
    return this.byPeople(this[side].countries(id).filter(([k]) => c < 0 || k === c));
  }

  /** [country, people] of the full name (f, l) in the pack countries (or in pack country c), most people first. */
  realCountries(f: number, l: number, c = -1): [string, number][] {
    return this.byPeople((this.realName(f, l)?.countries ?? []).filter(([k]) => c < 0 || k === c));
  }

  /** People with the full name (f, l) in the pack countries, or in pack country c (0: none). The part of (f, l)
   * must be read first (addPart). */
  real(f: number, l: number, c = -1): number {
    const name = this.realName(f, l);
    return c < 0 ? (name?.people ?? 0) : (name?.countries.find(([k]) => k === c)?.[1] ?? 0);
  }

  /** The number of each letter of a key, in the order of the pack alphabet. */
  private counts(key: string): Int32Array {
    const q = new Int32Array(26);
    for (const ch of key) q[this.rank[ch.charCodeAt(0) - 65]]++;
    return q;
  }

  /** The ids of the names of a side with the letters of the key (the names of one trie node). */
  private sameLetters(side: Side, key: string): [number, number] {
    const node = side.find(this.counts(key));
    return node < 0 ? [0, 0] : [side.nameStart[node], side.nameStart[node + 1]];
  }

  /** Id of the name with this key, or -1. */
  private lookup(side: Side, key: string): number {
    const [a, b] = this.sameLetters(side, key);
    for (let i = a; i < b; i++) if (this.key(side.name(i)) === key) return i;
    return -1;
  }

  /** The chunks with the names ids of a side that are not read yet, added to out as [side, chunk]. */
  needIds(side: "first" | "last", ids: Iterable<number>, out: [string, number][] = []): [string, number][] {
    const S = this[side], seen = new Set(out.filter(([s]) => s === side).map(([, c]) => c));
    for (const id of ids) {
      const c = Math.floor(id / S.chunkSize);
      if (!S.chunks[c] && !seen.has(c)) {
        seen.add(c);
        out.push([side, c]);
      }
    }
    return out;
  }

  /** The chunks to read (addChunk) before a search of the query: those of the anagrams and of the names of
   * the query. */
  need(query: string): [string, number][] {
    const [fs, ls] = this.anagrams(query);
    const out = this.needIds("last", ls, this.needIds("first", fs));
    const words = query.split(/\s+/).filter(Boolean);
    for (let i = 1; i < words.length; i++) {
      const [a, b] = this.sameLetters(this.first, this.key(words.slice(0, i).join(" ")));
      const [c, d] = this.sameLetters(this.last, this.key(words.slice(i).join(" ")));
      this.needIds("first", Array.from({ length: b - a }, (_, k) => a + k), out);
      this.needIds("last", Array.from({ length: d - c }, (_, k) => c + k), out);
    }
    return out;
  }

  /** The file of a chunk in the directory <pack>.names/ */
  chunkFile(side: string, c: number): string {
    return `${side}/${c.toString(16).padStart(4, "0")}.gz`;
  }

  /** Read the bytes of the file of a chunk. */
  async addChunk(side: "first" | "last", c: number, bytes: Uint8Array): Promise<void> {
    bytes = await gunzip(bytes);
    const S = this[side];
    const lens = new Float64Array(CHUNK_SECTIONS.length);
    let at = 0;
    for (let k = 0, v = 0, scale = 1; k < lens.length; at++) {
      v += (bytes[at] & 0x7f) * scale;
      if (bytes[at] < 0x80) [lens[k++], v, scale] = [v, 0, 1];
      else scale *= 128;
    }
    const parts = Array.from(lens, (len) => bytes.subarray(at, (at += len)));
    S.chunks[c] = new Chunk(Math.min(S.chunkSize, S.n - c * S.chunkSize), parts);
  }

  /** The most probable countries of the query, p(c | f, l), for the split of its words into a first and a last
   * name of the pack with the largest probability; null if no split gives two names of the pack. */
  origin(query: string, top = 5): Origin[] | null {
    const words = query.split(/\s+/).filter(Boolean);
    let best: Float64Array | null = null, bestLp = -Infinity;
    for (let i = 1; i < words.length; i++) {
      const f = this.lookup(this.first, this.key(words.slice(0, i).join(" ")));
      const l = this.lookup(this.last, this.key(words.slice(i).join(" ")));
      if (f < 0 || l < 0) continue;
      const fe = this.first.entry(f), le = this.last.entry(l);
      const t = Float64Array.from(this.logW, (w, c) => w + this.logP(fe, le, c));
      const lp = logSumExp(t);
      if (lp > bestLp) [best, bestLp] = [t, lp];
    }
    return best && this.byShare(best, bestLp, top);
  }

  /** The countries of the terms t of the mixture (sum: exp(lp)), most probable first. */
  private byShare(t: Float64Array, lp: number, top: number): Origin[] {
    const out = Array.from(t, (x, c) => ({ country: this.countries[c], p: Math.exp(x - lp) }));
    return out.sort((a, b) => b.p - a.p).slice(0, top);
  }

  /** logPMixture, faster: the exact terms of the countries where f or l is seen (about 12 of 105); in the other
   * countries neither name is seen, there is no pair term, and their terms sum to exp(a_f + a_l) times a constant
   * per country. */
  private mixture(f: Entry, l: Entry, expected: Float64Array | null = null): [number, number] {
    const fc = f.countries, lc = l.countries, terms = this.terms;
    let i = 0, j = 0, n = 0, best = -1, top = -Infinity, rest = this.baseSum;
    while (i < fc.length || j < lc.length) {
      const c = j === lc.length || (i < fc.length && fc[i] <= lc[j]) ? fc[i] : lc[j];
      const inF = i < fc.length && fc[i] === c, inL = j < lc.length && lc[j] === c;
      if (inF) i++;
      if (inL) j++;
      const t = (terms[n++] = this.logW[c] + this.logP(f, l, c));
      if (expected && inF && inL) expected[c] += Math.exp(t); // (P_c / P) p(f, l | c), where both names are seen
      if (t > top) [top, best] = [t, c];
      rest -= this.baseScaled[c];
    }
    const a = f.a + l.a;
    for (const c of this.baseOrder) { // the largest term of the other countries
      if (f.seen[c] || l.seen[c]) continue;
      if (a + this.base[c] > top) [top, best] = [a + this.base[c], c];
      break;
    }
    let sum = rest > 0 ? rest * Math.exp(a + this.baseMax - top) : 0;
    for (let k = 0; k < n; k++) sum += Math.exp(terms[k] - top);
    return [top + Math.log(sum), best];
  }

  /** log p(f, l) under the mixture of the pack countries, and the country with the largest term. */
  logPMixture(f: Entry, l: Entry): [number, number] {
    const t = new Float64Array(this.countries.length);
    let best = 0;
    for (let c = 0; c < t.length; c++) {
      t[c] = this.logW[c] + this.logP(f, l, c);
      if (t[c] > t[best]) best = c;
    }
    let sum = 0;
    for (let c = 0; c < t.length; c++) sum += Math.exp(t[c] - t[best]);
    return [t[best] + Math.log(sum), best];
  }

  /** The key of a text: its letters A-Z, accents removed, in upper case (Élodie -> ELODIE). */
  key(text: string): string {
    let out = "";
    for (const ch of text.normalize("NFC")) {
      const cp = ch.codePointAt(0)!;
      if (cp < this.table.length) for (const k of this.table[cp]) out += String.fromCharCode(65 + k);
    }
    return out;
  }

  /** All (first name id, last name id) pairs whose letters are the letters of the query. */
  anagrams(query: string): [number[], number[]] {
    const q = this.counts(this.key(query));
    const n = q.reduce((s, x) => s + x, 0);
    const F = this.first, L = this.last;
    const used = new Int32Array(26), rest = new Int32Array(26);
    const pathLetter: number[] = [], pathEnd: number[] = []; // the nodes from the root to node i
    const fs: number[] = [], ls: number[] = [];
    for (let i = 0; i < F.letters.length; ) {
      while (pathEnd.length && i >= pathEnd[pathEnd.length - 1]) {
        used[pathLetter.pop()!]--;
        pathEnd.pop();
      }
      const x = F.letters[i];
      if (used[x] === q[x] || pathEnd.length >= n - 2) {
        i = F.end[i]; // no letter x left, or fewer than 2 letters for the last name
        continue;
      }
      used[x]++;
      pathLetter.push(x);
      pathEnd.push(F.end[i]);
      if (F.nameStart[i + 1] > F.nameStart[i]) {
        for (let k = 0; k < 26; k++) rest[k] = q[k] - used[k];
        const node = L.find(rest);
        if (node >= 0) {
          for (let f = F.nameStart[i]; f < F.nameStart[i + 1]; f++) {
            for (let l = L.nameStart[node]; l < L.nameStart[node + 1]; l++) {
              fs.push(f);
              ls.push(l);
            }
          }
        }
      }
      i++;
    }
    return [fs, ls];
  }

  /** The anagram names of the query: the real full names first (most people first), then the most probable. */
  search(query: string, country: string | null = null, limit = 20): Search {
    const r = this.ranked(query, country);
    return { ...r.summary, results: r.order.slice(0, limit).map(r.result) };
  }

  /** The anagrams of the query, ranked once, to read page by page: the summary, the first n real full names, and
   * the first n other names. */
  ranking(query: string, country: string | null = null) {
    const { order, result, summary, byCountry, mass, model } = this.ranked(query, country);
    return {
      summary,
      real: (n: number) => order.slice(0, Math.min(n, summary.realNames)).map(result),
      possible: (n: number) => order.slice(summary.realNames, summary.realNames + n).map(result),
      model: (n: number) => model().slice(0, n).map(result), // all the names, the most probable first
      byCountry,
      mass, // sum of p(f, l) of all the names: times the population, the people expected
    };
  }

  /** The anagrams of the query, scored and sorted (the real full names first), and the result of one of them. */
  private ranked(query: string, country: string | null) {
    let [fs, ls] = this.anagrams(query);
    const c = country === null ? -1 : this.countries.indexOf(country);
    if (country !== null && c < 0) throw new Error(`country ${country} is not in the pack`);
    const firsts = new Map<number, Entry>(), lasts = new Map<number, Entry>();
    const get = (side: Side, cache: Map<number, Entry>, id: number): Entry => {
      let e = cache.get(id);
      if (!e) cache.set(id, (e = side.entry(id)));
      return e;
    };
    if (c >= 0) { // a country: the anagrams of the names seen in it
      const keep = fs.map((f, i) => get(this.first, firsts, f).seen[c] && get(this.last, lasts, ls[i]).seen[c]);
      [fs, ls] = [fs.filter((_, i) => keep[i]), ls.filter((_, i) => keep[i])];
    }
    let real = Float64Array.from(fs, (f, i) => this.real(f, ls[i], c));
    if (c >= 0) { // a country: a synthetic name has no real person anywhere (else it is real elsewhere)
      const keep = fs.map((f, i) => real[i] > 0 || this.real(f, ls[i], -1) === 0);
      [fs, ls, real] = [fs.filter((_, i) => keep[i]), ls.filter((_, i) => keep[i]), real.filter((_, i) => keep[i])];
    }
    const lp = new Float64Array(fs.length);
    const best = new Int32Array(fs.length).fill(c);
    const expected = c < 0 ? new Float64Array(this.countries.length) : null; // per country (the world only)
    for (let i = 0; i < fs.length; i++) {
      const f = get(this.first, firsts, fs[i]), l = get(this.last, lasts, ls[i]);
      if (c >= 0) lp[i] = this.logP(f, l, c);
      else [lp[i], best[i]] = this.mixture(f, l, real[i] > 0 ? null : expected); // synthetic: in all its countries
    }
    const order = Array.from(lp.keys())
      .sort((a, b) => real[b] - real[a] || lp[b] - lp[a] || fs[a] - fs[b] || ls[a] - ls[b]);
    const result = (i: number): Result => ({
      first: this.first.name(fs[i]),
      last: this.last.name(ls[i]),
      country: this.countries[best[i]],
      logP: lp[i],
      real: real[i],
      firstPeople: this.first.total[fs[i]],
      firstCountries: this.nameCountries("first", fs[i], c),
      lastPeople: this.last.total[ls[i]],
      lastCountries: this.nameCountries("last", ls[i], c),
      realCountries: this.realCountries(fs[i], ls[i], c),
    });
    const realNames = real.reduce((s, x) => s + (x > 0 ? 1 : 0), 0);
    const realPeople = real.reduce((s, x) => s + x, 0);
    // Per country, most first (computed once, when asked; the search in the world only): the real people of the
    // real full names, and the people that the model expects, as in the view of the country: the names whose first
    // and last name are seen in it, real there or real nowhere (synthetic); P_c p(f, l | c) people each.
    let counts: { real: [string, number][]; possible: [string, number][] } | null = null;
    const total = this.population.reduce((s, x) => s + x, 0);
    const byCountry = () => {
      if (counts) return counts;
      const people = new Float64Array(this.countries.length), synthetic = Float64Array.from(expected ?? []);
      const index = new Map(this.countries.map((cc, i) => [cc, i]));
      for (let i = 0; i < fs.length; i++) {
        if (!(real[i] > 0)) continue;
        const f = get(this.first, firsts, fs[i]), l = get(this.last, lasts, ls[i]);
        for (const [cc, x] of this.realCountries(fs[i], ls[i], c)) {
          const j = index.get(cc)!;
          people[j] += x;
          if (expected) synthetic[j] += Math.exp(this.logW[j] + this.logP(f, l, j)); // real there: the model too
        }
      }
      const list = (a: Float64Array, scale = 1) => Array.from(a, (x, i): [string, number] => [this.countries[i], x * scale])
        .filter(([, x]) => x > 1e-9).sort((p, q) => q[1] - p[1] || (p[0] < q[0] ? -1 : 1));
      return (counts = { real: list(people), possible: list(synthetic, total) });
    };
    // the probability of all the names together, in the people of the view (the world, or its country)
    let mass = 0;
    for (let i = 0; i < fs.length; i++) mass += Math.exp(lp[i]);
    // the order of the model: all the names, the most probable first (computed once, when asked)
    let byModel: number[] | null = null;
    const model = () => (byModel ??= Array.from(lp.keys()).sort((a, b) => lp[b] - lp[a] || fs[a] - fs[b] || ls[a] - ls[b]));
    return { order, result, byCountry, mass, model, summary: { query, country, found: fs.length, realNames, realPeople, origin: this.origin(query) } };
  }
}
