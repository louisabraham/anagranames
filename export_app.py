# Export the packs of the anagram apps (app/anagram.py, app/anagram.ts) to app/data/:
#   world.anag.gz      all names and countries of name_dataset (world mixture; a search in one country keeps the
#                      names seen in it)
#   world.names/       the display forms, people per country and vectors of the names of the world pack, in
#                      chunks of 1024 names: a search reads the chunks of the names it finds in the trie
#   world.real/        the real full names of the world pack, with their people per country, in 4096 parts
#   <pack>.check.json  log p of sample names computed with pairmodel.py, to test the apps
# Run: caffeinate -i .venv/bin/python export_app.py   (about 9 min)
#
# The only count filter is the one of the data (prepare_world.py, data/filter.json, metadata min_first and
# min_last): the names with fewer people in the world are not in the model. The world pack keeps all the names of
# the model, and all the full names of name_dataset with their people per country.
#
# The pack uses the world model with the counts of all people (train + valid + test). Its log Z_c for these
# counts takes one exact pass on the GPU (about 4 min), kept in <model>_logz_all_<data dir>.npy.
# A pack contains a subset of the names and countries of a model. The score of a name pair in a pack is its
# score in the full model.
# The world mixture weights each country by its population (population.tsv, metadata "population"), not by its
# people in name_dataset: name_dataset has 0.09% of its people in Japan and 4.9% in France, for 1.9% and 1.0% of
# the population of its countries. With these weights, the expected people with a full name in the world (the
# population of the pack countries times p) is at least the expected people in each country.
# A pack also lists the real full names: the pairs of its names that people of name_dataset have in its country.
# The world pack lists them (155 million) with their people in each country in the files world.real/<part>.gz,
# so that a search reads only the part of its letters.
#
# Pack format: gzip of
#   "ANAG", u32 version, u32 n, n bytes of JSON metadata, zero padding to a multiple of 4, sections.
#   metadata "sections" gives name -> [offset, length] of each section, from the end of the padding
#   (offsets are multiples of 4). Varints are LEB128. Per side ("first", "last"):
#   side.letters  u8 per trie node: index of its letter in metadata "alphabet" (the letters, the rarest in the
#                 names first). The trie holds the letter signatures of the names (key letters in the order of the
#                 alphabet); its nodes are in preorder, the root is not stored. A search skips the branches of
#                 the letters that its query does not have: with the rarest letters at the top, it skips more.
#   side.sizes    varint per node: nodes in its subtree, itself included
#   side.ends     varint per node: names whose signature ends at the node
#   side.names    UTF-8 display forms, one per line, in trie order (name id = line number)
#   side.total    varint per name: people in the model data (prior pi of the model)
#   side.entries  varint per name: pack countries where the name is seen
#   side.country  u8 per entry: index in the pack countries, increasing within a name
#   side.count    varint per entry: people with this name in this country. For a country of metadata "bases"
#                 (France: INSEE), the people of the base: its entries are the names of the base and the names seen
#                 in it in name_dataset (0 when not in the base). The model uses count * scale[c] (side metadata
#                 "scale": 1, or the people of the country in name_dataset per person of the base), and the pair
#                 term applies to the names of count > 0.
#   side.vec      1 bit per name, most significant bit first: the name has a vector
#   side.codes    u8 (names with a vector, groups): product-quantization codes
#   side.books    f32 (groups, 256, group size): the vector is books[j, codes[j]] for each group j
# side.names, entries, country, count, vec and codes are in the chunk files (side metadata "chunk": names per chunk)
# world.names/<side>/<chunk>.gz, chunk in 4 hexadecimal digits: the gzip of 6 varints (the length of each of these
# sections for the names of the chunk), then the 6 sections.
#
# Parts of a world pack. The part of a set of letters is the sum of part_hash[letter] over its letters, modulo
# parts (metadata). The anagrams of a query have the letters of the query: they are all in its part. The part of
# a full name is (h_f + h_l) mod parts, with h the part of each name. Thus for a first name f, the last names of
# part j are the class of the last names with h_l = (j - h_f) mod parts (74 names on average). The rank of a name
# is its position when the names of its side are sorted by people (side.total, most first), then by id. A part
# file is the gzip of 3 varints per full name, in increasing order of (rank of f, index of l in its class sorted
# by rank), then 2 varints per (full name, country) in the same order. Per full name: rank of f minus the rank
# of the previous full name (the rank itself for the first one); the index of l, minus the index of the previous
# full name plus 1 if the rank of f is the same; countries with this full name minus 1. Per country, in
# increasing order: its index minus the index of the previous country of the full name plus 1 (the index itself
# for the first one); people minus 1. The file starts with 1 varint: its number of full names.
import argparse
import gzip
import json
import os
import re
import struct
import time
import unicodedata
from collections import Counter

import numpy as np
from unidecode import unidecode

import namedata
import pairmodel as pm

WORLD_MODEL = "models/fr_insee/world_all_k16_wd0.1_u0_me20_s10000_gates_pool4096_cal_pq2.npz"  # France: INSEE base
POPULATION = "population.tsv"
VERSION = 1
PARTS = 4096  # parts of the real full names of a world pack
CHUNK = 1024  # names per chunk file of the world pack
PART_HASH = np.random.default_rng(1).integers(0, PARTS, 26)
RANGES = [(0x0080, 0x0250), (0x0300, 0x0370), (0x1E00, 0x1F00)]  # non-ASCII characters kept by prepare_world.py
NON_KEY = re.compile(r"[^A-Z]")


def key_of(name):
    """The A-Z key of prepare_world.py."""
    return NON_KEY.sub("", unidecode(unicodedata.normalize("NFC", name)).upper())


def letter_rules():
    """Key letters of the non-ASCII characters, for the queries of the apps."""
    return {str(cp): k for lo, hi in RANGES for cp in range(lo, hi) if (k := key_of(chr(cp)))}


def varints(a):
    """LEB128 bytes of non-negative integers."""
    a = np.asarray(a, np.int64)
    assert (a >= 0).all()
    size = 1 + sum((a >= 1 << (7 * k)).astype(np.int64) for k in range(1, 9))
    owner = np.repeat(np.arange(len(a)), size)
    j = np.arange(size.sum()) - np.repeat(np.cumsum(size) - size, size)  # byte index within the value
    out = ((a[owner] >> (7 * j)) & 0x7F).astype(np.uint8)
    out[j < size[owner] - 1] |= 0x80
    return out.tobytes()


def ranges(start, length):
    """The integer ranges [start_i, start_i + length_i), concatenated."""
    return np.arange(length.sum()) - np.repeat(np.cumsum(length) - length, length) + np.repeat(start, length)


def letter_order(keys):
    """The 26 letters, the rarest in the names first."""
    counts = Counter("".join(keys))
    return "".join(sorted(map(chr, range(65, 91)), key=lambda ch: (counts[ch], ch)))


def signature_trie(keys, people, displays, alphabet):
    """Order the names by signature (then most people first) and build the trie of the signatures in preorder.
    The signature of a name is its letters in the order of the alphabet. Returns the name order and, per node, the
    letter (its index in the alphabet), the subtree size and the number of names ending there."""
    rank = {ch: chr(65 + i) for i, ch in enumerate(alphabet)}
    sigs = ["".join(sorted(rank[ch] for ch in k)) for k in keys]
    order = sorted(range(len(keys)), key=lambda i: (sigs[i], -people[i], displays[i]))
    letters, sizes, ends, path, prev = [], [], [], [], ""
    for i in order:
        s = sigs[i]
        common = 0
        while common < min(len(prev), len(s)) and prev[common] == s[common]:
            common += 1
        while len(path) > common:
            j = path.pop()
            sizes[j] = len(letters) - j
        for ch in s[common:]:
            path.append(len(letters))
            letters.append(ord(ch) - 65)
            sizes.append(0)
            ends.append(0)
        ends[path[-1]] += 1
        prev = s
    while path:
        j = path.pop()
        sizes[j] = len(letters) - j
    return np.array(order), np.array(letters, np.uint8), np.array(sizes), np.array(ends)


class Names:
    """The names of one side of a model: keys, display forms, people (total) and people per country (CSR),
    vector rows (-1: no vector), codes and codebooks."""

    def __init__(self, keys, displays, total, ptr, country, count, vec_row, codes, books):
        self.keys, self.displays, self.total = keys, displays, np.asarray(total, np.int64)
        self.ptr, self.country, self.count = ptr, country, count
        self.vec_row, self.codes, self.books = vec_row, codes, books

    def pack(self, side, countries, alphabet, chunk_dir):
        """Sections of the names seen in the given countries (model country indices). Returns the sections, the side
        metadata and the model id of each pack name. With chunk_dir, the display forms, the people per country and
        the vectors are in the chunk files chunk_dir/<side>/<chunk>.gz instead."""
        pack_index = np.full(256, -1)
        pack_index[countries] = np.arange(len(countries))
        owner = np.repeat(np.arange(len(self.total)), np.diff(self.ptr))
        inside = pack_index[self.country] >= 0
        keep = np.bincount(owner[inside], minlength=len(self.total)) > 0
        sel = inside & keep[owner]
        ids = np.nonzero(keep)[0]
        local = np.cumsum(keep) - 1
        ptr = np.searchsorted(local[owner[sel]], np.arange(len(ids) + 1))
        country, count = pack_index[self.country[sel]], self.count[sel]

        order, letters, sizes, ends = signature_trie([self.keys[i] for i in ids], self.total[ids], [self.displays[i] for i in ids],
                                                     alphabet)
        lens = np.diff(ptr)[order]
        cells = ranges(ptr[:-1][order], lens)
        rows = self.vec_row[ids[order]]
        ids = ids[order]
        sections = {
            "letters": letters.tobytes(), "sizes": varints(sizes), "ends": varints(ends),
            "names": ("\n".join(self.displays[i] for i in ids) + "\n").encode(),
            "total": varints(self.total[ids]), "entries": varints(lens),
            "country": country[cells].astype(np.uint8).tobytes(), "count": varints(count[cells]),
            "vec": np.packbits(rows >= 0).tobytes(), "codes": self.codes[rows[rows >= 0]].tobytes(),
            "books": self.books.astype("<f4").tobytes(),
        }
        meta = {"names": len(ids), "nodes": len(letters), "vectors": int((rows >= 0).sum()),
                "groups": self.books.shape[0], "group_size": self.books.shape[2]}
        for k in CHUNK_SECTIONS:
            del sections[k]
        write_chunks(f"{chunk_dir}/{side}", [self.displays[i] for i in ids], lens, country[cells], count[cells],
                     rows, self.codes)
        meta["chunk"] = CHUNK
        return {f"{side}.{k}": v for k, v in sections.items()}, meta, ids


CHUNK_SECTIONS = ("names", "entries", "country", "count", "vec", "codes")


def write_chunks(out_dir, displays, lens, country, count, rows, codes):
    """Write the chunk files of the names of a side, CHUNK names per file, in the order of the names: each is the
    gzip of 6 varints (the length of each part), then the 6 parts of CHUNK_SECTIONS for its names."""
    os.makedirs(out_dir, exist_ok=True)
    cum = np.concatenate([[0], np.cumsum(lens)])
    for c in range(-(-len(displays) // CHUNK)):
        a, b = c * CHUNK, min((c + 1) * CHUNK, len(displays))
        e = slice(cum[a], cum[b])
        r = rows[a:b]
        parts = [("\n".join(displays[a:b]) + "\n").encode(), varints(lens[a:b]),
                 country[e].astype(np.uint8).tobytes(), varints(count[e]), np.packbits(r >= 0).tobytes(),
                 codes[r[r >= 0]].tobytes()]
        with open(f"{out_dir}/{c:04x}.gz", "wb") as fh:
            fh.write(gzip.compress(varints([len(x) for x in parts]) + b"".join(parts), 9, mtime=0))


def pack_pairs(f, l, count, ids, country=None):
    """The real full names (model name ids f, l, with count people in country) whose two names are in the pack
    (ids: model id of each pack name, per side): their keys first id * (number of last names) + last id in the
    pack, their people and their countries, in increasing order of key, then of country."""
    index = []
    for x, side in ((f, "first"), (l, "last")):
        m = np.full(max(int(x.max(initial=0)), int(ids[side].max(initial=0))) + 1, -1)
        m[ids[side]] = np.arange(len(ids[side]))
        index.append(m[x])
    ok = (index[0] >= 0) & (index[1] >= 0)
    key = index[0][ok] * len(ids["last"]) + index[1][ok]
    country = np.zeros(len(key), np.int64) if country is None else np.asarray(country)[ok]
    order = np.lexsort((country, key))
    return key[order], np.asarray(count)[ok][order], country[order]


def check_rows(key, count, n_first, n_last, rng, n=50):
    """Check rows: some real full names (pack keys, increasing, and people) and random pairs (mostly 0 people)."""
    some = key[rng.choice(len(key), min(n, len(key)), replace=False)] if len(key) else np.zeros(0, np.int64)
    rand = rng.integers(0, n_first, n) * n_last + rng.integers(0, n_last, n)
    rows = []
    for k in np.concatenate([some, rand]):
        i = np.searchsorted(key, k)
        rows.append({"first": int(k // n_last), "last": int(k % n_last),
                     "real": int(count[i]) if i < len(key) and key[i] == k else 0})
    return rows


def write_parts(out_dir, key, count, country, names, ids):
    """Write the real full names of the world pack to the part files out_dir/<part>.gz, part in 3 hexadecimal
    digits (format at the top of the file). One entry per (full name, country): pack key, people and country, in
    increasing order of key, then of country."""
    n_first, n_last = len(ids["first"]), len(ids["last"])
    h, total = {}, {}
    for side in ("first", "last"):
        keys = [names[side].keys[i] for i in ids[side]]
        letters = np.frombuffer("".join(keys).encode("ascii"), np.uint8) - 65
        h[side] = np.add.reduceat(PART_HASH[letters], np.cumsum([0] + [len(k) for k in keys[:-1]])) % PARTS
        total[side] = names[side].total[ids[side]]
    rank = np.empty(n_first, np.int64)
    rank[np.lexsort((np.arange(n_first), -total["first"]))] = np.arange(n_first)
    by_class = np.lexsort((np.arange(n_last), -total["last"], h["last"]))  # last names by class, then by rank
    index = np.empty(n_last, np.int64)  # index of a last name in its class
    index[by_class] = np.arange(n_last) - np.searchsorted(h["last"][by_class], h["last"][by_class])
    pair, start = np.unique(key, return_index=True)
    size = np.diff(np.r_[start, len(key)])  # countries of each full name
    f, l = pair // n_last, pair % n_last
    part, row, col = (h["first"][f] + h["last"][l]) % PARTS, rank[f], index[l]
    order = np.lexsort((col, row, part))
    part, row, col, start, size = part[order], row[order], col[order], start[order], size[order]
    first = np.r_[True, part[1:] != part[:-1]]
    new_row = first | np.r_[True, row[1:] != row[:-1]]
    d_row = np.where(first, row, row - np.r_[0, row[:-1]])
    d_col = np.where(new_row, col, col - np.r_[0, col[:-1]] - 1)
    cells = ranges(start, size)  # the entries in the order of the full names
    c, n = country[cells], count[cells]
    first_c = np.zeros(len(cells), bool)
    first_c[np.cumsum(size) - size] = True
    d_c = c - np.where(first_c, -1, np.r_[0, c[:-1]]) - 1
    bounds = np.searchsorted(part, np.arange(PARTS + 1))
    cell_bounds = np.r_[0, np.cumsum(size)][bounds]
    os.makedirs(out_dir, exist_ok=True)
    for j in range(PARTS):
        s, e = slice(bounds[j], bounds[j + 1]), slice(cell_bounds[j], cell_bounds[j + 1])
        data = np.concatenate([[s.stop - s.start], np.stack([d_row[s], d_col[s], size[s] - 1], 1).ravel(),
                               np.stack([d_c[e], n[e] - 1], 1).ravel()])
        with open(f"{out_dir}/{j:03x}.gz", "wb") as fh:
            fh.write(gzip.compress(varints(data), 9, mtime=0))


def write_pack(path, meta, sections):
    blob, where = bytearray(), {}
    for name, data in sections.items():
        blob += bytes(-len(blob) % 4)
        where[name] = [len(blob), len(data)]
        blob += data
    head = json.dumps(dict(meta, sections=where)).encode()
    raw = b"ANAG" + struct.pack("<II", VERSION, len(head)) + head
    with gzip.open(path, "wb", compresslevel=9) as fh:
        fh.write(raw + bytes(-len(raw) % 4) + blob)


def vector_rows(d, name, n):
    """Row in d[name_codes] of each of the n names, -1 for names without a vector."""
    has = np.unpackbits(d[f"{name}_rows"], count=n).astype(bool)
    rows = np.full(n, -1)
    rows[has] = np.arange(has.sum())
    return rows


def world_names(ds, d, data_dir):
    """Names of both sides of the world model, with the counts of ds.n_train. A country with a base (ds.base: INSEE
    for France) has the people of its base instead, and its names are the names of the base and the names seen in it
    in the data (people 0 when not in the base)."""
    out = {}
    for s, (side, ids, n, vec) in enumerate((("first", ds.f, ds.n_first, "U"), ("last", ds.l, ds.n_last, "V"))):
        vocab = [line.split("\t") for line in open(f"{data_dir}/vocab_{side}.tsv", encoding="utf-8").read().split("\n")[:-1]]
        dense = np.bincount(ids.astype(np.int64) * ds.C + ds.c, weights=ds.n_train, minlength=n * ds.C).reshape(n, ds.C)
        total = dense.sum(1)
        present = dense > 0
        for c, counts in ds.base.items():
            present[:, c] |= counts[s] > 0
            dense[:, c] = counts[s]
        cells = np.nonzero(present.reshape(-1))[0]
        ptr = np.searchsorted(cells // ds.C, np.arange(n + 1))
        out[side] = Names([v[0] for v in vocab], [v[1] for v in vocab], [int(v[2]) for v in vocab], ptr,
                          (cells % ds.C).astype(np.uint8), dense.reshape(-1)[cells].astype(np.int64),
                          vector_rows(d, vec, n), d[f"{vec}_codes"], d[f"{vec}_codebooks"])
        assert np.array_equal(total, out[side].total)
    return out


def population(countries):
    """Population of each country (population.tsv)."""
    pop = dict(line.split("\t") for line in open(POPULATION, encoding="utf-8").read().splitlines() if line[0] != "#")
    return np.array([int(pop[c]) for c in countries], np.int64)


def world_references(model, ds, log_z, pop):
    """A function giving, for a pack, log p(f, l | c) computed with the arrays of pairmodel.py: for people of the
    data (in the support) and random names (mostly outside it), and the world mixture for the world pack."""
    in_f, in_l = (m.reshape(-1, ds.C) for m in pm.support(ds))
    X, Y = np.array(model.X, np.float64), np.array(model.Y, np.float64)
    U, V, G, B = (np.array(a, np.float64) for a in (model.U, model.V, model.G, model.B))

    def log_p(f, l, c):
        inter = (U[f] * G[c] * V[l]).sum(-1) + B[c]
        return X[f, c] + Y[l, c] + (in_f[f, c] & in_l[l, c]) * inter - log_z[c]

    def references(countries, ids, rng, n=300):
        pack_f, pack_l = np.full(ds.n_first, -1), np.full(ds.n_last, -1)
        pack_f[ids["first"]], pack_l[ids["last"]] = np.arange(len(ids["first"])), np.arange(len(ids["last"]))

        def rows(f, l, lp, c=None):
            out = [{"first": int(pack_f[a]), "last": int(pack_l[b]), "log_p": float(v)} for a, b, v in zip(f, l, lp)]
            for r, k in zip(out, [] if c is None else c):
                r["country"] = ds.countries[k]
            return out

        j = rng.integers(*np.searchsorted(ds.c, [min(countries), max(countries) + 1]), 20 * n)
        j = j[(pack_f[ds.f[j]] >= 0) & (pack_l[ds.l[j]] >= 0)][:n]
        f = np.concatenate([ds.f[j], rng.choice(ids["first"], n)])
        l = np.concatenate([ds.l[j], rng.choice(ids["last"], n)])
        c = np.concatenate([ds.c[j], rng.choice(countries, n)])
        out = {"country": rows(f, l, log_p(f, l, c), c)}
        if len(countries) == ds.C:
            fw, lw = f[: n // 3, None], l[: n // 3, None]
            lp = np.logaddexp.reduce(np.log(pop / pop.sum()) + log_p(fw, lw, np.arange(ds.C)[None]), axis=1)
            out["world"] = rows(fw[:, 0], lw[:, 0], lp)
        return out

    return references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_fr_insee", help="data/: France with the base of name_dataset")
    parser.add_argument("--world-model", default=WORLD_MODEL)
    parser.add_argument("--out-dir", default="app/data")
    args = parser.parse_args()
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    letters = letter_rules()
    rng = np.random.default_rng(0)
    ds = namedata.load(args.data_dir, drop=False)  # all the people: the real full names do not change
    data_filter = json.load(open(f"{args.data_dir}/filter.json"))  # min_first, min_last of prepare_world.py
    alphabet = letter_order([line.split("\t", 1)[0] for side in ("first", "last")
                             for line in open(f"{args.data_dir}/vocab_{side}.tsv", encoding="utf-8")])
    ds.n_train = ds.n_train + ds.n_valid + ds.n_test  # all people (pm.support and log_marginals read n_train)
    d = np.load(args.world_model)
    names = world_names(ds, d, args.data_dir)
    model = pm.load_counts(args.world_model, ds, False)
    cache = f"{os.path.splitext(args.world_model)[0]}_logz_all_{os.path.basename(os.path.normpath(args.data_dir))}.npy"
    if not os.path.exists(cache):
        np.save(cache, np.array(pm.log_partition_restricted(model, ds), np.float64))
    log_z = np.load(cache)
    pop = population(ds.countries)
    references = world_references(model, ds, log_z, pop)
    people = np.bincount(ds.c, weights=ds.n_train, minlength=ds.C).astype(np.int64)
    print(f"world model and log Z ready ({time.time() - t0:.0f}s)", flush=True)

    cs = list(range(ds.C))
    sections, meta, ids = {}, {}, {}
    for side in ("first", "last"):
        s, meta[side], ids[side] = names[side].pack(side, cs, alphabet, f"{args.out_dir}/world.names")
        sections.update(s)
        n_all = len(names[side].total)
        scale = np.ones(ds.C)  # people of the base per person of the data: count * scale + alpha pi
        for c, counts in ds.base.items():
            scale[c] = people[c] / counts[("first", "last").index(side)].sum()
        meta[side].update(people=people.tolist(), prior_norm=float(names[side].total.sum() + 0.5 * n_all),
                          scale=scale.tolist())
    key, count, country = pack_pairs(ds.f, ds.l, ds.n_train, ids, ds.c)
    write_parts(f"{args.out_dir}/world.real", key, count, country, names, ids)
    entries = key, count, country
    key, start = np.unique(key, return_index=True)
    count = np.add.reduceat(count, start)  # people of each full name in all countries
    check_real = check_rows(key, count, len(ids["first"]), len(ids["last"]), rng)
    for r in check_real:  # and their people per country
        k = r["first"] * len(ids["last"]) + r["last"]
        i, j = np.searchsorted(entries[0], [k, k + 1])
        r["countries"] = [[ds.countries[c], int(x)] for c, x in zip(entries[2][i:j], entries[1][i:j])]
    write_pack(f"{args.out_dir}/world.anag.gz", dict(
        version=VERSION, source="name_dataset", model=os.path.basename(args.world_model), countries=ds.countries,
        bases={ds.countries[c]: "INSEE" for c in ds.base},
        people=people.tolist(), population=pop.tolist(), alpha=float(d["alpha"]), log_z=log_z.tolist(),
        gates=d["G"].astype(np.float64).tolist(), offsets=d["B"].astype(np.float64).tolist(), letters=letters,
        alphabet=alphabet, pairs=len(key), parts=PARTS, part_hash=PART_HASH.tolist(), **data_filter, **meta), sections)
    json.dump(dict(references(cs, ids, rng), real=check_real), open(f"{args.out_dir}/world.check.json", "w"))
    print(f"world: {meta['first']['names']:,} first, {meta['last']['names']:,} last names, {len(key):,} real full"
          f" names, {os.path.getsize(f'{args.out_dir}/world.anag.gz') / 1e6:.1f} MB ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
