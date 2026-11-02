# Anagram search for full names, ranked by a pair model. Python version (NumPy only); app/anagram.ts is the
# same algorithm in TypeScript. Data: the packs of app/data/, written by export_app.py.
#
#   python app/anagram.py "Elena Ourgaud"                        world: world mixture
#   python app/anagram.py "Elena Ourgaud" --country FR           world, France: the names seen in France
#   python app/anagram.py --check                                compare with the values of pairmodel.py
# Without a query, the queries are read from stdin, one per line.
#
# Search. The key of a name is its letters A-Z (accents removed, upper case); its signature is the key with
# the letters in the order of the pack alphabet, the rarest letters first. A pack stores the names in the
# preorder of a trie of their signatures. The search walks the trie of first names depth first and enters a
# node only if the query still has its letter, so it visits only signatures made of query letters. For each
# first-name signature, the remaining letters, in the order of the alphabet, give one path in the trie of last
# names: the last names at its end complete the query. The world pack keeps the data of its names (display
# forms, people per country, vectors) in chunk files of 1024 names; this version reads all of them.
#
# Rank. With N the people counts of the pack data,
#   log p(f, l | c) = X[f, c] + Y[l, c] + [N[f, c] > 0 and N[l, c] > 0] (u_f . (g_c * v_l) + b_c) - log Z_c
#   X[f, c] = log(N[f, c] + alpha pi_f) - log(N_c + alpha),   pi_f = (N_f + 1/2) / (sum of N_f + n_first / 2)
# For France in the world pack, N[f, FR] is the people of f in INSEE, scaled to the people of France in the data.
# and Y is the same for last names. In a pack of several countries without a country, the rank uses the
# mixture p(f, l) = sum_c (P_c / P) p(f, l | c), with P_c the population of c (not N_c: name_dataset does not
# have the same share of each population), and each result shows the country with the largest term.
#
# Real full names: a pack lists the full names of its names that people of name_dataset have; each result gives
# its people (0: none), and the results show them first (most people first), before the rank. The world pack
# keeps them, with their people per country, in the files world.real/<part>.gz, and a search reads the part of
# the letters of its query (export_app.py gives the format). Each result also gives the people with its first
# name, with its last name and with the full name, in the world and per country of the pack. The origin of the
# query is p(c | f, l), proportional to (P_c / P) p(f, l | c).
import argparse
import gzip
import json
import math
import os
import struct
import sys
import time
import unicodedata

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CHUNK = 1 << 15  # pairs scored at a time with the mixture


class KeyTable(dict):
    """str.translate table from a character to its key letters: ASCII letters in upper case, the pack rules
    (metadata "letters") for the other characters. Other characters are removed."""

    def __init__(self, letters):
        super().__init__({cp: chr(cp).upper() for cp in range(128) if chr(cp).isalpha()})
        self.update({int(cp): key for cp, key in letters.items()})

    def __missing__(self, cp):
        return None


def varints(b):
    """Values of concatenated LEB128 varints (uint8 array)."""
    if len(b) == 0:
        return np.zeros(0, np.int64)
    last = b < 0x80
    owner = np.concatenate([[0], np.cumsum(last)[:-1]])
    start = np.flatnonzero(np.concatenate([[True], last[:-1]]))
    return np.add.reduceat((b & 0x7F).astype(np.int64) << (7 * (np.arange(len(b)) - start[owner])), start)


def ranges(start, length):
    """The integer ranges [start_i, start_i + length_i), concatenated."""
    return np.arange(length.sum()) - np.repeat(np.cumsum(length) - length, length) + np.repeat(start, length)


def by_people(countries, country, count):
    """[country code, people] pairs, most people first, then by code."""
    return sorted(([countries[c], int(n)] for c, n in zip(country, count)), key=lambda x: (-x[1], x[0]))


CHUNK_SECTIONS = ("names", "entries", "country", "count", "vec", "codes")


def read_chunks(chunk_dir, side, m):
    """The sections CHUNK_SECTIONS of a side of the world pack, read from its chunk files."""
    out = [[] for _ in CHUNK_SECTIONS]
    for c in range(-(-m["names"] // m["chunk"])):
        with open(f"{chunk_dir}/{side}/{c:04x}.gz", "rb") as fh:
            raw = np.frombuffer(gzip.decompress(fh.read()), np.uint8)
        head = int(np.flatnonzero(raw < 0x80)[len(CHUNK_SECTIONS) - 1]) + 1  # the 6 varints of the lengths
        at = head + np.concatenate([[0], np.cumsum(varints(raw[:head]))])
        for k in range(len(CHUNK_SECTIONS)):
            out[k].append(raw[at[k] : at[k + 1]])
    return {f"{side}.{name}": np.concatenate(x) for name, x in zip(CHUNK_SECTIONS, out)}


def find(keys, values, k):
    """values[i] where keys[i] == k (keys increasing), else 0."""
    if len(keys) == 0:
        return np.zeros(len(k), np.int64)
    i = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
    return np.where(keys[i] == k, values[i], 0)


def read_pack(path):
    """Metadata and sections (name -> uint8 array) of a pack file."""
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    version, n = struct.unpack("<II", raw[4:12])
    assert raw[:4] == b"ANAG" and version == 1, f"{path} is not a pack of version 1"
    meta = json.loads(raw[12 : 12 + n])
    buf = np.frombuffer(raw, np.uint8)[12 + n + (-(12 + n) % 4) :]
    return meta, {name: buf[a : a + size] for name, (a, size) in meta["sections"].items()}


class Side:
    """The first or the last names of a pack: signature trie, display forms, people per country, vectors."""

    def __init__(self, meta, side, sections, chunk_dir):
        m = meta[side]
        sections = {**sections, **read_chunks(chunk_dir, side, m)}
        get = lambda name: sections[f"{side}.{name}"]
        self.n = m["names"]
        letters, end, ends = get("letters"), np.arange(m["nodes"]) + varints(get("sizes")), varints(get("ends"))
        self.letters = letters.tolist()
        self.end = end.tolist()  # first node after the subtree
        self.name_start = np.concatenate([[0], np.cumsum(ends)]).tolist()  # names of node i: [start[i], start[i + 1])
        self.names = get("names").tobytes().decode("utf-8").split("\n")[: self.n]

        self.ptr = np.concatenate([[0], np.cumsum(varints(get("entries")))])
        self.country = get("country")
        self.count = varints(get("count")).astype(np.float64)
        self.total = varints(get("total"))
        self.alpha_prior = meta["alpha"] * (self.total + 0.5) / m["prior_norm"]  # alpha pi
        # part of each name: the sum of part_hash over the letters of its node and of the nodes above it
        w = np.array(meta["part_hash"], np.int64)[[ord(ch) - 65 for ch in meta["alphabet"]]][letters]
        d = np.zeros(m["nodes"] + 1, np.int64)
        d[:-1] += w
        np.subtract.at(d, end, w)
        self.part = np.repeat(np.cumsum(d)[:-1], ends) % meta["parts"]
        self.by_rank = np.lexsort((np.arange(self.n), -self.total))  # most people first
        self.log_norm = np.log(np.array(m["people"], np.float64) + meta["alpha"])  # log(N_c + alpha)
        self.scale = np.array(m.get("scale", [1.0] * len(m["people"])), np.float64)  # people of the model per count

        has = np.unpackbits(get("vec"), count=self.n).astype(bool)
        self.row = np.full(self.n, -1)
        self.row[has] = np.arange(has.sum())
        self.codes = get("codes").reshape(-1, m["groups"])
        self.books = get("books").view("<f4").reshape(m["groups"], 256, m["group_size"]).astype(np.float64)

    def find(self, rest):
        """The trie node of the signature with the letter counts rest, or -1."""
        node, i, end = -1, 0, len(self.letters)  # the children of node are the nodes i, self.end[i], ... before end
        for x in range(26):
            for _ in range(rest[x]):
                while i < end and self.letters[i] < x:
                    i = self.end[i]
                if i == end or self.letters[i] != x:
                    return -1
                node, i, end = i, i + 1, self.end[i]
        return node

    def seen(self, ids, c):
        """Whether each name is seen in pack country c."""
        if not hasattr(self, "entry_key"):  # id * 256 + country of each (name, country), increasing
            self.entry_key = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.ptr)) * 256 + self.country
        k = np.asarray(ids, np.int64) * 256 + c
        i = np.minimum(np.searchsorted(self.entry_key, k), len(self.entry_key) - 1)
        return self.entry_key[i] == k

    def log_weights(self, ids):
        """X[ids, c] for all pack countries c, and whether each name has people in each country (count > 0: the
        pair term applies; a country with a base lists the names seen in the data too, with count 0)."""
        lens = self.ptr[ids + 1] - self.ptr[ids]
        rows = np.repeat(np.arange(len(ids)), lens)
        cells = ranges(self.ptr[ids], lens)
        counts = np.zeros((len(ids), len(self.log_norm)))
        counts[rows, self.country[cells]] = self.count[cells]
        with np.errstate(divide="ignore"):  # alpha = 0: a name has no weight where it is not seen
            return np.log(counts * self.scale + self.alpha_prior[ids, None]) - self.log_norm, counts > 0

    def vectors(self, ids):
        """Name vectors (len(ids), k); 0 for names without a vector (the null score)."""
        out = np.zeros((len(ids), self.books.shape[0] * self.books.shape[2]))
        r = self.row[ids]
        has = r >= 0
        out[has] = self.books[np.arange(len(self.books)), self.codes[r[has]]].reshape(has.sum(), out.shape[1])
        return out


class Model:
    def __init__(self, path):
        meta, sections = read_pack(path)
        self.source, self.countries = meta["source"], meta["countries"]
        pop = np.array(meta["population"], np.float64)
        self.log_w = np.log(pop / pop.sum())
        self.log_z = np.array(meta["log_z"])
        self.gates, self.offsets = np.array(meta["gates"]), np.array(meta["offsets"])
        self.table = KeyTable(meta["letters"])
        self.rank = [0] * 26  # index of each letter A-Z in the pack alphabet
        for i, ch in enumerate(meta["alphabet"]):
            self.rank[ord(ch) - 65] = i
        base = path.removesuffix(".gz").removesuffix(".anag")
        self.first, self.last = (Side(meta, side, sections, f"{base}.names") for side in ("first", "last"))
        self.parts = meta["parts"]
        self.part_dir = f"{base}.real"
        L = self.last
        self.by_class = np.lexsort((np.arange(L.n), -L.total, L.part))  # last names by class, then by rank
        self.class_start = np.searchsorted(L.part[self.by_class], np.arange(self.parts))
        self.part_cache = {}

    def log_p(self, f, l, c=None):
        """log p(f_i, l_i | c) for arrays of name ids: (m, len(c)), all pack countries when c is None."""
        c = np.arange(len(self.countries)) if c is None else np.atleast_1d(c)
        uf, fi = np.unique(f, return_inverse=True)
        ul, li = np.unique(l, return_inverse=True)
        (X, sx), (Y, sy) = self.first.log_weights(uf), self.last.log_weights(ul)
        inter = (self.first.vectors(uf)[fi] * self.last.vectors(ul)[li]) @ self.gates[c].T + self.offsets[c]
        seen = sx[fi][:, c] & sy[li][:, c]
        return X[fi][:, c] + Y[li][:, c] + seen * inter - self.log_z[c]

    def part(self, j):
        """The real full names of part j: keys (first id * n_last + last id, increasing), people, and people per
        country (ptr: entries of full name i in [ptr[i], ptr[i + 1]), country, count)."""
        if j not in self.part_cache:
            with open(f"{self.part_dir}/{j:03x}.gz", "rb") as fh:
                v = varints(np.frombuffer(gzip.decompress(fh.read()), np.uint8))
            m = int(v[0])
            a, b = v[1 : 1 + 3 * m].reshape(-1, 3), v[1 + 3 * m :].reshape(-1, 2)
            new_row = a[:, 0] > 0
            new_row[:1] = True
            t = a[:, 1] + ~new_row  # index of the last name, from the index before it + 1 in the same row
            cs = np.cumsum(t)
            col = cs - (cs - t)[new_row][np.cumsum(new_row) - 1]
            f = self.first.by_rank[np.cumsum(a[:, 0])]
            l = self.by_class[self.class_start[(j - self.first.part[f]) % self.parts] + col]
            size = a[:, 2] + 1
            start = np.cumsum(size) - size
            t = b[:, 0] + 1  # country, from the country before it + 1 in the same full name
            cs = np.cumsum(t)
            country = cs - (cs - t)[start][np.repeat(np.arange(m), size)] - 1
            count = b[:, 1] + 1
            key = f * self.last.n + l
            order = np.argsort(key)
            cells = ranges(start[order], size[order])
            self.part_cache[j] = {"key": key[order], "people": np.add.reduceat(count, start)[order],
                                  "ptr": np.concatenate([[0], np.cumsum(size[order])]),
                                  "country": country[cells], "count": count[cells],
                                  "entry_key": np.repeat(key[order], size[order]) * 256 + country[cells]}
        return self.part_cache[j]

    def real(self, f, l, c=None):
        """People with the full names (f_i, l_i) in the pack countries, or in pack country c (0: none)."""
        f, l = np.asarray(f), np.asarray(l)
        j = (self.first.part[f] + self.last.part[l]) % self.parts
        out = np.zeros(len(f), np.int64)
        for p in np.unique(j):
            sel = j == p
            part, k = self.part(int(p)), f[sel] * self.last.n + l[sel]
            out[sel] = find(part["key"], part["people"], k) if c is None else find(part["entry_key"], part["count"], k * 256 + c)
        return out

    def real_countries(self, f, l, c=None):
        """[country, people] of the full name (f, l) in the pack countries (or in pack country c), most people first."""
        if c is not None:
            n = int(self.real([f], [l], c)[0])
            return [[self.countries[c], n]] if n else []
        part, k = self.part(int((self.first.part[f] + self.last.part[l]) % self.parts)), f * self.last.n + l
        i = np.searchsorted(part["key"], k)
        if i == len(part["key"]) or part["key"][i] != k:
            return []
        a, b = part["ptr"][i], part["ptr"][i + 1]
        return by_people(self.countries, part["country"][a:b], part["count"][a:b])

    def name_countries(self, side, i, c=None):
        """[country, people] of name i of side ("first" or "last") in the pack countries (or in pack country c),
        most people first."""
        S = getattr(self, side)
        country, count = S.country[S.ptr[i] : S.ptr[i + 1]], S.count[S.ptr[i] : S.ptr[i + 1]]
        if c is not None:
            country, count = country[country == c], count[country == c]
        return by_people(self.countries, country, count)

    def lookup(self, side, key):
        """Id of the name with this key, or -1."""
        S = getattr(self, side)
        node = S.find(self.counts(key))
        if node < 0:
            return -1
        return next((i for i in range(S.name_start[node], S.name_start[node + 1]) if self.key(S.names[i]) == key), -1)

    def origin(self, query, top=5):
        """The most probable countries of the query, p(c | f, l), for the split of its words into a first and a
        last name of the pack with the largest probability; None if no split gives two names of the pack."""
        words, best = query.split(), None
        for i in range(1, len(words)):
            f, l = self.lookup("first", self.key(" ".join(words[:i]))), self.lookup("last", self.key(" ".join(words[i:])))
            if f >= 0 and l >= 0:
                t = self.log_p(np.array([f]), np.array([l]))[0] + self.log_w
                if best is None or np.logaddexp.reduce(t) > np.logaddexp.reduce(best):
                    best = t
        if best is None:
            return None
        p = np.exp(best - np.logaddexp.reduce(best))
        return [{"country": self.countries[c], "p": float(p[c])} for c in np.argsort(-p, kind="stable")[:top]]

    def log_p_mixture(self, f, l):
        """log p(f_i, l_i) under the mixture of the pack countries, and the country with the largest term."""
        lp, best = np.empty(len(f)), np.empty(len(f), np.int64)
        for i in range(0, len(f), CHUNK):
            t = self.log_p(f[i : i + CHUNK], l[i : i + CHUNK]) + self.log_w
            best[i : i + CHUNK] = t.argmax(1)
            lp[i : i + CHUNK] = np.logaddexp.reduce(t, axis=1)
        return lp, best

    def key(self, text):
        """The key of a text: its letters A-Z, accents removed, in upper case (Élodie -> ELODIE)."""
        return unicodedata.normalize("NFC", text).translate(self.table)

    def counts(self, key):
        """The number of each letter of a key, in the order of the pack alphabet."""
        q = [0] * 26
        for ch in key:
            q[self.rank[ord(ch) - 65]] += 1
        return q

    def anagrams(self, query):
        """All (first name id, last name id) pairs whose letters are the letters of the query."""
        q = self.counts(self.key(query))
        n, F, L = sum(q), self.first, self.last
        letters, end, start = F.letters, F.end, F.name_start
        used, path, blocks, i = [0] * 26, [], [], 0  # path: (letter, end) of the nodes from the root to node i
        while i < len(letters):
            while path and i >= path[-1][1]:
                used[path.pop()[0]] -= 1
            x = letters[i]
            if used[x] == q[x] or len(path) >= n - 2:  # no letter x left, or fewer than 2 letters for the last name
                i = end[i]
                continue
            used[x] += 1
            path.append((x, end[i]))
            if start[i + 1] > start[i]:
                node = L.find([a - b for a, b in zip(q, used)])
                if node >= 0:
                    blocks.append((start[i], start[i + 1], L.name_start[node], L.name_start[node + 1]))
            i += 1
        f = [np.repeat(np.arange(a, b), d - c) for a, b, c, d in blocks]
        l = [np.tile(np.arange(c, d), b - a) for a, b, c, d in blocks]
        return (np.concatenate(f), np.concatenate(l)) if blocks else (np.zeros(0, np.int64),) * 2

    def search(self, query, country=None, limit=20, by_model=False):
        """The anagram names of the query: the real full names first (most people first), then the most probable.
        With a country: the anagrams of the names seen in it, its real people, and its synthetic names: no real
        person anywhere in the pack. by_model: all the names, the most probable first."""
        f, l = self.anagrams(query)
        c = None if country is None else self.countries.index(country)
        if c is None:
            lp, best = self.log_p_mixture(f, l)
        else:
            keep = self.first.seen(f, c) & self.last.seen(l, c)
            f, l = f[keep], l[keep]
            lp, best = self.log_p(f, l, c)[:, 0], np.full(len(f), c)
        real = self.real(f, l, c)
        if c is not None:  # a country: a synthetic name has no real person anywhere
            keep = (real > 0) | (self.real(f, l, None) == 0)
            f, l, real, lp, best = f[keep], l[keep], real[keep], lp[keep], best[keep]
        top = (np.lexsort((l, f, -lp)) if by_model else np.lexsort((l, f, -lp, -real)))[:limit]
        results = [{"first": self.first.names[f[i]], "last": self.last.names[l[i]],
                    "country": self.countries[best[i]], "log_p": float(lp[i]), "real": int(real[i]),
                    "first_people": int(self.first.total[f[i]]), "first_countries": self.name_countries("first", f[i], c),
                    "last_people": int(self.last.total[l[i]]), "last_countries": self.name_countries("last", l[i], c),
                    "real_countries": self.real_countries(f[i], l[i], c)} for i in top]
        return {"query": query, "country": country, "found": len(f), "real_names": int((real > 0).sum()),
                "real_people": int(real.sum()), "origin": self.origin(query), "results": results}


def one_in(log_p):
    n = math.exp(-log_p)
    for unit, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if n >= unit:
            return f"1 in {n / unit:.1f} {word}"
    return f"1 in {round(n):,}"


def show(res, source):
    print(f"{res['query']} ({res['country'] or 'world'}, {source}): {res['found']:,} names, "
          f"{res['real_names']:,} real with {res['real_people']:,} people")
    if res["origin"] and not res["country"]:
        print("origin: " + ", ".join(f"{o['country']} {o['p']:.0%}" for o in res["origin"]))
    width = max((len(r["first"]) + len(r["last"]) for r in res["results"]), default=0) + 1
    for i, r in enumerate(res["results"], 1):
        name = f"{r['first']} {r['last']}"
        real = f"  real: {r['real']:,} {'person' if r['real'] == 1 else 'people'}" if r["real"] else ""
        print(f"{i:>4}. {name:<{width}}  {r['country']}  {one_in(r['log_p'])}{real}")


def check(model, path):
    """Compare log p with the values that export_app.py computed with pairmodel.py."""
    ref = json.load(open(path.replace(".anag.gz", ".check.json")))
    rows = ref["country"]
    f, l = (np.array([r[k] for r in rows]) for k in ("first", "last"))
    c = np.array([model.countries.index(r["country"]) for r in rows])
    errors = {"country": np.abs(model.log_p(f, l)[np.arange(len(rows)), c] - [r["log_p"] for r in rows]).max()}
    if "world" in ref:
        rows = ref["world"]
        f, l = (np.array([r[k] for r in rows]) for k in ("first", "last"))
        errors["world"] = np.abs(model.log_p_mixture(f, l)[0] - [r["log_p"] for r in rows]).max()
    print("largest difference with pairmodel.py, in nats: " + ", ".join(f"{v:.2e} ({k})" for k, v in errors.items()))
    rows = ref["real"]
    by_code = lambda pairs: sorted(map(tuple, pairs))
    wrong = sum(model.real([r["first"]], [r["last"]])[0] != r["real"]
                or "countries" in r and by_code(model.real_countries(r["first"], r["last"])) != by_code(r["countries"])
                for r in rows)
    print(f"real full names: {wrong} wrong of {len(rows)} ({sum(r['real'] > 0 for r in rows)} listed)")
    return max(errors.values()) < 1e-4 and wrong == 0


def pack_path(args):
    """The pack file of the options: --pack, else the world pack."""
    name = args.pack or "world"
    return name if os.path.exists(name) else f"{args.data}/{name}.anag.gz"


def main():
    parser = argparse.ArgumentParser(description="Full names whose letters are the letters of the query.")
    parser.add_argument("query", nargs="*", help="one query per argument (none: read lines from stdin)")
    parser.add_argument("--country", help="country code (e.g. FR): the names seen in it; default: world mixture")
    parser.add_argument("--pack", help="pack name or file; default: world")
    parser.add_argument("-n", "--limit", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="one JSON object per query")
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--check", action="store_true", help="compare with the values of pairmodel.py")
    args = parser.parse_args()

    t = time.time()
    path = pack_path(args)
    model = Model(path)
    print(f"loaded {os.path.basename(path)} in {time.time() - t:.1f}s", file=sys.stderr)
    if args.check:
        sys.exit(0 if check(model, path) else 1)
    for query in args.query or sys.stdin:
        if not query.strip():
            continue
        res = model.search(query.strip(), args.country, args.limit)
        if args.json:
            print(json.dumps(res, ensure_ascii=False))
        else:
            show(res, model.source)


if __name__ == "__main__":
    main()
