# Prepare world name data (name_dataset, 105 countries) for anagram search and pair models.
#
# A person is kept only if both names pass these rules (after NFC normalization):
#   - all characters are Latin: \u0000-ɏ, Ḁ-ỿ, ̀-ͯ
#   - only letters, combining marks, space, hyphen, apostrophe and period
#   - the match key has at least 2 letters
#   - the key occurs at least --min-first (first names) or --min-last (last names) times in the world, among
#     the people kept by the character rules. This is the only count filter of the data, the model and the apps:
#     it removes rare names, often fake ones. A kept name can have fewer people at the end, when some of its
#     people have a removed other name.
#
# Each name has a match key (A-Z only, accents removed, used for anagrams and as the
# vocabulary entry) and a display form (its most frequent spelling in the data).
#
# Output in data/:
#   vocab_first.tsv, vocab_last.tsv   key, display, count (sorted by count, row = id)
#   countries.txt                     country codes (row = id)
#   filter.json                       the values of --min-first and --min-last
#   pairs.npz                         unique (country, first, last) triples with counts
import argparse
import json
import os
import re
import time
import unicodedata
from array import array
from collections import Counter
from multiprocessing import Pool

import numpy as np
from unidecode import unidecode

DATA_DIR = "name_dataset/data"
OUT_DIR = "data"

LATIN = re.compile(r"^[\u0000-ɏḀ-ỿ̀-ͯ]+$")
NON_KEY = re.compile(r"[^A-Z]")
ALLOWED_PUNCT = set(" -'.")


def normalize(raw):
    """Return (display form, match key) for a raw name, or None if the name is rejected."""
    name = unicodedata.normalize("NFC", raw).replace("’", "'")
    name = " ".join(name.split())
    if not name or not LATIN.match(name):
        return None
    for ch in name:
        if ch not in ALLOWED_PUNCT and unicodedata.category(ch)[0] not in "LM":
            return None
    key = NON_KEY.sub("", unidecode(name).upper())
    if len(key) < 2:
        return None
    return name, key


class LocalVocab:
    """Vocabulary for one country file: key -> local id, with spelling counts."""

    def __init__(self):
        self.raw_to_id = {}  # raw string -> local id (-1 if rejected)
        self.raw_counts = Counter()  # raw string -> count (only accepted strings)
        self.raw_to_form = {}
        self.key_to_id = {}
        self.keys = []

    def lookup(self, raw):
        idx = self.raw_to_id.get(raw)
        if idx is None:
            res = normalize(raw)
            if res is None:
                idx = -1
            else:
                form, key = res
                idx = self.key_to_id.get(key)
                if idx is None:
                    idx = len(self.keys)
                    self.key_to_id[key] = idx
                    self.keys.append(key)
                self.raw_to_form[raw] = form
            self.raw_to_id[raw] = idx
        return idx

    def form_counts(self, counts_by_raw):
        """Return {key: Counter(form -> count)} from per-raw counts."""
        out = {}
        for raw, n in counts_by_raw.items():
            key = self.keys[self.raw_to_id[raw]]
            out.setdefault(key, Counter())[self.raw_to_form[raw]] += n
        return out


def process_country(path):
    """Read one country CSV. Return its unique (first, last) pairs with counts."""
    country = os.path.basename(path).removesuffix(".csv")
    vf, vl = LocalVocab(), LocalVocab()
    first_raw, last_raw = Counter(), Counter()
    codes = array("q")
    n_rows = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            n_rows += 1
            parts = line.split(",")
            if len(parts) != 4:
                continue
            f = vf.lookup(parts[0])
            if f < 0:
                continue
            l = vl.lookup(parts[1])
            if l < 0:
                continue
            first_raw[parts[0]] += 1
            last_raw[parts[1]] += 1
            codes.append((f << 32) | l)
    codes, counts = np.unique(np.frombuffer(codes, dtype=np.int64), return_counts=True)
    return {
        "country": country,
        "n_rows": n_rows,
        "first_keys": vf.keys,
        "last_keys": vl.keys,
        "first_forms": vf.form_counts(first_raw),
        "last_forms": vl.form_counts(last_raw),
        "f": (codes >> 32).astype(np.int32),
        "l": (codes & 0xFFFFFFFF).astype(np.int32),
        "n": counts.astype(np.uint32),
    }


def build_vocab(results, side, min_count):
    """Merge per-country vocabularies of one side ('first' or 'last').

    Returns (keys, displays, counts) sorted by count, and per-country maps
    local id -> global id (-1 if the key is removed).
    """
    total = Counter()
    forms = {}
    for r in results:
        local_counts = np.bincount(r[side[0]], weights=r["n"], minlength=len(r[f"{side}_keys"]))
        for key, n in zip(r[f"{side}_keys"], local_counts):
            total[key] += int(n)
        for key, fc in r[f"{side}_forms"].items():
            forms.setdefault(key, Counter()).update(fc)

    print(f"  {side}: {len(total):,} keys before filter")
    for t in [2, 5, 10, 50]:
        kept = [n for n in total.values() if n >= t]
        print(f"    min count {t:>2}: {len(kept):>10,} keys, {sum(kept) / sum(total.values()):.1%} of people")

    kept = sorted((k for k, n in total.items() if n >= min_count), key=lambda k: (-total[k], k))
    key_to_gid = {k: i for i, k in enumerate(kept)}
    displays = [forms[k].most_common(1)[0][0] for k in kept]
    counts = [total[k] for k in kept]
    maps = [
        np.array([key_to_gid.get(k, -1) for k in r[f"{side}_keys"]], dtype=np.int32)
        for r in results
    ]
    return kept, displays, counts, maps


def compact(ids, n, keys, displays):
    """Recount names from pairs, drop names with zero count, sort ids by count."""
    counts = np.bincount(ids, weights=n, minlength=len(keys)).astype(np.int64)
    order = np.lexsort((np.arange(len(keys)), -counts))
    order = order[counts[order] > 0]
    new_id = np.full(len(keys), -1, dtype=np.int32)
    new_id[order] = np.arange(len(order), dtype=np.int32)
    return new_id[ids], [keys[i] for i in order], [displays[i] for i in order], counts[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-first", type=int, default=5, help="people with a first name in the world")
    parser.add_argument("--min-last", type=int, default=5, help="people with a last name in the world")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    data_dir, out_dir = args.data_dir, args.out_dir

    paths = sorted(
        (os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")),
        key=os.path.getsize,
        reverse=True,
    )
    print(f"Reading {len(paths)} country files with {args.workers} workers...")
    t0 = time.time()
    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_country, paths), 1):
            results.append(r)
            print(f"  [{i:>3}/{len(paths)}] {r['country']}: {r['n_rows']:,} rows, "
                  f"{r['n'].sum():,} kept, {time.time() - t0:.0f}s", flush=True)
    results.sort(key=lambda r: r["country"])

    n_rows = sum(r["n_rows"] for r in results)
    n_latin = sum(int(r["n"].sum()) for r in results)
    print(f"Rows: {n_rows:,}, kept by character rules: {n_latin:,} ({n_latin / n_rows:.1%})")

    print("Building vocabularies...")
    first_keys, first_disp, first_counts, fmaps = build_vocab(results, "first", args.min_first)
    last_keys, last_disp, last_counts, lmaps = build_vocab(results, "last", args.min_last)

    print("Remapping pairs...")
    cs, fs, ls, ns = [], [], [], []
    for c, (r, fm, lm) in enumerate(zip(results, fmaps, lmaps)):
        f, l = fm[r["f"]], lm[r["l"]]
        keep = (f >= 0) & (l >= 0)
        cs.append(np.full(keep.sum(), c, dtype=np.uint8))
        fs.append(f[keep])
        ls.append(l[keep])
        ns.append(r["n"][keep])
    c, f, l, n = (np.concatenate(x) for x in (cs, fs, ls, ns))
    print(f"Kept {n.sum():,} people ({n.sum() / n_rows:.1%} of rows), {len(n):,} unique triples")

    # Marginals are recomputed from the kept pairs, so they match the pair data.
    # Names with no kept pair are removed, and ids are sorted again by count.
    f, first_keys, first_disp, first_counts = compact(f, n, first_keys, first_disp)
    l, last_keys, last_disp, last_counts = compact(l, n, last_keys, last_disp)
    print(f"Vocabulary: {len(first_keys):,} first names, {len(last_keys):,} last names")

    os.makedirs(out_dir, exist_ok=True)
    for side, keys, disp, counts in [
        ("first", first_keys, first_disp, first_counts),
        ("last", last_keys, last_disp, last_counts),
    ]:
        with open(f"{out_dir}/vocab_{side}.tsv", "w", encoding="utf-8") as fh:
            for k, d, cnt in zip(keys, disp, counts):
                fh.write(f"{k}\t{d}\t{cnt}\n")
    with open(f"{out_dir}/countries.txt", "w") as fh:
        fh.write("\n".join(r["country"] for r in results) + "\n")
    with open(f"{out_dir}/filter.json", "w") as fh:
        json.dump({"min_first": args.min_first, "min_last": args.min_last}, fh)
    np.savez(f"{out_dir}/pairs.npz", country=c, first=f, last=l, count=n)
    print(f"Saved to {out_dir}/ in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
