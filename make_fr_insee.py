# A data folder for a world model where France uses INSEE: the same files as data/ (links), and insee_fr.npz, the INSEE
# people of each name of the world vocabulary (0: not in INSEE). namedata.load then keeps, in France, only the people
# whose first and last names are both in INSEE (export_app.py keeps them all: drop=False), log_marginals takes INSEE
# as the base of France (scaled to the people of France in the data: the smoothing has the same weight as in the other
# countries), and the pair term of France applies to the names of INSEE (pairmodel.support).
# Run: .venv/bin/python make_fr_insee.py [--data-dir data] [--out data_fr_insee], then (the model of the site):
#   M=models/fr_insee/world_all_k16_wd0.1_u0_me20_s10000_gates_pool4096
#   caffeinate -i .venv/bin/python pairmodel.py world --data-dir data_fr_insee --alpha 100000 --k 16 --wd 0.1 \
#     --min-embed 20 --steps 10000 --gates --pool 4096 --restrict --save-dir models/fr_insee   (8 min)
#   caffeinate -i .venv/bin/python calibrate.py $M.npz --data-dir data_fr_insee   (90 min)
#   caffeinate -i .venv/bin/python quantize.py ${M}_cal.npz --data-dir data_fr_insee --save pq2
#   caffeinate -i .venv/bin/python export_app.py --packs world   (data_fr_insee and this model by default)
import argparse
import os
from collections import Counter

import numpy as np

from export_app import key_of

INSEE_FIRST, INSEE_LAST = "nat2022.csv", "noms2008nat_txt.txt"


def insee_counts(path, last):
    """{key: (display form, people)} from an INSEE file. The display form is the most frequent spelling."""
    forms = {}
    lines = open(path, encoding="utf-8").read().splitlines()[1:]
    for line in lines:
        if last:
            name, *n = line.rsplit(None, 11)
        else:
            _, name, _, n = line.split(";")
            n = [n]
        if name.startswith("_") or name == "AUTRES NOMS":  # rare names grouped together
            continue
        k = key_of(name)
        if len(k) >= 2:
            forms.setdefault(k, Counter())[name] += sum(map(int, n))
    return {k: (c.most_common(1)[0][0].title(), sum(c.values())) for k, c in forms.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data_fr_insee")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in os.listdir(args.data_dir):  # the same data: links
        link = f"{args.out}/{name}"
        if not os.path.exists(link) and name != "insee_fr.npz":
            os.symlink(os.path.abspath(f"{args.data_dir}/{name}"), link)
    arrays = {}
    for side, path in (("first", INSEE_FIRST), ("last", INSEE_LAST)):
        counts = insee_counts(path, side == "last")
        keys = [line.split("\t", 1)[0] for line in open(f"{args.data_dir}/vocab_{side}.tsv", encoding="utf-8")]
        arrays[side] = np.array([counts[k][1] if k in counts else 0 for k in keys], np.int64)
        found = arrays[side] > 0
        print(
            f"{side}: {len(counts):,} INSEE names, {found.sum():,} of them in the world vocabulary "
            f"({arrays[side].sum() / sum(n for _, n in counts.values()):.1%} of the INSEE people)"
        )
    np.savez(f"{args.out}/insee_fr.npz", **arrays)
    print(f"{args.out}/insee_fr.npz")


if __name__ == "__main__":
    main()
