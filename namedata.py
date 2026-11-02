# Load data/pairs.npz (see prepare_world.py) and split people into train/valid/test.
# NumPy only, so the MLX models and the JAX baseline (training.py) use the same split.
# A data folder with insee_fr.npz (make_fr_insee.py): France keeps only the people whose first and last names are in
# INSEE, and INSEE is its base (log_marginals). The split is made before, so the other countries do not change.
import os
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class Dataset:
    c: np.ndarray  # (T,) country of each unique triple, sorted
    f: np.ndarray  # (T,) first name id
    l: np.ndarray  # (T,) last name id
    n_train: np.ndarray  # (T,) people counts per split
    n_valid: np.ndarray
    n_test: np.ndarray
    n_first: int
    n_last: int
    countries: list
    base: dict = field(default_factory=dict)  # country index -> (people per first name, per last name): its base
    dropped: dict = field(default_factory=dict)  # the triples taken out of a country (f, l, n_train, n_valid, n_test)

    @property
    def C(self):
        return len(self.countries)


def load_dataset(path="data/pairs.npz", countries_path="data/countries.txt", seed=0):
    """Load unique triples and split people 80/10/10 into train/valid/test."""
    d = np.load(path)
    n = d["count"].astype(np.int64)
    rng = np.random.default_rng(seed)
    n_test = rng.binomial(n, 0.1)
    n_valid = rng.binomial(n - n_test, 0.1 / 0.9)
    n_train = n - n_test - n_valid
    with open(countries_path) as fh:
        countries = fh.read().split()
    c = d["country"].astype(np.int32)
    assert np.all(np.diff(c) >= 0), "triples must be sorted by country"
    return Dataset(
        c=c,
        f=d["first"].astype(np.int32),
        l=d["last"].astype(np.int32),
        n_train=n_train.astype(np.int32),
        n_valid=n_valid.astype(np.int32),
        n_test=n_test.astype(np.int32),
        n_first=int(d["first"].max()) + 1,
        n_last=int(d["last"].max()) + 1,
        countries=countries,
    )


def subset_country(ds, code):
    """Keep one country, and only names seen in its training split (as training.py did)."""
    keep = ds.c == ds.countries.index(code)
    f, l, n_train = ds.f[keep], ds.l[keep], ds.n_train[keep]
    f_seen = np.bincount(f, weights=n_train, minlength=ds.n_first) > 0
    l_seen = np.bincount(l, weights=n_train, minlength=ds.n_last) > 0
    f_map = np.cumsum(f_seen) - 1
    l_map = np.cumsum(l_seen) - 1
    ok = f_seen[f] & l_seen[l]
    return Dataset(
        c=np.zeros(ok.sum(), dtype=np.int32),
        f=f_map[f[ok]].astype(np.int32),
        l=l_map[l[ok]].astype(np.int32),
        n_train=n_train[ok],
        n_valid=ds.n_valid[keep][ok],
        n_test=ds.n_test[keep][ok],
        n_first=int(f_seen.sum()),
        n_last=int(l_seen.sum()),
        countries=[code],
    )


def with_insee(ds, path, drop=True):
    """France: INSEE as its base, and (drop) only the people whose first and last names are in INSEE. The export
    keeps all the people (drop=False): the real full names and the people of each name do not change."""
    d = np.load(path)
    first, last = d["first"], d["last"]
    c = ds.countries.index("FR")
    if not drop:
        return replace(ds, base={c: (first, last)})
    keep = (ds.c != c) | ((first[ds.f] > 0) & (last[ds.l] > 0))
    dropped = {k: getattr(ds, k)[~keep] for k in ("f", "l", "n_train", "n_valid", "n_test")}
    return replace(ds, c=ds.c[keep], f=ds.f[keep], l=ds.l[keep], n_train=ds.n_train[keep], n_valid=ds.n_valid[keep],
                   n_test=ds.n_test[keep], base={c: (first, last)}, dropped=dropped)


def people(ds):
    """People of each first name and of each last name in the world (weights n_train), the dropped triples too: the
    world marginals do not change when a country keeps only some of its people."""
    out = []
    for side, n in (("f", ds.n_first), ("l", ds.n_last)):
        x = np.bincount(getattr(ds, side), weights=ds.n_train, minlength=n)
        if ds.dropped:
            x += np.bincount(ds.dropped[side], weights=ds.dropped["n_train"], minlength=n)
        out.append(x)
    return out


def load(data_dir="data", country=None, drop=True):
    ds = load_dataset(f"{data_dir}/pairs.npz", f"{data_dir}/countries.txt")
    if os.path.exists(f"{data_dir}/insee_fr.npz"):
        ds = with_insee(ds, f"{data_dir}/insee_fr.npz", drop)
    return subset_country(ds, country) if country else ds


def describe(ds):
    return (f"{ds.C} countries, {ds.n_first:,} first, {ds.n_last:,} last names, {len(ds.f):,} triples, "
            f"people train/valid/test {int(ds.n_train.sum()):,}/{int(ds.n_valid.sum()):,}/{int(ds.n_test.sum()):,}")


def log_marginals(ds, alpha):
    """log p(f|c), log p(l|c) smoothed toward the global marginal with alpha pseudo-people. A country with a base
    (ds.base: INSEE for France) takes its people per name from it, scaled to its people in the data."""
    out = []
    world = people(ds)
    for side, (ids, n_names) in enumerate([(ds.f, ds.n_first), (ds.l, ds.n_last)]):
        N = np.bincount(ids.astype(np.int64) * ds.C + ds.c, weights=ds.n_train, minlength=n_names * ds.C)
        N = N.reshape(n_names, ds.C)
        g = world[side]
        g = (g + 0.5) / (g.sum() + 0.5 * n_names)
        for c, counts in ds.base.items():
            N[:, c] = counts[side] * (N[:, c].sum() / counts[side].sum())
        P = (N + alpha * g[:, None]) / (N.sum(0, keepdims=True) + alpha)
        out.append(np.log(P).astype(np.float32))
    return out
