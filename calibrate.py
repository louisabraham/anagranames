# Per-country calibration of a restricted pair model on the validation split.
#
# The interaction of country c becomes beta_c * s + b_c on the pairs of its support (beta_c scales the
# gates, b_c is an offset). beta_c = 0, b_c = 0 is the null, so the fit cannot be worse than the null
# on the validation people. Per person, in nats:
#   gain(beta, b) = beta * E[s 1_in] + b * P(in) - log(P_out + e^b M(beta)),  M(beta) = sum_in p e^(beta s)
# with E and P(in) over the people of a split and p the null. One exact pass per country gives a
# histogram of s weighted by p, hence M(beta) for any beta; the best b is in closed form.
# A country with a base (namedata.with_insee: France) keeps beta 1, b 0: its people in the data are only those whose
# names are in the base, not a sample of the country (all of them are in the support, so b goes to its bound).
# Run: caffeinate -i .venv/bin/python calibrate.py <model.npz> [--data-dir data]   (about 90 min for the world)
import argparse
import os
import time

import mlx.core as mx
import numpy as np

import namedata
import pairmodel

LO, HI, NB = -60.0, 60.0, 12000  # histogram of s, bins of 0.01
BETAS = np.round(np.arange(0, 2.0001, 0.01), 2)
B_MAX = 3.0


@mx.compile
def _chunk_hist(Uc, Vc, wx, wy):
    s = Uc @ Vc.T
    w = wx * wy.T
    b = mx.clip(((s - LO) * (NB / (HI - LO))).astype(mx.int32), 0, NB - 1)
    return mx.zeros((NB,)).at[b].add(w), mx.zeros((NB,)).at[b].add(w * s), mx.min(s), mx.max(s)


def null_histogram(model, ds, c, in_f, in_l, X, Y):
    """Mass of the null on the support of c, by bins of s: (mass, mass * s), and P_out."""
    fi, li = np.nonzero(in_f[:, c])[0], np.nonzero(in_l[:, c])[0]
    px, py = np.exp(X[:, c].astype(np.float64)), np.exp(Y[:, c].astype(np.float64))
    p_in = px[fi].sum() * py[li].sum()
    sx, sy = px[fi].max(), py[li].max()
    Uc = model.U[mx.array(fi)] * model.G[c]
    Vc = model.V[mx.array(li)]
    wy = mx.array((py[li] / sy).astype(np.float32))[:, None]
    h0, h1 = np.zeros(NB), np.zeros(NB)
    lo, hi = np.inf, -np.inf
    chunk = max(8, 2**26 // len(li))
    for i in range(0, len(fi), chunk):
        a, b, smin, smax = _chunk_hist(
            Uc[i : i + chunk], Vc, mx.array((px[fi[i : i + chunk]] / sx).astype(np.float32))[:, None], wy
        )
        h0 += np.array(a, dtype=np.float64)
        h1 += np.array(b, dtype=np.float64)
        lo, hi = min(lo, smin.item()), max(hi, smax.item())
    assert LO < lo and hi < HI, (lo, hi)
    scale = p_in / h0.sum()  # float32 sums inside a chunk: rescale to the exact total mass
    return h0 * scale, h1 * scale, max(1 - p_in, 0.0)


def split_stats(model, ds, c, in_f, in_l, split):
    """E[s 1_in] and P(in) over the people of a split in country c."""
    sel = np.arange(*np.searchsorted(ds.c, [c, c + 1]))
    n = getattr(ds, f"n_{split}")[sel].astype(np.float64)
    f, l = ds.f[sel], ds.l[sel]
    j = np.nonzero(in_f[f, c] & in_l[l, c] & (n > 0))[0]
    s = np.array(mx.sum(model.U[mx.array(f[j])] * model.G[c] * model.V[mx.array(l[j])], axis=1), dtype=np.float64)
    return (s * n[j]).sum() / n.sum(), n[j].sum() / n.sum()


def gains(h0, h1, p_out, es, p_in, b=None):
    """gain(beta, b) in nats for all BETAS, with the best b for these stats if b is None."""
    m = h0 > 0
    sbar = h1[m] / h0[m]
    M = np.array([np.sum(h0[m] * np.exp(beta * sbar)) for beta in BETAS])
    if b is None:
        if p_out == 0:
            b = np.zeros_like(M)  # b cancels in Z: no information
        else:
            with np.errstate(divide="ignore"):
                b = np.log(p_in * p_out / max(1 - p_in, 1e-12) / M)
            b = np.clip(b, -B_MAX, B_MAX)
    return BETAS * es + b * p_in - np.log(p_out + np.exp(b) * M), b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--no-offset", action="store_true", help="fit beta_c only (b_c = 0)")
    args = parser.parse_args()

    t0 = time.time()
    ds = namedata.load(args.data_dir)
    model = pairmodel.load_counts(args.model, ds, True)
    if model.G is None:
        model.G = mx.ones((ds.C, model.U.shape[1]))
    X, Y = np.array(model.X), np.array(model.Y)
    in_f, in_l = (m.reshape(-1, ds.C) for m in pairmodel.support(ds))
    beta, offset = np.ones(ds.C), np.zeros(ds.C)
    raw, cal, people = np.zeros(ds.C), np.zeros(ds.C), np.zeros(ds.C)
    i1 = int(np.nonzero(BETAS == 1)[0][0])
    print(f"{'cc':<3} {'beta':>5} {'b':>6}  valid gain raw -> fitted   test gain raw -> fitted (bits)", flush=True)
    for c in range(ds.C):
        if c in ds.base:
            print(f"{ds.countries[c]:<3} {1:5.2f} {0:+6.2f}  (a base: not calibrated)", flush=True)
            continue
        h0, h1, p_out = null_histogram(model, ds, c, in_f, in_l, X, Y)
        es_v, pin_v = split_stats(model, ds, c, in_f, in_l, "valid")
        es_t, pin_t = split_stats(model, ds, c, in_f, in_l, "test")
        zero = np.zeros(len(BETAS))
        gv, bv = gains(h0, h1, p_out, es_v, pin_v, zero if args.no_offset else None)
        k = int(np.argmax(gv))
        beta[c], offset[c] = BETAS[k], bv[k]
        gt_raw = gains(h0, h1, p_out, es_t, pin_t, zero)[0][i1]
        gt_fit = gains(h0, h1, p_out, es_t, pin_t, bv)[0][k]
        raw[c], cal[c] = gt_raw, gt_fit
        people[c] = ds.n_test[slice(*np.searchsorted(ds.c, [c, c + 1]))].sum()
        print(
            f"{ds.countries[c]:<3} {beta[c]:5.2f} {offset[c]:+6.2f}  {gv[i1] / np.log(2):+.3f} -> {gv[k] / np.log(2):+.3f}"
            f"   {gt_raw / np.log(2):+.3f} -> {gt_fit / np.log(2):+.3f}",
            flush=True,
        )
    w = people / people.sum()
    print(
        f"test gain (bits/person): raw {np.sum(w * raw) / np.log(2):+.4f}, calibrated {np.sum(w * cal) / np.log(2):+.4f}; "
        f"countries below the null: raw {int((raw < 0).sum())}, calibrated {int((cal < 0).sum())} "
        f"({time.time() - t0:.0f}s)"
    )
    d = dict(np.load(args.model))
    d["G"] = np.array(model.G) * beta[:, None].astype(np.float32)
    d["B"] = offset.astype(np.float32)
    out = os.path.splitext(args.model)[0] + ("_calbeta" if args.no_offset else "_cal") + ".npz"
    np.savez(out, **d)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
