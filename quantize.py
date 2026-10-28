# Compress the embeddings of a trained pair model and measure the cost in held-out bits per person.
# Every variant is scored with pairmodel.evaluate (exact log Z), against the same null model.
# Only the names with an embedding are stored (min_embed leaves the others at u = 0, the null score).
# The export orders names by count, so the stored names are a prefix and need no ids.
import argparse
import dataclasses
import json
import os
import time

import mlx.core as mx
import numpy as np

import namedata
import pairmodel as pm


def uniform_codes(W, bits, w, clips=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)):
    """Symmetric quantization with one scale per column; clip level chosen by weighted MSE.
    Returns the integer codes and the scale of each column."""
    qmax = 2 ** (bits - 1) - 1
    codes, scales = np.empty(W.shape, np.int8), np.empty(W.shape[1], np.float32)
    for j in range(W.shape[1]):
        col, best = W[:, j], None
        for clip in clips:
            scale = np.float32(clip * np.abs(col).max() / qmax)
            q = np.clip(np.round(col / scale), -qmax, qmax)
            err = np.sum(w * (q * scale - col) ** 2)
            if best is None or err < best[0]:
                best = (err, q, scale)
        codes[:, j], scales[j] = best[1], best[2]
    return codes, scales


def uniform(W, bits, w):
    codes, scales = uniform_codes(W, bits, w)
    return codes * scales


def save_quantized(path, model_path, method, U, V, rf, rl, cf, cl):
    """Write the model with compact codes for the names that have a vector (pairmodel.load_counts reads it).
    int8, int4: codes and one scale per column. pqG: 1 byte per group of G columns and the codebooks."""
    src = np.load(model_path)
    d = {key: src[key] for key in src.files if key not in ("U", "V", "G_logit")}
    for name, W, rows, w in (("U", U, rf, cf), ("V", V, rl, cl)):
        has = np.zeros(len(W), bool)
        has[rows] = True
        d[f"{name}_rows"] = np.packbits(has)
        if method.startswith("pq"):
            d[f"{name}_codes"], d[f"{name}_codebooks"] = product_codes(W[rows], int(method[2:]), w)
            continue
        codes, d[f"{name}_scales"] = uniform_codes(W[rows], int(method[3:]), w)
        if method == "int4":  # two codes per byte, offset by 8
            c = (codes + 8).astype(np.uint8)
            codes = (c[:, 0::2] << 4) | c[:, 1::2]
        d[f"{name}_codes"] = codes
    np.savez(path, method=method, **d)


def kmeans(X, w, n_centroids=256, iters=25, seed=0, chunk=2**18):
    """Weighted k-means on GPU. Returns the centroids and the centroid index of each row."""
    rng = np.random.default_rng(seed)
    p = w / w.sum()
    init = rng.choice(len(X), size=n_centroids, replace=False, p=p)
    Xm, wm = mx.array(X), mx.array(w.astype(np.float32))
    C = Xm[mx.array(init)]
    x2 = mx.sum(Xm * Xm, axis=1, keepdims=True)
    for _ in range(iters):
        assign = mx.concatenate([
            mx.argmin(x2[i : i + chunk] - 2 * Xm[i : i + chunk] @ C.T + mx.sum(C * C, axis=1), axis=1)
            for i in range(0, len(X), chunk)
        ])
        sums = mx.zeros(C.shape).at[assign].add(Xm * wm[:, None])
        tot = mx.zeros((n_centroids,)).at[assign].add(wm)
        C = mx.where(tot[:, None] > 0, sums / mx.maximum(tot[:, None], 1e-12), C)
        mx.eval(C, assign)
    return np.array(C), np.array(assign)


def product_codes(W, group, w):
    """Product quantization: 256 centroids (1 byte) per group of `group` columns.
    Returns the codes (n, k / group) and the codebooks (k / group, 256, group)."""
    parts = [kmeans(W[:, j : j + group], w) for j in range(0, W.shape[1], group)]
    return np.stack([a for _, a in parts], axis=1).astype(np.uint8), np.stack([C for C, _ in parts])


def product(W, group, w):
    codes, books = product_codes(W, group, w)
    return books[np.arange(len(books)), codes].reshape(len(W), -1)


def keep_top(W, counts, frac):
    """Keep the embeddings of the most frequent `frac` of names. The other names share one
    vector, their count-weighted mean (zero would drop the constant offset that u.v learns)."""
    n_keep = int(round(frac * len(W)))
    drop = np.argsort(-counts, kind="stable")[n_keep:]
    out = W.copy()
    w = counts[drop] + 1e-12
    out[drop] = (W[drop] * w[:, None]).sum(0) / w.sum()
    return out


def on_rows(W, rows, f):
    """f applied to the stored rows of W; the other rows stay at 0."""
    out = np.zeros_like(W)
    out[rows] = f(W[rows])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help=".npz saved by pairmodel.py --save-dir")
    parser.add_argument("--country", default=None, help="country of a 'country' model (e.g. FR)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default=None)
    parser.add_argument("--only", default=None, help="run only variants whose name contains this text")
    parser.add_argument("--save", choices=["int8", "int4", "pq2", "pq4", "pq8"], default=None,
                        help="instead of the comparison, write <model>_<save>.npz and evaluate that file")
    args = parser.parse_args()

    ds = namedata.load(args.data_dir, args.country)
    print(f"Data: {namedata.describe(ds)}", flush=True)
    alpha = float(np.load(args.model)["alpha"])
    base = pm.load_counts(args.model, ds, False)  # keeps the gates, offsets and restricted support
    U, V = np.array(base.U), np.array(base.V)
    k = U.shape[1]
    rf, rl = np.nonzero(np.abs(U).sum(1) > 0)[0], np.nonzero(np.abs(V).sum(1) > 0)[0]
    cf = np.bincount(ds.f, weights=ds.n_train, minlength=ds.n_first)[rf]
    cl = np.bincount(ds.l, weights=ds.n_train, minlength=ds.n_last)[rl]
    n_names = len(rf) + len(rl)
    country_bytes = 0 if base.G is None else base.G.size * 4 + (0 if base.B is None else base.B.size * 4)

    if args.save:
        path = f"{os.path.splitext(args.model)[0]}_{args.save}.npz"
        save_quantized(path, args.model, args.save, U, V, rf, rl, cf, cl)
        print(f"saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)
        null, res = pm.evaluate(pm.null_model(ds, alpha), ds), pm.evaluate(pm.load_counts(path, ds, False), ds)
        below = [c for c in ds.countries if res["per_country"][c] > null["per_country"][c]]
        print(f"  {os.path.basename(path)}: test {res['test_bits']:.4f}  gain {null['test_bits'] - res['test_bits']:+.4f}"
              f"  {len(below)} countries below null {' '.join(below)}  ({res['eval_s']:.0f}s)", flush=True)
        res.update(name=os.path.basename(path), file_mb=os.path.getsize(path) / 1e6, below_null=len(below))
        if args.out:
            with open(args.out, "w") as fh:
                json.dump([null, res], fh, indent=1)
        return

    results = []

    def run(name, Uq, Vq, bytes_per_name, n_stored=n_names, extra_bytes=0):
        t = time.time()
        if Uq is None:
            model = pm.null_model(ds, alpha)
        else:
            model = dataclasses.replace(base, U=mx.array(Uq.astype(np.float32)), V=mx.array(Vq.astype(np.float32)))
            extra_bytes += country_bytes
        res = pm.evaluate(model, ds)
        mb = (bytes_per_name * n_stored + extra_bytes) / 1e6
        null = results[0] if results else res
        below = sum(res["per_country"][c] > null["per_country"][c] for c in ds.countries)
        res.update(name=name, bytes_per_name=bytes_per_name, n_stored=n_stored, mb=mb, below_null=below)
        results.append(res)
        ref = results[1]["test_bits"] if len(results) > 1 else None
        loss = "" if ref is None else f"  vs fp32 {res['test_bits'] - ref:+.4f}"
        print(f"  {name:<34} {bytes_per_name:>5.1f} B/name {mb:>8.1f} MB  test {res['test_bits']:.4f}"
              f"  gain {null['test_bits'] - res['test_bits']:+.4f}{loss}  {below} below null"
              f"  ({time.time() - t:.0f}s)", flush=True)

    def want(name):
        return not args.only or args.only in name

    def quant(f_u, f_v):
        return on_rows(U, rf, f_u), on_rows(V, rl, f_v)

    run("null (no embeddings)", None, None, 0, 0)
    run("float32", U, V, 4 * k)
    if want("float16"):
        run("float16", U.astype(np.float16), V.astype(np.float16), 2 * k)
    if want("int8 per column"):
        run("int8 per column", *quant(lambda W: uniform(W, 8, cf), lambda W: uniform(W, 8, cl)), k)
    U4, V4 = quant(lambda W: uniform(W, 4, cf), lambda W: uniform(W, 4, cl))
    if want("int4 per column"):
        run("int4 per column", U4, V4, k / 2)
    for group in (2, 4, 8):
        if group < k and want(f"PQ {group} dims/byte"):
            cb = 2 * 256 * k * 4  # codebooks for U and V, float32
            run(f"PQ {group} dims/byte", *quant(lambda W: product(W, group, cf), lambda W: product(W, group, cl)),
                k / group, extra_bytes=cb)
    # Top names only (int4), the other names share their mean vector.
    for frac in (0.5, 0.25, 0.1, 0.05):
        if not want(f"int4, top {frac:.0%} of names"):
            continue
        Ut = on_rows(U4, rf, lambda W: keep_top(W, cf, frac))
        Vt = on_rows(V4, rl, lambda W: keep_top(W, cl, frac))
        run(f"int4, top {frac:.0%} of names", Ut, Vt, k / 2, int(round(frac * len(rf))) + int(round(frac * len(rl))) + 2)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
