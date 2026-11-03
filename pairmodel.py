# Pair models for full names (MLX), trained and evaluated on data/pairs.npz.
#
# All models have the form
#   p(f, l | c) = exp(X[f, c] + Y[l, c] + u_f . diag(g_c) . v_l + b_c) / Z_c
# and are scored by exact held-out negative log-likelihood, in bits per person.
# g_c = 1 unless the model has per-country gates, b_c = 0 unless it has per-country offsets.
# With "restrict", the interaction (with b_c) is applied only when both names occur in the
# training data of country c (training only sees such pairs); other pairs get the null score.
#
#   null      X, Y = log of smoothed per-country counts, no u, v
#             (first and last names independent in each country, like main.py)
#   counts    X, Y = same log counts (+ optional learned a_f, b_l), learned u, v.
#             Positives are sampled in proportion to counts, negatives from the
#             same country's marginals, so the sampled-softmax logit is a + b + u.v.
#   original  training.py's model, trained by train_original.py (one country).
#
# Z_c is computed exactly, in row chunks: over all n x m pairs for all countries at once,
# or, with "restrict", over each country's own names plus a closed form for the rest.
import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import namedata

LN2 = math.log(2)


# -----------------------------
# Exact evaluation
# -----------------------------
@mx.compile
def _chunk_log_z(Uc, eXc, V, eY):
    s = Uc @ V.T  # (chunk, m)
    smax = mx.max(s)
    return mx.log(mx.sum(eXc * (mx.exp(s - smax) @ eY), axis=0)) + smax


def log_partition(X, Y, U=None, V=None):
    """log Z_c = log sum_{f,l} exp(X[f,c] + Y[l,c] + u_f . v_l), for all c at once."""
    xm, ym = mx.max(X, axis=0), mx.max(Y, axis=0)
    eX, eY = mx.exp(X - xm), mx.exp(Y - ym)
    if U is None:
        return mx.log(mx.sum(eX, axis=0)) + mx.log(mx.sum(eY, axis=0)) + xm + ym
    chunk = max(8, 2**27 // V.shape[0])
    parts = []
    for i in range(0, U.shape[0], chunk):
        part = _chunk_log_z(U[i : i + chunk], eX[i : i + chunk], V, eY)
        mx.eval(part)
        parts.append(part)
    return mx.logsumexp(mx.stack(parts), axis=0) + xm + ym


@dataclass
class Model:
    X: mx.array  # (n_first, C) log first-name weights per country
    Y: mx.array  # (n_last, C)
    U: mx.array = None  # (n_first, k)
    V: mx.array = None  # (n_last, k)
    G: mx.array = None  # (C, k) per-country gates, or None for g_c = 1
    restrict: bool = False
    B: mx.array = None  # (C,) per-country offsets, or None for b_c = 0 (restricted models only).
    # Fit them on held-out people: on the train split they only grow, since all train pairs are in the support.


def support(ds):
    """Flat (name * C + c) masks of the names of country c: the names that occur in its training data, or for a
    country with a base (ds.base: INSEE for France), the names of its base."""
    if not hasattr(ds, "_support"):
        masks = []
        for side, (ids, n) in enumerate([(ds.f, ds.n_first), (ds.l, ds.n_last)]):
            m = np.bincount(ids.astype(np.int64) * ds.C + ds.c, weights=ds.n_train, minlength=n * ds.C) > 0
            m = m.reshape(n, ds.C)
            for c, counts in ds.base.items():
                m[:, c] = counts[side] > 0
            masks.append(m.reshape(-1))
        ds._support = tuple(masks)
    return ds._support


def log_partition_restricted(model, ds, countries=None):
    """log Z_c when the interaction only applies to pairs of names seen in country c
    (for all countries, or only the given country indices; the others get NaN)."""
    in_f, in_l = (m.reshape(-1, ds.C) for m in support(ds))
    X, Y = np.array(model.X), np.array(model.Y)
    logZ = np.full(ds.C, np.nan)
    for c in range(ds.C) if countries is None else countries:
        fi, li = np.nonzero(in_f[:, c])[0], np.nonzero(in_l[:, c])[0]
        xc, yc = X[:, c].astype(np.float64), Y[:, c].astype(np.float64)
        xm, ym = xc.max(), yc.max()
        ex, ey = np.exp(xc - xm), np.exp(yc - ym)
        sx_in, sy_in = ex[fi].sum(), ey[li].sum()
        sx_out, sy_out = ex[~in_f[:, c]].sum(), ey[~in_l[:, c]].sum()
        rest = sx_out * (sy_in + sy_out) + sx_in * sy_out  # pairs with a name outside the support
        Uc = model.U[mx.array(fi)]
        if model.G is not None:
            Uc = Uc * model.G[c]
        Vc = model.V[mx.array(li)]
        eXc = mx.array(ex[fi].astype(np.float32))[:, None]
        eYc = mx.array(ey[li].astype(np.float32))[:, None]
        chunk = max(8, 2**27 // len(li))
        parts = []
        for i in range(0, len(fi), chunk):
            part = _chunk_log_z(Uc[i : i + chunk], eXc[i : i + chunk], Vc, eYc)
            mx.eval(part)
            parts.append(part)
        log_in = float(mx.logsumexp(mx.concatenate(parts)))
        if model.B is not None:
            log_in += float(model.B[c])
        logZ[c] = np.logaddexp(log_in, np.log(rest) if rest > 0 else -np.inf) + xm + ym
    return mx.array(logZ.astype(np.float32))


def split_nll(model, ds, logZ, split, chunk=2**22):
    """Negative log2-likelihood of each person group in a split, and the group sizes."""
    n = getattr(ds, f"n_{split}")
    idx = np.nonzero(n > 0)[0]
    Xf, Yf = model.X.reshape(-1), model.Y.reshape(-1)
    if model.restrict:
        in_f, in_l = support(ds)
    s = np.empty(len(idx))
    for i in range(0, len(idx), chunk):
        j = idx[i : i + chunk]
        f, l, c = (mx.array(a[j]) for a in (ds.f, ds.l, ds.c))
        si = Xf[f * ds.C + c] + Yf[l * ds.C + c] - logZ[c]
        if model.U is not None:
            v = model.V[l] if model.G is None else model.V[l] * model.G[c]
            inter = mx.sum(model.U[f] * v, axis=1)
            if model.B is not None:
                inter = inter + model.B[c]
            if model.restrict:
                fc, lc = ds.f[j].astype(np.int64) * ds.C + ds.c[j], ds.l[j].astype(np.int64) * ds.C + ds.c[j]
                inter = inter * mx.array(in_f[fc] & in_l[lc])
            si = si + inter
        s[i : i + len(j)] = np.array(si)
    return idx, -s * n[idx] / LN2, n[idx]


def evaluate(model, ds):
    """Valid and test bits per person (one exact log Z), and test bits per country."""
    if isinstance(model, tuple):
        model = Model(*model)
    t0 = time.time()
    if model.U is not None and (model.restrict or model.G is not None or model.B is not None):
        assert model.restrict, "gated or offset models need the restricted evaluation"
        logZ = log_partition_restricted(model, ds)
    else:
        logZ = log_partition(model.X, model.Y, model.U, model.V)
    out = {}
    for split in ("valid", "test"):
        idx, nll, n = split_nll(model, ds, logZ, split)
        out[f"{split}_bits"] = float(nll.sum() / n.sum())
        if split == "test":
            people = np.bincount(ds.c[idx], weights=n, minlength=ds.C)
            per_c = np.bincount(ds.c[idx], weights=nll, minlength=ds.C) / np.maximum(people, 1)
            out["per_country"] = dict(zip(ds.countries, per_c.tolist()))
    out["eval_s"] = time.time() - t0
    return out


# -----------------------------
# Models
# -----------------------------
def null_model(ds, alpha):
    X, Y = namedata.log_marginals(ds, alpha)
    return Model(mx.array(X), mx.array(Y))


def load_original(path):
    d = np.load(path)
    return Model(mx.array(d["a"][:, None]), mx.array(d["b"][:, None]), mx.array(d["U"]), mx.array(d["V"]))


def load_counts(path, ds, restrict):
    """A model saved by --save-dir, on top of the smoothed log counts of ds."""
    d = np.load(path)
    X, Y = (mx.array(a) for a in namedata.log_marginals(ds, float(d["alpha"])))
    if "a" in d:
        X, Y = X + mx.array(d["a"])[:, None], Y + mx.array(d["b"])[:, None]
    U, V = (
        mx.array(dequantize(d, name, n) if f"{name}_codes" in d else d[name])
        for name, n in (("U", ds.n_first), ("V", ds.n_last))
    )
    G = mx.array(d["G"]) if "G" in d else None
    B = mx.array(d["B"]) if "B" in d else None
    return Model(X, Y, U, V, G, restrict or G is not None or B is not None, B)


def dequantize(d, name, n):
    """Name vectors written by quantize.py --save, 0 for names without a vector."""
    has = np.unpackbits(d[f"{name}_rows"], count=n).astype(bool)
    codes, method = d[f"{name}_codes"], str(d["method"])
    if method.startswith("pq"):  # one byte per group of columns: an index into the codebook of the group
        books = d[f"{name}_codebooks"]
        rec = books[np.arange(len(books)), codes].reshape(len(codes), -1)
    else:
        if method == "int4":
            codes = np.stack([codes >> 4, codes & 15], axis=-1).reshape(len(codes), -1).astype(np.int8) - 8
        rec = codes * d[f"{name}_scales"]
    W = np.zeros((n, rec.shape[1]), np.float32)
    W[has] = rec
    return W


class PairParams(nn.Module):
    def __init__(self, n_first, n_last, k, bias, n_countries=0):
        super().__init__()
        self.U = 0.01 * mx.random.normal((n_first, k))
        self.V = 0.01 * mx.random.normal((n_last, k))
        if bias:
            self.a = mx.zeros((n_first,))
            self.b = mx.zeros((n_last,))
        if n_countries:
            # Gates g = sigmoid(G_logit) in (0, 1): a country can shrink dimensions, not amplify them
            # (free gates grow and undo the weight decay on U and V).
            self.G_logit = mx.full((n_countries, k), 4.0)


def train_counts(
    ds,
    alpha,
    k=16,
    K_neg=16,
    batch=2**16,
    steps=10000,
    lr=1e-2,
    wd=1e-4,
    bias=False,
    neg_uniform=0.0,
    min_embed=0,
    gates=False,
    restrict=False,
    pool=0,
    seed=0,
    log_every=1000,
    eval_every=0,
):
    """Learn u, v (and optional a, b, per-country gates G) on top of per-country log counts.

    neg_uniform: fraction of negatives whose names come from a uniform draw over the
      country's unique training pairs (so rare names appear as negatives), with the
      sampled-softmax logits corrected by the mixture proposal density.
    min_embed: names with fewer training people have no embedding (u = 0, the null score).
    pool: if > 0, replace the sampled softmax (K_neg negatives per person, positive included)
      by a pooled estimate of each country's log Z from `pool` pairs per country and step, by
      self-normalized importance sampling from a mixture of three proposals: train people
      (joint counts), the null itself on the country's support, and names of unique train
      triples (flatter, so rare names occur). The joint part covers pairs that the model learns
      to score high, and the null part bounds the importance weights by 3. The loss is then an estimate of the train NLL minus
      the null's, in nats per person.
    """
    mx.random.seed(seed)
    model = PairParams(ds.n_first, ds.n_last, k, bias, ds.C if gates else 0)
    cf, cl = namedata.people(ds)  # the dropped triples too (namedata.with_insee): the same names get a vector
    mask_f = mx.array((cf >= min_embed).astype(np.float32))[:, None]
    mask_l = mx.array((cl >= min_embed).astype(np.float32))[:, None]
    K_u = int(round(neg_uniform * K_neg))
    if bias or gates:
        opt = optim.MultiOptimizer(
            [optim.AdamW(lr, weight_decay=wd, bias_correction=True), optim.Adam(lr, bias_correction=True)],
            [lambda path, _: path in ("U", "V")],
        )
    else:
        opt = optim.AdamW(lr, weight_decay=wd, bias_correction=True)
    opt.init(model.trainable_parameters())

    def loss_fn(model, f, l, c, correction):
        v = (model.V * mask_l)[l]
        if gates:
            v = v * mx.sigmoid(model.G_logit[c])[:, None, :]
        logits = mx.sum((model.U * mask_f)[f] * v, axis=-1) + correction
        if bias:
            logits = logits + model.a[f] + model.b[l]
        return mx.mean(mx.logsumexp(logits, axis=1) - logits[:, 0])

    def pooled_loss(model, f, l, c, fn, ln, lw, frac):
        g = mx.sigmoid(model.G_logit) if gates else mx.ones((ds.C, k))
        sp = mx.sum(model.U[f] * mask_f[f] * model.V[l] * mask_l[l] * g[c], axis=-1)
        sn = mx.sum(model.U[fn] * mask_f[fn] * model.V[ln] * mask_l[ln] * g[:, None, :], axis=-1)
        log_in = mx.logsumexp(sn + lw, axis=1) - mx.logsumexp(lw, axis=1) + log_pin
        return mx.sum(frac * mx.logaddexp(log_pout, log_in)) - mx.mean(sp)

    assert not (pool and bias), "the pooled loss has no a, b terms"
    loss_and_grad = nn.value_and_grad(model, pooled_loss if pool else loss_fn)

    def log_density(ids, n_names, weights):
        N = np.bincount(ids.astype(np.int64) * ds.C + ds.c, weights=weights, minlength=n_names * ds.C)
        N = N.reshape(n_names, ds.C)
        with np.errstate(divide="ignore"):
            return mx.array(np.log(N / N.sum(0, keepdims=True)).astype(np.float32).reshape(-1))

    # People are sampled through the cumulative train counts (triples sorted by country).
    cum = np.cumsum(ds.n_train.astype(np.int64))
    assert cum[-1] < 2**31
    people = np.bincount(ds.c, weights=ds.n_train, minlength=ds.C).astype(np.int64)
    F, L, Cn, cum, c_end = (mx.array(a.astype(np.int32)) for a in (ds.f, ds.l, ds.c, cum, np.cumsum(people)))
    c_start = c_end - mx.array(people.astype(np.int32))
    total = int(cum[-1])

    if K_u or pool:
        # Uniform proposal over training triples of each country, and the log densities of
        # both proposals (people marginals m, triple marginals t) for the logit correction.
        train_idx = np.nonzero(ds.n_train > 0)[0].astype(np.int32)
        tri = np.bincount(ds.c[train_idx], minlength=ds.C).astype(np.int64)
        t_end_np = np.cumsum(tri)
        t_idx, t_end = mx.array(train_idx), mx.array(t_end_np.astype(np.int32))
        t_start = t_end - mx.array(tri.astype(np.int32))
        one = (ds.n_train > 0).astype(np.float64)
        lm_f, lm_l = log_density(ds.f, ds.n_first, ds.n_train), log_density(ds.l, ds.n_last, ds.n_train)
        lt_f, lt_l = log_density(ds.f, ds.n_first, one), log_density(ds.l, ds.n_last, one)
    if K_u:
        log_mix = (math.log1p(-neg_uniform) if neg_uniform < 1 else -math.inf, math.log(neg_uniform))

    if pool:
        # Target: the smoothed null p (log lp_f + lp_l), with its mass inside (P_in) and outside
        # (P_out) each country's support. Proposal densities: t above, p on the support, and the
        # joint train counts, found by binary search in the sorted (c, f, l) keys.
        in_f, in_l = (m.reshape(-1, ds.C) for m in support(ds))
        Xn, Yn = namedata.log_marginals(ds, alpha)
        px_in, py_in = ((np.exp(P.astype(np.float64)) * m).sum(0) for P, m in ((Xn, in_f), (Yn, in_l)))
        p_in = px_in * py_in
        log_pin = mx.array(np.log(p_in).astype(np.float32))
        with np.errstate(divide="ignore"):
            log_pout = mx.array(np.log(np.maximum(1 - p_in, 0)).astype(np.float32))
        log_px_in, log_py_in = (mx.array(np.log(a).astype(np.float32)) for a in (px_in, py_in))
        lp_f, lp_l = mx.array(Xn.reshape(-1)), mx.array(Yn.reshape(-1))
        del Xn, Yn

        def support_sampler(ids, n_names, in_mask):
            """Names of each country's support with cumulative global weights 2 G + 1 (in
            proportion to the smoothed global marginal g), one segment per country, and the
            probability of drawing from the train people instead: p on the support is
            (N_fc + alpha g_f) / (N_c + alpha G_in,c), a mixture of m and g on the support."""
            G = np.bincount(ids, weights=ds.n_train, minlength=n_names)
            w = (2 * G + 1).astype(np.int64)
            names, cums, size = [], [], np.zeros(ds.C, np.int64)
            for c in range(ds.C):
                nm = np.nonzero(in_mask[:, c])[0]
                cw = np.cumsum(w[nm])
                names.append(nm)
                cums.append(cw + size[:c].sum())
                size[c] = cw[-1]
            g_in = size / (2 * G.sum() + n_names)
            pi = people / (people + alpha * g_in)
            start = np.concatenate([[0], np.cumsum(size)[:-1]])
            assert size.max() < 2**31
            return (
                mx.array(np.concatenate(names).astype(np.int32)),
                mx.array(np.concatenate(cums)),
                mx.array(start),
                mx.array(size.astype(np.int32)),
                mx.array(pi.astype(np.float32)),
            )

        samplers = [support_sampler(ids, n, m) for ids, n, m in ((ds.f, ds.n_first, in_f), (ds.l, ds.n_last, in_l))]
        key = (ds.c.astype(np.int64) << 43) | (ds.f.astype(np.int64) << 22) | ds.l
        order = np.argsort(key)
        keys = mx.array(key[order])
        with np.errstate(divide="ignore"):
            log_joint = mx.array(np.log(ds.n_train[order] / people[ds.c[order]]).astype(np.float32))
        del key, order
        rows = mx.arange(ds.C)[:, None]
        n3 = pool // 3

    state = [model.state, opt.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step():
        pos = mx.searchsorted(cum, mx.random.randint(0, total, (batch,), dtype=mx.int32), side="right")
        c = Cn[pos]
        if pool:

            def people_idx():  # (C, n3) triples drawn in proportion to train people
                return mx.searchsorted(
                    cum, mx.random.randint(c_start[:, None], c_end[:, None], (ds.C, n3)), side="right"
                )

            def triple_idx():  # (C, n3) unique train triples, uniform
                return t_idx[mx.random.randint(t_start[:, None], t_end[:, None], (ds.C, n3))]

            def null_idx(names_of_people, sampler):  # (C, n3) names drawn from p on the support
                names, cums, start, size, pi = sampler
                r = start[:, None] + mx.random.randint(0, size[:, None], (ds.C, n3)).astype(mx.int64)
                from_g = names[mx.searchsorted(cums, r, side="right")]
                return mx.where(
                    mx.random.uniform(shape=(ds.C, n3)) < pi[:, None], names_of_people[people_idx()], from_g
                )

            j = people_idx()
            fn = mx.concatenate([F[j], null_idx(F, samplers[0]), F[triple_idx()]], axis=1)
            ln = mx.concatenate([L[j], null_idx(L, samplers[1]), L[triple_idx()]], axis=1)
            i_f, i_l = fn * ds.C + rows, ln * ds.C + rows
            q = (rows.astype(mx.int64) << 43) | (fn.astype(mx.int64) << 22) | ln.astype(mx.int64)
            hit = mx.minimum(mx.searchsorted(keys, q), keys.size - 1)
            lj = mx.where(keys[hit] == q, log_joint[hit], -math.inf)
            l_null = lp_f[i_f] - log_px_in[:, None] + lp_l[i_l] - log_py_in[:, None]
            log_q = mx.logaddexp(mx.logaddexp(lj, l_null), lt_f[i_f] + lt_l[i_l]) - math.log(3)
            lw = lp_f[i_f] + lp_l[i_l] - log_q
            frac = mx.zeros((ds.C,)).at[c].add(1.0) / batch
            loss, grads = loss_and_grad(model, F[pos], L[pos], c, fn, ln, lw, frac)
            opt.update(model, grads)
            return loss
        start = c_start[c][:, None]
        width = (c_end - c_start)[c][:, None].astype(mx.float32)

        def same_country(K):
            r = start + (mx.random.uniform(shape=(batch, K)) * width).astype(mx.int32)
            return mx.searchsorted(cum, mx.minimum(r, c_end[c][:, None] - 1), side="right")

        def same_country_triples(K):
            ts, tw = t_start[c][:, None], (t_end - t_start)[c][:, None].astype(mx.float32)
            r = ts + (mx.random.uniform(shape=(batch, K)) * tw).astype(mx.int32)
            return t_idx[mx.minimum(r, t_end[c][:, None] - 1)]

        fs, ls = [F[pos][:, None], F[same_country(K_neg - K_u)]], [L[pos][:, None], L[same_country(K_neg - K_u)]]
        if K_u:
            fs.append(F[same_country_triples(K_u)])
            ls.append(L[same_country_triples(K_u)])
        f = mx.concatenate(fs, axis=1)  # column 0 = positive
        l = mx.concatenate(ls, axis=1)
        if K_u:
            # logit = score - log q, with q = (1 - eps) m(f) m(l) + eps t(f) t(l); score uses m.
            i_f, i_l = f * ds.C + c[:, None], l * ds.C + c[:, None]
            lm = lm_f[i_f] + lm_l[i_l]
            correction = lm - mx.logaddexp(log_mix[0] + lm, log_mix[1] + lt_f[i_f] + lt_l[i_l])
        else:
            correction = mx.zeros(f.shape)
        loss, grads = loss_and_grad(model, f, l, c, correction)
        opt.update(model, grads)
        return loss

    X0, Y0 = (mx.array(a) for a in namedata.log_marginals(ds, alpha))

    def as_model():
        X, Y = (X0 + model.a[:, None], Y0 + model.b[:, None]) if bias else (X0, Y0)
        G = mx.sigmoid(model.G_logit) if gates else None
        return Model(X, Y, model.U * mask_f, model.V * mask_l, G, restrict or gates or bool(pool))

    t0 = time.time()
    for i in range(1, steps + 1):
        loss = step()
        mx.eval(loss, state)
        if i % log_every == 0 or i == steps:
            msg = f"    step {i:>6} loss {loss.item():.4f} {time.time() - t0:.0f}s"
            if eval_every and i % eval_every == 0:
                msg += f"  valid {evaluate(as_model(), ds)['valid_bits']:.4f}"
            print(msg, flush=True)

    model.U, model.V = model.U * mask_f, model.V * mask_l
    return as_model(), model


# -----------------------------
# Experiments
# -----------------------------
def tune_alpha(ds, grid=(1, 10, 100, 1000, 10000, 100000, 1000000)):
    best = None
    for alpha in grid:
        v = evaluate(null_model(ds, alpha), ds)["valid_bits"]
        print(f"  null alpha={alpha:<6} valid {v:.4f} bits/person", flush=True)
        if best is None or v < best[1]:
            best = (alpha, v)
    return best[0]


def report(name, model, ds, results, null=None, **extra):
    res = {"name": name, **extra, **evaluate(model, ds)}
    gain = "" if null is None else f"  gain {null['test_bits'] - res['test_bits']:+.4f}"
    print(
        f"  {name:<36} valid {res['valid_bits']:.4f} test {res['test_bits']:.4f} bits/person{gain}"
        f"  (eval {res['eval_s']:.0f}s)",
        flush=True,
    )
    results.append(res)
    return res


def main():
    mx.set_cache_limit(8 << 30)  # freed GPU buffers of many shapes otherwise pile up in the cache
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["country", "world"])
    parser.add_argument("--country", default="FR", help="country for the 'country' experiment")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--original", default=None, help="evaluate a model saved by train_original.py")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--k", type=int, nargs="*", default=[16])
    parser.add_argument("--wd", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--neg", type=int, default=16)
    parser.add_argument("--pool", type=int, default=0, help="negatives per country and step for the pooled loss")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--neg-uniform", type=float, nargs="+", default=[0.0])
    parser.add_argument("--min-embed", type=int, nargs="+", default=[0])
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--gates", action="store_true", help="per-country gates on the interaction")
    parser.add_argument(
        "--restrict",
        action="store_true",
        help="interaction only between names seen in the country (implied by --gates)",
    )
    parser.add_argument("--load", nargs="*", default=[], help="evaluate models saved by --save-dir")
    parser.add_argument("--save-dir", default=None, help="save each counts model here")
    parser.add_argument("--alpha", type=float, default=None, help="smoothing of the bases (default: tuned on the data)")
    parser.add_argument("--out", default=None, help="append results to this JSON file")
    args = parser.parse_args()

    t0 = time.time()
    ds = namedata.load(args.data_dir, args.country if args.experiment == "country" else None)
    print(f"Data: {namedata.describe(ds)} ({time.time() - t0:.0f}s)", flush=True)

    results = []

    def save_results():
        if args.out:
            old = json.load(open(args.out)) if os.path.exists(args.out) else []
            with open(args.out, "w") as fh:
                json.dump(old + results[len(saved) :], fh, indent=1)
            saved.extend(results[len(saved) :])

    saved = []
    alpha = args.alpha if args.alpha is not None else tune_alpha(ds)
    null = report(f"null (alpha={alpha})", null_model(ds, alpha), ds, results)
    save_results()
    if args.original:
        report("original training.py recipe, k=8", load_original(args.original), ds, results, null)
    for path in args.load:
        model = load_counts(path, ds, args.restrict)
        report(os.path.basename(path) + (" restricted" if model.restrict else ""), model, ds, results, null)
        save_results()

    grid = [(k, wd, eps, me) for k in args.k for wd in args.wd for eps in args.neg_uniform for me in args.min_embed]
    for k, wd, eps, me in grid:
        neg = f"pool={args.pool}" if args.pool else f"neg={args.neg}"
        opts = f"k={k} wd={wd:g} {neg} unif={eps:g} min_embed={me} steps={args.steps}"
        name = "counts " + opts + (" +bias" if args.bias else "") + (" +gates" if args.gates else "")
        name += " restricted" if (args.restrict or args.gates or args.pool) else ""
        print(f"  training {name}...", flush=True)
        t = time.time()
        model, params = train_counts(
            ds,
            alpha,
            k=k,
            K_neg=args.neg,
            wd=wd,
            lr=args.lr,
            steps=args.steps,
            bias=args.bias,
            neg_uniform=eps,
            min_embed=me,
            gates=args.gates,
            restrict=args.restrict,
            pool=args.pool,
            eval_every=args.eval_every,
        )
        report(
            name,
            model,
            ds,
            results,
            null,
            k=k,
            wd=wd,
            neg_uniform=eps,
            min_embed=me,
            steps=args.steps,
            train_s=time.time() - t,
        )
        save_results()
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            tag = f"{args.experiment}_{args.country if args.experiment == 'country' else 'all'}"
            path = (
                f"{args.save_dir}/{tag}_k{k}_wd{wd:g}_u{eps:g}_me{me}_s{args.steps}"
                f"{'_bias' if args.bias else ''}{'_gates' if args.gates else ''}"
                f"{f'_pool{args.pool}' if args.pool else ''}.npz"
            )
            arrays = {key: np.array(v) for key, v in params.parameters().items()}
            if "G_logit" in arrays:
                arrays["G"] = np.array(mx.sigmoid(params.G_logit))
            np.savez(path, alpha=alpha, **arrays)
        mx.clear_cache()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
