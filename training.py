# JAX training code for a global joint model p(f,l) ∝ exp(s(f,l))
# with s(f,l) = a_f + b_l + u_f^T v_l
#
# Training objective: sampled softmax with importance correction under q(f,l)=qF(f)qL(l)
#
# Requires:
#   - arrays (edge_f, edge_l, edge_x) for the nnz positive pairs
#   - marginals (cf, cl) to build qF, qL
#
# Notes:
# - Works on CPU; on Apple Silicon GPU, install JAX Metal backend and set JAX_PLATFORM_NAME=metal.
# - The update is manual SGD (no full-size gradient materialization), using scatter-add on touched rows.

# Optimize CPU threading
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import random
from jax.scipy.special import logsumexp
from tqdm import tqdm


# -----------------------------
# Alias sampler (O(1) discrete sampling)
# -----------------------------
def build_alias_table(probs: np.ndarray):
    """Return (prob, alias) for Vose's alias method.
    probs: 1D array, nonnegative, not necessarily normalized.
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    K = probs.shape[0]

    q = probs * K
    prob = np.zeros(K, dtype=np.float32)
    alias = np.zeros(K, dtype=np.int32)

    small = [i for i in range(K) if q[i] < 1.0]
    large = [i for i in range(K) if q[i] >= 1.0]

    while small and large:
        s = small.pop()
        l = large.pop()

        prob[s] = q[s]
        alias[s] = l

        q[l] = q[l] - (1.0 - q[s])
        if q[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    for i in large + small:
        prob[i] = 1.0
        alias[i] = i

    return prob, alias


def alias_draw(key, prob, alias, shape):
    """Draw samples using alias table (JAX)."""
    K = prob.shape[0]
    k1, k2 = random.split(key, 2)
    kk = random.randint(k1, shape, 0, K, dtype=jnp.int32)
    uu = random.uniform(k2, shape, dtype=jnp.float32)
    return jnp.where(uu < prob[kk], kk, alias[kk])


# -----------------------------
# Model parameters and optimizer state
# -----------------------------
@struct.dataclass
class Params:
    U: jax.Array  # (n, k)
    V: jax.Array  # (m, k)
    a: jax.Array  # (n,)
    b: jax.Array  # (m,)


@struct.dataclass
class AdaGradState:
    """Accumulated squared gradients for AdaGrad."""

    G_U: jax.Array  # (n, k)
    G_V: jax.Array  # (m, k)
    G_a: jax.Array  # (n,)
    G_b: jax.Array  # (m,)


# -----------------------------
# Sampled-softmax training step (sparse AdaGrad)
# -----------------------------
@partial(jax.jit, static_argnames=("K_neg", "lr", "reg", "eps"))
def train_step(
    params: Params,
    adagrad_state: AdaGradState,
    key: jax.Array,
    f_pos: jax.Array,  # (B,)
    l_pos: jax.Array,  # (B,)
    w_pos: jax.Array,  # (B,)  positive weights, e.g. log1p(count)
    prob_f: jax.Array,
    alias_f: jax.Array,
    prob_l: jax.Array,
    alias_l: jax.Array,
    log_qf: jax.Array,
    log_ql: jax.Array,
    lr: float,
    reg: float,  # L2 on embeddings (and optionally biases)
    eps: float,  # AdaGrad epsilon for numerical stability
    K_neg: int,
):
    """One AdaGrad step on a batch, with K_neg negative pairs per positive.
    Uses sampled softmax with importance correction: logits = s(f,l) - log q(f,l).
    Sparse AdaGrad only updates touched embeddings.
    """
    B = f_pos.shape[0]

    # Sample negatives from q(f,l)=qF(f)qL(l)
    key, k1, k2 = random.split(key, 3)
    f_neg = alias_draw(k1, prob_f, alias_f, (B, K_neg))  # (B,K)
    l_neg = alias_draw(k2, prob_l, alias_l, (B, K_neg))  # (B,K)

    # Gather positives
    u_pos = params.U[f_pos]  # (B,k)
    v_pos = params.V[l_pos]  # (B,k)
    s_pos = params.a[f_pos] + params.b[l_pos] + jnp.sum(u_pos * v_pos, axis=1)  # (B,)
    logq_pos = log_qf[f_pos] + log_ql[l_pos]
    logit_pos = s_pos - logq_pos  # (B,)

    # Gather negatives (flatten for efficient gathers)
    f_neg_flat = f_neg.reshape(-1)  # (B*K,)
    l_neg_flat = l_neg.reshape(-1)  # (B*K,)
    u_neg = params.U[f_neg_flat]  # (B*K,k)
    v_neg = params.V[l_neg_flat]  # (B*K,k)
    s_neg = (
        params.a[f_neg_flat] + params.b[l_neg_flat] + jnp.sum(u_neg * v_neg, axis=1)
    )  # (B*K,)
    logq_neg = log_qf[f_neg_flat] + log_ql[l_neg_flat]
    logit_neg = (s_neg - logq_neg).reshape(B, K_neg)  # (B,K)

    # Sampled softmax loss: -w * log softmax(pos among {pos+negs})
    logits = jnp.concatenate([logit_pos[:, None], logit_neg], axis=1)  # (B, K+1)
    lse = logsumexp(logits, axis=1)  # (B,)
    log_p0 = logit_pos - lse  # (B,)
    loss = -jnp.mean(w_pos * log_p0)

    # Softmax probabilities for gradients
    p = jnp.exp(logits - lse[:, None])  # (B,K+1)
    g = p * w_pos[:, None]  # (B,K+1)
    g_pos = g[:, 0] - w_pos  # (B,)
    g_neg = g[:, 1:].reshape(-1)  # (B*K,)

    # Gradient contributions
    # For score s(f,l)=a_f+b_l+u_f^T v_l:
    # d/da_f += g, d/db_l += g, d/du_f += g*v_l, d/dv_l += g*u_f

    # Positive grads
    dU_pos = g_pos[:, None] * v_pos  # (B,k)
    dV_pos = g_pos[:, None] * u_pos  # (B,k)
    da_pos = g_pos  # (B,)
    db_pos = g_pos  # (B,)

    # Negative grads
    dU_neg = g_neg[:, None] * v_neg  # (B*K,k)
    dV_neg = g_neg[:, None] * u_neg  # (B*K,k)
    da_neg = g_neg  # (B*K,)
    db_neg = g_neg  # (B*K,)

    # L2 regularization (only for touched rows/cols; applied as grad += reg*param)
    if reg != 0.0:
        dU_pos = dU_pos + reg * u_pos
        dV_pos = dV_pos + reg * v_pos
        dU_neg = dU_neg + reg * u_neg
        dV_neg = dV_neg + reg * v_neg
        # Optional: biases regularization (usually small or zero). Keep off by default:
        # da_pos = da_pos + reg * params.a[f_pos]
        # db_pos = db_pos + reg * params.b[l_pos]
        # da_neg = da_neg + reg * params.a[f_neg_flat]
        # db_neg = db_neg + reg * params.b[l_neg_flat]

    # Sparse AdaGrad updates
    # Cast to parameter dtype to avoid warnings
    param_dtype = params.U.dtype

    # 1. Accumulate squared gradients (only for touched rows)
    G_U_new = adagrad_state.G_U.at[f_pos].add((dU_pos**2).astype(param_dtype))
    G_U_new = G_U_new.at[f_neg_flat].add((dU_neg**2).astype(param_dtype))

    G_V_new = adagrad_state.G_V.at[l_pos].add((dV_pos**2).astype(param_dtype))
    G_V_new = G_V_new.at[l_neg_flat].add((dV_neg**2).astype(param_dtype))

    G_a_new = adagrad_state.G_a.at[f_pos].add((da_pos**2).astype(param_dtype))
    G_a_new = G_a_new.at[f_neg_flat].add((da_neg**2).astype(param_dtype))

    G_b_new = adagrad_state.G_b.at[l_pos].add((db_pos**2).astype(param_dtype))
    G_b_new = G_b_new.at[l_neg_flat].add((db_neg**2).astype(param_dtype))

    # 2. Apply adaptive learning rate updates (only for touched rows)
    # lr_adapted = lr / sqrt(G + eps)
    update_U_pos = (-lr * dU_pos / (jnp.sqrt(G_U_new[f_pos]) + eps)).astype(param_dtype)
    update_U_neg = (-lr * dU_neg / (jnp.sqrt(G_U_new[f_neg_flat]) + eps)).astype(
        param_dtype
    )
    U_new = params.U.at[f_pos].add(update_U_pos)
    U_new = U_new.at[f_neg_flat].add(update_U_neg)

    update_V_pos = (-lr * dV_pos / (jnp.sqrt(G_V_new[l_pos]) + eps)).astype(param_dtype)
    update_V_neg = (-lr * dV_neg / (jnp.sqrt(G_V_new[l_neg_flat]) + eps)).astype(
        param_dtype
    )
    V_new = params.V.at[l_pos].add(update_V_pos)
    V_new = V_new.at[l_neg_flat].add(update_V_neg)

    update_a_pos = (-lr * da_pos / (jnp.sqrt(G_a_new[f_pos]) + eps)).astype(param_dtype)
    update_a_neg = (-lr * da_neg / (jnp.sqrt(G_a_new[f_neg_flat]) + eps)).astype(
        param_dtype
    )
    a_new = params.a.at[f_pos].add(update_a_pos)
    a_new = a_new.at[f_neg_flat].add(update_a_neg)

    update_b_pos = (-lr * db_pos / (jnp.sqrt(G_b_new[l_pos]) + eps)).astype(param_dtype)
    update_b_neg = (-lr * db_neg / (jnp.sqrt(G_b_new[l_neg_flat]) + eps)).astype(
        param_dtype
    )
    b_new = params.b.at[l_pos].add(update_b_pos)
    b_new = b_new.at[l_neg_flat].add(update_b_neg)

    new_params = Params(U=U_new, V=V_new, a=a_new, b=b_new)
    new_adagrad_state = AdaGradState(G_U=G_U_new, G_V=G_V_new, G_a=G_a_new, G_b=G_b_new)

    return new_params, new_adagrad_state, key, loss


# -----------------------------
# Scoring (global likelihood ranking)
# -----------------------------
@jax.jit
def score_pairs(params: Params, f_ids: jax.Array, l_ids: jax.Array):
    u = params.U[f_ids]
    v = params.V[l_ids]
    return params.a[f_ids] + params.b[l_ids] + jnp.sum(u * v, axis=1)


# -----------------------------
# Data loading utilities
# -----------------------------
def load_data(
    edge_f_path="edge_f.npy", edge_l_path="edge_l.npy", edge_x_path="edge_x.npy"
):
    """Load edge lists from .npy files."""
    print("Loading edge lists...")
    edge_f = np.load(edge_f_path)
    edge_l = np.load(edge_l_path)
    edge_x = np.load(edge_x_path)

    n = int(edge_f.max()) + 1
    m = int(edge_l.max()) + 1

    print(f"  n (first names): {n:,}")
    print(f"  m (last names): {m:,}")
    print(f"  nnz (edges): {len(edge_f):,}")
    print(f"  Total observations: {edge_x.sum():,.0f}")

    return edge_f, edge_l, edge_x, n, m


def load_vocabularies(first_path="first_names.txt", last_path="last_names.txt"):
    """Load name vocabularies from text files."""
    print("Loading vocabularies...")

    with open(first_path, "r", encoding="utf-8") as f:
        first_names = [line.strip() for line in f]

    with open(last_path, "r", encoding="utf-8") as f:
        last_names = [line.strip() for line in f]

    print(f"  First names: {len(first_names):,}")
    print(f"  Last names: {len(last_names):,}")

    return first_names, last_names


# -----------------------------
# Example end-to-end training loop
# -----------------------------
def main():
    # ---- Load data ----
    edge_f, edge_l, edge_x, n, m = load_data()
    first_names, last_names = load_vocabularies()

    # ---- Train model ----
    print("\nTraining model...")
    params, (cf, cl) = train(
        edge_f,
        edge_l,
        edge_x,
        n,
        m,
        k=8,
        K_neg=5,
        batch_size=2**16,
        steps=10000,
        lr=1.0,
        reg=1e-4,
        eps=1e-8,
        seed=0,
        dtype=jnp.float32,  # Full precision
    )

    # ---- Save model ----
    print("\nSaving model...")
    np.save("model_U.npy", np.array(params.U))
    np.save("model_V.npy", np.array(params.V))
    np.save("model_a.npy", np.array(params.a))
    np.save("model_b.npy", np.array(params.b))
    print("Model saved.")

    return params, first_names, last_names


def train(
    edge_f,
    edge_l,
    edge_x,
    n,
    m,
    k=32,
    K_neg=5,
    batch_size=65536,
    steps=2000,
    lr=0.1,
    reg=1e-4,
    eps=1e-8,
    seed=0,
    dtype=jnp.bfloat16,
):
    """Train on a fixed number of steps using sparse AdaGrad. For ~1 epoch, steps ≈ nnz / batch_size.

    Args:
        dtype: Parameter dtype. Use bfloat16 (default) for half memory, float32 for full precision.
    """
    nnz = int(edge_f.shape[0])

    # ---- Marginals for qF, qL ----
    # Use float64 accumulation for stability, then cast down.
    cf = np.bincount(
        edge_f.astype(np.int64), weights=edge_x.astype(np.float64), minlength=n
    )
    cl = np.bincount(
        edge_l.astype(np.int64), weights=edge_x.astype(np.float64), minlength=m
    )

    # Negative sampling distributions (power 0.75)
    qf = np.power(cf + 1e-12, 0.75)
    ql = np.power(cl + 1e-12, 0.75)
    qf = qf / qf.sum()
    ql = ql / ql.sum()

    prob_f_np, alias_f_np = build_alias_table(qf)
    prob_l_np, alias_l_np = build_alias_table(ql)

    # Move sampler tables + log q to device (keep float32 for sampling/numerical stability)
    prob_f = jnp.asarray(prob_f_np, dtype=jnp.float32)
    alias_f = jnp.asarray(alias_f_np)
    prob_l = jnp.asarray(prob_l_np, dtype=jnp.float32)
    alias_l = jnp.asarray(alias_l_np)
    log_qf = jnp.log(jnp.asarray(qf, dtype=jnp.float32) + 1e-30)
    log_ql = jnp.log(jnp.asarray(ql, dtype=jnp.float32) + 1e-30)

    # ---- Initialize parameters ----
    key = random.PRNGKey(seed)
    key, k1, k2 = random.split(key, 3)

    # Bias initialization from marginals helps global comparability.
    a0 = np.log1p(cf).astype(np.float32)
    b0 = np.log1p(cl).astype(np.float32)
    # Optional centering (doesn't change ranking if applied consistently, can help numerics)
    a0 = a0 - a0.mean()
    b0 = b0 - b0.mean()

    U0 = 0.01 * random.normal(k1, (n, k), dtype=dtype)
    V0 = 0.01 * random.normal(k2, (m, k), dtype=dtype)
    params = Params(
        U=U0, V=V0, a=jnp.asarray(a0, dtype=dtype), b=jnp.asarray(b0, dtype=dtype)
    )

    # Initialize AdaGrad state (zeros) - use same dtype as params
    adagrad_state = AdaGradState(
        G_U=jnp.zeros((n, k), dtype=dtype),
        G_V=jnp.zeros((m, k), dtype=dtype),
        G_a=jnp.zeros(n, dtype=dtype),
        G_b=jnp.zeros(m, dtype=dtype),
    )

    # ---- Batch sampler (uniform over nnz; weight by log1p(count) in loss) ----
    rng_np = np.random.default_rng(seed + 123)

    def sample_batch():
        idx = rng_np.integers(0, nnz, size=batch_size, dtype=np.int64)
        f = edge_f[idx].astype(np.int32)
        l = edge_l[idx].astype(np.int32)
        x = edge_x[idx]
        # Convert weights to dtype for consistency
        w = np.log1p(x.astype(np.float32))
        return f, l, w

    # ---- Training loop ----
    t0 = time.time()
    dtype_name = "bfloat16" if dtype == jnp.bfloat16 else "float32"
    pbar = tqdm(range(steps), desc=f"Training (AdaGrad, {dtype_name})", ncols=100)
    for step in pbar:
        f_b, l_b, w_b = sample_batch()
        f_b = jnp.asarray(f_b)
        l_b = jnp.asarray(l_b)
        w_b = jnp.asarray(w_b, dtype=dtype)

        params, adagrad_state, key, loss = train_step(
            params,
            adagrad_state,
            key,
            f_b,
            l_b,
            w_b,
            prob_f,
            alias_f,
            prob_l,
            alias_l,
            log_qf,
            log_ql,
            lr=lr,
            reg=reg,
            eps=eps,
            K_neg=K_neg,
        )

        # Update progress bar with loss
        pbar.set_postfix({"loss": f"{float(loss):.4f}"})

    return params, (cf, cl)


# -----------------------------
# Ranking 50 candidate full names
# -----------------------------
def rank_50(params: Params, candidate_f_ids: np.ndarray, candidate_l_ids: np.ndarray):
    f = jnp.asarray(candidate_f_ids.astype(np.int32))
    l = jnp.asarray(candidate_l_ids.astype(np.int32))
    s = score_pairs(params, f, l)  # higher = more likely
    order = jnp.argsort(-s)
    return np.array(order), np.array(s)


# -----------------------------
# Anagram finding utilities
# -----------------------------
def load_model(
    u_path="model_U.npy",
    v_path="model_V.npy",
    a_path="model_a.npy",
    b_path="model_b.npy",
):
    """Load a trained model from .npy files."""
    U = jnp.asarray(np.load(u_path))
    V = jnp.asarray(np.load(v_path))
    a = jnp.asarray(np.load(a_path))
    b = jnp.asarray(np.load(b_path))
    return Params(U=U, V=V, a=a, b=b)


def find_anagram_candidates(query, first_names, last_names):
    """Find all first+last name pairs that are anagrams of the query.

    Args:
        query: Target string to find anagrams for
        first_names: List of first names (normalized, uppercase)
        last_names: List of last names (normalized, uppercase)

    Returns:
        List of (first_idx, last_idx, firstname, lastname) tuples
    """
    from collections import Counter, defaultdict

    # Normalize query
    query_normalized = query.upper().replace(" ", "").replace("-", "").replace("'", "")
    query_count = Counter(query_normalized)

    # Build last name index
    last_index = defaultdict(list)
    for idx, lastname in enumerate(last_names):
        key = "".join(sorted(lastname))
        last_index[key].append(idx)

    # Find candidates
    candidates = []
    for first_idx, firstname in enumerate(first_names):
        first_count = Counter(firstname)
        if not first_count <= query_count:
            continue

        remaining = query_count - first_count
        remaining_key = "".join(sorted(remaining.elements()))

        for last_idx in last_index[remaining_key]:
            lastname = last_names[last_idx]
            candidates.append((first_idx, last_idx, firstname, lastname))

    return candidates


def rank_anagrams(query, first_names, last_names, params, top_k=50):
    """Find and rank anagram candidates using the trained model.

    Args:
        query: Target string to find anagrams for
        first_names: List of first names (normalized)
        last_names: List of last names (normalized)
        params: Trained model parameters
        top_k: Number of top results to return

    Returns:
        List of (score, firstname, lastname) tuples, sorted by score (descending)
    """
    candidates = find_anagram_candidates(query, first_names, last_names)

    if not candidates:
        return []

    # Extract indices
    first_ids = np.array([c[0] for c in candidates], dtype=np.int32)
    last_ids = np.array([c[1] for c in candidates], dtype=np.int32)

    # Score all candidates
    scores = score_pairs(params, jnp.asarray(first_ids), jnp.asarray(last_ids))
    scores = np.array(scores)

    # Sort by score
    order = np.argsort(-scores)[:top_k]

    results = []
    for idx in order:
        first_idx, last_idx, firstname, lastname = candidates[idx]
        score = scores[idx]
        results.append((float(score), firstname, lastname))

    return results


# -----------------------------
# Usage notes (non-code)
# -----------------------------
"""
To train in < a few hours on your laptop:

- Start with: k=32, K_neg=5, batch_size=65536.
- One "epoch" worth of positives is steps ≈ nnz / batch_size ≈ 14M / 65536 ≈ 214 steps.
- Try 1-2 epochs first; then increase if you still see improvement.

Candidate ranking (50 items) is trivial: just compute s(f,l)=a_f+b_l+u_f^T v_l and sort.

Apple GPU:
- If you have JAX Metal working, this training step is dominated by embedding gathers and dot products,
  which should benefit from GPU; the scatter updates can be the limiter.
- Keep dtype float32 initially.

Data format:
- edge_f, edge_l: uint32 (or int32)
- edge_x: uint32 or float32
- Loaded from .npy files created by prepare_data.py

Example usage:
    # Train
    params, first_names, last_names = main()

    # Or load a trained model
    params = load_model()
    first_names, last_names = load_vocabularies()

    # Find anagrams
    results = rank_anagrams("AMBRE ALISON", first_names, last_names, params, top_k=10)
    for score, firstname, lastname in results:
        print(f"{firstname.title()} {lastname.title()} (score: {score:.4f})")
"""


if __name__ == "__main__":
    main()
    params = load_model()
    first_names, last_names = load_vocabularies()

    # Find anagrams
    results = rank_anagrams("louis abraham", first_names, last_names, params, top_k=200)
    for score, firstname, lastname in results:
        print(f"{firstname.title()} {lastname.title()} (score: {score:.4f})")
