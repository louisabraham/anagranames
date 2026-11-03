# Train the original recipe (training.train() with the settings of training.main()) on the
# train split of one country, and save it for evaluation by pairmodel.py. Runs in .venv-jax.
import argparse
import os

import jax.numpy as jnp
import numpy as np

import namedata
import training

parser = argparse.ArgumentParser()
parser.add_argument("--country", default="FR")
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--out", default=None)
args = parser.parse_args()

ds = namedata.load(country=args.country)
print(namedata.describe(ds), flush=True)
keep = ds.n_train > 0
params, _ = training.train(
    ds.f[keep],
    ds.l[keep],
    ds.n_train[keep].astype(np.float32),
    ds.n_first,
    ds.n_last,
    k=8,
    K_neg=5,
    batch_size=2**16,
    steps=args.steps,
    lr=1.0,
    reg=1e-4,
    eps=1e-8,
    seed=0,
    dtype=jnp.float32,
)
out = args.out or f"models/original_{args.country}_k8.npz"
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out, a=np.asarray(params.a), b=np.asarray(params.b), U=np.asarray(params.U), V=np.asarray(params.V))
print(f"Saved {out}")
