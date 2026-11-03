# Build docs/world_model.pdf from the template docs/world_model.html and the results/ files.
# Run: uv run --no-project --with matplotlib --with numpy python docs/make_report.py
# The math is typeset by KaTeX (loaded from a CDN) and the page is printed by headless Chrome.
import json
import os
import shutil
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = "models/world_all_k16_wd0.1_u0_me20_s10000_gates_pool4096.npz"
CAL = "models/world_all_k16_wd0.1_u0_me20_s10000_gates_pool4096_cal.npz"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

INK, INK2, GRID = "#000000", "#444444", "#e2e2e2"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8d8c88"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": "#bbbbbb",
        "axes.linewidth": 0.6,
        "svg.fonttype": "path",
    }
)


def load(name):
    return json.load(open(f"{ROOT}/results/{name}.json"))


def gains(null, model):
    return {c: null["per_country"][c] - model["per_country"][c] for c in null["per_country"]}


def main():
    null, cal = load("world_cal")[:2]
    raw = load("world_pool3")[1]
    shared, gated = load("world")[1], load("world_gates")[1]
    quant = {e["name"]: e for e in load("quant_world_cal")}
    exp_null, exp = load("quant_world_cal_pq2")
    g_raw, cal_npz = np.load(f"{ROOT}/{RAW}")["G"], np.load(f"{ROOT}/{CAL}")
    beta = cal_npz["G"][:, 0] / g_raw[:, 0]

    total = lambda m: null["test_bits"] - m["test_bits"]
    below = lambda m: sum(v < 0 for v in gains(null, m).values())
    q_null, q_fp32 = quant["null (no embeddings)"], quant["float32"]
    loss = lambda name: quant[name]["test_bits"] - q_fp32["test_bits"]
    gain_cal = gains(null, cal)
    low = min(gain_cal, key=gain_cal.get)
    small = min((v for v in quant.values() if v["n_stored"]), key=lambda v: v["mb"])
    int8 = loss("int8 per column")
    num = {
        "gain_cal": f"{total(cal):.3f}",
        "null_bits": f"{null['test_bits']:.2f}",
        "gain_shared": f"{total(shared):+.3f}",
        "below_shared": below(shared),
        "gain_gated": f"{total(gated):+.3f}",
        "below_gated": below(gated),
        "gain_raw": f"{total(raw):+.3f}",
        "below_raw": below(raw),
        "below_cal": below(cal),
        "low_cal": f"{low} (+{gain_cal[low]:.3f})",
        "train_min": f"{raw['train_s'] / 60:.1f}",
        "beta_median": f"{np.median(beta):.2f}",
        "beta_lt1": int((beta < 1).sum()),
        "mb_fp32": f"{q_fp32['mb']:.0f}",
        "mb_pq2": f"{quant['PQ 2 dims/byte']['mb']:.0f}",
        "loss_int8": f"${int8 / 10 ** np.floor(np.log10(int8)):.1f} \\times 10^{{{int(np.floor(np.log10(int8)))}}}$",
        "loss_int4": f"{loss('int4 per column'):.3f}",
        "ratio_int": f"{loss('int4 per column') / int8:.0f}",
        "loss_pq2": f"{loss('PQ 2 dims/byte'):.4f}",
        "loss_pq4": f"{loss('PQ 4 dims/byte'):.3f}",
        "loss_pq8": f"{loss('PQ 8 dims/byte'):.3f}",
        "mb_top50": f"{quant['int4, top 50% of names']['mb']:.1f}",
        "loss_top50": f"{loss('int4, top 50% of names'):.3f}",
        "file_mb": f"{exp['file_mb']:.1f}",
        "gain_file": f"{exp_null['test_bits'] - exp['test_bits']:.3f}",
        "mb_small": f"{small['mb']:.1f}",
        "gain_small": f"{q_null['test_bits'] - small['test_bits']:.3f}",
    }
    assert exp["below_null"] == 0 and below(cal) == 0

    tmp = tempfile.mkdtemp()
    plot_countries(gains(null, raw), gain_cal, f"{tmp}/fig_countries.svg")
    plot_quant(quant, q_null, f"{tmp}/fig_quant.svg")
    html = open(f"{ROOT}/docs/world_model.html").read()
    for key, value in num.items():
        html = html.replace(f"@@{key}@@", str(value))
    assert "@@" not in html.split("-->", 1)[1], "a placeholder has no value"
    with open(f"{tmp}/world_model.html", "w") as fh:
        fh.write(html)
    out = f"{ROOT}/docs/world_model.pdf"
    subprocess.run(
        [
            CHROME,
            "--headless",
            "--no-pdf-header-footer",
            "--virtual-time-budget=20000",
            f"--print-to-pdf={out}",
            f"file://{tmp}/world_model.html",
        ],
        check=True,
        capture_output=True,
    )
    shutil.rmtree(tmp)
    print(f"saved {out}")


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_countries(gain_raw, gain_cal, path, n=25):
    order = sorted(gain_raw, key=gain_raw.get)[:n]
    fig, ax = plt.subplots(figsize=(6.6, 2.0))
    x = np.arange(n)
    r, c = np.array([gain_raw[k] for k in order]), np.array([gain_cal[k] for k in order])
    ax.vlines(x, r, c, color="#d0d0d0", lw=1.5, zorder=1)
    ax.scatter(x, r, s=22, facecolor="white", edgecolor=GRAY, lw=1.2, zorder=2, label="before calibration")
    ax.scatter(x, c, s=22, color=BLUE, edgecolor="white", lw=0.8, zorder=3, label="after calibration")
    ax.axhline(0, color=INK2, lw=0.7, zorder=0)
    ax.set_xticks(x, order, fontsize=7.5)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel("test gain (bits/person)")
    style(ax)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=8, handletextpad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def plot_quant(quant, q_null, path):
    groups = [
        (
            "scalar",
            BLUE,
            "o",
            [("float32", "fp32"), ("float16", "fp16"), ("int8 per column", "int8"), ("int4 per column", "int4")],
        ),
        ("product quantization", ORANGE, "s", [(f"PQ {g} dims/byte", f"$g={g}$") for g in (2, 4, 8)]),
        ("int4, top names", AQUA, "^", [(f"int4, top {p}% of names", f"top {p}%") for p in (50, 25, 10, 5)]),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    gain = lambda name: q_null["test_bits"] - quant[name]["test_bits"]
    ax.axhline(gain("float32"), color=INK2, lw=0.6, ls=(0, (4, 3)), zorder=0)
    points = []
    for label, color, marker, items in groups:
        mb, g = [quant[n]["mb"] for n, _ in items], [gain(n) for n, _ in items]
        ax.scatter(mb, g, s=24, color=color, marker=marker, edgecolor="white", lw=0.7, zorder=3, label=label)
        points += [(short, quant[n]["mb"], gain(n)) for n, short in items]
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlabel("size of the name vectors (MB, log scale)")
    ax.set_ylabel("test gain (bits/person)")
    mb, g = [p[1] for p in points], [p[2] for p in points]
    ax.set_xlim(min(mb) / 1.7, max(mb) * 2.0)
    ax.set_ylim(min(g) - 0.02, max(g) + 0.025)
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.legend(loc="lower right", frameon=False, fontsize=8, handletextpad=0.2)
    fig.tight_layout(pad=0.3)
    place_labels(fig, ax, points)
    fig.savefig(path)
    plt.close(fig)


def place_labels(fig, ax, points):
    """Label each point at the first offset where the text hits no other label, point or the legend."""
    r = fig.canvas.get_renderer()
    dots = [ax.transData.transform((x, y)) for _, x, y in points]
    taken = [ax.get_legend().get_window_extent(r)]
    frame = ax.get_window_extent(r)
    offsets = [
        (5, -3, "left"),
        (-5, -3, "right"),
        (4, 5, "left"),
        (4, -11, "left"),
        (-4, 5, "right"),
        (-4, -11, "right"),
        (0, 7, "center"),
        (0, -13, "center"),
    ]
    for name, x, y in points:
        for dx, dy, ha in offsets:
            t = ax.annotate(
                name,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                fontsize=7.5,
                color=INK2,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"),
            )
            box = t.get_window_extent(r).expanded(1.04, 1.1)
            hit = any(box.overlaps(b) for b in taken) or any(box.contains(*p) for p in dots)
            hit |= not (frame.contains(box.x0, box.y0) and frame.contains(box.x1, box.y1))
            if not hit:
                taken.append(box)
                break
            t.remove()
        else:
            raise RuntimeError(f"no free place for the label {name}")


if __name__ == "__main__":
    main()
