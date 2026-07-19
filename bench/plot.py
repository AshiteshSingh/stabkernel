"""Render stabkernel benchmark charts from bench/results.csv.

Writes SEVERAL PNGs, each with its own name, into bench/:
  throughput_gf2_xor.png     throughput vs row size for gf2_xor
  throughput_gf2_weight.png  throughput vs row size for gf2_weight
  throughput_gf2_inner.png   throughput vs row size for gf2_inner
  compute_bound_bars.png     per-kernel bars at a cache-resident size
  roofline_gf2_weight.png    gf2_weight vs size (memory-bound convergence)
  speedup_vs_scalar.png      SIMD speedup over the scalar baseline
  overview.png               combined 2-panel summary

Robust to whichever backends are present in the CSV (scalar / avx2 / avx512 /
neon), so it works the same on an AVX2 laptop or an AVX-512 server.

Usage:  python3 bench/plot.py            # reads bench/results.csv
        python3 bench/plot.py my.csv     # reads a specific CSV
"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results.csv")

rows = list(csv.DictReader(open(csv_path)))
if not rows:
    sys.exit("no rows in " + csv_path)
for r in rows:
    r["words"] = int(r["words"])
    r["gbps"]  = float(r["gbps"])

BACK_ORDER = ["scalar", "avx2", "avx512", "neon"]
KERN_ORDER = ["gf2_xor", "gf2_weight", "gf2_inner"]
colors = {"scalar": "#9aa0a6", "avx2": "#4285f4",
          "avx512": "#ea4335", "neon": "#34a853"}

backends = [b for b in BACK_ORDER if any(r["backend"] == b for r in rows)]
kernels  = [k for k in KERN_ORDER if any(r["kernel"]  == k for r in rows)]
sizes    = sorted({r["words"] for r in rows})

def gbps(kernel, backend, words):
    for r in rows:
        if r["kernel"] == kernel and r["backend"] == backend and r["words"] == words:
            return r["gbps"]
    return None

def kib(words):
    return words * 8 / 1024.0

written = []
def save(fig, name):
    path = os.path.join(here, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append(name)
    print("wrote", path)

# ---- 1) One throughput-vs-size line chart per kernel --------------------
for k in kernels:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for b in backends:
        pts = [(kib(w), gbps(k, b, w)) for w in sizes if gbps(k, b, w) is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", label=b, color=colors.get(b))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("row size (KiB, log scale)")
    ax.set_ylabel("throughput (GB/s, higher = better)")
    ax.set_title(f"stabkernel: {k} throughput vs working-set size")
    ax.legend(title="backend"); ax.grid(alpha=0.3)
    save(fig, f"throughput_{k}.png")

# ---- 2) Compute-bound bar chart at a cache-resident size ---------------
target = 512 if 512 in sizes else min(sizes)
sub_sizes = [target]
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(kernels)); w = 0.8 / max(1, len(backends))
for i, b in enumerate(backends):
    vals = [gbps(k, b, target) or 0.0 for k in kernels]
    off = (i - (len(backends) - 1) / 2) * w
    bars = ax.bar(x + off, vals, w, label=b, color=colors.get(b))
    for rect, v in zip(bars, vals):
        if v:
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(kernels)
ax.set_ylabel("throughput (GB/s)")
ax.set_title(f"Compute-bound ({kib(target):.0f} KiB rows, cache-resident)\n"
             "hand-tuned SIMD vs scalar")
ax.legend(title="backend"); ax.grid(axis="y", alpha=0.3)
save(fig, "compute_bound_bars.png")

# ---- 3) Roofline: gf2_weight vs size ----------------------------------
if "gf2_weight" in kernels:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for b in backends:
        pts = [(kib(w), gbps("gf2_weight", b, w)) for w in sizes
               if gbps("gf2_weight", b, w) is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", label=b, color=colors.get(b))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("row size (KiB, log scale)")
    ax.set_ylabel("gf2_weight throughput (GB/s)")
    ax.set_title("Memory-bound regime: backends converge (the roofline wall)")
    ax.legend(title="backend"); ax.grid(alpha=0.3)
    save(fig, "roofline_gf2_weight.png")

# ---- 4) Speedup over scalar at the compute-bound size -----------------
if "scalar" in backends and len(backends) > 1:
    simd = [b for b in backends if b != "scalar"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(kernels)); w = 0.8 / max(1, len(simd))
    for i, b in enumerate(simd):
        vals = []
        for k in kernels:
            base = gbps(k, "scalar", target); cur = gbps(k, b, target)
            vals.append((cur / base) if (base and cur) else 0.0)
        off = (i - (len(simd) - 1) / 2) * w
        bars = ax.bar(x + off, vals, w, label=b, color=colors.get(b))
        for rect, v in zip(bars, vals):
            if v:
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.05,
                        f"{v:.1f}x", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="#333", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(kernels)
    ax.set_ylabel("speedup vs scalar (x)")
    ax.set_title(f"SIMD speedup over scalar ({kib(target):.0f} KiB rows)")
    ax.legend(title="backend"); ax.grid(axis="y", alpha=0.3)
    save(fig, "speedup_vs_scalar.png")

# ---- 5) Combined 2-panel overview -------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
x = np.arange(len(kernels)); w = 0.8 / max(1, len(backends))
for i, b in enumerate(backends):
    vals = [gbps(k, b, target) or 0.0 for k in kernels]
    off = (i - (len(backends) - 1) / 2) * w
    ax1.bar(x + off, vals, w, label=b, color=colors.get(b))
ax1.set_xticks(x); ax1.set_xticklabels(kernels)
ax1.set_ylabel("throughput (GB/s)")
ax1.set_title(f"Compute-bound ({kib(target):.0f} KiB rows)")
ax1.legend(title="backend"); ax1.grid(axis="y", alpha=0.3)
if "gf2_weight" in kernels:
    for b in backends:
        pts = [(kib(w2), gbps("gf2_weight", b, w2)) for w2 in sizes
               if gbps("gf2_weight", b, w2) is not None]
        if pts:
            xs, ys = zip(*pts)
            ax2.plot(xs, ys, "o-", label=b, color=colors.get(b))
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("row size (KiB, log scale)")
    ax2.set_ylabel("gf2_weight throughput (GB/s)")
    ax2.set_title("Memory-bound: roofline convergence")
    ax2.legend(title="backend"); ax2.grid(alpha=0.3)
fig.suptitle("stabkernel: GF(2) bit-linear-algebra kernels",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
save(fig, "overview.png")

print(f"\nDone: {len(written)} PNG(s) written to {here}")
for n in written:
    print("  -", n)
