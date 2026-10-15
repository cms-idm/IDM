"""Overlay fit efficiency vs gen Lxy from two plot_dimuon_vertex.py summaries (e.g. old vs new fitter bounds).

  python3 compare_dimuon_vertex_runs.py OLD.json NEW.json OUTDIR [--labels old,new] [--xvar LxyFull] [--log]

Both summaries must come from the same inputs and carry the x-variable (LxyFull from the 740 cm
plot script on). Per group (overall, per mass, per lifetime, per sample), one PNG: fit efficiency
(valid / both matched, per event, 68.27% Clopper-Pearson) per combination, old open grey squares vs new
filled red circles, and
the both-accepted denominator per bin. Linear axes; --log adds a copy with a log y on the count panel.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_dimuon_vertex import clopper_pearson

COMBOS = ["HIER", "GG", "GO", "OO", "II", "IG", "IO"]


def series(entry, c):
    sl = slice(1, len(entry["edges"]) - 2)  # drop underflow and overflow
    m = np.array(entry["combos"][c]["matched"][sl])
    v = np.array(entry["combos"][c]["validGivenMatched"][sl])
    return m, v


def plot(group, old, new, labels, xvar, outdir, log):
    eo, en = old["x"][xvar], new["x"][xvar]
    if eo["edges"] != en["edges"]:
        raise RuntimeError(f"{group}: {xvar} binning differs between the summaries")
    edges = np.array([float(e) for e in en["edges"][1:-1]])
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axs = plt.subplots(2, 4, figsize=(20, 9))
    for ax, c in zip(axs.flat, COMBOS):
        if c not in eo["combos"] or c not in en["combos"]:
            ax.set_visible(False)
            continue
        for e, mk, col, lab in ((eo, "s", "#7f7f7f", labels[0]), (en, "o", "#d62728", labels[1])):
            m, v = series(e, c)
            eff, lo, hi = clopper_pearson(v, m)
            ok = m > 0
            ax.errorbar(centers[ok], eff[ok], yerr=[eff[ok] - lo[ok], hi[ok] - eff[ok]],
                        xerr=0.5 * np.diff(edges)[ok], fmt=mk, ms=4, color=col,
                        mfc="none" if mk == "s" else col,
                        label=f"{lab}: {int(v.sum())} valid / {int(m.sum())} matched")
        ax.set_title(c, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("fit efficiency (valid / both matched)")
        ax.legend(fontsize=7)
    ax = axs.flat[-1]
    sl = slice(1, len(en["edges"]) - 2)
    for e, ls, col, lab in ((eo, "--", "#7f7f7f", labels[0]), (en, "-", "#d62728", labels[1])):
        ax.stairs(np.array(e["bothAccepted"][sl]), edges, color=col, ls=ls, label=lab)
    ax.set_ylabel("events with both signal muons accepted per bin")
    ax.set_yscale("log" if log else "linear")
    ax.legend(fontsize=7)
    for ax in axs.flat:
        ax.set_xlabel("gen L_xy [cm]")
        ax.set_xlim(edges[0], edges[-1])
        ax.grid(alpha=0.3)
    fig.suptitle(f"{group} | fit efficiency per event, {labels[0]} (open squares) vs {labels[1]} (filled circles) | "
                 "mixed combos: either ordering | unweighted", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    d = os.path.join(outdir, group)
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, f"compare_fitEff_{xvar}{'_log' if log else ''}.png"), dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("outdir")
    ap.add_argument("--labels", default="old bounds (112 / 273.5 cm),new bounds (740 / 960 cm)")
    ap.add_argument("--xvar", default="LxyFull")
    ap.add_argument("--log", action="store_true", help="also write log-scale copies (*_log.png)")
    args = ap.parse_args()
    with open(args.old) as f:
        old = json.load(f)
    with open(args.new) as f:
        new = json.load(f)
    if old["samples"] != new["samples"]:
        raise RuntimeError("the summaries cover different samples")
    labels = args.labels.split(",")
    for group in new["groups"]:
        if group not in old["groups"] or args.xvar not in old["groups"][group]["x"]:
            raise RuntimeError(f"{group}: {args.xvar} missing from {args.old}")
        for log in ([False, True] if args.log else [False]):
            plot(group, old["groups"][group], new["groups"][group], labels, args.xvar, args.outdir, log)
        print(group, flush=True)


if __name__ == "__main__":
    main()
