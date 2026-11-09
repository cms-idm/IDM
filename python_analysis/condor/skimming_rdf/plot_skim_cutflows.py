#!/usr/bin/env python3

import os
import re
import glob
import argparse
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CUT_ORDER = [
    "initial",
    "after good vertex disabled",
    "after MET filters",
    "after HEM veto",
    "after MET trigger",
    "after PFMET > 200.0",
    "after nJet cut / final",
]

CUT_LABELS = {
    "initial": "Initial",
    "after good vertex disabled": "Good vertex\ndisabled",
    "after MET filters": "MET\nfilters",
    "after HEM veto": "HEM\nveto",
    "after MET trigger": "MET\ntrigger",
    "after PFMET > 200.0": r"$p_T^{miss}>200$",
    "after nJet cut / final": r"$N_{\mathrm{PFJet}}>0$",
}

SHORT_SAMPLE_LABELS = {
    "sig_Mchi-5p25_dMchi-0p5_ct-1": r"$m_\chi=5.25,\ \Delta m=0.5,\ c\tau=1$",
    "sig_Mchi-5p25_dMchi-0p5_ct-100": r"$m_\chi=5.25,\ \Delta m=0.5,\ c\tau=100$",
    "sig_Mchi-5p25_dMchi-0p5_ct-1000": r"$m_\chi=5.25,\ \Delta m=0.5,\ c\tau=1000$",
    "sig_Mchi-42p0_dMchi-4p0_ct-1": r"$m_\chi=42,\ \Delta m=4,\ c\tau=1$",
    "sig_Mchi-42p0_dMchi-4p0_ct-100": r"$m_\chi=42,\ \Delta m=4,\ c\tau=100$",
    "sig_Mchi-42p0_dMchi-4p0_ct-1000": r"$m_\chi=42,\ \Delta m=4,\ c\tau=1000$",
}


def safe_sample_label(sample):
    return SHORT_SAMPLE_LABELS.get(sample, sample.replace("_", r"\_"))


def sanitize_filename(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def discover_log_files(paths):
    files = []

    for path in paths:
        if os.path.isfile(path):
            files.append(path)
            continue

        if os.path.isdir(path):
            patterns = [
                os.path.join(path, "*.out"),
                os.path.join(path, "*.stdout"),
                os.path.join(path, "*.log"),
                os.path.join(path, "*.txt"),
                os.path.join(path, "*"),
            ]

            for pat in patterns:
                files.extend(glob.glob(pat))

    files = sorted(set(f for f in files if os.path.isfile(f)))

    return files


def normalize_sample_from_jobname(jobname):
    """
    Example:
      ntuples_sig_Mchi-5.25_dMchi-0.5_ct-1000_1
    becomes:
      sig_Mchi-5p25_dMchi-0p5_ct-1000

    Also handles:
      ntuples_sig_Mchi-42.0_dMchi-4.0_ct-1_0
    """
    j = jobname.strip()

    j = re.sub(r"^ntuples_", "", j)

    # Drop final job index suffix: _0, _1, ...
    j = re.sub(r"_[0-9]+$", "", j)

    # Convert decimal mass tokens to p notation.
    j = re.sub(r"Mchi-([0-9]+)\.([0-9]+)", r"Mchi-\1p\2", j)
    j = re.sub(r"dMchi-([0-9]+)\.([0-9]+)", r"dMchi-\1p\2", j)

    return j


def parse_one_log(path):
    with open(path, "r", errors="ignore") as f:
        text = f.read()

    if "Sequential cutflow" not in text:
        return None

    job_match = re.search(r"jobname\s*=\s*(\S+)", text)

    if not job_match:
        return None

    jobname = job_match.group(1)
    sample = normalize_sample_from_jobname(jobname)

    counts = OrderedDict()

    for cut in CUT_ORDER:
        # Example:
        # after MET trigger                             5096    step_eff = 0.254800
        pat = re.compile(rf"^{re.escape(cut)}\s+([0-9]+)", re.MULTILINE)
        m = pat.search(text)

        if m:
            counts[cut] = int(m.group(1))

    if not counts:
        return None

    final_match = re.search(r"^final\s*=\s*([0-9]+)", text, re.MULTILINE)
    final = int(final_match.group(1)) if final_match else counts.get("after nJet cut / final", None)

    return {
        "path": path,
        "jobname": jobname,
        "sample": sample,
        "counts": counts,
        "final": final,
    }


def aggregate_cutflows(records):
    agg = defaultdict(lambda: defaultdict(int))
    njobs = defaultdict(int)

    for rec in records:
        sample = rec["sample"]
        njobs[sample] += 1

        for cut, count in rec["counts"].items():
            agg[sample][cut] += count

    rows = []

    for sample in sorted(agg.keys()):
        row = {"sample": sample, "n_jobs": njobs[sample]}

        for cut in CUT_ORDER:
            row[cut] = agg[sample].get(cut, 0)

        initial = row["initial"]
        final = row["after nJet cut / final"]

        row["final_over_initial"] = final / initial if initial > 0 else np.nan
        row["met_trigger_eff"] = row["after MET trigger"] / row["after HEM veto"] if row["after HEM veto"] > 0 else np.nan
        row["met200_eff_after_trigger"] = row["after PFMET > 200.0"] / row["after MET trigger"] if row["after MET trigger"] > 0 else np.nan
        row["njet_eff_after_met200"] = final / row["after PFMET > 200.0"] if row["after PFMET > 200.0"] > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def select_samples(df, requested_samples, max_samples):
    available = list(df["sample"])

    if requested_samples:
        selected = []

        for req in requested_samples:
            matches = [s for s in available if req in s]

            if not matches:
                print(f"[WARN] requested sample pattern not found: {req}")
                continue

            selected.append(matches[0])

        return selected

    # Default: representative low/high mass and short/long ctau if available.
    preferred_patterns = [
        "Mchi-5p25_dMchi-0p5_ct-1",
        "Mchi-5p25_dMchi-0p5_ct-1000",
        "Mchi-42p0_dMchi-4p0_ct-1",
        "Mchi-42p0_dMchi-4p0_ct-1000",
    ]

    selected = []

    for pat in preferred_patterns:
        matches = [s for s in available if pat in s]
        if matches:
            selected.append(matches[0])

    if len(selected) == 0:
        selected = available[:max_samples]

    return selected[:max_samples]


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.savefig(path.replace(".png", ".pdf"))
    print(f"Saved: {path}")
    print(f"Saved: {path.replace('.png', '.pdf')}")


def plot_cutflow_counts(df, selected, outdir):
    for sample in selected:
        row = df[df["sample"] == sample].iloc[0]

        counts = np.array([row[c] for c in CUT_ORDER], dtype=float)
        x = np.arange(len(CUT_ORDER))

        plt.figure(figsize=(10, 6))
        plt.bar(x, counts)

        plt.xticks(x, [CUT_LABELS[c] for c in CUT_ORDER], rotation=0)
        plt.ylabel("Events")
        plt.title(f"RDF skim cutflow\n{safe_sample_label(sample)}")

        ymax = max(counts) if len(counts) else 1.0
        plt.ylim(0, 1.25 * ymax)

        for i, val in enumerate(counts):
            plt.text(i, val + 0.02 * ymax, f"{int(val)}", ha="center", va="bottom", fontsize=9)

        out = os.path.join(outdir, f"cutflow_counts_{sanitize_filename(sample)}.png")
        savefig(out)


def plot_cutflow_efficiencies(df, selected, outdir):
    for sample in selected:
        row = df[df["sample"] == sample].iloc[0]

        counts = np.array([row[c] for c in CUT_ORDER], dtype=float)
        initial = counts[0]

        eff = counts / initial if initial > 0 else np.zeros_like(counts)
        x = np.arange(len(CUT_ORDER))

        plt.figure(figsize=(10, 6))
        plt.bar(x, eff)

        plt.xticks(x, [CUT_LABELS[c] for c in CUT_ORDER], rotation=0)
        plt.ylabel("Cumulative efficiency")
        plt.title(f"RDF skim cumulative efficiency\n{safe_sample_label(sample)}")
        plt.ylim(0, 1.15)

        for i, val in enumerate(eff):
            plt.text(i, val + 0.025, f"{100.0*val:.1f}%", ha="center", va="bottom", fontsize=9)

        out = os.path.join(outdir, f"cutflow_cumulative_eff_{sanitize_filename(sample)}.png")
        savefig(out)


def plot_combined_final_eff(df, selected, outdir):
    labels = [safe_sample_label(s) for s in selected]
    final_eff = []
    njet_eff = []
    met_trigger_eff = []
    met200_eff_after_trigger = []

    for sample in selected:
        row = df[df["sample"] == sample].iloc[0]
        final_eff.append(row["final_over_initial"])
        njet_eff.append(row["njet_eff_after_met200"])
        met_trigger_eff.append(row["met_trigger_eff"])
        met200_eff_after_trigger.append(row["met200_eff_after_trigger"])

    x = np.arange(len(selected))
    width = 0.22

    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5 * width, met_trigger_eff, width, label="MET trigger / HEM")
    plt.bar(x - 0.5 * width, met200_eff_after_trigger, width, label="MET200 / trigger")
    plt.bar(x + 0.5 * width, njet_eff, width, label="nJet / MET200")
    plt.bar(x + 1.5 * width, final_eff, width, label="Final / initial")

    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Efficiency")
    plt.ylim(0, 1.15)
    plt.title("RDF skim efficiencies for selected signal points")
    plt.legend(frameon=False)

    out = os.path.join(outdir, "skim_efficiency_summary_selected_points.png")
    savefig(out)


def plot_combined_cutflow_lines(df, selected, outdir):
    x = np.arange(len(CUT_ORDER))

    plt.figure(figsize=(11, 6))

    for sample in selected:
        row = df[df["sample"] == sample].iloc[0]
        counts = np.array([row[c] for c in CUT_ORDER], dtype=float)
        initial = counts[0]
        eff = counts / initial if initial > 0 else np.zeros_like(counts)

        plt.plot(x, eff, marker="o", linewidth=2, label=safe_sample_label(sample))

    plt.xticks(x, [CUT_LABELS[c] for c in CUT_ORDER], rotation=0)
    plt.ylabel("Cumulative efficiency")
    plt.ylim(0, 1.1)
    plt.title("RDF skim cumulative efficiency comparison")
    plt.legend(frameon=False, fontsize=9)

    out = os.path.join(outdir, "cutflow_cumulative_eff_comparison.png")
    savefig(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dirs",
        nargs="+",
        default=["."],
        help="Directories/files to scan for skimmer logs",
    )
    parser.add_argument(
        "--outdir",
        default="plots_skim_cutflows_meeting",
        help="Output directory",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=[],
        help="Optional sample substrings to plot, e.g. Mchi-5p25_dMchi-0p5_ct-1000",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4,
        help="Maximum number of samples to plot if --samples is not provided",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    files = discover_log_files(args.log_dirs)
    print(f"Found {len(files)} candidate log files")

    records = []

    for f in files:
        rec = parse_one_log(f)
        if rec is not None:
            records.append(rec)

    print(f"Parsed {len(records)} skimmer job logs")

    if len(records) == 0:
        raise RuntimeError("No valid skimmer cutflow logs found.")

    df = aggregate_cutflows(records)

    csv_path = os.path.join(args.outdir, "skim_cutflow_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    selected = select_samples(df, args.samples, args.max_samples)

    print("Selected samples:")
    for s in selected:
        print(f"  {s}")

    selected_csv = os.path.join(args.outdir, "skim_cutflow_summary_selected.csv")
    df[df["sample"].isin(selected)].to_csv(selected_csv, index=False)
    print(f"Saved selected CSV: {selected_csv}")

    plot_cutflow_counts(df, selected, args.outdir)
    plot_cutflow_efficiencies(df, selected, args.outdir)
    plot_combined_final_eff(df, selected, args.outdir)
    plot_combined_cutflow_lines(df, selected, args.outdir)

    print("=" * 100)
    print("Summary for selected samples")
    print("=" * 100)

    cols = [
        "sample",
        "n_jobs",
        "initial",
        "after MET trigger",
        "after PFMET > 200.0",
        "after nJet cut / final",
        "final_over_initial",
        "njet_eff_after_met200",
    ]

    print(df[df["sample"].isin(selected)][cols].to_string(index=False))

    print("=" * 100)
    print(f"Plots written to: {args.outdir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
