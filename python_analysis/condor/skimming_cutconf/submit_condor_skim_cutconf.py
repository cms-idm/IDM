#!/usr/bin/env python3
"""
Submit condor jobs that skim ntuples down to the events passing a cut config up to
(and including) a given cut. Output is an exact copy of ntuples/outT (all branches)
for the passing events, so the Analyzer reads skims exactly like the original ntuples.

Output location: the ntuple version directory gets "_skimmed" appended and the rest
of the sample's path is kept, e.g.
    /store/group/lpcmetx/iDMe/Samples/Ntuples/signal_Jul2026noID/2024/M1-.../ctau-1/
 -> /store/group/lpcmetx/iDMe/Samples/Ntuples/signal_Jul2026noID_skimmed/2024/M1-.../ctau-1/
With the default -n -1 every sample is one job and one output file
(<input file name minus its counter>_skimmed.root). With -n N, samples with more than N files are split
into jobs whose outputs go to <out dir>/parts/ (invisible to the Analyzer, which only
lists the top-level .root files); merge_skims.py then combines them into <out dir>/.

A sample json pointing at the skimmed locations is written next to the input json
(<name>_skimmed.json). It keeps the ORIGINAL sum_wgt/num_events, since the event weights
must be normalized to the full sample, not to the skimmed subset.

Usage (from this directory, with a valid grid proxy):
  python3 submit_condor_skim_cutconf.py -s ../../configs/sample_configs/signal_2024_Jul2026noID_0p1.json \\
        -c ../../configs/cut_configs/an_selection.py -k cut8
  options: -n files per job, -p sample names, --dry-run, --force, --memory/--disk-gb
  resubmit jobs whose output is missing:
  python3 submit_condor_skim_cutconf.py --resubmit submissions_skim_cutconf/<jobset>
"""
import os
import re
import sys
import json
import math
import shutil
import tarfile
import argparse
import subprocess

from XRootD import client

EOS = "root://cmseos.fnal.gov"
HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_TOOLS = os.path.normpath(os.path.join(HERE, "../../analysisTools"))
xrd = client.FileSystem(EOS)


def eos_ls(path):
    """Names of the entries in an EOS directory ([] if it does not exist)."""
    status, listing = xrd.dirlist(path)
    return [e.name for e in listing] if status.ok else []


def list_files(sample):
    """Full xrootd URLs of the sample's input files, minus the blacklist."""
    blacklist = set(sample.get("blacklist", []))
    if "fileset" in sample:
        files = sample["fileset"]
    else:
        locs = sample["location"] if isinstance(sample["location"], list) else [sample["location"]]
        files = []
        for loc in locs:
            loc = re.sub(r"/+", "/", loc).rstrip("/")
            if loc.endswith(".root"):
                files.append(loc)
            else:
                files += [f"{loc}/{n}" for n in sorted(eos_ls(loc)) if n.endswith(".root")]
    files = [f for f in files if os.path.basename(f) not in blacklist]
    return [f if f.startswith("root://") else f"{EOS}/{f}" for f in files]


def skimmed_dir(sample):
    """Append _skimmed to the version directory right after .../Ntuples/, keep the rest."""
    loc = sample["location"][0] if isinstance(sample["location"], list) else sample["location"]
    loc = re.sub(r"/+", "/", loc).rstrip("/")
    if loc.endswith(".root"):
        loc = os.path.dirname(loc)
    m = re.match(r"(.*/Ntuples/)([^/]+)(/.*)?$", loc)
    if m is None:
        raise ValueError(f"can't find the version directory (.../Ntuples/<vers>/...) in {loc}")
    return f"{m.group(1)}{m.group(2)}_skimmed{m.group(3) or ''}"


def output_stem(files, fallback):
    """Common name of a sample's input files without the per-file counter, e.g.
    ntuples_M1-1_dM-0p05_mZD-3_ctau-1_flist_03.root -> ntuples_M1-1_dM-0p05_mZD-3_ctau-1
    DY_EE_merged_000.root                          -> DY_EE_merged
    Falls back to the sample name if the input files don't share one."""
    stems = {re.sub(r"(_flist)?_\d+$", "", os.path.basename(f)[:-len(".root")]) for f in files}
    return stems.pop() if len(stems) == 1 else fallback


def cut_names(cut_config):
    with open(cut_config) as f:
        return re.findall(r"^def (cut\d+)\s*\(", f.read(), flags=re.M)


def make_code_tarball(path, cut_config):
    """analysisTools/*.py (cut configs import from it), the cut config, and the skimmer."""
    with tarfile.open(path, "w:gz") as tar:
        for f in sorted(os.listdir(ANALYSIS_TOOLS)):
            if f.endswith(".py") and not f.startswith(("#", ".#")):
                tar.add(os.path.join(ANALYSIS_TOOLS, f), arcname=f"analysisTools/{f}")
        tar.add(cut_config, arcname=os.path.basename(cut_config))
        tar.add(os.path.join(HERE, "condor_skim_cutconf.py"), arcname="condor_skim_cutconf.py")


def condor_submit(jobset_dir, queue_file, args):
    cmd = ["condor_submit", "condor_skim_cutconf.jdl",
           "-batch-name", os.path.basename(os.path.normpath(jobset_dir)),
           "-append", f"request_memory = {args.memory}",
           "-append", f"request_disk = {int(args.disk_gb * 1024 ** 2)}"]
    if queue_file != "jobs.txt":
        # the jdl ends in "queue jobname from jobs.txt"; point it at another list instead
        with open(os.path.join(jobset_dir, "condor_skim_cutconf.jdl")) as f:
            jdl = f.read().replace("from jobs.txt", f"from {queue_file}")
        with open(os.path.join(jobset_dir, "condor_skim_cutconf_resubmit.jdl"), "w") as f:
            f.write(jdl)
        cmd[1] = "condor_skim_cutconf_resubmit.jdl"
    print(" ".join(cmd))
    # the LPC condor_submit is a shebang-less shell wrapper, which exec() rejects; run it via sh
    cmd = ["/bin/sh", shutil.which("condor_submit")] + cmd[1:]
    subprocess.run(cmd, cwd=jobset_dir, check=True)


def resubmit(args):
    """Resubmit the jobs of a job set whose output file is not on EOS."""
    with open(os.path.join(args.resubmit, "manifest.json")) as f:
        manifest = json.load(f)
    missing = []
    for samp in manifest["samples"].values():
        for job in samp["jobs"]:
            if job["out_file"] not in eos_ls(job["out_dir"]):
                missing.append(job["jobname"])
    print(f"{len(missing)} job(s) with missing output")
    if not missing:
        return
    for j in missing:
        print("  ", j)
    if args.dry_run:
        return
    with open(os.path.join(args.resubmit, "jobs_resubmit.txt"), "w") as f:
        f.write("\n".join(missing) + "\n")
    condor_submit(args.resubmit, "jobs_resubmit.txt", args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--samples", help="sample json (signal, bkg or data -- one type per json)")
    parser.add_argument("-c", "--cut-config", help="cut config .py, e.g. configs/cut_configs/an_selection.py")
    parser.add_argument("-k", "--cut", help="last cut to apply, e.g. cut8")
    parser.add_argument("-n", "--files-per-job", type=int, default=-1,
                        help="input files per job (-1: one job per sample, one output file)")
    parser.add_argument("-p", "--particular", nargs="+", default=None, help="only these sample names")
    parser.add_argument("--chunksize", type=int, default=50000, help="events per coffea chunk in the worker")
    parser.add_argument("--memory", type=int, default=4000, help="request_memory (MB)")
    parser.add_argument("--disk-gb", type=float, default=4, help="request_disk (GB)")
    parser.add_argument("--force", action="store_true", help="write into output dirs that already hold .root files")
    parser.add_argument("--dry-run", action="store_true", help="build everything but don't mkdir on EOS or submit")
    parser.add_argument("--resubmit", metavar="JOBSET_DIR", help="resubmit jobs of an existing job set with missing output")
    args = parser.parse_args()

    if args.resubmit:
        return resubmit(args)
    if not (args.samples and args.cut_config and args.cut):
        parser.error("-s, -c and -k are required (unless --resubmit)")
    if args.cut not in cut_names(args.cut_config):
        parser.error(f"{args.cut} is not defined in {args.cut_config} (has {cut_names(args.cut_config)})")
    if not args.dry_run and subprocess.run("voms-proxy-info -exists", shell=True).returncode != 0:
        sys.exit("no valid grid proxy: run voms-proxy-init --voms cms --valid 192:00")

    with open(args.samples) as f:
        samples = json.load(f)
    if len({s["type"] for s in samples}) > 1:
        sys.exit("the sample json mixes types (signal/bkg/data); split it up")
    if args.particular:
        unknown = set(args.particular) - {s["name"] for s in samples}
        if unknown:
            sys.exit(f"samples not in {args.samples}: {sorted(unknown)}")
        samples = [s for s in samples if s["name"] in args.particular]

    stem = os.path.splitext(os.path.basename(args.samples))[0]
    cut_stem = os.path.splitext(os.path.basename(args.cut_config))[0]
    jobset = f"{stem}_{cut_stem}-{args.cut}"
    jobset_dir = os.path.join(HERE, "submissions_skim_cutconf", jobset)
    skimmed_json = os.path.join(os.path.dirname(os.path.abspath(args.samples)), f"{stem}_skimmed.json")

    # job set directory: jdl + wrapper + code tarball + one json per job + jobs.txt
    if os.path.isdir(jobset_dir):
        shutil.rmtree(jobset_dir)
    for d in ("jobs", "logs"):
        os.makedirs(os.path.join(jobset_dir, d))
    for f in ("condor_skim_cutconf.jdl", "run_skim_cutconf.sh"):
        shutil.copy(os.path.join(HERE, f), jobset_dir)
    make_code_tarball(os.path.join(jobset_dir, "code.tar.gz"), args.cut_config)

    manifest = {"sample_json": os.path.abspath(args.samples), "cut_config": os.path.abspath(args.cut_config),
                "cut": args.cut, "skimmed_json": skimmed_json, "samples": {}}
    skimmed_entries = {}
    jobnames = []
    for samp in samples:
        name = samp["name"]
        safe = name.replace(".", "p")
        if not samp.get("location") and not samp.get("fileset"):
            print(f"SKIP {name}: empty location")
            continue
        if samp["type"] == "bkg" and "_slim" in str(samp["location"]):
            print(f"WARNING {name}: location looks like a slim ntuple, which lacks most branches")
        out_dir = skimmed_dir(samp)
        existing = [n for n in eos_ls(out_dir) if n.endswith(".root")] + \
                   [n for n in eos_ls(f"{out_dir}/parts") if n.endswith(".root")]
        if existing and not args.force:
            print(f"SKIP {name}: {out_dir} already holds {len(existing)} .root file(s) (use --force)")
            continue
        files = list_files(samp)
        if not files:
            print(f"SKIP {name}: no input files found")
            continue

        stem = output_stem(files, safe)
        n_per = len(files) if args.files_per_job <= 0 else args.files_per_job
        n_jobs = math.ceil(len(files) / n_per)
        chunks = [files[len(files) * i // n_jobs: len(files) * (i + 1) // n_jobs] for i in range(n_jobs)]
        jobs = []
        for i, chunk in enumerate(chunks):
            if n_jobs == 1:
                job_out_dir, out_file = out_dir, f"{stem}_skimmed.root"
            else:
                job_out_dir, out_file = f"{out_dir}/parts", f"{stem}_skimmed_part{i}.root"
            jobname = f"{safe}_{i}"
            job = {"sample": samp, "sample_name": name, "sample_json": os.path.basename(args.samples),
                   "cut_config": os.path.basename(args.cut_config), "cut": args.cut,
                   "chunksize": args.chunksize, "files": chunk, "out_dir": job_out_dir, "out_file": out_file}
            with open(os.path.join(jobset_dir, "jobs", f"{jobname}.json"), "w") as f:
                json.dump(job, f, indent=2)
            jobs.append({"jobname": jobname, "out_dir": job_out_dir, "out_file": out_file, "n_files": len(chunk)})
            jobnames.append(jobname)
        manifest["samples"][name] = {"out_dir": out_dir, "out_stem": stem, "merged": n_jobs == 1, "jobs": jobs}

        if not args.dry_run:
            status, _ = xrd.mkdir(jobs[0]["out_dir"], flags=client.flags.MkDirFlags.MAKEPATH)
            if not status.ok:
                sys.exit(f"could not create {jobs[0]['out_dir']}: {status.message}")

        entry = {k: v for k, v in samp.items() if k != "fileset"}
        entry["location"] = out_dir + "/"
        entry["nFiles"] = n_jobs
        entry["blacklist"] = []
        entry["skim"] = {"cut_config": os.path.basename(args.cut_config), "cut": args.cut,
                         "unskimmed_location": samp["location"], "unskimmed_nFiles": len(files)}
        skimmed_entries[name] = entry
        print(f"{name}: {len(files)} file(s) -> {n_jobs} job(s) -> {out_dir}")

    if not jobnames:
        print("nothing to submit")
        return
    with open(os.path.join(jobset_dir, "jobs.txt"), "w") as f:
        f.write("\n".join(jobnames) + "\n")
    with open(os.path.join(jobset_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"{len(jobnames)} job(s) in {jobset_dir}")

    if args.dry_run:
        print(f"dry run: not submitting; the skimmed sample json would be {skimmed_json}")
        return

    # write/update the skimmed sample json (entries are replaced by name, so -p keeps the others)
    merged = {}
    if os.path.exists(skimmed_json):
        with open(skimmed_json) as f:
            merged = {s["name"]: s for s in json.load(f)}
    merged.update(skimmed_entries)
    with open(skimmed_json, "w") as f:
        json.dump(list(merged.values()), f, indent=4)
    print(f"wrote {skimmed_json}")

    condor_submit(jobset_dir, "jobs.txt", args)


if __name__ == "__main__":
    main()
