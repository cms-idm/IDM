#!/usr/bin/env bash
# On LPC after compiling the corrected analyzer:
# bash run_shared_20260928.sh /absolute/path/to/previous_run/filelists
# Reuses the selected 20260928 MiniAOD lists, traverses each sample once, and
# produces independent duplicate diagnostics and shared-gen residual plots.
set -eo pipefail
work_dir="$(cd "$(dirname "$0")" && pwd)"
list_dir="$(cd "${1:?Pass the previous run filelists directory}" && pwd)"
cmssw_src="$(cd "$work_dir/../../../.." && pwd)"
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$cmssw_src"
eval "$(scram runtime -sh)"
set -u

run_dir="$work_dir/shared_fixed_20260928_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$run_dir"
mkdir "$run_dir/filelists"
shopt -s nullglob
lists=("$list_dir"/*.txt)
[[ ${#lists[@]} -gt 0 ]] || { echo "No input file lists: $list_dir" >&2; exit 1; }
cp "${lists[@]}" "$run_dir/filelists/"

echo "Results: $run_dir"
echo "Input lists: $list_dir"
echo "CMSSW: $CMSSW_VERSION; architecture: $SCRAM_ARCH"
git -C "$cmssw_src/iDMe" rev-parse --abbrev-ref HEAD || true
git -C "$cmssw_src/iDMe" rev-parse HEAD || true
python3 -c 'import ROOT; print("ROOT:", ROOT.gROOT.GetVersion())'

first=1
for list in "$run_dir"/filelists/*.txt; do
    [[ -s "$list" ]] || { echo "Empty input list: $list" >&2; exit 1; }
    sample="$(basename "$list" .txt)"
    sample_dir="$run_dir/$sample"
    mkdir "$sample_dir"
    echo "Starting $sample ($(wc -l < "$list") files), $(date -u +%FT%TZ)"
    (
        cd "$sample_dir"
        printf 'cmsRun %q flist=%q year=2022EE data=False signal=True numThreads=1 nEvents=-1 outfile=unused.root\n' \
          "$work_dir/your_diagnostic_cfg.py" "$list"
        cmsRun "$work_dir/your_diagnostic_cfg.py" \
          flist="$list" year=2022EE data=False signal=True \
          numThreads=1 nEvents=-1 outfile=unused.root > cmsRun.log 2>&1
        sed -n '/^=== displacedMuons double-counting diagnostic ===$/,/^===============================================/p' \
          cmsRun.log > shared_track_summary.txt
        python3 "$work_dir/displaced_shared_track_pt_residuals.py" \
          --input displaced_shared_tracks.root --outdir plots > plotting.log 2>&1
        python3 - "$first" <<'PY'
import json
import sys
with open("plots/residuals.json") as src:
    data = json.load(src)
counts = data["counts"]
print("COUNTS:", counts)
assert counts.get("inner", 0) == counts.get("global", 0) == counts.get("shared_gen", 0)
assert counts.get("best", 0) == counts.get("global_priority", 0)
assert sum(counts.get(f"best_choice_{j}", 0) for j in range(4)) == counts.get("shared_gen", 0)
if sys.argv[1] == "1" and not counts.get("shared_gen", 0):
    raise SystemExit("First sample has zero shared-gen matches: inspect cmsRun.log and plotting.log before continuing.")
for metric in ("res", "rel_res"):
    for percent in (68, 90):
        for policy in ("global_priority", "best"):
            name = f"g_shared_priority_best_{metric}_interval{percent}_{policy}"
            print(name, "points:", len(data["graphs"][name]["x"]))
PY
    )
    first=0
    echo "Completed $sample, $(date -u +%FT%TZ)"
done
echo "Finished: $run_dir"
