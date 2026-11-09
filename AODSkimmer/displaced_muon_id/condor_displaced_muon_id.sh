#!/usr/bin/env bash
# Condor transfer mode. Arguments: events FILELIST OUTROOT, or plots INPUTROOT OUTDIR.
set -euo pipefail
jobdir="$PWD"
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el8_amd64_gcc11
cd /cvmfs/cms.cern.ch/el8_amd64_gcc11/cms/cmssw/CMSSW_13_0_13/src
eval "$(scram runtime -sh)"
cd "$jobdir"
date -u
printf 'mode=%s input=%s output=%s revision=%s\n' "$1" "$2" "$3" "${ID_STUDY_REVISION:-unspecified}"
python3 --version
if [ "$1" = events ]; then
  bash run_displaced_muon_id.sh displaced_muon_id.C "$2" "$3"
elif [ "$1" = plots ]; then
  python3 plot_displaced_muon_id.py "$2" "$3"
else
  exit 2
fi
