#!/usr/bin/env bash
# Condor wrapper for DimuonVertexTruth. Arguments: SAMPLE FILELIST OUTROOT
# Transferred inputs: this script, dimuon_vertex_truth_cfg.py, FILELIST, cmssw_lib.tgz
# (lib/ of the user CMSSW_13_0_13 area, which holds the compiled plugin).
set -euo pipefail
jobdir="$PWD"
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el8_amd64_gcc11
scram project CMSSW CMSSW_13_0_13 > /dev/null
tar -xzf cmssw_lib.tgz -C CMSSW_13_0_13
cd CMSSW_13_0_13/src
eval "$(scram runtime -sh)"
cd "$jobdir"
date -u
printf 'sample=%s flist=%s output=%s revision=%s\n' "$1" "$2" "$3" "${DIMUVTX_REVISION:-unspecified}"
cmsRun dimuon_vertex_truth_cfg.py flist="$2" sample="$1" outFile="$3"
date -u
