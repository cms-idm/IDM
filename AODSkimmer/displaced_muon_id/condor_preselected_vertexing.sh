#!/usr/bin/env bash
# Existing dimuon Condor transfer pattern; args SAMPLE FILELIST OUTROOT REVISION.
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
cmsRun -j "$1.xml" preselected_vertexing_cfg.py sample="$1" flist="$2" outFile="$3" revision="$4"
date -u
