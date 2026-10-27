#!/usr/bin/env bash
# Existing dimuon Condor transfer pattern; args SAMPLE FILELIST OUTROOT REVISION [VERTEX_DIRECTIONS] [PF_NEAREST] [PF_STATION2] [PF_HIT_MATCH].
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
cmsRun -j "$1.xml" preselected_vertexing_cfg.py sample="$1" flist="$2" outFile="$3" revision="$4" vertexDirections="${5:-False}" pfNearest="${6:-False}" pfStation2="${7:-False}" pfHitMatch="${8:-False}"
date -u
