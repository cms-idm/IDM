#!/bin/bash
set -euo pipefail
workdir="$PWD"
finish() {
  status=$?
  trap - EXIT
  echo "WRAPPER_EXIT=$status"
  if [ -d "$workdir/test" ]; then
    tar -czf "$workdir/results.tar.gz" -C "$workdir/test" \
      --exclude='*.root' --exclude='CMSSW_*' . || status=1
    if [ "$status" -eq 0 ]; then
      tar -czf "$workdir/miniAOD.tar.gz" -C "$workdir/test" Run3Summer22EEMiniAOD.root || status=1
    else
      tar -czf "$workdir/miniAOD.tar.gz" --files-from /dev/null || status=1
    fi
  else
    tar -czf "$workdir/results.tar.gz" --files-from /dev/null || status=1
    tar -czf "$workdir/miniAOD.tar.gz" --files-from /dev/null || status=1
  fi
  exit "$status"
}
trap finish EXIT
set +u
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el8_amd64_gcc11
cd /cvmfs/cms.cern.ch/el8_amd64_gcc11/cms/cmssw/CMSSW_13_0_13/src
eval "$(scram runtime -sh)"
set -u
# Worker /etc/grid-security can lack the CMS VOMS LSC files, so voms-proxy-info fails on a valid proxy.
export X509_VOMS_DIR=/cvmfs/grid.cern.ch/etc/grid-security/vomsdir
cd "$workdir"
python3 prepare_wjets_test.py --chain WJets_2022EE_test_chain.json \
  --fragment WJets_2022EE_official_fragment.py --pileup-list pileup.txt \
  --output "$workdir/test"
cd "$workdir/test"
timeout 8h bash run_test.sh
