#!/bin/bash
# Usage: run_private_condor_production.sh <sample> <sample index> <job> <generated events> <EOS directory LFN> <attempt id>
# run_wjets_condor_production.sh generalized to any sample of the private 2022EE background production.
set -euo pipefail
sample="$1"
index="$2"
job="$3"
nevents="$4"
eosdir="$5"
attempt="$6"
name="$(basename "$eosdir")_${job}.root"
workdir="$PWD"
finish() {
  status=$?
  trap - EXIT
  if [ -d "$workdir/test" ]; then
    tar -czf "$workdir/results.tar.gz" -C "$workdir/test" \
      --exclude='*.root' --exclude='*.lhe' --exclude='CMSSW_*' . || status=1
  else
    tar -czf "$workdir/results.tar.gz" --files-from /dev/null || status=1
  fi
  echo "WRAPPER_EXIT=$status"
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
# Never replace an output (EOS rename overwrites): a job can fail after its copy, so check EOS before
# resubmitting it, and never run two attempts of one job index at once.
require_absent() {
  local out
  if out=$(xrdfs cmseos.fnal.gov stat "$1" 2>&1); then
    echo "Output already exists: $1"
    exit 1
  fi
  case "$out" in
    *"[3011]"*"No such file or directory"*) ;;
    *) echo "$out"; echo "Cannot confirm that $1 is absent"; exit 1 ;;
  esac
}
require_absent "$eosdir/$name"
python3 prepare_private_job.py --chain "${sample}_chain.json" --fragment "${sample}_fragment.py" \
  --pileup-list pileup.txt --output "$workdir/test" --sample-index "$index" --job "$job" --nevents "$nevents"
cd "$workdir/test"
timeout 8h bash run_test.sh
# End-to-end checksummed copy to a per-attempt temporary name, then rename: a failed copy never
# leaves a final-named file.
part="$eosdir/$name.part.$attempt"
xrdcp --nopbar --path --force --cksum adler32 Run3Summer22EEMiniAOD.root "root://cmseos.fnal.gov/$part"
require_absent "$eosdir/$name"
xrdfs cmseos.fnal.gov mv "$part" "$eosdir/$name"
echo "EOS_OUTPUT=$eosdir/$name"
