#!/bin/bash
# condor worker wrapper: set up an EL9 LCG view (ROOT + coffea + xrootd), unpack the
# code tarball (analysisTools/, the cut config, the skimmer) and skim one job json
set -e
job=$1

echo "host: $(hostname)   os: $(cat /etc/redhat-release)   job: ${job}"
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh
tar xzf code.tar.gz
python3 condor_skim_cutconf.py ${job}
