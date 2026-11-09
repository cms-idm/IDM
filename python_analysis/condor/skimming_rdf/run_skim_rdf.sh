#!/bin/bash
jobname=$1
outDir=$2
MET_cut=$3
nJet_cut=$4

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el8_amd64_gcc11

# Set up the CMSSW release on the worker node
scram p CMSSW CMSSW_13_0_13
cd CMSSW_13_0_13/src
eval `scramv1 runtime -sh`
cd -

# Explicitly run with python3, passing your arguments
tar xzf ${jobname}.tar.gz
python3 condor_skim_rdf.py -n ${jobname} -o ${outDir} -m ${MET_cut} -j ${nJet_cut}
#echo "xrdcp -f *.root root://cmseos.fnal.gov/${outDir}/"
#xrdcp -f *.root root://cmseos.fnal.gov/${outDir}/
