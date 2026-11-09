#!/bin/bash
jobname=$1
outDir=$2
MET_cut=$3
nJet_cut=$4

source /cvmfs/sft.cern.ch/lcg/views/LCG_103/x86_64-centos7-gcc11-opt/setup.sh
tar xzf ${jobname}.tar.gz
python condor_skim_rdf.py -n ${jobname} -o ${outDir} -m ${MET_cut} -j ${nJet_cut}
#echo "xrdcp -f *.root root://cmseos.fnal.gov/${outDir}/"
#xrdcp -f *.root root://cmseos.fnal.gov/${outDir}/