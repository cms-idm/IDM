#!/bin/bash

fname=$1
year=$2
nThreads=$3
isData=$4
isSignal=$5
outPath=$6
CMSSW=$7
compiled_CMSSW_envs=$8
slimOutPath=$9

xrdcp root://cmseos.fnal.gov//store/group/lpcmetx/iDMe//compiled_CMSSW_envs/${compiled_CMSSW_envs} .
tar -xzf ${compiled_CMSSW_envs}

mv ${fname}.txt ${CMSSW}/src/iDMe/AODSkimmer
cd ${CMSSW}/src/

pwd
ls

export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
source $VO_CMS_SW_DIR/cmsset_default.sh
scram b ProjectRename
eval `scram runtime -sh`
cd iDMe/AODSkimmer

pwd
ls

cmsRun scripts/ElectronNtuplizer_cfg.py flist=${fname}.txt data=${isData} signal=${isSignal} year=${year} numThreads=${nThreads}
# <<<<<<< HEAD
# mv test_output.root ${fname}.root
# xrdcp -f ${fname}.root root://cmseos.fnal.gov/${outPath}/${fname}.root
# echo "Copied ${fname}.root"
# echo "Done"

# # source submit_ElectronNtuplizer_condor.sh /uscms/home/reshmar/nobackup/sampleFactory/SampleFactory/TxtFiles2022/2022_Mchi-105p0_dMchi-10p0_ctau-100_20260722_153614_files.txt 2022 4 0 1 /eos/uscms/store/group/lpcmetx/iDMe/Samples/Ntuples/ CMSSW_13_0_13 ntuplizer_reshma_CMSSW_13_0_13.tar.gz

# # source submit_ElectronNtuplizer_condor.sh /uscms/home/reshmar/nobackup/IDMe_Run3_Collab/CMSSW_13_0_13/src/iDMe/AODSkimmer/condor/ElectronNtuplizer/TextFilesSignal/Mchi-99p0_dMchi-18p0_ctau-100.txt 2022 4 0 1 /eos/uscms/store/group/lpcmetx/iDMe/Samples/Ntuples/ CMSSW_13_0_13 ntuplizer_reshma_CMSSW_13_0_13.tar.gz
# =======
mv test_output.root ntuples_${fname}.root
xrdcp -f ntuples_${fname}.root root://cmseos.fnal.gov/${outPath}/ntuples_${fname}.root
echo "Copied ntuples_${fname}.root"

# Background jobs also produce a slim, all-events tree (test_output_slim.root); signal
# jobs never create one, so this is a no-op there.
if [ -f test_output_slim.root ]; then
	mv test_output_slim.root ntuples_${fname}_slim.root
	xrdcp -f ntuples_${fname}_slim.root root://cmseos.fnal.gov/${slimOutPath}/ntuples_${fname}_slim.root
	echo "Copied ntuples_${fname}_slim.root"
fi

echo "Done"

pwd
ls
# >>>>>>> pr8-source
