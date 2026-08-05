#!/bin/bash

dir="/uscms/home/reshmar/nobackup/IDMe_Run3_Collab/CMSSW_13_0_13/src/iDMe/AODSkimmer/condor/ElectronNtuplizer/TextFilesSignal"

for file in "$dir"/* ;
do
source submit_ElectronNtuplizer_condor.sh $file 2022 4 0 1 /store/group/lpcmetx/iDMe/Samples/Ntuples 10 CMSSW_13_0_13 ntuplizer_reshma_CMSSW_13_0_13.tar.gz
done