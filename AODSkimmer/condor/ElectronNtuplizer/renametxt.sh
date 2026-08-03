#!/bin/bash

#A bash script that gives the required/right name to the .txt files.

dir="/uscms/home/reshmar/nobackup/IDMe_Run3_Collab/CMSSW_13_0_13/src/iDMe/AODSkimmer/condor/ElectronNtuplizer/TextFilesSignal"

for file in "$dir"/*;
do
sample=$(basename $file)
sample=${sample#2022_}
sample=${sample%_[0-9]*_[0-9]*_files.txt}.txt

mv "$file" "$dir/$sample"


done