#!/bin/bash

#Incomplete

dir="/store/group/lpcmetx/iDMe/Samples/Ntuples/signal_/store/group/lpcmetx/iDMe/Samples/Ntuples/2022/Mchi-99p0_dMchi-18p0/ctau-100"
eos="root://cmseos.fnal.gov/"

xrdfs $eos ls $dir | while read -r file;
do
filename=$(basename "$file")
if [[ "$filename" == Signal_* ]] ;
then
echo "Deleting "$filename" "
xrdfs "$eos" rm "$filename"
fi

done


