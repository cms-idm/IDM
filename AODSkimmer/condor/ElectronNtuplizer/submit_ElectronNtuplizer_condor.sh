#!/bin/bash

flist=$1
year=$2
nThreads=$3
isData=$4
isSignal=$5
suffix=$6
nsplit=$7
CMSSW=$8
compiled_CMSSW_envs=$9

flist_full=`realpath $flist`
fname=`echo $flist_full | rev | cut -d "/" -f 1 | rev | cut -d "." -f 1`

mass=`echo $fname | cut -d "_" -f 1-3`
ctau=`echo $fname | cut -d "_" -f 4`
outDirName="${mass}/${ctau}"

mkdir -p split_fileLists
mkdir -p Logs
xrdfs root://cmseos.fnal.gov/ mkdir -p /store/group/lpcmetx/iDMe//Samples/Ntuples/signal_${suffix}/${year}/${outDirName}/
outPath=/store/group/lpcmetx/iDMe//Samples/Ntuples/signal_${suffix}/${year}/${outDirName}/

cp ${flist_full} .
split -d -l ${nsplit} --additional-suffix ".txt" ${fname}.txt ${fname}_
rm ${fname}.txt
for sublist in `ls ${fname}_*.txt`
do
	mv $sublist split_fileLists/$sublist
	sublist_name=`echo $sublist | cut -d "." -f 1`
	sublist_full=`realpath split_fileLists/$sublist`
	condor_submit ElectronNtuplizer_config.jdl -append "Arguments = ${sublist_name} ${year} ${nThreads} ${isData} ${isSignal} ${outPath} ${CMSSW} ${compiled_CMSSW_envs}" -append "transfer_input_files = ${sublist_full}" -append "request_cpus = ${nThreads}"
done
