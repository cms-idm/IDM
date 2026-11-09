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
category=${10:-signal}
outDirName=${11}

flist_full=$(realpath "$flist")
fname=`echo $flist_full | rev | cut -d "/" -f 1 | rev | cut -d "." -f 1`

# <<<<<<< HEAD

# # fname="${fname_full#2022_}"
# # echo "$fname"


# mass=`echo $fname | cut -d "_" -f 1-2`
# ctau=`echo $fname | cut -d "_" -f 3`
# outDirName="${mass}/${ctau}"
# =======
if [ -z "$outDirName" ]; then
	mass=`echo $fname | cut -d "_" -f 1-3`
	ctau=`echo $fname | cut -d "_" -f 4`
	outDirName="${mass}/${ctau}"
fi
# >>>>>>> pr8-source

echo "$outDirName"

mkdir -p split_fileLists
mkdir -p Logs
# <<<<<<< HEAD
# xrdfs root://cmseos.fnal.gov/ mkdir -p /store/group/lpcmetx/iDMe//Samples/Ntuples/signal_${suffix}/${year}/${outDirName}/
# outPath=/store/group/lpcmetx/iDMe//Samples/Ntuples/signal_${suffix}/${year}/${outDirName}/
# echo ${outPath}
# =======
xrdfs root://cmseos.fnal.gov/ mkdir -p /store/group/lpcmetx/iDMe//Samples/Ntuples/${category}_${suffix}/${year}/${outDirName}/
outPath=/store/group/lpcmetx/iDMe//Samples/Ntuples/${category}_${suffix}/${year}/${outDirName}/
# >>>>>>> pr8-source

# Background jobs also produce a slim, all-events tree alongside the full one (see
# ElectronSkimmer.cc); it gets its own parallel EOS directory. Signal jobs never
# produce a slim file, so skip creating a destination for it.
slimOutPath=""
if [ "$isSignal" != "True" ] && [ "$isSignal" != "1" ]; then
	xrdfs root://cmseos.fnal.gov/ mkdir -p /store/group/lpcmetx/iDMe//Samples/Ntuples/${category}_${suffix}_slim/${year}/${outDirName}/
	slimOutPath=/store/group/lpcmetx/iDMe//Samples/Ntuples/${category}_${suffix}_slim/${year}/${outDirName}/
fi

sort -V ${flist_full} > ${fname}.txt
split -d -l ${nsplit} --additional-suffix ".txt" ${fname}.txt ${fname}_
# <<<<<<< HEAD
# # rm ${fname}.txt
# =======
rm ${fname}.txt

itemdata=`realpath split_fileLists`/${fname}_items.txt
> ${itemdata}
# >>>>>>> pr8-source
for sublist in `ls ${fname}_*.txt`
do
    echo "$sublist"
	mv $sublist split_fileLists/$sublist
	sublist_name=`echo $sublist | cut -d "." -f 1`
	sublist_full=`realpath split_fileLists/$sublist`
# <<<<<<< HEAD
#     echo $sublist_full
# 	condor_submit ElectronNtuplizer_config.jdl -append "Arguments = ${sublist_name} ${year} ${nThreads} ${isData} ${isSignal} ${outPath} ${CMSSW} ${compiled_CMSSW_envs}" -append "transfer_input_files = ${sublist_full}" -append "request_cpus = ${nThreads}"
# =======
	echo "${sublist_name},${sublist_full}" >> ${itemdata}
# >>>>>>> pr8-source
done

# one condor_submit call queues every split file for this dataset as a single cluster
condor_submit ElectronNtuplizer_config_batch.jdl \
	-append 'Arguments = $(sublist_name) '"${year} ${nThreads} ${isData} ${isSignal} ${outPath} ${CMSSW} ${compiled_CMSSW_envs} ${slimOutPath}" \
	-append 'transfer_input_files = $(sublist_full)' \
	-append "request_cpus = ${nThreads}" \
	-queue "sublist_name,sublist_full from ${itemdata}"
