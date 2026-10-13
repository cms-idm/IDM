#!/usr/bin/env bash
set -euo pipefail

# Copy the full MiniAOD files. The event/pair/track labels describe the events
# of interest; they do not filter the events inside each ROOT file.
# Requires authenticated read access to FNAL EOS and write access to CERNBox.
# Run with: bash copy_shared_innerTrack_inputs_to_cernbox.sh [destination EOS path]
# Existing files are not overwritten.

cernbox_server='root://eosuser.cern.ch'
cernbox_folder=${1:-/eos/user/a/alabdelh/shared_innerTrack_event_display_inputs_20261007}
input_base='root://cmseos.fnal.gov//store/user/lpcmetx/iDMe/iDMmu_Run3_Production/2022EE/MINIAOD'

command -v xrdcp >/dev/null
command -v xrdfs >/dev/null
xrdfs "$cernbox_server" mkdir -p "$cernbox_folder"

copy_root() {
    local event_id=$1 pair=$2 track=$3 source=$4
    local run lumi event basename parent shard timestamp campaign filename
    IFS=: read -r run lumi event <<< "$event_id"
    basename=${source##*/}
    parent=${source%/*}
    shard=${parent##*/}
    parent=${parent%/*}
    timestamp=${parent##*/}
    parent=${parent%/*}
    campaign=${parent##*/}
    campaign=${campaign#2022EE_}
    filename="Run_${run}-Lumi_${lumi}-Event_${event}_Pair_${pair}_Track_${track}_${campaign}_${timestamp}_${shard}_${basename}"
    printf 'Copying %s, muon pair %s, track %s\n%s\n' "$event_id" "$pair" "$track" "$filename"
    xrdcp --streams 4 --cksum adler32 --posc "$source" "$cernbox_server/$cernbox_folder/$filename"
}

copy_root '1:10003:2485' '4-5' '4' "$input_base/Mchi-6p0_dMchi-2p0/ctau-1/iDMmu_pythiaGenFragment_ctau-1__chain_Run3Summer22EE/2022EE_Mchi-6p0_dMchi-2p0_ctau-1_20260928_143607/260928_193643/0000/Run3Summer22EEMiniAOD_3.root"
copy_root '1:10001:1341' '0-2' '1' "$input_base/Mchi-36p0_dMchi-12p0/ctau-100/iDMmu_pythiaGenFragment_ctau-100__chain_Run3Summer22EE/2022EE_Mchi-36p0_dMchi-12p0_ctau-100_20260928_150043/260928_200112/0000/Run3Summer22EEMiniAOD_1.root"
copy_root '1:10002:1375' '0-2' '2' "$input_base/Mchi-52p5_dMchi-5p0/ctau-100/iDMmu_pythiaGenFragment_ctau-100__chain_Run3Summer22EE/2022EE_Mchi-52p5_dMchi-5p0_ctau-100_20260928_150247/260928_200318/0000/Run3Summer22EEMiniAOD_2.root"
copy_root '1:10004:2063' '3-4' '3' "$input_base/Mchi-42p0_dMchi-4p0/ctau-100/iDMmu_pythiaGenFragment_ctau-100__chain_Run3Summer22EE/2022EE_Mchi-42p0_dMchi-4p0_ctau-100_20260928_150113/260928_200144/0000/Run3Summer22EEMiniAOD_4.root"
copy_root '1:10002:575' '2-3' '2' "$input_base/Mchi-12p0_dMchi-4p0/ctau-10/iDMmu_pythiaGenFragment_ctau-10__chain_Run3Summer22EE/2022EE_Mchi-12p0_dMchi-4p0_ctau-10_20260928_144802/260928_194832/0000/Run3Summer22EEMiniAOD_2.root"
copy_root '1:10003:1500' '0-1' '0' "$input_base/Mchi-24p0_dMchi-8p0/ctau-1000/iDMmu_pythiaGenFragment_ctau-1000__chain_Run3Summer22EE/2022EE_Mchi-24p0_dMchi-8p0_ctau-1000_20260928_150853/260928_200959/0000/Run3Summer22EEMiniAOD_3.root"
copy_root '1:10001:782' '0-4' '3' "$input_base/Mchi-31p5_dMchi-3p0/ctau-10/iDMmu_pythiaGenFragment_ctau-10__chain_Run3Summer22EE/2022EE_Mchi-31p5_dMchi-3p0_ctau-10_20260928_145002/260928_195035/0000/Run3Summer22EEMiniAOD_1.root"
copy_root '1:10003:1595' '1-2' '0' "$input_base/Mchi-55p0_dMchi-10p0/ctau-1/iDMmu_pythiaGenFragment_ctau-1__chain_Run3Summer22EE/2022EE_Mchi-55p0_dMchi-10p0_ctau-1_20260928_144425/260928_194455/0000/Run3Summer22EEMiniAOD_3.root"
copy_root '1:10001:832' '0-2' '1' "$input_base/Mchi-12p0_dMchi-4p0/ctau-1/iDMmu_pythiaGenFragment_ctau-1__chain_Run3Summer22EE/2022EE_Mchi-12p0_dMchi-4p0_ctau-1_20260928_143801/260928_193838/0000/Run3Summer22EEMiniAOD_1.root"

printf '\nCopied all 9 files to %s\n' "$cernbox_folder"
