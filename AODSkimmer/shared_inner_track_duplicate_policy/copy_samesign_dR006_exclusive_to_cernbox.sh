#!/usr/bin/env bash
set -euo pipefail

# Copy the full MiniAOD files of the 15 shared inner tracks matched exclusively to a same-sign gen muon
# with dR < 0.06 (11 crossedIP, 4 not). The event/pair/track labels describe the events of interest;
# they do not filter the events inside each ROOT file.
# Requires authenticated read access to FNAL EOS and write access to CERNBox.
# Run with: bash copy_samesign_dR006_exclusive_to_cernbox.sh [destination EOS path]
# Existing files are not overwritten.

cernbox_server='root://eosuser.cern.ch'
cernbox_folder=${1:-/eos/user/a/alabdelh/shared_innerTrack_samesign_dR006_exclusive_20261008}
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

# crossedIP
copy_root '1:10001:919' '0-4' '3' "$input_base/Mchi-42p0_dMchi-4p0/ctau-1000/iDMmu_pythiaGenFragment_ctau-1000__chain_Run3Summer22EE/2022EE_Mchi-42p0_dMchi-4p0_ctau-1000_20260928_151215/260928_201245/0000/Run3Summer22EEMiniAOD_1.root"
copy_root '1:10001:1341' '0-2' '1' "$input_base/Mchi-36p0_dMchi-12p0/ctau-100/iDMmu_pythiaGenFragment_ctau-100__chain_Run3Summer22EE/2022EE_Mchi-36p0_dMchi-12p0_ctau-100_20260928_150043/260928_200112/0000/Run3Summer22EEMiniAOD_1.root"
copy_root '1:10002:1363' '0-2' '1' "$input_base/Mchi-6p0_dMchi-2p0/ctau-100/iDMmu_pythiaGenFragment_ctau-100__chain_Run3Summer22EE/2022EE_Mchi-6p0_dMchi-2p0_ctau-100_20260928_145556/260928_195627/0000/Run3Summer22EEMiniAOD_2.root"
copy_root '1:10002:1375' '0-2' '2' "$input_base/Mchi-52p5_dMchi-5p0/ctau-100/iDMmu_pythiaGenFragment_ctau-100__chain_Run3Summer22EE/2022EE_Mchi-52p5_dMchi-5p0_ctau-100_20260928_150247/260928_200318/0000/Run3Summer22EEMiniAOD_2.root"
copy_root '1:10002:1477' '1-2' '2' "$input_base/Mchi-55p0_dMchi-10p0/ctau-1000/iDMmu_pythiaGenFragment_ctau-1000__chain_Run3Summer22EE/2022EE_Mchi-55p0_dMchi-10p0_ctau-1000_20260928_151420/260928_201452/0000/Run3Summer22EEMiniAOD_2.root"
copy_root '1:10003:237' '0-4' '4' "$input_base/Mchi-31p5_dMchi-3p0/ctau-100/iDMmu_pythiaGenFragment_ctau-100__chain_Run3Summer22EE/2022EE_Mchi-31p5_dMchi-3p0_ctau-100_20260928_145931/260928_200001/0000/Run3Summer22EEMiniAOD_3.root"
copy_root '1:10003:1638' '0-4' '3' "$input_base/Mchi-22p0_dMchi-4p0/ctau-1000/iDMmu_pythiaGenFragment_ctau-1000__chain_Run3Summer22EE/2022EE_Mchi-22p0_dMchi-4p0_ctau-1000_20260928_150809/260928_200852/0000/Run3Summer22EEMiniAOD_3.root"
copy_root '1:10003:1948' '0-3' '2' "$input_base/Mchi-22p0_dMchi-4p0/ctau-100/iDMmu_pythiaGenFragment_ctau-100__chain_Run3Summer22EE/2022EE_Mchi-22p0_dMchi-4p0_ctau-100_20260928_145830/260928_195901/0000/Run3Summer22EEMiniAOD_3.root"
copy_root '1:10004:857' '0-3' '2' "$input_base/Mchi-11p0_dMchi-2p0/ctau-100/iDMmu_pythiaGenFragment_ctau-100__chain_Run3Summer22EE/2022EE_Mchi-11p0_dMchi-2p0_ctau-100_20260928_145656/260928_195727/0000/Run3Summer22EEMiniAOD_4.root"
copy_root '1:10004:1475' '0-1' '0' "$input_base/Mchi-33p0_dMchi-6p0/ctau-100/iDMmu_pythiaGenFragment_ctau-100__chain_Run3Summer22EE/2022EE_Mchi-33p0_dMchi-6p0_ctau-100_20260928_150001/260928_200043/0000/Run3Summer22EEMiniAOD_4.root"
copy_root '1:10004:1586' '0-1' '0' "$input_base/Mchi-60p0_dMchi-20p0/ctau-100/iDMmu_pythiaGenFragment_ctau-100__chain_Run3Summer22EE/2022EE_Mchi-60p0_dMchi-20p0_ctau-100_20260928_150350/260928_200420/0000/Run3Summer22EEMiniAOD_4.root"

# not crossedIP
copy_root '1:10001:782' '0-4' '3' "$input_base/Mchi-31p5_dMchi-3p0/ctau-10/iDMmu_pythiaGenFragment_ctau-10__chain_Run3Summer22EE/2022EE_Mchi-31p5_dMchi-3p0_ctau-10_20260928_145002/260928_195035/0000/Run3Summer22EEMiniAOD_1.root"
copy_root '1:10001:832' '0-2' '1' "$input_base/Mchi-12p0_dMchi-4p0/ctau-1/iDMmu_pythiaGenFragment_ctau-1__chain_Run3Summer22EE/2022EE_Mchi-12p0_dMchi-4p0_ctau-1_20260928_143801/260928_193838/0000/Run3Summer22EEMiniAOD_1.root"
copy_root '1:10003:1500' '0-1' '0' "$input_base/Mchi-24p0_dMchi-8p0/ctau-1000/iDMmu_pythiaGenFragment_ctau-1000__chain_Run3Summer22EE/2022EE_Mchi-24p0_dMchi-8p0_ctau-1000_20260928_150853/260928_200959/0000/Run3Summer22EEMiniAOD_3.root"
copy_root '1:10003:1595' '1-2' '0' "$input_base/Mchi-55p0_dMchi-10p0/ctau-1/iDMmu_pythiaGenFragment_ctau-1__chain_Run3Summer22EE/2022EE_Mchi-55p0_dMchi-10p0_ctau-1_20260928_144425/260928_194455/0000/Run3Summer22EEMiniAOD_3.root"

printf '\nCopied all 15 files to %s\n' "$cernbox_folder"
