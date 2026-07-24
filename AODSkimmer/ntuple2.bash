#!/bin/bash

for flist in /uscms/home/reshmar/nobackup/sampleFactory/SampleFactory/EightyFour/*.txt; do
    sample=$(basename "$flist" .txt)
    sample=${sample%_Run3_2022_MINIAODfiles}

    echo "Running $sample..."

    cmsRun scripts/ElectronNtuplizer_cfg.py \
        year=2022 \
        data=0 \
        signal=1 \
        nEvents=100000 \
        flist="$flist" \
        outfile="Signal_${sample}_2022_output.root"

    echo "Finished $sample"
done
