for flist in /uscms/home/reshmar/nobackup/sampleFactory/SampleFactory/txtfiles/*.txt; do
    sample=$(basename "$flist" "_Run3Summer22MINIAOD_files.txt")

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
