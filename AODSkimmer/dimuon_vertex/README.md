# Truth-matched dimuon vertexing (vertexing step 1)

`plugins/DimuonVertexTruth.cc` + `dimuon_vertex_truth_cfg.py`. Definitions are in the plugin
header and in the `definitions` TNamed written to every output file.

- Signal muons: |pdgId|==13, status 1, last copy, first different-PDG ancestor 1000023; events with
  exactly one mu- and one mu+. Acceptance |eta|<2.4, 0<=vxy<740 cm, |vz|<960 cm, no pT minimum.
- Reco: unique inner/global/outer tracks of `displacedMuons`.
- Matching, per track type: station 2, same charge, dR<0.06 inner/global, <0.10 outer, greedy
  one-to-one. Fallback pass for gen muons failing station 2: inner at the ECAL inner boundary,
  global/outer at station 1 (`matchSurface` says which).
- Fits: KalmanVertexFitter(refit=true) for 9 ordered (mu- type, mu+ type) combinations.

Input: the 2026-09-23 MINIAOD listed in `fileLists/signal/2022EE/iDMmu/*_flist.txt`.
The 2026-05-06 production lacks `displacedMuons`.

## Running on LPC

Interactive check of one file (~30-120 s):

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /uscms_data/d3/abdelham/CMSSW_13_0_13/src && eval "$(scram runtime -sh)"
cmsRun iDMe/AODSkimmer/dimuon_vertex/dimuon_vertex_truth_cfg.py flist=<file.root or list.txt> sample=<name> outFile=out.root
```

Full sample on Condor, one job per sample, from a new run directory:

```bash
code=/uscms_data/d3/abdelham/CMSSW_13_0_13/src/iDMe/AODSkimmer/dimuon_vertex
run=/uscms_data/d3/abdelham/dimuon_vertex_<date>; mkdir -p $run/logs && cd $run
tar -czf cmssw_lib.tgz -C /uscms_data/d3/abdelham/CMSSW_13_0_13 lib
ls /uscms_data/d3/abdelham/CMSSW_13_0_13/src/iDMe/AODSkimmer/fileLists/signal/2022EE/iDMmu/*_flist.txt \
  | xargs -n1 basename | sed 's/_flist.txt//' > samples.txt
condor_submit DIMUVTX_REVISION=$(git -C $code rev-parse --short HEAD) $code/dimuon_vertex_truth.jdl
```

Plots and JSON (overall / mass / lifetime / sample; ~14 min on an interactive node for 72 samples):

```bash
python3 $code/plot_dimuon_vertex.py "$run/dimuon_vertex_*.root" $run/plots
```

Definitions (pooling of mixed combinations, panels, bins) are in the script docstring and in
`plots/dimuon_vertex_summary.json`.
