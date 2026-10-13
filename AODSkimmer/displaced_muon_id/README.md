# Displaced-muon ID comparison

This exploratory study uses the embedded inner/outer/global components of
`RecoDisplacedMuon` in the existing propagated signal ntuples. It changes no
production selection or ntuplizer. Outer means this object's outerTrack, not an
independently matched member of the raw DSA collection.

Signal gen muons are the saved GenSigMuon/AntiMuon scalars, explicitly requiring
status 1, last copy and absolute first different-PDG mother ID 1000023. Acceptance
is `abs(eta)<2.4`, `0<=vxy<740 cm`, `abs(vz)<960 cm`, with **no minimum pT**.
Gen propagation failures remain in the accepted denominator; no low-pT fallback
is introduced. Gen |d0| is `abs(vx*sin(phi)-vy*cos(phi))`, straight line to (0,0).

Matching uses station-2 position eta/phi, charge agreement, strict dR<0.06 for
inner/global and <0.10 for outer, and globally nearest greedy one-to-one matching
with compatible fallback. The component study independently matches each type.
Whole objects match using the smallest distance among any compatible components;
each object/gen can be assigned once. Categories are defined by valid components,
not by successful propagation. Baseline pre-ID matches are **held fixed** for
conditional ID efficiency. Reco efficiency selects ID-passing units first, then
reruns matching, permitting compatible alternatives. Both the fixed retention
and post-ID match counts are saved. Different reco objects sharing a TrackRef remain
different units; this study neither deduplicates them nor proves independent tracks.

Component IDs: inner requires its parent `isTracker`; global requires its parent
`isGlobal`; outer uses the exact `ElectronSkimmer::passesDisplacedID` conditions:
CSC+DT valid hits >12; CSC>0 or DT>18; normalized chi2<2.5; ptError/pt<1.
For the outer ID, zero/nonpositive pT fails the relative-error requirement.

Plots have reco efficiency with ID on top (matched after ID selection / all accepted
signal gen), and conditional ID efficiency below (passing / pre-ID matched units).
The three requested sets are:

- `components`: independent inner, outer, global curves, including components of mixed objects.
- `all_components`: every present component passes its ID; total plus global-present,
  inner-only, outer-only, and inner+outer-without-global categories. Category reco
  efficiencies share the all-accepted-gen denominator; ID denominators are category specific.
- `hierarchy_categories`: global-present requires isGlobal and ignores other IDs;
  inner-only tracker; outer-only DSA; mixed requires both, tracker, DSA, or either.
  Tracker/DSA requirements are inclusive, not exclusive of passing the other ID.
  `hierarchy` gives the corresponding four complete object-policy curves.

All sets have pT (starting at zero), eta, Lxy and |d0| axes, per sample, per mass
pair pooled over lifetime, per lifetime pooled over masses, and overall.
Pooling is unweighted. Error bars are 68.27% Clopper-Pearson intervals; zero
denominators have no point. Endpoint overflow/underflow is folded into end bins,
including pT>=100 GeV and values below log-axis minima; canvases disclose this.

## Running on LPC

Requires the existing CMSSW_13_0_13 ROOT/PyROOT and numpy runtime; never run project
code on the laptop. Source directory after Git deployment:
`/uscms_data/d3/abdelham/CMSSW_13_0_13/src/iDMe/AODSkimmer/displaced_muon_id`.
Existing campaign file list:
`/uscms_data/d3/abdelham/charge_flip_check_20261013/files_all.txt` (288 paths).
Outputs belong in a new scratch run directory; keep previous runs intact.

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /uscms_data/d3/abdelham/CMSSW_13_0_13/src
eval "$(scram runtime -sh)"
code=/uscms_data/d3/abdelham/CMSSW_13_0_13/src/iDMe/AODSkimmer/displaced_muon_id
run=/uscms_data/d3/abdelham/displaced_muon_id_$(date +%Y%m%d_%H%M%S)
mkdir -p "$run/logs"
cd "$run"
# For a bounded 1000-event validation; omit the final argument for a full chunk.
bash "$code/run_displaced_muon_id.sh" "$code/displaced_muon_id.C" \
  /uscms_data/d3/abdelham/charge_flip_check_20261013/files_all.txt validation.root 1000 \
  > logs/validation.log 2>&1
python3 "$code/plot_displaced_muon_id.py" validation.root validation_plots --overall-only \
  > logs/validation_plots.log 2>&1
# Long event processing must use Condor. Merge only successfully completed chunks:
hadd -f id_all.root out_chunk_*.root > logs/merge.log 2>&1
python3 "$code/plot_displaced_muon_id.py" id_all.root plots > logs/plots.log 2>&1
```

Retain the actual submission/wrapper and original input paths in the run directory,
timestamped logs with commands/runtime/commit, merged per-gen ROOT rows, PNGs, and
the central `id_all.root` and `id_all.json`. The ROOT contains accepted-gen rows,
every reconstructed displacedMuon (even events without accepted gen), component
ID/matching inputs, pre/post-ID any-component and first-present global→inner→outer
match indices, and all numerator/denominator histograms. The JSON contains definitions,
grouped binned counts, any-vs-priority count comparisons, and a per-sample/overall
audit of global-component presence versus isGlobal. It is also embedded in ROOT.
Priority matching has no fallback if the first present component fails propagation,
charge or dR. `matching_comparison` adds six-panel any-versus-priority reco-efficiency
plots (pre-ID and the five complete ID policies), with all grouping/x-axis breakdowns.
Changing the compatible pairs can change greedy assignments; per-gen priority-only
and any-only counts are retained rather than assuming a pointwise subset relation.
The plotting log contains pooled counts and efficiencies. Validate input completion
before merging; a partial failed job must not enter the results.

The saved `events.jdl` and `plots.jdl` in the actual run directory are the exact
Condor submissions; reproduce with `condor_submit events.jdl`, verify successful
completion, merge, then `condor_submit plots.jdl`. `condor_displaced_muon_id.sh`
uses the CVMFS CMSSW_13_0_13 runtime on EL8 and transferred inputs; submit with a
valid user-created CMS proxy. The run's logs record the source revisions actually used.
