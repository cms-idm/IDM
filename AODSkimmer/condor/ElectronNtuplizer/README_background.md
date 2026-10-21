# Background ntuples: slim/full split

Every background (`signal=0`) ntuplizer job writes **two** output files instead of one. Signal is unaffected and still writes a single file with every event.

| Stream | File | Events | Content |
|---|---|---|---|
| **full** | `ntuples_<job>.root` | only events passing the background preselection | the usual full `ntuples/outT` |
| **slim** | `ntuples_<job>_slim.root` | **every** event | event-level info only |

Most background events fail the MET trigger and could never pass the selection, so running vertexing and writing full branches for them wasted CPU and EOS space. The slim stream keeps the full, unselected event population so we can still normalize correctly and measure the trigger/MET-filter cuts. Without the slimming process, background datasets would have used over 17 TB of space; with the slim/full workflow, the current samples take up ~650GB (~500GB for the fulls, ~150GB for the slims).

## Preselection (gates the full stream)

In `plugins/ElectronSkimmer.cc` (called by `scripts/ElectronNtuplizer_cfg.py`):

- `selectionMode=hlt` (default): OR of `hltSelectionPaths`, which defaults to `HLT_PFMETNoMu120_PFMHTNoMu120_IDTight` and `HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60`.
- `selectionMode=metThreshold`: `PFMET_pt >= metThreshold`, which defaults to 200 GeV.
- **In both modes**, the event must also pass all MET filters (`METFiltersFailBits == 0`).

I recommend using the `hlt` mode as the offline cut doesn't save a ton of space and prevents us from easily changing the threshold later.

Events that fail are skipped before electrons, vertices and the other objects are reconstructed; those are only added in the slim tree.

Slim branches (`NtupleContainerV2::CreateSlimTreeBranches`) are:

- run/lumi/event numbers,
- all `trig_*` bits
- `METFiltersFailBits`
- PF and Calo MET
- `fixedGridRhoFastjetAll`
- electron counts (`nElectron`, `nLptElectron`, `nAllLptElectron`)
- `nPFJet`
- `PFHT`
- `PFJetPtLeading`
- `genWgt`
- gen PU

Both slim and full trees are stored as `ntuples/outT` in the root file, so generic tools such as `sumGenWgts.py` work on both streams. The slim file name comes from the main output name with `_slim` inserted before `.root`.

## Producing them

From this directory:

```bash
python3 ntuplize_bkg.py <key> [--subkey SUB] [--vers Aug2026noIDSep] [--nsplit 10] [--dry-run]
```

(Andrew uses version identifiers like Aug2026noIDSep, see notes in `python_analysis/configs/sample_configs/versnotes.txt`)
- `<key>` is a top-level key in `fileLists/background/MINIAOD/bkg_2024.json`, for example `QCD`, `DY`, `TT` or `Multiboson`. Each sub-entry maps a subsample name to a DAS dataset. The year is parsed from the JSON filename.
- The script queries DAS, writes the file lists to `fileLists/background/MINIAOD/fileLists/<year>/` and calls `submit_ElectronNtuplizer_condor.sh`, which submits one condor cluster per subsample.
- Output goes to two parallel EOS trees:
  ```
  /store/group/lpcmetx/iDMe/Samples/Ntuples/background_<vers>/<year>/<key>/<sub>/        # full
  /store/group/lpcmetx/iDMe/Samples/Ntuples/background_<vers>_slim/<year>/<key>/<sub>/   # slim
  ```
- `resubmit_failed.py <key>` finds full-stream outputs with a missing or broken tree and resubmits those jobs, which redoes both streams. `VERS` and `YEAR` are hardcoded at the top of the script, so update them first.
- `rename_eos_bkg.sh` merges one slim tree into another without overwriting files (`--dry-run`, `--force`, `--rmdir`). This is useful when combining partial productions. The source and destination paths are hardcoded.

### Code changes (ntuplizer)

- `4503704`: the core split.
  - **`ElectronSkimmer.cc`, `NtupleContainerV2`, `Helpers`**: add the background preselection and the slim tree.
  - **`ElectronNtuplizer_cfg.py`**: adds `selectionMode`, `metThreshold`, `hltSelectionPaths` and `slimOutfile`.
  - **`ntuplize_bkg.py`** and **`fileLists/make_file_list_bkg.py`** are new.
  - **`submit_ElectronNtuplizer_condor.sh`**: takes the output category (`background`) and output directory as arguments. It creates the `_slim` EOS directory and passes its path to the job as the 9th argument. It submits one cluster per dataset through `ElectronNtuplizer_config_batch.jdl`.
  - **`exec_ElectronNtuplizer.sh`**: copies the slim file to EOS when one was written.
- `ae77ec9`:
  - The full stream now also requires all MET filters to pass.
  - `resubmit_failed.py` handles background and the slim path.
  - `rename_eos_bkg.sh` and `xrd_rename_dir.py` are new. The latter renames an EOS directory, e.g. a version tag.
  - The `--nsplit` default drops from 50 to 10.
- `60925d5`:
  - `hlt` becomes the default `selectionMode` in the cfg instead of being hardcoded in `exec_ElectronNtuplizer.sh`.
  - Passing `selectionMode` with `signal=1` now raises an error.
  - The slim PFMET branches go back to full `float`; CaloMET stays `Float16_t`.
- `9774b78`, `d3ea374`:
  - `ntuplize_bkg.py` skips `_comment*` entries in the JSON and gains `--subkey`.
  - `resubmit_failed.py` takes `<key>`.

## Downstream (python_analysis)

Run this chain from `configs/sample_configs/` (see `jsonchain_bkg.sh`):

1. Merge the full stream with `haddMergeSamples.py`. Full files hold only a few events each, so this produces `background_<vers>_merged`. The slim stream is normally left unmerged.
2. Build two sample configs, one per stream, with `makeSignalConfigs.py -m bkg`, then `getSignalXsec.py` and `sumGenWgts.py`. For example: `bkg_2024_Aug2026.json` (merged full) and `bkg_2024_Aug2026-slim.json`.
3. For histograms, add the slimmed sample config to the Analyzer like so:
```python
   Analyzer(sample_config, hists_config, cuts_config,
            slimFileList=slimmed_config, processSlim=False)
   ```

**Normalization** (important): `sum_wgt` and `num_events` in the full config cover only the events that passed preselection, so they are **wrong** as normalization. When `slimFileList` is given, the `Analyzer` pairs each background sample with its slim entry by `bkg_<year>_<name>` and **overwrites** `sum_wgt` and `num_events` with the slim values for the whole sample. Always pass `slimFileList` for background, even if you don't process the slim files. Unpaired samples print a warning and keep the wrong values.

**`processSlim`:**
- `False`: normalization is still fixed, but the slim ntuples are never opened. Use this when you only need the main histograms.
- `True`: each slim sample is also run as a `<name>__slim` dataset, and the following changes:
  - The true `all` cutflow bin comes from the slim tree. The full tree's first bin is renamed `passHLT` ("Pass background HLT/MET preselection").
  - The **first two** cuts of the cut config are also applied to the slim tree and stored in `cutflow_slim`, `cutflow_cts_slim` and related entries. Those two cuts must use only slim branches, e.g. the MET filters and trigger. `utils.get_bkg_slim_cut_overrides` turns these into cutflow columns that replace the full-tree ones.
  - A histo config can define an optional `fillSlimHistos(events, histObj, samp, info, sum_wgt=...)` to fill inclusive histograms, such as MET, HT or trigger turn-ons, from the slim tree.

### Code changes (python_analysis)

**`configs/sample_configs/`**, commit `28f58ed`:
- `haddMergeSamples.py` (new): hadds a config's small files into files of about `--target-events` events each (default 300k), mirroring `<year>/<key>/<sub>/` under `-o`. Use `--dry-run` to see the plan. Restrict samples with `--samples` or `-g`. It sources cmsenv in a subshell to get `hadd`, so run it from the coffea env (optional, not critical, but useful for avoiding needing to process tons of tiny files)
- `jsonchain_bkg.sh` (new): runs the make, xsec and sumGenWgts chain for both the merged full stream and the slim stream.
- `getSignalXsec.py`: for bkg it now reads xsec and `filter_eff` from `bkg_<year>_xsec_filtereff_table.csv` (from `AODSkimmer/calcAuxilliaryData.py`), matching `<sample>_<subsample>`. It converts pb to fb. It falls back to `bkg_xsecs.json`.
- `makeSignalConfigs.py`: bkg and data output names now use the `-n` version name, e.g. `bkg_2024_Aug2026-slim.json`.

**`analysisTools/`**, commits `39e3daa` and `455fd47`:
- `analysisTools.py`:
  - `Analyzer` gains `slimFileList` and `processSlim`. The default is `processSlim=True`, so set it to `False` explicitly if you don't want the slim files read.
  - Pairing and the normalization override happen in `loadFiles()`. The original values are kept as `full_sum_wgt` and `full_num_events`.
  - File resolution moved into `_resolveSampleFiles()`, which both configs share.
  - `iDMeProcessor._processSlim()` handles the `__slim` datasets.
  - `genProcessor`, `bareProcessor` and `trigProcessor` skip `__slim` datasets.
  - The Runner now uses `skipbadfiles=True` and `xrootdtimeout=180`.
- `utils.py`: adds `get_bkg_slim_cut_overrides()`, described above.
- `analysisSubroutines.py`: fixes a crash in `vtxElectronConnection` for a single event with 0 vertices, which is now common in small files.

## Notes

- Andrew has only ntuplized TT, QCD, DY, Multiboson, and WtoLNu, and only for 2024. DYLowMass and Zto2Nu have not been ntuplized; Zgamma and SingleT only have MiniAODv6 samples available for 2024 which is not currently compatible with my CMSSW version.
- Best practice is to keep the two streams in sync. Normalization comes from the slim files that are present and not blacklisted. If a job's slim file is missing or blacklisted but its full file isn't, that sample is slightly over-weighted. 
- Signal configs have no slim counterpart and need none: signal files contain every event.
- Skimmed configs keep the `sum_wgt` of their parent full config, so the slim override is still required.

