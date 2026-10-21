# Background ntuples: slim/full split

Since `Aug2026noIDSep`, every background (`signal=0`) ntuplizer job writes **two** output files instead of one. Signal is unaffected and still writes a single file with every event.

| Stream | File | Events | Content |
|---|---|---|---|
| **full** | `ntuples_<job>.root` | only events passing the background preselection | the usual full `ntuples/outT` |
| **slim** | `ntuples_<job>_slim.root` | **every** event | event-level info only |

Why: most background events fail the MET trigger and could never pass the selection, so running vertexing and writing full branches for them wasted CPU and EOS space. The slim stream keeps the full, unselected event population so we can still normalize correctly and measure the trigger/MET-filter cuts.

## Preselection (gates the full stream)

In `plugins/ElectronSkimmer.cc`, and set via `scripts/ElectronNtuplizer_cfg.py`:

- `selectionMode=hlt` (default): OR of `hltSelectionPaths`, which defaults to `HLT_PFMETNoMu120_PFMHTNoMu120_IDTight` and `HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60`.
- `selectionMode=metThreshold`: `PFMET_pt >= metThreshold`, which defaults to 200 GeV.
- **In both modes**, the event must also pass all MET filters (`METFiltersFailBits == 0`).

Events that fail are skipped before electrons, vertices and the other objects are reconstructed. They show up only in the slim tree. If you pass `selectionMode` with `signal=1`, the config raises an error.

Slim branches (`NtupleContainerV2::CreateSlimTreeBranches`) are: run/lumi/event, all `trig_*` bits, `METFiltersFailBits`, PF and Calo MET, `fixedGridRhoFastjetAll`, electron counts (`nElectron`, `nLptElectron`, `nAllLptElectron`), `nPFJet`, `PFHT` and `PFJetPtLeading`, plus `genWgt` and gen PU for MC. The slim tree is also stored as `ntuples/outT`, so generic tools such as `sumGenWgts.py` work on both streams.

The slim file name comes from the main output name with `_slim` inserted before `.root`. You can override it with `slimOutfile=`.

## Producing them

From this directory:

```bash
python3 ntuplize_bkg.py <key> [--subkey SUB] [--vers Aug2026noIDSep] [--nsplit 10] [--dry-run]
```

- `<key>` is a top-level key in `fileLists/background/MINIAOD/bkg_2024.json`, for example `QCD`, `DY`, `TT` or `Multiboson`. Each sub-entry maps a subsample name to a DAS dataset. The year is parsed from the JSON filename.
- The script queries DAS, writes the file lists to `fileLists/background/MINIAOD/fileLists/<year>/` and calls `submit_ElectronNtuplizer_condor.sh`, which submits one condor cluster per subsample.
- Output goes to two parallel EOS trees:
  ```
  /store/group/lpcmetx/iDMe/Samples/Ntuples/background_<vers>/<year>/<key>/<sub>/        # full
  /store/group/lpcmetx/iDMe/Samples/Ntuples/background_<vers>_slim/<year>/<key>/<sub>/   # slim
  ```
- `resubmit_failed.py <key>` finds full-stream outputs with a missing or broken tree and resubmits those jobs, which redoes both streams. `VERS` and `YEAR` are hardcoded at the top of the script, so update them first.
- `rename_eos_bkg.sh` merges one slim tree into another without overwriting files (`--dry-run`, `--force`, `--rmdir`). This is useful when combining partial productions. The source and destination paths are hardcoded.

## Downstream (python_analysis)

Run this chain from `configs/sample_configs/` (see `jsonchain_bkg.sh`):

1. **Merge** the full stream with `haddMergeSamples.py`. Full files hold only a few events each, so this produces `background_<vers>_merged`. The slim stream is normally left unmerged.
2. Build **two sample configs**, one per stream, with `makeSignalConfigs.py -m bkg`, then `getSignalXsec.py` and `sumGenWgts.py`. For example: `bkg_2024_Aug2026.json` (merged full) and `bkg_2024_Aug2026-slim.json`.
3. Optionally **skim** the full config with `condor/skimming_cutconf/submit_condor_skim_cutconf.py`, which writes `*_skimmed.json`.
4. Run `histmaker_background.py`:
   ```python
   Analyzer(sample_config, hists_config, cuts_config,
            slimFileList=slimmed_config, processSlim=False)
   ```

**Normalization (important):** `sum_wgt` and `num_events` in the full config cover only the events that passed preselection, so they are **wrong** as normalization. When `slimFileList` is given, the `Analyzer` pairs each background sample with its slim entry by `bkg_<year>_<name>` and **overwrites** `sum_wgt` and `num_events` with the slim values for the whole sample. Always pass `slimFileList` for background, even if you don't process the slim files. Unpaired samples print a warning and keep the wrong values.

**`processSlim`:**
- `False`: normalization is still fixed, but the slim ntuples are never opened. Use this when you only need the main histograms.
- `True`: each slim sample is also run as a `<name>__slim` dataset, and the following changes:
  - The true `all` cutflow bin comes from the slim tree. The full tree's first bin is renamed `passHLT` ("Pass background HLT/MET preselection").
  - The cut config is also applied to the slim tree and stored in `cutflow_slim`, `cutflow_cts_slim` and related entries. Only cuts that use slim branches make sense here, such as the MET filters and trigger. `utils.get_bkg_slim_cut_overrides` turns these into cutflow columns that replace the full-tree ones.
  - A histo config can define an optional `fillSlimHistos(events, histObj, samp, info, sum_wgt=...)` to fill inclusive histograms, such as MET, HT or trigger turn-ons, from the slim tree.

## Gotchas

- **Keep the two streams in sync.** Normalization comes from the slim files that are present and not blacklisted. If a job's slim file is missing or blacklisted but its full file isn't, that sample is slightly over-weighted. For example, `DY_EE` currently has 1227 full vs. 1217 slim files, plus 10 blacklisted slim files. Compare file counts and blacklists after production.
- Signal configs have no slim counterpart and need none: signal files contain every event.
- Skimmed configs keep the `sum_wgt` of their parent full config, so the slim override is still required.
- Version notes are in `python_analysis/configs/sample_configs/versnotes.txt`.
