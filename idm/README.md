# `idm/` — Run 3 analysis package (SIDM-style coffea flow)

A pip-installable, **named-config** coffea analysis package for the IDM Run 3 (e+μ)
analysis, modelled on the sister SIDM analysis. It targets the modern stack
(coffea-2025 / awkward-2 / Python 3.11, the `idm_venv` from `SETUP_Run3.md`).

## Why this exists
- **Reproducible, distributed running** on LPC HTCondor from a notebook (`idm.tools.scaleout`).
- **Provenance** on every output (`idm.tools.metadata` writes a `.meta.yaml` sidecar).
- **Named configs, not forked code**: cuts / histograms (and, later, objects) are defined once
  as named callables (`idm/definitions/`) and selected *by name*, so each study varies
  *configuration* instead of copying analysis code. (Selection is via Python kwargs to
  `IdmProcessor` today; a YAML config layer like SIDM's is planned, not yet built.)

## The flow in three steps
1. **Define** analysis content once, by name, in `idm/definitions/`:
   - `cuts.py` — `cut_defs[name]` → `f(events)` returning a per-event mask
   - `hists.py` — `hist_defs[name]` → `{"axis": <hist.axis>, "fill": f(events) → array}`
   - `objects.py` — `obj_defs[name]` → `f(events)` returning a collection *(placeholder — not
     yet consumed by the processor; cuts/hists access collections like `e.Electron` directly)*
2. **Select** which cuts/hists to run by name when you build the processor.
3. **Run** the processor → it applies the cuts (in list order, cumulative cutflow) and fills the hists.

## Quick start

> 📓 **New here?** Start with the runnable walkthrough: `idm/tutorials/01_workflow_walkthrough.ipynb`.
```bash
# from the repo root, in idm_venv (see SETUP_Run3.md):
pip install -e .
```

### Runnable example (end-to-end, ~10 lines)
```python
from coffea.nanoevents import NanoEventsFactory
from idm.tools.processor import IdmProcessor

# load an ntuple with the IDM schema (the coffea-2025 schema still lives in python_analysis/,
# so this sys.path shim is needed until it's packaged; run from the repo root)
import sys; sys.path.insert(0, "python_analysis/analysisTools")
from mySchema_newCoffea import MySchema

events = NanoEventsFactory.from_root(
    {"my_ntuple.root": "ntuples/outT"}, schemaclass=MySchema,
).events()

# pick cuts + histograms BY NAME (defined in idm/definitions/) and run
p = IdmProcessor(cuts=["has_electron"], hists=["n_electron", "electron_pt", "muon_pt"])
out = p.process(events)

print(out["cutflow"])              # {"all": N, "has_electron": M}
out["hists"]["muon_pt"]            # a filled hist.Hist — plot or accumulate it
```
To run *distributed* over many files on LPC condor, build a client with
`idm.tools.scaleout.make_lpc_client()` and drive the processor with coffea's runner; record
provenance for the output with `idm.tools.metadata.write_run_metadata(...)`.

## Relationship to `python_analysis/`
This package is **additive** — it does **not** replace `python_analysis/`. The existing
`python_analysis/` (coffea-0.7) remains the working analysis and source of truth; `idm/` is the
coffea-2025 migration target that we build out incrementally. The two coexist while logic is
ported, so no one's in-progress work is disrupted.

## Layout
```
idm/
  tools/        pipeline helpers
    scaleout.py    LPCCondorCluster + dask Client (+ VOMS-proxy check); ships local idm/ to workers
    metadata.py    write/load .meta.yaml provenance sidecars for .coffea outputs
    processor.py   IdmProcessor: the named-config engine (apply named cuts, fill named hists)
    plotting.py    CMS-style plotting helpers (figsize/style/exp_label/save pdf+png)
    lint_plots.py  mechanical CMS/mplhep plotting linter (python -m idm.tools.lint_plots)
  definitions/  the named-config DSL (example defs included; filled in as logic is ported)
    cuts.py        cut_defs:   name -> f(events) -> mask          (wired into IdmProcessor)
    hists.py       hist_defs:  name -> {"axis": <hist.axis>, "fill": f(events) -> array}
    objects.py     obj_defs:  name -> f(events) -> collection     (placeholder; not yet wired)
  configs/      (planned) YAML configs to select definitions by name — not present yet;
                today you pass cut/hist names directly to IdmProcessor(...)
```

## Status
Done: the package + `scaleout` + `metadata` + `IdmProcessor` (with example defs) + CMS plotting
helpers (`plotting.py` / `lint_plots.py`) + the condor per-job-venv harness (`condor/`) + the
runnable tutorial notebook.
Next: port the real object/cut/hist definitions from `python_analysis/`; wire `obj_defs` into the
engine (or drop it); build the YAML config layer; add a studies/notebook scaffold.
