# `idm/` — Run 3 analysis package (SIDM-style coffea flow)

A pip-installable, **named-config** coffea analysis package for the IDM Run 3 (e+μ)
analysis, modelled on the sister SIDM analysis. It targets the modern stack
(coffea-2025 / awkward-2 / Python 3.11, the `idm_venv` from `SETUP_Run3.md`).

## Why this exists
- **Reproducible, distributed running** on LPC HTCondor from a notebook (`idm.tools.scaleout`).
- **Provenance** on every output (`idm.tools.metadata` writes a `.meta.yaml` sidecar).
- **Named configs, not forked code**: objects / cuts / histograms are defined once as
  named callables (`idm/definitions/`) and selected by name from YAML (`idm/configs/`),
  so each study varies *configuration* instead of copying analysis code.

## Relationship to `python_analysis/`
This package is **additive** — it does **not** replace `python_analysis/`. The existing
`python_analysis/` (coffea-0.7) remains the working analysis and source of truth; `idm/`
is the coffea-2025 migration target that we build out incrementally. The two coexist
while logic is ported, so no one's in-progress work is disrupted.

## Layout
```
idm/
  tools/        pipeline helpers
    scaleout.py   LPCCondorCluster + dask Client (+ VOMS-proxy check); ships local idm/ to workers
    metadata.py   write/load .meta.yaml provenance sidecars for .coffea outputs
  definitions/  the named-config DSL (additive scaffolding; filled in as logic is ported)
    objects.py    obj_defs: name -> f(events) -> collection
    cuts.py       cut_defs: name -> f(events) -> mask
    hists.py      hist_defs: name -> hist spec
  configs/      YAML configs that select definitions by name (samples, selections, hists)
```

## Quick use
```bash
# from the repo root, in idm_venv (see SETUP_Run3.md):
pip install -e .
```
```python
from idm.tools.scaleout import make_lpc_client          # distributed running on LPC condor
from idm.tools.metadata import write_run_metadata       # provenance sidecar for outputs
```

## Status
Stage 1 (this commit): package skeleton + `scaleout` + `metadata` + structure. Follow-ups
(Stage 2): port the processor and the object/cut/hist definitions from `python_analysis/`,
add the condor per-job-venv harness and a studies/notebook scaffold. See `NEXT_STEPS_Run3.md`.
