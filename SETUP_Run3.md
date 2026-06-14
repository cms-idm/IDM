# IDM Run 3 — setup & getting started

This branch (`MS_Run3`, based on `ACR_Run3_dev`) is the working line for the Run 3
inelastic-dark-matter analysis with **electrons and muons**. This document is the current,
verified setup for the coffea analysis environment on the Fermilab LPC.

> The older `python_analysis/README.md` describes the Run-2 / `cmslpc-sl7` / conda setup
> and is **out of date** — use this file for Run 3.

## 1. The analysis chain (orientation)

```
gridpack ──▶ UL_MCProduction ──▶ AOD ──▶ AODSkimmer (ElectronSkimmer) ──▶ flat ROOT ntuples
                                                                          (TTree: ntuples/outT)
                                                                                   │
                                                                                   ▼
                                                    python_analysis: coffea columnar analysis
                                                       ──▶ *.coffea histograms ──▶ SR / limits
```

- **Production / ntuplizing** (CMSSW): top-level `README.md`, `UL_MCProduction/`, `AODSkimmer/`.
- **Analysis** (coffea / Python): `python_analysis/` — this is what the venv below is for.

## 2. Build the analysis environment (one time, on LPC)

The analysis runs in a Python 3.11 virtualenv built on the cvmfs `LCG_107` view. On a
cmslpc **el9** node (`cmslpc-el9.fnal.gov`), from the repo root:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh
python -m venv idm_venv
unset PYTHONPATH                 # keep the venv isolated from the LCG view (important)
source idm_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:
- In future sessions, **just** `source idm_venv/bin/activate` — do *not* re-`source` the
  LCG view, or its packages shadow the pinned ones in the venv (you would silently get
  LCG's `coffea 2025.1.0` instead of the pinned `2025.5.0rc2`).
- `requirements.txt` pins the stack (coffea 2025.5.0rc2, awkward 2.8.7, uproot 5.7.4,
  hist 2.9.0, mplhep 1.2.0, dask 2025.3.0, correctionlib, xgboost, numba, fastjet,
  xrootd 6.0.3). `xrootd` is pinned to **6.0.3** because older versions (e.g. 5.7.0) try
  to compile from source on the node and fail; 6.0.3 installs from a wheel.
- Because `.gitignore` ignores `*.txt`, `requirements.txt` is tracked via `git add -f`.

Sanity check:

```bash
python -c "import coffea, awkward, uproot, hist, mplhep, correctionlib, xgboost; \
from XRootD import client; print('coffea', coffea.__version__, '— env OK')"
```

## 3. Run notebooks (SSH tunnel to an LPC login node)

Register the kernel once:

```bash
python -m ipykernel install --user --name idm_venv --display-name "IDM (LCG_107 Py3.11)"
```

Then, with `idm_venv` active on LPC:

```bash
jupyter lab --no-browser --port 8888     # on LPC
ssh -N -L 8888:localhost:8888 <user>@cmslpc-el9.fnal.gov   # on your laptop
# open the printed http://localhost:8888/?token=… URL and pick the "IDM (LCG_107 Py3.11)" kernel
```

## 4. Reading ntuples / EOS

Read ROOT files over xrootd, not the POSIX `/eos/uscms` mount:

```
root://cmseos.fnal.gov//store/group/lpcmetx/iDMe/<…>
```

Reads need a valid Kerberos ticket; writing to EOS needs a VOMS proxy
(`voms-proxy-init --valid 192:00 -voms cms`).

## 5. Status & known issues (Run 3 migration in progress)

- The coffea analysis code is currently **coffea-0.7-era** (`mySchema.py`,
  `processor.Runner`) and is mid-migration to the coffea-2025 API this venv provides;
  expect to port the Analyzer/processor + schema (a partial start is in
  `mySchema_newCoffea.py`). The venv is the migration target.
- The Condor submission JDLs reference an older singularity image and an absolute path from
  the original author's home area; these need repointing before batch submission works.
- Muon support (gen muons, PF + displaced-standalone muons, vertexing, IDs) is being added;
  coordinate on the collaborator branches before duplicating work.
