# IDM Run 3 — setup & getting started

`Run3_core` is the clean, pushed Run-3 line for the inelastic-dark-matter analysis with
**electrons and muons**. This document is the current, verified setup for the coffea analysis
environment on the Fermilab LPC.

> The older `python_analysis/README.md` describes the Run-2 / `cmslpc-sl7` / conda setup
> and is **out of date** — use this file for Run 3.

## 0. Prerequisites (one time)

- A Fermilab LPC account with a valid **Kerberos ticket** (`kinit <user>@FNAL.GOV`) — needed
  both to `ssh` to LPC and to read ntuples from EOS.
- Work on a **cmslpc el9** node (`cmslpc-el9.fnal.gov`).
- Clone the repo and check out the Run-3 line:

```bash
git clone https://github.com/cms-idm/IDM.git
cd IDM
git checkout Run3_core
```

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

### Old vs. new analysis workflow

**Old (Run-2):** AOD → `AODSkimmer` (C++) → flat ntuples → an **RDataFrame pre-skim**
(`python_analysis/condor/skimming_rdf/`) that applies the major iDM preselection (MET, ≥1 jet)
to produce *small* "skimmed" ntuples → coffea analysis run **locally** in a notebook (feasible
only because the skim shrank the event yield).

**New (Run-3, this framework):** AOD → `AODSkimmer` (C++) → flat ntuples → **coffea
`IdmProcessor` run distributed over LPC HTCondor (dask)** directly on the full ntuples. The
preselection that used to live in the RDataFrame skim becomes **named cuts**
(`idm/definitions/cuts.py`) applied in the processor — so there is **no separate skim stage**.

Distributed compute (`idm.tools.scaleout` / the `condor/` harness) makes running over full
statistics tractable, the selection lives in one place (config, not a forked C++ skim), and
every output carries provenance. Only the intermediate RDataFrame skim goes away — the C++
`AODSkimmer` ntuplization is unchanged.

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
python -m pip install -e .       # register the idm package, so `import idm` works
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

Then, each session — open the SSH tunnel from your laptop **first** (it just forwards the
port; the actual connection is made when your browser connects), then start the server on
LPC. Use the **same port on both ends**:

```bash
# 1) on your laptop — forward local 8888 to the LPC login node (leave this running):
ssh -N -L 8888:localhost:8888 <user>@cmslpc-el9.fnal.gov

# 2) in a separate LPC session, with idm_venv active:
jupyter lab --no-browser --port 8888

# 3) open the printed http://localhost:8888/?token=… URL on your laptop and pick the
#    "IDM (LCG_107 Py3.11)" kernel
```

**First run — see the framework end to end:** open `idm/tutorials/01_workflow_walkthrough.ipynb`
(load an ntuple → look at objects → pick named cuts/hists → run `IdmProcessor` → make a
CMS-style plot). It defaults to the example ntuple
`root://cmseos.fnal.gov//store/group/lpcmetx/IDM_Run3/examples/tutorial_DYJets_2022PostEE.root`
(override with `$IDM_TUTORIAL_NTUPLE`). To scale this analysis out over LPC HTCondor with dask,
see `idm/tutorials/02_lpc_dask_example.ipynb`. Run-3 samples + outputs live under
`/store/group/lpcmetx/IDM_Run3/` — read over xrootd (see §4).

## 4. Reading ntuples / EOS

Read ROOT files over xrootd, not the POSIX `/eos/uscms` mount:

```
root://cmseos.fnal.gov//store/group/lpcmetx/iDMe/<…>
```

Reads need a valid Kerberos ticket; writing to EOS needs a VOMS proxy
(`voms-proxy-init --valid 192:00 -voms cms`).
