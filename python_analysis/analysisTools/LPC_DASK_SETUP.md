# Running the coffea analysis on LPC HTCondor with dask

This lets `Analyzer.process(execr="dask", ...)` run over the full ntuple set on LPC batch
instead of on the interactive node, which is what currently dies with `BrokenProcessPool`.

You do not need to merge anything. Copy two files and apply three one-line fixes.

These live on branch `Menon27_allLPT_dask`, which branches from `Menon27_allLPT` and adds
only the files described here:

```bash
git fetch origin
git checkout origin/Menon27_allLPT_dask -- \
  python_analysis/analysisTools/lpc_dask.py \
  python_analysis/analysisTools/lpc_condor_config \
  python_analysis/analysisTools/LPC_DASK_SETUP.md
```

or copy them straight off LPC, where they are world-readable:

    /uscms_data/d3/murtazas/IDM/python_analysis/analysisTools/

## What is actually going wrong

`iDMeProcessor.process` retains about **90 MB per chunk**, process-globally, and
`gc.collect()` reclaims none of it (measured: steady state 482.6 -> 568.4 MB over
consecutive chunks, post-gc RSS identical to pre-gc). coffea 0.7 never spans a chunk across
files, so **one chunk == one file** and any single process climbs ~90 MB per file without
bound. A cmslpc interactive node has 11.4 GB total, no per-user cgroup cap, and typically
~4 GB free, so the kernel OOM killer reaps the process. A kernel SIGKILL is exactly what
surfaces as `BrokenProcessPool` instead of a Python traceback, and because the leak is
linear the death point is a reproducible chunk count. That is why it works on a subset and
always dies around the same fraction.

Two things this is **not**: it is not the accumulator (all 15 samples come to 1.94 MB of
histograms), and it is not xrootd (an I/O failure raises a normal Python exception with a
full traceback, which is not what you saw).

**Adding dask does not fix this by itself.** A dask worker is a long-lived process serving
unbounded tasks, and leaked memory is *unmanaged*, so dask cannot spill it: the worker
pauses at `0.8 * memory`, stops accepting tasks, and therefore never reaches the `0.95`
threshold that would trigger a nanny restart. It then parks below its condor RequestMemory,
so it does not even earn a hold, it just silently holds a slot. That would trade a loud
crash for a silent hang.

What fixes it is `--lifetime`, which recycles the worker **process** on a timer inside the
same condor job, resetting RSS every generation while the job itself is never resubmitted.
`make_lpc_client()` sets this up for you. Verified on real LPC condor: 5 worker generations,
RSS reset each time, 88 tasks, 0 failures, and shipped modules surviving every restart.

This is a workaround. The real fix is to find the retention inside the processor; bare
`NanoEventsFactory` materialization does not leak, so it is in the processor's own call
path. Until someone does that, recycling keeps memory bounded.

## Step 0: copy two files and fix three lines

Copy these into your own `python_analysis/analysisTools/`:

```bash
SRC=/uscms_data/d3/murtazas/IDM/python_analysis/analysisTools
DEST=~/nobackup/IDMe_Run3_Collab/CMSSW_13_0_13/src/iDMe/python_analysis/analysisTools
cp $SRC/lpc_dask.py $SRC/lpc_condor_config $SRC/LPC_DASK_SETUP.md $DEST/
```


- **`lpc_dask.py`** — the cluster helper. It has to sit beside `analysisTools.py`, because
  it defaults to shipping the modules from its own directory.
- **`lpc_condor_config`** — a 22-line HTCondor config, copied verbatim from `Run3_core`'s
  `condor/lpc_condor_config`. It mirrors `/etc/condor/config.d/01_cmslpc_interactive` but
  drops an `include` directive pointing at a per-user file that does not exist on all
  cmslpc-el9 nodes. `make_lpc_client()` looks for it next to `lpc_dask.py`.

Then apply these three one-line fixes to your `analysisTools.py`:

1. **Delete the `if info['type'] == "signal":` line** in `iDMeProcessor.process` (near line
   480, the one whose body is entirely commented out). Right now that `if` has no body, so
   the file raises `IndentationError` at the following line and **cannot be imported at
   all** in any executor. Deleting the `if` restores the behaviour the file had before,
   minus a `print("Good!")` you had already removed. `cutDesc[cutName] += ...` was always
   outside that `if`, so nothing else changes.
2. **Line 142: `root://cmsxrootd.fnal.gov/` -> `root://cmseos.fnal.gov/`.** You already made
   this change at line 147, but line 147 is the `type(loc) == list` branch and your sample
   configs use a plain string `location`, so execution takes line 142 and your fix never
   engaged. Both doors work; this just makes the string branch match what you intended.
3. **Line 905 in `getLumi`: `if year == 2022:` -> `if year == '2022':`.** The function does
   `year = str(year)` a few lines above, so the 2022 branch was comparing a string to an int
   and never fired. `getLumi(2022)` returned `(0, 0)`, which silently zeroed `cutflow_cts`
   in every 2022 output. See the note in "Things that will bite you".

Or take the whole file, which differs from your `Menon27_allLPT` copy by exactly those
three changes and nothing else:

```bash
SRC=/uscms_data/d3/murtazas/IDM/python_analysis/analysisTools
DEST=~/nobackup/IDMe_Run3_Collab/CMSSW_13_0_13/src/iDMe/python_analysis/analysisTools

diff $SRC/analysisTools.py $DEST/analysisTools.py     # check first: should be the 3 changes
cp   $SRC/analysisTools.py $DEST/analysisTools.py
```

## Step 1: a VOMS proxy the schedd can read

The proxy must **not** stay in `/tmp`. The LPC schedd is a separate host and cannot read
the login node's `/tmp`, which puts jobs on hold with code 13 subcode 2.

```bash
voms-proxy-init --valid 192:00 -voms cms
cp /tmp/x509up_u$(id -u) ~/x509up_proxy.pem
chmod 600 ~/x509up_proxy.pem
```

`lpc_dask` refuses to start if the proxy is in `/tmp`, missing, or expiring within the hour.

## Step 2: build a client venv inside the worker image

The client must run **inside the same image as the workers**. The newest `distributed`
installable on Python 3.8 is 2023.5.0, the image ships 2025.3.1, and the two cannot federate
(the worker dies with `KeyError: 'nanny-plugins'` and never joins). This is about the dask
wire protocol, not your analysis code.

```bash
export IMG=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-0.7-almalinux9:0.7.31-py3.11
export BINDS="-B /uscms_data -B /cvmfs -B /etc/grid-security -B /uscmst1b_scratch -B /uscms/home -B /uscms/homes"

mkdir -p ~/nobackup/dask_client && cd ~/nobackup/dask_client
apptainer exec $BINDS $IMG bash -c '
  python -m venv --system-site-packages .env
  source .env/bin/activate
  pip install --no-cache-dir "setuptools==70.3.0" "lpcjobqueue @ git+https://github.com/CoffeaTeam/lpcjobqueue.git@v0.6.0"
  python -c "import lpcjobqueue; print(lpcjobqueue.__version__)"
  python -m ipykernel install --user --name lpc_dask --display-name "LPC dask (in-image venv)"'
```

The kernel it registers records the **absolute** path of the venv interpreter, so if you
later move or delete `~/nobackup/dask_client` the kernel breaks with no hint in JupyterLab.
Re-run the `ipykernel install` line if you relocate it.

**This can take 20-30 minutes on a busy node** and prints nothing useful for most of it
(measured: 28 minutes, of which `python -m venv`'s ensurepip sat in uninterruptible disk
wait for over 17 minutes while /uscms_data was delivering 4.4 MB/s). It is not hung. Let it
finish; a Ctrl-C here leaves a half-built venv you will have to delete. It must print
`0.6.0` at the end.

`lpcjobqueue` prints `Condor configuration not found!` and suggests running `bootstrap.sh`
just before that. **Ignore it.** The image has no `/etc/condor`, and `make_lpc_client()`
sets `CONDOR_CONFIG` for you. Running `bootstrap.sh` would build a different environment.
The line that matters is the `0.6.0` right after.

Four things here are load-bearing:

- **lpcjobqueue must be v0.6.0.** Earlier versions do a bare `import htcondor`, and this
  image ships htcondor 25.11 with only the v2 bindings (`htcondor2`/`classad2`), so they die
  with `ModuleNotFoundError: No module named 'htcondor'`. v0.6.0 adds the shim.
- **`-B /uscmst1b_scratch` is required**, or `LPCCondorCluster.__init__` raises
  `OSError: [Errno 30] Read-only file system: '/uscmst1b_scratch'`.
- **`-B /uscms/homes` is required.** `/uscms/home/<user>` is a *symlink* into `/uscms/homes`
  (yours is `/uscms/home/reshmar -> /uscms/homes/r/reshmar`), so binding only `/uscms/home`
  leaves `~` dangling inside the container. apptainer then cannot even set the working
  directory (`WARNING: Error changing the container working directory. Using '/' instead`),
  the venv is created on the read-only image root, and pip silently falls back to an
  unreachable `~/.local`, so lpcjobqueue never installs.
- **setuptools must be pinned to 70.3.0.** `python -m venv` installs setuptools 79.0.1 into
  `.env`, shadowing the image's 70.3.0. The image's uproot 4.3.7 still calls
  `setuptools.extern.packaging`, which was removed after setuptools 71, so without the pin
  every client-side file open dies with
  `AttributeError: module 'setuptools' has no attribute 'extern'`. That breaks `futures` and
  `iterative` in the client venv; `dask` survives only because uproot runs on the workers.

## Step 3: launch Jupyter inside the image

Forward both ports in the *same* ssh so they land on the same load-balanced node:

```bash
ssh -L 8888:localhost:8888 -L 8787:localhost:8787 cmslpc-el9.fnal.gov
```

Then on LPC, start `tmux` **first** so a dropped ssh or a sleeping laptop does not kill a
multi-hour run and orphan 40 condor slots:

```bash
tmux new -s dask
```

Now, *inside that tmux shell*, export the variables and launch. The exports must happen
inside tmux: if a tmux server is already running, a new session inherits the **server's**
environment rather than your current shell's, so variables exported outside can arrive stale
or empty and the command degenerates to `apptainer exec bash -c ...`.

```bash
export IMG=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-0.7-almalinux9:0.7.31-py3.11
export BINDS="-B /uscms_data -B /cvmfs -B /etc/grid-security -B /uscmst1b_scratch -B /uscms/home -B /uscms/homes"

apptainer exec $BINDS $IMG bash -c '
  source ~/nobackup/dask_client/.env/bin/activate
  jupyter lab --no-browser --port=8888 --ip=127.0.0.1 --port-retries=0'
```

Detach with `Ctrl-b d`, reattach later with `tmux attach -t dask`.

`--port-retries=0` makes Jupyter fail loudly if 8888 is taken. Without it, it silently moves
to 8889+ and your `ssh -L 8888:localhost:8888` would tunnel to whatever else is on 8888,
possibly another user's server. If it does fail, pick a free port and use the same number in
both the `ssh -L` and the `--port` argument.

**In JupyterLab, select the "LPC dask (in-image venv)" kernel, not the default
"Python 3 (ipykernel)".** The image's default kernelspec hardcodes `/usr/local/bin/python`,
which cannot see the venv, so the default kernel fails at `make_lpc_client()` with
`ModuleNotFoundError: No module named 'lpcjobqueue'`. Activating the venv fixes the Lab
*server*, not the kernel; that is what the `ipykernel install` line in step 2 is for.

## Step 4: use it in the notebook

Set `REPO` to your own checkout and the rest follows. This block runs as pasted.

```python
import os, sys, json

REPO = "/uscms/home/reshmar/nobackup/IDMe_Run3_Collab/CMSSW_13_0_13/src/iDMe"
AT   = os.path.join(REPO, "python_analysis/analysisTools")
TP   = os.path.join(REPO, "python_analysis/thesis_plots/electron_xclean")

os.environ["X509_USER_PROXY"] = os.path.expanduser("~/x509up_proxy.pem")
sys.path.append(AT)

from lpc_dask import make_lpc_client, ship_analysis_modules, expected_entries
from analysisTools import Analyzer
import coffea.util as util

sample_config = os.path.join(REPO, "python_analysis/configs/sample_configs/signal_2022_Mchi_bigsample_ctau100_aEM.json")
cuts_config   = os.path.join(TP, "cut_configs/cuts_DSCB.py")
histos_config = os.path.join(TP, "histo_configs/histo_for_fig20-Lxy.py")
outdir        = os.path.join(REPO, "python_analysis/coffea/AllInOne_2022")
os.makedirs(outdir, exist_ok=True)

cluster, client = make_lpc_client(n_workers=40)
print("dashboard:", cluster.dashboard_link)

try:
    # On a busy pool the first worker can take 10-15 minutes to appear (measured 822 s).
    # A long pause here is queueing, not a hang.
    client.wait_for_workers(1, timeout=1800)

    ship_analysis_modules(client, configs=[cuts_config, histos_config])

    analyzer = Analyzer(sample_config, histos_config, cuts_config, max_samples=-1)
    out = analyzer.process(execr="dask", dask_client=client)

    acc, metrics = out                   # savemetrics=True gives a 2-tuple
    print("entries:", metrics["entries"], "expected:", expected_entries(sample_config))

    util.save(out, os.path.join(outdir, "Big_Resolution_dask.coffea"))
finally:
    client.close(); cluster.close()      # must run even if the wait times out,
                                         # or the condor jobs sit holding slots
```

If the printed dashboard port is not 8787 (it is often already taken on a shared login
node), open a second terminal with `ssh -L PORT:localhost:PORT cmslpc-el9.fnal.gov` using
the port it actually reported.

Your analysis code itself does not change, **with one exception**: the `systematics=` path
is not dask-safe. `analysisTools.py` and `corrections.py` build cwd-relative paths into
`analysisTools/corrections/` (~349 MB), which does not exist on a worker (cwd=/srv, no
shared filesystem), so every correctionlib/JEC call raises `FileNotFoundError`. Keep
systematics runs on futures or iterative until those paths are made absolute and the JSONs
are shipped.

## Bring it up in stages

Do not jump straight to 1086 files.

1. **2 files, futures.** Skip the `make_lpc_client` / `wait_for_workers` /
   `ship_analysis_modules` lines entirely — futures runs in your kernel and needs no
   cluster, and pasting them would submit 40 condor workers you never use. Set
   `max_samples=1, max_files_per_samp=2` in `Analyzer(...)` and use
   `execr="futures", workers=2`. Save as `Big_Resolution_futures.coffea`. This proves the
   in-image kernel runs your code at all. (The stack move itself is already checked: the
   same files under `execr="iterative"` give byte-identical histograms and cutflows in your
   py3.8 conda env and in the py3.11 image, so this is a plumbing check, not a physics one.)
2. **Same files, dask.** `make_lpc_client(n_workers=2)`, saving as
   `Big_Resolution_dask.coffea` so step 1's output survives for comparison. A missing-module
   failure shows up immediately here, before any physics. Use `max_samples=3,
   max_files_per_samp=1` for this step rather than one sample: with a single sample the
   `samp` axis trivially matches and the ordering hazard below cannot appear, whereas at
   three samples it shows up at once (observed: futures gave
   `['105p0','11p0','110p0']`, dask gave `['11p0','105p0','110p0']`). Compare against step 1
   on **histogram contents per named `(samp, cut)` cell**, never on raw arrays positionally,
   and not on cutflows alone (they are sums over the whole sample and cannot detect this).
3. **150 files.** `max_files_per_samp=10, max_samples=-1`. Watch the dashboard Workers tab:
   each worker's memory should **sawtooth**, climbing then dropping back to ~0.3 GB every
   few minutes. That sawtooth is the recycling working. If memory instead climbs
   monotonically and workers go orange or grey, check `print(cluster.job_script())` contains
   `--lifetime`.
4. **Full run.** Drop `max_files_per_samp`, `n_workers=40`.

## Before the full run: fix the resolution pairing

This one is not about dask, but it will decide whether the 1086-file run is worth doing.
In `histo_for_fig20-Lxy.py`, the residual subtracts two arrays that are ordered by different
things:

```python
Gen_pt_FLAT = ak.flatten(ak.concatenate([GenEle_pt[:, None], GenPos_pt[:, None]], axis=1))
Lpt_pt_flat = ak.flatten(AllLptElectron.pt[AllLptElectron.genMatched])
res_LPT     = (Lpt_pt_flat - Gen_pt_FLAT) / Gen_pt_FLAT
```

`Gen_pt_FLAT` is always `[electron, positron]` by construction. `Lpt_pt_flat` is in reco
collection order, which is pT-ordered. Pairing them by position assumes the leading reco
electron matched the gen *electron*, and nothing enforces that. Measured on
`Mchi-105p0_dMchi-10p0/ctau-100_00.root` (260 events pass the selection), the positron's
reco partner sits first in **51.2%** of events, so about half the entries subtract two
different particles. The same applies to `res_GED` with the `Electron` collection.

| pairing | mean | RMS | \|res\| > 0.3 |
| --- | --- | --- | --- |
| positional, `res_LPT` | +0.1276 | 0.7938 | 35.8% |
| index-based, `res_LPT` | -0.0336 | 0.1047 | 3.5% |
| positional, `res_GED` | +0.1381 | 0.8010 | 35.6% |
| index-based, `res_GED` | -0.0253 | 0.0971 | 2.3% |

The ntuple already carries the correct link. `GenEle.matchIdxAllLowPt` indexes into
`AllLptElectron`, and `GenEle.matchIdxLocal` indexes into `Electron` for this selection.
Replace the block from `Gen_pt = ...` through `res_LPT = ...` with:

```python
ev = events_new_very

# Gen side keeps its [electron, positron] order.
Gen_pt_FLAT  = ak.flatten(ak.concatenate(
    [ev.GenEle.pt[:, None],  ev.GenPos.pt[:, None]],  axis=1))
Gen_vxy_FLAT = ak.flatten(ak.concatenate(
    [ev.GenEle.vxy[:, None], ev.GenPos.vxy[:, None]], axis=1))

# Reco side: take each gen particle's OWN match by index, so the two arrays line up.
lpt, ged = ev.AllLptElectron.pt, ev.Electron.pt
Lpt_pt_flat = ak.flatten(ak.concatenate([
    lpt[ak.singletons(ev.GenEle.matchIdxAllLowPt)],
    lpt[ak.singletons(ev.GenPos.matchIdxAllLowPt)]], axis=1))
GED_pt_flat = ak.flatten(ak.concatenate([
    ged[ak.singletons(ev.GenEle.matchIdxLocal)],
    ged[ak.singletons(ev.GenPos.matchIdxLocal)]], axis=1))

res_GED = (GED_pt_flat - Gen_pt_FLAT) / (Gen_pt_FLAT)
res_LPT = (Lpt_pt_flat - Gen_pt_FLAT) / (Gen_pt_FLAT)
```

Everything downstream stays as it is: the fills still use `res_GED`, `res_LPT` and
`Gen_vxy_FLAT`, and the lengths still line up (all four arrays are 2 per selected event).

Two things worth knowing rather than taking on trust:

- **The Lxy binning is unaffected.** `GenEle.vxy` and `GenPos.vxy` are identical to machine
  precision because both come from the same decay vertex, so entries never moved between Lxy
  bins. Only the residual value was wrong.
- **`matchIdxLocal` is right for this selection specifically.** The mask pins
  `matchType == 'R'`, and for that population `matchIdxLocal` and `matchIdxGlobal` agree
  exactly and are both in range. If you change the selection to admit `matchType == 'L'`,
  re-check which index addresses which collection before reusing this.

Numbers above are one file, so treat the ratios as indicative rather than final. The
direction and the ~50% swap rate are not in doubt.

## Things that will bite you

**The entries audit only means anything on the full run.** While `max_files_per_samp` is
set, `expected_entries()` still reports the whole dataset, so staging steps show an
enormous bogus shortfall (at `max_samples=1, max_files_per_samp=2` it prints
`entries: 40000 expected: 19894000`). Ignore the `expected:` number until step 4.

**Audit `metrics["entries"]`.** Compare against `expected_entries(sample_config)`, which
sums the config's declared `num_events` — **not** `20000 * n_files`. File sizes are not
uniform: this config declares 19,894,000 events over 1086 files, and
`Mchi-99p0_dMchi-18p0/ctau-100` mixes two productions (files at 8k and at 20k entries,
averaging 11,667/file), so a flat 20k/file rule reports a bogus 41% shortfall on that sample
alone. A genuine shortfall means files were dropped.

**Sample-axis ordering is completion-ordered, and already is today.** The growth
`StrCategory` `samp` axis takes its category order from whichever chunk finishes first.
This is true of `FuturesExecutor` with `workers>1` as well as dask (coffea accumulates with
`FIRST_COMPLETED`); only `IterativeExecutor` is order-deterministic, and even then the order
is a property of that run, not something to rely on across coffea versions. Contents per named category
are bit-identical, but the raw arrays are permuted along axis 0. **Index histograms by
name.** Anything that indexes positionally, zips against a fixed sample list, or trusts
`list(h.axes["samp"])` order will silently scramble. This is invisible with one sample and
appears with three.

**Do not add `maxchunks` without re-checking which files run.** In coffea 0.7.21
`fileset.reverse()` reverses the flat *file* list, not just the dataset list; 0.7.31 differs.
`analysisTools.py` builds `processor.Runner(...)` without `maxchunks`, so today this only
permutes the sample axis and changes nothing else. The moment anyone adds `maxchunks`, the
two coffea versions would select *different files*, and the comparison above stops being
apples to apples.

**Old and new `.coffea` files are interchangeable.** A file written by `coffea.util.save`
in the image (py3.11 / coffea 0.7.31 / hist 2.10.1) loads in your py3.8 / coffea 0.7.21 /
hist 2.8.1 env with zero byte differences across all cutflows, all `(samp, cut)` cells and
`metrics["entries"]`, and plots normally with `mplhep`. So you can keep plotting in the
environment you already have.

**Bad files abort the run.** `Runner.skipbadfiles` defaults to False, so over 1086 files one
unreadable file kills the job. `DaskExecutor` does retry transient xrootd hiccups 3 times on
its own, so a single blip is already covered; what is not covered is a genuinely bad file.
If you set `skipbadfiles=True`, bad files are dropped *silently* and the entries audit is
your only signal.

**A short file list also fails silently.** `loadFiles()` discards the XRootD status from
`xrdClient.dirlist(loc)` and never checks the result length, so a failed or partial listing
just yields fewer files with no error anywhere. Another reason to run the entries audit.

**Your existing 2022 `.coffea` files have `cutflow_cts` identically zero.** `getLumi` did
`year = str(year)` and then tested `if year == 2022:` against an *int*, so the 2022 branch
never fired and every xsec*lumi-weighted count came out 0.0 (`cutflow_nevts` and `cutflow`
were unaffected). The copy of `analysisTools.py` referenced in step 0 fixes this
(`getLumi(2022)` now returns `(38.01, 0.53214)`), but anything you already produced for
2022 needs regenerating before those counts mean anything. There is still no 2023 entry.

**Always tear the cluster down** with `client.close(); cluster.close()`. An open cluster
keeps holding slots on a heavily contended shared pool.

**Keep worker-visible functions in imported `.py` files, never in a notebook cell.**
Functions defined in a cell pickle by value, which genuinely does fail across Python
versions. Anything imported from a module pickles by reference and is fine.

**Chunksize is not a lever.** `Runner`'s default is already 100000 and your files are
around 20k entries, so you are already at one chunk per file, the maximum coffea 0.7 can
produce. Raising it does nothing; *lowering* it to 5000 raised measured peak RSS from 1044
to 2255 MB and added 57% more I/O. Leave it alone.

**Do not raise `workers` on the futures path as a stopgap.** Each extra worker adds its own
~380 MB baseline plus its own leak, so more workers fail *earlier*, not later.

## If something goes wrong

| Symptom | Cause |
| --- | --- |
| Hold code 13.2, "reading from file /tmp/x509up_..." | proxy left in `/tmp`; see step 1 |
| `TypeError: expected string or bytes-like object, got 'NoneType'` at `schedd.py` | `CONDOR_CONFIG` not set before `import lpcjobqueue`; the image has no `/etc/condor`. Restart the kernel and call `make_lpc_client()` first |
| `ModuleNotFoundError: No module named 'htcondor'` | lpcjobqueue older than v0.6.0 against htcondor 25 |
| `OSError: [Errno 30] Read-only file system: '/uscmst1b_scratch'` | missing `-B /uscmst1b_scratch` bind |
| `ModuleNotFoundError` for one of your own modules on a worker | shipped out of dependency order, or a transitive local import not listed in `configs=` |
| `FileNotFoundError` under `.../analysisTools/corrections/...` | ran with `systematics=` under dask; not supported yet |
| Workers never appear | port ranges missing from `worker_extra_args`; check `print(cluster.job_script())` |
| Workers stall, no error, memory flat near the limit | the pause-not-restart failure; `--lifetime` is not engaging |
| Jobs vanish from the queue | LPC `SYSTEM_PERIODIC_REMOVE` deletes jobs after 10 restarts |

**Orphaned jobs.** If the kernel dies with the cluster still up, `close()` never runs and
the jobs sit in the queue holding slots. Clean up from any login node:

```bash
condor_q                          # see what is still yours
condor_rm <cluster_id>            # remove one submission
condor_rm $USER                   # remove all of yours, use with care
```

## Fallback if dask keeps fighting you

One fresh process per N files under plain condor is *structurally* immune to the leak rather
than merely bounded against it: a job running `execr="iterative"` over 5 files peaks around
1.2 GB against a 4 GB request, with no timer to tune, and it survives a dropped connection
because there is no scheduler living in your notebook kernel. It costs a merge step
afterwards with `coffea.processor.accumulate`. `Run3_core`'s `condor/` directory has a
working harness of this shape to crib from. Worth switching to if the in-image client is not
working within about half a day, or if you need runs to survive overnight.
