# idm condor batch

Run the `idm` coffea analysis over many ntuples on LPC HTCondor. Each job builds a fresh,
throwaway virtualenv on the worker (from `condor/requirements.txt`), runs one chunk of files
through `IdmProcessor`, and copies a `.coffea` + `.meta.yaml` provenance sidecar to EOS.
(For interactive distributed running from a notebook instead, use `idm.tools.scaleout`.)

## Files
- `run_idm_chunk.py` — runs `IdmProcessor` over a filelist → `.coffea` + `.meta.yaml`. Runs
  locally too (handy for testing): `python condor/run_idm_chunk.py --sample s --filelist f.txt --output o.coffea`.
- `run_job.sh` — the per-job wrapper (LCG_107 base → venv → pip install → run chunk → xrdcp).
- `make_job_args.py` — chunk per-sample filelists into jobs + write `job_args.txt`.
- `submit.sub` — the HTCondor submit description (**edit the EOS output dir** first).
- `requirements.txt` — the per-job venv pins (kept in sync with the repo-root `requirements.txt`).

## How to submit (from LPC)
```bash
# 0) a valid grid proxy is required for EOS writes:
voms-proxy-init --valid 192:00 -voms cms

# 1) put your input filelists in condor/filelists_in/<sample>.txt (one ROOT file per line)

# 2) make the per-chunk job args:
python condor/make_job_args.py --filelists-dir condor/filelists_in --files-per-job 5

# 3) tar the analysis code the workers need (idm/ + the chunk runner + the schema):
tar -czf condor/idm_code.tar.gz idm condor/run_idm_chunk.py \
    python_analysis/analysisTools/mySchema_newCoffea.py

# 4) edit the EOS output dir in submit.sub, then submit from inside condor/:
cd condor && mkdir -p logs && condor_submit submit.sub
```
Re-tar `idm_code.tar.gz` after any change under `idm/` or to `run_idm_chunk.py` before resubmitting.
