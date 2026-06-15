#!/bin/bash
# Per-job wrapper for IDM condor batch: build a throwaway venv from requirements.txt on the
# worker (in condor scratch), run one chunk with run_idm_chunk.py, and xrdcp the output to EOS.
set -euo pipefail
set -x

SAMPLE=$1
CHUNK=$2
FILELIST=$3
EOS_OUTDIR=$4
CUTS=${5:-}        # optional: comma-separated cut names
HISTS=${6:-}       # optional: comma-separated hist names (empty = all)

hostname; date
cd "${_CONDOR_SCRATCH_DIR}"
ls -lah

# 1. cvmfs LCG_107 base (Python 3.11), matching idm_venv
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh

# 2. throwaway venv in scratch; do NOT keep the LCG PYTHONPATH (it shadows the pinned versions)
python3 -m venv idm_venv
unset PYTHONPATH
source idm_venv/bin/activate
export PIP_CACHE_DIR="${_CONDOR_SCRATCH_DIR}/pip_cache"
python -m pip install --upgrade pip -q
python -m pip install --no-cache-dir -r requirements.txt

# 3. verify the stack before running
python - <<'PY'
import coffea, awkward, uproot, hist, yaml
print("coffea", coffea.__version__, "- imports OK")
PY

# 4. unpack the analysis code (idm/ incl. schema.py + condor/run_idm_chunk.py) and run the chunk
tar -xzf idm_code.tar.gz
export PYTHONPATH="${_CONDOR_SCRATCH_DIR}:${PYTHONPATH:-}"
FILELIST_BASENAME=$(basename "${FILELIST}")
OUTFILE="${SAMPLE}_${CHUNK}.coffea"
META="${SAMPLE}_${CHUNK}.meta.yaml"

python condor/run_idm_chunk.py \
    --sample "${SAMPLE}" \
    --filelist "${FILELIST_BASENAME}" \
    --output "${OUTFILE}" \
    --cuts "${CUTS}" \
    --hists "${HISTS}"

# 5. copy outputs to EOS
xrdcp -f "${OUTFILE}" "${EOS_OUTDIR}/${OUTFILE}"
xrdcp -f "${META}"    "${EOS_OUTDIR}/${META}" || echo "warning: no sidecar to copy"
rm -f "${OUTFILE}" "${META}"

date; echo "Done"
