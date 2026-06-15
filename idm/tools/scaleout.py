"""Scale-out helpers: run the coffea analysis distributed over LPC HTCondor.

Adapted from the SIDM analysis (sidm/tools/scaleout.py) for IDM. The client (your
notebook / login-node process) stays in ``idm_venv``; workers run as Condor jobs
inside the coffea-dask apptainer image. The local ``idm/`` tree is shipped to each
worker via ``UploadDirectory`` so uncommitted edits are visible without a git push.
"""

import os
import subprocess
from pathlib import Path

from dask.distributed import Client, PipInstall


def make_dask_client(address):
    """Create a dask client with a pip-install plugin (coffea-casa style)."""
    dependencies = [
        "git+https://github.com/cms-idm/IDM.git",
    ]
    client = Client(address)
    client.register_plugin(
        PipInstall(packages=dependencies, pip_options=["--upgrade", "--no-cache-dir"])
    )
    return client


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_LPC_IMAGE = (
    "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/"
    "coffea-dask-almalinux9:2025.5.0.rc2-py3.11"  # matches idm_venv (coffea 2025.5.0rc2, py3.11)
)
_DEFAULT_IDM_LOCAL_DIR = _REPO_ROOT / "idm"
_DEFAULT_LPC_CONFIG = _REPO_ROOT / "condor" / "lpc_condor_config"  # optional; see make_lpc_client
_PROXY_RENEW_CMD = "voms-proxy-init --valid 192:00 -voms cms"


def check_voms_proxy(min_seconds_left=3600):
    """Verify a CMS VOMS proxy exists and is not about to expire.

    Inspects ``$X509_USER_PROXY`` (or ``/tmp/x509up_u<UID>`` if unset), asks
    ``voms-proxy-info`` how much time is left, and raises ``RuntimeError`` (with the
    renewal command) if the proxy is missing or expiring within ``min_seconds_left``.
    Exports ``X509_USER_PROXY`` so LPCCondorCluster / condor_submit see the same file.
    Returns the proxy path on success.
    """
    proxy = os.environ.get("X509_USER_PROXY") or f"/tmp/x509up_u{os.getuid()}"
    if not os.path.isfile(proxy):
        raise RuntimeError(
            f"No VOMS proxy found at {proxy}. Renew it on cmslpc with:\n  {_PROXY_RENEW_CMD}"
        )
    result = subprocess.run(
        ["voms-proxy-info", "-file", proxy, "-timeleft"],
        capture_output=True, text=True,
    )
    try:
        remaining = int(result.stdout.strip())
    except ValueError:
        raise RuntimeError(
            f"voms-proxy-info on {proxy} did not return a parseable lifetime "
            f"(stdout={result.stdout!r}, stderr={result.stderr!r}). Renew with:\n  {_PROXY_RENEW_CMD}"
        )
    if remaining < min_seconds_left:
        hrs = remaining / 3600.0
        raise RuntimeError(
            f"VOMS proxy at {proxy} has only {remaining}s left ({hrs:.1f}h, "
            f"threshold {min_seconds_left}s). Renew with:\n  {_PROXY_RENEW_CMD}"
        )
    os.environ["X509_USER_PROXY"] = proxy
    return proxy


def make_lpc_client(
    min_workers=1,
    max_workers=10,
    memory="4GB",
    disk="4GB",
    cores=1,
    death_timeout=600,
    image=_DEFAULT_LPC_IMAGE,
    idm_local_dir=_DEFAULT_IDM_LOCAL_DIR,
    condor_config=None,
    **cluster_kwargs,
):
    """Create an LPCCondorCluster + Client to scale IDM jobs from a notebook on cmslpc.

    Workers run as Condor jobs inside the coffea-dask apptainer image (``image``);
    the client process stays in ``idm_venv`` outside the apptainer. The local
    ``idm/`` tree is uploaded to each worker so uncommitted edits are picked up
    without a git push.

    Args:
        min_workers/max_workers: passed to ``cluster.adapt()``.
        memory/disk/cores: per-worker resources.
        death_timeout: seconds a worker waits for the scheduler before self-terminating
            (raised from the dask default of 60 because LPC condor queues can be slow).
        image: apptainer image for workers; default matches ``idm_venv`` (coffea 2025.5.0rc2).
            Other coffea versions live under
            ``/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/``.
        idm_local_dir: local ``idm/`` source uploaded to workers (None to skip).
        condor_config: optional path to a CONDOR_CONFIG file. If None, the node default
            is used. A tuned LPC interactive config can be added under ``condor/`` later.
        **cluster_kwargs: forwarded to ``LPCCondorCluster``.

    Returns:
        ``(cluster, client)``. The caller is responsible for ``cluster.close()``.
    """
    check_voms_proxy()

    # htcondor caches its config at import time, so CONDOR_CONFIG must be set first.
    if condor_config is not None:
        os.environ["CONDOR_CONFIG"] = str(condor_config)

    from lpcjobqueue import LPCCondorCluster
    from distributed.diagnostics.plugin import UploadDirectory

    cluster = LPCCondorCluster(
        memory=memory,
        disk=disk,
        cores=cores,
        death_timeout=death_timeout,
        image=image,
        ship_env=False,
        **cluster_kwargs,
    )
    cluster.adapt(minimum=min_workers, maximum=max_workers)
    client = Client(cluster)

    if idm_local_dir is not None:
        client.register_plugin(
            UploadDirectory(str(idm_local_dir), restart_workers=False)
        )

    return cluster, client
