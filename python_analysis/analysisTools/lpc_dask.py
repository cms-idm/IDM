"""Run the iDMe coffea-0.7 analysis on LPC HTCondor via dask, from a notebook.

Quick start (see LPC_DASK_SETUP.md for the one-time client bootstrap)::

    from lpc_dask import make_lpc_client, ship_analysis_modules

    cluster, client = make_lpc_client(n_workers=40)
    ship_analysis_modules(client, configs=[cuts_config, histos_config])

    analyzer = Analyzer(sample_config, histos_config, cuts_config, max_samples=-1)
    out = analyzer.process(execr="dask", dask_client=client)

    client.close(); cluster.close()

SCOPE / WHAT IS NOT SUPPORTED
-----------------------------
``systematics=`` is NOT dask-safe. ``analysisTools.py`` (jsonPath) and ``corrections.py``
(corrPath) build **cwd-relative** paths into ``analysisTools/corrections/`` (~349 MB),
which does not exist on a worker (cwd=/srv, no shared filesystem), so every correctionlib
/ JEC call raises FileNotFoundError. Run systematics on the futures or iterative path
until those paths are made absolute and the JSONs are shipped.

``configs=`` ships the files you name, but NOT their transitive local imports. Several
histo configs in this repo do ``from myHisto import ...``; if yours imports another local
module, add that module to ``configs`` too (before the file that imports it). The failure
is a loud ModuleNotFoundError at ship time, not a silent wrong answer.

WHY THE WORKER LIFETIME FLAGS EXIST (read this before tuning anything)
---------------------------------------------------------------------
``iDMeProcessor.process`` retains roughly 90 MB per chunk, process-globally and immune to
``gc.collect()`` (measured: steady-state 482.6 -> 568.4 MB over consecutive chunks, with
post-gc RSS identical to pre-gc). coffea 0.7 never spans a chunk across files, so one
chunk == one file and a single long-lived process climbs without bound.

Dask does NOT fix this on its own. A distributed worker is a long-lived process serving
unbounded tasks, and the leaked memory is *unmanaged*, so dask cannot spill it: the worker
pauses at ``0.8 * memory``, stops accepting tasks, and therefore never reaches the ``0.95``
threshold that would make the nanny restart it. It parks below its condor RequestMemory,
so it does not even earn a hold; it just holds a slot. (The pause threshold and the
non-spillability are documented dask behaviour and the ~90 MB/chunk ratchet is measured;
the specific "27 chunks at 4GB" figure is arithmetic from those two, not a preserved run.)

The fix is ``--lifetime``, which recycles the worker PROCESS on a timer inside the same
condor job, so RSS resets to baseline every generation while the job itself is never
resubmitted (which also avoids LPC SYSTEM_PERIODIC_REMOVE, that deletes any job exceeding
10 restarts). Verified on real LPC condor: 5 worker generations, RSS reset each time, 88
tasks, 0 failures; and modules shipped with ``upload_file`` survive the restarts (22/22
tasks across 5 generations).

This is a WORKAROUND. The real fix is to find the retention inside the processor -- bare
``NanoEventsFactory`` materialization does not leak, so it is in the processor's own call
path. Until then, the recycling keeps memory bounded.
"""

import os
import sys
import subprocess

# Worker image. Ships coffea 0.7.31 / dask 2025.3.1 / py3.11, matching the coffea-0.7-era
# analysis code. The client MUST run inside this same image: the newest ``distributed``
# installable on py3.8 is 2023.5.0 and cannot federate with the image's 2025.3.1 (the
# worker dies on KeyError: 'nanny-plugins' and never joins).
LPC_IMAGE = (
    "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/"
    "coffea-0.7-almalinux9:0.7.31-py3.11"
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONDOR_CONFIG = os.path.join(_HERE, "lpc_condor_config")

# Order matters: client.upload_file IMPORTS each file as it lands, so a module must arrive
# only after everything it imports. Verified closed for the default processor:
# mySchema (coffea only) -> analysisSubroutines -> corrections -> analysisTools, which does
# `from mySchema import MySchema`, `import analysisSubroutines`, `import corrections`.
_MODULE_SHIP_ORDER = [
    "mySchema.py",
    "analysisSubroutines.py",
    "corrections.py",
    "analysisTools.py",
]

# Defaults from lpcjobqueue's config.yaml. worker_extra_args REPLACES these rather than
# merging, so anything we add has to carry them through or workers cannot reach the
# scheduler through the LPC firewall.
_LPC_PORT_ARGS = [
    "--worker-port", "10000:10070",
    "--nanny-port", "10070:10100",
    "--no-dashboard",
]


def check_voms_proxy(min_seconds_left=3600):
    """Return the proxy path, raising if it is missing, expiring, or unreachable.

    The proxy must NOT live in /tmp: the LPC schedd is a separate host and cannot read the
    login node's /tmp, which puts jobs on hold with code 13 subcode 2. A Jupyter kernel
    started non-interactively also does not source the profile that would normally set
    X509_USER_PROXY, so it is set explicitly here.
    """
    proxy = os.environ.get("X509_USER_PROXY") or "/tmp/x509up_u%d" % os.getuid()
    if proxy.startswith("/tmp/"):
        raise RuntimeError(
            "VOMS proxy is in /tmp (%s); the remote schedd cannot read it and jobs go on "
            "hold with code 13 subcode 2. Copy it somewhere the schedd can reach:\n"
            "    voms-proxy-init --valid 192:00 -voms cms\n"
            "    cp /tmp/x509up_u$(id -u) ~/x509up_proxy.pem\n"
            "    chmod 600 ~/x509up_proxy.pem\n"
            "then set os.environ['X509_USER_PROXY'] to that copy." % proxy
        )
    if not os.path.isfile(proxy):
        raise RuntimeError(
            "No VOMS proxy at %s. Renew with:\n"
            "    voms-proxy-init --valid 192:00 -voms cms" % proxy
        )
    try:
        result = subprocess.run(
            ["voms-proxy-info", "-file", proxy, "-timeleft"],
            capture_output=True, text=True,
        )
    except OSError as exc:
        raise RuntimeError(
            "could not run voms-proxy-info (%s). It ships in the coffea image and on the "
            "login nodes; if it is missing, your environment is not set up as expected."
            % exc
        )
    try:
        remaining = int(result.stdout.strip())
    except ValueError:
        raise RuntimeError(
            "voms-proxy-info on %s returned no parseable lifetime (rc=%s, stdout=%r, "
            "stderr=%r). Renew with: voms-proxy-init --valid 192:00 -voms cms"
            % (proxy, result.returncode, result.stdout, result.stderr)
        )
    if remaining < min_seconds_left:
        raise RuntimeError(
            "VOMS proxy at %s has only %ds left (threshold %ds). Renew with:\n"
            "    voms-proxy-init --valid 192:00 -voms cms"
            % (proxy, remaining, min_seconds_left)
        )
    os.environ["X509_USER_PROXY"] = proxy
    return proxy


def files_per_generation(memory="6GB", baseline_mib=700, leak_mib_per_file=91):
    """Estimate how many files one worker process can take before dask pauses it.

    dask parses ``memory`` as DECIMAL bytes ("6GB" -> 6e9 B -> 5722 MiB), then pauses the
    worker at 0.8x that. Getting this wrong by using GiB overestimates the budget by ~7%.
    """
    from dask.utils import parse_bytes
    pause_mib = 0.8 * parse_bytes(memory) / (1024.0 ** 2)
    return max(0.0, (pause_mib - baseline_mib) / float(leak_mib_per_file))


def make_lpc_client(
    n_workers=40,
    memory="6GB",
    disk="6GB",
    cores=1,
    lifetime="180s",
    lifetime_stagger="30s",
    death_timeout=600,
    image=LPC_IMAGE,
    condor_config=_DEFAULT_CONDOR_CONFIG,
    ship_env=False,
    worker_extra_args=None,
    **cluster_kwargs
):
    """Build an LPCCondorCluster + Client with worker-process recycling enabled.

    Args:
        n_workers: number of condor workers, via ``cluster.scale``. ``cluster.adapt`` is
            deliberately not used: adaptive scaling may read a recycling worker as a lost
            worker and churn against it (untested combination).
        memory/disk/cores: per-worker condor request. dask pauses a worker at 0.8x
            ``memory``; see :func:`files_per_generation` for the resulting file budget.
        lifetime/lifetime_stagger: worker process recycle timer, and the spread that keeps
            all workers from restarting together. Keep
            ``lifetime <= 0.6 * files_per_generation(memory) * seconds_per_file``.
            At the defaults (6GB -> ~42 files) and a measured ~9 s/file, that ceiling is
            ~230 s, so the 180 s default carries roughly 1.3x margin on time and 1.8x on
            memory.
        death_timeout: seconds a worker waits for the scheduler before giving up. Raised
            from the dask default of 60 because the LPC queue can be slow.
        image: worker apptainer image. Must match the client's coffea/dask versions.
        condor_config: CONDOR_CONFIG to export before importing lpcjobqueue. Resolved
            relative to this file by default. Inside the image this is REQUIRED, not
            cosmetic: the image has no /etc/condor, so htcondor would resolve a null
            config, FERMIHTC_REMOTE_POOL would be None, and lpcjobqueue raises
            "TypeError: expected string or bytes-like object, got 'NoneType'".
        ship_env: forwarded to LPCCondorCluster. Left False so workers use the image's
            python rather than a copy of your environment.
        worker_extra_args: overrides the default arg list entirely. If you pass this, you
            must include the LPC port ranges yourself (see ``_LPC_PORT_ARGS``) or workers
            cannot reach the scheduler.
        **cluster_kwargs: forwarded verbatim to LPCCondorCluster.

    Returns:
        ``(cluster, client)``. The caller must ``client.close(); cluster.close()`` when
        done, or the cluster keeps holding slots on a contended shared pool.
    """
    check_voms_proxy()

    # htcondor caches its config at import time, so CONDOR_CONFIG must be set BEFORE
    # lpcjobqueue (and therefore htcondor) is imported. If something already imported it,
    # say so plainly rather than producing a confusing NoneType failure later.
    if condor_config is not None:
        if not os.path.isfile(condor_config):
            raise RuntimeError("condor_config not found at %s" % condor_config)
        already = [m for m in ("htcondor", "htcondor2", "lpcjobqueue") if m in sys.modules]
        if already and os.environ.get("CONDOR_CONFIG") != str(condor_config):
            raise RuntimeError(
                "%s already imported before CONDOR_CONFIG was set, so htcondor has "
                "cached the wrong (or null) config. Restart the kernel and call "
                "make_lpc_client() BEFORE importing lpcjobqueue or htcondor."
                % ", ".join(already)
            )
        os.environ["CONDOR_CONFIG"] = str(condor_config)

    import dask
    from dask.distributed import Client
    from lpcjobqueue import LPCCondorCluster

    # lpcjobqueue rewrites the dashboard link to a jupyter-server-proxy relative route,
    # which 404s over a plain "ssh -L" tunnel. Undo it unless we really are under
    # JupyterHub. Must run AFTER the import, which is what sets it.
    if "JUPYTERHUB_SERVICE_PREFIX" not in os.environ:
        dask.config.set(
            {"distributed.dashboard.link": "{scheme}://localhost:{port}/status"}
        )

    if worker_extra_args is None:
        worker_extra_args = list(_LPC_PORT_ARGS) + [
            "--lifetime", lifetime,
            "--lifetime-stagger", lifetime_stagger,
            "--lifetime-restart",
        ]

    cluster = LPCCondorCluster(
        cores=cores,
        memory=memory,
        disk=disk,
        death_timeout=death_timeout,
        image=image,
        ship_env=ship_env,
        worker_extra_args=worker_extra_args,
        **cluster_kwargs
    )

    # Validate BEFORE submitting anything, so a failed check does not strand condor jobs
    # the caller has no handle to close.
    job_script = cluster.job_script()
    for required in ("--lifetime ", "--worker-port"):
        if required not in job_script:
            cluster.close()
            raise RuntimeError(
                "%r missing from the generated condor job script; worker recycling or "
                "scheduler connectivity would silently not work. Inspect "
                "cluster.job_script()." % required.strip()
            )

    cluster.scale(n_workers)
    client = Client(cluster)
    return cluster, client


def ship_analysis_modules(client, analysis_tools_dir=None, configs=(), verbose=True):
    """Send the analysis modules to every worker, in dependency order.

    LPC condor workers see NO shared filesystem: they run under ``apptainer exec
    --contain`` with only /cvmfs, /etc/hosts, /etc/grid-security and the job scratch dir
    bound in, cwd=/srv. /uscms, /uscms/home and /uscms_data are all absent, so a
    client-side ``sys.path.append`` means nothing there and each module has to travel.

    ``upload_file`` is used rather than ``UploadDirectory`` deliberately: these are bare
    top-level modules, not a package, so a directory upload would make them importable as
    ``analysisTools.analysisTools`` and break ``from analysisTools import Analyzer``. The
    directory is also ~350 MB, almost all of it ``corrections/``, which would then be
    broadcast to every worker.

    Args:
        client: the dask client from :func:`make_lpc_client`.
        analysis_tools_dir: directory holding the core modules. Defaults to this file's own
            directory, which is correct when lpc_dask.py sits beside analysisTools.py.
        configs: extra files to ship after the core modules, e.g. your cut and histogram
            config .py files. Shipped in the order given. Transitive local imports are NOT
            resolved: if a config does ``from myHisto import ...``, add myHisto.py here
            too, before the config that imports it.
    """
    src = analysis_tools_dir or _HERE
    to_ship = [os.path.join(src, m) for m in _MODULE_SHIP_ORDER] + list(configs)

    missing = [f for f in to_ship if not os.path.isfile(f)]
    if missing:
        raise RuntimeError("cannot ship, file(s) not found:\n  " + "\n  ".join(missing))

    for path in to_ship:
        client.upload_file(path)
        if verbose:
            print("shipped", os.path.basename(path))
    return to_ship


def expected_entries(sample_config, max_samples=-1):
    """Sum the ``num_events`` declared in a sample config, for the entries audit.

    Use this instead of ``20000 * n_files``: file sizes are NOT uniform in these ntuples.
    In ``signal_2022_Mchi_bigsample_ctau100_aEM.json`` the 15 samples declare 19,894,000
    events over 1086 files, and ``Mchi-99p0_dMchi-18p0/ctau-100`` mixes two productions
    (files at 8k and at 20k entries, averaging 11,667/file), so a flat 20k/file rule
    reports a bogus 41% shortfall on that sample alone.

    Note this is the config's own bookkeeping, not a read of the files. It is the right
    reference for "did I lose events", not an independent truth.
    """
    import json
    with open(sample_config) as handle:
        samples = json.load(handle)
    if max_samples > 0:
        samples = samples[:max_samples]
    return sum(s["num_events"] for s in samples)
