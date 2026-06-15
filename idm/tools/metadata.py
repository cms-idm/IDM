"""Provenance sidecars for ``.coffea`` outputs.

``write_run_metadata`` writes a ``.meta.yaml`` next to a ``.coffea`` recording what
produced it: the input ROOT fileset, the git commit, coffea version, schema, chunk
size, a UTC timestamp, and (optionally) the named selections / histogram collections
used. ``load_run_metadata`` reads it back. Adapted from SIDM (sidm/tools/metadata.py);
kept self-contained so it works before the configs/definitions are fully built out.
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import coffea
import yaml


def write_run_metadata(
    coffea_path,
    *,
    fileset,
    schema=None,
    chunksize=None,
    selections=None,
    hist_collections=None,
    extra=None,
    idm_root=None,
    created_utc=None,
):
    """Write a ``<name>.meta.yaml`` sidecar describing what produced ``<name>.coffea``.

    Args:
        coffea_path: path to the ``.coffea`` output (sidecar is ``foo.coffea`` -> ``foo.meta.yaml``).
        fileset: coffea-style ``{sample: {"files": [...], "metadata": {...}}}`` dict.
        schema: optional schema name (e.g. ``"MySchema"``).
        chunksize: optional chunk size.
        selections/hist_collections: optional lists of named selection / hist-collection
            names recorded verbatim (their full definitions can be expanded once
            ``idm/configs`` exists; kept as plain names here).
        extra: optional dict merged into the sidecar (e.g. cluster id, dashboard url).
        idm_root: optional repo-root override (default walks up from this file).
        created_utc: optional timestamp string; if None, stamped at call time.

    Returns:
        Path to the written sidecar.
    """
    idm_root = Path(idm_root) if idm_root else _find_idm_root()

    samples = []
    for name, info in fileset.items():
        files = list(info.get("files", [])) if isinstance(info, dict) else list(info)
        samples.append({
            "name": name,
            "n_files": len(files),
            "metadata": _yaml_safe(info.get("metadata", {})) if isinstance(info, dict) else {},
            "files": files,
        })

    meta = {
        "created_utc": created_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "coffea_path": str(coffea_path),
        "n_samples": len(samples),
        "schema": schema,
        "chunksize": chunksize,
        "selections": list(selections) if selections else [],
        "hist_collections": list(hist_collections) if hist_collections else [],
        "idm_commit": _git_rev(idm_root),
        "coffea_version": coffea.__version__,
        "samples": samples,
    }
    if extra:
        meta.update(extra)

    sidecar = _sidecar_path(coffea_path)
    with open(sidecar, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, default_flow_style=False, width=200)
    return sidecar


def load_run_metadata(coffea_path):
    """Load and return the ``.meta.yaml`` sidecar next to ``coffea_path``."""
    with open(_sidecar_path(coffea_path)) as f:
        return yaml.safe_load(f)


def _sidecar_path(coffea_path):
    p = str(coffea_path)
    if p.endswith(".coffea"):
        p = p[: -len(".coffea")]
    return p + ".meta.yaml"


def _find_idm_root():
    return Path(__file__).resolve().parent.parent.parent


def _git_rev(repo_root):
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def _yaml_safe(obj):
    """Make a structure safe for ``yaml.safe_dump`` (no Python-specific tags)."""
    if isinstance(obj, dict):
        return {k: _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(x) for x in obj]
    return obj
