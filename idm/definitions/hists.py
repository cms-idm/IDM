"""Histogram definitions (named-config DSL).

Each entry maps a NAME to a spec describing a ``hist.Hist`` axis (or axes) and the
quantity to fill from ``events``. A YAML hist-collection config (mirroring SIDM's
``configs/hist_collections.yaml``) groups named histograms into collections selected
per study, so plotting/booking is configured by name rather than by editing code.

Additive scaffolding; ``python_analysis/configs/histo_configs`` remains the source of
truth until ported. Plots produced from these should follow the shared CMS plotting
conventions in ``conventions/plotting.md`` (lint via ``conventions/lint_plots.py``).

Example shape:

    hist_defs = {
        "dimuon_mass": {"axis": ("m", 100, 0, 10, "m(mumu) [GeV]"),
                         "fill": lambda evts: ...},
    }
"""

# Add histogram definitions here.
hist_defs = {}
