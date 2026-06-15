"""Histogram definitions (named-config DSL).

Each entry maps a NAME to ``{"axis": <hist.axis>, "fill": f(events) -> array}``. The processor
builds a ``hist.Hist`` from ``axis`` and fills it with ``fill(selected_events)``. The fill may
return a per-event OR a per-object array — the processor flattens it — so a definition does not
have to worry about jaggedness.

Plots produced from these should follow CMS plotting conventions: build them with the
``idm.tools.plotting`` helpers and lint plotting scripts with ``python -m idm.tools.lint_plots``.
Per-study histograms are added as the analysis is ported.
"""

import awkward as ak
import hist

# --- EXAMPLE histograms (basic kinematics, for the framework demo / tutorial) ----------
hist_defs = {
    "n_electron": {
        "axis": hist.axis.Regular(8, 0, 8, name="n", label="N(electrons)"),
        "fill": lambda e: ak.num(e.Electron, axis=1),          # per-event count
    },
    "electron_pt": {
        "axis": hist.axis.Regular(50, 0, 100, name="pt", label=r"electron $p_{T}$ [GeV]"),
        "fill": lambda e: e.Electron.pt,                       # per-object (jagged), auto-flattened
    },
    "muon_pt": {
        "axis": hist.axis.Regular(50, 0, 100, name="pt", label=r"PF muon $p_{T}$ [GeV]"),
        "fill": lambda e: e.Muon.pt,                           # per-object (jagged), auto-flattened
    },
}
