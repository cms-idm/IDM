"""Cut / selection definitions (named-config DSL).

Each entry maps a NAME to a pure function of ``events`` returning a per-event boolean mask.
The processor (``idm.tools.processor.IdmProcessor``) AND-s the selected cuts into one event
selection. A YAML config can later compose named cuts into named selections (as SIDM does).

Channel-specific selections (electron ID, muon DSA ID, displaced-vertex cuts, ...) are added
by the channel owners as that logic is ported from ``python_analysis/``.
"""

import awkward as ak

# --- EXAMPLE cuts (basic, for the framework demo / tutorial; replace with real ones) ---
cut_defs = {
    "has_electron": lambda e: ak.num(e.Electron, axis=1) >= 1,  # >= 1 PF electron
    "has_muon":     lambda e: ak.num(e.Muon, axis=1) >= 1,      # >= 1 PF muon
}
