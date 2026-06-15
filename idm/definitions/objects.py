"""Object definitions (named-config DSL).

Each entry maps a NAME to a pure function of ``events`` returning a derived/selected collection.
NOTE: ``obj_defs`` is not yet consumed by ``IdmProcessor`` — cuts/hists currently access the
collections directly (``e.Electron`` / ``e.Muon``). These are placeholders for when object-
selection wiring lands. The authoritative object logic still lives in
``python_analysis/analysisTools/analysisSubroutines.py`` and is ported here incrementally;
channel-specific selections (electron x-cleaning, muon DSA displaced ID, ...) are owned by the
respective channel and added as that work lands.
"""

# --- EXAMPLE objects (passthroughs, to show the pattern; replace with real selections) ---
obj_defs = {
    "electrons": lambda e: e.Electron,
    "pf_muons":  lambda e: e.Muon,
}
