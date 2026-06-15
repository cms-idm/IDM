"""Object definitions (named-config DSL).

Each entry maps a NAME to a pure function of ``events`` returning a derived/selected collection.
A config selects objects by name, so the same processor runs many object definitions without
code edits. The authoritative object logic currently lives in
``python_analysis/analysisTools/analysisSubroutines.py`` and is ported here incrementally;
channel-specific selections (electron x-cleaning, muon DSA displaced ID, ...) are owned by the
respective channel and added as that work lands.
"""

# --- EXAMPLE objects (passthroughs, to show the pattern; replace with real selections) ---
obj_defs = {
    "electrons": lambda e: e.Electron,
    "pf_muons":  lambda e: e.Muon,
}
