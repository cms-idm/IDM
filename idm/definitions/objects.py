"""Object definitions (named-config DSL).

Each entry maps a NAME to a pure function of ``events`` (a coffea NanoEvents record
built with the IDM schema) returning a derived/masked collection. A config selects
objects by name, so the same processor runs many object definitions without code edits.

This is additive scaffolding for the coffea-2025 flow. The authoritative object logic
currently lives in ``python_analysis/analysisTools/analysisSubroutines.py`` and is
ported here incrementally. Channel-specific selections (e.g. muon DSA displaced ID,
PF<->DSA cross-clean) are owned by the respective channel and added as that work lands.

Example shape (illustrative; fill in as the analysis is ported):

    obj_defs = {
        "PFElectron":   lambda evts: evts.Electron[evts.Electron.IDmva90 == 1],
        "LowPtElectron": lambda evts: evts.LptElectron,
        # "PFMuon":  lambda evts: evts.Muon[...],      # TODO: muon channel
        # "DSAMuon": lambda evts: evts.DSAMuon[...],   # TODO: muon channel (DSA displaced ID)
    }
"""

# Add object definitions here. Keep each a pure function of `events`.
obj_defs = {}
