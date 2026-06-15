"""Cut / selection definitions (named-config DSL).

Each entry maps a NAME to a pure function of ``events`` returning a boolean mask
(event-level) or a per-object mask. A YAML selection config composes cuts by name
into named selections (mirroring SIDM's ``configs/selections.yaml``).

Additive scaffolding; the authoritative selection logic currently lives in
``python_analysis/`` (cut configs + ``analysisSubroutines.py``) and is ported here
incrementally. The published SR selections (ISR jet + MET + displaced di-lepton
vertex, etc.) are documented in ``ANALYSIS_ROADMAP.md`` and ``OBJECTS_Run3.md``.

Example shape:

    cut_defs = {
        "met_200":     lambda evts: evts.PFMET.pt > 200,
        "lead_jet_80": lambda evts: ak.firsts(evts.PFJet.pt) > 80,
    }
"""

# Add cut definitions here. Keep each a pure function of `events`.
cut_defs = {}
