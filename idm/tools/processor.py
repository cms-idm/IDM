"""idm.tools.processor — the named-config coffea processor (the analysis engine).

How the flow works, end to end
------------------------------
1. You DEFINE analysis content **once, by name**, in ``idm/definitions/``:
     - ``objects.py``: ``obj_defs[name]``  → a function of ``events`` returning a collection
     - ``cuts.py``:    ``cut_defs[name]``   → a function of ``events`` returning an event mask
     - ``hists.py``:   ``hist_defs[name]``  → ``{"axis": <hist.axis>, "fill": f(events) → array}``
2. You SELECT which ones to run **by name** (here, or later from a YAML config), e.g.::

       p = IdmProcessor(cuts=["has_muon"], hists=["muon_pt", "n_dsamuon"])

3. ``process(events)`` AND-s the chosen cuts into one event selection, fills the chosen
   histograms from the selected events, and returns ``{"hists": {...}, "cutflow": {...}}``.

So a *study* changes **configuration** (which names to run) instead of copying analysis code.

This is a small, working skeleton. The object/cut/hist *definitions* are filled in as the
existing ``python_analysis/`` logic is ported and as each channel adds its selections. A
runnable end-to-end example is in ``idm/README.md``.
"""

import awkward as ak
import hist
from coffea import processor

from idm.definitions.cuts import cut_defs
from idm.definitions.hists import hist_defs


class IdmProcessor(processor.ProcessorABC):
    """A named-config coffea processor.

    Args:
        cuts: list of cut names (keys of ``idm.definitions.cuts.cut_defs``) AND-ed together
            into the event selection. ``None``/empty → no selection (keep all events).
        hists: list of histogram names (keys of ``idm.definitions.hists.hist_defs``) to fill.
            ``None`` → fill every defined histogram.

    Unknown names raise immediately (a typo should fail loudly, not silently do nothing).
    """

    def __init__(self, cuts=None, hists=None):
        self._cuts = list(cuts) if cuts else []
        self._hists = list(hists) if hists is not None else list(hist_defs.keys())
        for c in self._cuts:
            if c not in cut_defs:
                raise KeyError(f"unknown cut '{c}'; defined cuts: {sorted(cut_defs)}")
        for h in self._hists:
            if h not in hist_defs:
                raise KeyError(f"unknown hist '{h}'; defined hists: {sorted(hist_defs)}")

    def process(self, events):
        # 1) build the event selection by AND-ing the named cuts, recording a cutflow
        sel = events
        cutflow = {"all": int(ak.num(events, axis=0))}
        for c in self._cuts:
            sel = sel[cut_defs[c](sel)]
            cutflow[c] = int(ak.num(sel, axis=0))

        # 2) fill the named histograms from the selected events
        out_hists = {}
        for name in self._hists:
            spec = hist_defs[name]
            h = hist.Hist(spec["axis"])
            values = spec["fill"](sel)
            # ak.ravel → a flat 1D array, so a fill lambda may return a per-event OR
            # per-object quantity without the definition having to flatten it itself.
            h.fill(ak.to_numpy(ak.ravel(values)))
            out_hists[name] = h

        return {"hists": out_hists, "cutflow": cutflow}

    def postprocess(self, accumulator):
        return accumulator
