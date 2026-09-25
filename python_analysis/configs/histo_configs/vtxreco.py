try:
    from histobins import *
except ModuleNotFoundError:
    from configs.hists.histobins import *
from hist import Hist
import hist
import numpy as np
import awkward as ak
from analysisTools.analysisSubroutines import defineGoodVertices, defineGoodLptVertices, defineGoodMergedCandidates

# Vertex reco efficiency vs gen-level ee quantities. For each variable these hists
# are filled at every plotted cut stage:
#   vtxreco_<name>_den : all events
#   vtxreco_lptvtx_<name>_num : events with >= 1 good lptvtx vertex (defineGoodLptVertices,
#                v15acr -- same cuts as hasGoodVertex / cut9 in an_selection.py, applied
#                to the all-low-pT lptvtx pool)
#   vtxreco_lptvtx_<name>_num_genmatch : events with >= 1 good lptvtx vertex that is also
#                gen-matched (lptvtx.isMatched, i.e. both vertex electrons matched to
#                the gen e+e-)
#   vtxreco_vtx_<name>_num : events with >= 1 good vtx vertex (defineGoodVertices,
#                v15acr -- the mixed vtx collection used by hasGoodVertex / cut9)
#   vtxreco_vtx_<name>_num_genmatch : events with >= 1 good vtx vertex that is also
#                gen-matched (vtx.isMatched)
#   vtxreco_merged_<name>_num : events with >= 1 good merged candidate
#                (merged_vtx from defineGoodMergedCandidates)
#   vtxreco_merged_<name>_num_genmatch : events with >= 1 good merged candidate that is
#                also gen-matched (merged_vtx.isGenMerged: matched to both gen electrons,
#                event in cat_merged)
# so that eff = <num hist> / vtxreco_<name>_den per bin (the denominator is shared).
# Evaluate at a cut stage *before* the good-vertex cut -- afterwards every event has
# a good vertex and eff is trivially 1.

# --- local axes (50 bins each) ---
gen_ee_pt     = Regular(50, 0, 50,     name="pt",  label="Gen $p_{T}(e^+e^-)$ [GeV]")
gen_lxy       = Regular(50, 0, 25,     name="lxy", label="Gen $L_{xy}$ [cm]")
gen_lxy_log   = Regular(50, 1e-3, 100, name="lxy", label="Gen $L_{xy}$ [cm]", transform=transform.log)
gen_ee_dr     = Regular(50, 0, 2,      name="dr",  label=r"Gen $\Delta R(e^+e^-)$")
gen_ee_dr_log = Regular(50, 1e-4, 10,  name="dr",  label=r"Gen $\Delta R(e^+e^-)$", transform=transform.log)
gen_ee_eta    = Regular(50, -2.5, 2.5, name="eta", label=r"Gen $\eta(e^+e^-)$")

VARIABLES = {
    # hist name       : (axis,          fill kwarg)
    "gen_ee_pt"       : (gen_ee_pt,     "pt"),
    "gen_lxy"         : (gen_lxy,       "lxy"),
    "gen_lxy_log"     : (gen_lxy_log,   "lxy"),
    "gen_ee_dr"       : (gen_ee_dr,     "dr"),
    "gen_ee_dr_log"   : (gen_ee_dr_log, "dr"),
    "gen_ee_eta"      : (gen_ee_eta,    "eta"),
}

# numerator hist prefix/suffix -> key into the event masks built in fillHistos
NUMERATORS = {
    ("vtxreco_lptvtx", "num")          : "lptvtx",
    ("vtxreco_lptvtx", "num_genmatch") : "lptvtx_genmatch",
    ("vtxreco_vtx",    "num")          : "regvtx",
    ("vtxreco_vtx",    "num_genmatch") : "regvtx_genmatch",
    ("vtxreco_merged", "num")          : "merged",
    ("vtxreco_merged", "num_genmatch") : "merged_genmatch",
}

# 2D gen Lxy vs gen ee pT (coarse variable bins), same num/den scheme:
#   vtxreco_gen_lxy_vs_pt_den, <prefix>_gen_lxy_vs_pt_<suffix>
gen_lxy_2d   = Variable([0, 1, 5, 10, 15, 100], name="lxy", label="Gen $L_{xy}$ [cm]")
gen_ee_pt_2d = Variable([0, 5, 10, 20, 100],    name="pt",  label="Gen $p_{T}(e^+e^-)$ [GeV]")
NAME_2D = "gen_lxy_vs_pt"

def make_histograms():
    histograms = {}
    for name, (axis, _) in VARIABLES.items():
        histograms[f"vtxreco_{name}_den"] = Hist(samp, cut, axis, storage=hist.storage.Weight())
        for prefix, suffix in NUMERATORS:
            histograms[f"{prefix}_{name}_{suffix}"] = Hist(samp, cut, axis, storage=hist.storage.Weight())
    histograms[f"vtxreco_{NAME_2D}_den"] = Hist(samp, cut, gen_lxy_2d, gen_ee_pt_2d, storage=hist.storage.Weight())
    for prefix, suffix in NUMERATORS:
        histograms[f"{prefix}_{NAME_2D}_{suffix}"] = Hist(samp, cut, gen_lxy_2d, gen_ee_pt_2d, storage=hist.storage.Weight())
    return histograms

subroutines = []

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    if info['type'] != "signal":
        return

    wgt = events.eventWgt/sum_wgt

    # Lxy is measured from the chi2 production vertex (true PV) rather than the
    # reco PV -- see genstudy.py / elerecoeff.py.
    chi2 = ak.firsts(events.GenPart[np.abs(events.GenPart.ID) == 1000023])
    lxy = np.sqrt((events.genEE.vx - chi2.vx)**2 + (events.genEE.vy - chi2.vy)**2)

    values = {
        "gen_ee_pt"     : events.genEE.pt,
        "gen_lxy"       : lxy,
        "gen_lxy_log"   : lxy,
        "gen_ee_dr"     : events.genEE.dr,
        "gen_ee_dr_log" : events.genEE.dr,
        "gen_ee_eta"    : events.genEE.eta,
    }

    defineGoodVertices(events, version='v15acr')
    defineGoodLptVertices(events, version='v15acr')
    defineGoodMergedCandidates(events)
    masks = {
        "lptvtx"          : events.nGoodLptVtx > 0,
        "lptvtx_genmatch" : ak.any(events.good_lptvtx.isMatched, axis=1),
        "regvtx"          : events.nGoodVtx > 0,
        "regvtx_genmatch" : ak.any(events.good_vtx.isMatched, axis=1),
        "merged"          : events.nMergedVtx > 0,
        "merged_genmatch" : ak.any(events.merged_vtx.isGenMerged, axis=1),
    }

    for name, (_, key) in VARIABLES.items():
        val = values[name]
        hists[f"vtxreco_{name}_den"].fill(samp=samp, cut=cut, **{key: val}, weight=wgt)
        for (prefix, suffix), mask_key in NUMERATORS.items():
            m = masks[mask_key]
            hists[f"{prefix}_{name}_{suffix}"].fill(samp=samp, cut=cut, **{key: val[m]}, weight=wgt[m])

    pt = events.genEE.pt
    hists[f"vtxreco_{NAME_2D}_den"].fill(samp=samp, cut=cut, lxy=lxy, pt=pt, weight=wgt)
    for (prefix, suffix), mask_key in NUMERATORS.items():
        m = masks[mask_key]
        hists[f"{prefix}_{NAME_2D}_{suffix}"].fill(samp=samp, cut=cut, lxy=lxy[m], pt=pt[m], weight=wgt[m])
