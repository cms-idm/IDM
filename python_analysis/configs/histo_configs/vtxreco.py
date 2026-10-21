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
#   vtxreco_lptvtx_<name>_num_genmatch_any : events with >= 1 gen-matched lptvtx vertex,
#                good or not (lptvtx.isMatched, i.e. both vertex electrons matched to
#                the gen e+e-)
#   vtxreco_lptvtx_<name>_num_genmatch_good : events with >= 1 gen-matched good lptvtx
#                vertex (defineGoodLptVertices, v15acr -- same cuts as hasGoodVertex /
#                cut9 in an_selection.py, applied to the all-low-pT lptvtx pool)
#   vtxreco_vtx_<name>_num_genmatch_any : events with >= 1 gen-matched vtx vertex, good
#                or not (vtx.isMatched)
#   vtxreco_vtx_<name>_num_genmatch_good : events with >= 1 gen-matched good vtx vertex
#                (defineGoodVertices, v15acr -- the mixed vtx collection used by
#                hasGoodVertex / cut9)
#   vtxreco_merged_<name>_num_genmatch_any : events with >= 1 gen-matched AllLptElectron
#                merged candidate, good or not (AllLptElectron.isGenMerged: matched to
#                both gen electrons, event in cat_merged)
#   vtxreco_merged_<name>_num_genmatch_good : events with >= 1 gen-matched good merged
#                candidate (merged_vtx from defineGoodMergedCandidates)
#   vtxreco_eereco_<name>_num_<pair> : di-electron reco efficiency -- events where both
#                gen electrons are gen-matched to reco electrons (no vertex requirement),
#                split by the matched collections (GenEle/GenPos.matchType, R = GED,
#                L = LPT):
#                  gedged   : (GED, GED)
#                  gedlpt   : (GED, LPT), either ordering
#                  lptlpt   : (LPT, LPT)
#                  sum      : any of the three above
#                  alptalpt : both matched to AllLptElectron (matchedAllLowPt)
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
    ("vtxreco_lptvtx", "num_genmatch_any")  : "lptvtx_genmatch_any",
    ("vtxreco_lptvtx", "num_genmatch_good") : "lptvtx_genmatch_good",
    ("vtxreco_vtx",    "num_genmatch_any")  : "regvtx_genmatch_any",
    ("vtxreco_vtx",    "num_genmatch_good") : "regvtx_genmatch_good",
    ("vtxreco_merged", "num_genmatch_any")  : "merged_genmatch_any",
    ("vtxreco_merged", "num_genmatch_good") : "merged_genmatch_good",
    ("vtxreco_eereco", "num_gedged")   : "eereco_gedged",
    ("vtxreco_eereco", "num_gedlpt")   : "eereco_gedlpt",
    ("vtxreco_eereco", "num_lptlpt")   : "eereco_lptlpt",
    ("vtxreco_eereco", "num_sum")      : "eereco_sum",
    ("vtxreco_eereco", "num_alptalpt") : "eereco_alptalpt",
}

# 2D gen Lxy vs gen ee pT (coarse variable bins), split into coarse gen ee dR bins
# via a third axis (one eff map per dR bin), same num/den scheme:
#   vtxreco_gen_lxy_vs_pt_den, <prefix>_gen_lxy_vs_pt_<suffix>
gen_lxy_2d   = Variable([0, 1, 5, 10, 15, 100],   name="lxy", label="Gen $L_{xy}$ [cm]")
gen_ee_pt_2d = Variable([0, 5, 10, 20, 100],      name="pt",  label="Gen $p_{T}(e^+e^-)$ [GeV]")
gen_ee_dr_2d = Variable([0, 0.01, 0.1, 1, 5],     name="dr",  label=r"Gen $\Delta R(e^+e^-)$")
NAME_2D = "gen_lxy_vs_pt"

def make_histograms():
    histograms = {}
    for name, (axis, _) in VARIABLES.items():
        histograms[f"vtxreco_{name}_den"] = Hist(samp, cut, axis, storage=hist.storage.Weight())
        for prefix, suffix in NUMERATORS:
            histograms[f"{prefix}_{name}_{suffix}"] = Hist(samp, cut, axis, storage=hist.storage.Weight())
    histograms[f"vtxreco_{NAME_2D}_den"] = Hist(samp, cut, gen_lxy_2d, gen_ee_pt_2d, gen_ee_dr_2d, storage=hist.storage.Weight())
    for prefix, suffix in NUMERATORS:
        histograms[f"{prefix}_{NAME_2D}_{suffix}"] = Hist(samp, cut, gen_lxy_2d, gen_ee_pt_2d, gen_ee_dr_2d, storage=hist.storage.Weight())
    return histograms

subroutines = []

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    if info['type'] != "signal":
        return

    # All hists are filled with raw (unweighted) event counts, so eff = num/den is the
    # plain fraction of signal events; eventWgt and sum_wgt are unused.
    wgt = np.ones(len(events))

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
    # AllLptElectron (and the lptvtx/merged_vtx pools built from it) isn't present in
    # every ntuple era (e.g. 2018) -- skip those pieces and fill their masks as all-False
    # rather than crashing, so the shared vtx-based hists still get filled.
    has_lpt = 'AllLptElectron' in events.fields
    if has_lpt:
        defineGoodLptVertices(events, version='v15acr')
        defineGoodMergedCandidates(events)
    no_lpt = np.zeros(len(events), dtype=bool)

    # di-electron reco: both gen electrons matched to a reco electron (R = GED, L = LPT)
    ele_ged = events.GenEle.matchType == 'R'
    ele_lpt = events.GenEle.matchType == 'L'
    pos_ged = events.GenPos.matchType == 'R'
    pos_lpt = events.GenPos.matchType == 'L'
    gedged = ele_ged & pos_ged
    gedlpt = (ele_ged & pos_lpt) | (ele_lpt & pos_ged)
    lptlpt = ele_lpt & pos_lpt
    has_alpt_match = 'matchedAllLowPt' in events.GenEle.fields
    alptalpt = (events.GenEle.matchedAllLowPt & events.GenPos.matchedAllLowPt) if has_alpt_match else no_lpt

    masks = {
        "lptvtx_genmatch_any"  : ak.any(events.lptvtx.isMatched, axis=1) if has_lpt else no_lpt,
        "lptvtx_genmatch_good" : ak.any(events.good_lptvtx.isMatched, axis=1) if has_lpt else no_lpt,
        "regvtx_genmatch_any"  : ak.any(events.vtx.isMatched, axis=1),
        "regvtx_genmatch_good" : ak.any(events.good_vtx.isMatched, axis=1),
        "merged_genmatch_any"  : ak.any(events.AllLptElectron.isGenMerged, axis=1) if has_lpt else no_lpt,
        "merged_genmatch_good" : ak.any(events.merged_vtx.isGenMerged, axis=1) if has_lpt else no_lpt,
        "eereco_gedged"        : gedged,
        "eereco_gedlpt"        : gedlpt,
        "eereco_lptlpt"        : lptlpt,
        "eereco_sum"           : gedged | gedlpt | lptlpt,
        "eereco_alptalpt"      : alptalpt,
    }

    for name, (_, key) in VARIABLES.items():
        val = values[name]
        hists[f"vtxreco_{name}_den"].fill(samp=samp, cut=cut, **{key: val}, weight=wgt)
        for (prefix, suffix), mask_key in NUMERATORS.items():
            m = masks[mask_key]
            hists[f"{prefix}_{name}_{suffix}"].fill(samp=samp, cut=cut, **{key: val[m]}, weight=wgt[m])

    pt = events.genEE.pt
    dr = events.genEE.dr
    hists[f"vtxreco_{NAME_2D}_den"].fill(samp=samp, cut=cut, lxy=lxy, pt=pt, dr=dr, weight=wgt)
    for (prefix, suffix), mask_key in NUMERATORS.items():
        m = masks[mask_key]
        hists[f"{prefix}_{NAME_2D}_{suffix}"].fill(samp=samp, cut=cut, lxy=lxy[m], pt=pt[m], dr=dr[m], weight=wgt[m])
