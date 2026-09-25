import numpy as np
import awkward as ak
from analysisTools.analysisSubroutines import getBtagWPs, hasGoodVertex

def _dphi(a, b):
    d = np.abs(a - b)
    return ak.where(d > np.pi, 2 * np.pi - d, d)

def cut1(events,info):
    name = "cut1"
    desc = r"Pass $\vec{p}_T^{miss}$ Filters"
    plots = True
    cut = events.METFiltersFailBits == 0
    return events[cut], name, desc, plots

def cut2(events,info):
    name = "cut2"
    desc = r"$\vec{p}_T^{miss}$ Trigger (120 GeV)"
    plots = True

    if info["year"] == 2016:
        cut = (events.trigFired16 & (1<<9)) == (1<<9)
    if info["year"] == 2017:
        cut = (events.trigFired17 & (1<<9)) == (1<<9)
    if info["year"] == 2022:
        cut = (events.trigFired18 & (1<<13)) == (1<<13)
    if info["year"] == 2024:
        #for f in events.trig.fields:
        #    if 'HLT' in f and 'PFMET' in f:
        #        print(f)
        #cut = (events.trig.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60)
        if 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight' not in events.trig.fields:
            for f in events.trig.fields:
                if 'HLT_PFMET' in f:
                    print(f)
            
        if 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60' in events.trig.fields and 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight' in events.trig.fields:
            cut = (events.trig.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight |
                   events.trig.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60)
        elif 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60' in events.trig.fields:
            cut = events.trig.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60
        elif 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight' in events.trig.fields:
            cut = events.trig.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
        
    return events[cut], name, desc, plots

def cut3(events,info):
    name = "cut3"
    desc = r"$\vec{p}_T^{miss}$ > 200 GeV"
    plots = True
    cut = events.PFMET.pt > 200
    return events[cut], name, desc, plots

def cut4(events,info):
    name = "cut4"
    desc = "nJets > 0 (pT > 30 GeV)"
    plots = False
    nJets = ak.count(events.PFJet.pt,axis=1)
    cut = (nJets > 0) 
    return events[cut], name, desc, plots

def cut5(events,info):
    # UL b-tag threshold recommendations from here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
    # using the medium WP, as in Andre's version of iDM
    name = "cut5"
    desc = "No b-tagged jets"
    plots = False
    bTag = events.PFJet.bTag

    # DeepFlavour working points for UL samples
    loose,med,tight = getBtagWPs(info["year"]) 
    wp = med # MEDIUM BTAG WP
    
    pass_bTag = bTag > wp
    n_bTag_Jets = ak.count(events.PFJet[pass_bTag].pt,axis=1)
    cut = n_bTag_Jets == 0
    return events[cut], name, desc, plots

def cut6(events, info):
    name = 'cut6'
    desc = r'J1 $p_T > 80$ GeV, $|\eta| < 2.4$'
    plots = False
    jets = events.PFJet
    sortJetsByPt = ak.argsort(jets.pt, ascending=False)
    j1 = jets[sortJetsByPt][:, 0]
    cut = (j1.pt > 80.) & (np.abs(j1.eta) < 2.4)
    return events[cut], name, desc, plots
    
def cut7(events, info):
    name = 'cut7'
    desc = r'$|\Delta \phi(\vec{p}_T^{miss},$ J1$)| > 1.5$'
    plots = False
    jets = events.PFJet
    sortJetsByPt = ak.argsort(jets.pt, ascending=False)
    j1 = jets[sortJetsByPt][:, 0]
    cut = np.abs(j1.METdPhi) > 1.5
    return events[cut], name, desc, plots
    
def cut8(events, info):
    name = 'cut8'
    desc = r'min$_{j \in \mathrm{jets}} |\Delta \phi(\vec{p}_T^{miss}, j)| > 0.75$'
    plots = True
    jets = events.PFJet
    cut = ak.min(abs(jets.METdPhi), axis=1) > 0.75
    return events[cut], name, desc, plots
    
def cut9(events, info):
    events = hasGoodVertex(events, info)
    name = 'cut9' 
    desc = 'Has Good ee Vertex' 
    plots = True 
    return events, name, desc, plots

def cut10(events,info):
    name = "cut10"
    desc = "SV(ee) OSSF"
    plots = False
    cut = events.sel_vtx.sign == -1
    return events[cut], name, desc, plots

def cut11(events,info):
    name = "cut11"
    desc = r"SV(ee) $cos(\theta_{coll}) > 0.4$"
    plots = False
    cut = events.sel_vtx.cos_collinear > 0.4
    return events[cut], name, desc, plots

def cut12(events,info):
    name = "cut12"
    desc = r"SV(ee) $\chi^2/ndf < 3$"
    plots = False
    cut = events.sel_vtx.reduced_chi2 < 3.
    return events[cut], name, desc, plots

def cut13(events, info):
    name = "cut13"
    desc = "Refit Vtx Mass < 7 GeV"
    plots = False
    cut = events.sel_vtx.refit_m < 7.
    return events[cut], name, desc, plots

def cut14(events, info):
    name = "cut14"
    desc = "Refit Vtx $\Delta R(ee) < 1$"
    plots = False
    cut = events.sel_vtx.refit_dR < 1.
    return events[cut], name, desc, plots

def cut15(events, info):
    name = "cut15"
    desc = "Vtx min $d_{xy} >0.15$ cm"
    plots = False
    cut = np.minimum(np.abs(events.sel_vtx.e1_refit_dxy), np.abs(events.sel_vtx.e2_refit_dxy)) > 0.15
    return events[cut], name, desc, plots

def cut16(events, info):
    name = "cut16"
    desc = "Vtx $L_{xy} >1$ cm"
    plots = False
    cut = events.sel_vtx.vxy_fromPV > 1.
    return events[cut], name, desc, plots

def cut17(events, info):
    name = "cut17"
    desc = "$\Delta \phi(ee, p_T^{miss}) < 1$"
    plots = False
    dphi_ee_met = _dphi(events.sel_vtx.phi, events.PFMET.phi)
    cut = dphi_ee_met < 1.
    return events[cut], name, desc, plots

def cut18(events, info):
    name = "cut18"
    desc = "$\eta(e_1)\times\eta(e_2) > 0$"
    plots = True
    cut = events.sel_vtx.e1.eta * events.sel_vtx.e2.eta > 0.
    return events[cut], name, desc, plots
