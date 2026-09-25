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
    desc = r"$\vec{p}_T^{miss}$ HLT and Offline > 200 GeV"
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

    events = events[cut]
    cut = events.PFMET.pt > 200
    return events[cut], name, desc, plots

def cut3(events,info):
    name = "cut3"
    desc = "Jets Cuts"
    plots = True
    nJets = ak.count(events.PFJet.pt,axis=1)
    # drop events with no jets before taking the leading jet
    events = events[nJets > 0]

    bTag = events.PFJet.bTag
    # DeepFlavour working points for UL samples
    loose,med,tight = getBtagWPs(info["year"]) 
    wp = med # MEDIUM BTAG WP
    
    pass_bTag = bTag > wp
    n_bTag_Jets = ak.count(events.PFJet[pass_bTag].pt,axis=1)

    jets = events.PFJet
    sortJetsByPt = ak.argsort(jets.pt, ascending=False)
    j1 = jets[sortJetsByPt][:, 0]
    
    cut = (n_bTag_Jets == 0) & (j1.pt > 80.) & (np.abs(j1.eta) < 2.4) & (np.abs(j1.METdPhi) > 1.5) & (ak.min(abs(jets.METdPhi), axis=1) > 0.75)
    return events[cut], name, desc, plots
    
def cut4(events, info):
    events = hasGoodVertex(events, info)
    name = 'cut4' 
    desc = 'Has Good ee Vertex' 
    plots = True 
    return events, name, desc, plots

def cut5(events,info):
    name = "cut5"
    desc = "Other vertex Cuts"
    plots = True
    cut = (events.sel_vtx.sign == -1) & (events.sel_vtx.cos_collinear > 0.4) & (events.sel_vtx.reduced_chi2 < 3.)
    return events[cut], name, desc, plots

def cut6(events, info):
    name = "cut6"
    desc = "BDT Approx. Cuts"
    plots = True

    part1 = (events.sel_vtx.refit_m < 7.) & (events.sel_vtx.refit_dR < 1.)
    part2 = (np.minimum(np.abs(events.sel_vtx.e1_refit_dxy), np.abs(events.sel_vtx.e2_refit_dxy)) > 0.15)
    part3 = (events.sel_vtx.vxy_fromPV > 1.) & (_dphi(events.sel_vtx.phi, events.PFMET.phi) < 1.)
    part4 = (events.sel_vtx.e1.eta * events.sel_vtx.e2.eta > 0.)
    
    cut = part1 & part2 & part3 & part4
    return events[cut], name, desc, plots

