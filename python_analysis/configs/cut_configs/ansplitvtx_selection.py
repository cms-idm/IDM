import numpy as np
import awkward as ak
from analysisTools.analysisSubroutines import getBtagWPs, hasGoodVertex, selectBestVertex

def cut1(events,info):
    name = "cut1"
    desc = r"Pass $\vec{p}_T^{miss}$ Filters"
    plots = False
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
    plots = True
    nJets = ak.count(events.PFJet.pt,axis=1)
    cut = (nJets > 0) 
    return events[cut], name, desc, plots

def cut5(events,info):
    # UL b-tag threshold recommendations from here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
    # using the medium WP, as in Andre's version of iDM
    name = "cut5"
    desc = "No b-tagged jets"
    plots = True
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
    plots = True
    jets = events.PFJet
    sortJetsByPt = ak.argsort(jets.pt, ascending=False)
    j1 = jets[sortJetsByPt][:, 0]
    cut = (j1.pt > 80.) & (np.abs(j1.eta) < 2.4)
    return events[cut], name, desc, plots
    
def cut7(events, info):
    name = 'cut7'
    desc = r'$|\Delta \phi(\vec{p}_T^{miss},$ J1$)| > 1.5$'
    plots = True
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

"""
def defineGoodVertices(events,version='v9',ele_id='dR'):
    # Selecting electrons that pass basic pT and eta cuts                                                                                                                              
    if ele_id == 'basic':
        IDcut = events.vtx.e1.passIDBasic & events.vtx.e2.passIDBasic
    if ele_id == 'dR':
        IDcut = events.vtx.e1.passID & events.vtx.e2.passID
    ossf = events.vtx.sign == -1
    chi2 = events.vtx.reduced_chi2 < 5
    chi2_loose = events.vtx.reduced_chi2 < 15
    mass = events.vtx.refit_m < 20
    eleDphi = events.vtx.eleDphi < 2
    mindxy = events.vtx.min_dxy > 0.01
    mindxy_refit = np.minimum(np.abs(events.vtx.e1.refit_dxy), np.abs(events.vtx.e2.refit_dxy)) > 0.001
    mindxy_refit_tight = np.minimum(np.abs(events.vtx.e1.refit_dxy), np.abs(events.vtx.e2.refit_dxy)) > 0.01
    mindxyLoose = events.vtx.min_dxy > 0.005
    maxMiniIso = np.maximum(events.vtx.e1.miniRelIsoEleCorr,events.vtx.e2.miniRelIsoEleCorr) < 0.9
    passConvVeto = events.vtx.e1.conversionVeto & events.vtx.e2.conversionVeto
    mass_lo = events.vtx.m > 0.325
    mass_lo_refit = events.vtx.refit_m > 0.1
    logdxydz = np.minimum(np.log10(np.abs(events.vtx.e1.dxy/events.vtx.e1.dz)), np.log10(np.abs(events.vtx.e2.dxy/events.vtx.e2.dz))) > -1.25
    logdxydz_loose = np.minimum(np.log10(np.abs(events.vtx.e1.dxy/events.vtx.e1.dz)), np.log10(np.abs(events.vtx.e2.dxy/events.vtx.e2.dz))) > -2
    if version == 'v15acr':
	events["vtx","isGood"] = IDcut & maxMiniIso & chi2_loose & mindxy_refit & passConvVeto & mass_lo_refit & logdxydz_loose # up-to-date with AN 
"""

#def cut9(events, info):
#    events = hasGoodVertex(events, info)
#    name = 'cut9' 
#    desc = 'Has Good ee Vertex' 
#    plots = True 
#    return events, name, desc, plots

def cut9(events, info):
    name = "cut9"
    desc = "Vtx: Ele ID Cut"
    plots = True
    cut = ak.any(events.vtx.e1.passID & events.vtx.e2.passID, axis=1)
    return events[cut], name, desc, plots

def cut10(events, info):
    name = "cut10"
    desc = "Vtx: maxMiniIso < 0.9"
    plots = True
    cut = ak.any(np.maximum(events.vtx.e1.miniRelIsoEleCorr, events.vtx.e2.miniRelIsoEleCorr) < 0.9, axis=1)
    return events[cut], name, desc, plots

def cut11(events, info):
    name = "cut11"
    desc = "Vtx: reduced chi^2 < 15"
    plots = True
    cut = ak.any(events.vtx.reduced_chi2 < 15, axis=1)
    return events[cut], name, desc, plots

def cut12(events, info):
    name = "cut12"
    desc = "Vtx: min dxy > 0.001"
    plots = True
    cut = ak.any(np.minimum(np.abs(events.vtx.e1.refit_dxy), np.abs(events.vtx.e2.refit_dxy)) > 0.001, axis=1)
    return events[cut], name, desc, plots

def cut13(events, info):
    name = "cut13"
    desc = "Vtx: pass conversion veto"
    plots = True
    cut = ak.any(events.vtx.e1.conversionVeto & events.vtx.e2.conversionVeto, axis=1)
    return events[cut], name, desc, plots

def cut14(events, info):
    name = "cut14"
    desc = "Vtx: refitted mass > 0.1 GeV"
    plots = True
    cut = ak.any(events.vtx.refit_m > 0.1, axis=1)
    return events[cut], name, desc, plots

def cut15(events, info):
    name = "cut15"
    desc = "Vtx: min log(dxy/dz) > -2"
    plots = True
    cut = ak.any(np.minimum(np.log10(np.abs(events.vtx.e1.dxy/events.vtx.e1.dz)), np.log10(np.abs(events.vtx.e2.dxy/events.vtx.e2.dz))) > -2, axis=1)
    return events[cut], name, desc, plots

def cut16(events, info):
    events = hasGoodVertex(events, info)
    name = 'cut16' 
    desc = 'Has Good ee Vertex' 
    plots = True 
    return events, name, desc, plots

def cut17(events,info):
    name = "cut17"
    desc = "SV(ee) OSSF"
    plots = True
    cut = events.sel_vtx.sign == -1
    return events[cut], name, desc, plots

def cut18(events,info):
    name = "cut18"
    desc = r"SV(ee) $cos(\theta_{coll}) > 0.4$"
    plots = True
    cut = events.sel_vtx.cos_collinear > 0.4
    return events[cut], name, desc, plots

def cut19(events,info):
    name = "cut19"
    desc = r"SV(ee) $\chi^2/ndf < 3$"
    plots = True
    cut = events.sel_vtx.reduced_chi2 < 3.
    return events[cut], name, desc, plots

