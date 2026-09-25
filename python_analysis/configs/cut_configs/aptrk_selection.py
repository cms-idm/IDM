import numpy as np
import awkward as ak
from analysisTools.analysisSubroutines import getBtagWPs, hasGoodVertex, selectMergedElectronCandidates

def _dphi(a, b):
    d = np.abs(a - b)
    return ak.where(d > np.pi, 2 * np.pi - d, d)

def _dr(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2)**2 + _dphi(phi1, phi2)**2)

def _pt_rel(obj_pt, gen_pt):
    return np.abs(obj_pt - gen_pt) / gen_pt < 0.2


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
    plots = False
    jets = events.PFJet
    cut = ak.min(abs(jets.METdPhi), axis=1) > 0.75
    return events[cut], name, desc, plots
    
def cut9(events, info):
    name = 'cut9'
    desc = r'has merged candidate'
    plots = True

    cand = selectMergedElectronCandidates(events)
    cut = ~ak.is_none(cand)
    return events[cut], name, desc, plots

def cut10(events,info):
    name = "cut10"
    desc = r"min$\Delta \phi(e, j) > 1.0$"
    plots = False

    cand = selectMergedElectronCandidates(events)
    min_dphi_e_j = ak.fill_none(ak.min(cand.dPhiJets, axis=-1), 999)
    cut = min_dphi_e_j > 1.0
    return events[cut], name, desc, plots

def cut11(events,info):
    name = "cut11"
    desc = r"cos$(\Delta \phi(e, p_T^{miss})) > 0.4$"
    plots = False
    
    # all events have one now
    cand = selectMergedElectronCandidates(events)
    dphi_e_met = _dphi(cand.phi, events.PFMET.phi)
    cut = np.cos(dphi_e_met) > 0.4
    return events[cut], name, desc, plots

def cut12(events,info):
    name = "cut12"
    desc = r"Candidate $d_{xy} > 0.001$cm"
    plots = False
    cand = selectMergedElectronCandidates(events)
    cut = cand.dxy > 0.001
    return events[cut], name, desc, plots

def cut13(events,info):
    name = "cut13"
    desc = r"Candidate $L_{xy}^{est} > 0.1cm$"
    plots = True
    cand = selectMergedElectronCandidates(events)
    dphi_e_met = _dphi(cand.phi, events.PFMET.phi)
    lxyEst     = np.abs(cand.dxy) / np.sin(dphi_e_met)
    cut = lxyEst > 0.1
    return events[cut], name, desc, plots

def cut14(events, info):
    name = "cut14"
    desc = r"Log$(|d_{xy}/d_z|) > -1.25"
    plots = False
    cand = selectMergedElectronCandidates(events)
    cut = np.log10(np.abs(cand.dxy / cand.dz)) > -1.25
    return events[cut], name, desc, plots

def cut15(events, info):
    name = "cut15"
    desc = r"Electron Track $\chi^2/$df < 5"
    plots = False
    cand = selectMergedElectronCandidates(events)
    cut = cand.trkChi2 < 5.
    return events[cut], name, desc, plots

def cut16(events, info):
    name = 'cut16'
    desc = 'Mini Rel Iso < 0.9'
    plots = True
    cand = selectMergedElectronCandidates(events)
    cut = cand.miniRelIsoEleCorr < 0.9
    return events[cut], name, desc, plots
