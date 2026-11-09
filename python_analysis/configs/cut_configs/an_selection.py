import numpy as np
import awkward as ak
import json
import os
try:
    from analysisTools.analysisSubroutines import getBtagWPs, hasGoodVertex, getBDTscore_kfold
except ModuleNotFoundError:  # run with ../analysisTools/ on sys.path (e.g. trainBDT/skimmerTools)
    from analysisSubroutines import getBtagWPs, hasGoodVertex, getBDTscore_kfold

# k-fold BDT (trainBDT/BDTutils_kfold.train_bdt). Note: any module-level name containing "cut" is picked up as a cut
bdt_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../trainBDT/models/variables_BDT_AN_TestOct28.json')
# key of this selection's threshold in the model JSON's thres_opt (written by trainBDT/step3_optimizeThreshold.py)
sel_name = os.path.splitext(os.path.basename(__file__))[0]

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
    if info["year"] in (2018, 2022):
        if "trigFired18" in events.fields:
            cut = (events.trigFired18 & (1<<13)) == (1<<13)
        else:
            # newer ntuples (NtupleContainerV2) don't fill trigFired18,
            # so fall back to the named HLT branch (bit 13 of trigFired18
            # corresponds to HLT_PFMET120_PFMHT120_IDTight)
            cut = events.trig.HLT_PFMET120_PFMHT120_IDTight
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
    desc = "BDT score > optimized threshold"
    plots = True
    with open(bdt_json) as fin:
        vars_dict = json.load(fin)
    if sel_name not in vars_dict.get('thres_opt', {}):
        raise KeyError(f'{bdt_json} has no thres_opt["{sel_name}"]; run trainBDT/step3_optimizeThreshold.py with cut_config = {sel_name}.py')
    thres = vars_dict['thres_opt'][sel_name]['value']
    events['BDTScore'] = getBDTscore_kfold(events, bdt_json)
    cut = events.BDTScore > thres
    return events[cut], name, desc, plots
