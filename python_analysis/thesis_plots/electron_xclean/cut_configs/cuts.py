import numpy as np
import awkward as ak
import sys

import analysisSubroutines as routines


# import numpy as np
# import awkward as ak
# from analysisTools.analysisSubroutines import getBtagWPs, hasGoodVertex

def cut1(events,info):
    name = "cut1"
    desc = r"Pass $\vec{p}_T^{miss}$ Filters"
    plots = True
    cut = events.METFiltersFailBits == 0
    return events[cut], name, desc, plots

def cut2(events,info):
    name = "cut2"
    desc = "MET Trigger (120 GeV)"
    plots = True
    cut = events.trig.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
    return events[cut], name, desc, plots

# def cut2(events,info):
#     name = "cut2"
#     desc = r"$\vec{p}_T^{miss}$ Trigger (120 GeV)"
#     plots = True

#     if info["year"] == 2016:
#         cut = (events.trigFired16 & (1<<9)) == (1<<9)
#     if info["year"] == 2017:
#         cut = (events.trigFired17 & (1<<9)) == (1<<9)
#     if info["year"] == 2022:
#         cut = (events.trigFired18 & (1<<13)) == (1<<13)
#     if info["year"] == 2024:
#         #for f in events.trig.fields:
#         #    if 'HLT' in f and 'PFMET' in f:
#         #        print(f)
#         #cut = (events.trig.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60)
#         if 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight' not in events.trig.fields:
#             for f in events.trig.fields:
#                 if 'HLT_PFMET' in f:
#                     print(f)
            
#         if 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60' in events.trig.fields and 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight' in events.trig.fields:
#             cut = (events.trig.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight |
#                    events.trig.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60)
#         elif 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60' in events.trig.fields:
#             cut = events.trig.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60
#         elif 'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight' in events.trig.fields:
#             cut = events.trig.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
        
#     return events[cut], name, desc, plots

def cut3(events,info):
    name = "cut3"
    desc = r"N GED/Lpt Electron >= 2"
    plots = True
    cut = (ak.num(events.Electron) + ak.num(events.LptElectron)) >= 2
    return events[cut], name, desc, plots

# def cut4(events,info):
#     name = "cut4"
#     desc = r"$\vec{p}_T^{miss}$ > 200 GeV"
#     plots = True
#     cut = events.PFMET.pt > 200
#     return events[cut], name, desc, plots

# def cut5(events,info):
#     name = "cut5"
#     desc = "nJets > 0 (pT > 30 GeV)"
#     plots = True
#     nJets = ak.count(events.PFJet.pt,axis=1)
#     cut = (nJets > 0) 
#     return events[cut], name, desc, plots

# def cut6(events,info):
#     # UL b-tag threshold recommendations from here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
#     # using the medium WP, as in Andre's version of iDM
#     name = "cut6"
#     desc = "No b-tagged jets"
#     plots = True
#     bTag = events.PFJet.bTag

#     # DeepFlavour working points for UL samples
#     loose,med,tight = getBtagWPs(info["year"]) 
#     wp = med # MEDIUM BTAG WP
    
#     pass_bTag = bTag > wp
#     n_bTag_Jets = ak.count(events.PFJet[pass_bTag].pt,axis=1)
#     cut = n_bTag_Jets == 0
#     return events[cut], name, desc, plots

# def cut7(events, info):
#     name = 'cut7'
#     desc = r'J1 $p_T > 80$ GeV, $|\eta| < 2.4$'
#     plots = True
#     jets = events.PFJet
#     sortJetsByPt = ak.argsort(jets.pt, ascending=False)
#     j1 = jets[sortJetsByPt][:, 0]
#     cut = (j1.pt > 80.) & (np.abs(j1.eta) < 2.4)
#     return events[cut], name, desc, plots
    
# def cut8(events, info):
#     name = 'cut8'
#     desc = r'$|\Delta \phi(\vec{p}_T^{miss},$ J1$)| > 1.5$'
#     plots = True
#     jets = events.PFJet
#     sortJetsByPt = ak.argsort(jets.pt, ascending=False)
#     j1 = jets[sortJetsByPt][:, 0]
#     cut = np.abs(j1.METdPhi) > 1.5
#     return events[cut], name, desc, plots
    
# def cut9(events, info):
#     name = 'cut9'
#     desc = r'min$_{j \in \mathrm{jets}} |\Delta \phi(\vec{p}_T^{miss}, j)| > 0.75$'
#     plots = True
#     jets = events.PFJet
#     cut = ak.min(abs(jets.METdPhi), axis=1) > 0.75
#     return events[cut], name, desc, plots
    
# def cut10(events, info):
#     events = hasGoodVertex(events, info)
#     name = 'cut10' 
#     desc = 'Has Good ee Vertex' 
#     plots = True 
#     return events, name, desc, plots

# def cut11(events,info):
#     name = "cut11"
#     desc = "SV(ee) OSSF"
#     plots = True
#     cut = events.sel_vtx.sign == -1
#     return events[cut], name, desc, plots

# def cut12(events,info):
#     name = "cut12"
#     desc = r"SV(ee) $cos(\theta_{coll}) > 0.4$"
#     plots = True
#     cut = events.sel_vtx.cos_collinear > 0.4
#     return events[cut], name, desc, plots

# def cut13(events,info):
#     name = "cut13"
#     desc = r"SV(ee) $\chi^2/ndf < 3$"
#     plots = True
#     cut = events.sel_vtx.reduced_chi2 < 3.
#     return events[cut], name, desc, plots



















# def cut0(events,info):
#     name = "cut0"
#     desc = r"Pass $\vec{p}_T^{miss}$ Filters"
#     plots = True
#     cut = events.METFiltersFailBits == 0
#     return events[cut], name, desc, plots

# def cut0(events,info):
#     name = "cut0"
#     desc = "Dummycut"
#     plots = True
#     return events, name, desc, plots

# def cut0(events,info):
#     name = "cut0"
#     desc = "Lxy cut0"
#     plots = True
#     cut = (events.GenEle.vxy < 1) 
#     return events[cut], name, desc, plots


# def cut0(events,info):
#     name = "cut0"
#     desc = "Lxy cut0"
#     plots = True
#     cut = (events.GenEle.vxy > 10) & (events.GenEle.vxy <15)
#     return events[cut], name, desc, plots

# def cut1(events,info):
#     name = "cut1"
#     desc = "pT cut1"
#     plots = True
#     cut = 5<events.GenEle.pt <10
#     return events[cut], name, desc, plots

# def cut2(events,info):
#     name = "cut2"
#     desc = "pT cut2"
#     plots = True
#     cut = (events.GenEle.pt>10) 
#     return events[cut], name, desc, plots

# def cut3(events,info):
#     name = "cut3"
#     desc = "pT cut3"
#     plots = True
#     cut = (events.GenEle.pt>15) & (events.GenEle.pt<20)
#     return events[cut], name, desc, plots


# def cut4(events,info):
#     name = "cut4"
#     desc = "pT cut4"
#     plots = True
#     cut = (events.GenEle.pt>20) 
#     return events[cut], name, desc, plots

# def cut1(events,info):
#     name = "cut1"
#     desc = "Pass MET Filters"
#     plots = True
#     cut = events.METFiltersFailBits == 0
#     return events[cut], name, desc, plots

# def cut2(events,info):
#     name = "cut2"
#     desc = "HEM Veto"
#     plots = True
#     if info["year"] == 2018:
#         cut = ~events.HEM.flag
#         return events[cut], name, desc, plots
#     else:
#         return events, name, desc, plots

# def cut3(events,info):
#     name = "cut3"
#     desc = "MET Trigger (120 GeV)"
#     plots = True
#     cut = events.trig.HLT_PFMET120_PFMHT120_IDTight
#     return events[cut], name, desc, plots

# def cut4(events,info):
#     name = "cut4"
#     desc = "MET > 200 GeV"
#     plots = True
#     cut = events.PFMET.pt > 200
#     return events[cut], name, desc, plots

# def cut5(events,info):
#     # UL b-tag threshold recommendations from here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
#     # using the medium WP, as in Andre's version of iDM
#     name = "cut5"
#     desc = "No b-tagged jets"
#     plots = True
#     bTag = events.PFJet.bTag
#     # DeepFlavour working points for UL samples
#     if info["year"] == 2018:
#         # twiki : https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18
#         #wp = 0.0490 # loose
#         wp = 0.2783 # medium
#         #wp = 0.7100 # tight
#     if info["year"] == 2017:
#         # twiki : https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17
#         #wp = 0.0532 # loose
#         wp = 0.3040 # medium
#         #wp = 0.7476 # tight
#     if info["year"] == "2016_preVFP":
#         # twiki : https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16preVFP
#         #wp = 0.0508 # loose
#         wp = 0.2598 # medium
#         #wp = 0.6502 # tight
#     if info["year"] == "2016_postVFP":
#         # twiki : https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16postVFP
#         #wp = 0.0480 # loose
#         wp = 0.2489 # medium
#         #wp = 0.6377 # tight
#     pass_bTag = bTag > wp
#     n_bTag_Jets = ak.count(events.PFJet[pass_bTag].pt,axis=1)
#     cut = n_bTag_Jets == 0
#     return events[cut], name, desc, plots

# def cut6(events,info):
#     name = "cut6"
#     desc = "Leading jet |eta| < 2.4"
#     plots = True
#     cut = np.abs(events.PFJet.eta[:,0]) < 2.4
#     return events[cut], name, desc, plots

# def cut7(events,info):
#     name = "cut7"
#     desc = "Leading jet pT > 80 GeV"
#     plots = True
#     cut = events.PFJet.pt[:,0] > 80
#     return events[cut], name, desc, plots
