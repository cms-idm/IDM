import numpy as np
import awkward as ak
import sys
sys.path.append("../../../analysisTools/")
import analysisSubroutines as routines

def cut0(events,info):
    # using the medium WP, as in Andre's version of iDM
    name = "cut0"
    desc = "Njets < 3"
    plots = False
    cut = events.nJets < 3
    return events[cut], name, desc, plots

def cut1(events,info):
    # using the medium WP, as in Andre's version of iDM
    name = "cut1"
    desc = "No b-tagged jets"
    plots = False
    n_bTag_Jets = ak.count_nonzero(events.PFJet.passMedID,axis=1)
    cut = n_bTag_Jets == 0
    return events[cut], name, desc, plots

def cut2(events,info):
    name = "cut2"
    desc = "Leading jet |eta| < 2.4"
    plots = False
    cut = np.abs(events.PFJet.eta[:,0]) < 2.4
    return events[cut], name, desc, plots

def cut3(events,info):
    name = "cut3"
    desc = "Leading jet pT > 80 GeV"
    plots = False
    cut = events.PFJet.pt[:,0] > 80
    return events[cut], name, desc, plots

def cut4(events,info):
    name = "cut4"
    desc = "dPhi(MET,leading jet) > 2"
    plots = False
    cut = np.abs(events.PFJet.METdPhi[:,0]) > 2
    return events[cut], name, desc, plots

def cut5(events,info):
    name = "cut5"
    desc = "dPhi(MET,all jets) > 0.75"
    plots = False
    cut = ak.all(np.abs(events.PFJet.METdPhi) > 0.75,axis=1)
    return events[cut], name, desc, plots

def cut6(events,info):
    name = "cut6"
    desc = "Refit M(ee) > 0.1 GeV"
    plots = False
    cut = events.sel_vtx.refit_m > 0.1
    return events[cut], name, desc, plots

def cut7(events,info):
    name = "cut7"
    desc = "|PFMET - CaloMET|/CaloMET < 0.7"
    plots = False
    cut = np.abs(events.PFMET.pt-events.CaloMET.pt)/events.CaloMET.pt < 0.7
    return events[cut], name, desc, plots

def cut8(events,info):
    # pre-computing BDT score so I can make plots with it
    name = 'cut8'
    desc = 'computing BDT score'
    plots = False
    
    if len(events) != 0:
        input = routines.makeBDTinputs(events)

        # Using the same BDT inputs, but retrained
        # Retrained with MINIAOD + new x-cleaning + v5 good_vtx definition + Medium-BJet Veto
        #model = './models/BDT_BJetMedium_NJetL4.json'
        model = './models/BDT_NJetL3.json'
        score_BDT = routines.getBDTClassifierScore(input, model)
        events['BDTScore'] = score_BDT
    else:
        events['BDTScore'] = ak.zeros_like(events.eventWgt)

    return events, name, desc, plots

def cut9(events,info):
    name = "cut9"
    desc = "BDTLoose"
    plots = True
    thres = 0.994421124458313
    if len(events) != 0:
        cut = events['BDTScore'] > thres
    else:
        cut = []
    return events[cut], name, desc, plots

def cut10(events,info):
    name = "cut10"
    desc = "BDTMed"
    plots = True
    thres = 0.9968745708465576
    if len(events) != 0:
        cut = events['BDTScore'] > thres
    else:
        cut = []
    return events[cut], name, desc, plots

def cut11(events,info):
    name = "cut11"
    desc = "BDTTight"
    plots = True
    thres = 0.9988095164299011
    if len(events) != 0:
        cut = events['BDTScore'] > thres
    else:
        cut = []
    return events[cut], name, desc, plots