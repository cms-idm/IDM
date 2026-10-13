import numpy as np
import awkward as ak
import sys

import analysisSubroutines as routines

def cut0(events,info):
    name = "cut0"
    desc = "Dummy cut"
    plots = True
    return events, name, desc, plots

# def cut1(events,info):
#     name = "cut1"
#     desc = r"Pass $\vec{p}_T^{miss}$ Filters"
#     plots = True
#     cut = events.METFiltersFailBits == 0
#     return events[cut], name, desc, plots

# def cut2(events,info):
#     name = "cut2"
#     desc = "dR cut0"
#     plots = True
#     cut = (events.genEE.dr > 0.01) & (events.genEE.dr < 0.1)
#     return events[cut], name, desc, plots


# def cut2(events,info):
#     name = "cut2"
#     desc = "dR cut0"
#     plots = True
#     cut = (events.genEE.dr > 0.01) & (events.genEE.dr < 0.1)
#     return events[cut], name, desc, plots



    
# def cut2(events,info):
#     name = "cut2"
#     desc = "MET Trigger (120 GeV)"
#     plots = True
#     cut = events.trig.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
#     return events[cut], name, desc, plots


# def cut3(events,info):
#     name = "cut3"
#     desc = r"N GED/Lpt Electron >= 2"
#     plots = True
#     cut = (ak.num(events.Electron) + ak.num(events.LptElectron)) >= 2
#     return events[cut], name, desc, plots



# def cut2(events,info):
#     name = "cut2"
#     desc = "pT cut2"
#     plots = True
#     cut = (events.GenEle.pt<5) & (events.GenPos.pt<5)
#     return events[cut], name, desc, plots

# def cut2(events,info):
#     name = "cut2"
#     desc = "pT cut2"
#     plots = True
#     cut = (events.GenEle.pt>5) & (events.GenEle.pt<10)
#     return events[cut], name, desc, plots


# def cut2(events,info):
#     name = "cut2"
#     desc = "pT cut4"
#     plots = True
#     cut = (events.GenEle.pt>20) 
#     return events[cut], name, desc, plots