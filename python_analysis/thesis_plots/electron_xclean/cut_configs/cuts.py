import numpy as np
import awkward as ak
import sys

import analysisSubroutines as routines



def cut1(events,info):
    name = "cut1"
    desc = r"Pass $\vec{p}_T^{miss}$ Filters"
    plots = True
    cut = events.METFiltersFailBits == 0
    return events[cut], name, desc, plots

# def cut2(events,info):
#     name = "cut2"
#     desc = "Lxy cut0"
#     plots = True
#     cut = (events.GenEle.vxy < 1) 
#     return events[cut], name, desc, plots


# def cut2(events,info):
#     name = "cut2"
#     desc = "Lxy cut0"
#     plots = True
#     cut = ((events.GenEle.vxy > 0) & (events.GenEle.vxy <1)) | ((events.GenPos.vxy >0)) & (events.GenPos.vxy <1) )
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
#     cut = (events.GenEle.pt>10) 
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