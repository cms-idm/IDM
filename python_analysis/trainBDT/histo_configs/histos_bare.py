import numpy as np
import awkward as ak
from myHisto import myHisto

def make_histograms(info):
    h = myHisto()
    
    # selected vertex
    h.make("sel_vtx_dR",'dR')
    h.make("bdtScore",'bdtScore')

    return h

subroutines = []

def fillHistos(events,h,samp,cut,info,sum_wgt=1):
    h.samp = samp
    h.cut = cut
    if info["type"] == "signal" or info["type"] == "bkg":
        wgt = events.eventWgt/sum_wgt
    else:
        wgt = 1

    sel_vtx = events.sel_vtx
    ### FILLING HISTOGRAMS ###
    
    h.fill("sel_vtx_dR",dR=sel_vtx.dR,weight=wgt)
    h.fill("bdtScore",bdtScore=events.BDTScore,weight=wgt)