from histobins import *
from hist import Hist
import hist
import numpy as np
import awkward as ak

def make_histograms():
    histograms = {
        "bdt_score" : Hist(samp,cut,bdt_score,storage=hist.storage.Weight()),
        
        }
    return histograms

subroutines = []
    

def fillHistos(events,histos,samp,cut,info,sum_wgt=1):
    e1 = events.sel_vtx.e1
    e2 = events.sel_vtx.e2

    min_dxy = np.minimum(np.abs(e1.dxy),np.abs(e2.dxy))
    max_dxy = np.maximum(np.abs(e1.dxy),np.abs(e2.dxy))
    mean_dxy = np.mean(np.abs(e1.dxy),np.abs(e2.dxy))
    delta_dxy = np.abs(np.abs(e1.dxy)-np.abs(e2.dxy))

    min_dz = np.minimum(np.abs(e1.dz),np.abs(e2.dz))
    delta_dz = np.abs(np.abs(e1.dz)-np.abs(e2.dz))
    
    max_pfiso = ak.where(e1.PFRelIso<e2.PFRelIso,e2.PFRelIso,e1.PFRelIso)
    
    wgt = events.eventWgt/sum_wgt
    vtx = events.sel_vtx

    histos["bdt_score"].fill(samp=samp,cut=cut,bdt_score=events.BDTScore,weight=wgt)