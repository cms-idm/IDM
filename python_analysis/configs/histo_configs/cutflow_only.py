import numpy as np
import awkward as ak

# Blank histogram config for running cutflows only. The processor adds the
# cutflow dicts (cutflow, cutflow_cts, ...) to the output itself, so no
# histograms need to be booked or filled here.

def make_histograms(info=None):
    histograms = {}
    return histograms

subroutines = []

def fillHistos(events,histos,samp,cut,info,sum_wgt=1):
    pass
