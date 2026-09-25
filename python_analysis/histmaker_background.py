# imports
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

import sys
#sys.path.append("../../analysisTools/")
from analysisTools.analysisTools import Analyzer
from analysisTools.analysisTools import loadSchema
import analysisTools.analysisTools as tools
import analysisTools.analysisSubroutines as routines
import importlib
import coffea.util as util

import time
import json
import os
import glob

# ---- SETTINGS ----
vers = 'Aug2026'
cuts = 'an'
hists = 'appearingtrack'


# ---- FILES ----
cuts_config = f"configs/cut_configs/{cuts}_selection.py"
#hists_config = "configs/hists/genstudy.py"
hists_config = f"configs/histo_configs/{hists}.py"
outdir = 'workarea'

# ---- SIGNAL -------------------------------------
sample_config = f"configs/sample_configs/bkg_2024_{vers}_skimmed.json"
slimmed_config = f"configs/sample_configs/bkg_2024_{vers}-slim.json"

# slim ntuples are only needed to fill the cutflow_slim/'passHLT' entries -- set to
# False to skip opening them entirely when only the main histograms are needed
process_slim = False

#analyzer = Analyzer(sample_config, hists_config, cuts_config, model_config) # If using BDT in cuts
analyzer = Analyzer(sample_config, hists_config, cuts_config, slimFileList=slimmed_config, processSlim=process_slim) # If not using BDT in cuts

t1 = time.time()
#out = analyzer.process(execr='iterative')
out = analyzer.process(execr='futures', workers=4)
t2 = time.time()

print("Runtime: {:.2f} minutes".format((t2-t1)/60))
util.save(out,f"{outdir}/hists_bkg{vers}_{cuts}-sel_{hists}.coffea")

del out, analyzer

