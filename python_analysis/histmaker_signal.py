7# imports
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
vers = 'Jul2026noID'
cuts = 'anmatchvtx'
hists = 'lxyanalysis'


# ---- FILES ----
cuts_config = f"configs/cut_configs/{cuts}_selection.py"
#hists_config = "configs/hists/genstudy.py"
hists_config = f"configs/histo_configs/{hists}.py"
outdir = 'workarea'

# ---- SIGNAL -------------------------------------
sample_config = f"configs/sample_configs/signal_2024_{vers}_aEM.json"

#analyzer = Analyzer(sample_config, hists_config, cuts_config, model_config) # If using BDT in cuts
analyzer = Analyzer(sample_config, hists_config, cuts_config) # If not using BDT in cuts

t1 = time.time()
#out = analyzer.process(execr='iterative')
out = analyzer.process(execr='futures', workers=8)
t2 = time.time()

print("Runtime: {:.2f} minutes".format((t2-t1)/60))
util.save(out,f"{outdir}/hists_sig{vers}_{cuts}-sel_{hists}.coffea")

del out, analyzer

