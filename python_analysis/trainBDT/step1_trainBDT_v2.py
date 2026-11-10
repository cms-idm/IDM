import json
import uproot
from XRootD import client
import xgboost as xgb
import os
import sys
sys.path.append("../analysisTools/")
from skimmerTools import Skimmer
import coffea
import coffea.util as util
import numpy as np
import awkward as ak
import time
import h5py
import BDTutils_v2 as bdt

# ---------------------------------------------------------------------------
# Skims
# ---------------------------------------------------------------------------
suffix = "_TestOct8"
cut_config = "../configs/cut_configs/anabrv_selection.py"
last_cut = "cut5" # apply cuts in cut_config up to and including this one (e.g. "cut5"), None = all cuts
cut_tag = f"_{last_cut}" if last_cut is not None else ""
signal_json = "../configs/sample_configs/signal_2024_Jul2026noID_0p1_skimmed.json"
bkg_json = "../configs/sample_configs/bkg_2024_Aug2026_skimmed.json"

# skim signal
samp = signal_json.split("/")[-1].split(".")[0]
print(f'Start analyzing: {samp}')
skim = Skimmer(signal_json, cut_config, last_cut=last_cut)
t1 = time.time()
out = skim.process(execr='futures',workers=4)
t2 = time.time()
sig_file = f"{samp}{suffix}{cut_tag}"
bdt.write_h5(out[0],f"h5/{sig_file}.h5")

print('Completed in {:.2f} min\n'.format((t2-t1)/60))
del out, skim

# skim bkg
samp = bkg_json.split("/")[-1].split(".")[0]
print(f'Start analyzing: {samp}')
skim = Skimmer(bkg_json, cut_config, last_cut=last_cut)
t1 = time.time()
out = skim.process(execr='futures',workers=4)
t2 = time.time()
bkg_file = f"{samp}{suffix}{cut_tag}"
bdt.write_h5(out[0],f"h5/{bkg_file}.h5")

print('Completed in {:.2f} min\n'.format((t2-t1)/60))
del out, skim

# ---------------------------------------------------------------------------
# BDT 
# ---------------------------------------------------------------------------
#variables = ['sel_vtx_chi2','sel_vtx_METdPhi','sel_vtx_refit_m','sel_vtx_refit_dR','sel_vtx_corrMinDxy','vxy','vxy_signif',\
#             'sel_vtx_cos_collinear', 'sel_vtx_prod_eta', 'met_leadPt_ratio', 'jetMETdPhi', 'minJetMETdPhi']
#bdt_name = "BDT_NJetL3"
bdt_name = "AN"
variables = ['sel_vtx_refit_m', 'sel_vtx_refit_dR', 'sel_vtx_METdPhi', 'sel_vtx_minDxy_refit', 'vxy_corr', 'sel_vtx_prod_eta']

bst, tpr_WP, thres_WP = bdt.train_bdt(bdt_name,sig_file,bkg_file,variables)

print(tpr_WP)
print(thres_WP)

