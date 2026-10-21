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
import BDTutils_kfold as bdt
from feature_sets import all_features

# ---------------------------------------------------------------------------
# Skims
# ---------------------------------------------------------------------------
suffix = "_TestRun2"
cut_config = "../configs/cut_configs/anabrv_selection.py"
last_cut = "cut4" # apply cuts in cut_config up to and including this one (e.g. "cut5"), None = all cuts
cut_tag = f"_{last_cut}" if last_cut is not None else ""
#signal_json = "../configs/sample_configs/signal_2024_Jul2026noID_0p1_skimmed.json"
signal_json = "../configs/sample_configs/signal_2018_official_Preapproval_aEM.json"
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
bdt.check_features([f"h5/{sig_file}.h5"],all_features())
sig_fold = bdt.get_fold(out[0]['eventNum'].value)
print(f'Signal events per fold (eventNum % {bdt.K_FOLDS}, before gen matching): {np.bincount(sig_fold,minlength=bdt.K_FOLDS)}')

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
bdt.check_features([f"h5/{bkg_file}.h5"],all_features())
bkg_fold = bdt.get_fold(out[0]['eventNum'].value)
print(f'Background events per fold (eventNum % {bdt.K_FOLDS}): {np.bincount(bkg_fold,minlength=bdt.K_FOLDS)}')

print('Completed in {:.2f} min\n'.format((t2-t1)/60))
del out, skim

