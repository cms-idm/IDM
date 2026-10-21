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
from feature_sets import FEATURE_SETS
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Skims
# ---------------------------------------------------------------------------
suffix = "_TestRun2"
cut_config = "../configs/cut_configs/anabrv_selection.py"
last_cut = "cut4" # apply cuts in cut_config up to and including this one (e.g. "cut5"), None = all cuts
cut_tag = f"_{last_cut}" if last_cut is not None else ""
signal_json = "../configs/sample_configs/signal_2018_official_Preapproval_aEM.json"
bkg_json = "../configs/sample_configs/bkg_2024_Aug2026_skimmed.json"

# skim signal
samp_sig = signal_json.split("/")[-1].split(".")[0]
sig_file = f"{samp_sig}{suffix}{cut_tag}"

# skim bkg
samp_bkg = bkg_json.split("/")[-1].split(".")[0]
bkg_file = f"{samp_bkg}{suffix}{cut_tag}"

# ---------------------------------------------------------------------------
# BDT 
# ---------------------------------------------------------------------------
bdt_name = "BDT"
feature_sets = ["AN"] # keys of FEATURE_SETS in feature_sets.py; one model is trained per set, named f"{bdt_name}_{fs}"

exclude_points = [{'m1': 0.05}]

for fs in feature_sets:
    name = f"{bdt_name}_{fs}"
    variables = FEATURE_SETS[fs]
    print(f'\n===== Training {name}: {variables} =====')

    # k-fold cross training: models[i] is trained on folds != i (fold = eventNum % bdt.K_FOLDS)
    models, tpr_WP, thres_WP, oof = bdt.train_bdt(name,sig_file,bkg_file,variables,exclude_points=exclude_points)

    print(tpr_WP)
    print(thres_WP)

    # -----------------------------------------------------------------------
    # Performance: AUC, and S/B, S/sqrt(B) per working point
    # -----------------------------------------------------------------------
    # out-of-fold scores: each event is scored by the model that did not train on it
    # (all signal incl. unmatched vertices, excluded points removed); S and B below are unweighted event counts
    sig_score = oof['sig_score']
    bkg_score = oof['bkg_score']
    sig_point = oof['sig_point']

    # weighted, matching the AUC shown on the ROC curve plot
    y_true = np.concatenate((np.ones(len(sig_score)),np.zeros(len(bkg_score))))
    y_score = np.concatenate((sig_score,bkg_score))
    auc = roc_auc_score(y_true,y_score,sample_weight=np.concatenate((oof['sig_wgt'],oof['bkg_wgt'])))
    print(f'\n{name} out-of-fold AUC: {auc:.4f}')

    points = np.unique(np.stack([sig_point['m1'],sig_point['delta'],sig_point['ctau']],axis=1),axis=0)
    for wp, thres in thres_WP.items():
        B = np.sum(bkg_score > thres)
        print(f'\n{wp} WP (threshold {thres:.4f}, target sig eff {100*tpr_WP[wp]:.0f}%): B = {B} (unweighted)')
        print(f'{"m1":>6} {"delta":>6} {"ctau":>6} {"S":>12} {"S/B":>12} {"S/sqrt(B)":>12}')
        for m1, delta, ctau in points:
            in_point = (sig_point['m1'] == m1) & (sig_point['delta'] == delta) & (sig_point['ctau'] == ctau)
            S = np.sum(in_point & (sig_score > thres))
            s_over_b = S/B if B > 0 else np.inf
            s_over_sqrtb = S/np.sqrt(B) if B > 0 else np.inf
            print(f'{m1:>6g} {delta:>6g} {ctau:>6g} {S:>12d} {s_over_b:>12.4g} {s_over_sqrtb:>12.4g}')

