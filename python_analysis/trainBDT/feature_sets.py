# Named versions of the BDT input features. Every feature listed here must be written to the h5 by
# skimmerTools.makeBDTInputs; step1_makeh5.py checks this after each skim.
FEATURE_SETS = {
    # --- VERTEXED ---
    "AN": ['sel_vtx_refit_m', 'sel_vtx_refit_dR', 'sel_vtx_METdPhi', 'sel_vtx_minDxy_refit', 'vxy_corr', 'sel_vtx_prod_eta'],
    # BDT_10vars_comb11, see makeBDTinputs in analysisSubroutines.py
    "comb11": ['sel_vtx_chi2', 'sel_vtx_METdPhi', 'sel_vtx_refit_m', 'sel_vtx_refit_dR', 'sel_vtx_corrMinDxy', 'vxy', 'vxy_signif',
               'sel_vtx_cos_collinear', 'sel_vtx_prod_eta', 'met_leadPt_ratio', 'jetMETdPhi', 'minJetMETdPhi'],
    # Comb 4, see makeBDTinputs_ABCD in analysisSubroutines.py
    "ABCD_comb4": ['sel_vtx_chi2', 'sel_vtx_METdPhi', 'sel_vtx_refit_m', 'sel_vtx_refit_dR', 'sel_vtx_minDxy_refit',
                   'vxy_corr', 'sel_vtx_prod_eta', 'met_leadPt_ratio'],
    
    # --- MERGED ---
    "merged1": [],
}

def all_features():
    # union of all feature sets, in first-seen order
    feats = []
    for fs in FEATURE_SETS.values():
        feats += [f for f in fs if f not in feats]
    return feats
