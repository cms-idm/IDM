# Named versions of the BDT input features. Every feature listed here must be written to the h5 by
# skimmerTools.makeBDTInputs; step1_makeh5.py checks this after each skim. For inference, each feature also needs a
# definition in analysisSubroutines.BDT_VARIABLES (used by analysisSubroutines.makeBDTinputs).
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

# which feature sets belong to which kind of h5 (skimmerTools.Skimmer bdt_type)
BDT_TYPE_SETS = {
    "vertexed": ["AN", "comb11", "ABCD_comb4"],
    "merged": ["merged1"],
}

def all_features(bdt_type="vertexed"):
    # union of the feature sets for this bdt_type, in first-seen order
    feats = []
    for name in BDT_TYPE_SETS[bdt_type]:
        feats += [f for f in FEATURE_SETS[name] if f not in feats]
    return feats
