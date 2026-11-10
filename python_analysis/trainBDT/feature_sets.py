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
    # all of _MELE_VARS in configs/histo_configs/corrmatrix.py, see makeMergedBDTInputs in skimmerTools.py
    "merged1": ['mele_log10_dxy',  'mele_log10_dz', 'mele_logDxyDz', 'mele_cosMETdPhi', 'mele_log10_lxyEst', 'mele_mindRj', 'mele_mindPhiJ',
                'mele_dRj0', 'mele_log10_trkChi2', 'mele_trkProb', 'mele_numTrackerHits', 'mele_numPixHits', 'mele_ID',
                'mele_log10_miniRelIso', 'mele_log10_trkRelIso', 'mele_HoverE', 'mele_EoverPInv',
                'mele_missingInnerHits', 'mele_conversionVeto', 'mele_svProxyVxy', 'met_leadPt_ratio', 'met_pt'], # AUC 0.976 
    "merged2": ['mele_log10_lxyEst', 'mele_cosMETdPhi', 'mele_log10_dz', 'mele_log10_miniRelIso',
                'mele_mindPhiJ', 'mele_log10_trkRelIso', 'mele_ID', 'met_leadPt_ratio',
                'mele_missingInnerHits', 'mele_logDxyDz', 'mele_scPhiWidth', 'mele_EoverPInv'], # AUC 0.972
    "merged3": ['mele_log10_lxyEst', 'mele_cosMETdPhi', 'mele_log10_dz', 'mele_log10_miniRelIso',
                'mele_mindPhiJ', 'mele_log10_trkRelIso', 'mele_ID', 'met_leadPt_ratio',
                'mele_missingInnerHits',  'mele_scPhiWidth'], # AUC 0.9831 after excluding m1=50
    "merged4": ['mele_log10_lxyEst', 'mele_cosMETdPhi', 'mele_log10_dz', 'mele_log10_miniRelIso',
                'mele_mindPhiJ', 'mele_log10_trkRelIso', 'mele_ID', 'met_leadPt_ratio',
                'mele_missingInnerHits',  'mele_scPhiWidth', 'HT', 'nJets',
                'mele_dEtaSeed', 'mele_dPhiIn', 'mele_sigmaIetaIeta', 'mele_scEtaWidth', 'mele_scPhiWidth',
                'mele_r9', 'mele_sigmaIphiIphi', 'mele_e1x5Frac', 'mele_e2x5Frac',
                'mele_scNClus', 'mele_scEoverP', 'mele_scRawFrac', 'mele_log10_dEtaSCTrk'], # AUC 0.9865
    "merged5": ['mele_log10_lxyEst', 'mele_cosMETdPhi', 'mele_log10_dz', 'mele_log10_miniRelIso',
                'mele_mindPhiJ', 'mele_log10_trkRelIso', 'mele_ID', 'met_leadPt_ratio',
                'mele_missingInnerHits',  'mele_scPhiWidth',
                'mele_r9', 'mele_e1x5Frac', 'mele_log10_dEtaSCTrk', 'mele_sigmaIphiIphi'], # AUC 0.9833 hmmmm
    "merged6": ['mele_log10_lxyEst', 'mele_cosMETdPhi', 'mele_log10_dz', 'mele_log10_miniRelIso',
                'mele_mindPhiJ', 'mele_log10_trkRelIso', 'mele_ID', 'met_leadPt_ratio',
                'mele_missingInnerHits',  'mele_scPhiWidth',
                'mele_r9', 'mele_e1x5Frac', 'mele_log10_dEtaSCTrk', 'mele_sigmaIphiIphi'],
    "merged7": ['mele_log10_lxyEst', 'mele_cosMETdPhi', 'mele_log10_dz', 'mele_log10_miniRelIso',
                'mele_mindPhiJ', 'mele_log10_trkRelIso', 'mele_ID', 'met_leadPt_ratio',
                'mele_missingInnerHits',  'mele_scPhiWidth', 'mele_dPhiIn'], 


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
