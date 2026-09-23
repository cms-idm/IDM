from hist import Hist
from hist.axis import Variable, Regular, StrCategory, IntCategory, Integer
import hist
import numpy as np
import awkward as ak

# When True, fill histograms with raw event counts (weight=1) instead of the
# physical xsec/lumi/genWgt weight -- useful for diagnosing bins dominated by
# a small number of high-weighted events (low MC effective statistics).
RAW_COUNTS = True

# With RAW_COUNTS=True every fill weight is exactly 1, so a Weight() storage's
# sumw2 accumulator is just a duplicate of sumw -- use the cheaper Double()
# storage in that case. Falls back to Weight() (proper variance tracking) if
# RAW_COUNTS is ever set back to False.
_STORAGE = hist.storage.Double() if RAW_COUNTS else hist.storage.Weight()

# ── Axes (copied from configs/histo_configs/histobins.py, standalone so this
# module carries no dependency on it; rebinned to ~50 bins each) ─────────────
samp = StrCategory([], name="samp", label="Sample Name", growth=True)
cut  = StrCategory([], name="cut",  label="Cut Applied",  growth=True)

vtx_sign = IntCategory([-1, 1], name="sign", label="Vertex sign (q1*q2)")

ele_pt        = Regular(50, 0,   50,  name="pt",          label="$p_{T}$ [GeV]")
ele_eta       = Regular(50, -3,   3,  name="eta",         label=r"$\eta$")
ele_phi       = Regular(50, -3.2, 3.2, name="phi",        label=r"$\phi$")
ele_trkHits   = Regular(50, 0,   30,  name="numTrkHits",   label="Number of Tracker Hits")
ele_pixHits   = Regular(50, 0,   10,  name="numPixHits",   label="Number of Pixel Hits")
ele_stripHits = Regular(50, 0,   25,  name="numStripHits", label="Number of Strip Hits")
ele_chi2      = Regular(50, 0,  100,  name="chi2",         label=r"Track $\chi^2/df$")
ele_trkIso    = Regular(50, 0,  100,  name="trkIso",       label="Tracker Iso")
ele_trkRelIso = Regular(50, 0,    5,  name="relIso",       label="Tracker Relative Iso")
ele_PFRelIso  = Regular(50, 0,   10,  name="relIso",       label="PF Relative Iso")
ele_PFIso     = Regular(50, 0,   20,  name="iso",          label="PF Isolation")
ele_miniIso        = Regular(50, 0, 100, name="iso", label="Mini Iso")
ele_miniRelIso     = Regular(50, 0,  10, name="iso", label="Mini Relative Iso")
ele_miniIsoCorr    = Regular(50, 0, 100, name="iso", label="Corrected Mini Iso")
ele_miniRelIsoCorr = Regular(50, 0,  10, name="iso", label="Corrected Mini Relative Iso")
ele_prob    = Regular(50, 0, 1,   name="prob",   label=r"Electron Track $\chi^2$ Probability")
ele_angRes  = Regular(50, 0, 0.1, name="angRes", label=r"Angular Resolution $\sqrt{\sigma_\eta^2 + \sigma_\phi^2}$")
ele_dxy     = Regular(50, 0, 40,  name="dxy",    label="Electron Track $d_{xy}$ [cm]")
ele_dz      = Regular(50, 0, 5,   name="dz",     label="Electron Track $d_{z}$ [cm]")
ele_id      = Regular(50, -1, 4,  name="ele_id", label="Low $p_T$ electron ID Score")

ee_dr = Regular(50, 0, 1, name='dr', label=r"$\Delta R$")

vxy_coarse = Regular(50, 0, 50, name="vxy", label="$v_{xy}$ [cm]")
vz_coarse  = Regular(50, 0, 50, name="vz",  label="$v_{z}$ [cm]")
vtx_chi2   = Regular(50, 0, 30,  name="chi2", label=r"Vertex Fit $\chi^2/df$")
vtx_pt     = Regular(50, 0, 200, name="pt",   label="Selected Vertex $p_T$")
vtx_mass   = Regular(50, 0, 5,   name="mass", label="$m_{e^+e^-}$ [GeV]")

dphiJ    = Regular(50, 0, 3.2, name="dphiJ",   label=r"$\Delta \phi$")
dR       = Regular(50, 0, 6,   name='dr',      label=r"$\Delta R$")
dR_zoom  = Regular(50, 0, 1,   name='dr_zoom', label=r"$\Delta R$")
dRj      = Regular(50, 0, 6,   name='drj',     label=r"$\Delta R$")
dxy_fine = Regular(50, 0, 0.2, name='dxy',     label="$d_{xy}$ [cm]")

# ── Extra axes not in histobins ──────────────────────────────────────────────
_e_energy  = Regular(50,  0, 100,  name='e',        label='E [GeV]')
_calIso    = Regular(50,  0, 100,  name='iso',       label='Calo Iso [GeV]')
_calRelIso = Regular(50,  0,   5,  name='relIso',    label='Calo Relative Iso')
_rhoEA     = Regular(50,  0,  20,  name='rhoEA',     label=r'$\rho \times EA$ [GeV]')
_sieie     = Regular(50,  0,0.05,  name='sieie',     label=r'$\sigma_{i\eta i\eta}$')
_dEtaSeed  = Regular(50,  0,0.01,  name='dEtaSeed',  label=r'$|\Delta\eta_\mathrm{seed}|$')
_dPhiIn    = Regular(50,  0, 0.1,  name='dPhiIn',    label=r'$|\Delta\phi_\mathrm{in}|$')
_HoE       = Regular(50,  0,   1,  name='HoE',       label='H/E')
_invEmP    = Regular(50,  0, 0.1,  name='invEmP',    label=r'$|1/E - 1/p|$ [GeV$^{-1}$]')
_missHits  = Integer(0,    5,       name='missHits',  label='Exp. Missing Inner Hits')
_charge    = IntCategory([-1, 1],   name='charge',    label='Charge')
_drNearestEle = Regular(50, 0, 6,  name='drNearestEle', label=r'Min $\Delta R(e, e_{\mathrm{other\ lpt}})$')

# Relational axes: an electron (resolved leg or vertex leg) vs. its nearest
# other AllLptElectron -- large dPt/dVxy/dVz/opposite charge point to a
# genuine second electron, while near-zero values across the board point to
# a duplicate GSF track.
_dPt      = Regular(50, -50, 50, name='dpt',    label=r'$p_T(e_{\mathrm{sel}}) - p_T(e_{\mathrm{near}})$ [GeV]')
_dPtRel   = Regular(50,  -2,  2, name='dptRel', label=r'$(p_T(e_{\mathrm{sel}}) - p_T(e_{\mathrm{near}}))/p_T(e_{\mathrm{sel}})$')
_dVxyNear = Regular(50,   0, 10, name='dvxy',   label=r'$|v_{xy}(e_{\mathrm{sel}}) - v_{xy}(e_{\mathrm{near}})|$ [cm]')
_dVzNear  = Regular(50,   0, 10, name='dvz',    label=r'$|v_z(e_{\mathrm{sel}}) - v_z(e_{\mathrm{near}})|$ [cm]')
_dDxyNear = Regular(50,   0,  5, name='ddxy',   label=r'$|d_{xy}(e_{\mathrm{sel}}) - d_{xy}(e_{\mathrm{near}})|$ [cm]')
_dDzNear  = Regular(50,   0,  5, name='ddz',    label=r'$|d_z(e_{\mathrm{sel}}) - d_z(e_{\mathrm{near}})|$ [cm]')

# Axes for the vertex (vtx) record's own scalar fields -- see _VTX_HISTS and
# _VTXLEG_HISTS below.
_vtxVxyErr    = Regular(50, 0, 10,  name='sigma_vxy', label='$v_{xy}$ Error [cm]')
_vtxProb      = Regular(50, 0, 1,   name='prob',       label=r'Vertex $\chi^2$ Probability')
_metDphi      = Regular(50, 0, 3.2, name='METdPhi',     label=r'$\Delta\phi(\mathrm{vtx}, E_T^{\mathrm{miss}})$')
_eleDphi      = Regular(50, 0, 3.2, name='eleDphi',    label=r'$\Delta\phi(e_1, e_2)$')
_cosCollinear = Regular(50, -1, 1,  name='cos_collinear', label=r'$\cos(\theta_{\mathrm{collinear}})$')
_vtxType      = StrCategory(['LL', 'LR', 'RR'], name='type',     label='Vertex Type')
_legType      = StrCategory(['L', 'R'],         name='ele_type', label='Leg Collection Type')
_legMatchType = IntCategory([], name='matchType', label='Leg Gen Match Type', growth=True)
_matchSignAx  = IntCategory([], name='matchSign', label='Vertex Gen-Match Sign', growth=True)
_boolFlag     = IntCategory([0, 1], name='flag', label='Boolean Flag')
_vtxLegDxyErr = Regular(50, 0, 5, name='dxyErr', label=r'Refit $d_{xy}$ Error [cm]')
_vtxLegDzErr  = Regular(50, 0, 5, name='dzErr',  label=r'Refit $d_z$ Error [cm]')

# Same variable-width style as histobins.py's ele_lxy_res, extended out to 150 cm.
_gen_lxy = Variable(
    [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150],
    name='lxy', label='$L_{xy}$ [cm]',
)

# Every per-electron reco block below (res_*, vtx_ele_*, and their
# *_nearestEle_* realism checks) is filled twice, tagged by which of the
# event's two electrons has the higher RECO pt -- not by gen charge/flavor,
# since a real analysis (data included) only ever knows reco kinematics.
_LEAD_TAGS = ('lead', 'sublead')

# (field_name_on_AllLptElectron, hist_axis)
_ELE_HISTS = [
    ('pt',                  ele_pt),
    ('eta',                 ele_eta),
    ('phi',                 ele_phi),
    ('e',                   _e_energy),
    ('ID',                  ele_id),
    ('angRes',              ele_angRes),
    #('vxy',                 vxy_coarse),
    #('vz',                  vz_coarse),
    ('dxy',                 ele_dxy),
    ('dz',                  ele_dz),
    #('trkChi2',             ele_chi2),
    #('trkIso',              ele_trkIso),
    #('trkRelIso',           ele_trkRelIso),
    #('calIso',              _calIso),
    #('calRelIso',           _calRelIso),
    #('PFIso',               ele_PFIso),
    #('PFRelIso',            ele_PFRelIso),
    #('miniIso',             ele_miniIso),
    #('miniRelIso',          ele_miniRelIso),
    #('PFIsoEleCorr',        ele_PFIso),
    #('PFRelIsoEleCorr',     ele_PFRelIso),
    #('miniIsoEleCorr',      ele_miniIsoCorr),
    #('miniRelIsoEleCorr',   ele_miniRelIsoCorr),
    #('chadIso',             ele_PFIso),
    #('nhadIso',             ele_PFIso),
    #('phoIso',              ele_PFIso),
    #('rhoEA',               _rhoEA),
    #('trkProb',             ele_prob),
    ('numTrackerHits',      ele_trkHits),
    ('numPixHits',          ele_pixHits),
    ('numStripHits',        ele_stripHits),
    ('charge',              _charge),
    ('minDRtoReg',          dR),
    ('drNearestEle',        _drNearestEle),
    ('mindRj',              dRj),
    ('mindPhiJ',            dphiJ),
    ('full5x5sigmaIetaIeta', _sieie),
    #('absdEtaSeed',         _dEtaSeed),
    #('absdPhiIn',           _dPhiIn),
    ('HoverE',              _HoE),
    ('abs1overEm1overP',    _invEmP),
    ('expMissingInnerHits', _missHits),
    ('IDscore',             ele_id),
]

# vtx.e1/e2 (Electron/LptElectron legs of a vertex, set up by
# vtxElectronConnection) support most of _ELE_HISTS' fields directly, but
# lack the AllLptElectron-only minDRtoReg/drNearestEle quantities, and expose
# the ID score only as IDscore (no separate 'ID' field to alias to, unlike
# AllLptElectron) -- drop those three entries for the vertexed category's
# per-leg reco hists.
_VTXELE_EXCLUDE = {'minDRtoReg', 'drNearestEle', 'ID'}
_VTXELE_HISTS = [(f, a) for f, a in _ELE_HISTS if f not in _VTXELE_EXCLUDE]

# Vertex-level (i.e. not per-leg) scalar fields stored directly on the vtx
# record itself (analysisTools/analysisSubroutines.py's vtxElectronConnection/
# defineGoodVertices/projectLxy*), not already covered by the vtx_mass/
# vtx_chi2/vtx_pt/vtx_eta/vtx_phi/vtx_dr/vtx_min_dxy hists above (those use
# the vertex-constrained refit_* variants). (hist_key, vtx_field, axis) --
# hist_key differs from vtx_field only where the raw (pre-refit) quantity
# would otherwise collide with an existing refit-based hist name.
# Deliberately NOT included: vx/vy/px/py/pz/energy (redundant with vxy/vz
# and pt/eta/phi/m -- same 4-vector, would just double the axis count for no
# new information), dRJets/dPhiJets (jagged per-jet arrays, already reduced
# into mindRj/mindPhiJ), e{1,2}_idx (a plain array index, not a physical
# quantity).
_VTX_HISTS = [
    ('vxy',        'vxy',        vxy_coarse),     # reco transverse vertex displacement [cm]
    ('sigmavxy',   'sigmavxy',   _vtxVxyErr),      # uncertainty on vxy [cm]
    ('vz',         'vz',         vz_coarse),       # reco vertex z position [cm]
    ('prob',       'prob',       _vtxProb),        # vertex fit chi2 probability
    ('dr_raw',     'dR',         ee_dr),           # e1-e2 dR, pre-refit
    ('sign',       'sign',       vtx_sign),        # q(e1)*q(e2)
    ('min_dxy_raw','minDxy',     dxy_fine),        # ntuple-level min(|dxy|) of the two legs, pre-refit
    ('METdPhi',    'METdPhi',    _metDphi),        # Δφ(vertex p4, MET)
    ('mass_raw',   'm',          vtx_mass),        # dilepton mass, pre-refit
    ('pt_raw',     'pt',         vtx_pt),          # dilepton pt, pre-refit
    ('eta_raw',    'eta',        ele_eta),         # dilepton eta, pre-refit
    ('phi_raw',    'phi',        ele_phi),         # dilepton phi, pre-refit
    ('mindRj',     'mindRj',     dRj),             # min dR to nearest PFJet
    ('mindPhiJ',   'mindPhiJ',   dphiJ),           # min dPhi to nearest PFJet
    ('eleDphi',    'eleDphi',    _eleDphi),        # |Δφ(e1, e2)|
    ('cos_collinear',              'cos_collinear',              _cosCollinear),
    ('cos_collinear_fromPV',       'cos_collinear_fromPV',       _cosCollinear),
    ('cos_collinear_fromPV_refit', 'cos_collinear_fromPV_refit', _cosCollinear),
    ('gen_cos_collinear_fromPV',   'gen_cos_collinear_fromPV',   _cosCollinear),  # gen-level analog, signal MC only
    ('projectedLxy', 'projectedLxy', vxy_coarse),  # vxy projected onto the collinear direction [cm]
    ('vxy_fromPV',   'vxy_fromPV',   vxy_coarse),  # sqrt'd below -- ntuple stores it squared
    ('typ',        'typ',        _vtxType),        # leg-collection combination (LL/LR/RR)
    ('isGood',     'isGood',     _boolFlag),
    ('isMatched',  'isMatched',  _boolFlag),
    ('matchSign',  'matchSign',  _matchSignAx),
    ('bothElePassID',      'bothElePassID',      _boolFlag),
    ('bothElePassIDBasic', 'bothElePassIDBasic', _boolFlag),
]
_VTX_SQRT_FIELDS = {'vxy_fromPV'}
_VTX_BOOL_FIELDS = {'isGood', 'isMatched', 'bothElePassID', 'bothElePassIDBasic'}

# Flat e{1,2}_-prefixed leg fields stored directly on the vtx record itself
# -- distinct from the nested e1/e2 electron objects' own fields already
# covered by _VTXELE_HISTS (e.g. vtx.e1_refit_dxy, the vertex-constrained
# fit result, vs. vtx.e1.dxy, the electron's own pre-fit reco dxy).
# Deliberately NOT included: e{1,2}_idx (array index, not physical).
_VTXLEG_HISTS = [
    ('typ',          _legType),        # which collection (Electron="R"/LptElectron="L") this leg came from
    ('isMatched',    _boolFlag),       # leg-level gen-match flag
    ('matchType',    _legMatchType),   # leg-level gen-match type code
    ('refit_dxy',    dxy_fine),        # vertex-constrained refit dxy [cm]
    ('refit_dxyErr', _vtxLegDxyErr),   # uncertainty on refit dxy [cm]
    ('refit_dz',     ele_dz),          # vertex-constrained refit dz [cm]
    ('refit_dzErr',  _vtxLegDzErr),    # uncertainty on refit dz [cm]
    ('refit_chi2',   ele_chi2),        # vertex-constrained refit leg chi2
]
_VTXLEG_ABS_FIELDS = {'refit_dxy', 'refit_dz'}
_VTXLEG_BOOL_FIELDS = {'isMatched'}

# Fields that are filled as np.abs(...) (signed in the ntuple, but
# histogrammed unsigned).
_ABS_FIELDS = {'vxy', 'vz', 'dxy', 'dz'}

# _ELE_HISTS has two hist-key entries ('ID' and 'IDscore') that both read the
# same underlying 'ID' field -- 'IDscore' isn't itself a field on the ntuple.
# full5x5sigmaIetaIeta is likewise a hist-key alias, not an AllLptElectron
# field on its own -- the real (legacy-named) field is full55sigmaIetaIeta;
# renamed at the hist-key level only so it matches mergedcats.py's mele_/
# mpho_-side naming for the same quantity, for 1:1 cross-category overlay.
_FIELD_ALIASES = {'IDscore': 'ID', 'full5x5sigmaIetaIeta': 'full55sigmaIetaIeta'}

# vtx.e1/e2 alias the same new hist key to the same real field as
# AllLptElectron (both use full55sigmaIetaIeta natively), but unlike
# AllLptElectron they expose the ID score only as a literal 'IDscore' field
# (no separate 'ID' to alias from -- see _VTXELE_HISTS comment), so this
# can't just reuse _FIELD_ALIASES, which would wrongly redirect IDscore->ID.
_VTXELE_FIELD_ALIASES = {'full5x5sigmaIetaIeta': 'full55sigmaIetaIeta'}

# Approximate electron mass [GeV], used only to build a Lorentz vector for
# the two resolved reco legs (see _make_p4) -- negligible next to detector
# resolution at the pt scales histogrammed here, included anyway since it's free.
_ELECTRON_MASS = 0.000511

# ── "Resolved but not vertexed" categorization ───────────────────────────────
# For every cat_sep event, diagnoses why the two separately gen-matched
# AllLptElectron legs (ge_reco/gp_reco below) didn't also make the event
# cat_vertexed -- see analysisSubroutines.computeMergedCatVars for that
# definition (nGoodLptVtx>0 & sel_lptvtx.isMatched, both built from the
# lptvtx pool of AllLptElectron pairs) and AODSkimmer/plugins/
# ElectronSkimmer.cc's computeVertices lambda for how lptvtx candidates are
# built in the first place. Not mutually exclusive -- e.g. a pair can fail
# both the mass and conversion-veto v15acr sub-cuts at once, or dR<0.01 and
# "no candidate" together (the former causes the latter) -- so
# _fillResNotVtxReasons fills res_notVtx_reason once per applicable reason,
# per event.
_resNotVtxReasons = [
    'reco dR < 0.01',
    'no lptvtx object and dr > 0.01',
    'mass < 0.1 GeV',
    'conversion veto FP',
    'other vertex criteria',
    'wrong lptvtx candidate',
    'truth-matching difference',
]
_resNotVtxReasonAx = StrCategory(_resNotVtxReasons, name='reason', label='Why resolved event failed vertexing')

def _fillResNotVtxReasons(hists, events, cat_sep, ge_reco, gp_reco, ge_idx, gp_idx, w_res, samp, cut, dr_fn):
    """See _resNotVtxReasons above. ge_reco/gp_reco/ge_idx/gp_idx are the
    resolved category's two gen-matched AllLptElectron legs (and their
    AllLptElectron indices), already sliced to the cat_sep subset -- same
    convention as the rest of fillHistos's resolved-category block."""
    # ge_idx/gp_idx are None for events where the corresponding gen lepton
    # matched to no AllLptElectron at all (e.g. zero-candidate events at low
    # mass/ctau). Sentinel to -1 (never a real AllLptElectron index) before
    # building lo_idx/hi_idx so the lptvtx_cs.e{1,2}_idx comparisons below
    # broadcast to real False instead of None -- a None there silently
    # Kleene-propagates through every downstream reason_* mask, and indexing
    # w_res with a mask that contains None (rather than pure bool) leaves a
    # stray None entry in the weight array passed to hist.fill(), which is
    # what boost_histogram's "spans must have compatible lengths" error was
    # coming from. -1 correctly falls into the "no lptvtx candidate" reason.
    ge_idx = ak.fill_none(ge_idx, -1)
    gp_idx = ak.fill_none(gp_idx, -1)
    lo_idx = np.minimum(ge_idx, gp_idx)
    hi_idx = np.maximum(ge_idx, gp_idx)

    # (1) The two legs are within the dR<0.01 near-duplicate veto that keeps
    # ElectronSkimmer.cc's computeVertices from ever forming a vertex
    # candidate for this exact pair.
    dr_legs = dr_fn(ge_reco.eta, ge_reco.phi, gp_reco.eta, gp_reco.phi)
    reason_dr = ak.fill_none(dr_legs < 0.01, False)

    # (2) No lptvtx entry at all for this (ge_idx, gp_idx) pair, for legs
    # that don't already fall under (1) -- covers a failed/non-convergent
    # KVF fit (ElectronSkimmer.cc: tv.isValid()). Legs within the dR<0.01
    # veto also have no candidate by construction, but those are counted
    # under reason_dr instead so the two reasons stay mutually exclusive.
    lptvtx_cs   = events.lptvtx[cat_sep]
    match_mask  = (lptvtx_cs.e1_idx == lo_idx) & (lptvtx_cs.e2_idx == hi_idx)
    has_cand    = ak.any(match_mask, axis=1)
    reason_noCand = ~has_cand & ~reason_dr

    candidate = ak.firsts(lptvtx_cs[match_mask])

    # (3)-(5) The v15acr good-vertex sub-cuts (defineGoodLptVertices), split
    # into their own dedicated categories (mass, conversion veto) plus a
    # catch-all ('other') for every remaining sub-cut. Reuses
    # candidate.isGood (rather than re-deriving the full v15acr formula
    # here) so this stays in sync with any future change to that definition.
    # mass is checked first so an event failing both mass and conversion
    # veto lands only in reason_mass, keeping the categories exclusive.
    cand_mass_ok     = ak.fill_none(candidate.refit_m > 0.1, False)
    cand_convveto_ok = ak.fill_none(candidate.e1.conversionVeto & candidate.e2.conversionVeto, False)
    cand_good        = ak.fill_none(candidate.isGood, False)

    reason_mass     = has_cand & ~cand_mass_ok
    reason_convVeto = has_cand & cand_mass_ok & ~cand_convveto_ok
    reason_other    = has_cand & ~cand_good & cand_mass_ok & cand_convveto_ok

    # (6)-(7) The candidate is a genuinely good vertex (so nGoodLptVtx>0 for
    # this event) -- but the event still landed in cat_sep, which by
    # computeMergedCatVars's definition can only mean either (6) it lost the
    # min-chi2 selectBestLptVertex race to some other good vertex in the
    # event, or (7) it *was* selected but sel_lptvtx.isMatched came back
    # False -- i.e. the ntuple-level, dR-only/unique AllLowPt truth-match
    # assignment (ElectronSkimmer.cc) disagreed with the charge+pT-filtered
    # gen-match used to define ge_reco/gp_reco/cat_sep here.
    sel_vtx = events.sel_lptvtx[cat_sep]
    sel_e1  = ak.fill_none(sel_vtx.e1_idx, -1)
    sel_e2  = ak.fill_none(sel_vtx.e2_idx, -1)
    is_sel  = (sel_e1 == lo_idx) & (sel_e2 == hi_idx)

    reason_wrongVtx  = has_cand & cand_good & ~is_sel
    reason_truthDiff = has_cand & cand_good & is_sel

    _masks = [reason_dr, reason_noCand, reason_mass, reason_convVeto,
              reason_other, reason_wrongVtx, reason_truthDiff]
    for reason, mask in zip(_resNotVtxReasons, _masks):
        n = ak.sum(mask)
        if n > 0:
            # hist.fill needs at least one array-valued axis input to infer
            # the fill length -- with samp/cut/reason all scalar strings and
            # only weight array-valued, boost_histogram raises "spans must
            # have compatible lengths" instead of broadcasting the scalars.
            # Broadcasting reason into an array (matching weight's length)
            # sidesteps that.
            hists['res_notVtx_reason'].fill(samp=samp, cut=cut, reason=[reason] * n, weight=w_res[mask])

def _pt_order(a, b):
    """Split two per-event, same-schema electron records into (lead, sublead)
    by reco pt -- the convention used throughout this module for splitting
    per-electron hists, since only reco pt is available in data."""
    a_is_lead = a.pt >= b.pt
    return ak.where(a_is_lead, a, b), ak.where(a_is_lead, b, a)

def _fill_single(hists, prefix, obj, evt_weight, samp, cut, field_hists):
    """Fill `{prefix}_{field}` hists with one (already-selected, non-jagged)
    reco object per event.
    """
    for field, axis in field_hists:
        val = getattr(obj, _FIELD_ALIASES.get(field, field))
        if field in _ABS_FIELDS:
            val = np.abs(val)
        hists[f'{prefix}_{field}'].fill(
            samp=samp, cut=cut, weight=evt_weight,
            **{axis.name: val},
        )

def _fill_vtxele(hists, prefix, obj, evt_weight, samp, cut, field_hists):
    """Like _fill_single, but for vtx.e1/e2 objects, using _VTXELE_FIELD_ALIASES
    (not _FIELD_ALIASES -- see _VTXELE_HISTS comment) since IDscore needs no
    alias here but full5x5sigmaIetaIeta still does."""
    for field, axis in field_hists:
        val = getattr(obj, _VTXELE_FIELD_ALIASES.get(field, field))
        if field in _ABS_FIELDS:
            val = np.abs(val)
        hists[f'{prefix}_{field}'].fill(
            samp=samp, cut=cut, weight=evt_weight,
            **{axis.name: val},
        )

def _fill_resolved_ele(hists, prefix, re, dr_other, evt_weight, samp, cut):
    """Fill `{prefix}_{field}` hists for one resolved (gen-matched)
    AllLptElectron, substituting the caller-supplied dr_other (dR to the
    nearest *non*-gen-matched AllLptElectron) for the generic drNearestEle
    field -- AllLptElectron.drNearestEle (computeMergedCatVars) measures dR
    to the nearest AllLptElectron of any kind, which for a resolved leg is
    usually its own gen-matched partner, not a genuine nearby extra track.
    """
    for field, axis in _ELE_HISTS:
        if field == 'drNearestEle':
            val = dr_other
        else:
            val = getattr(re, _FIELD_ALIASES.get(field, field))
            if field in _ABS_FIELDS:
                val = np.abs(val)
        hists[f'{prefix}_{field}'].fill(
            samp=samp, cut=cut, weight=evt_weight,
            **{axis.name: val},
        )

def _fill_vtx(hists, prefix, v, evt_weight, samp, cut, field_hists):
    """Fill `{prefix}_{key}` hists with one vertex-level (i.e. not per-leg)
    quantity stored directly on the vtx record -- see _VTX_HISTS."""
    for key, field, axis in field_hists:
        val = getattr(v, field)
        if field in _VTX_SQRT_FIELDS:
            val = np.sqrt(val)
        elif field in _VTX_BOOL_FIELDS:
            val = ak.values_astype(val, np.int32)
        hists[f'{prefix}_{key}'].fill(
            samp=samp, cut=cut, weight=evt_weight,
            **{axis.name: val},
        )

def _fill_vtxleg_flat(hists, prefix, v, pick_e1, evt_weight, samp, cut, field_hists):
    """Fill `{prefix}_{field}` hists from the flat vtx.e{1,2}_{field}
    branches (leg-level scalars stored on the vertex record itself -- see
    _VTXLEG_HISTS), picking leg 1 or 2 per-event via the pick_e1 mask."""
    for field, axis in field_hists:
        val = ak.where(pick_e1, getattr(v, f'e1_{field}'), getattr(v, f'e2_{field}'))
        if field in _VTXLEG_ABS_FIELDS:
            val = np.abs(val)
        elif field in _VTXLEG_BOOL_FIELDS:
            val = ak.values_astype(val, np.int32)
        hists[f'{prefix}_{field}'].fill(
            samp=samp, cut=cut, weight=evt_weight,
            **{axis.name: val},
        )

def _make_p4(obj):
    """Build a Lorentz vector for a bare pt/eta/phi electron record so two
    legs can be added to get the resolved category's reco dilepton
    kinematics (mass/pt/eta/phi) -- there's no fitted vertex for this
    category, so unlike vtx.m/pt/eta/phi this has to be built by hand here.
    Mirrors the ak.with_name(...,'PtEtaPhiMLorentzVector') pattern used for
    jets in analysisTools/corrections.py."""
    return ak.with_name(
        ak.zip({'pt': obj.pt, 'eta': obj.eta, 'phi': obj.phi, 'mass': ak.full_like(obj.pt, _ELECTRON_MASS)}),
        'PtEtaPhiMLorentzVector',
    )

def make_histograms():
    histograms = {}

    # ── Resolved category (AllLptElectron reco variables), split by reco pt
    # ordering into a leading and a subleading electron ───────────────────────
    for tag in _LEAD_TAGS:
        for field, axis in _ELE_HISTS:
            histograms[f'res_{tag}_{field}'] = Hist(samp, cut, axis, storage=_STORAGE)
    histograms['res_gen_lxy']        = Hist(samp, cut, _gen_lxy, storage=_STORAGE)
    histograms['res_gen_ee_pt']      = Hist(samp, cut, ele_pt,      storage=_STORAGE)
    histograms['res_gen_lead_pt']    = Hist(samp, cut, ele_pt,      storage=_STORAGE)
    histograms['res_gen_sublead_pt'] = Hist(samp, cut, ele_pt,      storage=_STORAGE)
    histograms['res_gen_ee_dr']      = Hist(samp, cut, ee_dr,       storage=_STORAGE)
    histograms['res_gen_ee_eta']     = Hist(samp, cut, ele_eta,     storage=_STORAGE)
    histograms['res_gen_lead_eta']   = Hist(samp, cut, ele_eta,     storage=_STORAGE)
    histograms['res_gen_sublead_eta']= Hist(samp, cut, ele_eta,     storage=_STORAGE)

    # ── Resolved "nearest other electron" realism check: same reco variables
    # (_ELE_HISTS), but for the single AllLptElectron closest to each resolved
    # (gen-matched) electron (the object responsible for the low-dR entries
    # in res_{lead,sublead}_drNearestEle) — pt/trkChi2/hit-count/IDscore etc.
    # tell us whether that nearby object looks like a real, well-reconstructed
    # electron or a low-quality/duplicate GSF track candidate.
    for tag in _LEAD_TAGS:
        for field, axis in _ELE_HISTS:
            histograms[f'res_{tag}_nearestEle_{field}'] = Hist(samp, cut, axis, storage=_STORAGE)

    # ── Resolved category: reco dilepton (pair-level) quantities, analogous
    # to the vtx object's own vertex-level fields (_VTX_HISTS) -- but there's
    # no fitted vertex here, so only what's reconstructible from the two
    # resolved legs' own 4-vectors is available: no vxy/vz/prob/sigmavxy/typ/
    # isGood/isMatched/matchSign/cos_collinear*/projectedLxy/refit_* analog
    # exists, since all of those require an actual vertex fit.
    # Named res_ee_* (mirroring the existing gen-level res_gen_ee_* pair
    # naming), not bare res_* -- res_pt/eta/phi/etc. are reserved for the
    # per-electron res_{lead,sublead}_* comparison mergedcat_plots.py expects
    # (mirroring mele_pt/mpho_pt), a different quantity than the pair's own
    # pt/eta/phi, so reusing that bare name here would collide with it.
    histograms['res_ee_dr']      = Hist(samp, cut, ee_dr,    storage=_STORAGE)
    histograms['res_ee_sign']    = Hist(samp, cut, vtx_sign, storage=_STORAGE)
    histograms['res_ee_eleDphi'] = Hist(samp, cut, _eleDphi, storage=_STORAGE)
    histograms['res_ee_mass']    = Hist(samp, cut, vtx_mass, storage=_STORAGE)
    histograms['res_ee_pt']      = Hist(samp, cut, vtx_pt,   storage=_STORAGE)
    histograms['res_ee_eta']     = Hist(samp, cut, ele_eta,  storage=_STORAGE)
    histograms['res_ee_phi']     = Hist(samp, cut, ele_phi,  storage=_STORAGE)
    histograms['res_ee_METdPhi'] = Hist(samp, cut, _metDphi, storage=_STORAGE)

    # ── Resolved-but-not-vertexed diagnostic: why didn't this event's
    # resolved pair also make it into the vertexed category? See
    # _resNotVtxReasons / _fillResNotVtxReasons above.
    histograms['res_notVtx_reason'] = Hist(samp, cut, _resNotVtxReasonAx, storage=_STORAGE)

    # ── Vertexed category (gen kinematics only — supersedes all other cats) ──
    histograms['vtx_gen_lxy']        = Hist(samp, cut, _gen_lxy, storage=_STORAGE)
    histograms['vtx_gen_ee_pt']      = Hist(samp, cut, ele_pt,      storage=_STORAGE)
    histograms['vtx_gen_lead_pt']    = Hist(samp, cut, ele_pt,      storage=_STORAGE)
    histograms['vtx_gen_sublead_pt'] = Hist(samp, cut, ele_pt,      storage=_STORAGE)
    histograms['vtx_gen_ee_dr']      = Hist(samp, cut, ee_dr,       storage=_STORAGE)
    histograms['vtx_gen_ee_eta']     = Hist(samp, cut, ele_eta,     storage=_STORAGE)
    histograms['vtx_gen_lead_eta']   = Hist(samp, cut, ele_eta,     storage=_STORAGE)
    histograms['vtx_gen_sublead_eta']= Hist(samp, cut, ele_eta,     storage=_STORAGE)

    # ── Vertexed category: vertex-level reco quantities ───────────────────────
    histograms['vtx_mass']    = Hist(samp, cut, vtx_mass,  storage=_STORAGE)
    histograms['vtx_chi2']    = Hist(samp, cut, vtx_chi2,  storage=_STORAGE)
    histograms['vtx_pt']      = Hist(samp, cut, vtx_pt,    storage=_STORAGE)
    histograms['vtx_eta']     = Hist(samp, cut, ele_eta,   storage=_STORAGE)
    histograms['vtx_phi']     = Hist(samp, cut, ele_phi,   storage=_STORAGE)
    histograms['vtx_dr']      = Hist(samp, cut, ee_dr,     storage=_STORAGE)
    histograms['vtx_min_dxy'] = Hist(samp, cut, dxy_fine,  storage=_STORAGE)

    # ── Vertexed category: every other vertex-level scalar field stored on
    # the vtx record itself, not already covered above -- see _VTX_HISTS.
    for key, field, axis in _VTX_HISTS:
        histograms[f'vtx_{key}'] = Hist(samp, cut, axis, storage=_STORAGE)

    # ── Vertexed category: per-leg reco quantities, split by reco pt ordering
    # into a leading and a subleading leg (mirrors configs/histo_configs/
    # vtxvars.py's per-leg fields, but pt-ordered instead of unified) ──────────
    for tag in _LEAD_TAGS:
        for field, axis in _VTXELE_HISTS:
            histograms[f'vtx_ele_{tag}_{field}'] = Hist(samp, cut, axis, storage=_STORAGE)

    # ── Vertexed category: flat e{1,2}_-prefixed leg fields stored on the vtx
    # record itself (distinct from the nested e1/e2 electron objects' own
    # fields just above), also split lead/sublead by reco pt -- see
    # _VTXLEG_HISTS.
    for tag in _LEAD_TAGS:
        for field, axis in _VTXLEG_HISTS:
            histograms[f'vtx_ele_{tag}_{field}'] = Hist(samp, cut, axis, storage=_STORAGE)

    # ── Vertexed "nearest other electron" realism check / relational vars:
    # same rationale as the res_*_nearestEle_* hists above -- for each
    # pt-ordered vertex leg, the single nearest AllLptElectron not already
    # gen-matched to either gen (coll.is_matched, a category-independent flag
    # -- see computeMergedCatVars -- so it naturally excludes both vertex
    # legs themselves whenever they are also AllLptElectron entries, since a
    # truth-matched vertex's legs are by construction gen-matched).
    for tag in _LEAD_TAGS:
        histograms[f'vtx_ele_{tag}_drNearestEle'] = Hist(samp, cut, _drNearestEle, storage=_STORAGE)
        for field, axis in _ELE_HISTS:
            histograms[f'vtx_ele_{tag}_nearestEle_{field}'] = Hist(samp, cut, axis, storage=_STORAGE)
        histograms[f'vtx_ele_{tag}_nearestEle_dPt']        = Hist(samp, cut, _dPt,      storage=_STORAGE)
        histograms[f'vtx_ele_{tag}_nearestEle_dPtRel']     = Hist(samp, cut, _dPtRel,   storage=_STORAGE)
        histograms[f'vtx_ele_{tag}_nearestEle_chargeProd'] = Hist(samp, cut, vtx_sign,  storage=_STORAGE)
        histograms[f'vtx_ele_{tag}_nearestEle_dVxy']       = Hist(samp, cut, _dVxyNear, storage=_STORAGE)
        histograms[f'vtx_ele_{tag}_nearestEle_dVz']        = Hist(samp, cut, _dVzNear,  storage=_STORAGE)
        histograms[f'vtx_ele_{tag}_nearestEle_dDxy']       = Hist(samp, cut, _dDxyNear, storage=_STORAGE)
        histograms[f'vtx_ele_{tag}_nearestEle_dDz']        = Hist(samp, cut, _dDzNear,  storage=_STORAGE)
        histograms[f'vtx_ele_{tag}_nearestEle_drZoom']     = Hist(samp, cut, dR_zoom,   storage=_STORAGE)

    return histograms

# computeMergedCatVars (analysisTools/analysisSubroutines.py) runs once per
# chunk, before the cut loop, and attaches the vertexed/resolved
# categorization (cat_vertexed, vertexed_vtx, cat_sep, gen_lxy,
# AllLptElectron.{is_matched,dr_to_ge,dr_to_gp}) that fillHistos below reads
# back at each cut stage, instead of recomputing it from scratch every time
# (an_selection.py currently fills at 6 stages). cat_vertexed (a good,
# truth-matched ee vertex, v15acr -- see analysisSubroutines.computeMergedCatVars)
# supersedes cat_sep: cat_sep is only True for events that are *not* vertexed.
# vertexed_vtx is the flat, per-event vertex record (only meaningful where
# cat_vertexed is True) used to fill this category's own vtx_*/vtx_ele_*
# reco hists below.
subroutines = ['computeMergedCatVars']

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    if RAW_COUNTS:
        wgt = ak.ones_like(events.eventWgt)
    else:
        wgt = events.eventWgt / sum_wgt

    # gen_lxy and the resolved categorization (cat_sep, plus the
    # AllLptElectron.{is_matched,dr_to_ge,dr_to_gp} fields used below) are
    # computed once per chunk by computeMergedCatVars
    # (analysisTools/analysisSubroutines.py, registered in this module's
    # `subroutines` list) rather than recomputed here on every
    # savePlots=True cut stage (an_selection.py currently has 6).
    gen_lxy = events.gen_lxy
    coll    = events.AllLptElectron

    _cat_sep = events.cat_sep

    # ── Helpers (used below on already-category-selected, small subsets) ────
    def _dphi(a, b):
        d = np.abs(a - b)
        return ak.where(d > np.pi, 2 * np.pi - d, d)

    def _dr(eta1, phi1, eta2, phi2):
        return np.sqrt((eta1 - eta2)**2 + _dphi(phi1, phi2)**2)

    # ── Resolved fills ─────────────────────────────────────────────────────────
    # Gen kinematics for all resolved events (classical + ambiguous).
    if ak.sum(_cat_sep) > 0:
        w_res   = wgt[_cat_sep]
        lxy_res = gen_lxy[_cat_sep]
        ge_res  = events.GenEle[_cat_sep]
        gp_res  = events.GenPos[_cat_sep]

        lead_is_e_res   = ge_res.pt >= gp_res.pt
        lead_pt_res     = ak.where(lead_is_e_res, ge_res.pt,  gp_res.pt)
        sublead_pt_res  = ak.where(lead_is_e_res, gp_res.pt,  ge_res.pt)
        lead_eta_res    = ak.where(lead_is_e_res, ge_res.eta, gp_res.eta)
        sublead_eta_res = ak.where(lead_is_e_res, gp_res.eta, ge_res.eta)

        hists['res_gen_lxy'        ].fill(samp=samp, cut=cut, lxy=lxy_res,                          weight=w_res)
        hists['res_gen_ee_pt'      ].fill(samp=samp, cut=cut, pt=events.genEE.pt[_cat_sep],         weight=w_res)
        hists['res_gen_lead_pt'    ].fill(samp=samp, cut=cut, pt=lead_pt_res,                       weight=w_res)
        hists['res_gen_sublead_pt' ].fill(samp=samp, cut=cut, pt=sublead_pt_res,                    weight=w_res)
        hists['res_gen_ee_dr'      ].fill(samp=samp, cut=cut, dr=events.genEE.dr[_cat_sep],         weight=w_res)
        hists['res_gen_ee_eta'     ].fill(samp=samp, cut=cut, eta=events.genEE.eta[_cat_sep],        weight=w_res)
        hists['res_gen_lead_eta'   ].fill(samp=samp, cut=cut, eta=lead_eta_res,                     weight=w_res)
        hists['res_gen_sublead_eta'].fill(samp=samp, cut=cut, eta=sublead_eta_res,                  weight=w_res)

        # Reco fills: closest matched reco to each gen particle.
        # Covers both classical (exclusive one-to-one) and ambiguous (merged + extra) subcases.
        dr_ge_masked = ak.where(coll.is_matched, coll.dr_to_ge, np.inf)
        dr_gp_masked = ak.where(coll.is_matched, coll.dr_to_gp, np.inf)
        ge_reco = ak.firsts(coll[ak.argmin(dr_ge_masked, axis=1, keepdims=True)])[_cat_sep]
        gp_reco = ak.firsts(coll[ak.argmin(dr_gp_masked, axis=1, keepdims=True)])[_cat_sep]

        # AllLptElectron indices of ge_reco/gp_reco -- needed below to find
        # the corresponding lptvtx candidate (if any) for _fillResNotVtxReasons.
        idx_coll = ak.local_index(coll, axis=1)
        ge_idx = ak.firsts(idx_coll[ak.argmin(dr_ge_masked, axis=1, keepdims=True)])[_cat_sep]
        gp_idx = ak.firsts(idx_coll[ak.argmin(dr_gp_masked, axis=1, keepdims=True)])[_cat_sep]

        # Split the two gen-matched reco electrons by reco pt, not by which
        # gen particle they matched -- see _LEAD_TAGS.
        lead_reco, sublead_reco = _pt_order(ge_reco, gp_reco)

        # AllLptElectron objects not matched to either gen -- ge_reco/gp_reco
        # are gen-matched by definition, so their nearest-electron dR should
        # be measured against a genuine third (non-gen-matched) electron, not
        # against each other.
        res_other_ele = coll[~coll.is_matched][_cat_sep]

        for tag, re in zip(_LEAD_TAGS, (lead_reco, sublead_reco)):
            _dr_re_other = _dr(res_other_ele.eta, res_other_ele.phi, re.eta, re.phi)
            dr_re_other  = ak.fill_none(ak.min(_dr_re_other, axis=1), 999)

            _fill_resolved_ele(hists, f'res_{tag}', re, dr_re_other, w_res, samp, cut)

            # Realism check on the single nearest other AllLptElectron (the
            # one realizing the min in res_{tag}_drNearestEle), restricted to
            # dR<0.2 to isolate the near-duplicate population behind the
            # res_{tag}_drNearestEle spike at dR~0.
            _near_res_mask = dr_re_other < 0.2
            if ak.sum(_near_res_mask) > 0:
                _near_idx_re  = ak.argmin(_dr_re_other, axis=1, keepdims=True)
                _nearest_res  = ak.firsts(res_other_ele[_near_idx_re])[_near_res_mask]
                _fill_single(hists, f'res_{tag}_nearestEle', _nearest_res, w_res[_near_res_mask], samp, cut, _ELE_HISTS)

        # Reco dilepton (pair-level) quantities, analogous to the vtx
        # object's own vertex-level fields where a fitted vertex isn't
        # required to define them -- see _VTX_HISTS comment for what has no
        # analog here (anything needing an actual vertex fit).
        res_pair = _make_p4(lead_reco) + _make_p4(sublead_reco)
        res_dr_ll     = _dr(lead_reco.eta, lead_reco.phi, sublead_reco.eta, sublead_reco.phi)
        res_dphi_ll   = _dphi(lead_reco.phi, sublead_reco.phi)
        res_sign_ll   = lead_reco.charge * sublead_reco.charge
        res_dphi_met  = _dphi(res_pair.phi, events.PFMET.phi[_cat_sep])

        hists['res_ee_dr'     ].fill(samp=samp, cut=cut, dr=res_dr_ll,          weight=w_res)
        hists['res_ee_sign'   ].fill(samp=samp, cut=cut, sign=res_sign_ll,      weight=w_res)
        hists['res_ee_eleDphi'].fill(samp=samp, cut=cut, eleDphi=res_dphi_ll,   weight=w_res)
        hists['res_ee_mass'   ].fill(samp=samp, cut=cut, mass=res_pair.mass,    weight=w_res)
        hists['res_ee_pt'     ].fill(samp=samp, cut=cut, pt=res_pair.pt,        weight=w_res)
        hists['res_ee_eta'    ].fill(samp=samp, cut=cut, eta=res_pair.eta,      weight=w_res)
        hists['res_ee_phi'    ].fill(samp=samp, cut=cut, phi=res_pair.phi,      weight=w_res)
        hists['res_ee_METdPhi'].fill(samp=samp, cut=cut, METdPhi=res_dphi_met,  weight=w_res)

        # Why this resolved event isn't also vertexed -- see
        # _resNotVtxReasons / _fillResNotVtxReasons.
        _fillResNotVtxReasons(hists, events, _cat_sep, ge_reco, gp_reco, ge_idx, gp_idx, w_res, samp, cut, _dr)

    # ── Vertexed fills ─────────────────────────────────────────────────────────
    # Definition from computeMergedCatVars: a good (v15acr), truth-matched ee
    # vertex. Supersedes cat_sep (and cat_merged/cat_merged_photon/cat_zero,
    # not used by this config).
    _cat_vertexed = events.cat_vertexed
    if ak.sum(_cat_vertexed) > 0:
        w_vtx   = wgt[_cat_vertexed]
        lxy_vtx = gen_lxy[_cat_vertexed]
        ge_vtx  = events.GenEle[_cat_vertexed]
        gp_vtx  = events.GenPos[_cat_vertexed]

        lead_is_e_vtx   = ge_vtx.pt >= gp_vtx.pt
        lead_pt_vtx     = ak.where(lead_is_e_vtx, ge_vtx.pt,  gp_vtx.pt)
        sublead_pt_vtx  = ak.where(lead_is_e_vtx, gp_vtx.pt,  ge_vtx.pt)
        lead_eta_vtx    = ak.where(lead_is_e_vtx, ge_vtx.eta, gp_vtx.eta)
        sublead_eta_vtx = ak.where(lead_is_e_vtx, gp_vtx.eta, ge_vtx.eta)

        hists['vtx_gen_lxy'        ].fill(samp=samp, cut=cut, lxy=lxy_vtx,                          weight=w_vtx)
        hists['vtx_gen_ee_pt'      ].fill(samp=samp, cut=cut, pt=events.genEE.pt[_cat_vertexed],    weight=w_vtx)
        hists['vtx_gen_lead_pt'    ].fill(samp=samp, cut=cut, pt=lead_pt_vtx,                       weight=w_vtx)
        hists['vtx_gen_sublead_pt' ].fill(samp=samp, cut=cut, pt=sublead_pt_vtx,                    weight=w_vtx)
        hists['vtx_gen_ee_dr'      ].fill(samp=samp, cut=cut, dr=events.genEE.dr[_cat_vertexed],    weight=w_vtx)
        hists['vtx_gen_ee_eta'     ].fill(samp=samp, cut=cut, eta=events.genEE.eta[_cat_vertexed],   weight=w_vtx)
        hists['vtx_gen_lead_eta'   ].fill(samp=samp, cut=cut, eta=lead_eta_vtx,                     weight=w_vtx)
        hists['vtx_gen_sublead_eta'].fill(samp=samp, cut=cut, eta=sublead_eta_vtx,                  weight=w_vtx)

        # Reco fills: the actual good, truth-matched vertex (v15acr) selected
        # for each vertexed event -- see analysisSubroutines.computeMergedCatVars.
        v = events.vertexed_vtx[_cat_vertexed]

        hists['vtx_mass'   ].fill(samp=samp, cut=cut, mass=v.refit_m,                      weight=w_vtx)
        hists['vtx_chi2'   ].fill(samp=samp, cut=cut, chi2=v.reduced_chi2,                 weight=w_vtx)
        hists['vtx_pt'     ].fill(samp=samp, cut=cut, pt=v.refit_pt,                       weight=w_vtx)
        hists['vtx_eta'    ].fill(samp=samp, cut=cut, eta=v.refit_eta,                     weight=w_vtx)
        hists['vtx_phi'    ].fill(samp=samp, cut=cut, phi=v.refit_phi,                     weight=w_vtx)
        hists['vtx_dr'     ].fill(samp=samp, cut=cut, dr=v.refit_dR,                        weight=w_vtx)
        hists['vtx_min_dxy'].fill(samp=samp, cut=cut,
                                   dxy=np.minimum(np.abs(v.e1_refit_dxy), np.abs(v.e2_refit_dxy)), weight=w_vtx)

        # Every other vertex-level scalar field stored on the vtx record
        # itself -- see _VTX_HISTS.
        _fill_vtx(hists, 'vtx', v, w_vtx, samp, cut, _VTX_HISTS)

        # Split the two vertex legs by reco pt -- see _LEAD_TAGS.
        lead_leg, sublead_leg = _pt_order(v.e1, v.e2)
        _e1_is_lead_vtx = v.e1.pt >= v.e2.pt

        _fill_vtxele(hists, 'vtx_ele_lead',    lead_leg,    w_vtx, samp, cut, _VTXELE_HISTS)
        _fill_vtxele(hists, 'vtx_ele_sublead', sublead_leg, w_vtx, samp, cut, _VTXELE_HISTS)

        # Flat e{1,2}_-prefixed leg fields stored on the vtx record itself,
        # pt-ordered the same way as the nested e1/e2 objects just above --
        # see _VTXLEG_HISTS.
        _fill_vtxleg_flat(hists, 'vtx_ele_lead',    v, _e1_is_lead_vtx,  w_vtx, samp, cut, _VTXLEG_HISTS)
        _fill_vtxleg_flat(hists, 'vtx_ele_sublead', v, ~_e1_is_lead_vtx, w_vtx, samp, cut, _VTXLEG_HISTS)

        # "Nearest other electron" realism check: for each pt-ordered vertex
        # leg, the single nearest AllLptElectron not already gen-matched to
        # either gen (coll.is_matched is computed once, independent of
        # category, by computeMergedCatVars -- so it excludes both vertex
        # legs themselves whenever they are also AllLptElectron entries,
        # since a truth-matched vertex's legs are by construction
        # gen-matched). Restricted to dR<0.2, same as res_*_nearestEle, to
        # isolate the near-duplicate population behind the
        # vtx_ele_{tag}_drNearestEle spike at dR~0.
        _vtx_other_ele = coll[~coll.is_matched][_cat_vertexed]
        if ak.sum(ak.num(_vtx_other_ele, axis=1)) > 0:
            for tag, _leg in zip(_LEAD_TAGS, (lead_leg, sublead_leg)):
                _dr_leg_other = _dr(_vtx_other_ele.eta, _vtx_other_ele.phi, _leg.eta, _leg.phi)
                dr_leg_other  = ak.fill_none(ak.min(_dr_leg_other, axis=1), 999)
                hists[f'vtx_ele_{tag}_drNearestEle'].fill(samp=samp, cut=cut, drNearestEle=dr_leg_other, weight=w_vtx)

                _near_vtx_mask = dr_leg_other < 0.2
                if ak.sum(_near_vtx_mask) > 0:
                    _near_idx_leg = ak.argmin(_dr_leg_other, axis=1, keepdims=True)
                    _nearest_vtx  = ak.firsts(_vtx_other_ele[_near_idx_leg])[_near_vtx_mask]
                    w_near_vtx    = w_vtx[_near_vtx_mask]
                    _fill_single(hists, f'vtx_ele_{tag}_nearestEle', _nearest_vtx, w_near_vtx, samp, cut, _ELE_HISTS)

                    # Relational variables vs. the vertex leg itself.
                    _leg_near       = _leg[_near_vtx_mask]
                    _dpt_vtx        = _leg_near.pt - _nearest_vtx.pt
                    _chargeProd_vtx = _leg_near.charge * _nearest_vtx.charge
                    hists[f'vtx_ele_{tag}_nearestEle_dPt'       ].fill(samp=samp, cut=cut, dpt=_dpt_vtx, weight=w_near_vtx)
                    hists[f'vtx_ele_{tag}_nearestEle_dPtRel'    ].fill(samp=samp, cut=cut, dptRel=_dpt_vtx / _leg_near.pt, weight=w_near_vtx)
                    hists[f'vtx_ele_{tag}_nearestEle_chargeProd'].fill(samp=samp, cut=cut, sign=_chargeProd_vtx, weight=w_near_vtx)
                    hists[f'vtx_ele_{tag}_nearestEle_dVxy'      ].fill(samp=samp, cut=cut, dvxy=np.abs(_leg_near.vxy - _nearest_vtx.vxy), weight=w_near_vtx)
                    hists[f'vtx_ele_{tag}_nearestEle_dVz'       ].fill(samp=samp, cut=cut, dvz=np.abs(_leg_near.vz  - _nearest_vtx.vz),   weight=w_near_vtx)
                    hists[f'vtx_ele_{tag}_nearestEle_dDxy'      ].fill(samp=samp, cut=cut, ddxy=np.abs(_leg_near.dxy - _nearest_vtx.dxy), weight=w_near_vtx)
                    hists[f'vtx_ele_{tag}_nearestEle_dDz'       ].fill(samp=samp, cut=cut, ddz=np.abs(_leg_near.dz  - _nearest_vtx.dz),   weight=w_near_vtx)
                    hists[f'vtx_ele_{tag}_nearestEle_drZoom'    ].fill(samp=samp, cut=cut, dr_zoom=dr_leg_other[_near_vtx_mask],          weight=w_near_vtx)
