try:
    from histobins import *
except ModuleNotFoundError:
    from configs.histo_configs.histobins import *
from hist import Hist
from hist.axis import Regular, Variable, StrCategory, IntCategory
import hist
import numpy as np
import awkward as ak

# ptmiss studies: trigger efficiency, resolution/response, and a comparison of
# the ptmiss definitions stored in the ntuples.
#
# Two entry points, both called by iDMeProcessor (analysisTools/analysisTools.py):
#
#   fillSlimHistos : bkg "{name}__slim" datasets, i.e. every generated event with
#                    no HLT/ptmiss gate. This is the only inclusive background
#                    population, so all bkg trigger efficiencies come from here.
#                    Filled under the 'cut' labels in SLIM_CUTS.
#   fillHistos     : full trees (signal, and bkg/data downstream of the gate),
#                    after every cut with savePlots=True. Trigger efficiencies are
#                    filled here for signal only (bkg/data full trees already sit
#                    behind the METNoMu120 OR), and only at TRIG_EFF_CUTS.
#
# Run the background with histmaker_background.py and process_slim = True (with
# a slim sample config next to the main one); signal with histmaker_signal.py.
#
# ptmiss definitions (MET_DEFS). The ones marked "new" were added in commit
# 36e4455 and only exist in ntuples made after it; every fill checks for the
# branch and skips it when absent, so this config also runs on older ntuples
# (pf and calo only, no gen-lepton split).
#   pf       : Type-1 PF ptmiss                               PFMET_pt
#   pfraw    : raw PF ptmiss                          (new)   PFMET_rawPt
#   puppi    : Type-1 PUPPI ptmiss                    (new)   PuppiMET_pt
#   calo     : calo ptmiss                                    CaloMET_pt
#   pfnomu   : Type-1 PF ptmiss + PF-muon pT          (new)   PFMETNoMu_pt
#   mhtnomu  : PF MHT without muons, tight-ID jets    (new)   PFMHTNoMu_pt
#   trignomu : min(pfnomu, mhtnomu), the offline analogue of the
#              HLT_PFMETNoMu120_PFMHTNoMu120 thresholds   (derived)
# Slim trees store only pt for the new definitions (phi is full-tree only), and
# only GenMET_pt (no phi). So the vector studies (met_upar, met_uperp,
# met_dphi_gen) fill from full trees only.
#
# Gen-lepton split (axis 'genlep', MC with the new GenPromptLep_* branches):
# prompt e/mu (isPromptFinalState, not from tau decays) and prompt taus. '0l',
# '1e', '1mu', '1tau', '2l+'; 'na' when the branches are missing (old ntuples,
# data). Signal electrons from chi2 -> chi1 ee are also prompt, so signal ends
# up mostly in '2l+'.
#
# Efficiencies: eff = h[{'passTrig': 1}] / h[{'passTrig': sum}], for ptmiss >=
# TRIG_MIN_MET (lower bins are left empty). In trig_eff also choose the
# reference triggers: sum over passIsoMu27/passEle32 for the plain MC
# efficiency, or select 1 on one of them. The hists are
# weighted (Weight storage), so for Clopper-Pearson errors use effective counts
# n_eff = sum(w)^2 / sum(w^2) for the numerator and the denominator.
#
# Resolution: genMetTrue counts neutrinos and BSM invisibles but not muons, so
# the NoMu definitions are compared with GenMET only in events with no prompt
# gen muon (GenPromptLep_nMu == 0).

# ---- configuration ----

MET_DEFS = {
    # name: (collection, pt field, phi field)
    'pf':      ('PFMET',     'pt',    'phi'),
    'pfraw':   ('PFMET',     'rawPt', 'rawPhi'),
    'puppi':   ('PuppiMET',  'pt',    'phi'),
    'calo':    ('CaloMET',   'pt',    'phi'),
    'pfnomu':  ('PFMETNoMu', 'pt',    'phi'),
    'mhtnomu': ('PFMHTNoMu', 'pt',    'phi'),
}
MET_VARS = list(MET_DEFS) + ['trignomu']
NOMU_VARS = {'pfnomu', 'mhtnomu', 'trignomu'}

# for plotting scripts
MET_LABELS = {
    'pf':       r'$p_T^{miss}$ (PF)',
    'pfraw':    r'$p_T^{miss}$ (raw PF)',
    'puppi':    r'$p_T^{miss}$ (PUPPI)',
    'calo':     r'$p_T^{miss}$ (Calo)',
    'pfnomu':   r'$p_{T,\mathrm{no}\,\mu}^{miss}$ (PF)',
    'mhtnomu':  r'$H_{T,\mathrm{no}\,\mu}^{miss}$ (PF)',
    'trignomu': r'min($p_{T,\mathrm{no}\,\mu}^{miss}$, $H_{T,\mathrm{no}\,\mu}^{miss}$)',
    'gen':      r'$p_T^{miss,gen}$',
}

# trigger name on the axis: list of HLT paths, ORed. Paths missing from a
# sample's trig_* branches are dropped; a trigger with none present is skipped.
TRIGGERS = {
    # analysis trigger (cut2) and the bkg full-tree gate (hltSelectionPaths)
    'METNoMu120_OR':         ['HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
                              'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60'],
    #'PFMETNoMu110_FilterHF': ['HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF'],
    'PFMETNoMu120':          ['HLT_PFMETNoMu120_PFMHTNoMu120_IDTight'],
    'PFMETNoMu120_PFHT60':   ['HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60'],
    #'PFMETNoMu130':          ['HLT_PFMETNoMu130_PFMHTNoMu130_IDTight'],
    #'PFMETNoMu140':          ['HLT_PFMETNoMu140_PFMHTNoMu140_IDTight'],
    #'PFMET120':              ['HLT_PFMET120_PFMHT120_IDTight'],
    #'PFMETTypeOne140':       ['HLT_PFMETTypeOne140_PFMHT140_IDTight'],
    #'PFMET105_IsoTrk50':     ['HLT_PFMET105_IsoTrk50'],
}
# subset used in the 2D (HT, nPV, gen-lepton) efficiency hists
TRIGGERS_MAIN = ['METNoMu120_OR', 'PFMETNoMu120', 'PFMETNoMu120_PFHT60']

# reference (orthogonal) triggers, one 0/1 axis each in trig_eff, named
# 'pass<key>' (passIsoMu27, passEle32). Sum over both for the plain MC
# efficiency; select 1 on one of them for the reference-trigger method used in
# data (filled in MC for closure). A path missing from a sample fills as 0.
REF_TRIGGERS = {
    'IsoMu27': 'HLT_IsoMu27',
    'Ele32':   'HLT_Ele32_WPTight_Gsf',
}

# trigger hists are filled only for ptmiss >= TRIG_MIN_MET (GeV), per
# definition, where the efficiency is ~0 anyway; this skips most events and
# saves most of the fill time. Must be a bin edge of _TURNON_EDGES; the bins
# below it stay empty.
TRIG_MIN_MET = 60

# slim-tree selections ('cut' axis labels); prefixed so they never collide with
# the full tree's cutN labels for the same sample
SLIM_CUTS = ['slim_filters'] #, 'slim_filters_jet']

# full-tree cuts where signal trigger efficiencies are filled: after the ptmiss
# filters (cut1), before the trigger (cut2)
TRIG_EFF_CUTS = ('cut1',)

# full-tree cuts where the non-trigger hists are filled (filters, trigger,
# ptmiss > 200); None = every cut with savePlots=True. Each extra cut adds
# ~1.8 MB per sample to the output.
FULL_FILL_CUTS = ('cut1', 'cut2', 'cut3')

GENLEP_CATS = ['0l', '1e', '1mu', '1tau', '2l+', 'na']

CORR_PAIRS = [('pf', 'puppi'), ('pf', 'calo'), ('pf', 'pfraw'), ('pf', 'pfnomu'),
              ('pfnomu', 'mhtnomu'), ('pf', 'gen'), ('puppi', 'gen')]

# ---- axes ----

_TURNON_EDGES = np.concatenate([np.arange(0, 100, 20), np.arange(100, 300, 10),
                                np.arange(300, 500, 25), np.arange(500, 1001, 100)])
_GENMET_EDGES = [0, 25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 700, 1000]

# Fixed category axes have overflow=False: every fill uses a listed value, and
# the unused overflow bin would multiply the size of the dense histograms
def _ax_metvar(vals=None):
    return StrCategory(vals or MET_VARS, name='metvar', label=r'$p_T^{miss}$ definition', overflow=False)

def _ax_trigger(vals=None):
    return StrCategory(vals or list(TRIGGERS), name='trigger', label='HLT path', overflow=False)

_ax_refs   = lambda: [IntCategory([0, 1], name=f'pass{r}', label=f'Pass {p}', overflow=False)
                      for r, p in REF_TRIGGERS.items()]
_ax_pass   = lambda: IntCategory([0, 1], name='passTrig', label='Pass trigger', overflow=False)
_ax_genlep = lambda: StrCategory(GENLEP_CATS, name='genlep', label='Prompt gen leptons', overflow=False)
_ax_turnon = lambda: Variable(_TURNON_EDGES, name='met', label=r'$p_T^{miss}$ (GeV)')
_ax_met    = lambda: Regular(100, 0, 1000, name='met', label=r'$p_T^{miss}$ (GeV)')
_ax_genmet = lambda: Variable(_GENMET_EDGES, name='genmet', label=r'$p_T^{miss,gen}$ (GeV)')
_ax_ht     = lambda: Variable([0, 30, 60, 100, 150, 200, 300, 500, 1000], name='ht', label=r'$H_T$ (GeV)')
_ax_npv    = lambda: Variable([0, 20, 30, 40, 50, 60, 70, 100], name='npv', label='Number of primary vertices')
_ax_resp   = lambda: Regular(150, 0, 3, name='response', label=r'$p_T^{miss}/p_T^{miss,gen}$')
_ax_diff   = lambda: Regular(160, -200, 200, name='diff', label=r'$p_T^{miss} - p_T^{miss,gen}$ (GeV)')

def make_histograms(info=None):
    W = hist.storage.Weight
    h = {}

    # --- trigger efficiency ---
    h['trig_eff'] = Hist(samp, cut, *_ax_refs(), _ax_trigger(), _ax_metvar(), _ax_turnon(), _ax_pass(),
                         storage=W())
    h['trig_eff_ht'] = Hist(samp, cut, _ax_trigger(TRIGGERS_MAIN), _ax_metvar(['pf', 'trignomu']),
                            _ax_turnon(), _ax_ht(), _ax_pass(), storage=W())
    # nPV (pileup) dependence: disabled for now to save run time
    # h['trig_eff_npv'] = Hist(samp, cut, _ax_trigger(TRIGGERS_MAIN), _ax_metvar(['pf', 'trignomu']),
    #                          _ax_turnon(), _ax_npv(), _ax_pass(), storage=W())
    h['trig_eff_genlep'] = Hist(samp, cut, _ax_genlep(), _ax_trigger(TRIGGERS_MAIN),
                                _ax_metvar(['pf', 'pfnomu', 'trignomu']), _ax_turnon(), _ax_pass(), storage=W())

    # --- distributions / definition comparison ---
    h['met_dist'] = Hist(samp, cut, _ax_genlep(), _ax_metvar(), _ax_met(), storage=W())
    h['genmet_dist'] = Hist(samp, cut, _ax_genlep(),
                            Regular(100, 0, 1000, name='met', label=r'$p_T^{miss,gen}$ (GeV)'), storage=W())
    h['met_corr'] = Hist(samp, cut,
                         StrCategory([f'{a}_{b}' for a, b in CORR_PAIRS], name='pair', label='x_y definitions',
                                     overflow=False),
                         Regular(60, 0, 600, name='x', label=r'$p_T^{miss}$ (GeV)'),
                         Regular(60, 0, 600, name='y', label=r'$p_T^{miss}$ (GeV)'), storage=W())
    h['met_significance'] = Hist(samp, cut, _ax_genlep(),
                                 Regular(80, 1e-2, 1e3, name='sig', label=r'$p_T^{miss}$ significance',
                                         transform=hist.axis.transform.log), storage=W())
    h['met_sig_vs_met'] = Hist(samp, cut, Regular(50, 0, 500, name='met', label=r'$p_T^{miss}$ (GeV)'),
                               Regular(60, 1e-2, 1e3, name='sig', label=r'$p_T^{miss}$ significance',
                                       transform=hist.axis.transform.log), storage=W())
    h['pf_calo_ratio'] = Hist(samp, cut, Regular(100, 0, 5, name='ratio',
                              label=r'$|p_T^{miss,PF} - p_T^{miss,Calo}|/p_T^{miss,Calo}$'), storage=W())

    # --- resolution / response vs GenMET (MC) ---
    h['met_response'] = Hist(samp, cut, _ax_metvar(), _ax_genmet(), _ax_resp(), storage=W())
    h['met_diff'] = Hist(samp, cut, _ax_metvar(), _ax_genmet(), _ax_diff(), storage=W())
    # nPV (pileup) dependence: disabled for now to save run time
    # h['met_diff_npv'] = Hist(samp, cut, _ax_metvar(), _ax_npv(), _ax_diff(), storage=W())
    # vector components w.r.t. the GenMET direction (full trees only: need phi)
    phi_vars = list(MET_DEFS)
    h['met_upar'] = Hist(samp, cut, _ax_metvar(phi_vars), _ax_genmet(),
                         Regular(160, -200, 200, name='upar',
                                 label=r'$p_{T,\parallel}^{miss} - p_T^{miss,gen}$ (GeV)'), storage=W())
    h['met_uperp'] = Hist(samp, cut, _ax_metvar(phi_vars), _ax_genmet(),
                          Regular(160, -200, 200, name='uperp',
                                  label=r'$p_{T,\perp}^{miss}$ (GeV)'), storage=W())
    h['met_dphi_gen'] = Hist(samp, cut, _ax_metvar(phi_vars), _ax_genmet(),
                             Regular(64, 0, np.pi, name='dphi',
                                     label=r'$|\Delta\phi(\vec{p}_T^{miss}, \vec{p}_T^{miss,gen})|$'), storage=W())
    return h

subroutines = []

# ---- helpers ----

def _get(events, coll, fld=None):
    """Flat float64 numpy array of events[coll][fld], sentinels (< -900) -> nan; None if absent."""
    if coll not in events.fields:
        return None
    arr = events[coll]
    if fld is not None:
        if fld not in arr.fields:
            return None
        arr = arr[fld]
    x = ak.to_numpy(arr).astype(np.float64)
    return np.where(x > -900, x, np.nan)

def _trig(events, paths):
    """Bool array: OR of the paths present in events.trig; None if none are present."""
    if 'trig' not in events.fields:
        return None
    present = [p for p in paths if p in events.trig.fields]
    if not present:
        return None
    out = np.zeros(len(events), dtype=bool)
    for p in present:
        out |= ak.to_numpy(events.trig[p]).astype(bool)
    return out

def _met_values(events):
    """{metvar: (pt, phi or None)} for the definitions present in these events."""
    out = {}
    for var, (coll, fpt, fphi) in MET_DEFS.items():
        pt = _get(events, coll, fpt)
        if pt is None:
            continue
        out[var] = (pt, _get(events, coll, fphi))
    if 'pfnomu' in out and 'mhtnomu' in out:
        # np.minimum propagates nan, so trignomu is undefined if either input is
        out['trignomu'] = (np.minimum(out['pfnomu'][0], out['mhtnomu'][0]), None)
    return out

def _genlep(events):
    """(category string array, nMu array or None)."""
    n = len(events)
    if 'GenPromptLep' not in events.fields:
        return np.full(n, 'na', dtype='<U4'), None
    ne = ak.to_numpy(events.GenPromptLep.nEle)
    nm = ak.to_numpy(events.GenPromptLep.nMu)
    nt = ak.to_numpy(events.GenPromptLep.nTau)
    tot = ne + nm + nt
    cat = np.full(n, '2l+', dtype='<U4')
    cat[tot == 0] = '0l'
    cat[(tot == 1) & (ne == 1)] = '1e'
    cat[(tot == 1) & (nm == 1)] = '1mu'
    cat[(tot == 1) & (nt == 1)] = '1tau'
    return cat, nm

def _fill(h, mask, **kw):
    if not np.any(mask):
        return
    h.fill(**{k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in kw.items()})

def _absdphi(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.where(d > np.pi, 2 * np.pi - d, d)

# ---- fillers ----

def _fill_trigger(ev, hists, samp, cut, wgt, mets, genlep):
    # keep events with at least one definition >= TRIG_MIN_MET; each fill below
    # then also requires its own definition >= TRIG_MIN_MET (nan compares False)
    hi = np.zeros(len(ev), dtype=bool)
    for pt, _ in mets.values():
        hi |= pt >= TRIG_MIN_MET
    if not np.any(hi):
        return
    ev, wgt, genlep = ev[hi], wgt[hi], genlep[hi]
    mets = {var: (pt[hi], None) for var, (pt, _) in mets.items()}

    ht = _get(ev, 'PFHT')
    # npv = _get(ev, 'numPV')  # trig_eff_npv disabled
    passes = {t: _trig(ev, paths) for t, paths in TRIGGERS.items()}
    passes = {t: p.astype(int) for t, p in passes.items() if p is not None}

    refs = {}
    for r, path in REF_TRIGGERS.items():
        fired = _trig(ev, [path])
        refs[f'pass{r}'] = np.zeros(len(ev), dtype=int) if fired is None else fired.astype(int)

    for t, p in passes.items():
        for var, (pt, _) in mets.items():
            _fill(hists['trig_eff'], pt >= TRIG_MIN_MET, samp=samp, cut=cut, trigger=t, metvar=var,
                  met=pt, passTrig=p, weight=wgt, **refs)

    for t in TRIGGERS_MAIN:
        if t not in passes:
            continue
        p = passes[t]
        for var in ('pf', 'pfnomu', 'trignomu'):
            if var not in mets:
                continue
            pt = mets[var][0]
            ok = pt >= TRIG_MIN_MET
            _fill(hists['trig_eff_genlep'], ok, samp=samp, cut=cut, genlep=genlep, trigger=t,
                  metvar=var, met=pt, passTrig=p, weight=wgt)
            if var == 'pfnomu':
                continue
            if ht is not None:
                _fill(hists['trig_eff_ht'], ok & np.isfinite(ht), samp=samp, cut=cut, trigger=t,
                      metvar=var, met=pt, ht=ht, passTrig=p, weight=wgt)
            # if npv is not None:
            #     _fill(hists['trig_eff_npv'], ok & np.isfinite(npv), samp=samp, cut=cut, trigger=t,
            #           metvar=var, met=pt, npv=npv, passTrig=p, weight=wgt)

def _fill_ptmiss(ev, hists, samp, cut, wgt, mets, genlep, nmu):
    n = len(ev)
    all_ok = np.ones(n, dtype=bool)

    for var, (pt, _) in mets.items():
        _fill(hists['met_dist'], np.isfinite(pt), samp=samp, cut=cut, genlep=genlep, metvar=var,
              met=pt, weight=wgt)

    gen_pt = _get(ev, 'GenMET', 'pt')
    gen_phi = _get(ev, 'GenMET', 'phi')
    if gen_pt is not None:
        _fill(hists['genmet_dist'], np.isfinite(gen_pt), samp=samp, cut=cut, genlep=genlep,
              met=gen_pt, weight=wgt)

    vals = {v: pt for v, (pt, _) in mets.items()}
    if gen_pt is not None:
        vals['gen'] = gen_pt
    for a, b in CORR_PAIRS:
        if a in vals and b in vals:
            _fill(hists['met_corr'], np.isfinite(vals[a]) & np.isfinite(vals[b]), samp=samp, cut=cut,
                  pair=f'{a}_{b}', x=vals[a], y=vals[b], weight=wgt)

    sig = _get(ev, 'PFMET', 'significance')
    if sig is not None:
        ok = np.isfinite(sig)
        _fill(hists['met_significance'], ok, samp=samp, cut=cut, genlep=genlep, sig=sig, weight=wgt)
        if 'pf' in mets:
            _fill(hists['met_sig_vs_met'], ok & np.isfinite(mets['pf'][0]), samp=samp, cut=cut,
                  met=mets['pf'][0], sig=sig, weight=wgt)

    if 'pf' in mets and 'calo' in mets:
        pf, calo = mets['pf'][0], mets['calo'][0]
        ok = np.isfinite(pf) & np.isfinite(calo) & (calo > 0)
        ratio = np.abs(pf - calo) / np.where(ok, calo, 1.0)
        _fill(hists['pf_calo_ratio'], ok, samp=samp, cut=cut, ratio=ratio, weight=wgt)

    # --- resolution vs GenMET ---
    if gen_pt is None:
        return
    # npv = _get(ev, 'numPV')  # met_diff_npv disabled
    gen_ok = np.isfinite(gen_pt) & (gen_pt > 0)
    nomu_ok = (nmu == 0) if nmu is not None else all_ok
    for var, (pt, phi) in mets.items():
        ok = gen_ok & np.isfinite(pt)
        if var in NOMU_VARS:
            ok &= nomu_ok
        resp = pt / np.where(ok, gen_pt, 1.0)
        diff = pt - gen_pt
        _fill(hists['met_response'], ok, samp=samp, cut=cut, metvar=var, genmet=gen_pt,
              response=resp, weight=wgt)
        _fill(hists['met_diff'], ok, samp=samp, cut=cut, metvar=var, genmet=gen_pt, diff=diff, weight=wgt)
        # if npv is not None:
        #     _fill(hists['met_diff_npv'], ok & np.isfinite(npv), samp=samp, cut=cut, metvar=var, npv=npv,
        #           diff=diff, weight=wgt)
        if phi is None or gen_phi is None:
            continue
        vok = ok & np.isfinite(phi) & np.isfinite(gen_phi)
        dphi = phi - gen_phi
        _fill(hists['met_upar'], vok, samp=samp, cut=cut, metvar=var, genmet=gen_pt,
              upar=pt * np.cos(dphi) - gen_pt, weight=wgt)
        _fill(hists['met_uperp'], vok, samp=samp, cut=cut, metvar=var, genmet=gen_pt,
              uperp=pt * np.sin(dphi), weight=wgt)
        _fill(hists['met_dphi_gen'], vok, samp=samp, cut=cut, metvar=var, genmet=gen_pt,
              dphi=_absdphi(phi, gen_phi), weight=wgt)

def _event_weight(events, sum_wgt):
    if 'eventWgt' in events.fields:
        return ak.to_numpy(events.eventWgt / sum_wgt).astype(np.float64)
    return np.ones(len(events))

def fillHistos(events, hists, samp, cut, info, sum_wgt=1):
    do_trig = info['type'] == 'signal' and cut in TRIG_EFF_CUTS
    do_rest = FULL_FILL_CUTS is None or cut in FULL_FILL_CUTS
    if not (do_trig or do_rest) or len(events) == 0:
        return
    wgt = _event_weight(events, sum_wgt)
    mets = _met_values(events)
    genlep, nmu = _genlep(events)
    if do_trig:
        _fill_trigger(events, hists, samp, cut, wgt, mets, genlep)
    if do_rest:
        _fill_ptmiss(events, hists, samp, cut, wgt, mets, genlep, nmu)

def fillSlimHistos(events, hists, samp, info, sum_wgt=1):
    # samp is the base sample name (no "__slim"); the 'cut' labels keep these
    # separate from the same sample's full-tree fills
    if len(events) == 0:
        return
    wgt_all = _event_weight(events, sum_wgt)
    filt = ak.to_numpy(events.METFiltersFailBits == 0) if 'METFiltersFailBits' in events.fields \
        else np.ones(len(events), dtype=bool)
    jet = _get(events, 'PFJetPtLeading')
    jet = (np.nan_to_num(jet, nan=-1.0) > 30) if jet is not None else np.ones(len(events), dtype=bool)
    masks = {
        'slim_filters':     filt,
        #'slim_filters_jet': filt & jet,
    }
    for cut in SLIM_CUTS:
        m = masks[cut]
        ev = events[m]
        if len(ev) == 0:
            continue
        wgt = wgt_all[m]
        mets = _met_values(ev)
        genlep, nmu = _genlep(ev)
        _fill_trigger(ev, hists, samp, cut, wgt, mets, genlep)
        _fill_ptmiss(ev, hists, samp, cut, wgt, mets, genlep, nmu)
