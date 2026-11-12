"""Muon-channel coffea processors for the shared Analyzer (procType='mumu').

MuMuPairRecoveryProcessor reproduces the reco-only dimuon pair recovery study from the ntuple alone: the
mumuvtx pair candidates, the RecoDisplacedMuon fixed components and the mumusig truth labels written by
AODSkimmer/plugins/MuMuVtx.h (port of the recoOnly PreselectedVertexing producer). Selection and counters are
those of the pilot python_analysis/studies/MuonPairRecovery/reco_pair_recovery_coffea.py (Alaa_Dev), which
reproduced AODSkimmer/displaced_muon_id/reco_pair_recovery.py exactly: cumulative stages opposite sign ->
status 3 -> finite chi2, ndof with chi2 >= 0, ndof > 0, chi2/ndof < 4 -> PCA OK -> finite 0 <= PCA dR < 0.9 ->
valid angle and |dphi(Type-1 PF MET, PCA mumu)| < 0.5 (math.remainder wrapping). Correct pair = mumusigTwoAcc and
both legs carry a truth label. Raw, unweighted; no event weights are applied.

Only coffea.processor, awkward and numpy are used, so it runs under coffea 0.7 and the calendar-versioned coffea.
The output is a plain dict keyed by (dataset, filename, entrystart), so chunks never merge by addition; the
caller sums the integer counters and concatenates the per-event and per-pair arrays.
"""
import math

import awkward as ak
import numpy as np
from coffea import processor

STAGES = ('eligible_pair', 'opposite_sign', 'valid_vertex', 'chi2', 'pca_valid', 'dr', 'met_angle')
RECORD_FIELDS = ('run', 'lumi', 'event', 'T', 'nPassing', 'nCorrect', 'disjoint')
PAIR_RECORD_FIELDS = ('run', 'lumi', 'event', 'rawA', 'rawB', 'status', 'chi2', 'ndof')
TWO_PI = 2 * math.pi


def _np(array, dtype=None):
    out = ak.to_numpy(array)
    return out.astype(dtype) if dtype is not None else out


def wrapped_abs_dphi(a, b):
    """abs(math.remainder(a - b, 2*pi)) for |a - b| < 3*pi (one shift picks the nearest multiple exactly)."""
    d = a - b
    if np.any(np.abs(d) >= 3 * math.pi):
        raise RuntimeError('phi difference >= 3*pi; single-shift wrapping would differ from math.remainder')
    d = np.where(d > math.pi, d - TWO_PI, np.where(d < -math.pi, d + TWO_PI, d))
    return np.abs(d)


def pair_stage(q1, q2, status, chi2, ndof, pca_ok, dr, mumu_pt, mumu_phi, met_pt, met_phi):
    """Last passed cumulative stage per pair (0..6). Float inputs are promoted to float64 first."""
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2_ok = (np.isfinite(chi2) & np.isfinite(ndof) & (chi2 >= 0) & (ndof > 0)
                   & (np.where(ndof > 0, chi2 / ndof, np.inf) < 4))
    angle_valid = (np.isfinite(met_pt) & (met_pt > 0) & np.isfinite(met_phi)
                   & np.isfinite(mumu_pt) & (mumu_pt > 0) & np.isfinite(mumu_phi))
    dphi = np.full(len(chi2), np.inf)
    dphi[angle_valid] = wrapped_abs_dphi(met_phi[angle_valid], mumu_phi[angle_valid])
    cuts = [q1.astype(np.int64) * q2 < 0,
            status == 3,
            chi2_ok,
            pca_ok.astype(bool),
            np.isfinite(dr) & (dr >= 0) & (dr < .9),
            angle_valid & (dphi < .5)]
    passed = np.ones(len(chi2), dtype=bool)
    stage = np.zeros(len(chi2), dtype=np.int64)
    for cut in cuts:
        passed = passed & cut
        stage += passed
    return stage


def _check_pairs(n_obj, fixed, i1, i2, counts):
    """Per event: complete enumeration of eligible objects, legs in range, distinct, i1 < i2, no repeats."""
    n_elig = _np(ak.sum(fixed != -1, axis=1), np.int64)
    if not np.array_equal(counts, n_elig * (n_elig - 1) // 2):
        raise RuntimeError('mumuvtx: incomplete pair enumeration')
    n_obj_flat = np.repeat(n_obj, counts)
    a, b = _np(ak.flatten(i1), np.int64), _np(ak.flatten(i2), np.int64)
    if np.any(a < 0) or np.any(b < 0) or np.any(a >= n_obj_flat) or np.any(b >= n_obj_flat):
        raise RuntimeError('mumuvtx: leg index out of range')
    if np.any(a >= b):
        raise RuntimeError('mumuvtx: legs not in producer order i < j')
    ev = np.repeat(np.arange(len(counts)), counts)
    keys = np.stack([ev, a, b], axis=1)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise RuntimeError('mumuvtx: repeated pair')
    fixed_flat_a = _np(ak.flatten(fixed[i1]), np.int64)
    fixed_flat_b = _np(ak.flatten(fixed[i2]), np.int64)
    if np.any(fixed_flat_a == -1) or np.any(fixed_flat_b == -1):
        raise RuntimeError('mumuvtx: leg without a fixed component')


class MuMuPairRecoveryProcessor(processor.ProcessorABC):
    """Reco-pair-recovery counters, per-event records and per-pair identities from the ntuple."""

    def __init__(self, samples=None, sampleInfo=None):
        self.samples = samples
        self.sampleInfo = sampleInfo

    def process(self, events):
        meta = events.metadata
        key = (meta['dataset'], meta['filename'], int(meta.get('entrystart', 0)))
        n = len(events)
        if n and not ak.all(events.mumuvtxAvailable == 1):
            raise RuntimeError('mumuvtx not available (no cleaned displacedMuons) in %s' % meta['filename'])

        mu = events.RecoDisplacedMuon
        fixed = mu.fixedComp
        label = mu.mumusigIdx
        n_obj = _np(ak.num(fixed, axis=1), np.int64)
        if not np.array_equal(n_obj, _np(ak.num(mu.originalIndex, axis=1), np.int64)):
            raise RuntimeError('RecoDisplacedMuon field lengths differ')
        vtx = events.mumuvtx
        counts = _np(ak.num(vtx.status, axis=1), np.int64)  # MySchema turns nmumuvtx into these offsets
        i1, i2 = vtx.m1_idx, vtx.m2_idx
        _check_pairs(n_obj, fixed, i1, i2, counts)

        t = _np(events.mumusigTwoAcc, bool)
        met_pt = _np(events.PFMET.pt, np.float64)
        met_phi = _np(events.PFMET.phi, np.float64)
        p_event = np.repeat(np.arange(n), counts)
        lab1 = _np(ak.flatten(label[i1]), np.int64)
        lab2 = _np(ak.flatten(label[i2]), np.int64)
        flat = {k: _np(ak.flatten(vtx[k])) for k in ('m1_charge', 'm2_charge', 'status', 'chi2', 'ndof', 'pcaOK',
                                                     'pcaDR', 'pcaMuMuPt', 'pcaMuMuPhi')}
        stage_flat = pair_stage(flat['m1_charge'], flat['m2_charge'], flat['status'].astype(np.int64),
                                flat['chi2'].astype(np.float64), flat['ndof'].astype(np.float64), flat['pcaOK'],
                                flat['pcaDR'].astype(np.float64), flat['pcaMuMuPt'].astype(np.float64),
                                flat['pcaMuMuPhi'].astype(np.float64), met_pt[p_event], met_phi[p_event])
        # Pair labels as PreselectedVertexing: pIsBB = both legs are the two labelled objects (each label is on at
        # most one object, so both legs labelled means exactly those two); pMatchedLegs = labelled legs.
        is_bb_flat = (lab1 >= 0) & (lab2 >= 0)
        legs_flat = (lab1 >= 0).astype(np.int64) + (lab2 >= 0)
        correct_flat = t[p_event] & is_bb_flat
        passing_flat = stage_flat == 6

        out = {}
        out['events'] = n
        out['truth_accepted'] = int(t.sum())
        sig_q = ak.fill_none(ak.pad_none(events.mumusig.charge, 2, clip=True), 0)
        n_sig = _np(ak.num(events.mumusig.charge, axis=1), np.int64)  # nmumusig -> offsets in MySchema
        q_prod = _np(sig_q[:, 0], np.int64) * _np(sig_q[:, 1], np.int64)
        has_signal_pair = (n_sig == 2) & (q_prod < 0)
        out['has_signal_pair'] = int(has_signal_pair.sum())
        for i, s in enumerate(STAGES):
            out['candidates_' + s] = int((stage_flat >= i).sum())
            out['correct_candidates_' + s] = int(((stage_flat >= i) & correct_flat).sum())
        t_flat = t[p_event]
        out['passing_no_qualifying_truth_pair'] = int((passing_flat & ~t_flat).sum())
        out['passing_correct'] = int((passing_flat & correct_flat).sum())
        out['passing_partial'] = int((passing_flat & t_flat & ~correct_flat & (legs_flat == 1)).sum())
        out['passing_wrong'] = int((passing_flat & t_flat & ~correct_flat & (legs_flat != 1)).sum())

        stage = ak.unflatten(stage_flat, counts)
        correct = ak.unflatten(correct_flat, counts)
        passing = ak.unflatten(passing_flat, counts)
        any_stage = _np(ak.fill_none(ak.max(stage, axis=1), -1), np.int64)
        best_correct = _np(ak.fill_none(ak.max(ak.where(correct, stage, -1), axis=1), -1), np.int64)
        for i, s in enumerate(STAGES):
            out['events_' + s] = int((any_stage >= i).sum())
            out['truth_recovered_' + s] = int((best_correct >= i).sum())
        n_pass = _np(ak.sum(passing, axis=1), np.int64)
        n_corr = _np(ak.sum(passing & correct, axis=1), np.int64)
        obj = ak.zip({'a': i1, 'b': i2})
        combos = ak.combinations(obj[passing], 2, fields=['x', 'y'])
        disjoint_combo = ((combos.x.a != combos.y.a) & (combos.x.a != combos.y.b)
                          & (combos.x.b != combos.y.a) & (combos.x.b != combos.y.b))
        disjoint = _np(ak.fill_none(ak.any(disjoint_combo, axis=1), False), bool)
        out['passing_pairs'] = int(n_pass.sum())
        out['correct_pairs'] = int(n_corr.sum())
        sel = n_pass > 0
        rec = n_corr > 0
        out['selected_events'] = int(sel.sum())
        out['selected_truth_events'] = int((t & sel).sum())
        out['selected_has_signal_pair'] = int((sel & has_signal_pair).sum())
        out['recovered_events'] = int(rec.sum())
        out['selected_without_T'] = int((sel & ~t).sum())
        out['zero_passing'] = int((n_pass == 0).sum())
        out['one_passing'] = int((n_pass == 1).sum())
        out['multiple_passing'] = int((n_pass > 1).sum())
        out['multiple_with_disjoint_pair_pair'] = int(disjoint.sum())
        out['multiple_all_overlapping'] = int(((n_pass > 1) & ~disjoint).sum())
        out['correct_only'] = int((rec & (n_pass == n_corr)).sum())
        out['correct_plus_incorrect'] = int((rec & (n_pass > n_corr)).sum())
        out['T_incorrect_only'] = int((t & sel & ~rec).sum())
        out['T_no_passing'] = int((t & ~sel).sum())
        lost = t & (best_correct < 6)
        out['loss_reconstruction_ID_eta_propagation_matching'] = int((lost & (best_correct < 0)).sum())
        for i in range(6):
            out['loss_' + STAGES[i + 1]] = int((lost & (best_correct == i)).sum())
        if not np.array_equal(rec.astype(int) + (t & sel & ~rec) + (t & ~sel), t.astype(int)):
            raise RuntimeError('T partition fails')
        if out['passing_correct'] + out['passing_partial'] + out['passing_wrong'] + \
                out['passing_no_qualifying_truth_pair'] != out['passing_pairs']:
            raise RuntimeError('Candidate partition fails')

        run = _np(events.runNum, np.int64)
        lumi = _np(events.lumiSec, np.int64)
        evt = _np(events.eventNum, np.int64)
        records = {'run': run, 'lumi': lumi, 'event': evt, 'T': t.astype(np.int64), 'nPassing': n_pass,
                   'nCorrect': n_corr, 'disjoint': disjoint.astype(np.int64)}
        # Pair identities in the producer's raw displacedMuons indices, for the pair-identity check.
        raw = mu.originalIndex
        pairs = {'run': run[p_event], 'lumi': lumi[p_event], 'event': evt[p_event],
                 'rawA': _np(ak.flatten(raw[i1]), np.int64), 'rawB': _np(ak.flatten(raw[i2]), np.int64),
                 'status': flat['status'].astype(np.int64), 'chi2': flat['chi2'], 'ndof': flat['ndof']}
        return {key: {'counts': out, 'records': records, 'pairs': pairs}}

    def postprocess(self, accumulator):
        return accumulator
