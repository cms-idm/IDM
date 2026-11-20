import json
import os
import sys

import numpy as np
import matplotlib
# Non-interactive backend: with the default qtagg, every closed figure leaks
# ~10 MB (Qt frees the canvas only from its event loop, which a batch script
# never runs), so a full run gets OOM-killed. We only save PNGs anyway.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)  # python_analysis/
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import analysisTools.plotTools as ptools

# Event maps (phi horizontal, eta vertical) for the events dumped by
# AODSkimmer/scripts/dump_eventmap_objects.py (step 1, run under cmsenv).
# One plot per event, zoomed on the gen ee system: an eta range of 2*DR_WINDOW,
# phi widened so that eta and phi have the same scale (dR circles are round) and
# the frame is filled, with the gen ee at EE_POS in the frame (the centre).
# Objects outside the window are clipped.
# Shown: reco/gen AK4 jets, reco low-pT / gen signal electrons (coloured by
# charge) with lines to their ECAL-entrance points, the low-pT electrons' own
# superclusters (open squares in the electron's colour), the gen ee 4-vector
# sum, ECAL superclusters, and reco (PF, PUPPI) / gen ptmiss. ptmiss has no eta,
# so it is drawn as a vertical line at its phi.
# Plus the near_gen_ee objects (only within the dump's dR of the gen ee):
# track-like objects (charged PF candidates, GSF / lost / lost-electron /
# isolated / displaced tracks) as lines from their momentum direction at the
# PCA (small dot) to their ECAL-entrance point; neutral PF candidates, photons,
# conversions, secondary vertices and EB/EE/ES clusters as markers; ECAL rechits
# as crystal-sized boxes on a semi-transparent viridis (log energy) background,
# without a colorbar.
# One colour per object family (CMS 10-colour palette); linestyle / marker tell
# the collections within a family apart.
# The legend is not on the event maps: it is one standalone image,
# plotdir/hist_eventmap_legend.png.
# phi is unwrapped around the gen ee, so axis values can run past +-pi.
#
# Run from python_analysis/ in the coffea env:  python eventmaps/eventmap_plots.py

ptools.set_plot_style()

infile  = 'eventmaps/outputs/eventmap_objects_merged_events_sigNov2026_anabrv-sel_cut3.json'
plotdir = 'plots/eventmaps'
seltag  = 'anabrv-sel_cut3'   # subdir under plotdir; matches the event-list CSV
max_events = -1              # -1 = every event in infile

DR_WINDOW   = 0.2    # eta range is 2*DR_WINDOW; phi range follows from equal scaling
EE_POS      = (0.5, 0.5)   # gen ee position in the frame (axes fraction from left, from bottom)
TRACKER_ETA = 2.5    # guide lines at the tracker acceptance
JET_R       = 0.4    # AK4 cone radius drawn around each jet
LABEL_FS    = 18     # pT / energy label font size
LEGEND_FS   = 20     # standalone legend font size

# One colour per object family, electrons split by charge (10 colour-coded
# series); reco vs gen is told apart by filled vs open markers / solid vs dashed
# lines. The gen ee sum is black (a "total" of the two gen electrons).
_colors = ptools.cms_colors(10)
COL = {'ele-': _colors[0], 'met': _colors[1], 'ele+': _colors[2], 'sc': _colors[3], 'jet': _colors[4],
       'trk': _colors[5],    # charged PF candidates, lost / isolated / displaced tracks
       'etrk': _colors[6],   # electron tracks: GSF, lost electron tracks
       'neu': _colors[7],    # photons, neutral PF candidates
       'conv': _colors[8],   # conversions
       'sv': _colors[9]}     # secondary vertices
# EB/EE and ES clusters share the ECAL supercluster colour ('sc').

# Track-like near_gen_ee collections: key -> (colour key, linestyle, legend label).
# Drawn as a line from the momentum direction at the PCA to the ECAL entrance.
TRACKS = {
    'pf_cands':         ('trk',  '-',  'Charged PF candidate'),
    'lost_tracks':      ('trk',  '--', 'Lost track'),
    'isolated_tracks':  ('trk',  ':',  'Isolated track'),
    'displaced_tracks': ('trk',  '-.', 'Displaced track'),
    'gsf_tracks':       ('etrk', '-',  'GSF track'),
    'lost_ele_tracks':  ('etrk', '--', 'Lost electron track'),
}
TRK_LW, TRK_MS = 1.5, 5   # track line width, PCA-direction dot size
PF_ALPHA = 0.5            # transparency of all PF candidates (charged lines and neutral markers)
PF_PHOTON_LABEL_E = 1.    # PF photons above this energy (GeV) get an energy label

# Point-like near_gen_ee collections: key -> (colour key, marker, filled, size, legend label).
MARKERS = {
    'photons':                ('neu',  'h', True,  13, 'Photon'),
    'conversions':            ('conv', 'D', True,  10, 'Conversion'),
    'single_leg_conversions': ('conv', 'D', False, 10, 'Single-leg conversion'),
    'open_conversions':       ('conv', '>', True,  11, 'Open conversion (GSF pair)'),
    'secondary_vertices':     ('sv',   'P', True,  14, 'Secondary vertex (IVF)'),
    'ebee_clusters':          ('sc',   'p', False, 12, 'ECAL EB/EE cluster'),
    'es_clusters':            ('sc',   '1', False, 16, 'ES cluster'),
}
# Neutral PF candidates (no ECAL line): photon vs everything else.
PF_NEUTRAL = {
    'pf_photon': ('neu', '^', True, 10, 'PF photon'),
    'pf_nhad':   ('neu', 'v', True, 10, 'PF neutral hadron'),
}

# ECAL rechit boxes: crystal size in (eta, phi). EB crystals are 0.0174 x 0.0174;
# EE crystals are 2.862 cm wide in x and y at the front face (|z| = 315.4 cm),
# converted to eta/phi at each hit's eta.
EB_CRYSTAL     = 0.0174
EE_CRYSTAL_CM  = 2.862
EE_FRONT_Z_CM  = 315.4
RECHIT_CMAP    = 'viridis'
RECHIT_ALPHA   = 0.5
MET_STYLES = {           # key in objects['met'] -> (linestyle, legend label)
    'pf':    ('-',  r'Reco PF $p_T^\mathrm{miss}$'),
    'puppi': ('-.', r'Reco PUPPI $p_T^\mathrm{miss}$'),
    'gen':   ('--', r'Gen $p_T^\mathrm{miss}$'),
}
ELE_MS = {'reco': 12, 'gen': 22}   # momentum-direction marker sizes
ECAL_MARKER = 'X'                  # marker at the ECAL-entrance end of the line


def _fmt_pt(pt):
    return f'{pt:.3g}'


def _wrap(dphi):
    """dphi wrapped into [-pi, pi)."""
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def _ele_charge(e, gen):
    # gen: pdgId 11 = e- (charge -1), -11 = e+
    return -np.sign(e['pdgId']) if gen else e['charge']


def _ele_col(e, gen):
    return COL['ele+'] if _ele_charge(e, gen) > 0 else COL['ele-']


class _Window:
    """Plot coordinates around the gen ee: x = phi unwrapped around phi0, y = eta."""

    def __init__(self, eta0, phi0):
        self.eta0, self.phi0 = eta0, phi0

    def x(self, phi):
        return self.phi0 + _wrap(phi - self.phi0)

    def contains(self, ax, x, y):
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        return x0 <= x <= x1 and y0 <= y <= y1


def _draw_jets(ax, win, jets, gen):
    for j in jets:
        if gen:
            c = Circle((win.x(j['phi']), j['eta']), JET_R, fill=False, edgecolor=COL['jet'],
                       linestyle='--', linewidth=2, zorder=2)
        else:
            c = Circle((win.x(j['phi']), j['eta']), JET_R, facecolor=COL['jet'], edgecolor=COL['jet'],
                       alpha=0.35, linewidth=2, zorder=1)
        ax.add_patch(c)


def _draw_eles(ax, win, eles, gen):
    """Electron at its momentum direction, plus a line to its ECAL-entrance point
    (when the propagation reached the ECAL)."""
    for e in eles:
        col = _ele_col(e, gen)
        x, y = win.x(e['phi']), e['eta']
        ecal = e.get('ecal')
        if ecal is not None:
            xe = x + _wrap(ecal['phi'] - e['phi'])
            ax.plot([x, xe], [y, ecal['eta']], color=col, linestyle='--' if gen else '-',
                    linewidth=2, zorder=3)
            ax.plot(xe, ecal['eta'], linestyle='none', marker=ECAL_MARKER, markersize=12,
                    markerfacecolor='none' if gen else col, markeredgecolor=col,
                    markeredgewidth=1.5, zorder=4)
        if gen:
            ax.plot(x, y, linestyle='none', marker='o', markersize=ELE_MS['gen'], markerfacecolor='none',
                    markeredgecolor=col, markeredgewidth=2.5, zorder=4)
        else:
            ax.plot(x, y, linestyle='none', marker='o', markersize=ELE_MS['reco'], color=col, zorder=5)
            sc = e.get('sc')
            if sc is not None:   # the electron's own supercluster
                ax.plot(win.x(sc['phi']), sc['eta'], linestyle='none', marker='s', markersize=16,
                        markerfacecolor='none', markeredgecolor=col, markeredgewidth=2.5, zorder=3)


def _draw_tracks(ax, win, trks, col, ls, alpha=1.):
    """Track-like objects: a dot at the momentum direction at the PCA and a line
    from there to the ECAL-entrance point (just the dot if the propagation
    didn't reach the ECAL)."""
    for t in trks:
        x, y = win.x(t['phi']), t['eta']
        ecal = t.get('ecal')
        if ecal is not None:
            xe = x + _wrap(ecal['phi'] - t['phi'])
            ax.plot([x, xe], [y, ecal['eta']], color=col, linestyle=ls, linewidth=TRK_LW, alpha=alpha,
                    zorder=2.5)
        ax.plot(x, y, linestyle='none', marker='o', markersize=TRK_MS, color=col, alpha=alpha, zorder=2.5)


def _draw_markers(ax, win, objs, col, marker, filled, ms, alpha=1.):
    if not objs:
        return
    ax.plot([win.x(o['phi']) for o in objs], [o['eta'] for o in objs], linestyle='none', marker=marker,
            markersize=ms, markerfacecolor=col if filled else 'none', markeredgecolor=col,
            markeredgewidth=2, alpha=alpha, zorder=3)


def _crystal_size(hit):
    """(deta, dphi) of the crystal a rechit sits in."""
    if hit['det'] == 'EB':
        return EB_CRYSTAL, EB_CRYSTAL
    z, s = EE_FRONT_Z_CM, EE_CRYSTAL_CM
    r = z / np.sinh(abs(hit['eta']))
    # deta = s * |d eta / d r| at fixed z; dphi = s / r
    return s * z / (r * np.hypot(r, z)), s / r


def _draw_rechits(ax, win, hits):
    """ECAL rechits as crystal-sized boxes, semi-transparent viridis on a log
    energy scale (per event; no colorbar). Hits with E <= 0 are skipped."""
    hits = [h for h in hits if h['energy'] > 0]
    if not hits:
        return
    rects = []
    for h in hits:
        deta, dphi = _crystal_size(h)
        rects.append(Rectangle((win.x(h['phi']) - dphi / 2, h['eta'] - deta / 2), dphi, deta))
    e = np.array([h['energy'] for h in hits])
    pc = PatchCollection(rects, cmap=RECHIT_CMAP, norm=LogNorm(e.min(), max(e.max(), 1.01 * e.min())),
                         alpha=RECHIT_ALPHA, edgecolor='none', zorder=0.5)
    pc.set_array(e)
    ax.add_collection(pc)


def _draw_near_gen_ee(ax, win, near):
    """The near_gen_ee objects (empty/None for dumps made before they existed)."""
    if not near:
        return
    _draw_rechits(ax, win, near.get('ecal_rechits', []))
    for key, (ckey, ls, _) in TRACKS.items():
        trks = near.get(key, [])
        if key == 'pf_cands':
            trks = [c for c in trks if c['charge'] != 0]
        _draw_tracks(ax, win, trks, COL[ckey], ls, alpha=PF_ALPHA if key == 'pf_cands' else 1.)
    neutral = [c for c in near.get('pf_cands', []) if c['charge'] == 0]
    for kind, objs in (('pf_photon', [c for c in neutral if c['pdgId'] == 22]),
                       ('pf_nhad', [c for c in neutral if c['pdgId'] != 22])):
        ckey, marker, filled, ms, _ = PF_NEUTRAL[kind]
        _draw_markers(ax, win, objs, COL[ckey], marker, filled, ms, alpha=PF_ALPHA)
    for key, (ckey, marker, filled, ms, _) in MARKERS.items():
        _draw_markers(ax, win, near.get(key, []), COL[ckey], marker, filled, ms)


def _draw_gen_ee(ax, win, ee):
    if ee is not None:
        ax.plot(win.x(ee['phi']), ee['eta'], linestyle='none', marker='*', markersize=24,
                color='black', zorder=6)


def _draw_scs(ax, win, scs):
    for s in scs:
        ax.plot(win.x(s['phi']), s['eta'], linestyle='none', marker='s', markersize=16,
                markerfacecolor='none', markeredgecolor=COL['sc'], markeredgewidth=2.5, zorder=3)


def _draw_met(ax, win, met):
    """Vertical ptmiss lines."""
    for key, (ls, _) in MET_STYLES.items():
        m = met.get(key)
        if m is not None:
            x = win.x(m['phi'])
            ax.plot([x, x], [-10, 10], color=COL['met'], linestyle=ls, linewidth=2.5, zorder=3)


def _place_labels(ax, labels):
    """Annotate each (x, y, text, ha, gen, color) with its text, in its object's colour (italic for gen
    objects), nudging labels up/down in
    steps of one text height until they neither overlap a label already placed nor leave the frame.
    Displaced labels get a thin leader line back to their object."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_to_pt = 72. / fig.dpi
    frame = ax.get_window_extent(renderer)
    leg = ax.get_legend()
    placed = [leg.get_window_extent(renderer)] if leg is not None else []
    kw = dict(textcoords='offset points', va='bottom', fontsize=LABEL_FS, zorder=7)
    for x, y, text, ha, gen, color in labels:
        dx = 8 if ha == 'left' else -8
        kw['fontstyle'] = 'italic' if gen else 'normal'
        kw['color'] = color
        ann = ax.annotate(text, (x, y), xytext=(dx, 6), ha=ha, **kw)
        step = ann.get_window_extent(renderer).height * px_to_pt + 2
        for k in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, 7, 8]:   # 0, up 1, down 1, up 2, ...
            ann.xyann = (dx, 6 + k * step)
            bb = ann.get_window_extent(renderer)
            if (frame.y0 <= bb.y0 and bb.y1 <= frame.y1
                    and not any(bb.padded(2).overlaps(p) for p in placed)):
                break
        else:
            k = 0   # nowhere clear: keep the default spot
            ann.xyann = (dx, 6)
        if k != 0:
            ann.remove()
            ann = ax.annotate(text, (x, y), xytext=(dx, 6 + k * step), ha=ha,
                              arrowprops={'arrowstyle': '-', 'color': color, 'linewidth': 0.8}, **kw)
        placed.append(ann.get_window_extent(renderer).padded(2))


def _legend_handles():
    h = [
        Patch(facecolor=COL['jet'], edgecolor=COL['jet'], alpha=0.35, label='Reco AK4 jet'),
        Patch(fill=False, edgecolor=COL['jet'], linestyle='--', linewidth=2, label='Gen jet'),
    ]
    for q, sym in (('-', '$e^-$'), ('+', '$e^+$')):
        col = COL['ele' + q]
        h += [Line2D([], [], linestyle='none', marker='o', markersize=ELE_MS['reco'], color=col,
                     label=f'Reco low-$p_T$ {sym}'),
              Line2D([], [], linestyle='none', marker='o', markersize=18, markerfacecolor='none',
                     markeredgecolor=col, markeredgewidth=2.5, label=rf'Gen {sym} ($\chi_2$ decay)')]
    h += [
        Line2D([], [], linestyle='none', marker='*', markersize=20, color='black', label=r'Gen $e^+e^-$ sum'),
        Line2D([], [], color='black', linestyle='-', linewidth=2, marker=ECAL_MARKER, markersize=12,
               markevery=[1], label=r'Reco $\to$ ECAL entrance'),
        Line2D([], [], color='black', linestyle='--', linewidth=2, marker=ECAL_MARKER, markersize=12,
               markerfacecolor='none', markevery=[1], label=r'Gen $\to$ ECAL entrance'),
        Line2D([], [], linestyle='none', marker='s', markersize=14, markerfacecolor='none',
               markeredgecolor='black', markeredgewidth=2.5, label=r'Low-$p_T$ $e$ SC (colour of its $e$)'),
        Line2D([], [], linestyle='none', marker='s', markersize=14, markerfacecolor='none',
               markeredgecolor=COL['sc'], markeredgewidth=2.5, label='ECAL supercluster'),
    ]
    h += [Line2D([], [], color=COL['met'], linestyle=ls, linewidth=2.5, label=lab)
          for ls, lab in MET_STYLES.values()]
    # near_gen_ee objects
    h += [Line2D([], [], color=COL[ckey], linestyle=ls, linewidth=TRK_LW, marker='o', markersize=TRK_MS,
                 markevery=[0], alpha=PF_ALPHA if key == 'pf_cands' else 1.,
                 label=rf'{lab} (PCA $\to$ ECAL)')
          for key, (ckey, ls, lab) in TRACKS.items()]
    h += [Line2D([], [], linestyle='none', marker=marker, markersize=ms,
                 markerfacecolor=COL[ckey] if filled else 'none', markeredgecolor=COL[ckey],
                 markeredgewidth=2, alpha=PF_ALPHA if key in PF_NEUTRAL else 1., label=lab)
          for key, (ckey, marker, filled, ms, lab) in list(PF_NEUTRAL.items()) + list(MARKERS.items())]
    h += [Patch(facecolor=plt.get_cmap(RECHIT_CMAP)(0.6), alpha=RECHIT_ALPHA, edgecolor='none',
                label=r'ECAL rechit (colour: log $E$)')]
    return h


def _title(rec):
    first = f"Event Map: {rec['run']}:{rec['lumi']}:{rec['event']}"
    second = (rf"$m_1$={rec['m1']:g} GeV, $\delta$={ptools.fmt_delta(rec['delta'])}, "
              rf"c$\tau$={rec['ctau']:g} mm")
    if rec.get('gen_lxy') is not None:
        second += rf", $L_{{xy}}^\mathrm{{gen}}$={rec['gen_lxy']:.3g} cm"
    return f'{first}\n({second})'


def _equal_scale(ax, win):
    """Set the phi range so one unit of phi is as long on screen as one unit of
    eta (dR circles are round), with the gen ee at EE_POS[0] of the width.
    Keeps the eta range."""
    ax.figure.canvas.draw()
    bb = ax.get_window_extent()
    lo, hi = ax.get_ylim()
    span = (hi - lo) * bb.width / bb.height
    fx = EE_POS[0]
    ax.set_xlim(win.phi0 - fx * span, win.phi0 + (1 - fx) * span)


def _finish(ax, win, rec):
    # Gen ee at EE_POS; no legend on the map (see save_legend).
    fy = EE_POS[1]
    ax.set_ylim(win.eta0 - 2 * fy * DR_WINDOW, win.eta0 + 2 * (1 - fy) * DR_WINDOW)
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\eta$')
    ax.set_title(_title(rec))
    plt.tight_layout()
    _equal_scale(ax, win)   # last: nothing may change the axes box after this


def save_legend(dr):
    """The event-map legend as its own image, plotdir/hist_eventmap_legend.png.
    dr: the dump's dR window for the near_gen_ee objects (None if unknown)."""
    title = (r'Labels: $p_T$ (GeV), SC: $E$ (GeV), '
             rf'PF photon: $E$ (GeV, if > {PF_PHOTON_LABEL_E:g}); gen in italics')
    if dr is not None:
        title += '\n' + rf'Tracks, PF cands, photons, conversions, SVs, clusters, rechits: $\Delta R$ < {dr:g} of gen $e^+e^-$'
    fig = plt.figure(figsize=ptools.FIGSIZE)
    fig.legend(handles=_legend_handles(), loc='center', ncol=2, fontsize=LEGEND_FS,
               title=title, title_fontsize=LEGEND_FS)
    os.makedirs(plotdir, exist_ok=True)
    fname = os.path.join(plotdir, 'hist_eventmap_legend.png')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    return fname


def _pstr(v):
    """0.05 -> '0p05', 1.0 -> '1'."""
    return f'{v:g}'.replace('.', 'p')


def _sample_dir(rec):
    """e.g. sig_2024_M1-0p5_delta-0p05_ctau-1 (named by m1/delta/ctau, not Mchi/dMchi)."""
    return f"sig_{rec['year']}_M1-{_pstr(rec['m1'])}_delta-{_pstr(rec['delta'])}_ctau-{_pstr(rec['ctau'])}"


def plot_event(rec):
    obj = rec['objects']
    ee = obj['gen_ee']
    if ee is None:
        print(f"  skipping {rec['run']}:{rec['lumi']}:{rec['event']}: no gen ee (need exactly 2 gen electrons)")
        return None
    win = _Window(ee['eta'], ee['phi'])
    fig, ax = ptools.new_fig(rec.get('year'))
    for y in (-TRACKER_ETA, TRACKER_ETA):
        ax.axhline(y, color='black', linestyle=':', linewidth=1.5, zorder=0)

    _draw_jets(ax, win, obj['reco_jets'], gen=False)
    _draw_jets(ax, win, obj['gen_jets'], gen=True)
    _draw_near_gen_ee(ax, win, obj.get('near_gen_ee'))
    _draw_scs(ax, win, obj['ecal_scs'])
    _draw_eles(ax, win, obj['gen_eles'], gen=True)
    _draw_eles(ax, win, obj['reco_lpt_eles'], gen=False)
    _draw_gen_ee(ax, win, ee)
    _draw_met(ax, win, obj['met'])

    # Axis limits must be final before the labels are placed in display space.
    _finish(ax, win, rec)
    # Each label is drawn in the colour of its object.
    labels = [(win.x(o['phi']), o['eta'], _fmt_pt(o['pt']), 'left', key.startswith('gen'), COL['jet'])
              for key in ('reco_jets', 'gen_jets') for o in obj[key]]
    labels += [(win.x(o['phi']), o['eta'], _fmt_pt(o['pt']), 'left', key == 'gen_eles',
                _ele_col(o, key == 'gen_eles'))
               for key in ('reco_lpt_eles', 'gen_eles') for o in obj[key]]
    labels += [(win.x(s['phi']), s['eta'], _fmt_pt(s['energy']), 'right', False, COL['sc'])
               for s in obj['ecal_scs']]
    labels += [(win.x(e['sc']['phi']), e['sc']['eta'], _fmt_pt(e['sc']['energy']), 'right', False,
                _ele_col(e, False))
               for e in obj['reco_lpt_eles'] if (e.get('sc') or {}).get('energy') is not None]
    labels += [(win.x(c['phi']), c['eta'], _fmt_pt(c['energy']), 'left', False, COL['neu'])
               for c in (obj.get('near_gen_ee') or {}).get('pf_cands', [])
               if c['charge'] == 0 and c['pdgId'] == 22 and c['energy'] > PF_PHOTON_LABEL_E]
    labels = [l for l in labels if win.contains(ax, l[0], l[1])]
    ybot = ax.get_ylim()[0]
    labels += [(win.x(m['phi']), ybot, _fmt_pt(m['pt']), 'left', k == 'gen', COL['met'])
               for k, m in ((k, obj['met'].get(k)) for k in MET_STYLES)
               if m is not None and win.contains(ax, win.x(m['phi']), ybot)]
    _place_labels(ax, labels)

    outdir = os.path.join(plotdir, seltag, _sample_dir(rec))
    os.makedirs(outdir, exist_ok=True)
    fname = f"{outdir}/hist_eventmap_{rec['run']}-{rec['lumi']}-{rec['event']}.png"
    plt.savefig(fname)
    plt.close(fig)
    return fname


with open(infile) as f:
    dump = json.load(f)
records = dump['events']
if max_events >= 0:
    records = records[:max_events]
print(f'Legend: {save_legend(dump["meta"].get("dr"))}')
print(f'Plotting {len(records)} events from {infile}')
for i, rec in enumerate(records):
    fname = plot_event(rec)
    if fname is not None:
        print(f'  [{i + 1}/{len(records)}] {fname}')
print(f'Plots written to {plotdir}/{seltag}/')
