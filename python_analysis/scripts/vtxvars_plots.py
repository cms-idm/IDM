# imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import awkward as ak
import mplhep as hep

import sys
from analysisTools.analysisTools import Analyzer
from analysisTools.analysisTools import loadSchema
import analysisTools.analysisTools as tools
import analysisTools.analysisSubroutines as routines
import importlib
import coffea.util as util
import analysisTools.utils as utils
import analysisTools.plotTools as ptools
import os

outdir = 'workarea'
vers = 'May2026'
selection = 'anmatchvtx'
hists_tag = 'vtxvars-match'
saved_signal_hists = f"{outdir}/hists_sig{vers}_{selection}-sel_{hists_tag}.coffea"
plottag = f'sig{vers}_{selection}-sel-best'

s_hists = util.load(saved_signal_hists)[0]
s_pts   = utils.get_signal_point_dict(s_hists)

size = (16, 12)
cut  = 'cut9'

style_dict_base = {
    'rebin': 1j, 'xlim': None,
    'doLogy': False, 'doLogx': False, 'doDensity': True, 'doYerr': False,
    'xlabel': None, 'ylabel': None, 'label': None, 'flow': None,
    'doSave': False,
    'doCMSLabel': False,
}

# Mass point series matching cutflow_plots.py
series_list = [
    {'m1s': [0.05, 0.5, 5, 50], 'deltas': [0.1],           'ctaus': [10],        'tag': f'ctau-10_delta-0p1_m1-wide'},
    {'m1s': [0.5, 1, 2, 5],     'deltas': [0.1],           'ctaus': [10],        'tag': f'ctau-10_delta-0p1_m1-narrow'},
    {'m1s': [0.5],              'deltas': [0.1],           'ctaus': [1, 10, 100],'tag': f'm1-0p5_delta-0p1'},
    {'m1s': [0.5],              'deltas': [0.05, 0.1, 0.2],'ctaus': [10],        'tag': f'm1-0p5_ctau-10'},
    {'m1s': [5],                'deltas': [0.1],           'ctaus': [1, 10, 100],'tag': f'm1-5_delta-0p1'},
    {'m1s': [5],                'deltas': [0.05, 0.1, 0.2],'ctaus': [10],        'tag': f'm1-5_ctau-10'},
]

# Plot specifications:
#   (title, outname, variables, xlim, doLogy)
# variables with 2 elements => 2 lines per signal point (solid/dotted), legend labels solid_label/dotted_label
# variables with 1 element  => single line per signal point
plot_specs = [
    # per-electron leading/subleading
    (r'Vtx Electron $p_T$',               'vtx_ele_pt',          ['vtx_leading_ele_pt',    'vtx_subleading_ele_pt'],    False, 'Leading',   'Subleading'       ),
    (r'Vtx Electron $\eta$',              'vtx_ele_eta',         ['vtx_leading_ele_eta',   'vtx_subleading_ele_eta'],   False, 'Leading',   'Subleading'       ),
    (r'Vtx Electron min $\Delta R(e,j)$', 'vtx_ele_mindrj',      ['vtx_leading_ele_mindRj','vtx_subleading_ele_mindRj'],False, 'Leading',   'Subleading'       ),
    # per-electron paired
    (r'Vtx Electron ID Score',            'vtx_ele_id',          ['vtx_ele_ID'],                                        False, None,        None               ),
    (r'Vtx Electron miniIso',             'vtx_ele_miniiso',     ['vtx_ele_miniIso',     'vtx_ele_miniIsoCorr'],        True,  'Regular',   'Corrected'        ),
    (r'Vtx Electron miniRelIso',          'vtx_ele_minireliso',  ['vtx_ele_miniRelIso',  'vtx_ele_miniRelIsoCorr'],     True,  'Regular',   'Corrected'        ),
    (r'Vtx Electron $d_{xy}$',            'vtx_ele_dxy',         ['vtx_ele_dxy',         'vtx_ele_refit_dxy'],          True,  r'$d_{xy}$', r'Refit $d_{xy}$'  ),
    (r'Vtx Electron $|d_{xy}|/|d_z|$',    'vtx_ele_dxydz',       ['vtx_ele_dxydz'],                                     False, None,        None               ),
    (r'Vertex $\log(|d_{xy}|/|d_z|)$',    'vtx_ele_logdxydz',    ['vtx_ele_logdxydz',    'vtx_min_logdxydz'],           False, r'per-$e$',  'Vertex min'       ),
    (r'Vertex $\chi^2/df$',               'vtx_chi2',            ['vtx_ele_track_chi2',  'vtx_reduced_chi2'],           True,  'Ele track', 'Vertex'           ),
    # per-vertex aggregates (single variable)                                                                           
    (r'Vertex Max Mini Rel Iso (corr)',   'vtx_maxminiiso',      ['vtx_maxMiniIso'],                                    True,  None,        None               ),
    # vertex paired nominal/refit                                                                                       
    (r'Vertex Min $d_{xy}$',              'vtx_min_refit_dxy',   ['vtx_min_dxy',  'vtx_min_refit_dxy'],                 True,  'Nominal',   'Refit'            ),
    (r'Vertex $m_{e^+e^-}$',              'vtx_mass',            ['vtx_mass',     'vtx_refit_mass'],                    False, 'Nominal',   'Refit'            ),
    (r'Vertex $\Delta R$',                'vtx_dr',              ['vtx_dr',       'vtx_refit_dr'],                      False, 'Nominal',   'Refit'            ),
    (r'Vertex $p_T$',                     'vtx_pt',              ['vtx_pt',       'vtx_refit_pt'],                      False, 'Nominal',   'Refit'            ),
    (r'Vertex $\eta$',                    'vtx_eta',             ['vtx_eta',      'vtx_refit_eta'],                     False, 'Nominal',   'Refit'            ),
    (r'Vertex $\phi$',                    'vtx_phi',             ['vtx_phi',      'vtx_refit_phi'],                     False, 'Nominal',   'Refit'            ),
]

os.makedirs('plots', exist_ok=True)

for title, outname, variables, doLogy, solid_label, dotted_label in plot_specs:
    is_paired = len(variables) > 1
    plot_dict = {
        'variable': variables if is_paired else variables[0],
        'year': 2024,
        'cut': cut,
    }
    ls = ['-', '--'] if is_paired else ['-']

    for series in series_list:
        m1s    = series['m1s']
        deltas = series['deltas']
        ctaus  = series['ctaus']
        tag    = series['tag']

        fig, ax = plt.subplots(figsize=size)
        style_dict = {
            **style_dict_base,
            'fig': fig, 'ax': ax,
            'ls': ls,
            'doLogy': doLogy,
        }

        cmap_idx = 0
        for m1 in m1s:
            for delta in deltas:
                for ctau in ctaus:
                    ptools.plot_signal_1D(s_hists, m1, delta, ctau, plot_dict, style_dict, cmap_idx=cmap_idx)
                    cmap_idx += 1

        # CMS label once per plot
        hep.cms.label('Private Work', data=True, year=plot_dict['year'], com='13.6', ax=ax)

        # legend: signal point handles from ax, plus line-style legend entries when paired
        handles, labels = ax.get_legend_handles_labels()
        if is_paired and solid_label is not None:
            handles += [
                mlines.Line2D([], [], color='black', ls='-',  label=solid_label),
                mlines.Line2D([], [], color='black', ls='--', label=dotted_label),
            ]
            labels += [solid_label, dotted_label]

        ax.legend(handles=handles, labels=labels)

        plt.title(rf'{title}: $M_1$ = {m1s}, $\Delta$ = {deltas}, c$\tau$ = {ctaus}mm')
        plt.tight_layout()
        plt.savefig(f'plots/hist_{plottag}_{outname}_{tag}.png')
        plt.close()
