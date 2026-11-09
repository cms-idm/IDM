import hist
import coffea.util as util
from coffea.processor import accumulate
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import re
import os
import sys
import mplhep as hep
hep.style.use("CMS")
plt.rcParams['font.size'] = 16.0
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize, LogNorm
import utils

from mplhep.styles.cms import cmap_petroff

cmap = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"] # cms-recommended version of 10-color scheme

bkg_cmap = {
    "QCD":cmap_petroff[0],
    "WJets":cmap_petroff[1],
    "ZJets":cmap_petroff[2],
    "DY":cmap_petroff[3],
    "Top":cmap_petroff[4],
    "Multiboson":cmap_petroff[5],
    "ZGamma":"darkkhaki",
    "DYLowMass":"tab:brown"
}

'''
# When using bkg list with extra processes
bkg_cmap = {
    "QCD": cmap[0],
    "WJets": cmap[1],
    "ZJets": cmap[2],
    "DY-M4to50": cmap[3],
    "DY-M50": cmap[4],
    "TTbar": cmap[5],
    "SingleTop": cmap[6],
    "Multiboson": cmap[7],
    "ZGamma": cmap[8],
    "TTX": cmap[9]
}
'''

selected_signals = [
    "sig_2018_Mchi-10p5_dMchi-1p0_ctau-1",
    "sig_2018_Mchi-11p0_dMchi-2p0_ctau-100",
    "sig_2018_Mchi-52p5_dMchi-5p0_ctau-10",
    "sig_2018_Mchi-77p0_dMchi-14p0_ctau-100"
]
selected_signals_cmap = {
    "sig_2018_Mchi-10p5_dMchi-1p0_ctau-1":"k",
    "sig_2018_Mchi-11p0_dMchi-2p0_ctau-100":"g",
    "sig_2018_Mchi-52p5_dMchi-5p0_ctau-10":"c",
    "sig_2018_Mchi-77p0_dMchi-14p0_ctau-100":"b"
}

class histContainer:
    def __init__(self,path,noMeta=False,bkg=False):
        self.histos, self.metadata = self.loadCoffeaFiles(path,noMeta)
        self.hnames = list(self.histos.keys())
        if bkg:
            self.catSamps,self.catNames = utils.bkg_categories(self.histos['cutflow'])
            self.cats = list(self.catSamps.keys())
    
    def loadCoffeaFiles(self,path,noMeta):
        if type(path) != list:
            path = [path]
        histos,metas = [], []
        for p in path:
            if noMeta:
                h,m = util.load(p), None
            else:
                h,m = util.load(p)
            histos.append(h)
            metas.append(m)
        histos = accumulate(histos)
        metadata = None if noMeta else accumulate(metas)
        return histos, metadata

    def load(self,hname):
        return self.histos[hname]
    def load(self,hname):
        return self.histos[hname]
    def names(self,spec=None):
        if spec is not None:
            print("\n".join([h for h in self.hnames if spec in h]))
        else:
            print("\n".join(self.hnames))

def getPresentSamples(h,possibleSamples):
    for a in h.axes:
        if a.name=='samp':
            ax = a
            break
    nbins = len(ax.centers)
    avail = [ax.bin(i) for i in range(nbins)]
    return [p for p in possibleSamples if p in avail]

def make_label(row):
    m1 = int(row['m1'])
    delta = int(100*row['delta'])
    ct = int(row['ctau'])
    label = f"{delta}%, {m1} GeV, {ct} mm"
    return label

def makeNMinus1(h1,h2,lessThan=False):
    assert len(h1.axes)==1 and h1.axes == h2.axes
    ax = h1.axes[0]
    if lessThan:
        s1 = np.cumsum(h1.counts(flow=True))
        s2 = np.cumsum(h2.counts(flow=True))
    else:
        s1 = np.cumsum(h1.counts(flow=True)[::-1])[::-1]
        s2 = np.cumsum(h2.counts(flow=True)[::-1])[::-1]
    x = ax.edges
    dx = x[1]-x[0]
    x = np.append(x,[x[-1]+dx])
    signif = np.where(s2>0,s1/np.sqrt(s2),-1)
    signif[(signif==-1) & (s1>0)] = np.inf
    signif[(signif==-1) & (s1==0)] = 0
    return x, signif

def makeNMinus1_multiBkg(hnum,hdens,lessThan=False):
    assert len(hnum.axes)==1 and hnum.axes == hdens[0].axes
    ax = hnum.axes[0]
    if lessThan:
        snum = np.cumsum(hnum.counts(flow=True))
        sden = sum([np.cumsum(hd.counts(flow=True)) for hd in hdens])
    else:
        snum = np.cumsum(hnum.counts(flow=True)[::-1])[::-1]
        sden = sum([np.cumsum(hd.counts(flow=True)[::-1])[::-1] for hd in hdens])
    x = ax.edges
    dx = x[1]-x[0]
    x = np.append(x,[x[-1]+dx])
    signif = np.where(sden>0,snum/np.sqrt(sden),-1)
    signif[(signif==-1) & (snum>0)] = np.inf
    signif[(signif==-1) & (snum==0)] = 0
    return x, signif

def makeCutEff(h,lessThan=False):
    ax = h.axes[0]
    if lessThan:
        s = np.cumsum(h.counts(flow=True)[:-1])
    else:
        s = np.cumsum(h.counts(flow=True)[1:][::-1])[::-1]
    x = ax.edges
    #dx = x[1]-x[0]
    #x = np.append(x,[x[-1]+dx])
    eff = s/np.sum(h.counts(flow=True))
    return x, eff, s
    

def getSampleInfo(histos,hname="ele_kinematics"):
    samps = [s.name for s in histos[hname].axis("sample").identifiers()]
    info = {}
    masses = []
    cts = []
    for s in samps:
        ct = re.findall("ctau-(\d+)",s)[0]
        m, dm = re.findall("Mchi-(\d+p\d)_dMchi-(\d+p\d)",s)[0]
        m = m.replace("p",".")
        dm = dm.replace("p",".")
        entry = "{0}-{1}".format(m,dm)
        
        if entry not in info.keys():
            info[entry] = {}
        info[entry][ct] = s

        if entry not in masses:
            masses.append(entry)
        if ct not in cts:
            cts.append(ct)
    return info, masses, cts

def reduceSampleName(name,lifetime=False,mass=False,full=False,verbosity=0):
    output = name
    m, dm = re.findall("Mchi-(\d+p\d)_dMchi-(\d+p\d)",name)[0]
    ct = re.findall("ctau-(\d+)",name)[0]
    m = m.replace("p",".")
    dm = dm.replace("p",".")
    if full:
        output = r"$m_\chi = {0}$ GeV, $\Delta m_\chi = {1}$ GeV, $c\tau = {2}$ mm".format(m,dm,ct)
    if lifetime:
        output = r"$c\tau = {0}$ mm".format(ct)
    if mass:
        if verbosity == 0: output = r"$({0}, {1})$ GeV".format(m,dm)
        if verbosity == 1: output = r"$(m_\chi, \Delta m_\chi) = ({0}, {1})$ GeV".format(m,dm)
        if verbosity == 2: output = r"$m_\chi = {0}$ GeV, $\Delta m_\chi = {1}$ GeV".format(m,dm)
    return output

def signalPoint(name):
    a = re.search('Mchi-(\d+p\d+)_dMchi-(\d+p\d+)_ctau-(\d+)',name)
    mchi = float(a.group(1).replace("p","."))
    dmchi = float(a.group(2).replace("p","."))
    ctau = float(a.group(3).replace("p","."))
    m1 = mchi - dmchi/2
    m2 = mchi + dmchi/2
    delta = dmchi/m1
    return {"mchi":mchi, "dmchi":dmchi, "ctau":ctau, "m1":m1, "m2":m2, "delta":delta, "name":name}

def getCut(label,n=2):
    name = ""
    while label[0:n]!=label[n:2*n] and n<len(label):
        name=label[0:n+1]
        n+=1
    return name

def getHTlow(sampName):
    ht = int(re.search("HT(\d+)to",sampName).group(1))
    return ht

def setDefaultStyle(fontsize=14):
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["figure.figsize"] = (10,8)

def hget(h,samp,cut):
    return h[{"samp":samp,"cut":cut}]

def makeCDF(h,start,stop,bins=100,right=True,nevents=False,category=False):
    x = np.linspace(start,stop,bins)
    n_tot = h.sum(flow=True).value
    if not category:
        if right:
            yields = np.array([h[complex(f"{xi}j")::sum].value for xi in x])
        else:
            yields = np.array([h[:complex(f"{xi}j"):sum].value for xi in x])
    else:
        edges_ordered = h.axes[0].edges[::-1] if right else h.axes[0].edges
        yields_raw = np.cumsum(h.values()[::-1]) if right else np.cumsum(h.values())
        # add extra point to yields to draw steps
        yields = []
        x = []
        for i,y in enumerate(yields_raw):
            yields.extend([y,y])
            x.extend([edges_ordered[i],edges_ordered[i+1]])
        yields = np.array(yields)
        x = np.array(x)
    effs = yields/n_tot
    if nevents:
        return x,yields
    else:
        return x, effs

def overlay(h,overlay,label_key=None,**kwargs):
    axes = h.axes
    targ = None
    for a in axes:
        if a.name == overlay:
            targ = a
    if targ is None:
        print("can't find overlay axis!")
        return
    n_overlay = len(targ.centers)
    labels = [targ.value(i) for i in range(n_overlay)]
    histos = [h[{overlay:l}] for l in labels]
    if label_key is not None:
        labels = [label_key[targ.value(i)] for i in range(n_overlay)]
    hep.histplot(histos,label=labels,**kwargs)

# Plot efficiency type stuff
def plot_signal_efficiency(sig_histo, df, plot_dict_sig_eff):
    """
    Example plot_dict_sig_eff

    plot_dict_sig_eff = {
    
    # Select signal points to display
    'm1s': [5, 20, 30, 50, 80, 100],
    'deltas': [0.2],
    'ctaus': [1],

    # Plot display styling
    'ylim': [1, 1e+6], # None for default
    'doLog': True,
    
    'ylabel': 'Events', # None for default
    'title': rf"Cutflow: $\Delta$ = {deltas[0]}, c$\tau$ = {ctaus[0]}mm",
    'label': None,

    # Plot saving
    'doSave': False,
    'outDir': './plots/',
    'outName': f'Cutflow_SR_signal_delta_{deltas[0]}_ct_{ctaus[0]}.png'
    }

    """
    
    cuts = utils.get_signal_list_of_cuts(sig_histo)

    m1_list = []
    for point in df.index.values:
        sig_dict = signalPoint(point)
        m1 = int(sig_dict['m1'])
        m1_list.append(m1)

    df['m1'] = m1_list
    df = df.sort_values(by=['m1']) # sort by m1
    df.pop('m1')
    
    for point in df.index.values:
        sig_dict = signalPoint(point)
        m1 = int(sig_dict['m1'])
        delta = sig_dict['delta']
        dmchi = sig_dict['dmchi']
        ctau = int(sig_dict['ctau'])
        
        if (m1 in plot_dict_sig_eff['m1s']) and (delta in plot_dict_sig_eff['deltas']):
            if ctau in plot_dict_sig_eff['ctaus']:
                if plot_dict_sig_eff['label'] == None:
                    label = rf"($M_{1}$, $\Delta$) = ({m1:.0f}, {dmchi:.0f}) GeV, c$\tau$ = {int(ctau)}mm"
                else:
                    label = plot_dict_sig_eff['label']
                plt.plot(cuts, df.loc[point], label=label)

    if plot_dict_sig_eff['doLog']:
        plt.yscale('log')

    if plot_dict_sig_eff['ylim'] != None:
        plt.ylim(plot_dict_sig_eff['ylim'][0], plot_dict_sig_eff['ylim'][1])

    
    plt.grid()
    
    plt.ylabel(plot_dict_sig_eff['ylabel'])
    plt.title(plot_dict_sig_eff['title'])
    
    plt.xticks(ticks = np.arange(len(cuts)), labels = cuts, rotation = 45, ha = 'right')
    
    plt.legend(loc='upper right')
    
    if plot_dict_sig_eff['doSave']:
        os.makedirs(plot_dict_sig_eff['outDir'], exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{plot_dict_sig_eff['outDir']}/{plot_dict_sig_eff['outName']}")
        print(f"Saved: {plot_dict_sig_eff['outDir']}/{plot_dict_sig_eff['outName']}")
    

def plot_bkg_efficiency(bkg_histos, df, plot_dict_bkg_eff):
    """
    Example:

    plot_dict_bkg_eff = {

    # Select processes
    'processes': 'all', # Otherwise, give as a list; ['WJets', 'ZJets', 'Total']

    # Plot display styling
    'ylim': None, # None for default; otherwise [ymin, ymax]
    'doLog': True,
    
    'ylabel': 'Events', # None for default
    'title': rf"Cutflow", 
    'label': None,
    'color': None,

    # Plot saving
    'doSave': True,
    'outDir': './plots/cutflow/',
    'outName': ''
    }

    """
    if plot_dict_bkg_eff['processes'] == 'all':
        processes = df.index.values.tolist()
    else:
        processes = plot_dict_bkg_eff['processes']
    cuts = utils.get_bkg_list_of_cuts(bkg_histos)

    # Color map for each process
    for process in processes:
        if plot_dict_bkg_eff['label'] != None:
            label = plot_dict_bkg_eff['label']
        else:
            label = plot_dict_bkg_eff
        
        if 'Total' in process:
            if plot_dict_bkg_eff['color'] != None:
                color = plot_dict_bkg_eff['color']
            else:
                color = 'black'
            plt.plot(cuts, df.loc[process], label=label, color=color)
        else:
            plt.plot(cuts, df.loc[process], label=label, color = bkg_cmap[process])

    if plot_dict_bkg_eff['doLog']:
        plt.yscale('log')

    if plot_dict_bkg_eff['ylim'] != None:
        plt.ylim(plot_dict_bkg_eff['ylim'][0], plot_dict_bkg_eff['ylim'][1])
    
    plt.grid()
    
    plt.ylabel(plot_dict_bkg_eff['ylabel'])
    plt.title(plot_dict_bkg_eff['title'])
    
    plt.xticks(ticks = np.arange(len(cuts)), labels = cuts, rotation = 45, ha = 'right')
    
    plt.legend(loc='upper right')
    
    if plot_dict_bkg_eff['doSave']:
        os.makedirs(plot_dict_bkg_eff['outDir'], exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{plot_dict_bkg_eff['outDir']}/{plot_dict_bkg_eff['outName']}")
        print(f"Saved: {plot_dict_bkg_eff['outDir']}/{plot_dict_bkg_eff['outName']}")
    
    plt.show()


def plot_bkg_efficiency_legacy(bkg_histos, df, doLog = True, ylabel = '', title = '', isLegacy = True):
    processes = df.index.values.tolist()
    cuts = utils.get_bkg_list_of_cuts(bkg_histos, isLegacy = isLegacy)

    # Color map for each process
    # cmap = mpl.colormaps['Set3'].colors
    # cmap = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"] # cms-recommended for 6-color scheme
    cmap = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"] # cms-recommended
    
    colors = { 'W+jets': cmap[0],
               'Z+jets': cmap[1],
               'QCD': cmap[2],
               'DY': cmap[3],
               'Top': cmap[4],
               'TTJetsDiLept': cmap[5],
               'Diboson': cmap[6],
               'Triboson': cmap[7],
               'Total': cmap[8]
    }

    for process in processes:
        plt.plot(cuts, df.loc[process], label=process, color = colors[process])

    plt.grid()

    if doLog:
        plt.yscale('log')
    
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.xticks(ticks = np.arange(len(cuts)), labels = cuts, rotation = 45, ha = 'right')
    
    plt.legend()
    plt.show()

# Plot kinematics
def plot_signal_1D(sig_histo, m1, delta, ctau, plot_dict, style_dict):
    """
    Example:

    plot_dict = {
    'variable': 'sel_vtx_vxy10',
    'cut': 'cut7',
    'year': 2018
    }
    
    style_dict = {
        'fig': fig,
        'ax': ax,
        'rebin': 1j,
        'xlim': None,     # if None, the default will show up; otherwise give as a list, i.e. [0, 10]
        'doLogy': True, 
        'doLogx': False,
        'doDensity': False,
        'doYerr': False, 
        'xlabel': r"$L_{xy}$ [cm]",   # if None, the default will show up; otherwise give as a string, i.e. 'Electron dxy'
        'ylabel': 'Events/0.1cm',   # if None, the default will show up; otherwise give as a string, i.e. 'Efficiency'
        'label': None,    # if None, the default will show up; otherwise give as a string, i.e. 'Highest ctau signal samples'
        'flow': None,     # overflow
        'doSave': False,
        'outDir': './plots/',
        'outName': f'background_cut7_Lxy_max10.png'
    }

    """

    fig = style_dict['fig']
    ax = style_dict['ax']
    
    hep.cms.label('', data=False, year=plot_dict['year'])
    
    # get signal point info
    si = utils.get_signal_point_dict(sig_histo)
    samp_df = si[(si.m1 == m1) & (si.delta == delta) & (si.ctau == ctau)]
    
    samp = samp_df.name[0]

    m1 = samp_df.m1[0]
    dmchi = samp_df.dmchi[0]
    ctau = samp_df.ctau[0]
    label = rf"$(m_\chi, \Delta m_\chi) = ({m1:.0f}, {dmchi:.0f})$ GeV"

    if style_dict['label'] != None:
        label = style_dict['label']
    
    # get histogram from coffea output
    histo = sig_histo[plot_dict['variable']][{"samp":samp, "cut": plot_dict['cut']}]

    # rebinning
    histo = histo[::style_dict['rebin']]

    # set x range manually
    if style_dict['xlim'] != None:
        xlim = style_dict['xlim']
        xbin_range = np.where((histo.axes.edges[0] > xlim[0]) & (histo.axes.edges[0] < xlim[1]))[0]
        histo = histo[ int(xbin_range[0])-1:int(xbin_range[-1]+1) ]

    # x and y labels
    if style_dict['xlabel'] != None:
        ax.set_xlabel(style_dict['xlabel'])

    if style_dict['ylabel'] != None:
        ax.set_ylabel(style_dict['ylabel'])
    else:   
        binwidth = histo.axes.widths[0][0]
        if style_dict['doDensity']:
            ax.set_ylabel(f'A.U./{binwidth:.3f}')
        else:
            ax.set_ylabel(f'Events/{binwidth:.3f}')

    # x,y scale
    if style_dict['doLogx']:
        ax.set_xscale('log')
    if style_dict['doLogy']:
        ax.set_yscale('log')

    # Plot
    hep.histplot(histo, yerr=style_dict['doYerr'], density=style_dict['doDensity'], ax=ax, histtype='step', flow=style_dict['flow'], label = label)

    plt.legend()
    
    if style_dict['doSave']:
        os.makedirs(style_dict['outDir'], exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{style_dict['outDir']}/{style_dict['outName']}")
        print(f"Saved: {style_dict['outDir']}/{style_dict['outName']}")
    

def plot_signal_2D(sig_histo, m1, delta, ctau, plot_dict, style_dict):
    """
    Example:

    plot_dict = {
        'variable': 'sel_vtx_vx_vs_vy',
        'cut': 'cut9',
        'year': 2018
    }
    
    style_2d_dict = {
        'fig': fig,
        'ax': ax,
        'xrebin': 1j,
        'yrebin': 1j,
        'xlim': None,     # if None, the default will show up; otherwise give as a list, i.e. [0, 10]  
        'ylim': None,     # if None, the default will show up; otherwise give as a list, i.e. [0, 10]
        'doLogy': False, 
        'doLogx': False,
        'doLogz': True,
        'xlabel': r"$v_{x}$ [cm]",   # if None, the default will show up; otherwise give as a string, i.e. 'Electron dxy'
        'ylabel': r"$v_{y}$ [cm]",   # if None, the default will show up; otherwise give as a string, i.e. 'Efficiency'
        'zlabel': 'Events',   
        'flow': None,     # overflow
        'doSave': True,
        'outDir': './plots/',
        'outName': f'signal_cut7_vx_vs_vy_m1_{m1}_delta_{delta}_ctau_{ctau}.png'
    }

    """

    fig = style_dict['fig']
    ax = style_dict['ax']
    
    hep.cms.label('', data=False, year=plot_dict['year'])
    
    # get signal point info
    si = utils.get_signal_point_dict(sig_histo)
    samp_df = si[(si.m1 == m1) & (si.delta == delta) & (si.ctau == ctau)]
    
    samp = samp_df.name[0]

    m1 = samp_df.m1[0]
    dmchi = samp_df.dmchi[0]
    ctau = samp_df.ctau[0]
    label = f'({m1}, {dmchi}) GeV, ctau = {int(ctau)}mm'
    
    # get histogram from coffea output
    histo = sig_histo[plot_dict['variable']][{"samp":samp, "cut": plot_dict['cut']}]

    # rebinning
    histo = histo[::style_dict['xrebin'],::style_dict['yrebin']]

    # set x range manually
    if style_dict['xlim'] != None:
        xlim = style_dict['xlim']
        xbin_range = np.where((histo.axes.edges[0] > xlim[0]) & (histo.axes.edges[0] < xlim[1]))[0]
        histo = histo[ int(xbin_range[0])-1:int(xbin_range[-1]+1), : ]
    if style_dict['ylim'] != None:
        ylim = style_dict['ylim']
        ybin_range = np.where((histo.axes.edges[1] > ylim[0]) & (histo.axes.edges[1] < ylim[1]))[1]
        histo = histo[ :, int(ybin_range[0]):int(ybin_range[-1]+1) ]
    
    # x and y labels
    if style_dict['xlabel'] != None:
        ax.set_xlabel(style_dict['xlabel'])
    if style_dict['ylabel'] != None:
        ax.set_ylabel(style_dict['ylabel'])

    # x,y scale
    if style_dict['doLogx']:
        ax.set_xscale('log')
    if style_dict['doLogy']:
        ax.set_yscale('log')
    
    # Plot
    if style_dict['doLogz']:
        hep.hist2dplot(histo, flow=style_dict['flow'], norm=mpl.colors.LogNorm(), ax=ax, cbarextend=True)
    else:
        hep.hist2dplot(histo, flow=style_dict['flow'], ax=ax, cbarextend=True)

    # z label
    if style_dict['zlabel'] != None:
        fig.get_axes()[-1].set_ylabel(style_dict['zlabel'])
    
    if style_dict['doSave']:
        os.makedirs(style_dict['outDir'], exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{style_dict['outDir']}/{style_dict['outName']}")
        print(f"Saved: {style_dict['outDir']}/{style_dict['outName']}")



def get_bkg_histo_1d(bkg_histos, plot_dict, style_dict, processes = 'all'):
    if processes == 'all':
        processes = list(set(utils.get_bkg_point_dict(bkg_histos).loc[:, 'Process']))
        
    subprocess = {process: [] for process in processes} # initialize the dictionary of bkg processes
        
    availSubCat = list(bkg_histos[plot_dict['variable']].axes['samp']) # get the list of subprocesses available for the histogram
    for samp in availSubCat:
        process = utils.get_bkg_point_dict(bkg_histos).loc[samp][0]            
        if process in processes:
            subprocess[process].append(samp) # fill out the bkg process list with the available subprocesses
            
    # Get histogram for each process
    bkg={}
    bkg[plot_dict['variable']] = {process:bkg_histos[plot_dict['variable']][{"samp":subprocess[process]}][{"samp": sum}] for process in processes}
        
    # sort the histograms by the entries and stack
    for process in processes:
        entries = {process: bkg[plot_dict['variable']][process].sum().value for process in processes}
        
    sorted_entries = dict(sorted(entries.items(), key=lambda x:x[1], reverse = False))

    # histogram
    bkg_stack = {}
        
    # add histos to stack after rebinning and range setting
    for process in sorted_entries.keys():
        bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][plot_dict['cut'],::style_dict['rebin']]
        
        # set x range manually
        if style_dict['xlim'] != None:
            xlim = style_dict['xlim']
            xbin_range = np.where((bkg[plot_dict['variable']][process].axes.edges[0] > xlim[0]) & (bkg[plot_dict['variable']][process].axes.edges[0] < xlim[1]))[0]
            bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][ int(xbin_range[0])-1:int(xbin_range[-1]+1) ]
        
        bkg_stack[process] = bkg[plot_dict['variable']][process]
        
    hb = hist.Stack.from_dict(bkg_stack)

    return hb

def get_bkg_histo_stacked_1d(bkg_histos, plot_dict, style_dict, processes = 'all'):
    if processes == 'all':
        processes = list(set(utils.get_bkg_point_dict(bkg_histos).loc[:, 'Process']))
        
    subprocess = {process: [] for process in processes} # initialize the dictionary of bkg processes
        
    availSubCat = list(bkg_histos[plot_dict['variable']].axes['samp']) # get the list of subprocesses available for the histogram
    for samp in availSubCat:
        process = utils.get_bkg_point_dict(bkg_histos).loc[samp][0]            
        if process in processes:
            subprocess[process].append(samp) # fill out the bkg process list with the available subprocesses
            
    # Get histogram for each process
    bkg={}
    bkg[plot_dict['variable']] = {process:bkg_histos[plot_dict['variable']][{"samp":subprocess[process]}][{"samp": sum}] for process in processes}
        
    # sort the histograms by the entries and stack
    for process in processes:
        entries = {process: bkg[plot_dict['variable']][process].sum().value for process in processes}
        
    sorted_entries = dict(sorted(entries.items(), key=lambda x:x[1], reverse = False))

    bkg_stack = {}
        
    # add histos to stack after rebinning and range setting
    for idx, process in enumerate(sorted_entries.keys()):
        bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][plot_dict['cut'],::style_dict['rebin']]
        
        # set x range manually
        if style_dict['xlim'] != None:
            xlim = style_dict['xlim']
            xbin_range = np.where((bkg[plot_dict['variable']][process].axes.edges[0] > xlim[0]) & (bkg[plot_dict['variable']][process].axes.edges[0] < xlim[1]))[0]
            bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][ int(xbin_range[0])-1:int(xbin_range[-1]+1) ]
        
        if idx == 0:
            bkg_stack = bkg[plot_dict['variable']][process]
        else:
            bkg_stack += bkg[plot_dict['variable']][process]

    return bkg_stack




def get_data_histo_1d(data_histo, plot_dict, style_dict):
    runs = list(data_histo['cutflow_cts'].keys())

    for idx, run in enumerate(runs):
        try:
            if idx == 0:
                histo = data_histo[plot_dict['variable']][{"samp":run, "cut": plot_dict['cut']}]
            else:
                histo += data_histo[plot_dict['variable']][{"samp":run, "cut": plot_dict['cut']}]
        except:
            print('No run')

    # rebinning
    histo = histo[::style_dict['rebin']]

    # set x range manually
    if style_dict['xlim'] != None:
        xlim = style_dict['xlim']
        xbin_range = np.where((histo.axes.edges[0] > xlim[0]) & (histo.axes.edges[0] < xlim[1]))[0]
        histo = histo[ int(xbin_range[0])-1:int(xbin_range[-1]+1) ]

    return histo

def plot_bkg_1d(bkg_histos, plot_dict, style_dict, isLegacy = False, processes = 'all'):
    """
    Example:

    plot_dict = {
    'variable': 'sel_vtx_vxy10',
    'cut': 'cut7',
    'year': 2018
    }
    
    style_dict = {
        'fig': fig,
        'ax': ax,
        'rebin': 1j,
        'xlim': None,     # if None, the default will show up; otherwise give as a list, i.e. [0, 10]
        'doLogy': True, 
        'doLogx': False,
        'doDensity': False,
        'doYerr': False, 
        'xlabel': r"$L_{xy}$ [cm]",   # if None, the default will show up; otherwise give as a string, i.e. 'Electron dxy'
        'ylabel': 'Events/0.1cm',   # if None, the default will show up; otherwise give as a string, i.e. 'Efficiency'
        'label': None,    # if None, the default will show up; otherwise give as a string, i.e. 'Highest ctau signal samples'
        'flow': None,     # overflow
        'doSave': False,
        'outDir': './plots/',
        'outName': f'background_cut7_Lxy_max10.png'
    }

    """
    fig = style_dict['fig']
    ax = style_dict['ax']

    # CMS styling
    #hep.cms.label(r"$\mathrm{Private Work}$", data=False, year=plot_dict['year'])
    hep.cms.label('', data=False, year=plot_dict['year'])
    
    if isLegacy:
        return plot_bkg_1d_legacy(ax, bkg_histos, plot_dict, style_dict, processes, isLegacy)
    else:
        # if process is given as a list, i.e. ['DY', 'W+jets'], plot only these processes in the list; otherwise, plot all as default
        if processes == 'all':
            processes = list(set(utils.get_bkg_point_dict(bkg_histos).loc[:, 'Process']))
        
        subprocess = {process: [] for process in processes} # initialize the dictionary of bkg processes
        
        availSubCat = list(bkg_histos[plot_dict['variable']].axes['samp']) # get the list of subprocesses available for the histogram
        for samp in availSubCat:
            process = utils.get_bkg_point_dict(bkg_histos).loc[samp][0]            
            if process in processes:
                subprocess[process].append(samp) # fill out the bkg process list with the available subprocesses
            
        # Get histogram for each process
        bkg={}
        bkg[plot_dict['variable']] = {process:bkg_histos[plot_dict['variable']][{"samp":subprocess[process]}][{"samp": sum}] for process in processes}
        
        # sort the histograms by the entries and stack
        for process in processes:
            entries = {process: bkg[plot_dict['variable']][process].sum().value for process in processes}
        
        sorted_entries = dict(sorted(entries.items(), key=lambda x:x[1], reverse = False))
        
        # histogram
        bkg_stack = {}
        
        # add histos to stack after rebinning and range setting
        for process in sorted_entries.keys():
            bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][plot_dict['cut'],::style_dict['rebin']]
        
            # set x range manually
            if style_dict['xlim'] != None:
                xlim = style_dict['xlim']
                xbin_range = np.where((bkg[plot_dict['variable']][process].axes.edges[0] > xlim[0]) & (bkg[plot_dict['variable']][process].axes.edges[0] < xlim[1]))[0]
                bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][ int(xbin_range[0])-1:int(xbin_range[-1]+1) ]
        
            bkg_stack[process] = bkg[plot_dict['variable']][process]
        
        hb = hist.Stack.from_dict(bkg_stack)
            
        color_list = [bkg_cmap[process] for process in sorted_entries.keys()]
        
        # x and y labels
        if style_dict['xlabel'] != None:
            ax.set_xlabel(style_dict['xlabel'])
        
        if style_dict['ylabel'] != None:
            ax.set_ylabel(style_dict['ylabel'])
        else:
            binwidth = hb[0].axes.widths[0][0]
                
            if style_dict['doDensity']:
                ax.set_ylabel(f'A.U./{binwidth:.3f}')
            else:
                ax.set_ylabel(f'Events/{binwidth:.3f}')
        
        # x,y scale
        if style_dict['doLogx']:
            ax.set_xscale('log')
        if style_dict['doLogy']:
            ax.set_yscale('log')
            
        # Plot
        hb.plot(stack=True, yerr=style_dict['doYerr'], density=style_dict['doDensity'], flow=style_dict['flow'], histtype='fill', color=color_list)
        
        # legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])

        if style_dict['doSave']:
            os.makedirs(style_dict['outDir'], exist_ok=True)
            plt.tight_layout()
            plt.savefig(f"{style_dict['outDir']}/{style_dict['outName']}")
            print(f"Saved: {style_dict['outDir']}/{style_dict['outName']}")

def plot_bkg_1d_legacy(ax, bkg_histos, plot_dict, style_dict, processes = 'all', isLegacy = True):  
    
    if processes == 'all':
        #processes = bkg_histos.keys()

        list_cut_index = utils.get_bkg_list_of_cuts(bkg_histos, get_cut_idx=True, isLegacy = isLegacy)
        list_cut_name = utils.get_bkg_list_of_cuts(bkg_histos, get_cut_idx=False, isLegacy = isLegacy)
        
        cut_name = plot_dict['cut']
        
        df = utils.get_bkg_cutflow_df(bkg_histos, 'cutflow_cts', isLegacy = isLegacy).iloc[:-1]
        
        df = df[list_cut_name[list_cut_index.index(cut_name)]]
        
        processes = df.index[df != 0].to_list()
    
    # if process is given as a list, i.e. ['DY', 'W+jets'], plot only these processes in the list; otherwise, plot all as default

    bkg={}
    bkg[plot_dict['variable']] = {process:bkg_histos[process][plot_dict['variable']][{"samp":sum}] for process in processes}
    
    # sort the histograms by the entries and stack
    for process in processes:
        entries = {process: bkg[plot_dict['variable']][process].sum().value for process in processes}
    
    sorted_entries = dict(sorted(entries.items(), key=lambda x:x[1], reverse = False))

    # histogram
    bkg_stack = {}
    
    # add histos to stack after rebinning and range setting
    for process in sorted_entries.keys():
        bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][plot_dict['cut'],::style_dict['rebin']]

        # set x range manually
        if style_dict['xlim'] != None:
            xlim = style_dict['xlim']
            xbin_range = np.where((bkg[plot_dict['variable']][process].axes.edges[0] > xlim[0]) & (bkg[plot_dict['variable']][process].axes.edges[0] < xlim[1]))[0]
            bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][ int(xbin_range[0])-1:int(xbin_range[-1]+1) ]

        bkg_stack[process] = bkg[plot_dict['variable']][process]

    hb = hist.Stack.from_dict(bkg_stack)
    
    # Color map for each process
    # cmap = mpl.colormaps['Set3'].colors
    # cmap = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"] # cms-recommended for 6-color scheme
    cmap = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"] # cms-recommended
    
    colors = { 'W+jets': cmap[0],
               'Z+jets': cmap[1],
               'QCD': cmap[2],
               'DY': cmap[3],
               'Top': cmap[4],
               'TTJetsDiLept': cmap[5],
               'Diboson': cmap[6],
               'Triboson': cmap[7],
    }
    
    color_list = [colors[process] for process in sorted_entries.keys()]

    # x and y labels
    if style_dict['xlabel'] != None:
        ax.set_xlabel(style_dict['xlabel'])

    if style_dict['ylabel'] != None:
        ax.set_ylabel(style_dict['ylabel'])
    else:
        binwidth = hb[0].axes.widths[0][0]
        
        if style_dict['doDensity']:
            ax.set_ylabel(f'A.U./{binwidth:.3f}')
        else:
            ax.set_ylabel(f'Events/{binwidth:.3f}')

    # x,y scale
    if style_dict['doLogx']:
        ax.set_xscale('log')
    if style_dict['doLogy']:
        ax.set_yscale('log')
    
    # Plot
    hb.plot(stack=True, yerr=style_dict['doYerr'], density=style_dict['doDensity'], flow=style_dict['flow'], histtype='fill', color=color_list)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])


def plot_bkg_1d_stacked(bkg_histos, plot_dict, style_dict, isLegacy = False, processes = 'all'):
    """
    Example:

    plot_dict = {
    'variable': 'sel_vtx_vxy10',
    'cut': 'cut7',
    'year': 2018
    }
    
    style_dict = {
        'fig': fig,
        'ax': ax,
        'rebin': 1j,
        'xlim': None,     # if None, the default will show up; otherwise give as a list, i.e. [0, 10]
        'doLogy': True, 
        'doLogx': False,
        'doDensity': False,
        'doYerr': False, 
        'xlabel': r"$L_{xy}$ [cm]",   # if None, the default will show up; otherwise give as a string, i.e. 'Electron dxy'
        'ylabel': 'Events/0.1cm',   # if None, the default will show up; otherwise give as a string, i.e. 'Efficiency'
        'label': None,    # if None, the default will show up; otherwise give as a string, i.e. 'Highest ctau signal samples'
        'flow': None,     # overflow
        'doSave': False,
        'outDir': './plots/',
        'outName': f'background_cut7_Lxy_max10.png'
    }

    """
    fig = style_dict['fig']
    ax = style_dict['ax']
    
    hep.cms.label('', data=False, year=plot_dict['year'])
    
    if isLegacy:
        return plot_bkg_1d_stacked_legacy(ax, bkg_histos, plot_dict, style_dict, processes = 'all', isLegacy = isLegacy)
    else:
        # if process is given as a list, i.e. ['DY', 'W+jets'], plot only these processes in the list; otherwise, plot all as default
        if processes == 'all':
            processes = list(set(utils.get_bkg_point_dict(bkg_histos).loc[:, 'Process']))
        
        subprocess = {process: [] for process in processes} # initialize the dictionary of bkg processes
        
        availSubCat = list(bkg_histos[plot_dict['variable']].axes['samp']) # get the list of subprocesses available for the histogram
        for samp in availSubCat:
            process = utils.get_bkg_point_dict(bkg_histos).loc[samp][0]
            if process in processes:
                subprocess[process].append(samp) # fill out the bkg process list with the available subprocesses
            
        # Get histogram for each process
        bkg={}
        bkg[plot_dict['variable']] = {process:bkg_histos[plot_dict['variable']][{"samp":subprocess[process]}][{"samp": sum}] for process in processes}
        
        # sort the histograms by the entries and stack
        for process in processes:
            entries = {process: bkg[plot_dict['variable']][process].sum().value for process in processes}
        
        sorted_entries = dict(sorted(entries.items(), key=lambda x:x[1], reverse = False))
        
        # histogram
        bkg_stack = {}
        
        # add histos to stack after rebinning and range setting
        for idx, process in enumerate(sorted_entries.keys()):
            bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][plot_dict['cut'],::style_dict['rebin']]
        
            # set x range manually
            if style_dict['xlim'] != None:
                xlim = style_dict['xlim']
                xbin_range = np.where((bkg[plot_dict['variable']][process].axes.edges[0] > xlim[0]) & (bkg[plot_dict['variable']][process].axes.edges[0] < xlim[1]))[0]
                bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][ int(xbin_range[0])-1:int(xbin_range[-1]+1) ]
        
            if idx == 0:
                bkg_stack = bkg[plot_dict['variable']][process]
            else:
                bkg_stack += bkg[plot_dict['variable']][process]
        
        # x and y labels
        if style_dict['xlabel'] != None:
            ax.set_xlabel(style_dict['xlabel'])
    
        if style_dict['ylabel'] != None:
            ax.set_ylabel(style_dict['ylabel'])
        else:
            binwidth = bkg_stack.axes.widths[0][0]
            
            if style_dict['doDensity']:
                ax.set_ylabel(f'A.U./{binwidth:.3f}')
            else:
                ax.set_ylabel(f'Events/{binwidth:.3f}')
    
        # x,y scale
        if style_dict['doLogx']:
            ax.set_xscale('log')
        if style_dict['doLogy']:
            ax.set_yscale('log')  
        
        # Plot
        hep.histplot(bkg_stack, yerr=style_dict['doYerr'], density=style_dict['doDensity'], ax=ax, histtype='step', flow=style_dict['flow'], label = style_dict['label'])
        
        # legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])

        if style_dict['doSave']:
            os.makedirs(style_dict['outDir'], exist_ok=True)
            plt.tight_layout()
            plt.savefig(f"{style_dict['outDir']}/{style_dict['outName']}")
            print(f"Saved: {style_dict['outDir']}/{style_dict['outName']}")
        

def plot_bkg_1d_stacked_legacy(ax, bkg_histos, plot_dict, style_dict, processes = 'all', isLegacy = True):  
    if processes == 'all':
        #processes = bkg_histos.keys()

        list_cut_index = utils.get_bkg_list_of_cuts(bkg_histos, get_cut_idx=True, isLegacy = isLegacy)
        list_cut_name = utils.get_bkg_list_of_cuts(bkg_histos, get_cut_idx=False, isLegacy = isLegacy)
        
        cut_name = plot_dict['cut']
        
        df = utils.get_bkg_cutflow_df(bkg_histos, 'cutflow_cts', isLegacy = isLegacy).iloc[:-1]
        
        df = df[list_cut_name[list_cut_index.index(cut_name)]]
        
        processes = df.index[df != 0].to_list()
    # if process is given as a list, i.e. ['DY', 'W+jets'], plot only these processes in the list; otherwise, plot all as default
    
    bkg={}
    bkg[plot_dict['variable']] = {process:bkg_histos[process][plot_dict['variable']][{"samp":sum}] for process in processes}
    
    # sort the histograms by the entries and stack
    for process in processes:
        entries = {process: bkg[plot_dict['variable']][process].sum().value for process in processes}
    
    sorted_entries = dict(sorted(entries.items(), key=lambda x:x[1], reverse = False))

    # histogram
    # add histos to stack after rebinning and range setting
    for idx, process in enumerate(sorted_entries.keys()):
        bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][plot_dict['cut'],::style_dict['rebin']]

        # set x range manually
        if style_dict['xlim'] != None:
            xlim = style_dict['xlim']
            xbin_range = np.where((bkg[plot_dict['variable']][process].axes.edges[0] > xlim[0]) & (bkg[plot_dict['variable']][process].axes.edges[0] < xlim[1]))[0]
            bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][ int(xbin_range[0])-1:int(xbin_range[-1]+1) ]
            
        if idx == 0:
            bkg_stack = bkg[plot_dict['variable']][process]
        else:
            bkg_stack += bkg[plot_dict['variable']][process]
    
    # x and y labels
    if style_dict['xlabel'] != None:
        ax.set_xlabel(style_dict['xlabel'])

    if style_dict['ylabel'] != None:
        ax.set_ylabel(style_dict['ylabel'])
    else:
        binwidth = bkg_stack.axes.widths[0][0]
        
        if style_dict['doDensity']:
            ax.set_ylabel(f'A.U./{binwidth:.3f}')
        else:
            ax.set_ylabel(f'Events/{binwidth:.3f}')

    # x,y scale
    if style_dict['doLogx']:
        ax.set_xscale('log')
    if style_dict['doLogy']:
        ax.set_yscale('log')  
    
    # Plot
    hep.histplot(bkg_stack, yerr=style_dict['doYerr'], density=style_dict['doDensity'], ax=ax, histtype='step', flow=style_dict['flow'], label = style_dict['label'])
    
    # legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

def plot_bkg_2D(bkg_histos, plot_dict, style_dict, isLegacy=False, processes = 'all'):
    """
    Example:

    
    plot_dict = {
        'variable': 'sel_vtx_vx_vs_vy',
        'cut': 'cut9',
        'year': 2018
    }
    
    style_2d_dict = {
        'fig': fig,
        'ax': ax,
        'xrebin': 1j,
        'yrebin': 1j,
        'xlim': None,     # if None, the default will show up; otherwise give as a list, i.e. [0, 10]  
        'ylim': None,     # if None, the default will show up; otherwise give as a list, i.e. [0, 10]
        'doLogy': False, 
        'doLogx': False,
        'doLogz': True,
        'xlabel': r"$v_{x}$ [cm]",   # if None, the default will show up; otherwise give as a string, i.e. 'Electron dxy'
        'ylabel': r"$v_{y}$ [cm]",   # if None, the default will show up; otherwise give as a string, i.e. 'Efficiency'
        'zlabel': 'Events',   
        'flow': None,     # overflow
        'doSave': False,
        'outDir': './plots/',
        'outName': f'background_lxy_mass.png'
    }

    """
    
    hep.cms.label('', data=False, year=plot_dict['year'])

    fig = style_dict['fig']
    ax = style_dict['ax']
    
    if isLegacy:
        return plot_bkg_2D_legacy(ax, bkg_histos, plot_dict, style_dict, processes = 'all', isLegacy = isLegacy)
    else:
        # if process is given as a list, i.e. ['DY', 'W+jets'], plot only these processes in the list; otherwise, plot all as default
        if processes == 'all':
            processes = list(set(utils.get_bkg_point_dict(bkg_histos).loc[:, 'Process']))
        
        subprocess = {process: [] for process in processes} # initialize the dictionary of bkg processes
        
        availSubCat = list(bkg_histos[plot_dict['variable']].axes['samp']) # get the list of subprocesses available for the histogram
        for samp in availSubCat:
            process = utils.get_bkg_point_dict(bkg_histos).loc[samp][0]            
            if process in processes:
                subprocess[process].append(samp) # fill out the bkg process list with the available subprocesses
            
        # Get histogram for each process
        bkg={}
        bkg[plot_dict['variable']] = {process:bkg_histos[plot_dict['variable']][{"samp":subprocess[process]}][{"samp": sum}] for process in processes}
        
        # sort the histograms by the entries and stack
        for process in processes:
            entries = {process: bkg[plot_dict['variable']][process].sum().value for process in processes}
        
        sorted_entries = dict(sorted(entries.items(), key=lambda x:x[1], reverse = False))

        # histogram
        # add histos to stack after rebinning and range setting
        for idx, process in enumerate(sorted_entries.keys()):
            bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][plot_dict['cut'], ::style_dict['xrebin'], ::style_dict['yrebin']]
            if style_dict['xlim'] != None:
                xlim = style_dict['xlim']
                xbin_range = np.where((bkg[plot_dict['variable']][process].axes.edges[0] > xlim[0]) & (bkg[plot_dict['variable']][process].axes.edges[0] < xlim[1]))[0]
                bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][ int(xbin_range[0])-1:int(xbin_range[-1]+1), : ]
            if style_dict['ylim'] != None:
                ylim = style_dict['ylim']
                ybin_range = np.where((bkg[plot_dict['variable']][process].axes.edges[1] > ylim[0]) & (bkg[plot_dict['variable']][process].axes.edges[1] < ylim[1]))[1]
                bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][ :, int(ybin_range[0]):int(ybin_range[-1]+1) ]
                
            if idx == 0:
                bkg_stack = bkg[plot_dict['variable']][process]
            else:
                bkg_stack += bkg[plot_dict['variable']][process]
    
        # x y labels
        if style_dict['xlabel'] != None:
            ax.set_xlabel(style_dict['xlabel'])
    
        if style_dict['ylabel'] != None:
            ax.set_ylabel(style_dict['ylabel'])
    
        # x,y scale
        if style_dict['doLogx']:
            ax.set_xscale('log')
        if style_dict['doLogy']:
            ax.set_yscale('log')
        
        # Plot
        if style_dict['doLogz']:
            hep.hist2dplot(bkg_stack, flow=style_dict['flow'], norm=mpl.colors.LogNorm(), ax=ax, cbarextend=True)
        else:
            hep.hist2dplot(bkg_stack, flow=style_dict['flow'], ax=ax, cbarextend=True)


        # z label
        if style_dict['zlabel'] != None:
            fig.get_axes()[-1].set_ylabel(style_dict['zlabel'])
        
        # legend
        #handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles[::-1], labels[::-1])

        if style_dict['doSave']:
            os.makedirs(style_dict['outDir'], exist_ok=True)
            plt.tight_layout()
            plt.savefig(f"{style_dict['outDir']}/{style_dict['outName']}")
            print(f"Saved: {style_dict['outDir']}/{style_dict['outName']}")


def plot_bkg_2D_legacy(ax, bkg_histos, plot_dict, style_dict, processes = 'all', isLegacy = True):  

    processes_list = processes
    
    if processes == 'all':
        #processes = bkg_histos.keys()

        list_cut_index = utils.get_bkg_list_of_cuts(bkg_histos, get_cut_idx=True, isLegacy = isLegacy)
        list_cut_name = utils.get_bkg_list_of_cuts(bkg_histos, get_cut_idx=False, isLegacy = isLegacy)
        
        cut_name = plot_dict['cut']
        
        df = utils.get_bkg_cutflow_df(bkg_histos, 'cutflow_cts', isLegacy = isLegacy).iloc[:-1]
        
        df = df[list_cut_name[list_cut_index.index(cut_name)]]
        
        processes = df.index[df != 0].to_list()
    # if process is given as a list, i.e. ['DY', 'W+jets'], plot only these processes in the list; otherwise, plot all as default
    
    bkg={}
    bkg[plot_dict['variable']] = {process:bkg_histos[process][plot_dict['variable']][{"samp":sum}] for process in processes}
    
    # sort the histograms by the entries and stack
    for process in processes:
        entries = {process: bkg[plot_dict['variable']][process].sum().value for process in processes}
    
    sorted_entries = dict(sorted(entries.items(), key=lambda x:x[1], reverse = False))

    # histogram
    # add histos to stack after rebinning and range setting
    for idx, process in enumerate(sorted_entries.keys()):
        bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][plot_dict['cut'], ::style_dict['xrebin'], ::style_dict['yrebin']]
    
        if style_dict['xlim'] != None:
            xlim = style_dict['xlim']
            xbin_range = np.where((bkg[plot_dict['variable']][process].axes.edges[0] > xlim[0]) & (bkg[plot_dict['variable']][process].axes.edges[0] < xlim[1]))[0]
            bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][ int(xbin_range[0])-1:int(xbin_range[-1]+1), : ]
        if style_dict['ylim'] != None:
            ylim = style_dict['ylim']
            ybin_range = np.where((bkg[plot_dict['variable']][process].axes.edges[1] > ylim[0]) & (bkg[plot_dict['variable']][process].axes.edges[1] < ylim[1]))[1]
            bkg[plot_dict['variable']][process] = bkg[plot_dict['variable']][process][ :, int(ybin_range[0]):int(ybin_range[-1]+1) ]
            
        if idx == 0:
            bkg_stack = bkg[plot_dict['variable']][process]
        else:
            bkg_stack += bkg[plot_dict['variable']][process]

    # x and y labels
    if style_dict['xlabel'] != None:
        ax.set_xlabel(style_dict['xlabel'])

    if style_dict['ylabel'] != None:
        ax.set_ylabel(style_dict['ylabel'])

    # x,y scale
    if style_dict['doLogx']:
        ax.set_xscale('log')
    if style_dict['doLogy']:
        ax.set_yscale('log')
    
    # Plot
    if style_dict['doLogz']:
        hep.hist2dplot(bkg_stack, flow=style_dict['flow'], norm=mpl.colors.LogNorm(), ax=ax)
    else:
        hep.hist2dplot(bkg_stack, flow=style_dict['flow'], ax=ax)
    
    # legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

def plot_data_1d(data_histo, plot_dict, style_dict):
        
    #hep.cms.label('', data=False, year=plot_dict['year'])

    fig = style_dict['fig']
    ax = style_dict['ax']

    # Get list of data
    runs = list(data_histo['cutflow_cts'].keys())

    for idx, run in enumerate(runs):
        try:
            if idx == 0:
                histo = data_histo[plot_dict['variable']][{"samp":run, "cut": plot_dict['cut']}]
            else:
                histo += data_histo[plot_dict['variable']][{"samp":run, "cut": plot_dict['cut']}]
        except:
            print('No run')

    # rebinning
    histo = histo[::style_dict['rebin']]

    # set x range manually
    if style_dict['xlim'] != None:
        xlim = style_dict['xlim']
        xbin_range = np.where((histo.axes.edges[0] > xlim[0]) & (histo.axes.edges[0] < xlim[1]))[0]
        histo = histo[ int(xbin_range[0])-1:int(xbin_range[-1]+1) ]

    # x and y labels
    if style_dict['xlabel'] != None:
        ax.set_xlabel(style_dict['xlabel'])

    if style_dict['ylabel'] != None:
        ax.set_ylabel(style_dict['ylabel'])
    else:    
        binwidth = histo.axes.widths[0][0]
        if style_dict['doDensity']:
            ax.set_ylabel(f'A.U./{binwidth:.3f}')
        else:
            ax.set_ylabel(f'Events/{binwidth:.3f}')

    # x,y scale
    if style_dict['doLogx']:
        ax.set_xscale('log')
    if style_dict['doLogy']:
        ax.set_yscale('log')

    # Plot
    hep.histplot(histo, yerr=style_dict['doYerr'], density=style_dict['doDensity'], ax=ax, histtype='errorbar', flow=style_dict['flow'], label = style_dict['label'], color='black')

    # legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])


def get_data_histo_2D(data_histo, plot_dict, style_dict):
    fig = style_dict['fig']
    ax = style_dict['ax']
    
    #hep.cms.label('', data=True, year=plot_dict['year'])
    
    # Get list of data
    runs = list(data_histo['cutflow_cts'].keys())

    for idx, run in enumerate(runs):
        try:
            if idx == 0:
                histo = data_histo[plot_dict['variable']][{"samp":run, "cut": plot_dict['cut']}]
            else:
                histo += data_histo[plot_dict['variable']][{"samp":run, "cut": plot_dict['cut']}]
        except:
            print('No run')

    # set x range manually
    if style_dict['xlim'] != None:
        xlim = style_dict['xlim']
        xbin_range = np.where((histo.axes.edges[0] > xlim[0]) & (histo.axes.edges[0] < xlim[1]))[0]
        histo = histo[ int(xbin_range[0])-1:int(xbin_range[-1]+1), : ]
    if style_dict['ylim'] != None:
        ylim = style_dict['ylim']
        ybin_range = np.where((histo.axes.edges[1] > ylim[0]) & (histo.axes.edges[1] < ylim[1]))[1]
        histo = histo[ :, int(ybin_range[0]):int(ybin_range[-1]+1) ]

    return histo


def plot_data_2D(data_histo, plot_dict, style_dict):
    """
    Example:

    """

    fig = style_dict['fig']
    ax = style_dict['ax']
    
    #hep.cms.label('', data=True, year=plot_dict['year'])
    hep.cms.label('', data=False, llabel='Private Work', rlabel='')
    
    # Get list of data
    runs = list(data_histo['cutflow_cts'].keys())

    for idx, run in enumerate(runs):
        if idx == 0:
            histo = data_histo[plot_dict['variable']][{"samp":run, "cut": plot_dict['cut']}]
        else:
            histo += data_histo[plot_dict['variable']][{"samp":run, "cut": plot_dict['cut']}]

    # rebinning
    histo = histo[::style_dict['xrebin'],::style_dict['yrebin']]

    # set x range manually
    if style_dict['xlim'] != None:
        xlim = style_dict['xlim']
        xbin_range = np.where((histo.axes.edges[0] > xlim[0]) & (histo.axes.edges[0] < xlim[1]))[0]
        histo = histo[ int(xbin_range[0])-1:int(xbin_range[-1]+1), : ]
    if style_dict['ylim'] != None:
        ylim = style_dict['ylim']
        ybin_range = np.where((histo.axes.edges[1] > ylim[0]) & (histo.axes.edges[1] < ylim[1]))[1]
        histo = histo[ :, int(ybin_range[0]):int(ybin_range[-1]+1) ]
    
    # x and y labels
    if style_dict['xlabel'] != None:
        ax.set_xlabel(style_dict['xlabel'])
    if style_dict['ylabel'] != None:
        ax.set_ylabel(style_dict['ylabel'])

    # x,y scale
    if style_dict['doLogx']:
        ax.set_xscale('log')
    if style_dict['doLogy']:
        ax.set_yscale('log')
    
    # Plot
    if style_dict['doLogz']:
        hep.hist2dplot(histo, flow=style_dict['flow'], norm=mpl.colors.LogNorm(), ax=ax, cbarextend=True)
    else:
        hep.hist2dplot(histo, flow=style_dict['flow'], ax=ax, cbarextend=True)

    # z label
    if style_dict['zlabel'] != None:
        fig.get_axes()[-1].set_ylabel(style_dict['zlabel'])
    
    if style_dict['doSave']:
        os.makedirs(style_dict['outDir'], exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{style_dict['outDir']}/{style_dict['outName']}")
        print(f"Saved: {style_dict['outDir']}/{style_dict['outName']}")

def plot_data_MC_ratio(data_histo, bkg_histo, plot_dict, style_dict):
    """
    Plot data and background MC
    """
    
    fig = style_dict['fig']
    ax = style_dict['ax']
    
    plot_bkg_1d(bkg_histo, plot_dict, style_dict, processes = 'all')
    plot_bkg_1d_stacked_errbar(bkg_histo, plot_dict, style_dict, processes = 'all')
    plot_data_1d(data_histo, plot_dict, style_dict)

    """
    Ratio plot
    """
    # Calculate ratio
    hist_bkg = get_bkg_histo_stacked_1d(bkg_histo, plot_dict, style_dict, processes = 'all')
    hist_data = get_data_histo_1d(data_histo, plot_dict, style_dict)

    ratio = hist_data.values()/hist_bkg.values()
    ratio[np.isnan(ratio)] = 0
    ratio[ratio == 0] = np.inf

    # Add axis for ratio
    ratio_length = (ax.get_position().y1 - ax.get_position().y0) / 3
    
    ax_ratio = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - ratio_length * 1.2, \
                             ax.get_position().x1 - ax.get_position().x0, ratio_length]) 

    ax.get_shared_x_axes().join(ax, ax_ratio)
    ax.xaxis.set_ticklabels([])

    ax_ratio.set_xlabel(ax.get_xlabel())
    ax.set_xlabel('')
    ax_ratio.set_ylabel('Data/MC')
    ax_ratio.set_ylim([0,2.5])

    xbin_centers = hist_bkg.axes.edges[0][:-1] + hist_bkg.axes.widths[0]/2
    
    # data error bar
    data_err = np.sqrt(hist_data.values())/hist_data.values()
    ax_ratio.errorbar(xbin_centers, ratio, yerr=data_err, fmt='o', color='black')
    #ax_ratio.plot(xbin_centers, ratio, 'o', color='black')

    # bkg Error bars
    bkg_err = np.sqrt(hist_bkg.values())/hist_bkg.values()
    bkg_err[np.isnan(bkg_err)] = 0

    y_upper = np.ones(len(bkg_err)) + bkg_err
    y_lower = np.ones(len(bkg_err)) - bkg_err

    error_band_args = { 
        #"edges": (range(len(ratio)+1) * binwidth) + xmin, "facecolor": "none", "linewidth": 0.5,
        "edges": hist_bkg.axes.edges[0], 
        "facecolor": "none", "linewidth": 0.5,
        "alpha": .5, "color": "grey", "hatch": "///"
    }
    ax_ratio.stairs(y_upper, baseline=y_lower, **error_band_args)
    ax_ratio.stairs(y_upper, baseline=y_lower, **error_band_args)
    
    ax_ratio.axhline(y=1, color='black', linestyle='--', linewidth=0.8)

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    
    if style_dict['doSave']:
        os.makedirs(style_dict['outDir'], exist_ok=True)
        plt.savefig(f"{style_dict['outDir']}/{style_dict['outName']}", bbox_inches='tight', bbox_extra_artists=[ax_ratio], pad_inches=0.3)
        print(f"Saved: {style_dict['outDir']}/{style_dict['outName']}")

def plot_samples_sigBkg(loader_sig,loader_bkg,hname,selection,samples,labels,outName,outD,
                 xlabel=None,ylabel=None,title=None,xlim=None,ylim=None,
                 ncol_leg=1,figsize=None,logy=False,logx=False,rebin=1j,
                 save=True,show=False,density=False,yerr=True,histtype='errorbar',heplabel="Private Work",
                 legend_loc='best',legend_fontsize=12,label_fontsize=16,bkgOnly=False,sigOnly=False):
    if not figsize:
        figsize=(8,6)
    plt.figure(figsize=figsize)
    if not bkgOnly:
        # load signal histograms
        h_sig = loader_sig.load(hname)[selection]
        histos_sig = [h_sig[{"samp":s}][::rebin] for s in samples]
        labels_sig = labels
        colors_sig = ['k','g','b','c']
        if len(labels_sig) < len(colors_sig):
            colors_sig = colors_sig[:len(labels_sig)]
    if not sigOnly:
        # load bkg histograms
        h_bkg = loader_bkg.load(hname)[selection]
        labels_bkg = loader_bkg.cats
        histos_bkg = []
        counts_bkg = []
        colors_bkg = []
        for cat in labels_bkg:
            trueSel = getPresentSamples(h_bkg,loader_bkg.catSamps[cat])
            hsel = h_bkg[{"samp":trueSel}][{"samp":sum}]
            if hsel.sum(flow=True).value == 0:
                continue
            histos_bkg.append(hsel[::rebin])
            colors_bkg.append(bkg_cmap[cat])
            counts_bkg.append(hsel.sum(flow=True).value)
        histos_bkg = [h for h,_ in sorted(zip(histos_bkg,counts_bkg),key=lambda p: p[1],reverse=True)]
        colors_bkg = [c for c,_ in sorted(zip(colors_bkg,counts_bkg),key=lambda p: p[1],reverse=True)]
        labels_bkg = [l for l,_ in sorted(zip(labels_bkg,counts_bkg),key=lambda p: p[1],reverse=True)]
    # plot histograms
    if not sigOnly:
        hep.histplot(histos_bkg,label=labels_bkg,density=density,yerr=yerr,stack=True,histtype='fill',color=colors_bkg)
    if not bkgOnly:
        hep.histplot(histos_sig,label=labels_sig,density=density,yerr=yerr,color=colors_sig,histtype=histtype,lw=2)
    if xlabel:
        plt.xlabel(xlabel,fontsize=label_fontsize)
    if ylabel:
        plt.ylabel(ylabel,fontsize=label_fontsize)
    else:
        if density:
            plt.ylabel("A.U.",fontsize=label_fontsize)
        else:
            plt.ylabel("Events",fontsize=label_fontsize)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if title:
        plt.title(title)
    if logy:
        plt.yscale('log')
    if logx:
        plt.xscale('log')
    plt.legend(ncol=ncol_leg,fontsize=legend_fontsize,loc=legend_loc)
    hep.cms.text(heplabel)
    #hep.cms.label("Simulation", data=False, year=2018)
    plt.tight_layout()
    if save:
        os.makedirs(outD,exist_ok=True)
        if density:
            plt.savefig(f"{outD}/{outName}_density.pdf")
        else:
            plt.savefig(f"{outD}/{outName}.pdf")
    if not show:
        plt.close()

def make_cdf_summary_sigVsBkg(dfs,loader_sig,loader_bkg,hname_sig,hname_bkg,selection,outName,outD,right=True,
                 xlabel=None,ylabel=None,title=None,xlim=None,ylim=None,
                 ncol_leg=1,figsize=None,logy=False,logx=False,alpha=1,
                 save=True,show=False,legend_loc='best',category=False,bkgOnly=False,sigOnly=False):
    if not figsize:
        figsize=(8,6)
    fig,axes = plt.subplots(1,1,figsize=figsize)
    handles = []
    if not bkgOnly:
        hsig = loader_sig.load(hname_sig)[selection]
        for s in dfs.name:
            hsamp = hsig[{"samp":s}]
            if xlim:
                edges = xlim
            else:
                edges = hsamp.axes[0].edges[:-1] if right else hsamp.axes[0].edges[1:]
            x,eff_real = makeCDF(hsamp,edges[0],edges[-1],right=right,category=category)
            plt.plot(x,eff_real,color='green',alpha=0.5,lw=2)
        handles.append(Line2D([],[],lw=2,color='green',label="Signals"))
    if not sigOnly:
        hbkg = loader_bkg.load(hname_bkg)[selection]
        for bkg_cat in loader_bkg.cats:
            trueSel = getPresentSamples(hbkg,loader_bkg.catSamps[bkg_cat])
            hsamp = hbkg[{"samp":trueSel}][{"samp":sum}]
            x,eff_fake = makeCDF(hsamp,edges[0],edges[-1],right=right,category=category)
            h, = plt.plot(x,eff_fake,alpha=1,lw=2,label=bkg_cat,color=bkg_cmap[bkg_cat])
            handles.append(h)
    plt.legend(handles=handles,loc=legend_loc,ncol=ncol_leg)
    if xlabel:
        plt.xlabel(xlabel)
    if right:
        plt.ylabel("Cumulative Distribution (Right)")
    else:
        plt.ylabel("Cumulative Distribution (Left)")
    if ylim:
        plt.ylim(ylim)
    if title:
        plt.title(title)
    if logy:
        plt.yscale('log')
    if logx:
        plt.xscale('log')
    plt.grid()
    plt.tight_layout()
    if save:
        os.makedirs(outD,exist_ok=True)
        plt.savefig(f"{outD}/{outName}.pdf")
    if not show:
        plt.close()

def make_Nminus1_sigVsBkg(dfs,loader_sig,loader_bkg,hname_sig,hname_bkg,selection,outName,outD,right=True,
                 xlabel=None,ylabel=None,title=None,xlim=None,ylim=None,
                 ncol_leg=1,figsize=None,logy=False,logx=False,alpha=1,
                 save=True,show=False,legend_loc='best',category=False):
    if not figsize:
        figsize=(8,6)
    fig,axes = plt.subplots(1,1,figsize=figsize)
    hsig = loader_sig.load(hname_sig)[selection]
    hbkg = loader_bkg.load(hname_bkg)[selection]
    
    # get bkg yields as a function of cut
    bkg_yields = []
    for bkg_cat in loader_bkg.cats:
        trueSel = getPresentSamples(hbkg,loader_bkg.catSamps[bkg_cat])
        hsamp = hbkg[{"samp":trueSel}][{"samp":sum}]
        if xlim:
            edges = xlim
        else:
            edges = hsamp.axes[0].edges[:-1] if right else hsamp.axes[0].edges[1:]
        x,nbkg = makeCDF(hsamp,edges[0],edges[-1],right=right,category=category,nevents=True)
        bkg_yields.append(nbkg)
    tot_bkg = sum(bkg_yields)
    
    for s in dfs.name:
        hsamp = hsig[{"samp":s}]
        if xlim:
            edges = xlim
        else:
            edges = hsamp.axes[0].edges[:-1] if right else hsamp.axes[0].edges[1:]
        x,sig_yields = makeCDF(hsamp,edges[0],edges[-1],right=right,category=category,nevents=True)
        plt.plot(x,sig_yields/tot_bkg,color='green',alpha=0.5,lw=2)
        
    handles = [Line2D([],[],lw=2,color='green',label="Signals")]
    plt.legend(handles=handles,loc=legend_loc,ncol=ncol_leg)
    if xlabel:
        plt.xlabel(xlabel)
    if right:
        plt.ylabel(r"$S/\sqrt{B}$ (Right)")
    else:
        plt.ylabel(r"$S/\sqrt{B}$ (Left)")
    if ylim:
        plt.ylim(ylim)
    if title:
        plt.title(title)
    if logy:
        plt.yscale('log')
    if logx:
        plt.xscale('log')
    plt.grid()
    plt.tight_layout()
    if save:
        os.makedirs(outD,exist_ok=True)
        plt.savefig(f"{outD}/{outName}.pdf")
    if not show:
        plt.close()

def getBkgComposition(loader_bkg,hname,selection):
    h_bkg = loader_bkg.load(hname)[selection]
    labels_bkg = loader_bkg.cats
    histos_bkg = []
    counts_bkg = []
    for cat in labels_bkg:
        trueSel = getPresentSamples(h_bkg,loader_bkg.catSamps[cat])
        histos_bkg.append(h_bkg[{"samp":trueSel}][{"samp":sum}])
        counts_bkg.append(h_bkg[{"samp":trueSel}][{"samp":sum}].value)
    print("Background Composition is:")
    for ct, label in zip(counts_bkg,labels_bkg):
        print(f"\t{label} : {ct:.4f} ({100*ct/np.sum(counts_bkg):.4f}%)")

def summedBkgCutflow(loader_bkg,cfname,cut):
    labels_bkg = loader_bkg.cats
    cf = loader_bkg.load(cfname)
    output = {cat:0 for cat in labels_bkg}
    for key,value in cf.items():
        cat = key.split("_")[2]
        output[cat] += value[cut]
    return output
