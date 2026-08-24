from __future__ import with_statement
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea import processor
import uproot
import awkward as ak
#import vector
#vector.register_awkward()
import numba as nb
import awkward.numba
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
import importlib
import pandas as pd
from XRootD import client
NanoAODSchema.warn_missing_crossrefs = False
import xgboost as xgb
import correctionlib

# Helper functions
def deltaPhi(v1,v2):
    # copy of the ROOT RVec DeltaPhi function
    # see here https://root.cern/doc/master/RVec_8hxx_source.html#l02742
    M_PI = 3.14159265358979323846264338328
    dPhi = np.fmod(v1-v2,2*M_PI)
    under = ak.values_astype(dPhi < -1*M_PI,np.float32)
    over = ak.values_astype(dPhi > M_PI,np.float32)
    fine = ak.values_astype((dPhi <= M_PI) & (dPhi >= -1*M_PI),np.float32)
    output = fine*dPhi + under*(dPhi + 2.0*M_PI) + over*(dPhi - 2.0*M_PI)
    return output

def deltaR(eta1,phi1,eta2,phi2):
    return np.sqrt((eta1-eta2)**2 + deltaPhi(phi1,phi2)**2)

@nb.njit()
def deltaPhiSingle(phi1,phi2):
    M_PI = 3.14159265358979323846264338328
    dPhi = np.fmod(phi1-phi2,2*M_PI)
    if dPhi < -1*M_PI:
        dPhi += 2*M_PI
    elif dPhi > M_PI:
        dPhi -= 2*M_PI
    return dPhi

@nb.njit()
def deltaRSingle(eta1,phi1,eta2,phi2):
    return np.sqrt((eta1-eta2)**2 + deltaPhiSingle(phi1,phi2)**2)

def _numbaSafe(x):
    """Detach a coffea NanoEvents array before handing it to a numba kernel.

    Two compounding leaks are triggered by passing a coffea array into an njit
    kernel, and both are closed here:

    1. awkward-1 builds the *name* of the numba ArrayViewType from
       repr(behavior) (awkward/_connect/_numba/__init__.py:77-78), and coffea
       stamps behavior["__events_factory__"] = self
       (coffea/nanoevents/factory.py:387). NanoEventsFactory has no __repr__, so
       that name embeds a fresh heap address every chunk. numba compares types by
       name, so it sees a brand-new type each chunk, compiles a brand-new
       specialization, and caches it forever on the module-level dispatcher. The
       cached type holds behavior -> factory -> UprootSourceMapping -> the open
       uproot file and its array_cache. Everything stays reachable, which is why
       gc.collect() reclaims nothing.
    2. ak.packed() materializes lazy VirtualArrays. awkward's numba bridge
       Py_IncRef's any layout it is handed
       (awkward/_connect/_numba/layout.py:2502,2565,2574) with no matching
       Py_DecRef, so a lazy layout is refcount-leaked outright, keeping that
       chunk's uproot source alive invisibly to gc.

    ak.Array(...) drops the behavior; ak.packed(...) removes the lazy nodes.
    Stripping behavior alone leaves ~40 MB/chunk on the table.

    Any future njit call site that bypasses runJitOutput reintroduces both leaks.
    """
    if isinstance(x, ak.Array):
        return ak.Array(ak.packed(ak.Array(x.layout), highlevel=False))
    return x

def runJitOutput(func,*args):
    b = ak.ArrayBuilder()
    func(b,*[_numbaSafe(x) for x in args])
    out = b.snapshot()
    del b
    return out

# Routines to produce additional variables (e.g. best vertex, good electrons, etc) for cuts
def electronJetSeparation(events):
    """
    Function to compute angular separations between electrons and jets
    """
    j0_eta = events.PFJet.eta[:,0]
    j0_phi = events.PFJet.phi[:,0]
    j1_eta = ak.fill_none(ak.pad_none(events.PFJet.eta,2),-999)[:,1]
    j1_phi = ak.fill_none(ak.pad_none(events.PFJet.phi,2),-999)[:,1]

    # Defining some extra branches for electron/low-pT electrons
    events["Electron","dRj0"] = np.sqrt(deltaPhi(events.Electron.phi,j0_phi)**2 + (events.Electron.eta-j0_eta)**2)
    events["Electron","dPhij0"] = np.abs(deltaPhi(events.Electron.phi,j0_phi))
    events["Electron","dRj1"] = np.sqrt(deltaPhi(events.Electron.phi,j1_phi)**2 + (events.Electron.eta-j1_eta)**2)
    events["Electron","dPhij1"] = ak.where(j1_phi != -999, ak.fill_none(ak.firsts(np.abs(deltaPhi(events.Electron.phi,j1_phi))),999), 999)
    events["Electron","mindRj"] = np.minimum(events.Electron.dRj0,events.Electron.dRj1)
    events["Electron","mindPhiJ"] = np.minimum(events.Electron.dPhij0,events.Electron.dPhij1)
    
    events["LptElectron","dRj0"] = np.sqrt(deltaPhi(events.LptElectron.phi,j0_phi)**2 + (events.LptElectron.eta-j0_eta)**2)
    events["LptElectron","dPhij0"] = np.abs(deltaPhi(events.LptElectron.phi,j0_phi))
    events["LptElectron","dRj1"] = np.sqrt(deltaPhi(events.LptElectron.phi,j1_phi)**2 + (events.LptElectron.eta-j1_eta)**2)
    events["LptElectron","dPhij1"] = ak.where(j1_phi != -999, ak.fill_none(ak.firsts(np.abs(deltaPhi(events.LptElectron.phi,j1_phi))),999), 999)
    events["LptElectron","mindRj"] = np.minimum(events.LptElectron.dRj0,events.LptElectron.dRj1)
    events["LptElectron","mindPhiJ"] = np.minimum(events.LptElectron.dPhij0,events.LptElectron.dPhij1)

def electronID(events,info):
    """
    Function to tag electrons as good/bad according to cuts we define
    """
    eles = events.Electron
    lpt_eles = events.LptElectron
    
    # define branches with the same name to store relevant MVA ID score for electrons/low-pT electrons
    events['Electron','IDscore'] = ak.zeros_like(eles.pt) # 'loose' mva score for regular electrons (just a dummy value)
    events['LptElectron','IDscore'] = lpt_eles.ID # use the actual MVA score for low-pT electrons

    # lowpT ID
    lpt_ele_kinematic_cut = (lpt_eles.pt > 1) & (np.abs(lpt_eles.eta) < 2.4)
    lpt_ele_id_cut = lpt_eles.pt > 0 # dummy always True
    events["LptElectron","passID"] = lpt_ele_kinematic_cut & lpt_ele_id_cut & (lpt_eles.mindRj > 0.4)
    events["LptElectron","passIDBasic"] = lpt_ele_kinematic_cut & lpt_ele_id_cut
    
    # regular ID
    """if info['type'] == 'signal':
        ele_kinematic_cut = (eles.pt > 1) & (np.abs(eles.eta) < 2.4)
        ele_id_cut = eles.IDcutLoose==1
        getLptMatchInfoForReg(events)
        events["Electron","passIDTemp"] = ele_kinematic_cut & ele_id_cut & (eles.mindRj > 0.4)
        events["Electron","passIDBasicTemp"] = ele_kinematic_cut & ele_id_cut
        
        events["Electron","passEitherID"] = ak.values_astype(events.Electron.passIDTemp | events.Electron.lptMatchPassID,bool)
        events["Electron","passEitherIDBasic"] = ak.values_astype(events.Electron.passIDBasicTemp | events.Electron.lptMatchPassIDBasic,bool)
        events["LptElectron","passEitherID"] = ak.values_astype(events.LptElectron.passID,bool)
        events["LptElectron","passEitherIDBasic"] = ak.values_astype(events.LptElectron.passIDBasic,bool)

        events["Electron","passID"] = events.Electron.passEitherID
        events["Electron","passIDBasic"] = events.Electron.passEitherIDBasic
    else:
        ele_kinematic_cut = (eles.pt > 1) & (np.abs(eles.eta) < 2.4)
        ele_id_cut = eles.IDcutLoose==1
        events["Electron","passID"] = ele_kinematic_cut & ele_id_cut & (eles.mindRj > 0.4)
        events["Electron","passIDBasic"] = ele_kinematic_cut & ele_id_cut"""
    ele_kinematic_cut = (eles.pt > 1) & (np.abs(eles.eta) < 2.4)
    #ele_id_cut = eles.IDcutLoose == 1
    ele_id_cut = eles.IDmva90 == 1
    events["Electron","passID"] = ele_kinematic_cut & ele_id_cut & (eles.mindRj > 0.4)
    events["Electron","passIDBasic"] = ele_kinematic_cut & ele_id_cut
    
# def jetBtag(events,year):
#     year = str(year)
#     loose,med,tight = getBtagWPs(year)
#     events["PFJet","passLooseID"] = events.PFJet.bTag > loose
#     events["PFJet","passMedID"] = events.PFJet.bTag > med
#     events["PFJet","passTightID"] = events.PFJet.bTag > tight

def jetBtag(events,year):
    loose,med,tight = getBtagWPs(year)
    events["PFJet","passLooseID"] = events.PFJet.bTag > loose
    events["PFJet","passMedID"] = events.PFJet.bTag > med
    events["PFJet","passTightID"] = events.PFJet.bTag > tight


def getBtagWPs(year):
    if year == 2022:
       loose,med,tight = 0.0583, 0.3086, 0.7183
    if year == 2018:
        loose,med,tight = 0.0490, 0.2783, 0.7100
    if year == 2017:
        loose,med,tight = 0.0532, 0.3040, 0.7476
    if year == 2016:
        loose,med,tight = 0.0480, 0.2489, 0.6377
    if year == "2016APV":
         loose,med,tight = 0.0508, 0.2598, 0.6502
    return loose,med,tight

def electronIsoConePtSum(events):
    all_eles = ak.concatenate((events.Electron,events.LptElectron),axis=1)
    lpt_eles = events.LptElectron
    reg_eles = events.Electron
    events["LptElectron","elePtSumIsoCone"] = runJitOutput(sumPtInCone,lpt_eles.eta,lpt_eles.phi,all_eles.eta,all_eles.phi,all_eles.pt)
    events["Electron","elePtSumIsoCone"] = runJitOutput(sumPtInCone,reg_eles.eta,reg_eles.phi,all_eles.eta,all_eles.phi,all_eles.pt)

def vtxElectronConnection(events):
    """
    Function to associate electrons to vertices, so they can be easily accessed as vtx.e1 and vtx.e2
    """
    all_eles = ak.concatenate((events.Electron,events.LptElectron),axis=1)
    n_reg = ak.count(events.Electron.pt,axis=1)
    vtx = events.vtx

    # shitty fix for weird bug when there's just 1 event with 1 vertex
    if len(events) == 1 and len(vtx.pt) == 1:
        events["vtx","e1"] = events.LptElectron[vtx.e1_idx] if vtx.e1_typ[0][0]=="L" else events.Electron[vtx.e1_idx]
        events["vtx","e2"] = events.LptElectron[vtx.e2_idx] if vtx.e2_typ[0][0]=="L" else events.Electron[vtx.e2_idx]
    else:
        isLow1 = ak.values_astype(vtx.e1_typ=="L","int")
        isLow2 = ak.values_astype(vtx.e2_typ=="L","int")
    
        vtx_e1_flatIdx = events.vtx.e1_idx + isLow1*n_reg
        vtx_e2_flatIdx = events.vtx.e2_idx + isLow2*n_reg

        events["vtx","e1"] = all_eles[vtx_e1_flatIdx]
        events["vtx","e2"] = all_eles[vtx_e2_flatIdx]

    events["vtx","e1","refit_dxy"] = vtx.e1_refit_dxy
    events["vtx","e2","refit_dxy"] = vtx.e2_refit_dxy

def getDeltaR(e1, e2):
     dR_sq = (e1.eta - e2.eta)**2 + (e1.phi - e2.phi)**2
     print(dR_sq)
 
     return np.sqrt(dR_sq)
 
def getInvariantMass(pts, etas, phis):
     mass_sq = (2 * pts[0] * pts[1]) * (np.cosh(etas[0] - etas[1]) - np.cos(phis[0] - phis[1]))
 
     return np.sqrt(mass_sq)
 
def getBtagInfo(events):
     isTrueB = events.PFJet.truth == 5
     isTrueC = events.PFJet.truth == 4
     isTrueLight = events.PFJet.truth == 0
 
     taggedMed = events.PFJet.passMedID
 
     isBtagB = isTrueB & taggedMed
     isCtagB = isTrueC & taggedMed
     isLightTagB = isTrueLight & taggedMed
 
     btagNum = ak.count_nonzero(isBtagB)
     btagDenom = ak.count_nonzero(isTrueB)
     btagEff = btagNum / btagDenom
 
     #print(f'b tagging eff: {btagNum}/{btagDenom}={btagEff}')
 
     events["PFJet","btagNum"] = btagNum
     events["PFJet","btagDenom"] = btagDenom

def defineGoodVertices(events,version='none',ele_id='dR'):
    # Selecting electrons that pass basic pT and eta cuts
    if ele_id == 'basic':
        IDcut = events.vtx.e1.passIDBasic & events.vtx.e2.passIDBasic
    if ele_id == 'dR':
        IDcut = events.vtx.e1.passID & events.vtx.e2.passID
    ossf = events.vtx.sign == -1
    chi2 = events.vtx.reduced_chi2 < 5
    mass = events.vtx.refit_m < 20
    eleDphi = events.vtx.eleDphi < 2
    mindxy = events.vtx.min_dxy > 0.01
    mindxy_refit = np.minimum(np.abs(events.vtx.e1.refit_dxy), np.abs(events.vtx.e2.refit_dxy)) > 0.001
    mindxy_refit_tight = np.minimum(np.abs(events.vtx.e1.refit_dxy), np.abs(events.vtx.e2.refit_dxy)) > 0.01
    mindxyLoose = events.vtx.min_dxy > 0.005
    maxMiniIso = np.maximum(events.vtx.e1.miniRelIsoEleCorr,events.vtx.e2.miniRelIsoEleCorr) < 0.9 
    passConvVeto = events.vtx.e1.conversionVeto & events.vtx.e2.conversionVeto
    mass_lo = events.vtx.m > 0.325
    mass_lo_refit = events.vtx.refit_m > 0.1
    logdxydz = np.minimum(np.log10(np.abs(events.vtx.e1.dxy/events.vtx.e1.dz)), np.log10(np.abs(events.vtx.e2.dxy/events.vtx.e2.dz))) > -1.25
    logdxydz_loose = np.minimum(np.log10(np.abs(events.vtx.e1.dxy/events.vtx.e1.dz)), np.log10(np.abs(events.vtx.e2.dxy/events.vtx.e2.dz))) > -2
    if version == 'none':
        events['vtx','isGood'] = ak.values_astype(ak.ones_like(events.vtx.m),bool)
    if version == 'default':
        events["vtx","isGood"] = IDcut & ossf # base definition
    if version == 'v1':
        events["vtx","isGood"] = IDcut & ossf & chi2 # v1 definition
    if version == 'v2':
        events["vtx","isGood"] = IDcut & ossf & chi2 & mass # v2 definition
    if version == 'v3':
        events["vtx","isGood"] = IDcut & ossf & chi2 & mass & mindxy # v3 definition
    if version == 'v4':
        events["vtx","isGood"] = IDcut & ossf & chi2 & mass & mindxy & maxMiniIso # v4 definition
    if version == 'v5':
        events['vtx','isGood'] = IDcut & ossf & chi2 & mass & mindxy & maxMiniIso & passConvVeto # v5 definition
    if version == 'v6':
        events['vtx','isGood'] = IDcut & ossf & chi2 & mass & mindxy & maxMiniIso & passConvVeto & mass_lo # v6 definition
    if version == 'v7':
        events['vtx','isGood'] = IDcut & ossf & chi2 & mindxyLoose & maxMiniIso & passConvVeto # v7 definition
    if version == 'v8':
        events["vtx","isGood"] = IDcut & ossf & chi2 & mindxyLoose & maxMiniIso & passConvVeto & mass_lo_refit # v8 definition
    if version == "v9":
        events["vtx","isGood"] = IDcut & ossf & chi2 & maxMiniIso & passConvVeto & mass_lo_refit & mindxy_refit
    if version == "v10":
        events["vtx","isGood"] = IDcut & chi2 & maxMiniIso & passConvVeto & mass_lo_refit & mindxy_refit
    if version == 'v11':
         events["vtx","isGood"] = IDcut & chi2 & maxMiniIso & passConvVeto & mass_lo_refit & mindxy_refit # remove OSSF

    if version == 'v11_LL':
        vtx_type = ((events.vtx.e1_typ == 'L') & (events.vtx.e2_typ == 'L'))
        events["vtx","isGood"] = IDcut & chi2 & maxMiniIso & passConvVeto & mass_lo_refit & mindxy_refit & vtx_type # remove OSSF
    if version == 'v11_RR':
        vtx_type = ((events.vtx.e1_typ == 'R') & (events.vtx.e2_typ == 'R'))
        events["vtx","isGood"] = IDcut & chi2 & maxMiniIso & passConvVeto & mass_lo_refit & mindxy_refit & vtx_type # remove OSSF
    if version == 'v11_RL':
        vtx_type = ((events.vtx.e1_typ == 'R') & (events.vtx.e2_typ == 'L'))|((events.vtx.e1_typ == 'L') & (events.vtx.e2_typ == 'R'))
        events["vtx","isGood"] = IDcut & chi2 & maxMiniIso & passConvVeto & mass_lo_refit & mindxy_refit # remove OSSF
        
    if version == 'v12':
         events["vtx","isGood"] = IDcut & chi2 & maxMiniIso & passConvVeto & mass_lo_refit & mindxy_refit_tight # tighten mindxy
    if version == 'v13':
         events["vtx","isGood"] = IDcut & chi2 & maxMiniIso & passConvVeto & mass_lo_refit & mindxy_refit_tight & logdxydz # add logdxydz
    if version == 'v14':
         events["vtx","isGood"] = IDcut & chi2 & maxMiniIso & passConvVeto & mass_lo_refit & mindxy_refit & logdxydz # loosen mindxy

    if version == 'v15':
         events["vtx","isGood"] = IDcut & chi2 & maxMiniIso & passConvVeto & mass_lo_refit & mindxy_refit & logdxydz_loose # loosen mindxy
    
    if version == 'v40': # For SF studies
         vtx_type = ((events.vtx.e1_typ == 'L') & (events.vtx.e2_typ == 'L'))
         #vtx_type = ((events.vtx.e1_typ == 'R') & (events.vtx.e2_typ == 'R'))
         #vtx_type = ((events.vtx.e1_typ == 'R') & (events.vtx.e2_typ == 'L'))|((events.vtx.e1_typ == 'L') & (events.vtx.e2_typ == 'R'))
         events['vtx','isGood'] =  IDcut & chi2 & maxMiniIso & mindxy_refit & vtx_type

    events.__setitem__("good_vtx",events.vtx[events.vtx.isGood])
    events.__setitem__("nGoodVtx",ak.count(events.good_vtx.vxy,axis=1))

def selectBestVertex(events):
    sel_vtx = ak.flatten(events.good_vtx[ak.argmin(events.good_vtx.reduced_chi2,axis=1,keepdims=True)])
    events.__setitem__("sel_vtx",sel_vtx)

def computeExtraVariables(events,info):
    events['Electron','mindRj'] = ak.fill_none(ak.min(events.Electron.dRJets,axis=-1),999)
    events['Electron','mindPhiJ'] = ak.fill_none(ak.min(np.abs(events.Electron.dPhiJets),axis=-1),999)
    events['LptElectron','mindRj'] = ak.fill_none(ak.min(events.LptElectron.dRJets,axis=-1),999)
    events['LptElectron','mindPhiJ'] = ak.fill_none(ak.min(np.abs(events.LptElectron.dPhiJets),axis=-1),999)
    events['vtx','mindRj'] = ak.fill_none(ak.min(events.vtx.dRJets,axis=-1),999)
    events['vtx','mindPhiJ'] = ak.fill_none(ak.min(np.abs(events.vtx.dPhiJets),axis=-1),999)
    events['vtx','corrMinDxy'] = ak.fill_none(np.minimum(np.abs(events.vtx.e1_refit_dxy),np.abs(events.vtx.e2_refit_dxy)),999)
    projectLxy(events)
    electronID(events,info) # electron kinematic/ID definition
    jetBtag(events,info['year'])
    if info['type'] == "signal":
        events['GenJetMETdPhi'] = np.abs(deltaPhi(events.GenJet.phi[:,0],events.GenMET.phi))
        events['GenEle','dr'] = events.genEE.dr
        events['GenPos','dr'] = events.genEE.dr
        if "vxy" not in events.genEE.fields:
            events['genEE','vxy'] = events.GenEle.vxy
        genElectronKinematicBins(events)
        genMatchRecoQuantities(events)
    # associate electrons to vertices after all electron-related stuff has been computed
    vtxElectronConnection(events) # associate electrons to vertices
    events['vtx','min_dxy'] = np.minimum(np.abs(events.vtx.e1.dxy),np.abs(events.vtx.e2.dxy))
    events['vtx','eleDphi'] = np.abs(deltaPhi(events.vtx.e1.phi,events.vtx.e2.phi))
    events['vtx','vxy_fromPV'] = ((events.vtx.vx-events.PV.x)**2+(events.vtx.vy-events.PV.y)**2)
    if info['type'] == 'signal':
        projectGenLxy(events)
        events['GenJetMETdPhi'] = np.abs(deltaPhi(events.GenJet.phi[:,0],events.GenMET.phi))
    return events

def miscExtraVariablesSignal(events):
    # record if signal is reconstructed
    events["signalReco"] = events.GenEle.matched & events.GenPos.matched

    # separation between jets and gen e+/e-
    genj_phi_pt30 = ak.fill_none(ak.pad_none(events.GenJet.phi[events.GenJet.pt>30],1),999)
    genj_eta_pt30 = ak.fill_none(ak.pad_none(events.GenJet.eta[events.GenJet.pt>30],1),999)
    events["GenEle","mindRj"] = ak.min(np.sqrt(deltaPhi(events.PFJet.phi,events.GenEle.phi)**2 + (events.PFJet.eta-events.GenEle.eta)**2),axis=1)
    events["GenEle","mindPhiJ"] = ak.min(np.abs(deltaPhi(events.PFJet.phi,events.GenEle.phi)),axis=1)
    events["GenEle","mindRjGen"] = ak.min(np.sqrt(deltaPhi(genj_phi_pt30,events.GenEle.phi)**2 + (genj_eta_pt30-events.GenEle.eta)**2),axis=1)
    events["GenEle","mindPhiJGen"] = ak.min(ak.where(genj_phi_pt30 != 999,np.abs(deltaPhi(genj_phi_pt30,events.GenEle.phi)),999),axis=1)
    
    events["GenPos","mindRj"] = ak.min(np.sqrt(deltaPhi(events.PFJet.phi,events.GenPos.phi)**2 + (events.PFJet.eta-events.GenPos.eta)**2),axis=1)
    events["GenPos","mindPhiJ"] = ak.min(np.abs(deltaPhi(events.PFJet.phi,events.GenPos.phi)),axis=1)
    events["GenPos","mindRjGen"] = ak.min(np.sqrt(deltaPhi(genj_phi_pt30,events.GenPos.phi)**2 + (genj_eta_pt30-events.GenPos.eta)**2),axis=1)
    events["GenPos","mindPhiJGen"] = ak.min(ak.where(genj_phi_pt30 != 999,np.abs(deltaPhi(genj_phi_pt30,events.GenPos.phi)),999),axis=1)

    events["genEE","mindRj"] = ak.min(np.sqrt(deltaPhi(events.PFJet.phi,events.genEE.phi)**2 + (events.PFJet.eta-events.genEE.eta)**2),axis=1)
    events["genEE","mindPhiJ"] = ak.min(np.abs(deltaPhi(events.PFJet.phi,events.genEE.phi)),axis=1)
    events["genEE","mindRjGen"] = ak.min(np.sqrt(deltaPhi(genj_phi_pt30,events.genEE.phi)**2 + (genj_eta_pt30-events.genEE.eta)**2),axis=1)
    events["genEE","mindPhiJGen"] = ak.min(ak.where(genj_phi_pt30 != 999,np.abs(deltaPhi(genj_phi_pt30,events.genEE.phi)),999),axis=1)

    # collinear angle & projectedLxy, calculating ctau with gen info
    projectLxy(events)
    #calculateCtau(events)

def projectGenLxy(events):
    ee_vx_fromPV = events.GenEle.vx - events.GenPart.vx[events.GenPart.ID == 1000023]
    ee_vy_fromPV = events.GenEle.vy - events.GenPart.vy[events.GenPart.ID == 1000023]
    ee_px = events.GenEle.px + events.GenPos.px
    ee_py = events.GenEle.py + events.GenPos.py

    ee_dotprod = ee_vx_fromPV * ee_px + ee_vy_fromPV * ee_py
    ee_vxy_fromPV = np.sqrt(ee_vx_fromPV*ee_vx_fromPV + ee_vy_fromPV*ee_vy_fromPV)
    ee_pxy = np.sqrt(ee_px*ee_px + ee_py*ee_py)
    cos = ee_dotprod / (ee_vxy_fromPV*ee_pxy)
    #print(cos)

    events["vtx","gen_cos_collinear_fromPV"] = ak.flatten(cos)

def projectLxyFromPV(events):
    vtx = events.vtx

    # (vx, vy)
    vx_fromPV = vtx.vx - events.PV.x
    vy_fromPV = vtx.vy - events.PV.y

    vxy_mag_fromPV = np.sqrt(vx_fromPV * vx_fromPV + vy_fromPV * vy_fromPV)

    # (px, py) Refitted
    px_refit = vtx.refit_pt * np.cos(vtx.refit_phi)
    py_refit = vtx.refit_pt * np.sin(vtx.refit_phi)

    vx_fromPV_px_refit = vx_fromPV * px_refit
    vy_fromPV_py_refit = vy_fromPV * py_refit

    dotprod_fromPV_refit = vx_fromPV_px_refit + vy_fromPV_py_refit

    pxy_mag_refit = np.sqrt(px_refit * px_refit + py_refit * py_refit)
    cos_fromPV_refit = dotprod_fromPV_refit / (vxy_mag_fromPV * pxy_mag_refit)
    
    # (px, py)
    vx_fromPV_px = vx_fromPV * vtx.px
    vy_fromPV_py = vy_fromPV * vtx.py
    
    dotprod_fromPV = vx_fromPV_px + vy_fromPV_py

    pxy_mag = np.sqrt(vtx.px * vtx.px + vtx.py * vtx.py)
    cos_fromPV = dotprod_fromPV / (vxy_mag_fromPV * pxy_mag)
    
    #mask_dotprod_neg = dotprod < 0

    events["vtx","cos_collinear_fromPV"] = cos_fromPV
    events["vtx","cos_collinear_fromPV_refit"] = cos_fromPV_refit

def projectLxy(events):
    vtx = events.vtx

    vxpx = vtx.vx * vtx.px
    vypy = vtx.vy * vtx.py
    dotprod = vxpx + vypy
    vxy_mag = np.sqrt(vtx.vx * vtx.vx + vtx.vy * vtx.vy)
    pxy_mag = np.sqrt(vtx.px * vtx.px + vtx.py * vtx.py)
    cos = dotprod / (vxy_mag * pxy_mag)
    events["vtx","cos_collinear"] = cos
    events["vtx","projectedLxy"] =  vtx.vxy * cos

    projectLxyFromPV(events) # cancel out beamspot mis-modeling effect

def calculateCtau(events):
    # chi2
    mask_chi2 = events.GenPart.ID == 1000023
    vx_chi2 = ak.flatten(events.GenPart.vx[mask_chi2])
    vy_chi2 = ak.flatten(events.GenPart.vy[mask_chi2])
    vz_chi2 = ak.flatten(events.GenPart.vz[mask_chi2])

    gamma_chi2 = ak.flatten(events.GenPart.e[mask_chi2])/ak.flatten(events.GenPart.mass[mask_chi2])

    # gen e-
    mask_genele = events.GenPart.ID == 11
    vx_genele = ak.flatten(events.GenPart.vx[mask_genele])
    vy_genele = ak.flatten(events.GenPart.vy[mask_genele])
    vz_genele = ak.flatten(events.GenPart.vz[mask_genele])

    #gamma_genele = ak.flatten(events.GenPart.e[mask_genele])/ak.flatten(events.GenPart.mass[mask_genele])

    # gen e+
    mask_genpos = events.GenPart.ID == -11
    vx_genpos = ak.flatten(events.GenPart.vx[mask_genpos])
    vy_genpos = ak.flatten(events.GenPart.vy[mask_genpos])
    vz_genpos = ak.flatten(events.GenPart.vz[mask_genpos])

    #gamma_genpos = ak.flatten(events.GenPart.e[mask_genpos])/ak.flatten(events.GenPart.mass[mask_genpos])

    # decay length of chi2 in the lab frame
    decaylength_chi2 = 10 * np.sqrt( (vx_genele-vx_chi2)**2 + (vy_genele-vy_chi2)**2 + (vz_genele-vz_chi2)**2 ) # in [mm]

    ctau_chi2 = decaylength_chi2 / gamma_chi2

    events.__setitem__("ctau",ctau_chi2)
    
@nb.njit
def ele_dRcategory(b,ele_arr):
    nEvents = len(ele_arr)
    for n in range(nEvents):
        if ele_arr[n] <= 0.1:
            b.append(0)
        elif ele_arr[n] > 0.1 and ele_arr[n] <= 0.5:
            b.append(1)
        else:
            b.append(2)

@nb.njit
def ele_vxyCategory(b,ele_arr):
    nEvents = len(ele_arr)
    for n in range(nEvents):
        if ele_arr[n] <= 1:
            b.append(0)
        elif ele_arr[n] > 1 and ele_arr[n] <= 5:
            b.append(1)
        elif ele_arr[n] > 5 and ele_arr[n] <= 10:
            b.append(2)
        elif ele_arr[n] > 10 and ele_arr[n] <= 15:
            b.append(3)
        else:
            b.append(4)

@nb.njit
def ele_ptCategory(b,ele_arr):
    nEvents = len(ele_arr)
    for n in range(nEvents):
        if ele_arr[n] <= 5:
            b.append(0)
        elif ele_arr[n] > 5 and ele_arr[n] <= 10:
            b.append(1)
        elif ele_arr[n] > 10 and ele_arr[n] <= 20:
            b.append(2)
        else:
            b.append(3)

def genElectronKinematicBins(events):
    dR_map = ak.Array(["0to0p1","0p1to0p5","0p5toInf"])
    vxy_map = ak.Array(["0to1","1to5","5to10","10to15","15toInf"])
    pt_map = ak.Array(["0to5","5to10","10to20","20toInf"])

    events["GenEle","dRbin"] = dR_map[runJitOutput(ele_dRcategory,events.GenEle.dr)]
    events["GenEle","vxyBin"] = vxy_map[runJitOutput(ele_vxyCategory,events.GenEle.vxy)]
    events["GenEle","ptBin"] = pt_map[runJitOutput(ele_ptCategory,events.GenEle.pt)]

    events["GenPos","dRbin"] = dR_map[runJitOutput(ele_dRcategory,events.GenPos.dr)]
    events["GenPos","vxyBin"] = vxy_map[runJitOutput(ele_vxyCategory,events.GenPos.vxy)]
    events["GenPos","ptBin"] = pt_map[runJitOutput(ele_ptCategory,events.GenPos.pt)]

@nb.njit
def regEle_lptMatchID(b,ele_lptMatchIdx,lpt_ele_id):
    nEvents = len(ele_lptMatchIdx)
    for n in range(nEvents):
        b.begin_list()
        n_ele = len(ele_lptMatchIdx[n])
        for ie in range(n_ele):
            if ele_lptMatchIdx[n][ie] >= 0:
                b.append(int(lpt_ele_id[n][ele_lptMatchIdx[n][ie]]))
            else:
                b.append(0)
        b.end_list()

def getLptMatchInfoForReg(events):
    events["Electron","lptMatchPassID"] = runJitOutput(regEle_lptMatchID,events.Electron.lptMatchIdx,events.LptElectron.passID)
    events["Electron","lptMatchPassIDBasic"] = runJitOutput(regEle_lptMatchID,events.Electron.lptMatchIdx,events.LptElectron.passIDBasic)

@nb.njit
def getGenMatchRecoQuantity(b,genMatchTyp,genMatchIdx,RegQuantity,LptQuantity,unMatched_value):
    nEvents = len(genMatchTyp)
    for n in range(nEvents):
        typ = genMatchTyp[n]
        idx = genMatchIdx[n]
        if typ == "None":
            b.append(unMatched_value)
        elif typ == "R":
            b.append(int(RegQuantity[n][idx]))
        elif typ == "L":
            b.append(int(LptQuantity[n][idx]))

def genMatchRecoQuantities(events):
    #events["GenEle","matchPassID"] = runJitOutput(getGenMatchRecoQuantity,events.GenEle.matchType.to_numpy(),events.GenEle.matchIdxLocal.to_numpy(),events.Electron.passEitherID,events.LptElectron.passID,0)
    #events["GenEle","matchPassIDBasic"] = runJitOutput(getGenMatchRecoQuantity,events.GenEle.matchType.to_numpy(),events.GenEle.matchIdxLocal.to_numpy(),events.Electron.passEitherIDBasic,events.LptElectron.passIDBasic,0)

    #events["GenPos","matchPassID"] = runJitOutput(getGenMatchRecoQuantity,events.GenPos.matchType.to_numpy(),events.GenPos.matchIdxLocal.to_numpy(),events.Electron.passEitherID,events.LptElectron.passID,0)
    #events["GenPos","matchPassIDBasic"] = runJitOutput(getGenMatchRecoQuantity,events.GenPos.matchType.to_numpy(),events.GenPos.matchIdxLocal.to_numpy(),events.Electron.passEitherIDBasic,events.LptElectron.passIDBasic,0)

    events["GenEle","matchPassID"] = runJitOutput(getGenMatchRecoQuantity,events.GenEle.matchType.to_numpy(),events.GenEle.matchIdxLocal.to_numpy(),events.Electron.passID,events.LptElectron.passID,0)
    events["GenEle","matchPassIDBasic"] = runJitOutput(getGenMatchRecoQuantity,events.GenEle.matchType.to_numpy(),events.GenEle.matchIdxLocal.to_numpy(),events.Electron.passIDBasic,events.LptElectron.passIDBasic,0)

    events["GenPos","matchPassID"] = runJitOutput(getGenMatchRecoQuantity,events.GenPos.matchType.to_numpy(),events.GenPos.matchIdxLocal.to_numpy(),events.Electron.passID,events.LptElectron.passID,0)
    events["GenPos","matchPassIDBasic"] = runJitOutput(getGenMatchRecoQuantity,events.GenPos.matchType.to_numpy(),events.GenPos.matchIdxLocal.to_numpy(),events.Electron.passIDBasic,events.LptElectron.passIDBasic,0)

@nb.njit
def getGenMatchedVtxQuantity(b,vtxIsMatched,vtxQuantity,unMatched_value):
    nEvents = len(vtxIsMatched)
    for n in range(nEvents):
        nvtx = len(vtxIsMatched[n])
        hasMatch = False
        for nv in range(nvtx):
            if vtxIsMatched[n][nv]:
                hasMatch = True
                b.append(vtxQuantity[n][nv])
                break
        if not hasMatch:
            b.append(unMatched_value)


def genMatchExtraVtxVariables(events):
    #events["vtx","bothElePassID"] = events.vtx.e1.passEitherID & events.vtx.e2.passEitherID
    #events["vtx","bothElePassIDBasic"] = events.vtx.e1.passEitherIDBasic & events.vtx.e2.passEitherIDBasic
    #events["matchedVtxPassID"] = runJitOutput(getGenMatchedVtxQuantity,ak.Array(ak.values_astype(events.vtx.isMatched,int).tolist()),events.vtx.bothElePassID,0)
    #events["matchedVtxPassIDBasic"] = runJitOutput(getGenMatchedVtxQuantity,ak.Array(ak.values_astype(events.vtx.isMatched,int).tolist()),events.vtx.bothElePassIDBasic,0)
    events["vtx","bothElePassID"] = events.vtx.e1.passID & events.vtx.e2.passID
    events["vtx","bothElePassIDBasic"] = events.vtx.e1.passIDBasic & events.vtx.e2.passIDBasic
    events["matchedVtxPassID"] = runJitOutput(getGenMatchedVtxQuantity,ak.Array(ak.values_astype(events.vtx.isMatched,int).tolist()),events.vtx.bothElePassID,0)
    events["matchedVtxPassIDBasic"] = runJitOutput(getGenMatchedVtxQuantity,ak.Array(ak.values_astype(events.vtx.isMatched,int).tolist()),events.vtx.bothElePassIDBasic,0)

def matchedVertexElectron(events,i):
    vtx = events.sel_vtx

    e1match = events.sel_vtx.e1_isMatched == True
    e2match = events.sel_vtx.e2_isMatched == True

    matched = ak.where(e1match,-1,0)
    matched = ak.where(e2match,1,matched)

    return matched

@nb.njit()
def sumPtInCone(b,e1_eta,e1_phi,e2_eta,e2_phi,e2_pt):
    nEvents = len(e1_eta)
    for n in range(nEvents):
        b.begin_list() # start list to record pt sums for each electron in e1
        n_e1 = len(e1_eta[n])
        n_e2 = len(e2_eta[n])
        for i in range(n_e1):
            pt_sum_cone = 0
            for j in range(n_e2):
                if e1_eta[n][i] == e2_eta[n][j] and e1_phi[n][i] == e2_phi[n][j]:
                    # don't include in sum if it's the same electron (same pt and phi)
                    continue
                dR = deltaRSingle(e1_eta[n][i],e1_phi[n][i],e2_eta[n][j],e2_phi[n][j])
                if dR < 0.3:
                    pt_sum_cone += e2_pt[n][j]
            b.real(pt_sum_cone)
        b.end_list() # end list

def checkHEMjet(events):
    # If there is a jet in the event that satisfies the following requirements, reject the event
    # if ((jet.pt() > 30) && (jet.eta() > -3.0) && (jet.eta() < -1.3) && (jet.phi() > -1.57) && (jet.phi() < -0.87))
    
    filter_jet_pt = events.PFJet.pt > 30
    filter_jet_etaBug = (events.PFJet.eta > -3.0) & (events.PFJet.eta < -1.4) # with bug in eta boundary -- what is implemented in the ntuplizer
    filter_jet_eta = (events.PFJet.eta > -3.0) & (events.PFJet.eta < -1.3) 
    filter_jet_phi = (events.PFJet.phi > -1.57) & (events.PFJet.phi < -0.87)

    isHEM = filter_jet_pt & filter_jet_eta
    isHEM = isHEM & filter_jet_phi

    hasHEMjet = ak.flatten(ak.sum(isHEM, axis=-1, keepdims=True))

    events['hasHEMjet'] = hasHEMjet

    isHEMbug = filter_jet_pt & filter_jet_etaBug
    isHEMbug = isHEMbug & filter_jet_phi

    hasHEMjetBug = ak.flatten(ak.sum(isHEMbug, axis=-1, keepdims=True))
    
    events['hasHEMjetBug'] = hasHEMjetBug

def checkHEMelectron(events):
    # PF 
    filter_elecPF_pt = events.Electron.pt > 5
    filter_elecPF_eta = (events.Electron.eta > -3.0) & (events.Electron.eta < -1.3)
    filter_elecPF_phi = (events.Electron.phi > -1.57) & (events.Electron.phi < -0.87)

    isHEMelecPF = filter_elecPF_pt & filter_elecPF_eta
    isHEMelecPF = isHEMelecPF & filter_elecPF_phi

    hasHEMelecPF = ak.flatten(ak.sum(isHEMelecPF, axis=-1, keepdims=True))
    
    events['hasHEMelecPF'] = hasHEMelecPF

    # Low pT
    filter_elecLpt_pt = events.LptElectron.pt > 1
    filter_elecLpt_eta = (events.LptElectron.eta > -3.0) & (events.LptElectron.eta < -1.3)
    filter_elecLpt_phi = (events.LptElectron.phi > -1.57) & (events.LptElectron.phi < -0.87)

    isHEMelecLpt = filter_elecLpt_pt & filter_elecLpt_eta
    isHEMelecLpt = isHEMelecLpt & filter_elecLpt_phi

    hasHEMelecLpt = ak.flatten(ak.sum(isHEMelecLpt, axis=-1, keepdims=True))
    
    events['hasHEMelecLpt'] = hasHEMelecLpt


def makeBDTv1Inputs(events): # Legacy BDT; not used anymore 
    '''
    variables = ['lead_jet_pt','lead_jet_eta','minJetMETdPhi','jetMETdPhi',
                 'sel_vtx_sign', 'sel_vtx_chi2','sel_vtx_METdPhi','sel_vtx_m','sel_vtx_dR','sel_vtx_minDxy']
    '''

    lead_jet_pt_arr = events.PFJet.pt[:,0].to_numpy()
    lead_jet_eta_arr = events.PFJet.eta[:,0].to_numpy()
    minJetMETdPhi_arr = ak.min(np.abs(events.PFJet.METdPhi),axis=1).to_numpy()
    jetMETdPhi_arr = np.abs(events.PFJet.METdPhi[:,0]).to_numpy()
    sel_vtx_sign_arr = events.sel_vtx.sign.to_numpy()
    sel_vtx_chi2_arr = events.sel_vtx.reduced_chi2.to_numpy()
    sel_vtx_METdPhi_arr = np.abs(events.sel_vtx.METdPhi).to_numpy()
    sel_vtx_m_arr = events.sel_vtx.m.to_numpy()
    sel_vtx_dR_arr = events.sel_vtx.dR.to_numpy()
    sel_vtx_minDxy_arr = np.minimum(np.abs(events.sel_vtx.e1.dxy),np.abs(events.sel_vtx.e2.dxy)).to_numpy()
    
    input_arrs = (lead_jet_pt_arr, lead_jet_eta_arr, minJetMETdPhi_arr, jetMETdPhi_arr, \
                  sel_vtx_sign_arr, sel_vtx_chi2_arr, sel_vtx_METdPhi_arr, sel_vtx_m_arr, sel_vtx_dR_arr, sel_vtx_minDxy_arr)
    
    input = np.stack(input_arrs, axis=1)

    return input


def makeBDTv2Inputs(events): # Legacy BDT; not used anymore
    '''
    variables = ['sel_vtx_chi2','sel_vtx_METdPhi','sel_vtx_m','sel_vtx_dR','sel_vtx_minDxy','vxy_signif']
    '''

    #sel_vtx_sign_arr = events.sel_vtx.sign.to_numpy()
    sel_vtx_chi2_arr = events.sel_vtx.reduced_chi2.to_numpy()
    sel_vtx_METdPhi_arr = np.abs(events.sel_vtx.METdPhi).to_numpy()
    sel_vtx_m_arr = events.sel_vtx.m.to_numpy()
    sel_vtx_dR_arr = events.sel_vtx.dR.to_numpy()
    sel_vtx_minDxy_arr = np.minimum(np.abs(events.sel_vtx.e1.dxy),np.abs(events.sel_vtx.e2.dxy)).to_numpy()
    vxy_signif_arr = (events.sel_vtx.vxy/events.sel_vtx.sigmavxy).to_numpy()
    
    input_arrs = (sel_vtx_chi2_arr, sel_vtx_METdPhi_arr, sel_vtx_m_arr, sel_vtx_dR_arr, sel_vtx_minDxy_arr, vxy_signif_arr)
    
    input = np.stack(input_arrs, axis=1)

    return input

def makeBDTinputs(events): # Current BDT for SR vtx cut
    '''
    # BDT_10vars_comb11 (ROC-AUC, PR-AUC) = (0.9958, 0.9959)
    variables = ['sel_vtx_chi2','sel_vtx_METdPhi','sel_vtx_refit_m','sel_vtx_refit_dR','sel_vtx_corrMinDxy','vxy','vxy_signif',\
                 'sel_vtx_cos_collinear', 'sel_vtx_prod_eta', 'met_leadPt_ratio'
]
    '''

    mindxy = np.minimum(np.abs(events.sel_vtx.e1.dxy),np.abs(events.sel_vtx.e2.dxy))
    maxdxy = np.maximum(np.abs(events.sel_vtx.e1.dxy),np.abs(events.sel_vtx.e2.dxy))

    deltadxy = np.abs(np.abs(events.sel_vtx.e1.dxy) - np.abs(events.sel_vtx.e2.dxy))

    sel_vtx_chi2_arr = events.sel_vtx.reduced_chi2.to_numpy()
    sel_vtx_METdPhi_arr = np.abs(events.sel_vtx.METdPhi).to_numpy()
    sel_vtx_m_arr = events.sel_vtx.m.to_numpy()
    sel_vtx_refit_m_arr = events.sel_vtx.refit_m.to_numpy()
    sel_vtx_dR_arr = events.sel_vtx.dR.to_numpy()
    sel_vtx_refit_dR_arr = events.sel_vtx.refit_dR.to_numpy()
    sel_vtx_minDxy_arr = mindxy.to_numpy()
    sel_vtx_corrMinDxy_arr = events.sel_vtx.corrMinDxy.to_numpy()
    sel_vtx_vxy_arr = events.sel_vtx.vxy.to_numpy()
    vxy_signif_arr = (events.sel_vtx.vxy/events.sel_vtx.sigmavxy).to_numpy()
    cos_collinear_arr = events.sel_vtx.cos_collinear.to_numpy()
    sel_vtx_prod_eta_arr = (events.sel_vtx.e1.eta * events.sel_vtx.e2.eta).to_numpy()
    met_leadPt_ratio_arr = (events.PFMET.pt/events.PFJet.pt[:,0]).to_numpy()
    jetMETdPhi = np.abs(events.PFJet.METdPhi[:,0]).to_numpy()
    minJetMETdPhi = ak.min(np.abs(events.PFJet.METdPhi),axis=1).to_numpy()

    input_arrs = (sel_vtx_chi2_arr, sel_vtx_METdPhi_arr, sel_vtx_refit_m_arr, sel_vtx_refit_dR_arr, \
                  sel_vtx_corrMinDxy_arr, sel_vtx_vxy_arr, vxy_signif_arr, \
                  cos_collinear_arr, sel_vtx_prod_eta_arr, met_leadPt_ratio_arr, jetMETdPhi, minJetMETdPhi)

    input = np.stack(input_arrs, axis=1)

    return input


def makeBDTinputs_ABCD(events): # Current BDT for SR vtx cut
    '''
    # Comb 4
    variables = ['sel_vtx_chi2',
             'sel_vtx_METdPhi', # refit phi?
             'sel_vtx_m_refit',
             'sel_vtx_dR_refit',
             'sel_vtx_minDxy_refit',
             'vxy_corr',
             'sel_vtx_prod_eta', # refit eta?
             'met_leadPt_ratio',
            ]
    '''
    mindxy = np.minimum(np.abs(events.sel_vtx.e1.refit_dxy), np.abs(events.sel_vtx.e2.refit_dxy))
    sel_vtx_chi2_arr = events.sel_vtx.reduced_chi2.to_numpy()
    sel_vtx_METdPhi_arr = np.abs(events.sel_vtx.METdPhi).to_numpy()
    sel_vtx_m_arr = events.sel_vtx.refit_m.to_numpy()
    sel_vtx_dR_arr = events.sel_vtx.refit_dR.to_numpy()
    sel_vtx_minDxy_arr = mindxy.to_numpy()
    sel_vtx_vxy_arr = events.sel_vtx.vxy.to_numpy()
    vxy_signif_arr = (events.sel_vtx.vxy/events.sel_vtx.sigmavxy).to_numpy()
    cos_collinear_arr = events.sel_vtx.cos_collinear_fromPV_refit.to_numpy()
    sel_vtx_prod_eta_arr = (events.sel_vtx.e1.eta * events.sel_vtx.e2.eta).to_numpy()
    met_leadPt_ratio_arr = (events.PFMET.pt/events.PFJet.pt[:,0]).to_numpy()
    jetMETdPhi_arr = np.abs(events.PFJet.METdPhi[:,0]).to_numpy()
    minJetMETdPhi_arr = ak.min(np.abs(events.PFJet.METdPhi),axis=1).to_numpy()
    logdxydz_arr = np.minimum(np.log10(np.abs(events.sel_vtx.e1.dxy/events.sel_vtx.e1.dz)), np.log10(np.abs(events.sel_vtx.e2.dxy/events.sel_vtx.e2.dz)))

#    input_arrs = (sel_vtx_chi2_arr, sel_vtx_METdPhi_arr, sel_vtx_m_arr, sel_vtx_dR_arr, \
#                  sel_vtx_minDxy_arr, sel_vtx_vxy_arr, sel_vtx_prod_eta_arr, met_leadPt_ratio_arr
#                 )

#    input_arrs = (sel_vtx_chi2_arr, sel_vtx_METdPhi_arr, sel_vtx_m_arr, sel_vtx_dR_arr, \
#                  sel_vtx_minDxy_arr, sel_vtx_vxy_arr, sel_vtx_prod_eta_arr
#                 )

    input_arrs = (sel_vtx_chi2_arr, sel_vtx_METdPhi_arr, sel_vtx_m_arr, sel_vtx_dR_arr, \
                  sel_vtx_minDxy_arr, sel_vtx_vxy_arr, sel_vtx_prod_eta_arr, logdxydz_arr
                   )

    input = np.stack(input_arrs, axis=1)
    return input

def getBDTscore(arr, model):
    # load the pre-trained model
    trained_model = xgb.XGBRegressor()
    trained_model.load_model(model)

    # get BDT score
    score = trained_model.predict(arr)

    return score

def prepareBDT(events, model):
    if (len(events) != 0) and (model != None):
        input = makeBDTinputs_ABCD(events)
        score_BDT = getBDTscore(input, model)
        
        events['BDTScore'] = score_BDT

def getEventsSelVtxIsTruthMatched(events):
# for signal MC, return the events where selected vertex (lowest chi2) passes the truth-matching (gen-matching)
    e1_match = matchedVertexElectron(events,1)
    e2_match = matchedVertexElectron(events,2)
    events["sel_vtx","match"] = ak.sum(np.stack((e1_match, e2_match)), axis=0)   
    #events["sel_vtx","match"] = ak.values_astype(ak.where(e1_match*e2_match == -1,2,ak.where(np.abs(e1_match)+np.abs(e2_match) > 0,1,0)),np.int32)
 
    return events[events.sel_vtx.match == 2]

def getTrueVertex(events, evt_vtx):
    e1match = evt_vtx.e1_isMatched == True
    e2match = evt_vtx.e2_isMatched == True

    has_matched_vtx = (ak.sum(e1match & e2match, axis=-1, keepdims=True) != 0)
    idx_vtxMatch = ak.argmin(ak.where(e1match & e2match == True, 0, 1),axis=1,keepdims=True)

    return idx_vtxMatch, has_matched_vtx

def selectTrueVertex(events, evt_vtx): 
    '''
    Select vertex formed with the reco e+,e- that are closest to gen e+,e- (true vertex); instead of the vertex with the lowest chi2.
    dR(gen, reco) < 0.1 gen matching cut is not applied when doGenMatch is set to False.

    - To call this function, in analysisTools; replace the line of `routines.selectBestVertex(events)` like the following

    if info['type'] == "signal":
        events = routines.selectTrueVertex(events, events.good_vtx)           # select the true vertex

    - To run it without electron cuts applied to vertex electrons, add the following lines before `events = events[events.nGoodVtx > 0]` cut
    
    if info['type'] == "signal":
        events = routines.selectTrueVertex(events, events.vtx)           # select the true vertex

    '''
    # Sam edits 5/31/24: changing this to use the vtx_isMatched field that I added to the ntuples - simplifies this function
    # first slim down to events with a matched vertex
    cut_hasVtx = ak.any(evt_vtx.isMatched,axis=1)
    events = events[cut_hasVtx]
    evt_vtx = evt_vtx[cut_hasVtx]

    # next, select the matched vertex
    sel_vtx = ak.flatten(evt_vtx[evt_vtx.isMatched])
    events.__setitem__("sel_vtx",sel_vtx)

    return events

def selectTrueAndMinChi2Vtx(events, evt_vtx, doGenMatch=False): 
    '''
    Select vertex formed with the reco e+,e- that are closest to gen e+,e- (true vertex) that is also the lowest chi2 vertex.
    dR(gen, reco) < 0.1 gen matching cut is not applied when doGenMatch is set to False.

    - To call this function, in analysisTools; replace the line of `routines.selectBestVertex(events)` like the following

    if info['type'] == "signal":
        events = routines.selectTrueAndMinChi2Vertex(events, events.good_vtx, doGenMatch=False)      # select the true & min chi2 vertex

    - To run it without electron cuts applied to vertex electrons, add the following lines before `events = events[events.nGoodVtx > 0]` cut
    
    if info['type'] == "signal":
        events = routines.selectTrueAndMinChi2Vertex(events, events.vtx, doGenMatch=False)           # select the true vertex

    '''
    
    # Index of true vertex in each event; for events that have the true vertex
    idx_vtxMatch, has_matched_vtx = getTrueVertex(events, evt_vtx, doGenMatch=doGenMatch)
    idx_vtxMatch = idx_vtxMatch[has_matched_vtx]
    
    # Index of vertex with the lowest chi2 in each event; for events that have the true vertex
    idx_minChi2 = ak.argmin(evt_vtx.reduced_chi2,axis=1,keepdims=True)[has_matched_vtx]

    events = events[ak.flatten(idx_vtxMatch == idx_minChi2)] # reject the events where the true vtx is not the same as the min chi2 vtx

    return events


#############################################
########## Specialized routines #############
#############################################

# Must specify which ones you want to run in the histogram config (will need to run the ones that make the vars for your histograms)

def eleVtxGenMatching(events):
    # Electron collections
    ele = events.Electron
    lpt_ele = events.LptElectron
    cand_ele = events.EleCand
    gen_ele = events.GenEle
    gen_pos = events.GenPos

    # Electron gen-matches
    ele_matches = events.GenEleMatches
    pos_matches = events.GenPosMatches
    ele_match = events.GenEleMatch
    pos_match = events.GenPosMatch
    ele_matchC = events.GenEleMatchC
    pos_matchC = events.GenPosMatchC

    # Find unique matches with different collections / uniqueness schemes
    ematch, pmatch = findUniqueMatches(ele_matches,ele_match,pos_matches,pos_match)
    ematch11, pmatch11 = findUniqueMatches(ele_matches,ele_match,pos_matches,pos_match,allow11=True)
    ematchC, pmatchC = findUniqueMatches(ele_matches,ele_matchC,pos_matches,pos_matchC,useCands=True)
    ematchC11, pmatchC11 = findUniqueMatches(ele_matches,ele_matchC,pos_matches,pos_matchC,allow11=True,useCands=True)

    events.__setitem__("ematch",ematch)
    events.__setitem__("pmatch",pmatch)
    events.__setitem__("ematch11",ematch11)
    events.__setitem__("pmatch11",pmatch11)
    events.__setitem__("ematchC",ematchC)
    events.__setitem__("pmatchC",pmatchC)
    events.__setitem__("ematchC11",ematchC11)
    events.__setitem__("pmatchC11",pmatchC11)

    #####################################
    ####### VERTEX MATCHING STUFF #######
    #####################################

    # masks for different vertex match types, low/reg only
    noMatch_RL = (ematch.typ == 0) & (pmatch.typ == 0)
    matchRR_RL = (ematch.typ == 1) & (pmatch.typ == 1)
    matchLR_RL = ((ematch.typ == 1) & (pmatch.typ == 2)) | ((ematch.typ == 2) & (pmatch.typ == 1))
    matchLL_RL = (ematch.typ == 2) & (pmatch.typ == 2)
    ones = ak.ones_like(ematch.typ)
    zeros = ak.zeros_like(ematch.typ)
    matchType_RL = ak.where(noMatch_RL,zeros,-1*ones)
    matchType_RL = ak.where(matchRR_RL,ones,matchType_RL)
    matchType_RL = ak.where(matchLR_RL,2*ones,matchType_RL)
    matchType_RL = ak.where(matchLL_RL,3*ones,matchType_RL)

    events.__setitem__("eeMatchRR_RL",matchRR_RL)
    events.__setitem__("eeMatchLR_RL",matchLR_RL)
    events.__setitem__("eeMatchLL_RL",matchLL_RL)
    events.__setitem__("eeMatchType_RL",matchType_RL)

    # masks for different vertex match types, low/reg/cand
    noMatch_RLC = (ematchC.typ == 0) & (pmatchC.typ == 0)
    matchRR_RLC = (ematchC.typ == 1) & (pmatchC.typ == 1)
    matchRC_RLC = ((ematchC.typ == 1) & (pmatchC.typ == 3)) | ((ematchC.typ == 3) & (pmatchC.typ == 1))
    matchLR_RLC = ((ematchC.typ == 1) & (pmatchC.typ == 2)) | ((ematchC.typ == 2) & (pmatchC.typ == 1))
    matchLC_RLC = ((ematchC.typ == 2) & (pmatchC.typ == 3)) | ((ematchC.typ == 3) & (pmatchC.typ == 2))
    matchLL_RLC = (ematchC.typ == 2) & (pmatchC.typ == 2)
    matchCC_RLC = (ematchC.typ == 3) & (pmatchC.typ == 3)
    ones = ak.ones_like(ematchC.typ)
    zeros = ak.zeros_like(ematchC.typ)
    matchType_RLC = ak.where(noMatch_RLC,zeros,-1*ones)
    matchType_RLC = ak.where(matchRR_RLC,ones,matchType_RLC)
    matchType_RLC = ak.where(matchLR_RLC,2*ones,matchType_RLC)
    matchType_RLC = ak.where(matchLL_RLC,3*ones,matchType_RLC)
    matchType_RLC = ak.where(matchRC_RLC,4*ones,matchType_RLC)
    matchType_RLC = ak.where(matchLC_RLC,5*ones,matchType_RLC)
    matchType_RLC = ak.where(matchCC_RLC,6*ones,matchType_RLC)

    events.__setitem__("eeMatchRR_RLC",matchRR_RLC)
    events.__setitem__("eeMatchRC_RLC",matchRC_RLC)
    events.__setitem__("eeMatchLR_RLC",matchLR_RLC)
    events.__setitem__("eeMatchLC_RLC",matchLC_RLC)
    events.__setitem__("eeMatchLL_RLC",matchLL_RLC)
    events.__setitem__("eeMatchCC_RLC",matchCC_RLC)
    events.__setitem__("eeMatchType_RLC",matchType_RLC)

    #######################################
    ####### ELECTRON MATCHING STUFF #######
    #######################################
    # calculating other useful quantities
    nEmatch = ak.count(ele_matches[ele_matches.typ != 3].dr,axis=1)
    nPmatch = ak.count(pos_matches[pos_matches.typ != 3].dr,axis=1)
    nEmatchC = ak.count(ele_matches.dr,axis=1)
    nPmatchC = ak.count(pos_matches.dr,axis=1)

    events.__setitem__("nEmatch",nEmatch)
    events.__setitem__("nPmatch",nPmatch)
    events.__setitem__("nEmatchC",nEmatchC)
    events.__setitem__("nPmatchC",nPmatchC)

    # determining e+e- "match classes" for each event
    zeros = ak.zeros_like(ematch.typ) # generic vector of zeros for each event
    ones = ak.ones_like(ematch.typ) # generic vector fo ones for each event
    
    noMatch = (ematch11.typ == 0) & (pmatch11.typ == 0)
    oneMatch = ((ematch11.typ != 0) & (pmatch11.typ == 0)) | ((ematch11.typ == 0) & (pmatch11.typ != 0))
    degenMatch = (ematch11.typ != 0) & (pmatch11.typ != 0) & ((ematch11.typ == pmatch11.typ) & (ematch11.ind == pmatch11.ind))
    uniqMatch = (ematch11.typ != 0) & (pmatch11.typ != 0) & ((ematch11.typ != pmatch11.typ) | (ematch11.ind != pmatch11.ind))
    matchClass = ak.where(noMatch,zeros,-1*ones)
    matchClass = ak.where(oneMatch,ones,matchClass)
    matchClass = ak.where(degenMatch,2*ones,matchClass)
    matchClass = ak.where(uniqMatch,3*ones,matchClass)

    events.__setitem__("eleMatchClass",matchClass)

    noMatchC = (ematchC11.typ == 0) & (pmatchC11.typ == 0)
    oneMatchC = ((ematchC11.typ != 0) & (pmatchC11.typ == 0)) | ((ematchC11.typ == 0) & (pmatchC11.typ != 0))
    degenMatchC = (ematchC11.typ != 0) & (pmatchC11.typ != 0) & ((ematchC11.typ == pmatchC11.typ) & (ematchC11.ind == pmatchC11.ind))
    uniqMatchC = (ematchC11.typ != 0) & (pmatchC11.typ != 0) & ((ematchC11.typ != pmatchC11.typ) | (ematchC11.ind != pmatchC11.ind))
    matchClassC = ak.where(noMatchC,zeros,-1*ones)
    matchClassC = ak.where(oneMatchC,ones,matchClassC)
    matchClassC = ak.where(degenMatchC,2*ones,matchClassC)
    matchClassC = ak.where(uniqMatchC,3*ones,matchClassC)

    events.__setitem__("eleMatchClassC",matchClassC)
    
def genParticles(events):
    gen_eles = ak.concatenate([events.GenEle,events.GenPos])
    gen_ele = events.GenEle
    gen_pos = events.GenPos
    ee = events.genEE

    """histos['ele_kinematics'].fill(sample=samp,cut=cut,ele_type="Generator",
                        pt=gen_eles.pt,eta=gen_eles.eta,phi=gen_eles.phi)
    histos['gen_displacement'].fill(sample=samp,cut=cut,
                        vxy=gen_eles.vxy,
                        vz=gen_eles.vz)
    histos["gen_ee_kinematics"].fill(sample=samp,cut=cut,
                mass=ee.mass,
                dR=ee.dr)"""

def conversions(events,histos,samp,cut):
    eles = events.Electron
    lpt_eles = events.LptElectron
    convs = events.Conversion
    gen_eles = events.GenPart[np.abs(events.GenPart.ID) == 11]
    gen_ele = ak.flatten(events.GenPart[events.GenPart.ID == 11])
    gen_pos = ak.flatten(events.GenPart[events.GenPart.ID == -11])

    eles_dir = ak.zip(
        {
            "x":eles.eta,
            "y":eles.phi
        },
        with_name="TwoVector"
    )
    lpt_eles_dir = ak.zip(
        {
            "x":lpt_eles.eta,
            "y":lpt_eles.phi
        },
        with_name="TwoVector"
    )
    conv1_dir = ak.zip(
        {
            "x":convs.trk1_innerEta,
            "y":convs.trk1_innerPhi,
        },
        with_name="TwoVector"
    )
    conv2_dir = ak.zip(
        {
            "x":convs.trk2_innerEta,
            "y":convs.trk2_innerPhi,
        },
        with_name="TwoVector"
    )
    gen_ele_dir = ak.zip(
        {
            "x":gen_ele.eta,
            "y":gen_ele.phi,
        },
        with_name="TwoVector"
    )
    gen_pos_dir = ak.zip(
        {
            "x":gen_pos.eta,
            "y":gen_pos.phi,
        },
        with_name="TwoVector"
    )

    all_eles_dir = ak.concatenate([eles_dir,lpt_eles_dir,conv1_dir,conv2_dir],axis=1)
    all_eles_labels = ak.concatenate([ak.ones_like(eles.eta),2*ak.ones_like(lpt_eles.eta),
                                        3*ak.ones_like(convs.trk1_innerEta),3*ak.ones_like(convs.trk2_innerEta)],axis=1)

    dR_e = (all_eles_dir - gen_ele_dir).r
    dR_p = (all_eles_dir - gen_pos_dir).r
    nearest_e = ak.argmin(dR_e,axis=1,keepdims=True)
    nearest_p = ak.argmin(dR_p,axis=1,keepdims=True)
    mindR_e = ak.fill_none(ak.flatten(dR_e[nearest_e]),999)
    mindR_p = ak.fill_none(ak.flatten(dR_p[nearest_p]),999)
    type_e_nearest = ak.fill_none(ak.flatten(all_eles_labels[nearest_e]),0)
    type_p_nearest = ak.fill_none(ak.flatten(all_eles_labels[nearest_p]),0)
    
    match_e = mindR_e < 0.1
    match_p = mindR_p < 0.1
    type_e_nearest = type_e_nearest*ak.values_astype(match_e,int) # sets match type to 0 (no match) where dR < 0.1 criterion not satisfied
    type_p_nearest = type_p_nearest*ak.values_astype(match_p,int)
    histos['match_matrix'].fill(sample=samp,
                    e_matchType=type_e_nearest,
                    p_matchType=type_p_nearest)

    ## Conversion-specific studies
    d1e = (conv1_dir - gen_ele_dir).r
    d1p = (conv1_dir - gen_pos_dir).r
    d2e = (conv2_dir - gen_ele_dir).r
    d2p = (conv2_dir - gen_pos_dir).r
    ## Matching to most nearby gen object (possible for both conv eles matched to same object)
    d1e_min = ak.fill_none(ak.flatten(d1e[ak.argmin(d1e,axis=1,keepdims=True)]),999)
    d1p_min = ak.fill_none(ak.flatten(d1p[ak.argmin(d1p,axis=1,keepdims=True)]),999)
    d2e_min = ak.fill_none(ak.flatten(d2e[ak.argmin(d2e,axis=1,keepdims=True)]),999)
    d2p_min = ak.fill_none(ak.flatten(d2p[ak.argmin(d2p,axis=1,keepdims=True)]),999)
    matchType_conv1 = ak.values_astype(d1e_min<0.1,int) + 2*ak.values_astype(d1p_min<0.1,int) #0 if no match, 1 if match to e, 2 if match to p, 3 if match to both
    matchType_conv2 = ak.values_astype(d2e_min<0.1,int) + 2*ak.values_astype(d2p_min<0.1,int)
    fullUniqMatch = ((matchType_conv1 == 1) & (matchType_conv2 == 2)) | ((matchType_conv1 == 2) & (matchType_conv2 == 1))
    fullMatch = (matchType_conv1 > 0) & (matchType_conv2 > 0)
    anyMatch = (matchType_conv1 > 0) | (matchType_conv2 > 0)
    histos['conv_match_matrix'].fill(sample=samp,cut=cut,
                        match="all",
                        c1_matchType=matchType_conv1,
                        c2_matchType=matchType_conv2)
    histos['conv_match_matrix'].fill(sample=samp,cut=cut,
                        match="fullUnique",
                        c1_matchType=matchType_conv1[fullUniqMatch],
                        c2_matchType=matchType_conv2[fullUniqMatch])
    histos['conv_match_matrix'].fill(sample=samp,cut=cut,
                        match="full",
                        c1_matchType=matchType_conv1[fullMatch],
                        c2_matchType=matchType_conv2[fullMatch])
    histos['conv_match_matrix'].fill(sample=samp,cut=cut,
                        match="any",
                        c1_matchType=matchType_conv1[anyMatch],
                        c2_matchType=matchType_conv2[anyMatch])
    ## Attempting un-ambiguous matches - i.e. "best" matches
    ## If e.g. both conv electrons are best matched to the same object, but one is *better* matched and the 
    ## other could also be matched to the other gen object, just not as well
    both_c1 = (matchType_conv1 == 3)
    both_c2 = (matchType_conv2 == 3)
    # zeroing out the "both" matches
    uniqMatch_conv1 = matchType_conv1*ak.values_astype(~both_c1,int)
    uniqMatch_conv2 = matchType_conv2*ak.values_astype(~both_c2,int)
    # setting to 1 or 2 based on smaller dR, only where there was a "both" match
    uniqMatch_conv1 = uniqMatch_conv1 + ak.values_astype(both_c1,int)*(ak.values_astype(d1e_min < d1p_min,int) + 2*ak.values_astype(d1e_min > d1p_min,int))
    uniqMatch_conv2  = uniqMatch_conv2 + ak.values_astype(both_c2,int)*(ak.values_astype(d2e_min < d2p_min,int) + 2*ak.values_astype(d2e_min > d2p_min,int))
    # Disambiguate some "double" matches -- e.g. conv1 and conv2 both matched to same gen object
    c1c2_e = (uniqMatch_conv1 == uniqMatch_conv2) & (uniqMatch_conv1 == 1)
    c1c2_p = (uniqMatch_conv1 == uniqMatch_conv2) & (uniqMatch_conv1 == 2)
    # zeroing out the "double" matches
    oldMatch_conv1 = 1*uniqMatch_conv1
    oldMatch_conv2 = 1*uniqMatch_conv2
    uniqMatch_conv1 = uniqMatch_conv1*ak.values_astype(~c1c2_e,int)*ak.values_astype(~c1c2_p,int)
    uniqMatch_conv2 = uniqMatch_conv2*ak.values_astype(~c1c2_e,int)*ak.values_astype(~c1c2_p,int)
    # When c1 and c2 are matched to the same object, see if there's a distinct matching possible instead
    switch1_e2p = (c1c2_e) & (d1p_min < 0.1) & ( (d1e_min >= d2e_min) | (d2p_min > 0.1) )
    switch1_p2e = (c1c2_p) & (d1e_min < 0.1) & ( (d1p_min >= d2p_min) | (d2e_min > 0.1) )
    switch2_e2p = (c1c2_e) & (d2p_min < 0.1) & ( (d2e_min > d1e_min) | (d1p_min > 0.1) )
    switch2_p2e = (c1c2_p) & (d2e_min < 0.1) & ( (d2p_min > d1p_min) | (d1e_min > 0.1) )
    no_switch1 = (c1c2_e & (~switch1_e2p)) | (c1c2_p & (~switch1_p2e))
    no_switch2 = (c1c2_e & (~switch2_e2p)) | (c1c2_p & (~switch2_p2e))
    uniqMatch_conv1 = uniqMatch_conv1 + ak.values_astype(switch1_p2e,int) + 2*ak.values_astype(switch1_e2p,int) + oldMatch_conv1*ak.values_astype(no_switch1,int)
    uniqMatch_conv2 = uniqMatch_conv2 + ak.values_astype(switch2_p2e,int) + 2*ak.values_astype(switch2_e2p,int) + oldMatch_conv2*ak.values_astype(no_switch2,int)

    histos['conv_match_matrix_uniq'].fill(sample=samp,cut=cut,
                                match="all",
                                c1_matchType=uniqMatch_conv1,
                                c2_matchType=uniqMatch_conv2)
    histos['conv_match_matrix_uniq'].fill(sample=samp,cut=cut,
                                match="full",
                                c1_matchType=uniqMatch_conv1[fullMatch],
                                c2_matchType=uniqMatch_conv2[fullMatch])
    histos['conv_match_matrix_uniq'].fill(sample=samp,cut=cut,
                                match="any",
                                c1_matchType=uniqMatch_conv1[anyMatch],
                                c2_matchType=uniqMatch_conv2[anyMatch])

##### Helper functions for other routines #####

def findNearestReco(gen,reco,types,thresh=0.1):
    # construct eta-phi vectors
    gen_r = ak.zip({"x":gen.eta,"y":gen.phi},with_name="TwoVector")
    reco_r = ak.zip({"x":reco.eta,"y":reco.phi},with_name="TwoVector")
    # compute dRs
    dr = (reco_r - gen_r).r  
    # sort by ascending dR
    isort = ak.argsort(dr,axis=1)
    dr = dr[isort]
    types = types[isort]
    # keep only matches with dR < threshold
    match = dr < thresh
    dr = dr[match]
    types = types[match]
    imatch = isort[match] 

    return imatch, types, dr

def findMatches(gen1,gen2,reco,recoType):
    i1, t1, dr1 = findNearestReco(gen1,reco,recoType)
    i2, t2, dr2 = findNearestReco(gen2,reco,recoType)
    return i1, t1, dr1, i2, t2, dr2

def findUniqueMatches(ele_matches,best_ele_match,pos_matches,best_pos_match,useCands=False,allow11=False):
    if not useCands:
        ele_matches = ele_matches[ele_matches.typ != 3]
        pos_matches = pos_matches[pos_matches.typ != 3]
    numE = ak.count(ele_matches.typ,axis=1)
    numP = ak.count(pos_matches.typ,axis=1)
    idx = ak.Array(np.arange(len(best_ele_match)))

    partitions = []
    ele = []
    pos = []

    for i in range(1):
        # removing events where one or more objects has no match
        cutA = (numE > 0) & (numP > 0)
        ele.append(best_ele_match[~cutA])
        pos.append(best_pos_match[~cutA])
        partitions.append(idx[~cutA])
        ele_matches = ele_matches[cutA]
        best_ele_match = best_ele_match[cutA]
        pos_matches = pos_matches[cutA]
        best_pos_match = best_pos_match[cutA]
        idx = idx[cutA]
        numE = numE[cutA]
        numP = numP[cutA]

        if len(numE) == 0:
            break

        # remove events with distinct match
        cutB = (best_ele_match.typ == best_pos_match.typ) & (best_ele_match.ind == best_pos_match.ind)
        ele.append(best_ele_match[~cutB])
        pos.append(best_pos_match[~cutB])
        partitions.append(idx[~cutB])
        ele_matches = ele_matches[cutB]
        best_ele_match = best_ele_match[cutB]
        pos_matches = pos_matches[cutB]
        best_pos_match = best_pos_match[cutB]
        idx = idx[cutB]
        numE = numE[cutB]
        numP = numP[cutB]

        if len(numP) == 0:
            break

        # sorting remaining matches by dR
        sort_ele = ak.argsort(ele_matches.dr,axis=1)
        sort_pos = ak.argsort(pos_matches.dr,axis=1)
        ele_matches = ele_matches[sort_ele]
        pos_matches = pos_matches[sort_pos]
        
        cutC = (numE == 1) & (numP == 1)
        cutD = (numE >= 2) & (numP == 1)
        cutE = (numE == 1) & (numP >= 2)
        cutF = (numE >= 2) & (numP >= 2)

        if ak.count_nonzero(cutC) > 0:
            ele_best = best_ele_match[cutC]
            pos_best = best_pos_match[cutC]
            if not allow11:
                ele_best = ak.where(ele_best.dr < pos_best.dr,ele_best,ak.full_like(ele_best,0))
                pos_best = ak.where(pos_best.dr < ele_best.dr,pos_best,ak.full_like(pos_best,0))
            ele.append(ele_best)
            pos.append(pos_best)
            partitions.append(idx[cutC])
        if ak.count_nonzero(cutD) > 0:
            ele_best = ele_matches[cutD][:,1]
            pos_best = best_pos_match[cutD]
            ele.append(ele_best)
            pos.append(pos_best)
            partitions.append(idx[cutD])
        if ak.count_nonzero(cutE) > 0:
            ele_best = best_ele_match[cutE]
            pos_best = pos_matches[cutE][:,1]
            ele.append(ele_best)
            pos.append(pos_best)
            partitions.append(idx[cutE])
        if ak.count_nonzero(cutF) > 0:
            ele_best = ak.where(best_ele_match[cutF].dr < best_pos_match[cutF].dr,ele_matches[cutF][:,0],ele_matches[cutF][:,1])
            pos_best = ak.where(best_pos_match[cutF].dr < best_ele_match[cutF].dr,pos_matches[cutF][:,0],pos_matches[cutF][:,1])
            ele.append(ele_best)
            pos.append(pos_best)
            partitions.append(idx[cutF])
    
    all_partitions = ak.concatenate(partitions)
    reorder = ak.argsort(all_partitions)
    ele_out = ak.concatenate(ele)[reorder]
    pos_out = ak.concatenate(pos)[reorder]

    return ele_out, pos_out

