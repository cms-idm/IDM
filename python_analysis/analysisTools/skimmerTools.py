from __future__ import with_statement
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from mySchema_newCoffea import MySchema
from coffea import processor
from coffea.nanoevents.methods import vector
from coffea.processor import column_accumulator
import uproot
import awkward as ak
ak.behavior.update(vector.behavior)
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
import importlib
import pandas as pd
from XRootD import client
import re
NanoAODSchema.warn_missing_crossrefs = False
import analysisSubroutines as routines
import sys
from collections import defaultdict
from analysisTools import getLumi, deltaPhi
import plotTools as ptools

class Skimmer:
    def __init__(self,fileList,cuts,max_samples=-1,max_files_per_samp=-1,last_cut=None):
        # load in file config
        if type(fileList) == str and ".json" in fileList:
            with open(fileList) as f:
                self.fileList = json.load(f)
        else:
            self.fileList = fileList

        # load in cuts config (should be a path to a .py file with cut methods)
        self.cuts = cuts
        self.last_cut = last_cut # apply cuts up to and including this one (e.g. "cut5"), None = all cuts

        self.sample_names = [] # list of sample names, readable names are generated from data in the fileList json
        self.sample_locs = {} # dictionary mapping sample name to file/directory location
        self.sample_info = {} # dictionary with sample metadata
        self.max_samples = max_samples
        self.max_files_per_samp = max_files_per_samp
        self.nEvents = {}
        self.nEventsProcessed = {}
        self.totalEvents = 0
        self.mode = None
        
        self.loadFiles()
    
    def loadFiles(self):
        loaded = 0
        for sample in self.fileList:
            if self.max_samples > 0 and loaded == self.max_samples:
                break
            mode = sample['type']
            if mode == 'signal':
                name = "sig_{0}_Mchi-{1}_dMchi-{2}_ctau-{3}".format(sample['year'],str(sample['Mchi']).replace(".","p"),str(sample['dMchi']).replace(".","p"),sample['ctau'])
            elif mode == 'bkg':
                name = "bkg_{0}_{1}".format(sample['year'],sample['name'])
            elif mode == 'data':
                name = "data_{0}_{1}".format(sample['year'],sample['name'])
            
            if self.mode is None:
                self.mode = mode
            else:
                if self.mode != mode:
                    print("Error! You're mixing samples of differing types (e.g. signal and bkg, signal and data, etc)")
                    print("Please split up different kinds of samples into different configs")
                    exit()
            
            loc = sample['location']
            if '.root' in loc:
                # if the location is just a single file, load it in
                self.sample_locs[name] = [sample['location']]
            elif 'fileset' in sample.keys():
                self.sample_locs[name] = [f for f in sample['fileset'] if f.split("/")[-1] not in sample['blacklist']]
            else:
                # if the location is a directory, use the xrootd client to get a list of files
                xrdClient = client.FileSystem("root://cmseos.fnal.gov")
                if type(loc) != list:
                    status, flist = xrdClient.dirlist(loc)
                    fullList = ["root://cmsxrootd.fnal.gov/"+loc+"/"+item.name for item in flist if (('.root' in item.name) and (item.name not in sample['blacklist']))]
                else:
                    fullList = []
                    for l in loc:
                        status, flist = xrdClient.dirlist(l)
                        fullList.extend(["root://cmsxrootd.fnal.gov/"+l+"/"+item.name for item in flist if (('.root' in item.name) and (item.name not in sample['blacklist']))])
                if self.max_files_per_samp > 0:
                    fullList = fullList[:self.max_files_per_samp] if len(fullList) > self.max_files_per_samp else fullList
                self.sample_locs[name] = fullList
            
            self.sample_info[name] = sample
            self.sample_names.append(name)
            loaded += 1

    def process(self,treename='ntuples/outT',execr="iterative",workers=4):
        fileset = self.sample_locs
        proc = makeBDTInputs(self.sample_names,self.sample_info,self.sample_locs,self.cuts,mode=self.mode,last_cut=self.last_cut)
        if execr == "iterative":
            executor = processor.IterativeExecutor()
        elif execr == "futures":
            executor = processor.FuturesExecutor(workers=workers)
        else:
            print("Invalid executor type specification!")
            exit
        runner = processor.Runner(executor=executor,schema=MySchema,savemetrics=True)
        accumulator = runner(fileset,
                            treename=treename,
                            processor_instance=proc)
        return accumulator

class makeBDTInputs(processor.ProcessorABC):
    def __init__(self,samples,sampleInfo,fileSet,cutFile,mode='signal',last_cut=None):
        self.samples = samples
        self.sampleInfo = sampleInfo
        self.sampleLocs = fileSet
        self.mode = mode

        # load in cuts module
        self.cutFile = cutFile
        if "/" in self.cutFile: # if cut file is in a different directory
            sys.path.append("/".join(self.cutFile.split("/")[:-1]))
            cutFileName = self.cutFile.split("/")[-1].split(".")[0]
        else: # cut file is in the same directory (e.g. running on condor)
            cutFileName = self.cutFile.split(".")[0]
        self.cutLib = importlib.import_module(cutFileName)
        cutList = [c for c in dir(self.cutLib) if re.fullmatch(r"cut\d+",c)]
        cutList = sorted(cutList,key=lambda x: int(x[3:])) # make sure cuts are ordered as they are in the file
        if last_cut is not None: # only apply cuts up to and including last_cut
            if last_cut not in cutList:
                raise ValueError(f"last_cut '{last_cut}' not found in {self.cutFile} (available: {cutList})")
            cutList = cutList[:cutList.index(last_cut)+1]
        self.cuts = [getattr(self.cutLib,c) for c in cutList]

    
    def process(self,events):
        samp = events.metadata["dataset"]
        outputs = {}
        info = self.sampleInfo[samp]
        isMC = info["type"] == "signal" or info["type"] == "bkg"
        info['defineGoodVertices'] = routines.defineGoodVertices
        info['selectBestVertex'] = routines.selectBestVertex
        if isMC:
            sum_wgt = info["sum_wgt"]
            lumi, unc = getLumi(info['year'])
            xsec = info['xsec']
            if info['type'] == 'signal':
                xsec = xsec*info['filter_eff']
            # register event weight branch
            events.__setitem__("eventWgt",xsec*lumi*events.genWgt)
            events.__setitem__("eventWgtNorm",xsec*lumi*events.genWgt/sum_wgt)
        else:
            sum_wgt = info["num_events"]

        # register event weight branch
        events.__setitem__("eventWgt",xsec*lumi*events.genWgt)
        events.__setitem__("eventWgtNorm",xsec*lumi*events.genWgt/sum_wgt)

        if info['type'] == "signal":
            events.__setitem__("m1",ptools.signalPoint(samp)['m1'])
            events.__setitem__("delta",ptools.signalPoint(samp)['delta'])
            events.__setitem__("ctau",ptools.signalPoint(samp)['ctau'])

        #################################
        ## Calculating Additional Vars ##
        #################################
        events = routines.computeExtraVariables(events,info)
        
        #################################
        ##### Hard-coded basic cuts #####
        #################################
        # 1 or 2 jets in the event
        nJets = ak.count(events.PFJet.pt,axis=1)
        #events = events[(nJets>0) & (nJets<4)]
        events["nJets"] = nJets
        events = events[nJets>0]
        # needs a good vertex
        routines.defineGoodVertices(events,version='v15acr') # define "good" vertices based on whether associated electrons pass ID cuts
        events = events[events.nGoodVtx > 0]
        # define "selected" vertex based on selection criteria in the routine (nominally: lowest chi2)
        # for signal, only select the events where the good vertex is the true vertex
        if info['type'] == "signal":
            events = routines.selectTrueVertex(events, events.good_vtx)
        else:
            routines.selectBestVertex(events)

        ###############################
        ######## CUTS & HISTOS ########
        ###############################
        for cut in self.cuts:
            events, cutName, cutDesc, savePlots = cut(events,info) # apply cuts on jets, BDT will be only used for vertex related variables
        
        # filling outputs dictionary
        e1 = events.sel_vtx.e1
        e2 = events.sel_vtx.e2

        deltadxy = np.abs(np.abs(e1.dxy) - np.abs(e2.dxy))
        mindxy = np.minimum(np.abs(e1.dxy),np.abs(e2.dxy))
        maxdxy = np.maximum(np.abs(e1.dxy),np.abs(e2.dxy))
        meandxy = (np.abs(e1.dxy) + np.abs(e2.dxy))/2
        abs_deta = np.abs(e1.eta - e2.eta)
        abs_dphi = np.abs(deltaPhi(e1.phi, e2.phi))
        
        outputs['wgt'] = column_accumulator(events["eventWgt"].to_numpy())
        xsec = info['xsec']
        if info['type'] == 'signal':
            xsec = xsec*info['filter_eff']
        outputs['xsec'] = column_accumulator(xsec*np.ones(len(events)))
        #outputs['samp'] = column_accumulator(np.full(len(events),samp))
        outputs['wgt_norm'] = column_accumulator(events["eventWgtNorm"].to_numpy())
        # event IDs, used for the k-fold assignment in BDT training (fold = eventNum % k)
        outputs['eventNum'] = column_accumulator(events.eventNum.to_numpy())
        outputs['lumiSec'] = column_accumulator(events.lumiSec.to_numpy())
        outputs['runNum'] = column_accumulator(events.runNum.to_numpy())
        outputs['lead_jet_eta'] = column_accumulator(events.PFJet.eta[:,0].to_numpy())
        outputs['lead_jet_pt'] = column_accumulator(events.PFJet.pt[:,0].to_numpy())
        outputs['jetMETdPhi'] = column_accumulator(np.abs(events.PFJet.METdPhi[:,0]).to_numpy())
        outputs['minJetMETdPhi'] = column_accumulator(ak.min(np.abs(events.PFJet.METdPhi),axis=1).to_numpy())
        outputs['sel_vtx_sign'] = column_accumulator(events.sel_vtx.sign.to_numpy())
        outputs['sel_vtx_chi2'] = column_accumulator(events.sel_vtx.reduced_chi2.to_numpy())
        outputs['sel_vtx_METdPhi'] = column_accumulator(np.abs(events.sel_vtx.METdPhi).to_numpy())
        outputs['sel_vtx_m'] = column_accumulator(events.sel_vtx.m.to_numpy())
        outputs['sel_vtx_refit_m'] = column_accumulator(events.sel_vtx.refit_m.to_numpy())
        outputs['sel_vtx_dR'] = column_accumulator(events.sel_vtx.dR.to_numpy())
        outputs['sel_vtx_refit_dR'] = column_accumulator(events.sel_vtx.refit_dR.to_numpy())
        outputs['sel_vtx_minDxy'] = column_accumulator(np.minimum(np.abs(events.sel_vtx.e1.dxy),np.abs(events.sel_vtx.e2.dxy)).to_numpy())
        outputs['sel_vtx_corrMinDxy'] = column_accumulator(events.sel_vtx.corrMinDxy.to_numpy())
        outputs['met_leadPt_ratio'] = column_accumulator((events.PFMET.pt/events.PFJet.pt[:,0]).to_numpy())
        outputs["vxy"] = column_accumulator(events.sel_vtx.vxy.to_numpy())
        outputs["vxy_signif"] = column_accumulator((events.sel_vtx.vxy/events.sel_vtx.sigmavxy).to_numpy())
        outputs["sel_e1_dxy"] = column_accumulator(np.abs(e1.dxy).to_numpy())
        outputs["sel_e1_dxySignif"] = column_accumulator((np.abs(e1.dxy)/e1.dxyErr).to_numpy())
        outputs["sel_e2_dxy"] = column_accumulator(np.abs(e2.dxy).to_numpy())
        outputs["sel_e2_dxySignif"] = column_accumulator((np.abs(e2.dxy)/e2.dxyErr).to_numpy())
        
        if info['type'] == 'signal':
            # sel_vtx is re-selected (lowest chi2) inside the cuts, so compute the match count here
            outputs['sel_vtx_match'] = column_accumulator(ak.to_numpy(ak.values_astype(events.sel_vtx.e1_isMatched,int) + ak.values_astype(events.sel_vtx.e2_isMatched,int)))
            outputs['sel_vtx_isMatched'] = column_accumulator(ak.values_astype(events.sel_vtx.isMatched,int).to_numpy())
            outputs['m1'] = column_accumulator(events["m1"].to_numpy())
            outputs['delta'] = column_accumulator(events["delta"].to_numpy())
            outputs['ctau'] = column_accumulator(events["ctau"].to_numpy())

        outputs["sel_vtx_cos_collinear"] = column_accumulator(events.sel_vtx.cos_collinear.to_numpy())
        outputs["sel_vtx_prod_eta"] = column_accumulator((e1.eta*e2.eta).to_numpy())
        outputs["sel_vtx_sign_prod_eta"] = column_accumulator(((e1.eta*e2.eta)/np.abs(e1.eta*e2.eta)).to_numpy())

        outputs["sel_vtx_pt"] = column_accumulator(events.sel_vtx.pt.to_numpy())
        outputs["sel_vtx_projectedLxy"] = column_accumulator(events.sel_vtx.projectedLxy.to_numpy())
        outputs["sel_vtx_pt_e1_over_pt_e2"] = column_accumulator((np.minimum(e1.pt, e2.pt)/np.maximum(e1.pt, e2.pt)).to_numpy())
        outputs["sel_vtx_pt_over_m"] = column_accumulator((events.sel_vtx.pt/events.sel_vtx.m).to_numpy())
        outputs["delta_dxy_over_mindxy"] = column_accumulator((deltadxy/mindxy).to_numpy())
        outputs["delta_dxy_over_maxdxy"] = column_accumulator((deltadxy/maxdxy).to_numpy())
        outputs["delta_dxy_over_meandxy"] = column_accumulator((deltadxy/meandxy).to_numpy())
        outputs["delta_eta_over_delta_phi"] = column_accumulator((abs_deta/abs_dphi).to_numpy())
        outputs["log_delta_eta_over_delta_phi"] = column_accumulator(np.log10(abs_deta/abs_dphi).to_numpy())
        outputs["logdxydz"] = column_accumulator(np.minimum(np.log10(np.abs(events.sel_vtx.e1.dxy/events.sel_vtx.e1.dz)),np.log10(np.abs(events.sel_vtx.e2.dxy/events.sel_vtx.e2.dz))).to_numpy())
       
        outputs["sel_vtx_CosThetaColl_fromPV_refit"] = column_accumulator(events.sel_vtx.cos_collinear_fromPV_refit.to_numpy())
        outputs["sel_vtx_ThetaColl_fromPV_refit"] = column_accumulator(np.arccos(events.sel_vtx.cos_collinear_fromPV_refit.to_numpy()))
        outputs["sel_vtx_CosThetaColl_fromPV"] = column_accumulator(events.sel_vtx.cos_collinear_fromPV.to_numpy())
        outputs["vxy_corr"] = column_accumulator(np.sqrt((events.sel_vtx.vx-events.PV.x)**2+(events.sel_vtx.vy-events.PV.y)**2).to_numpy())
        outputs["sel_vtx_minDxy_refit"] = column_accumulator(np.minimum(np.abs(events.sel_vtx.e1.refit_dxy), np.abs(events.sel_vtx.e2.refit_dxy)).to_numpy())
        outputs["sel_vtx_METdPhi_corr"] = column_accumulator(np.abs(deltaPhi(events.PFMET.phi, events.sel_vtx.refit_phi)).to_numpy())


        return outputs

    def postprocess(self, accumulator):
        return accumulator
    
