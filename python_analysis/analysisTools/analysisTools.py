from __future__ import with_statement
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from mySchema import MySchema
from coffea import processor

#from coffea.dataset_tools import (
#    apply_to_fileset,
#    max_chunks,
#    preprocess,
#)
#import dask

import uproot
import awkward as ak
#import vector
#vector.register_awkward()
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
from hist import Hist
from hist.axis import StrCategory, Regular, Integer, IntCategory
import hist
import corrections

match_names = {"Default":"match0","lowpt":"match1"}
vxy_range = {1:[0,20],10:[0,50],100:[0,50],1000:[0,50]}
vxy_rebin = {1:5,10:20,100:20,1000:20}

class Analyzer:
    def __init__(self,fileList,histoList,cuts,model_json=None,systematics=None,max_samples=-1,max_files_per_samp=-1,newCoffea=False,nJet_isNominal=None,isSFstudies=False,good_vtx='v11'):
        # flag to see if we're using new coffea
        self.newCoffea = newCoffea

        # load in file config
        if type(fileList) == str and ".json" in fileList:
            with open(fileList) as f:
                self.fileList = json.load(f)
        else:
            self.fileList = fileList

        # systematics
        if systematics != None:
            with open(systematics) as s:
                sys_list = json.load(s)
            self.systematics = {}
            for idx, item in enumerate(sys_list):
                self.systematics[sys_list[idx]["name"]] = sys_list[idx]["type"]
        else:
            self.systematics = None

        self.isSFstudies = isSFstudies

        #load in histogram config
        if "/" in histoList: # if cut file is in a different directory
            sys.path.append("/".join(histoList.split("/")[:-1]))
            self.histoFile = histoList.split("/")[-1].split(".")[0]
        else: # cut file is in the same directory (e.g. running on condor)
            self.histoFile = histoList.split(".")[0]

        # load in cuts config (should be a path to a .py file with cut methods)
        self.cuts = cuts

        self.sample_names = [] # list of sample names, readable names are generated from data in the fileList json
        self.sample_locs = {} # dictionary mapping sample name to file/directory location
        self.sample_info = {} # dictionary with sample metadata
        self.max_samples = max_samples
        self.max_files_per_samp = max_files_per_samp
        self.nEvents = {}
        self.nEventsProcessed = {}
        self.totalEvents = 0
        self.mode = None

        self.model = model_json # BDT model for inference (if used in selections)
        self.nJet_isNom = nJet_isNominal # nomial njet range of NJet > 0 and NJet < 3
        self.good_vtx = good_vtx
        
        self.loadFiles()
    
    def loadFiles(self):
        loaded = 0
        for sample in self.fileList:
            if self.max_samples > 0 and loaded == self.max_samples:
                break
            mode = sample['type']
            if mode == 'signal':
                if 'mZD' in sample['name']:
                    name = "sig_{0}_Mchi-{1}_dMchi-{2}_ctau-{3}_{4}".format(sample['year'],str(sample['Mchi']).replace(".","p"),str(sample['dMchi']).replace(".","p"),sample['ctau'],sample['name'].split('_')[-1])
                else:
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
                if self.newCoffea:
                    self.sample_locs[name] = {'files':{sample['location']:'ntuples/outT'}}
                else:
                    self.sample_locs[name] = [sample['location']]
            elif 'fileset' in sample.keys():
                if self.newCoffea:
                    self.sample_locs[name] = {'files':{f:'ntuples/outT' for f in sample['fileset'] if f.split("/")[-1] not in sample['blacklist']}}
                    if self.max_files_per_samp > 0 and len(self.sample_locs[name]['files']) > self.max_files_per_samp:
                        self.sample_locs[name]['files'] = {k:self.sample_locs[name]['files'][k] for k in list(self.sample_locs[name]['files'].keys())[:self.max_files_per_samp]}
                else:
                    self.sample_locs[name] = [f for f in sample['fileset'] if f.split("/")[-1] not in sample['blacklist']]
                    if self.max_files_per_samp > 0 and len(self.sample_locs[name]) > self.max_files_per_samp:
                        self.sample_locs[name] = self.sample_locs[name][:self.max_files_per_samp]
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
                if self.max_files_per_samp > 0 and len(fullList) > self.max_files_per_samp:
                    fullList = fullList[:self.max_files_per_samp]
                if self.newCoffea:
                    self.sample_locs[name] = {'files':{f:'ntuples/outT' for f in fullList}}
                else:
                    self.sample_locs[name] = fullList
            
            self.sample_info[name] = sample
            self.sample_names.append(name)
            loaded += 1

    def process(self,treename='ntuples/outT',execr="iterative",workers=4,merging=False,dask_client=None,procType='default',**kwargs):
        fileset = self.sample_locs
        if procType == 'default':
            proc = iDMeProcessor(self.sample_names,self.sample_info,self.sample_locs,self.histoFile,self.cuts,mode=self.mode,model_json=self.model,nJet_isNom=self.nJet_isNom,isSFstudies=self.isSFstudies,good_vtx=self.good_vtx,systematics=self.systematics,**kwargs)
        elif procType == 'gen':
            proc = genProcessor(self.sample_names,self.sample_info,self.sample_locs,self.histoFile,self.cuts,mode=self.mode,**kwargs)
        elif procType == 'trig':
            proc = trigProcessor(self.sample_names,self.sample_info,self.sample_locs,self.histoFile,self.cuts,mode=self.mode,**kwargs)
        elif procType == 'bare':
            proc = bareProcessor(self.sample_names,self.sample_info,self.sample_locs,self.histoFile,self.cuts,mode=self.mode,**kwargs)
        
        if not self.newCoffea:
            if execr == "iterative":
                executor = processor.IterativeExecutor()
            elif execr == "futures":
                executor = processor.FuturesExecutor(workers=workers, merging=merging)
            elif execr == "dask":
                if dask_client is None:
                    print("Need to supply a dask client!")
                    return
                else:
                    executor = processor.DaskExecutor(client=dask_client)
            else:
                print("Invalid executor type specification!")
                return
            runner = processor.Runner(executor=executor,schema=MySchema,savemetrics=True)
            accumulator = runner(fileset,
                                treename=treename,
                                processor_instance=proc)
        else:
            print("Preprocessing")
            dataset_runnable, dataset_updated = preprocess(fileset,step_size=100_000,files_per_batch=1)
            print("Done Preprocessing")
            to_compute = apply_to_fileset(proc,dataset_runnable,schemaclass=MySchema)
            (accumulator,) = dask.compute(to_compute)
        
        return accumulator

class iDMeProcessor(processor.ProcessorABC):
    def __init__(self,samples,sampleInfo,fileSet,histoFile,cutFile,mode='signal',model_json=None,nJet_isNom=None,isSFstudies=False,good_vtx='v11',systematics=None,**kwargs):
        self.samples = samples
        self.sampleInfo = sampleInfo
        self.sampleLocs = fileSet
        self.mode = mode
        self.model = model_json
        self.nJet_isNom = nJet_isNom
        self.good_vtx = good_vtx
        self.systematics = systematics
        print('Running with systematics: ', self.systematics)
        self.isSFstudies = isSFstudies

        # load in histogram config
        self.histoMod = importlib.import_module(histoFile)
        self.histoFill = self.histoMod.fillHistos
        self.subroutines = self.histoMod.subroutines
        
        # load in cuts module
        self.cutFile = cutFile
        if "/" in self.cutFile: # if cut file is in a different directory
            sys.path.append("/".join(self.cutFile.split("/")[:-1]))
            cutFileName = self.cutFile.split("/")[-1].split(".")[0]
            self.cutLib = importlib.import_module(cutFileName)
            cutList = [c for c in dir(self.cutLib) if "cut" in c]
            cutList = sorted(cutList,key=lambda x: int(x[3:])) # make sure cuts are ordered as they are in the file
            self.cuts = [getattr(self.cutLib,c) for c in cutList]
        else: # cut file is in the same directory (e.g. running on condor)
            cutFileName = self.cutFile.split(".")[0]
            self.cutLib = importlib.import_module(cutFileName)
            cutList = [c for c in dir(self.cutLib) if "cut" in c]
            cutList = sorted(cutList,key=lambda x: int(x[3:])) # make sure cuts are ordered as they are in the file
            self.cuts = [getattr(self.cutLib,c) for c in cutList]

        self.extraStuff = {}
        for k,v in kwargs.items():
            self.extraStuff[k] = v
            print(f"Registering extra input {k} = {v}")
    
    def process(self,events):
        samp = events.metadata["dataset"]
        info = self.sampleInfo[samp]
        isMC = info["type"] == "signal" or info["type"] == "bkg"
        info['defineGoodVertices'] = routines.defineGoodVertices
        info['selectBestVertex'] = routines.selectBestVertex
        for k,v in self.extraStuff.items():
            info[f"extras_{k}"] = v
        
        #histos = self.histoMod.make_histograms()
        #histos['cutDesc'] = defaultdict(str)
        histObj = self.histoMod.make_histograms(info)
        cutDesc = defaultdict(str)

        cutflow = defaultdict(float)               # efficiency
        cutflow_counts = defaultdict(float)        # xsec-weighted event counts
        cutflow_nevts = defaultdict(int)           # raw event counts
        cutflow_vtx_matched = defaultdict(float)   # (for signal MC) fraction that the selected vertex is truth-matched
        # (for signal MC) also check the above, but only counting the events where both gen ee are reconstructed: dR(reco,gen) < 0.1

        if isMC:
            sum_wgt = info["sum_wgt"]
            lumi, unc = getLumi(info['year'])
            xsec = info['xsec']
            # Apply NLO k-factor for W/Z
            if 'DY' in info['name']:
                if self.isSFstudies == False:
                    xsec = xsec * 1.23
            elif 'WJet' in info['name']:
                xsec = xsec * 1.21
            elif 'ZJet' in info['name']:
                xsec = xsec * 1.23
                
            if info['type'] == 'signal':
                xsec = xsec*info['filter_eff']
            # register event weight branch
            events.__setitem__("eventWgt",xsec*lumi*events.genWgt)
        else:
            sum_wgt = info["num_events"]

        # Initial number of events
        if isMC:
            cutflow['all'] += ak.sum(events.genWgt)/sum_wgt
        else:
            cutflow['all'] += len(events)/sum_wgt
        cutflow_nevts['all'] += len(events)

        if info['type'] == "signal":
            cutflow_vtx_matched['all'] += 1 # dummy value before selecting a vertex
            
        cutDesc['all'] = 'No cuts@'

        ######################################################################################
        ## Add HEM flags to Event (before applying any quality cuts to jet, electrons ##
        ######################################################################################

        if str(info['year']) == '2018':
            routines.checkHEMjet(events)
            routines.checkHEMelectron(events)

        #################################
        ## Calculating Additional Vars ##
        #################################
        events = routines.computeExtraVariables(events,info)

        #################################
        ## HEM Veto for 2018 ##
        #################################
 
        # Veto HEM jets and electrons for 2018 data and MC
        if str(info['year']) == '2018':
            events = events[events.hasHEMjet == 0]
            events = events[events.hasHEMelecPF == 0]
            events = events[events.hasHEMelecLpt == 0]

        #################################
        ## Applying systematics and SF ##
        #################################
        iov = str(info['year'])
        jsonPath = f"../../../analysisTools/corrections/{info['year']}/"

        apply_vtx_SF = False
        if self.systematics != None:
            #print('systematics')
            
            if self.systematics['PU'] != 'None':
                if isMC:
                    sf_PU = corrections.get_sf_PU(iov, jsonPath, nTrueInt=events.genPU.true, type=self.systematics['PU'])
                    events["eventWgt"] = events["eventWgt"] * np.array(sf_PU)
            if self.systematics['electron'] != 'None':
                if isMC:
                    sf_vtx = corrections.get_sf_ee_vtx(iov, jsonPath, events.vtx.vxy_fromPV, vtx_type = events.vtx.typ, type=self.systematics['electron'])
                    #sf_vtx = routines.vtxElectronSF(events, type=self.systematics['electron']) # shitty hack
                    events['vtx','sf'] = sf_vtx
                    apply_vtx_SF = True
            if self.systematics['btag'] != 'None':
                if isMC:
                    sf_btag = corrections.apply_btagSF(iov, jsonPath, events, type=self.systematics['btag'])
                    events["eventWgt"] = events["eventWgt"] * np.array(sf_btag)
            if self.systematics['jec'] != 'None':
                corrections.apply_JEC(isMC, iov, events, type=self.systematics['jec'])
            if self.systematics['met'] != 'None':
                corrections.correct_MET(events, type=self.systematics['met'])
            
            if self.systematics['trigger_MC'] != 'None':
                if isMC:
                    sf_trig_MC = corrections.get_trigger_MC(iov, jsonPath, events.PFMET.pt, isMC, type=self.systematics['trigger_MC'])
                    events['eventWgt'] = events['eventWgt'] * sf_trig_MC
            if self.systematics['trigger_data'] != 'None':
                if not isMC:
                    sf_trig_data = corrections.get_trigger_MC(iov, jsonPath, events.PFMET.pt, isMC, type=self.systematics['trigger_data'])
                    events['eventWgt'] = events['eventWgt'] * sf_trig_data
            

        #################################
        ##### Hard-coded basic cuts #####
        #################################
        # 1 or 2 jets in the event
        nJets = ak.count(events.PFJet.pt,axis=1)
        #events = events[(nJets>0) & (nJets<3)] # Nominal NJet requirement for SR
        events["nJets"] = nJets
        events = events[nJets>0]

        if self.nJet_isNom != None: # If applying Njet cut (legacy: deprecated after fixing the nJet bug in pythia)
            if self.nJet_isNom:
                events = events[(nJets>0) & (nJets<3)] # Nominal NJet requirement for SR & VR2
                cutName = '0 < NJet < 3@'
                print('Njet nominal')
            elif self.nJet_isNom:
                events = events[(nJets>2)] # Reverse NJet requirement for VR1
                cutName = 'NJet > 2@'
                print('Njet flipped')

            # Initial number of events
            if isMC:
                cutflow['njet'] += ak.sum(events.genWgt)/sum_wgt
            else:
                cutflow['njet'] += len(events)/sum_wgt
            cutflow_nevts['njet'] += len(events)

            if info['type'] == "signal":
                cutflow_vtx_matched['njet'] += 1 # dummy value before selecting a vertex
                cutflow_vtx_matched_fromReco['njet'] += 1

            cutDesc['njet'] = cutName

        # needs a good vertex
        routines.defineGoodVertices(events,version=self.good_vtx) # define "good" vertices based on whether associated electrons pass ID cuts
        events = events[events.nGoodVtx > 0]
        # define "selected" vertex based on selection criteria in the routine (nominally: lowest chi2)
        #routines.selectBestVertex(events)
        if info['type'] == "signal":
             events = routines.selectTrueVertex(events,events.good_vtx)
             #routines.selectBestVertex(events)
        else:
             routines.selectBestVertex(events)
        routines.prepareBDT(events, self.model) # prepare BDT inference if the cuts include BDT-based cut

        # Vtx SF related stuff (shitty hack)
        if apply_vtx_SF == True:
             events["eventWgt"] = events["eventWgt"] * events.sel_vtx.sf

        # Fill cutflow after baseline selection
        if isMC:
            cutflow['hasVtx'] += ak.sum(events.genWgt)/sum_wgt
        else:
            cutflow['hasVtx'] += len(events)/sum_wgt
        cutflow_nevts['hasVtx'] += len(events)
        cutDesc['hasVtx'] = 'Baseline Selection@'

        # For signal, (1) check if the vertex ee are gen-matched (2) check if the event has ee that are gen-matched
        if info['type'] == "signal":
            routines.projectGenLxy(events)
            vtx_matched_events = events[events.sel_vtx.isMatched]
            cutflow_vtx_matched['hasVtx'] += ak.sum(vtx_matched_events.genWgt)/ak.sum(events.genWgt)

        # computing any extra quantities specified in the histogram config file
        for subroutine in self.subroutines:
            getattr(routines,subroutine)(events)

        
        # SF studies
        if self.isSFstudies:
            if isMC:
                if 'ZGamma' in info['name']:
                    print('Before cutting on ZG pT', len(events))
                    events = events[events.sel_vtx.refit_pt > 17]
                    print('After cutting on ZG pT', len(events[events.sel_vtx.refit_pt > 17]))
                elif 'DY' in info['name']:
                    events = events[events.sel_vtx.refit_pt < 17]
        
        
        ###############################
        ######## CUTS & HISTOS ########
        ###############################
        for cut in self.cuts:
            events, cutName, cutDescription, savePlots = cut(events,info)
            if isMC:
                cutflow[cutName] += ak.sum(events.genWgt)/sum_wgt
            else:
                cutflow[cutName] += len(events)/sum_wgt
            cutflow_nevts[cutName] += len(events)            
            if info['type'] == "signal":
                vtx_matched_events = events[events.sel_vtx.isMatched]
                cutflow_vtx_matched[cutName] += ak.sum(vtx_matched_events.genWgt)/ak.sum(events.genWgt)
            cutDesc[cutName] += cutDescription + "@"

            # Fill histograms
            if savePlots and len(events) > 0: # fixes some bugginess trying to fill histograms with empty arrays
                self.histoFill(events,histObj,samp,cutName,info,sum_wgt=sum_wgt)
        
        for k in cutflow.keys():
            if isMC:
                cutflow_counts[k] = xsec*lumi*cutflow[k]
            else:
                cutflow_counts[k] = sum_wgt*cutflow[k]
        
        histos = histObj.histograms
        histos['cutDesc'] = cutDesc
        histos['cutflow'] = {samp:cutflow}
        histos['cutflow_cts'] = {samp:cutflow_counts}
        histos['cutflow_nevts'] = {samp:cutflow_nevts}
        histos['cutflow_vtx_matched'] = {samp:cutflow_vtx_matched}

        return histos

    def postprocess(self, accumulator):
        # only need one description per cut name -- adds many during parallel execution
        for cutName in list(accumulator['cutDesc'].keys()):
            accumulator['cutDesc'][cutName] = accumulator['cutDesc'][cutName].split("@")[0]
        return accumulator

# processor for extracting gen/truth-matched signal plots - no selection
class genProcessor(iDMeProcessor):
    def process(self,events):
        samp = events.metadata["dataset"]
        info = self.sampleInfo[samp]
        
        #histos = self.histoMod.make_histograms()
        #histos['cutDesc'] = defaultdict(str)
        histObj = self.histoMod.make_histograms(info)
        
        cutDesc = defaultdict(str)
        cutflow = defaultdict(float)               # efficiency
        cutflow_counts = defaultdict(float)        # xsec-weighted event counts
        cutflow_nevts = defaultdict(int)           # raw event counts
        
        sum_wgt = info["sum_wgt"]
        lumi, unc = getLumi(info['year'])
        xsec = info['xsec']
        if info['type'] == 'signal':
            xsec = xsec*info['filter_eff']

        # register event weight branch
        events.__setitem__("eventWgt",xsec*lumi*events.genWgt)

        # Initial number of events
        cutflow['all'] += ak.sum(events.genWgt)/sum_wgt
        cutflow_nevts['all'] += len(events)
        cutDesc['all'] = 'No cuts'

        #################################
        ## Calculating Additional Vars ##
        #################################
        routines.jetBtag(events,info['year'])
        routines.vtxElectronConnection(events)
        events = routines.computeExtraVariables(events,info)
        #if info['type'] == 'signal':
        #    routines.genMatchExtraVtxVariables(events)

        # initial histogram fill
        self.histoFill(events,histObj,samp,"no_presel",info,sum_wgt=sum_wgt)

        #################################
        #### Hard-coded basic cuts ######
        #################################
        # 1 or 2 jets in the event
        #nJets = ak.count(events.PFJet.pt,axis=1)
        #events = events[(nJets>0) & (nJets<3)]
        
        #################################
        #### Demand >= 1 ee vertices ####
        #################################
        #routines.defineGoodVertices(events) # define "good" vertices based on whether associated electrons pass ID cuts
        routines.defineGoodVertices(events,version='none') # define "good" vertices based on whether associated electrons pass ID cuts

        #events.__setitem__("nGoodVtx",ak.count(events.good_vtx.vxy,axis=1))
        #events = events[events.nGoodVtx > 0]
        # define "selected" vertex based on selection criteria in the routine (nominally: lowest chi2)
        routines.selectBestVertex(events)

        # Fill cutflow after baseline selection
        cutflow['hasVtx'] += ak.sum(events.genWgt)/sum_wgt
        cutflow_nevts['hasVtx'] += len(events)
        cutDesc['hasVtx'] = 'Baseline Selection'
        
        # computing any extra quantities specified in the histogram config file
        for subroutine in self.subroutines:
            getattr(routines,subroutine)(events)
        
        ###############################
        ######## CUTS & HISTOS ########
        ###############################
        for cut in self.cuts:
            events, cutName, cutDescription, savePlots = cut(events,info)
            cutflow[cutName] += ak.sum(events.genWgt)/sum_wgt
            cutflow_nevts[cutName] += len(events)            
            cutDesc[cutName] += cutDescription + "@"
            # Fill histograms
            if savePlots and len(events) > 0: # fixes some bugginess trying to fill histograms with empty arrays
                self.histoFill(events,histObj,samp,cutName,info,sum_wgt=sum_wgt)
        
        for k in cutflow.keys():
            cutflow_counts[k] = xsec*lumi*cutflow[k]
        
        
        histos = histObj.histograms
        histos['cutDesc'] = cutDesc
        histos['cutflow'] = {samp:cutflow}
        histos['cutflow_cts'] = {samp:cutflow_counts}
        histos['cutflow_nevts'] = {samp:cutflow_nevts}
        
        return histos

# processor for doing nothing but filling histos
class bareProcessor(iDMeProcessor):
    def process(self,events):
        samp = events.metadata["dataset"]
        info = self.sampleInfo[samp]
        isMC = info["type"] == "signal" or info["type"] == "bkg"
        info['defineGoodVertices'] = routines.defineGoodVertices
        info['selectBestVertex'] = routines.selectBestVertex
        for k,v in self.extraStuff.items():
            info[f"extras_{k}"] = v
        
        #histos = self.histoMod.make_histograms()
        #histos['cutDesc'] = defaultdict(str)
        histObj = self.histoMod.make_histograms(info)
        cutDesc = defaultdict(str)
        
        if isMC:
            sum_wgt = info["sum_wgt"]
            lumi, unc = getLumi(info['year'])
            xsec = info['xsec']
            if info['type'] == 'signal':
                xsec = xsec*info['filter_eff']
            # register event weight branch
            events.__setitem__("eventWgt",xsec*lumi*events.genWgt)
        else:
            sum_wgt = info["num_events"]

        #################################
        ## Calculating Additional Vars ##
        #################################
        #events = routines.computeExtraVariables(events,info)
        
        #################################
        ##### Hard-coded basic cuts #####
        #################################
        # 1 or 2 jets in the event
        #nJets = ak.count(events.PFJet.pt,axis=1)
        #events = events[(nJets>0) & (nJets<4)]
        #events = events[nJets>0]
        #events["nJets"] = nJets
        # needs a good vertex
        #routines.defineGoodVertices(events,version='v8') # define "good" vertices based on whether associated electrons pass ID cuts
        #events = events[events.nGoodVtx > 0]
        # define "selected" vertex based on selection criteria in the routine (nominally: lowest chi2)
        #routines.selectBestVertex(events)

        self.histoFill(events,histObj,samp,"all",info,sum_wgt=sum_wgt)
        
        histos = histObj.histograms

        return histos
    
    def postprocess(self, accumulator):
        # only need one description per cut name -- adds many during parallel execution
        return accumulator

class trigProcessor(iDMeProcessor):
    def process(self,events):
        samp = events.metadata["dataset"]

        info = self.sampleInfo[samp]
        isMC = info["type"] == "signal" or info["type"] == "bkg"
        if isMC:
            sum_wgt = info["sum_wgt"]
            lumi, unc = getLumi(info['year'])
            xsec = info['xsec']
            if info['type'] == 'signal':
                xsec = xsec*info['filter_eff']
            # register event weight branch
            events.__setitem__("eventWgt",xsec*lumi*events.genWgt)
        else:
            sum_wgt = info["num_events"]

        # fill histo
        if isMC:
            events['wgt'] = events.eventWgt/sum_wgt
        else:
            events['wgt'] = 1
        
        # define histo
        MET_passTrig = Hist(StrCategory([],name="samp",label="Sample Name",growth=True),
                               Regular(600,0,600,name="met",label="met"),
                               IntCategory([0,1],name="passTrig",label="passTrig"),
                               storage=hist.storage.Weight())

        MET_all = Hist(StrCategory([],name="samp",label="Sample Name",growth=True),
                               Regular(600,0,600,name="met",label="met"),
                               storage=hist.storage.Weight())

        MET_passTrig_all = Hist(StrCategory([],name="samp",label="Sample Name",growth=True),
                               Regular(600,0,600,name="met",label="met"),
                               IntCategory([0,1],name="passTrig",label="passTrig"),
                               storage=hist.storage.Weight())

        jet_pt_passTrig = Hist(StrCategory([],name="samp",label="Sample Name",growth=True),
                               Regular(500,0,500,name="pt",label="pt]"),
                               IntCategory([0,1],name="passTrig",label="passTrig"),
                               storage=hist.storage.Weight())

        jet_pt_all = Hist(StrCategory([],name="samp",label="Sample Name",growth=True),
                               Regular(500,0,500,name="pt",label="pt]"),
                               storage=hist.storage.Weight())

        # histograms w/o selection
        MET_all.fill(samp=samp,met=events.PFMET.pt,weight=events.wgt)
        MET_passTrig_all.fill(samp=samp,met=events.PFMET.pt,
                              passTrig=ak.values_astype(events.trig.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight,int),
                              weight=events.wgt)
    
        # require jets
        nJets = ak.count(events.PFJet.pt,axis=1)
        #events = events[(nJets>0)&(nJets<3)]
        events = events[nJets>0]
        jet_pt_all.fill(samp=samp,pt=events.PFJet.pt[:,0],weight=events.wgt)
        events = events[np.abs(events.PFMET.pt-events.CaloMET.pt)/events.CaloMET.pt < 0.5]

        # iDM-like jet selection
        if info['type'] == 'data' or info['type'] == "bkg":
            lead_pt = events.PFJet.pt[:,0]
            lead_eta = np.abs(events.PFJet.eta[:,0])
            cut = (lead_pt > 80) & (lead_eta < 2.4)
            
            ele_cut = (events.Electron.pt > 30) & (np.abs(events.Electron.eta) < 2.4)
            #lpt_ele_cut = (events.LptElectron.pt > 30)
            n_ele_cut = ak.any(ele_cut,axis=1)# | ak.any(lpt_ele_cut,axis=1)
            
            cut = cut & n_ele_cut
            events = events[cut]
        
        # require reference trigger - try several
        refTrigs = [
            "HLT_DoubleEle27_CaloIdL_MW",
            "HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350",
            "HLT_Ele17_CaloIdM_TrackIdM_PFJet30",
            "HLT_Ele40_WPTight_Gsf",
            "HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350",
            "HLT_Ele32_WPTight_Gsf",
            "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
            "HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL",
            "HLT_Ele35_WPTight_Gsf_L1EGMT",
            "HLT_Ele23_CaloIdM_TrackIdM_PFJet30",
            "HLT_Ele38_WPTight_Gsf",
            "HLT_DoubleEle24_eta2p1_WPTight_Gsf",
            "HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned",
            "HLT_Ele8_CaloIdM_TrackIdM_PFJet30",
            "HLT_Ele35_WPTight_Gsf",
            "HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30",
            "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "HLT_Ele28_eta2p1_WPTight_Gsf_HT150",
            "HLT_DoubleEle33_CaloIdL_MW",
            "HLT_Ele32_WPTight_Gsf_L1DoubleEG",
            "HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30"
            "HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
            "HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30"
        ]

        MET_passTrig_refTrig = Hist(StrCategory([],name="samp",label="Sample Name",growth=True),
                                    StrCategory([],name="refTrig",label="ref trigger",growth=True),
                                    Regular(600,0,600,name="met",label="met"),
                                    IntCategory([0,1],name="passMET",label="passMET"),
                                    IntCategory([0,1],name="passRef",label="passRef"),
                                    storage=hist.storage.Weight())
        for refTrig in refTrigs:
            if refTrig in events.trig.fields:
                MET_passTrig_refTrig.fill(samp=samp,met=events.PFMET.pt,refTrig=refTrig,
                                        passMET=ak.values_astype(events.trig.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight,int),
                                        passRef=ak.values_astype(events.trig[refTrig],int))

        #events = events[events.trig.HLT_Ele20_WPTight_Gsf==1]

        # fill histos
        #MET_passTrig.fill(samp=samp,
        #                     met=events.PFMET.pt,
        #                     passTrig=ak.values_astype(events.trig.HLT_PFMET120_PFMHT120_IDTight,int),
        #                     weight=events.wgt)
        #jet_pt_passTrig.fill(samp=samp,pt=events.PFJet.pt[:,0],passTrig=ak.values_astype(events.trig.HLT_PFMET120_PFMHT120_IDTight,int),weight=events.wgt)

        output = {"MET_passTrig":MET_passTrig,"MET_all":MET_all, "MET_passTrig_all":MET_passTrig_all,
                  "jet_pt_all":jet_pt_all, "jet_pt_passTrig":jet_pt_passTrig,"MET_passTrig_refTrig":MET_passTrig_refTrig}

        return output

    def postprocess(self, accumulator):
        return accumulator

class fileSkimmer:
    def __init__(self,sampFile,sampleInfo,cutFile,mode='signal'):
        self.sampleInfo = sampleInfo
        self.sampFile = sampFile
        fname = sampFile.split("/")[-1]
        self.outFileName = fname.replace(".root","_skimmed.root")
        self.mode = mode
        
        # load in cuts module
        self.cutFile = cutFile
        if "/" in self.cutFile: # if cut file is in a different directory
            sys.path.append("/".join(self.cutFile.split("/")[:-1]))
            cutFileName = self.cutFile.split("/")[-1].split(".")[0]
            self.cutLib = importlib.import_module(cutFileName)
            cutList = [c for c in dir(self.cutLib) if "cut" in c]
            cutList = sorted(cutList,key=lambda x: int(x[3:])) # make sure cuts are ordered as they are in the file
            self.cuts = [getattr(self.cutLib,c) for c in cutList]
        else: # cut file is in the same directory (e.g. running on condor)
            cutFileName = self.cutFile.split(".")[0]
            self.cutLib = importlib.import_module(cutFileName)
            cutList = [c for c in dir(self.cutLib) if "cut" in c]
            cutList = sorted(cutList,key=lambda x: int(x[3:])) # make sure cuts are ordered as they are in the file
            self.cuts = [getattr(self.cutLib,c) for c in cutList]
    
    def skim(self):
        with uproot.open(self.sampFile) as input_file:
            events = NanoEventsFactory.from_root(input_file,treepath="ntuples/outT",schemaclass=MySchema).events()
            info = self.sampleInfo
            sum_wgt = info["sum_wgt"]
            lumi, unc = getLumi(info['year'])
            xsec = self.sampleInfo['xsec']
            if info['type'] == 'signal':
                xsec = xsec*info['filter_eff']

            # register event weight branch
            events.__setitem__("eventWgt",xsec*lumi*events.genWgt/sum_wgt)
            # Preselection
            routines.selectGoodElesAndVertices(events)
            events.__setitem__("nGoodVtx",ak.count(events.good_vtx.vxy,axis=1))
            events = events[events.nGoodVtx > 0]
            
            # pre-computing quantities for cuts
            routines.selectBestVertex(events)
            # computing some signal-only diagnostic quantities
            if info['type'] == "signal":
                e1_match = routines.matchedVertexElectron(events,1)
                e2_match = routines.matchedVertexElectron(events,2)
                events["sel_vtx","match"] = ak.values_astype(ak.where(e1_match*e2_match == -1,2,ak.where(np.abs(e1_match)+np.abs(e2_match) > 0,1,0)),np.int32)
            else:
                events["sel_vtx","match"] = ak.zeros_like(events.sel_vtx.pt,dtype=np.int32)

            ###############################
            ######## CUTS & HISTOS ########
            ###############################
            for cut in self.cuts:
                events, cutName, cutDesc, savePlots = cut(events,info)
            
            if len(events.genWgt) == 0:
                pass
            else:
                output_tree = {}
                vtx_vars = [f for f in events.sel_vtx.fields if f!="e1" and f!="e2"]
                for v in vtx_vars:
                    if v == "e1_typ" or v == "e2_typ":
                        output_tree[f"sel_vtx_{v}"] = ak.Array(ak.where(events.sel_vtx[v]=="R",1,2).to_list())
                    elif v == "typ":
                        output_tree[f"sel_vtx_{v}"] = ak.Array(ak.where(events.sel_vtx[v]=="RR",1,ak.where(events.sel_vtx[v]=="LR",2,3)).to_list())
                    else:
                        output_tree[f"sel_vtx_{v}"] = ak.Array(events.sel_vtx[v].to_list())
                ele_fields = events.sel_vtx.e1.fields
                for ef in ele_fields:
                    output_tree[f"sel_e1_{ef}"] = ak.Array(events.sel_vtx.e1[ef].to_list())
                    output_tree[f"sel_e2_{ef}"] = ak.Array(events.sel_vtx.e2[ef].to_list())
                output_tree['genWgt'] = ak.Array(events.genWgt.to_list())
                output_tree['eventWgt'] = ak.Array(events.eventWgt.to_list())
                additional_fields = {
                                        'CaloMET':['ET','pt','phi'],
                                        'Photon':['et','eta','phi'],
                                        'PFMET':['ET','pt','phi'],
                                        'PFJet':['pt','eta','phi','bTag','METdPhi'],
                                        'Electron':["*"],
                                        'LptElectron':["*"],
                                        'vtx':["*"]
                                    }
                for af in additional_fields.keys():
                    subfields = additional_fields[af]
                    if subfields == ["*"]:
                        subfields = events[af].fields
                    allvars = {}
                    for subf in subfields:
                        if af == "vtx" and ("e1" in subf or "e2" in subf):
                            continue
                        if af == "vtx" and subf=="typ":
                            allvars[subf] = ak.Array(ak.where(events[af][subf]=="RR",1,ak.where(events[af][subf]=="LR",2,3)).to_list())
                        else:
                            allvars[subf] = ak.Array(events[af][subf].to_list())
                    output_tree[af] = ak.zip(allvars)
                with uproot.recreate(self.outFileName) as outfile:
                    outfile['outT'] = output_tree

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

def getLumi(year):
    # recommendations from https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2
    year = str(year)

    lumi, unc = 0, 0
    if year == '2016':
        lumi = 16.8
        unc = 0.012*lumi # 1.2 percent
    if year == '2016APV':
        lumi = 19.5
        unc = 0.012*lumi # 1.2 percent
    if year == '2017':
        lumi = 41.48
        unc = 0.023*lumi # 2.3 percent
    if year == '2018':
        lumi = 59.83
        unc = 0.025*lumi # 2.5 percent
    return lumi, unc

def loadSchema(fileLoc):
    loc = uproot.open(fileLoc)
    tree = NanoEventsFactory.from_root(loc,treepath="ntuples/outT",schemaclass=MySchema).events()
    return tree

def loadNano(fileLoc):
    loc = uproot.open(fileLoc)
    tree = NanoEventsFactory.from_root(loc,treepath="ntuples/outT",schemaclass=NanoAODSchema).events()
    return tree

def flatten_fillNone(arr,val):
    return ak.fill_none(ak.flatten(arr),val)
