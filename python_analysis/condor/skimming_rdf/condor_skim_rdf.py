#!/usr/bin/env python
import os, stat
import ROOT
import json
import sys
import numpy as np
import time
from argparse import ArgumentParser

elePassCut = '''
std::vector<bool> elePassCut(ROOT::VecOps::RVec<float> pt, ROOT::VecOps::RVec<float> eta) {
    std::vector<bool> v;
    for (int i = 0; i < pt.size(); i++) {
        v.push_back((pt.at(i) > 1) && (std::abs(eta.at(i)) < 2.4));
    }
    return v;
}
'''
isGoodVtx = '''
std::vector<bool> isGoodVtx(ROOT::VecOps::RVec<string> e1_typ, ROOT::VecOps::RVec<string> e2_typ, ROOT::VecOps::RVec<int> e1_ind, ROOT::VecOps::RVec<int> e2_ind, ROOT::VecOps::RVec<bool> good_reg, ROOT::VecOps::RVec<bool> good_lpt) {
    vector<bool> good_vtx;
    for (int i = 0; i < e1_typ.size(); i++) {
        bool good_e1 = false;
        if (e1_typ.at(i) == "R") {
            if (good_reg.at(e1_ind.at(i))) {
                good_e1 = true;
            }
        } 
        else {
            if (good_lpt.at(e1_ind.at(i))) {
                good_e1 = true;
            }
        }
        
        bool good_e2 = false;
        if (e2_typ.at(i) == "R") {
            if (good_reg.at(e2_ind.at(i))) {
                good_e2 = true;
            }
        } 
        else {
            if (good_lpt.at(e2_ind.at(i))) {
                good_e2 = true;
            }
        }
        
        good_vtx.push_back(good_e1 && good_e2);
    }
    return good_vtx;
}
'''
passHEM = '''
bool passHEM(int year, bool HEM_flag) {
    bool pass;
    if (year == 2018) {
        pass = !HEM_flag;
    }
    else {
        pass = true;
    }
    return pass;
}
'''
passMETtrig = '''
bool passMETtrig(int year, unsigned int fired16, unsigned int fired17, unsigned int fired18) {
    bool pass;
    if (year == 2016) {
        pass = ((fired16 & (1<<9)) == (1<<9));
    }
    else if (year == 2017) {
        pass = ((fired17 & (1<<9)) == (1<<9));
    }
    else if (year == 2018) {
        pass = ((fired18 & (1<<13)) == (1<<13));
    }
    return pass;
}
'''
passbTagLoose = '''
vector<bool> passbTagLoose(int year, ROOT::VecOps::RVec<float> btag, bool APV) {
    float wp;
    if ((year==2016) && APV) wp = 0.0508;
    if ((year==2016) && !APV) wp = 0.0480;
    if (year==2017) wp = 0.0532;
    if (year==2018) wp = 0.0490;
    vector<bool> pass;
    for (int i = 0; i < btag.size(); i++) {
        pass.push_back(btag.at(i) > wp);
    }
    return pass;
}
'''
passbTagMed = '''
vector<bool> passbTagMed(int year, ROOT::VecOps::RVec<float> btag, bool APV) {
    float wp;
    if ((year==2016) && APV) wp = 0.2598;
    if ((year==2016) && !APV) wp = 0.2489;
    if (year==2017) wp = 0.3040;
    if (year==2018) wp = 0.2783;
    vector<bool> pass;
    for (int i = 0; i < btag.size(); i++) {
        pass.push_back(btag.at(i) > wp);
    }
    return pass;
}
'''
passbTagTight = '''
vector<bool> passbTagTight(int year, ROOT::VecOps::RVec<float> btag, bool APV) {
    float wp;
    if ((year==2016) && APV) wp = 0.6502;
    if ((year==2016) && !APV) wp = 0.6377;
    if (year==2017) wp = 0.7476;
    if (year==2018) wp = 0.7100;
    vector<bool> pass;
    for (int i = 0; i < btag.size(); i++) {
        pass.push_back(btag.at(i) > wp);
    }
    return pass;
}
'''
anyTrue = '''
bool anyTrue(ROOT::VecOps::RVec<bool> vals) {
    bool any = false;
    for (int i = 0; i < vals.size(); i++) {
        if (vals.at(i)) {
            any = true;
            break;
        }
    }
    return any;
}
'''
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n","--jobname",required=True)
    parser.add_argument("-o","--outdir",required=True)
    parser.add_argument("-m","--met_cut",type=float,required=True)
    parser.add_argument("-j","--njet_cut",required=True,type=int)
    args = parser.parse_args()

    jobname = args.jobname
    outDir = args.outdir
    MET_cut = args.met_cut
    nJet_cut = args.njet_cut

    t = time.time()
    ROOT.gInterpreter.GenerateDictionary("ROOT::VecOps::RVec<vector<float> >", "vector;ROOT/RVec.hxx")
    ROOT.gInterpreter.GenerateDictionary("ROOT::VecOps::RVec<string>", "string;ROOT/RVec.hxx")
    ROOT.gInterpreter.Declare(elePassCut)
    ROOT.gInterpreter.Declare(isGoodVtx)
    ROOT.gInterpreter.Declare(passHEM)
    #ROOT.gInterpreter.Declare(passMETtrig)
    ROOT.gInterpreter.Declare(passbTagLoose)
    ROOT.gInterpreter.Declare(passbTagMed)
    ROOT.gInterpreter.Declare(passbTagTight)
    ROOT.gInterpreter.Declare(anyTrue)
    ROOT.gInterpreter.GenerateDictionary("vector<vector<float> >","vector")
    ROOT.EnableImplicitMT()
    print(f"set up root in {(time.time() - t)/60} mins")
    t = time.time()

    with open('samples.json','r') as fin:
        samps = json.load(fin)
    files = samps['fileset']
    files = [f for f in files if f.split("/")[-1] not in samps['blacklist']]
    year = samps['year']
    d = ROOT.RDataFrame("ntuples/outT",files)
    print(f"loaded RDF in {(time.time() - t)/60} mins")
    t = time.time()
    if year == "2016APV":
        d = d.Define("APV","true")
    else:
        d = d.Define("APV","false")
    d = d.Define("year",f"{int(year)}")
    d = d.Define("Electron_passCut","elePassCut(Electron_pt,Electron_eta)")
    d = d.Define("LptElectron_passCut","elePassCut(LptElectron_pt,LptElectron_eta)")
    d = d.Define("vtx_isGood","isGoodVtx(vtx_e1_typ, vtx_e2_typ, vtx_e1_idx, vtx_e2_idx, Electron_passCut, LptElectron_passCut)")
    d = d.Define("passHEMveto","passHEM(year,HEM_flag)")
    d = d.Define("PFJet_bLoose","passbTagLoose(year,PFJet_bTag,APV)")
    d = d.Define("PFJet_bMed","passbTagMed(year,PFJet_bTag,APV)")
    d = d.Define("PFJet_bTight","passbTagTight(year,PFJet_bTag,APV)")
    d = d.Define("anyB_loose","anyTrue(PFJet_bLoose)")
    d = d.Define("anyB_med","anyTrue(PFJet_bMed)")
    d = d.Define("anyB_tight","anyTrue(PFJet_bTight)")
    print(f"set up branches {(time.time() - t)/60} mins")
    t = time.time()
    print(f"initial = {d.Count().GetValue()}")
    
    if nJet_cut > 0:
        njet_filter = f"(nPFJet > 0) && (nPFJet < {nJet_cut})"
    else:
        njet_filter = "nPFJet > 0"
    d = d.Filter("anyTrue(vtx_isGood)") \
        .Filter("METFiltersFailBits == 0") \
        .Filter("passHEMveto") \
        .Filter("trig_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight == 1")
        .Filter(f"PFMET_pt > {MET_cut}") \
        .Filter(njet_filter) \
    final = d.Count().GetValue()
    print(f"final = {final}")
    if final > 0:
        d.Snapshot("ntuples/outT",f"root://cmseos.fnal.gov/{outDir}/{jobname}.root")
    print(f"filtered and output in {(time.time() - t)/60} mins")
    del d
