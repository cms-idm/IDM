import ROOT
import json
import time
from argparse import ArgumentParser


elePassCut = r'''
std::vector<bool> elePassCut(
    const ROOT::VecOps::RVec<float>& pt,
    const ROOT::VecOps::RVec<float>& eta
) {
    std::vector<bool> v;

    for (size_t i = 0; i < pt.size(); i++) {
        v.push_back((pt.at(i) > 1.0) && (std::abs(eta.at(i)) < 2.4));
    }

    return v;
}
'''


isGoodVtx = r'''
bool passBoolIndex(int idx, const std::vector<bool>& vals) {
    if (idx < 0) {
        return false;
    }

    if (static_cast<size_t>(idx) >= vals.size()) {
        return false;
    }

    return vals.at(idx);
}

std::vector<bool> isGoodVtx(
    const ROOT::VecOps::RVec<std::string>& e1_typ,
    const ROOT::VecOps::RVec<std::string>& e2_typ,
    const ROOT::VecOps::RVec<int>& e1_ind,
    const ROOT::VecOps::RVec<int>& e2_ind,
    const std::vector<bool>& good_reg,
    const std::vector<bool>& good_lpt
) {
    std::vector<bool> good_vtx;

    for (size_t i = 0; i < e1_typ.size(); i++) {
        bool good_e1 = false;

        if (e1_typ.at(i) == "R") {
            good_e1 = passBoolIndex(e1_ind.at(i), good_reg);
        }
        else {
            good_e1 = passBoolIndex(e1_ind.at(i), good_lpt);
        }

        bool good_e2 = false;

        if (e2_typ.at(i) == "R") {
            good_e2 = passBoolIndex(e2_ind.at(i), good_reg);
        }
        else {
            good_e2 = passBoolIndex(e2_ind.at(i), good_lpt);
        }

        good_vtx.push_back(good_e1 && good_e2);
    }

    return good_vtx;
}
'''


passHEM = r'''
bool passHEM(int year, bool HEM_flag) {
    if (year == 2018) {
        return !HEM_flag;
    }

    return true;
}
'''


passbTagLoose = r'''
std::vector<bool> passbTagLoose(
    int year,
    const ROOT::VecOps::RVec<float>& btag,
    bool APV
) {
    float wp = 0.0;

    if ((year == 2016) && APV)  wp = 0.0508;
    if ((year == 2016) && !APV) wp = 0.0480;
    if (year == 2017)           wp = 0.0532;
    if (year == 2018)           wp = 0.0490;
    if (year == 2022)           wp = 0.0583;

    std::vector<bool> pass;

    for (size_t i = 0; i < btag.size(); i++) {
        pass.push_back(btag.at(i) > wp);
    }

    return pass;
}
'''


passbTagMed = r'''
std::vector<bool> passbTagMed(
    int year,
    const ROOT::VecOps::RVec<float>& btag,
    bool APV
) {
    float wp = 0.0;

    if ((year == 2016) && APV)  wp = 0.2598;
    if ((year == 2016) && !APV) wp = 0.2489;
    if (year == 2017)           wp = 0.3040;
    if (year == 2018)           wp = 0.2783;
    if (year == 2022)           wp = 0.3086;

    std::vector<bool> pass;

    for (size_t i = 0; i < btag.size(); i++) {
        pass.push_back(btag.at(i) > wp);
    }

    return pass;
}
'''


passbTagTight = r'''
std::vector<bool> passbTagTight(
    int year,
    const ROOT::VecOps::RVec<float>& btag,
    bool APV
) {
    float wp = 0.0;

    if ((year == 2016) && APV)  wp = 0.6502;
    if ((year == 2016) && !APV) wp = 0.6377;
    if (year == 2017)           wp = 0.7476;
    if (year == 2018)           wp = 0.7100;
    if (year == 2022)           wp = 0.7183;

    std::vector<bool> pass;

    for (size_t i = 0; i < btag.size(); i++) {
        pass.push_back(btag.at(i) > wp);
    }

    return pass;
}
'''


anyTrue = r'''
bool anyTrue(const ROOT::VecOps::RVec<bool>& vals) {
    for (size_t i = 0; i < vals.size(); i++) 
    {
        if (vals.at(i)) 
        {
            return true;
        }
    }

    return false;
}

bool anyTrue(const std::vector<bool>& vals) {
    for (size_t i = 0; i < vals.size(); i++) {
        if (vals.at(i)) {
            return true;
        }
    }

    return false;
}
'''


def year_for_cuts(year_label):
    year_string = str(year_label)

    if year_string == "2016APV":
        return 2016

    if year_string.startswith("2022"):
        return 2022

    return int(year_string)


def count_and_print(label, rdf, previous=None):
    count = rdf.Count().GetValue()

    if previous is None:
        print(f"{label:<45} {count}")
    else:
        if previous > 0:
            eff = count / previous
            print(f"{label:<45} {count}    step_eff = {eff:.6f}")
        else:
            print(f"{label:<45} {count}    step_eff = undefined")

    return count


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--jobname", required=True)
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("-m", "--met_cut", type=float, required=True)
    parser.add_argument("-j", "--njet_cut", required=True, type=int)

    args = parser.parse_args()

    jobname = args.jobname
    outDir = args.outdir.rstrip("/")
    MET_cut = args.met_cut
    nJet_cut = args.njet_cut

    t = time.time()

    ROOT.gInterpreter.GenerateDictionary(
        "ROOT::VecOps::RVec<std::vector<float> >",
        "vector;ROOT/RVec.hxx",
    )

    ROOT.gInterpreter.GenerateDictionary(
        "ROOT::VecOps::RVec<std::string>",
        "string;ROOT/RVec.hxx",
    )

    ROOT.gInterpreter.GenerateDictionary(
        "std::vector<std::vector<float> >",
        "vector",
    )

    ROOT.gInterpreter.Declare(elePassCut)
    ROOT.gInterpreter.Declare(isGoodVtx)
    ROOT.gInterpreter.Declare(passHEM)
    ROOT.gInterpreter.Declare(passbTagLoose)
    ROOT.gInterpreter.Declare(passbTagMed)
    ROOT.gInterpreter.Declare(passbTagTight)
    ROOT.gInterpreter.Declare(anyTrue)

    ROOT.EnableImplicitMT()

    print(f"set up ROOT in {(time.time() - t) / 60.0} mins")

    t = time.time()

    with open("samples.json", "r") as fin:
        samps = json.load(fin)

    files = samps["fileset"]

    blacklist = samps.get("blacklist", [])
    files = [f for f in files if f.split("/")[-1] not in blacklist]

    year_label = samps["year"]
    year = year_for_cuts(year_label)
    is_apv = str(year_label) == "2016APV"

    print("========================================")
    print("Input sample info")
    print("========================================")
    print(f"jobname             = {jobname}")
    print(f"outDir              = {outDir}")
    print(f"MET_cut             = {MET_cut}")
    print(f"nJet_cut            = {nJet_cut}")
    print(f"sample year label   = {year_label}")
    print(f"year used for cuts  = {year}")
    print(f"is APV              = {is_apv}")
    print(f"number of files     = {len(files)}")
    print("========================================")

    d = ROOT.RDataFrame("ntuples/outT", files)

    print(f"loaded RDF in {(time.time() - t) / 60.0} mins")

    t = time.time()

    if is_apv:
        d = d.Define("APV", "true")
    else:
        d = d.Define("APV", "false")

    d = d.Define("year", f"{year}")

    d = d.Define(
        "Electron_passCut",
        "elePassCut(Electron_pt, Electron_eta)",
    )

    d = d.Define(
        "LptElectron_passCut",
        "elePassCut(LptElectron_pt, LptElectron_eta)",
    )

    # Keep this branch only for diagnostics.
    # For the iDMmu skim, we do NOT require this electron-based vertex condition.
    d = d.Define(
        "vtx_isGood",
        "isGoodVtx("
        "vtx_e1_typ, "
        "vtx_e2_typ, "
        "vtx_e1_idx, "
        "vtx_e2_idx, "
        "Electron_passCut, "
        "LptElectron_passCut"
        ")",
    )

    d = d.Define(
        "passHEMveto",
        "passHEM(year, HEM_flag)",
    )

    d = d.Define(
        "PFJet_bLoose",
        "passbTagLoose(year, PFJet_bTag, APV)",
    )

    d = d.Define(
        "PFJet_bMed",
        "passbTagMed(year, PFJet_bTag, APV)",
    )

    d = d.Define(
        "PFJet_bTight",
        "passbTagTight(year, PFJet_bTag, APV)",
    )

    d = d.Define(
        "anyB_loose",
        "anyTrue(PFJet_bLoose)",
    )

    d = d.Define(
        "anyB_med",
        "anyTrue(PFJet_bMed)",
    )

    d = d.Define(
        "anyB_tight",
        "anyTrue(PFJet_bTight)",
    )

    print(f"set up branches in {(time.time() - t) / 60.0} mins")

    if nJet_cut > 0:
        njet_filter = f"(nPFJet > 0) && (nPFJet < {nJet_cut})"
    else:
        # Keep current behavior:
        # -j 0 means nPFJet > 0, matching the current nJetsG0 label.
        njet_filter = "nPFJet > 0"

    # For iDMmu, disable electron good-vertex requirement.
    # We keep a diagnostic count of anyTrue(vtx_isGood), but it is not applied.
    good_vertex_filter = "1"

    print("========================================")
    print("Cut definitions")
    print("========================================")
    print("good vertex cut  = DISABLED for iDMmu; using 1")
    print("good vertex diagnostic only = anyTrue(vtx_isGood)")
    print("MET filter cut   = METFiltersFailBits == 0")
    print("HEM veto cut     = passHEMveto")
    print("MET trigger cut  = trig_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight == 1")
    print(f"MET cut          = PFMET_pt > {MET_cut}")
    print(f"nJet cut         = {njet_filter}")
    print("========================================")

    print("========================================")
    print("Individual cut counts from initial sample")
    print("========================================")

    initial = count_and_print("initial", d)

    count_and_print(
        "pass good vertex diagnostic alone",
        d.Filter("anyTrue(vtx_isGood)"),
        initial,
    )

    count_and_print(
        "pass good vertex applied alone",
        d.Filter(good_vertex_filter),
        initial,
    )

    count_and_print(
        "pass MET filters alone",
        d.Filter("METFiltersFailBits == 0"),
        initial,
    )

    count_and_print(
        "pass HEM veto alone",
        d.Filter("passHEMveto"),
        initial,
    )

    count_and_print(
        "pass MET trigger alone",
        d.Filter("trig_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight == 1"),
        initial,
    )

    count_and_print(
        f"pass PFMET > {MET_cut} alone",
        d.Filter(f"PFMET_pt > {MET_cut}"),
        initial,
    )

    count_and_print(
        "pass nJet cut alone",
        d.Filter(njet_filter),
        initial,
    )

    print("========================================")
    print("Sequential cutflow")
    print("========================================")

    d0 = d
    n0 = count_and_print("initial", d0)

    # Good vertex disabled for iDMmu.
    d1 = d0.Filter(good_vertex_filter)
    n1 = count_and_print("after good vertex disabled", d1, n0)

    d2 = d1.Filter("METFiltersFailBits == 0")
    n2 = count_and_print("after MET filters", d2, n1)

    d3 = d2.Filter("passHEMveto")
    n3 = count_and_print("after HEM veto", d3, n2)

    d4 = d3.Filter("trig_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight == 1")
    n4 = count_and_print("after MET trigger", d4, n3)

    d5 = d4.Filter(f"PFMET_pt > {MET_cut}")
    n5 = count_and_print(f"after PFMET > {MET_cut}", d5, n4)

    d6 = d5.Filter(njet_filter)
    final = count_and_print("after nJet cut / final", d6, n5)

    print("========================================")
    print(f"final = {final}")
    print("========================================")

    if final > 0:
        output_file = f"root://cmseos.fnal.gov/{outDir}/{jobname}.root"
        print(f"writing output: {output_file}")
        d6.Snapshot("ntuples/outT", output_file)
    else:
        print("final = 0, skipping Snapshot")

    print(f"filtered and output in {(time.time() - t) / 60.0} mins")

    del d




# #!/usr/bin/env python
# import os, stat
# import ROOT
# import json
# import sys
# import numpy as np
# import time
# from argparse import ArgumentParser

# elePassCut = '''
# std::vector<bool> elePassCut(ROOT::VecOps::RVec<float> pt, ROOT::VecOps::RVec<float> eta) {
#     std::vector<bool> v;
#     for (int i = 0; i < pt.size(); i++) {
#         v.push_back((pt.at(i) > 1) && (std::abs(eta.at(i)) < 2.4));
#     }
#     return v;
# }
# '''
# isGoodVtx = '''
# std::vector<bool> isGoodVtx(ROOT::VecOps::RVec<string> e1_typ, ROOT::VecOps::RVec<string> e2_typ, ROOT::VecOps::RVec<int> e1_ind, ROOT::VecOps::RVec<int> e2_ind, ROOT::VecOps::RVec<bool> good_reg, ROOT::VecOps::RVec<bool> good_lpt) {
#     vector<bool> good_vtx;
#     for (int i = 0; i < e1_typ.size(); i++) {
#         bool good_e1 = false;
#         if (e1_typ.at(i) == "R") {
#             if (good_reg.at(e1_ind.at(i))) {
#                 good_e1 = true;
#             }
#         } 
#         else {
#             if (good_lpt.at(e1_ind.at(i))) {
#                 good_e1 = true;
#             }
#         }
        
#         bool good_e2 = false;
#         if (e2_typ.at(i) == "R") {
#             if (good_reg.at(e2_ind.at(i))) {
#                 good_e2 = true;
#             }
#         } 
#         else {
#             if (good_lpt.at(e2_ind.at(i))) {
#                 good_e2 = true;
#             }
#         }
        
#         good_vtx.push_back(good_e1 && good_e2);
#     }
#     return good_vtx;
# }
# '''
# passHEM = '''
# bool passHEM(int year, bool HEM_flag) {
#     bool pass;
#     if (year == 2018) {
#         pass = !HEM_flag;
#     }
#     else {
#         pass = true;
#     }
#     return pass;
# }
# '''
# passMETtrig = '''
# bool passMETtrig(int year, unsigned int fired16, unsigned int fired17, unsigned int fired18) {
#     bool pass;
#     if (year == 2016) {
#         pass = ((fired16 & (1<<9)) == (1<<9));
#     }
#     else if (year == 2017) {
#         pass = ((fired17 & (1<<9)) == (1<<9));
#     }
#     else if (year == 2018) {
#         pass = ((fired18 & (1<<13)) == (1<<13));
#     }
#     return pass;
# }
# '''
# passbTagLoose = '''
# vector<bool> passbTagLoose(int year, ROOT::VecOps::RVec<float> btag, bool APV) {
#     float wp;
#     if ((year==2016) && APV) wp = 0.0508;
#     if ((year==2016) && !APV) wp = 0.0480;
#     if (year==2017) wp = 0.0532;
#     if (year==2018) wp = 0.0490;
#     vector<bool> pass;
#     for (int i = 0; i < btag.size(); i++) {
#         pass.push_back(btag.at(i) > wp);
#     }
#     return pass;
# }
# '''
# passbTagMed = '''
# vector<bool> passbTagMed(int year, ROOT::VecOps::RVec<float> btag, bool APV) {
#     float wp;
#     if ((year==2016) && APV) wp = 0.2598;
#     if ((year==2016) && !APV) wp = 0.2489;
#     if (year==2017) wp = 0.3040;
#     if (year==2018) wp = 0.2783;
#     vector<bool> pass;
#     for (int i = 0; i < btag.size(); i++) {
#         pass.push_back(btag.at(i) > wp);
#     }
#     return pass;
# }
# '''
# passbTagTight = '''
# vector<bool> passbTagTight(int year, ROOT::VecOps::RVec<float> btag, bool APV) {
#     float wp;
#     if ((year==2016) && APV) wp = 0.6502;
#     if ((year==2016) && !APV) wp = 0.6377;
#     if (year==2017) wp = 0.7476;
#     if (year==2018) wp = 0.7100;
#     vector<bool> pass;
#     for (int i = 0; i < btag.size(); i++) {
#         pass.push_back(btag.at(i) > wp);
#     }
#     return pass;
# }
# '''
# anyTrue = '''
# bool anyTrue(ROOT::VecOps::RVec<bool> vals) {
#     bool any = false;
#     for (int i = 0; i < vals.size(); i++) {
#         if (vals.at(i)) {
#             any = true;
#             break;
#         }
#     }
#     return any;
# }
# '''
# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("-n","--jobname",required=True)
#     parser.add_argument("-o","--outdir",required=True)
#     parser.add_argument("-m","--met_cut",type=float,required=True)
#     parser.add_argument("-j","--njet_cut",required=True,type=int)
#     args = parser.parse_args()

#     jobname = args.jobname
#     outDir = args.outdir
#     MET_cut = args.met_cut
#     nJet_cut = args.njet_cut

#     t = time.time()
#     ROOT.gInterpreter.GenerateDictionary("ROOT::VecOps::RVec<vector<float> >", "vector;ROOT/RVec.hxx")
#     ROOT.gInterpreter.GenerateDictionary("ROOT::VecOps::RVec<string>", "string;ROOT/RVec.hxx")
#     ROOT.gInterpreter.Declare(elePassCut)
#     ROOT.gInterpreter.Declare(isGoodVtx)
#     ROOT.gInterpreter.Declare(passHEM)
#     #ROOT.gInterpreter.Declare(passMETtrig)
#     ROOT.gInterpreter.Declare(passbTagLoose)
#     ROOT.gInterpreter.Declare(passbTagMed)
#     ROOT.gInterpreter.Declare(passbTagTight)
#     ROOT.gInterpreter.Declare(anyTrue)
#     ROOT.gInterpreter.GenerateDictionary("vector<vector<float> >","vector")
#     ROOT.EnableImplicitMT()
#     print(f"set up root in {(time.time() - t)/60} mins")
#     t = time.time()

#     with open('samples.json','r') as fin:
#         samps = json.load(fin)
#     files = samps['fileset']
#     files = [f for f in files if f.split("/")[-1] not in samps['blacklist']]
#     year = samps['year']
#     d = ROOT.RDataFrame("ntuples/outT",files)
#     print(f"loaded RDF in {(time.time() - t)/60} mins")
#     t = time.time()
#     if year == "2016APV":
#         d = d.Define("APV","true")
#     else:
#         d = d.Define("APV","false")
#     d = d.Define("year",f"{int(year)}")
#     d = d.Define("Electron_passCut","elePassCut(Electron_pt,Electron_eta)")
#     d = d.Define("LptElectron_passCut","elePassCut(LptElectron_pt,LptElectron_eta)")
#     d = d.Define("vtx_isGood","isGoodVtx(vtx_e1_typ, vtx_e2_typ, vtx_e1_idx, vtx_e2_idx, Electron_passCut, LptElectron_passCut)")
#     d = d.Define("passHEMveto","passHEM(year,HEM_flag)")
#     d = d.Define("PFJet_bLoose","passbTagLoose(year,PFJet_bTag,APV)")
#     d = d.Define("PFJet_bMed","passbTagMed(year,PFJet_bTag,APV)")
#     d = d.Define("PFJet_bTight","passbTagTight(year,PFJet_bTag,APV)")
#     d = d.Define("anyB_loose","anyTrue(PFJet_bLoose)")
#     d = d.Define("anyB_med","anyTrue(PFJet_bMed)")
#     d = d.Define("anyB_tight","anyTrue(PFJet_bTight)")
#     print(f"set up branches {(time.time() - t)/60} mins")
#     t = time.time()
#     print(f"initial = {d.Count().GetValue()}")
    
#     if nJet_cut > 0:
#         njet_filter = f"(nPFJet > 0) && (nPFJet < {nJet_cut})"
#     else:
#         njet_filter = "nPFJet > 0"
#     d = d.Filter("anyTrue(vtx_isGood)") \
#         .Filter("METFiltersFailBits == 0") \
#         .Filter("passHEMveto") \
#         .Filter("trig_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight == 1")
#         .Filter(f"PFMET_pt > {MET_cut}") \
#         .Filter(njet_filter) \
#     final = d.Count().GetValue()
#     print(f"final = {final}")
#     if final > 0:
#         d.Snapshot("ntuples/outT",f"root://cmseos.fnal.gov/{outDir}/{jobname}.root")
#     print(f"filtered and output in {(time.time() - t)/60} mins")
#     del d
