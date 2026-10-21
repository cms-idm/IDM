from histobins import *
from hist import Hist
import hist
import numpy as np
import awkward as ak

def make_histograms():
    histograms = {
        "bdt_score" : Hist(samp,cut,bdt_score,storage=hist.storage.Weight()),
        "abs1overEm1overP" : Hist(samp,cut,abs1overEm1overP,storage=hist.storage.Weight()),
        
        # selected vertex displacement
        "sel_vtx_vxy": Hist(samp,cut,vxy,storage=hist.storage.Weight()),
        "sel_vtx_vxy_projected": Hist(samp,cut,vxy_projected,storage=hist.storage.Weight()),
        "sel_vtx_minDxy": Hist(samp,cut,ele_dxy,storage=hist.storage.Weight()),
        "sel_vtx_dxy1": Hist(samp,cut,ele_dxy,storage=hist.storage.Weight()),
        "sel_vtx_dxy2": Hist(samp,cut,ele_dxy,storage=hist.storage.Weight()),
        "sel_vtx_minDz": Hist(samp,cut,ele_dz,storage=hist.storage.Weight()),
        "sel_vtx_deltaDxy": Hist(samp,cut,ele_dxy,storage=hist.storage.Weight()),
        "sel_vtx_deltaDz": Hist(samp,cut,ele_dz,storage=hist.storage.Weight()),
        "sel_vtx_vxySignif": Hist(samp,cut,vtx_vxySignif,storage=hist.storage.Weight()),

        # selected vertex quantities
        "sel_vtx_type": Hist(samp,cut,vtx_type,storage=hist.storage.Weight()),
        "sel_vtx_matchType" : Hist(samp,cut,vtx_matchType,storage=hist.storage.Weight()),
        "sel_vtx_dR_zoom": Hist(samp,cut,dR_zoom,storage=hist.storage.Weight()),
        "sel_vtx_chi2": Hist(samp,cut,vtx_chi2,storage=hist.storage.Weight()),
        "sel_vtx_dR": Hist(samp,cut,dR,storage=hist.storage.Weight()),
        "sel_vtx_dEta": Hist(samp,cut,deta,storage=hist.storage.Weight()),
        "sel_vtx_dPhi": Hist(samp,cut,dphi,storage=hist.storage.Weight()),
        "sel_vtx_sign_eta": Hist(samp,cut,sign_eta,storage=hist.storage.Weight()),
        "sel_vtx_prod_eta": Hist(samp,cut,prod_eta,storage=hist.storage.Weight()),
        "sel_vtx_METdPhi": Hist(samp,cut,dphi,storage=hist.storage.Weight()),
        "sel_vtx_pt": Hist(samp,cut,ele_pt,storage=hist.storage.Weight()),
        "sel_vtx_eta": Hist(samp,cut,ele_eta,storage=hist.storage.Weight()),
        "sel_vtx_phi": Hist(samp,cut,ele_phi,storage=hist.storage.Weight()),
        "sel_vtx_mass": Hist(samp,cut,mass,storage=hist.storage.Weight()),

        "sel_vtx_pt_e1_over_pt_e2": Hist(samp,cut,ratio,storage=hist.storage.Weight()),
        "delta_dxy_over_mindxy": Hist(samp,cut,ratio_big,storage=hist.storage.Weight()),
        "delta_dxy_over_maxdxy": Hist(samp,cut,ratio,storage=hist.storage.Weight()),
        "delta_dxy_over_meandxy": Hist(samp,cut,ratio_big,storage=hist.storage.Weight()),

        "sel_vtx_pt_over_m": Hist(samp,cut,sel_vtx_pt_over_m,storage=hist.storage.Weight()),
        "sel_vtx_dEta_over_dPhi": Hist(samp,cut,deta_over_dphi,storage=hist.storage.Weight()),
        "sel_vtx_log_dEta_over_dPhi": Hist(samp,cut,deta_over_dphi,storage=hist.storage.Weight()),

        # e+/-
        "nLpt_Electron": Hist(samp,cut,hist.axis.Integer(0,10,name="num_ele"),storage=hist.storage.Weight()),
        "nPF_Electron": Hist(samp,cut,hist.axis.Integer(0,10,name="num_ele"),storage=hist.storage.Weight()),
        "nElectron": Hist(samp,cut,hist.axis.Integer(0,10,name="num_ele"),storage=hist.storage.Weight()),

        "nGoodVtx": Hist(samp,cut,hist.axis.Integer(0,10,name="num_good_vtx"),storage=hist.storage.Weight()),

        "met_over_lead_jet_pt": Hist(samp,cut,met_over_pt,storage=hist.storage.Weight()),

        "ctau": Hist(samp,cut,ctau,storage=hist.storage.Weight()),
        "cos_collinear": Hist(samp,cut,cos_collinear,storage=hist.storage.Weight()),
        "projectedLxy": Hist(samp,cut,vxy_projected,storage=hist.storage.Weight()),
        "abs_cos_collinear": Hist(samp,cut,cos_collinear,storage=hist.storage.Weight()),
        # gen ee dxy, vxy

        "gen_ee_pt": Hist(samp,cut,ele_pt,storage=hist.storage.Weight()),
        "gen_chi2_pt": Hist(samp,cut,chi2_pt,storage=hist.storage.Weight()),
        "gen_chi2_ee_pt_ratio": Hist(samp,cut,ratio,storage=hist.storage.Weight()),

        # 2D selected vertex histos
        "ctau_vs_Lxy": Hist(samp,cut,ctau,vxy,storage=hist.storage.Weight()),
        "gen_chi2_pt_vs_gen_ee_pt": Hist(samp,cut,chi2_pt,ele_pt,storage=hist.storage.Weight()),
        
        "sel_vtx_vxy_vs_mindxy" : Hist(samp,cut,vxy,dxy,storage=hist.storage.Weight()),

        "dxy1_vs_dxy2" : Hist(samp,cut,dxy1,dxy2,storage=hist.storage.Weight()),
        "sel_vtx_vxy_vs_matchType" : Hist(samp,cut,vxy,vtx_matchType,storage=hist.storage.Weight()),

        "sel_vtx_vxy_vs_sel_vtx_chi2" : Hist(samp,cut,vxy,vtx_chi2,storage=hist.storage.Weight()),
        "sel_vtx_vxy_vs_sel_vtx_dR" : Hist(samp,cut,vxy,dR,storage=hist.storage.Weight()),
        "sel_vtx_vxy_vs_sel_vtx_mass" : Hist(samp,cut,vxy,mass,storage=hist.storage.Weight()),
        "sel_vtx_vxy_vs_sel_vtx_METdPhi" : Hist(samp,cut,vxy,dphi,storage=hist.storage.Weight()),
        "sel_vtx_vxy_vs_sel_vtx_vxySignif" : Hist(samp,cut,vxy,vtx_vxySignif,storage=hist.storage.Weight()),
        "sel_vtx_vxy_vs_sel_vtx_pt" : Hist(samp,cut,vxy,ele_pt,storage=hist.storage.Weight()),
        "sel_vtx_vxy_vs_sel_vtx_sign_eta" : Hist(samp,cut,vxy,sign_eta,storage=hist.storage.Weight()),
        
        "sel_vtx_minDxy_vs_matchType" : Hist(samp,cut,ele_dxy,vtx_matchType,storage=hist.storage.Weight()),
        "sel_vtx_minDz_vs_matchType" : Hist(samp,cut,ele_dz,vtx_matchType,storage=hist.storage.Weight()),
        "sel_vtx_deltaDxy_vs_matchType" : Hist(samp,cut,ele_dxy,vtx_matchType,storage=hist.storage.Weight()),
        "sel_vtx_deltaDz_vs_matchType" : Hist(samp,cut,ele_dz,vtx_matchType,storage=hist.storage.Weight()),
        "sel_vtx_vxySignif_vs_matchType" : Hist(samp,cut,vtx_vxySignif,vtx_matchType,storage=hist.storage.Weight()),
        "sel_vtx_chi2_vs_matchType" : Hist(samp,cut,vtx_chi2,vtx_matchType,storage=hist.storage.Weight()),
        "sel_vtx_dR_vs_matchType" : Hist(samp,cut,dR,vtx_matchType,storage=hist.storage.Weight()),
        "sel_vtx_dEta_vs_matchType" : Hist(samp,cut,deta,vtx_matchType,storage=hist.storage.Weight()),
        "sel_vtx_dPhi_vs_matchType" : Hist(samp,cut,dphi,vtx_matchType,storage=hist.storage.Weight()),
        "sel_vtx_METdPhi_vs_matchType" : Hist(samp,cut,dphi,vtx_matchType,storage=hist.storage.Weight()),
        "sel_vtx_pt_vs_matchType" : Hist(samp,cut,ele_pt,vtx_matchType,storage=hist.storage.Weight()),
        "sel_vtx_eta_vs_matchType" : Hist(samp,cut,ele_eta,vtx_matchType,storage=hist.storage.Weight()),
        "sel_vtx_phi_vs_matchType" : Hist(samp,cut,ele_phi,vtx_matchType,storage=hist.storage.Weight()),
        "sel_vtx_mass_vs_matchType" : Hist(samp,cut,mass,vtx_matchType,storage=hist.storage.Weight()),
        "sel_vtx_type_vs_matchType" : Hist(samp,cut,vtx_type,vtx_matchType,storage=hist.storage.Weight()),

        "sel_vtx_sign_eta_vs_matchType" : Hist(samp,cut,sign_eta,vtx_matchType,storage=hist.storage.Weight()),

        "nLpt_Electron_vs_matchType" : Hist(samp,cut,hist.axis.Integer(0,10,name="num_ele"),vtx_matchType,storage=hist.storage.Weight()),
        "nPF_Electron_vs_matchType" : Hist(samp,cut,hist.axis.Integer(0,10,name="num_ele"),vtx_matchType,storage=hist.storage.Weight()),
        "nElectron_vs_matchType" : Hist(samp,cut,hist.axis.Integer(0,10,name="num_ele"),vtx_matchType,storage=hist.storage.Weight()),
        "nGoodVtx_vs_matchType" : Hist(samp,cut,hist.axis.Integer(0,10,name="num_good_vtx"),vtx_matchType,storage=hist.storage.Weight()),
        "nGoodVtx_vs_sel_vtx_type" : Hist(samp,cut,hist.axis.Integer(0,10,name="num_good_vtx"),vtx_type,storage=hist.storage.Weight()),
        
        "sel_vtx_dR_vs_sel_vtx_vxy" : Hist(samp,cut,dR,vxy,storage=hist.storage.Weight()),
        "sel_vtx_dR_vs_sel_vtx_mass" : Hist(samp,cut,dR,mass,storage=hist.storage.Weight()),
        "sel_vtx_dR_vs_sel_vtx_minDxy" : Hist(samp,cut,dR,ele_dxy,storage=hist.storage.Weight()),
        "sel_vtx_dR_vs_sel_vtx_chi2" : Hist(samp,cut,dR,vtx_chi2,storage=hist.storage.Weight()),
        "sel_vtx_dR_vs_sel_vtx_vxySignif" : Hist(samp,cut,dR,vtx_vxySignif,storage=hist.storage.Weight()),
        "sel_vtx_dR_vs_sel_vtx_METdPhi" : Hist(samp,cut,dR,dphi,storage=hist.storage.Weight()),
        "sel_vtx_dR_vs_bdt_score" : Hist(samp,cut,dR,bdt_score,storage=hist.storage.Weight()),
        "sel_vtx_dR_vs_sel_vtx_pt_e1_over_pt_e2" : Hist(samp,cut,dR,ratio,storage=hist.storage.Weight()),
        "sel_vtx_dR_vs_sel_vtx_type": Hist(samp,cut,dR,vtx_type,storage=hist.storage.Weight()),

        "sel_vtx_dR_zoom_vs_sel_vtx_vxy" : Hist(samp,cut,dR_zoom,vxy,storage=hist.storage.Weight()),
        "sel_vtx_dR_zoom_vs_sel_vtx_mass" : Hist(samp,cut,dR_zoom,mass,storage=hist.storage.Weight()),
        "sel_vtx_dR_zoom_vs_sel_vtx_minDxy" : Hist(samp,cut,dR_zoom,ele_dxy,storage=hist.storage.Weight()),
        "sel_vtx_dR_zoom_vs_sel_vtx_chi2" : Hist(samp,cut,dR_zoom,vtx_chi2,storage=hist.storage.Weight()),
        "sel_vtx_dR_zoom_vs_sel_vtx_vxySignif" : Hist(samp,cut,dR_zoom,vtx_vxySignif,storage=hist.storage.Weight()),
        "sel_vtx_dR_zoom_vs_sel_vtx_METdPhi" : Hist(samp,cut,dR_zoom,dphi,storage=hist.storage.Weight()),
        "sel_vtx_dR_zoom_vs_bdt_score" : Hist(samp,cut,dR_zoom,bdt_score,storage=hist.storage.Weight()),
        "sel_vtx_dR_zoom_vs_sel_vtx_pt_e1_over_pt_e2" : Hist(samp,cut,dR_zoom,ratio,storage=hist.storage.Weight()),
        "sel_vtx_dR_zoom_vs_sel_vtx_type": Hist(samp,cut,dR_zoom,vtx_type,storage=hist.storage.Weight()),
        
        }
    return histograms

subroutines = []
    

def fillHistos(events,histos,samp,cut,info,sum_wgt=1):
    e1 = events.sel_vtx.e1
    e2 = events.sel_vtx.e2

    min_dxy = np.minimum(np.abs(e1.dxy),np.abs(e2.dxy))
    max_dxy = np.maximum(np.abs(e1.dxy),np.abs(e2.dxy))
    mean_dxy = np.mean(np.abs(e1.dxy),np.abs(e2.dxy))
    delta_dxy = np.abs(np.abs(e1.dxy)-np.abs(e2.dxy))

    min_dz = np.minimum(np.abs(e1.dz),np.abs(e2.dz))
    delta_dz = np.abs(np.abs(e1.dz)-np.abs(e2.dz))
    
    max_pfiso = ak.where(e1.PFRelIso<e2.PFRelIso,e2.PFRelIso,e1.PFRelIso)
    
    wgt = events.eventWgt/sum_wgt
    vtx = events.sel_vtx

    histos["bdt_score"].fill(samp=samp,cut=cut,bdt_score=events.BDTScore,weight=wgt)
    #histos["sel_vtx_dR_zoom"].fill(samp=samp,cut=cut,dr_zoom=vtx.dR,weight=wgt)

    #histos["abs1overEm1overP"].fill(samp=samp,cut=cut,abs1overEm1overP=e1.abs1overEm1overP,weight=wgt)
    
    histos["sel_vtx_vxy"].fill(samp=samp,cut=cut,vxy=vtx.vxy,weight=wgt)
    histos["sel_vtx_vxy_projected"].fill(samp=samp,cut=cut,vxy_projected=events.projectedLxy,weight=wgt)
    #histos["sel_vtx_vxy_zoomed"].fill(samp=samp,cut=cut,vxy_zoomed=vtx.vxy,weight=wgt)
    histos["sel_vtx_minDxy"].fill(samp=samp,cut=cut,dxy=min_dxy,weight=wgt)
    #histos["sel_vtx_dxy1"].fill(samp=samp,cut=cut,dxy=np.abs(e1.dxy),weight=wgt)
    #histos["sel_vtx_dxy2"].fill(samp=samp,cut=cut,dxy=np.abs(e2.dxy),weight=wgt)
    
    #histos["sel_vtx_minDz"].fill(samp=samp,cut=cut,dz=min_dz,weight=wgt)

    #histos["sel_vtx_deltaDxy"].fill(samp=samp,cut=cut,dxy=delta_dxy,weight=wgt)
    #histos["sel_vtx_deltaDz"].fill(samp=samp,cut=cut,dz=delta_dz,weight=wgt)
    
    histos["sel_vtx_vxySignif"].fill(samp=samp,cut=cut,vxy_signif=vtx.vxy/vtx.sigmavxy,weight=wgt)

    histos["sel_vtx_type"].fill(samp=samp,cut=cut,type=vtx.typ,weight=wgt)

    histos["met_over_lead_jet_pt"].fill(samp=samp,cut=cut,met_over_pt=events.PFMET.pt/events.PFJet.pt[:,0],weight=wgt)
    
    histos["sel_vtx_chi2"].fill(samp=samp,cut=cut,chi2=vtx.reduced_chi2,weight=wgt)
    histos["sel_vtx_dR"].fill(samp=samp,cut=cut,dr=vtx.dR,weight=wgt)

    histos["sel_vtx_dEta"].fill(samp=samp,cut=cut,deta=np.abs(e1.eta-e2.eta),weight=wgt)
    histos["sel_vtx_dPhi"].fill(samp=samp,cut=cut,dphi=np.abs(e1.phi-e2.phi),weight=wgt)
    histos["sel_vtx_dEta_over_dPhi"].fill(samp=samp,cut=cut,deta_over_dphi=np.abs(e1.eta-e2.eta)/np.abs(e1.phi-e2.phi),weight=wgt)
    histos["sel_vtx_log_dEta_over_dPhi"].fill(samp=samp,cut=cut,deta_over_dphi=np.log10(np.abs(e1.eta-e2.eta)/np.abs(e1.phi-e2.phi)),weight=wgt)

    histos["sel_vtx_sign_eta"].fill(samp=samp,cut=cut,sign_eta=(e1.eta*e2.eta)/np.abs(e1.eta*e2.eta),weight=wgt)
    histos["sel_vtx_prod_eta"].fill(samp=samp,cut=cut,prod_eta=e1.eta*e2.eta,weight=wgt)
    
    histos["sel_vtx_METdPhi"].fill(samp=samp,cut=cut,dphi=np.abs(vtx.METdPhi),weight=wgt)

    histos["sel_vtx_pt"].fill(samp=samp,cut=cut,pt=vtx.pt,weight=wgt)
    histos["sel_vtx_eta"].fill(samp=samp,cut=cut,eta=vtx.eta,weight=wgt)
    histos["sel_vtx_phi"].fill(samp=samp,cut=cut,phi=vtx.phi,weight=wgt)
    histos["sel_vtx_mass"].fill(samp=samp,cut=cut,mass=vtx.m,weight=wgt)

    histos["sel_vtx_pt_e1_over_pt_e2"].fill(samp=samp,cut=cut,ratio=np.minimum(e1.pt, e2.pt)/np.maximum(e1.pt, e2.pt),weight=wgt)
    histos["sel_vtx_pt_over_m"].fill(samp=samp,cut=cut,sel_vtx_pt_over_m=vtx.pt/vtx.m,weight=wgt)
    histos["met_over_lead_jet_pt"].fill(samp=samp,cut=cut,met_over_pt=events.PFMET.pt/events.PFJet.pt[:,0],weight=wgt)

    histos["delta_dxy_over_mindxy"].fill(samp=samp,cut=cut,ratio_big=delta_dxy/min_dxy,weight=wgt)
    histos["delta_dxy_over_maxdxy"].fill(samp=samp,cut=cut,ratio=delta_dxy/max_dxy,weight=wgt)
    histos["delta_dxy_over_meandxy"].fill(samp=samp,cut=cut,ratio_big=delta_dxy/mean_dxy,weight=wgt)

    histos['cos_collinear'].fill(samp=samp,cut=cut,cos_collinear=events.cos_collinear,weight=wgt)
    histos['projectedLxy'].fill(samp=samp,cut=cut,vxy_projected=events.projectedLxy,weight=wgt)

    #histos['nLpt_Electron'].fill(samp=samp,cut=cut,num_ele=ak.count(events.LptElectron.pt,axis=1),weight=wgt)
    #histos['nPF_Electron'].fill(samp=samp,cut=cut,num_ele=ak.count(events.Electron.pt,axis=1),weight=wgt)
    #histos['nElectron'].fill(samp=samp,cut=cut,num_ele=ak.count(events.LptElectron.pt,axis=1)+ak.count(events.Electron.pt,axis=1),weight=wgt)

    #histos['nGoodVtx'].fill(samp=samp,cut=cut,num_good_vtx=ak.count(events.good_vtx.vxy,axis=1),weight=wgt)

    '''
    histos['sel_vtx_vxy_vs_mindxy'].fill(samp=samp,cut=cut,vxy=vtx.vxy,dxy=min_dxy,weight=wgt)
    
    histos['sel_vtx_vxy_vs_sel_vtx_chi2'].fill(samp=samp,cut=cut,vxy=vtx.vxy,chi2=vtx.reduced_chi2,weight=wgt)
    histos['sel_vtx_vxy_vs_sel_vtx_dR'].fill(samp=samp,cut=cut,vxy=vtx.vxy,dr=vtx.dR,weight=wgt)
    histos['sel_vtx_vxy_vs_sel_vtx_mass'].fill(samp=samp,cut=cut,vxy=vtx.vxy,mass=vtx.m,weight=wgt)
    histos['sel_vtx_vxy_vs_sel_vtx_METdPhi'].fill(samp=samp,cut=cut,vxy=vtx.vxy,dphi=np.abs(vtx.METdPhi),weight=wgt)
    histos['sel_vtx_vxy_vs_sel_vtx_vxySignif'].fill(samp=samp,cut=cut,vxy=vtx.vxy,vxy_signif=vtx.vxy/vtx.sigmavxy,weight=wgt)
    histos['sel_vtx_vxy_vs_sel_vtx_pt'].fill(samp=samp,cut=cut,vxy=vtx.vxy,pt=vtx.pt,weight=wgt)
    histos['sel_vtx_vxy_vs_sel_vtx_sign_eta'].fill(samp=samp,cut=cut,vxy=vtx.vxy,sign_eta=(e1.eta*e2.eta)/np.abs(e1.eta*e2.eta),weight=wgt)
    
    histos['dxy1_vs_dxy2'].fill(samp=samp,cut=cut,dxy1=np.abs(e1.dxy),dxy2=np.abs(e2.dxy),weight=wgt)

    histos['cos_collinear'].fill(samp=samp,cut=cut,cos_collinear=events.cos_collinear,weight=wgt)
    
    histos['abs_cos_collinear'].fill(samp=samp,cut=cut,cos_collinear=np.abs(events.cos_collinear),weight=wgt)
    
    histos['sel_vtx_dR_vs_sel_vtx_vxy'].fill(samp=samp,cut=cut,dr=vtx.dR,vxy=vtx.vxy,weight=wgt)
    histos['sel_vtx_dR_vs_sel_vtx_mass'].fill(samp=samp,cut=cut,dr=vtx.dR,mass=vtx.m,weight=wgt)
    histos['sel_vtx_dR_vs_sel_vtx_minDxy'].fill(samp=samp,cut=cut,dr=vtx.dR,dxy=min_dxy,weight=wgt)
    histos['sel_vtx_dR_vs_sel_vtx_chi2'].fill(samp=samp,cut=cut,dr=vtx.dR,chi2=vtx.reduced_chi2,weight=wgt)
    histos['sel_vtx_dR_vs_sel_vtx_vxySignif'].fill(samp=samp,cut=cut,dr=vtx.dR,vxy_signif=vtx.vxy/vtx.sigmavxy,weight=wgt)
    histos['sel_vtx_dR_vs_sel_vtx_METdPhi'].fill(samp=samp,cut=cut,dr=vtx.dR,dphi=np.abs(vtx.METdPhi),weight=wgt)
    histos['sel_vtx_dR_vs_bdt_score'].fill(samp=samp,cut=cut,dr=vtx.dR,bdt_score=events.BDTScore,weight=wgt)

    histos['sel_vtx_dR_vs_sel_vtx_pt_e1_over_pt_e2'].fill(samp=samp,cut=cut,dr=vtx.dR,ratio=e1.pt/e2.pt,weight=wgt)
    histos['sel_vtx_dR_vs_sel_vtx_type'].fill(samp=samp,cut=cut,dr=vtx.dR,type=vtx.typ,weight=wgt)

    histos['sel_vtx_dR_zoom_vs_sel_vtx_vxy'].fill(samp=samp,cut=cut,dr_zoom=vtx.dR,vxy=vtx.vxy,weight=wgt)
    histos['sel_vtx_dR_zoom_vs_sel_vtx_mass'].fill(samp=samp,cut=cut,dr_zoom=vtx.dR,mass=vtx.m,weight=wgt)
    histos['sel_vtx_dR_zoom_vs_sel_vtx_minDxy'].fill(samp=samp,cut=cut,dr_zoom=vtx.dR,dxy=min_dxy,weight=wgt)
    histos['sel_vtx_dR_zoom_vs_sel_vtx_chi2'].fill(samp=samp,cut=cut,dr_zoom=vtx.dR,chi2=vtx.reduced_chi2,weight=wgt)
    histos['sel_vtx_dR_zoom_vs_sel_vtx_vxySignif'].fill(samp=samp,cut=cut,dr_zoom=vtx.dR,vxy_signif=vtx.vxy/vtx.sigmavxy,weight=wgt)
    histos['sel_vtx_dR_zoom_vs_sel_vtx_METdPhi'].fill(samp=samp,cut=cut,dr_zoom=vtx.dR,dphi=np.abs(vtx.METdPhi),weight=wgt)
    histos['sel_vtx_dR_zoom_vs_bdt_score'].fill(samp=samp,cut=cut,dr_zoom=vtx.dR,bdt_score=events.BDTScore,weight=wgt)

    histos['sel_vtx_dR_zoom_vs_sel_vtx_pt_e1_over_pt_e2'].fill(samp=samp,cut=cut,dr_zoom=vtx.dR,ratio=e1.pt/e2.pt,weight=wgt)
    histos['sel_vtx_dR_zoom_vs_sel_vtx_type'].fill(samp=samp,cut=cut,dr_zoom=vtx.dR,type=vtx.typ,weight=wgt)
    
    if info["type"] == "signal":
        
        #histos['ctau'].fill(samp=samp,cut=cut,ctau=events.ctau,weight=wgt)

        #histos['ctau_vs_Lxy'].fill(samp=samp,cut=cut,ctau=events.ctau,vxy=vtx.vxy,weight=wgt)

        
        histos["sel_vtx_matchType"].fill(samp=samp,cut=cut,mtype=vtx.match,weight=wgt)
        histos['sel_vtx_vxy_vs_matchType'].fill(samp=samp,cut=cut,vxy=vtx.vxy,mtype=vtx.match,weight=wgt)
        histos['sel_vtx_minDxy_vs_matchType'].fill(samp=samp,cut=cut,dxy=min_dxy,mtype=vtx.match,weight=wgt)
        histos['sel_vtx_minDz_vs_matchType'].fill(samp=samp,cut=cut,dz=min_dz,mtype=vtx.match,weight=wgt)
        histos['sel_vtx_deltaDxy_vs_matchType'].fill(samp=samp,cut=cut,dxy=delta_dxy,mtype=vtx.match,weight=wgt)
        histos['sel_vtx_deltaDz_vs_matchType'].fill(samp=samp,cut=cut,dz=delta_dz,mtype=vtx.match,weight=wgt)

        histos['sel_vtx_vxySignif_vs_matchType'].fill(samp=samp,cut=cut,vxy_signif=vtx.vxy/vtx.sigmavxy,mtype=vtx.match,weight=wgt)
        histos['sel_vtx_chi2_vs_matchType'].fill(samp=samp,cut=cut,chi2=vtx.reduced_chi2,mtype=vtx.match,weight=wgt)
        histos['sel_vtx_dR_vs_matchType'].fill(samp=samp,cut=cut,dr=vtx.dR,mtype=vtx.match,weight=wgt)
        histos['sel_vtx_dEta_vs_matchType'].fill(samp=samp,cut=cut,deta=np.abs(e1.eta-e2.eta),mtype=vtx.match,weight=wgt)
        histos['sel_vtx_dPhi_vs_matchType'].fill(samp=samp,cut=cut,dphi=np.abs(e1.phi-e2.phi),mtype=vtx.match,weight=wgt)

        histos["sel_vtx_sign_eta_vs_matchType"].fill(samp=samp,cut=cut,sign_eta=(e1.eta*e2.eta)/np.abs(e1.eta*e2.eta),mtype=vtx.match,weight=wgt)
        
        histos['sel_vtx_METdPhi_vs_matchType'].fill(samp=samp,cut=cut,dphi=np.abs(vtx.METdPhi),mtype=vtx.match,weight=wgt)
        histos['sel_vtx_pt_vs_matchType'].fill(samp=samp,cut=cut,pt=vtx.pt,mtype=vtx.match,weight=wgt)
        histos['sel_vtx_eta_vs_matchType'].fill(samp=samp,cut=cut,eta=vtx.eta,mtype=vtx.match,weight=wgt)
        histos['sel_vtx_phi_vs_matchType'].fill(samp=samp,cut=cut,phi=vtx.phi,mtype=vtx.match,weight=wgt)
        histos['sel_vtx_mass_vs_matchType'].fill(samp=samp,cut=cut,mass=vtx.m,mtype=vtx.match,weight=wgt)
        histos['sel_vtx_type_vs_matchType'].fill(samp=samp,cut=cut,type=vtx.typ,mtype=vtx.match,weight=wgt)

        histos['nLpt_Electron_vs_matchType'].fill(samp=samp,cut=cut,num_ele=ak.count(events.LptElectron.pt,axis=1),mtype=vtx.match,weight=wgt)
        histos['nPF_Electron_vs_matchType'].fill(samp=samp,cut=cut,num_ele=ak.count(events.Electron.pt,axis=1),mtype=vtx.match,weight=wgt)
        histos['nElectron_vs_matchType'].fill(samp=samp,cut=cut,num_ele=ak.count(events.LptElectron.pt,axis=1)+ak.count(events.Electron.pt,axis=1)\
                                              ,mtype=vtx.match,weight=wgt)
        histos['nGoodVtx_vs_matchType'].fill(samp=samp,cut=cut,num_good_vtx=ak.count(events.good_vtx.vxy,axis=1),mtype=vtx.match,weight=wgt)
        histos['nGoodVtx_vs_sel_vtx_type'].fill(samp=samp,cut=cut,num_good_vtx=ak.count(events.good_vtx.vxy,axis=1),type=vtx.typ,weight=wgt)
    '''