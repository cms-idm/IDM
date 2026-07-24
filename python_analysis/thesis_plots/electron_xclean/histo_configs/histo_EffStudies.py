from hist import Hist
from hist.axis import StrCategory, Regular, Integer, IntCategory, Variable
import hist
import numpy as np
import awkward as ak
import analysisSubroutines as sub

# General Purpose
samp = StrCategory([],name="samp",label="Sample Name",growth=True)
cut = StrCategory([],name="cut",label="Cut Applied",growth=True)


# functions to make histograms
class myHisto:
    def __init__(self):
        self.histograms = {}
        self.samp = "NO_SAMPLE"
        self.cut = "NO_CUT"

        # axis compendium
        self.samp = StrCategory([],name="samp",label="Sample Name",growth=True)
        self.cut = StrCategory([],name="cut",label="Cut Applied",growth=True)
        self.ele_type = self.parse_axis(('ele_type',['L','R']))
        self.match_type = self.parse_axis(('match_type',['L','R']))
        self.match = self.parse_axis(('match',[0,1]))
        self.met = self.parse_axis(('met',60,50,300))
        self.dR = self.parse_axis(('dR',200,0,1)) 
        self.mindR = self.parse_axis(('mindR',60,0,0.06)) 
        

        self.vxy1 = self.parse_axis(('vxy',[0,1,5,10,15,20]))   
        self.vxy10 = self.parse_axis(('vxy',[0,2,4,6,8,10,14,18,24,30,40,50,60,70,80]))  #Lxy 10, 100
        
        self.vxy100 = self.parse_axis(('vxy',[0,2,4,6,8,10,14,18,24,30,40,50]))  #Lxy 10, 100
       
        self.PT_GED = self.parse_axis(("PT_GED",[0,1,2,3,4,5,8,10,12,14,16,18,20,25,30]))  #Good for low pT electrons for 1000mm
        self.PT_Lpt = self.parse_axis(("PT_Lpt",[0,1,2,3,4,5,8,10,12,14,16,18,20,25,30]))  #Good for low pT electrons for 1000mm

        self.ele_pt = self.parse_axis(("pt",[0,1,2,3,4,5,8,10,12,14,16,18,20,25,30]))   #Good for GED electrons
       
        self.dphi = self.parse_axis(("phi",64,-3.2,3.2))
        self.phi = self.parse_axis(("phi",64,-3.2,3.2))
        self.abs_dphi = self.parse_axis(('phi',100,0,3))
        self.eta = self.parse_axis(('eta',50,-2.5,2.5))

        self.dxy = self.parse_axis(('dxy',500,0,5))
        self.r3 = self.parse_axis(('r3',500,0,100))
        self.dxy_signif = self.parse_axis(('signif',100,0,20))
        self.trk_chi2 = self.parse_axis(('chi2',100,0,5))
        self.vtx_chi2 = self.parse_axis(('chi2',100,0,5))
        self.iso = self.parse_axis(('iso',100,0,5))
        self.sieie = self.parse_axis(('sieie',100,0,0.1))
        self.detaseed = self.parse_axis(('detaSeed',100,0,5))
        self.HoverE = self.parse_axis(('hoe',100,0,200))
        self.EinvMinusPinv = self.parse_axis(('emp',100,0,5))
        self.numMissingHits = self.parse_axis(('missing',10,0,10))
        self.passConvVeto = self.parse_axis(('passVeto',[0,1]))
        self.vxy_relDiff = self.parse_axis(('rel_diff',50,-2,2))
        self.isMatched = self.parse_axis(('matched',[0,1]))
        self.isPF = self.parse_axis(('isPF',[0,1]))
        self.numHits = self.parse_axis(('numHits',20,0,20))
        self.trkProb = self.parse_axis(('prob',100,0,1))
        self.IDScore = self.parse_axis(('id',100,-1,3))
        self.ele_passID = self.parse_axis(('passID',[0,1]))
        #self.vtx_type = self.parse_axis(('vtype',['LL','LR','RR']))

        self.vtx_mass = self.parse_axis(('mass',100,0,30))
        self.vtx_sign = self.parse_axis(('sign',[-1,1]))
        self.vtx_pt = self.parse_axis(('pt',100,0,50))
        self.ele_ptRes = self.parse_axis(('ptres',100,-2,2))
        self.sigReco = self.parse_axis(('reco',[0,1]))
        self.vtxMatch = self.parse_axis(('match',[0,1]))

    def make(self,name,*args,**hist_kwargs):
        if name in self.histograms.keys():
            print(f"Histogram {name} already exists! Skipping")
            return
        axes = [self.samp,self.cut]
        for ax in args:
            if type(ax) == tuple:
                axes.append(self.parse_axis(ax))
            else:
                axes.append(getattr(self,ax))
        self.histograms[name] = Hist(*axes,storage=hist.storage.Weight(),**hist_kwargs)
    
    def fill(self,name,**kwargs):
        self.histograms[name].fill(samp=self.samp,cut=self.cut,**kwargs)
    
   
    #I made it a bit CASE-SPECIFIC

    def parse_axis(self,a):
        name = a[0]
        if type(a[1]) == list:
            assert len(a) == 2
            if (len(a[1]) > 2):
                axis = Variable(a[1],name=name,label=name)
            else:
                if type(a[1][0]) == str:
                    axis = StrCategory(a[1],name=name,label=name)
                if type(a[1][0]) == int: 
                    axis = IntCategory(a[1],name=name,label=name)
           

        else:
            assert len(a) == 4
            axis = Regular(a[1],a[2],a[3],name=name,label=name)
        return axis

def make_histograms():
    h = myHisto()
    
    # quantities associated w/ gen objects
   
    h.make("gen_vxy1",'vxy1')
    h.make("gen_vxy10",'vxy10')
    h.make("gen_vxy100",'vxy100')

  
   
    h.make('gen_ele_pt','ele_pt')
    h.make('pT_genElePos_GED', 'PT_GED')
    h.make('pT_genElePos_Lpt', 'PT_Lpt')

    
    h.make('gen_ele_eta','eta')
    h.make('gen_ele_phi','phi')
    h.make('gen_ele_dR','dR')    
    h.make('gen_ele_vxy1','vxy1')
    h.make('gen_ele_vxy10','vxy10')
    h.make('gen_ele_vxy100','vxy100')
    h.make("mindR", 'mindR')

    # matched reco electrons, corresponding gen object variables
    h.make("match_ele_gen_pt",'match_type','ele_passID','ele_pt')
    
    h.make('match_ele_gen_vxy1','match_type','ele_passID','vxy1')    
    h.make('match_ele_gen_vxy10','match_type','ele_passID','vxy10')
    h.make('match_ele_gen_vxy100','match_type','ele_passID','vxy100')
    
    h.make("gen_ele_pt_vs_vxy1",'ele_pt','vxy1')
    h.make("match_ele_gen_pt_vs_vxy1",'match_type','ele_passID','ele_pt','vxy1') 
    
  
    # misc other quantities
    h.make("PFMET",'met')
    h.make("PFMET_vs_genMET",'met',('genmet',100,50,300))


    return h

subroutines = []


def fillHistos(events,h,samp,cut,info,sum_wgt=1):
    h.samp = samp
    h.cut = cut
    wgt = events.eventWgt/sum_wgt
    
    if info["type"] == "signal":
   
        mask_genele = events.GenEle.matched
        events_with_genEle_matched = events[mask_genele]
        
        mask_R = (events_with_genEle_matched.GenEle.matchType == 'R')  
        mask_L = (events_with_genEle_matched.GenEle.matchType == 'L')        

        pt_genele_GED = events_with_genEle_matched.GenEle.pt[mask_R] 
        pt_genele_Lpt = events_with_genEle_matched.GenEle.pt[mask_L]
       
        mask_genpos = events.GenPos.matched        
        events_with_genpos_matched = events[mask_genpos]
        
        mask_R_pos = (events_with_genpos_matched.GenPos.matchType == 'R')   
        mask_L_pos = (events_with_genpos_matched.GenPos.matchType == 'L')        

        pt_genpos_GED = events_with_genpos_matched.GenPos.pt[mask_R_pos] 
        pt_genpos_Lpt = events_with_genpos_matched.GenPos.pt[mask_L_pos] 

       
        
        
        
        h.fill("gen_ele_pt",pt=events.GenEle.pt)
        h.fill("gen_ele_pt",pt=events.GenPos.pt)

        h.fill("pT_genElePos_GED", PT_GED = pt_genele_GED)
        h.fill("pT_genElePos_GED", PT_GED = pt_genpos_GED)

        h.fill("pT_genElePos_Lpt", PT_Lpt = pt_genele_Lpt)
        h.fill("pT_genElePos_Lpt", PT_Lpt = pt_genpos_Lpt)
       

        
    

        
        
        

        


        








        
        # hasMatch_pf = (ak.count(events.Electron.pt,axis=1)>0) &\
        #               (ak.count_nonzero(events.Electron.genMatched,axis=1)>0)

        # hasMatch_lpt = (ak.count(events.LptElectron.pt,axis=1)>0) &\
        #                (ak.count_nonzero(events.LptElectron.genMatched,axis=1)>0)

        # # print ("events[hasMatch_pf].GenEle:", events[hasMatch_pf].GenEle)
        # match_pf = events[hasMatch_pf].Electron
        # match_pf = match_pf[match_pf.genMatched]        
        # # print (" match_pf = match_pf[match_pf.genMatched]:", match_pf)


        # genObj_pf = ak.where(match_pf.matchType==-1,events[hasMatch_pf].GenEle,events[hasMatch_pf].GenPos)
        # match_pf = ak.flatten(match_pf)
        # genObj_pf = ak.flatten(genObj_pf)

        # match_pf_passID = ak.values_astype(match_pf.passID,int)


        # match_lpt = events[hasMatch_lpt].LptElectron
        # match_lpt = match_lpt[match_lpt.genMatched]
        # genObj_lpt = ak.where(match_lpt.matchType==-1,events[hasMatch_lpt].GenEle,events[hasMatch_lpt].GenPos)
        # match_lpt = ak.flatten(match_lpt)
        # genObj_lpt = ak.flatten(genObj_lpt)
        # match_lpt_passID = ak.values_astype(match_lpt.passID,int)


        

        # sub.ctaucalculate(events)
      
       
       
        # h.fill("mindR", mindR=ak.flatten(events.LptElectron.minDRtoReg))              
      
               

        # h.fill("match_ele_gen_pt",match_type='R',passID=match_pf_passID,pt=genObj_pf.pt)       
        # h.fill("match_ele_gen_pt",match_type='L',passID=match_lpt_passID,pt=genObj_lpt.pt)    

          
        # h.fill("match_ele_gen_vxy1",match_type='R',passID=match_pf_passID,vxy=genObj_pf.vxy)
        # h.fill("match_ele_gen_vxy10",match_type='R',passID=match_pf_passID,vxy=genObj_pf.vxy)
        # h.fill("match_ele_gen_vxy100",match_type='R',passID=match_pf_passID,vxy=genObj_pf.vxy)

        # h.fill("match_ele_gen_vxy1",match_type='L',passID=match_lpt_passID,vxy=genObj_lpt.vxy)
        # h.fill("match_ele_gen_vxy10",match_type='L',passID=match_lpt_passID,vxy=genObj_lpt.vxy)
        # h.fill("match_ele_gen_vxy100",match_type='L',passID=match_lpt_passID,vxy=genObj_lpt.vxy)

        # h.fill("gen_ele_pt_vs_vxy1",pt=events.GenEle.pt,vxy=events.GenEle.vxy)
        # h.fill("gen_ele_pt_vs_vxy1",pt=events.GenPos.pt,vxy=events.GenPos.vxy)
        
        # h.fill("match_ele_gen_pt_vs_vxy1",match_type='L',passID=match_lpt_passID,pt=genObj_lpt.pt, vxy=genObj_lpt.vxy)
        
        # h.fill("match_ele_gen_pt_vs_vxy1",match_type='R',passID=match_pf_passID,pt=genObj_pf.pt, vxy=genObj_pf.vxy)

