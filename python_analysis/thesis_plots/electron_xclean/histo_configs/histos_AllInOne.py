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
        self.dR = self.parse_axis(('dR',50,0,5)) 
        self.mindR = self.parse_axis(('mindR',60,0,0.06)) 
        
        #For eff studies
        self.vxy1 = self.parse_axis(('vxy',[0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,30]))  #Lxy 10, 100

        self.ele_pt = self.parse_axis(("pt",[0,1,2,3,4,5,8,10,12,14,16,18,20,30])) 
        self.PT_GED = self.parse_axis(("PT_GED",[0,1,2,3,4,5,8,10,12,14,16,18,20,30]))  
        self.PT_Lpt = self.parse_axis(("PT_Lpt",[0,1,2,3,4,5,8,10,12,14,16,18,20,30]))  
        self.PT_Lpt_noxclean = self.parse_axis(("PT_Lpt_noxclean",[0,1,2,3,4,5,8,10,12,14,16,18,20,30]))  


        self.VXY_GED = self.parse_axis(("VXY_GED",[0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,30]))  
        self.VXY_Lpt = self.parse_axis(("VXY_Lpt",[0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,30]))  

        self.VXY_Lpt_noxclean = self.parse_axis(("VXY_Lpt_noxclean",[0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,30]))  


       
        self.IDScore = self.parse_axis(('id',100,-1,3))
        self.ele_passID = self.parse_axis(('passID',[0,1]))

        self.dRgenGEDGED = self.parse_axis(("dRgenGEDGED",50,0,5)) 
        self.dRgenGEDLpt = self.parse_axis(("dRgenGEDLpt",50,0,5)) 
        self.dRgenLptLpt = self.parse_axis(("dRgenLptLpt",50,0,5)) 
        self.dRgenLptGED = self.parse_axis(("dRgenLptGED",50,0,5)) 



      

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
    
    h.make('gen_dR', 'dR')
    h.make('dR_gen_GED_GED', 'dRgenGEDGED')
    h.make('dR_gen_Lpt_Lpt', 'dRgenLptLpt')
    h.make('dR_gen_GED_Lpt', 'dRgenGEDLpt')
    h.make('dR_gen_Lpt_GED', 'dRgenLptGED')

           
    h.make('gen_ele_pt','ele_pt')
    h.make('gen_ele_vxy1','vxy1')

    
    h.make('pT_genElePos_GED', 'PT_GED')
    h.make('pT_genElePos_Lpt', 'PT_Lpt')

    h.make('vxy_genElePos_GED', 'VXY_GED')
    h.make('vxy_genElePos_Lpt', 'VXY_Lpt')
    


    #No xclean
    h.make("pT_genElePos_Lpt_Noxclean", 'PT_Lpt_noxclean')
    h.make("vxy_genElePos_Lpt_Noxclean", 'VXY_Lpt_noxclean')

    

    return h

subroutines = []


def fillHistos(events,h,samp,cut,info,sum_wgt=1):
    h.samp = samp
    h.cut = cut
    wgt = events.eventWgt/sum_wgt
    
    if info["type"] == "signal":
        

        #When no lpt ele with GED match (with-xclean)

        #GenEle########################
        mask_genele = events.GenEle.matched
        events_with_genEle_matched = events[mask_genele]
        
        mask_R = (events_with_genEle_matched.GenEle.matchType == 'R')  
        mask_L = (events_with_genEle_matched.GenEle.matchType == 'L')        

        #pT cases
        pt_genele_GED = events_with_genEle_matched.GenEle.pt[mask_R] 
        pt_genele_Lpt = events_with_genEle_matched.GenEle.pt[mask_L]
        
        #vxy cases
        vxy_genele_GED = events_with_genEle_matched.GenEle.vxy[mask_R] 
        vxy_genele_Lpt = events_with_genEle_matched.GenEle.vxy[mask_L]

       
        
        #GenPos###########################
        mask_genpos = events.GenPos.matched        
        events_with_genpos_matched = events[mask_genpos]
        
        mask_R_pos = (events_with_genpos_matched.GenPos.matchType == 'R')   
        mask_L_pos = (events_with_genpos_matched.GenPos.matchType == 'L')        

        #pT cases
        pt_genpos_GED = events_with_genpos_matched.GenPos.pt[mask_R_pos] 
        pt_genpos_Lpt = events_with_genpos_matched.GenPos.pt[mask_L_pos] 

                
        #vxy cases
        vxy_genpos_GED = events_with_genpos_matched.GenPos.vxy[mask_R_pos] 
        vxy_genpos_Lpt = events_with_genpos_matched.GenPos.vxy[mask_L_pos] 


        #dR cases: Only x-cleaned ele
        MASK = (mask_genele) & (mask_genpos)
        events_mask = events[MASK]

        mask_GED_GED = (events_mask.GenEle.matchType == 'R') &  (events_mask.GenPos.matchType == 'R')
        mask_Lpt_Lpt = (events_mask.GenEle.matchType == 'L') &  (events_mask.GenPos.matchType == 'L')
        mask_GED_Lpt = (events_mask.GenEle.matchType == 'R') &  (events_mask.GenPos.matchType == 'L')
        mask_Lpt_GED = (events_mask.GenEle.matchType == 'L') &  (events_mask.GenPos.matchType == 'R')
       
        
        dR_gen_GED_GED = events_mask.genEE.dr[mask_GED_GED]
        dR_gen_Lpt_Lpt = events_mask.genEE.dr[mask_Lpt_Lpt]
        dR_gen_GED_Lpt = events_mask.genEE.dr[mask_GED_Lpt]
        dR_gen_Lpt_GED = events_mask.genEE.dr[mask_Lpt_GED]
      
       

        

        #All Lpt ele (No xclean)
        mask_allLpt_e = events.GenEle.matchedAllLowPt 
        
        pt_genele_Lpt_ALL = events.GenEle.pt[mask_allLpt_e]
        vxy_genele_Lpt_ALL = events.GenEle.vxy[mask_allLpt_e]


        mask_allLpt_p = events.GenPos.matchedAllLowPt
        
        pt_genpos_Lpt_ALL = events.GenPos.pt[mask_allLpt_p]
        vxy_genpos_Lpt_ALL = events.GenPos.vxy[mask_allLpt_p]

        
        

       
        
        
        
        h.fill("gen_ele_pt",pt=events.GenEle.pt)
        h.fill("gen_ele_pt",pt=events.GenPos.pt)


        h.fill("pT_genElePos_GED", PT_GED = pt_genele_GED)
        h.fill("pT_genElePos_GED", PT_GED = pt_genpos_GED)

        h.fill("pT_genElePos_Lpt", PT_Lpt = pt_genele_Lpt)
        h.fill("pT_genElePos_Lpt", PT_Lpt = pt_genpos_Lpt)   

                

        h.fill("pT_genElePos_Lpt_Noxclean", PT_Lpt_noxclean = pt_genele_Lpt_ALL)
        h.fill("pT_genElePos_Lpt_Noxclean", PT_Lpt_noxclean = pt_genpos_Lpt_ALL)

###################################vxy cases##############

        h.fill("gen_ele_vxy1",vxy=events.GenEle.vxy)
        h.fill("gen_ele_vxy1",vxy=events.GenPos.vxy)
        
        h.fill("vxy_genElePos_GED", VXY_GED = vxy_genele_GED)
        h.fill("vxy_genElePos_GED", VXY_GED = vxy_genpos_GED)

        h.fill("vxy_genElePos_Lpt", VXY_Lpt = vxy_genele_Lpt)
        h.fill("vxy_genElePos_Lpt", VXY_Lpt = vxy_genpos_Lpt)


        h.fill("vxy_genElePos_Lpt_Noxclean", VXY_Lpt_noxclean = vxy_genele_Lpt_ALL)
        h.fill("vxy_genElePos_Lpt_Noxclean", VXY_Lpt_noxclean = vxy_genpos_Lpt_ALL)

####dR cases############################

        h.fill("gen_dR",dR=events.genEE.dr)
        
        h.fill("dR_gen_GED_GED", dRgenGEDGED=dR_gen_GED_GED)
        h.fill("dR_gen_Lpt_Lpt", dRgenLptLpt=dR_gen_Lpt_Lpt)
        h.fill("dR_gen_GED_Lpt", dRgenGEDLpt=dR_gen_GED_Lpt)
        h.fill("dR_gen_Lpt_GED", dRgenLptGED=dR_gen_Lpt_GED)
     