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
        
        #For eff studies
        self.ele_pt = self.parse_axis(("pt",[0,1,2,3,4,5,8,10,12,14,16,18,20,25]))   

        self.PT_GED = self.parse_axis(("PT_GED",[0,1,2,3,4,5,8,10,12,14,16,18,20,25]))  
        self.PT_Lpt = self.parse_axis(("PT_Lpt",[0,1,2,3,4,5,8,10,12,14,16,18,20,25]))  

        self.PT_Lpt_noxclean = self.parse_axis(("PT_Lpt_noxclean",[0,1,2,3,4,5,8,10,12,14,16,18,20,25]))  

       
        self.IDScore = self.parse_axis(('id',100,-1,3))
        self.ele_passID = self.parse_axis(('passID',[0,1]))

      

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
    
    
    h.make('gen_ele_pt','ele_pt')
    h.make('pT_genElePos_GED', 'PT_GED')
    h.make('pT_genElePos_Lpt', 'PT_Lpt')

    #No xclean
    h.make("pT_genElePos_Lpt_Noxclean", 'PT_Lpt_noxclean')

    

    return h

subroutines = []


def fillHistos(events,h,samp,cut,info,sum_wgt=1):
    h.samp = samp
    h.cut = cut
    wgt = events.eventWgt/sum_wgt
    
    if info["type"] == "signal":

        #When no lpt ele with GED match (with-xclean)
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


        #All Lpt ele (No xclean)
        mask_allLpt_e = events.GenEle.matchedAllLowPt 
        print ("mask_allLpt_e=", mask_allLpt_e)
        pt_genele_Lpt_ALL = events.GenEle.pt[mask_allLpt_e]

        mask_allLpt_p = events.GenPos.matchedAllLowPt
        print ("mask_allLpt_p=", mask_allLpt_p)
        pt_genpos_Lpt_ALL = events.GenPos.pt[mask_allLpt_p]

        
        

       
        
        
        
        h.fill("gen_ele_pt",pt=events.GenEle.pt)
        h.fill("gen_ele_pt",pt=events.GenPos.pt)

        h.fill("pT_genElePos_GED", PT_GED = pt_genele_GED)
        h.fill("pT_genElePos_GED", PT_GED = pt_genpos_GED)

        h.fill("pT_genElePos_Lpt", PT_Lpt = pt_genele_Lpt)
        h.fill("pT_genElePos_Lpt", PT_Lpt = pt_genpos_Lpt)

        h.fill("pT_genElePos_Lpt_Noxclean", PT_Lpt_noxclean = pt_genele_Lpt_ALL)
        h.fill("pT_genElePos_Lpt_Noxclean", PT_Lpt_noxclean = pt_genpos_Lpt_ALL)




        
       
