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
        

        self.PT_GED = self.parse_axis(("PT_GED",[0,1,2,3,4,5,8,10,12,14,16,18,20,25,30]))  #Good for low pT electrons for 1000mm
        self.PT_Lpt = self.parse_axis(("PT_Lpt",[0,1,2,3,4,5,8,10,12,14,16,18,20,25,30]))  #Good for low pT electrons for 1000mm        
        self.ele_pt = self.parse_axis(("pt",[0,1,2,3,4,5,8,10,12,14,16,18,20,25,30]))   #Good for GED electrons

        self.VXY_GED = self.parse_axis(("VXY_GED",[0,2,4,7,10,15,22,30]))  #Good for low pT electrons for 1000mm
        self.VXY_Lpt = self.parse_axis(("VXY_Lpt",[0,2,4,7,10,15,22,30]))  #Good for low pT electrons for 1000mm        
        self.ele_vxy = self.parse_axis(("vxy",[0,2,4,7,10,15,22,30]))   #Good for GED electrons


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

    h.make('gen_ele_vxy','ele_vxy')
    h.make('vxy_genElePos_GED', 'VXY_GED')
    h.make('vxy_genElePos_Lpt', 'VXY_Lpt')


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

        # pt_genele_GED = events_with_genEle_matched.GenEle.pt[mask_R] 
        # pt_genele_Lpt = events_with_genEle_matched.GenEle.pt[mask_L]
       
        # mask_genpos = events.GenPos.matched        
        # events_with_genpos_matched = events[mask_genpos]
        
        # mask_R_pos = (events_with_genpos_matched.GenPos.matchType == 'R')   
        # mask_L_pos = (events_with_genpos_matched.GenPos.matchType == 'L')        

        # pt_genpos_GED = events_with_genpos_matched.GenPos.pt[mask_R_pos] 
        # pt_genpos_Lpt = events_with_genpos_matched.GenPos.pt[mask_L_pos] 


        #################################################################
        vxy_genele_GED = events_with_genEle_matched.GenEle.vxy[mask_R] 
        vxy_genele_Lpt = events_with_genEle_matched.GenEle.vxy[mask_L]
       
        mask_genpos = events.GenPos.matched        
        events_with_genpos_matched = events[mask_genpos]
        
        mask_R_pos = (events_with_genpos_matched.GenPos.matchType == 'R')   
        mask_L_pos = (events_with_genpos_matched.GenPos.matchType == 'L')        

        vxy_genpos_GED = events_with_genpos_matched.GenPos.vxy[mask_R_pos] 
        vxy_genpos_Lpt = events_with_genpos_matched.GenPos.vxy[mask_L_pos] 
        ####################################################################
        
        
        h.fill("gen_ele_pt",pt=events.GenEle.pt)
        h.fill("gen_ele_pt",pt=events.GenPos.pt)

        h.fill("gen_ele_vxy",vxy=events.GenEle.vxy)
        h.fill("gen_ele_vxy",vxy=events.GenPos.vxy)

        h.fill("vxy_genElePos_GED", VXY_GED = vxy_genele_GED)
        h.fill("vxy_genElePos_GED", VXY_GED = vxy_genpos_GED)

        h.fill("vxy_genElePos_Lpt", VXY_Lpt = vxy_genele_Lpt)
        h.fill("vxy_genElePos_Lpt", VXY_Lpt = vxy_genpos_Lpt)

        # h.fill("pT_genElePos_GED", PT_GED = pt_genele_GED)
        # h.fill("pT_genElePos_GED", PT_GED = pt_genpos_GED)

        # h.fill("pT_genElePos_Lpt", PT_Lpt = pt_genele_Lpt)
        # h.fill("pT_genElePos_Lpt", PT_Lpt = pt_genpos_Lpt)
       



