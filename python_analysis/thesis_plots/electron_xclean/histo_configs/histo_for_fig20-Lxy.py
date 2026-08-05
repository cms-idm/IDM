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
        
        #For Resolution studies
        self.Res_LPT = self.parse_axis(('Res_LPT',500,-0.8,0.8))  #-0.5-0.5
        self.Res_GED = self.parse_axis(('Res_GED',500,-0.8,0.8)) 
        
        self.Gen_pt = self.parse_axis(('Gen_pt',[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,25])) 
        self.Gen_vxy = self.parse_axis(('Gen_vxy',[0,1,3,6,10,15])) 


        
        #For Eff studies
        self.vxy1 = self.parse_axis(('vxy',[0,1,3,6,10,15]))  #Lxy 10, 100
        # self.ele_pt = self.parse_axis(("pt",[0,5,10,20,30])) 
        self.ele_pt = self.parse_axis(("pt",[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,25])) 
        
        
       
      

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
    h.make('gen_ele_vxy1','vxy1')

    h.make("Gen_ElePos_pt", 'Gen_pt')
    h.make("Gen_ElePos_vxy", 'Gen_vxy')

    h.make('res_GED_gen','Res_GED')
    h.make('res_LPT_gen','Res_LPT')

   

    h.make('res_GED_gen_ptbin', 'Gen_vxy', 'Res_GED')
    h.make('res_LPT_gen_ptbin', 'Gen_vxy', 'Res_LPT')

    
    return h

subroutines = []


def fillHistos(events,h,samp,cut,info,sum_wgt=1):
    h.samp = samp
    h.cut = cut
    wgt = events.eventWgt/sum_wgt
    
    if info["type"] == "signal":


        #Resolution STUDY:

        mask = ((events.GenEle.matchedAllLowPt) & (events.GenEle.matchType == 'R')) & ((events.GenPos.matchedAllLowPt) & (events.GenPos.matchType == 'R'))
        events_new = events[mask]
        

        # mask_56 = ((events_new.GenEle.pt > 5) & (events_new.GenEle.pt < 6)) & ((events_new.GenPos.pt > 5) & (events_new.GenPos.pt < 6))
        events_new_very = events_new

        GenEle_pt = events_new_very.GenEle.pt
        GenEle_vxy = events_new_very.GenEle.vxy
        
        
        
        GenPos_pt = events_new_very.GenPos.pt
        GenPos_vxy = events_new_very.GenPos.vxy
        

        
        Gen_pt = ak.concatenate([GenEle_pt[:, None],GenPos_pt[:, None]],  axis=1) #IMP
        Gen_vxy = ak.concatenate([GenEle_vxy[:, None],GenPos_vxy[:, None]],  axis=1) #IMP
                
        
        Gen_pt_FLAT = ak.flatten(Gen_pt)
        Gen_vxy_FLAT = ak.flatten(Gen_vxy)



        Lpt_pt = events_new_very.AllLptElectron.pt[(events_new_very.AllLptElectron.genMatched) ] #IMP

        Lpt_pt_flat = ak.flatten(Lpt_pt)        
        
        GED_pt  = events_new_very.Electron.pt[(events_new_very.Electron.genMatched)] 

        GED_pt_flat = ak.flatten(GED_pt)  

        
        
        #Residual calculation:
        res_GED = (GED_pt_flat - Gen_pt_FLAT)/(Gen_pt_FLAT) 

        res_LPT = (Lpt_pt_flat - Gen_pt_FLAT)/(Gen_pt_FLAT) 


        #Resolution plots
       
        h.fill("res_GED_gen", Res_GED = res_GED)
        h.fill("res_LPT_gen", Res_LPT = res_LPT)

        h.fill("Gen_ElePos_pt", Gen_pt = GenEle_pt)
        h.fill("Gen_ElePos_pt", Gen_pt = GenPos_pt)

        h.fill("res_GED_gen_ptbin", Gen_vxy=Gen_vxy_FLAT, Res_GED=res_GED)

        h.fill("res_LPT_gen_ptbin",    Gen_vxy=Gen_vxy_FLAT,    Res_LPT=res_LPT)
        


      