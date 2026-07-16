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
       

        self.Res_LPT_lxy = self.parse_axis(('Res_LPT_vxy',200,-0.7,0.7))  #-0.5-0.5
        self.Res_GED_lxy = self.parse_axis(('Res_GED_vxy',200,-0.7,0.7)) 
        
        #For Eff studies
        self.vxy1 = self.parse_axis(('vxy',[0,1,2,3,4,5,6,8,10,12,14,16,18,20]))  #Lxy 10, 100
        # self.ele_pt = self.parse_axis(("pt",[0,5,10,20,30])) 
        self.ele_pt = self.parse_axis(("pt",[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])) 
        
        self.PT_GED = self.parse_axis(("PT_GED",[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]))
        self.PT_Lpt = self.parse_axis(("PT_Lpt",[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])) 
        self.PT_Lpt_noxclean = self.parse_axis(("PT_Lpt_noxclean",[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]))


        self.VXY_GED = self.parse_axis(("VXY_GED",[0,1,2,3,4,5,6,8,10,12,14,16,18,20]))  
        self.VXY_Lpt = self.parse_axis(("VXY_Lpt",[0,1,2,3,4,5,6,8,10,12,14,16,18,20]))  

        self.VXY_Lpt_noxclean = self.parse_axis(("VXY_Lpt_noxclean",[0,1,2,3,4,5,6,8,10,12,14,16,18,20]))  


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
    h.make('gen_ele_vxy1','vxy1')

  

    h.make('res_GED_gen_lxy','Res_GED_lxy')
    h.make('res_LPT_gen_lxy','Res_LPT_lxy')

    
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

        mask_56 = ((events_new.GenEle.vxy > 1) & (events_new.GenEle.vxy < 2)) & ((events_new.GenPos.vxy > 1) & (events_new.GenPos.vxy < 2))
        events_new_very = events_new[mask_56]

        GenEle_vxy = events_new_very.GenEle.vxy
        print ("GenEle_vxy=", GenEle_vxy)
        print (len(GenEle_vxy))
        
        
        GenPos_vxy = events_new_very.GenPos.vxy
        print ("GenPos_vxy=", GenPos_vxy)
        print (len(GenPos_vxy))
        

        
        Gen_vxy = ak.concatenate([GenEle_vxy[:, None],GenPos_vxy[:, None]],  axis=1) #IMP
        print ("Gen_vxy=", Gen_vxy)
        
        
        Gen_vxy_FLAT = ak.flatten(Gen_vxy)
        print ("len(Gen_vxyt_FLAT)=", len(Gen_vxy_FLAT))



        Lpt_vxy = events_new_very.AllLptElectron.vxy[(events_new_very.AllLptElectron.genMatched) ] #IMP
        print ("Lpt_vxy=", Lpt_vxy)

        Lpt_vxy_flat = ak.flatten(Lpt_vxy)        
        print ("len(Lpt_vxy_flat)=", len(Lpt_vxy_flat)) 
        
        GED_vxy  = events_new_very.Electron.vxy[(events_new_very.Electron.genMatched)] 
        print ("GED_vxy=", GED_vxy)

        GED_vxy_flat = ak.flatten(GED_vxy)  
        print ("len(GED_vxy_flat)=", len(GED_vxy_flat))

        
        
        #Residual calculation:
        res_GED_vxy = (GED_vxy_flat - Gen_vxy_FLAT)/(Gen_vxy_FLAT) 
        print ("Res_GED=", res_GED_vxy)

        res_LPT_vxy = (Lpt_vxy_flat - Gen_vxy_FLAT)/(Gen_vxy_FLAT) 
        print("Res_LPT=", res_LPT_vxy)


        #Resolution plots
       
        h.fill("res_GED_gen_lxy", Res_GED_vxy = res_GED_vxy)
        h.fill("res_LPT_gen_lxy", Res_LPT_vxy = res_LPT_vxy)

      