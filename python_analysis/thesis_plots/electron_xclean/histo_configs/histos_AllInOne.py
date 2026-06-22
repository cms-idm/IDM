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
        self.Res_LPT = self.parse_axis(('Res_LPT',[-1,1])) 
        self.Res_GED = self.parse_axis(('Res_GED',[-1,1])) 

        self.vxy1 = self.parse_axis(('vxy',[0,1,2,3,4,5,6,8,10,12,14,16,18,20]))  #Lxy 10, 100
        # self.ele_pt = self.parse_axis(("pt",[0,5,10,20,30])) 
        self.ele_pt = self.parse_axis(("pt",[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])) 
        
        self.PT_GED = self.parse_axis(("PT_GED",[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]))
        self.PT_Lpt = self.parse_axis(("PT_Lpt",[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])) 
        self.PT_Lpt_noxclean = self.parse_axis(("PT_Lpt_noxclean",[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]))


        self.VXY_GED = self.parse_axis(("VXY_GED",[0,1,2,3,4,5,6,8,10,12,14,16,18,20]))  
        self.VXY_Lpt = self.parse_axis(("VXY_Lpt",[0,1,2,3,4,5,6,8,10,12,14,16,18,20]))  

        self.VXY_Lpt_noxclean = self.parse_axis(("VXY_Lpt_noxclean",[0,1,2,3,4,5,6,8,10,12,14,16,18,20]))  


        ############################################
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
    
   
           
    h.make('gen_ele_pt','ele_pt')
    h.make('gen_ele_vxy1','vxy1')

    #Resolution studies
    h.make('res_lowpT_gen','Res_LPT')
    
    
    h.make('res_GED_gen','Res_GED')
    
    return h

subroutines = []


def fillHistos(events,h,samp,cut,info,sum_wgt=1):
    h.samp = samp
    h.cut = cut
    wgt = events.eventWgt/sum_wgt
    
    if info["type"] == "signal":


        #Resolution STUDY:

        mask = (events.GenEle.matchedAllLowPt) & (events.GenPos.matchedAllLowPt)
        events_new = events[mask]

        mask_dr = events_new.AllLptElectron.minDRtoReg < 0.05

        mask_dr_event = ak.any(mask_dr, axis=1)

        events_new_very = events_new[mask_dr_event] #We have now chosen events that have atleast a mindr<0.05

        Gen_Ele_pt = events_new_very.GenEle.pt
        print ("Gen_Ele_pt=", Gen_Ele_pt)
        Gen_Pos_pt = events_new_very.GenPos.pt
        print ("Gen_Pos_pt=", Gen_Pos_pt)

        Lpt_dr = events_new_very.AllLptElectron.minDRtoReg
        print (Lpt_dr)

        Lpt_mask_dr = events_new_very.AllLptElectron.minDRtoReg < 0.05
        print (events_new_very.AllLptElectron.minDRtoReg[Lpt_mask_dr])

        Lpt_pt = events_new_very.AllLptElectron.pt[Lpt_mask_dr] #These are the Lpt Electrons that have a nearby GED
        print ("Lpt_pt=", Lpt_pt)
        print (len(Lpt_pt))

        # GED_pt = events_new_very.Electron.pt[(events_new_very.Electron.genMatched) & (events_new_very.Electron.hasAllLptMatch)]
        # print ("GED_pt=", GED_pt)
        # mask = ak.num(GED_pt, axis=1) == ak.num(Lpt_pt, axis=1)
        # print(mask)
        # n_bad = ak.sum(~mask)
        # print(n_bad)
        
        

        # mask_GED = events_new_very.Electron.genMatched
        # GED_pt = events_new_very.Electron.pt[mask_GED]
        # print ("GED_pt=", GED_pt)
        # print (len(GED_pt))

        print(ak.all(ak.num(GED_pt, axis=1) == ak.num(Lpt_pt, axis=1)))


        # mask = (events.AllLptElectron.genMatched) & (events.AllLptElectron.minDRtoReg < 0.05) #Select events where low-pt ele are all gen matched and they have a nearby GED.
        # Lpt_pt = events.AllLptElectron.pt[mask]
        # print (Lpt_pt)
        # GED_pt = events.Electron.pt[mask]
        # events_new = events[mask] #We have events where low-pT electrons are gen matched

        # #Now, let us choose events with mindr<0.05
        # mask_dr = events_new.AllLptElectron.minDRtoReg < 0.05
        # mask_dr_event = ak.any(mask_dr, axis=1)

        # events_new_very = events_new[mask_dr_event] 
        # dr = events_new_very.AllLptElectron.minDRtoReg
        # print ("dr=", dr)



        
      
       
        









                