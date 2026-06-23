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

        mask_gen_alllpt = events.GenEle.matchedAllLowPt
        event_new = events[mask_gen_alllpt]
        
        mask_dr = event_new.AllLptElectron.minDRtoReg < 0.05
        print (mask_dr)        
        mask_event = ak.any(mask_dr, axis=1)    
        
        event_new_very = event_new[mask_event]
        
        print ("dr check=", event_new_very.AllLptElectron.minDRtoReg)
        Gen_Ele_pt = event_new_very.GenEle.pt  #This is the Gen Ele pt to be used
        print ("match check=", event_new_very.GenEle.matchType)
        
        print ("LPt pt = ", event_new_very.AllLptElectron.pt)
        mask_dr_lpt = event_new_very.AllLptElectron.minDRtoReg < 0.05

        Lpt_pt = event_new_very.AllLptElectron.pt[mask_dr_lpt]
        FLAT_lpt = ak.flatten(Lpt_pt)
        
        print ("Length LPt pt= ", Lpt_pt)
        
        GED_pt = event_new_very.Electron.pt
        FLAT_GED = ak.flatten(GED_pt)
        print ("Length GED pt=", GED_pt)

        print(ak.all(ak.num(GED_pt, axis=1) == ak.num(Lpt_pt, axis=1)))
##################################################################################

        mask_gen_pos_alllpt = events.GenPos.matchedAllLowPt
        event_new_pos = events[mask_gen_pos_alllpt]

        mask_dr_pos = event_new_pos.AllLptElectron.minDRtoReg < 0.05

        mask_event_pos = ak.any(mask_dr_pos, axis=1)        
        event_new_pos_very = event_new_pos[mask_event_pos]

        GenPos_pt = event_new_pos_very.GenPos.pt

        mask_dr_lpt_pos = event_new_pos_very.AllLptElectron.minDRtoReg < 0.05
        print ("LPt_pos = ", event_new_pos_very.AllLptElectron.pt[mask_dr_lpt_pos])

        print ("GED_pos pt=", event_new_pos_very.Electron.pt)

###########################################
        
        # dr = event_new.AllLptElectron.minDRtoReg[mask_dr]
        # print("dr=", dr)
        # Allpt_pt = event_new.AllLptElectron.pt[mask_dr]
        # # GED_pt = event_new.Electron.pt[mask_dr]
        # # Gen_Ele_pt = event_new.GenEle.pt[mask_dr]
        # print ("Allpt_pt=", Allpt_pt)
        # # print ("GED_pt=", GED_pt)
        # # flat_lpt_pt = ak.flatten(Allpt_pt)
        # # print ("len(flat_lpt_pt)=", len(flat_lpt_pt))
        # # print ("len(Gen_Ele_pt)=", len(Gen_Ele_pt))

        # print (event_new.GenEle.matchType)
        
        
        
        

        # mask_allLpt_e = events.GenEle.matchedAllLowPt 
        # type_new = events.GenEle.matchType[mask_allLpt_e]
        # print (type_new)
        
       
        # lpt = ak.Array([[3,4],[],[5,7],[],[],[7,8]])
        # mask1 = lpt > 3
        # print ("mask1=", mask1)
        # mask=  ak.any (mask1, axis=1)
        # print ("mask=", mask)
        # # ged = [[5],[],[],[],[],[9]]

        
        # trial = (ak.num(lpt, axis=1) > 0) & (ak.num(ged, axis=1) > 0)
        # print (trial)     


        
        # mask_genele_matched = events.GenEle.matched
        
        # events_with_genele_matched = events[mask_genele_matched]

        # events_with_gen_matchedtoLPT = events_with_genele_matched[events_with_genele_matched.GenEle.matchType == 'L']

        # #Resolution study:
        # MINdr = events_with_gen_matchedtoLPT.AllLptElectron.minDRtoReg
        # print ("MINdr=", MINdr)
        # MASK_mindr_condition = MINdr < 0.05
        # print ("MASK_mindr_condition=", MASK_mindr_condition)  
        # MASK_togetevents = ak.any(MASK_mindr_condition, axis=1)
        # print ("MASK_togetevents=", MASK_togetevents)

        # event_new = events_with_gen_matchedtoLPT[MASK_togetevents]
        
        # genele_pt = event_new.GenEle.pt
        # print ("genele_pt=", genele_pt)
        # print ("len(genele_pt)=", len(genele_pt))
        
        # ged_pt = event_new.Electron.pt        
        # print ("ged_pt=", ged_pt)
        # print ("ak.num(ged_pt, axis=1)", ak.num(ged_pt, axis=1))
        # trial =ak.num(ged_pt, axis=1)
        # mask_try = ak.any(trial !=1)
        # print ("yes/no=", mask_try)
        
                       
        
        
        # FLAT_ged_pt = ak.flatten(ged_pt)
        # print ("len(FLAT_ged_pt)=", len(FLAT_ged_pt))

        # alllpt_pt = event_new.AllLptElectron.pt
        # print ("lpt_pt=", alllpt_pt)
       

        # lpt_pt = event_new.LptElectron.pt
        # print ("LptElectron.pt=", lpt_pt)

        # common = alllpt_pt[ak.is_in(alllpt_pt, lpt_pt)]
        # print ("common=", common)
        # print (ak.num(common, axis=1))

        
#Almost there a bit more


        
        # pt_ged = events_with_gen_matchedtoLPT.Electron.pt
        # print ("pt_ged = ", pt_ged)
        
        
        
        


        
        

        

       
        