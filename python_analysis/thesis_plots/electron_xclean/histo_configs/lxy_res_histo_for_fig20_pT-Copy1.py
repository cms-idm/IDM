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
        
        self.dxy_gen = self.parse_axis(('dxy_gen',500,0,5))  #just added
        self.GED_dxy_flat = self.parse_axis(('GED_dxy_flat',500,0,5)) 

        
        #For Resolution studies
        self.Res_LPT = self.parse_axis(('Res_LPT',1000,-10,10))  #-0.5-0.5
        self.Res_GED = self.parse_axis(('Res_GED',1000,-10,10)) 
        
        self.Gen_pt = self.parse_axis(('Gen_pt',[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,30])) 
        
        self.Gen_dxy_res = self.parse_axis(('Gen_dxy_res',[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,30])) #Gen_pt_FLAT in this case

        self.Gen_vxy = self.parse_axis(('Gen_vxy',[0,1,3,6,15])) 
        
        self.Gen_vxy_FLAT = self.parse_axis(('Gen_vxy_FLAT',100,0,10)) 
        self.GED_vxy_flat = self.parse_axis(('GED_vxy_flat',100,0,10)) 
        self.Lpt_vxy_flat = self.parse_axis(('Lpt_vxy_flat',100,0,10)) 


        
        #For Eff studies
        self.vxy1 = self.parse_axis(('vxy',[0,1,3,6,10,15]))  #Lxy 10, 100
        # self.ele_pt = self.parse_axis(("pt",[0,5,10,20,30])) 
        self.ele_pt = self.parse_axis(("pt",[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,30])) 
        
        
       
      

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
    
    h.make('gen_ele_vxy','Gen_vxy_FLAT')
    h.make('GED_vxy_flat', 'GED_vxy_flat')
    h.make('Lpt_vxy_flat', 'Lpt_vxy_flat')
    
    h.make('dxy_gen','dxy_gen')
    h.make('GED_dxy_flat', 'GED_dxy_flat')
    

    h.make("Gen_ElePos_pt", 'Gen_pt')
    h.make("Gen_ElePos_vxy", 'Gen_vxy')

    h.make('res_GED_gen','Res_GED')
    h.make('res_LPT_gen','Res_LPT')

   

    h.make('res_GED_gen_ptbin', 'Gen_dxy_res', 'Res_GED')
    h.make('res_LPT_gen_ptbin', 'Gen_dxy_res', 'Res_LPT')

    
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
        

        ev = events_new

        GenEle_pt = ev.GenEle.pt
        GenEle_vxy = ev.GenEle.vxy

        GenPos_pt = ev.GenPos.pt
        GenPos_vxy = ev.GenPos.vxy        

       
        # Gen side keeps its [electron, positron] order.
        Gen_pt_FLAT  = ak.flatten(ak.concatenate(    [ev.GenEle.pt[:, None],  ev.GenPos.pt[:, None]],  axis=1))
        Gen_vxy_FLAT = ak.flatten(ak.concatenate(    [ev.GenEle.vxy[:, None], ev.GenPos.vxy[:, None]], axis=1))

        def dxy(obj, ref_x,ref_y):
            """Return transverse distance between obj and ref at their point of closest approach"""            
            return (-(obj.vx - ref_x)*obj.py + (obj.vy - ref_y)*obj.px)/obj.pt

        
        Dxy_gen_ele = dxy(ev.GenEle, ev.PV.x, ev.PV.y)
        print ("Dxy_gen_ele=", Dxy_gen_ele)
        Dxy_gen_pos = dxy(ev.GenPos, ev.PV.x, ev.PV.y)
        print ("Dxy_gen_pos=", Dxy_gen_pos)

        Gen_dxy_FLAT  = ak.flatten(ak.concatenate(    [Dxy_gen_ele[:, None],  Dxy_gen_pos[:, None]],  axis=1))

       
        # Reco side: take each gen particle's OWN match by index, so the two arrays line up.
        lpt, ged = ev.AllLptElectron.vxy, ev.Electron.vxy
        Lpt_vxy_flat = ak.flatten(ak.concatenate([    lpt[ak.singletons(ev.GenEle.matchIdxAllLowPt)],    lpt[ak.singletons(ev.GenPos.matchIdxAllLowPt)]], axis=1))
        GED_vxy_flat = ak.flatten(ak.concatenate([    ged[ak.singletons(ev.GenEle.matchIdxLocal)],    ged[ak.singletons(ev.GenPos.matchIdxLocal)]], axis=1))

        GED_dxy_flat = ak.flatten(ak.concatenate([    ged[ak.singletons(ev.GenEle.matchIdxLocal)],    ged[ak.singletons(ev.GenPos.matchIdxLocal)]], axis=1))

        res_GED = (GED_vxy_flat - Gen_vxy_FLAT) / (Gen_vxy_FLAT)
        res_LPT = (Lpt_vxy_flat - Gen_vxy_FLAT) / (Gen_vxy_FLAT)

        # Check ref for dxy. Where is the PV? Make sure PV is same for both gen and reco.


       


        # Resolution plots
       
        # h.fill("res_GED_gen", Res_GED = res_GED)
        # h.fill("res_LPT_gen", Res_LPT = res_LPT)
        

        # h.fill("gen_ele_vxy", Gen_vxy_FLAT=Gen_vxy_FLAT)
        # h.fill("GED_vxy_flat", GED_vxy_flat=GED_vxy_flat)
        # h.fill("Lpt_vxy_flat", Lpt_vxy_flat=Lpt_vxy_flat)

        # h.fill("dxy_gen", dxy_gen = Dxy_gen_ele)  #New added
        # h.fill("dxy_gen", dxy_gen = Dxy_gen_pos)  #New added



        # h.fill("res_GED_gen_ptbin", Gen_dxy_res=Gen_pt_FLAT, Res_GED=res_GED) #changed to dxy IMP

        # h.fill("res_LPT_gen_ptbin",    Gen_dxy_res=Gen_pt_FLAT,    Res_LPT=res_LPT) #changed to dxy IMP
        


      