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

        #Raw GEN, GED, LPT:
        self.Gen_vxy_RAW = self.parse_axis(('Gen_vxy_RAW',1000,0,30))  
        self.Electron_vxy_RAW = self.parse_axis(('Electron_vxy_RAW',1000,0,30))  #GED before cuts
        self.LptElectron_vxy_RAW = self.parse_axis(('LptElectron_vxy_RAW',1000,0,30))  #GED before cuts

        
        self.dxy_gen = self.parse_axis(('dxy_gen',500,-3,3))  

        
        #For Resolution studies
        self.Res_LPT = self.parse_axis(('Res_LPT',500,-1,1))  
        self.Res_GED = self.parse_axis(('Res_GED',500,-1,1)) 
        
        self.Gen_pt = self.parse_axis(('Gen_pt',[0,5,10,20,30])) 
        
        # self.Gen_dxy_res = self.parse_axis(('Gen_dxy_res',[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,30])) #Gen_vxy_FLAT in this case
        self.Gen_dxy_res = self.parse_axis(('Gen_dxy_res',[0,1,3,6,10,30])) #Gen_vxy_FLAT in this case


        #With ev events:
        self.Gen_vxy_FLAT = self.parse_axis(('Gen_vxy_FLAT',1000,0,30)) #With cuts
        self.GED_vxy_flat = self.parse_axis(('GED_vxy_flat',1000,0,30)) #With cuts
        self.Lpt_vxy_flat = self.parse_axis(('Lpt_vxy_flat',1000,0,30)) #With cuts
        
        
        #For vtx Eff studies
        self.pt = self.parse_axis(('pt',[0,5,10,20,30])) 

        self.vxy = self.parse_axis(('vxy',[0,1,5,10,15,20]))  

        self.pt_den = self.parse_axis(('pt_den',[0,5,10,20,30])) 

        self.vxy_den = self.parse_axis(('vxy_den',[0,1,5,10,15,20]))  

        
        self.Lxy_genEE = self.parse_axis(('Lxy_genEE',1000,0,100))  
        self.Pt_genEE = self.parse_axis(('Pt_genEE',1000,0,100)) 
        self.dR_genEE = self.parse_axis(('dR_genEE',500,-5,5)) 

        
        
        
       
      

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
    
   


    #ev quantities: 
    h.make('gen_ele_vxy','Gen_vxy_FLAT')    
    h.make('GED_vxy_flat', 'GED_vxy_flat')
    h.make('Lpt_vxy_flat', 'Lpt_vxy_flat')
    
    

    #Residual:
    h.make('res_GED_gen','Res_GED')
    h.make('res_LPT_gen','Res_LPT')

   
    #Resolution in bins of a quantity:
    h.make('res_GED_gen_ptbin', 'Gen_dxy_res', 'Res_GED')
    h.make('res_LPT_gen_ptbin', 'Gen_dxy_res', 'Res_LPT')

    #Raw quantities:
    h.make('Gen_vxy_RAW', 'Gen_vxy_RAW')
    h.make('Electron_vxy_RAW', 'Electron_vxy_RAW')
    h.make('LptElectron_vxy_RAW', 'LptElectron_vxy_RAW')
    
    h.make("genvtx_pt_vxy",'pt','vxy')
    
    h.make("genvtx_pt_vxy_den",'pt_den','vxy_den')
    h.make('genEELxy','Lxy_genEE')  
    h.make('genEEPt','Pt_genEE') 
    h.make('genEEdR','dR_genEE') 
    

    
    return h

subroutines = []


def fillHistos(events,h,samp,cut,info,sum_wgt=1):
    h.samp = samp
    h.cut = cut
    wgt = events.eventWgt/sum_wgt
    
    if info["type"] == "signal":


        #Resolution STUDY:
        
        
        h.fill("genvtx_pt_vxy_den",pt_den=events.genEE.pt,vxy_den=events.genEE.vxy) #denominator: All the gen vertcies

        Lxy = events.GenEle.vxy  #Lxy for the gen electrons
        Pt =  events.genEE.pt
        dR =  events.genEE.dr

        h.fill("genEELxy",Lxy_genEE=Lxy)
        h.fill("genEEPt",Pt_genEE=Pt)
        h.fill("genEEdR",dR_genEE=dR)
        

        mask1 = ( (events.GenEle.matchType == 'R') & (events.GenPos.matchType == 'R') )        

        ev  = events[mask1]

        mask2 =  ak.any(ev.vtx.isMatched, axis=1)
        ev_new = ev[mask2] #Events which has atleast one matched vertex

        gen_vxy = ev_new.genEE.vxy

        gen_ee_pt = ev_new.genEE.pt 

        
        h.fill("genvtx_pt_vxy",pt=gen_ee_pt,vxy=gen_vxy)   #numerator: Only the gen vertices matched to LL 

       
