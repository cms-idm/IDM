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
        
        self.dxy_gen = self.parse_axis(('dxy_gen',500,-3,3))  

        
        #For Resolution studies
        self.Res_LPT = self.parse_axis(('Res_LPT',500,-1,1))  
        self.Res_GED = self.parse_axis(('Res_GED',500,-1,1)) 
        
        self.Gen_pt = self.parse_axis(('Gen_pt',[0,5,10,20,30]))         
        self.Gen_dxy_res = self.parse_axis(('Gen_dxy_res',[0,1,3,6,10,30])) #Gen_vxy_FLAT in this case


        #With ev events:
        self.Gen_vxy_FLAT = self.parse_axis(('Gen_vxy_FLAT',1000,0,30)) #With cuts
        self.GED_vxy_flat = self.parse_axis(('GED_vxy_flat',1000,0,30)) #With cuts
        self.Lpt_vxy_flat = self.parse_axis(('Lpt_vxy_flat',1000,0,30)) #With cuts
        
        
        #For vtx Eff studies
        self.pt_LL = self.parse_axis(('pt_LL',[0,5,10,20,100])) 
        self.vxy_LL = self.parse_axis(('vxy_LL',[0,1,5,10,15,20]))  

        self.pt_RR = self.parse_axis(('pt_RR',[0,5,10,20,100])) 
        self.vxy_RR = self.parse_axis(('vxy_RR',[0,1,5,10,15,20])) 

        self.pt_LR = self.parse_axis(('pt_LR',[0,5,10,20,100])) 
        self.vxy_LR = self.parse_axis(('vxy_LR',[0,1,5,10,15,20]))  

        self.pt_den = self.parse_axis(('pt_den',[0,5,10,20,100])) 
        self.vxy_den = self.parse_axis(('vxy_den',[0,1,5,10,15,20]))  

        
        self.Lxy_genEE = self.parse_axis(('Lxy_genEE',1000,0,100))  
        self.Pt_genEE = self.parse_axis(('Pt_genEE',1000,0,100)) 
        self.dR_genEE = self.parse_axis(('dR_genEE',500,0,5)) 
        self.eta_genEE = self.parse_axis(('eta_genEE',500,-5,5))
        self.phi_genEE = self.parse_axis(('phi_genEE',500,-5,5))
        
        self.METdPhi = self.parse_axis(('METdPhi',500,-5,5))

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
    
    h.make("genvtx_pt_vxy_LL",'pt_LL','vxy_LL')
    h.make("genvtx_pt_vxy_RR",'pt_RR','vxy_RR')
    h.make("genvtx_pt_vxy_LR",'pt_LR','vxy_LR')

    
    h.make("genvtx_pt_vxy_den",'pt_den','vxy_den')
    h.make('genEELxy','Lxy_genEE')  
    h.make('genEEPt','Pt_genEE') 
    h.make('genEEdR','dR_genEE') 
    h.make('genEEeta','eta_genEE') 
    h.make('genEEphi','phi_genEE') 
    h.make('genEEMETdPhi', 'METdPhi')
    

    
    return h

subroutines = []


def fillHistos(events,h,samp,cut,info,sum_wgt=1):
    h.samp = samp
    h.cut = cut
    wgt = events.eventWgt/sum_wgt
    
    if info["type"] == "signal":


        #Resolution STUDY:

        # dr =  events.genEE.dr

        # cut = (dr>1) & (dr < 5)
        # events = events[cut]
        
        #Fill the denominator: All the gen vertices passing the dr cut
        h.fill("genvtx_pt_vxy_den",pt_den=events.genEE.pt,vxy_den=events.genEE.vxy) 

        #Have the basic quantities as well:
        Lxy = events.GenEle.vxy  
        Pt =  events.genEE.pt
        dR =  events.genEE.dr
        eta = events.genEE.eta
        phi = events.genEE.phi
        metdphi = events.genEE.METdPhi

        h.fill("genEELxy",Lxy_genEE=Lxy)
        h.fill("genEEPt",Pt_genEE=Pt)
        h.fill("genEEdR",dR_genEE=dR)
        h.fill("genEEeta",eta_genEE=eta)
        h.fill("genEEphi",phi_genEE=phi)
        h.fill("genEEMETdPhi",METdPhi = metdphi)
        

        #Start the numerator:
        
        #For LL combination:
        maskLL = ( (events.GenEle.matchType == 'L') & (events.GenPos.matchType == 'L') )        

        evLL  = events[maskLL]

        mask2LL =  ak.any(evLL.vtx.isMatched, axis=1)
        ev_newLL = evLL[mask2LL] #Events which has atleast one matched vertex

        gen_vxy_LL = ev_newLL.genEE.vxy
        gen_ee_pt_LL = ev_newLL.genEE.pt

        
        h.fill("genvtx_pt_vxy_LL",pt_LL=gen_ee_pt_LL, vxy_LL=gen_vxy_LL)   #numerator: Only the gen vertices matched to LL 

        #For RR combination:
        maskRR = ( (events.GenEle.matchType == 'R') & (events.GenPos.matchType == 'R') )        

        evRR  = events[maskRR]

        mask2RR =  ak.any(evRR.vtx.isMatched, axis=1)
        ev_newRR = evRR[mask2RR] #Events which has atleast one matched vertex

        gen_vxy_RR = ev_newRR.genEE.vxy
        gen_ee_pt_RR = ev_newRR.genEE.pt
        
        
        h.fill("genvtx_pt_vxy_RR",pt_RR=gen_ee_pt_RR,vxy_RR=gen_vxy_RR)   #numerator: Only the gen vertices matched to RR

        #LR combination:
        mask1LR = ( (events.GenEle.matchType == 'L') & (events.GenPos.matchType == 'R') )
        evLR  = events[mask1LR]

        mask2LR =  ak.any(evLR.vtx.isMatched, axis=1)
        ev_newLR = evLR[mask2LR] #Events which has atleast one matched vertex

        gen_vxy_LR = ev_newLR.genEE.vxy

        gen_ee_pt_LR = ev_newLR.genEE.pt

        h.fill("genvtx_pt_vxy_LR",pt_LR=gen_ee_pt_LR,vxy_LR=gen_vxy_LR)   #numerator: Only the gen vertices matched to LR

