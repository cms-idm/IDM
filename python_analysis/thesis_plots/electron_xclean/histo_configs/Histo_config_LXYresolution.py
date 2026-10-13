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
        
        self.Gen_pt = self.parse_axis(('Gen_pt',[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,30])) 
        
        # self.Gen_dxy_res = self.parse_axis(('Gen_dxy_res',[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,30])) #Gen_vxy_FLAT in this case
        self.Gen_dxy_res = self.parse_axis(('Gen_dxy_res',[0,1,3,6,10,30])) #Gen_vxy_FLAT in this case


        #With ev events:
        self.Gen_vxy_FLAT = self.parse_axis(('Gen_vxy_FLAT',1000,0,30)) #With cuts
        self.GED_vxy_flat = self.parse_axis(('GED_vxy_flat',1000,0,30)) #With cuts
        self.Lpt_vxy_flat = self.parse_axis(('Lpt_vxy_flat',1000,0,30)) #With cuts
        
        # self.lxy = self.parse_axis(('lxy',1000,0,30))  #GEN Lxy with cuts



        
        #For Eff studies
        # self.vxy1 = self.parse_axis(('vxy',[0,1,3,6,10,15]))  #Lxy 10, 100
        
        
       
      

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

    
    return h

subroutines = []


def fillHistos(events,h,samp,cut,info,sum_wgt=1):
    h.samp = samp
    h.cut = cut
    wgt = events.eventWgt/sum_wgt
    
    if info["type"] == "signal":


        #Resolution STUDY:
        Gen_vxy_RAW = ak.flatten(ak.concatenate(    [events.GenEle.vxy[:, None], events.GenPos.vxy[:, None]], axis=1)) #Gen vxy raw
        
        Electron_vxy_RAW = ak.flatten(events.Electron.vxy)  #GED vxy raw

        LptElectron_vxy_RAW = ak.flatten(events.AllLptElectron.vxy)  #GED vxy raw


        mask = ((events.GenEle.matchedAllLowPt) & (events.GenEle.matchType == 'R')) & ((events.GenPos.matchedAllLowPt) & (events.GenPos.matchType == 'R'))
        events_new = events[mask]
        

        ev = events_new

        GenEle_pt = ev.GenEle.pt
        GenEle_vxy = ev.GenEle.vxy

        GenPos_pt = ev.GenPos.pt
        GenPos_vxy = ev.GenPos.vxy     

        # Table section
        # gen_pt = ak.concatenate(    [ev.GenEle.pt[:, None], ev.GenPos.pt[:, None]],     axis=1 )
        gen_vxy = ak.concatenate(    [ev.GenEle.vxy[:, None], ev.GenPos.vxy[:, None]],     axis=1 )


        # lpt, ged = ev.AllLptElectron.pt, ev.Electron.pt
        lpt, ged = ev.AllLptElectron.vxy, ev.Electron.vxy

        # lpt_pt = ak.concatenate(    [        lpt[ak.singletons(ev.GenEle.matchIdxAllLowPt)],        lpt[ak.singletons(ev.GenPos.matchIdxAllLowPt)]    ],    axis=1)
        # ged_pt = ak.concatenate(    [        ged[ak.singletons(ev.GenEle.matchIdxLocal)],        ged[ak.singletons(ev.GenPos.matchIdxLocal)]    ],    axis=1)



        lpt_vxy = ak.concatenate(    [        lpt[ak.singletons(ev.GenEle.matchIdxAllLowPt)],        lpt[ak.singletons(ev.GenPos.matchIdxAllLowPt)]    ],    axis=1)
        ged_vxy = ak.concatenate(    [        ged[ak.singletons(ev.GenEle.matchIdxLocal)],        ged[ak.singletons(ev.GenPos.matchIdxLocal)]    ],    axis=1)

        # print(f"{'Event':<8} {'Particle':<10} {'Gen_pt':<15} {'GED_pt':<15} {'Lpt_pt':<15}")
        # print("-" * 65)

        # for i in range(len(ev)):
        #     for j, particle in enumerate(["Ele", "Pos"]):
        #         print( f"{i:<8} " f"{particle:<10} "  f"{float(gen_pt[i][j]):<15.5f} "    f"{float(ged_pt[i][j]):<15.5f} "  f"{float(lpt_pt[i][j]):<15.5f}"  )
                

        print(f"{'Event':<8} {'Particle':<10} {'Gen_vxy':<15} {'GED_vxy':<15} {'Lpt_vxy':<15}")
        print("-" * 65)

        for i in range(len(ev)):
            for j, particle in enumerate(["Ele", "Pos"]):
                print( f"{i:<8} " f"{particle:<10} "  f"{float(gen_vxy[i][j]):<15.5f} "    f"{float(ged_vxy[i][j]):<15.5f} "  f"{float(lpt_vxy[i][j]):<15.5f}"  )
                
       
       #  Gen_pt_FLAT  = ak.flatten(ak.concatenate(    [ev.GenEle.pt[:, None],  ev.GenPos.pt[:, None]],  axis=1))
       #  Gen_vxy_FLAT = ak.flatten(ak.concatenate(    [ev.GenEle.vxy[:, None], ev.GenPos.vxy[:, None]], axis=1))

        

       
       
       #  # Reco side: take each gen particle's OWN match by index, so the two arrays line up.
       #  lpt, ged = ev.AllLptElectron.vxy, ev.Electron.vxy
        
       #  Lpt_vxy_flat = ak.flatten(ak.concatenate([    lpt[ak.singletons(ev.GenEle.matchIdxAllLowPt)],    lpt[ak.singletons(ev.GenPos.matchIdxAllLowPt)]], axis=1))
       #  GED_vxy_flat = ak.flatten(ak.concatenate([    ged[ak.singletons(ev.GenEle.matchIdxLocal)],    ged[ak.singletons(ev.GenPos.matchIdxLocal)]], axis=1))



       # # Sort everything according to Gen_vxy
       #  order = ak.argsort(Gen_vxy_FLAT)

       #  Gen_sorted = Gen_vxy_FLAT[order]
       #  GED_sorted = GED_vxy_flat[order]
       #  Lpt_sorted = Lpt_vxy_flat[order]

       #  print(f"{'Index':<8} {'Gen_vxy':<15} {'GED_vxy':<15} {'Lpt_vxy':<15}")
       #  print("-" * 53)

       #  for i, (gen, ged, lpt) in enumerate(zip(Gen_sorted, GED_sorted, Lpt_sorted)):    print(
       #  f"{i:<8} "
       #  f"{float(gen):<15.5f} "
       #  f"{float(ged):<15.5f} "
       #  f"{float(lpt):<15.5f}"
       #  )
        

        # res_GED = (GED_vxy_flat - Gen_vxy_FLAT) / (Gen_vxy_FLAT)
        
        # res_LPT = (Lpt_vxy_flat - Gen_vxy_FLAT) / (Gen_vxy_FLAT)
        
        # # Resolution plots       
        # h.fill("res_GED_gen", Res_GED = res_GED)
        # h.fill("res_LPT_gen", Res_LPT = res_LPT)
        

        # #Raw Gen, GED, Lpt
        # h.fill("Gen_vxy_RAW", Gen_vxy_RAW =Gen_vxy_RAW) #Raw Electron vxy
        # h.fill("Electron_vxy_RAW", Electron_vxy_RAW =Electron_vxy_RAW) #Raw Electron vxy
        # h.fill("LptElectron_vxy_RAW", LptElectron_vxy_RAW =LptElectron_vxy_RAW) #Raw LPt vxy
      
        # #ev plots
        # h.fill("gen_ele_vxy", Gen_vxy_FLAT=Gen_vxy_FLAT) #ev
        # h.fill("GED_vxy_flat", GED_vxy_flat=GED_vxy_flat) #ev      
        # h.fill("Lpt_vxy_flat", Lpt_vxy_flat=Lpt_vxy_flat) #ev

        


        # h.fill("res_GED_gen_ptbin", Gen_dxy_res=Gen_vxy_FLAT, Res_GED=res_GED) #changed to vxy IMP

        # h.fill("res_LPT_gen_ptbin",    Gen_dxy_res=Gen_vxy_FLAT,    Res_LPT=res_LPT) #changed to vxy IMP
        


      