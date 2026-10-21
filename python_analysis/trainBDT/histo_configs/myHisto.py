from hist import Hist
from hist.axis import StrCategory, Regular, Integer, IntCategory
import hist

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
        self.met = self.parse_axis(('met',100,50,300))
        self.dR = self.parse_axis(('dR',100,0,5))
        self.vxy1 = self.parse_axis(('vxy',100,0,1))
        self.vxy10 = self.parse_axis(('vxy',100,0,10))
        self.vxy100 = self.parse_axis(('vxy',100,0,100))
        self.ele_pt = self.parse_axis(("pt",50,0,50))
        self.dphi = self.parse_axis(("phi",64,-3.2,3.2))
        self.phi = self.parse_axis(("phi",64,-3.2,3.2))
        self.abs_dphi = self.parse_axis(('abs_dphi',100,0,3.2))
        self.eta = self.parse_axis(('eta',100,-2.5,2.5))
        self.dxy = self.parse_axis(('dxy',500,0,5))
        self.r3 = self.parse_axis(('r3',500,0,100))
        self.dxy_signif = self.parse_axis(('signif',100,0,20))
        self.trk_chi2 = self.parse_axis(('chi2',100,0,5))
        self.vtx_chi2 = self.parse_axis(('chi2',100,0,5))
        self.iso = self.parse_axis(('iso',100,0,5))
        self.sieie = self.parse_axis(('sieie',100,0,0.1))
        self.detaseed = self.parse_axis(('detaSeed',100,0,5))
        self.HoverE = self.parse_axis(('hoe',100,0,200))
        self.EinvMinusPinv = self.parse_axis(('emp',100,0,5))
        self.numMissingHits = self.parse_axis(('missing',10,0,10))
        self.passConvVeto = self.parse_axis(('passVeto',[0,1]))
        self.vxy_relDiff = self.parse_axis(('rel_diff',50,-2,2))
        self.isMatched = self.parse_axis(('matched',[0,1]))
        self.isPF = self.parse_axis(('isPF',[0,1]))
        self.numHits = self.parse_axis(('numHits',20,0,20))
        self.trkProb = self.parse_axis(('prob',100,0,1))
        self.IDScore = self.parse_axis(('id',100,-1,3))
        self.ele_passID = self.parse_axis(('passID',[0,1]))
        self.vtx_type = self.parse_axis(('vtype',['LL','LR','RR']))
        self.vtx_mass = self.parse_axis(('mass',100,0,30))
        self.vtx_sign = self.parse_axis(('sign',[-1,1]))
        self.vtx_pt = self.parse_axis(('pt',100,0,50))
        self.ele_ptRes = self.parse_axis(('ptres',100,-2,2))
        self.sigReco = self.parse_axis(('reco',[0,1]))
        self.vtxMatch = self.parse_axis(('match',[0,1]))
        self.dRCategories = self.parse_axis(('dRCat',['0to0p1','0p1to0p5','0p5toInf']))
        self.vxyCategories = self.parse_axis(('vxyCat',['0to1','1to5','5to10','10to15','15toInf']))
        self.ptCategories = self.parse_axis(('ptCat',['0to5','5to10','10to20','20toInf']))
        self.nJets = self.parse_axis(('nJets',10,0,10))
        self.vtxPurity = self.parse_axis(('purity',['total','n_matched_any','n_matched_sel','n_matched_good',
                                                    'n_eeReco_any','n_eeReco_passID',
                                                    'n_vtxReco_any','n_vtxReco_good']))
        self.recoCategories = self.parse_axis(('recoCat',['total','noEEreco','eeReco_noID','eeReco_ID_noVtx',
                                                          'eeReco_ID_hasVtx_bad','eeReco_ID_hasVtx_good_unSel','eeReco_ID_hasVtx_good_sel']))
        self.jet_pt = self.parse_axis(('jet_pt',100,80,1000))
        self.mass_low = self.parse_axis(('mass_low',500,0,5))
        self.mindxy_low = self.parse_axis(('mindxy_low',1000,0,0.1))
        self.sign_etaProd = self.parse_axis(('sign_etaProd',[-1,1]))
        self.cosTheta = self.parse_axis(('cosTheta',100,-1,1))
        self.LxyCosTheta = self.parse_axis(('LxyCosTheta',100,-50,50))
        self.LxyCosThetaZoom = self.parse_axis(('LxyCosThetaZoom',100,-5,5))
        self.LxyCosThetaZoomZoom = self.parse_axis(('LxyCosThetaZoomZoom',100,-1,1))
        self.jetMETratio = self.parse_axis(('jetMETratio',100,0,2))
        self.chi2Rank = self.parse_axis(('chi2Rank',[0,1,2,3,4,5,6,7,8,9,10]))
        self.vx = self.parse_axis(('vx',300,-15,15))
        self.vy = self.parse_axis(('vy',300,-15,15))
        self.vxy20 = self.parse_axis(('vxy',200,0,20))
        self.vxyRes = self.parse_axis(('vxyRes',100,0,5))
        self.vxySignif = self.parse_axis(('vxySignif',500,0,50))
        self.bdtScore = self.parse_axis(('bdtScore',1000,0,1))

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
    
    def parse_axis(self,a):
        name = a[0]
        if type(a[1]) == list:
            assert len(a) == 2
            if type(a[1][0]) == str:
                axis = StrCategory(a[1],name=name,label=name)
            else:
                axis = IntCategory(a[1],name=name,label=name)
        else:
            assert len(a) == 4
            axis = Regular(a[1],a[2],a[3],name=name,label=name)
        return axis