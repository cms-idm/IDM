from hist.axis import StrCategory, Regular, Integer, IntCategory

############################
##### Categorical Axes #####
############################
# General Purpose
samp = StrCategory([],name="samp",label="Sample Name",growth=True)
cut = StrCategory([],name="cut",label="Cut Applied",growth=True)
# For electrons
ele_type = StrCategory(["L","R"],name="ele_type",label="Electron Type")
# For vertices
vtx_sign = IntCategory([-1,1],name="sign",label="Vertex sign (q1*q2)")
vtx_type = StrCategory(["LL","LR","RR"],name="type",label="Vertex Type")
vtx_matchType = IntCategory([0,1,2],name="mtype",label="Vertex Gen Match Type")
# Misc
angleDot = Regular(100,-1,1,name='dot',label=r"$\hat{n}_1 \cdot \hat{n}_2$")
bdtScore = Regular(100,0,1,name='score',label='BDT Score')
ratio = Regular(100,0,1,name="ratio",label="ratio")
ratio_big = Regular(1000,0,50,name="ratio_big",label="ratio")
muonEtFrac= Regular(100,0,1,name="muonEtFrac",label="muonEtFrac")

############################
###### Numerical Axes ######
############################
# Electrons
ele_pt = Regular(100,0,50,name="pt",label="$p_{T}$ [GeV]")
ele_pt_zoom_fine = Regular(250,0,25,name="pt",label="$p_{T}$ [GeV]")
ele_eta = Regular(60,-3,3,name="eta",label=r"$\eta$")
ele_eta_fine = Regular(300,-3,3,name="eta",label=r"$\eta$")
ele_phi = Regular(64,-3.2,3.2,name="phi",label=r"$\phi$")
ele_trkHits = Regular(30,0,30,name="numTrkHits",label="Number of Tracker Hits")
ele_pixHits = Regular(10,0,10,name="numPixHits",label="Number of Pixel Hits")
ele_stripHits = Regular(25,0,25,name="numStripHits",label="Number of Strip Hits")
ele_chi2 = Regular(200,0,100,name="chi2",label=r"Track $\chi^2/df$")
ele_trkIso = Regular(100,0,100,name="trkIso",label="Tracker Iso")
ele_trkRelIso = Regular(100,0,5,name="relIso",label="Tracker Relative Iso")
ele_PFRelIso = Regular(100,0,10,name="relIso",label="PF Relative Iso")
ele_PFIso = Regular(100,0,100,name="iso",label="PF Isolation")
ele_PFRelIsoM = Regular(100,0,200,name="isoM",label=r"$I_{PF}^{rel} \times m_{e^+e^-}$")
ele_PFIsoM = Regular(100,0,200,name="isoM",label=r"$I_{PF} \times m_{e^+e^-}$")
ele_prob = Regular(100,0,1,name="prob",label=r"Electron Track $\chi^2$ Probability")
ele_angRes = Regular(100,0,0.1,name="angRes",label=r"Angular Resolution $\sqrt{\sigma_\eta^2 + \sigma_\phi^2}$")
ele_dxy = Regular(100,0,20,name="dxy",label="Electron Track $d_{xy}$ [cm]")
ele_dxySignif = Regular(150,0,150,name="dxy_signif",label=r"Electron Track $d_{xy}/\sigma_{d_{xy}}$")
ele_dz = Regular(100,0,5,name="dz",label="Electron Track $d_{z}$ [cm]")

# Dielectrons
ee_mass = Regular(100,0,4,name="mass",label="$m_{e^+e^-}$ [GeV]")
ee_dr = Regular(100,0,1,name='dr',label=r"$\Delta R$")
ee_dr_narrow = Regular(100,0,0.2,name='dr',label=r"$\Delta R$")
ee_ctau = Regular(100, 0, 1000, name="ctau", label=r"$c\tau$ [mm]")
ee_ctau_pr = Regular(100, 0, 500, name="ctau", label=r"$c\tau$ [mm]")

# Vertices
vxy = Regular(2000,0,50,name="vxy",label="$v_{xy}$ [cm]")
vxy_coarse = Regular(100,0,50,name="vxy",label="$v_{xy}$ [cm]")
vxy_zoom_fine = Regular(250,0,25,name="vxy",label="$v_{xy}$ [cm]")
vxy_zoom = Regular(200,0,20,name="vxy",label="$v_{xy}$ [cm]")
vxy_zoomzoom = Regular(500,0,5,name="vxy",label="$v_{xy}$ [cm]")
vxy_projected = Regular(4000,-50,50,name="vxy_projected",label="$v_{xy}$ [cm]")
vz = Regular(2000,0,50,name="vz",label="$v_{z}$ [cm]")
vz_coarse = Regular(100,0,50,name="vz",label="$v_{z}$ [cm]")
vz_fine = Regular(250,0,50,name="vz",label="$v_{z}$ [cm]")
mass = Regular(100,0,400,name="mass",label="$m_{e^+e^-}$ [GeV]")
energy = Regular(200,0,100,name="energy",label="Dielectron Energy [GeV]")
dR_over_pT = Regular(100,0,10,name="dR_over_pT",label=r"$\Delta R/p_T$")
dR_over_m = Regular(100,0,5,name="dR_over_m",label=r"$\Delta R/m$")
dR_over_pTm = Regular(100,0,20,name="dR_over_pTm",label=r"$\Delta R/(p_T/m)$")
dR_over_mpT = Regular(100,0,10,name="dR_over_mpT",label=r"$\Delta R/(m/p_T)$")
METdPhi_over_pT = Regular(100,0,10,name="METdPhi_over_pT",label=r"$\Delta \phi/p_T$")
METdPhi_over_m = Regular(100,0,20,name="METdPhi_over_m",label=r"$\Delta \phi/m$")
METdPhi_over_pTm = Regular(100,0,10,name="METdPhi_over_pTm",label=r"$\Delta \phi/(p_T/m)$")
METdPhi_over_mpT = Regular(100,0,50,name="METdPhi_over_mpT",label=r"$\Delta \phi/(m/p_T)$")
vtx_vxyErr = Regular(100,0,10,name="sigma_vxy",label="$v_{xy}$ Error")
vtx_chi2 = Regular(150,0,30,name="chi2",label=r"Vertex Fit $\chi^2/df$")
vtx_vxySignif = Regular(100,0,100,name="vxy_signif",label="Vertex $v_{xy}$ Significance")
vtx_prob = Regular(100,0,1,name="prob",label=r"Vertex $\chi^2$ Probability")
vtx_pt = Regular(100,0,200,name="pt",label="Selected Vertex $p_T$")
vtx_mass = Regular(100,0,1,name="mass",label="$m_{e^+e^-}$ [GeV]")
ee_met_dphi = Regular(100,0,0.5,name="dphi",label=r"$\Delta \phi$")

# Gen-matching
etype = Regular(4,0,4,name="Etype",label="Electron Match Type")
ptype = Regular(4,0,4,name="Ptype",label="Positron Match Type")
nEmatch = Regular(5,0,5,name="nEmatch",label="Number of Gen Electron Matches")
nPmatch = Regular(5,0,5,name="nPmatch",label="Number of Gen Positron Matches")
matchClass = Regular(4,0,4,name="matchClass",label="Match Class")

# Generic quantities
sel_vtx_pt_over_m = Regular(5000,0,1000,name='sel_vtx_pt_over_m',label="sel_vtx_pt_over_m")
met_over_pt = Regular(100,0,2,name='met_over_pt',label=r"$p_T^\mathrm{miss}/p_T^{j_1}$")
deta = Regular(64,0,6.4,name='deta',label=r"$\Delta \eta$")
sign_eta = IntCategory([-1,1],name="sign_eta",label="eta product sign")
prod_eta = Regular(100,-10,10,name='prod_eta',label=r"$\eta_{e_1}*\eta_{e_2}$")
dphi = Regular(64,0,3.2,name="dphi",label=r"$\Delta \phi$")
dphiJ = Regular(64,0,3.2,name="dphiJ",label=r"$\Delta \phi$")
dR = Regular(100,0,6,name='dr',label=r"$\Delta R$")
dR_zoom = Regular(100,0,1,name='dr_zoom',label=r"$\Delta R$")
dRj = Regular(100,0,6,name='drj',label=r"$\Delta R$")
dphi = Regular(64,0,3.2,name="dphi",label=r"$\Delta \phi$")
dxy = Regular(40,0,0.2,name='dxy',label="$d_{xy}$ [cm]")
dxy_fine = Regular(100,0,0.2,name='dxy',label="$d_{xy}$ [cm]")
pfiso = Regular(100,0,1,name='pfiso',label=r"$I_{\mathrm{PF}}^{\mathrm{rel}}$")
#pfmet_pt = Regular(100,0,300,name='pfmet_pt',label="$I_{\mathrm{PF}}^{\mathrm{met}}$") #new
dphi_generic = Regular(32,0,3.2,name="dphi",label=r"$\Delta \phi$")
met_pt = Regular(100,0,400,name="met_pt",label="$p_T^{miss}$ [GeV]")
njets = Integer(0,8,name="njets",label="$N_{jets}$")
neles = Integer(0,4,name="neles",label="$N_{electrons}$")
nphos = Integer(0,4,name="nphos",label="$N_{photons}$")
btag = Regular(100,0,1,name="btag",label="DeepJet B-Tag Score")
jet_abseta = Regular(30,0,3,name="eta",label=r"$Leading Jet |\eta|$")
jet_pt = Regular(100,50,300,name="pt",label="Leading Jet $p_T$")
met = Regular(100,50,300,name='met',label='met')
cos_collinear = Regular(1000, -1, 1, name="cos_collinear", label=r"$cos(\theta_{collinear})$")
ctau = Regular(10000, 0, 1000, name="ctau", label=r"$c\tau$ [mm]")

# Electron ID
ele_id = Regular(50,-1,4,name="ele_id",label="Low $p_T$ electron ID Score")

#triggercut = Integer(0,1,name="triggercut",label="triggercut")
