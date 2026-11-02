import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import FWCore.Utilities.FileUtils as FileUtils
from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.Eras.Era_Run2_2016_HIPM_cff import Run2_2016_HIPM
from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL

#Run3 imports
from Configuration.Eras.Era_Run3_cff import Run3                   #corresponds to Run3 2022 (maybe)
from Configuration.Eras.Era_Run3_2023_cff import Run3_2023         #corresponds to Run3 2023
from Configuration.Eras.Era_Run3_2024_cff import Run3_2024         # 2024

import json
import sys

# argument parsing
options = VarParsing.VarParsing('analysis')
options.register('data',
        False,
        VarParsing.VarParsing.multiplicity.singleton,
        VarParsing.VarParsing.varType.bool,
        "Run on data (1) or MC (0)"
        )
options.register('signal',
        True,
        VarParsing.VarParsing.multiplicity.singleton,
        VarParsing.VarParsing.varType.bool,
        "Run on signal (1) or not (0")
options.register('year',
        "2022",

        VarParsing.VarParsing.multiplicity.singleton,
        VarParsing.VarParsing.varType.string,
        "Data/MC year")
options.register('numThreads',
        8,
        VarParsing.VarParsing.multiplicity.singleton,
        VarParsing.VarParsing.varType.int,
        "Number of threads (for CRAB vs non-CRAB execution)")
options.register("nEvents",
	-1,
	VarParsing.VarParsing.multiplicity.singleton,

        VarParsing.VarParsing.varType.int,
	"Number of events to process (defaults to all)")
options.register('flist',
        "",
        VarParsing.VarParsing.multiplicity.singleton,
        VarParsing.VarParsing.varType.string,
        "File list to ntuplize")
options.register('outfile',
        "test_output.root",
        VarParsing.VarParsing.multiplicity.singleton,
        VarParsing.VarParsing.varType.string,
        "Output file name")
options.register('selectionMode',
        "hlt",
        VarParsing.VarParsing.multiplicity.singleton,
        VarParsing.VarParsing.varType.string,
        "Criterion gating the full-event background stream; background only, must not be "
        "set when signal=1. 'metThreshold': ptmiss >= metThreshold. 'hlt': OR of hltSelectionPaths.")
options.register('metThreshold',
        200.0,
        VarParsing.VarParsing.multiplicity.singleton,
        VarParsing.VarParsing.varType.float,
        "ptmiss threshold (GeV) gating the full-event background stream, used when "
        "selectionMode == 'metThreshold'; irrelevant for signal")
options.register('hltSelectionPaths',
        "",
        VarParsing.VarParsing.multiplicity.list,
        VarParsing.VarParsing.varType.string,
        "Trigger paths ORed together when selectionMode == 'hlt'")
# VarParsing.register() appends a list default as a single nested item rather than
# extending, so the real default is set here via attribute assignment instead.
options.hltSelectionPaths = ["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60"]
options.register('slimOutfile',
        "",
        VarParsing.VarParsing.multiplicity.singleton,
        VarParsing.VarParsing.varType.string,
        "Output file for the slim, all-events background stream (background only). "
        "If left empty, derived by inserting '_slim' before the '.root' extension of outfile.")

options.parseArguments()

# selectionMode (and its associated knobs, metThreshold/hltSelectionPaths) only gate the
# full-readout background stream -- the plugin ignores it entirely for signal
# (passesPtMiss = isSignal || passesBkgSelection in ElectronSkimmer.cc). Reject an explicit
# override so a signal job doesn't silently carry a no-op selectionMode configuration.
_explicitArgs = set(arg.split('=', 1)[0] for arg in sys.argv[1:])
if options.signal and 'selectionMode' in _explicitArgs:
    raise ValueError("selectionMode is a background-only option and has no effect on signal "
                      "samples; remove it or set signal=0")

# file list
if ".txt" in options.flist:
    # list of files
    print("reading input file list: "+options.flist)
    options.inputFiles = FileUtils.loadListFromFile(options.flist)

else:
    # we have passed a file name directly
    options.inputFiles = options.flist

# globaltag
globaltag = ''
if options.year == '2016APV':
    globaltag = '106X_dataRun2_v37' if options.data else '106X_mcRun2_asymptotic_preVFP_v11'
    era = Run2_2016_HIPM
    recoEgammaTools_era = '2016preVFP-UL'
elif options.year == '2016':
    globaltag = '106X_dataRun2_v37' if options.data else '106X_mcRun2_asymptotic_v17'
    era = Run2_2016
    recoEgammaTools_era = '2016postVFP-UL'
elif options.year == '2017':
    globaltag = '106X_dataRun2_v37' if options.data else '106X_mc2017_realistic_v10'
    era = Run2_2017
    recoEgammaTools_era = '2017-UL'
elif options.year == '2018':
    globaltag = '106X_dataRun2_v37' if options.data else '106X_upgrade2018_realistic_v16_L1v1'
    era = Run2_2018
    recoEgammaTools_era = '2018-UL'

# Run3 below options following https://twiki.cern.ch/twiki/bin/viewauth/CMS/MultivariateElectronIdentificationRun3
elif options.year == '2022':
    globaltag = '130X_dataRun3_v2' if options.data else '130X_mcRun3_2022_realistic_v5'  
    #For data, you can also use 124X_dataRun3_PromptAnalysis_v1 
    era = Run3
    recoEgammaTools_era = '2022-Prompt' #XYZ FIX
    
elif options.year == '2023':
    globaltag = '130X_dataRun3_PromptAnalysis_v1' if options.data else '130X_mcRun3_2023_realistic_v14'
    era = Run3_2023
    recoEgammaTools_era = '2022-Prompt' #XYZ FIX

elif options.year == '2024': #XYZ FIX
    globaltag = '' if options.data else '150X_mcRun3_2024_realistic_v2'
    era = Run3_2024
    recoEgammaTools_era = '2022-Prompt' #XYZ FIX


else:
    print("Invalid year: {0}".format(options.year))

    exit

#######################
##### MET Filters #####
#######################
metFilters = []
if options.year == '2016' or options.year == '2016APV':
    metFilters = [
        "Flag_goodVertices",
        "Flag_globalSuperTightHalo2016Filter",
        "Flag_HBHENoiseFilter",
        "Flag_HBHENoiseIsoFilter",
        "EcalDeadCellTriggerPrimitiveFilter",
        "Flag_BadPFMuonFilter",
        "Flag_BadPFMuonDzFilter",
        "Flag_eeBadScFilter",
        "Flag_hfNoisyHitsFilter"
    ]
elif options.year in ['2017', '2018', '2022', '2023', '2024']:

    metFilters = [
        "Flag_goodVertices",
        "Flag_globalSuperTightHalo2016Filter",
        "Flag_HBHENoiseFilter",
        "Flag_HBHENoiseIsoFilter",
        "Flag_EcalDeadCellTriggerPrimitiveFilter",
        "Flag_BadPFMuonFilter",
        "Flag_BadPFMuonDzFilter",
        "Flag_hfNoisyHitsFilter",
        "Flag_eeBadScFilter",
        "Flag_ecalBadCalibFilter"
    ]

#######################
###### Triggers #######
#######################

# record all trigger paths that might be useful acrcoss all years - some will not always be available,
# but what's available will get written out to the ntuples
# Jet triggers (for MET trigger eff)
metTrigs = [
    #"HLT_PFMET90_PFMHT90_IDTight", # not included in Run 3
    #"HLT_PFMET100_PFMHT100_IDTight", # not included in Run 3
    #"HLT_PFMET110_PFMHT110_IDTight", # not included in Run 3
    "HLT_PFMET120_PFMHT120_IDTight",
    "HLT_PFMET120_PFMHT120_IDTight_PFHT60",
    "HLT_PFMET130_PFMHT130_IDTight",
    "HLT_PFMET140_PFMHT140_IDTight",
    "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF",
    "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight",
    "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF",
    "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60",
    "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight",
    "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_FilterHF",
    "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight",
    "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_FilterHF",
    #"HLT_PFMETTypeOne110_PFMHT110_IDTight", # not included in Run 3
    #"HLT_PFMETTypeOne120_PFMHT120_IDTight", # not included in Run 3
    #"HLT_PFMETTypeOne130_PFMHT130_IDTight", # not included in Run 3
    "HLT_PFMETTypeOne140_PFMHT140_IDTight",
    #"HLT_PFMET100_PFMHT100_IDTight_PFHT60_v9", # not included in Run 3
    "HLT_PFMET105_IsoTrk50",
]


jetTrigs = [
    #"HLT_PFJet15", # not included in Run 3
    #"HLT_PFJet25", # not included in Run 3
    "HLT_PFJet40",
    "HLT_PFJet60",
    "HLT_PFJet80",
    "HLT_PFJet140",
    "HLT_PFJet200",
    "HLT_PFJet260",
    "HLT_PFJet320",
    "HLT_PFJet400",
    "HLT_PFJet450",
    "HLT_PFJet500",
    "HLT_PFJet550",
    #"HLT_AK4PFJet30", # not included in Run 3
    #"HLT_AK4PFJet50", # not included in Run 3
    #"HLT_AK4PFJet80", # not included in Run 3
    #"HLT_AK4PFJet100", # not included in Run 3
    #"HLT_AK4PFJet120", # not included in Run 3
]
eleTrigs = list(set([
    "HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
    "HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned",
    "HLT_Ele28_eta2p1_WPTight_Gsf_HT150",
    #"HLT_Ele27_Ele37_CaloIdL_MW",
    #"HLT_DoubleEle25_CaloIdL_MW",
    "HLT_DoubleEle27_CaloIdL_MW",
    "HLT_DoubleEle33_CaloIdL_MW",
    "HLT_DoubleEle24_eta2p1_WPTight_Gsf",
    #"HLT_Ele20_WPTight_Gsf", # not included in Run 3
    #"HLT_Ele15_WPLoose_Gsf", # not included in Run 3
    #"HLT_Ele17_WPLoose_Gsf", # not included in Run 3
    #"HLT_Ele20_WPLoose_Gsf", # not included in Run 3
    "HLT_Ele27_WPTight_Gsf",
    "HLT_Ele28_WPTight_Gsf",
    "HLT_Ele30_WPTight_Gsf",
    "HLT_Ele32_WPTight_Gsf",
    "HLT_Ele32_WPTight_Gsf_L1DoubleEG",
    "HLT_Ele35_WPTight_Gsf",
    "HLT_Ele35_WPTight_Gsf_L1EGMT",
    "HLT_Ele38_WPTight_Gsf",
    "HLT_Ele40_WPTight_Gsf",
    "HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL",
    "HLT_Ele15_Ele8_CaloIdL_TrackIdL_IsoVL",
    "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
    "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
    "HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30",
    "HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30",
    #"HLT_Ele15_CaloIdL_TrackIdL_IsoVL_PFJet30", # not included in Run 3
    "HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30",
    "HLT_Ele8_CaloIdM_TrackIdM_PFJet30",
    "HLT_Ele17_CaloIdM_TrackIdM_PFJet30",
    "HLT_Ele23_CaloIdM_TrackIdM_PFJet30",
    "HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
    "HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350",
    "HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350"
]))

muTrigs = [
     "HLT_IsoMu27"
]

triggerPaths = metTrigs + jetTrigs + eleTrigs + muTrigs

# Electron effective area input file for PU-corrected PF isolation calculations
effAreaInputPath = "RecoEgamma/ElectronIdentification/data/Run3_Winter22/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_122X.txt"

process = cms.Process("USER",era)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.EventContent.EventContent_cff")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, globaltag, '')

process.options = cms.untracked.PSet()
process.options.numberOfThreads=cms.untracked.uint32(options.numThreads)
process.options.numberOfStreams=cms.untracked.uint32(0)
process.options.numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(1)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.nEvents)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles),
    #skipBadFiles = cms.untracked.bool(True),
    #cacheSize = cms.untracked.uint32(0)
    skipBadFiles = cms.untracked.bool(True)
)
process.TFileService = cms.Service("TFileService",
    fileName = cms.string(options.outfile),
    closeFileFast = cms.untracked.bool(True)
)

##############################
###### Main iDM analyzer #####
##############################
# import ntuplizer
from iDMe.AODSkimmer.ElectronSkimmer_cfi import ElectronSkimmer
process.ntuples = ElectronSkimmer.clone(
    isData = cms.bool(options.data),
    isSignal = cms.bool(options.signal),
    year = options.year,
    selectionMode = cms.string(options.selectionMode),
    metThreshold = cms.double(options.metThreshold),
    hltSelectionPaths = cms.vstring(options.hltSelectionPaths),
    slimOutfile = cms.string(options.slimOutfile),
    metFilters = cms.vstring(metFilters),
    triggerPaths = cms.vstring(triggerPaths),
    effAreasConfigFile = cms.FileInPath(effAreaInputPath),
    displacedStandAloneMuons = cms.InputTag("displacedStandAloneMuons")
)

# import EGamma postreco tools
# old version
#from RecoEgamma.EgammaTools.EgammaPostRecoTools import setupEgammaPostRecoSeq
# Run3 working version?
from EgammaUser.EgammaPostRecoTools.EgammaPostRecoTools import setupEgammaPostRecoSeq
# Run3 following https://twiki.cern.ch/twiki/bin/viewauth/CMS/MultivariateElectronIdentificationRun3
setupEgammaPostRecoSeq(process,
                       runEnergyCorrections=False, # XYZ deactivated bc not working for Run3 yet, I think?
                       runVID=True, #saves CPU time by not needlessly re-running VID, if you want the Fall17V2 IDs, set this to True or remove (default is True)
                       era=recoEgammaTools_era,
                       eleIDModules=['RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_iso_V1_cff',
                                     'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_noIso_V1_cff',
                                     'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Winter22_122X_V1_cff']
                       )


# load nanoAOD producer chain for low-pT electrons -- computes mini iso
process.load('PhysicsTools.NanoAOD.lowPtElectrons_cff')
process.lowPtNanoElectronSequence = cms.Sequence(process.modifiedLowPtElectrons+\
                                         process.updatedLowPtElectrons+\
                                         process.lowPtPATElectronID+\
                                         process.isoForLowPtEle+\
                                         process.updatedLowPtElectronsWithUserData)

# load nanoAOD producer for electrons - modify to just compute PFiso and miniIso
process.load('PhysicsTools.NanoAOD.electrons_cff')
process.isoForEleRelative = process.isoForEle.clone( # configure isolation to compute relative isolation by default
    relative = cms.bool(True)
)
process.slimmedElectronsWithUserDataMinimal = process.slimmedElectronsWithUserData.clone( # remove need to save non-isolation value maps
    src = cms.InputTag("slimmedElectrons"),
    userFloats = cms.PSet(
        miniIsoChg = cms.InputTag("isoForEleRelative:miniIsoChg"),
        miniIsoAll = cms.InputTag("isoForEleRelative:miniIsoAll"),
        #PFIsoChg = cms.InputTag("isoForEleRelative:PFIsoChg"), # Doesn't work for Run 3
        #PFIsoAll = cms.InputTag("isoForEleRelative:PFIsoAll"), # Doesn't work for Run 3
        #PFIsoAll04 = cms.InputTag("isoForEleRelative:PFIsoAll04"), # Doesn't work for Run 3
    ),
    userIntFromBools = cms.PSet(),
    userInts = cms.PSet(),
    userCands = cms.PSet()
)
process.nanoElectronSequence = cms.Sequence(process.isoForEleRelative+\
                                            process.slimmedElectronsWithUserDataMinimal)

## Define paths and schedule
process.ntupleSequence = cms.Sequence(process.ntuples)
process.ntuplePath = cms.Path(process.ntupleSequence)

process.iDMEgammaPostRecoSequence = cms.Sequence(process.egammaPostRecoSeq)
process.iDMEgammaPostReco = cms.Path(process.iDMEgammaPostRecoSequence)

process.iDMNanoElectronSequence = cms.Sequence(process.lowPtNanoElectronSequence + process.nanoElectronSequence)
process.iDMNanoElectron = cms.Path(process.iDMNanoElectronSequence)

process.schedule = cms.Schedule(process.iDMEgammaPostReco,process.iDMNanoElectron,process.ntuplePath)

