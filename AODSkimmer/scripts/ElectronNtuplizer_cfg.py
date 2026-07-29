import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import FWCore.Utilities.FileUtils as FileUtils
from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.Eras.Era_Run2_2016_HIPM_cff import Run2_2016_HIPM

# Run3 imports
from Configuration.Eras.Era_Run3_cff import Run3

import sys


# argument parsing
options = VarParsing.VarParsing("analysis")

options.register(
    "data",
    False,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.bool,
    "Run on data (1) or MC (0)"
)

options.register(
    "signal",
    True,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.bool,
    "Run on signal (1) or not (0)"
)

options.register(
    "year",
    "2022EE",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Data/MC year"
)

options.register(
    "numThreads",
    8,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of threads"
)

options.register(
    "nEvents",
    -1,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of events to process"
)

options.register(
    "flist",
    "",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Input file list or single input file"
)

options.register(
    "outfile",
    "test_output.root",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Output ROOT file"
)

options.parseArguments()


# input files
if ".txt" in options.flist:
    print("reading input file list: " + options.flist)
    options.inputFiles = FileUtils.loadListFromFile(options.flist)
else:
    options.inputFiles = options.flist


# global tag / era
globaltag = ""

if options.year == "2016APV":
    globaltag = "106X_dataRun2_v37" if options.data else "106X_mcRun2_asymptotic_preVFP_v11"
    era = Run2_2016_HIPM
    recoEgammaTools_era = "2016preVFP-UL"

elif options.year == "2016":
    globaltag = "106X_dataRun2_v37" if options.data else "106X_mcRun2_asymptotic_v17"
    era = Run2_2016
    recoEgammaTools_era = "2016postVFP-UL"

elif options.year == "2017":
    globaltag = "106X_dataRun2_v37" if options.data else "106X_mc2017_realistic_v10"
    era = Run2_2017
    recoEgammaTools_era = "2017-UL"

elif options.year == "2018":
    globaltag = "106X_dataRun2_v37" if options.data else "106X_upgrade2018_realistic_v16_L1v1"
    era = Run2_2018
    recoEgammaTools_era = "2018-UL"

elif options.year in ["2022", "2022EE"]:
    globaltag = "130X_dataRun3_v2" if options.data else "130X_mcRun3_2022_realistic_v5"
    era = Run3
    recoEgammaTools_era = "2022-Prompt"

else:
    print("Invalid or unsupported year for this CMSSW_13_0_13 cfg: {0}".format(options.year))
    print("Supported years here: 2016APV, 2016, 2017, 2018, 2022, 2022EE")
    sys.exit(1)


#######################
##### MET Filters #####
#######################

metFilters = []

if options.year in ["2016", "2016APV"]:
    metFilters = [
        "Flag_goodVertices",
        "Flag_globalSuperTightHalo2016Filter",
        "Flag_HBHENoiseFilter",
        "Flag_HBHENoiseIsoFilter",
        "EcalDeadCellTriggerPrimitiveFilter",
        "Flag_BadPFMuonFilter",
        "Flag_BadPFMuonDzFilter",
        "Flag_eeBadScFilter",
        "Flag_hfNoisyHitsFilter",
    ]

elif options.year in ["2017", "2018", "2022", "2022EE"]:
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
        "Flag_ecalBadCalibFilter",
    ]


#######################
###### Triggers #######
#######################

metTrigs = [
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
    "HLT_PFMETTypeOne140_PFMHT140_IDTight",
    "HLT_PFMET105_IsoTrk50",
]

jetTrigs = [
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
]

eleTrigs = list(set([
    "HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
    "HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned",
    "HLT_Ele28_eta2p1_WPTight_Gsf_HT150",
    "HLT_DoubleEle27_CaloIdL_MW",
    "HLT_DoubleEle33_CaloIdL_MW",
    "HLT_DoubleEle24_eta2p1_WPTight_Gsf",
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
    "HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30",
    "HLT_Ele8_CaloIdM_TrackIdM_PFJet30",
    "HLT_Ele17_CaloIdM_TrackIdM_PFJet30",
    "HLT_Ele23_CaloIdM_TrackIdM_PFJet30",
    "HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350",
    "HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350",
]))

muTrigs = [
    "HLT_IsoMu27",
]

triggerPaths = metTrigs + jetTrigs + eleTrigs + muTrigs


# Electron effective area input file for PU-corrected PF isolation calculations
effAreaInputPath = (
    "RecoEgamma/ElectronIdentification/data/Run3_Winter22/"
    "effAreaElectrons_cone03_pfNeuHadronsAndPhotons_122X.txt"
)


#######################
###### Process ########
#######################

process = cms.Process("USER", era)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# Required for both:
#   1. PropagateToMuon station-1/station-2 diagnostics
#   2. direct propagation to the DSA outermost-hit surface
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, globaltag, "")

process.options = cms.untracked.PSet()
process.options.numberOfThreads = cms.untracked.uint32(options.numThreads)
process.options.numberOfStreams = cms.untracked.uint32(0)
process.options.numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.nEvents)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles),
    skipBadFiles = cms.untracked.bool(True)
)

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string(options.outfile),
    closeFileFast = cms.untracked.bool(True)
)


##############################
###### Main iDM analyzer #####
##############################

from iDMe.AODSkimmer.ElectronSkimmer_cfi import ElectronSkimmer

process.ntuples = ElectronSkimmer.clone(
    isData = cms.bool(options.data),
    isSignal = cms.bool(options.signal),
    year = cms.string(options.year),
    metFilters = cms.vstring(metFilters),
    triggerPaths = cms.vstring(triggerPaths),
    effAreasConfigFile = cms.FileInPath(effAreaInputPath),
    displacedStandAloneMuons = cms.InputTag("displacedStandAloneMuons"),

    # Strict propagation to station 1.
    genMuonPropagatorSt1 = cms.PSet(
        useSimpleGeometry = cms.bool(True),
        useStation2 = cms.bool(False),
        fallbackToME1 = cms.bool(False),
        cosmicPropagationHypothesis = cms.bool(False),
        useMB2InOverlap = cms.bool(False),
        useTrack = cms.string("none"),
        useState = cms.string("atVertex"),
        propagatorAlong = cms.ESInputTag(
            "", "SteppingHelixPropagatorAlong"
        ),
        propagatorAny = cms.ESInputTag(
            "", "SteppingHelixPropagatorAny"
        ),
        propagatorOpposite = cms.ESInputTag(
            "", "SteppingHelixPropagatorOpposite"
        ),
    ),

    # Strict propagation to station 2 without ME1 fallback.
    genMuonPropagatorSt2 = cms.PSet(
        useSimpleGeometry = cms.bool(True),
        useStation2 = cms.bool(True),
        fallbackToME1 = cms.bool(False),
        cosmicPropagationHypothesis = cms.bool(False),
        useMB2InOverlap = cms.bool(False),
        useTrack = cms.string("none"),
        useState = cms.string("atVertex"),
        propagatorAlong = cms.ESInputTag(
            "", "SteppingHelixPropagatorAlong"
        ),
        propagatorAny = cms.ESInputTag(
            "", "SteppingHelixPropagatorAny"
        ),
        propagatorOpposite = cms.ESInputTag(
            "", "SteppingHelixPropagatorOpposite"
        ),
    ),

    # Used explicitly for MB3/ME3 and MB4/ME4.
    stationPropagatorAlong = cms.ESInputTag(
        "", "SteppingHelixPropagatorAlong"
    ),
)


##############################
###### EGamma post-reco ######
##############################

from RecoEgamma.EgammaTools.EgammaPostRecoTools import setupEgammaPostRecoSeq

setupEgammaPostRecoSeq(
    process,
    runEnergyCorrections = False,
    runVID = True,
    era = recoEgammaTools_era,
    eleIDModules = [
        "RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_iso_V1_cff",
        "RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_noIso_V1_cff",
        "RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Winter22_122X_V1_cff",
    ]
)


##################################
###### Low-pT electron chain #####
##################################

process.load("PhysicsTools.NanoAOD.lowPtElectrons_cff")

process.lowPtNanoElectronSequence = cms.Sequence(
    process.modifiedLowPtElectrons +
    process.updatedLowPtElectrons +
    process.lowPtPATElectronID +
    process.isoForLowPtEle +
    process.updatedLowPtElectronsWithUserData
)


##################################
###### Regular electron chain ####
##################################

process.load("PhysicsTools.NanoAOD.electrons_cff")

process.isoForEleRelative = process.isoForEle.clone(
    relative = cms.bool(True)
)

process.slimmedElectronsWithUserDataMinimal = process.slimmedElectronsWithUserData.clone(
    src = cms.InputTag("slimmedElectrons"),
    userFloats = cms.PSet(
        miniIsoChg = cms.InputTag("isoForEleRelative:miniIsoChg"),
        miniIsoAll = cms.InputTag("isoForEleRelative:miniIsoAll"),
    ),
    userIntFromBools = cms.PSet(),
    userInts = cms.PSet(),
    userCands = cms.PSet()
)

process.nanoElectronSequence = cms.Sequence(
    process.isoForEleRelative +
    process.slimmedElectronsWithUserDataMinimal
)


##############################
###### Paths and schedule ####
##############################

process.ntupleSequence = cms.Sequence(process.ntuples)
process.ntuplePath = cms.Path(process.ntupleSequence)

process.iDMEgammaPostRecoSequence = cms.Sequence(process.egammaPostRecoSeq)
process.iDMEgammaPostReco = cms.Path(process.iDMEgammaPostRecoSequence)

process.iDMNanoElectronSequence = cms.Sequence(
    process.lowPtNanoElectronSequence +
    process.nanoElectronSequence
)
process.iDMNanoElectron = cms.Path(process.iDMNanoElectronSequence)

process.schedule = cms.Schedule(
    process.iDMEgammaPostReco,
    process.iDMNanoElectron,
    process.ntuplePath
)