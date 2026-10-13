import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.AlCa.GlobalTag import GlobalTag


process = cms.Process("INSPECT", Run3)

process.load(
    "Configuration.StandardSequences.GeometryRecoDB_cff"
)

process.load(
    "Configuration.StandardSequences.MagneticField_cff"
)

process.load(
    "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff"
)


process.GlobalTag = GlobalTag(
    process.GlobalTag,
    "130X_mcRun3_2022_realistic_postEE_v6",
    ""
)


process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(1)
)


process.source = cms.Source(
    "PoolSource",

    fileNames=cms.untracked.vstring(
        "root://cmseos.fnal.gov//store/user/lpcmetx/iDMe/"
        "iDMmu_Run3_Production/2022EE/MINIAOD/"
        "Mchi-12p0_dMchi-4p0/ctau-1/"
        "iDMmu_pythiaGenFragment_ctau-1__chain_Run3Summer22EE/"
        "2022EE_Mchi-12p0_dMchi-4p0_ctau-1_20260928_143801/"
        "260928_193838/0000/"
        "Run3Summer22EEMiniAOD_1.root"
    ),

    eventsToProcess=cms.untracked.VEventRange(
        "1:10001:832"
    ),
)


process.inspectSharedMuonDuplicate = cms.EDAnalyzer(
    "InspectSharedMuonDuplicate",

    muons=cms.InputTag(
        "displacedMuons"
    ),

    # The shared-inner-track pair in event 1:10001:832.
    firstIndex=cms.uint32(0),
    secondIndex=cms.uint32(2),
)


process.p = cms.Path(
    process.inspectSharedMuonDuplicate
)