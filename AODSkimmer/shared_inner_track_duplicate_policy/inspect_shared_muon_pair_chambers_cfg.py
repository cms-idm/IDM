"""cmsRun config for InspectSharedMuonPairChambers (iDMe/AODSkimmer/plugins/InspectSharedMuonPairChambers.cc).

The 4 shared-innerTrack displacedMuon pairs that do not cross the IP and are matched exclusively to a same-sign
gen muon (dR < 0.06 at the ECAL inner boundary). genIndex = that gen muon's index in prunedGenParticles.

  cmsRun inspect_shared_muon_pair_chambers_cfg.py 2>&1 | tee inspect_pair_chambers.log
"""
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.AlCa.GlobalTag import GlobalTag

base = "root://cmseos.fnal.gov//store/user/lpcmetx/iDMe/iDMmu_Run3_Production/2022EE/MINIAOD/"
# event, A, B, genIndex, file
pairs = [
    ("1:10001:832", 0, 2, 19, "Mchi-12p0_dMchi-4p0/ctau-1/iDMmu_pythiaGenFragment_ctau-1__chain_Run3Summer22EE/2022EE_Mchi-12p0_dMchi-4p0_ctau-1_20260928_143801/260928_193838/0000/Run3Summer22EEMiniAOD_1.root"),
    ("1:10003:1500", 0, 1, 27, "Mchi-24p0_dMchi-8p0/ctau-1000/iDMmu_pythiaGenFragment_ctau-1000__chain_Run3Summer22EE/2022EE_Mchi-24p0_dMchi-8p0_ctau-1000_20260928_150853/260928_200959/0000/Run3Summer22EEMiniAOD_3.root"),
    ("1:10001:782", 0, 4, 12, "Mchi-31p5_dMchi-3p0/ctau-10/iDMmu_pythiaGenFragment_ctau-10__chain_Run3Summer22EE/2022EE_Mchi-31p5_dMchi-3p0_ctau-10_20260928_145002/260928_195035/0000/Run3Summer22EEMiniAOD_1.root"),
    ("1:10003:1595", 1, 2, 22, "Mchi-55p0_dMchi-10p0/ctau-1/iDMmu_pythiaGenFragment_ctau-1__chain_Run3Summer22EE/2022EE_Mchi-55p0_dMchi-10p0_ctau-1_20260928_144425/260928_194455/0000/Run3Summer22EEMiniAOD_3.root"),
]

process = cms.Process("INSPECTPAIR", Run3)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag = GlobalTag(process.GlobalTag, "130X_mcRun3_2022_realistic_postEE_v6", "")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")

process.source = cms.Source("PoolSource",
    fileNames=cms.untracked.vstring(sorted({base + f for _, _, _, _, f in pairs})),
    eventsToProcess=cms.untracked.VEventRange([e for e, _, _, _, _ in pairs]))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))

process.inspectSharedMuonPairChambers = cms.EDAnalyzer("InspectSharedMuonPairChambers",
    muons=cms.InputTag("displacedMuons"),
    genParticles=cms.InputTag("prunedGenParticles"),
    pairs=cms.VPSet([cms.PSet(event=cms.string(e), A=cms.uint32(a), B=cms.uint32(b), genIndex=cms.int32(g))
                     for e, a, b, g, _ in pairs]),
)
process.p = cms.Path(process.inspectSharedMuonPairChambers)
