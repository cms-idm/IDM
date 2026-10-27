"""Hierarchy matching with and without the MiniAOD displaced-muon filter (study note: kb/studies/slimmed-displaced-muon-loss.md).
cmsRun slimmed_loss_cfg.py flist=<original-list|root> sample=<name> outFile=<new.root> revision=<sha> [nEvents=N]
Case A = preselected_vertexing_cfg.py unchanged (displacedMuons -> cleanedDisplacedMuons -> preselectedVertexing).
Case B = displacedMuons -> filteredDisplacedMuons (CMSSW DisplacedMuonFilterProducer, its cfi defaults, the input of
slimmedDisplacedMuons) -> cleanedFilteredDisplacedMuons -> preselectedVertexingB (same matching parameters).
slimmedCheck validates the filter replication and filtered <-> stored slimmedDisplacedMuons per event.
"""
import os
import FWCore.ParameterSet.Config as cms

exec(open(os.path.join(os.environ["CMSSW_BASE"], "src/iDMe/AODSkimmer/displaced_muon_id/preselected_vertexing_cfg.py")).read())

process.load("PhysicsTools.PatAlgos.recoLayer0.filteredDisplacedMuons_cfi")
process.filteredDisplacedMuons.srcMuons = cms.InputTag("displacedMuons", "", "RECO")
process.cleanedFilteredDisplacedMuons = process.cleanedDisplacedMuons.clone(muons=cms.InputTag("filteredDisplacedMuons"))
process.preselectedVertexingB = process.preselectedVertexing.clone(
    muons=cms.InputTag("cleanedFilteredDisplacedMuons"),
    originalIndex=cms.InputTag("cleanedFilteredDisplacedMuons", "originalIndex"))
process.slimmedCheck = cms.EDAnalyzer("SlimmedDisplacedFilterCheck",
    raw=cms.InputTag("displacedMuons", "", "RECO"), filtered=cms.InputTag("filteredDisplacedMuons"),
    slimmed=cms.InputTag("slimmedDisplacedMuons"),
    minPtSTA=process.filteredDisplacedMuons.minPtSTA, minPtTK=process.filteredDisplacedMuons.minPtTK,
    minMatches=process.filteredDisplacedMuons.minMatches)
process.p = cms.Path(process.cleanedDisplacedMuons + process.preselectedVertexing + process.filteredDisplacedMuons +
                     process.cleanedFilteredDisplacedMuons + process.preselectedVertexingB + process.slimmedCheck)
