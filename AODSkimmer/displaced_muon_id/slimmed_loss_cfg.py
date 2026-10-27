"""Hierarchy matching with and without the MiniAOD displaced-muon filter (study note: kb/studies/slimmed-displaced-muon-loss.md).
cmsRun slimmed_loss_cfg.py flist=<original-list|root> sample=<name> outFile=<new.root> revision=<sha> [nEvents=N]
Case A = preselected_vertexing_cfg.py unchanged (displacedMuons -> cleanedDisplacedMuons -> preselectedVertexing).
Case B = displacedMuons -> filteredDisplacedMuons (SlimmedDisplacedFilterCheck: the raw muons with an exact partner in the stored
slimmedDisplacedMuons, i.e. what the MiniAOD filter kept; cuts replicated only as diagnostics) -> cleanedFilteredDisplacedMuons ->
preselectedVertexingB (same matching parameters).
"""
import os
import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.recoLayer0.filteredDisplacedMuons_cfi import filteredDisplacedMuons as _cmsswFilter

exec(open(os.path.join(os.environ["CMSSW_BASE"], "src/iDMe/AODSkimmer/displaced_muon_id/preselected_vertexing_cfg.py")).read())

process.filteredDisplacedMuons = cms.EDProducer("SlimmedDisplacedFilterCheck",
    raw=cms.InputTag("displacedMuons", "", "RECO"), slimmed=cms.InputTag("slimmedDisplacedMuons"),
    minPtSTA=_cmsswFilter.minPtSTA, minPtTK=_cmsswFilter.minPtTK, minMatches=_cmsswFilter.minMatches)
process.cleanedFilteredDisplacedMuons = process.cleanedDisplacedMuons.clone(muons=cms.InputTag("filteredDisplacedMuons"))
process.preselectedVertexingB = process.preselectedVertexing.clone(
    muons=cms.InputTag("cleanedFilteredDisplacedMuons"),
    originalIndex=cms.InputTag("cleanedFilteredDisplacedMuons", "originalIndex"))
process.p = cms.Path(process.cleanedDisplacedMuons + process.preselectedVertexing + process.filteredDisplacedMuons +
                     process.cleanedFilteredDisplacedMuons + process.preselectedVertexingB)
