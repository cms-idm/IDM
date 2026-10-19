"""Deduplicated displacedMuons, the default muon input for analysis (kb/DECISIONS.md 2026-10-19).

Load it and put process.cleanedDisplacedMuons first in the Path; consumers read
cms.InputTag("cleanedDisplacedMuons") (plus instances "originalIndex" and "nRaw").
Duplicate studies keep reading raw displacedMuons.
"""
import FWCore.ParameterSet.Config as cms

cleanedDisplacedMuons = cms.EDProducer(
    "DisplacedMuonDuplicateCleaner",
    muons=cms.InputTag("displacedMuons", "", "RECO"),
    validate=cms.bool(False),
)
