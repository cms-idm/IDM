"""cmsRun config for SharedInnerTrackDuplicatePolicy (iDMe/AODSkimmer/plugins/SharedInnerTrackDuplicatePolicy.cc).

  cmsRun shared_inner_track_duplicate_policy_cfg.py flist=files.txt [nEvents=-1] [outPrefix=shared_inner_track_duplicate_policy]
  cmsRun shared_inner_track_duplicate_policy_cfg.py eventsCsv=shared_inner_track_muon_locations.csv
      (only the files and run:lumi:event listed in that CSV; uses its file_url, run, lumi, event columns)
  python3 summarize_shared_inner_track_duplicate_policy.py --all <outPrefix>.csv --genmatched <outPrefix>_genmatched.csv

Input: the Run3Summer22EE MiniAOD files (CMSSW_13_0_13, GT 130X_mcRun3_2022_realistic_postEE_v6).
Gen: prunedGenParticles, status 1, |pdgId| == 13, no ancestry requirement; matched at the ECAL inner boundary
exactly as in shared_track_trajectory_check_cfg.py. The largest drThreshold is the gen-matched CSV inclusion cut.
Exclusive matching: a gen muon is eligible only if the shared track is its closest displacedMuon inner track.
Forward/behind: the exclusive gen muon propagated to each copy's chambers (along / opposite propagators).
"""
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3

options = VarParsing("analysis")
options.register("flist", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "text file, one input file per line")
options.register("eventsCsv", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "CSV with file_url,run,lumi,event")
options.register("nEvents", -1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "events to process")
options.register("outPrefix", "shared_inner_track_duplicate_policy", VarParsing.multiplicity.singleton, VarParsing.varType.string, "output prefix")
options.parseArguments()

process = cms.Process("DUPPOLICY", Run3)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10000
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "130X_mcRun3_2022_realistic_postEE_v6", "")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
process.load("TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff")

if options.eventsCsv:
    import csv
    with open(options.eventsCsv) as src:
        rows = list(csv.DictReader(src))
    files = sorted({r["file_url"] for r in rows})
    events = sorted({(int(r["run"]), int(r["lumi"]), int(r["event"])) for r in rows})
    process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(files),
                                eventsToProcess=cms.untracked.VEventRange([f"{r}:{l}:{e}" for r, l, e in events]))
    print(f"eventsCsv: {len(files)} files, {len(events)} run:lumi:event")
else:
    with open(options.flist) as src:
        files = [line.strip() for line in src if line.strip() and not line.startswith("#")]
    process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(files))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.nEvents))

process.sharedInnerTrackDuplicatePolicy = cms.EDAnalyzer(
    "SharedInnerTrackDuplicatePolicy",
    muons=cms.InputTag("displacedMuons"),
    genParticles=cms.InputTag("prunedGenParticles"),
    drThresholds=cms.vdouble(0.10, 0.06),
    policyFile=cms.string(options.outPrefix + ".csv"),
    genMatchedFile=cms.string(options.outPrefix + "_genmatched.csv"),
)
process.p = cms.Path(process.sharedInnerTrackDuplicatePolicy)
