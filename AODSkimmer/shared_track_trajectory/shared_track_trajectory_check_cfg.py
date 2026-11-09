"""cmsRun config for SharedTrackTrajectoryCheck (IDM/AODSkimmer/plugins/SharedTrackTrajectoryCheck.cc).

  cmsRun shared_track_trajectory_check_cfg.py flist=files.txt [nEvents=-1] [outPrefix=shared_track_trajectory]
  cmsRun shared_track_trajectory_check_cfg.py eventsCsv=shared_inner_track_muon_locations.csv
      (only the files and run:lumi:event listed in that CSV; uses its file_url, run, lumi, event columns)
  python3 summarize_shared_track_trajectory.py --tracks <outPrefix>_tracks.csv --items <outPrefix>_items.csv

Input: the Run3Summer22EE MiniAOD files (CMSSW_13_0_13, GT 130X_mcRun3_2022_realistic_postEE_v6).
Gen: prunedGenParticles, status 1, |pdgId| == 13, no ancestry requirement.
Tolerances are read from this release's muons1stStep config (displacedMuons1stStep is a clone of it).
The displacedMuons associations were made in CMSSW_12_4_14_patch3: compare the printed values with that release.
"""
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3
from RecoMuon.MuonIdentification.muons1stStep_cfi import muons1stStep

options = VarParsing("analysis")
options.register("flist", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "text file, one input file per line")
options.register("eventsCsv", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "CSV with file_url,run,lumi,event")
options.register("nEvents", -1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "events to process")
options.register("outPrefix", "shared_track_trajectory", VarParsing.multiplicity.singleton, VarParsing.varType.string, "output prefix")
options.parseArguments()

process = cms.Process("SHAREDTRAJ", Run3)
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

tap = muons1stStep.TrackAssociatorParameters
process.sharedTrackTrajectoryCheck = cms.EDAnalyzer(
    "SharedTrackTrajectoryCheck",
    muons=cms.InputTag("displacedMuons"),
    genParticles=cms.InputTag("prunedGenParticles"),
    drThresholds=cms.vdouble(0.1, 0.06),
    maxAbsDx=muons1stStep.maxAbsDx,                # segment-to-track x criterion
    maxAbsDy=muons1stStep.maxAbsDy,                # segment-to-track y criterion
    muonMaxDistanceX=tap.muonMaxDistanceX,         # track-to-chamber edge criterion
    muonMaxDistanceY=tap.muonMaxDistanceY,
    rpcMaxAbsDx=cms.double(20.0),                  # hard-coded in MuonIdProducer::fillMuonId (absDx <= 20)
    tracksFile=cms.string(options.outPrefix + "_tracks.csv"),
    itemsFile=cms.string(options.outPrefix + "_items.csv"),
)
process.p = cms.Path(process.sharedTrackTrajectoryCheck)
