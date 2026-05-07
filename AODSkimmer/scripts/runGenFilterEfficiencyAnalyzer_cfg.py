import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import FWCore.Utilities.FileUtils as FileUtils

options = VarParsing.VarParsing('analysis')
options.register('flist',
        "",
        VarParsing.VarParsing.multiplicity.singleton,
        VarParsing.VarParsing.varType.string,
        "File list to ntuplize")
options.parseArguments()

# file list
if ".txt" in options.flist:
    # list of files
    print("reading input file list: "+options.flist)
    options.inputFiles = FileUtils.loadListFromFile(options.flist)
else:
    # we have passed a file name directly
    options.inputFiles = options.flist

process = cms.Process("GenFilterEfficiency")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:gen.root')
)

process.dummy = cms.EDAnalyzer("GenFilterEfficiencyAnalyzer",
                               genFilterInfoTag = cms.InputTag("genFilterEfficiencyProducer")
)

process.p = cms.Path(process.dummy)

