import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
import FWCore.Utilities.FileUtils as FileUtils

options = VarParsing ('analysis')
options.register('flist',
        "",
        VarParsing.multiplicity.singleton,
        VarParsing.varType.string,
        "File list to run over")
options.parseArguments()

# file list
if options.flist != "":
    if ".txt" in options.flist:
        # list of files
        print("reading input file list: "+options.flist)
        options.inputFiles = FileUtils.loadListFromFile(options.flist)
    else:
        # we have passed a file name directly
        options.inputFiles = cms.untracked.vstring(options.flist)

process = cms.Process('XSec')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100000

secFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles),
    secondaryFileNames = secFiles)
process.xsec = cms.EDAnalyzer("GenXSecAnalyzer")

process.ana = cms.Path(process.xsec)
process.schedule = cms.Schedule(process.ana)
