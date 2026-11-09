from CRABAPI.RawCommand import crabCommand
from CRABClient.ClientExceptions import ClientException
import CRABClient.UserUtilities
from XRootD import client
import XRootD.client.flags as flags
from optparse import OptionParser
from multiprocessing import Process
import os
import json
from datetime import datetime
import sys

xrdClient = client.FileSystem("root://cmseos.fnal.gov")

def parseArguments():
    parser = OptionParser()
    
    parser.add_option('-c', '--crabCmd',
                      dest = 'crabCmd',
                      default = '',
                      help = "crab command",
                      metavar = 'CMD')

    parser.add_option('-y', '--year',
                      dest = 'year',
                      default = '',
                      help = "Which year to process ('2023'/'2022'/'2018'(default)/'2017'/'2016'/'2016APV')",
                      metavar = 'YEAR')

    parser.add_option("-f","--inFile",
                      dest = 'inFile',
                      default = '',
                      help = 'JSON file with input data or MC lists',
                      metavar = 'INFILE')

    parser.add_option("-t","--type",
                      dest="samp_type",
                      default='',
                      help='Sample type, Data (1) or MC (0)',
                      metavar="TYPE")

    parser.add_option('-s','--signal',
                     dest="signal",
                     default='',
                     help='Is running signal (1) or not (0)',
                     metavar='SIGNAL')

    parser.add_option('-n','--name',
                      dest='name',
                      default='',
                      help='name for this particular run (will be saved in dir background_NAME)',
                      metavar='NAME')
    
    parser.add_option('-p','--particular',
                      dest='particular',
                      default='',
                      help='particular subsample to process',
                      metavar='PARTICULAR')
    
    (options, arguments) = parser.parse_args()
    
    if arguments:
        parser.error("Found positional argument(s): %s." % (arguments))
    if not options.crabCmd:
        parser.error("(-c CMD, --crabCmd=CMD) option not provided.")
    if not options.year:
        parser.error("-y year option not provided")
    if not options.inFile:
        parser.error("-f input file option not provided")
    if not options.samp_type:
        parser.error("-t type data (1) or mc (0) option not specified")
    if not options.signal:
        parser.error("-s : is running on signal (1) or not (0) option not specified")
    if not options.name:
        parser.error("-n : name for run not specified")

    return options

def main():
    options = parseArguments()
    
    if 'CMSSW_BASE' not in os.environ.keys():
        print ("Run cmsenv first!")

        return
    base_dir = os.environ['CMSSW_BASE']

    year = options.year
    samp_type = int(options.samp_type) # 1 = data, 0 = MC
    isSignal = int(options.signal) # 1 = signal, 0 = bkg
    if samp_type == 1 and isSignal == 1:
        print("Makes no sense - configured to be data and signal at the same time")
        sys.exit()
    run_name = options.name
    if options.particular == '':
        particular = None
    else:
        particular = options.particular

    config = CRABClient.UserUtilities.config()

    # Basic settings common to all runs 
    config.General.workArea = base_dir+'/src/iDMe/AODSkimmer/crab/submissions_ElectronNtuplizer_/' + options.name

    config.General.transferOutputs = True
    config.General.transferLogs = False
    config.JobType.pluginName = 'Analysis'
    config.JobType.psetName = base_dir+'/src/iDMe/AODSkimmer/scripts/ElectronNtuplizer_cfg.py'
    config.JobType.allowUndistributedCMSSW = True
    config.JobType.numCores = 1
    config.Data.splitting = 'Automatic'

    config.JobType.maxMemoryMB = 2500
    #config.Data.totalUnits = 1
    #config.Data.splitting = 'EventAwareLumiBased'
    #config.Data.unitsPerJob = 10000
    config.Data.publication = False
    #config.Site.whitelist = ['T2_US_*', 'T2_DE_*', 'T2_EE_*', 'T2_ES_*', 'T2_GR_*', 'T2_HU_*', 'T2_IT_*', 'T2_RU_*', 'T2_UK_*']
    config.Site.storageSite = 'T3_US_FNALLPC'
    if samp_type == 1:
        print("Running on data! Adding the golden json as a lumimask")
        if options.year == "2018":
            config.Data.lumiMask = "../data_info/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"
        if options.year == "2017":
            config.Data.lumiMask = "../data_info/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"
        if options.year == "2016" or options.year == "2016APV":
            config.Data.lumiMask = "../data_info/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"

    inFile = options.inFile
    samples = {}
    with open(inFile) as f:
        samples = json.load(f)
    for samp in samples.keys():
        for subsample, dataset_info in samples[samp].items():
            if particular is not None and subsample != particular:
                continue
            
            items_to_process = []
            
            if isinstance(dataset_info, dict):
                if not dataset_info:
                    continue
                for era, dataset in dataset_info.items():
                    items_to_process.append((f"{subsample}_{era}", dataset))
            elif isinstance(dataset_info, str):
                items_to_process.append((subsample, dataset_info))
                
            for current_subsample, dataset in items_to_process:
                if samp_type == 0:
                    if isSignal == 0:
                        output_base = '/store/group/lpcmetx/iDMe/Samples/Ntuples/background_{0}/{1}/{2}/{3}/'.format(run_name,year,samp,current_subsample)
                    else:
                        output_base = '/store/group/lpcmetx/iDMe/Samples/Ntuples/signal_{0}/{1}/{2}/{3}/'.format(run_name,year,samp,current_subsample)
                else:
                    output_base = '/store/group/lpcmetx/iDMe/Samples/Ntuples/data_{0}/{1}/{2}/{3}/'.format(run_name,year,samp,current_subsample)
                xrdClient.mkdir(output_base,flags.MkDirFlags.MAKEPATH)
                config.Data.outLFNDirBase = output_base
                config.Data.inputDataset = dataset
                config.General.requestName = 'iDMe_' + current_subsample + datetime.now().strftime("_%Y_%m_%d-%H_%M")
                config.JobType.outputFiles = ['{0}.root'.format(current_subsample)]
                config.JobType.pyCfgParams = ['numThreads=1',
                                              'outfile={0}.root'.format(current_subsample),
                                              'data={0}'.format(samp_type),
                                              'signal={0}'.format(isSignal)]
                print ('Submitting for input dataset {0}'.format(current_subsample))

                #crabCommand(options.crabCmd, config = config)
                kwargs = {'config':config}
                p = Process(target=crabCommand,args=(options.crabCmd,),kwargs=kwargs)
                p.start()
                p.join()

if __name__ == '__main__':
    main()
