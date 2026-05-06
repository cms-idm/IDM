import uproot
import numpy as np
import json
import sys
from XRootD import client
from tqdm import tqdm
from multiprocessing import Process, Value, Array, Manager
import multiprocessing as mp
import traceback

def sum_weights(fileList,sums,blacklist,nevts,isData):
    sum_wgt = 0
    sum_nevts = 0
    for f in tqdm(fileList):
        try:
            with uproot.open(f)['ntuples/outT'] as tree:
                if tree.num_entries == 0:
                    continue
                else:
                    if isData:
                        sum_wgt += 0
                    else:
                        sum_wgt += np.sum(tree['genWgt'].array())
                    sum_nevts += tree.num_entries
        except Exception:
            print(f'Blacklisting {f.split("/")[-1]}; error:')
            print(traceback.format_exc())
            blacklist.append(f.split("/")[-1])
    sums.append(sum_wgt)
    nevts.append(sum_nevts)

inputJson = sys.argv[1]
with open(inputJson) as f:
    samples = json.load(f)
num_cpus = mp.cpu_count()

if len(sys.argv) == 3:
    isData = bool(int(sys.argv[2]))
else:
    isData = False

for samp in samples:
    if 'designation' in samp.keys():
        print(f"Running on {samp['designation']}")
    else:
        print(f"Running on {samp['name']}")
    loc = samp["location"]
    nFiles = -1
    has_blacklist = True if "blacklist" in samp.keys() else False
    blacklist = [] # track files that can't be opened
    sum_wgt = []
    sum_evt = []
    if '.root' in loc:
        nFiles = 1
        try:
            tree = uproot.open(loc)['ntuples/outT']
            if tree.num_entries == 0:
                sum_wgt = 0
            else:
                if not isData:
                    sum_wgt = np.sum(tree['genWgt'].array())
                else:
                    sum_wgt = 0
                sum_evt = tree.num_entries
        except:
            blacklist.append(loc)
            sum_wgt = 0
            sum_evt = 0
            
    else:
        xrdClient = client.FileSystem("root://cmseos.fnal.gov")
        if type(loc) != list:
            status, flist = xrdClient.dirlist(loc)
            fullList = ["root://cmsxrootd.fnal.gov/"+loc+"/"+item.name for item in flist if '.root' in item.name]
        else:
            fullList = []
            for l in loc:
                status, flist = xrdClient.dirlist(l)
                fullList.extend(["root://cmsxrootd.fnal.gov/"+l+"/"+item.name for item in flist if '.root' in item.name])
        nFiles = len(fullList)
        if has_blacklist:
            fullList = [f for f in fullList if f not in samp["blacklist"]]
        if nFiles < 2*num_cpus:
        #if nFiles < 0: # always do parallel mode
            print("Serial Mode")
            sum_weights(fullList,sum_wgt,blacklist,sum_evt,isData)
            sum_wgt = np.sum(sum_wgt)
            sum_evt = np.sum(sum_evt)
        else:
            print("Parallel Mode")
            subLists = [list(l) for l in np.array_split(fullList,num_cpus)]
            print ("num_cpus:", num_cpus)
            
            with Manager() as manager:
                m_blacklist = manager.list()
                m_sums = manager.list()
                m_nevts = manager.list()
                processes = []
                for subList in subLists:
                    processes.append(Process(target=sum_weights,args=(subList,m_sums,m_blacklist,m_nevts,isData)))
                for p in processes:
                    p.start()
                for p in processes:
                    p.join()
                blacklist = [b for b in m_blacklist]
                sum_wgt = np.sum([s for s in m_sums])
                sum_evt = np.sum([s for s in m_nevts])

    print('Blacklisted {0} files in {1}'.format(len(blacklist),samp['name']))
    samp['sum_wgt'] = float(sum_wgt)
    samp['num_events'] = int(sum_evt)

    print('sum_wgt:', samp['sum_wgt'])
    print('num_events:',samp['num_events'])

    if 'blacklist' in samp.keys():
        samp['blacklist'].extend(blacklist)
        samp['blacklist'] = list(set(samp['blacklist']))
    else:
        samp['blacklist'] = blacklist
    samp['nFiles'] = nFiles - len(blacklist)

    #checkpointing
    with open(inputJson,'w') as f:
        json.dump(samples,f,indent=4)
