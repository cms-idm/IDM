p3 makeSignalConfigs_officialSignalMC.py -m sig -y 2018 -a aEM -p /store/group/lpcmetx/iDMe/Samples/Ntuples/signal_2018_official_Preapproval/ -n official_Preapproval
p3 getSignalXsec.py signal_2018_official_Preapproval_aEM.json
p3 sumGenWgts.py signal_2018_official_Preapproval_aEM.json
