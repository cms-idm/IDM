p3 makeSignalConfigs.py -m sig -y 2024 -a aEM -p /store/group/lpcmetx/iDMe/Samples/Ntuples/signal_Jun2026noID/ -n Jun2026noID
p3 getSignalXsec.py signal_2024_Jun2026noID_aEM.json
p3 sumGenWgts.py signal_2024_Jun2026noID_aEM.json
