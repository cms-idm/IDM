p3 makeSignalConfigs.py -m sig -y 2024 -a 0p1 -p /store/group/lpcmetx/iDMe/Samples/Ntuples/signal_Jul2026noID/ -n Jul2026noID
p3 getSignalXsec.py signal_2024_Jul2026noID_0p1.json
p3 sumGenWgts.py signal_2024_Jul2026noID_0p1.json
