p3 makeSignalConfigs.py -m sig -y 2024 -a aEM -p /store/group/lpcmetx/iDMe/Samples/Ntuples/signal_May2026/ -n May2026
p3 getSignalXsec.py signal_2024_May2026_aEM.json
p3 sumGenWgts.py signal_2024_May2026_aEM.json
