p3 makeSignalConfigs.py -m sig -y 2024 -a aEM -p /store/group/lpcmetx/iDMe/Samples/Ntuples/signal_Apr2026/ -n Apr2026
p3 getSignalXsec.py signal_2024_Apr2026_aEM.json
p3 sumGenWgts.py signal_2024_Apr2026_aEM.json
