p3 makeSignalConfigs.py -m sig -y 2024 -a 0p1 -p /store/group/lpcmetx/iDMe/Samples/Ntuples/signal_Oct2026noIDdR/ -n Oct2026
p3 getSignalXsec.py signal_2024_Oct2026_0p1.json
p3 sumGenWgts.py signal_2024_Oct2026_0p1.json
