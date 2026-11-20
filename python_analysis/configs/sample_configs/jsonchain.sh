vers=Nov2026
p3 makeSignalConfigs.py -m sig -y 2024 -a 0p1 -p /store/group/lpcmetx/iDMe/Samples/Ntuples/signal_${vers}noIDdR/ -n ${vers}
p3 getSignalXsec.py signal_2024_${vers}_0p1.json
p3 sumGenWgts.py signal_2024_${vers}_0p1.json

