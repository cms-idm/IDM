p3 makeSignalConfigs.py -m bkg -y 2024 -a aEM -p /store/group/lpcmetx/iDMe/Samples/Ntuples/background_Aug2026noIDSep_merged/ -n Aug2026
p3 getSignalXsec.py bkg_2024_Aug2026.json
p3 sumGenWgts.py bkg_2024_Aug2026.json

p3 makeSignalConfigs.py -m bkg -y 2024 -a aEM -p /store/group/lpcmetx/iDMe/Samples/Ntuples/background_Aug2026noIDSep_slim/ -n Aug2026-slim
p3 getSignalXsec.py bkg_2024_Aug2026-slim.json
p3 sumGenWgts.py bkg_2024_Aug2026-slim.json

