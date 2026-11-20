vers=Nov2026
p3 makeSignalConfigs.py -m bkg -y 2024 -a aEM -p /store/group/lpcmetx/iDMe/Samples/Ntuples/background_${vers}noIDdR/ -n ${vers}-full
p3 getSignalXsec.py bkg_2024_${vers}-full.json
p3 sumGenWgts.py bkg_2024_${vers}-full.json

python3 haddMergeSamples.py -c bkg_2024_${vers}-full.json -i /store/group/lpcmetx/iDMe/Samples/Ntuples/background_${vers}noIDdR/ -o /store/group/lpcmetx/iDMe/Samples/Ntuples/background_${vers}noIDdR_merged

p3 makeSignalConfigs.py -m bkg -y 2024 -a aEM -p /store/group/lpcmetx/iDMe/Samples/Ntuples/background_${vers}noIDdR_merged/ -n ${vers}-fm
p3 getSignalXsec.py bkg_2024_${vers}-fm.json
p3 sumGenWgts.py bkg_2024_${vers}-fm.json

p3 makeSignalConfigs.py -m bkg -y 2024 -a aEM -p /store/group/lpcmetx/iDMe/Samples/Ntuples/background_${vers}noIDdR_slim/ -n ${vers}-slim
p3 getSignalXsec.py bkg_2024_${vers}-slim.json
p3 sumGenWgts.py bkg_2024_${vers}-slim.json

