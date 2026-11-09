import os, sys

def stringfy_friendly(num):
    if isinstance(num, int):
        return str(num)
    elif isinstance(num, float):
        if int(num*1000) > 0:
            num = round(num, 3)
            return str(num).replace('.', 'p') if '.' in str(num) else str(num)
        else:
            num = '%.3e' % num
            return num.replace('.', 'p')
    else:
        raise ValueError("{0} is not a number!".format(num))

vers = 'Jul2026noID'
CMSSW = 'CMSSW_14_0_21'
compiled_CMSSW_envs='ntuplizer_CMSSW_14_0_21_acrobert.tar.gz'

m1l = [0.05, 0.5, 1, 2, 5, 50]
dml = [0.05, 0.1, 0.2] 
ctaul = [1, 10, 100]

for m1 in m1l:
    for dm in dml:
        for ctau in ctaul:

            m2 = round(m1*(1+dm), 5)
            mchi = round((m1+m2)/2, 5)
            dmchi = round(m2-m1, 5)
            med = round(3 * m1, 5)
            stfm1 = stringfy_friendly(m1)
            stfdm = stringfy_friendly(dm)
            stfmed = stringfy_friendly(med)

            procstring = f'M1-{stfm1}_dM-{stfdm}_mZD-{stfmed}_ctau-{ctau}'
            cmd = f'source submit_ElectronNtuplizer_condor.sh ../../fileLists/signal/2024/{procstring}_flist.txt 2024 4 False True {vers} 5 {CMSSW} {compiled_CMSSW_envs}'
            #cmd = f'cmsRun scripts/ElectronNtuplizer_cfg.py year=2024 data=0 signal=1 flist="{flist}" outfile={outfile}'
            os.system(f'bash -c "{cmd}"')
