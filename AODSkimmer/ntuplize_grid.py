import os

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

vers = 'Apr2026'
    
m1l = [0.05, 0.5, 1, 2, 5, 50]
dml = [0.05, 0.1, 0.2] 
ctaul = [1, 10, 100]

for m1 in m1l:
    for dm in dml:
        for ctau in ctaul:

            m2 = m1*(1+dm)
            mchi = (m1+m2)/2
            dmchi = m2-m1
            med = 3 * m1
            stfm1 = stringfy_friendly(m1)
            stfdm = stringfy_friendly(dm)
            stfmed = stringfy_friendly(med)

            procstring = f'M1-{stfm1}_dM-{stfdm}_mZD-{stfmed}_ctau-{ctau}'
            flist = f'fileLists/signal/2024/{procstring}_flist.txt'
            outfile = f'ntuples/signal/2024/iDMe_run3_ntuples_{vers}_{procstring}.root'
            cmd = f'cmsRun scripts/ElectronNtuplizer_cfg.py year=2024 data=0 signal=1 flist="{flist}" outfile={outfile}'
            print(cmd)
            os.system(cmd)
            exit()
