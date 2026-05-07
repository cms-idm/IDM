import os

fls = os.popen(f'xrdfs root://cmseos.fnal.gov/ ls /store/group/lpcmetx/iDMe/Samples/acrobert/signal/2024/MiniAOD/').read()
def condition(filename):
    return True

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


#m1l = [0.05, 0.5, 5, 50]
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

            #procname = f'iDMe_run3_M1-{stfm1}_dM-{stfdm}_mZD-{stfmed}_ctau-{ctau}_1jet_xptj80_{int(nevents/1000)}k-evts_{year}'
            procstring = f'M1-{stfm1}_dM-{stfdm}_mZD-{stfmed}_ctau-{ctau}_'
            # iDMe_run3_M1-5_dM-0p1_mZD-15_ctau-1_1jet_xptj80_MiniAOD_2024_17_6346evts.root

            files = [f for f in fls.split('\n') if procstring in f]
            if len(files) >= 15:
                with open(f'signal/2024/{procstring}flist.txt',"w") as fout:
                    for f in files:
                        fout.write(f+"\n")

