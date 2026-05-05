import pandas as pd
import sys
import os
import subprocess

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

def cmsRunScript(flist, script):

    maxEvents = "-1"

    filelist = open(flist, 'r').readlines()
    inputFiles = ""
    for rootfile in filelist:
        if('root' in rootfile):
            inputFiles += ' inputFiles='+rootfile.rstrip('\n') + ' '

    # compute cross section
    command = 'cmsRun {} {} maxEvents=-1 2>&1'.format(script, inputFiles)
    result = subprocess.check_output(command, shell=True, text=True).split('\n')
    return result
    

def calculate_xsec(flist):

    result = cmsRunScript(flist, 'scripts/genXSecAnalyzer_cfg.py')
    for line in result:
        if 'final cross section' in line:
            elements = line.split(' ')
            return float(elements[6])

def calculate_filter_eff(flist):

    result = cmsRunScript(flist, 'scripts/genFilterEfficiency_cfg.py')
    for line in result:
        if 'Filter efficiency' in line:
            elements = line.split(' ')
            return float(elements[3])
    

vers = 'Feb2026'

m1l = [0.05, 0.5, 1, 2, 5, 50]
dml = [0.05, 0.1, 0.2]
ctaul = [1, 10, 100]

# columns
col_m1 = []
col_m2 = []
col_dm = []
col_mzd = []
col_ctau = []
col_mchi = []
col_dmchi = []

col_xs = []
col_eff = []

for m1 in m1l:
    for dm in dml:
        for ctau in ctaul:

            m2 = round(m1*(1+dm), 5)
            mchi = round((m1+m2)/2, 5)
            dmchi = round(m2-m1, 5)
            mzd = round(3 * m1, 5)
            
            col_m1.append(m1); col_m2.append(m2); col_dm.append(dm); col_mzd.append(mzd)
            col_mchi.append(mchi); col_dmchi.append(dmchi); col_ctau.append(ctau)
            
            stfm1 = stringfy_friendly(m1)
            stfdm = stringfy_friendly(dm)
            stfmzd = stringfy_friendly(mzd)

            flist = f'fileLists/signal/2024/M1-{stfm1}_dM-{stfdm}_mZD-{stfmzd}_ctau-{ctau}_flist.txt'

            xsec = calculate_xsec(flist)
            col_xs.append(xsec)
            fileff = calculate_filter_eff(flist)
            col_eff.append(fileff)

            print(m1, dm, ctau, xsec, fileff)

            
#df[(df["Mchi"] == mchi) & (df["dMchi"] == dmchi) & (df["ct"] == 100) & (df["alphaD"] == aD) & (df['mA/m1'] == 3)]
        
# Create a sample DataFrame
data = {"M1": col_m1, "M2": col_m2, "dM": col_dm, "mZD": col_mzd,
        "Mchi": col_mchi, "dMchi": col_dmchi, "ct": col_ctau,
        "alphaD": ["aEM" for m in col_m1], "mA/m1": [3 for m in col_m1],
        "xsec(pb)": col_xs, "filter_eff": col_eff}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('signal_xsec_filtereff_table.csv', index=False)
