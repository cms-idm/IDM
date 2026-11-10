import numpy as np
import h5py
import matplotlib.pyplot as plt
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import mplhep as hep
import json

def write_h5(data,fname):
    with h5py.File(fname,'w') as f:
        for k in data.keys():
            arr = data[k].value
            if type(arr) == np.ma.core.MaskedArray:
                arr = arr.data
            f.create_dataset(k,data=arr)

def process_signal_inputs(sig_files,variables):
    sig_data = []
    sig_xsec_norm = []
    sig_point = {'m1':[], 'delta':[], 'ctau':[]}

    m1s = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ctaus = [1, 10, 100]
    deltas = [0.1, 0.2]

    for sf in sig_files:
        with h5py.File(sf,"r") as fin:
            entries = len(fin['wgt'])

            match = fin["sel_vtx_isMatched"][()]
            mask = match==1

            entries_genMatched = sum(mask)
            print(f'Signal events {entries} -> {entries_genMatched} after gen matching (raw counts)\n')

            sig_data.append(np.concatenate([fin[v][()][mask].reshape(-1,1) for v in variables],axis=1))
            sig_xsec_norm.append(fin['wgt_norm'][()][mask])

            sig_point['m1'].append(fin['m1'][()][mask])
            sig_point['delta'].append(fin['delta'][()][mask])
            sig_point['ctau'].append(fin['ctau'][()][mask])

            # NOTE: this loop is disabled. It was carried over from BDTutils.py (v1), where it built
            # per-subprocess train/test splits ([:idx_train] / [idx_train:]). Here in v2 there is no
            # train/test split, so mask_sig_point[:] selects the full subprocess slice, which is a subset
            # of the full mask block already appended above -- re-appending it duplicated every signal
            # event (and its m1/delta/ctau/wgt_norm entries) a second time.
            # for m1 in m1s:
            #     for delta in deltas:
            #         for ctau in ctaus:
            #             mask_sig_point = (fin['m1'][()][mask] == m1) & (fin['delta'][()][mask] == delta) & (fin['ctau'][()][mask] == ctau)
            #             sig_data.append(np.concatenate([fin[v][()][mask][mask_sig_point][:].reshape(-1,1) for v in variables],axis=1))
            #
            #             sig_point['m1'].append(fin['m1'][()][mask][mask_sig_point][:])
            #             sig_point['delta'].append(fin['delta'][()][mask][mask_sig_point][:])
            #             sig_point['ctau'].append(fin['ctau'][()][mask][mask_sig_point][:])
            #
            #             sig_xsec_norm.append(fin['wgt_norm'][()][mask][mask_sig_point][:])

    sig_data = np.concatenate(sig_data,axis=0)
    sig_xsec_norm = np.concatenate(sig_xsec_norm,axis=0)
    sig_point['m1'] = np.concatenate(sig_point['m1'], axis=0)
    sig_point['delta'] = np.concatenate(sig_point['delta'], axis=0)
    sig_point['ctau'] = np.concatenate(sig_point['ctau'], axis=0)
    
    return sig_data, sig_xsec_norm, sig_point

def process_bkg_inputs(bkg_files,variables):
    nBkg = 0
    bkg_data = []
    bkg_xsec_norm = []

    for bf in bkg_files:
        with h5py.File(bf,"r") as fin:
            process = bf.split('_')[3]
            entries = len(fin['wgt'])
            print(f'Raw bkg counts: {entries}')
            nBkg += len(fin['wgt'])

            bkg_data.append(np.concatenate([fin[v][()].reshape(-1,1) for v in variables],axis=1))
            bkg_xsec_norm.append(fin['wgt_norm'][()])

    bkg_xsec_norm = np.concatenate(bkg_xsec_norm,axis=0)
    print(f'Number of total background events (raw counts): {nBkg}')
    bkg_data = np.concatenate(bkg_data,axis=0)
    
    return bkg_data, bkg_xsec_norm

def reweight_bkg(bkg_files):
    bkg_nev_raw = 0
    bkg_sumWgts = 0

    for bf in bkg_files:
        with h5py.File(bf,"r") as fin:
            evtWeights = fin['wgt_norm'][()]
            bkg_nev_raw += len(evtWeights)
            bkg_sumWgts += evtWeights.sum()
    print(f'Total background counts (xsec weighted): {bkg_sumWgts:.2f}')
    print(f'Total background counts (raw): {bkg_nev_raw}')

    bkg_sf = []
    bkg_xsec_norm = []
    for bf in bkg_files:
        with h5py.File(bf,"r") as fin:
            bkg_sf.append(fin['wgt_norm'][()] / bkg_sumWgts * bkg_nev_raw)
            bkg_xsec_norm.append(fin['wgt_norm'][()])
    bkg_sf = np.concatenate(bkg_sf,axis=0)
    bkg_xsec_norm = np.concatenate(bkg_xsec_norm)

    print("Sum bkg sf ",bkg_sf.sum())
    
    return bkg_nev_raw, bkg_sumWgts, bkg_sf, bkg_xsec_norm

def reweight_signal(sig_data,bkg_nev_raw,nSamp,bkg_sf,sig_subprocess,bkg_xsec_norm):
    sig_nev_raw = len(sig_data)
    bkg_to_sig = bkg_nev_raw/sig_nev_raw

    print(f'Signal sample size (all subprocesses summed): {sig_nev_raw}')
    print(f'Background sample size (all subprocesses summed): {bkg_nev_raw}')
    print(f'{bkg_to_sig} more background than signal samples')
    
    print(f'There are {nSamp} subprocesses in signal, i.e. per m1/delta/ctau points.')
    print(f'\nIdeally, sig and bkg sample size for BDT input should be the same.')
    print(f'\nFor background, we care about each background process contribution to the total, i.e. QCD having higher xsec than Diboson.')
    print(f'This should be taken into account, so we got the SF for background input that will correct for this relative xsec contribution.')
    print(f'\nFor signal, we want BDT to equally "see" each subprocess. For example, delta=0.2 splitting has lower xsec than delta=0.1.')
    print(f'But we want the BDT to "equally" see them. Therefore, we reweigh signal such that each subprocess relative contribution is the same.')
    print(f'For signal, we also get the overall SF against bkg, because right now we have 10 times less signal input than background.')
    print(f'This means, for each one of {nSamp} signal subprocesses, there should be [n(background sample size)/n(number of signal subprocess)] = {len(bkg_xsec_norm)/nSamp}')
    print(f'We will get the SF for each signal subprocess such that their weighted count corresponds to {bkg_nev_raw/nSamp}')
    
    sig_sf = np.zeros(len(sig_data))
    total_weighted_bkg = bkg_sf.sum()
    nSubprocessWeighted = total_weighted_bkg/nSamp

    for samp, idx in sig_subprocess.items():
        samp_idx = np.nonzero(idx)[0]
        nSubprocess = np.sum(idx) # unweighted

        if nSubprocess == 0:
            continue
        sf = nSubprocessWeighted/nSubprocess

        #print(f'{samp}: {sf}')

        sig_sf[samp_idx] = sf
    
    return sig_sf

def run_training(bdt_name,variables,sig_data,sig_sf_arr,sig_xsec_norm,sig_point,
                 bkg_data,bkg_sf_arr,bkg_xsec_norm,
                 n_estimators=1000,max_depth=8,learning_rate=0.05):
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split

    train = np.concatenate((sig_data,bkg_data),axis=0)
    train_sf = np.concatenate((sig_sf_arr,bkg_sf_arr),axis=0)
    train_sf = abs(train_sf) 
    train_xsec_norm = np.concatenate((sig_xsec_norm, bkg_xsec_norm), axis=0)
    train_m1 = np.concatenate((sig_point['m1'],np.zeros(bkg_data.shape[0])),axis=0)
    train_delta = np.concatenate((sig_point['delta'],np.zeros(bkg_data.shape[0])),axis=0)
    train_ctau = np.concatenate((sig_point['ctau'],np.zeros(bkg_data.shape[0])),axis=0)
    y_train = np.concatenate((np.ones(sig_data.shape[0]),np.zeros(bkg_data.shape[0])),axis=0)
                    
    rng = np.random.default_rng(seed=438290)
    perm = rng.permutation(len(train))
    train = train[perm]
    y_train = y_train[perm]
    dtrain = xgb.DMatrix(train,label=y_train,feature_names=variables)

    train_m1 = train_m1[perm]
    train_delta = train_delta[perm]
    train_ctau = train_ctau[perm]
    train_xsec_norm = train_xsec_norm[perm]
    
    # grid search
    import sklearn.metrics as skmetrics
    #regressor = xgb.XGBRegressor()
    classifier = xgb.XGBClassifier()
    print("Starting grid search")
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingRandomSearchCV, HalvingGridSearchCV
    
    from scipy.stats import uniform, randint
    #param_grid = {
    #    'learning_rate':[0.01,0.05,0.1],
    #    'n_estimators':[200,400,600,800,1000,1200,1400],
    #    'max_depth':[2,4,6,8,10,12]
    #}
    #grid_search = HalvingGridSearchCV(estimator=classifier, param_grid=param_grid, 
    #                           cv=3, scoring='neg_log_loss', verbose=1,refit=True,
    #                          n_jobs=-1)
    
    param_grid = {
        'max_depth': [7],
        'n_estimators': [800],
        'learning_rate': [0.01]
    }
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, 
                                cv=3, scoring='neg_log_loss', verbose=2,refit=True,
                                n_jobs=-1)
    
    grid_search.fit(train, y_train,sample_weight=train_sf)
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    bst = grid_search.best_estimator_
    
    #print("training bdt")
    ## train bdt
    #bst = xgb.XGBRegressor(objective=skmetrics.log_loss,**best_params)
    #bst.fit(train, y_train, sample_weight=train_sf, 
    #    eval_set=[(test, y_test)], eval_metric="rmse", early_stopping_rounds=10,verbose=50)
    
    # make plots
    preds = bst.predict_proba(train)[:,1]
    
    plt.figure(figsize=(8,6))
    bins = 50
    h,bins,_ = plt.hist(preds[y_train==0],bins=bins,density=True, histtype='stepfilled', label='Background', alpha=0.2, color='red')
    h,bins,_ = plt.hist(preds[y_train==1],bins=bins,density=True, histtype='stepfilled', label='Signal', alpha=0.2, color='blue')
    plt.legend()
    #plt.title(bdt_name)
    plt.xlabel('BDT Score')
    plt.ylabel('A.U.')
    plt.yscale('log')
    hep.cms.text("Private Work")
    os.makedirs(f"plots/{bdt_name}/",exist_ok=True)
    plt.savefig(f'plots/{bdt_name}/{bdt_name}_scores_maxDepth{max_depth}_nEst{n_estimators}_lr{learning_rate}.pdf')

    # importance
    plt.figure(figsize=(8,6))
    label_dict = {}
    for idx, var in enumerate(variables):
        label_dict['f{}'.format(idx)] = var
    plt.figure(figsize=(8,6))
    xgb.plot_importance(bst,ax=plt.gca())
    ticks = [ item.get_text() for item in plt.gca().get_yticklabels() ]
    plt.close()
    plt.figure(figsize=(8,6))
    relabel = [ label_dict[tick] for tick in ticks ]
    plt.figure(figsize=(8,6))
    xgb.plot_importance(bst,ax=plt.gca())
    plt.title("")
    plt.gca().set_yticklabels(relabel)
    hep.cms.text("Private Work")
    plt.savefig(f'plots/{bdt_name}/{bdt_name}_importance.pdf', bbox_inches = "tight")
    
    # ROC
    from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve, auc
    plt.figure(figsize=(8,8))
    # test predictions
    fpr, tpr, thresholds = roc_curve(y_train, preds)
    auc = roc_auc_score(y_train, preds)
    plt.plot(fpr, tpr, color = "red", label = F"AUC = {auc:.4f}")
    plt.plot([0,1], [0,1] , color = "black", ls = "--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('FPR' , fontsize=12)
    plt.ylabel('TPR' , fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend( prop={'size':12} , loc = 4)
    hep.cms.text("Private Work")
    plt.savefig(f'plots/{bdt_name}/{bdt_name}_roc_curve.pdf')
    
    # PR
    # train predictions
    plt.figure(figsize=(8,8))
    from sklearn.metrics import auc
    # test predictions
    precision, recall, thresholds2 = precision_recall_curve(y_train, preds)
    plt.plot(recall, precision, color = "red", label=f"PR: {auc(recall, precision):.4f}")
    plt.ylabel('Precision (TP/(TP+FP))', fontsize=12)
    plt.xlabel('Signal Efficiency (TPR)', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend( prop={'size':12} , loc = 3)
    hep.cms.text("Private Work")
    plt.savefig(f'plots/{bdt_name}/{bdt_name}_PR_curve.pdf')
    
    # save model
    bst.save_model(f'models/{bdt_name}.json')
    
    # get WPs
    idx_valid = thresholds < 1
    tpr = tpr[idx_valid]
    thresholds = thresholds[idx_valid]
    tpr_WP = {'tight': 0.7, 'medium': 0.85, 'loose': 0.9}
    thres_WP = {'tight': 0, 'medium': 0, 'loose': 0}
    epsilon = 0.005
    for wp in tpr_WP.keys():    
        idx = np.where((tpr > tpr_WP[wp] - epsilon) & (tpr < tpr_WP[wp] + epsilon))

        thres_WP[wp] = thresholds[idx].mean()

        print(f'{wp} threshold ({100*tpr_WP[wp]}% sig eff): {thres_WP[wp]}')
        
    return bst, tpr_WP, thres_WP
    
def train_bdt(bdt_name,signal_h5,bkg_h5,variables,max_depth=6,n_estimators=500,lr=0.01,split_first=False):
    # load signal
    sig_files = [f"h5/{signal_h5}.h5"]
    sig_data, sig_xsec_norm, sig_point = process_signal_inputs(sig_files,variables)
    
    #print('Signal input statistics (unweighted)')
    sig_subprocess = {}
    for ctau in [1, 10, 100]:
        for delta in [0.1, 0.2]:
            for m1 in [5.,10.,20.,30.,40.,50.,60.,70.,80.,90.,100.]:
                idx = ((sig_point['m1'] == m1) & (sig_point['delta'] == delta))&(sig_point['ctau'] == ctau)
                point = f'm1_{m1}_delta_{delta}_ctau_{ctau}'
                sig_subprocess[point] = idx
                #print(f'{point}: {np.sum(idx)}')
    nSamp = 0
    for samp, count in sig_subprocess.items():
        if np.sum(count) != 0:
            nSamp += 1
        else:
            print(f'{samp} has zero counts')
    print(f'Number of signal subprocesses (training set) with non-zero count: {nSamp}')
    
    # load bkg
    bkg_files = [f"h5/{bkg_h5}.h5"]
    bkg_data, bkg_xsec_norm = process_bkg_inputs(bkg_files,variables)
    
    # reweight bkg
    bkg_nev_raw, bkg_sumWgts, bkg_sf, bkg_xsec_norm = reweight_bkg(bkg_files)
    
    # reweight signal
    sig_sf = reweight_signal(sig_data,bkg_nev_raw,nSamp,bkg_sf,sig_subprocess,bkg_xsec_norm)
    
    # prep weights
    bkg_sf_arr = np.array(bkg_sf)
    sig_sf_arr = sig_sf
    
    # train bdt
    bst, tpr_WP, thres_WP = run_training(bdt_name,variables,sig_data,sig_sf_arr,sig_xsec_norm,sig_point,
                                         bkg_data,bkg_sf_arr,bkg_xsec_norm,
                                         n_estimators=n_estimators,max_depth=max_depth,learning_rate=lr)
    
    # vars dict 
    vars_dict = {"variables":variables}
    with open(f"models/variables_{bdt_name}.json","w") as fout:
        json.dump(vars_dict,fout,indent=4)
        
    return bst, tpr_WP, thres_WP