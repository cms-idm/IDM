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
    sig_data_train = []
    sig_data_test = []
    
    sig_xsec_norm = []
    sig_xsec_norm_train = []
    sig_xsec_norm_test = []
    
    sig_point = {'m1':[], 'delta':[], 'ctau':[]}
    sig_point_train = {'m1':[], 'delta':[], 'ctau':[]}
    sig_point_test = {'m1':[], 'delta':[], 'ctau':[]}

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

            # for loop for each signal point to get the sig_data_test and sig_data_train
            for m1 in m1s:
                for delta in deltas:
                    for ctau in ctaus:
                        mask_sig_point = (fin['m1'][()][mask] == m1) & (fin['delta'][()][mask] == delta) & (fin['ctau'][()][mask] == ctau)
                        idx_train = int(0.8*len(fin['m1'][()][mask][mask_sig_point]))
                        #print(fin['m1'][()][mask][mask_sig_point][:idx_train])
                        sig_data_train.append(np.concatenate([fin[v][()][mask][mask_sig_point][:idx_train].reshape(-1,1) for v in variables],axis=1))
                        sig_data_test.append(np.concatenate([fin[v][()][mask][mask_sig_point][idx_train:].reshape(-1,1) for v in variables],axis=1))

                        sig_point_train['m1'].append(fin['m1'][()][mask][mask_sig_point][:idx_train])
                        sig_point_train['delta'].append(fin['delta'][()][mask][mask_sig_point][:idx_train])
                        sig_point_train['ctau'].append(fin['ctau'][()][mask][mask_sig_point][:idx_train])

                        sig_point_test['m1'].append(fin['m1'][()][mask][mask_sig_point][idx_train:])
                        sig_point_test['delta'].append(fin['delta'][()][mask][mask_sig_point][idx_train:])
                        sig_point_test['ctau'].append(fin['ctau'][()][mask][mask_sig_point][idx_train:])

                        sig_xsec_norm_train.append(fin['wgt_norm'][()][mask][mask_sig_point][:idx_train])
                        sig_xsec_norm_test.append(fin['wgt_norm'][()][mask][mask_sig_point][idx_train:])


    sig_data_train = np.concatenate(sig_data_train, axis=0)
    sig_data_test = np.concatenate(sig_data_test, axis=0)

    sig_data = np.concatenate(sig_data,axis=0)
    sig_xsec_norm = np.concatenate(sig_xsec_norm,axis=0)

    sig_xsec_norm_train = np.concatenate(sig_xsec_norm_train, axis=0)
    sig_xsec_norm_test = np.concatenate(sig_xsec_norm_test, axis=0)

    sig_point_train['m1'] = np.concatenate(sig_point_train['m1'], axis=0)
    sig_point_train['delta'] = np.concatenate(sig_point_train['delta'], axis=0)
    sig_point_train['ctau'] = np.concatenate(sig_point_train['ctau'], axis=0)

    sig_point_test['m1'] = np.concatenate(sig_point_test['m1'], axis=0)
    sig_point_test['delta'] = np.concatenate(sig_point_test['delta'], axis=0)
    sig_point_test['ctau'] = np.concatenate(sig_point_test['ctau'], axis=0)
    
    return sig_data, sig_data_train, sig_data_test, sig_xsec_norm, sig_xsec_norm_train, sig_xsec_norm_test, sig_point, sig_point_train, sig_point_test

def process_bkg_inputs(bkg_files,variables):
    nBkg = 0
    bkg_data = []
    bkg_xsec_norm = []

    for bf in bkg_files:
        with h5py.File(bf,"r") as fin:
            process = bf.split('_')[3]
            entries = len(fin['wgt'])
            print(f'Raw counts: {entries}')
            nBkg += len(fin['wgt'])

            bkg_data.append(np.concatenate([fin[v][()].reshape(-1,1) for v in variables],axis=1))
            bkg_xsec_norm.append(fin['wgt_norm'][()])

    bkg_xsec_norm = np.concatenate(bkg_xsec_norm,axis=0)
    print(f'\nNumber of total background events (raw counts): {nBkg}')
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

def reweight_signal(sig_data,sig_data_train,bkg_nev_raw,nSamp,bkg_sf,sig_subprocess,bkg_xsec_norm):
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
    
    sig_sf = np.zeros(len(sig_data_train))
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

def run_training(bdt_name,variables,sig_data_train,sig_data_test,sig_sf_arr,sig_xsec_norm_train,sig_xsec_norm_test,
                 sig_point_train,sig_point_test,bkg_data,bkg_sf_arr,bkg_xsec_norm,
                 n_estimators=1000,max_depth=8,learning_rate=0.05):
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed=438290)
    random_state = rng.integers(0,100000,1)[0]

    sig_train = shuffle(sig_data_train, random_state=random_state)
    sig_test = shuffle(sig_data_test, random_state=random_state)
    sig_train_sf = shuffle(sig_sf_arr, random_state=random_state)

    sig_train_xsec_norm = shuffle(sig_xsec_norm_train, random_state=random_state)
    sig_test_xsec_norm = shuffle(sig_xsec_norm_test, random_state=random_state)

    sig_train_m1 = shuffle(sig_point_train['m1'], random_state=random_state)
    sig_test_m1 = shuffle(sig_point_test['m1'], random_state=random_state)

    sig_train_delta = shuffle(sig_point_train['delta'], random_state=random_state)
    sig_test_delta = shuffle(sig_point_test['delta'], random_state=random_state)

    sig_train_ctau = shuffle(sig_point_train['ctau'], random_state=random_state)
    sig_test_ctau = shuffle(sig_point_test['ctau'], random_state=random_state)

    bkg_train, bkg_test = train_test_split(bkg_data,train_size=0.8,random_state=random_state)
    bkg_train_sf, bkg_test_sf = train_test_split(bkg_sf_arr, train_size=0.8, random_state=random_state)
    bkg_train_xsec_norm, bkg_test_xsec_norm = train_test_split(bkg_xsec_norm, train_size=0.8, random_state=random_state)

    bkg_train_m1, bkg_test_m1 = train_test_split(np.zeros(bkg_sf_arr.shape[0]), train_size=0.8, random_state=random_state)
    bkg_train_delta, bkg_test_delta = train_test_split(np.zeros(bkg_sf_arr.shape[0]), train_size=0.8, random_state=random_state)
    bkg_train_ctau, bkg_test_ctau = train_test_split(np.zeros(bkg_sf_arr.shape[0]), train_size=0.8, random_state=random_state)

    train = np.concatenate((sig_train,bkg_train),axis=0)
    train_sf = np.concatenate((sig_train_sf,bkg_train_sf),axis=0)
    train_sf = abs(train_sf)  # training sf cannot have negative values, genWgts have negative values, take abs for now, which is not correct :( 

    train_xsec_norm = np.concatenate((sig_train_xsec_norm, bkg_train_xsec_norm), axis=0)
    train_m1 = np.concatenate((sig_train_m1,bkg_train_m1),axis=0)
    train_delta = np.concatenate((sig_train_delta,bkg_train_delta),axis=0)
    train_ctau = np.concatenate((sig_train_ctau,bkg_train_ctau),axis=0)

    y_train = np.concatenate((np.ones(len(sig_train)),np.zeros(len(bkg_train))),axis=0)

    perm = rng.permutation(len(train))
    train = train[perm]
    y_train = y_train[perm]
    dtrain = xgb.DMatrix(train,label=y_train,feature_names=variables)

    train_m1 = train_m1[perm]
    train_delta = train_delta[perm]
    train_ctau = train_ctau[perm]

    train_xsec_norm = train_xsec_norm[perm]

    test = np.concatenate((sig_test,bkg_test),axis=0)

    test_xsec_norm = np.concatenate((sig_test_xsec_norm, bkg_test_xsec_norm), axis=0)

    test_m1 = np.concatenate((sig_test_m1,bkg_test_m1),axis=0)
    test_delta = np.concatenate((sig_test_delta,bkg_test_delta),axis=0)
    test_ctau = np.concatenate((sig_test_ctau,bkg_test_ctau),axis=0)

    y_test = np.concatenate((np.ones(len(sig_test)),np.zeros(len(bkg_test))),axis=0)

    perm = rng.permutation(len(test))
    test = test[perm]
    y_test = y_test[perm]
    test_xsec_norm = test_xsec_norm[perm]
    test_m1 = test_m1[perm]
    test_delta = test_delta[perm]
    test_ctau = test_ctau[perm]

    dtest = xgb.DMatrix(test,label=y_test,feature_names=variables)
    
    # grid search
    import sklearn.metrics as skmetrics
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'max_depth': [5],
        'n_estimators': [600, 800],
        'learning_rate': [0.01]
    }
    #regressor = xgb.XGBRegressor()
    classifier = xgb.XGBClassifier()
    print("Starting grid search")
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, 
                               cv=3, scoring='neg_log_loss', verbose=2,refit=True,
                              n_jobs=-1,)
    grid_search.fit(train, y_train, sample_weight=train_sf)
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    bst = grid_search.best_estimator_
    
    #print("training bdt")
    ## train bdt
    #bst = xgb.XGBRegressor(objective=skmetrics.log_loss,**best_params)
    #bst.fit(train, y_train, sample_weight=train_sf, 
    #    eval_set=[(test, y_test)], eval_metric="rmse", early_stopping_rounds=10,verbose=50)
    
    # make plots
    pred_test = bst.predict(test)
    pred_train = bst.predict(train)
    
    plt.figure(figsize=(8,6))
    bins = 50
    h,bins,_ = plt.hist(pred_test[y_test==0],bins=bins,density=True, histtype='stepfilled', label='Test set: Background', alpha=0.2, color='red')
    h,bins,_ = plt.hist(pred_test[y_test==1],bins=bins,density=True, histtype='stepfilled', label='Test set: Signal', alpha=0.2, color='blue')
    h,bins,_ = plt.hist(pred_train[y_train==0],bins=bins,density=True, histtype='step', label='Train set: Background',  color='red')
    h,bins,_ = plt.hist(pred_train[y_train==1],bins=bins,density=True, histtype='step', label='Train set: Signal', color='blue')
    plt.legend()
    plt.title(bdt_name)
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
    plt.gca().set_yticklabels(relabel)
    hep.cms.text("Private Work")
    plt.savefig(f'plots/{bdt_name}/{bdt_name}_importance.pdf', bbox_inches = "tight")
    
    # ROC
    from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve, auc
    plt.figure(figsize=(8,8))
    fpr, tpr, thresholds = roc_curve(y_train, pred_train)
    auc = roc_auc_score(y_train, pred_train)
    plt.plot(fpr, tpr, color = "green", label = F"Training ROC-AUC = {auc:.4f}")
    # test predictions
    fpr, tpr, thresholds = roc_curve(y_test, pred_test)
    auc = roc_auc_score(y_test, pred_test)
    plt.plot(fpr, tpr, color = "red", label = F"Test ROC-AUC = {auc:.4f}")
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
    precision, recall, thresholds2 = precision_recall_curve(y_train, pred_train)
    from sklearn.metrics import auc
    plt.plot(recall, precision, color = "green", label=f"Training PR-AUC: {auc(recall, precision):.4f}")
    # test predictions
    precision, recall, thresholds2 = precision_recall_curve(y_test, pred_test)
    plt.plot(recall, precision, color = "red", label=f"Test PR-AUC: {auc(recall, precision):.4f}")
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
    sig_data, sig_data_train, sig_data_test,\
    sig_xsec_norm, sig_xsec_norm_train, sig_xsec_norm_test,\
    sig_point, sig_point_train, sig_point_test = process_signal_inputs(sig_files,variables)
    
    #print('Signal input statistics (unweighted)')
    sig_subprocess = {}
    for ctau in [1, 10, 100]:
        for delta in [0.1, 0.2]:
            for m1 in [5.,10.,20.,30.,40.,50.,60.,70.,80.,90.,100.]:
                idx = ((sig_point_train['m1'] == m1) & (sig_point_train['delta'] == delta))&(sig_point_train['ctau'] == ctau)
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
    sig_sf = reweight_signal(sig_data,sig_data_train,bkg_nev_raw,nSamp,bkg_sf,sig_subprocess,bkg_xsec_norm)
    
    # prep weights
    bkg_sf_arr = np.array(bkg_sf)
    sig_sf_arr = sig_sf
    
    # train bdt
    bst, tpr_WP, thres_WP = run_training(bdt_name,variables,sig_data_train,sig_data_test,sig_sf_arr,sig_xsec_norm_train,sig_xsec_norm_test,
                 sig_point_train,sig_point_test,bkg_data,bkg_sf_arr,bkg_xsec_norm,
                 n_estimators=n_estimators,max_depth=max_depth,learning_rate=lr)
    
    # vars dict 
    vars_dict = {"variables":variables}
    with open(f"models/variables_{bdt_name}.json","w") as fout:
        json.dump(vars_dict,fout,indent=4)
        
    return bst, tpr_WP, thres_WP