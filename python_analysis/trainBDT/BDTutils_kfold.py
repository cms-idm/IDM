import numpy as np
import h5py
import matplotlib.pyplot as plt
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import mplhep as hep
import json

# k-fold cross training: every event is assigned to fold = eventNum % K_FOLDS. The model for fold i is
# trained on all folds != i and is the only model used to score events in fold i, so every MC event gets
# an unbiased (out-of-fold) score and the full MC sample can be used for the analysis.
K_FOLDS = 5

def get_fold(event_num, k=K_FOLDS):
    return (np.asarray(event_num).astype(np.uint64) % np.uint64(k)).astype(int)

def read_event_num(fin, fname):
    if 'eventNum' not in fin:
        raise KeyError(f'{fname} has no eventNum dataset, which is needed for the k-fold assignment. '
                       f'Regenerate it with step1 (skimmerTools now writes eventNum/lumiSec/runNum).')
    return fin['eventNum'][()]

def write_h5(data,fname):
    with h5py.File(fname,'w') as f:
        for k in data.keys():
            arr = data[k].value
            if type(arr) == np.ma.core.MaskedArray:
                arr = arr.data
            f.create_dataset(k,data=arr)

def check_features(h5_files,variables):
    # raise once, listing every missing column per file, instead of a KeyError partway through loading
    missing = {}
    for fname in h5_files:
        with h5py.File(fname,'r') as f:
            miss = [v for v in variables if v not in f]
        if miss:
            missing[fname] = miss
    if missing:
        msg = '\n'.join(f'  {fname}: {miss}' for fname, miss in missing.items())
        raise KeyError(f'Missing features in h5 (add them to skimmerTools.makeBDTInputs and rerun step1_makeh5.py):\n{msg}')

def process_signal_inputs(sig_files,variables):
    # sig_point carries the per-event fold assignment ('fold') alongside m1/delta/ctau, so anything that
    # filters sig_point (e.g. exclude_signal_points) keeps the folds aligned with the data
    sig_data = []
    sig_xsec_norm = []
    sig_point = {'m1':[], 'delta':[], 'ctau':[], 'eventNum':[], 'fold':[]}

    for sf in sig_files:
        with h5py.File(sf,"r") as fin:
            # keep all signal events, whether or not the selected vertex is gen-matched, so training and
            # evaluation see the same signal population as the analysis
            entries = len(fin['wgt'])
            entries_genMatched = np.sum(fin["sel_vtx_isMatched"][()]==1)
            print(f'Signal events: {entries} ({entries_genMatched} with gen-matched selected vertex, raw counts)\n')

            sig_data.append(np.concatenate([fin[v][()].reshape(-1,1) for v in variables],axis=1))
            sig_xsec_norm.append(fin['wgt_norm'][()])

            sig_point['m1'].append(fin['m1'][()])
            sig_point['delta'].append(fin['delta'][()])
            sig_point['ctau'].append(fin['ctau'][()])

            event_num = read_event_num(fin,sf)
            sig_point['eventNum'].append(event_num)
            sig_point['fold'].append(get_fold(event_num))

    sig_data = np.concatenate(sig_data,axis=0)
    sig_xsec_norm = np.concatenate(sig_xsec_norm,axis=0)
    sig_point = {k: np.concatenate(v,axis=0) for k, v in sig_point.items()}

    return sig_data, sig_xsec_norm, sig_point

def process_bkg_inputs(bkg_files,variables):
    # returns the per-event fold assignment as a third output
    nBkg = 0
    bkg_data = []
    bkg_xsec_norm = []
    bkg_fold = []

    for bf in bkg_files:
        with h5py.File(bf,"r") as fin:
            process = bf.split('_')[3]
            entries = len(fin['wgt'])
            print(f'Raw bkg counts: {entries}')
            nBkg += len(fin['wgt'])

            bkg_data.append(np.concatenate([fin[v][()].reshape(-1,1) for v in variables],axis=1))
            bkg_xsec_norm.append(fin['wgt_norm'][()])
            bkg_fold.append(get_fold(read_event_num(fin,bf)))

    bkg_xsec_norm = np.concatenate(bkg_xsec_norm,axis=0)
    print(f'Number of total background events (raw counts): {nBkg}')
    bkg_data = np.concatenate(bkg_data,axis=0)
    bkg_fold = np.concatenate(bkg_fold,axis=0)

    return bkg_data, bkg_xsec_norm, bkg_fold

def reweight_bkg_arrays(bkg_xsec_norm):
    # same normalization as reweight_bkg, but on an in-memory array so it can be applied to the training folds only.
    # negative-weight events get zero weight, and the rest are normalized so the bkg sf sum to the raw count
    bkg_nev_raw = len(bkg_xsec_norm)
    bkg_wgt_pos = np.clip(bkg_xsec_norm, 0, None)
    bkg_sumWgts = bkg_wgt_pos.sum()
    print(f'Total background counts (xsec weighted, negative weights zeroed): {bkg_sumWgts:.2f}')
    print(f'Total background counts (raw): {bkg_nev_raw}, of which {np.sum(bkg_xsec_norm < 0)} with negative weight set to 0')

    bkg_sf = bkg_wgt_pos / bkg_sumWgts * bkg_nev_raw
    print("Sum bkg sf ",bkg_sf.sum())

    return bkg_nev_raw, bkg_sumWgts, bkg_sf, bkg_xsec_norm

def reweight_bkg(bkg_files):
    # full-sample reweighting; train_bdt reweights the training folds with reweight_bkg_arrays instead
    bkg_xsec_norm = []
    for bf in bkg_files:
        with h5py.File(bf,"r") as fin:
            bkg_xsec_norm.append(fin['wgt_norm'][()])
    bkg_xsec_norm = np.concatenate(bkg_xsec_norm)

    return reweight_bkg_arrays(bkg_xsec_norm)

def build_sig_subprocess(sig_point):
    # one subprocess per unique (m1, delta, ctau) point present in the signal sample
    sig_subprocess = {}
    points = np.unique(np.stack([sig_point['m1'],sig_point['delta'],sig_point['ctau']],axis=1),axis=0)
    for m1, delta, ctau in points:
        idx = (sig_point['m1'] == m1) & (sig_point['delta'] == delta) & (sig_point['ctau'] == ctau)
        point = f'm1_{m1:g}_delta_{delta:g}_ctau_{ctau:g}'
        sig_subprocess[point] = idx
    nSamp = 0
    for samp, count in sig_subprocess.items():
        if np.sum(count) != 0:
            nSamp += 1
        else:
            print(f'{samp} has zero counts')
    print(f'Number of signal subprocesses with non-zero count: {nSamp}')
    return sig_subprocess, nSamp

def print_fold_balance(sig_point,bkg_fold,k=K_FOLDS):
    sig_subprocess, _ = build_sig_subprocess(sig_point)
    print(f'\nEvents per fold (fold = eventNum % {k}, raw counts)')
    print(f'{"":>30} ' + ' '.join(f'{f"fold{i}":>7}' for i in range(k)))
    print(f'{"background":>30} ' + ' '.join(f'{np.sum(bkg_fold == i):>7d}' for i in range(k)))
    print(f'{"signal (all points)":>30} ' + ' '.join(f'{np.sum(sig_point["fold"] == i):>7d}' for i in range(k)))
    for point, idx in sig_subprocess.items():
        print(f'{point:>30} ' + ' '.join(f'{np.sum(idx & (sig_point["fold"] == i)):>7d}' for i in range(k)))

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

## 'learning_rate': 0.01, 'max_depth': 6, 'n_estimators': 600
### 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 700
#### 'learning_rate': 0.012, 'max_depth': 3, 'n_estimators': 700
##### 'learning_rate': 0.015, 'max_depth': 2, 'n_estimators': 800 BDT out-of-fold AUC: 0.9920
###### 'learning_rate': 0.018, 'max_depth': 2, 'n_estimators': 800 BDT out-of-fold AUC: 0.9919
####### 'learning_rate': 0.018, 'max_depth': 2, 'n_estimators': 800 BDT out-of-fold AUC: 0.9919
PARAM_GRID = {
    'max_depth': [2, 4, 6, 8],
    'n_estimators': [600, 800, 1000, 1200],
    'learning_rate': [0.01, 0.018, 0.02, 0.03]
}

def build_training_set(sig_data,sig_sf_arr,bkg_data,bkg_sf_arr):
    # stacks signal (label 1) and bkg (label 0) into one shuffled training set with abs weights (a no-op now that
    # reweight_bkg_arrays zeroes negative bkg weights; kept as a safeguard)
    train = np.concatenate((sig_data,bkg_data),axis=0)
    train_sf = np.concatenate((sig_sf_arr,bkg_sf_arr),axis=0)
    train_sf = abs(train_sf)
    y_train = np.concatenate((np.ones(sig_data.shape[0]),np.zeros(bkg_data.shape[0])),axis=0)

    rng = np.random.default_rng(seed=438290)
    perm = rng.permutation(len(train))
    return train[perm], y_train[perm], train_sf[perm]

def grid_search_fold(train,y_train,train_sf,param_grid=PARAM_GRID):
    # grid search on one fold's training events (its CV only splits the training folds, i.e. it is nested inside
    # the k-fold). No refit: train_bdt pools the CV scores of all folds in pick_common_params and trains every
    # fold model with the same parameters. Returns GridSearchCV.cv_results_
    from sklearn.model_selection import GridSearchCV
    print("Starting grid search")
    grid_search = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=param_grid,
                                cv=3, scoring='neg_log_loss', verbose=2,refit=False,
                                n_jobs=1)
    grid_search.fit(train, y_train,sample_weight=train_sf)
    return grid_search.cv_results_

def pick_common_params(cv_results,n_show=5):
    # cv_results: one GridSearchCV.cv_results_ per fold, all on the same grid. Averages each parameter set's CV
    # score over the folds and returns the best, so all fold models share the same hyperparameters
    params = cv_results[0]['params']
    if any(r['params'] != params for r in cv_results):
        raise ValueError('grid searches of the folds were run on different parameter grids')
    scores = np.array([r['mean_test_score'] for r in cv_results]) # (k, n_params)
    mean_score = scores.mean(axis=0)
    best = int(np.argmax(mean_score))

    print('\nPer-fold best parameters (not used):')
    for i, s in enumerate(scores):
        print(f'fold{i}: {params[int(np.argmax(s))]}  score = {s.max():.5f}')
    print(f'\nTop {n_show} parameter sets by fold-averaged CV score (neg_log_loss):')
    print(f'{"mean":>9} ' + ' '.join(f'{f"fold{i}":>9}' for i in range(len(scores))) + '  params')
    for j in np.argsort(mean_score)[::-1][:n_show]:
        print(f'{mean_score[j]:>9.5f} ' + ' '.join(f'{s:>9.5f}' for s in scores[:,j]) + f'  {params[j]}')
    print(f'\nCommon parameters for all folds: {params[best]}')
    return params[best]

def run_training(bdt_name,variables,sig_data,sig_sf_arr,sig_xsec_norm,sig_point,
                 bkg_data,bkg_sf_arr,bkg_xsec_norm,params):
    # trains the model for a single fold on the given (training-fold) inputs with fixed hyperparameters params
    # (from pick_common_params), saves it to models/{bdt_name}.json and returns it. Performance plots and WPs are
    # made from out-of-fold scores in train_bdt.
    train, y_train, train_sf = build_training_set(sig_data,sig_sf_arr,bkg_data,bkg_sf_arr)

    print(f"Training with {params}")
    bst = xgb.XGBClassifier(**params)
    bst.fit(train, y_train, sample_weight=train_sf)

    # importance
    os.makedirs(f"plots/{bdt_name}/",exist_ok=True)
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
    plt.close('all')

    # save model
    os.makedirs("models",exist_ok=True)
    bst.save_model(f'models/{bdt_name}.json')

    return bst

def predict_kfold(models,data,folds):
    # score each event with the model that did not train on it (models[i] for events with fold == i)
    scores = np.full(len(data),-1.0)
    for i, bst in enumerate(models):
        in_fold = folds == i
        if np.sum(in_fold) > 0:
            scores[in_fold] = bst.predict_proba(data[in_fold])[:,1]
    return scores

def compute_WPs(y,scores,weights=None,tpr_WP=None,epsilon=0.005):
    # weights: per-event weights for the ROC, so the WP signal efficiencies are weighted efficiencies
    if tpr_WP is None:
        tpr_WP = {'tight': 0.7, 'medium': 0.8, 'loose': 0.9}
    fpr, tpr, thresholds = roc_curve(y, scores, sample_weight=weights)
    idx_valid = thresholds < 1
    tpr = tpr[idx_valid]
    thresholds = thresholds[idx_valid]
    thres_WP = {}
    for wp in tpr_WP.keys():
        idx = np.where((tpr > tpr_WP[wp] - epsilon) & (tpr < tpr_WP[wp] + epsilon))
        thres_WP[wp] = thresholds[idx].mean()
        print(f'{wp} threshold ({100*tpr_WP[wp]}% sig eff): {thres_WP[wp]}')
    return tpr_WP, thres_WP

def make_eval_plots(bdt_name,y,scores,train_y=None,train_scores=None,folds=None,weights=None,train_weights=None):
    # y/scores: out-of-fold labels and scores. train_y/train_scores (optional): each fold model's scores on its
    # own training events, overlaid as an overtraining check. folds (optional): per-event fold assignment aligned
    # with y/scores, used to overlay the ROC/PR curve of each individual fold. weights/train_weights (optional):
    # per-event weights aligned with scores/train_scores, applied to the score histograms and ROC/PR curves
    from sklearn.metrics import precision_recall_curve, auc
    os.makedirs(f"plots/{bdt_name}/",exist_ok=True)
    fold_ids = np.unique(folds) if folds is not None else []
    fold_colors = plt.cm.viridis(np.linspace(0,0.9,max(len(fold_ids),1)))
    if weights is None:
        weights = np.ones(len(scores))
    if train_scores is not None and train_weights is None:
        train_weights = np.ones(len(train_scores))

    # score distributions
    plt.figure(figsize=(8,6))
    bins = np.linspace(0,1,51)
    plt.hist(scores[y==0],bins=bins,weights=weights[y==0],density=True, histtype='stepfilled', label='Out-of-fold: Background', alpha=0.2, color='red')
    plt.hist(scores[y==1],bins=bins,weights=weights[y==1],density=True, histtype='stepfilled', label='Out-of-fold: Signal', alpha=0.2, color='blue')
    if train_scores is not None:
        plt.hist(train_scores[train_y==0],bins=bins,weights=train_weights[train_y==0],density=True, histtype='step', label='Training: Background', color='red')
        plt.hist(train_scores[train_y==1],bins=bins,weights=train_weights[train_y==1],density=True, histtype='step', label='Training: Signal', color='blue')
    plt.legend()
    plt.xlabel('BDT Score')
    plt.ylabel('A.U.')
    plt.yscale('log')
    hep.cms.text("Private Work")
    plt.savefig(f'plots/{bdt_name}/{bdt_name}_scores.pdf')

    # ROC
    plt.figure(figsize=(8,8))
    fpr, tpr, thresholds = roc_curve(y, scores, sample_weight=weights)
    plt.plot(fpr, tpr, color = "red", label = F"Out-of-fold AUC = {roc_auc_score(y, scores, sample_weight=weights):.4f}")
    if train_scores is not None:
        fpr_tr, tpr_tr, _ = roc_curve(train_y, train_scores, sample_weight=train_weights)
        plt.plot(fpr_tr, tpr_tr, color = "blue", ls = ":", label = F"Training AUC = {roc_auc_score(train_y, train_scores, sample_weight=train_weights):.4f}")
    for f, c in zip(fold_ids, fold_colors):
        in_fold = folds == f
        fpr_f, tpr_f, _ = roc_curve(y[in_fold], scores[in_fold], sample_weight=weights[in_fold])
        plt.plot(fpr_f, tpr_f, color = c, lw = 1, alpha = 0.7, label = F"Fold {f} AUC = {roc_auc_score(y[in_fold], scores[in_fold], sample_weight=weights[in_fold]):.4f}")
    diag = np.logspace(-4, 0, 100)
    plt.plot(diag, diag, color = "black", ls = "--")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1e-4, 1])
    plt.ylim([1e-4, 1])
    plt.xlabel('FPR' , fontsize=12)
    plt.ylabel('TPR' , fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend( prop={'size':12} , loc = 4)
    hep.cms.text("Private Work")
    plt.savefig(f'plots/{bdt_name}/{bdt_name}_roc_curve.pdf')

    # PR
    plt.figure(figsize=(8,8))
    precision, recall, thresholds2 = precision_recall_curve(y, scores, sample_weight=weights)
    plt.plot(recall, precision, color = "red", label=f"Out-of-fold PR: {auc(recall, precision):.4f}")
    for f, c in zip(fold_ids, fold_colors):
        in_fold = folds == f
        precision_f, recall_f, _ = precision_recall_curve(y[in_fold], scores[in_fold], sample_weight=weights[in_fold])
        plt.plot(recall_f, precision_f, color = c, lw = 1, alpha = 0.7, label=f"Fold {f} PR: {auc(recall_f, precision_f):.4f}")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1e-4, 1])
    plt.ylim([1e-4, 1])
    plt.ylabel('Precision (TP/(TP+FP))', fontsize=12)
    plt.xlabel('Signal Efficiency (TPR)', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend( prop={'size':12} , loc = 3)
    hep.cms.text("Private Work")
    plt.savefig(f'plots/{bdt_name}/{bdt_name}_PR_curve.pdf')
    plt.close('all')

def make_importance_plot(bdt_name,models,variables):
    # per-fold feature importance side by side, same F score (importance_type='weight') as xgb.plot_importance
    os.makedirs(f"plots/{bdt_name}/",exist_ok=True)
    imp = np.array([[bst.get_booster().get_score(importance_type='weight').get(f'f{idx}',0) for idx in range(len(variables))]
                    for bst in models])
    order = np.argsort(imp.mean(axis=0))
    k = len(models)
    width = 0.8/k
    colors = plt.cm.viridis(np.linspace(0,0.9,k))
    plt.figure(figsize=(8,6))
    ypos = np.arange(len(variables))
    for i in range(k):
        plt.barh(ypos - 0.4 + (i+0.5)*width, imp[i][order], height=width, color=colors[i], label=f'Fold {i}')
    plt.yticks(ypos, [variables[j] for j in order])
    plt.xlabel('F score')
    plt.ylabel('Features')
    plt.legend()
    hep.cms.text("Private Work")
    plt.savefig(f'plots/{bdt_name}/{bdt_name}_importance.pdf', bbox_inches = "tight")
    plt.close('all')

def save_oof_scores(fname,scores,folds,event_num,sig_point=None):
    with h5py.File(fname,'w') as f:
        f.create_dataset('score',data=scores)
        f.create_dataset('fold',data=folds)
        f.create_dataset('eventNum',data=event_num)
        if sig_point is not None:
            for k in ['m1','delta','ctau']:
                f.create_dataset(k,data=sig_point[k])

def exclude_signal_points(sig_data,sig_xsec_norm,sig_point,exclude_points):
    # exclude_points: list of dicts with any subset of keys 'm1', 'delta', 'ctau'.
    # An event is dropped if it matches every key given in any one dict, e.g.
    # [{'m1':0.05}, {'delta':0.05,'ctau':100}] drops all m1=0.05 points and all delta=0.05, ctau=100 points.
    # Every entry of sig_point (including 'eventNum' and 'fold') is filtered with the same mask.
    drop =np.zeros(len(sig_data),dtype=bool)
    for excl in exclude_points:
        unknown = set(excl) - {'m1','delta','ctau'}
        if unknown:
            raise ValueError(f'Unknown keys in exclude_points entry {excl}: {unknown}')
        match = np.ones(len(sig_data),dtype=bool)
        for k, v in excl.items():
            match &= np.isclose(sig_point[k],v)
        if np.sum(match) == 0:
            print(f'Warning: exclusion {excl} matches no signal events')
        drop |= match
    keep = ~drop
    print(f'Excluding {np.sum(drop)} of {len(sig_data)} signal events via exclude_points')
    sig_point = {k: v[keep] for k, v in sig_point.items()}
    return sig_data[keep], sig_xsec_norm[keep], sig_point

def train_bdt(bdt_name,signal_h5,bkg_h5,variables,max_depth=6,n_estimators=500,lr=0.01,split_first=False,exclude_points=None):
    # K_FOLDS-fold cross training. Returns the fold models (models[i] is trained on folds != i and must only be
    # used on events with fold == i), the WPs from out-of-fold scores, and a dict with the out-of-fold scores.
    k = K_FOLDS

    sig_files = [f"h5/{signal_h5}.h5"]
    bkg_files = [f"h5/{bkg_h5}.h5"]
    check_features(sig_files+bkg_files,variables)

    # load signal
    sig_data, sig_xsec_norm, sig_point = process_signal_inputs(sig_files,variables)
    if exclude_points:
        sig_data, sig_xsec_norm, sig_point = exclude_signal_points(sig_data,sig_xsec_norm,sig_point,exclude_points)
    sig_fold = sig_point['fold']

    # load bkg
    bkg_data, bkg_xsec_norm, bkg_fold = process_bkg_inputs(bkg_files,variables)
    bkg_event_num = []
    for bf in bkg_files:
        with h5py.File(bf,"r") as fin:
            bkg_event_num.append(read_event_num(fin,bf))
    bkg_event_num = np.concatenate(bkg_event_num)

    print_fold_balance(sig_point,bkg_fold,k)

    # pass 1: grid search on each fold's training events, then one common parameter set for all folds
    fold_inputs, cv_results = [], []
    for i in range(k):
        print(f'\n========== Fold {i}: grid search on folds != {i} ==========')
        sig_train = sig_fold != i
        bkg_train = bkg_fold != i
        sig_point_train = {key: v[sig_train] for key, v in sig_point.items()}
        sig_subprocess, nSamp = build_sig_subprocess(sig_point_train)

        # reweight using the training folds only
        bkg_nev_raw, bkg_sumWgts, bkg_sf, bkg_xsec_norm_train = reweight_bkg_arrays(bkg_xsec_norm[bkg_train])
        sig_sf = reweight_signal(sig_data[sig_train],bkg_nev_raw,nSamp,bkg_sf,sig_subprocess,bkg_xsec_norm_train)
        fold_inputs.append((sig_train,bkg_train,sig_point_train,sig_sf,np.array(bkg_sf),bkg_xsec_norm_train))

        cv_results.append(grid_search_fold(*build_training_set(sig_data[sig_train],sig_sf,bkg_data[bkg_train],np.array(bkg_sf))))
    params = pick_common_params(cv_results)

    # pass 2: train every fold model with the common parameters
    models = []
    train_y, train_scores, train_wgts = [], [], []
    for i in range(k):
        print(f'\n========== Fold {i}: training on folds != {i} ==========')
        sig_train, bkg_train, sig_point_train, sig_sf, bkg_sf, bkg_xsec_norm_train = fold_inputs[i]

        # train bdt
        bst = run_training(f'{bdt_name}_fold{i}',variables,sig_data[sig_train],sig_sf,sig_xsec_norm[sig_train],sig_point_train,
                           bkg_data[bkg_train],bkg_sf,bkg_xsec_norm_train,params)
        models.append(bst)

        # scores on this model's own training events, for the overtraining check
        train_scores.append(bst.predict_proba(np.concatenate((sig_data[sig_train],bkg_data[bkg_train]),axis=0))[:,1])
        train_y.append(np.concatenate((np.ones(np.sum(sig_train)),np.zeros(np.sum(bkg_train)))))
        train_wgts.append(np.abs(np.concatenate((sig_sf,bkg_sf))))

    # out-of-fold scores
    sig_score = predict_kfold(models,sig_data,sig_fold)
    bkg_score = predict_kfold(models,bkg_data,bkg_fold)
    y = np.concatenate((np.ones(len(sig_score)),np.zeros(len(bkg_score))))
    scores = np.concatenate((sig_score,bkg_score))
    folds = np.concatenate((sig_fold,bkg_fold))

    # evaluation weights: same scheme as training (xsec-weighted bkg with negative weights zeroed, equal weight per
    # signal point), computed on the full sample so every out-of-fold event has one
    print('\nComputing full-sample weights for the out-of-fold evaluation')
    sig_subprocess_all, nSamp_all = build_sig_subprocess(sig_point)
    bkg_nev_all, _, bkg_wgt, _ = reweight_bkg_arrays(bkg_xsec_norm)
    sig_wgt = reweight_signal(sig_data,bkg_nev_all,nSamp_all,bkg_wgt,sig_subprocess_all,bkg_xsec_norm)
    sig_wgt, bkg_wgt = np.abs(sig_wgt), np.abs(bkg_wgt)
    weights = np.concatenate((sig_wgt,bkg_wgt))

    print('\nOut-of-fold AUC (weighted)')
    for i in range(k):
        in_fold = folds == i
        print(f'fold{i}: {roc_auc_score(y[in_fold],scores[in_fold],sample_weight=weights[in_fold]):.4f}')
    print(f'all folds: {roc_auc_score(y,scores,sample_weight=weights):.4f}')

    make_eval_plots(f'{bdt_name}_kfold',y,scores,np.concatenate(train_y),np.concatenate(train_scores),folds=folds,
                    weights=weights,train_weights=np.concatenate(train_wgts))
    make_importance_plot(f'{bdt_name}_kfold',models,variables)
    tpr_WP, thres_WP = compute_WPs(y,scores,weights=weights)

    save_oof_scores(f"h5/{bdt_name}_oof_scores_sig.h5",sig_score,sig_fold,sig_point['eventNum'],sig_point)
    save_oof_scores(f"h5/{bdt_name}_oof_scores_bkg.h5",bkg_score,bkg_fold,bkg_event_num)

    # vars dict
    # WPs from the weighted out-of-fold ROC: thres_WP[wp] is the score cut giving signal efficiency tpr_WP[wp]
    vars_dict = {"variables":variables, "k":k, "fold_rule":"eventNum % k", "params":params,
                 "models":[f"models/{bdt_name}_fold{i}.json" for i in range(k)],
                 "tpr_WP":{wp: float(v) for wp, v in tpr_WP.items()},
                 "thres_WP":{wp: float(v) for wp, v in thres_WP.items()}}
    with open(f"models/variables_{bdt_name}.json","w") as fout:
        json.dump(vars_dict,fout,indent=4)

    oof = {'sig_score':sig_score, 'bkg_score':bkg_score, 'sig_fold':sig_fold, 'bkg_fold':bkg_fold, 'sig_point':sig_point,
           'sig_wgt':sig_wgt, 'bkg_wgt':bkg_wgt}

    return models, tpr_WP, thres_WP, oof
