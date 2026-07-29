# Inelastic Dark Matter with Leptons (this README is out-of-date, needs to be modified for Run 3)

This repository contains all the code I've developed for generating events, skimming AOD, and running analysis with [coffea](https://coffeateam.github.io/coffea/). Each subfolder has its own README (if it doesn't, it means I haven't gotten to it yet and you should bug me about it!).

This is all designed to run in `CMSSW 10_6_26`, and ideally on the Fermilab LPC with the condor job queue. I've tried to avoid hard-coding any user-specific paths etc. into any of the code, but it's certainly not perfect. Please let me know if something is broken for you. Most things tha

## Getting set up
To use this repository you'll need the `CMSSW 10_6_26` release, which you can get by running
```bash
cmsrel CMSSW_10_6_26
cd CMSSW_10_6_26/src
cmsenv
git cms-init
```
Then, clone this repository by running
```bash
git clone https://github.com/SamBT/iDMe.git
```

## Cloning CMSSW submodules and applying patches
I've made a few minor adjustments to some of the MINIAOD-producing code in CMSSW, mostly to do with generating the low- $p_T$ electrons. The low- $p_T$ ID included in MINIAOD is "sub-optimal" (see [Rob's twiki](https://twiki.cern.ch/twiki/bin/view/Main/RobBainbridgeSandBox#UltraLegacy_data_sets) for details), so I've replaced it with the "optimal" one ("2020Nov28"). Energy regression is also not applied at MINIAOD level for some unthinkably ridiculous reason, so I've had to add that as well. To get these corrections, you'll need to checkout a few CMSSW submodules and apply the patch file. The patch file also includes a [tracker dimensions extension](https://www.classe.cornell.edu/~skb93/iDMe/git_patches/extendTrackerDimensions.patch) taken from the [original iDM analysis](https://github.com/afrankenthal/iDMAnalysis/tree/master/skimmer) to aid displaced vertexing, but I'm not sure this is necessary any more (can't hurt!). From the `src` directory, run these lines to checkout the modules:
```bash
git cms-addpkg PhysicsTools/PatAlgos
git cms-addpkg RecoEgamma/EgammaElectronProducers
git cms-addpkg RecoVertex/KalmanVertexFit
git cms-addpkg RecoVertex/VertexTools
```
and these lines to apply the patch:
```bash
git apply iDMe/patches/lowPtElectrons_miniAOD_IDRegression.patch
```

## Installing EGamma PostRecoTools
We need to run EGamma "PostRecoTools" according to [EGamma POG recommendations](https://twiki.cern.ch/twiki/bin/view/CMS/EgammaUL2016To2018#Recipe_for_running_scales_and_sm). Run these commands (copied from the linked twiki) to set it up:
```bash
git cms-addpkg RecoEgamma/EgammaTools
git clone https://github.com/cms-egamma/EgammaPostRecoTools.git
mv EgammaPostRecoTools/python/EgammaPostRecoTools.py RecoEgamma/EgammaTools/python/.
git clone -b ULSSfiles_correctScaleSysMC https://github.com/jainshilpi/EgammaAnalysis-ElectronTools.git EgammaAnalysis/ElectronTools/data/
git cms-addpkg EgammaAnalysis/ElectronTools
```

## Building
Lastly, build everything with 
```bash
scram b -j8
```
And that should do it! If you make changes to the modules inside `iDMe`, you'll obviously need to rebuild with `scram`.

## What the submodules do
Here's a brief overview of the most important modules in this repo:
1. `UL_MCProduction` : framework for submitting condor jobs to produce UL AOD for signal Monte Carlo, starting from gridpacks. Currently only working for UL 2018, but will be extended for 2016/17 soon. This is the workhorse for our private MC production. Check out the README inside for instructions.
2. `AODSkimmer` : Code to run on top of AOD and produce flat ntuples for analysis. Works by generating the MINIAOD products "on the fly", with the correct energy regression + ID for the low- $p_T$ electrons. Can be run on condor (for samples that live on the LPC, i.e. privately produced signal) or crab (for centrally produced samples). Check out the README inside for instructions.
3. `python_analysis` : Contains all the `coffea` analysis scripts and tools. There are `condor` submission scripts, but lately I've been running analysis directly from a jupyter notebook using an `ssh` tunnel to an LPC login node (this only makes sense time-wise using the slimmed ntuples I've made, which have the major iDM cuts -- e.g. $E_T^\mathrm{miss}$ -- as preselection, and thus have much lower event yields). Check out the README inside for more details

And here's an overview of the less important modules:
1. `iDMeSkimmer` : Old version of the ntuplizer designed to run on MINIAOD. Not used any more, as we prefer AOD for a number of reasons. Still exists as skeleton code we can work with if we ever return to MINIAOD.
2. `CustomTools` : Some helper functions/modules for `AODSkimmer`. Some tools are defunct and no longer used, e.g. the stuff I tried out a while back for finding displaced electrons in the `isolatedTracks` collection.

## Contributing code
If you'd like to contribute code, please fork this repo! If you're doing something wacky/experimental/that will potentially break the project, please do it on a separate branch (let me know and I can make the corresponding branch in my repo). For core analysis development, you can submit pull requests to `main` after pulling or rebasing any upstream changes.
