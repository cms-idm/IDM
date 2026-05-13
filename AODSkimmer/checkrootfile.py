import uproot
import awkward as ak
import numpy as np

#['Electron_genMatched', 'Electron_hasLptMatch', 'Electron_lptMatchIdx', 'Electron_hasAllLptMatch', 'Electron_AllLptMatchIdx', 'LptElectron_genMatched', 'LptElectron_gedIsMatched', 'AllLptElectron_genMatched', 'AllLptElectron_gedIsMatched', 'vtx_isMatched', 'vtx_e1_isMatched', 'vtx_e2_isMatched']

# Open a ROOT file
with uproot.open("htctest.root") as f:

    # List available keys (trees, histograms, etc.)
    print("Keys:", f.keys())

    # Access a tree
    tree = f["ntuples/outT"]

    # Print available branches
    print("Match Branches:", [f for f in tree.keys() if 'Match' in f])

    # Read specific branches as arrays
    arrays = tree.arrays(["AllLptElectron_pt", "LptElectron_pt", "Electron_pt", "PFJet_pt", "nPFJet", "nPFJetAll",
                          'AllLptElectron_genMatched', 'AllLptElectron_gedIsMatched', 'Electron_hasAllLptMatch', 'Electron_AllLptMatchIdx',
                          'AllLptElectron_xCleaned', 'AllLptElectron_gedIdx', 'GenEle_matchedAllLowPt', 'GenEle_matched'], library="ak")

    # Print first few entries
    print("AllLpt:", arrays["AllLptElectron_pt"][:10])
    print("Lpt:", arrays["LptElectron_pt"][:10])
    print("GED:", arrays["Electron_pt"][:10])
    print("PFJet:", arrays["PFJet_pt"][:10])
    print("nPFJet:", arrays["nPFJet"][:10])
    print("nPFJetAll:", arrays["nPFJetAll"][:10])

    print("AllLptGenM:", arrays["AllLptElectron_genMatched"][:10])
    print("AllLptGEDM:", arrays["AllLptElectron_gedIsMatched"][:10])
    print("GEDhasAllLptM:", arrays["Electron_hasAllLptMatch"][:10])
    print("GEDAllLptMidx:", arrays["Electron_AllLptMatchIdx"][:10])

    print("AllLptXC:", arrays["AllLptElectron_xCleaned"][:10])
    print("AllLptGEDidx:", arrays["AllLptElectron_gedIdx"][:10])
    print("GenEleMatched:", arrays["GenEle_matched"][:10])
    print("GenEleMatchedAll:", arrays["GenEle_matchedAllLowPt"][:10])

    disagree = arrays["GenEle_matched"] != arrays["GenEle_matchedAllLowPt"]
    print("GenEleMatched:", arrays["GenEle_matched"][disagree][:10])
    print("GenEleMatchedAll:", arrays["GenEle_matchedAllLowPt"][disagree][:10])

