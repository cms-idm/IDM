import uproot
import awkward as ak
import numpy as np

# Open a ROOT file
with uproot.open("htctest.root") as f:

    # List available keys (trees, histograms, etc.)
    print("Keys:", f.keys())

    # Access a tree
    tree = f["ntuples/outT"]

    # Print available branches
    print("Branches:", tree.keys())

    # Read specific branches as arrays
    arrays = tree.arrays(["XCLptElectron_pt", "LptElectron_pt", "Electron_pt"], library="ak")

    # Print first few entries
    print("XCLpt:", arrays["XCLptElectron_pt"][:10])
    print("Lpt:", arrays["LptElectron_pt"][:10])
    print("GED:", arrays["Electron_pt"][:10])

