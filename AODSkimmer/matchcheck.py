"""
Standalone check (plain uproot, no coffea) for whether gen electron / gen positron
matching in the ntuple ever grabs the WRONG-CHARGE reco track.

This is the suspected root cause of the gen-dxy vs GED/Lpt-dxy sign flip:
in AODSkimmer/plugins/ElectronSkimmer.cc, the gen-to-reco matching loop only
uses angular distance (dR) to decide a match -- it never checks that the
matched track's charge is actually correct (-1 for the electron, +1 for the
positron). If the wrong-charge sibling gets matched, dxy comes out with the
opposite sign from the gen-level expectation, since dxy sign depends on charge.

Usage:
    python check_charge_matching.py /path/to/your_ntuple.root
"""

import sys
import uproot
import awkward as ak
import numpy as np

def main(filepath):
    f = uproot.open(filepath)
    tree = f["ntuples/outT"]

    branches = [
        "GenEle_matchType", "GenEle_matchIdxLocal",
        "GenEle_matchedAllLowPt", "GenEle_matchIdxAllLowPt",
        "GenPos_matchType", "GenPos_matchIdxLocal",
        "GenPos_matchedAllLowPt", "GenPos_matchIdxAllLowPt",
        "Electron_charge", "LptElectron_charge", "AllLptElectron_charge",
    ]
    arrays = tree.arrays(branches, library="ak")

    def check(gen_label, match_type, idx_local, idx_alllowpt, matched_alllowpt,
              expected_charge, ele_charge, lpt_charge, alllpt_charge):
        """
        expected_charge: -1 for GenEle, +1 for GenPos
        """
        n_checked_RL = 0
        n_wrong_RL = 0

        # --- Regular ("R") / LowPt ("L") match, via matchIdxLocal ---
        is_R = match_type == "R"
        is_L = match_type == "L"

        # Regular electron matches
        r_idx = idx_local[is_R]
        r_charge_true = ele_charge[is_R]
        matched_charge_R = np.array([
            row[i] if 0 <= i < len(row) else None
            for row, i in zip(r_charge_true.tolist(), r_idx.tolist())
        ])
        n_R = np.sum(matched_charge_R != None)
        wrong_R = np.sum([c == -expected_charge for c in matched_charge_R if c is not None])

        # LowPt electron matches
        l_idx = idx_local[is_L]
        l_charge_true = lpt_charge[is_L]
        matched_charge_L = np.array([
            row[i] if 0 <= i < len(row) else None
            for row, i in zip(l_charge_true.tolist(), l_idx.tolist())
        ])
        n_L = np.sum(matched_charge_L != None)
        wrong_L = np.sum([c == -expected_charge for c in matched_charge_L if c is not None])

        n_checked_RL = n_R + n_L
        n_wrong_RL = wrong_R + wrong_L

        # --- AllLowPt match, via matchIdxAllLowPt ---
        has_all = matched_alllowpt == True
        all_idx = idx_alllowpt[has_all]
        all_charge_true = alllpt_charge[has_all]
        matched_charge_ALL = np.array([
            row[i] if 0 <= i < len(row) else None
            for row, i in zip(all_charge_true.tolist(), all_idx.tolist())
        ])
        n_ALL = np.sum(matched_charge_ALL != None)
        wrong_ALL = np.sum([c == -expected_charge for c in matched_charge_ALL if c is not None])

        print(f"--- {gen_label} (expected charge {expected_charge:+d}) ---")
        print(f"  R/L match (matchIdxLocal):      {n_wrong_RL} / {n_checked_RL} wrong-charge matches"
              f" ({100*n_wrong_RL/n_checked_RL:.2f}%)" if n_checked_RL else "  R/L match: no matched entries")
        print(f"  AllLowPt match (matchIdxAllLowPt): {wrong_ALL} / {n_ALL} wrong-charge matches"
              f" ({100*wrong_ALL/n_ALL:.2f}%)" if n_ALL else "  AllLowPt match: no matched entries")
        print()

    check(
        "GenEle", arrays["GenEle_matchType"], arrays["GenEle_matchIdxLocal"],
        arrays["GenEle_matchIdxAllLowPt"], arrays["GenEle_matchedAllLowPt"],
        expected_charge=-1,
        ele_charge=arrays["Electron_charge"], lpt_charge=arrays["LptElectron_charge"],
        alllpt_charge=arrays["AllLptElectron_charge"],
    )

    check(
        "GenPos", arrays["GenPos_matchType"], arrays["GenPos_matchIdxLocal"],
        arrays["GenPos_matchIdxAllLowPt"], arrays["GenPos_matchedAllLowPt"],
        expected_charge=+1,
        ele_charge=arrays["Electron_charge"], lpt_charge=arrays["LptElectron_charge"],
        alllpt_charge=arrays["AllLptElectron_charge"],
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_charge_matching.py /path/to/ntuple.root")
        sys.exit(1)
    main(sys.argv[1])