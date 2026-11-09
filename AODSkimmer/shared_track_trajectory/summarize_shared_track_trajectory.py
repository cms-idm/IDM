#!/usr/bin/env python3
"""Summarize SharedTrackTrajectoryCheck output (one or more *_tracks.csv files).

  python3 summarize_shared_track_trajectory.py --tracks run1_tracks.csv [run2_tracks.csv ...]

Counting unit: one distinct shared innerTrack (event + ProductID + key). A two-user track is one A/B pair
(A = lower displacedMuon index). Tracks with more users are listed, not put in the A/B tables.
Per dR threshold:
  matched   = at least one eligible gen muon within dR at the ECAL inner boundary (no charge requirement)
  ambiguous = two or more gen muons within dR (listed; not in the A/B tables)
  unique    = exactly one; split by charge relation (same / opposite) between track and gen muon.
Tables: raw forward-score preference (A / B / tie) versus the trajectory conclusion, chamber level and
hit level (segments + RPC hits) kept separate. Tie-breaking (global, then index) is reported separately.
"""
import argparse
import csv
from collections import Counter

CONCLUSIONS = ["A_only", "B_only", "both", "neither", "untestable"]
PREFS = ["A", "B", "tie"]


def table(rows, column):
    grid = Counter((r["raw_preference"], r[column]) for r in rows)
    width = max(len(c) for c in CONCLUSIONS) + 2
    lines = ["    " + "score\\trajectory".ljust(20) + "".join(c.rjust(width) for c in CONCLUSIONS) + "total".rjust(width)]
    for p in PREFS:
        counts = [grid[p, c] for c in CONCLUSIONS]
        lines.append(f"    {p:<20}" + "".join(str(n).rjust(width) for n in counts) + str(sum(counts)).rjust(width))
    return "\n".join(lines)


def agreement(rows, column):
    tally = Counter()
    for r in rows:
        p, c = r["raw_preference"], r[column]
        if c == "untestable":
            tally["untestable"] += 1
        elif p == "tie":
            tally["score tie"] += 1
        elif c in ("both", "neither"):
            tally[f"trajectory {c}"] += 1
        elif (p, c) in (("A", "A_only"), ("B", "B_only")):
            tally["agree"] += 1
        else:
            tally["disagree"] += 1
    return ", ".join(f"{k}={tally[k]}" for k in
                     ["agree", "disagree", "score tie", "trajectory both", "trajectory neither", "untestable"])


def ident(r):
    return f"{r['run']}:{r['lumi']}:{r['event']} track({r['track_product']},{r['track_key']})"


def describe(r):
    return (f"  {ident(r)} gen pdgId={r['gen_pdgId']} q={r['gen_charge']} pt={r['gen_pt']} vxy={r['gen_vxy']} "
            f"vz={r['gen_vz']} mother={r['gen_first_other_mother'] or 'none'} dR={r['gen_dR_ecal']} | "
            f"A[{r['A_index']}] global={r['A_global']} score={r['A_score']} chamber={r['A_chamber_verdict']} "
            f"hit={r['A_hit_verdict']} | B[{r['B_index']}] global={r['B_global']} score={r['B_score']} "
            f"chamber={r['B_chamber_verdict']} hit={r['B_hit_verdict']} | identical chambers={r['identical_hit_chambers']} "
            f"crossedIP={r['crossedIP'] or 'n/a'} first-hit r.p={r['first_hit_rdotp'] or 'n/a'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", nargs="+", required=True)
    args = parser.parse_args()
    rows = [r for path in args.tracks for r in csv.DictReader(open(path))]
    thresholds = [c[len("n_gen_dR_lt_"):] for c in rows[0] if c.startswith("n_gen_dR_lt_")] if rows else []

    print(f"Distinct shared innerTracks: {len(rows)}")
    print("  users per track: " + ", ".join(f"{k} users={v}" for k, v in sorted(Counter(r["n_users"] for r in rows).items())))
    for r in rows:
        if r["n_users"] != "2":
            print(f"  >2 users (not in A/B tables): {ident(r)} users={r['user_indices']}")
    status = Counter(r["reco_ecal_status"] for r in rows)
    print("Track at ECAL inner boundary (along momentum from last hit): " + ", ".join(f"{k}={v}" for k, v in sorted(status.items())))
    print(f"Tracks in events where at least one gen muon failed ECAL propagation: "
          f"{sum(int(r['n_gen_ecal_failure']) > 0 for r in rows)}; "
          f"where at least one gen muon starts beyond the boundary: {sum(int(r['n_gen_ecal_start_beyond']) > 0 for r in rows)}")

    for t in thresholds:
        key = f"n_gen_dR_lt_{t}"
        ok = [r for r in rows if r["reco_ecal_status"] == "ok"]
        matched = [r for r in ok if int(r[key]) >= 1]
        ambiguous = [r for r in matched if int(r[key]) >= 2]
        unique = [r for r in matched if int(r[key]) == 1]
        unmatched = [r for r in ok if int(r[key]) == 0]
        print(f"\n=== dR < {t} ===")
        print(f"  track ECAL point ok: {len(ok)} | matched: {len(matched)} (unique {len(unique)}, ambiguous {len(ambiguous)}) "
              f"| unmatched: {len(unmatched)} (of which in events with a gen ECAL failure: "
              f"{sum(int(r['n_gen_ecal_failure']) > 0 for r in unmatched)}) "
              f"| track ECAL propagation not ok: {len(rows) - len(ok)}")
        print("  unique matches by charge: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(r["charge_relation"] for r in unique).items())))
        for charge in ("same", "opposite"):
            pairs = [r for r in unique if r["charge_relation"] == charge and r["n_users"] == "2"]
            print(f"\n  [{charge}-charge unique matches, two-user tracks: {len(pairs)}]")
            for column, label in (("chamber_conclusion", "chamber bounds"), ("hit_conclusion", "segments + RPC hits")):
                print(f"   {label}:")
                print(table(pairs, column))
                print("    " + agreement(pairs, column))
            disagree = [r for r in pairs for c in ("chamber_conclusion", "hit_conclusion")
                        if (r["raw_preference"], r[c]) in (("A", "B_only"), ("B", "A_only"))]
            seen = set()
            if disagree:
                print("   score-trajectory disagreements:")
            for r in disagree:
                if ident(r) not in seen:
                    seen.add(ident(r))
                    print(describe(r))
            ties = [r for r in pairs if r["raw_preference"] == "tie"]
            if ties:
                print("   score ties (choice after tie-break: global, then earlier index):")
            for r in ties:
                print(f"    -> keeps {r['tiebreak_preference']}; " + describe(r).strip())
            gem = sum(int(r["A_gem_hits_not_tested"] or 0) + int(r["B_gem_hits_not_tested"] or 0) for r in pairs)
            if gem:
                print(f"   GEM rechit matches not tested at hit level: {gem}")
        if ambiguous:
            print("\n  ambiguous (two or more gen muons within dR; nearest shown):")
            for r in ambiguous:
                print(f"  {ident(r)} n_gen={r[key]} nearest dR={r['gen_dR_ecal']}")


if __name__ == "__main__":
    main()
