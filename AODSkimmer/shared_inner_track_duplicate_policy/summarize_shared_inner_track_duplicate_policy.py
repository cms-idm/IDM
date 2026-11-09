#!/usr/bin/env python3
"""Summarize SharedInnerTrackDuplicatePolicy output.

  python3 summarize_shared_inner_track_duplicate_policy.py --all shared_inner_track_duplicate_policy.csv [...] \
      [--genmatched shared_inner_track_duplicate_policy_genmatched.csv [...]]

Counting unit: one distinct shared innerTrack (event + ProductID + key). Two-user tracks are one A/B pair
(A = lower displacedMuon index); tracks with more users are listed, never put into the policy counts.
Rules:
  original = components (strict superset of outer/global) first, else forward score; equal score = tie
  new      = crossedIP: forward score; otherwise components; anything else = tie
Tie fraction = ties / two-user tracks in that category.
Gen-matched file: categories any-charge / same-sign at each dR threshold, for
  nearest   = nearest gen muon of that kind within dR (unchanged definition)
  exclusive = only gen muons whose closest displacedMuon inner track is this shared track
Forward/behind (exclusive only): the exclusive gen muon propagated to each copy's chambers; it prefers a copy
when that copy's chambers are all forward and the other's all behind, otherwise it is undecided.
"""
import argparse
import csv
from collections import Counter

CROSSED = [("1", "crossedIP = true"), ("0", "crossedIP = false"), ("", "crossedIP = n/a (no TrackExtra)")]


def read(paths):
    return [r for path in paths for r in csv.DictReader(open(path))]


def frac(n, d):
    return f"{n}/{d} = {n / d:.3f}" if d else f"{n}/0 = n/a"


def ident(r):
    return f"{r['run']}:{r['lumi']}:{r['event']} track({r['track_product']},{r['track_key']})"


def new_rule(r):
    if r["crossedIP"] == "1":
        return r["score_preference"]
    return r["final_choice"] if r["decision_stage"] == "components" else "tie"


def policy(rows, indent):
    """rows: two-user tracks."""
    comp = [r for r in rows if r["decision_stage"] == "components"]
    score = [r for r in rows if r["decision_stage"] == "score"]
    wins = Counter(r["final_choice"] for r in score)
    relation = Counter(r["component_relation"] for r in score)
    print(f"{indent}two-user shared tracks: {len(rows)}")
    print(f"{indent}original rule: A dominates {sum(r['component_relation'] == 'A_dominates' for r in comp)}, "
          f"B dominates {sum(r['component_relation'] == 'B_dominates' for r in comp)}, "
          f"score fallback {len(score)} (equal components {relation['equal']}, incomparable {relation['incomparable']}): "
          f"A wins {wins['A']}, B wins {wins['B']}, ties {wins['tie']}; tie fraction {frac(wins['tie'], len(rows))}")
    new = Counter(new_rule(r) for r in rows)
    print(f"{indent}new rule: A {new['A']}, B {new['B']}, ties {new['tie']}; "
          f"differs from original in {sum(new_rule(r) != r['final_choice'] for r in rows)}")
    for value, label in CROSSED:
        sub = [r for r in rows if r["crossedIP"] == value]
        if not sub and value == "":
            continue
        sub_score = [r for r in sub if r["decision_stage"] == "score"]
        ties = sum(r["final_choice"] == "tie" for r in sub_score)
        orig = Counter(r["final_choice"] for r in sub)
        new = Counter(new_rule(r) for r in sub)
        print(f"{indent}{label}: total {len(sub)}, component decisions {len(sub) - len(sub_score)}, "
              f"score fallbacks {len(sub_score)}, score ties {ties} | original A/B/tie {orig['A']}/{orig['B']}/{orig['tie']}"
              f" | new A/B/tie {new['A']}/{new['B']}/{new['tie']}")


def side(f, b, x):
    if f + b + x == 0:
        return "none"
    if x:
        return "failure"
    return "forward" if b == 0 else "behind" if f == 0 else "split"


def forward_behind(r, kind):
    v = [r[f"{kind}_excl_{c}"] for c in ("A_forward", "A_behind", "A_failure", "B_forward", "B_behind", "B_failure")]
    if "" in v:
        return "n/a", "n/a"
    af, ab, ax, bf, bb, bx = map(int, v)
    sa, sb = side(af, ab, ax), side(bf, bb, bx)
    verdict = "A" if (sa, sb) == ("forward", "behind") else "B" if (sa, sb) == ("behind", "forward") else "undecided"
    return verdict, f"{sa}/{sb}"


def directions(rows, kind, indent):
    """rows: two-user tracks matched exclusively."""
    for value, label in CROSSED:
        sub = [r for r in rows if r["crossedIP"] == value]
        if not sub:
            continue
        res = [(r, *forward_behind(r, kind)) for r in sub]
        pattern = Counter(p for _, _, p in res)
        decided = [(r, v) for r, v, _ in res if v in ("A", "B")]
        agree_orig = sum(r["final_choice"] == v for r, v in decided)
        agree_new = sum(new_rule(r) == v for r, v in decided)
        print(f"{indent}{label} forward/behind (A/B copies): " + ", ".join(f"{k} {n}" for k, n in sorted(pattern.items())))
        print(f"{indent}  decided {len(decided)}: agrees with original rule {agree_orig}, with new rule {agree_new}")
        for r, v in decided:
            if r["final_choice"] != v or new_rule(r) != v:
                print(f"{indent}    {ident(r)} forward/behind={v} original={r['final_choice']} new={new_rule(r)} "
                      f"components={r['user_components']} scores={r['user_scores']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", nargs="+", required=True, help="shared_inner_track_duplicate_policy.csv file(s)")
    parser.add_argument("--genmatched", nargs="+", default=[], help="..._genmatched.csv file(s)")
    args = parser.parse_args()

    rows = read(args.all)
    two = [r for r in rows if r["n_users"] == "2"]
    multi = [r for r in rows if r["n_users"] != "2"]
    print("=== All shared innerTracks ===")
    print(f"  total distinct shared innerTracks: {len(rows)}")
    print(f"  two-user shared tracks: {len(two)}")
    print(f"  >2-user tracks: {len(multi)} (not reduced to A/B)")
    for r in multi:
        print(f"    {ident(r)} users={r['user_indices']} components={r['user_components']} "
              f"scores={r['user_scores']} crossedIP={r['crossedIP'] or 'n/a'}")
    print("  overall:")
    policy(two, "    ")

    if not args.genmatched:
        return
    gen = read(args.genmatched)
    if not gen:
        print("\n=== Gen-matched shared innerTracks: none ===")
        return
    thresholds = [c[len("n_gen_anycharge_dR_lt_"):] for c in gen[0] if c.startswith("n_gen_anycharge_dR_lt_")]
    print(f"\n=== Gen-matched shared innerTracks (file rows: {len(gen)}) ===")
    print("  standalone-only displacedMuons (not reco candidates) in these events: "
          + ", ".join(f"{k} in {v} tracks" for k, v in sorted(Counter(r["n_standalone_only_muons"] for r in gen).items())))
    print("  displacedMuon inner tracks without an ECAL point in these events: "
          + ", ".join(f"{k} in {v} tracks" for k, v in sorted(Counter(r["n_reco_inner_ecal_not_ok"] for r in gen).items())))
    for kind in ("anycharge", "samesign"):
        for t in thresholds:
            near = [r for r in gen if int(r[f"n_gen_{kind}_dR_lt_{t}"]) >= 1]
            excl = [r for r in gen if int(r[f"n_gen_{kind}_excl_dR_lt_{t}"]) >= 1]
            for label, matched, key in (("nearest", near, f"n_gen_{kind}_dR_lt_{t}"),
                                        ("exclusive", excl, f"n_gen_{kind}_excl_dR_lt_{t}")):
                m_two = [r for r in matched if r["n_users"] == "2"]
                print(f"\n  [{kind} dR < {t}, {label}] matched shared tracks: {len(matched)} "
                      f"(two-user {len(m_two)}, >2-user {len(matched) - len(m_two)}, "
                      f"two or more gen muons within dR {sum(int(r[key]) >= 2 for r in matched)})")
                policy(m_two, "    ")
                if label == "exclusive":
                    directions(m_two, kind, "    ")
            lost = [r for r in near if int(r[f"n_gen_{kind}_excl_dR_lt_{t}"]) == 0]
            if lost:
                print(f"\n  [{kind} dR < {t}] matched by nearest but not exclusively "
                      f"(gen muon is closer to another displacedMuon inner track): {len(lost)}")
                for r in lost:
                    print(f"    {ident(r)} crossedIP={r['crossedIP'] or 'n/a'} nearest dR={r[f'nearest_{kind}_dR_ecal']} "
                          f"owner={r[f'nearest_{kind}_owner']} owner dR={r[f'nearest_{kind}_owner_dR_ecal']}")


if __name__ == "__main__":
    main()
