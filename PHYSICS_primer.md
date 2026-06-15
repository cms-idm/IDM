# IDM — physics primer (on-ramp)

A short, plain-language introduction to the physics of the inelastic-dark-matter (iDM)
analysis, for anyone joining the team. For how to set up and run the analysis, see
`SETUP_Run3.md`. Primary reference: the CMS displaced-dimuon iDM search,
[arXiv:2305.11649](https://arxiv.org/abs/2305.11649) (PRL 132, 041802).

## The model
We look for a **dark sector** with its own force: a broken `U(1)_D` gauge symmetry whose
force-carrier is a **dark photon `A'`** that mixes weakly with the ordinary photon (kinetic
mixing strength `eps`). The dark matter comes as **two nearly-degenerate states**, `chi1`
(lighter, stable, invisible) and `chi2` (heavier, unstable). "Inelastic" means the dark
photon couples the two *different* states (`A' chi1 chi2`) rather than a state to itself —
so producing dark matter naturally makes a `chi1`–`chi2` pair.

Two numbers drive everything:
- **`m1` = m(chi1)** — the dark-matter mass (we scan ~3–80 GeV; `mA' = 3 m1`).
- **`Delta = m(chi2) − m(chi1)`** — the small mass splitting (we scan `Delta/m1 ∈ {0.1, 0.4}`).

## The signal at the LHC
1. **Production:** `pp -> A' -> chi1 chi2`, recoiling against initial-state-radiation (ISR).
2. **Decay:** `chi2 -> chi1 + l+ l-` through a virtual `A'` (a 3-body decay), where `l = mu` or `e`.
3. **What we see:** `chi1` escapes the detector (→ **missing transverse momentum, MET**), the ISR
   shows up as a **hard jet**, and the lepton pair is the visible handle.

So the experimental signature is: **one hard ISR jet + large MET + a low-mass, opposite-sign
lepton pair**.

## Why the leptons are *soft* and *displaced* (the crux)
The splitting `Delta` sets the energy released in `chi2 -> chi1 l l`:
- The dilepton invariant mass is **bounded by `Delta`** (`m_ll < Delta`), so for small `Delta`
  the leptons are **soft** (low momentum) and **non-resonant** (no mass peak).
- Small `Delta` also makes `chi2` **long-lived** — it travels a measurable distance before
  decaying, so the lepton pair comes from a **displaced vertex** (up to metres from the beamline).
- The production rate scales steeply, roughly `~1/Delta^5`, so smaller splittings are rarer but
  more striking (more displaced).

These two facts — *soft* and *displaced* — are why standard prompt lepton reconstruction
struggles, and why we need special objects (below).

## Two channels, special objects
- **Di-muon:** displaced muons can fall outside the inner tracker's reach, so standard
  tracker-seeded muons fail. We use **DSA (displaced-standalone) muons**, reconstructed from
  the muon spectrometer alone with no requirement to point back to the collision. Prompt PF
  muons recover the small-displacement regime.
- **Di-electron:** we use **low-pT / displaced electrons** (the `lowPtGsfElectron` collection
  with an optimized BDT ID + energy regression, restored from AOD) alongside standard electrons.

## The search strategy in one line
Because the leptons can't trigger, we **trigger on the ISR jet + MET**; then we require a
**displaced, opposite-sign, low-mass lepton pair forming a good common vertex**, and we estimate
the background **from data** (an ABCD method on displacement vs. isolation). No excess →
limits on the production cross-section × branching fraction across the `(m1, Delta, ctau)` grid.

## Where this analysis sits
This is the **Run 3** extension, to **both muons and electrons**, of (a) the published Run-2
displaced-dimuon search and (b) the Run-2 displaced-electron ("iDMe") analysis. Sister analysis:
SIDM (self-interacting DM), which shares much of the muon/DSA tooling.
