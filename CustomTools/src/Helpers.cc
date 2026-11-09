#include "iDMe/CustomTools/interface/Helpers.hh"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include <cmath>
#include <string>

Helper::Helper() {}
Helper::~Helper() {}

bool Helper::JetID(const pat::Jet &jet, std::string year) {
    auto eta = jet.eta();
    auto absEta = std::abs(eta);

    auto neutEmFrac = jet.neutralEmEnergyFraction();
    auto neutHadFrac = jet.neutralHadronEnergyFraction();
    auto nConstit = jet.nConstituents();
    auto chargedHadFrac = jet.chargedHadronEnergyFraction();
    auto chargedMult = jet.chargedMultiplicity();
    auto neutMult = jet.neutralMultiplicity();
    auto muonFrac = jet.muonEnergyFraction();
    auto chEmFrac = jet.chargedEmEnergyFraction();

    bool passID = false;

    /*
      Accept Run 3 year labels such as:
        2022
        2022EE
        2022preEE / 2022PreEE
        2023
        2023BPix
        2024

      The previous exact check only accepted "2022", "2023", "2024".
      If the ntuplizer passed "2022EE", passID stayed false for every jet,
      which caused nPFJetAll > 0 but nPFJet == 0.
    */
    const bool isRun3 =
        (year.rfind("2022", 0) == 0) ||
        (year.rfind("2023", 0) == 0) ||
        (year.rfind("2024", 0) == 0);

    if (isRun3) {
        // Run 3 jet ID placeholder based on the existing Run 3-style cuts.
        // TODO: update with the final official Run 3 recommendation if needed.
        if (absEta <= 2.6) {
            passID =
                (neutHadFrac < 0.9) &&
                (neutEmFrac < 0.9) &&
                (nConstit > 1) &&
                (muonFrac < 0.8) &&
                (chargedHadFrac > 0) &&
                (chargedMult > 0) &&
                (chEmFrac < 0.80);
        }
        else if ((absEta > 2.6) && (absEta <= 2.7)) {
            passID =
                (neutHadFrac < 0.9) &&
                (neutEmFrac < 0.99) &&
                (muonFrac < 0.8) &&
                (chargedMult > 0) &&
                (chEmFrac < 0.8);
        }
        else if ((absEta > 2.7) && (absEta <= 3.0)) {
            passID =
                (neutEmFrac < 0.99) &&
                (neutEmFrac > 0.01) &&
                (neutMult > 1);
        }
        else if ((absEta >= 3.0) && (absEta < 5.0)) {
            passID =
                (neutHadFrac > 0.2) &&
                (neutEmFrac < 0.9) &&
                (neutMult > 10);
        }
    }

    // Apply additional cuts to leading jet (see monojet analysis: https://arxiv.org/pdf/1703.01651.pdf)
    // Run3 not in current Run2 AN so ignoring
    //if (idx == 0) {
    //    passID = passID && (chargedHadFrac > 0.1) && (neutHadFrac < 0.8);
    //}

    return passID;
}

// Approximate Phase-1 CMS tracker layer/disk positions in cm, used only to place a coarse
// SV-proxy point at the innermost valid hit -- not meant to reproduce exact module geometry.
namespace {
    constexpr float kBPixRadiusCm[4] = {2.9f, 6.8f, 10.9f, 16.0f};
    constexpr float kTIBRadiusCm[4]  = {25.5f, 33.9f, 41.9f, 49.8f};
    constexpr float kTOBRadiusCm[6]  = {60.8f, 69.2f, 78.0f, 86.8f, 96.5f, 108.0f};
    constexpr float kFPixDiskZCm[3]  = {29.1f, 39.6f, 51.6f};
    constexpr float kTIDDiskZCm[3]   = {74.0f, 84.0f, 94.0f};
    constexpr float kTECDiskZCm[9]   = {130.f, 145.f, 166.f, 183.f, 210.f, 232.f, 254.f, 272.f, 290.f};
}

HitBasedSVProxy Helper::EstimateSVProxyFromInnermostHit(const reco::Track &track) {
    HitBasedSVProxy proxy;

    // theta is conserved along the helix (pT and pz are each separately conserved in a
    // uniform solenoidal field), so evaluating it at the PCA to the beam axis is fine.
    // phi is NOT conserved -- it precesses with arc length as the track curves -- so
    // track.phi() (defined at the PCA to the beam axis) would be rotated relative to the
    // true azimuth at the innermost hit, worse for lower-pT/more-displaced tracks. Use the
    // state at the innermost hit itself when available to avoid that bias.
    float theta = track.theta();
    float phi = track.phi();
    // innerOk() reads the TrackExtra, which some MINIAOD inputs (e.g. the iDMmu 2022EE
    // production) do not store; without it, keep the PCA direction above.
    if (track.extra().isNonnull() && track.extra().isAvailable() && track.innerOk()) {
        theta = track.innerMomentum().theta();
        phi = track.innerMomentum().phi();
    }

    const reco::HitPattern &hp = track.hitPattern();
    uint16_t firstValidHit = 0;
    bool foundValidHit = false;
    for (int i = 0; i < hp.numberOfAllHits(reco::HitPattern::TRACK_HITS); i++) {
        uint16_t hit = hp.getHitPattern(reco::HitPattern::TRACK_HITS, i);
        if (!reco::HitPattern::validHitFilter(hit)) continue;
        firstValidHit = hit;
        foundValidHit = true;
        break;
    }
    if (!foundValidHit) return proxy;

    uint32_t subDet = reco::HitPattern::getSubStructure(firstValidHit);
    uint32_t layer = reco::HitPattern::getLayer(firstValidHit); // 1-indexed

    float radius = -1.f; // set if the innermost hit is in a barrel layer
    float diskZ = -1.f;  // set if the innermost hit is in an endcap disk

    if (subDet == PixelSubdetector::PixelBarrel && layer >= 1 && layer <= 4) {
        radius = kBPixRadiusCm[layer - 1];
    } else if (subDet == StripSubdetector::TIB && layer >= 1 && layer <= 4) {
        radius = kTIBRadiusCm[layer - 1];
    } else if (subDet == StripSubdetector::TOB && layer >= 1 && layer <= 6) {
        radius = kTOBRadiusCm[layer - 1];
    } else if (subDet == PixelSubdetector::PixelEndcap && layer >= 1 && layer <= 3) {
        diskZ = kFPixDiskZCm[layer - 1];
    } else if (subDet == StripSubdetector::TID && layer >= 1 && layer <= 3) {
        diskZ = kTIDDiskZCm[layer - 1];
    } else if (subDet == StripSubdetector::TEC && layer >= 1 && layer <= 9) {
        diskZ = kTECDiskZCm[layer - 1];
    } else {
        return proxy; // unrecognized subdetector/layer combination
    }

    // Project the hardcoded layer radius/disk z-position onto a straight line from the
    // origin along the track's direction at the innermost state -- rho/z = tan(theta).
    float rho, z;
    if (radius > 0.f) {
        rho = radius;
        z = rho / std::tan(theta);
    } else {
        z = std::copysign(diskZ, std::cos(theta));
        rho = std::abs(z * std::tan(theta));
    }

    proxy.valid = true;
    proxy.x = rho * std::cos(phi);
    proxy.y = rho * std::sin(phi);
    proxy.z = z;
    proxy.vxy = rho;
    return proxy;
}
