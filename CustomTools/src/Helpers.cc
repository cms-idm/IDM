#include "iDMe/CustomTools/interface/Helpers.hh"

Helper::Helper() {}
Helper::~Helper() {}

bool Helper::JetID(const pat::Jet &jet, std::string year) {
    auto eta = jet.eta();
    auto neutEmFrac = jet.neutralEmEnergyFraction();
    auto neutHadFrac = jet.neutralHadronEnergyFraction();
    auto nConstit = jet.nConstituents();
    auto chargedHadFrac = jet.chargedHadronEnergyFraction();
    auto chargedMult = jet.chargedMultiplicity();
    auto neutMult = jet.neutralMultiplicity();
    auto muonFrac = jet.muonEnergyFraction();
    auto chEmFrac = jet.chargedEmEnergyFraction();

    bool passID = false;

    if ((year == "2022") || (year == "2023") || (year == "2024")) {
       // Run3 XYZ need to update!
       if (abs(eta) <= 2.6) {
            passID = (neutHadFrac < 0.9) && (neutEmFrac < 0.9) && (nConstit > 1) && (muonFrac < 0.8) && (chargedHadFrac > 0) && (chargedMult > 0) && (chEmFrac < 0.80);

        }
        else if ((abs(eta) > 2.6) && (abs(eta) <= 2.7)) {
            passID = (neutHadFrac < 0.9) && (neutEmFrac < 0.99) && (muonFrac < 0.8) && (chargedMult > 0) && (chEmFrac < 0.8);
        }
        else if ((abs(eta) > 2.7) && (abs(eta) <= 3.0)) {
            passID = (neutEmFrac < 0.99) && (neutEmFrac > 0.01) && (neutMult > 1);
        }
        else if ((abs(eta) >= 3.0) && (abs(eta) < 5.0)) {
            passID = (neutHadFrac > 0.2) && (neutEmFrac < 0.9) && (neutMult > 10);
        }
    }

    // Apply additional cuts to leading jet (see monojet analysis: https://arxiv.org/pdf/1703.01651.pdf)
    // Run3 not in current Run2 AN so ignoring
    //if (idx == 0) {
    //    passID = passID && (chargedHadFrac > 0.1) && (neutHadFrac < 0.8);
    //}

    return passID;
}
