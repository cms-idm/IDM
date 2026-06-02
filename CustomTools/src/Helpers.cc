#include "iDMe/CustomTools/interface/Helpers.hh"

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
        // Run3 using 2017/18 UL criteria from Run2 AN
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
    // Apply additional cuts to leading jet (see monojet analysis: https://arxiv.org/pdf/1703.01651.pdf)
    // Run3 not in current Run2 AN so ignoring
    //if (idx == 0) {
    //    passID = passID && (chargedHadFrac > 0.1) && (neutHadFrac < 0.8);
    //}

    return passID;
}
