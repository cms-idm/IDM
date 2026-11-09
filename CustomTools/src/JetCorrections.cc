#include "iDMe/CustomTools/interface/JetCorrections.hh"

JetCorrections::~JetCorrections() {}

JetCorrections::JetCorrections(edm::Handle<vector<reco::PFJet> > &recoJetHandle_, edm::Handle<reco::JetCorrector> &jetCorrectorHandle_, NtupleContainer &nt_, edm::Handle<reco::JetTagCollection> &bTagProbbHandle_, edm::Handle<reco::JetTagCollection> &bTagProbbbHandle_, edm::Handle<vector<reco::GsfElectron> > recoElectronHandle_, int year_, bool isData_) 
:
jetHandle(recoJetHandle_), corrHandle(jetCorrectorHandle_), eleHandle(recoElectronHandle_), nt(nt_), bTagProbbHandle(bTagProbbHandle_), bTagProbbbHandle(bTagProbbbHandle_)
{
    year = year_;
    isData = isData_;
}

bool JetCorrections::passJetID(const reco::PFJet &jet, int idx) {
    // Apply Jet ID (from https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVUL) for AK4 CHS Jets
    // REMOVING the optional "lepton-veto/LepVeto" requirements (thresholds on muon and EM energy fractions), since we will
    // want to make sure that *all* potential jets are far away from the lepton system in iDM signal
    auto eta = jet.eta();
    auto neutEmFrac = jet.neutralEmEnergyFraction();
    auto neutHadFrac = jet.neutralHadronEnergyFraction();
    auto nConstit = jet.nConstituents();
    auto chargedHadFrac = jet.chargedHadronEnergyFraction();
    auto chargedMult = jet.chargedMultiplicity();
    auto muonFrac = jet.muonEnergyFraction();
    auto neutMult = jet.neutralMultiplicity();
    auto chEmFrac = jet.chargedEmEnergyFraction();

    bool passID = false;

    // Run3 using Run2 AN JetID for 2017/2018 UL
    if ((year == 2022) || (year == 2023) || (year == 2024)) {
        if (abs(eta) <= 2.6) {
	    passID = (neutHadFrac < 0.9) && (neutEmFrac < 0.9) && (nConstit > 1) && (muonFrac < 0.8) && (chargedHadFrac > 0) && (chargedMult > 0)  && (chEmFrac < 0.8);
        }
        else if ((abs(eta) >= 2.6) && (abs(eta) <= 2.7)) {
 	    passID = (neutHadFrac < 0.9) && (neutEmFrac < 0.99) && (muonFrac < 0.8) && (chargedMult > 0) && (chEmFrac < 0.8);
        }
        else if ((abs(eta) >= 2.7) && (abs(eta) <= 3.0)) {
            passID = (neutEmFrac < 0.99) && (neutEmFrac > 0.01) && (neutMult > 1);
        }
        else if ((abs(eta) >= 3.0) && (abs(eta) < 5.0)) {
	    passID = (neutHadFrac > 0.2) && (neutEmFrac < 0.9) && (neutMult > 10);
        }
    }

    // Apply additional cuts to leading jet (see monojet analysis: https://arxiv.org/pdf/1703.01651.pdf)
    // Run3 - not currently in AN, so skipping
    //if (idx == 0) {
    //    passID = passID && (chargedHadFrac > 0.1) && (neutHadFrac < 0.8);
    //}

    return passID;
}

void JetCorrections::Correct(JME::JetResolution &resolution, JME::JetResolutionScaleFactor &resolution_sf, JetCorrectionUncertainty &jecUnc, edm::Handle<double> &rhoHandle, edm::Handle<vector<reco::GenJet> > &genJetHandle, reco::PFMET &PFMET) {
    // Apply Jet ID recommendations for UL anaylses (https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVUL)
    vector<bool> jetIDResults;
    int jet_idx = 0;
    for (auto & jet : *jetHandle) {
        bool jetIDResult = passJetID(jet,jet_idx);
        jetIDResults.push_back(jetIDResult);
        
        // Check if jet falls in bad HEM region (will only be relevant for subset of 2018 data)
        float pt = jet.pt(), eta = jet.eta(), phi = jet.phi();
        if ((pt > 30) && (eta > -3.0) && (eta < -1.4) && (phi > -1.57) && (phi < -0.87)) {
            nt.PFHEMFlag_ = true;
        }

        jet_idx++;
    }

    // Cross-clean jets against the regular electrons in the event
    // Don't need to cross-clean against low-pT electrons, as they do not go into jets
    for (size_t i = 0; i < jetHandle->size(); i++) {
        if (!jetIDResults[i]) continue;
        auto jet = jetHandle->at(i);
        for (size_t j = 0; j < eleHandle->size(); j++) {
            auto ele = eleHandle->at(j);
            if (reco::deltaR(jet,ele) < 0.4) {
                jetIDResults[i] = false;
            }
        }
    }

    // Apply JEC+JER to jets that pass ID
    // Need to re-order jets by pT after this
    // vector key: index that will be changed after re-ordering
    // vector value1: corrected pT, value2: original key to refer back to
    vector<std::pair<std::unique_ptr<reco::PFJet>, size_t>> correctedJets;
    vector<std::pair<std::unique_ptr<reco::PFJet>, size_t>> correctedJetsJESUp;
    vector<std::pair<std::unique_ptr<reco::PFJet>, size_t>> correctedJetsJESDown;
    vector<std::pair<std::unique_ptr<reco::PFJet>, size_t>> correctedJetsJERUp;
    vector<std::pair<std::unique_ptr<reco::PFJet>, size_t>> correctedJetsJERDown;

    // Initialize MET corrections due to JES/JER
    corr_METpx = PFMET.px(); corr_METpy = PFMET.py();
    corr_METpx_JESUp = PFMET.px(); corr_METpy_JESUp = PFMET.py();
    corr_METpx_JESDown = PFMET.px(); corr_METpy_JESDown = PFMET.py();
    corr_METpx_JERUp = PFMET.px(); corr_METpy_JERUp = PFMET.py();
    corr_METpx_JERDown = PFMET.px(); corr_METpy_JERDown = PFMET.py();

    for (size_t i =0; i < jetHandle->size(); i++) {
        reco::PFJet jet_i = jetHandle->at(i);

        // Get JEC and initialize corrected jet objects with JES/JER uncertainties
        double jec = corrHandle->correction(jet_i);
        std::unique_ptr<reco::PFJet> corr_jet_i(jet_i.clone());
        std::unique_ptr<reco::PFJet> corr_jet_i_jes_up(jet_i.clone());
        std::unique_ptr<reco::PFJet> corr_jet_i_jes_down(jet_i.clone());
        std::unique_ptr<reco::PFJet> corr_jet_i_jer_up(jet_i.clone());
        std::unique_ptr<reco::PFJet> corr_jet_i_jer_down(jet_i.clone());
        // For MET corrections due to jet smearing
        std::unique_ptr<reco::PFJet> corr_jet_i_jes_only(jet_i.clone());

        // Apply JEC to all objects
        corr_jet_i->scaleEnergy(jec);
        corr_jet_i_jes_only->scaleEnergy(jec);
        corr_jet_i_jes_up->scaleEnergy(jec);
        corr_jet_i_jes_down->scaleEnergy(jec);
        corr_jet_i_jer_up->scaleEnergy(jec);
        corr_jet_i_jer_down->scaleEnergy(jec);

        // Evaluate JEC uncertainties
        jecUnc.setJetEta(corr_jet_i->eta());
        jecUnc.setJetPt(corr_jet_i->pt());
        double uncUp = jecUnc.getUncertainty(true); // true: Up direction
        jecUnc.setJetEta(corr_jet_i->eta());
        jecUnc.setJetPt(corr_jet_i->pt());
        double uncDown = jecUnc.getUncertainty(false); // false: Down direction

        // Apply JES uncertainties
        double jesUp = 1 + uncUp;
        double jesDown = 1 - uncDown;
        corr_jet_i_jes_up->scaleEnergy(jesUp);
        corr_jet_i_jes_down->scaleEnergy(jesDown);

        // Calculate jet energy resolution smearing factors https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution
        // Should only be != 1 if running on MC
        double smearFactor = 1., smearFactor_up = 1., smearFactor_down = 1.;
        if (!isData) {
            JME::JetParameters jparams;
            jparams.setJetPt(corr_jet_i->pt()).setJetEta(corr_jet_i->eta()).setRho(*rhoHandle);
            double jet_resolution = resolution.getResolution(jparams);
            double jer_sf = resolution_sf.getScaleFactor(jparams, Variation::NOMINAL);
            double jer_sf_up = resolution_sf.getScaleFactor(jparams, Variation::UP);
            double jer_sf_down = resolution_sf.getScaleFactor(jparams, Variation::DOWN);

            // Try matching with gen jet
            double min_dR = std::numeric_limits<double>::infinity();
            const reco::GenJet* matched_genJet = nullptr;
            for (const auto& genJet: *genJetHandle) {
                double dR = deltaR(genJet, *corr_jet_i);
                if (dR > min_dR) {
                    continue;
                }
                if (dR < 0.2) { // 0.2 = R_cone / 2
                    double dPt = std::abs(genJet.pt() - corr_jet_i->pt());
                    if (abs(dPt) > 3*jet_resolution*corr_jet_i->pt()) {
                        continue;
                    }
                    matched_genJet = &genJet;
                }
            }

            // Applying "hybrid" smearing procedure (https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution)
            if (matched_genJet) {
                double dPt = corr_jet_i->pt() - matched_genJet->pt();
                smearFactor = 1 + (jer_sf - 1.) * dPt / corr_jet_i->pt();
                smearFactor_up = 1 + (jer_sf_up - 1.) * dPt / corr_jet_i->pt();
                smearFactor_down = 1 + (jer_sf_down - 1.) * dPt / corr_jet_i->pt();
            }
            if (!matched_genJet && jer_sf > 1) { 
                double sigma = jet_resolution * std::sqrt(jer_sf * jer_sf - 1);
                std::normal_distribution<> d(0, sigma);
                smearFactor = 1. + d(m_random_generator);
            }
            if (!matched_genJet && jer_sf_up > 1) {
                double sigma_up = jet_resolution * std::sqrt(jer_sf_up * jer_sf_up - 1);
                std::normal_distribution<> d_up(0, sigma_up);
                smearFactor_up = 1. + d_up(m_random_generator);
            }
            if (!matched_genJet && jer_sf_down > 1) {
                double sigma_down = jet_resolution * std::sqrt(jer_sf_down * jer_sf_down - 1);
                std::normal_distribution<> d_down(0, sigma_down);
                smearFactor_down = 1. + d_down(m_random_generator);
            }

            if (corr_jet_i->energy() * smearFactor < 0.01) {
                double newSmearFactor = 0.01 / corr_jet_i->energy();
                smearFactor = newSmearFactor;
            }
            if (corr_jet_i->energy() * smearFactor_up < 0.01) {
                double newSmearFactor = 0.01 / corr_jet_i->energy();
                smearFactor_up = newSmearFactor;
            }
            if (corr_jet_i->energy() * smearFactor_down < 0.01) {
                double newSmearFactor = 0.01 / corr_jet_i->energy();
                smearFactor_down = newSmearFactor;
            }
        }

        // Apply smearing
        corr_jet_i->scaleEnergy(smearFactor);
        corr_jet_i_jes_up->scaleEnergy(smearFactor);
        corr_jet_i_jes_down->scaleEnergy(smearFactor);
        corr_jet_i_jer_up->scaleEnergy(smearFactor_up);
        corr_jet_i_jer_down->scaleEnergy(smearFactor_down);

        // Apply MET corrections due to smearing (JEC corrections already present in Type-I MET)
        corr_METpx -= (corr_jet_i->px() - corr_jet_i_jes_only->px());
        corr_METpy -= (corr_jet_i->py() - corr_jet_i_jes_only->py());
        corr_METpx_JESUp -= (corr_jet_i_jes_up->px() - corr_jet_i_jes_only->px());
        corr_METpy_JESUp -= (corr_jet_i_jes_up->py() - corr_jet_i_jes_only->py());
        corr_METpx_JESDown -= (corr_jet_i_jes_down->px() - corr_jet_i_jes_only->px());
        corr_METpy_JESDown -= (corr_jet_i_jes_down->py() - corr_jet_i_jes_only->py());
        corr_METpx_JERUp -= (corr_jet_i_jer_up->px() - corr_jet_i_jes_only->px());
        corr_METpy_JERUp -= (corr_jet_i_jer_up->py() - corr_jet_i_jes_only->py());
        corr_METpx_JERDown -= (corr_jet_i_jer_down->px() - corr_jet_i_jes_only->px());
        corr_METpy_JERDown -= (corr_jet_i_jer_down->py() - corr_jet_i_jes_only->py());

        correctedJets.push_back(std::make_pair(std::move(corr_jet_i), i));
        correctedJetsJESUp.push_back(std::make_pair(std::move(corr_jet_i_jes_up), i));
        correctedJetsJESDown.push_back(std::make_pair(std::move(corr_jet_i_jes_down), i));
        correctedJetsJERUp.push_back(std::make_pair(std::move(corr_jet_i_jer_up), i));
        correctedJetsJERDown.push_back(std::make_pair(std::move(corr_jet_i_jer_down), i));
    } // end jet loop

    // Before sorting the jet collections by pT, calculate MET corrections for each as well
    nt.PFMETSmearingOnlyPt_ = std::sqrt(corr_METpx*corr_METpx + corr_METpy*corr_METpy);
    nt.PFMETSmearingOnlyPhi_ = atan2(corr_METpy, corr_METpx);
    nt.PFMETCorrectedPt_ = std::sqrt(corr_METpx*corr_METpx + corr_METpy*corr_METpy);
    nt.PFMETCorrectedPhi_ = atan2(corr_METpy, corr_METpx);
    nt.PFMETJESUpPt_ = std::sqrt(corr_METpx_JESUp*corr_METpx_JESUp + corr_METpy_JESUp*corr_METpy_JESUp);
    nt.PFMETJESUpPhi_ = atan2(corr_METpy_JESUp, corr_METpx_JESUp);
    nt.PFMETJESDownPt_ = std::sqrt(corr_METpx_JESDown*corr_METpx_JESDown + corr_METpy_JESDown*corr_METpy_JESDown);
    nt.PFMETJESDownPhi_ = atan2(corr_METpy_JESDown, corr_METpx_JESDown);
    nt.PFMETJERUpPt_ = std::sqrt(corr_METpx_JERUp*corr_METpx_JERUp + corr_METpy_JERUp*corr_METpy_JERUp);
    nt.PFMETJERUpPhi_ = atan2(corr_METpy_JERUp, corr_METpx_JERUp);
    nt.PFMETJERDownPt_ = std::sqrt(corr_METpx_JERDown*corr_METpx_JERDown + corr_METpy_JERDown*corr_METpy_JERDown);
    nt.PFMETJERDownPhi_ = atan2(corr_METpy_JERDown, corr_METpx_JERDown);

    // Sort jets by decreasing pT
    auto reverseSortJets = [](const std::pair<std::unique_ptr<reco::PFJet>, size_t> &a, const std::pair<std::unique_ptr<reco::PFJet>, size_t> &b) {
        return (a.first->pt() > b.first->pt());
    };
    sort(correctedJets.begin(), correctedJets.end(), reverseSortJets);
    sort(correctedJetsJESUp.begin(), correctedJetsJESUp.end(), reverseSortJets);
    sort(correctedJetsJESDown.begin(), correctedJetsJESDown.end(), reverseSortJets);
    sort(correctedJetsJERUp.begin(), correctedJetsJERUp.end(), reverseSortJets);
    sort(correctedJetsJERDown.begin(), correctedJetsJERDown.end(), reverseSortJets);

    // Get 10 top leading jets info, sorted by corrected pT
    // Only pick jets that have passed loose ID and cross-cleaning
    nt.PFNJet_ = jetHandle->size(); 
    nt.PFNPassIDJet_ = 0;
    nt.PFNHighPtJet_ = 0;

    for (size_t i = 0; i < correctedJets.size(); i++) {
        size_t index = correctedJets[i].second;
        auto & jet_i = *(correctedJets[i].first);

        // Exclude jets that didn't pass ID above (ID checked against uncorrected jets)
        if (!jetIDResults[index]) continue;
        nt.PFNPassIDJet_++;

        if (jet_i.pt() > 30) {
            nt.PFNHighPtJet_++;
            nt.PFJetCorrectedPt_.push_back(jet_i.pt());
            nt.PFJetCorrectedEta_.push_back(jet_i.eta());
            nt.PFJetCorrectedPhi_.push_back(jet_i.phi());
            // TODO: figure out how BtagProbb(b) collections actually behave
            // FIXME this might be problematic with the jet corrections, keep in mind
            if (bTagProbbHandle->size() > i && bTagProbbbHandle->size() > i)
                nt.PFJetCorrectedBTag_.push_back((*bTagProbbHandle)[index].second +\
                (*bTagProbbbHandle)[index].second);
            else 
                nt.PFJetCorrectedBTag_.push_back(-9999);
            nt.PFJetCorrectedCHEF_.push_back(jet_i.chargedHadronEnergyFraction());
            nt.PFJetCorrectedNHEF_.push_back(jet_i.neutralHadronEnergyFraction());
            nt.PFJetCorrectedCEEF_.push_back(jet_i.chargedEmEnergyFraction());
            nt.PFJetCorrectedNEEF_.push_back(jet_i.neutralEmEnergyFraction());
            nt.PFJetCorrectedNumDaughters_.push_back(jet_i.numberOfDaughters());
            nt.PFJetCorrectedChargedMultiplicity_.push_back(jet_i.chargedMultiplicity());
        
            // Add pt, eta, phi information for syst. uncert. collections
            jet_i = *(correctedJetsJESUp[i].first);
            nt.PFJetCorrectedJESUpPt_.push_back(jet_i.pt());
            nt.PFJetCorrectedJESUpEta_.push_back(jet_i.eta());
            nt.PFJetCorrectedJESUpPhi_.push_back(jet_i.phi());
        
            jet_i = *(correctedJetsJESDown[i].first);
            nt.PFJetCorrectedJESDownPt_.push_back(jet_i.pt());
            nt.PFJetCorrectedJESDownEta_.push_back(jet_i.eta());
            nt.PFJetCorrectedJESDownPhi_.push_back(jet_i.phi());

            jet_i = *(correctedJetsJERUp[i].first);
            nt.PFJetCorrectedJERUpPt_.push_back(jet_i.pt());
            nt.PFJetCorrectedJERUpEta_.push_back(jet_i.eta());
            nt.PFJetCorrectedJERUpPhi_.push_back(jet_i.phi());

            jet_i = *(correctedJetsJERDown[i].first);
            nt.PFJetCorrectedJERDownPt_.push_back(jet_i.pt());
            nt.PFJetCorrectedJERDownEta_.push_back(jet_i.eta());
            nt.PFJetCorrectedJERDownPhi_.push_back(jet_i.phi());

            // Also add uncorrected jet info for completeness
            auto jet_ii = jetHandle->at(index);
            nt.PFJetPt_.push_back(jet_ii.pt());
            nt.PFJetEta_.push_back(jet_ii.eta());
            nt.PFJetPhi_.push_back(jet_ii.phi());
        }
    }
}
