#include "iDMe/CustomTools/interface/NtupleContainerV2.hh"

#include <array>
#include <cctype>
#include <stdexcept>

NtupleContainerV2::NtupleContainerV2() {}

NtupleContainerV2::~NtupleContainerV2() {}

namespace {
// Flat components of a LorentzVector; eta, phi = -999 for zero-pt (absent or sentinel) entries.
inline void flatComponents(const math::XYZTLorentzVector& v, float& pt, float& eta, float& phi, float& mass) {
    pt = v.Pt();
    mass = v.M();
    if (v.Pt() > 0) { eta = v.Eta(); phi = v.Phi(); }
    else { eta = -999.f; phi = -999.f; }
}
// Scalar availability flag outside the collection prefix, so coffea MySchema keeps it as an event-level
// field: "RecoDisplacedMuon_outerTrack" -> "RecoDisplacedMuonOuterTrackAvailable".
inline std::string camelJoin(const std::string& name) {
    std::string out;
    bool upper = false;
    for (const char c : name) {
        if (c == '_') { upper = true; continue; }
        out += upper ? static_cast<char>(std::toupper(static_cast<unsigned char>(c))) : c;
        upper = false;
    }
    return out;
}
inline std::string availableName(const std::string& name) { return camelJoin(name) + "Available"; }
// LorentzVector branches are split into fCoordinates sub-branches, which coffea MySchema can't zip into a
// collection: the p4 itself goes outside the collection prefix, "X_y_p4" -> "XYP4", "X_yP4" -> "XYP4".
inline std::string p4BranchName(const std::string& name) {
    const bool snake = name.size() >= 3 && name.compare(name.size() - 3, 3, "_p4") == 0;
    const bool camel = !snake && name.size() >= 2 && name.compare(name.size() - 2, 2, "P4") == 0;
    if (!snake && !camel)
        throw std::runtime_error("NtupleContainerV2::branchP4: name must end in _p4 or P4: " + name);
    return camelJoin(name.substr(0, name.size() - (snake ? 3 : 2))) + "P4";
}
// Flat components stay in the collection: "X_p4" -> ("X_pt", ...); "XP4" -> ("XPt", ...).
inline std::array<std::string, 4> flatNames(const std::string& name) {
    const bool snake = name.size() >= 3 && name.compare(name.size() - 3, 3, "_p4") == 0;
    const bool camel = !snake && name.size() >= 2 && name.compare(name.size() - 2, 2, "P4") == 0;
    if (!snake && !camel)
        throw std::runtime_error("NtupleContainerV2::branchP4: name must end in _p4 or P4: " + name);
    const std::string base = name.substr(0, name.size() - (snake ? 3 : 2));
    if (snake) return {{base + "_pt", base + "_eta", base + "_phi", base + "_mass"}};
    return {{base + "Pt", base + "Eta", base + "Phi", base + "Mass"}};
}
}  // namespace

void NtupleContainerV2::branchP4(const std::string& name, vector<math::XYZTLorentzVector>* p4, bool flat) {
    outT->Branch(p4BranchName(name).c_str(), p4);
    if (!flat) return;
    flatP4Vectors_.push_back(FlatP4Vector{p4, {}, {}, {}, {}});
    auto& f = flatP4Vectors_.back();
    const auto names = flatNames(name);
    outT->Branch(names[0].c_str(), &f.pt);
    outT->Branch(names[1].c_str(), &f.eta);
    outT->Branch(names[2].c_str(), &f.phi);
    outT->Branch(names[3].c_str(), &f.mass);
}

void NtupleContainerV2::branchP4(const std::string& name, math::XYZTLorentzVector* p4, bool flat) {
    outT->Branch(p4BranchName(name).c_str(), p4);
    if (!flat) return;
    flatP4Scalars_.push_back(FlatP4Scalar{p4, -999.f, -999.f, -999.f, -999.f});
    auto& f = flatP4Scalars_.back();
    const auto names = flatNames(name);
    outT->Branch(names[0].c_str(), &f.pt);
    outT->Branch(names[1].c_str(), &f.eta);
    outT->Branch(names[2].c_str(), &f.phi);
    outT->Branch(names[3].c_str(), &f.mass);
}

void NtupleContainerV2::FillFlatP4() {
    for (auto& f : flatP4Vectors_) {
        const size_t n = f.p4->size();
        f.pt.resize(n); f.eta.resize(n); f.phi.resize(n); f.mass.resize(n);
        for (size_t i = 0; i < n; ++i)
            flatComponents((*f.p4)[i], f.pt[i], f.eta[i], f.phi[i], f.mass[i]);
    }
    for (auto& f : flatP4Scalars_)
        flatComponents(*f.p4, f.pt, f.eta, f.phi, f.mass);
}

void NtupleContainerV2::SetTree(TTree *tree) { outT = tree; }
void NtupleContainerV2::SetSlimTree(TTree *tree) { outT_slim = tree; }

void NtupleContainerV2::CreateTreeBranches() {

    // Reco information
    outT->Branch("trigFired",&fired_);
    outT->Branch("eventNum", &eventNum_);
    outT->Branch("lumiSec", &lumiSec_);
    outT->Branch("runNum", &runNum_);
    for (int i = 0; i < numTrigs_; i++) {
        TString branchName = TString::Format("trig_%s", trigNames_[i].c_str());
        outT->Branch(branchName,&trigPassed_[i],branchName+"/O");
    }

    // MET Filters
    outT->Branch("METFiltersFailBits",&METFiltersFailBits_);

    // Muons
    outT->Branch("nMuon",&nMuon_);
    outT->Branch("Muon_patMuonIdx",&recoMuonPatIdx_);
    outT->Branch("Muon_pt",&recoMuonPt_);
    outT->Branch("Muon_eta",&recoMuonEta_);
    outT->Branch("Muon_phi",&recoMuonPhi_);
    outT->Branch("Muon_e",&recoMuonEnergy_);
    outT->Branch("Muon_charge",&recoMuonCharge_);
    outT->Branch("Muon_IDcutLoose",&recoMuonIDcutBasedLoose_);
    outT->Branch("Muon_IDcutMedium",&recoMuonIDcutBasedMedium_);
    outT->Branch("Muon_IDcutMediumPrompt",&recoMuonIDcutBasedMediumPrompt_);
    outT->Branch("Muon_IDcutTight",&recoMuonIDcutBasedTight_);
    outT->Branch("Muon_isPFMuon",&recoMuonIsPFMuon_);
    outT->Branch("Muon_isGlobalMuon",&recoMuonIsGlobalMuon_);
    outT->Branch("Muon_isStandAloneMuon",&recoMuonIsStandAloneMuon_);

    auto branchDisplacedTrack = [this](const std::string& name, DisplacedTrackFields& fields) {
        outT->Branch(availableName(name).c_str(), &fields.available);
        outT->Branch(("n" + name).c_str(), &fields.n);
        branchP4((name + "_p4").c_str(), &fields.p4);
        outT->Branch((name + "_valid").c_str(), &fields.valid);
        outT->Branch((name + "_charge").c_str(), &fields.charge);
        outT->Branch((name + "_extraAvailable").c_str(), &fields.extraAvailable);
        outT->Branch((name + "_ptError").c_str(), &fields.ptError);
        outT->Branch((name + "_vxy").c_str(), &fields.vxy);
        outT->Branch((name + "_vz").c_str(), &fields.vz);
        outT->Branch((name + "_dxy").c_str(), &fields.dxy);
        outT->Branch((name + "_dz").c_str(), &fields.dz);
        outT->Branch((name + "_normalizedChi2").c_str(), &fields.normalizedChi2);
        outT->Branch((name + "_nMuonHits").c_str(), &fields.nMuonHits);
        outT->Branch((name + "_nCSCHits").c_str(), &fields.nCSCHits);
        outT->Branch((name + "_nDTHits").c_str(), &fields.nDTHits);
        outT->Branch((name + "_nTrackerHits").c_str(), &fields.nTrackerHits);
        outT->Branch((name + "_nPixelHits").c_str(), &fields.nPixelHits);
        outT->Branch((name + "_nStripHits").c_str(), &fields.nStripHits);
    };
    auto branchDisplacedMuon = [this, &branchDisplacedTrack](
        const std::string& name, DisplacedMuonFields& fields) {
        outT->Branch(availableName(name).c_str(), &fields.available);
        outT->Branch(("n" + name).c_str(), &fields.n);
        branchP4((name + "_p4").c_str(), &fields.p4);
        outT->Branch((name + "_charge").c_str(), &fields.charge);
        outT->Branch((name + "_isStandAlone").c_str(), &fields.isStandAlone);
        outT->Branch((name + "_isTracker").c_str(), &fields.isTracker);
        outT->Branch((name + "_isGlobal").c_str(), &fields.isGlobal);
        outT->Branch((name + "_isPF").c_str(), &fields.isPF);
        outT->Branch((name + "_nMatchedStations").c_str(), &fields.nMatchedStations);
        outT->Branch((name + "_timeValid").c_str(), &fields.timeValid);
        outT->Branch((name + "_timeAtIpInOut").c_str(), &fields.timeAtIpInOut);
        branchDisplacedTrack(name + "_outerTrack", fields.outer);
        branchDisplacedTrack(name + "_innerTrack", fields.inner);
        branchDisplacedTrack(name + "_globalTrack", fields.global);
    };
    branchDisplacedTrack("DisplacedTrack", displacedTrack_);
    branchDisplacedTrack("DisplacedGlobalTrack", displacedGlobalTrack_);
    branchDisplacedMuon("SlimmedMuon", slimmedMuon_);
    branchDisplacedMuon("RecoDisplacedMuon", recoDisplacedMuon_);
    outT->Branch("RecoDisplacedMuon_originalIndex", &recoDisplacedMuonOriginalIndex_);
    outT->Branch("nRecoDisplacedMuonRaw", &nRecoDisplacedMuonRaw_);
    branchDisplacedMuon("SlimmedDisplacedMuon", slimmedDisplacedMuon_);

    auto branchDisplacedTrackPropagation = [this](
        const std::string& name, DisplacedTrackFields& fields) {
        auto branchStation = [this](const std::string& prefix,
                                   DisplacedTrackPropagationFields& station) {
            outT->Branch((prefix + "Status").c_str(), &station.status);
            branchP4((prefix + "P4").c_str(), &station.p4);
            outT->Branch((prefix + "PositionEta").c_str(), &station.positionEta);
            outT->Branch((prefix + "PositionPhi").c_str(), &station.positionPhi);
        };
        branchStation(name + "_propSt1", fields.propSt1);
        branchStation(name + "_propSt2", fields.propSt2);
        branchStation(name + "_propSt3", fields.propSt3);
        branchStation(name + "_propSt4", fields.propSt4);
    };
    branchDisplacedTrackPropagation("DisplacedTrack", displacedTrack_);
    branchDisplacedTrackPropagation("DisplacedGlobalTrack", displacedGlobalTrack_);
    auto branchDisplacedMuonPropagation = [&branchDisplacedTrackPropagation](
        const std::string& name, DisplacedMuonFields& fields) {
        branchDisplacedTrackPropagation(name + "_outerTrack", fields.outer);
        branchDisplacedTrackPropagation(name + "_innerTrack", fields.inner);
        branchDisplacedTrackPropagation(name + "_globalTrack", fields.global);
    };
    branchDisplacedMuonPropagation("SlimmedMuon", slimmedMuon_);
    branchDisplacedMuonPropagation("RecoDisplacedMuon", recoDisplacedMuon_);
    branchDisplacedMuonPropagation("SlimmedDisplacedMuon", slimmedDisplacedMuon_);

    // All PF muons and successful Station-1/2/3/4 propagated PF-muon views.
    outT->Branch("nPFMuon",&nPFMuon_);
    outT->Branch("PFMuon_patMuonIdx",&pfMuonPatIdx_);
    branchP4("PFMuon_p4",&pfMuonP4_);
    outT->Branch("PFMuon_charge",&pfMuonCharge_);
    outT->Branch("PFMuon_IDcutLoose",&pfMuonIDcutBasedLoose_);
    outT->Branch("PFMuon_IDcutMedium",&pfMuonIDcutBasedMedium_);
    outT->Branch("PFMuon_IDcutMediumPrompt",&pfMuonIDcutBasedMediumPrompt_);
    outT->Branch("PFMuon_IDcutTight",&pfMuonIDcutBasedTight_);
    outT->Branch("PFMuon_isGlobalMuon",&pfMuonIsGlobalMuon_);
    outT->Branch("PFMuon_isStandAloneMuon",&pfMuonIsStandAloneMuon_);
    outT->Branch("PFMuon_propagationTrackType",&pfMuonPropagationTrackType_);
    outT->Branch("PFMuon_trkNumValidMuonHits",&pfMuonTrkNumValidMuonHits_);
    outT->Branch("PFMuon_trkNumValidTrackerHits",&pfMuonTrkNumValidTrackerHits_);
    outT->Branch("PFMuon_trkNumValidPixelHits",&pfMuonTrkNumValidPixelHits_);
    outT->Branch("PFMuon_trkNumValidStripHits",&pfMuonTrkNumValidStripHits_);
    outT->Branch("PFMuon_numMatchedStations",&pfMuonNumMatchedStations_);
    outT->Branch("PFMuon_trkNumMuonStations",&pfMuonTrkNumMuonStations_);
    outT->Branch("PFMuon_trkNumCSCHits",&pfMuonTrkNumCSCHits_);
    outT->Branch("PFMuon_trkNumDTHits",&pfMuonTrkNumDTHits_);
    outT->Branch("PFMuon_trkPt",&pfMuonTrkPt_);
    outT->Branch("PFMuon_trkEta",&pfMuonTrkEta_);
    outT->Branch("PFMuon_trkEtaError",&pfMuonTrkEtaError_);
    outT->Branch("PFMuon_trkPhi",&pfMuonTrkPhi_);
    outT->Branch("PFMuon_trkPhiError",&pfMuonTrkPhiError_);
    outT->Branch("PFMuon_trkOuterEta",&pfMuonTrkOuterEta_);
    outT->Branch("PFMuon_trkOuterPhi",&pfMuonTrkOuterPhi_);
    outT->Branch("PFMuon_trkNormalizedChi2",&pfMuonTrkNormalizedChi2_);
    outT->Branch("PFMuon_trkPtError",&pfMuonTrkPtError_);
    outT->Branch("PFMuon_trkVxy",&pfMuonTrkVxy_);
    outT->Branch("PFMuon_trkVz",&pfMuonTrkVz_);
    outT->Branch("PFMuon_trkDxy",&pfMuonTrkDxy_);
    outT->Branch("PFMuon_trkDxyError",&pfMuonTrkDxyError_);
    outT->Branch("PFMuon_trkDz",&pfMuonTrkDz_);
    outT->Branch("PFMuon_trkDzError",&pfMuonTrkDzError_);
    outT->Branch("PFMuon_trkProb",&pfMuonTrkProb_);
    outT->Branch("PFMuon_propSt1Status",&pfMuonPropSt1Status_);
    outT->Branch("PFMuon_propSt1Idx",&pfMuonPropSt1Idx_);
    outT->Branch("PFMuon_propSt2Status",&pfMuonPropSt2Status_);
    outT->Branch("PFMuon_propSt2Idx",&pfMuonPropSt2Idx_);

    outT->Branch("nPropPFMuonSt1",&nPropPFMuonSt1_);
    outT->Branch("PropPFMuonSt1_pfMuonIdx",&propPFMuonSt1PFMuonIdx_);
    branchP4("PropPFMuonSt1_p4",&propPFMuonSt1P4_);
    outT->Branch("PropPFMuonSt1_positionEta",&propPFMuonSt1PositionEta_);
    outT->Branch("PropPFMuonSt1_positionPhi",&propPFMuonSt1PositionPhi_);

    outT->Branch("nPropPFMuonSt2",&nPropPFMuonSt2_);
    outT->Branch("PropPFMuonSt2_pfMuonIdx",&propPFMuonSt2PFMuonIdx_);
    branchP4("PropPFMuonSt2_p4",&propPFMuonSt2P4_);
    outT->Branch("PropPFMuonSt2_positionEta",&propPFMuonSt2PositionEta_);
    outT->Branch("PropPFMuonSt2_positionPhi",&propPFMuonSt2PositionPhi_);

    outT->Branch("PFMuon_propSt3Status",&pfMuonPropSt3Status_);
    outT->Branch("PFMuon_propSt3Idx",&pfMuonPropSt3Idx_);
    outT->Branch("nPropPFMuonSt3",&nPropPFMuonSt3_);
    outT->Branch("PropPFMuonSt3_pfMuonIdx",&propPFMuonSt3PFMuonIdx_);
    branchP4("PropPFMuonSt3_p4",&propPFMuonSt3P4_);
    outT->Branch("PropPFMuonSt3_positionEta",&propPFMuonSt3PositionEta_);
    outT->Branch("PropPFMuonSt3_positionPhi",&propPFMuonSt3PositionPhi_);
    outT->Branch("PFMuon_propSt4Status",&pfMuonPropSt4Status_);
    outT->Branch("PFMuon_propSt4Idx",&pfMuonPropSt4Idx_);
    outT->Branch("nPropPFMuonSt4",&nPropPFMuonSt4_);
    outT->Branch("PropPFMuonSt4_pfMuonIdx",&propPFMuonSt4PFMuonIdx_);
    branchP4("PropPFMuonSt4_p4",&propPFMuonSt4P4_);
    outT->Branch("PropPFMuonSt4_positionEta",&propPFMuonSt4PositionEta_);
    outT->Branch("PropPFMuonSt4_positionPhi",&propPFMuonSt4PositionPhi_);

    // Normal Electrons
    outT->Branch("nElectron",&nElectronDefault_);
    outT->Branch("Electron_pt",&recoElectronPt_);
    outT->Branch("Electron_eta",&recoElectronEta_);
    outT->Branch("Electron_etaErr",&recoElectronEtaError_);
    outT->Branch("Electron_phi",&recoElectronPhi_);
    outT->Branch("Electron_phiErr",&recoElectronPhiError_);
    outT->Branch("Electron_IDcutVeto",&recoElectronID_cutVeto_);
    outT->Branch("Electron_IDcutLoose",&recoElectronID_cutLoose_);
    outT->Branch("Electron_IDcutMed",&recoElectronID_cutMed_);
    outT->Branch("Electron_IDcutTight",&recoElectronID_cutTight_);
    outT->Branch("Electron_IDcutVetoInt",&recoElectronID_cutVetoInt_);
    outT->Branch("Electron_IDcutLooseInt",&recoElectronID_cutLooseInt_);
    outT->Branch("Electron_IDcutMedInt",&recoElectronID_cutMedInt_);
    outT->Branch("Electron_IDcutTightInt",&recoElectronID_cutTightInt_);
    outT->Branch("Electron_IDmvaIso90",&recoElectronID_mvaIso90_);
    outT->Branch("Electron_IDmvaIso80",&recoElectronID_mvaIso80_);
    outT->Branch("Electron_IDmvaIsoLoose",&recoElectronID_mvaIsoLoose_);
    outT->Branch("Electron_IDmva90",&recoElectronID_mva90_);
    outT->Branch("Electron_IDmva80",&recoElectronID_mva80_);
    outT->Branch("Electron_IDmvaLoose",&recoElectronID_mvaLoose_);
    outT->Branch("Electron_angRes",&recoElectronAngularRes_);
    outT->Branch("Electron_e",&recoElectronE_);
    outT->Branch("Electron_vxy",&recoElectronVxy_);
    outT->Branch("Electron_vz",&recoElectronVz_);
    outT->Branch("Electron_svProxyValid",&recoElectronSVProxyValid_);
    outT->Branch("Electron_svProxyX",&recoElectronSVProxyX_);
    outT->Branch("Electron_svProxyY",&recoElectronSVProxyY_);
    outT->Branch("Electron_svProxyZ",&recoElectronSVProxyZ_);
    outT->Branch("Electron_svProxyVxy",&recoElectronSVProxyVxy_);
    outT->Branch("Electron_dxy",&recoElectronDxy_);
    outT->Branch("Electron_dxyErr",&recoElectronDxyError_);
    outT->Branch("Electron_dz",&recoElectronDz_);
    outT->Branch("Electron_dzErr",&recoElectronDzError_);
    outT->Branch("Electron_trkChi2",&recoElectronTrkChi2_);
    outT->Branch("Electron_trkIso",&recoElectronTrkIso_);
    outT->Branch("Electron_trkRelIso",&recoElectronTrkRelIso_);
    outT->Branch("Electron_calIso",&recoElectronCaloIso_);
    outT->Branch("Electron_calRelIso",&recoElectronCaloRelIso_);
    outT->Branch("Electron_PFIso",&recoElectronPFIso_);
    outT->Branch("Electron_PFRelIso",&recoElectronPFRelIso_);
    outT->Branch("Electron_miniIso",&recoElectronMiniIso_);
    outT->Branch("Electron_miniRelIso",&recoElectronMiniRelIso_);
    outT->Branch("Electron_PFIsoEleCorr",&recoElectronPFIsoEleCorr_);
    outT->Branch("Electron_PFRelIsoEleCorr",&recoElectronPFRelIsoEleCorr_);
    outT->Branch("Electron_miniIsoEleCorr",&recoElectronMiniIsoEleCorr_);
    outT->Branch("Electron_miniRelIsoEleCorr",&recoElectronMiniRelIsoEleCorr_);
    outT->Branch("Electron_chadIso",&recoElectronChadIso_);
    outT->Branch("Electron_nhadIso",&recoElectronNhadIso_);
    outT->Branch("Electron_phoIso",&recoElectronPhoIso_);
    outT->Branch("Electron_rhoEA",&recoElectronRhoEA_);
    outT->Branch("Electron_trkProb",&recoElectronTrkProb_);
    outT->Branch("Electron_numTrackerHits",&recoElectronTrkNumTrackerHits_);
    outT->Branch("Electron_numPixHits",&recoElectronTrkNumPixHits_);
    outT->Branch("Electron_numStripHits",&recoElectronTrkNumStripHits_);
    outT->Branch("Electron_charge",&recoElectronCharge_);
    outT->Branch("Electron_isPF",&recoElectronIsPF_);
    outT->Branch("Electron_genMatched",&recoElectronGenMatched_);
    outT->Branch("Electron_matchType",&recoElectronMatchType_);
    outT->Branch("Electron_dRJets",&recoElectronDrToJets_);
    outT->Branch("Electron_dPhiJets",&recoElectronDphiToJets_);
    outT->Branch("Electron_full55sigmaIetaIeta",&recoElectronFull5x5_sigmaIetaIeta_);
    outT->Branch("Electron_absdEtaSeed",&recoElectronAbsdEtaSeed_);
    outT->Branch("Electron_absdPhiIn",&recoElectronAbsdPhiIn_);
    outT->Branch("Electron_HoverE",&recoElectronHoverE_);
    outT->Branch("Electron_abs1overEm1overP",&recoElectronAbs1overEm1overP_);
    outT->Branch("Electron_expMissingInnerHits",&recoElectronExpMissingInnerHits_);
    outT->Branch("Electron_conversionVeto",&recoElectronConversionVeto_);
    outT->Branch("Electron_isEE",&recoElectronIsEE_);
    // supercluster
    outT->Branch("Electron_scRawE",&recoElectronScRawE_);
    outT->Branch("Electron_scEnergy",&recoElectronScEnergy_);
    outT->Branch("Electron_scEta",&recoElectronScEta_);
    outT->Branch("Electron_scPhi",&recoElectronScPhi_);
    outT->Branch("Electron_scEtaWidth",&recoElectronScEtaWidth_);
    outT->Branch("Electron_scPhiWidth",&recoElectronScPhiWidth_);
    outT->Branch("Electron_scClustersSize",&recoElectronScClustersSize_);
    // additional shower shapes
    outT->Branch("Electron_full55sigmaIetaIphi",&recoElectronFull5x5SigmaIetaIphi_);
    outT->Branch("Electron_seedEoverScRawE",&recoElectronSeedEoverScRawE_);
    outT->Branch("Electron_full55eMaxOverScRawE",&recoElectronFull5x5EMaxOverScRawE_);
    outT->Branch("Electron_full55e2ndOverScRawE",&recoElectronFull5x5E2ndOverScRawE_);
    // special vars for x-clean study
    outT->Branch("Electron_hasLptMatch",&recoElectronHasLptMatch_);
    outT->Branch("Electron_lptMatchIdx",&recoElectronLptMatchIdx_);
    outT->Branch("Electron_hasAllLptMatch",&recoElectronHasAllLptMatch_);
    outT->Branch("Electron_AllLptMatchIdx",&recoElectronAllLptMatchIdx_);

    // Low pT electrons
    outT->Branch("nLptElectron",&nElectronLowPt_);
    outT->Branch("LptElectron_pt",&recoLowPtElectronPt_);
    outT->Branch("LptElectron_eta",&recoLowPtElectronEta_);
    outT->Branch("LptElectron_etaErr",&recoLowPtElectronEtaError_);
    outT->Branch("LptElectron_phi",&recoLowPtElectronPhi_);
    outT->Branch("LptElectron_phiErr",&recoLowPtElectronPhiError_);
    outT->Branch("LptElectron_ID",&recoLowPtElectronID_);
    outT->Branch("LptElectron_angRes",&recoLowPtElectronAngularRes_);
    outT->Branch("LptElectron_e",&recoLowPtElectronE_);
    outT->Branch("LptElectron_vxy",&recoLowPtElectronVxy_);
    outT->Branch("LptElectron_vz",&recoLowPtElectronVz_);
    outT->Branch("LptElectron_svProxyValid",&recoLowPtElectronSVProxyValid_);
    outT->Branch("LptElectron_svProxyX",&recoLowPtElectronSVProxyX_);
    outT->Branch("LptElectron_svProxyY",&recoLowPtElectronSVProxyY_);
    outT->Branch("LptElectron_svProxyZ",&recoLowPtElectronSVProxyZ_);
    outT->Branch("LptElectron_svProxyVxy",&recoLowPtElectronSVProxyVxy_);
    outT->Branch("LptElectron_dxy",&recoLowPtElectronDxy_);
    outT->Branch("LptElectron_dxyErr",&recoLowPtElectronDxyError_);
    outT->Branch("LptElectron_dz",&recoLowPtElectronDz_);
    outT->Branch("LptElectron_dzErr",&recoLowPtElectronDzError_);
    outT->Branch("LptElectron_trkChi2",&recoLowPtElectronTrkChi2_);
    outT->Branch("LptElectron_trkIso",&recoLowPtElectronTrkIso_);
    outT->Branch("LptElectron_trkRelIso",&recoLowPtElectronTrkRelIso_);
    outT->Branch("LptElectron_calIso",&recoLowPtElectronCaloIso_);
    outT->Branch("LptElectron_calRelIso",&recoLowPtElectronCaloRelIso_);
    outT->Branch("LptElectron_PFIso",&recoLowPtElectronPFIso_);
    outT->Branch("LptElectron_PFRelIso",&recoLowPtElectronPFRelIso_);
    outT->Branch("LptElectron_miniIso",&recoLowPtElectronMiniIso_);
    outT->Branch("LptElectron_miniRelIso",&recoLowPtElectronMiniRelIso_);
    outT->Branch("LptElectron_PFIsoEleCorr",&recoLowPtElectronPFIsoEleCorr_);
    outT->Branch("LptElectron_PFRelIsoEleCorr",&recoLowPtElectronPFRelIsoEleCorr_);
    outT->Branch("LptElectron_miniIsoEleCorr",&recoLowPtElectronMiniIsoEleCorr_);
    outT->Branch("LptElectron_miniRelIsoEleCorr",&recoLowPtElectronMiniRelIsoEleCorr_);
    outT->Branch("LptElectron_chadIso",&recoLowPtElectronChadIso_);
    outT->Branch("LptElectron_nhadIso",&recoLowPtElectronNhadIso_);
    outT->Branch("LptElectron_phoIso",&recoLowPtElectronPhoIso_);
    outT->Branch("LptElectron_rhoEA",&recoLowPtElectronRhoEA_);
    outT->Branch("LptElectron_trkProb",&recoLowPtElectronTrkProb_);
    outT->Branch("LptElectron_numTrackerHits",&recoLowPtElectronTrkNumTrackerHits_);
    outT->Branch("LptElectron_numPixHits",&recoLowPtElectronTrkNumPixHits_);
    outT->Branch("LptElectron_numStripHits",&recoLowPtElectronTrkNumStripHits_);
    outT->Branch("LptElectron_charge",&recoLowPtElectronCharge_);
    outT->Branch("LptElectron_minDRtoReg",&recoLowPtElectronMinDrToReg_);
    outT->Branch("LptElectron_isPF",&recoLowPtElectronIsPF_);
    outT->Branch("LptElectron_genMatched",&recoLowPtElectronGenMatched_);
    outT->Branch("LptElectron_matchType",&recoLowPtElectronMatchType_);
    outT->Branch("LptElectron_dRJets",&recoLowPtElectronDrToJets_);
    outT->Branch("LptElectron_dPhiJets",&recoLowPtElectronDphiToJets_);
    outT->Branch("LptElectron_full55sigmaIetaIeta",&recoLowPtElectronFull5x5_sigmaIetaIeta_);
    outT->Branch("LptElectron_absdEtaSeed",&recoLowPtElectronAbsdEtaSeed_);
    outT->Branch("LptElectron_absdPhiIn",&recoLowPtElectronAbsdPhiIn_);
    outT->Branch("LptElectron_HoverE",&recoLowPtElectronHoverE_);
    outT->Branch("LptElectron_abs1overEm1overP",&recoLowPtElectronAbs1overEm1overP_);
    outT->Branch("LptElectron_expMissingInnerHits",&recoLowPtElectronExpMissingInnerHits_);
    outT->Branch("LptElectron_conversionVeto",&recoLowPtElectronConversionVeto_);
    outT->Branch("LptElectron_isEE",&recoLowPtElectronIsEE_);
    // supercluster
    outT->Branch("LptElectron_scRawE",&recoLowPtElectronScRawE_);
    outT->Branch("LptElectron_scEnergy",&recoLowPtElectronScEnergy_);
    outT->Branch("LptElectron_scEta",&recoLowPtElectronScEta_);
    outT->Branch("LptElectron_scPhi",&recoLowPtElectronScPhi_);
    outT->Branch("LptElectron_scEtaWidth",&recoLowPtElectronScEtaWidth_);
    outT->Branch("LptElectron_scPhiWidth",&recoLowPtElectronScPhiWidth_);
    outT->Branch("LptElectron_scClustersSize",&recoLowPtElectronScClustersSize_);
    // additional shower shapes
    outT->Branch("LptElectron_full55r9",&recoLowPtElectronFull5x5R9_);
    outT->Branch("LptElectron_full55e5x5",&recoLowPtElectronFull5x5E5x5_);
    outT->Branch("LptElectron_full55e1x5",&recoLowPtElectronFull5x5E1x5_);
    outT->Branch("LptElectron_full55e2x5Max",&recoLowPtElectronFull5x5E2x5Max_);
    outT->Branch("LptElectron_full55sigmaIphiIphi",&recoLowPtElectronFull5x5SigmaIphiIphi_);
    outT->Branch("LptElectron_full55hcalDepth1OverEcal",&recoLowPtElectronFull5x5HcalDepth1OverEcal_);
    outT->Branch("LptElectron_full55hcalDepth2OverEcal",&recoLowPtElectronFull5x5HcalDepth2OverEcal_);
    outT->Branch("LptElectron_full55sigmaIetaIphi",&recoLowPtElectronFull5x5SigmaIetaIphi_);
    outT->Branch("LptElectron_seedEoverScRawE",&recoLowPtElectronSeedEoverScRawE_);
    outT->Branch("LptElectron_full55eMaxOverScRawE",&recoLowPtElectronFull5x5EMaxOverScRawE_);
    outT->Branch("LptElectron_full55e2ndOverScRawE",&recoLowPtElectronFull5x5E2ndOverScRawE_);
    // special vars for x-cleaning study
    outT->Branch("LptElectron_xCleaned",&recoLowPtElectronIsXCleaned_);
    outT->Branch("LptElectron_gedIdx",&recoLowPtElectronGEDidx_);
    outT->Branch("LptElectron_gedIsMatched",&recoLowPtElectronGEDisMatched_);

    // All Low pT electrons (includes cross-cleaned)
    outT->Branch("nAllLptElectron",&nElectronAllLowPt_);
    outT->Branch("AllLptElectron_pt",&recoAllLowPtElectronPt_);
    outT->Branch("AllLptElectron_eta",&recoAllLowPtElectronEta_);
    outT->Branch("AllLptElectron_etaErr",&recoAllLowPtElectronEtaError_);
    outT->Branch("AllLptElectron_phi",&recoAllLowPtElectronPhi_);
    outT->Branch("AllLptElectron_phiErr",&recoAllLowPtElectronPhiError_);
    outT->Branch("AllLptElectron_ID",&recoAllLowPtElectronID_);
    outT->Branch("AllLptElectron_angRes",&recoAllLowPtElectronAngularRes_);
    outT->Branch("AllLptElectron_e",&recoAllLowPtElectronE_);
    outT->Branch("AllLptElectron_vxy",&recoAllLowPtElectronVxy_);
    outT->Branch("AllLptElectron_vz",&recoAllLowPtElectronVz_);
    outT->Branch("AllLptElectron_svProxyValid",&recoAllLowPtElectronSVProxyValid_);
    outT->Branch("AllLptElectron_svProxyX",&recoAllLowPtElectronSVProxyX_);
    outT->Branch("AllLptElectron_svProxyY",&recoAllLowPtElectronSVProxyY_);
    outT->Branch("AllLptElectron_svProxyZ",&recoAllLowPtElectronSVProxyZ_);
    outT->Branch("AllLptElectron_svProxyVxy",&recoAllLowPtElectronSVProxyVxy_);
    outT->Branch("AllLptElectron_dxy",&recoAllLowPtElectronDxy_);
    outT->Branch("AllLptElectron_dxyErr",&recoAllLowPtElectronDxyError_);
    outT->Branch("AllLptElectron_dz",&recoAllLowPtElectronDz_);
    outT->Branch("AllLptElectron_dzErr",&recoAllLowPtElectronDzError_);
    outT->Branch("AllLptElectron_trkChi2",&recoAllLowPtElectronTrkChi2_);
    outT->Branch("AllLptElectron_trkIso",&recoAllLowPtElectronTrkIso_);
    outT->Branch("AllLptElectron_trkRelIso",&recoAllLowPtElectronTrkRelIso_);
    outT->Branch("AllLptElectron_calIso",&recoAllLowPtElectronCaloIso_);
    outT->Branch("AllLptElectron_calRelIso",&recoAllLowPtElectronCaloRelIso_);
    outT->Branch("AllLptElectron_PFIso",&recoAllLowPtElectronPFIso_);
    outT->Branch("AllLptElectron_PFRelIso",&recoAllLowPtElectronPFRelIso_);
    outT->Branch("AllLptElectron_miniIso",&recoAllLowPtElectronMiniIso_);
    outT->Branch("AllLptElectron_miniRelIso",&recoAllLowPtElectronMiniRelIso_);
    outT->Branch("AllLptElectron_PFIsoEleCorr",&recoAllLowPtElectronPFIsoEleCorr_);
    outT->Branch("AllLptElectron_PFRelIsoEleCorr",&recoAllLowPtElectronPFRelIsoEleCorr_);
    outT->Branch("AllLptElectron_miniIsoEleCorr",&recoAllLowPtElectronMiniIsoEleCorr_);
    outT->Branch("AllLptElectron_miniRelIsoEleCorr",&recoAllLowPtElectronMiniRelIsoEleCorr_);
    outT->Branch("AllLptElectron_chadIso",&recoAllLowPtElectronChadIso_);
    outT->Branch("AllLptElectron_nhadIso",&recoAllLowPtElectronNhadIso_);
    outT->Branch("AllLptElectron_phoIso",&recoAllLowPtElectronPhoIso_);
    outT->Branch("AllLptElectron_rhoEA",&recoAllLowPtElectronRhoEA_);
    outT->Branch("AllLptElectron_trkProb",&recoAllLowPtElectronTrkProb_);
    outT->Branch("AllLptElectron_numTrackerHits",&recoAllLowPtElectronTrkNumTrackerHits_);
    outT->Branch("AllLptElectron_numPixHits",&recoAllLowPtElectronTrkNumPixHits_);
    outT->Branch("AllLptElectron_numStripHits",&recoAllLowPtElectronTrkNumStripHits_);
    outT->Branch("AllLptElectron_charge",&recoAllLowPtElectronCharge_);
    outT->Branch("AllLptElectron_minDRtoReg",&recoAllLowPtElectronMinDrToReg_);
    outT->Branch("AllLptElectron_isPF",&recoAllLowPtElectronIsPF_);
    outT->Branch("AllLptElectron_genMatched",&recoAllLowPtElectronGenMatched_);
    outT->Branch("AllLptElectron_matchType",&recoAllLowPtElectronMatchType_);
    outT->Branch("AllLptElectron_dRJets",&recoAllLowPtElectronDrToJets_);
    outT->Branch("AllLptElectron_dPhiJets",&recoAllLowPtElectronDphiToJets_);
    outT->Branch("AllLptElectron_full55sigmaIetaIeta",&recoAllLowPtElectronFull5x5_sigmaIetaIeta_);
    outT->Branch("AllLptElectron_absdEtaSeed",&recoAllLowPtElectronAbsdEtaSeed_);
    outT->Branch("AllLptElectron_absdPhiIn",&recoAllLowPtElectronAbsdPhiIn_);
    outT->Branch("AllLptElectron_HoverE",&recoAllLowPtElectronHoverE_);
    outT->Branch("AllLptElectron_abs1overEm1overP",&recoAllLowPtElectronAbs1overEm1overP_);
    outT->Branch("AllLptElectron_expMissingInnerHits",&recoAllLowPtElectronExpMissingInnerHits_);
    outT->Branch("AllLptElectron_conversionVeto",&recoAllLowPtElectronConversionVeto_);
    outT->Branch("AllLptElectron_isEE",&recoAllLowPtElectronIsEE_);
    // supercluster
    outT->Branch("AllLptElectron_scRawE",&recoAllLowPtElectronScRawE_);
    outT->Branch("AllLptElectron_scEnergy",&recoAllLowPtElectronScEnergy_);
    outT->Branch("AllLptElectron_scEta",&recoAllLowPtElectronScEta_);
    outT->Branch("AllLptElectron_scPhi",&recoAllLowPtElectronScPhi_);
    outT->Branch("AllLptElectron_scEtaWidth",&recoAllLowPtElectronScEtaWidth_);
    outT->Branch("AllLptElectron_scPhiWidth",&recoAllLowPtElectronScPhiWidth_);
    outT->Branch("AllLptElectron_scClustersSize",&recoAllLowPtElectronScClustersSize_);
    // additional shower shapes
    outT->Branch("AllLptElectron_full55r9",&recoAllLowPtElectronFull5x5R9_);
    outT->Branch("AllLptElectron_full55e5x5",&recoAllLowPtElectronFull5x5E5x5_);
    outT->Branch("AllLptElectron_full55e1x5",&recoAllLowPtElectronFull5x5E1x5_);
    outT->Branch("AllLptElectron_full55e2x5Max",&recoAllLowPtElectronFull5x5E2x5Max_);
    outT->Branch("AllLptElectron_full55sigmaIphiIphi",&recoAllLowPtElectronFull5x5SigmaIphiIphi_);
    outT->Branch("AllLptElectron_full55hcalDepth1OverEcal",&recoAllLowPtElectronFull5x5HcalDepth1OverEcal_);
    outT->Branch("AllLptElectron_full55hcalDepth2OverEcal",&recoAllLowPtElectronFull5x5HcalDepth2OverEcal_);
    outT->Branch("AllLptElectron_full55sigmaIetaIphi",&recoAllLowPtElectronFull5x5SigmaIetaIphi_);
    outT->Branch("AllLptElectron_seedEoverScRawE",&recoAllLowPtElectronSeedEoverScRawE_);
    outT->Branch("AllLptElectron_full55eMaxOverScRawE",&recoAllLowPtElectronFull5x5EMaxOverScRawE_);
    outT->Branch("AllLptElectron_full55e2ndOverScRawE",&recoAllLowPtElectronFull5x5E2ndOverScRawE_);
    // x-cleaning status and GED match info
    outT->Branch("AllLptElectron_xCleaned",&recoAllLowPtElectronIsXCleaned_);
    outT->Branch("AllLptElectron_gedIdx",&recoAllLowPtElectronGEDidx_);
    outT->Branch("AllLptElectron_gedIsMatched",&recoAllLowPtElectronGEDisMatched_);

    // Standard standalone (STA) muons.
    outT->Branch("STAMuonAvailable", &staMuonAvailable_);
    outT->Branch("nSTAMuon", &nSTAMuon_);
    outT->Branch("recoSTAMuonPt", &recoSTAMuonPt_);
    outT->Branch("recoSTAMuonPtErr", &recoSTAMuonPtErr_);
    outT->Branch("recoSTAMuonEta", &recoSTAMuonEta_);
    outT->Branch("recoSTAMuonEtaErr", &recoSTAMuonEtaErr_);
    outT->Branch("recoSTAMuonPhi", &recoSTAMuonPhi_);
    outT->Branch("recoSTAMuonPhiErr", &recoSTAMuonPhiErr_);
    outT->Branch("recoSTAMuonOuterEta", &recoSTAMuonOuterEta_);
    outT->Branch("recoSTAMuonOuterPhi", &recoSTAMuonOuterPhi_);
    outT->Branch("recoSTAMuonE", &recoSTAMuonE_);
    outT->Branch("recoSTAMuonPx", &recoSTAMuonPx_);
    outT->Branch("recoSTAMuonPy", &recoSTAMuonPy_);
    outT->Branch("recoSTAMuonPz", &recoSTAMuonPz_);
    outT->Branch("recoSTAMuonVxy", &recoSTAMuonVxy_);
    outT->Branch("recoSTAMuonVz", &recoSTAMuonVz_);
    outT->Branch("recoSTAMuonDxy", &recoSTAMuonDxy_);
    outT->Branch("recoSTAMuonDxyError", &recoSTAMuonDxyError_);
    outT->Branch("recoSTAMuonDz", &recoSTAMuonDz_);
    outT->Branch("recoSTAMuonDzError", &recoSTAMuonDzError_);
    outT->Branch("recoSTAMuonTrkChi2", &recoSTAMuonTrkChi2_);
    outT->Branch("recoSTAMuonTrkProb", &recoSTAMuonTrkProb_);
    outT->Branch("recoSTAMuonTrkNumTrackerHits", &recoSTAMuonTrkNumTrackerHits_);
    outT->Branch("recoSTAMuonTrkNumPixHits", &recoSTAMuonTrkNumPixHits_);
    outT->Branch("recoSTAMuonTrkNumStripHits", &recoSTAMuonTrkNumStripHits_);
    outT->Branch("recoSTAMuonCharge", &recoSTAMuonCharge_);
    outT->Branch("recoSTAMuonDisplacedId", &recoSTAMuonDisplacedId_);
    outT->Branch("recoSTAMuonTrkNumCSCHits", &recoSTAMuonTrkNumCSCHits_);
    outT->Branch("recoSTAMuonTrkNumHits", &recoSTAMuonTrkNumHits_);
    outT->Branch("recoSTAMuonTrkNumPlanes", &recoSTAMuonTrkNumPlanes_);
    outT->Branch("recoSTAMuonTrkNumDTHits", &recoSTAMuonTrkNumDTHits_);
    outT->Branch("recoSTAMuonIdx", &recoSTAMuonIdx_);
    branchP4("recoSTAMuonP4", &recoSTAMuonP4_, false);  // flat pt/eta/phi (or position eta/phi) already branched

    outT->Branch("recoSTAMuonPropSt1Valid", &recoSTAMuonPropSt1Valid_);
    outT->Branch("recoSTAMuonPropSt1Eta", &recoSTAMuonPropSt1Eta_);
    outT->Branch("recoSTAMuonPropSt1Phi", &recoSTAMuonPropSt1Phi_);
    outT->Branch("recoSTAMuonPropSt1MomEta", &recoSTAMuonPropSt1MomEta_);
    outT->Branch("recoSTAMuonPropSt1MomPhi", &recoSTAMuonPropSt1MomPhi_);
    outT->Branch("recoSTAMuonPropSt1Idx", &recoSTAMuonPropSt1Idx_);

    outT->Branch("recoSTAMuonPropSt2Valid", &recoSTAMuonPropSt2Valid_);
    outT->Branch("recoSTAMuonPropSt2Eta", &recoSTAMuonPropSt2Eta_);
    outT->Branch("recoSTAMuonPropSt2Phi", &recoSTAMuonPropSt2Phi_);
    outT->Branch("recoSTAMuonPropSt2MomEta", &recoSTAMuonPropSt2MomEta_);
    outT->Branch("recoSTAMuonPropSt2MomPhi", &recoSTAMuonPropSt2MomPhi_);
    outT->Branch("recoSTAMuonPropSt2Idx", &recoSTAMuonPropSt2Idx_);

    outT->Branch("nPropSTAMuonSt1", &nPropSTAMuonSt1_);
    outT->Branch("PropSTAMuonSt1_staMuonIdx", &propSTAMuonSt1STAMuonIdx_);
    branchP4("PropSTAMuonSt1_p4", &propSTAMuonSt1P4_);
    outT->Branch("PropSTAMuonSt1_positionEta", &propSTAMuonSt1PositionEta_);
    outT->Branch("PropSTAMuonSt1_positionPhi", &propSTAMuonSt1PositionPhi_);

    outT->Branch("nPropSTAMuonSt2", &nPropSTAMuonSt2_);
    outT->Branch("PropSTAMuonSt2_staMuonIdx", &propSTAMuonSt2STAMuonIdx_);
    branchP4("PropSTAMuonSt2_p4", &propSTAMuonSt2P4_);
    outT->Branch("PropSTAMuonSt2_positionEta", &propSTAMuonSt2PositionEta_);
    outT->Branch("PropSTAMuonSt2_positionPhi", &propSTAMuonSt2PositionPhi_);

    outT->Branch("recoSTAMuonPropSt3Valid", &recoSTAMuonPropSt3Valid_);
    outT->Branch("recoSTAMuonPropSt3Eta", &recoSTAMuonPropSt3Eta_);
    outT->Branch("recoSTAMuonPropSt3Phi", &recoSTAMuonPropSt3Phi_);
    outT->Branch("recoSTAMuonPropSt3MomEta", &recoSTAMuonPropSt3MomEta_);
    outT->Branch("recoSTAMuonPropSt3MomPhi", &recoSTAMuonPropSt3MomPhi_);
    outT->Branch("recoSTAMuonPropSt3Idx", &recoSTAMuonPropSt3Idx_);
    outT->Branch("nPropSTAMuonSt3", &nPropSTAMuonSt3_);
    outT->Branch("PropSTAMuonSt3_staMuonIdx", &propSTAMuonSt3STAMuonIdx_);
    branchP4("PropSTAMuonSt3_p4", &propSTAMuonSt3P4_);
    outT->Branch("PropSTAMuonSt3_positionEta", &propSTAMuonSt3PositionEta_);
    outT->Branch("PropSTAMuonSt3_positionPhi", &propSTAMuonSt3PositionPhi_);
    outT->Branch("recoSTAMuonPropSt4Valid", &recoSTAMuonPropSt4Valid_);
    outT->Branch("recoSTAMuonPropSt4Eta", &recoSTAMuonPropSt4Eta_);
    outT->Branch("recoSTAMuonPropSt4Phi", &recoSTAMuonPropSt4Phi_);
    outT->Branch("recoSTAMuonPropSt4MomEta", &recoSTAMuonPropSt4MomEta_);
    outT->Branch("recoSTAMuonPropSt4MomPhi", &recoSTAMuonPropSt4MomPhi_);
    outT->Branch("recoSTAMuonPropSt4Idx", &recoSTAMuonPropSt4Idx_);
    outT->Branch("nPropSTAMuonSt4", &nPropSTAMuonSt4_);
    outT->Branch("PropSTAMuonSt4_staMuonIdx", &propSTAMuonSt4STAMuonIdx_);
    branchP4("PropSTAMuonSt4_p4", &propSTAMuonSt4P4_);
    outT->Branch("PropSTAMuonSt4_positionEta", &propSTAMuonSt4PositionEta_);
    outT->Branch("PropSTAMuonSt4_positionPhi", &propSTAMuonSt4PositionPhi_);

    // DSA muons
    outT->Branch("DSAMuonAvailable", &dsaMuonAvailable_);
    outT->Branch("nDSAMuon", &nDSAMuon_);
    outT->Branch("recoDSAMuonPt", &recoDSAMuonPt_);
    outT->Branch("recoDSAMuonPtErr", &recoDSAMuonPtErr_);
    outT->Branch("recoDSAMuonEta", &recoDSAMuonEta_);
    outT->Branch("recoDSAMuonEtaErr", &recoDSAMuonEtaErr_);
    outT->Branch("recoDSAMuonPhi", &recoDSAMuonPhi_);
    outT->Branch("recoDSAMuonPhiErr", &recoDSAMuonPhiErr_);
    outT->Branch("recoDSAMuonOuterEta", &recoDSAMuonOuterEta_);
    outT->Branch("recoDSAMuonOuterPhi", &recoDSAMuonOuterPhi_);
    outT->Branch("recoDSAMuonE", &recoDSAMuonE_);
    outT->Branch("recoDSAMuonPx", &recoDSAMuonPx_);
    outT->Branch("recoDSAMuonPy", &recoDSAMuonPy_);
    outT->Branch("recoDSAMuonPz", &recoDSAMuonPz_);
    outT->Branch("recoDSAMuonVxy", &recoDSAMuonVxy_);
    outT->Branch("recoDSAMuonVz", &recoDSAMuonVz_);
    outT->Branch("recoDSAMuonDxy", &recoDSAMuonDxy_);
    outT->Branch("recoDSAMuonDxyError", &recoDSAMuonDxyError_);
    outT->Branch("recoDSAMuonDz", &recoDSAMuonDz_);
    outT->Branch("recoDSAMuonDzError", &recoDSAMuonDzError_);
    outT->Branch("recoDSAMuonTrkChi2", &recoDSAMuonTrkChi2_);
    outT->Branch("recoDSAMuonTrkProb", &recoDSAMuonTrkProb_);
    outT->Branch("recoDSAMuonTrkNumTrackerHits", &recoDSAMuonTrkNumTrackerHits_);
    outT->Branch("recoDSAMuonTrkNumPixHits", &recoDSAMuonTrkNumPixHits_);
    outT->Branch("recoDSAMuonTrkNumStripHits", &recoDSAMuonTrkNumStripHits_);
    outT->Branch("recoDSAMuonCharge", &recoDSAMuonCharge_);
    outT->Branch("recoDSAMuonDisplacedId", &recoDSAMuonDisplacedId_);
    outT->Branch("recoDSAMuonTrkNumCSCHits", &recoDSAMuonTrkNumCSCHits_);
    outT->Branch("recoDSAMuonTrkNumHits", &recoDSAMuonTrkNumHits_);
    outT->Branch("recoDSAMuonTrkNumPlanes", &recoDSAMuonTrkNumPlanes_);
    outT->Branch("recoDSAMuonTrkNumDTHits", &recoDSAMuonTrkNumDTHits_);
    outT->Branch("recoDSAMuonIdx", &recoDSAMuonIdx_);
    branchP4("recoDSAMuonP4", &recoDSAMuonP4_, false);  // flat pt/eta/phi (or position eta/phi) already branched

    outT->Branch("recoDSAMuonPropSt1Valid", &recoDSAMuonPropSt1Valid_);
    outT->Branch("recoDSAMuonPropSt1Eta", &recoDSAMuonPropSt1Eta_);
    outT->Branch("recoDSAMuonPropSt1Phi", &recoDSAMuonPropSt1Phi_);
    outT->Branch("recoDSAMuonPropSt1MomEta", &recoDSAMuonPropSt1MomEta_);
    outT->Branch("recoDSAMuonPropSt1MomPhi", &recoDSAMuonPropSt1MomPhi_);
    outT->Branch("recoDSAMuonPropSt1Idx", &recoDSAMuonPropSt1Idx_);

    outT->Branch("recoDSAMuonPropSt2Valid", &recoDSAMuonPropSt2Valid_);
    outT->Branch("recoDSAMuonPropSt2Eta", &recoDSAMuonPropSt2Eta_);
    outT->Branch("recoDSAMuonPropSt2Phi", &recoDSAMuonPropSt2Phi_);
    outT->Branch("recoDSAMuonPropSt2MomEta", &recoDSAMuonPropSt2MomEta_);
    outT->Branch("recoDSAMuonPropSt2MomPhi", &recoDSAMuonPropSt2MomPhi_);
    outT->Branch("recoDSAMuonPropSt2Idx", &recoDSAMuonPropSt2Idx_);

    outT->Branch("recoDSAMuonPropSt3Valid", &recoDSAMuonPropSt3Valid_);
    outT->Branch("recoDSAMuonPropSt3Eta", &recoDSAMuonPropSt3Eta_);
    outT->Branch("recoDSAMuonPropSt3Phi", &recoDSAMuonPropSt3Phi_);
    outT->Branch("recoDSAMuonPropSt3MomEta", &recoDSAMuonPropSt3MomEta_);
    outT->Branch("recoDSAMuonPropSt3MomPhi", &recoDSAMuonPropSt3MomPhi_);

    outT->Branch("recoDSAMuonPropSt4Valid", &recoDSAMuonPropSt4Valid_);
    outT->Branch("recoDSAMuonPropSt4Eta", &recoDSAMuonPropSt4Eta_);
    outT->Branch("recoDSAMuonPropSt4Phi", &recoDSAMuonPropSt4Phi_);
    outT->Branch("recoDSAMuonPropSt4MomEta", &recoDSAMuonPropSt4MomEta_);
    outT->Branch("recoDSAMuonPropSt4MomPhi", &recoDSAMuonPropSt4MomPhi_);

    outT->Branch("nPropDSAMuonSt1", &nPropDSAMuonSt1_);
    outT->Branch("PropDSAMuonSt1_dsaMuonIdx", &propDSAMuonSt1DSAMuonIdx_);
    branchP4("PropDSAMuonSt1_p4", &propDSAMuonSt1P4_);
    outT->Branch("PropDSAMuonSt1_positionEta", &propDSAMuonSt1PositionEta_);
    outT->Branch("PropDSAMuonSt1_positionPhi", &propDSAMuonSt1PositionPhi_);

    outT->Branch("nPropDSAMuonSt2", &nPropDSAMuonSt2_);
    outT->Branch("PropDSAMuonSt2_dsaMuonIdx", &propDSAMuonSt2DSAMuonIdx_);
    branchP4("PropDSAMuonSt2_p4", &propDSAMuonSt2P4_);
    outT->Branch("PropDSAMuonSt2_positionEta", &propDSAMuonSt2PositionEta_);
    outT->Branch("PropDSAMuonSt2_positionPhi", &propDSAMuonSt2PositionPhi_);

    outT->Branch("recoDSAMuonPropSt3Idx", &recoDSAMuonPropSt3Idx_);
    outT->Branch("nPropDSAMuonSt3", &nPropDSAMuonSt3_);
    outT->Branch("PropDSAMuonSt3_dsaMuonIdx", &propDSAMuonSt3DSAMuonIdx_);
    branchP4("PropDSAMuonSt3_p4", &propDSAMuonSt3P4_);
    outT->Branch("PropDSAMuonSt3_positionEta", &propDSAMuonSt3PositionEta_);
    outT->Branch("PropDSAMuonSt3_positionPhi", &propDSAMuonSt3PositionPhi_);
    outT->Branch("recoDSAMuonPropSt4Idx", &recoDSAMuonPropSt4Idx_);
    outT->Branch("nPropDSAMuonSt4", &nPropDSAMuonSt4_);
    outT->Branch("PropDSAMuonSt4_dsaMuonIdx", &propDSAMuonSt4DSAMuonIdx_);
    branchP4("PropDSAMuonSt4_p4", &propDSAMuonSt4P4_);
    outT->Branch("PropDSAMuonSt4_positionEta", &propDSAMuonSt4PositionEta_);
    outT->Branch("PropDSAMuonSt4_positionPhi", &propDSAMuonSt4PositionPhi_);

    outT->Branch("recoDSAMuonOuterHitValid", &recoDSAMuonOuterHitValid_);
    outT->Branch("recoDSAMuonOuterHitTrackExtraAvailable", &recoDSAMuonOuterHitTrackExtraAvailable_);
    outT->Branch("recoDSAMuonOuterHitRecHitsSize", &recoDSAMuonOuterHitRecHitsSize_);
    outT->Branch("recoDSAMuonOuterHitNValidMuonHits", &recoDSAMuonOuterHitNValidMuonHits_);
    outT->Branch("recoDSAMuonOuterHitDetId", &recoDSAMuonOuterHitDetId_);
    outT->Branch("recoDSAMuonOuterHitX", &recoDSAMuonOuterHitX_);
    outT->Branch("recoDSAMuonOuterHitY", &recoDSAMuonOuterHitY_);
    outT->Branch("recoDSAMuonOuterHitZ", &recoDSAMuonOuterHitZ_);
    outT->Branch("recoDSAMuonOuterHitR", &recoDSAMuonOuterHitR_);
    outT->Branch("recoDSAMuonOuterHitEta", &recoDSAMuonOuterHitEta_);
    outT->Branch("recoDSAMuonOuterHitPhi", &recoDSAMuonOuterHitPhi_);

    outT->Branch("recoDSAMuonMinDrToGenSigMuonOuterHit", &recoDSAMuonMinDrToGenSigMuonOuterHit_);
    outT->Branch("recoDSAMuonMatchGenSigMuonOuterHitIdx", &recoDSAMuonMatchGenSigMuonOuterHitIdx_);
    outT->Branch("recoDSAMuonMatchGenSigMuonOuterHitPdgId", &recoDSAMuonMatchGenSigMuonOuterHitPdgId_);
    outT->Branch("recoDSAMuonMatchGenSigMuonOuterHitPropValid", &recoDSAMuonMatchGenSigMuonOuterHitPropValid_);
    outT->Branch("recoDSAMuonMatchGenSigMuonOuterHitPropEta", &recoDSAMuonMatchGenSigMuonOuterHitPropEta_);
    outT->Branch("recoDSAMuonMatchGenSigMuonOuterHitPropPhi", &recoDSAMuonMatchGenSigMuonOuterHitPropPhi_);
    outT->Branch("recoDSAMuonMatchGenSigMuonOuterHitPropMomEta", &recoDSAMuonMatchGenSigMuonOuterHitPropMomEta_);
    outT->Branch("recoDSAMuonMatchGenSigMuonOuterHitPropMomPhi", &recoDSAMuonMatchGenSigMuonOuterHitPropMomPhi_);
    
    // Photons
    outT->Branch("nPhoton",&nPhotons_);
    outT->Branch("Photon_et",&PhotonEt_);
    outT->Branch("Photon_eta",&PhotonEta_);
    outT->Branch("Photon_phi",&PhotonPhi_);
    outT->Branch("Photon_pt",&PhotonPt_);

    // OOT Photons
    outT->Branch("nootPhoton",&nOOTPhotons_);
    outT->Branch("ootPhoton_et",&ootPhotonEt_);
    outT->Branch("ootPhoton_eta",&ootPhotonEta_);
    outT->Branch("ootPhoton_phi",&ootPhotonPhi_);
    outT->Branch("ootPhoton_pt",&ootPhotonPt_);

    // Isolated Tracks
    outT->Branch("nIsoTrack",&nIsoTrack_);
    outT->Branch("IsoTrack_pt",&isoTrackPt_);
    outT->Branch("IsoTrack_eta",&isoTrackEta_);
    outT->Branch("IsoTrack_phi",&isoTrackPhi_);

    // PF Candidates
    outT->Branch("nPFCand",&nPFCand_);
    outT->Branch("PFCand_pt",&pfCandPt_);
    outT->Branch("PFCand_eta",&pfCandEta_);
    outT->Branch("PFCand_phi",&pfCandPhi_);

    // Lost Tracks
    outT->Branch("nLostTrack",&nLostTrack_);
    outT->Branch("LostTrack_pt",&lostTrackPt_);
    outT->Branch("LostTrack_eta",&lostTrackEta_);
    outT->Branch("LostTrack_phi",&lostTrackPhi_);

    // Photon conversions (unused -- fill loop is disabled in ElectronSkimmer.cc, see nt.nConversions_)
    /*
    outT->Branch("nConversion",&nConversions_);
    outT->Branch("Conversion_pt",&conversionPt_);
    outT->Branch("Conversion_eta",&conversionEta_);
    outT->Branch("Conversion_phi",&conversionPhi_);
    outT->Branch("Conversion_energy",&conversionE_);
    outT->Branch("Conversion_px",&conversionPx_);
    outT->Branch("Conversion_py",&conversionPy_);
    outT->Branch("Conversion_pz",&conversionPz_);
    outT->Branch("Conversion_vxy",&conversionVxy_);
    outT->Branch("Conversion_vz",&conversionVz_);
    outT->Branch("Conversion_x",&conversionX_);
    outT->Branch("Conversion_y",&conversionY_);
    outT->Branch("Conversion_z",&conversionZ_);
    outT->Branch("Conversion_lxy",&conversionLxy_);
    outT->Branch("Conversion_lz",&conversionLz_);
    outT->Branch("Conversion_lxyPV",&conversionLxyPV_);
    outT->Branch("Conversion_lzPV",&conversionLzPV_);
    outT->Branch("Conversion_dxy",&conversionDxy_);
    outT->Branch("Conversion_dz",&conversionDz_);
    outT->Branch("Conversion_dxyPV",&conversionDxyPV_);
    outT->Branch("Conversion_dzPV",&conversionDzPV_);
    outT->Branch("Conversion_EoverP",&conversionEoverP_);
    outT->Branch("Conversion_EoverPrefit",&conversionEoverPrefit_);
    outT->Branch("Conversion_m",&conversionM_);
    outT->Branch("Conversion_dR",&conversionDr_);
    outT->Branch("Conversion_chi2",&conversionChi2_);
    outT->Branch("Conversion_tk1_nHitsBeforeVtx",&conversion_Trk1nHitsVtx_);
    outT->Branch("Conversion_tk1_pt",&conversion_Trk1Pt_);
    outT->Branch("Conversion_tk1_eta",&conversion_Trk1Eta_);
    outT->Branch("Conversion_tk1_phi",&conversion_Trk1Phi_);
    outT->Branch("Conversion_tk1_chi2",&conversion_Trk1Chi2_);
    outT->Branch("Conversion_tk1_nValidHits",&conversion_Trk1NValidHits_);
    outT->Branch("Conversion_tk1_nLostHits",&conversion_Trk1numLostHits_);
    outT->Branch("Conversion_tk1_dxy",&conversion_Trk1dxy_);
    outT->Branch("Conversion_tk1_dxyBS",&conversion_Trk1dxyBS_);
    outT->Branch("Conversion_tk1_dxyPV",&conversion_Trk1dxyPV_);
    outT->Branch("Conversion_tk1_dz",&conversion_Trk1dz_);
    outT->Branch("Conversion_tk1_dzPV",&conversion_Trk1dzPV_);
    outT->Branch("Conversion_tk2_nHitsBeforeVtx",&conversion_Trk2nHitsVtx_);
    outT->Branch("Conversion_tk2_pt",&conversion_Trk2Pt_);
    outT->Branch("Conversion_tk2_eta",&conversion_Trk2Eta_);
    outT->Branch("Conversion_tk2_phi",&conversion_Trk2Phi_);
    outT->Branch("Conversion_tk2_chi2",&conversion_Trk2Chi2_);
    outT->Branch("Conversion_tk2_nValidHits",&conversion_Trk2NValidHits_);
    outT->Branch("Conversion_tk2_nLostHits",&conversion_Trk2numLostHits_);
    outT->Branch("Conversion_tk2_dxy",&conversion_Trk2dxy_);
    outT->Branch("Conversion_tk2_dxyBS",&conversion_Trk2dxyBS_);
    outT->Branch("Conversion_tk2_dxyPV",&conversion_Trk2dxyPV_);
    outT->Branch("Conversion_tk2_dz",&conversion_Trk2dz_);
    outT->Branch("Conversion_tk2_dzPV",&conversion_Trk2dzPV_);
    */

    // Jets
    outT->Branch("nPFJetAll",&PFNJetAll_);
    outT->Branch("nPFJet",&PFNJet_);
    outT->Branch("numPFJetTrueB",&PFNbJetTrue_);
    outT->Branch("numPFJetTaggedB",&PFNbJetTagged_);
    outT->Branch("PFJet_matchedGenJetPt",&PFJet_matchedGenJetPt_);
    outT->Branch("PFJet_matchedGenJetEta",&PFJet_matchedGenJetEta_);
    outT->Branch("PFJet_matchedGenJetPhi",&PFJet_matchedGenJetPhi_);
    outT->Branch("PFJet_rawFactor",&PFJetRawFactor_);
    outT->Branch("PFJet_massRaw",&PFJetMassRaw_);
    outT->Branch("PFJet_ptRaw",&PFJetPtRaw_);
    outT->Branch("PFJet_area",&PFJetArea_);
    outT->Branch("PFJet_energy",&PFJetEnergy_);
    outT->Branch("PFJet_energyRaw",&PFJetEnergyRaw_);
    outT->Branch("PFJet_mass",&PFJetMass_);
    outT->Branch("PFJet_pt",&PFJetPt_);
    outT->Branch("PFJet_eta",&PFJetEta_);
    outT->Branch("PFJet_phi",&PFJetPhi_);
    outT->Branch("PFJet_bTag",&PFJetBTag_);
    outT->Branch("PFJet_truth",&PFJetTruth_);
    outT->Branch("PFJet_bTagEffDenomPt",&PFJetEffDenomPt_);
    outT->Branch("PFJet_bTagEffNumPt",&PFJetEffNumPt_);
    outT->Branch("PFJet_METdPhi",&PFJetMETdPhi_);
    outT->Branch("PFJet_CHEF",&PFJetCorrectedCHEF_);
    outT->Branch("PFJet_NHEF",&PFJetCorrectedNHEF_);
    outT->Branch("PFJet_CEEF",&PFJetCorrectedCEEF_);
    outT->Branch("PFJet_NEEF",&PFJetCorrectedNEEF_);
    outT->Branch("PFJet_corrNumDaughters",&PFJetCorrectedNumDaughters_);
    outT->Branch("PFJet_corrCHM",&PFJetCorrectedChargedMultiplicity_);
    outT->Branch("PFJet_corr_pt",&PFJetCorrectedPt_);
    outT->Branch("PFJet_corrJESUp_pt",&PFJetCorrectedJESUpPt_);
    outT->Branch("PFJet_corrJESUp_eta",&PFJetCorrectedJESUpEta_);
    outT->Branch("PFJet_corrJESUp_phi",&PFJetCorrectedJESUpPhi_);
    outT->Branch("PFJet_corrJESDown_pt",&PFJetCorrectedJESDownPt_);
    outT->Branch("PFJet_corrJESDown_eta",&PFJetCorrectedJESDownEta_);
    outT->Branch("PFJet_corrJESDown_phi",&PFJetCorrectedJESDownPhi_);
    outT->Branch("PFJet_corrJERUp_pt",&PFJetCorrectedJERUpPt_);
    outT->Branch("PFJet_corrJERUp_eta",&PFJetCorrectedJERUpEta_);
    outT->Branch("PFJet_corrJERUp_phi",&PFJetCorrectedJERUpPhi_);
    outT->Branch("PFJet_corrJERDown_pt",&PFJetCorrectedJERDownPt_);
    outT->Branch("PFJet_corrJERDown_eta",&PFJetCorrectedJERDownEta_);
    outT->Branch("PFJet_corrJERDown_phi",&PFJetCorrectedJERDownPhi_);
    outT->Branch("HEM_flag",&PFHEMFlag_);
    outT->Branch("PFHT",&PFHT_);
    outT->Branch("PFJetPtLeading",&PFJetPtLeading_);

    // MET
    outT->Branch("PFMET_ET",&PFMET_ET_);
    outT->Branch("PFMET_pt",&PFMET_Pt_);
    outT->Branch("PFMET_phi",&PFMET_Phi_);
    outT->Branch("PFMET_JESUpPt",&PFMETJESUpPt_);
    outT->Branch("PFMET_JESUpPhi",&PFMETJESUpPhi_);
    outT->Branch("PFMET_JESDownPt",&PFMETJESDownPt_);
    outT->Branch("PFMET_JESDownPhi",&PFMETJESDownPhi_);
    outT->Branch("PFMET_JERUpPt",&PFMETJERUpPt_);
    outT->Branch("PFMET_JERUpPhi",&PFMETJERUpPhi_);
    outT->Branch("PFMET_JERDownPt",&PFMETJERDownPt_);
    outT->Branch("PFMET_JERDownPhi",&PFMETJERDownPhi_);
    outT->Branch("PFMET_UnclusteredUpPt",&PFMETUnclusteredUpPt_);
    outT->Branch("PFMET_UnclusteredUpPhi",&PFMETUnclusteredUpPhi_);
    outT->Branch("PFMET_UnclusteredDownPt",&PFMETUnclusteredDownPt_);
    outT->Branch("PFMET_UnclusteredDownPhi",&PFMETUnclusteredDownPhi_);
    outT->Branch("PFMET_JetResUpSmearPt",&PFMETJetResUpSmearPt_);
    outT->Branch("PFMET_JetResUpSmearPhi",&PFMETJetResUpSmearPhi_);
    outT->Branch("PFMET_JetResDownSmearPt",&PFMETJetResDownSmearPt_);
    outT->Branch("PFMET_JetResDownSmearPhi",&PFMETJetResDownSmearPhi_);

    outT->Branch("CaloMET_ET",&CaloMET_ET_);
    outT->Branch("CaloMET_pt",&CaloMET_Pt_);
    outT->Branch("CaloMET_phi",&CaloMET_Phi_);
    
    // Pileup density
    outT->Branch("rho",&rho_);
    outT->Branch("fixedGridRhoFastjetAll",&fixedGridRhoFastjetAll_);

    // PV
    outT->Branch("PV_x",&PV_x_);
    outT->Branch("PV_y",&PV_y_);
    outT->Branch("PV_z",&PV_z_);
    outT->Branch("numPV",&numPV_);

    // Electron-positron vertex branches
    outT->Branch("nvtx",&nvtx_);
    outT->Branch("vtx_typ",&vtx_type_);
    outT->Branch("vtx_vxy", &vtx_recoVtxVxy_);
    outT->Branch("vtx_sigmavxy", &vtx_recoVtxSigmaVxy_);
    outT->Branch("vtx_vx",&vtx_recoVtxVx_);
    outT->Branch("vtx_vy",&vtx_recoVtxVy_);
    outT->Branch("vtx_vz",&vtx_recoVtxVz_);
    outT->Branch("vtx_reduced_chi2", &vtx_recoVtxReducedChi2_);
    outT->Branch("vtx_prob", &vtx_prob_);
    outT->Branch("vtx_dR",  &vtx_recoVtxDr_);
    outT->Branch("vtx_sign",&vtx_recoVtxSign_);
    outT->Branch("vtx_minDxy",&vtx_minDxy_);
    outT->Branch("vtx_sharedSC",&vtx_sharedSC_);
    outT->Branch("vtx_sharedSeed",&vtx_sharedSeed_);
    outT->Branch("vtx_dPtRel",&vtx_dPtRel_);
    outT->Branch("vtx_dDxy",&vtx_dDxy_);
    outT->Branch("vtx_dDz",&vtx_dDz_);
    outT->Branch("vtx_sharedCtf",&vtx_sharedCtf_);
    outT->Branch("vtx_e1_ctfGsfOverlap",&vtx_e1_ctfGsfOverlap_);
    outT->Branch("vtx_e2_ctfGsfOverlap",&vtx_e2_ctfGsfOverlap_);
    outT->Branch("vtx_METdPhi",&vtx_METdPhi_);
    outT->Branch("vtx_pt",&vtx_ll_pt_);
    outT->Branch("vtx_eta",&vtx_ll_eta_);
    outT->Branch("vtx_phi",&vtx_ll_phi_);
    outT->Branch("vtx_energy",&vtx_ll_e_);
    outT->Branch("vtx_m",&vtx_ll_m_);
    outT->Branch("vtx_px",&vtx_ll_px_);
    outT->Branch("vtx_py",&vtx_ll_py_);
    outT->Branch("vtx_pz",&vtx_ll_pz_);
    outT->Branch("vtx_refit_m",&vtx_refit_m_);
    outT->Branch("vtx_refit_pt",&vtx_refit_pt_);
    outT->Branch("vtx_refit_eta",&vtx_refit_eta_);
    outT->Branch("vtx_refit_phi",&vtx_refit_phi_);
    outT->Branch("vtx_refit_dR",&vtx_refit_dr_);
    outT->Branch("vtx_isMatched",&vtx_isMatched_);
    outT->Branch("vtx_matchSign",&vtx_matchSign_);
    outT->Branch("vtx_dRJets",&vtx_dRtoJets_);
    outT->Branch("vtx_dPhiJets",&vtx_dPhiToJets_);

    outT->Branch("vtx_e1_typ",&vtx_e1_type_);
    outT->Branch("vtx_e1_idx",&vtx_e1_idx_);
    outT->Branch("vtx_e1_isMatched",&vtx_e1_isMatched_);
    outT->Branch("vtx_e1_matchType",&vtx_e1_matchType_);
    outT->Branch("vtx_e1_refit_dxy",&vtx_e1_refitDxy_);
    outT->Branch("vtx_e1_refit_dxyErr",&vtx_e1_refitDxyErr_);
    outT->Branch("vtx_e1_refit_dz",&vtx_e1_refitDz_);
    outT->Branch("vtx_e1_refit_dzErr",&vtx_e1_refitDzErr_);
    outT->Branch("vtx_e1_refit_chi2",&vtx_e1_refitChi2_);

    outT->Branch("vtx_e2_typ",&vtx_e2_type_);
    outT->Branch("vtx_e2_idx",&vtx_e2_idx_);
    outT->Branch("vtx_e2_isMatched",&vtx_e2_isMatched_);
    outT->Branch("vtx_e2_matchType",&vtx_e2_matchType_);
    outT->Branch("vtx_e2_refit_dxy",&vtx_e2_refitDxy_);
    outT->Branch("vtx_e2_refit_dxyErr",&vtx_e2_refitDxyErr_);
    outT->Branch("vtx_e2_refit_dz",&vtx_e2_refitDz_);
    outT->Branch("vtx_e2_refit_dzErr",&vtx_e2_refitDzErr_);
    outT->Branch("vtx_e2_refit_chi2",&vtx_e2_refitChi2_);

    // LowPt-only Electron-positron vertex branches
    outT->Branch("nlptvtx",&nlptvtx_);
    outT->Branch("lptvtx_typ",&lptvtx_type_);
    outT->Branch("lptvtx_vxy", &lptvtx_recoVtxVxy_);
    outT->Branch("lptvtx_sigmavxy", &lptvtx_recoVtxSigmaVxy_);
    outT->Branch("lptvtx_vx",&lptvtx_recoVtxVx_);
    outT->Branch("lptvtx_vy",&lptvtx_recoVtxVy_);
    outT->Branch("lptvtx_vz",&lptvtx_recoVtxVz_);
    outT->Branch("lptvtx_reduced_chi2", &lptvtx_recoVtxReducedChi2_);
    outT->Branch("lptvtx_prob", &lptvtx_prob_);
    outT->Branch("lptvtx_dR",  &lptvtx_recoVtxDr_);
    outT->Branch("lptvtx_sign",&lptvtx_recoVtxSign_);
    outT->Branch("lptvtx_minDxy",&lptvtx_minDxy_);
    outT->Branch("lptvtx_sharedSC",&lptvtx_sharedSC_);
    outT->Branch("lptvtx_sharedSeed",&lptvtx_sharedSeed_);
    outT->Branch("lptvtx_dPtRel",&lptvtx_dPtRel_);
    outT->Branch("lptvtx_dDxy",&lptvtx_dDxy_);
    outT->Branch("lptvtx_dDz",&lptvtx_dDz_);
    outT->Branch("lptvtx_sharedCtf",&lptvtx_sharedCtf_);
    outT->Branch("lptvtx_e1_ctfGsfOverlap",&lptvtx_e1_ctfGsfOverlap_);
    outT->Branch("lptvtx_e2_ctfGsfOverlap",&lptvtx_e2_ctfGsfOverlap_);
    outT->Branch("lptvtx_METdPhi",&lptvtx_METdPhi_);
    outT->Branch("lptvtx_pt",&lptvtx_ll_pt_);
    outT->Branch("lptvtx_eta",&lptvtx_ll_eta_);
    outT->Branch("lptvtx_phi",&lptvtx_ll_phi_);
    outT->Branch("lptvtx_energy",&lptvtx_ll_e_);
    outT->Branch("lptvtx_m",&lptvtx_ll_m_);
    outT->Branch("lptvtx_px",&lptvtx_ll_px_);
    outT->Branch("lptvtx_py",&lptvtx_ll_py_);
    outT->Branch("lptvtx_pz",&lptvtx_ll_pz_);
    outT->Branch("lptvtx_refit_m",&lptvtx_refit_m_);
    outT->Branch("lptvtx_refit_pt",&lptvtx_refit_pt_);
    outT->Branch("lptvtx_refit_eta",&lptvtx_refit_eta_);
    outT->Branch("lptvtx_refit_phi",&lptvtx_refit_phi_);
    outT->Branch("lptvtx_refit_dR",&lptvtx_refit_dr_);
    outT->Branch("lptvtx_isMatched",&lptvtx_isMatched_);
    outT->Branch("lptvtx_matchSign",&lptvtx_matchSign_);
    outT->Branch("lptvtx_dRJets",&lptvtx_dRtoJets_);
    outT->Branch("lptvtx_dPhiJets",&lptvtx_dPhiToJets_);

    outT->Branch("lptvtx_e1_typ",&lptvtx_e1_type_);
    outT->Branch("lptvtx_e1_idx",&lptvtx_e1_idx_);
    outT->Branch("lptvtx_e1_isMatched",&lptvtx_e1_isMatched_);
    outT->Branch("lptvtx_e1_matchType",&lptvtx_e1_matchType_);
    outT->Branch("lptvtx_e1_refit_dxy",&lptvtx_e1_refitDxy_);
    outT->Branch("lptvtx_e1_refit_dxyErr",&lptvtx_e1_refitDxyErr_);
    outT->Branch("lptvtx_e1_refit_dz",&lptvtx_e1_refitDz_);
    outT->Branch("lptvtx_e1_refit_dzErr",&lptvtx_e1_refitDzErr_);
    outT->Branch("lptvtx_e1_refit_chi2",&lptvtx_e1_refitChi2_);

    outT->Branch("lptvtx_e2_typ",&lptvtx_e2_type_);
    outT->Branch("lptvtx_e2_idx",&lptvtx_e2_idx_);
    outT->Branch("lptvtx_e2_isMatched",&lptvtx_e2_isMatched_);
    outT->Branch("lptvtx_e2_matchType",&lptvtx_e2_matchType_);
    outT->Branch("lptvtx_e2_refit_dxy",&lptvtx_e2_refitDxy_);
    outT->Branch("lptvtx_e2_refit_dxyErr",&lptvtx_e2_refitDxyErr_);
    outT->Branch("lptvtx_e2_refit_dz",&lptvtx_e2_refitDz_);
    outT->Branch("lptvtx_e2_refit_dzErr",&lptvtx_e2_refitDzErr_);
    outT->Branch("lptvtx_e2_refit_chi2",&lptvtx_e2_refitChi2_);

    // Gen information
    if (!isData_) {
        outT->Branch("genWgt", &genwgt_);
        outT->Branch("genPU_obs", &genpuobs_);
        outT->Branch("genPU_true", &genputrue_);
        outT->Branch("nGenJet",&nGenJet_);
        outT->Branch("GenJet_pt", &genJetPt_);
        outT->Branch("GenJet_eta", &genJetEta_);
        outT->Branch("GenJet_phi", &genJetPhi_);
        outT->Branch("GenJet_METdPhi",&genJetMETdPhi_);
        outT->Branch("GenMET_pt", &genLeadMETPt_);
        outT->Branch("GenMET_phi", &genLeadMETPhi_);
        outT->Branch("GenMET_px",&genLeadMETPx_);
        outT->Branch("GenMET_py",&genLeadMETPy_);
        outT->Branch("GenMET_ET",&genLeadMETET_);
    }
    if (!isData_) {
        // Authoritative complete gen-particle collection. GenLepton_* below is
        // the all-lepton view linked back by genParticleIdx.
        outT->Branch("nGenPart",&nGen_);
        outT->Branch("GenPart_ID",&genID_);
        outT->Branch("GenPart_motherID",&genMotherID_);
        outT->Branch("GenPart_firstDifferentMotherID",&genPartFirstDifferentMotherID_);
        outT->Branch("GenPart_status",&genPartStatus_);
        outT->Branch("GenPart_charge",&genCharge_);
        outT->Branch("GenPart_pt",&genPt_);
        outT->Branch("GenPart_eta",&genEta_);
        outT->Branch("GenPart_phi",&genPhi_);
        outT->Branch("GenPart_e",&genEn_);
        outT->Branch("GenPart_px",&genPx_);
        outT->Branch("GenPart_py",&genPy_);
        outT->Branch("GenPart_pz",&genPz_);
        outT->Branch("GenPart_vxy",&genVxy_);
        outT->Branch("GenPart_vx",&genVx_);
        outT->Branch("GenPart_vy",&genVy_);
        outT->Branch("GenPart_vz",&genVz_);
        outT->Branch("GenPart_mass",&genMass_);
        outT->Branch("GenPart_isFirstCopy",&genPartIsFirstCopy_);
        outT->Branch("GenPart_isLastCopy",&genPartIsLastCopy_);
        outT->Branch("GenPart_isLastCopyBeforeFSR",&genPartIsLastCopyBeforeFSR_);
        outT->Branch("GenPart_isHardProcess",&genPartIsHardProcess_);
        outT->Branch("GenPart_fromHardProcessFinalState",&genPartFromHardProcessFinalState_);
        outT->Branch("GenPart_fromHardProcessBeforeFSR",&genPartFromHardProcessBeforeFSR_);
        outT->Branch("GenPart_isPromptFinalState",&genPartIsPromptFinalState_);

        // All gen-lepton entries, including charged leptons and neutrinos.
        outT->Branch("nGenLepton",&nGenLepton_);
        outT->Branch("GenLepton_genParticleIdx",&genLeptonGenParticleIdx_);
        outT->Branch("GenLepton_ID",&genLeptonID_);
        outT->Branch("GenLepton_motherID",&genLeptonMotherID_);
        outT->Branch("GenLepton_firstDifferentMotherID",&genLeptonFirstDifferentMotherID_);
        outT->Branch("GenLepton_status",&genLeptonStatus_);
        outT->Branch("GenLepton_charge",&genLeptonCharge_);
        branchP4("GenLepton_p4",&genLeptonP4_);
        outT->Branch("GenLepton_vxy",&genLeptonVxy_);
        outT->Branch("GenLepton_vx",&genLeptonVx_);
        outT->Branch("GenLepton_vy",&genLeptonVy_);
        outT->Branch("GenLepton_vz",&genLeptonVz_);
        outT->Branch("GenLepton_isFirstCopy",&genLeptonIsFirstCopy_);
        outT->Branch("GenLepton_isLastCopy",&genLeptonIsLastCopy_);
        outT->Branch("GenLepton_isLastCopyBeforeFSR",&genLeptonIsLastCopyBeforeFSR_);
        outT->Branch("GenLepton_isHardProcess",&genLeptonIsHardProcess_);
        outT->Branch("GenLepton_fromHardProcessFinalState",&genLeptonFromHardProcessFinalState_);
        outT->Branch("GenLepton_fromHardProcessBeforeFSR",&genLeptonFromHardProcessBeforeFSR_);
        outT->Branch("GenLepton_isPromptFinalState",&genLeptonIsPromptFinalState_);
        outT->Branch("GenLepton_isSignal",&genLeptonIsSignal_);
        outT->Branch("GenLepton_propSt2Status",&genLeptonPropSt2Status_);
        outT->Branch("GenLepton_propSt2Idx",&genLeptonPropSt2Idx_);

        outT->Branch("nPropGenLeptonSt2",&nPropGenLeptonSt2_);
        outT->Branch("PropGenLeptonSt2_genLeptonIdx",&propGenLeptonSt2GenLeptonIdx_);
        branchP4("PropGenLeptonSt2_p4",&propGenLeptonSt2P4_);
        outT->Branch("PropGenLeptonSt2_positionEta",&propGenLeptonSt2PositionEta_);
        outT->Branch("PropGenLeptonSt2_positionPhi",&propGenLeptonSt2PositionPhi_);

        if (isSignal_) {
            outT->Branch("nGenSigLepton",&nGenSigLepton_);
            outT->Branch("GenSigLepton_genLeptonIdx",&genSigLeptonGenLeptonIdx_);
            outT->Branch("GenSigLepton_propSt2Idx",&genSigLeptonPropSt2Idx_);

            outT->Branch("nPropGenSigLeptonSt2",&nPropGenSigLeptonSt2_);
            outT->Branch(
                "PropGenSigLeptonSt2_genSigLeptonIdx",
                &propGenSigLeptonSt2GenSigLeptonIdx_
            );
            outT->Branch(
                "PropGenSigLeptonSt2_genLeptonIdx",
                &propGenSigLeptonSt2GenLeptonIdx_
            );
            branchP4("PropGenSigLeptonSt2_p4",&propGenSigLeptonSt2P4_);
            outT->Branch(
                "PropGenSigLeptonSt2_positionEta",
                &propGenSigLeptonSt2PositionEta_
            );
            outT->Branch(
                "PropGenSigLeptonSt2_positionPhi",
                &propGenSigLeptonSt2PositionPhi_
            );

            outT->Branch("GenEle_charge",&genEleCharge_);
            outT->Branch("GenEle_motherID",&genEleMotherID_);
            outT->Branch("GenEle_pt",&genElePt_);
            outT->Branch("GenEle_eta",&genEleEta_);
            outT->Branch("GenEle_phi",&genElePhi_);
            outT->Branch("GenEle_energy",&genEleEn_);
            outT->Branch("GenEle_px",&genElePx_);
            outT->Branch("GenEle_py",&genElePy_);
            outT->Branch("GenEle_pz",&genElePz_);
            outT->Branch("GenEle_vxy",&genEleVxy_);
            outT->Branch("GenEle_vz",&genEleVz_);
            outT->Branch("GenEle_vx",&genEleVx_);
            outT->Branch("GenEle_vy",&genEleVy_);
            outT->Branch("GenEle_matched",&genEleMatched_);
            outT->Branch("GenEle_matchType",&genEleMatchType_);
            outT->Branch("GenEle_matchIdxLocal",&genEleMatchIdxLocal_);
            outT->Branch("GenEle_matchIdxGlobal",&genEleMatchIdxGlobal_);
            outT->Branch("GenEle_matchedAllLowPt",&genEleMatchedAllLowPt_);
            outT->Branch("GenEle_matchIdxAllLowPt",&genEleMatchIdxAllLowPt_);

            outT->Branch("GenEle_GEDPtRes",&genEleGEDPtRes_);
            outT->Branch("GenEle_GEDPtResRel",&genEleGEDPtResRel_);
            outT->Branch("GenEle_LptPtRes",&genEleLptPtRes_);
            outT->Branch("GenEle_LptPtResRel",&genEleLptPtResRel_);

            outT->Branch("GenPos_charge",&genPosCharge_);
            outT->Branch("GenPos_motherID",&genPosMotherID_);
            outT->Branch("GenPos_pt",&genPosPt_);
            outT->Branch("GenPos_eta",&genPosEta_);
            outT->Branch("GenPos_phi",&genPosPhi_);
            outT->Branch("GenPos_energy",&genPosEn_);
            outT->Branch("GenPos_px",&genPosPx_);
            outT->Branch("GenPos_py",&genPosPy_);
            outT->Branch("GenPos_pz",&genPosPz_);
            outT->Branch("GenPos_vxy",&genPosVxy_);
            outT->Branch("GenPos_vz",&genPosVz_);
            outT->Branch("GenPos_vx",&genPosVx_);
            outT->Branch("GenPos_vy",&genPosVy_);
            outT->Branch("GenPos_matched",&genPosMatched_);
            outT->Branch("GenPos_matchType",&genPosMatchType_);
            outT->Branch("GenPos_matchIdxLocal",&genPosMatchIdxLocal_);
            outT->Branch("GenPos_matchIdxGlobal",&genPosMatchIdxGlobal_);
            outT->Branch("GenPos_matchedAllLowPt",&genPosMatchedAllLowPt_);
            outT->Branch("GenPos_matchIdxAllLowPt",&genPosMatchIdxAllLowPt_);

            outT->Branch("GenPos_GEDPtRes",&genPosGEDPtRes_);
            outT->Branch("GenPos_GEDPtResRel",&genPosGEDPtResRel_);
            outT->Branch("GenPos_LptPtRes",&genPosLptPtRes_);
            outT->Branch("GenPos_LptPtResRel",&genPosLptPtResRel_);

            // Signal reco info
            outT->Branch("signalReconstructed",&signalReconstructed_);
            outT->Branch("signalVertexReconstructed",&signalVertexReconstructed_);

            outT->Branch("genEE_pt",&genEEPt_);
            outT->Branch("genEE_eta",&genEEEta_);
            outT->Branch("genEE_phi",&genEEPhi_);
            outT->Branch("genEE_energy",&genEEEn_);
            outT->Branch("genEE_mass",&genEEMass_);
            outT->Branch("genEE_dr",&genEEdR_);
            outT->Branch("genEE_METdPhi",&genEEMETdPhi_);
            outT->Branch("genEE_vxy",&genEEVxy_);
            outT->Branch("genEE_vz",&genEEVz_);
            outT->Branch("genEE_vx",&genEEVx_);
            outT->Branch("genEE_vy",&genEEVy_);

            // Signal muon and anti-muon: status == 1, last copy, first different mother == chi2.
            outT->Branch("nGenSigMuonFinal",&nGenSigMuonFinal_);
            outT->Branch("GenSigMuon_isValid",&genSigMuonIsValid_);
            outT->Branch("GenSigAntiMuon_isValid",&genSigAntiMuonIsValid_);
            outT->Branch("GenSigDimuon_isValid",&genSigDimuonIsValid_);
            outT->Branch("GenSigMuon_charge",&genSigMuonCharge_);
            outT->Branch("GenSigMuon_motherID",&genSigMuonMotherID_);
            outT->Branch("GenSigMuon_status",&genSigMuonStatus_);
            outT->Branch("GenSigMuon_pt",&genSigMuonPt_);
            outT->Branch("GenSigMuon_eta",&genSigMuonEta_);
            outT->Branch("GenSigMuon_phi",&genSigMuonPhi_);
            outT->Branch("GenSigMuon_energy",&genSigMuonEn_);
            outT->Branch("GenSigMuon_mass",&genSigMuonMass_);
            outT->Branch("GenSigMuon_px",&genSigMuonPx_);
            outT->Branch("GenSigMuon_py",&genSigMuonPy_);
            outT->Branch("GenSigMuon_pz",&genSigMuonPz_);
            outT->Branch("GenSigMuon_vxy",&genSigMuonVxy_);
            outT->Branch("GenSigMuon_vz",&genSigMuonVz_);
            outT->Branch("GenSigMuon_vx",&genSigMuonVx_);
            outT->Branch("GenSigMuon_vy",&genSigMuonVy_);
            outT->Branch("GenSigMuon_matched",&genSigMuonMatched_);
            outT->Branch("GenSigMuon_matchType",&genSigMuonMatchType_);
            outT->Branch("GenSigMuon_matchIdxLocal",&genSigMuonMatchIdxLocal_);
            outT->Branch("GenSigMuon_matchIdxGlobal",&genSigMuonMatchIdxGlobal_);
            outT->Branch("GenSigMuon_immediateMotherID",&genSigMuonImmediateMotherID_);
            outT->Branch("GenSigMuon_firstDifferentMotherID",&genSigMuonFirstDifferentMotherID_);
            outT->Branch("GenSigMuon_isFirstCopy",&genSigMuonIsFirstCopy_);
            outT->Branch("GenSigMuon_isLastCopy",&genSigMuonIsLastCopy_);
            outT->Branch("GenSigMuon_isLastCopyBeforeFSR",&genSigMuonIsLastCopyBeforeFSR_);
            outT->Branch("GenSigMuon_isHardProcess",&genSigMuonIsHardProcess_);
            outT->Branch("GenSigMuon_fromHardProcessFinalState",&genSigMuonFromHardProcessFinalState_);
            outT->Branch("GenSigMuon_fromHardProcessBeforeFSR",&genSigMuonFromHardProcessBeforeFSR_);
            outT->Branch("GenSigMuon_isPromptFinalState",&genSigMuonIsPromptFinalState_);
            outT->Branch("GenSigMuon_minDrToRecoMuon",&genSigMuonMinDrToRecoMuon_);
            outT->Branch("GenSigMuon_matchRecoMuonIdx",&genSigMuonMatchRecoMuonIdx_);
            outT->Branch("GenSigMuon_minDrToSTAMuon",&genSigMuonMinDrToSTAMuon_);
            outT->Branch("GenSigMuon_matchSTAMuonIdx",&genSigMuonMatchSTAMuonIdx_);
            outT->Branch("GenSigMuon_minDrToDSAMuon",&genSigMuonMinDrToDSAMuon_);
            outT->Branch("GenSigMuon_matchDSAMuonIdx",&genSigMuonMatchDSAMuonIdx_);

             outT->Branch("GenSigMuon_propSt1Valid", &genSigMuonPropSt1Valid_);
            outT->Branch("GenSigMuon_propSt1Eta", &genSigMuonPropSt1Eta_);
            outT->Branch("GenSigMuon_propSt1Phi", &genSigMuonPropSt1Phi_);
            outT->Branch("GenSigMuon_propSt1MomEta", &genSigMuonPropSt1MomEta_);
            outT->Branch("GenSigMuon_propSt1MomPhi", &genSigMuonPropSt1MomPhi_);
            branchP4("GenSigMuon_propSt1P4", &genSigMuonPropSt1P4_, false);  // flat pt/eta/phi (or position eta/phi) already branched

            outT->Branch("GenSigMuon_propSt2Valid", &genSigMuonPropSt2Valid_);
            outT->Branch("GenSigMuon_propSt2Eta", &genSigMuonPropSt2Eta_);
            outT->Branch("GenSigMuon_propSt2Phi", &genSigMuonPropSt2Phi_);
            outT->Branch("GenSigMuon_propSt2MomEta", &genSigMuonPropSt2MomEta_);
            outT->Branch("GenSigMuon_propSt2MomPhi", &genSigMuonPropSt2MomPhi_);
            branchP4("GenSigMuon_propSt2P4", &genSigMuonPropSt2P4_, false);  // flat pt/eta/phi (or position eta/phi) already branched

            outT->Branch("GenSigMuon_propSt3Valid", &genSigMuonPropSt3Valid_);
            outT->Branch("GenSigMuon_propSt3Eta", &genSigMuonPropSt3Eta_);
            outT->Branch("GenSigMuon_propSt3Phi", &genSigMuonPropSt3Phi_);
            outT->Branch("GenSigMuon_propSt3MomEta", &genSigMuonPropSt3MomEta_);
            outT->Branch("GenSigMuon_propSt3MomPhi", &genSigMuonPropSt3MomPhi_);
            branchP4("GenSigMuon_propSt3P4", &genSigMuonPropSt3P4_, false);  // flat pt/eta/phi (or position eta/phi) already branched

            outT->Branch("GenSigMuon_propSt4Valid", &genSigMuonPropSt4Valid_);
            outT->Branch("GenSigMuon_propSt4Eta", &genSigMuonPropSt4Eta_);
            outT->Branch("GenSigMuon_propSt4Phi", &genSigMuonPropSt4Phi_);
            outT->Branch("GenSigMuon_propSt4MomEta", &genSigMuonPropSt4MomEta_);
            outT->Branch("GenSigMuon_propSt4MomPhi", &genSigMuonPropSt4MomPhi_);
            branchP4("GenSigMuon_propSt4P4", &genSigMuonPropSt4P4_, false);  // flat pt/eta/phi (or position eta/phi) already branched

            outT->Branch("GenSigMuon_minDrToSTAMuonPropSt1", &genSigMuonMinDrToSTAMuonPropSt1_);
            outT->Branch("GenSigMuon_matchSTAMuonPropSt1Idx", &genSigMuonMatchSTAMuonPropSt1Idx_);
            outT->Branch("GenSigMuon_minDrToSTAMuonPropSt2", &genSigMuonMinDrToSTAMuonPropSt2_);
            outT->Branch("GenSigMuon_matchSTAMuonPropSt2Idx", &genSigMuonMatchSTAMuonPropSt2Idx_);

            outT->Branch("GenSigMuon_minDrToDSAMuonPropSt1", &genSigMuonMinDrToDSAMuonPropSt1_);
            outT->Branch("GenSigMuon_matchDSAMuonPropSt1Idx", &genSigMuonMatchDSAMuonPropSt1Idx_);
            outT->Branch("GenSigMuon_minDrToDSAMuonPropSt2", &genSigMuonMinDrToDSAMuonPropSt2_);
            outT->Branch("GenSigMuon_matchDSAMuonPropSt2Idx", &genSigMuonMatchDSAMuonPropSt2Idx_);
            outT->Branch("GenSigMuon_minDrToDSAMuonPropSt3", &genSigMuonMinDrToDSAMuonPropSt3_);
            outT->Branch("GenSigMuon_matchDSAMuonPropSt3Idx", &genSigMuonMatchDSAMuonPropSt3Idx_);
            outT->Branch("GenSigMuon_minDrToDSAMuonPropSt4", &genSigMuonMinDrToDSAMuonPropSt4_);
            outT->Branch("GenSigMuon_matchDSAMuonPropSt4Idx", &genSigMuonMatchDSAMuonPropSt4Idx_);
            outT->Branch("GenSigMuon_minDrToDSAMuonOuterHit", &genSigMuonMinDrToDSAMuonOuterHit_);
            outT->Branch("GenSigMuon_matchDSAMuonOuterHitIdx", &genSigMuonMatchDSAMuonOuterHitIdx_);

            outT->Branch("GenSigAntiMuon_charge",&genSigAntiMuonCharge_);
            outT->Branch("GenSigAntiMuon_motherID",&genSigAntiMuonMotherID_);
            outT->Branch("GenSigAntiMuon_status",&genSigAntiMuonStatus_);
            outT->Branch("GenSigAntiMuon_pt",&genSigAntiMuonPt_);
            outT->Branch("GenSigAntiMuon_eta",&genSigAntiMuonEta_);
            outT->Branch("GenSigAntiMuon_phi",&genSigAntiMuonPhi_);
            outT->Branch("GenSigAntiMuon_energy",&genSigAntiMuonEn_);
            outT->Branch("GenSigAntiMuon_mass",&genSigAntiMuonMass_);
            outT->Branch("GenSigAntiMuon_px",&genSigAntiMuonPx_);
            outT->Branch("GenSigAntiMuon_py",&genSigAntiMuonPy_);
            outT->Branch("GenSigAntiMuon_pz",&genSigAntiMuonPz_);
            outT->Branch("GenSigAntiMuon_vxy",&genSigAntiMuonVxy_);
            outT->Branch("GenSigAntiMuon_vz",&genSigAntiMuonVz_);
            outT->Branch("GenSigAntiMuon_vx",&genSigAntiMuonVx_);
            outT->Branch("GenSigAntiMuon_vy",&genSigAntiMuonVy_);
            outT->Branch("GenSigAntiMuon_matched",&genSigAntiMuonMatched_);
            outT->Branch("GenSigAntiMuon_matchType",&genSigAntiMuonMatchType_);
            outT->Branch("GenSigAntiMuon_matchIdxLocal",&genSigAntiMuonMatchIdxLocal_);
            outT->Branch("GenSigAntiMuon_matchIdxGlobal",&genSigAntiMuonMatchIdxGlobal_);
            outT->Branch("GenSigAntiMuon_immediateMotherID",&genSigAntiMuonImmediateMotherID_);
            outT->Branch("GenSigAntiMuon_firstDifferentMotherID",&genSigAntiMuonFirstDifferentMotherID_);
            outT->Branch("GenSigAntiMuon_isFirstCopy",&genSigAntiMuonIsFirstCopy_);
            outT->Branch("GenSigAntiMuon_isLastCopy",&genSigAntiMuonIsLastCopy_);
            outT->Branch("GenSigAntiMuon_isLastCopyBeforeFSR",&genSigAntiMuonIsLastCopyBeforeFSR_);
            outT->Branch("GenSigAntiMuon_isHardProcess",&genSigAntiMuonIsHardProcess_);
            outT->Branch("GenSigAntiMuon_fromHardProcessFinalState",&genSigAntiMuonFromHardProcessFinalState_);
            outT->Branch("GenSigAntiMuon_fromHardProcessBeforeFSR",&genSigAntiMuonFromHardProcessBeforeFSR_);
            outT->Branch("GenSigAntiMuon_isPromptFinalState",&genSigAntiMuonIsPromptFinalState_);
            outT->Branch("GenSigAntiMuon_minDrToRecoMuon",&genSigAntiMuonMinDrToRecoMuon_);
            outT->Branch("GenSigAntiMuon_matchRecoMuonIdx",&genSigAntiMuonMatchRecoMuonIdx_);
            outT->Branch("GenSigAntiMuon_minDrToSTAMuon",&genSigAntiMuonMinDrToSTAMuon_);
            outT->Branch("GenSigAntiMuon_matchSTAMuonIdx",&genSigAntiMuonMatchSTAMuonIdx_);
            outT->Branch("GenSigAntiMuon_minDrToDSAMuon",&genSigAntiMuonMinDrToDSAMuon_);
            outT->Branch("GenSigAntiMuon_matchDSAMuonIdx",&genSigAntiMuonMatchDSAMuonIdx_);


            outT->Branch("GenSigAntiMuon_propSt1Valid", &genSigAntiMuonPropSt1Valid_);
            outT->Branch("GenSigAntiMuon_propSt1Eta", &genSigAntiMuonPropSt1Eta_);
            outT->Branch("GenSigAntiMuon_propSt1Phi", &genSigAntiMuonPropSt1Phi_);
            outT->Branch("GenSigAntiMuon_propSt1MomEta", &genSigAntiMuonPropSt1MomEta_);
            outT->Branch("GenSigAntiMuon_propSt1MomPhi", &genSigAntiMuonPropSt1MomPhi_);
            branchP4("GenSigAntiMuon_propSt1P4", &genSigAntiMuonPropSt1P4_, false);  // flat pt/eta/phi (or position eta/phi) already branched

            outT->Branch("GenSigAntiMuon_propSt2Valid", &genSigAntiMuonPropSt2Valid_);
            outT->Branch("GenSigAntiMuon_propSt2Eta", &genSigAntiMuonPropSt2Eta_);
            outT->Branch("GenSigAntiMuon_propSt2Phi", &genSigAntiMuonPropSt2Phi_);
            outT->Branch("GenSigAntiMuon_propSt2MomEta", &genSigAntiMuonPropSt2MomEta_);
            outT->Branch("GenSigAntiMuon_propSt2MomPhi", &genSigAntiMuonPropSt2MomPhi_);
            branchP4("GenSigAntiMuon_propSt2P4", &genSigAntiMuonPropSt2P4_, false);  // flat pt/eta/phi (or position eta/phi) already branched

            outT->Branch("GenSigAntiMuon_propSt3Valid", &genSigAntiMuonPropSt3Valid_);
            outT->Branch("GenSigAntiMuon_propSt3Eta", &genSigAntiMuonPropSt3Eta_);
            outT->Branch("GenSigAntiMuon_propSt3Phi", &genSigAntiMuonPropSt3Phi_);
            outT->Branch("GenSigAntiMuon_propSt3MomEta", &genSigAntiMuonPropSt3MomEta_);
            outT->Branch("GenSigAntiMuon_propSt3MomPhi", &genSigAntiMuonPropSt3MomPhi_);
            branchP4("GenSigAntiMuon_propSt3P4", &genSigAntiMuonPropSt3P4_, false);  // flat pt/eta/phi (or position eta/phi) already branched

            outT->Branch("GenSigAntiMuon_propSt4Valid", &genSigAntiMuonPropSt4Valid_);
            outT->Branch("GenSigAntiMuon_propSt4Eta", &genSigAntiMuonPropSt4Eta_);
            outT->Branch("GenSigAntiMuon_propSt4Phi", &genSigAntiMuonPropSt4Phi_);
            outT->Branch("GenSigAntiMuon_propSt4MomEta", &genSigAntiMuonPropSt4MomEta_);
            outT->Branch("GenSigAntiMuon_propSt4MomPhi", &genSigAntiMuonPropSt4MomPhi_);
            branchP4("GenSigAntiMuon_propSt4P4", &genSigAntiMuonPropSt4P4_, false);  // flat pt/eta/phi (or position eta/phi) already branched

            outT->Branch("GenSigAntiMuon_minDrToSTAMuonPropSt1", &genSigAntiMuonMinDrToSTAMuonPropSt1_);
            outT->Branch("GenSigAntiMuon_matchSTAMuonPropSt1Idx", &genSigAntiMuonMatchSTAMuonPropSt1Idx_);
            outT->Branch("GenSigAntiMuon_minDrToSTAMuonPropSt2", &genSigAntiMuonMinDrToSTAMuonPropSt2_);
            outT->Branch("GenSigAntiMuon_matchSTAMuonPropSt2Idx", &genSigAntiMuonMatchSTAMuonPropSt2Idx_);

            outT->Branch("GenSigAntiMuon_minDrToDSAMuonPropSt1", &genSigAntiMuonMinDrToDSAMuonPropSt1_);
            outT->Branch("GenSigAntiMuon_matchDSAMuonPropSt1Idx", &genSigAntiMuonMatchDSAMuonPropSt1Idx_);
            outT->Branch("GenSigAntiMuon_minDrToDSAMuonPropSt2", &genSigAntiMuonMinDrToDSAMuonPropSt2_);
            outT->Branch("GenSigAntiMuon_matchDSAMuonPropSt2Idx", &genSigAntiMuonMatchDSAMuonPropSt2Idx_);
            outT->Branch("GenSigAntiMuon_minDrToDSAMuonPropSt3", &genSigAntiMuonMinDrToDSAMuonPropSt3_);
            outT->Branch("GenSigAntiMuon_matchDSAMuonPropSt3Idx", &genSigAntiMuonMatchDSAMuonPropSt3Idx_);
            outT->Branch("GenSigAntiMuon_minDrToDSAMuonPropSt4", &genSigAntiMuonMinDrToDSAMuonPropSt4_);
            outT->Branch("GenSigAntiMuon_matchDSAMuonPropSt4Idx", &genSigAntiMuonMatchDSAMuonPropSt4Idx_);
            outT->Branch("GenSigAntiMuon_minDrToDSAMuonOuterHit", &genSigAntiMuonMinDrToDSAMuonOuterHit_);
            outT->Branch("GenSigAntiMuon_matchDSAMuonOuterHitIdx", &genSigAntiMuonMatchDSAMuonOuterHitIdx_);

            outT->Branch("genSigDimuon_pt",&genSigDimuonPt_);
            outT->Branch("genSigDimuon_eta",&genSigDimuonEta_);
            outT->Branch("genSigDimuon_phi",&genSigDimuonPhi_);
            outT->Branch("genSigDimuon_energy",&genSigDimuonEn_);
            outT->Branch("genSigDimuon_mass",&genSigDimuonMass_);
            outT->Branch("genSigDimuon_dr",&genSigDimuonDr_);
            outT->Branch("genSigDimuon_METdPhi",&genSigDimuonMETdPhi_);
            outT->Branch("genSigDimuon_vxy",&genSigDimuonVxy_);
            outT->Branch("genSigDimuon_vz",&genSigDimuonVz_);
            outT->Branch("genSigDimuon_vx",&genSigDimuonVx_);
            outT->Branch("genSigDimuon_vy",&genSigDimuonVy_);
            outT->Branch("signalDimuonReconstructed",&signalDimuonReconstructed_);
        }
    }


    // Dimuon pair candidates (mumuvtx), fixed vertexing component and signal truth labels (mumusig);
    // filled by AODSkimmer/plugins/MuMuVtx.h. Legs index RecoDisplacedMuon (local positions).
    outT->Branch("mumuvtxAvailable", &mumuvtxAvailable_);
    outT->Branch("nmumuvtx", &nmumuvtx_);
    outT->Branch("mumuvtx_m1_idx", &mumuvtx_m1_idx_);
    outT->Branch("mumuvtx_m2_idx", &mumuvtx_m2_idx_);
    outT->Branch("mumuvtx_m1_typ", &mumuvtx_m1_typ_);
    outT->Branch("mumuvtx_m2_typ", &mumuvtx_m2_typ_);
    outT->Branch("mumuvtx_m1_charge", &mumuvtx_m1_charge_);
    outT->Branch("mumuvtx_m2_charge", &mumuvtx_m2_charge_);
    outT->Branch("mumuvtx_status", &mumuvtx_status_);
    outT->Branch("mumuvtx_chi2", &mumuvtx_chi2_);
    outT->Branch("mumuvtx_ndof", &mumuvtx_ndof_);
    outT->Branch("mumuvtx_prob", &mumuvtx_prob_);
    outT->Branch("mumuvtx_vx", &mumuvtx_vx_);
    outT->Branch("mumuvtx_vy", &mumuvtx_vy_);
    outT->Branch("mumuvtx_vz", &mumuvtx_vz_);
    outT->Branch("mumuvtx_lxyBS", &mumuvtx_lxyBS_);
    outT->Branch("mumuvtx_mass", &mumuvtx_mass_);
    outT->Branch("mumuvtx_pcaOK", &mumuvtx_pcaOK_);
    outT->Branch("mumuvtx_pcaDR", &mumuvtx_pcaDR_);
    outT->Branch("mumuvtx_pcaMuMuPt", &mumuvtx_pcaMuMuPt_);
    outT->Branch("mumuvtx_pcaMuMuPhi", &mumuvtx_pcaMuMuPhi_);
    outT->Branch("mumuvtx_m1_pcaPt", &mumuvtx_m1_pcaPt_);
    outT->Branch("mumuvtx_m1_pcaEta", &mumuvtx_m1_pcaEta_);
    outT->Branch("mumuvtx_m1_pcaPhi", &mumuvtx_m1_pcaPhi_);
    outT->Branch("mumuvtx_m2_pcaPt", &mumuvtx_m2_pcaPt_);
    outT->Branch("mumuvtx_m2_pcaEta", &mumuvtx_m2_pcaEta_);
    outT->Branch("mumuvtx_m2_pcaPhi", &mumuvtx_m2_pcaPhi_);
    outT->Branch("RecoDisplacedMuon_fixedComp", &recoDisplacedMuonFixedComp_);
    outT->Branch("RecoDisplacedMuon_mumusigIdx", &recoDisplacedMuonMumusigIdx_);
    outT->Branch("RecoDisplacedMuon_innerTrack_idPass", &recoDisplacedMuonInnerTrackIdPass_);
    outT->Branch("RecoDisplacedMuon_outerTrack_idPass", &recoDisplacedMuonOuterTrackIdPass_);
    outT->Branch("RecoDisplacedMuon_globalTrack_idPass", &recoDisplacedMuonGlobalTrackIdPass_);
    outT->Branch("RecoDisplacedMuon_innerTrack_inEta", &recoDisplacedMuonInnerTrackInEta_);
    outT->Branch("RecoDisplacedMuon_outerTrack_inEta", &recoDisplacedMuonOuterTrackInEta_);
    outT->Branch("RecoDisplacedMuon_globalTrack_inEta", &recoDisplacedMuonGlobalTrackInEta_);
    outT->Branch("nmumusig", &nmumusig_);
    outT->Branch("mumusig_genIdx", &mumusig_genIdx_);
    outT->Branch("mumusig_charge", &mumusig_charge_);
    outT->Branch("mumusig_accepted", &mumusig_accepted_);
    outT->Branch("mumusigTwoAcc", &mumusigTwoAcc_);
}

// Mirrors the subset of CreateTreeBranches() that ElectronSkimmer::analyze() fills
// unconditionally, before the ptmiss preselection gate. The electron-count and jet
// summary branches (nElectron, nLptElectron, nAllLptElectron, nPFJet, PFHT,
// PFJetPtLeading) are filled via a lightweight pass for events that fail the
// preselection, and via the full reconstruction otherwise -- see analyze().
void NtupleContainerV2::CreateSlimTreeBranches() {

    outT_slim->Branch("trigFired",&fired_);
    outT_slim->Branch("eventNum", &eventNum_);
    outT_slim->Branch("lumiSec", &lumiSec_);
    outT_slim->Branch("runNum", &runNum_);
    for (int i = 0; i < numTrigs_; i++) {
        TString branchName = TString::Format("trig_%s", trigNames_[i].c_str());
        outT_slim->Branch(branchName,&trigPassed_[i],branchName+"/O");
    }

    outT_slim->Branch("METFiltersFailBits",&METFiltersFailBits_);

    outT_slim->Branch("PFMET_ET",&PFMET_ET_);
    outT_slim->Branch("PFMET_pt",&PFMET_Pt_);
    outT_slim->Branch("PFMET_phi",&PFMET_Phi_);

    // Stored as Float16_t (leaflist "/f") -- these are event-level bookkeeping
    // quantities only (preselection efficiency/normalization), so the reduced
    // mantissa precision (~1e-4 relative) is not analysis-sensitive here.
    outT_slim->Branch("CaloMET_ET",&CaloMET_ET_,"CaloMET_ET/f");
    outT_slim->Branch("CaloMET_pt",&CaloMET_Pt_,"CaloMET_pt/f");
    outT_slim->Branch("CaloMET_phi",&CaloMET_Phi_,"CaloMET_phi/f");

    outT_slim->Branch("fixedGridRhoFastjetAll",&fixedGridRhoFastjetAll_,"fixedGridRhoFastjetAll/f");

    outT_slim->Branch("nElectron",&nElectronDefault_);
    outT_slim->Branch("nLptElectron",&nElectronLowPt_);
    outT_slim->Branch("nAllLptElectron",&nElectronAllLowPt_);
    outT_slim->Branch("nMuon",&nMuon_);
    outT_slim->Branch("nDSAMuon",&nDSAMuon_);
    outT_slim->Branch("nPFJet",&PFNJet_);
    outT_slim->Branch("PFHT",&PFHT_,"PFHT/f");
    outT_slim->Branch("PFJetPtLeading",&PFJetPtLeading_,"PFJetPtLeading/f");

    if (!isData_) {
        outT_slim->Branch("genWgt", &genwgt_);
        outT_slim->Branch("genPU_obs", &genpuobs_);
        outT_slim->Branch("genPU_true", &genputrue_);
    }
}

void NtupleContainerV2::ClearTreeBranches() {
    fired_ = 0;
    for (int i = 0; i < numTrigs_; i++) {
        trigPassed_[i] = false;
    }

    METFiltersFailBits_ = 0;

    // Gen particles

    nGen_ = 0;
    genID_.clear();
    genMotherID_.clear();
    genPartFirstDifferentMotherID_.clear();
    genPartStatus_.clear();
    genCharge_.clear();
    genPt_.clear();
    genEta_.clear();
    genPhi_.clear();
    genEn_.clear();
    genPx_.clear();
    genPy_.clear();
    genPz_.clear();
    genVxy_.clear();
    genVx_.clear();
    genVy_.clear();
    genVz_.clear();
    genMass_.clear();
    genPartIsFirstCopy_.clear();
    genPartIsLastCopy_.clear();
    genPartIsLastCopyBeforeFSR_.clear();
    genPartIsHardProcess_.clear();
    genPartFromHardProcessFinalState_.clear();
    genPartFromHardProcessBeforeFSR_.clear();
    genPartIsPromptFinalState_.clear();

    nGenLepton_ = 0;
    genLeptonGenParticleIdx_.clear();
    genLeptonID_.clear();
    genLeptonMotherID_.clear();
    genLeptonFirstDifferentMotherID_.clear();
    genLeptonStatus_.clear();
    genLeptonCharge_.clear();
    genLeptonP4_.clear();
    genLeptonVxy_.clear();
    genLeptonVx_.clear();
    genLeptonVy_.clear();
    genLeptonVz_.clear();
    genLeptonIsFirstCopy_.clear();
    genLeptonIsLastCopy_.clear();
    genLeptonIsLastCopyBeforeFSR_.clear();
    genLeptonIsHardProcess_.clear();
    genLeptonFromHardProcessFinalState_.clear();
    genLeptonFromHardProcessBeforeFSR_.clear();
    genLeptonIsPromptFinalState_.clear();
    genLeptonIsSignal_.clear();
    genLeptonPropSt2Status_.clear();
    genLeptonPropSt2Idx_.clear();

    nPropGenLeptonSt2_ = 0;
    propGenLeptonSt2GenLeptonIdx_.clear();
    propGenLeptonSt2P4_.clear();
    propGenLeptonSt2PositionEta_.clear();
    propGenLeptonSt2PositionPhi_.clear();

    nGenSigLepton_ = 0;
    genSigLeptonGenLeptonIdx_.clear();
    genSigLeptonPropSt2Idx_.clear();

    nPropGenSigLeptonSt2_ = 0;
    propGenSigLeptonSt2GenSigLeptonIdx_.clear();
    propGenSigLeptonSt2GenLeptonIdx_.clear();
    propGenSigLeptonSt2P4_.clear();
    propGenSigLeptonSt2PositionEta_.clear();
    propGenSigLeptonSt2PositionPhi_.clear();

    // Gen Signal Muon
    nGenSigMuonFinal_ = 0;
    genSigMuonIsValid_ = false;
    genSigAntiMuonIsValid_ = false;
    genSigDimuonIsValid_ = false;
    genSigMuonMotherID_ = 0;
    genSigMuonStatus_ = 0;
    genSigMuonCharge_ = 0;
    genSigMuonPt_ = -999;
    genSigMuonEta_ = -999;
    genSigMuonPhi_ = -999;
    genSigMuonEn_ = -999;
    genSigMuonMass_ = -999;
    genSigMuonPx_ = -999;
    genSigMuonPy_ = -999;
    genSigMuonPz_ = -999;
    genSigMuonVxy_ = -999;
    genSigMuonVz_ = -999;
    genSigMuonVx_ = -999;
    genSigMuonVy_ = -999;
    genSigMuonMatched_ = false;
    genSigMuonMatchType_ = "None";
    genSigMuonMatchIdxLocal_ = -999;
    genSigMuonMatchIdxGlobal_ = -999;
    genSigMuonImmediateMotherID_ = 0;
    genSigMuonFirstDifferentMotherID_ = 0;
    genSigMuonIsFirstCopy_ = 0;
    genSigMuonIsLastCopy_ = 0;
    genSigMuonIsLastCopyBeforeFSR_ = 0;
    genSigMuonIsHardProcess_ = 0;
    genSigMuonFromHardProcessFinalState_ = 0;
    genSigMuonFromHardProcessBeforeFSR_ = 0;
    genSigMuonIsPromptFinalState_ = 0;
    genSigMuonMinDrToRecoMuon_ = 999.0;
    genSigMuonMatchRecoMuonIdx_ = -1;
    genSigMuonMinDrToSTAMuon_ = 999.0;
    genSigMuonMatchSTAMuonIdx_ = -1;
    genSigMuonMinDrToDSAMuon_ = 999.0;
    genSigMuonMatchDSAMuonIdx_ = -1;

    genSigMuonPropSt1Valid_ = 0;
    genSigMuonPropSt1Eta_ = -999;
    genSigMuonPropSt1Phi_ = -999;
    genSigMuonPropSt1MomEta_ = -999;
    genSigMuonPropSt1MomPhi_ = -999;
    genSigMuonPropSt1P4_ = math::XYZTLorentzVector(0., 0., 0., 0.);

    genSigMuonPropSt2Valid_ = 0;
    genSigMuonPropSt2Eta_ = -999;
    genSigMuonPropSt2Phi_ = -999;
    genSigMuonPropSt2MomEta_ = -999;
    genSigMuonPropSt2MomPhi_ = -999;
    genSigMuonPropSt2P4_ = math::XYZTLorentzVector(0., 0., 0., 0.);

    genSigMuonPropSt3Valid_ = 0;
    genSigMuonPropSt3Eta_ = -999;
    genSigMuonPropSt3Phi_ = -999;
    genSigMuonPropSt3MomEta_ = -999;
    genSigMuonPropSt3MomPhi_ = -999;
    genSigMuonPropSt3P4_ = math::XYZTLorentzVector(0., 0., 0., 0.);

    genSigMuonPropSt4Valid_ = 0;
    genSigMuonPropSt4Eta_ = -999;
    genSigMuonPropSt4Phi_ = -999;
    genSigMuonPropSt4MomEta_ = -999;
    genSigMuonPropSt4MomPhi_ = -999;
    genSigMuonPropSt4P4_ = math::XYZTLorentzVector(0., 0., 0., 0.);

    genSigMuonMinDrToSTAMuonPropSt1_ = 999.0;
    genSigMuonMatchSTAMuonPropSt1Idx_ = -1;
    genSigMuonMinDrToSTAMuonPropSt2_ = 999.0;
    genSigMuonMatchSTAMuonPropSt2Idx_ = -1;

    genSigMuonMinDrToDSAMuonPropSt1_ = 999.0;
    genSigMuonMatchDSAMuonPropSt1Idx_ = -1;
    genSigMuonMinDrToDSAMuonPropSt2_ = 999.0;
    genSigMuonMatchDSAMuonPropSt2Idx_ = -1;
    genSigMuonMinDrToDSAMuonPropSt3_ = 999.0;
    genSigMuonMatchDSAMuonPropSt3Idx_ = -1;
    genSigMuonMinDrToDSAMuonPropSt4_ = 999.0;
    genSigMuonMatchDSAMuonPropSt4Idx_ = -1;
    genSigMuonMinDrToDSAMuonOuterHit_ = 999.0;
    genSigMuonMatchDSAMuonOuterHitIdx_ = -1;

    // Gen Signal Anti-Muon
    genSigAntiMuonMotherID_ = 0;
    genSigAntiMuonStatus_ = 0;
    genSigAntiMuonCharge_ = 0;
    genSigAntiMuonPt_ = -999;
    genSigAntiMuonEta_ = -999;
    genSigAntiMuonPhi_ = -999;
    genSigAntiMuonEn_ = -999;
    genSigAntiMuonMass_ = -999;
    genSigAntiMuonPx_ = -999;
    genSigAntiMuonPy_ = -999;
    genSigAntiMuonPz_ = -999;
    genSigAntiMuonVxy_ = -999;
    genSigAntiMuonVz_ = -999;
    genSigAntiMuonVx_ = -999;
    genSigAntiMuonVy_ = -999;
    genSigAntiMuonMatched_ = false;
    genSigAntiMuonMatchType_ = "None";
    genSigAntiMuonMatchIdxLocal_ = -999;
    genSigAntiMuonMatchIdxGlobal_ = -999;
    genSigAntiMuonImmediateMotherID_ = 0;
    genSigAntiMuonFirstDifferentMotherID_ = 0;
    genSigAntiMuonIsFirstCopy_ = 0;
    genSigAntiMuonIsLastCopy_ = 0;
    genSigAntiMuonIsLastCopyBeforeFSR_ = 0;
    genSigAntiMuonIsHardProcess_ = 0;
    genSigAntiMuonFromHardProcessFinalState_ = 0;
    genSigAntiMuonFromHardProcessBeforeFSR_ = 0;
    genSigAntiMuonIsPromptFinalState_ = 0;
    genSigAntiMuonMinDrToRecoMuon_ = 999.0;
    genSigAntiMuonMatchRecoMuonIdx_ = -1;
    genSigAntiMuonMinDrToSTAMuon_ = 999.0;
    genSigAntiMuonMatchSTAMuonIdx_ = -1;
    genSigAntiMuonMinDrToDSAMuon_ = 999.0;
    genSigAntiMuonMatchDSAMuonIdx_ = -1;

    genSigAntiMuonPropSt1Valid_ = 0;
    genSigAntiMuonPropSt1Eta_ = -999;
    genSigAntiMuonPropSt1Phi_ = -999;
    genSigAntiMuonPropSt1MomEta_ = -999;
    genSigAntiMuonPropSt1MomPhi_ = -999;
    genSigAntiMuonPropSt1P4_ = math::XYZTLorentzVector(0., 0., 0., 0.);

    genSigAntiMuonPropSt2Valid_ = 0;
    genSigAntiMuonPropSt2Eta_ = -999;
    genSigAntiMuonPropSt2Phi_ = -999;
    genSigAntiMuonPropSt2MomEta_ = -999;
    genSigAntiMuonPropSt2MomPhi_ = -999;
    genSigAntiMuonPropSt2P4_ = math::XYZTLorentzVector(0., 0., 0., 0.);

    genSigAntiMuonPropSt3Valid_ = 0;
    genSigAntiMuonPropSt3Eta_ = -999;
    genSigAntiMuonPropSt3Phi_ = -999;
    genSigAntiMuonPropSt3MomEta_ = -999;
    genSigAntiMuonPropSt3MomPhi_ = -999;
    genSigAntiMuonPropSt3P4_ = math::XYZTLorentzVector(0., 0., 0., 0.);

    genSigAntiMuonPropSt4Valid_ = 0;
    genSigAntiMuonPropSt4Eta_ = -999;
    genSigAntiMuonPropSt4Phi_ = -999;
    genSigAntiMuonPropSt4MomEta_ = -999;
    genSigAntiMuonPropSt4MomPhi_ = -999;
    genSigAntiMuonPropSt4P4_ = math::XYZTLorentzVector(0., 0., 0., 0.);

    genSigAntiMuonMinDrToSTAMuonPropSt1_ = 999.0;
    genSigAntiMuonMatchSTAMuonPropSt1Idx_ = -1;
    genSigAntiMuonMinDrToSTAMuonPropSt2_ = 999.0;
    genSigAntiMuonMatchSTAMuonPropSt2Idx_ = -1;

    genSigAntiMuonMinDrToDSAMuonPropSt1_ = 999.0;
    genSigAntiMuonMatchDSAMuonPropSt1Idx_ = -1;
    genSigAntiMuonMinDrToDSAMuonPropSt2_ = 999.0;
    genSigAntiMuonMatchDSAMuonPropSt2Idx_ = -1;
    genSigAntiMuonMinDrToDSAMuonPropSt3_ = 999.0;
    genSigAntiMuonMatchDSAMuonPropSt3Idx_ = -1;
    genSigAntiMuonMinDrToDSAMuonPropSt4_ = 999.0;
    genSigAntiMuonMatchDSAMuonPropSt4Idx_ = -1;
    genSigAntiMuonMinDrToDSAMuonOuterHit_ = 999.0;
    genSigAntiMuonMatchDSAMuonOuterHitIdx_ = -1;


    genSigDimuonPt_ = -999;
    genSigDimuonEta_ = -999;
    genSigDimuonPhi_ = -999;
    genSigDimuonEn_ = -999;
    genSigDimuonMass_ = -999;
    genSigDimuonDr_ = -999;
    genSigDimuonMETdPhi_ = -999;
    genSigDimuonVxy_ = -999;
    genSigDimuonVz_ = -999;
    genSigDimuonVx_ = -999;
    genSigDimuonVy_ = -999;
    signalDimuonReconstructed_ = false;

    // Gen Electron & Positron from iDM signal
    genEleCharge_ = 0;
    genEleMotherID_ = 0;
    genElePt_ = -999;
    genEleEta_ = -999;
    genElePhi_ = -999;
    genEleEn_ = -999;
    genElePx_ = -999;
    genElePy_ = -999;
    genElePz_ = -999;
    genEleVxy_ = -999;
    genEleVz_ = -999;
    genEleVx_ = -999;
    genEleVy_ = -999;
    genEleMatched_ = false;
    genEleMatchType_ = "None";
    genEleMatchIdxLocal_ = -999;
    genEleMatchIdxGlobal_ = -999;
    genEleMatchedAllLowPt_ = false;
    genEleMatchIdxAllLowPt_ = -999;

    genPosCharge_ = 0;
    genPosMotherID_ = 0;
    genPosPt_ = -999;
    genPosEta_ = -999;
    genPosPhi_ = -999;
    genPosEn_ = -999;
    genPosPx_ = -999;
    genPosPy_ = -999;
    genPosPz_ = -999;
    genPosVxy_ = -999;
    genPosVz_ = -999;
    genPosVx_ = -999;
    genPosVy_ = -999;
    genPosMatched_ = false;
    genPosMatchType_ = "None";
    genPosMatchIdxLocal_ = -999;
    genPosMatchIdxGlobal_ = -999;
    genPosMatchedAllLowPt_ = false;
    genPosMatchIdxAllLowPt_ = -999;

    genEleGEDPtRes_ = -999;
    genEleGEDPtResRel_ = -999;
    genEleLptPtRes_ = -999;
    genEleLptPtResRel_ = -999;
    genPosGEDPtRes_ = -999;
    genPosGEDPtResRel_ = -999;
    genPosLptPtRes_ = -999;
    genPosLptPtResRel_ = -999;

    // Signal reconstruction info
    signalReconstructed_ = false;
    signalVertexReconstructed_ = false;

    genEEPt_ = -999;
    genEEEta_ = -999;
    genEEPhi_ = -999;
    genEEEn_ = -999;
    genEEMass_ = -999;
    genEEdR_ = -999;
    genEEMETdPhi_ = -999;
    genEEVxy_ = -999;
    genEEVz_ = -999;
    genEEVx_ = -999;
    genEEVy_ = -999;
    
    // Gen jet
    nGenJet_ = 0;
    genJetPt_.clear();
    genJetEta_.clear();
    genJetPhi_.clear();
    genJetMETdPhi_.clear();

    //Gen MET
    genLeadMETPt_ = -9999;
    genLeadMETPhi_ = -9999;
    genLeadMETET_ = -9999;
    genLeadMETPx_ = -9999;
    genLeadMETPy_ = -9999;

    // Muons
    nMuon_ = 0;
    recoMuonPatIdx_.clear();
    recoMuonPt_.clear();
    recoMuonEta_.clear();
    recoMuonPhi_.clear();
    recoMuonEnergy_.clear();
    recoMuonCharge_.clear();
    recoMuonIDcutBasedLoose_.clear();
    recoMuonIDcutBasedMedium_.clear();
    recoMuonIDcutBasedMediumPrompt_.clear();
    recoMuonIDcutBasedTight_.clear();
    recoMuonIsPFMuon_.clear();
    recoMuonIsGlobalMuon_.clear();
    recoMuonIsStandAloneMuon_.clear();
    displacedTrack_.clear();
    displacedGlobalTrack_.clear();
    slimmedMuon_.clear();
    recoDisplacedMuon_.clear();
    recoDisplacedMuonOriginalIndex_.clear();
    nRecoDisplacedMuonRaw_ = -1;
    mumuvtxAvailable_ = 0;
    nmumuvtx_ = 0;
    for (auto* v : {&mumuvtx_m1_idx_, &mumuvtx_m2_idx_, &mumuvtx_m1_typ_, &mumuvtx_m2_typ_, &mumuvtx_m1_charge_,
                    &mumuvtx_m2_charge_, &mumuvtx_status_, &mumuvtx_pcaOK_, &recoDisplacedMuonFixedComp_,
                    &recoDisplacedMuonMumusigIdx_, &recoDisplacedMuonInnerTrackIdPass_,
                    &recoDisplacedMuonOuterTrackIdPass_, &recoDisplacedMuonGlobalTrackIdPass_,
                    &recoDisplacedMuonInnerTrackInEta_, &recoDisplacedMuonOuterTrackInEta_,
                    &recoDisplacedMuonGlobalTrackInEta_, &mumusig_genIdx_, &mumusig_charge_, &mumusig_accepted_})
        v->clear();
    for (auto* v : {&mumuvtx_chi2_, &mumuvtx_ndof_, &mumuvtx_prob_, &mumuvtx_vx_, &mumuvtx_vy_, &mumuvtx_vz_,
                    &mumuvtx_lxyBS_, &mumuvtx_mass_, &mumuvtx_pcaDR_, &mumuvtx_pcaMuMuPt_, &mumuvtx_pcaMuMuPhi_,
                    &mumuvtx_m1_pcaPt_, &mumuvtx_m1_pcaEta_, &mumuvtx_m1_pcaPhi_, &mumuvtx_m2_pcaPt_,
                    &mumuvtx_m2_pcaEta_, &mumuvtx_m2_pcaPhi_})
        v->clear();
    nmumusig_ = 0;
    mumusigTwoAcc_ = 0;
    slimmedDisplacedMuon_.clear();

    nPFMuon_ = 0;
    pfMuonPatIdx_.clear();
    pfMuonP4_.clear();
    pfMuonCharge_.clear();
    pfMuonIDcutBasedLoose_.clear();
    pfMuonIDcutBasedMedium_.clear();
    pfMuonIDcutBasedMediumPrompt_.clear();
    pfMuonIDcutBasedTight_.clear();
    pfMuonIsGlobalMuon_.clear();
    pfMuonIsStandAloneMuon_.clear();
    pfMuonPropagationTrackType_.clear();
    pfMuonTrkNumValidMuonHits_.clear();
    pfMuonTrkNumValidTrackerHits_.clear();
    pfMuonTrkNumValidPixelHits_.clear();
    pfMuonTrkNumValidStripHits_.clear();
    pfMuonNumMatchedStations_.clear();
    pfMuonTrkNumMuonStations_.clear();
    pfMuonTrkNumCSCHits_.clear();
    pfMuonTrkNumDTHits_.clear();
    pfMuonTrkPt_.clear();
    pfMuonTrkEta_.clear();
    pfMuonTrkEtaError_.clear();
    pfMuonTrkPhi_.clear();
    pfMuonTrkPhiError_.clear();
    pfMuonTrkOuterEta_.clear();
    pfMuonTrkOuterPhi_.clear();
    pfMuonTrkNormalizedChi2_.clear();
    pfMuonTrkPtError_.clear();
    pfMuonTrkVxy_.clear();
    pfMuonTrkVz_.clear();
    pfMuonTrkDxy_.clear();
    pfMuonTrkDxyError_.clear();
    pfMuonTrkDz_.clear();
    pfMuonTrkDzError_.clear();
    pfMuonTrkProb_.clear();
    pfMuonPropSt1Status_.clear();
    pfMuonPropSt1Idx_.clear();
    pfMuonPropSt2Status_.clear();
    pfMuonPropSt2Idx_.clear();

    nPropPFMuonSt1_ = 0;
    propPFMuonSt1PFMuonIdx_.clear();
    propPFMuonSt1P4_.clear();
    propPFMuonSt1PositionEta_.clear();
    propPFMuonSt1PositionPhi_.clear();

    nPropPFMuonSt2_ = 0;
    propPFMuonSt2PFMuonIdx_.clear();
    propPFMuonSt2P4_.clear();
    propPFMuonSt2PositionEta_.clear();
    propPFMuonSt2PositionPhi_.clear();
    pfMuonPropSt3Status_.clear();
    pfMuonPropSt3Idx_.clear();
    nPropPFMuonSt3_ = 0;
    propPFMuonSt3PFMuonIdx_.clear();
    propPFMuonSt3P4_.clear();
    propPFMuonSt3PositionEta_.clear();
    propPFMuonSt3PositionPhi_.clear();
    pfMuonPropSt4Status_.clear();
    pfMuonPropSt4Idx_.clear();
    nPropPFMuonSt4_ = 0;
    propPFMuonSt4PFMuonIdx_.clear();
    propPFMuonSt4P4_.clear();
    propPFMuonSt4PositionEta_.clear();
    propPFMuonSt4PositionPhi_.clear();

    // Electrons
    nElectronDefault_ = 0;
    recoElectronPt_.clear();
    recoElectronEta_.clear();
    recoElectronEtaError_.clear();
    recoElectronPhi_.clear();
    recoElectronPhiError_.clear();
    recoElectronID_cutVeto_.clear();
    recoElectronID_cutLoose_.clear();
    recoElectronID_cutMed_.clear();
    recoElectronID_cutTight_.clear();
    recoElectronID_cutVetoInt_.clear();
    recoElectronID_cutLooseInt_.clear();
    recoElectronID_cutMedInt_.clear();
    recoElectronID_cutTightInt_.clear();
    recoElectronID_mvaIso90_.clear();
    recoElectronID_mvaIso80_.clear();
    recoElectronID_mvaIsoLoose_.clear();
    recoElectronID_mva90_.clear();
    recoElectronID_mva80_.clear();
    recoElectronID_mvaLoose_.clear();
    recoElectronAngularRes_.clear();
    recoElectronE_.clear();
    recoElectronVxy_.clear();
    recoElectronVz_.clear();
    recoElectronSVProxyValid_.clear();
    recoElectronSVProxyX_.clear();
    recoElectronSVProxyY_.clear();
    recoElectronSVProxyZ_.clear();
    recoElectronSVProxyVxy_.clear();
    recoElectronDxy_.clear();
    recoElectronDxyError_.clear();
    recoElectronDz_.clear();
    recoElectronDzError_.clear();
    recoElectronTrkChi2_.clear();
    recoElectronTrkIso_.clear();
    recoElectronTrkRelIso_.clear();
    recoElectronCaloIso_.clear();
    recoElectronCaloRelIso_.clear();
    recoElectronPFIso_.clear();
    recoElectronPFRelIso_.clear();
    recoElectronMiniIso_.clear();
    recoElectronMiniRelIso_.clear();
    recoElectronPFIsoEleCorr_.clear();
    recoElectronPFRelIsoEleCorr_.clear();
    recoElectronMiniIsoEleCorr_.clear();
    recoElectronMiniRelIsoEleCorr_.clear();
    recoElectronChadIso_.clear();
    recoElectronNhadIso_.clear();
    recoElectronPhoIso_.clear();
    recoElectronRhoEA_.clear();
    recoElectronTrkProb_.clear();
    recoElectronTrkNumTrackerHits_.clear();
    recoElectronTrkNumPixHits_.clear();
    recoElectronTrkNumStripHits_.clear();
    recoElectronCharge_.clear();
    recoElectronIsPF_.clear();
    recoElectronGenMatched_.clear();
    recoElectronMatchType_.clear();
    recoElectronDrToJets_.clear();
    recoElectronDphiToJets_.clear();
    recoElectronFull5x5_sigmaIetaIeta_.clear();
    recoElectronAbsdEtaSeed_.clear();
    recoElectronAbsdPhiIn_.clear();
    recoElectronHoverE_.clear();
    recoElectronAbs1overEm1overP_.clear();
    recoElectronExpMissingInnerHits_.clear();
    recoElectronConversionVeto_.clear();
    recoElectronIsEE_.clear();
    // supercluster
    recoElectronScRawE_.clear();
    recoElectronScEnergy_.clear();
    recoElectronScEta_.clear();
    recoElectronScPhi_.clear();
    recoElectronScEtaWidth_.clear();
    recoElectronScPhiWidth_.clear();
    recoElectronScClustersSize_.clear();
    // additional shower shapes
    recoElectronFull5x5SigmaIetaIphi_.clear();
    recoElectronSeedEoverScRawE_.clear();
    recoElectronFull5x5EMaxOverScRawE_.clear();
    recoElectronFull5x5E2ndOverScRawE_.clear();
    // special vars for x-cleaning study
    recoElectronHasLptMatch_.clear();
    recoElectronLptMatchIdx_.clear();
    recoElectronHasAllLptMatch_.clear();
    recoElectronAllLptMatchIdx_.clear();

    // Low pT electrons
    nElectronLowPt_ = 0;
    recoLowPtElectronPt_.clear();
    recoLowPtElectronEta_.clear();
    recoLowPtElectronEtaError_.clear();
    recoLowPtElectronPhi_.clear();
    recoLowPtElectronPhiError_.clear();
    recoLowPtElectronID_.clear();
    recoLowPtElectronAngularRes_.clear();
    recoLowPtElectronE_.clear();
    recoLowPtElectronVxy_.clear();
    recoLowPtElectronVz_.clear();
    recoLowPtElectronSVProxyValid_.clear();
    recoLowPtElectronSVProxyX_.clear();
    recoLowPtElectronSVProxyY_.clear();
    recoLowPtElectronSVProxyZ_.clear();
    recoLowPtElectronSVProxyVxy_.clear();
    recoLowPtElectronDxy_.clear();
    recoLowPtElectronDxyError_.clear();
    recoLowPtElectronDz_.clear();
    recoLowPtElectronDzError_.clear();
    recoLowPtElectronTrkChi2_.clear();
    recoLowPtElectronTrkIso_.clear();
    recoLowPtElectronTrkRelIso_.clear();
    recoLowPtElectronCaloIso_.clear();
    recoLowPtElectronCaloRelIso_.clear();
    recoLowPtElectronPFIso_.clear();
    recoLowPtElectronPFRelIso_.clear();
    recoLowPtElectronMiniIso_.clear();
    recoLowPtElectronMiniRelIso_.clear();
    recoLowPtElectronPFIsoEleCorr_.clear();
    recoLowPtElectronPFRelIsoEleCorr_.clear();
    recoLowPtElectronMiniIsoEleCorr_.clear();
    recoLowPtElectronMiniRelIsoEleCorr_.clear();
    recoLowPtElectronChadIso_.clear();
    recoLowPtElectronNhadIso_.clear();
    recoLowPtElectronPhoIso_.clear();
    recoLowPtElectronRhoEA_.clear();
    recoLowPtElectronTrkProb_.clear();
    recoLowPtElectronTrkNumTrackerHits_.clear();
    recoLowPtElectronTrkNumPixHits_.clear();
    recoLowPtElectronTrkNumStripHits_.clear();
    recoLowPtElectronCharge_.clear();
    recoLowPtElectronMinDrToReg_.clear();
    recoLowPtElectronIsPF_.clear();
    recoLowPtElectronGenMatched_.clear();
    recoLowPtElectronMatchType_.clear();
    recoLowPtElectronDrToJets_.clear();
    recoLowPtElectronDphiToJets_.clear();
    recoLowPtElectronFull5x5_sigmaIetaIeta_.clear();
    recoLowPtElectronAbsdEtaSeed_.clear();
    recoLowPtElectronAbsdPhiIn_.clear();
    recoLowPtElectronHoverE_.clear();
    recoLowPtElectronAbs1overEm1overP_.clear();
    recoLowPtElectronExpMissingInnerHits_.clear();
    recoLowPtElectronConversionVeto_.clear();
    recoLowPtElectronIsEE_.clear();
    // supercluster
    recoLowPtElectronScRawE_.clear();
    recoLowPtElectronScEnergy_.clear();
    recoLowPtElectronScEta_.clear();
    recoLowPtElectronScPhi_.clear();
    recoLowPtElectronScEtaWidth_.clear();
    recoLowPtElectronScPhiWidth_.clear();
    recoLowPtElectronScClustersSize_.clear();
    // additional shower shapes
    recoLowPtElectronFull5x5R9_.clear();
    recoLowPtElectronFull5x5E5x5_.clear();
    recoLowPtElectronFull5x5E1x5_.clear();
    recoLowPtElectronFull5x5E2x5Max_.clear();
    recoLowPtElectronFull5x5SigmaIphiIphi_.clear();
    recoLowPtElectronFull5x5HcalDepth1OverEcal_.clear();
    recoLowPtElectronFull5x5HcalDepth2OverEcal_.clear();
    recoLowPtElectronFull5x5SigmaIetaIphi_.clear();
    recoLowPtElectronSeedEoverScRawE_.clear();
    recoLowPtElectronFull5x5EMaxOverScRawE_.clear();
    recoLowPtElectronFull5x5E2ndOverScRawE_.clear();
    // special vars for x-cleaning study
    recoLowPtElectronIsXCleaned_.clear();
    recoLowPtElectronGEDidx_.clear();
    recoLowPtElectronGEDisMatched_.clear();

    // Low pT electrons
    nElectronAllLowPt_ = 0;
    recoAllLowPtElectronPt_.clear();
    recoAllLowPtElectronEta_.clear();
    recoAllLowPtElectronEtaError_.clear();
    recoAllLowPtElectronPhi_.clear();
    recoAllLowPtElectronPhiError_.clear();
    recoAllLowPtElectronID_.clear();
    recoAllLowPtElectronAngularRes_.clear();
    recoAllLowPtElectronE_.clear();
    recoAllLowPtElectronVxy_.clear();
    recoAllLowPtElectronVz_.clear();
    recoAllLowPtElectronSVProxyValid_.clear();
    recoAllLowPtElectronSVProxyX_.clear();
    recoAllLowPtElectronSVProxyY_.clear();
    recoAllLowPtElectronSVProxyZ_.clear();
    recoAllLowPtElectronSVProxyVxy_.clear();
    recoAllLowPtElectronDxy_.clear();
    recoAllLowPtElectronDxyError_.clear();
    recoAllLowPtElectronDz_.clear();
    recoAllLowPtElectronDzError_.clear();
    recoAllLowPtElectronTrkChi2_.clear();
    recoAllLowPtElectronTrkIso_.clear();
    recoAllLowPtElectronTrkRelIso_.clear();
    recoAllLowPtElectronCaloIso_.clear();
    recoAllLowPtElectronCaloRelIso_.clear();
    recoAllLowPtElectronPFIso_.clear();
    recoAllLowPtElectronPFRelIso_.clear();
    recoAllLowPtElectronMiniIso_.clear();
    recoAllLowPtElectronMiniRelIso_.clear();
    recoAllLowPtElectronPFIsoEleCorr_.clear();
    recoAllLowPtElectronPFRelIsoEleCorr_.clear();
    recoAllLowPtElectronMiniIsoEleCorr_.clear();
    recoAllLowPtElectronMiniRelIsoEleCorr_.clear();
    recoAllLowPtElectronChadIso_.clear();
    recoAllLowPtElectronNhadIso_.clear();
    recoAllLowPtElectronPhoIso_.clear();
    recoAllLowPtElectronRhoEA_.clear();
    recoAllLowPtElectronTrkProb_.clear();
    recoAllLowPtElectronTrkNumTrackerHits_.clear();
    recoAllLowPtElectronTrkNumPixHits_.clear();
    recoAllLowPtElectronTrkNumStripHits_.clear();
    recoAllLowPtElectronCharge_.clear();
    recoAllLowPtElectronMinDrToReg_.clear();
    recoAllLowPtElectronIsPF_.clear();
    recoAllLowPtElectronGenMatched_.clear();
    recoAllLowPtElectronMatchType_.clear();
    recoAllLowPtElectronDrToJets_.clear();
    recoAllLowPtElectronDphiToJets_.clear();
    recoAllLowPtElectronFull5x5_sigmaIetaIeta_.clear();
    recoAllLowPtElectronAbsdEtaSeed_.clear();
    recoAllLowPtElectronAbsdPhiIn_.clear();
    recoAllLowPtElectronHoverE_.clear();
    recoAllLowPtElectronAbs1overEm1overP_.clear();
    recoAllLowPtElectronExpMissingInnerHits_.clear();
    recoAllLowPtElectronConversionVeto_.clear();
    recoAllLowPtElectronIsEE_.clear();
    // supercluster
    recoAllLowPtElectronScRawE_.clear();
    recoAllLowPtElectronScEnergy_.clear();
    recoAllLowPtElectronScEta_.clear();
    recoAllLowPtElectronScPhi_.clear();
    recoAllLowPtElectronScEtaWidth_.clear();
    recoAllLowPtElectronScPhiWidth_.clear();
    recoAllLowPtElectronScClustersSize_.clear();
    // additional shower shapes
    recoAllLowPtElectronFull5x5R9_.clear();
    recoAllLowPtElectronFull5x5E5x5_.clear();
    recoAllLowPtElectronFull5x5E1x5_.clear();
    recoAllLowPtElectronFull5x5E2x5Max_.clear();
    recoAllLowPtElectronFull5x5SigmaIphiIphi_.clear();
    recoAllLowPtElectronFull5x5HcalDepth1OverEcal_.clear();
    recoAllLowPtElectronFull5x5HcalDepth2OverEcal_.clear();
    recoAllLowPtElectronFull5x5SigmaIetaIphi_.clear();
    recoAllLowPtElectronSeedEoverScRawE_.clear();
    recoAllLowPtElectronFull5x5EMaxOverScRawE_.clear();
    recoAllLowPtElectronFull5x5E2ndOverScRawE_.clear();
    recoAllLowPtElectronIsXCleaned_.clear();
    recoAllLowPtElectronGEDidx_.clear();
    recoAllLowPtElectronGEDisMatched_.clear();

    // Gen weight and pileup
    genwgt_ = 0;
    genpuobs_ = -9999;
    genputrue_ = -9999;

    // Standard standalone (STA) muons
    staMuonAvailable_ = 0;
    nSTAMuon_ = 0;
    recoSTAMuonPt_.clear();
    recoSTAMuonPtErr_.clear();
    recoSTAMuonEta_.clear();
    recoSTAMuonEtaErr_.clear();
    recoSTAMuonPhi_.clear();
    recoSTAMuonPhiErr_.clear();
    recoSTAMuonOuterEta_.clear();
    recoSTAMuonOuterPhi_.clear();
    recoSTAMuonE_.clear();
    recoSTAMuonPx_.clear();
    recoSTAMuonPy_.clear();
    recoSTAMuonPz_.clear();
    recoSTAMuonVxy_.clear();
    recoSTAMuonVz_.clear();
    recoSTAMuonDxy_.clear();
    recoSTAMuonDxyError_.clear();
    recoSTAMuonDz_.clear();
    recoSTAMuonDzError_.clear();
    recoSTAMuonTrkChi2_.clear();
    recoSTAMuonTrkProb_.clear();
    recoSTAMuonTrkNumTrackerHits_.clear();
    recoSTAMuonTrkNumPixHits_.clear();
    recoSTAMuonTrkNumStripHits_.clear();
    recoSTAMuonCharge_.clear();
    recoSTAMuonDisplacedId_.clear();
    recoSTAMuonTrkNumCSCHits_.clear();
    recoSTAMuonTrkNumHits_.clear();
    recoSTAMuonTrkNumPlanes_.clear();
    recoSTAMuonTrkNumDTHits_.clear();
    recoSTAMuonIdx_.clear();
    recoSTAMuonP4_.clear();

    recoSTAMuonPropSt1Valid_.clear();
    recoSTAMuonPropSt1Eta_.clear();
    recoSTAMuonPropSt1Phi_.clear();
    recoSTAMuonPropSt1MomEta_.clear();
    recoSTAMuonPropSt1MomPhi_.clear();
    recoSTAMuonPropSt1Idx_.clear();

    recoSTAMuonPropSt2Valid_.clear();
    recoSTAMuonPropSt2Eta_.clear();
    recoSTAMuonPropSt2Phi_.clear();
    recoSTAMuonPropSt2MomEta_.clear();
    recoSTAMuonPropSt2MomPhi_.clear();
    recoSTAMuonPropSt2Idx_.clear();

    nPropSTAMuonSt1_ = 0;
    propSTAMuonSt1STAMuonIdx_.clear();
    propSTAMuonSt1P4_.clear();
    propSTAMuonSt1PositionEta_.clear();
    propSTAMuonSt1PositionPhi_.clear();

    nPropSTAMuonSt2_ = 0;
    propSTAMuonSt2STAMuonIdx_.clear();
    propSTAMuonSt2P4_.clear();
    propSTAMuonSt2PositionEta_.clear();
    propSTAMuonSt2PositionPhi_.clear();
    recoSTAMuonPropSt3Valid_.clear();
    recoSTAMuonPropSt3Eta_.clear();
    recoSTAMuonPropSt3Phi_.clear();
    recoSTAMuonPropSt3MomEta_.clear();
    recoSTAMuonPropSt3MomPhi_.clear();
    recoSTAMuonPropSt3Idx_.clear();
    nPropSTAMuonSt3_ = 0;
    propSTAMuonSt3STAMuonIdx_.clear();
    propSTAMuonSt3P4_.clear();
    propSTAMuonSt3PositionEta_.clear();
    propSTAMuonSt3PositionPhi_.clear();
    recoSTAMuonPropSt4Valid_.clear();
    recoSTAMuonPropSt4Eta_.clear();
    recoSTAMuonPropSt4Phi_.clear();
    recoSTAMuonPropSt4MomEta_.clear();
    recoSTAMuonPropSt4MomPhi_.clear();
    recoSTAMuonPropSt4Idx_.clear();
    nPropSTAMuonSt4_ = 0;
    propSTAMuonSt4STAMuonIdx_.clear();
    propSTAMuonSt4P4_.clear();
    propSTAMuonSt4PositionEta_.clear();
    propSTAMuonSt4PositionPhi_.clear();

    // DSA muons
    dsaMuonAvailable_ = 0;
    nDSAMuon_ = 0;
    recoDSAMuonPt_.clear();
    recoDSAMuonEta_.clear();
    recoDSAMuonPtErr_.clear();
    recoDSAMuonEtaErr_.clear();
    recoDSAMuonPhi_.clear();
    recoDSAMuonPhiErr_.clear();
    recoDSAMuonOuterEta_.clear();
    recoDSAMuonOuterPhi_.clear();
    recoDSAMuonE_.clear();
    recoDSAMuonPx_.clear();
    recoDSAMuonPy_.clear();
    recoDSAMuonPz_.clear();
    recoDSAMuonVxy_.clear();
    recoDSAMuonVz_.clear();
    recoDSAMuonDxy_.clear();
    recoDSAMuonDxyError_.clear();
    recoDSAMuonDz_.clear();
    recoDSAMuonDzError_.clear();
    recoDSAMuonTrkChi2_.clear();
    recoDSAMuonTrkProb_.clear();
    recoDSAMuonTrkNumTrackerHits_.clear();
    recoDSAMuonTrkNumPixHits_.clear();
    recoDSAMuonTrkNumStripHits_.clear();
    recoDSAMuonCharge_.clear();
    recoDSAMuonDisplacedId_.clear();
    recoDSAMuonTrkNumCSCHits_.clear();
    recoDSAMuonTrkNumHits_.clear();
    recoDSAMuonTrkNumPlanes_.clear();
    recoDSAMuonTrkNumDTHits_.clear();
    recoDSAMuonIdx_.clear();
    recoDSAMuonP4_.clear();


    recoDSAMuonPropSt1Valid_.clear();
    recoDSAMuonPropSt1Eta_.clear();
    recoDSAMuonPropSt1Phi_.clear();
    recoDSAMuonPropSt1MomEta_.clear();
    recoDSAMuonPropSt1MomPhi_.clear();
    recoDSAMuonPropSt1Idx_.clear();

    recoDSAMuonPropSt2Valid_.clear();
    recoDSAMuonPropSt2Eta_.clear();
    recoDSAMuonPropSt2Phi_.clear();
    recoDSAMuonPropSt2MomEta_.clear();
    recoDSAMuonPropSt2MomPhi_.clear();
    recoDSAMuonPropSt2Idx_.clear();

    recoDSAMuonPropSt3Valid_.clear();
    recoDSAMuonPropSt3Eta_.clear();
    recoDSAMuonPropSt3Phi_.clear();
    recoDSAMuonPropSt3MomEta_.clear();
    recoDSAMuonPropSt3MomPhi_.clear();

    recoDSAMuonPropSt4Valid_.clear();
    recoDSAMuonPropSt4Eta_.clear();
    recoDSAMuonPropSt4Phi_.clear();
    recoDSAMuonPropSt4MomEta_.clear();
    recoDSAMuonPropSt4MomPhi_.clear();

    nPropDSAMuonSt1_ = 0;
    propDSAMuonSt1DSAMuonIdx_.clear();
    propDSAMuonSt1P4_.clear();
    propDSAMuonSt1PositionEta_.clear();
    propDSAMuonSt1PositionPhi_.clear();

    nPropDSAMuonSt2_ = 0;
    propDSAMuonSt2DSAMuonIdx_.clear();
    propDSAMuonSt2P4_.clear();
    propDSAMuonSt2PositionEta_.clear();
    propDSAMuonSt2PositionPhi_.clear();
    recoDSAMuonPropSt3Idx_.clear();
    nPropDSAMuonSt3_ = 0;
    propDSAMuonSt3DSAMuonIdx_.clear();
    propDSAMuonSt3P4_.clear();
    propDSAMuonSt3PositionEta_.clear();
    propDSAMuonSt3PositionPhi_.clear();
    recoDSAMuonPropSt4Idx_.clear();
    nPropDSAMuonSt4_ = 0;
    propDSAMuonSt4DSAMuonIdx_.clear();
    propDSAMuonSt4P4_.clear();
    propDSAMuonSt4PositionEta_.clear();
    propDSAMuonSt4PositionPhi_.clear();

    recoDSAMuonOuterHitValid_.clear();
    recoDSAMuonOuterHitTrackExtraAvailable_.clear();
    recoDSAMuonOuterHitRecHitsSize_.clear();
    recoDSAMuonOuterHitNValidMuonHits_.clear();
    recoDSAMuonOuterHitDetId_.clear();
    recoDSAMuonOuterHitX_.clear();
    recoDSAMuonOuterHitY_.clear();
    recoDSAMuonOuterHitZ_.clear();
    recoDSAMuonOuterHitR_.clear();
    recoDSAMuonOuterHitEta_.clear();
    recoDSAMuonOuterHitPhi_.clear();

    recoDSAMuonMinDrToGenSigMuonOuterHit_.clear();
    recoDSAMuonMatchGenSigMuonOuterHitIdx_.clear();
    recoDSAMuonMatchGenSigMuonOuterHitPdgId_.clear();
    recoDSAMuonMatchGenSigMuonOuterHitPropValid_.clear();
    recoDSAMuonMatchGenSigMuonOuterHitPropEta_.clear();
    recoDSAMuonMatchGenSigMuonOuterHitPropPhi_.clear();
    recoDSAMuonMatchGenSigMuonOuterHitPropMomEta_.clear();
    recoDSAMuonMatchGenSigMuonOuterHitPropMomPhi_.clear();


    // Photons
    nPhotons_ = 0;
    PhotonEt_.clear();
    PhotonEta_.clear();
    PhotonPhi_.clear();
    PhotonPt_.clear();

    // OOT Photons
    nOOTPhotons_ = 0;
    ootPhotonEt_.clear();
    ootPhotonEta_.clear();
    ootPhotonPhi_.clear();
    ootPhotonPt_.clear();

    // Isolated Tracks
    nIsoTrack_ = 0;
    isoTrackPt_.clear();
    isoTrackEta_.clear();
    isoTrackPhi_.clear();

    // PF Candidates
    nPFCand_ = 0;
    pfCandPt_.clear();
    pfCandEta_.clear();
    pfCandPhi_.clear();

    // Lost Tracks
    nLostTrack_ = 0;
    lostTrackPt_.clear();
    lostTrackEta_.clear();
    lostTrackPhi_.clear();

    // Photon conversions
    nConversions_ = 0;
    conversionPt_.clear();
    conversionEta_.clear();
    conversionPhi_.clear();
    conversionE_.clear();
    conversionPx_.clear();
    conversionPy_.clear();
    conversionPz_.clear();
    conversionVxy_.clear();
    conversionVz_.clear();
    conversionX_.clear();
    conversionY_.clear();
    conversionZ_.clear();
    conversionLxy_.clear();
    conversionLz_.clear();
    conversionLxyPV_.clear();
    conversionLzPV_.clear();
    conversionDxy_.clear();
    conversionDz_.clear();
    conversionDxyPV_.clear();
    conversionDzPV_.clear();
    conversionEoverP_.clear();
    conversionEoverPrefit_.clear();
    conversionNSharedHits_.clear();
    conversionM_.clear();
    conversionDr_.clear();
    conversionChi2_.clear();
    conversion_Trk1nHitsVtx_.clear();
    conversion_Trk1Pt_.clear();
    conversion_Trk1Eta_.clear();
    conversion_Trk1Phi_.clear();
    conversion_Trk1Chi2_.clear();
    conversion_Trk1NValidHits_.clear();
    conversion_Trk1numLostHits_.clear();
    conversion_Trk1dxy_.clear();
    conversion_Trk1dxyBS_.clear();
    conversion_Trk1dxyPV_.clear();
    conversion_Trk1dz_.clear();
    conversion_Trk1dzPV_.clear();
    conversion_Trk2nHitsVtx_.clear();
    conversion_Trk2Pt_.clear();
    conversion_Trk2Eta_.clear();
    conversion_Trk2Phi_.clear();
    conversion_Trk2Chi2_.clear();
    conversion_Trk2NValidHits_.clear();
    conversion_Trk2numLostHits_.clear();
    conversion_Trk2dxy_.clear();
    conversion_Trk2dxyBS_.clear();
    conversion_Trk2dxyPV_.clear();
    conversion_Trk2dz_.clear();
    conversion_Trk2dzPV_.clear();

    // Jets
    PFNJet_ = 0;
    PFNJetAll_ = 0;
    PFNbJetTrue_ = 0;
    PFNbJetTagged_ = 0;
    PFJet_matchedGenJetPt_.clear();
    PFJet_matchedGenJetEta_.clear();
    PFJet_matchedGenJetPhi_.clear();
    PFJetRawFactor_.clear();
    PFJetMassRaw_.clear();
    PFJetPtRaw_.clear();
    PFJetArea_.clear();
    PFJetMass_.clear();
    PFJetEnergy_.clear();
    PFJetEnergyRaw_.clear();
    PFJetPt_.clear();
    PFJetEta_.clear();
    PFJetPhi_.clear();
    PFJetBTag_.clear();
    PFJetTruth_.clear();
    PFJetEffDenomPt_.clear();
    PFJetEffNumPt_.clear();
    PFJetMETdPhi_.clear();
    PFJetCorrectedCHEF_.clear();
    PFJetCorrectedNHEF_.clear();
    PFJetCorrectedCEEF_.clear();
    PFJetCorrectedNEEF_.clear();
    PFJetCorrectedNumDaughters_.clear();
    PFJetCorrectedChargedMultiplicity_.clear();
    PFJetCorrectedPt_.clear();
    PFJetCorrectedJESUpPt_.clear();
    PFJetCorrectedJESUpEta_.clear();
    PFJetCorrectedJESUpPhi_.clear();
    PFJetCorrectedJESDownPt_.clear();
    PFJetCorrectedJESDownEta_.clear();
    PFJetCorrectedJESDownPhi_.clear();
    PFJetCorrectedJERUpPt_.clear();
    PFJetCorrectedJERUpEta_.clear();
    PFJetCorrectedJERUpPhi_.clear();
    PFJetCorrectedJERDownPt_.clear();
    PFJetCorrectedJERDownEta_.clear();
    PFJetCorrectedJERDownPhi_.clear();
    PFHEMFlag_ = false;
    PFHT_ = 0;
    PFJetPtLeading_ = -999;

    // MET 
    PFMET_ET_ = -999;
    PFMET_Pt_ = -999;
    PFMET_Phi_ = -999;
    PFMETJESUpPt_ = -9999;
    PFMETJESUpPhi_ = -9999;
    PFMETJESDownPt_ = -9999;
    PFMETJESDownPhi_ = -9999;
    PFMETJERUpPt_ = -9999;
    PFMETJERUpPhi_ = -9999;
    PFMETJERDownPt_ = -9999;
    PFMETJERDownPhi_ = -9999;
    PFMETUnclusteredUpPt_ = -9999;
    PFMETUnclusteredUpPhi_ = -9999;
    PFMETUnclusteredDownPt_ = -9999;
    PFMETUnclusteredDownPhi_ = -9999;
    PFMETJetResUpSmearPt_ = -9999;
    PFMETJetResUpSmearPhi_ = -9999;
    PFMETJetResDownSmearPt_ = -9999;
    PFMETJetResDownSmearPhi_ = -9999;
    
    CaloMET_ET_ = -999;
    CaloMET_Pt_ = -999;
    CaloMET_Phi_ = -999;

    rho_ = -9999;

    fixedGridRhoFastjetAll_ = -999.;

    // PV
    PV_x_ = -999.;
    PV_y_ = -999.;
    PV_z_ = -999.;
    numPV_ = 0;

    // Electron-positron vertices
    nvtx_ = 0;
    vtx_type_.clear();
    vtx_recoVtxVxy_.clear();
    vtx_recoVtxVz_.clear();
    vtx_recoVtxSigmaVxy_.clear();
    vtx_recoVtxVx_.clear();
    vtx_recoVtxVy_.clear();
    vtx_recoVtxVz_.clear();
    vtx_recoVtxReducedChi2_.clear();
    vtx_prob_.clear();
    vtx_recoVtxDr_.clear();
    vtx_recoVtxSign_.clear();
    vtx_minDxy_.clear();
    vtx_sharedSC_.clear();
    vtx_sharedSeed_.clear();
    vtx_dPtRel_.clear();
    vtx_dDxy_.clear();
    vtx_dDz_.clear();
    vtx_sharedCtf_.clear();
    vtx_e1_ctfGsfOverlap_.clear();
    vtx_e2_ctfGsfOverlap_.clear();
    vtx_METdPhi_.clear();
    vtx_ll_pt_.clear();
    vtx_ll_eta_.clear();
    vtx_ll_phi_.clear();
    vtx_ll_e_.clear();
    vtx_ll_m_.clear();
    vtx_ll_px_.clear();
    vtx_ll_py_.clear();
    vtx_ll_pz_.clear();
    vtx_refit_m_.clear();
    vtx_refit_pt_.clear();
    vtx_refit_eta_.clear();
    vtx_refit_phi_.clear();
    vtx_refit_dr_.clear();
    vtx_isMatched_.clear();
    vtx_matchSign_.clear();
    vtx_dRtoJets_.clear();
    vtx_dPhiToJets_.clear();

    vtx_e1_type_.clear();
    vtx_e1_idx_.clear();
    vtx_e1_isMatched_.clear();
    vtx_e1_matchType_.clear();
    vtx_e1_refitDxy_.clear();
    vtx_e1_refitDxyErr_.clear();
    vtx_e1_refitDz_.clear();
    vtx_e1_refitDzErr_.clear();
    vtx_e1_refitChi2_.clear();

    vtx_e2_type_.clear();
    vtx_e2_idx_.clear();
    vtx_e2_isMatched_.clear();
    vtx_e2_matchType_.clear();
    vtx_e2_refitDxy_.clear();
    vtx_e2_refitDxyErr_.clear();
    vtx_e2_refitDz_.clear();
    vtx_e2_refitDzErr_.clear();
    vtx_e2_refitChi2_.clear();

    // LowPt-only Electron-positron vertices
    nlptvtx_ = 0;
    lptvtx_type_.clear();
    lptvtx_recoVtxVxy_.clear();
    lptvtx_recoVtxVz_.clear();
    lptvtx_recoVtxSigmaVxy_.clear();
    lptvtx_recoVtxVx_.clear();
    lptvtx_recoVtxVy_.clear();
    lptvtx_recoVtxVz_.clear();
    lptvtx_recoVtxReducedChi2_.clear();
    lptvtx_prob_.clear();
    lptvtx_recoVtxDr_.clear();
    lptvtx_recoVtxSign_.clear();
    lptvtx_minDxy_.clear();
    lptvtx_sharedSC_.clear();
    lptvtx_sharedSeed_.clear();
    lptvtx_dPtRel_.clear();
    lptvtx_dDxy_.clear();
    lptvtx_dDz_.clear();
    lptvtx_sharedCtf_.clear();
    lptvtx_e1_ctfGsfOverlap_.clear();
    lptvtx_e2_ctfGsfOverlap_.clear();
    lptvtx_METdPhi_.clear();
    lptvtx_ll_pt_.clear();
    lptvtx_ll_eta_.clear();
    lptvtx_ll_phi_.clear();
    lptvtx_ll_e_.clear();
    lptvtx_ll_m_.clear();
    lptvtx_ll_px_.clear();
    lptvtx_ll_py_.clear();
    lptvtx_ll_pz_.clear();
    lptvtx_refit_m_.clear();
    lptvtx_refit_pt_.clear();
    lptvtx_refit_eta_.clear();
    lptvtx_refit_phi_.clear();
    lptvtx_refit_dr_.clear();
    lptvtx_isMatched_.clear();
    lptvtx_matchSign_.clear();
    lptvtx_dRtoJets_.clear();
    lptvtx_dPhiToJets_.clear();

    lptvtx_e1_type_.clear();
    lptvtx_e1_idx_.clear();
    lptvtx_e1_isMatched_.clear();
    lptvtx_e1_matchType_.clear();
    lptvtx_e1_refitDxy_.clear();
    lptvtx_e1_refitDxyErr_.clear();
    lptvtx_e1_refitDz_.clear();
    lptvtx_e1_refitDzErr_.clear();
    lptvtx_e1_refitChi2_.clear();

    lptvtx_e2_type_.clear();
    lptvtx_e2_idx_.clear();
    lptvtx_e2_isMatched_.clear();
    lptvtx_e2_matchType_.clear();
    lptvtx_e2_refitDxy_.clear();
    lptvtx_e2_refitDxyErr_.clear();
    lptvtx_e2_refitDz_.clear();
    lptvtx_e2_refitDzErr_.clear();
    lptvtx_e2_refitChi2_.clear();
}
