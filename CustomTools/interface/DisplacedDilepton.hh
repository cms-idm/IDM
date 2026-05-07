#ifndef DISPLACEDDILEPTON_HH
#define DISPLACEDDILEPTON_HH

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonPFIsolation.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/PFIsolation.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/MET.h"

#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"

#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"
#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "iDMe/CustomTools/interface/llCandidate.h"
#include "iDMe/CustomTools/interface/trackPair.h"

#include <string>
#include <iostream>
#include <vector>
#include <algorithm>

#include "TLorentzVector.h"
#include "TTree.h"
#include "TFile.h"
#include "TVector3.h"

using std::vector;

const int nIsoTrackMax = 500;
const int nPhotonMax = 100;
const int nElectronMax = 100;
const int nElectronCandidateMax = 1000;

class DisplacedDileptons {

public:
    // Run3 changed
    DisplacedDileptons(reco::Vertex &pv_, reco::BeamSpot &beamspot_, edm::Handle<std::vector<pat::IsolatedTrack> > &isotracks_, edm::Handle<std::vector<pat::Photon> > &photons_, edm::Handle<std::vector<pat::PackedCandidate> > &packedPFCandidates_, const edm::ESHandle<TransientTrackBuilder> &theTransientTrackBuilder_);
    ~DisplacedDileptons();
    void findDileptons();
    float getDeltaR(float phi1, float eta1, float phi2, float eta2);
    bool passIsotrackSelection(const pat::IsolatedTrack &track);
    bool passPhotonSelection(const pat::Photon &photon);
    bool passBaselineSelection(llCandidate llc);
    bool passBaselineSelection(trackPair ttp);
    float computeRelIso(const reco::Track & track,  edm::Handle<std::vector<pat::PackedCandidate> > & pfs, bool isPF);

    edm::Handle<std::vector<pat::IsolatedTrack> > isoTracks;
    edm::Handle<std::vector<pat::Photon> > photons;
    edm::Handle<std::vector<pat::PackedCandidate> > packedPFCandidates;
    reco::Vertex pv;
    reco::BeamSpot beamspot;
    edm::ESHandle<TransientTrackBuilder> theTransientTrackBuilder;

    //-> PRIMARY VERTEX SELECTION
    int nPV;
    int nTruePV;
    int PV_passAcceptance;
    float PV_vx;
    float PV_vy;
    float PV_vz;

    //-> BEAM SPOT
    float BeamSpot_x0;
    float BeamSpot_y0;
    float BeamSpot_z0;
    float BeamSpot_BeamWidthX;
    float BeamSpot_BeamWidthY;


    //-> ISOTRACK SELECTION
    int nIsoTrack;
    // Primitive:
    float IsoTrackSel_pt[nIsoTrackMax];
    float IsoTrackSel_eta[nIsoTrackMax];
    float IsoTrackSel_etaExtra[nIsoTrackMax];
    float IsoTrackSel_phiExtra[nIsoTrackMax];
    float IsoTrackSel_phi[nIsoTrackMax];
    int IsoTrackSel_charge[nIsoTrackMax];
    float IsoTrackSel_dxy[nIsoTrackMax];
    float IsoTrackSel_dxyError[nIsoTrackMax];
    float IsoTrackSel_dxy_PV[nIsoTrackMax];
    float IsoTrackSel_dxyError_PV[nIsoTrackMax];
    float IsoTrackSel_dxy_0[nIsoTrackMax];
    float IsoTrackSel_dxyError_0[nIsoTrackMax];
    float IsoTrackSel_dxy_BS[nIsoTrackMax];
    float IsoTrackSel_dxyError_BS[nIsoTrackMax];
    float IsoTrackSel_dz[nIsoTrackMax];
    float IsoTrackSel_dzError[nIsoTrackMax];
    float IsoTrackSel_vx[nIsoTrackMax];
    float IsoTrackSel_vy[nIsoTrackMax];
    float IsoTrackSel_vz[nIsoTrackMax];
    float IsoTrackSel_pfIsolationDR03[nIsoTrackMax];
    float IsoTrackSel_miniPFIsolation[nIsoTrackMax];
    float IsoTrackSel_relPfIsolationDR03[nIsoTrackMax];
    float IsoTrackSel_relMiniPFIsolation[nIsoTrackMax];
    int IsoTrackSel_isHighPurityTrack[nIsoTrackMax];
    int IsoTrackSel_numberOfValidTrackerHits[nIsoTrackMax];
    int IsoTrackSel_numberOfValidPixelHits[nIsoTrackMax];
    int IsoTrackSel_numberOfValidPixelBarrelHits[nIsoTrackMax];
    int IsoTrackSel_numberOfValidPixelEndcapHits[nIsoTrackMax];
    int IsoTrackSel_numberOfValidStripHits[nIsoTrackMax];
    int IsoTrackSel_numberOfValidStripTIBHits[nIsoTrackMax];
    int IsoTrackSel_numberOfValidStripTIDHits[nIsoTrackMax];
    int IsoTrackSel_numberOfValidStripTOBHits[nIsoTrackMax];
    int IsoTrackSel_numberOfValidStripTECHits[nIsoTrackMax];
    int IsoTrackSel_fromPV[nIsoTrackMax];
    float IsoTrackSel_PVx[nIsoTrackMax];
    float IsoTrackSel_PVy[nIsoTrackMax];
    float IsoTrackSel_PVz[nIsoTrackMax];

    //-> PHOTON SELECTION
    int nPhoton;
    float PhotonSel_et[nPhotonMax];
    float PhotonSel_eta[nPhotonMax];
    float PhotonSel_phi[nPhotonMax];
    float PhotonSel_hadronicOverEm[nPhotonMax];
    float PhotonSel_full5x5_sigmaIetaIeta[nPhotonMax];
    int PhotonSel_isEB[nPhotonMax];
    int PhotonSel_isEE[nPhotonMax];
    float PhotonSel_r9[nPhotonMax];
    float PhotonSel_ecalIso[nPhotonMax];
    float PhotonSel_hcalIso[nPhotonMax];
    float PhotonSel_caloIso[nPhotonMax];
    float PhotonSel_relIso[nPhotonMax];

    //-> ELECTRON SELECTION
    int nElectron;
    float ElectronSel_pt[nElectronMax];
    float ElectronSel_et[nElectronMax];
    float ElectronSel_eta[nElectronMax];
    float ElectronSel_phi[nElectronMax];
    float ElectronSel_dxy[nElectronMax];
    float ElectronSel_dxyError[nElectronMax];
    float ElectronSel_dxySignificance[nElectronMax];
    float ElectronSel_dB[nElectronMax];
    float ElectronSel_edB[nElectronMax];
    float ElectronSel_isLoose[nElectronMax];
    float ElectronSel_isMedium[nElectronMax];
    float ElectronSel_isTight[nElectronMax];

    //-> ELECTRON CANDIDATE SELECTION
    int nElectronCandidate;
    float ElectronCandidate_pt[nElectronCandidateMax];
    float ElectronCandidate_et[nElectronCandidateMax];
    float ElectronCandidate_eta[nElectronCandidateMax];
    float ElectronCandidate_phi[nElectronCandidateMax];
    float ElectronCandidate_dxy[nElectronCandidateMax];
    float ElectronCandidate_dxyError[nElectronCandidateMax];
    float ElectronCandidate_dxy_PV[nElectronCandidateMax];
    float ElectronCandidate_dxyError_PV[nElectronCandidateMax];
    float ElectronCandidate_dxy_0[nElectronCandidateMax];
    float ElectronCandidate_dxyError_0[nElectronCandidateMax];
    float ElectronCandidate_dxy_BS[nElectronCandidateMax];
    float ElectronCandidate_dxyError_BS[nElectronCandidateMax];
    float ElectronCandidate_relPFiso[nElectronCandidateMax];
    float ElectronCandidate_relTrkiso[nElectronCandidateMax];
    int ElectronCandidate_photonIdx[nElectronCandidateMax];
    int ElectronCandidate_isotrackIdx[nElectronCandidateMax];
    int ElectronCandidate_pvAssociationQuality[nElectronCandidateMax];
    float ElectronCandidate_ptDiff[nElectronCandidateMax];

    // -> All EE candidates
    int nEE;
    int EE_idxA[20];
    int EE_idxB[20];
    float EE_Lxy_PV[20];
    float EE_Ixy_PV[20];
    float EE_Lxy_0[20];
    float EE_Ixy_0[20];
    float EE_Lxy_BS[20];
    float EE_Ixy_BS[20];
    float EE_trackDxy[20];
    float EE_trackIxy[20];
    float EE_trackDxy_PV[20];
    float EE_trackIxy_PV[20];
    float EE_trackDxy_0[20];
    float EE_trackIxy_0[20];
    float EE_trackDxy_BS[20];
    float EE_trackIxy_BS[20];
    float EE_vx[20];
    float EE_vy[20];
    float EE_mass[20];
    float EE_normalizedChi2[20];
    float EE_leadingPt[20];
    float EE_subleadingPt[20];
    float EE_leadingEt[20];
    float EE_subleadingEt[20];
    float EE_cosAlpha[20];
    float EE_dR[20];
    float EE_dPhi[20];
    float EE_lldPhi[20];
    float EE_relisoA[20];
    float EE_relisoB[20];

    // -> EE candidates that survive to baseline selection
    int nEEBase;
    int EEBase_maxIxy;
    int EEBase_idxA[20];
    int EEBase_idxB[20];
    float EEBase_Lxy[20];
    float EEBase_Ixy[20];
    float EEBase_trackDxy[20];
    float EEBase_trackIxy[20];
    float EEBase_vx[20];
    float EEBase_vy[20];
    float EEBase_mass[20];
    float EEBase_normalizedChi2[20];
    float EEBase_leadingPt[20];
    float EEBase_subleadingPt[20];
    float EEBase_leadingEt[20];
    float EEBase_subleadingEt[20];
    float EEBase_cosAlpha[20];
    float EEBase_dPhi[20];
    float EEBase_relisoA[20];
    float EEBase_relisoB[20];
    float EEBase_refittedDxy[20];
    float EEBase_refittedIxy[20];
    int EEBase_fromPVA[20];
    int EEBase_fromPVB[20];
    int EEBase_PVAssociation[20];
};

#endif
