#include "iDMe/CustomTools/interface/DisplacedDilepton.hh"

DisplacedDileptons::DisplacedDileptons(reco::Vertex &pv_, reco::BeamSpot &beamspot_, edm::Handle<std::vector<pat::IsolatedTrack> > &isotracks_, edm::Handle<std::vector<pat::Photon> > &photons_, edm::Handle<std::vector<pat::PackedCandidate> > &packedPFCandidates_, const edm::ESHandle<TransientTrackBuilder> &theTransientTrackBuilder_) {
    isoTracks = isotracks_;
    photons = photons_;
    packedPFCandidates = packedPFCandidates_;
    pv = pv_;
    theTransientTrackBuilder = theTransientTrackBuilder_;
    beamspot = beamspot_;

    PV_vx = pv.x();
    PV_vy = pv.y();
    PV_vz = pv.z();
    PV_passAcceptance = false;
    if (!pv.isFake() && pv.ndof() > 4 && fabs(pv.z()) < 25 && pv.position().rho() <= 2) {
        PV_passAcceptance = true;
    }

    BeamSpot_x0 = beamspot.x0();
    BeamSpot_y0 = beamspot.y0();
    BeamSpot_z0 = beamspot.z0();
    BeamSpot_BeamWidthX = beamspot.BeamWidthX();
    BeamSpot_BeamWidthY = beamspot.BeamWidthY();
}

DisplacedDileptons::~DisplacedDileptons() {}

float DisplacedDileptons::getDeltaR(float phi1, float eta1, float phi2, float eta2) {
    float dPhi = fabs(phi1 - phi2);
    if (dPhi > 3.14) {dPhi = 2*3.14 - dPhi;}
    float dEta = eta1 - eta2;
    float dR = sqrt(dPhi*dPhi + dEta*dEta);
    return dR;
}

bool DisplacedDileptons::passIsotrackSelection(const pat::IsolatedTrack &track) {
    // Quality cuts:
    const reco::HitPattern &hits = track.hitPattern();
    if (hits.numberOfValidTrackerHits() < 6) { return false; }
    if (!track.isHighPurityTrack()) { return false;}
    // Isotrack must have packed candidate:
    const pat::PackedCandidateRef &pckCand = track.packedCandRef();
    if (!pckCand.isNonnull()) { return false; }
    // Preselection cuts:
    if (track.pt() < 5) { return false; }
    if (fabs(track.eta()) > 2.4) { return false; }
    return true;
}

bool DisplacedDileptons::passPhotonSelection(const pat::Photon &photon) {
    // Quality cuts:
    if (photon.hadronicOverEm() > 0.05) { return false; } 
    if (photon.isEE() && photon.full5x5_sigmaIetaIeta() > 0.0425) { return false; }
    if (photon.isEB() && photon.full5x5_sigmaIetaIeta() > 0.0112) { return false; }
    // Preselection cuts:
    if (photon.et() < 3) {return false; }
    return true;
}

bool DisplacedDileptons::passBaselineSelection(llCandidate llc) {
    // Electron selection:
    if (llc.type == 0) {

        if ( llc.leadingPt < 5 ) { return false; }
        if ( llc.subleadingPt < 5 ) { return false; }
        if ( llc.leadingEt < 5 ) { return false; }
        if ( llc.subleadingEt < 5 ) { return false; }
        if ( fabs(llc.etaA) > 1.442 || fabs(llc.etaB) > 1.442 ) { return false; }
        if ( fabs(llc.relisoA) > 0.2 || fabs(llc.relisoB) > 0.2 ) { return false; }
        if ( llc.normalizedChi2 > 10 ) { return false; }
        //if ( llc.mass < 15 ) { return false; }

        return true;

    }
    // Warning if not electron or muon.
    std::cout << "Warning: The selected llCandidate is not an electron or muon" << std::endl;
    return false;
}

bool DisplacedDileptons::passBaselineSelection(trackPair ttp) {
   if ( ttp.leadingPt < 5 ) { return false; }
   if ( ttp.subleadingPt < 5 ) { return false; }
   if ( fabs(ttp.etaA) > 2.0 || fabs(ttp.etaB) > 2.0 ) { return false; }
   if ( fabs(ttp.relisoA) > 0.1 || fabs(ttp.relisoB) > 0.1 ) { return false; }
   if ( ttp.normalizedChi2 > 5 ) { return false; }
   //if ( ttp.mass < 15 ) { return false; }
   return true;
}

float DisplacedDileptons::computeRelIso(const reco::Track & track, edm::Handle<std::vector<pat::PackedCandidate> > & pfs, bool isPF) {
    // Contributions to isolation:
    double charged = 0, neutral = 0, pileup  = 0, trackiso = 0;
    for (unsigned int i = 0, n = pfs->size(); i < n; ++i) {
        const pat::PackedCandidate &pf = (*pfs)[i];
        // Reject pf candidate if it is the same track:
        if (fabs(pf.pt() - track.pt()) < 0.01) { continue; }
        // Only count tracks within a 0.3 cone
        double _dR = getDeltaR(track.phi(), track.eta(), pf.phi(), pf.eta());
        if (_dR > 0.3 || _dR < 0.03) { continue; }
        // PF
        if (pf.charge() == 0) {
            if (pf.pt() > 0.5) neutral += pf.pt();
        } else if (pf.fromPV() >= 2) {
            charged += pf.pt();
        } else {
            if (pf.pt() > 0.5) pileup += pf.pt();
        }
        // track
        if (pf.charge() != 0 and pf.fromPV() >= 2) {trackiso += pf.pt(); }
    }
    // do deltaBeta:
    double iso = charged + std::max(0.0, neutral-0.5*pileup);
    if (isPF){
        return iso/track.pt();
    } else {
        return trackiso/track.pt();
    }
}

void DisplacedDileptons::findDileptons() {
    // Setting up variables used later
    GlobalPoint _PVpoint(pv.x(), pv.y(), pv.z());
    GlobalPoint _BSpoint(beamspot.x0(), beamspot.y0(), beamspot.z0());
    GlobalPoint _0point(0.0, 0.0, 0.0);
    bool _BSMode = false;

    std::vector<int> iT; // track indexes
    int i_tk = 0;
    for (auto & isotrack : *isoTracks){
        if (!passIsotrackSelection(isotrack)){ continue; } // some quality cuts
        iT.push_back(i_tk);
        i_tk++;
    }
    nIsoTrack = iT.size(); // number of isotracks

    // Sort the isotracks by pt:
    std::sort( std::begin(iT), std::end(iT), [&](int i1, int i2){ return (*isoTracks).at(i1).pt() > (*isoTracks).at(i2).pt(); });

    // Loop over the isotracks:
    for (size_t i = 0; i < iT.size(); ++i){
        const pat::IsolatedTrack & isotrack = (*isoTracks).at(iT.at(i));
        // Basic features:
        IsoTrackSel_pt[i] = isotrack.pt();
        IsoTrackSel_eta[i] = isotrack.eta();
        IsoTrackSel_phi[i] = isotrack.phi();
        IsoTrackSel_charge[i] = isotrack.charge();      

        // Track extrapolations to ECAL surface (used to do the electron matching)
        IsoTrackSel_etaExtra[i] = isotrack.eta() + isotrack.deltaEta();
        IsoTrackSel_phiExtra[i] = isotrack.phi() + isotrack.deltaPhi();
    
        // Isolation info:
        const pat::PFIsolation &pfiso = isotrack.pfIsolationDR03();
        const pat::PFIsolation &minipfiso = isotrack.miniPFIsolation();

        double neutralIso = fmax(0.0, pfiso.photonIso() + pfiso.neutralHadronIso() - 0.5*pfiso.puChargedHadronIso());
        double chargedIso = pfiso.chargedHadronIso();
        IsoTrackSel_pfIsolationDR03[i] = neutralIso + chargedIso;
        IsoTrackSel_relPfIsolationDR03[i] = IsoTrackSel_pfIsolationDR03[i]/isotrack.pt();

        double miniNeutralIso = fmax(0.0, minipfiso.photonIso() + minipfiso.neutralHadronIso() - 0.5*minipfiso.puChargedHadronIso());
        double miniChargedIso = minipfiso.chargedHadronIso();
        IsoTrackSel_miniPFIsolation[i] = miniNeutralIso + miniChargedIso;
        IsoTrackSel_relMiniPFIsolation[i] = IsoTrackSel_miniPFIsolation[i]/isotrack.pt();

        // Quality info:
        IsoTrackSel_isHighPurityTrack[i] = isotrack.isHighPurityTrack();

        // Hit info:
        const reco::HitPattern &hits = isotrack.hitPattern();

        IsoTrackSel_numberOfValidTrackerHits[i] = hits.numberOfValidTrackerHits();
        IsoTrackSel_numberOfValidPixelHits[i] = hits.numberOfValidPixelHits();
        IsoTrackSel_numberOfValidPixelBarrelHits[i] = hits.numberOfValidPixelBarrelHits();
        IsoTrackSel_numberOfValidPixelEndcapHits[i] = hits.numberOfValidPixelEndcapHits();
        IsoTrackSel_numberOfValidStripHits[i] = hits.numberOfValidStripHits();
        IsoTrackSel_numberOfValidStripTIBHits[i] = hits.numberOfValidStripTIBHits();
        IsoTrackSel_numberOfValidStripTIDHits[i] = hits.numberOfValidStripTIDHits();
        IsoTrackSel_numberOfValidStripTOBHits[i] = hits.numberOfValidStripTOBHits();
        IsoTrackSel_numberOfValidStripTECHits[i] = hits.numberOfValidStripTECHits();

        // Info extracted form the packedCandidate of the isotrack
        //fromPV: 0 NoPV, 1 PVLoose, 2 PVTight, 3 used in the PV fit
        IsoTrackSel_fromPV[i] = isotrack.fromPV(); 
        
        const pat::PackedCandidateRef &pckCand = isotrack.packedCandRef(); // access the packed candidate
            
        //This catches all the isotracks (selected at this point)
        if (isotrack.fromPV() > -1){ // check it has a PV

            IsoTrackSel_vx[i] = (*pckCand).vx();
            IsoTrackSel_vy[i] = (*pckCand).vy();
            IsoTrackSel_vz[i] = (*pckCand).vz();

            const reco::VertexRef &PV = (*pckCand).vertexRef(); // access the PV of the candidate
            IsoTrackSel_PVx[i] = (*PV).x();
            IsoTrackSel_PVy[i] = (*PV).y();
            IsoTrackSel_PVz[i] = (*PV).z();

            // Access to the track and computation of impact parameters:
            const reco::Track *trref = (*pckCand).bestTrack();
            const reco::Track &ctr = *trref;
            reco::TransientTrack _isotk = theTransientTrackBuilder->build(ctr);
            TrajectoryStateClosestToPoint _trajPV = _isotk.trajectoryStateClosestToPoint( _PVpoint );
            TrajectoryStateClosestToPoint _trajBS = _isotk.trajectoryStateClosestToPoint( _BSpoint );
            TrajectoryStateClosestToPoint _traj0 = _isotk.trajectoryStateClosestToPoint( _0point );
                    
            // Impact parameter info:
            IsoTrackSel_dxy[i] = (*pckCand).dxy();
            IsoTrackSel_dxyError[i] = (*pckCand).dxyError();
            IsoTrackSel_dxy_PV[i] = (*pckCand).dxy(pv.position());
            IsoTrackSel_dxyError_PV[i] = _trajPV.perigeeError().transverseImpactParameterError();
            IsoTrackSel_dxy_BS[i] = ctr.dxy(beamspot);
            IsoTrackSel_dxyError_BS[i] = _trajBS.perigeeError().transverseImpactParameterError();
            IsoTrackSel_dxy_0[i] = -_traj0.perigeeParameters().transverseImpactParameter();
            IsoTrackSel_dxyError_0[i] = _traj0.perigeeError().transverseImpactParameterError();
            IsoTrackSel_dz[i] = (*pckCand).dz(pv.position());
            IsoTrackSel_dzError[i] = (*pckCand).dzError(); 
        }
        else {

            IsoTrackSel_vx[i] = -99;
            IsoTrackSel_vy[i] = -99;
            IsoTrackSel_vz[i] = -99;

            IsoTrackSel_PVx[i] = -99;
            IsoTrackSel_PVy[i] = -99;
            IsoTrackSel_PVz[i] = -99;

        }
    }

    //// ----------------------------
    //// --
    //// ---- CMS Photon Collection
    //// --
    //// ----------------------------
    std::vector<int> iP; // photon indices
    // Select good photons
    int i_ph = 0;
    for (auto & photon : *photons){       
        if (!passPhotonSelection(photon)) continue;
        iP.push_back(i_ph);
        i_ph++;
    }
    // Sort good photon indices by pt
    std::sort( std::begin(iP), std::end(iP), [&](int i1, int i2){ return (*photons).at(i1).et() > (*photons).at(i2).et(); });
    // Loop over the photons
    nPhoton = iP.size();
    for (size_t i = 0; i < iP.size(); i++){
        const pat::Photon & photon = (*photons)[iP.at(i)];
        PhotonSel_et[i] = photon.et();
        PhotonSel_eta[i] = photon.eta();
        PhotonSel_phi[i] = photon.phi();
        PhotonSel_hadronicOverEm[i] = photon.hadronicOverEm();
        PhotonSel_full5x5_sigmaIetaIeta[i] = photon.full5x5_sigmaIetaIeta();
        PhotonSel_isEB[i] = photon.isEB();
        PhotonSel_isEE[i] = photon.isEE();
        PhotonSel_r9[i] = photon.r9();
    }

    //// -----------------------------------
    //// --
    //// ---- EE Candidates reconstruction
    //// --
    //// -----------------------------------


    // Variable initiallization:
    std::vector<int> matched_tracks, matched_SC, matched_triggerObjects; // std vectors with matched objects to avoid overlapping
    float dRMin = 99999; // dR to minimize as high as possible in the beginning
    float dRThreshold = 0.1; // Maximum dR to do the lepton matching
    float dR; // Computation of dR
    int tmin, scmin, li; // minimum track, minimum SC, minimum trigger object, reconstructed lepton index
    // while loop that looks for the minimum matchings and stops when the dRThreshold is reached:
    while (1){
        dRMin = 99999;
        // Loop over the tracks
        for (size_t t = 0; t < iT.size(); t++){
            const pat::IsolatedTrack & isotrack = (*isoTracks).at(iT.at(t));
            // pass if the track is associated already:
            if(std::find(matched_tracks.begin(), matched_tracks.end(), t) != matched_tracks.end()){ continue; }
            // pass if the track does not fulfil the prerequisites:
            if(!passIsotrackSelection(isotrack)){ continue; }
            // Reject tracks falling in Barrel-Endcap transition (only to do the matching)
            if (fabs(isotrack.eta()) > 1.4442 and fabs(isotrack.eta()) < 1.566) { continue; }
            // Loop over the superclusters:
            for (size_t sc = 0; sc < iP.size(); sc++){
                const pat::Photon & photon = (*photons).at(iP.at(sc));
                // pass if the SC is associated to other track:
                if(std::find(matched_SC.begin(), matched_SC.end(), sc) != matched_SC.end()){ continue; }
                // pass if the SC does not fulfil the prerequisites:              
                // ------------ SC matching -------------
                dR = getDeltaR(isotrack.phi() + isotrack.deltaPhi(), isotrack.eta() + isotrack.deltaEta(), photon.phi(), photon.eta());      
                if (dR < dRMin){
                    dRMin = dR;
                    scmin = sc;
                    tmin = t;
                }
            }
        }
        if (dRMin > dRThreshold){ break; } // Here we go out the while loop
        li = matched_SC.size();
        ElectronCandidate_pt[li] = (*isoTracks).at(iT.at(tmin)).pt();
        ElectronCandidate_eta[li] = (*isoTracks).at(iT.at(tmin)).eta();
        ElectronCandidate_phi[li] = (*isoTracks).at(iT.at(tmin)).phi();
        ElectronCandidate_et[li] = (*photons).at(iP.at(scmin)).et();
        ElectronCandidate_photonIdx[li] = scmin;
        ElectronCandidate_isotrackIdx[li] = tmin;
        ElectronCandidate_dxy[li] = IsoTrackSel_dxy[tmin];
        ElectronCandidate_dxyError[li] = IsoTrackSel_dxyError[tmin];
        ElectronCandidate_dxy_PV[li] = IsoTrackSel_dxy_PV[tmin];
        ElectronCandidate_dxyError_PV[li] = IsoTrackSel_dxyError_PV[tmin];
        ElectronCandidate_dxy_0[li] = IsoTrackSel_dxy_0[tmin];
        ElectronCandidate_dxyError_0[li] = IsoTrackSel_dxyError_0[tmin];
        ElectronCandidate_dxy_BS[li] = IsoTrackSel_dxy_BS[tmin];
        ElectronCandidate_dxyError_BS[li] = IsoTrackSel_dxyError_BS[tmin];

        // re-compute isolation
        const pat::PackedCandidateRef &e_pck = (*isoTracks).at(iT.at(tmin)).packedCandRef();
        ElectronCandidate_relPFiso[li] = computeRelIso(*(*e_pck).bestTrack(), packedPFCandidates, true);
        ElectronCandidate_relTrkiso[li] = computeRelIso(*(*e_pck).bestTrack(), packedPFCandidates, false);
        ElectronCandidate_pvAssociationQuality[li] = (*isoTracks).at(iT.at(tmin)).packedCandRef()->pvAssociationQuality();
        ElectronCandidate_ptDiff[li] = (*isoTracks).at(iT.at(tmin)).pt() - (*isoTracks).at(iT.at(tmin)).packedCandRef()->pseudoTrack().pt();

        matched_SC.push_back(scmin); matched_tracks.push_back(tmin);
    }
    nElectronCandidate = matched_SC.size();

    ////////////////////////////////////////////////////////////////////////////////////////////
    //// ---------------------------------------------------------------------------------- ////
    //// -------------------------- LL CANDIDATES RECONSTRUCTION -------------------------- ////                                   
    //// ---------------------------------------------------------------------------------- ////
    ////////////////////////////////////////////////////////////////////////////////////////////
    double minChi2 = 10000;
    int min_i = 99;
    int min_j = 99;
    std::vector<pat::IsolatedTrack> leptonTracks;
    /////////////////////////////////////////////////////////
    // --------------------------------------------------- //
    // ----------- eeCandidates reconstruction ----------- //
    // --------------------------------------------------- //
    /////////////////////////////////////////////////////////
    nEE = 0;
    nEEBase = 0;
    EEBase_maxIxy = 0;
    std::vector<double> pairedE; // electrons that are already paired


    while(2*nEE < nElectronCandidate - 1 ){

        // Init control variables:
        minChi2 = 10000;
        min_i = 99;
        min_j = 99;

        for (int i = 0; i < nElectronCandidate; i++) {
            for (int j = i+1; j < nElectronCandidate; j++) {
        
                if (i == j) { continue; }
                if ( std::find(pairedE.begin(), pairedE.end(), i) != pairedE.end() ) {continue;}
                if ( std::find(pairedE.begin(), pairedE.end(), j) != pairedE.end() ) {continue;}

                const pat::IsolatedTrack & it_i = (*isoTracks).at(iT.at(ElectronCandidate_isotrackIdx[i]));
                const pat::IsolatedTrack & it_j = (*isoTracks).at(iT.at(ElectronCandidate_isotrackIdx[j]));
                const pat::PackedCandidateRef &pckCand_i = it_i.packedCandRef(); 
                const pat::PackedCandidateRef &pckCand_j = it_j.packedCandRef(); 
                const reco::Track *itrref_i = (*pckCand_i).bestTrack();
                const reco::Track *itrref_j = (*pckCand_j).bestTrack();
                const reco::Track &itr_i = *itrref_i;
                const reco::Track &itr_j = *itrref_j;

                //llCandidate testcandidate(pv, theTransientTrackBuilder, it_i, it_j, true);
                trackPair testcandidate(pv, beamspot, theTransientTrackBuilder, itr_i, itr_j, true);

                if (!testcandidate.hasValidVertex) { continue ;} 

                // Check if the Chi2 is lower:
                if (testcandidate.normalizedChi2 < minChi2) {
                    minChi2 = testcandidate.normalizedChi2;
                    min_i = i;
                    min_j = j;
                }

            } // end j electron loop
        } // end i electron loop

        if (min_i == 99 || min_j == 99) { break; }
        pairedE.push_back(min_i);
        pairedE.push_back(min_j);

        // -> Get LLP Candidate variables:
        const pat::IsolatedTrack &it_A = (*isoTracks).at(iT.at(ElectronCandidate_isotrackIdx[min_i]));
        const pat::IsolatedTrack &it_B = (*isoTracks).at(iT.at(ElectronCandidate_isotrackIdx[min_j])); 
        const pat::PackedCandidateRef &pckCand_A = it_A.packedCandRef(); 
        const pat::PackedCandidateRef &pckCand_B = it_B.packedCandRef(); 
        const reco::Track *itrref_A = (*pckCand_A).bestTrack();
        const reco::Track *itrref_B = (*pckCand_B).bestTrack();
        const reco::Track &itr_A = *itrref_A;
        const reco::Track &itr_B = *itrref_B;

        trackPair eeCandidate(pv, beamspot, theTransientTrackBuilder, itr_A, itr_B, true);

        // Additionally, for electrons we have to redefine:
        eeCandidate.leadingEt = (ElectronCandidate_et[min_i] > ElectronCandidate_et[min_j])? ElectronCandidate_et[min_i]: ElectronCandidate_et[min_j];
        eeCandidate.subleadingEt = (ElectronCandidate_et[min_i] < ElectronCandidate_et[min_j])? ElectronCandidate_et[min_i]: ElectronCandidate_et[min_j];

        eeCandidate.relisoA = ElectronCandidate_relTrkiso[min_i];
        eeCandidate.relisoB = ElectronCandidate_relTrkiso[min_j];

        eeCandidate.trackDxy = (fabs(ElectronCandidate_dxy[min_i])/ElectronCandidate_dxyError[min_i] < fabs(ElectronCandidate_dxy[min_j])/ElectronCandidate_dxyError[min_j]) ? ElectronCandidate_dxy[min_i] : ElectronCandidate_dxy[min_j];
        eeCandidate.trackIxy = (fabs(ElectronCandidate_dxy[min_i])/ElectronCandidate_dxyError[min_i] < fabs(ElectronCandidate_dxy[min_j])/ElectronCandidate_dxyError[min_j]) ? fabs(ElectronCandidate_dxy[min_i])/ElectronCandidate_dxyError[min_i] : fabs(ElectronCandidate_dxy[min_j])/ElectronCandidate_dxyError[min_j];

        eeCandidate.trackDxy_PV = (fabs(ElectronCandidate_dxy_PV[min_i])/ElectronCandidate_dxyError_PV[min_i] < fabs(ElectronCandidate_dxy_PV[min_j])/ElectronCandidate_dxyError_PV[min_j]) ? ElectronCandidate_dxy_PV[min_i] : ElectronCandidate_dxy_PV[min_j];
        eeCandidate.trackIxy_PV = (fabs(ElectronCandidate_dxy_PV[min_i])/ElectronCandidate_dxyError_PV[min_i] < fabs(ElectronCandidate_dxy_PV[min_j])/ElectronCandidate_dxyError_PV[min_j]) ? fabs(ElectronCandidate_dxy_PV[min_i])/ElectronCandidate_dxyError_PV[min_i] : fabs(ElectronCandidate_dxy_PV[min_j])/ElectronCandidate_dxyError_PV[min_j];

        eeCandidate.trackDxy_0 = (fabs(ElectronCandidate_dxy_0[min_i])/ElectronCandidate_dxyError_0[min_i] < fabs(ElectronCandidate_dxy_0[min_j])/ElectronCandidate_dxyError_0[min_j]) ? ElectronCandidate_dxy_0[min_i] : ElectronCandidate_dxy_0[min_j];
        eeCandidate.trackIxy_0 = (fabs(ElectronCandidate_dxy_0[min_i])/ElectronCandidate_dxyError_0[min_i] < fabs(ElectronCandidate_dxy_0[min_j])/ElectronCandidate_dxyError_0[min_j]) ? fabs(ElectronCandidate_dxy_0[min_i])/ElectronCandidate_dxyError_0[min_i] : fabs(ElectronCandidate_dxy_0[min_j])/ElectronCandidate_dxyError_0[min_j];
        
        eeCandidate.trackDxy_BS = (fabs(ElectronCandidate_dxy_BS[min_i])/ElectronCandidate_dxyError_BS[min_i] < fabs(ElectronCandidate_dxy_BS[min_j])/ElectronCandidate_dxyError_BS[min_j]) ? ElectronCandidate_dxy_BS[min_i] : ElectronCandidate_dxy_BS[min_j];
        eeCandidate.trackIxy_BS = (fabs(ElectronCandidate_dxy_BS[min_i])/ElectronCandidate_dxyError_BS[min_i] < fabs(ElectronCandidate_dxy_BS[min_j])/ElectronCandidate_dxyError_BS[min_j]) ? fabs(ElectronCandidate_dxy_BS[min_i])/ElectronCandidate_dxyError_BS[min_i] : fabs(ElectronCandidate_dxy_BS[min_j])/ElectronCandidate_dxyError_BS[min_j];

        TLorentzVector l1 = TLorentzVector(); 
        TLorentzVector l2 = TLorentzVector();
        l1.SetPtEtaPhiM(ElectronCandidate_pt[min_i], ElectronCandidate_eta[min_i], ElectronCandidate_phi[min_i], 0.501/1000.0);
        l2.SetPtEtaPhiM(ElectronCandidate_pt[min_j], ElectronCandidate_eta[min_j], ElectronCandidate_phi[min_j], 0.501/1000.0);
        TVector3 vl1 = l1.Vect(); 
        TVector3 vl2 = l2.Vect(); 
        TVector3 vl1l2 = vl1 + vl2;
        TVector3 vec = TVector3(eeCandidate.vx - PV_vx, eeCandidate.vy - PV_vy, 0.0);

        eeCandidate.mass = (l1 + l2).M();
        eeCandidate.dPhi = fabs(vec.DeltaPhi(vl1l2));
        eeCandidate.lldPhi = fabs(l1.DeltaPhi(l2));
        eeCandidate.dR = fabs(l1.DeltaR(l2));

        if (!_BSMode){

            EE_idxA[nEE] = min_i;
            EE_idxB[nEE] = min_j;
            EE_Lxy_PV[nEE] = eeCandidate.Lxy_PV;
            EE_Ixy_PV[nEE] = eeCandidate.Ixy_PV;
            EE_Lxy_0[nEE] = eeCandidate.Lxy_0;
            EE_Ixy_0[nEE] = eeCandidate.Ixy_0;
            EE_Lxy_BS[nEE] = eeCandidate.Lxy_BS;
            EE_Ixy_BS[nEE] = eeCandidate.Ixy_BS;
            EE_trackDxy[nEE] = eeCandidate.trackDxy;
            EE_trackIxy[nEE] = eeCandidate.trackIxy;
            EE_trackDxy_0[nEE] = eeCandidate.trackDxy_0;
            EE_trackIxy_0[nEE] = eeCandidate.trackIxy_0;
            EE_trackDxy_PV[nEE] = eeCandidate.trackDxy_PV;
            EE_trackIxy_PV[nEE] = eeCandidate.trackIxy_PV;
            EE_trackDxy_BS[nEE] = eeCandidate.trackDxy_BS;
            EE_trackIxy_BS[nEE] = eeCandidate.trackIxy_BS;
            EE_vx[nEE] = eeCandidate.vx;
            EE_vy[nEE] = eeCandidate.vy;
            EE_normalizedChi2[nEE] = eeCandidate.normalizedChi2;
            EE_mass[nEE] = eeCandidate.mass;
            EE_leadingPt[nEE] = eeCandidate.leadingPt;
            EE_subleadingPt[nEE] = eeCandidate.subleadingPt;
            EE_cosAlpha[nEE] = eeCandidate.cosAlpha;
            EE_dPhi[nEE] = eeCandidate.dPhi;
            EE_lldPhi[nEE] = eeCandidate.lldPhi;
            EE_dR[nEE] = eeCandidate.dR;
            EE_relisoA[nEE] = eeCandidate.relisoA;
            EE_relisoB[nEE] = eeCandidate.relisoB;
            EE_leadingEt[nEE] = eeCandidate.leadingEt;
            EE_subleadingEt[nEE] = eeCandidate.subleadingEt;

        }

        nEE++;

        // -> Fill candidates that pass baseline selection:
        if ( passBaselineSelection(eeCandidate) ) {

            if (_BSMode) {

            leptonTracks.push_back(it_A); leptonTracks.push_back(it_B);

            EEBase_idxA[nEEBase] = min_i;
            EEBase_idxB[nEEBase] = min_j;
            EEBase_Lxy[nEEBase] = eeCandidate.Lxy_PV;
            EEBase_Ixy[nEEBase] = eeCandidate.Ixy_PV;
            EEBase_trackDxy[nEEBase] = eeCandidate.trackDxy;
            EEBase_trackIxy[nEEBase] = eeCandidate.trackIxy;
            EEBase_vx[nEEBase] = eeCandidate.vx;
            EEBase_vy[nEEBase] = eeCandidate.vy;
            EEBase_normalizedChi2[nEEBase] = eeCandidate.normalizedChi2;
            EEBase_mass[nEEBase] = eeCandidate.mass;
            EEBase_leadingPt[nEEBase] = eeCandidate.leadingPt;
            EEBase_subleadingPt[nEEBase] = eeCandidate.subleadingPt;
            EEBase_cosAlpha[nEEBase] = eeCandidate.cosAlpha;
            EEBase_dPhi[nEEBase] = eeCandidate.dPhi;
            EEBase_relisoA[nEEBase] = eeCandidate.relisoA;
            EEBase_relisoB[nEEBase] = eeCandidate.relisoB;
            EEBase_leadingEt[nEEBase] = eeCandidate.leadingEt;
            EEBase_subleadingEt[nEEBase] = eeCandidate.subleadingEt;
            EEBase_fromPVA[nEEBase] = eeCandidate.fromPVA;
            EEBase_fromPVB[nEEBase] = eeCandidate.fromPVB;
            EEBase_PVAssociation[nEEBase] = eeCandidate.PVAssociation;

            if ( fabs(EEBase_trackIxy[nEEBase]) > fabs(EEBase_trackIxy[EEBase_maxIxy]) ) { EEBase_maxIxy = nEEBase; }

            }
            nEEBase++;

        }

    } // end while 

}
