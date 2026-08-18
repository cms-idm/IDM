#ifndef HELPERS_HH
#define HELPERS_HH

#include <algorithm>
#include <cmath> 
#include <memory>
#include <random>
#include <vector>
#include <boost/format.hpp>
#include <boost/any.hpp>

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"

// Coarse xyz estimate of a track's displacement, derived from the tracker layer
// of its innermost valid hit rather than a fitted/refit vertex.
struct HitBasedSVProxy {
    bool valid = false;
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
    float vxy = 0.f;
};

class Helper {
    public:
        Helper();
        ~Helper();
        bool JetID(const pat::Jet &jet, std::string year);
        HitBasedSVProxy EstimateSVProxyFromInnermostHit(const reco::Track &track);
};

#endif