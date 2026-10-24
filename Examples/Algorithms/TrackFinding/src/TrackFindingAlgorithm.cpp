// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//This include has been modified to include a struct for 
//storing candidate hits on surfaces along the track 
//trajectory
#include "ActsExamples/TrackFinding/TrackFindingAlgorithm.hpp"

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Direction.hpp"
#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/ParticleHypothesis.hpp"
#include "Acts/EventData/ProxyAccessor.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include "Acts/EventData/TrackContainer.hpp"
#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/StandardAborters.hpp"
#include "Acts/Propagator/SympyStepper.hpp"
#include "Acts/Surfaces/PerigeeSurface.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/TrackFinding/TrackStateCreator.hpp"
#include "Acts/TrackFitting/BetheHeitlerApprox.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/TrackFitting/GsfMixtureReduction.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/TrackHelpers.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/Measurement.hpp"
#include "ActsExamples/EventData/MeasurementCalibration.hpp"
#include "ActsExamples/EventData/Seed.hpp"
#include "ActsExamples/EventData/SpacePoint.hpp"
#include "ActsExamples/EventData/Track.hpp"
#include "ActsExamples/Framework/AlgorithmContext.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ActsExamples {

namespace {

class MeasurementSelector {
 public:
  using Traj = Acts::VectorMultiTrajectory;

  explicit MeasurementSelector(Acts::MeasurementSelector selector, const TrackFindingAlgorithm::Config& config)
      : m_candidate_accumulator("allCandidates"), m_selector(std::move(selector)), m_cfg(&config) {}

  mutable ActsExamples::candidate_properties cand_props;
  Acts::ProxyAccessor<ActsExamples::candidate_properties> m_candidate_accumulator;

  //Set the event data as attributes we can access on calls to MeasurementSelector
  void setEventData(const MeasurementSubset* measurements, const ClusterContainer* clusters){
    m_measurements = measurements;
    m_clusters = clusters;
  }

  /// Resolve the seed to measurement indices once, so the per candidate check
  /// does not have to walk space points and unpack source links.
  void setSeed(const std::optional<ConstSeedProxy>& seed) {
    m_seedIndices.clear();
    if (!seed.has_value()) {
      return;
    }
    for (const ConstSpacePointProxy sp : seed->spacePoints()) {
      for (const Acts::SourceLink& sl : sp.sourceLinks()) {
        m_seedIndices.push_back(sl.get<IndexSourceLink>().index());
      }
    }
  }

  Acts::Result<std::pair<std::vector<Traj::TrackStateProxy>::iterator,
                         std::vector<Traj::TrackStateProxy>::iterator>>
  select(std::vector<Traj::TrackStateProxy>& candidates, bool& isOutlier,
         const Acts::Logger& logger) const {
    if (!m_seedIndices.empty()) {
      // the selector sorts by chi2 anyway, so partitioning in place is enough
      auto rest = std::partition(
          candidates.begin(), candidates.end(),
          [this](const auto& candidate) { return isSeedCandidate(candidate); });
      if (rest != candidates.begin()) {
        candidates.erase(rest, candidates.end());
      }
    }

    //Collect the candidates and clear the struct
    auto passing_candidates =  m_selector.select<Acts::VectorMultiTrajectory>(candidates, isOutlier, logger);
    cand_props.clear();

    if (passing_candidates.ok()){
        //Collect all passing candidates 
        auto[passing_begin, passing_end] = passing_candidates.value();
        //Loop through ALL candidates
        for (auto it = candidates.begin(); it != candidates.end(); it++){
          //Define the set of attributes to be readout
          bool selected_cand = 0;
          bool outlier_cand = 0;
          bool rejected_cand = 0;
          int hit_id = -999;
          double chi2 = -999;
          double charge = 0;
          int nchannels = 0;
          int sizeLoc0 = 0;
          int sizeLoc1 = 0;
          double globalEta = 0;
          double globalPhi = 0;
          double incidentEta = 0;
          double incidentPhi = 0;
          double loc0 = 0;
          double loc1 = 0;
          double varloc0 = -999;
          double varloc1 = -999;

          if (it >= passing_end){
            rejected_cand = 1;
          }
          else if (it <= passing_begin){
            selected_cand = 1;
          }
          else{
            outlier_cand = 1;
          }
          //Check for a reference to an uncalibrated
          //structure
          if (it->hasUncalibratedSourceLink()){
            const Acts::SourceLink& source_link = it->getUncalibratedSourceLink();
            //Follow the reference 
            if (const auto* index_link = source_link.getPtr<ActsExamples::IndexSourceLink>()){
              //Update the hit id using the global index
              //of the reference
              hit_id = static_cast<int>(index_link->index());
              //Retrieve Chi2
              chi2 = it->chi2();
              //Check that m_clusters is not a nullptr
              //recall that m_clusters is a pointer to 
              //the cluster memory space. Also use the
              //fact that clusters are 1<->1 mapped with
              //measurements according to Cluster.hpp
              if (m_clusters!=nullptr && hit_id >= 0 && static_cast<std::size_t>(hit_id) < m_clusters->size()){
                //A write protected view of the cluster
                const auto& cluster = (*m_clusters)[hit_id];
                //Grab the total cluster charge
                charge = cluster.sumActivations();
                //Number of channels contributing
                nchannels = static_cast<int>(cluster.channels.size());
                //Number of channels along local coordinates (l0,l1)
                sizeLoc0 = static_cast<int>(cluster.sizeLoc0);
                sizeLoc1 = static_cast<int>(cluster.sizeLoc1);
                globalEta = cluster.globalEta;
                globalPhi = cluster.globalPhi;
                incidentEta = cluster.etaAngle;
                incidentPhi = cluster.phiAngle;
              }
              else{
                if (m_clusters == nullptr){
                  std::cout<< "CLUSTERS IS NULL" <<std::endl;
                }
              }
            if (m_measurements!=nullptr && hit_id >= 0 && static_cast<std::size_t>(hit_id) < m_measurements->size()){
              const auto& measurement = (*m_measurements).at(static_cast<std::size_t>(hit_id));
              //Measurements contain parameters and covariances
              //parameters = local coordinates indexed with [n]
              //covariances = matrix objects indexed with (n,m)
              const auto locs = measurement.parameters();
              const auto cov = measurement.covariance();
              loc0 = locs[0];
              varloc0 = cov(0,0);
              if (locs.size() > 1){
                loc1 = locs[1];
                varloc1 = cov(1,1);
              }
            }
            else{
              if (m_measurements==nullptr){
                std::cout << "MEASUREMENTS IS NULL" << std::endl;
              }
            }
            }
          }
          //Pushback to the containers
          cand_props.hit_idx.push_back(hit_id);
          cand_props.chi2.push_back(chi2);
          cand_props.selected_cand.push_back(selected_cand);
          cand_props.outlier_cand.push_back(outlier_cand);
          cand_props.rejected_cand.push_back(rejected_cand);
          cand_props.charge_dep.push_back(charge);
          cand_props.nchannels.push_back(nchannels);
          cand_props.sizeLoc0.push_back(sizeLoc0);
          cand_props.sizeLoc1.push_back(sizeLoc1);
          cand_props.globalEta.push_back(globalEta);
          cand_props.globalPhi.push_back(globalPhi);
          cand_props.incidentEta.push_back(incidentEta);
          cand_props.incidentPhi.push_back(incidentPhi);
          cand_props.Loc0.push_back(loc0);
          cand_props.VarLoc0.push_back(varloc0);
          cand_props.Loc1.push_back(loc1);
          cand_props.VarLoc1.push_back(varloc1);
        }
    }

    //Handle the case of missing candidates so that the track
    //structure can be aligned by counting surfaces
    if (!passing_candidates.ok() || candidates.empty()){
      cand_props.hit_idx.push_back(-999);
      cand_props.chi2.push_back(-999);
      cand_props.selected_cand.push_back(0);
      cand_props.outlier_cand.push_back(0);
      cand_props.rejected_cand.push_back(0);
      cand_props.charge_dep.push_back(0);
      cand_props.nchannels.push_back(0);
      cand_props.sizeLoc0.push_back(0);
      cand_props.sizeLoc1.push_back(0);
      cand_props.globalEta.push_back(-999);
      cand_props.globalPhi.push_back(-999);
      cand_props.incidentEta.push_back(-999);
      cand_props.incidentPhi.push_back(-999);
      cand_props.Loc0.push_back(0);
      cand_props.VarLoc0.push_back(-999);
      cand_props.Loc1.push_back(0);
      cand_props.VarLoc1.push_back(-999);
    }

    for (auto& cand_proxy : candidates){
      m_candidate_accumulator(cand_proxy) = cand_props;
    }

    //Pass the candidates as normal
    return passing_candidates;

  }

 private:
  Acts::MeasurementSelector m_selector;
  /// measurement indices of the current seed, at most a handful
  std::vector<Index> m_seedIndices;

  //Pointers to the config for input names, 
  //clusters, and measurements
  const TrackFindingAlgorithm::Config* m_cfg{nullptr};
  const MeasurementSubset* m_measurements{nullptr};
  const ClusterContainer* m_clusters{nullptr};


  bool isSeedCandidate(const Traj::TrackStateProxy& candidate) const {
    assert(candidate.hasUncalibratedSourceLink());

    const Index index =
        candidate.getUncalibratedSourceLink().get<IndexSourceLink>().index();
    return std::ranges::find(m_seedIndices, index) != m_seedIndices.end();
  }
};

/// Flags the seeds whose measurements are a subset of an already found track.
///
/// In case of strip seeds only the first source link of the pair is used.
class SeedCoverage {
 public:
  /// Index the seeds by their measurements.
  ///
  /// @param seeds The seeds to index.
  void index(const SeedContainer& seeds) {
    m_seedSize.assign(seeds.size(), 0);
    m_covered.assign(seeds.size(), false);
    m_counts.assign(seeds.size(), 0);

    for (std::size_t iSeed = 0; iSeed < seeds.size(); ++iSeed) {
      const ConstSeedProxy seed = seeds.at(iSeed);
      for (const SpacePointIndex spIndex : seed.spacePointIndices()) {
        const ConstSpacePointProxy sp = seeds.spacePointContainer().at(spIndex);
        if (sp.sourceLinks().empty()) {
          continue;
        }
        const Index measurement =
            sp.sourceLinks().front().get<IndexSourceLink>().index();
        m_measurementToSeeds[measurement].push_back(iSeed);
        ++m_seedSize[iSeed];
      }
    }
  }

  /// Flag every seed whose measurements the track covers.
  ///
  /// @param track The track that was found.
  void addCoverageFrom(const TrackProxy& track) {
    for (const auto& trackState : track.trackStatesReversed()) {
      if (!trackState.hasUncalibratedSourceLink()) {
        continue;
      }
      const Index measurement =
          trackState.getUncalibratedSourceLink().get<IndexSourceLink>().index();
      const auto it = m_measurementToSeeds.find(measurement);
      if (it == m_measurementToSeeds.end()) {
        continue;
      }
      for (const std::size_t iSeed : it->second) {
        if (m_counts[iSeed] == 0) {
          m_touched.push_back(iSeed);
        }
        ++m_counts[iSeed];
        if (m_counts[iSeed] >= m_seedSize[iSeed]) {
          m_covered[iSeed] = true;
        }
      }
    }

    for (const std::size_t iSeed : m_touched) {
      m_counts[iSeed] = 0;
    }
    m_touched.clear();
  }

  /// Whether a previously found track covers the seed.
  ///
  /// @param iSeed The index of the seed.
  /// @return True if the seed is covered.
  bool isCovered(std::size_t iSeed) const { return m_covered.at(iSeed); }

 private:
  std::unordered_map<Index, std::vector<std::size_t>> m_measurementToSeeds;
  std::vector<std::uint32_t> m_seedSize;
  std::vector<bool> m_covered;

  /// per track scratch, reset through `m_touched`
  std::vector<std::uint32_t> m_counts;
  std::vector<std::size_t> m_touched;
};

class BranchStopper {
 public:
  using BranchStopperResult =
      Acts::CombinatorialKalmanFilterBranchStopperResult;

  struct BranchState {
    std::size_t nPixelHoles = 0;
    std::size_t nStripHoles = 0;
  };

  static constexpr Acts::ProxyAccessor<BranchState> branchStateAccessor =
      Acts::ProxyAccessor<BranchState>(Acts::hashString("MyBranchState"));

  mutable std::atomic<std::size_t> m_nStoppedBranches{0};

  explicit BranchStopper(const TrackFindingAlgorithm::Config& config)
      : m_cfg(config) {}

  BranchStopperResult operator()(
      const TrackContainer::TrackProxy& track,
      const TrackContainer::TrackStateProxy& trackState) const {
    if (!m_cfg.trackSelectorCfg.has_value()) {
      return BranchStopperResult::Continue;
    }

    const Acts::TrackSelector::Config* singleConfig = std::visit(
        [&](const auto& config) -> const Acts::TrackSelector::Config* {
          using T = std::decay_t<decltype(config)>;
          if constexpr (std::is_same_v<T, Acts::TrackSelector::Config>) {
            return &config;
          } else if constexpr (std::is_same_v<
                                   T, Acts::TrackSelector::EtaBinnedConfig>) {
            double theta = trackState.parameters()[Acts::eBoundTheta];
            double eta = Acts::AngleHelpers::etaFromTheta(theta);
            return config.hasCuts(eta) ? &config.getCuts(eta) : nullptr;
          }
        },
        *m_cfg.trackSelectorCfg);

    if (singleConfig == nullptr) {
      ++m_nStoppedBranches;
      return BranchStopperResult::StopAndDrop;
    }

    bool tooManyHolesPS = false;
    if (!(m_cfg.pixelVolumeIds.empty() && m_cfg.stripVolumeIds.empty())) {
      auto& branchState = branchStateAccessor(track);
      // count both holes and outliers as holes for pixel/strip counts
      if (trackState.typeFlags().isHole() ||
          trackState.typeFlags().isOutlier()) {
        auto volumeId = trackState.referenceSurface().geometryId().volume();
        if (Acts::rangeContainsValue(m_cfg.pixelVolumeIds, volumeId)) {
          ++branchState.nPixelHoles;
        } else if (Acts::rangeContainsValue(m_cfg.stripVolumeIds, volumeId)) {
          ++branchState.nStripHoles;
        }
      }
      tooManyHolesPS = branchState.nPixelHoles > m_cfg.maxPixelHoles ||
                       branchState.nStripHoles > m_cfg.maxStripHoles;
    }

    bool enoughMeasurements =
        track.nMeasurements() >= singleConfig->minMeasurements;
    bool tooManyHoles =
        track.nHoles() > singleConfig->maxHoles || tooManyHolesPS;
    bool tooManyOutliers = track.nOutliers() > singleConfig->maxOutliers;
    bool tooManyHolesAndOutliers = (track.nHoles() + track.nOutliers()) >
                                   singleConfig->maxHolesAndOutliers;

    if (tooManyHoles || tooManyOutliers || tooManyHolesAndOutliers) {
      ++m_nStoppedBranches;
      return enoughMeasurements ? BranchStopperResult::StopAndKeep
                                : BranchStopperResult::StopAndDrop;
    }

    return BranchStopperResult::Continue;
  }

 private:
  const TrackFindingAlgorithm::Config& m_cfg;
};

}  // namespace

TrackFindingAlgorithm::TrackFindingAlgorithm(
    Config config, std::unique_ptr<const Acts::Logger> logger)
    : IAlgorithm("TrackFindingAlgorithm", std::move(logger)),
      m_cfg(std::move(config)) {
  if (m_cfg.inputMeasurements.empty()) {
    throw std::invalid_argument("Missing measurements input collection");
  }
  if (m_cfg.inputInitialTrackParameters.empty()) {
    throw std::invalid_argument(
        "Missing initial track parameters input collection");
  }
  if (m_cfg.outputTracks.empty()) {
    throw std::invalid_argument("Missing tracks output collection");
  }

  if (m_cfg.seedDeduplication && m_cfg.inputSeeds.empty()) {
    throw std::invalid_argument(
        "Missing seeds input collection. This is "
        "required for seed deduplication.");
  }
  if (m_cfg.stayOnSeed && m_cfg.inputSeeds.empty()) {
    throw std::invalid_argument(
        "Missing seeds input collection. This is "
        "required for staying on seed.");
  }

  if (m_cfg.trackSelectorCfg.has_value()) {
    m_trackSelector = std::visit(
        [](const auto& cfg) -> std::optional<Acts::TrackSelector> {
          return Acts::TrackSelector(cfg);
        },
        m_cfg.trackSelectorCfg.value());
  }

  m_inputMeasurements.initialize(m_cfg.inputMeasurements);
  m_inputInitialTrackParameters.initialize(m_cfg.inputInitialTrackParameters);
  m_inputSeeds.maybeInitialize(m_cfg.inputSeeds);
  m_outputTracks.initialize(m_cfg.outputTracks);
  //Cluster handle initialization
  m_inputClusters.initialize(m_cfg.inputClusters);
}

ProcessCode TrackFindingAlgorithm::execute(const AlgorithmContext& ctx) const {
  // Read input data
  const auto& measurements = m_inputMeasurements(ctx);
  const auto& initialParameters = m_inputInitialTrackParameters(ctx);
  const SeedContainer* seeds = nullptr;
  //Use the handle to access the data indexed
  //by the event context
  const auto& clusters = m_inputClusters(ctx);

  if (m_inputSeeds.isInitialized()) {
    seeds = &m_inputSeeds(ctx);

    if (initialParameters.size() != seeds->size()) {
      ACTS_ERROR("Number of initial parameters and seeds do not match. "
                 << initialParameters.size() << " != " << seeds->size());
    }
  }

  // Construct a perigee surface as the target surface
  auto pSurface = Acts::Surface::makeShared<Acts::PerigeeSurface>(
      Acts::Vector3{0., 0., 0.});

  PassThroughCalibrator pcalibrator;
  MeasurementCalibratorAdapter calibrator(pcalibrator,
                                          measurements.container());
  Acts::GainMatrixUpdater kfUpdater(m_cfg.useJosephFormulation);

  using Extensions = Acts::CombinatorialKalmanFilterExtensions<TrackContainer>;

  BranchStopper branchStopper(m_cfg);
  MeasurementSelector measSel{
      Acts::MeasurementSelector(m_cfg.measurementSelectorCfg),
    m_cfg};

  measSel.setEventData(&measurements, &clusters);

  IndexSourceLinkAccessor slAccessor;
  slAccessor.container = &measurements.orderedIndices();

  using TrackStateCreatorType =
      Acts::TrackStateCreator<IndexSourceLinkAccessor::Iterator,
                              TrackContainer>;
  TrackStateCreatorType trackStateCreator;
  trackStateCreator.sourceLinkAccessor
      .template connect<&IndexSourceLinkAccessor::range>(&slAccessor);
  trackStateCreator.calibrator
      .template connect<&MeasurementCalibratorAdapter::calibrate>(&calibrator);
  trackStateCreator.measurementSelector
      .template connect<&MeasurementSelector::select>(&measSel);

  Extensions extensions;
  extensions.updater.connect<&Acts::GainMatrixUpdater::operator()<
      typename TrackContainer::TrackStateContainerBackend>>(&kfUpdater);
  extensions.branchStopper.connect<&BranchStopper::operator()>(&branchStopper);
  extensions.createTrackStates
      .template connect<&TrackStateCreatorType ::createTrackStates>(
          &trackStateCreator);
  extensions.mixtureReducer.connect<&Acts::reduceMixtureWithKLDistance>();

  Acts::PropagatorPlainOptions firstPropOptions(ctx.recoGeoContext,
                                                ctx.magFieldContext);
  firstPropOptions.maxSteps = m_cfg.maxSteps;
  firstPropOptions.direction = m_cfg.reverseSearch ? Acts::Direction::Backward()
                                                   : Acts::Direction::Forward();
  firstPropOptions.constrainToVolumeIds = m_cfg.constrainToVolumeIds;
  firstPropOptions.endOfWorldVolumeIds = m_cfg.endOfWorldVolumeIds;

  Acts::PropagatorPlainOptions secondPropOptions(ctx.recoGeoContext,
                                                 ctx.magFieldContext);
  secondPropOptions.maxSteps = m_cfg.maxSteps;
  secondPropOptions.direction = firstPropOptions.direction.invert();
  secondPropOptions.constrainToVolumeIds = m_cfg.constrainToVolumeIds;
  secondPropOptions.endOfWorldVolumeIds = m_cfg.endOfWorldVolumeIds;

  // Set the CombinatorialKalmanFilter options
  TrackFinderOptions firstOptions(ctx.recoGeoContext, ctx.magFieldContext,
                                  ctx.calibContext, extensions,
                                  firstPropOptions);

  firstOptions.targetSurface = m_cfg.reverseSearch ? pSurface.get() : nullptr;
  firstOptions.recordMaterialStates = m_cfg.recordMaterialStates;
  firstOptions.betheHeitlerApprox =
      std::make_shared<Acts::PolynomialBetheHeitlerApprox>(
          Acts::makeDefaultBetheHeitlerApprox());

  TrackFinderOptions secondOptions(ctx.recoGeoContext, ctx.magFieldContext,
                                   ctx.calibContext, extensions,
                                   secondPropOptions);
  secondOptions.targetSurface = m_cfg.reverseSearch ? nullptr : pSurface.get();
  secondOptions.skipPrePropagationUpdate = true;
  secondOptions.recordMaterialStates = m_cfg.recordMaterialStates;
  secondOptions.betheHeitlerApprox = firstOptions.betheHeitlerApprox;

  using Extrapolator = Acts::Propagator<Acts::SympyStepper, Acts::Navigator>;
  using ExtrapolatorOptions = Extrapolator::template Options<
      Acts::ActorList<Acts::MaterialInteractor, Acts::EndOfWorldReached>>;

  Extrapolator extrapolator(
      Acts::SympyStepper(m_cfg.magneticField),
      Acts::Navigator({m_cfg.trackingGeometry},
                      logger().cloneWithSuffix("Navigator")),
      logger().cloneWithSuffix("Propagator"));

  ExtrapolatorOptions extrapolationOptions(ctx.recoGeoContext,
                                           ctx.magFieldContext);
  extrapolationOptions.constrainToVolumeIds = m_cfg.constrainToVolumeIds;
  extrapolationOptions.endOfWorldVolumeIds = m_cfg.endOfWorldVolumeIds;

  // Perform the track finding for all initial parameters
  ACTS_DEBUG("Invoke track finding with " << initialParameters.size()
                                          << " seeds.");

  auto trackContainer = std::make_shared<Acts::VectorTrackContainer>();
  auto trackStateContainer = std::make_shared<Acts::VectorMultiTrajectory>();

  auto trackContainerTemp = std::make_shared<Acts::VectorTrackContainer>();
  auto trackStateContainerTemp = std::make_shared<Acts::VectorMultiTrajectory>();

  //Add all canddiates to the track whiteboard
  trackStateContainer -> addColumn<ActsExamples::candidate_properties>("allCandidates");
  trackStateContainerTemp -> addColumn<ActsExamples::candidate_properties>("allCandidates");
  //Post the track candidate properties to the whiteboard with a proxy name
  Acts::ProxyAccessor<ActsExamples::candidate_properties> candidate_accumulator ("allCandidates");

  TrackContainer tracks(trackContainer, trackStateContainer);
  TrackContainer tracksTemp(trackContainerTemp, trackStateContainerTemp);

  // Note that not all backends support PODs as column types
  tracks.addColumn<BranchStopper::BranchState>("MyBranchState");
  tracksTemp.addColumn<BranchStopper::BranchState>("MyBranchState");

  tracks.addColumn<unsigned int>("trackGroup");
  tracksTemp.addColumn<unsigned int>("trackGroup");
  Acts::ProxyAccessor<unsigned int> seedNumber("trackGroup");

  unsigned int nSeed = 0;

  // A map indicating whether a seed has been discovered already
  SeedCoverage seedCoverage;

  auto addTrack = [&](const TrackProxy& track) {
    ++m_nFoundTracks;

    // trim the track if requested
    if (m_cfg.trimTracks) {
      Acts::trimTrack(track, true, true, true, true);
    }
    Acts::calculateTrackQuantities(track);

    if (m_trackSelector.has_value() && !m_trackSelector->isValidTrack(track)) {
      return;
    }

    // flag seeds which are covered by the track
    seedCoverage.addCoverageFrom(track);

    ++m_nSelectedTracks;

    auto destProxy = tracks.makeTrack();
    // make sure we copy track states!
    destProxy.copyFrom(track);

  };

  if (seeds != nullptr && m_cfg.seedDeduplication) {
    // Index the seeds for deduplication
    seedCoverage.index(*seeds);
  }

  for (std::size_t iSeed = 0; iSeed < initialParameters.size(); ++iSeed) {
    m_nTotalSeeds++;

    //Clear the candidates from the previous seeds
    measSel.cand_props.clear();

    if (seeds != nullptr) {
      const ConstSeedProxy seed = seeds->at(iSeed);

      if (m_cfg.seedDeduplication) {
        // check if an already found track covers the seed
        if (seedCoverage.isCovered(iSeed)) {
          m_nDeduplicatedSeeds++;
          ACTS_VERBOSE("Skipping seed " << iSeed << " due to deduplication.");
          continue;
        }
      }

      if (m_cfg.stayOnSeed) {
        measSel.setSeed(seed);
      }
    }

    // Clear trackContainerTemp and trackStateContainerTemp
    tracksTemp.clear();

    const Acts::BoundTrackParameters& firstInitialParameters =
        initialParameters.at(iSeed);
    ACTS_VERBOSE("Processing seed " << iSeed << " with initial parameters "
                                    << firstInitialParameters);

    const bool useBrem = m_cfg.findTracksBrem != nullptr &&
                         firstInitialParameters.particleHypothesis() ==
                             Acts::ParticleHypothesis::electron();
    const TrackFinderFunction& findTracks =
        useBrem ? *m_cfg.findTracksBrem : *m_cfg.findTracks;

    auto firstRootBranch = tracksTemp.makeTrack();
    auto firstResult = findTracks(firstInitialParameters, firstOptions,
                                  tracksTemp, firstRootBranch);
    nSeed++;

    if (!firstResult.ok()) {
      m_nFailedSeeds++;
      ACTS_WARNING("Track finding failed for seed " << iSeed << " with error"
                                                    << firstResult.error());
      continue;
    }

    auto& firstTracksForSeed = firstResult.value();
    for (auto& firstTrack : firstTracksForSeed) {
      // TODO a copy of the track should not be necessary but is the safest way
      //      with the current EDM
      // TODO a lightweight copy without copying all the track state components
      //      might be a solution
      auto trackCandidate = tracksTemp.makeTrack();
      trackCandidate.copyFrom(firstTrack);

      Acts::Result<void> firstSmoothingResult{
          Acts::smoothTrack(ctx.recoGeoContext, trackCandidate, logger())};
      if (!firstSmoothingResult.ok()) {
        m_nFailedSmoothing++;
        ACTS_ERROR("First smoothing for seed "
                   << iSeed << " and track " << firstTrack.index()
                   << " failed with error " << firstSmoothingResult.error());
        continue;
      }

      // number of second tracks found
      std::size_t nSecond = 0;

      // Set the seed number, this number decrease by 1 since the seed number
      // has already been updated
      seedNumber(trackCandidate) = nSeed - 1;

      if (m_cfg.twoWay) {
        std::optional<Acts::VectorMultiTrajectory::TrackStateProxy>
            firstMeasurementOpt;
        for (auto trackState : trackCandidate.trackStatesReversed()) {
          // We are excluding non measurement states and outlier here. Those can
          // decrease resolution because only the smoothing corrected the very
          // first prediction as filtering is not possible.
          if (trackState.typeFlags().isMeasurement()) {
            firstMeasurementOpt = trackState;
          }
        }

        if (firstMeasurementOpt.has_value()) {
          TrackContainer::TrackStateProxy firstMeasurement{
              firstMeasurementOpt.value()};
          TrackContainer::ConstTrackStateProxy firstMeasurementConst{
              firstMeasurement};

          Acts::BoundTrackParameters secondInitialParameters =
              trackCandidate.createParametersFromState(firstMeasurementConst);

          if (!secondInitialParameters.referenceSurface().insideBounds(
                  secondInitialParameters.localPosition())) {
            m_nSkippedSecondPass++;
            ACTS_DEBUG(
                "Smoothing of first pass fit produced out-of-bounds parameters "
                "relative to the surface. Skipping second pass.");
            continue;
          }

          auto secondRootBranch = tracksTemp.makeTrack();
          secondRootBranch.copyFromWithoutStates(trackCandidate);
          auto secondResult = findTracks(secondInitialParameters, secondOptions,
                                         tracksTemp, secondRootBranch);

          if (!secondResult.ok()) {
            ACTS_WARNING("Second track finding failed for seed "
                         << iSeed << " with error" << secondResult.error());
          } else {
            // store the original previous state to restore it later
            auto originalFirstMeasurementPrevious = firstMeasurement.previous();

            auto& secondTracksForSeed = secondResult.value();
            for (auto& secondTrack : secondTracksForSeed) {
              // TODO a copy of the track should not be necessary but is the
              //      safest way with the current EDM
              // TODO a lightweight copy without copying all the track state
              //      components might be a solution
              auto secondTrackCopy = tracksTemp.makeTrack();
              secondTrackCopy.copyFrom(secondTrack);

              // Note that this is only valid if there are no branches
              // We disallow this by breaking this look after a second track was
              // processed
              secondTrackCopy.reverseTrackStates(true);

              firstMeasurement.previous() =
                  secondTrackCopy.outermostTrackState().index();

              // Retain tip and stem index of the first track
              auto tipIndex = trackCandidate.tipIndex();
              auto stemIndex = trackCandidate.stemIndex();
              trackCandidate.copyFromWithoutStates(secondTrackCopy);
              trackCandidate.tipIndex() = tipIndex;
              trackCandidate.stemIndex() = stemIndex;

              // finalize the track candidate

              bool doExtrapolate = true;

              if (!m_cfg.reverseSearch) {
                // these parameters are already extrapolated by the CKF and have
                // the optimal resolution. note that we did not smooth all the
                // states.

                // only extrapolate if we did not do it already
                doExtrapolate = !trackCandidate.hasReferenceSurface();
              } else {
                // smooth the full track and extrapolate to the reference

                auto secondSmoothingResult = Acts::smoothTrack(
                    ctx.recoGeoContext, trackCandidate, logger());
                if (!secondSmoothingResult.ok()) {
                  m_nFailedSmoothing++;
                  ACTS_ERROR("Second smoothing for seed "
                             << iSeed << " and track " << secondTrack.index()
                             << " failed with error "
                             << secondSmoothingResult.error());
                  continue;
                }

                trackCandidate.reverseTrackStates(true);
              }

              if (doExtrapolate) {
                auto secondExtrapolationResult =
                    Acts::extrapolateTrackToReferenceSurface(
                        trackCandidate, *pSurface, extrapolator,
                        extrapolationOptions, m_cfg.extrapolationStrategy,
                        logger());
                if (!secondExtrapolationResult.ok()) {
                  m_nFailedExtrapolation++;
                  ACTS_ERROR("Second extrapolation for seed "
                             << iSeed << " and track " << secondTrack.index()
                             << " failed with error "
                             << secondExtrapolationResult.error());
                  continue;
                }
              }

              addTrack(trackCandidate);

              ++nSecond;
            }

            // restore the original previous state
            firstMeasurement.previous() = originalFirstMeasurementPrevious;
          }
        }
      }

      // if no second track was found, we will use only the first track
      if (nSecond == 0) {
        // restore the track to the original state
        auto tipIndex = trackCandidate.tipIndex();
        auto stemIndex = trackCandidate.stemIndex();
        trackCandidate.copyFromWithoutStates(firstTrack);
        trackCandidate.tipIndex() = tipIndex;
        trackCandidate.stemIndex() = stemIndex;

        auto firstExtrapolationResult =
            Acts::extrapolateTrackToReferenceSurface(
                trackCandidate, *pSurface, extrapolator, extrapolationOptions,
                m_cfg.extrapolationStrategy, logger());
        if (!firstExtrapolationResult.ok()) {
          m_nFailedExtrapolation++;
          ACTS_ERROR("Extrapolation for seed "
                     << iSeed << " and track " << firstTrack.index()
                     << " failed with error "
                     << firstExtrapolationResult.error());
          continue;
        }

        addTrack(trackCandidate);
      }
    }
  }

  // Compute shared hits from all the reconstructed tracks
  if (m_cfg.computeSharedHits) {
    computeSharedHits(tracks, measurements);
  }

  ACTS_DEBUG("Finalized track finding with " << tracks.size()
                                             << " track candidates.");

  m_nStoppedBranches += branchStopper.m_nStoppedBranches;

  m_memoryStatistics.local().hist +=
      tracks.trackStateContainer().statistics().hist;

  auto constTrackStateContainer =
      std::make_shared<Acts::ConstVectorMultiTrajectory>(
          std::move(*trackStateContainer));

  auto constTrackContainer = std::make_shared<Acts::ConstVectorTrackContainer>(
      std::move(*trackContainer));

  ConstTrackContainer constTracks{constTrackContainer,
                                  constTrackStateContainer};

  m_outputTracks(ctx, std::move(constTracks));
  return ProcessCode::SUCCESS;
}

ProcessCode TrackFindingAlgorithm::finalize() {
  ACTS_INFO("TrackFindingAlgorithm statistics:");
  ACTS_INFO("- total seeds: " << m_nTotalSeeds);
  ACTS_INFO("- deduplicated seeds: " << m_nDeduplicatedSeeds);
  ACTS_INFO("- failed seeds: " << m_nFailedSeeds);
  ACTS_INFO("- failed smoothing: " << m_nFailedSmoothing);
  ACTS_INFO("- failed extrapolation: " << m_nFailedExtrapolation);
  ACTS_INFO("- failure ratio seeds: " << static_cast<double>(m_nFailedSeeds) /
                                             m_nTotalSeeds);
  ACTS_INFO("- found tracks: " << m_nFoundTracks);
  ACTS_INFO("- selected tracks: " << m_nSelectedTracks);
  ACTS_INFO("- stopped branches: " << m_nStoppedBranches);
  ACTS_INFO("- skipped second pass: " << m_nSkippedSecondPass);

  auto memoryStatistics =
      m_memoryStatistics.combine([](const auto& a, const auto& b) {
        Acts::VectorMultiTrajectory::Statistics c;
        c.hist = a.hist + b.hist;
        return c;
      });
  std::stringstream ss;
  memoryStatistics.toStream(ss);
  ACTS_DEBUG("Track State memory statistics (averaged):\n" << ss.str());
  return ProcessCode::SUCCESS;
}

// TODO this is somewhat duplicated in AmbiguityResolutionAlgorithm.cpp
// TODO we should make a common implementation in the core at some point
void TrackFindingAlgorithm::computeSharedHits(
    TrackContainer& tracks, const MeasurementSubset& measurements) const {
  // Compute shared hits from all the reconstructed tracks
  // Compute nSharedhits and Update ckf results
  // hit index -> list of multi traj indexes [traj, meas]

  std::vector<std::size_t> firstTrackOnTheHit(
      measurements.container().size(), std::numeric_limits<std::size_t>::max());
  std::vector<std::size_t> firstStateOnTheHit(
      measurements.container().size(), std::numeric_limits<std::size_t>::max());

  for (auto track : tracks) {
    for (auto state : track.trackStatesReversed()) {
      if (!state.typeFlags().isMeasurement()) {
        continue;
      }

      std::size_t hitIndex = state.getUncalibratedSourceLink()
                                 .template get<IndexSourceLink>()
                                 .index();

      // Check if hit not already used
      if (firstTrackOnTheHit.at(hitIndex) ==
          std::numeric_limits<std::size_t>::max()) {
        firstTrackOnTheHit.at(hitIndex) = track.index();
        firstStateOnTheHit.at(hitIndex) = state.index();
        continue;
      }

      // if already used, control if first track state has been marked
      // as shared
      std::size_t indexFirstTrack = firstTrackOnTheHit.at(hitIndex);
      std::size_t indexFirstState = firstStateOnTheHit.at(hitIndex);

      auto firstState = tracks.getTrack(indexFirstTrack)
                            .container()
                            .trackStateContainer()
                            .getTrackState(indexFirstState);
      firstState.typeFlags().setIsSharedHit();

      // Decorate this track state
      state.typeFlags().setIsSharedHit();
    }
  }
}

}  // namespace ActsExamples
