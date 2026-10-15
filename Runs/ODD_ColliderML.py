import os
import sys
import pathlib
import acts
#IO/Sim/etc. are implemented in
#`examples`
import acts.examples
import acts.examples.root
#Will only need to reconstruct the 
#simulated data from EDM4HEP files 
from acts.examples.reconstruction import (
    addSeeding,
    SeedingAlgorithm, 
    CkfConfig,
    addCKFTracks,
    TrackSelectorConfig,
    addAmbiguityResolution,
    AmbiguityResolutionConfig,
    addVertexFitting,
    VertexFinder
)
#Need simulation modules for handling 
#true particle hits
from acts.examples.simulation import (
    MomentumConfig,
    EtaConfig,
    PhiConfig,
    ParticleConfig,
    ParticleSelectorConfig,
    addSimParticleSelection,
    addDigitization,
    addDigiParticleSelection
)

#Modules for processing the EDM4HEP
import acts.examples.edm4hep
from acts.examples.edm4hep import PodioReader

#Grab the detector geometry and material maps
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory

#Units package for conversions on the fly
u = acts.UnitConstants

#Params 

#Number of events
NEVENTS = 1
#Magnetic field along the beam axis in Tesla
B_Z = acts.ConstantBField(acts.Vector3(0.0, 0.0, 3.0 * u.T))
#Input path
INPUT_PATH = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/ttbar/v1/runs/1/edm4hep.root"
#Output path
OUT_PATH = "../ColliderML_processed"
#use graph based track seeding
USE_GBTS = False

#Script for processing EDM4HEP files for the ODD Detector with 3T magnetic field
#Much of this can be found in the very general script 
#/Examples/Scripts/Python/full_chain_odd.py

#Geometry and acts directories
geoDir = getOpenDataDetectorDirectory()
actsDir = pathlib.Path(__file__).parent.parent

print(geoDir, actsDir)

#Creating file pointers to the necessary detector configs
oddMaterialMap = ( geoDir / "data/odd-material-maps.root" )
oddDigiConfig = ( actsDir / "Examples/Configs/odd-digi-smearing-config.json" )
oddGbtsLayerMap = ( actsDir / "Examples/Configs/odd-gbts-layer-map.json" )
oddGbtsConnectionTable = ( actsDir / "Examples/Configs/odd-gbts-connection-table.json" )
oddSeedingSel = ( actsDir / "Examples/Configs/odd-seeding-config.json" )
oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

#The detector is the geometry plus the material, combine them
detector = getOpenDataDetector(odd_dir=geoDir, materialDecorator=oddMaterialDeco)
trackingGeometry = detector.trackingGeometry()
decorators = detector.contextDecorators()
rnd = acts.examples.RandomNumbers(seed=42)

#Build the sequencer used to prcoess the events
seq = acts.examples.Sequencer( events = NEVENTS, numThreads = 1,
                                outputDir = OUT_PATH )


seq.addReader( PodioReader(level=acts.logging.DEBUG, inputPath=INPUT_PATH, outputFrame="events", category="events") )

edm4hepReader = acts.examples.edm4hep.EDM4hepSimInputConverter(inputFrame="events",
                inputSimHits=["PixelBarrelReadout",
                            "PixelEndcapReadout",
                            "ShortStripBarrelReadout",
                            "ShortStripEndcapReadout",
                            "LongStripBarrelReadout",
                            "LongStripEndcapReadout"],
                outputParticlesGenerator="particles_generated",
                outputParticlesSimulation="particles_simulated",
                outputSimHits="simhits",
                outputSimVertices="vertices_truth",
                dd4hepDetector=detector,
                trackingGeometry=trackingGeometry,
                sortSimHitsInTime=False,
                particleRMax=1080 * u.mm,
                particleZ=(-3030 * u.mm, 3030 * u.mm),
                particlePtMin=150 * u.MeV,
                level=acts.logging.DEBUG,
)

#Adding EDM4HEP configs
seq.addAlgorithm(edm4hepReader)

#Whiteboard is the `Store` for ACTS
#allowing for thread-safe access
seq.addWhiteboardAlias("particles", edm4hepReader.config.outputParticlesSimulation)

addSimParticleSelection(seq, 
                ParticleSelectorConfig(rho=(0.0, 24 * u.mm), absZ=(0.0, 1.0 * u.m), eta=(-3.0, 3.0), removeNeutral=True) )


addDigitization(seq, trackingGeometry, B_Z, digiConfigFile=oddDigiConfig, 
                outputDirRoot=OUT_PATH, outputDirCsv=OUT_PATH, rnd=rnd)

addDigiParticleSelection(seq, 
                ParticleSelectorConfig(pt=(1.0 * u.GeV, None), eta=(-3.0, 3.0), measurements=(9, None), removeNeutral=True))


#Seeding algorithm checking triplets with track consistent curvature 
addSeeding(seq, trackingGeometry, B_Z, **(dict( seedingAlgorithm=SeedingAlgorithm.Gbts, 
                                                layerMappingConfigFile=oddGbtsLayerMap,
                                                connectorInputConfigFile=oddGbtsConnectionTable) if USE_GBTS else {}),
                initialSigmas=[1 * u.mm, 1 * u.mm, 1 * u.degree, 1 * u.degree, 0 * u.e / u.GeV, 1 * u.ns],
                initialSigmaQoverPt=0.1 * u.e / u.GeV,
                initialSigmaPtRel=0.1,
                initialVarInflation=[1.0] * 6,
                particleHypothesis=acts.ParticleHypothesis.muon,
                geoSelectionConfigFile=oddSeedingSel,
                outputDirRoot=OUT_PATH,
                outputDirCsv=OUT_PATH)


#CKF Configurations constraining the minimum track momentum
#including the Chi2 cutoffs
addCKFTracks(seq, trackingGeometry, B_Z, TrackSelectorConfig(pt=(1.0 * u.GeV, None),
                                                            absEta=(None, 3.0),
                                                            loc0=(-4.0 * u.mm, 4.0 * u.mm),
                                                            nMeasurementsMin=7,
                                                            maxHoles=2,
                                                            maxOutliers=2,
                                                            ),
            CkfConfig(
                chi2CutOffMeasurement=15.0,
                chi2CutOffOutlier=25.0,
                numMeasurementsCutOff=2,
                seedDeduplication=True,
                stayOnSeed=True,
                pixelVolumes=[16, 17, 18],
                stripVolumes=[23, 24, 25],
                maxPixelHoles=1,
                maxStripHoles=2,
                constrainToVolumes=[
                    2,  # beam pipe
                    32,
                    4,  # beam pipe gap
                    16,
                    17,
                    18,  # pixel
                    20,  # PST
                    23,
                    24,
                    25,  # short strip
                    26,
                    8,  # long strip gap
                    28,
                    29,
                    30,  # long strip
                ],
            ),
            outputDirRoot=OUT_PATH, outputDirCsv=OUT_PATH, writeCovMat=True, writeTrackStates=True)

addAmbiguityResolution(seq, AmbiguityResolutionConfig(maximumSharedHits=3, maximumIterations=1000000, nMeasurementsMin=7),
                       outputDirRoot=OUT_PATH, outputDirCsv=OUT_PATH, writeCovMat=True)

addVertexFitting(seq, B_Z, vertexFinder=VertexFinder.AMVF, outputDirRoot=OUT_PATH, outputDirCsv=OUT_PATH)


seq.run()