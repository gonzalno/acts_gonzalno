#include <torch/torch.h>
#include <iostream>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TChain.h"
#include "ROOT/RDataFrame.hxx"
#include "TCanvas.h"
#include "TInterpreter.h"
#include "TSystem.h"

int main(int argc, char** argv){

    //Create a structure for storing hit properties
    //to store in an unordered map
    struct props{
        int hit_id;
        std::vector<double> chi2;
        double charge;
        int nchannels;
        int sizeLoc0;
        int sizeLoc1;
        double Loc0;
        double Loc1;
        double VarLoc0;
        double VarLoc1;
        std::vector<double> resLoc0;
        std::vector<double> resLoc1;
        int passing_flag;
        int outlier_flag;
        int failing_flag;
    };

    //Conversion factor for nfC -> nElectrons
    double nElectrons_per_fC = 1e-15*1e+19/1.602;
    //Conversion factor for nElectrons -> keV
    double fC_to_keV = nElectrons_per_fC*3.6*1e-3;

    //argv wraps all input arguments
    std::string input_dir = argv[1];
    std::string tree_name = argv[2];
    std::string hist_output = argv[3];
    std::string torch_tensor_output = argv[4];

    // Redirect temporary Cling dictionary files to ./tmp/root_dicts
    gSystem->SetBuildDir("./tmp/root_dicts", true);

    // Generate dictionaries for 2D vectors cleanly
    gInterpreter->GenerateDictionary("std::vector<std::vector<int>>", "vector");
    gInterpreter->GenerateDictionary("std::vector<std::vector<double>>", "vector");
    gInterpreter->GenerateDictionary("std::vector<std::vector<size_t>>", "vector");

    TH1::AddDirectory(kFALSE);

    //Path where all trackstate data is stored
    std::string path =  (input_dir).c_str();

    //Make histogram objects to collect stats

    //Chi2
    TH1F *chi2_passing = new TH1F("Chi2_passing", "; Chi2; Counts", 100, 0, 20);
    TH1F *chi2_outlier = new TH1F("Chi2_outlier", "; Chi2; Counts", 100, 0, 20);
    TH1F *chi2_failing = new TH1F("Chi2_failing", "; Chi2; Counts", 100, 0, 20);

    //Cluster properties

    //Charge Depostied
    TH1F *charge_passing = new TH1F("Charge_passing", "; nElectrons; Counts", 40, 1e2, 1e4);
    TH1F *charge_outlier = new TH1F("Charge_outlier", "; nElectrons; Counts", 40, 1e2, 1e4);
    TH1F *charge_failing = new TH1F("Charge_failing", "; nElectrons; Counts", 40, 1e2, 1e4);
    //Cluster size in nchannels
    TH1F *nchannels_passing = new TH1F("nchannels_passing", "; nchannels; Counts", 10, 0, 10);
    TH1F *nchannels_outlier = new TH1F("nchannels_outlier", "; nchannels; Counts", 10, 0, 10);
    TH1F *nchannels_failing = new TH1F("nchannels_failing", "; nchannels; Counts", 10, 0, 10);
    //Cluster size in Local Coordinate L0 
    TH1F *l0_size_passing = new TH1F("l0_size_passing", "; L0 Size; Counts", 10, 0, 10);
    TH1F *l0_size_outlier = new TH1F("l0_size_outlier", "; L0 Size; Counts", 10, 0, 10);
    TH1F *l0_size_failing = new TH1F("l0_size_failing", "; L0 Size; Counts", 10, 0, 10);
    //Cluster size in Local Coordinate L0 
    TH1F *l1_size_passing = new TH1F("l1_size_passing", "; L1 Size; Counts", 10, 0, 10);
    TH1F *l1_size_outlier = new TH1F("l1_size_outlier", "; L1 Size; Counts", 10, 0, 10);
    TH1F *l1_size_failing = new TH1F("l1_size_failing", "; L1 Size; Counts", 10, 0, 10);
    //L0 for clusters
    TH1F *l0_passing = new TH1F("l0_passing", "; L0 Cluster; Counts", 40, -1e2, 1e2);
    TH1F *l0_outlier = new TH1F("l0_outlier", "; L0 Cluster; Counts", 40, -1e2, 1e2);
    TH1F *l0_failing = new TH1F("l0_failing", "; L0 Cluster; Counts", 40, -1e2, 1e2);
    //L1 for clusters
    TH1F *l1_passing = new TH1F("l1_passing", "; L1 Cluster; Counts", 40, -1e2, 1e2);
    TH1F *l1_outlier = new TH1F("l1_outlier", "; L1 Cluster; Counts", 40, -1e2, 1e2);
    TH1F *l1_failing = new TH1F("l1_failing", "; L1 Cluster; Counts", 40, -1e2, 1e2);
    //Residuals computed from l0 cluster - l0 predicted
    TH1F *l0_res_passing = new TH1F("l0_res_passing", "; L0 Cluster - L0 Pred.; Counts", 40, -10, 10);
    TH1F *l0_res_outlier = new TH1F("l0_res_outlier", "; L0 Cluster - L0 Pred.; Counts", 40, -10, 10);
    TH1F *l0_res_failing = new TH1F("l0_res_failing", "; L0 Cluster - L0 Pred.; Counts", 40, -10, 10);
    //Variance of the measurements on L0 
    TH1F *l0_var_passing = new TH1F("var_passing", "; L0 Variance; Counts", 20, 0, 5e-4);
    TH1F *l0_var_outlier = new TH1F("var_outlier", "; L0 Variance; Counts", 20, 0, 5e-4);
    TH1F *l0_var_failing = new TH1F("var_failing", "; L0 Variance; Counts", 20, 0, 5e-4);


    //Loop through the set of files
    for (const auto & f : std::filesystem::directory_iterator(path)){
        std::cout << "Processing file " << f.path().c_str() << std::endl;
	//Splitting the output file path for unique naming of the torch
	//outputs
	int k = 0;
	std::string file_name_id;
	for (auto& sub_path : f.path()){
		if (k!=4){
			k+=1;
		}
		else{
			file_name_id = sub_path.stem().string();
		}
	}
        //Container for tensors used in ML training
        std::vector<torch::Tensor> features;
        std::vector<torch::Tensor> labels;
        std::vector<torch::Tensor> regression;
        //Create an unordered map
        std::unordered_map<int, props> hit_map;
        //Open the file
        TFile data(f.path().c_str());
        //Register the tree
        TTree *tree = (TTree*)data.Get(tree_name.c_str());
        //Set the branch addresses
        //Custom
        std::vector<std::vector<int>>*cand_hit_idx = nullptr;
        std::vector<std::vector<int>>*cand_passing_bool = nullptr;
        std::vector<std::vector<int>>*cand_outlier_bool = nullptr;
        std::vector<std::vector<int>>*cand_failed_bool = nullptr;
        std::vector<std::vector<double>>*cand_chi2 = nullptr;
        std::vector<std::vector<double>>*cand_charge_dep = nullptr;
        std::vector<std::vector<int>>*cand_nchannels = nullptr;
        std::vector<std::vector<int>>*cand_sizeLoc0 = nullptr;
        std::vector<std::vector<int>>*cand_sizeLoc1 = nullptr;
        std::vector<std::vector<int>>*cand_incidentEta = nullptr;
        std::vector<std::vector<int>>*cand_incidentPhi = nullptr;
        std::vector<std::vector<double>>*cand_Loc0 = nullptr;
        std::vector<std::vector<double>>*cand_Loc1 = nullptr;
        std::vector<std::vector<double>>*cand_VarLoc0 = nullptr;
        std::vector<std::vector<double>>*cand_VarLoc1 = nullptr;
        //Default branches
        unsigned int nStates = 0;
        std::vector<double>*track_eQoP_pred = nullptr;
        std::vector<double>*track_eTHETA_pred = nullptr;
        std::vector<double>*track_ePHI_pred = nullptr;
        std::vector<double>*track_loc0_pred = nullptr;
        std::vector<double>*track_loc1_pred = nullptr;
        //Assign the vector pointers to the branch names
        tree -> SetBranchAddress("all_cand_hit_idx", &cand_hit_idx);
        tree -> SetBranchAddress("isCand_passing", &cand_passing_bool);
        tree -> SetBranchAddress("isCand_outlier", &cand_outlier_bool);
        tree -> SetBranchAddress("isCand_failure", &cand_failed_bool);
        tree -> SetBranchAddress("all_cand_chi2", &cand_chi2);
        tree -> SetBranchAddress("all_cand_charge_dep", &cand_charge_dep);
        tree -> SetBranchAddress("all_cand_nchannels", &cand_nchannels);
        tree -> SetBranchAddress("all_cand_sizeLoc0", &cand_sizeLoc0);
        tree -> SetBranchAddress("all_cand_sizeLoc1", &cand_sizeLoc1);
        tree -> SetBranchAddress("all_cand_Loc0", &cand_Loc0);
        tree -> SetBranchAddress("all_cand_VarLoc0", &cand_VarLoc0);
        tree -> SetBranchAddress("all_cand_Loc1", &cand_Loc1);
        tree -> SetBranchAddress("all_cand_VarLoc1", &cand_VarLoc1);
        tree -> SetBranchAddress("nStates", &nStates);
        tree -> SetBranchAddress("eQOP_prt", &track_eQoP_pred);
        tree -> SetBranchAddress("eTHETA_prt", &track_eTHETA_pred);
        tree -> SetBranchAddress("ePHI_prt", &track_ePHI_pred);
        tree -> SetBranchAddress("eLOC0_prt", &track_loc0_pred);
        tree -> SetBranchAddress("eLOC1_prt", &track_loc1_pred);

        int ntracks = tree -> GetEntries();

        //Loop through the tracks
        for ( int entry = 0; entry < ntracks; entry++){
            tree -> GetEntry(entry);
            //Loop through track states
            for (int state = 0; state < nStates; state++){ 
                //Grab track properties
                double track_loc0 = (*track_loc0_pred)[state];
                double track_loc1 = (*track_loc1_pred)[state];
                double track_eQoP = (*track_eQoP_pred)[state];
                double track_ePHI = (*track_ePHI_pred)[state];
                double track_eTHETA = (*track_eTHETA_pred)[state];
                //Loop through candidates           
                for ( int cand_indx = 0; cand_indx < (*cand_passing_bool)[state].size(); cand_indx++){
                    //Grab hit id
                    int indx = (*cand_hit_idx)[state][cand_indx];
                    //skip if the idx is invalid
                    if (indx==-999){
                        continue;
                    }
                    //Create a feature vector for ML output
                    std::vector<float> feature_vec;
                    //Create a regression target, the chi2 for 
                    //basline
                    regression.push_back(torch::tensor({(*cand_chi2)[state][cand_indx]}));

                    //Pushback to the feature vector 
                    //Track Features
                    // q/p, track phi, track theta,
                    //Candidate features
                    //res loc0, res loc1,
                    //Var Loc 0, Var Loc 1, size Loc 0, size Loc 1
                    //nchannels, charge depositon, chi2
                    feature_vec.push_back(track_eQoP);
                    feature_vec.push_back(track_ePHI);
                    feature_vec.push_back(track_eTHETA);
                    feature_vec.push_back(track_loc0 - (*cand_Loc0)[state][cand_indx]);
                    feature_vec.push_back(track_loc1 - (*cand_Loc1)[state][cand_indx]);
                    feature_vec.push_back((*cand_VarLoc0)[state][cand_indx]);
                    feature_vec.push_back((*cand_VarLoc1)[state][cand_indx]);
                    feature_vec.push_back((*cand_sizeLoc0)[state][cand_indx]);
                    feature_vec.push_back((*cand_sizeLoc1)[state][cand_indx]);
                    feature_vec.push_back((*cand_nchannels)[state][cand_indx]);
                    feature_vec.push_back((*cand_charge_dep)[state][cand_indx]);
		            feature_vec.push_back((*cand_chi2)[state][cand_indx]);
                    //Convert the cluster feature_vector to torch tensor
                    torch::Tensor feature_tensor = torch::tensor(feature_vec);

                    //Create a class label item for ML training
                    int class_label;


                    //Create the mapping
                    int pass_bool = (*cand_passing_bool)[state][cand_indx];
                    int outlier_bool = (*cand_outlier_bool)[state][cand_indx];
                    int failing_bool = (*cand_failed_bool)[state][cand_indx];

                    //Passing 
                    if (pass_bool==1){
                        class_label = 1;
                    }
                    //Outlier  or Failing
                    if (outlier_bool==1 || failing_bool==1){
                        class_label = 0; 
                    }

                    features.push_back(feature_tensor);
                    labels.push_back(torch::tensor({class_label}));

                    if (hit_map.find(indx)==hit_map.end()){
                        //Create structure for cands
                        props cand_props;
                        //Attach the properties
                        cand_props.hit_id = indx;
                        cand_props.charge = (*cand_charge_dep)[state][cand_indx];
                        cand_props.nchannels = (*cand_nchannels)[state][cand_indx];
                        cand_props.sizeLoc0 = (*cand_sizeLoc0)[state][cand_indx];
                        cand_props.sizeLoc1 = (*cand_sizeLoc1)[state][cand_indx];
                        cand_props.Loc0 = (*cand_Loc0)[state][cand_indx];
                        cand_props.Loc1 = (*cand_Loc1)[state][cand_indx];
                        cand_props.VarLoc0 = (*cand_VarLoc0)[state][cand_indx];
                        cand_props.VarLoc1 =  (*cand_VarLoc1)[state][cand_indx];
                        //Residuals depend on the track and since more than one track can attempt
                        //to claim the same hit, residuals is a vector where each index <-> track
                        cand_props.chi2.push_back( (*cand_chi2)[state][cand_indx] );
                        cand_props.resLoc0.push_back( (*cand_Loc0)[state][cand_indx] - track_loc0 );
                        cand_props.resLoc1.push_back( (*cand_Loc1)[state][cand_indx] - track_loc1 );
                        //Similar logic for the residuals applies to the booleans but we can 
                        //keep track of status with an integer flag
                        cand_props.passing_flag = pass_bool;
                        cand_props.outlier_flag = outlier_bool;
                        cand_props.failing_flag = failing_bool;
                        //Add to the map
                        hit_map[indx] = cand_props;
                    }

                    //Handle the case where a hit is used twice
                    else {
                        //Only update the boolean flags
                        //and add the residuals
                        hit_map[indx].chi2.push_back( (*cand_chi2)[state][cand_indx] );
                        hit_map[indx].resLoc0.push_back( (*cand_Loc0)[state][cand_indx] - track_loc0 );
                        hit_map[indx].resLoc1.push_back( (*cand_Loc1)[state][cand_indx] - track_loc1 );
                        hit_map[indx].passing_flag += pass_bool;
                        hit_map[indx].outlier_flag += outlier_bool;
                        hit_map[indx].failing_flag += failing_bool;
                    }
                }
            }
        }

        //Loop through the map and add to histograms in the hierarchy
        //passing -> outlier -> failing so that a hit passes if it passed
        //for any track, an outlier if it was an outlier for any track and did 
        //not pass for any track, and failing if it failed for all tracks
        for (const auto[indx, cand] : hit_map){
            if (cand.passing_flag > 0){
                //Fill
                chi2_passing -> Fill(*(std::min_element( cand.chi2.begin(), cand.chi2.end() )));
                charge_passing -> Fill(cand.charge*nElectrons_per_fC);
                nchannels_passing -> Fill(cand.nchannels);
                l0_size_passing -> Fill(cand.sizeLoc0);
                l1_size_passing -> Fill(cand.sizeLoc1);
                l0_passing -> Fill(cand.Loc0);
                l1_passing -> Fill(cand.Loc1);
                l0_res_passing -> Fill(*(std::min_element( cand.resLoc0.begin(), cand.resLoc0.end() )));
                l0_var_passing -> Fill(cand.VarLoc0);
            }
            else if (cand.passing_flag==0 && cand.outlier_flag > 0){
                //Fill
                chi2_outlier -> Fill(*(std::min_element( cand.chi2.begin(), cand.chi2.end() )));
                charge_outlier -> Fill(cand.charge*nElectrons_per_fC);
                nchannels_outlier -> Fill(cand.nchannels);
                l0_size_outlier -> Fill(cand.sizeLoc0);
                l1_size_outlier -> Fill(cand.sizeLoc1);
                l0_outlier-> Fill(cand.Loc0);
                l1_outlier-> Fill(cand.Loc1);
                l0_res_outlier -> Fill(*(std::min_element( cand.resLoc0.begin(), cand.resLoc0.end() )));
                l0_var_outlier -> Fill(cand.VarLoc0);
            }
            else if (cand.failing_flag == cand.resLoc0.size()){
                //Fill
                chi2_failing -> Fill(*(std::min_element( cand.chi2.begin(), cand.chi2.end() )));
                charge_failing -> Fill(cand.charge*nElectrons_per_fC);
                nchannels_failing -> Fill(cand.nchannels);
                l0_size_failing -> Fill(cand.sizeLoc0);
                l1_size_failing -> Fill(cand.sizeLoc1);
                l0_failing -> Fill(cand.Loc0);
                l1_failing -> Fill(cand.Loc1);
                l0_res_failing -> Fill(*(std::min_element( cand.resLoc0.begin(), cand.resLoc0.end() )));
                l0_var_failing -> Fill(cand.VarLoc0);
            }
        }
    std::cout << labels.size() << std::endl;
    std::cout << regression.size() << std::endl;
    std::cout << features.size() << std::endl;

    //Write to the torch output files for ML training
    torch::save(labels, (torch_tensor_output+"_"+file_name_id+"_labels.pt").c_str() );
    torch::save(regression, (torch_tensor_output+"_"+file_name_id+"_regression.pt").c_str() );
    torch::save(features, (torch_tensor_output+"_"+file_name_id+"_features.pt").c_str() );
    //End of loop through files 
    }

    //Make output file for histograms 
    TFile *hist_output_file = TFile::Open((hist_output+".root").c_str(), "RECREATE");
    hist_output_file -> cd();
    //Write passing histograms
    chi2_passing -> Write();
    charge_passing -> Write();
    nchannels_passing -> Write();
    l0_size_passing -> Write();
    l1_size_passing -> Write();
    l0_passing -> Write();
    l1_passing -> Write();
    l0_res_passing -> Write();
    l0_var_passing -> Write();
    //Write outlier histograms
    chi2_outlier -> Write();
    charge_outlier -> Write();
    nchannels_outlier -> Write();
    l0_size_outlier -> Write();
    l1_size_outlier -> Write();
    l0_outlier -> Write();
    l1_outlier -> Write();
    l0_res_outlier-> Write();
    l0_var_outlier -> Write();
    //Write failing histograms
    chi2_failing -> Write();
    charge_failing -> Write();
    nchannels_failing -> Write();
    l0_size_failing -> Write();
    l1_size_failing -> Write();
    l0_failing -> Write();
    l1_failing -> Write();
    l0_res_failing -> Write();
    l0_var_failing -> Write();
    //Output the file and close
    hist_output_file -> Write();
    hist_output_file -> Close();
    //End of program
    return 0;
}
