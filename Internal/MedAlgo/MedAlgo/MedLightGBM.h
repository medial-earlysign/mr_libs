#pragma once

#ifndef __MED_LIGHT_GBM__
#define __MED_LIGHT_GBM__

#include <MedAlgo/MedAlgo/MedAlgo.h>
#include <LightGBM/c_api.h>

#include <vector>
#include <string>
#include <map>

//=============================================
// MedLightGBM Params
//=============================================

struct MedLightGBMParams : public SerializableObject
{
    MedLightGBMParams()
    {
        defaults = "";
        defaults += "boosting_type=gbdt;";
        defaults += "objective=binary;";
        defaults += "metric=binary_logloss,auc;";
        defaults += "metric_freq=1;";
        defaults += "is_training_metric=true;";
        defaults += "max_bin=255;";
        defaults += "num_iterations=200;";
        defaults += "learning_rate=0.05;";
        defaults += "tree_learner=serial;";
        defaults += "num_threads=12;";
        defaults += "min_data_in_leaf=50;";
        defaults += "min_sum_hessian_in_leaf=5.0;";
        defaults += "is_enable_sparse=false;";
        defaults += "num_machines=1;";
        defaults += "feature_fraction=0.8;";
        defaults += "bagging_fraction=0.25;";
        defaults += "bagging_freq=4;";
        defaults += "is_unbalance=true;";
        defaults += "num_leaves=80"; 
    }

    string defaults = "";
    string user_params = "";

    ADD_CLASS_NAME(MedLightGBMParams)
    ADD_SERIALIZATION_FUNCS(defaults, user_params);
};

//=============================================
// MedLightGBM Predictor
//=============================================

class MedLightGBM : public MedPredictor
{
public:
    MedLightGBMParams params;

    // Initialization methods
    int init(map<string, string> &initialization_map);
    int set_params(map<string, string> &initialization_map);
    int init_from_string(string init_str);

    MedLightGBM();
    ~MedLightGBM();

    // Core ML functions
    int Learn(float *x, float *y, const float *w, int nsamples, int nftrs);
    int Learn(float *x, float *y, int nsamples, int nftrs) { return Learn(x, y, NULL, nsamples, nftrs); }
    
    int Predict(float *x, float *&preds, int nsamples, int nftrs) const;
    void predict_single(const vector<float> &x, vector<float> &preds) const;
    void predict_single(const vector<double> &x, vector<double> &preds) const;
    void prepare_predict_single();

    // Feature Importance & SHAP
    void calc_feature_importance(vector<float> &features_importance_scores, const string &general_params, const MedFeatures *features);
    void calc_feature_contribs(MedMat<float> &x, MedMat<float> &contribs) const;

    int n_preds_per_sample() const;
    void export_predictor(const string &output_fname);

    // Serialization
    void pre_serialization();
    void post_deserialization();
    void print(FILE *fp, const string &prefix, int level = 0) const;
    string get_model_str();

    ADD_CLASS_NAME(MedLightGBM)
    ADD_SERIALIZATION_FUNCS(classifier_type, params, model_as_string, model_features, features_count, _mark_learn_done)

private:
    bool _mark_learn_done = false;
    string model_as_string;
    bool prepared_single;

    int num_preds;
    
    // The core C API Handle
    BoosterHandle booster_handle_;

    // Helper to generate LightGBM C API parameter strings
    string build_c_api_param_string() const;
    int get_num_iterations() const;
};

MEDSERIALIZE_SUPPORT(MedLightGBMParams);
MEDSERIALIZE_SUPPORT(MedLightGBM);

#endif