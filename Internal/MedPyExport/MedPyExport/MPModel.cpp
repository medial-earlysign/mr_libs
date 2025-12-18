#include "MPModel.h"
#include "MPSamples.h"

#include "InfraMed/InfraMed/MedConvert.h"
#include "InfraMed/InfraMed/InfraMed.h"
#include "InfraMed/InfraMed/Utils.h"
#include "MedUtils/MedUtils/MedUtils.h"
#include "InfraMed/InfraMed/MedPidRepository.h"
#include "MedProcessTools/MedProcessTools/MedModel.h"
//#include "MedProcessTools/MedProcessTools/SampleFilter.h"

const int MPModelStage::LEARN_REP_PROCESSORS = MED_MDL_LEARN_REP_PROCESSORS;
const int MPModelStage::LEARN_FTR_GENERATORS = MED_MDL_LEARN_FTR_GENERATORS;
const int MPModelStage::APPLY_FTR_GENERATORS = MED_MDL_APPLY_FTR_GENERATORS;
const int MPModelStage::LEARN_FTR_PROCESSORS = MED_MDL_LEARN_FTR_PROCESSORS;
const int MPModelStage::APPLY_FTR_PROCESSORS = MED_MDL_APPLY_FTR_PROCESSORS;
const int MPModelStage::LEARN_PREDICTOR = MED_MDL_LEARN_PREDICTOR;
const int MPModelStage::APPLY_PREDICTOR = MED_MDL_APPLY_PREDICTOR;
const int MPModelStage::INSERT_PREDS = MED_MDL_INSERT_PREDS;
const int MPModelStage::LEARN_POST_PROCESSORS = MED_MDL_LEARN_POST_PROCESSORS;
const int MPModelStage::APPLY_POST_PROCESSORS = MED_MDL_APPLY_POST_PROCESSORS;
const int MPModelStage::END = MED_MDL_END;

static_assert(MPModelStage::END == 10, "med model was changed");

MPModel::MPModel() { o = new MedModel(); };
MPModel::~MPModel() { delete o; o = nullptr; };
void MPModel::init_from_json_file(const std::string& fname) { o->init_from_json_file(fname); };
std::vector<std::string> MPModel::init_from_json_file_with_alterations(const std::string& fname, std::vector<std::string> json_alt) { o->init_from_json_file_with_alterations(fname, json_alt); return json_alt; };
void MPModel::add_pre_processors_json_string_to_model(const std::string &in_json, const std::string &fname, bool add_rep_first) {
	std::vector<std::string> empty_alt; 
	o->add_pre_processors_json_string_to_model(in_json, fname, empty_alt, add_rep_first); 
}
std::vector<std::string> MPModel::get_required_signal_names() { std::vector<std::string> ret; o->get_required_signal_names(ret); return ret; };
int MPModel::learn(MPPidRepository* rep, MPSamples* samples) { 
#ifdef AM_API_FOR_CLIENT
	rep->finish_load_data();
#endif
	return o->learn(*((MedPidRepository*)(rep->o)), (MedSamples*)(samples->o)); 
};
int MPModel::apply(MPPidRepository* rep, MPSamples* samples) { 
#ifdef AM_API_FOR_CLIENT
	rep->finish_load_data();
#endif
	return o->apply(*((MedPidRepository*)(rep->o)), *((MedSamples*)(samples->o))); 
};
int MPModel::learn(MPPidRepository* rep, MPSamples* samples, int start_stage, int end_stage) { 
#ifdef AM_API_FOR_CLIENT
	rep->finish_load_data();
#endif
	return o->learn(*((MedPidRepository*)(rep->o)), (MedSamples*)(samples->o),(MedModelStage)start_stage, (MedModelStage)end_stage);
};
int MPModel::apply(MPPidRepository* rep, MPSamples* samples, int start_stage, int end_stage) { 
#ifdef AM_API_FOR_CLIENT
	rep->finish_load_data();
#endif
	return o->apply(*((MedPidRepository*)(rep->o)), *((MedSamples*)(samples->o)), (MedModelStage)start_stage, (MedModelStage)end_stage);
};

int MPModel::write_to_file(const std::string &fname) { return val_or_exception(o->write_to_file(fname),"Could not write to file"); };
int MPModel::read_from_file(const std::string &fname) { return val_or_exception(o->read_from_file(fname),"Could not read from file"); };

MPFeatures MPModel::MEDPY_GET_features() { return MPFeatures(&o->features); };
MPPredictor MPModel::MEDPY_GET_predictor() { 
	MPPredictor pred= MPPredictor();
	pred.o = o->predictor;
	return pred; 
}

void MPModel::clear() { o->clear();  };

int MPModel::MEDPY_GET_verbosity() { return o->verbosity; };
void MPModel::MEDPY_SET_verbosity(int new_vval) { o->verbosity = new_vval; };

void MPModel::add_feature_generators(std::string& name, std::vector<std::string>& signals) { o->add_feature_generators(name, signals); };
void MPModel::add_feature_generators(std::string& name, std::vector<std::string>& signals, std::string init_string) { o->add_feature_generators(name, signals, init_string); };
void MPModel::add_feature_generator(std::string& name, std::string& signal) { o->add_feature_generator(name, signal); };
void MPModel::add_feature_generators(std::string& name, std::string& signal, std::string init_string) { o->add_feature_generators(name, signal, init_string); };
void MPModel::add_age() { o->add_age(); };
void MPModel::add_gender() { o->add_gender(); };
void MPModel::get_all_features_names(std::vector<std::string> &feat_names, int before_process_set) { o->get_all_features_names(feat_names, before_process_set); };
void MPModel::add_normalizers() { o->add_normalizers(); };
void MPModel::add_normalizers(std::string init_string) { o->add_normalizers(init_string); };
void MPModel::add_normalizers(std::vector<std::string>& features) { o->add_normalizers(features); };
void MPModel::add_normalizers(std::vector<std::string>& features, string init_string) { o->add_normalizers(features, init_string); };
void MPModel::add_imputers() { o->add_imputers(); };
void MPModel::add_imputers(std::string init_string) { o->add_imputers(init_string); };
void MPModel::add_imputers(std::vector<std::string>& features) { o->add_imputers(features); };
void MPModel::add_imputers(std::vector<std::string>& features, std::string init_string) { o->add_imputers(features, init_string); };
void MPModel::add_rep_processor_to_set(int i_set, const std::string &init_string) { o->add_rep_processor_to_set(i_set, init_string); };
void MPModel::add_feature_generator_to_set(int i_set, const std::string &init_string) { o->add_feature_generator_to_set(i_set, init_string); };
void MPModel::add_feature_processor_to_set(int i_set, int duplicate, const std::string &init_string) { o->add_feature_processor_to_set(i_set, duplicate, init_string); };
void MPModel::add_process_to_set(int i_set, int duplicate, const std::string &init_string) { o->add_process_to_set(i_set, duplicate, init_string); };
void MPModel::add_process_to_set(int i_set, const std::string &init_string) { o->add_process_to_set(i_set, init_string); };
void MPModel::set_predictor(MPPredictor& _predictor) { o->set_predictor(_predictor.o); }
void MPModel::make_predictor(std::string name) { o->set_predictor(name); };
void MPModel::set_predictor(std::string name, std::string init_string) { o->set_predictor(name, init_string); };
int MPModel::collect_and_add_virtual_signals(MPPidRepository &rep) { return o->collect_and_add_virtual_signals(*(rep.o)); };
int MPModel::quick_learn_rep_processors(MPPidRepository& rep, MPSamples& samples) { return o->quick_learn_rep_processors(*(rep.o), *(samples.o)); };
int MPModel::learn_rep_processors(MPPidRepository& rep, MPSamples& samples) { return o->learn_rep_processors(*(rep.o), *(samples.o)); };
void MPModel::filter_rep_processors() { o->filter_rep_processors(); };
int MPModel::learn_feature_generators(MPPidRepository &rep, MPSamples *learn_samples) { return o->learn_feature_generators(*(rep.o), (*learn_samples).o); };
int MPModel::generate_all_features(MPPidRepository &rep, MPSamples *samples, MPFeatures &features, std::vector<std::string> req_feature_generators) { 
	unordered_set<string> req_feature_generators_uos;
	for (auto& s : req_feature_generators)
		req_feature_generators_uos.emplace(s);
	return o->generate_all_features(*(rep.o), (*samples).o, *(features.o), req_feature_generators_uos);
};
int MPModel::learn_and_apply_feature_processors(MPFeatures &features) { return o->learn_and_apply_feature_processors(*(features.o)); };
int MPModel::learn_feature_processors(MPFeatures &features) { return o->learn_feature_processors(*(features.o)); };
int MPModel::apply_feature_processors(MPFeatures &features, bool learning) { return o->apply_feature_processors(*(features.o), learning); };

void MPModel::dprint_process(const std::string &pref, int rp_flag, int fg_flag, int fp_flag, bool pp_flag, bool predictor_type) { return o->dprint_process(pref, rp_flag, fg_flag, fp_flag, pp_flag, predictor_type); };
int MPModel::write_feature_matrix(const std::string mat_fname) { return o->write_feature_matrix(mat_fname); };
MPSerializableObject MPModel::asSerializable() { return MPSerializableObject(o); }
void MPModel::fit_for_repository(MPPidRepository &rep) { o->fit_for_repository(*rep.o); }
void MPModel::calc_contribs(MPMat &mat, MPMat &mat_out) { o->predictor->calc_feature_contribs(*mat.o, *mat_out.o); }
//void MPModel::calc_feature_contribs_conditional(MPMat &mat_x_in, const std::vector<std::string> &features_cond_string, const vector<float> &features_cond_float, MPMat &mat_x_out, MPMat &mat_contribs) {
void MPModel::calc_feature_contribs_conditional(MPMat &mat_x_in, const std::string& features_cond_string, float features_cond_float, MPMat &mat_x_out, MPMat &mat_contribs) {
	unordered_map<string, float> tmp_map;
	tmp_map[features_cond_string] = features_cond_float;
	/*for (int i = 0; i < features_cond_string.size(); i++)
	{
		tmp_map[features_cond_string[i]] = features_cond_float[i];
	}*/
	
	o->predictor->calc_feature_contribs_conditional(*mat_x_in.o, tmp_map, *mat_x_out.o, *mat_contribs.o);
}

void MPModel::apply_model_change(const std::string &change_json_content) {
	vector<ChangeModelInfo> change_reqs;
	ChangeModelInfo::parse_json_string(change_json_content, change_reqs);
	o->change_model(change_reqs);
}

void MPModel::add_post_processors_json_string_to_model(const std::string &in_json,const std::string &fname) {
	o->add_post_processors_json_string_to_model(in_json, fname);
}

void MPModel::train_rep_processor_by_index(int index, MPPidRepository &rep, MPSamples &samples)
{
	if (index < 0)
		throw runtime_error("Invalid index " + to_string(index) + ". Must be positive index");
	if (index >= o->rep_processors.size())
		throw runtime_error("Invalid index " + to_string(index) + ". There are only " + to_string(o->rep_processors.size()) + " rep processors");
	RepProcessor *repository_processor = o->rep_processors[index];

	repository_processor->learn(*rep.o, *samples.o);
}

std::string MPModel::print_rep_processor_by_index(int index) 
{
	if (index < 0)
		throw runtime_error("Invalid index " + to_string(index) + ". Must be positive index");
	if (index >= o->rep_processors.size())
		throw runtime_error("Invalid index " + to_string(index) + ". There are only " + to_string(o->rep_processors.size()) + " rep processors");
	RepProcessor *repository_processor = o->rep_processors[index];
	return repository_processor->object_json();
}

void MPModel::delete_rep_processor_by_index(int index) {
	if (index < 0)
		throw runtime_error("Invalid index " + to_string(index) + ". Must be positive index");
	if (index >= o->rep_processors.size())
		throw runtime_error("Invalid index " + to_string(index) + ". There are only " + to_string(o->rep_processors.size()) + " rep processors");
	delete o->rep_processors[index];
	o->rep_processors.erase(o->rep_processors.begin() + index);
}

std::string MPModel::print_feature_generator_by_index(int index) {
	if (index < 0)
		throw runtime_error("Invalid index " + to_string(index) + ". Must be positive index");
	if (index >= o->generators.size())
		throw runtime_error("Invalid index " + to_string(index) + ". There are only " + to_string(o->generators.size()) + " generators");
	FeatureGenerator *generator = o->generators[index];
	return generator->object_json();
}
std::string MPModel::print_feature_processor_by_index(int index){
	if (index < 0)
		throw runtime_error("Invalid index " + to_string(index) + ". Must be positive index");
	if (index >= o->feature_processors.size())
		throw runtime_error("Invalid index " + to_string(index) + ". There are only " + to_string(o->feature_processors.size()) + " feature_processors");
	FeatureProcessor *ftr_processor = o->feature_processors[index];
	return ftr_processor->object_json();
}

void MPModel::delete_feature_processor_by_index(int index) {
	if (index < 0)
		throw runtime_error("Invalid index " + to_string(index) + ". Must be positive index");
	if (index >= o->feature_processors.size())
		throw runtime_error("Invalid index " + to_string(index) + ". There are only " + to_string(o->feature_processors.size()) + " feature_processors");
	delete o->feature_processors[index];
	o->feature_processors.erase(o->feature_processors.begin() + index);
}

std::string MPModel::print_post_processor_by_index(int index) {
	if (index < 0)
		throw runtime_error("Invalid index " + to_string(index) + ". Must be positive index");
	if (index >= o->post_processors.size())
		throw runtime_error("Invalid index " + to_string(index) + ". There are only " + to_string(o->post_processors.size()) + " post_processors");
	return o->post_processors[index]->object_json();
}

std::string MPModel::get_model_weights_json() {
	return o->object_json();
}

void MPModel::delete_post_processor_by_index(int index) {
	if (index < 0)
		throw runtime_error("Invalid index " + to_string(index) + ". Must be positive index");
	if (index >= o->post_processors.size())
		throw runtime_error("Invalid index " + to_string(index) + ". There are only " + to_string(o->post_processors.size()) + " post_processors");
	delete o->post_processors[index];
	o->post_processors.erase(o->post_processors.begin() + index);
}

void MPModel::train_post_processor_by_index(int index, MPFeatures &features) {
	if (index < 0)
		throw runtime_error("Invalid index " + to_string(index) + ". Must be positive index");
	if (index >= o->post_processors.size())
		throw runtime_error("Invalid index " + to_string(index) + ". There are only " + to_string(o->post_processors.size()) + " post_processors");
	PostProcessor *post_processor = o->post_processors[index];
	post_processor->Learn(*features.o);
}

std::string MPModel::get_model_processors_info() {
	nlohmann::ordered_json js;
	js["rep_processors"] = o->rep_processors.size();
	js["feature_generators"] = o->generators.size();
	js["feature_processors"] = o->feature_processors.size();
	js["post_processors"] = o->post_processors.size();
	return js.dump();
}