#include "MedLightGBM.h"

#include <LightGBM/application.h>

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/text_reader.h>

#include <LightGBM/network.h>
#include <LightGBM/dataset.h>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/boosting.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/metric.h>
#include <LightGBM/c_api.h>

// #include "predictor.hpp"

#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/../../src/application/predictor.hpp>

#include <cstdio>
#include <ctime>

#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Logger/Logger/Logger.h>
#define LOCAL_SECTION LOG_MEDALGO
#define LOCAL_LEVEL LOG_DEF_LEVEL
extern MedLogger global_logger;

namespace LightGBM_MES
{
	using Log = LightGBM::Log;

	Application::Application(int argc, char **argv)
	{
		LoadParameters(argc, argv);
		// set number of threads for openmp
		OMP_SET_NUM_THREADS(config_.num_threads);
		if (config_.data.size() == 0 && config_.task != LightGBM::TaskType::kConvertModel)
		{
			LightGBM::Log::Fatal("No training/prediction data, application quit");
		}

		if (config_.device_type == std::string("cuda"))
		{
			LightGBM::LGBM_config_::current_device = lgbm_device_cuda;
		}
	}

	Application::~Application()
	{
		if (config_.is_parallel)
		{
			LightGBM::Network::Dispose();
		}
	}

	void Application::Run()
	{
		if (config_.task == LightGBM::TaskType::kPredict || config_.task == LightGBM::TaskType::KRefitTree)
		{
			InitPredict();
			Predict();
		}
		else if (config_.task == LightGBM::TaskType::kConvertModel)
		{
			ConvertModel();
		}
		else
		{
			InitTrain();
			Train();
		}
	}

	void Application::LoadParameters(int argc, char **argv)
	{
		std::unordered_map<std::string, std::vector<std::string>> all_params;
		std::unordered_map<std::string, std::string> params;
		for (int i = 1; i < argc; ++i)
		{
			LightGBM::Config::KV2Map(&all_params, argv[i]);
		}
		// read parameters from config file
		bool config_file_ok = true;
		if (all_params.count("config") > 0)
		{
			LightGBM::TextReader<size_t> config_reader(all_params["config"][0].c_str(), false);
			config_reader.ReadAllLines();
			if (!config_reader.Lines().empty())
			{
				for (auto &line : config_reader.Lines())
				{
					// remove str after "#"
					if (line.size() > 0 && std::string::npos != line.find_first_of("#"))
					{
						line.erase(line.find_first_of("#"));
					}
					line = LightGBM::Common::Trim(line);
					if (line.size() == 0)
					{
						continue;
					}
					LightGBM::Config::KV2Map(&all_params, line.c_str());
				}
			}
			else
			{
				config_file_ok = false;
			}
		}
		LightGBM::Config::SetVerbosity(all_params);
		// de-duplicate params
		LightGBM::Config::KeepFirstValues(all_params, &params);
		if (!config_file_ok)
		{
			LightGBM::Log::Warning("Config file %s doesn't exist, will ignore", params["config"].c_str());
		}
		LightGBM::ParameterAlias::KeyAliasTransform(&params);
		config_.Set(params);
		LightGBM::Log::Info("Finished loading parameters");
	}

	void Application::LoadData()
	{
		auto start_time = std::chrono::high_resolution_clock::now();
		std::unique_ptr<LightGBM::Predictor> predictor;
		// prediction is needed if using input initial model(continued train)
		LightGBM::PredictFunction predict_fun = nullptr;
		// need to continue training
		if (boosting_->NumberOfTotalModel() > 0 && config_.task != LightGBM::TaskType::KRefitTree)
		{
			predictor.reset(new LightGBM::Predictor(boosting_.get(), 0, -1, true, false, false, false, -1, -1));
			predict_fun = predictor->GetPredictFunction();
		}

		// sync up random seed for data partition
		if (config_.is_data_based_parallel)
		{
			config_.data_random_seed = LightGBM::Network::GlobalSyncUpByMin(config_.data_random_seed);
		}

		LightGBM::Log::Debug("Loading train file...");
		LightGBM::DatasetLoader dataset_loader(config_, predict_fun,
									 config_.num_class, config_.data.c_str());
		// load Training data
		if (config_.is_data_based_parallel)
		{
			// load data for distributed training
			train_data_.reset(dataset_loader.LoadFromFile(config_.data.c_str(),
														  LightGBM::Network::rank(), LightGBM::Network::num_machines()));
		}
		else
		{
			// load data for single machine
			train_data_.reset(dataset_loader.LoadFromFile(config_.data.c_str(), 0, 1));
		}
		// need save binary file
		if (config_.save_binary)
		{
			train_data_->SaveBinaryFile(nullptr);
		}
		// create training metric
		if (config_.is_provide_training_metric)
		{
			for (auto metric_type : config_.metric)
			{
				auto metric = std::unique_ptr<LightGBM::Metric>(LightGBM::Metric::CreateMetric(metric_type, config_));
				if (metric == nullptr)
				{
					continue;
				}
				metric->Init(train_data_->metadata(), train_data_->num_data());
				train_metric_.push_back(std::move(metric));
			}
		}
		train_metric_.shrink_to_fit();

		if (!config_.metric.empty())
		{
			// only when have metrics then need to construct validation data

			// Add validation data, if it exists
			for (size_t i = 0; i < config_.valid.size(); ++i)
			{
				LightGBM::Log::Debug("Loading validation file #%zu...", (i + 1));
				// add
				auto new_dataset = std::unique_ptr<LightGBM::Dataset>(
					dataset_loader.LoadFromFileAlignWithOtherDataset(
						config_.valid[i].c_str(),
						train_data_.get()));
				valid_datas_.push_back(std::move(new_dataset));
				// need save binary file
				if (config_.save_binary)
				{
					valid_datas_.back()->SaveBinaryFile(nullptr);
				}

				// add metric for validation data
				valid_metrics_.emplace_back();
				for (auto metric_type : config_.metric)
				{
					auto metric = std::unique_ptr<LightGBM::Metric>(LightGBM::Metric::CreateMetric(metric_type, config_));
					if (metric == nullptr)
					{
						continue;
					}
					metric->Init(valid_datas_.back()->metadata(),
								 valid_datas_.back()->num_data());
					valid_metrics_.back().push_back(std::move(metric));
				}
				valid_metrics_.back().shrink_to_fit();
			}
			valid_datas_.shrink_to_fit();
			valid_metrics_.shrink_to_fit();
		}
		auto end_time = std::chrono::high_resolution_clock::now();
		// output used time on each iteration
		LightGBM::Log::Info("Finished loading data in %f seconds",
				  std::chrono::duration<double, std::milli>(end_time - start_time) * 1e-3);
	}

	void Application::InitTrain()
	{
		if (config_.is_parallel)
		{
			// need init network
			LightGBM::Network::Init(config_);
			LightGBM::Log::Info("Finished initializing network");
			config_.feature_fraction_seed =
				LightGBM::Network::GlobalSyncUpByMin(config_.feature_fraction_seed);
			config_.feature_fraction =
				LightGBM::Network::GlobalSyncUpByMin(config_.feature_fraction);
			config_.drop_seed =
				LightGBM::Network::GlobalSyncUpByMin(config_.drop_seed);
		}

		// create boosting
		boosting_.reset(
			LightGBM::Boosting::CreateBoosting(config_.boosting,
									 config_.input_model.c_str()));
		// create objective function
		objective_fun_.reset(
			LightGBM::ObjectiveFunction::CreateObjectiveFunction(config_.objective,
													   config_));
		// load training data
		LoadData();
		if (config_.task == LightGBM::TaskType::kSaveBinary)
		{
			LightGBM::Log::Info("Save data as binary finished, exit");
			exit(0);
		}
		// initialize the objective function
		objective_fun_->Init(train_data_->metadata(), train_data_->num_data());
		// initialize the boosting
		boosting_->Init(&config_, train_data_.get(), objective_fun_.get(),
						LightGBM::Common::ConstPtrInVectorWrapper<LightGBM::Metric>(train_metric_));
		// add validation data into boosting
		for (size_t i = 0; i < valid_datas_.size(); ++i)
		{
			boosting_->AddValidDataset(valid_datas_[i].get(),
									   LightGBM::Common::ConstPtrInVectorWrapper<LightGBM::Metric>(valid_metrics_[i]));
			LightGBM::Log::Debug("Number of data points in validation set #%zu: %d", i + 1, valid_datas_[i]->num_data());
		}
		LightGBM::Log::Info("Finished initializing training");
	}

	void Application::Train()
	{
		LightGBM::Log::Info("Started training...");
		boosting_->Train(config_.snapshot_freq, config_.output_model);
		boosting_->SaveModelToFile(0, -1, config_.saved_feature_importance_type,
								   config_.output_model.c_str());
		// convert model to if-else statement code
		if (config_.convert_model_language == std::string("cpp"))
		{
			boosting_->SaveModelToIfElse(-1, config_.convert_model.c_str());
		}
		LightGBM::Log::Info("Finished training");
	}

	void Application::Predict()
	{
		if (config_.task == LightGBM::TaskType::KRefitTree)
		{
			// create predictor
			LightGBM::Predictor predictor(boosting_.get(), 0, -1, false, true, false, false, 1, 1);
			predictor.Predict(config_.data.c_str(), config_.output_result.c_str(), config_.header, config_.predict_disable_shape_check,
							  config_.precise_float_parser);
			LightGBM::TextReader<int> result_reader(config_.output_result.c_str(), false);
			result_reader.ReadAllLines();

			size_t nrow = result_reader.Lines().size();
			size_t ncol = 0;
			if (nrow > 0)
			{
				ncol = LightGBM::Common::StringToArray<int>(result_reader.Lines()[0], '\t').size();
			}
			std::vector<int> pred_leaf;
			pred_leaf.resize(nrow * ncol);

#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
			for (int irow = 0; irow < static_cast<int>(nrow); ++irow)
			{
				auto line_vec = LightGBM::Common::StringToArray<int>(result_reader.Lines()[irow], '\t');
				CHECK_EQ(line_vec.size(), ncol);
				for (int i_row_item = 0; i_row_item < static_cast<int>(ncol); ++i_row_item)
				{
					pred_leaf[irow * ncol + i_row_item] = line_vec[i_row_item];
				}
				// Free memory
				result_reader.Lines()[irow].clear();
			}
			LightGBM::DatasetLoader dataset_loader(config_, nullptr,
										 config_.num_class, config_.data.c_str());
			train_data_.reset(dataset_loader.LoadFromFile(config_.data.c_str(), 0, 1));
			train_metric_.clear();
			objective_fun_.reset(LightGBM::ObjectiveFunction::CreateObjectiveFunction(config_.objective,
																			config_));
			objective_fun_->Init(train_data_->metadata(), train_data_->num_data());
			boosting_->Init(&config_, train_data_.get(), objective_fun_.get(),
							LightGBM::Common::ConstPtrInVectorWrapper<LightGBM::Metric>(train_metric_));

			boosting_->RefitTree(pred_leaf.data(), nrow, ncol);
			boosting_->SaveModelToFile(0, -1, config_.saved_feature_importance_type,
									   config_.output_model.c_str());
			LightGBM::Log::Info("Finished RefitTree");
		}
		else
		{
			// create predictor
			LightGBM::Predictor predictor(boosting_.get(), config_.start_iteration_predict, config_.num_iteration_predict, config_.predict_raw_score,
								config_.predict_leaf_index, config_.predict_contrib,
								config_.pred_early_stop, config_.pred_early_stop_freq,
								config_.pred_early_stop_margin);
			predictor.Predict(config_.data.c_str(),
							  config_.output_result.c_str(), config_.header, config_.predict_disable_shape_check,
							  config_.precise_float_parser);
			LightGBM::Log::Info("Finished prediction");
		}
	}

	void Application::InitPredict()
	{
		boosting_.reset(
			LightGBM::Boosting::CreateBoosting("gbdt", config_.input_model.c_str()));
		LightGBM::Log::Info("Finished initializing prediction, total used %d iterations", boosting_->GetCurrentIteration());
	}

	void Application::ConvertModel()
	{
		boosting_.reset(
			LightGBM::Boosting::CreateBoosting(config_.boosting, config_.input_model.c_str()));
		boosting_->SaveModelToIfElse(-1, config_.convert_model.c_str());
	}

	std::function<std::vector<double>(int row_idx)> RowFunctionFromDenseMatric(const void *data, int num_row, int num_col, int data_type, int is_row_major);
	std::function<std::vector<std::pair<int, double>>(int row_idx)> RowPairFunctionFromDenseMatric(const void *data, int num_row, int num_col, int data_type, int is_row_major);
	//-------------------------------------------------------------------------------------------------
	int MemApp::set_params(map<string, string> &init_params)
	{
		bool prev_silent = is_silent;
		is_silent = false;
		only_fatal = false;
		is_silent = init_params.find("verbose") != init_params.end() && stoi(init_params.at("verbose")) <= 0;
		is_silent |= (init_params.find("verbosity") != init_params.end() && stoi(init_params.at("verbosity")) <= 0);
		is_silent |= (init_params.find("silent") != init_params.end() && stoi(init_params.at("silent")) > 0);
		if (global_logger.levels[LOG_MEDALGO] > LOG_DEF_LEVEL || is_silent)
			LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Warning);
		if (init_params.find("silent") != init_params.end() && stoi(init_params.at("silent")) > 1)
		{
			LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Fatal);
			only_fatal = true;
		}

		unordered_map<string, string> params;
		for (auto &e : init_params)
			params[e.first] = e.second;
		LightGBM::ParameterAlias::KeyAliasTransform(&params);
		// load configs
		config_.Set(params);

		if (!prev_silent)
			LightGBM::Log::Info("Finished loading parameters");
		return 0;
	}

	//-------------------------------------------------------------------------------------------------
	int MemApp::InitTrainData(float *xdata, float *ydata, const float *weight, int nrows, int ncols)
	{
		if (global_logger.levels[LOG_MEDALGO] > LOG_DEF_LEVEL || is_silent)
			LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Warning);
		if (only_fatal)
			LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Fatal);
		LightGBM::Log::Info("init train data %d x %d", nrows, ncols);
		if (config_.num_threads > 0)
			omp_set_num_threads(config_.num_threads);

		std::unique_ptr<LightGBM::Dataset> ret;
		auto get_row_fun = RowFunctionFromDenseMatric(xdata, nrows, ncols, C_API_DTYPE_FLOAT32, 1);

		// sample data first
		LightGBM::Random rand(config_.seed);
		int sample_cnt = static_cast<int>(nrows < config_.bin_construct_sample_cnt ? nrows : config_.bin_construct_sample_cnt);
		auto sample_indices = rand.Sample(nrows, sample_cnt);
		sample_cnt = static_cast<int>(sample_indices.size());
		std::vector<std::vector<double>> sample_values(ncols);
		std::vector<std::vector<int>> sample_idx(ncols);
		for (size_t i = 0; i < sample_indices.size(); ++i)
		{
			auto idx = sample_indices[i];
			auto row = get_row_fun(static_cast<int>(idx));
			for (size_t j = 0; j < row.size(); ++j)
			{
				if (std::fabs(row[j]) > LightGBM::kEpsilon)
				{
					sample_values[j].emplace_back(row[j]);
					sample_idx[j].emplace_back(static_cast<int>(i));
				}
			}
		}
		LightGBM::DatasetLoader loader(config_, nullptr, 1, nullptr);
		train_data_.reset(loader.ConstructFromSampleData(LightGBM::Common::Vector2Ptr<double>(&sample_values).data(),
														 LightGBM::Common::Vector2Ptr<int>(&sample_idx).data(),
														 static_cast<int>(sample_values.size()),
														 LightGBM::Common::VectorSize<double>(sample_values).data(),
														 static_cast<size_t>(sample_cnt),
														 static_cast<LightGBM::data_size_t>(nrows),
														 static_cast<int64_t>(nrows) * ncols * sizeof(float)));

		OMP_INIT_EX();
#pragma omp parallel for schedule(static)
		for (int i = 0; i < nrows; ++i)
		{
			OMP_LOOP_EX_BEGIN();
			const int tid = omp_get_thread_num();
			auto one_row = get_row_fun(i);
			train_data_->PushOneRow(tid, i, one_row);
			OMP_LOOP_EX_END();
		}
		OMP_THROW_EX();
		train_data_->FinishLoad();

		// load label
		train_data_->SetFloatField("label", ydata, nrows);

		// load weight
		if (weight != NULL)
			train_data_->SetFloatField("weight", weight, nrows);

		// create training metric
		LightGBM::Log::Info("training eval bit %d", config_.is_provide_training_metric);
		if (config_.is_provide_training_metric)
		{
			LightGBM::Log::Info("Creating training metrics: types %zu [%s]",
								config_.metric.size(), medial::io::get_list(config_.metric, ", ").c_str());
			for (auto metric_type : config_.metric)
			{
				auto metric = std::unique_ptr<LightGBM::Metric>(LightGBM::Metric::CreateMetric(metric_type, config_));
				if (metric == nullptr)
				{
					continue;
				}
				metric->Init(train_data_->metadata(), train_data_->num_data());
				train_metric_.push_back(std::move(metric));
			}
		}
		train_metric_.shrink_to_fit();

		LightGBM::Log::Info("finished loading train mat");
		return 0;
	}

	//-------------------------------------------------------------------------------------------------
	int MemApp::InitTrain(float *xdata, float *ydata, const float *weight, int nrows, int ncols)
	{
		if (global_logger.levels[LOG_MEDALGO] > LOG_DEF_LEVEL || is_silent)
			LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Warning);
		if (only_fatal)
			LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Fatal);
		if (config_.is_parallel)
		{
			LightGBM::Log::Info("parallel mode not supported yet for MedLightGBM !!");
			return -1;
		}

		// create boosting
		boosting_.reset(LightGBM::Boosting::CreateBoosting(config_.boosting, config_.input_model.c_str()));

		// create objective function
		objective_fun_.reset(LightGBM::ObjectiveFunction::CreateObjectiveFunction(config_.objective, config_));

		// load training data
		InitTrainData(xdata, ydata, weight, nrows, ncols);

		// initialize the objective function
		objective_fun_->Init(train_data_->metadata(), train_data_->num_data());

		// initialize the boosting
		boosting_->Init(&config_, train_data_.get(), objective_fun_.get(), LightGBM::Common::ConstPtrInVectorWrapper<LightGBM::Metric>(train_metric_));

		// add validation data into boosting ==> Currently not used, as we do not allow loading validation data at this stage in MedLightGBM
		// for (size_t i = 0; i < valid_datas_.size(); ++i)
		//	boosting_->AddValidDataset(valid_datas_[i].get(), Common::ConstPtrInVectorWrapper<Metric>(valid_metrics_[i]));

		LightGBM::Log::Info("Finished initializing training");
		return 0;
	}

	//-------------------------------------------------------------------------------------------------
	void MemApp::Train()
	{
		if (global_logger.levels[LOG_MEDALGO] > LOG_DEF_LEVEL || is_silent)
			LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Warning);
		if (only_fatal)
			LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Fatal);
		LightGBM::Log::Info("Started training...");
		int total_iter = config_.num_iterations;
		bool is_finished = false;
		bool need_eval = true;
		auto start_time = std::chrono::steady_clock::now();
		LightGBM::Log::Info("total_iter %d is_finished %d need_eval %d", total_iter, (int)is_finished, (int)need_eval);
		for (int iter = 0; iter < total_iter && !is_finished; ++iter)
		{
			is_finished = boosting_->TrainOneIter(nullptr, nullptr);
			auto end_time = std::chrono::steady_clock::now();
			// output used time per iteration
			if ((((iter + 1) % config_.metric_freq) == 0) || (iter == total_iter - 1))
			{

				if (!config_.metric.empty())
				{
					vector<double> m_res = boosting_->GetEvalAt(0);
					stringstream eval_str;

					eval_str << config_.metric[0] << "=";
					if (!m_res.empty())
						eval_str << m_res[0];
					for (size_t i = 1; i < config_.metric.size(); ++i)
					{
						eval_str << ", " << config_.metric[i] << "=";
						if (i < m_res.size())
							eval_str << m_res[i];
					}

					LightGBM::Log::Info("%f seconds elapsed, finished iteration %d. [%s]",
										(std::chrono::duration<double, std::milli>(end_time - start_time) * 1e-3).count(), iter + 1,
										eval_str.str().c_str());
				}
				else
					LightGBM::Log::Info("%f seconds elapsed, finished iteration %d", (std::chrono::duration<double, std::milli>(end_time - start_time) * 1e-3).count(), iter + 1);
			}
		}
		LightGBM::Log::Info("Finished training");
	}

	void MemApp::fetch_boosting(LightGBM::Boosting *&res)
	{
		res = boosting_.get();
	}

	void MemApp::fetch_early_stop(LightGBM::PredictionEarlyStopInstance &early_stop_)
	{
		LightGBM::Boosting *_boosting = boosting_.get();

		early_stop_ = CreatePredictionEarlyStopInstance("none", LightGBM::PredictionEarlyStopConfig());
		if (config_.pred_early_stop && !_boosting->NeedAccuratePrediction())
		{
			LightGBM::PredictionEarlyStopConfig pred_early_stop_config;
			pred_early_stop_config.margin_threshold = config_.pred_early_stop_margin;
			pred_early_stop_config.round_period = config_.pred_early_stop_freq;
			if (_boosting->NumberOfClasses() == 1)
			{
				early_stop_ = CreatePredictionEarlyStopInstance("binary", pred_early_stop_config);
			}
			else
			{
				early_stop_ = CreatePredictionEarlyStopInstance("multiclass", pred_early_stop_config);
			}
		}
	}

	int MemApp::serialize_to_string(string &str) const
	{
		str = boosting_->SaveModelToString(0, -1, 0);
		return 0;
	}

	int MemApp::deserialize_from_string(string &str)
	{
		std::unique_ptr<LightGBM::Boosting> ret;
		string type = config_.boosting; // use boosting_type for older lightgbm version
		if (type == std::string("gbdt"))
			ret.reset(new LightGBM::GBDT());
		else if (type == std::string("dart"))
			ret.reset(new LightGBM::DART());
		else if (type == std::string("goss"))
			ret.reset(new LightGBM::GBDT());
		else
		{
			fprintf(stderr, "deserialize MedLightGBM ERROR: unknown boosting type %s\n", type.c_str());
			return -1;
		}
		if (!ret.get()->LoadModelFromString(str.c_str(), str.length()))
			return -1;
		boosting_.reset(ret.release());
		return 0;
	}
	//-------------------------------------------------------------------------------------------------
	void MemApp::Predict(float *x, int nrows, int ncols, float *&preds) const
	{
		auto get_row_fun = RowPairFunctionFromDenseMatric(x, nrows, ncols, C_API_DTYPE_FLOAT32, 1);

		// create boosting
		LightGBM::Predictor predictor(boosting_.get(), 0, config_.num_iteration_predict, config_.predict_raw_score, config_.predict_leaf_index, config_.predict_contrib,
									  config_.pred_early_stop, config_.pred_early_stop_freq, config_.pred_early_stop_margin);
		int64_t num_pred_in_one_row = boosting_->NumPredictOneRow(0, config_.num_iteration_predict, config_.predict_leaf_index, config_.predict_contrib);
		auto pred_fun = predictor.GetPredictFunction();

		// string str;
		// serialize_to_string(str);

		int64_t len_res = nrows * num_pred_in_one_row;
		// MLOG("[MedLightGBM] predict: nrows %d , num_pred %d , len_res %d\n", nrows, num_pred_in_one_row, len_res);
		// MLOG("[MedLightGBM] predict: num_iter %d , is_raw %d , is_leaf %d\n",
		//	config_.io_config.num_iteration_predict, config_.io_config.is_predict_raw_score ? 1 : 0, config_.io_config.is_predict_leaf_index ? 1:0);
		vector<double> out_result_vec(len_res);
		double *out_result = &out_result_vec[0];
		if (preds == NULL)
			preds = new float[len_res];

		OMP_INIT_EX();
#pragma omp parallel for schedule(static)
		for (int i = 0; i < nrows; ++i)
		{
			OMP_LOOP_EX_BEGIN();
			auto one_row = get_row_fun(i);
			auto pred_wrt_ptr = out_result + static_cast<size_t>(num_pred_in_one_row) * i;
			pred_fun(one_row, pred_wrt_ptr);
			OMP_LOOP_EX_END();
		}
		OMP_THROW_EX();

		for (int64_t i = 0; i < len_res; i++)
			preds[i] = (float)out_result[i];
	}

	//-------------------------------------------------------------------------------------------------
	void MemApp::PredictShap(float *x, int nrows, int ncols, float *&shap_vals) const
	{
		auto get_row_fun = RowPairFunctionFromDenseMatric(x, nrows, ncols, C_API_DTYPE_FLOAT32, 1);

		// create boosting
		LightGBM::Predictor predictor(boosting_.get(), 0, config_.num_iteration_predict, config_.predict_raw_score, config_.predict_leaf_index, true,
									  config_.pred_early_stop, config_.pred_early_stop_freq, config_.pred_early_stop_margin);
		int64_t num_pred_in_one_row = boosting_->NumPredictOneRow(0, config_.num_iteration_predict, config_.predict_leaf_index, true);
		auto pred_fun = predictor.GetPredictFunction();

		int64_t len_res = nrows * num_pred_in_one_row;

		vector<double> out_result_vec(len_res);
		double *out_result = &out_result_vec[0];
		if (shap_vals == NULL)
			MTHROW_AND_ERR("Error MemApp::PredictShap - shap_vals was NULL\n");

		OMP_INIT_EX();
#pragma omp parallel for schedule(static)
		for (int i = 0; i < nrows; ++i)
		{
			OMP_LOOP_EX_BEGIN();
			auto one_row = get_row_fun(i);
			auto pred_wrt_ptr = out_result + static_cast<size_t>(num_pred_in_one_row) * i;
			pred_fun(one_row, pred_wrt_ptr);
			for (int j = 0; j < num_pred_in_one_row; j++)
				shap_vals[i * num_pred_in_one_row + j] = (float)pred_wrt_ptr[j];
			OMP_LOOP_EX_END();
		}
		OMP_THROW_EX();
	}

	//-----------------------------------------------------------------------------------------------------------
	//----- start of some help functions
	//-----------------------------------------------------------------------------------------------------------
	std::function<std::vector<std::pair<int, double>>(int row_idx)>
	RowPairFunctionFromDenseMatric(const void *data, int num_row, int num_col, int data_type, int is_row_major)
	{
		auto inner_function = RowFunctionFromDenseMatric(data, num_row, num_col, data_type, is_row_major);
		if (inner_function != nullptr)
		{
			return [inner_function](int row_idx)
			{
				auto raw_values = inner_function(row_idx);
				std::vector<std::pair<int, double>> ret;
				for (int i = 0; i < static_cast<int>(raw_values.size()); ++i)
				{
					if (std::fabs(raw_values[i]) > 1e-15)
					{
						ret.emplace_back(i, raw_values[i]);
					}
				}
				return ret;
			};
		}
		return nullptr;
	}

	std::function<std::vector<double>(int row_idx)>
	RowFunctionFromDenseMatric(const void *data, int num_row, int num_col, int data_type, int is_row_major)
	{
		if (data_type == C_API_DTYPE_FLOAT32)
		{
			const float *data_ptr = reinterpret_cast<const float *>(data);
			if (is_row_major)
			{
				return [data_ptr, num_col, num_row](int row_idx)
				{
					std::vector<double> ret(num_col);
					auto tmp_ptr = data_ptr + static_cast<size_t>(num_col) * row_idx;
					for (int i = 0; i < num_col; ++i)
					{
						ret[i] = static_cast<double>(*(tmp_ptr + i));
						if (std::isnan(ret[i]))
						{
							ret[i] = 0.0f;
						}
					}
					return ret;
				};
			}
			else
			{
				return [data_ptr, num_col, num_row](int row_idx)
				{
					std::vector<double> ret(num_col);
					for (int i = 0; i < num_col; ++i)
					{
						ret[i] = static_cast<double>(*(data_ptr + static_cast<size_t>(num_row) * i + row_idx));
						if (std::isnan(ret[i]))
						{
							ret[i] = 0.0f;
						}
					}
					return ret;
				};
			}
		}
		else if (data_type == C_API_DTYPE_FLOAT64)
		{
			const double *data_ptr = reinterpret_cast<const double *>(data);
			if (is_row_major)
			{
				return [data_ptr, num_col, num_row](int row_idx)
				{
					std::vector<double> ret(num_col);
					auto tmp_ptr = data_ptr + static_cast<size_t>(num_col) * row_idx;
					for (int i = 0; i < num_col; ++i)
					{
						ret[i] = static_cast<double>(*(tmp_ptr + i));
						if (std::isnan(ret[i]))
						{
							ret[i] = 0.0f;
						}
					}
					return ret;
				};
			}
			else
			{
				return [data_ptr, num_col, num_row](int row_idx)
				{
					std::vector<double> ret(num_col);
					for (int i = 0; i < num_col; ++i)
					{
						ret[i] = static_cast<double>(*(data_ptr + static_cast<size_t>(num_row) * i + row_idx));
						if (std::isnan(ret[i]))
						{
							ret[i] = 0.0f;
						}
					}
					return ret;
				};
			}
		}
		throw std::runtime_error("unknown data type in RowFunctionFromDenseMatric");
	}

	void MemApp::calc_feature_importance(vector<float> &features_importance_scores,
										 const string &general_params, int max_feature_idx_)
	{

		map<string, string> params;
		MedSerialize::init_map_from_string(general_params, params);
		string importance_type = "gain"; //"frequency"; //"gain";
		if (params.find("importance_type") != params.end())
			importance_type = params.at("importance_type");

		MTHROW_AND_ERR("Not supported");
	}

}

//===============================================================================================
// MedLightGBM
//===============================================================================================

void MedLightGBM::calc_feature_importance(vector<float> &features_importance_scores,
										  const string &general_params, const MedFeatures *features)
{
	if (!_mark_learn_done)
		MTHROW_AND_ERR("ERROR:: Requested calc_feature_importance before running learn\n");

	map<string, string> params;

	unordered_set<string> local_types = {"gain", "split"};
	unordered_set<string> legal_types = {"gain", "split", "shap"};

	MedSerialize::initialization_text_to_map(general_params, params);
	string importance_type = "gain"; // default
	for (auto it = params.begin(); it != params.end(); ++it)
		if (it->first == "importance_type")
			importance_type = it->second;
		else
			MTHROW_AND_ERR("Unsupported calc_feature_importance param \"%s\"\n", it->first.c_str());

	if (legal_types.find(importance_type) == legal_types.end())
		MTHROW_AND_ERR("Ilegal importance_type value \"%s\" "
					   "- should by one of [%s]\n",
					   importance_type.c_str(), medial::io::get_list(legal_types, ", ").c_str());

	features_importance_scores.resize(model_features.empty() ? features_count : (int)model_features.size());

	if (local_types.count(importance_type) > 0)
	{
		mem_app.calc_feature_importance(features_importance_scores, "importance_type=" + importance_type,
										(model_features.empty() ? features_count : (int)model_features.size()));
		return;
	}

	// shap option
	MedMat<float> feat_mat, contribs_mat;
	if (features == NULL)
		MTHROW_AND_ERR("SHAP values feature importance requires features \n");

	features->get_as_matrix(feat_mat);
	calc_feature_contribs(feat_mat, contribs_mat);
#pragma omp parallel for
	for (int j = 0; j < contribs_mat.ncols; ++j)
	{
		float col_sum = 0;

		for (int i = 0; i < contribs_mat.nrows; ++i)
		{
			col_sum += abs(contribs_mat.get(i, j));
		}
		features_importance_scores[j] = col_sum / (float)contribs_mat.nrows;
	}
}

void MedLightGBM::calc_feature_contribs(MedMat<float> &x, MedMat<float> &contribs)
{
	int nrows = x.nrows;
	int ncols = x.ncols;

	contribs.resize(nrows, ncols + 1);
	// copy metadata
	contribs.signals.insert(contribs.signals.end(), x.signals.begin(), x.signals.end());
	contribs.signals.push_back("b0");
	contribs.recordsMetadata.insert(contribs.recordsMetadata.end(), x.recordsMetadata.begin(), x.recordsMetadata.end());

	float *contribs_ptr = contribs.data_ptr();
	float *x_ptr = x.data_ptr();
	mem_app.PredictShap(x_ptr, nrows, ncols, contribs_ptr);
}

void MedLightGBM::prepare_predict_single()
{
	if (!prepared_single)
	{
		num_preds = n_preds_per_sample();

		mem_app.fetch_boosting(_boosting);
		mem_app.fetch_early_stop(early_stop_);
		prepared_single = true;
	}
}

void MedLightGBM::predict_single(const vector<float> &x, vector<float> &preds) const
{
	int n_ftrs = (int)x.size();
	vector<double> one_row(n_ftrs);
	for (int i = 0; i < n_ftrs; ++i)
		one_row[i] = static_cast<double>(x[i]);

	vector<double> out_result_vec(num_preds);
	predict_single(one_row, out_result_vec);

	preds.resize(num_preds);
	for (int64_t i = 0; i < num_preds; i++)
		preds[i] = (float)out_result_vec[i];
}

void MedLightGBM::predict_single(const vector<double> &x, vector<double> &preds) const
{
	preds.resize(num_preds);
	double *out_result = &preds[0];
	_boosting->Predict(x.data(), out_result, &early_stop_);
}

void MedLightGBM::export_predictor(const string &output_fname)
{
	string predictor_str;
	mem_app.serialize_to_string(predictor_str);
	ofstream ofs(output_fname, std::ios::binary);
	if (!ofs)
	{
		MTHROW_AND_ERR("MedLightGBM::export_predictor failed. couldn't write %s \n", output_fname.c_str());
	}
	ofs << predictor_str;
	ofs.close();
}

void MedLightGBM::print(FILE *fp, const string &prefix, int level) const
{

	if (level == 0)
		fprintf(fp, "%s: MedLightGBM ()\n", prefix.c_str());
	else
	{
		string predictor_str;
		mem_app.serialize_to_string(predictor_str);
		fprintf(fp, "%s: MedLightGBM ()\n%s\n", prefix.c_str(), predictor_str.c_str());
	}
}