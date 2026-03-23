#include <iostream>
#include <string>
#include <vector>
#include <map>

using namespace std;

int auto_time_convert(long long ts, int to_type)
{
	long long date_t = 0;
	long long hhmm = 0;

	// MLOG("auto time convert: Date is %d , ts %lld , to_type %d\n", MedTime::Date, ts, to_type);

	if ((ts / (long long)1000000000) == 0)
	{
		date_t = ts; // yyyymmdd
		hhmm = 0;
	}
	else if (((ts / (long long)100000000000) == 0))
	{
		date_t = ts / 100; // yyyymmddhh
		hhmm = 60 * (ts % 100);
	}
	else if (((ts / (long long)10000000000000) == 0))
	{
		date_t = ts / 10000; // yyyymmddhhmm
		hhmm = 60 * ((ts % 10000) / 100) + (ts % 100);
	}
	else
	{
		date_t = ts / 1000000; // yyyymmddhhmmss
		hhmm = 60 * ((ts % 1000000) / 10000) + ((ts % 10000) / 100);
	}

	if (to_type == 1)
	{
		// MLOG("auto time convert: date_t %d\n", date_t);
		// Ensure valid date:
		int year = int(date_t / 10000);
		if (year < 1900 || year > 3000)
		{
			// MTHROW_AND_ERR("Error invalid date %lld\n", ts);
			return -1;
		}
		return (int)date_t;
	}

	return 0;
}

vector<pair<int, string>> AddData_data(int patient_id, const char *signalName, int TimeStamps_len, long long *TimeStamps, int Values_len, float *Values,
									   int n_time_channels, int n_val_channels)
{
	// At the moment MedialInfraAlgoMarker only loads timestamps given as ints.
	// This may change in the future as needed.
	vector<pair<int, string>> ret;
	int *i_times = NULL;
	vector<int> times_int;
	vector<float> values_filter;
	vector<int> skip_e_index;
	int time_ch_count, val_ch_count;
	string sig = string(signalName);

	int tu = 1;

	if (TimeStamps_len > 0)
	{
		times_int.resize(TimeStamps_len);
		int write_index = 0;

		// currently assuming we only work with dates ... will have to change this when we'll move to other units
		int i = 0;
		while (i < TimeStamps_len)
		{
			times_int[write_index] = auto_time_convert(TimeStamps[i], tu);
			if (times_int[write_index] < 0)
			{
				char buff[5000];
				snprintf(buff, sizeof(buff), "Error in AddData :: patient %d, signals %s, timestamp %lld is ilegal",
						 patient_id, signalName, TimeStamps[i]);
				cout << buff << endl;

				ret.push_back(pair<int, string>(1, string(buff)));
				if (skip_e_index.empty())
				{
					time_ch_count = n_time_channels;
					val_ch_count = n_val_channels;
					values_filter.reserve(Values_len);
				}

				// Need to skip this element index and create a complete new signal:
				int current_element_index = int(i / time_ch_count);
				skip_e_index.push_back(current_element_index);
				int time_pos = i % time_ch_count;
				write_index = current_element_index * time_ch_count;

				// skip current element
				i += time_ch_count - time_pos;
				continue;
			}
			++write_index;
			++i;
		}
		i_times = &times_int[0];
	}
	if (!skip_e_index.empty())
	{
		// Rewrite values_filter from Values based on skipping skip_e_index elements. Nujm of channels is: val_ch_count
		int i = 0, j = 0;
		while (i < Values_len)
		{
			int current_element_index = int(i / val_ch_count);
			if (j < (int)skip_e_index.size() && current_element_index == skip_e_index[j])
			{
				++j;
				i += val_ch_count;
				continue;
			}
			else
			{
				values_filter.push_back(Values[i]);
			}
			++i;
		}
		Values = &values_filter[0];

		TimeStamps_len -= (int)skip_e_index.size() * time_ch_count;
		Values_len = (int)values_filter.size();
	}

	// print call to  i_times, TimeStamps_len, Values, Values_len
	for (size_t i = 0; i < TimeStamps_len; ++i)
	{
		cout << "Time " << i << " " << i_times[i] << endl;
	}
	for (size_t i = 0; i < Values_len; ++i)
	{
		cout << "Value " << i << " " << Values[i] << endl;
	}
	cout << "#################" << endl;
	return ret;
}

int main()
{
	// Proof of testing new logic.
	vector<long long> time = {20130101};
	vector<float> vals = {13};
	AddData_data(1, "Hemoglobin", time.size(), &time[0], vals.size(), &vals[0], 1, 1);

	time = {123};
	AddData_data(1, "Hemoglobin", time.size(), &time[0], vals.size(), &vals[0], 1, 1);

	time = {20130101, 20140101, 20150101, 20160101};
	vals = {13, 14};
	AddData_data(1, "Hemoglobin", time.size(), &time[0], vals.size(), &vals[0], 2, 1);

	time = {20130101, 123, 20150101, 20160101};
	AddData_data(1, "Hemoglobin", time.size(), &time[0], vals.size(), &vals[0], 2, 1);

	time = {123, 20140101, 20150101, 20160101};
	AddData_data(1, "Hemoglobin", time.size(), &time[0], vals.size(), &vals[0], 2, 1);

	time = {20130101, 20140101};
	vals = {13, 14};
	AddData_data(1, "Hemoglobin", time.size(), &time[0], vals.size(), &vals[0], 2, 2);

	time = {20130101, 123};
	vals = {13, 14};
	AddData_data(1, "Hemoglobin", time.size(), &time[0], vals.size(), &vals[0], 2, 2);

	time = {20130101, 123, 20150101, 20160101};
	vals = {13, 14, 15,16,17,18,19,20};
	AddData_data(1, "Hemoglobin", time.size(), &time[0], vals.size(), &vals[0], 1, 2);

	return 0;
}