import numpy as np
import pickle
import itertools

import pandas as pd
from exp_mixture_model import EMM
from simulate_dns_arrival_sample import simulate_probe_samples
import datetime
import multiprocessing

extended_window_size = 50
max_sample_num_for_mix_exp = 300


def get_exp_estimation_features(probe_sample, window_range=0):
    # print(window_range)
    component_num = 1
    interval_list = [x[0] for x in probe_sample]
    for i in range(window_range):
        interval_list.append([])
        interval_list.insert(0, [])
    features = []
    for i in range(window_range, len(interval_list)-window_range):
        intervals = list(itertools.chain(*interval_list[i-window_range: i+window_range+1]))
        if len(intervals) == 0:
            features.append([0, len(intervals)])
        else:
            if component_num == 1:
                rate = 1/np.mean(np.array(intervals))
            else:
                model = EMM(k=component_num)
                pi, mu = model.fit(np.array(intervals))
                rate = 1 / np.sum(pi * mu)
            features.append([rate, len(intervals)])
    features = np.array(features)
    return features


def get_mix_exp_rate(probe_sample,TTL):
    estimated_rate = 2

    interval_list = [x[0] for x in probe_sample]
    intervals = np.array(list(itertools.chain(*interval_list))).tolist()
    repeat_num = 5
    for _ in range(repeat_num):
        intervals = [get_float_interval(int(x), estimated_rate) for x in intervals]
        # intervals = np.floor(intervals)+np.random.random(len(intervals))
        # model = EMM(k=component_num)
        model = EMM()
        pi, mu = model.fit(np.array(intervals))
        rate = np.sum(1/mu*pi*(mu+TTL)) / np.sum(pi * (mu+TTL))
        rate_1 = 1 / np.mean(intervals)
        estimated_rate = rate
    print(estimated_rate)
    return rate


def get_features(probe_sample):
    exp_feature_windows = list(range(extended_window_size))
    features = []
    for window_size in exp_feature_windows:
        features_1 = get_exp_estimation_features(probe_sample, window_size)
        features.append(features_1)
    features_3 = np.array(list(range(len(probe_sample))))
    features_3 = (features_3 / np.max(features_3)).reshape((-1, 1))
    features.append(features_3)
    features = np.hstack(features)
    return features


def get_features_in_estimation(probe_sample, estimated_rates):
    for i in range(len(probe_sample)):
        intervals = probe_sample[i][0]
        intervals = [int(x) for x in intervals]
        estimated_rate = estimated_rates[i]
        intervals = [get_float_interval(x, estimated_rate) for x in intervals]
        probe_sample[i][0] = intervals

    exp_feature_windows = list(range(extended_window_size))
    features = []
    for window_size in exp_feature_windows:
        features_1 = get_exp_estimation_features(probe_sample, window_size)
        features.append(features_1)
    features_3 = np.array(list(range(len(probe_sample))))
    features_3 = (features_3 / np.max(features_3)).reshape((-1, 1))
    features.append(features_3)
    features = np.hstack(features)
    return features


def get_float_interval(interval, rate):
    a = interval
    rate = np.maximum(rate, 0.0001)
    x = a * np.exp(-rate * a) - (a + 1) * np.exp(-rate * (a + 1)) + (np.exp(-a * rate) - np.exp(-rate * (a + 1))) / rate
    y = np.exp(-rate * a) - np.exp(-rate * (a + 1))
    if x != 0 and y != 0:
        float_interval = x / y
    else:
        float_interval = np.floor(interval) + 0.5
    return float_interval


def get_random_parameters(TTL, rate_upper_bound):
    rate = np.random.random(1)[0]*rate_upper_bound
    return rate, TTL


def run(task_infos):
    task_list = []
    for task_info in task_infos:
        TTL = task_info[0]
        rate_upper_bound = task_info[1]
        min_rate = task_info[2]
        arrival_num = int(task_info[3])
        rate_num = int(task_info[4])
        m_num = int(task_info[5])
        sample_path = task_info[6]
        for i in range(m_num):
            rate, TTL = get_random_parameters(TTL, rate_upper_bound)
            task_list.append([rate, TTL, arrival_num, rate_num, min_rate, sample_path, rate_upper_bound])

    pool = multiprocessing.Pool(processes=my_proc_num)
    for _ in pool.imap_unordered(generate_trace, task_list):
        pass


def generate_trace(task_info):
    rate, TTL, arrival_num, rate_num, min_rate, sample_path, rate_upper_bound = task_info
    probe_samples = simulate_probe_samples(max_rate=rate, TTL=TTL, arrival_num=arrival_num, rate_num=rate_num, min_rate=min_rate)
    for probe_sample, max_rate, TTL, rate_tag in probe_samples:
        try:
            features = get_features(probe_sample)
            rates = [x[1] for x in probe_sample]
            time_stamp = str(datetime.datetime.now().timestamp())
            sample_file_name = sample_path + 'sample_' + str(rate_upper_bound) + '_' + str(TTL) + '_' + str(
                rate_tag) + '_' + time_stamp + '.pkl'
            with open(sample_file_name, 'wb') as file:
                pickle.dump([features, rates, probe_sample, max_rate, TTL], file)
                print('output ' + sample_file_name)
        except Exception as exc:
            print(exc)


def generate_trace_direct(task_info):
    probe_sample, max_rate, TTL, rate_tag = task_info
    features = get_features(probe_sample)
    rates = [x[1] for x in probe_sample]
    return features, rates, probe_sample, max_rate, TTL, rate_tag


if __name__ == '__main__':
    my_proc_num = 8
    # # my_proc_num = 3
    # # task_info_file_name = r'simulated_task_info.csv'
    # # task_info_file_name = r'simulated_task_info_3.csv'
    # task_info_file_name = r'simulated_task_info_51.csv'
    # my_task_infos = np.array(pd.read_csv(task_info_file_name, header=None)).tolist()
    # run(my_task_infos)
    # print('Done!')