import os.path
import random
import sys
from os import walk

import numpy as np
import pickle
import itertools

import pandas as pd
from exp_mixture_model import EMM
from exp_mixture_model import generate_emm
from simulate_dns_arrival_sample_multiple import simulate_probe_from_file
import datetime
import multiprocessing

max_sample_num_for_mix_exp = 300

extended_window_size = 50

input_path = r'../data/campus_1500/'
output_path = r'../data/real_trace_multiple/'

if os.path.exists(output_path) is False:
    os.mkdir(output_path)

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


def get_mix_exp_estimation_features(probe_sample, big_window_range=6*24):
    # print(big_window_range)
    interval_list = [x[0] for x in probe_sample]
    window_num = int(6*24/big_window_range)
    features = []
    for i in range(window_num):
        intervals = list(itertools.chain(*interval_list[i*big_window_range: (i + 1) * big_window_range]))
        random.shuffle(intervals)
        intervals = intervals[:max_sample_num_for_mix_exp]
        try:
            model = EMM()
            pi, mu = model.fit(np.array(intervals))
            rate = 1 / np.sum(pi * mu)
        except:
            print('mix exp error')
            rate = 1 / np.mean(np.array(intervals))

        for j in range(i*big_window_range,(i + 1) * big_window_range):
            features.append([rate, len(intervals)])

    features = np.array(features)
    return features


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
    x = a * np.exp(-rate * a) - (a + 1) * np.exp(-rate * (a + 1)) + (np.exp(-a * rate) - np.exp(-rate * (a + 1))) / rate
    y = np.exp(-rate * a) - np.exp(-rate * (a + 1))
    if x != 0 and y != 0:
        float_interval = x / y
    else:
        float_interval = np.floor(interval) + 0.5
    return float_interval


def generate_trace(task_info):
    trace_file_name, TTL, cache_num = task_info
    print(trace_file_name)

    result_list = []

    # if trace_file_name != '/Users/jfli/ftp/dataset1/jd2008.jd.com_202.117.0.20.txt':
    #     return result_list

    probe_sample_list = simulate_probe_from_file(trace_file_name, cache_num=cache_num, TTL=TTL, second_precision=False)
    if probe_sample_list is None:
        return result_list
    for i in range(len(probe_sample_list)):
        cache_probe_samples = probe_sample_list[i]
        cache_result_list = []
        for probe_sample, max_rate, TTL, trace_tag in cache_probe_samples:
            features = get_features(probe_sample)
            rates = [x[1] for x in probe_sample]
            cache_result_list.append([features, rates, probe_sample, max_rate, TTL])
        result_list.append([cache_result_list, max_rate, TTL, trace_tag])
    return result_list


def list_trace_file(sample_path, dns_table, domain_table, existing_domain_dns):
    file_list = []
    for dir_path, folders, files in walk(sample_path):
        for file_name in files:
            if file_name.__contains__('.txt') is False:
                continue
            domain_name = file_name.split('_')[0]
            dns_server = file_name.replace('.txt','').split('_')[-1]
            if dns_table.__contains__(dns_server) is False:
                continue
            if domain_table.__contains__(domain_name) is False:
                continue
            if existing_domain_dns.__contains__((domain_name, dns_server)):
                continue
            file_list.append([sample_path + file_name, domain_table[domain_name], my_cache_num])
    return file_list


def get_existing_domain_dns(sample_path):
    existing_domain_dns = {}
    for dir_path, folders, files in walk(sample_path):
        for file_name in files:
            if file_name.__contains__('.pkl') is False:
                continue
            strs = file_name.split('_')
            domain = strs[-3]
            dns = strs[-2]
            existing_domain_dns[(domain, dns)] = 1
    return existing_domain_dns


def run():
    dns_list = ['202.117.0.20']
    domain_list = np.array(pd.read_csv(r'my_domain_list_ttl.txt', header=None)).tolist()
    dns_table = {x: 1 for x in dns_list}
    domain_table = {x[0]: x[1] for x in domain_list}
    existing_domain_dns = get_existing_domain_dns(output_path)
    task_list = list_trace_file(input_path, dns_table, domain_table, existing_domain_dns)
    pool = multiprocessing.Pool(processes=my_proc_num)
    cache_sample_output_path = output_path + str(my_cache_num)+'/'
    if os.path.exists(cache_sample_output_path) is False:
        os.mkdir(cache_sample_output_path)
    for result_list in pool.imap_unordered(generate_trace, task_list):
        for i in range(len(result_list)):
            cache_result_list, max_rate, TTL, trace_tag = result_list[i]
            sample_file_name = cache_sample_output_path + 'sample_' + str(max_rate) + '_' + str(TTL) + '_' + str(trace_tag) +'_'+str(i)+ '.pkl'
            with open(sample_file_name, 'wb') as file:
                pickle.dump(cache_result_list, file)
                print('output ' + sample_file_name)


if __name__ == '__main__':
    my_proc_num = 5
    my_cache_num = 2
    if len(sys.argv) > 1:
        my_cache_num = int(sys.argv[1])
    run()
    print('Done!')