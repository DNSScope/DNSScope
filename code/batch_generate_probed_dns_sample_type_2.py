import os
from os import walk

import numpy as np
import pickle
import itertools

import pandas as pd
from exp_mixture_model import EMM
from simulate_dns_arrival_sample import simulate_probe_samples
import datetime
import multiprocessing
from os.path import exists


my_output_path = r'../data/probe_sample_type_2/'

if exists(my_output_path) is False:
    os.mkdir(my_output_path)

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
    # print(estimated_rate)
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
    pool = multiprocessing.Pool(processes=my_proc_num)
    for _ in pool.imap_unordered(generate_trace, task_infos):
        pass

def get_probe_sample(cri_file_names, rate_file_name):
    period = 86400
    window_size = 600
    if exists(rate_file_name) is False:
        return None, None, None
    for cri_file_name in cri_file_names:
        if exists(cri_file_name) is False:
            return None, None, None
    xs = np.array(pd.read_csv(rate_file_name, header=None))
    query_times = xs[:, -1].tolist()
    start_time = datetime.datetime.fromtimestamp(query_times[0])
    query_times = [(datetime.datetime.fromtimestamp(x)-start_time).total_seconds() for x in query_times]
    query_times = [x for x in query_times if x <= period]
    rate_list = [0]*int(np.ceil(period/window_size))
    for query_time in query_times:
        window_index = int(query_time/window_size)
        rate_list[window_index] += 1/window_size

    probe_samples = []
    for cri_file_name in cri_file_names:
        probe_sample = [[[], rate_list[i]/len(cri_file_names)] for i in range(int(np.ceil(period / window_size)))]

        xs = np.array(pd.read_csv(cri_file_name, header=None))
        times = xs[:, 2].tolist()
        cris = xs[:, 3].tolist()
        for i in range(len(times)):
            timestamp = times[i]
            cri_time = (datetime.datetime.fromtimestamp(timestamp) - start_time).total_seconds()
            window_index = int(cri_time / window_size)
            if window_index >= len(probe_sample):
                break
            probe_sample[window_index][0].append(cris[i])
        probe_samples.append(probe_sample)

    max_rate = np.max(rate_list)
    rate_tag = np.random.random(1)[0]
    return probe_samples, max_rate, rate_tag


def generate_trace(task_info):
    domain_name, ttl, cri_file_names, rate_file_name, dns_server = task_info
    trace_tag = domain_name + '_' + dns_server
    probe_samples, max_rate, rate_tag = get_probe_sample(cri_file_names, rate_file_name)
    if probe_samples is None:
        return
    samples = []
    for probe_sample in probe_samples:
        try:
            features = get_features(probe_sample)
            rates = [x[1] for x in probe_sample]
            # time_stamp = str(datetime.datetime.now().timestamp())
            samples.append([features, rates, probe_sample, max_rate, ttl])
        except Exception as exc:
            print(exc)
    sample_file_name = my_output_path + 'sample_' + str(max_rate) + '_' + str(ttl) + '_' + str(
        trace_tag) + '_' + str(0) + '.pkl'

    with open(sample_file_name, 'wb') as file:
        pickle.dump(samples, file)
        print('output ' + sample_file_name)



def get_task_info(sample_path):
    ttl_table = {}
    for dir_path, folders, files in walk(sample_path):
        for folder in folders:
            ttl = -1
            try:
                ttl = int(folder)
                ttl_table[ttl] = [sample_path + str(ttl)+r'/', []]
            except Exception as exc:
                print(exc)
        break
    task_infos = []
    for ttl in ttl_table.keys():
        # if ttl != 10:
        #     continue
        #     print('nihao')
        for dir_path, folders, files in walk(ttl_table[ttl][0]):
            for folder in folders:
                if folder.__contains__('.txt') is False:
                    continue
                trace_tag = folder[:-4]
                strs = trace_tag.split('_')
                domain_name = strs[0]
                dns_server = strs[1]
                domain_name_path = ttl_table[ttl][0]+r'/'+folder+r'/'
                # cri_file_name = domain_name_path + r'PROBE_CRIs_realendtime.txt'
                cri_file_names = []
                cri_file_name = domain_name_path + r'PROBE_CRIs_realendtime_HOST0.txt'
                cri_file_names.append(cri_file_name)
                cri_file_name = domain_name_path + r'PROBE_CRIs_realendtime_HOST1.txt'
                cri_file_names.append(cri_file_name)
                # cri_file_name = domain_name_path + r'PROBE_CRIs.txt'
                # cri_file_name = domain_name_path + r'QUERY_CRIs.txt'
                rate_file_name = domain_name_path + r'USEFUL_QUERY.txt'
                ttl_table[ttl][1].append([domain_name, cri_file_names, rate_file_name, dns_server])
            break
    for ttl in ttl_table.keys():
        domain_list = ttl_table[ttl][1]
        for domain_name, cri_file_name, rate_file_name, dns_server in domain_list:
            task_infos.append([domain_name, ttl, cri_file_name, rate_file_name, dns_server])
    return task_infos



if __name__ == '__main__':
    # my_proc_num = 8
    my_proc_num = 3

    my_sample_path = r'../data/hybrid_r_dns_80/CRIs_type2/'

    my_task_infos = get_task_info(my_sample_path)
    run(my_task_infos)
    print('Done!')