import os.path
import random

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pandas as pd

def simulate_GBM_rate(mu=0.5, sigma=1, max_rate=1, sample_num=1):
    S0 = 1
    u = mu
    T = 1  # 年化时间长度
    I = sample_num
    M = int(86400 / 60)  # 年化时间分段数目
    dt = T / M  # 模拟的每小步步长
    S = np.zeros((M + 1, I))
    S[0] = S0
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((u - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * npr.standard_normal(I))
    rate_samples = []
    for i in range(sample_num):
        S[:, i] = S[:, i]/np.max(S[:, i])*max_rate
        rate_samples.append(S[:, i].tolist())
    return rate_samples


def simulate_constant_rate(mu=0.5, sigma=1, max_rate=1, sample_num=1):
    S0 = 1
    u = mu
    T = 1  # 年化时间长度
    I = sample_num
    M = int(86400 / 60)  # 年化时间分段数目
    dt = T / M  # 模拟的每小步步长
    S = np.zeros((M + 1, I))
    S[0] = S0
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((u - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * npr.standard_normal(I))
    rate_samples = []
    for i in range(sample_num):
        S[:, i] = S[:, i]/np.max(S[:, i])*max_rate
        rate_samples.append([max_rate]*len(S[:, i]))
    return rate_samples


def simulate_probe(arrival_sample, TTL, second_precision=False):
    cache_end_time = -1
    cache_start_time = -1
    samples = []
    for i in range(len(arrival_sample)):
        if arrival_sample[i] > cache_end_time:
            cache_start_time = arrival_sample[i]
            if second_precision is True:
                cache_start_time = int(cache_start_time)
            interval = cache_start_time - cache_end_time

            # if interval == 0:
            #     print('nihao')

            t = (cache_start_time + cache_end_time)/2
            samples.append([t, interval])
            cache_end_time = cache_start_time + TTL
    if len(samples) > 0:
        del samples[0]
    return samples


def simulate_probe_samples(max_rate=1, TTL=300, second_precision=False, arrival_num=1, rate_num=1, min_rate=-1):
    period = 86400
    rate_step = 60
    window_size = 600

    rate_samples = simulate_GBM_rate(max_rate=max_rate, sample_num=rate_num)
    # rate_samples = simulate_constant_rate(max_rate=max_rate, sample_num=rate_num)
    if min_rate >= 0:
        rate_samples = np.array(rate_samples)
        # a = np.min(rate_samples, axis=1)
        rate_samples = rate_samples - np.min(rate_samples, axis=1).reshape((-1, 1)) + min_rate
        rate_samples = rate_samples.tolist()
    arrival_samples = []
    rate_tags = np.random.random(len(rate_samples)).tolist()
    for i in range(len(rate_samples)):
        rate_sample = rate_samples[i]
        for j in range(arrival_num):
            t = 0
            arrival_sample = []
            while t <= period:
                t = t + np.random.exponential(1/rate_sample[int(t/rate_step)], 1)[0]
                arrival_sample.append(t)
            if len(arrival_sample) > 0:
                del arrival_sample[-1]
            arrival_samples.append([arrival_sample, rate_tags[i]])
    probe_sample_list = []
    for arrival_sample, rate_tag in arrival_samples:
        probe_samples = [[[], 0] for i in range(int(period/window_size))]
        for DNS_arrival in arrival_sample:
            probe_samples[int(DNS_arrival/window_size)][1] += 1/window_size
        rates = np.array([x[1] for x in probe_samples])
        max_rate = np.max(rates)
        min_rate = np.min(rates)
        # print('min rate:' + str(min_rate) + ', max rate:' + str(max_rate))
        cache_samples = simulate_probe(arrival_sample, TTL, second_precision)
        for i in range(len(cache_samples)):
            probe_samples[int(cache_samples[i][0] / window_size)][0].append(cache_samples[i][1])
        probe_sample_list.append([probe_samples, max_rate, TTL, rate_tag])
    return probe_sample_list


def simulate_probe_samples_from_rates(rate_samples, TTL=300, second_precision=False, arrival_num=1):
    period = 86400
    rate_step = 600
    window_size = 600
    arrival_samples = []
    rate_tags = np.random.random(len(rate_samples)).tolist()
    for i in range(len(rate_samples)):
        rate_sample = rate_samples[i]
        rate_sample = [max(x, 0.0001) for x in rate_sample]
        for j in range(arrival_num):
            t = 0
            arrival_sample = []
            while t <= period:
                t = t + np.random.exponential(1/rate_sample[int(t/rate_step)], 1)[0]
                arrival_sample.append(t)
            if len(arrival_sample) > 0:
                del arrival_sample[-1]
            arrival_samples.append([arrival_sample, rate_tags[i]])
    probe_sample_list = []
    for arrival_sample, rate_tag in arrival_samples:
        probe_samples = [[[], 0] for i in range(int(period/window_size))]
        for DNS_arrival in arrival_sample:
            probe_samples[int(DNS_arrival/window_size)][1] += 1/window_size
        rates = np.array([x[1] for x in probe_samples])
        max_rate = np.max(rates)
        min_rate = np.min(rates)
        # print('min rate:' + str(min_rate) + ', max rate:' + str(max_rate))
        cache_samples = simulate_probe(arrival_sample, TTL, second_precision)
        for i in range(len(cache_samples)):
            probe_samples[int(cache_samples[i][0] / window_size)][0].append(cache_samples[i][1])
        probe_sample_list.append([probe_samples, max_rate, TTL, rate_tag])
    return probe_sample_list


def simulate_probe_from_file(trace_file_name, cache_num=2, TTL=300, second_precision=True):
    period = 86400
    window_size = 600
    if trace_file_name.__contains__('.txt') is False:
        return None
    DNS_traces = np.array(pd.read_csv(trace_file_name, header=None))
    arrival_sample = DNS_traces[:, 0].tolist()
    all_arrival_sample = list(filter(lambda x: x >= 0, arrival_sample))
    all_arrival_sample = list(sorted(all_arrival_sample))
    arrival_sample_list = get_arrival_samples(all_arrival_sample, cache_num)

    temp_probe_sample_list = []
    for i in range(len(arrival_sample_list)):
        temp_probe_sample_list.append([])
        cache_arrival_sample = arrival_sample_list[i]
        cache_probe_sample_list = get_probe_sample(cache_arrival_sample, window_size, TTL, second_precision, period, trace_file_name)
        for j in range(len(cache_probe_sample_list)):
            temp_probe_sample_list[i].append(cache_probe_sample_list[j])
        # probe_sample_list.append(cache_probe_sample_list)
    days = int(np.max([len(x) for x in temp_probe_sample_list]))
    probe_sample_list = [[] for i in range(days)]
    for i in range(len(temp_probe_sample_list)):
        for j in range(len(temp_probe_sample_list[i])):
            probe_sample_list[j].append(temp_probe_sample_list[i][j])
    return probe_sample_list


def get_arrival_samples(all_arrival_sample, sample_num=2):
    arrival_sample_list = []
    indexed_all_arrival_sample = np.array(all_arrival_sample)
    arrival_sample_index = np.random.randint(0, sample_num, len(all_arrival_sample))
    for k in range(sample_num):
        cache_sample_index = np.argwhere(arrival_sample_index == k).flatten().tolist()
        cache_sample_index = sorted(cache_sample_index)
        cache_arrival_sample = indexed_all_arrival_sample[cache_sample_index].tolist()
        # if cache_arrival_sample[-1] != np.max(cache_arrival_sample):
        #     print('nihao')
        arrival_sample_list.append(cache_arrival_sample)
    return arrival_sample_list


def get_probe_sample(arrival_sample, window_size, TTL, second_precision, period, trace_file_name):
    probe_samples = [[[], 0] for i in range(int(np.ceil(arrival_sample[-1] / window_size)))]
    # probe_samples = [[[], 0] for i in range(int(arrival_sample[-1] / window_size)+1)]
    for DNS_arrival in arrival_sample:
        if int(DNS_arrival / window_size) >= len(probe_samples):
            print('nihao')
        probe_samples[int(DNS_arrival / window_size)][1] += 1 / window_size
    rates = np.array([x[1] for x in probe_samples])
    max_rate = np.max(rates)
    min_rate = np.min(rates)
    print('min rate:' + str(min_rate) + ', max rate:' + str(max_rate))
    cache_samples = simulate_probe(arrival_sample, TTL, second_precision)
    for i in range(len(cache_samples)):
        if cache_samples[i][1] == 0:
            print('nihao')
        probe_samples[int(cache_samples[i][0] / window_size)][0].append(cache_samples[i][1])
    days = int(np.ceil(len(probe_samples) / (period / window_size)))
    trace_tag = os.path.split(trace_file_name)[1]
    trace_tag = trace_tag[:-4]
    probe_sample_list = [[[], max_rate, TTL, trace_tag] for i in range(days)]

    for i in range(len(probe_samples)):
        day_index = int(i / (period / window_size))
        probe_sample_list[day_index][0].append(probe_samples[i])
    if len(probe_sample_list[-1][0]) != period / window_size:
        del probe_sample_list[-1]
    return probe_sample_list


if __name__ == '__main__':
    # simulate_probe_samples()
    file_name = r'D:/work/happy_dns_estimation_data/dns_data_1/dns_data_1/zyx.qq.com_202.117.0.20.txt'
    simulate_probe_from_file(file_name)
