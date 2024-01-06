import os.path
import numpy as np
import numpy.random as npr
import pandas as pd


def simulate_GBM_rate(mu=0.5, sigma=1, max_rate=1, sample_num=1):
    S0 = 1
    u = mu
    T = 1
    I = sample_num
    M = int(86400 / 60)
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((u - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * npr.standard_normal(I))
    rate_samples = []
    for i in range(sample_num):
        S[:, i] = S[:, i]/np.max(S[:, i])*max_rate
        rate_samples.append(S[:, i].tolist())
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


def simulate_probe_from_file(trace_file_name, TTL=300, second_precision=True):
    period = 86400
    window_size = 600
    if trace_file_name.__contains__('.txt') is False:
        return None
    DNS_traces = np.array(pd.read_csv(trace_file_name, header=None))
    arrival_sample = DNS_traces[:, 0].tolist()
    arrival_sample = list(filter(lambda x:x>=0, arrival_sample))

    probe_samples = [[[], 0] for i in range(int(np.ceil(arrival_sample[-1] / window_size)))]
    for DNS_arrival in arrival_sample:
        probe_samples[int(DNS_arrival / window_size)][1] += 1 / window_size
    rates = np.array([x[1] for x in probe_samples])
    max_rate = np.max(rates)
    min_rate = np.min(rates)
    print('min rate:' + str(min_rate) + ', max rate:' + str(max_rate))
    cache_samples = simulate_probe(arrival_sample, TTL, second_precision)
    for i in range(len(cache_samples)):
        probe_samples[int(cache_samples[i][0] / window_size)][0].append(cache_samples[i][1])
    days = int(np.ceil(len(probe_samples)/(period/window_size)))
    trace_tag = os.path.split(trace_file_name)[1]
    trace_tag = trace_tag[:-4]
    probe_sample_list = [[[], max_rate, TTL, trace_tag] for i in range(days)]

    for i in range(len(probe_samples)):
        day_index = int(i/(period/window_size))
        probe_sample_list[day_index][0].append(probe_samples[i])
    if len(probe_sample_list[-1][0]) != period/window_size:
        del probe_sample_list[-1]
    return probe_sample_list

