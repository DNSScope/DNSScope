import collections
import copy
import datetime
import multiprocessing
import os
from os import walk

import numpy as np
import torch
from torch import nn
import random
import pandas as pd
import pickle
from batch_generate_dns_sample_probe import get_features_in_estimation, generate_trace_direct, get_mix_exp_rate
from simulate_dns_arrival_sample import simulate_probe_samples_from_rates
from os.path import exists
from torch.autograd import Variable
import sys

z_dimension = 10
# occupy_task = False
occupy_task = True

my_dns_list = ['202.117.0.20']

max_error_ratio = 5
min_error_ratio = -1

extended_window_size = 10
retrain_trace_num = 1

retrain_proc_num = 8

retrain_num = 1

max_feature_dim = 101
feature_index = list(range(0, max_feature_dim, 2))
rate_index = list(range(extended_window_size))

normalized_lower_bound = 0.005

is_retrain = True
batch_size = 10
validation_step = 10

# min_epoch = 20
min_epoch = 50

# model_path = r'/Users/jfli/ftp/model/'
# # testing_sample_path = r'../happy_dns_estimation_data/probe_sample_type_3_20230628/'
# testing_sample_path = r'../happy_dns_estimation_data/probe_sample_type_3_20230701/'

model_path = r'../model/'
testing_sample_path = r'../data/probe_sample_type_3/'


def load_model_config():
    with open(config_file_name, 'r') as file:
        contents = file.readlines()
    contents = [x.strip().split(';') for x in contents]
    config_table = {}
    for i in range(len(contents)):
        config_items = [eval(x) for x in contents[i]]
        config_index, TTL, active_feature_index, loss_weight = config_items
        config_table[config_index] = [TTL, active_feature_index, loss_weight]
    return config_table


class ArrivalRateEstimator(nn.Module):
    def __init__(self, input_size, active_feature_index):
        super(ArrivalRateEstimator, self).__init__()
        self.input_size = input_size
        self.hidden_size = 100
        self.out_size = 1
        self.active_feature_index = active_feature_index
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            dropout=0.1,
            bidirectional=True
        )

        self.out = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),nn.Dropout(p=0.1),nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),nn.PReLU(),
            # nn.Linear(self.hidden_size, self.hidden_size), nn.PReLU(),
                                 nn.Linear(self.hidden_size, self.out_size))

    def forward(self, x):
        r_out, h_state = self.rnn(x[:, :, self.active_feature_index], None)
        my_outs = self.out(r_out)
        return my_outs, h_state


def train(training_samples, validation_samples, my_predictor, loss_weight):

    learning_rate = 0.001
    optimizer = torch.optim.Adam(my_predictor.parameters(), lr=learning_rate)

    loss_f = nn.MSELoss().to(device)
    repeat_num = 100000
    n = 50

    # min_rate = torch.from_numpy(np.array(0.0001)).float()
    rate_lower_bound = torch.from_numpy(np.array(normalized_lower_bound)).float()
    batch_feature_list = []
    batch_rate_list = []
    batch_normalized_rate_list = []
    for i in range(len(training_samples)):
        features = training_samples[i][0]
        features = fix_feature(features)
        features = features[:, feature_index]
        xs = torch.from_numpy(np.array(features)).float()
        ys = torch.from_numpy(np.array(training_samples[i][1])).float().reshape(-1, 1)
        zs = torch.maximum(ys, rate_lower_bound)
        # data_list_train.append([xs, ys])
        max_rate = np.max(training_samples[i][1])
        min_rate = np.min(training_samples[i][1])
        # print('min rate:'+str(min_rate)+', max rate:'+str(max_rate))
        batch_index = int(i/batch_size)
        if batch_index >= len(batch_feature_list):
            batch_feature_list.append([])
            batch_rate_list.append([])
            batch_normalized_rate_list.append([])
        batch_feature_list[batch_index].append(xs)
        batch_rate_list[batch_index].append(ys)
        batch_normalized_rate_list[batch_index].append(zs)

    for i in range(len(batch_feature_list)):
        batch_feature_list[i] = torch.stack(batch_feature_list[i], dim=1).to(device)
        batch_rate_list[i] = torch.stack(batch_rate_list[i], dim=1).to(device)
        batch_normalized_rate_list[i] = torch.stack(batch_normalized_rate_list[i], dim=1)
        batch_normalized_rate_list[i] = torch.mean(batch_normalized_rate_list[i], dim=0).to(device)

    validation_features = []
    validation_rates = []
    validation_normalized_rates = []
    for i in range(len(validation_samples)):
        features = validation_samples[i][0]
        features = fix_feature(features)
        features = features[:, feature_index]
        xs = torch.from_numpy(np.array(features)).float()
        ys = torch.from_numpy(np.array(validation_samples[i][1])).float().reshape(-1, 1)
        zs = torch.maximum(ys, rate_lower_bound)
        validation_features.append(xs)
        validation_rates.append(ys)
        validation_normalized_rates.append(zs)

    validation_features = torch.stack(validation_features, dim=1).to(device)
    validation_rates = torch.stack(validation_rates, dim=1).to(device)
    validation_normalized_rates = torch.stack(validation_normalized_rates, dim=1)
    validation_normalized_rates = torch.mean(validation_normalized_rates, dim=0).to(device)

    validation_loss_list = []
    stop_counter = 0
    timestamp = str(datetime.datetime.now().timestamp())
    model_list = []
    for epoch in range(repeat_num):
        # print(str(epoch))
        my_predictor.train()

        total_loss = 0
        for batch_index in range(len(batch_feature_list)):
            my_outs, h_state = my_predictor(batch_feature_list[batch_index])
            # input = my_outs
            # target = batch_rate_list[batch_index]

            input = torch.div(my_outs, batch_normalized_rate_list[batch_index])
            target = torch.div(batch_rate_list[batch_index], batch_normalized_rate_list[batch_index])

            loss = loss_f(my_outs, batch_rate_list[batch_index])*(1-loss_weight) + loss_f(input, target) * loss_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().numpy()

        # torch.save(my_predictor, model_file_name)
        model_list.append(copy.deepcopy(my_predictor))
        # print('training loss:'+str(total_loss))
        my_predictor.eval()
        my_outs, h_state = my_predictor(validation_features)
        # input = my_outs
        # target = validation_rates

        input = torch.div(my_outs, validation_normalized_rates)
        target = torch.div(validation_rates, validation_normalized_rates)

        validation_loss = (loss_f(my_outs, validation_rates)*(1-loss_weight) + loss_f(input, target) * loss_weight).cpu().detach().numpy()

        # print('validation loss:' + str(validation_loss))
        validation_loss_list.append(validation_loss)
        if validation_loss <= np.min(np.array(validation_loss_list)):
            stop_counter = 0
        else:
            stop_counter += 1
        if stop_counter > validation_step and epoch > min_epoch:
            break
        # print('-'*100)
    optimal_model_index = np.argmin(np.array(validation_loss_list))
    my_predictor = model_list[optimal_model_index]
    my_predictor.eval()
    return my_predictor


def fusion_estimation_grid(estimation_list, testing_samples):
    alpha_list = np.arange(-10, 10, 1).tolist()
    beta_list = np.arange(0, 5, 1).tolist()
    likelihood_list = []
    for alpha in alpha_list:
        for beta in beta_list:
            likelihood, estimated_rate_list = get_likelihood_np(estimation_list, testing_samples, alpha, beta)
            likelihood_list.append([[alpha, beta], likelihood, estimated_rate_list])

    likelihood_list = sorted(likelihood_list, key=lambda x: x[1])
    estimated_rate_list = likelihood_list[0][2]
    return estimated_rate_list


def smooth_relative_errors(relative_errors):
    new_relative_errors = []
    r = 1
    m = len(relative_errors)
    for i in range(m):
        start_index = max(i-r, 0)
        end_index = min(i+r, m-1)
        new_relative_errors.append(np.mean(relative_errors[list(range(start_index, end_index+1))]))
    # print('nihao')
    return np.array(new_relative_errors)


def generate_rate_samples_guided(my_mapper, estimated_rates, average_rate):
    new_samples = np.array(estimated_rates)
    generated_rates_list = []
    repeat_num = 100
    # repeat_num = 10
    good_candidate_num = 10
    estimated_data = torch.from_numpy(new_samples).float()
    estimated_data = estimated_data.to(device)
    estimated_data = estimated_data.unsqueeze(0).unsqueeze(2)
    rate_candidates = []
    for i in range(repeat_num):
        # error_ratios = my_mapper.shift(estimated_data, 0.01)
        error_ratios = my_mapper.shift(estimated_data, my_drift_factor)

        # error_ratios[len(error_ratios)-1] = 0

        error_ratios = np.minimum(np.maximum(error_ratios, min_error_ratio), max_error_ratio)
        estimated_rates = estimated_data[0, :, 0].cpu().detach().numpy()
        # relative_errors = smooth_relative_errors(relative_errors)
        fake_data = estimated_rates*(1+error_ratios)
        mean_error = np.mean(error_ratios)
        rate_candidates.append([fake_data, np.abs(np.mean(fake_data)-average_rate), mean_error])
    rate_candidates = sorted(rate_candidates, key=lambda x: x[1])
    generated_rates_list = rate_candidates[:good_candidate_num]
    relative_errors = [x[2] for x in generated_rates_list]
    # for i in range(len(generated_rates_list)):
    #     relative_errors.append(np.mean((generated_rates_list[i][0]-new_samples)/new_samples))
    generated_rates_list = [x[0].tolist() for x in generated_rates_list]
    print(relative_errors)
    return generated_rates_list


def generate_rate_samples(my_mapper, estimated_rates):
    new_samples = np.array(estimated_rates)
    generated_rates_list = []
    repeat_num = 1
    # repeat_num = 50
    estimated_data = torch.from_numpy(new_samples).float()
    estimated_data = estimated_data.to(device)
    estimated_data = estimated_data.unsqueeze(0).unsqueeze(2)
    for i in range(repeat_num):
        error_ratios = my_mapper.shift(copy.deepcopy(estimated_data), my_drift_factor)
        error_ratios = np.minimum(np.maximum(error_ratios, min_error_ratio), max_error_ratio)
        estimated_rates = estimated_data[0, :, 0].detach().numpy()
        # relative_errors = smooth_relative_errors(relative_errors)
        fake_data = estimated_rates*(1+error_ratios)
        generated_rates_list.append(fake_data.tolist())
    return generated_rates_list


def rate_estimation_int_interval(task_info):
    # print('#'*100)
    all_testing_samples, my_predictor, model_tag, loss_weight, TTL, trace_tag, sample_percentage = task_info
    # output_file_name = result_path + trace_tag + '_' + str(model_tag) + '.txt'
    output_file_name = result_path + trace_tag + '_' + str(model_tag) + '_' + str(sample_percentage) + '.txt'
    if exists(output_file_name) and is_force_estimate is False:
        print('skip '+output_file_name)
        return
    if occupy_task is True:
        with open(output_file_name, 'w') as file:
            file.writelines([])

    predictor_raw = copy.deepcopy(my_predictor).to(device)
    ae_model_file_name = model_path + 'AE_GRU_' + str(TTL) + '_' + str(model_tag) + '.model'
    if gpu_index == -1:
        my_mapper = torch.load(ae_model_file_name, map_location=torch.device('cpu'))
    else:
        my_mapper = torch.load(ae_model_file_name)
    my_mapper = my_mapper.to(device)
    my_mapper.device = device
    my_mapper.encoder.device = device
    my_mapper.decoder.device = device

    my_cache_num = len(all_testing_samples[0])
    all_result_list = []
    for k in range(my_cache_num):
        testing_samples = []
        for cache_testing_samples in all_testing_samples:
            if k < len(cache_testing_samples):
                testing_samples.append(cache_testing_samples[k])

        testing_rates, estimation_list, ML_estimation_list, UB_estimation_list, estimation_info_1 = rate_estimate(
            predictor_raw, copy.deepcopy(testing_samples), trace_tag)
        result_list = []

        for i in range(estimation_list.shape[1]):
            results = []
            results.append(estimation_list[:, i].reshape((-1, 1)))
            for j in range(ML_estimation_list.shape[2]):
                results.append(ML_estimation_list[:, i, j].reshape((-1, 1)))
                results.append(UB_estimation_list[:, i, j].reshape((-1, 1)))
            result_list.append(results)

        estimation_info_2 = ''
        if is_retrain:
            mix_exp_rates = [0 for _ in range(len(testing_samples))]

            task_list = []
            for i in range(len(testing_samples)):
                task_list.append([i, testing_samples[i], TTL])

            with multiprocessing.Pool(processes=retrain_proc_num) as pool:
                for task_index, average_rate in pool.imap_unordered(compute_mix_exp_rate, task_list):
                    # print(average_rate)
                    mix_exp_rates[task_index] = average_rate

            for _ in range(retrain_num):
                rate_samples = []
                for i in range(estimation_list.shape[1]):
                    # rate_samples.extend(generate_rate_samples(my_mapper, estimation_list[:, i]))
                    rate_samples.extend(
                        generate_rate_samples_guided(my_mapper, estimation_list[:, i], mix_exp_rates[i]))
                probe_sample_list = simulate_probe_samples_from_rates(rate_samples, TTL=int(TTL),
                                                                      arrival_num=retrain_trace_num)
                new_samples = []

                with multiprocessing.Pool(processes=retrain_proc_num) as pool:
                    for new_sample in pool.imap_unordered(generate_trace_direct, probe_sample_list):
                        new_samples.append(new_sample)

                random.shuffle(new_samples)
                n = int(len(new_samples) / 3)
                training_samples = new_samples[n:]
                validation_samples = new_samples[:n]
                predictor = train(training_samples, validation_samples, copy.deepcopy(predictor_raw), loss_weight)
                testing_rates, retrain_estimation_list, _, _, estimation_info_2 = rate_estimate(predictor,
                                                                                                copy.deepcopy(
                                                                                                    testing_samples),
                                                                                                trace_tag)
                for i in range(retrain_estimation_list.shape[1]):
                    result_list[i].insert(0, retrain_estimation_list[:, i].reshape((-1, 1)))
                    result_list[i].insert(0, np.array(testing_rates[i]).reshape((-1, 1)))
        for i in range(len(result_list)):
            result_list[i] = np.hstack(result_list[i])
        result_list = np.vstack(result_list)

        all_result_list.append(result_list[:, [0, 1, 2]])
    result_length = np.max([x.shape[0] for x in all_result_list])
    merged_results = np.zeros((result_length, 3))
    for results in all_result_list:
        merged_results[:results.shape[0], :results.shape[1]] = merged_results[:results.shape[0],
                                                               :results.shape[1]] + np.maximum(results, 0)
    get_estimation_error(merged_results, trace_tag)
    df = pd.DataFrame(merged_results)
    df.to_csv(output_file_name, header=False, index=False)
    print('output ' + output_file_name)


def get_estimation_error(results, trace_tag):
    rates = results[:, 0]
    estimated_rates_1 = results[:, 1]
    estimated_rates_2 = results[:, 2]
    error_1 = np.mean(np.abs(estimated_rates_1 - rates))
    error_2 = np.mean(np.abs(estimated_rates_2 - rates))
    print(trace_tag+'*'*100)
    print('DNS estimation error 1:'+str(error_1)+', error 2:'+str(error_2))


def compute_mix_exp_rate(task_info):
    task_index, testing_sample, TTL = task_info
    average_rate = get_mix_exp_rate(testing_sample[2], TTL)
    print('real:' + str(np.mean(testing_sample[1]))+', estimated:'+str(average_rate))
    return task_index, average_rate


def rate_estimate(my_predictor, testing_samples, trace_tag):
    testing_rates = []
    estimation_list = np.ones((testing_samples[0][0].shape[0], len(testing_samples)))
    refine_step = 10

    ML_estimation_list = []
    UB_estimation_list = []
    for k in range(refine_step):
        # print(k)
        testing_features = []
        testing_rates = []
        for i in range(len(testing_samples)):
            # print(estimation_list.reshape((1, -1)))
            probe_sample = testing_samples[i][2]
            features = get_features_in_estimation(probe_sample, estimation_list[:, i])
            if k == 0:
                ML_estimation_list.append(copy.deepcopy(features)[:, feature_index[:-1]])
            features = fix_feature(features)
            if k == 0:
                UB_estimation_list.append(copy.deepcopy(features)[:, feature_index[:-1]])

            xs = torch.from_numpy(np.array(features[:, feature_index])).float()
            ys = testing_samples[i][1]
            testing_features.append(xs)
            testing_rates.append(ys)

        testing_features = torch.stack(testing_features, dim=1).to(device)
        my_outs, h_state = my_predictor(testing_features)
        estimation_list = my_outs.cpu().detach().numpy()[:, :, 0]

    ML_estimation_list = np.stack(ML_estimation_list, axis=1)
    UB_estimation_list = np.stack(UB_estimation_list, axis=1)

    reference_rate_list = ML_estimation_list[:, :, rate_index]

    error_list_1 = []
    error_list_2 = []
    error_list_3 = []
    error_list_4 = []
    # result_list = []
    max_rate_info = ''
    for i in range(estimation_list.shape[1]):
        estimation = estimation_list[:, i].flatten()
        reference_rate = reference_rate_list[:, i]
        # result = np.stack([estimation, testing_rates[i], reference_rate[:, 0]], axis=0)
        error_1 = np.mean(np.abs(estimation - np.array(testing_rates[i])))
        error_3 = np.mean((estimation - np.array(testing_rates[i])))
        max_rate_info += str(np.max(np.array(testing_rates[i]))) + ','
        error_2 = []
        error_4 = []
        for j in range(reference_rate.shape[1]):
            positive_index = np.argwhere(reference_rate[:, j] > 0).flatten().tolist()
            error_2.append(
                np.mean(np.abs(reference_rate[positive_index, j] - np.array(testing_rates[i])[positive_index])))
            error_4.append(np.mean((reference_rate[positive_index, j] - np.array(testing_rates[i])[positive_index])))
        error_list_1.append(error_1)
        error_list_2.append(error_2)
        error_list_3.append(error_3)
        error_list_4.append(error_4)
        # result_list.append(result)
    error_list_2 = np.stack(error_list_2, axis=0)

    info_1 = 'testing error:' + str(np.mean(error_list_1)) + ', reference error:' + str(np.mean(error_list_2, axis=0))
    info_1 += '\n' + 'mean testing error:' + str(np.mean(error_list_3)) + ', mean reference error:' + str(np.mean(error_list_4, axis=0))
    return testing_rates, estimation_list, ML_estimation_list, UB_estimation_list, info_1


def get_likelihood(estimation_list, testing_samples, alpha, beta):
    day_num = estimation_list.shape[1]
    estimated_rate_list = []
    test_sample_list = []
    for i in range(day_num):
        weights = (1-torch.exp(alpha)/(1+torch.exp(alpha)))*torch.exp(-torch.pow(beta, 2)*torch.abs(torch.from_numpy(np.array(list(range(day_num))))-i))/day_num
        weights[i] = torch.exp(alpha)/(1+torch.exp(alpha))
        weights = weights/torch.sum(weights)
        estimated_rates = torch.sum(estimation_list * weights, axis=1)
        estimated_rate_list.append(estimated_rates)
        probe_sample = testing_samples[i][2]
        for j in range(len(probe_sample)):
            for interval in probe_sample[j][0]:
                test_sample = 1 - torch.exp(-estimated_rates[j]*torch.from_numpy(np.array(interval)))
                if 0 < test_sample < 1:
                    test_sample_list.append(test_sample)
    test_sample_list = torch.sort(torch.stack(test_sample_list))[0]

    S = 0
    N = len(test_sample_list)
    for i in range(1, N+1):
        S = S + (2*i-1)/N*(torch.log(test_sample_list[i-1])+torch.log(1-test_sample_list[N-i]))
    A2 = -N-S
    likelihood = A2
    estimated_rate_list = torch.stack(estimated_rate_list, axis=1).detach().numpy()
    return likelihood, estimated_rate_list


def get_likelihood_np(estimation_list, testing_samples, alpha, beta):
    day_num = estimation_list.shape[1]
    estimated_rate_list = []
    test_sample_list = []
    for i in range(day_num):
        weights = (1-np.exp(alpha)/(1+np.exp(alpha)))*np.exp(-np.power(beta, 2)*np.abs(np.array(list(range(day_num)))-i))/day_num
        weights[i] = np.exp(alpha)/(1+np.exp(alpha))
        weights = weights/np.sum(weights)
        estimated_rates = np.sum(estimation_list * weights, axis=1)
        estimated_rate_list.append(estimated_rates)
        probe_sample = testing_samples[i][2]
        for j in range(len(probe_sample)):
            for interval in probe_sample[j][0]:
                test_sample = 1 - np.exp(-estimated_rates[j]*np.array(interval))
                if 0 < test_sample < 1:
                    test_sample_list.append(test_sample)
    test_sample_list = np.sort(np.stack(test_sample_list))

    S = 0
    N = len(test_sample_list)
    for i in range(1, N+1):
        S = S + (2*i-1)/N*(np.log(test_sample_list[i-1])+np.log(1-test_sample_list[N-i]))
    A2 = -N-S
    likelihood = A2
    estimated_rate_list = np.stack(estimated_rate_list, axis=1)
    return likelihood, estimated_rate_list


def load_model():
    model_table = collections.defaultdict(list)
    for config_index in my_config_table.keys():
        active_TTL, active_feature_index, loss_weight = my_config_table[config_index]
        model_tag = config_index
        for sample_percentage in my_percentage_list:
            model_file_name = model_path + 'model_' + str(active_TTL) + '_' + str(model_tag)+'_'+str(sample_percentage) + '.model'
            if exists(model_file_name) is False:
                continue
            my_model = torch.load(model_file_name, map_location=device)
            my_model = my_model.to(device)
            my_model.eval()
            model_table[active_TTL].append([my_model, model_tag, loss_weight, sample_percentage])
    # model_file_name = model_path + 'model_' + str(active_TTL) + '.model'
    return model_table


def evaluate_model(testing_sample_table):
    model_table = load_model()
    for trace_tag in testing_sample_table.keys():
        strs = trace_tag.split('_')
        active_ttl = int(strs[0])
        if model_table.__contains__(active_ttl) is False:
            continue
        model_list = model_table[active_ttl]
        for my_model, model_tag, loss_weight, sample_percentage in model_list:
            task_info = [testing_sample_table[trace_tag], my_model, model_tag, loss_weight, active_ttl, trace_tag, sample_percentage]
            rate_estimation_int_interval(task_info)


def evaluate(my_predictor, features, rates):
    my_predictor.eval()
    my_outs, h_state = my_predictor(features)
    estimation_list = my_outs.detach().numpy()
    error_list_1 = []
    error_list_2 = []
    result_list = []

    reference_rate_list = features[:, :, rate_index].detach().numpy()
    for i in range(estimation_list.shape[1]):
        estimation = estimation_list[:,i,:].flatten()
        reference_rate = reference_rate_list[:, i]
        result = np.stack([estimation, rates[i]], axis=0)
        error_1 = np.mean(np.abs(estimation - np.array(rates[i])))
        error_2 = np.mean(np.abs(reference_rate - np.array(rates[i]).reshape((-1, 1))), axis=0)
        error_list_1.append(error_1)
        error_list_2.append(error_2)
        result_list.append(result)
    error_list_2 = np.stack(error_list_2, axis=0)
    print('testing error:'+str(np.mean(error_list_1))+', reference error:'+str(np.mean(error_list_2, axis=0)))


def list_file(sample_path, active_TTL):
    file_name_list = []
    for dir_path, folders, files in walk(sample_path):
        for file_name in files:
            if file_name.__contains__('.pkl') is False:
                continue
            strs = file_name.split('_')
            if strs[2] != str(active_TTL):
                continue
            file_name_list.append(sample_path + file_name)
    return file_name_list


def fix_feature(raw_features):
    # a = copy.deepcopy(raw_features)
    for i in feature_index[:-1]:
        positive_index = np.argwhere(raw_features[:, i] > 0).flatten().tolist()

        for j in positive_index:
            interval_num = raw_features[j, i + 1]
            if interval_num > 1:
                raw_features[j, i] = raw_features[j, i]*(interval_num-1)/interval_num
    return raw_features


def load_data(sample_path, active_TTL):
    sample_files = list_file(sample_path, active_TTL)
    samples = []
    for sample_file_name in sample_files:
        with open(sample_file_name, 'rb') as file:
            samples.append(pickle.load(file))
    return samples


def list_testing_file(sample_path):
    file_name_table = collections.defaultdict(list)
    for dir_path, folders, files in walk(sample_path):
        for file_name in files:
            if file_name.__contains__('.pkl') is False:
                continue
            strs = file_name.split('_')
            max_rate = strs[1]
            ttl = strs[2]
            domain = strs[3]
            dns = strs[4]
            if domain not in my_domain_list:
                continue
            if dns not in my_dns_list:
                continue
            trace_tag = ttl + '_' + domain + '_' + dns + '_' + max_rate
            file_name_table[trace_tag].append(sample_path + file_name)
    return file_name_table


def load_testing_data(sample_path):
    sample_file_table = list_testing_file(sample_path)
    # for trace_tag in sample_file_table.keys():
    #     sample_file_table[trace_tag] = [sample_file_table[trace_tag][0]]

    sample_table = collections.defaultdict(list)
    for trace_tag in sample_file_table.keys():
        for sample_file_name in sample_file_table[trace_tag]:
            with open(sample_file_name, 'rb') as file:
                sample = pickle.load(file)
                sample_table[trace_tag].append(sample)
    return sample_table


def run():
    testing_sample_table = load_testing_data(testing_sample_path)
    print(str(len(testing_sample_table))+' traces')
    evaluate_model(testing_sample_table)


if __name__ == '__main__':
    gpu_index = -1
    my_drift_factor = 0.0005
    my_sample_percentage = 0.5
    if len(sys.argv) > 1:
        gpu_index = int(sys.argv[1])
    if len(sys.argv) > 2:
        my_drift_factor = float(sys.argv[2])
    if len(sys.argv) > 3:
        my_sample_percentage = float(sys.argv[3])
    is_force_estimate = False
    # is_force_estimate = True

    my_percentage_list = [my_sample_percentage]

    result_path = r'../result/hybrid_r_dns_80_type_3/'
    if exists(result_path) is False:
        os.mkdir(result_path)
    config_file_name = r'model_config_card_all.txt'
    device = 'cpu'
    if gpu_index == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif gpu_index == 1:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    my_config_table = load_model_config()
    my_domain_list = np.array(pd.read_csv(r'my_domain_list_probe.txt', header=None)).flatten()
    my_index = list(range(len(my_domain_list)))
    my_domain_list = my_domain_list[my_index].tolist()
    run()

    print('Done!')