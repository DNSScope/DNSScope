import collections
import copy
import datetime
import multiprocessing
import sys
from os import walk
from os.path import exists

import numpy as np
import torch
from torch import nn
import random
import pandas as pd
import pickle
from batch_generate_dns_sample import get_features_in_estimation


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

occupy_task = True
is_force_train = False

batch_size = 10
validation_step = 20
min_epoch = 100
test_proc_num = 5
normalized_lower_bound = 0.005

model_path = r'../model/'
temp_path = r'../temp/'
training_sample_path = r'../data/simulated_trace/'
testing_sample_path = r'../data/simulated_trace_test/'

max_feature_dim = 101
# feature_index = list(range(0, 15, 2))
feature_index = list(range(0, max_feature_dim, 2))

rate_index = [0, 1, 2, 3, 4]


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


def train(training_samples, validation_samples, testing_sample_table, active_TTL, active_feature_index, loss_weight, model_tag):
    optimal_model_file_name = model_path + 'model_' + str(active_TTL) + '_' + str(model_tag) + '.model'
    if exists(optimal_model_file_name) and is_force_train is False:
        print('skip '+optimal_model_file_name)
        return
    if occupy_task is True:
        with open(optimal_model_file_name, 'w') as file:
            file.writelines([])

    testing_samples = []
    for trace_tag in testing_sample_table.keys():
        testing_samples.extend(testing_sample_table[trace_tag])

    my_predictor = ArrivalRateEstimator(len(active_feature_index), active_feature_index).to(device)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(my_predictor.parameters(), lr=learning_rate)

    loss_f = nn.MSELoss().to(device)
    repeat_num = 100000

    rate_lower_bound = torch.from_numpy(np.array(normalized_lower_bound)).float()
    batch_feature_list = []
    batch_rate_list = []
    batch_normalized_rate_list = []
    for i in range(len(training_samples)):
        features = training_samples[i][0]
        features = fix_feature(features)
        features = features[:, feature_index]
        xs = torch.from_numpy(np.array(features)).float()
        ys = torch.from_numpy(np.array(training_samples[i][1])).float().reshape(-1,1)
        zs = torch.maximum(ys, rate_lower_bound)
        # data_list_train.append([xs, ys])
        max_rate = np.max(training_samples[i][1])
        min_rate = np.min(training_samples[i][1])
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

    print('-'*100)
    testing_features = []
    testing_rates = []
    raw_testing_features = []
    for i in range(len(testing_samples)):
        features = testing_samples[i][0]
        raw_testing_features.append(copy.deepcopy(features)[:, feature_index])
        features = fix_feature(features)
        features = features[:, feature_index]
        xs = torch.from_numpy(np.array(features)).float()
        ys = testing_samples[i][1]
        testing_features.append(xs)
        testing_rates.append(ys)

    testing_features = torch.stack(testing_features, dim=1).to(device)
    raw_testing_features = np.stack(raw_testing_features, axis=1)
    reference_rate_list = raw_testing_features[:, :, rate_index]

    validation_loss_list = []
    stop_counter = 0
    for epoch in range(repeat_num):
        print(str(epoch))
        model_file_name = temp_path + 'model_'+str(active_TTL)+'_' + str(model_tag) + '_'+str(epoch)+'.model'
        my_predictor.train()

        total_loss = 0
        for batch_index in range(len(batch_feature_list)):
            # h_state = None
            # h_state = h_state.to(device)
            # my_outs, h_state = my_predictor(batch_feature_list[batch_index], h_state)
            my_outs, h_state = my_predictor(batch_feature_list[batch_index])

            input = torch.div(my_outs, batch_normalized_rate_list[batch_index])
            target = torch.div(batch_rate_list[batch_index], batch_normalized_rate_list[batch_index])

            loss = loss_f(my_outs, batch_rate_list[batch_index])*(1-loss_weight) + loss_f(input, target) * loss_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().detach().numpy()

        torch.save(my_predictor, model_file_name)
        print('training loss:'+str(total_loss))
        my_predictor.eval()
        h_state = None
        # my_outs, h_state = my_predictor(validation_features, h_state)
        my_outs, h_state = my_predictor(validation_features)

        input = torch.div(my_outs, validation_normalized_rates)
        target = torch.div(validation_rates, validation_normalized_rates)
        validation_loss = (loss_f(my_outs, validation_rates)*(1-loss_weight) + loss_f(input, target) * loss_weight).cpu().detach().numpy()

        print('validation loss:' + str(validation_loss))
        validation_loss_list.append(validation_loss)
        if validation_loss <= np.min(np.array(validation_loss_list)):
            stop_counter = 0
        else:
            stop_counter += 1
        if stop_counter > validation_step and epoch > min_epoch:
            break
        if epoch % 20 == 0:
            evaluate(my_predictor, testing_features, reference_rate_list, testing_rates)
        print('-'*100)
    optimal_model_index = np.argmin(np.array(validation_loss_list))
    model_file_name = temp_path + 'model_' + str(active_TTL) + '_' + str(model_tag) + '_' + str(optimal_model_index) + '.model'
    my_model = torch.load(model_file_name)

    torch.save(my_model, optimal_model_file_name)
    print('output '+optimal_model_file_name)


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


def rate_estimation_int_interval(task_info):
    my_predictor, testing_samples = task_info

    testing_features = []
    for i in range(len(testing_samples)):
        xs = torch.from_numpy(np.array(testing_samples[i][0][:, feature_index])).float()
        testing_features.append(xs)
    testing_features = torch.stack(testing_features, dim=1)
    reference_rate_list = testing_features[:, :, rate_index].detach().numpy()

    estimation_list = np.ones((testing_samples[0][0].shape[0], len(testing_samples)))

    refine_step = 10
    for k in range(refine_step):
        # print(k)
        testing_features = []
        testing_rates = []
        for i in range(len(testing_samples)):
            probe_sample = testing_samples[i][2]
            features = get_features_in_estimation(probe_sample, estimation_list[:, i])
            xs = torch.from_numpy(np.array(features[:, feature_index])).float()
            ys = testing_samples[i][1]
            testing_features.append(xs)
            testing_rates.append(ys)

        testing_features = torch.stack(testing_features, dim=1)

        h_state = None
        my_outs, h_state = my_predictor(testing_features)
        estimation_list = my_outs.detach().numpy()[:, :, 0]

    error_list_1 = []
    error_list_2 = []
    error_list_3 = []
    error_list_4 = []
    result_list = []
    max_rate_info = ''
    for i in range(estimation_list.shape[1]):
        estimation = estimation_list[:, i].flatten()
        reference_rate = reference_rate_list[:, i]
        result = np.stack([estimation, testing_rates[i], reference_rate[:, 0]], axis=0)
        error_1 = np.mean(np.abs(estimation - np.array(testing_rates[i])))
        error_3 = np.mean((estimation - np.array(testing_rates[i])))
        max_rate_info += str(np.max(np.array(testing_rates[i])))+','
        # error_2 = np.mean(np.abs(reference_rate - np.array(testing_rates[i]).reshape((-1, 1))), axis=0)
        error_2 = []
        error_4 = []
        for j in range(reference_rate.shape[1]):
            positive_index = np.argwhere(reference_rate[:, j] > 0).flatten().tolist()
            error_2.append(np.mean(np.abs(reference_rate[positive_index, j] - np.array(testing_rates[i])[positive_index])))
            error_4.append(np.mean((reference_rate[positive_index, j] - np.array(testing_rates[i])[positive_index])))
        error_list_1.append(error_1)
        error_list_2.append(error_2)
        error_list_3.append(error_3)
        error_list_4.append(error_4)
        result_list.append(result)
    error_list_2 = np.stack(error_list_2, axis=0)

    info_1 = 'testing error:' + str(np.mean(error_list_1)) + ', reference error:' + str(np.mean(error_list_2, axis=0))
    info_1 += '\n'+'mean testing error:' + str(np.mean(error_list_3)) + ', mean reference error:' + str(np.mean(error_list_4, axis=0))
    estimated_rate_list = fusion_estimation_grid(estimation_list, testing_samples)
    error_list_1 = []
    error_list_3 = []
    for i in range(estimated_rate_list.shape[1]):
        estimation = estimated_rate_list[:, i].flatten()
        result = np.stack([estimation, testing_rates[i]], axis=0)
        # result_list.append(result)
        error_1 = np.mean(np.abs(estimation - np.array(testing_rates[i])))
        error_list_1.append(error_1)
        error_3 = np.mean((estimation - np.array(testing_rates[i])))
        error_list_3.append(error_3)
    info_2 = 'fusion estimation error:' + str(np.mean(error_list_1))
    info_2 += '\n' + 'mean fusion estimation error:' + str(np.mean(error_list_3))
    return result_list


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


def evaluate(my_predictor, features, reference_rate_list, rates):
    my_predictor.eval()
    h_state = None
    my_outs, h_state = my_predictor(features)
    estimation_list = my_outs.cpu().detach().numpy()
    error_list_1 = []
    error_list_2 = []
    result_list = []

    # reference_rate_list = features[:, :, rate_index].detach().numpy()
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
    # print('testing error:'+str(np.mean(error_list_1))+', reference error:'+str(np.mean(error_list_2, axis=0)))


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


def list_testing_file(sample_path, active_TTL):
    file_name_table = collections.defaultdict(list)
    for dir_path, folders, files in walk(sample_path):
        for file_name in files:
            if file_name.__contains__('.pkl') is False:
                continue
            strs = file_name.split('_')
            if strs[2] != str(active_TTL):
                continue
            trace_tag = strs[3]
            file_name_table[trace_tag].append(sample_path + file_name)
    return file_name_table


def load_testing_data(sample_path, active_TTL):
    sample_file_table = list_testing_file(sample_path, active_TTL)
    sample_table = collections.defaultdict(list)
    for trace_tag in sample_file_table.keys():
        for sample_file_name in sample_file_table[trace_tag]:
            with open(sample_file_name, 'rb') as file:
                sample_table[trace_tag].append(pickle.load(file))
    return sample_table


def run(task_info):
    active_TTL, active_feature_index, loss_weight, model_tag, sample_percentage = task_info
    samples = load_data(training_sample_path, active_TTL)
    random.shuffle(samples)
    m = int(np.ceil(len(samples) * sample_percentage))
    samples = samples[:m]
    print('sample num:'+str(len(samples)))
    n = int(len(samples)/3)
    training_samples = samples[n:]
    validation_samples = samples[:n]
    testing_sample_table = load_testing_data(testing_sample_path, active_TTL)
    train(training_samples, validation_samples, testing_sample_table, active_TTL, active_feature_index, loss_weight, str(model_tag)+'_'+str(sample_percentage))


def load_model_config():
    config_file_name = r'model_config_card_all.txt'
    with open(config_file_name, 'r') as file:
        contents = file.readlines()
    contents = [x.strip().split(';') for x in contents]
    config_table = {}
    for i in range(len(contents)):
        config_items = [eval(x) for x in contents[i]]
        config_index, TTL, active_feature_index, loss_weight = config_items
        config_table[config_index] = [TTL, active_feature_index, loss_weight]
    return config_table


if __name__ == '__main__':
    gpu_index = 0
    if len(sys.argv) > 1:
        gpu_index = int(sys.argv[1])

    device = 'cpu'
    if gpu_index == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif gpu_index == 1:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    my_percentage_list = [0.5]
    my_config_table = load_model_config()
    task_list = []
    for my_config_index in my_config_table.keys():
        my_task_info = my_config_table[my_config_index]
        my_task_info.append(my_config_index)
        for my_percentage in my_percentage_list:
            task_info = copy.deepcopy(my_task_info)
            task_info.append(my_percentage)
            task_list.append(task_info)

    for my_task_info in task_list:
        run(my_task_info)

    print('Done!')