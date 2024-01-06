import collections
import copy
import datetime
import multiprocessing
import os.path
from os import walk

import numpy as np
import torch
from torch import nn
import random
import pandas as pd
import pickle
from batch_generate_dns_sample import get_features_in_estimation
from torch.utils.data import DataLoader, TensorDataset
from AE_model_GRU import LSTMAutoEncoder

force_training = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

batch_size = 10
validation_step = 20
min_epoch = 100
test_proc_num = 5
normalized_lower_bound = 0.005

feature_proc_num = 8
z_dimension = 10
# use_existing_sample = True
use_existing_sample = False


model_path = r'../model/'
temp_path = r'../temp/'
training_sample_path = r'../data/simulated_trace/'
testing_sample_path = r'../data/simulated_trace_test/'

max_feature_dim = 101
# feature_index = list(range(0, 15, 2))
feature_index = list(range(0, max_feature_dim, 2))

rate_index = [0, 1, 2, 3, 4]

my_sample_percentage = 0.5


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


def generate_comparison_samples(my_predictor, testing_samples):
    my_predictor.eval()
    testing_rates, estimation_list, _, _, info = rate_estimate(my_predictor, testing_samples, None)
    # print(info)
    estimated_rates = []
    real_rates = []
    for i in range(estimation_list.shape[1]):
        estimated_rates.append(estimation_list[:, i].reshape((-1, 1)))
        real_rates.append(testing_rates[:, i].reshape((-1, 1)))
    estimated_rates = np.hstack(estimated_rates)
    real_rates = np.hstack(real_rates)
    return estimated_rates, real_rates


def generate_features(task_info):
    index, probe_sample, reference_rates = task_info
    features = get_features_in_estimation(probe_sample, reference_rates)
    return index, features


def rate_estimate(my_predictor, testing_samples, trace_tag):
    testing_rates = []
    estimation_list = np.ones((testing_samples[0][0].shape[0], len(testing_samples)))
    refine_step = 10
    # refine_step = 1

    ML_estimation_list = []
    UB_estimation_list = []

    for k in range(refine_step):
        print(k)
        testing_features = []
        testing_rates = []
        features_list = [[] for _ in range(len(testing_samples))]
        feature_task_list = [[i, testing_samples[i][2], estimation_list[:, i]] for i in range(len(testing_samples))]
        with multiprocessing.Pool(processes=feature_proc_num) as pool:
            for sample_index, features in pool.imap_unordered(generate_features, feature_task_list):
                features_list[sample_index] = features
        for i in range(len(testing_samples)):
            # probe_sample = testing_samples[i][2]
            # features = get_features_in_estimation(probe_sample, estimation_list[:, i])
            features = features_list[i]
            if k == 0:
                ML_estimation_list.append(copy.deepcopy(features)[:, feature_index[:-1]])
            features = fix_feature(features)
            if k == 0:
                UB_estimation_list.append(copy.deepcopy(features)[:, feature_index[:-1]])
            xs = torch.from_numpy(np.array(features[:, feature_index])).float()
            ys = testing_samples[i][1]
            testing_features.append(xs)
            testing_rates.append(ys)

        # testing_features = [torch.from_numpy(np.array(testing_samples[i][0][:, feature_index])).float() for i in range(len(testing_samples))]

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
    testing_rates = [np.array(x).reshape((-1, 1)) for x in testing_rates]
    testing_rates = np.hstack(testing_rates)
    return testing_rates, estimation_list, ML_estimation_list, UB_estimation_list, info_1


def run(task_info):
    active_TTL, active_feature_index, loss_weight, model_tag = task_info
    model_file_name = model_path + 'model_' + str(active_TTL) + '_' + str(model_tag)+'_'+str(my_sample_percentage) + '.model'
    my_predictor = torch.load(model_file_name, map_location=device)
    my_predictor = my_predictor.to(device)
    my_predictor.eval()
    testing_sample_table = load_testing_data(testing_sample_path, active_TTL)
    testing_samples = []
    for trace_tag in testing_sample_table.keys():
        testing_samples.extend(testing_sample_table[trace_tag])
    sample_file_name = temp_path + 'sample_' + str(active_TTL) + '_' + str(model_tag) + '.pkl'
    if os.path.exists(sample_file_name) is False or use_existing_sample is False:
        estimated_rates, real_rates = generate_comparison_samples(my_predictor, testing_samples)
        with open(sample_file_name, 'wb') as file:
            pickle.dump([estimated_rates, real_rates], file)
            print('output ' + sample_file_name)
    with open(sample_file_name, 'rb') as file:
        estimated_rates, real_rates = pickle.load(file)

    optimal_model_file_name = model_path + 'AE_GRU_' + str(active_TTL) + '_' + str(model_tag) + '.model'
    if os.path.exists(optimal_model_file_name) is True and force_training is False:
        print('skip '+optimal_model_file_name)
        return

    # repeat_num = 100000
    repeat_num = 200
    sample_index = list(range(estimated_rates.shape[1]))
    random.shuffle(sample_index)
    m = int(len(sample_index)*2/3)
    training_index = sorted(sample_index[:m])
    validation_index = sorted(sample_index[m:])
    training_estimated_rates = np.transpose(estimated_rates[:, training_index])
    training_real_rates = np.transpose(real_rates[:, training_index])

    validation_estimated_rates = torch.from_numpy(np.transpose(estimated_rates[:, validation_index])).float().unsqueeze(2).to(device)
    validation_real_rates = torch.from_numpy(np.transpose(real_rates[:, validation_index])).float().unsqueeze(2).to(device)

    data_iter = DataLoader(TensorDataset(torch.from_numpy(training_estimated_rates).float().unsqueeze(2).to(device), torch.from_numpy(training_real_rates).float().unsqueeze(2).to(device)),
                           shuffle=True, batch_size=20)
    # learning_rate = 0.001
    learning_rate = 0.01
    my_mapper = LSTMAutoEncoder(2, 5, 1, device=device)
    my_mapper.train()
    my_mapper = my_mapper.to(device)
    optimizer = torch.optim.Adam(my_mapper.parameters(), lr=learning_rate)

    loss_f = nn.MSELoss().to(device)
    # model_list = []
    validation_loss_list = []
    for epoch in range(repeat_num):
        print('-'*100)
        print(epoch)
        total_loss = 0
        my_mapper.train()
        model_file_name = temp_path + 'AE_GRU_' + str(active_TTL) + '_' + str(model_tag) + '_' + str(epoch) + '.model'
        for i, (rates_1, rates_2) in enumerate(data_iter):
            ys_1, embeddings_1 = my_mapper(rates_1)
            ys_2, embeddings_2 = my_mapper(rates_2)
            # embeddings_1 = torch.cat(list(embeddings_1), dim=2)
            embeddings_1 = embeddings_1.permute(1, 2, 0)
            # embeddings_2 = torch.cat(list(embeddings_2), dim=2)
            embeddings_2 = embeddings_2.permute(1, 2, 0)
            loss_1 = loss_f(ys_1, rates_1)
            loss_2 = loss_f(ys_2, rates_2)
            loss_3 = loss_f(embeddings_2, embeddings_1)
            loss = loss_1 + loss_2 + loss_3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().numpy()

            # torch.save(my_predictor, model_file_name)
        my_mapper.eval()
        torch.save(my_mapper, model_file_name)
        print('training loss:'+str(total_loss))

        ys_1, embeddings_1 = my_mapper(validation_estimated_rates)
        ys_2, embeddings_2 = my_mapper(validation_real_rates)
        # embeddings_1 = torch.cat(list(embeddings_1), dim=2)
        embeddings_1 = embeddings_1.permute(1, 2, 0)
        # embeddings_2 = torch.cat(list(embeddings_2), dim=2)
        embeddings_2 = embeddings_2.permute(1, 2, 0)
        loss_1 = loss_f(ys_1, validation_estimated_rates)
        loss_2 = loss_f(ys_2, validation_real_rates)
        loss_3 = loss_f(embeddings_2, embeddings_1)
        validation_loss = (loss_1 + loss_2 + loss_3).cpu().detach().numpy()
        print('validation loss:' + str(validation_loss))
        validation_loss_list.append(validation_loss)
        if validation_loss <= np.min(np.array(validation_loss_list)):
            stop_counter = 0
        else:
            stop_counter += 1
        if stop_counter > validation_step and epoch > min_epoch:
            break
        # print('-'*100)
    optimal_model_index = np.argmin(np.array(validation_loss_list))
    model_file_name = temp_path + 'AE_GRU_' + str(active_TTL) + '_' + str(model_tag) + '_' + str(optimal_model_index) + '.model'
    my_mapper = torch.load(model_file_name, map_location=device)
    my_mapper.eval()
    optimal_model_file_name = model_path + 'AE_GRU_' + str(active_TTL) + '_' + str(model_tag) + '.model'
    torch.save(my_mapper, optimal_model_file_name)
    print('output ' + optimal_model_file_name)


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
    my_config_table = load_model_config()
    task_list = []
    for my_config_index in my_config_table.keys():
        my_task_info = my_config_table[my_config_index]
        my_task_info.append(my_config_index)
        task_list.append(my_task_info)

    for my_task_info in task_list:
        run(my_task_info)

    print('Done!')