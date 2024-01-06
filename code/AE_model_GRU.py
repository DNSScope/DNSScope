import copy

import numpy as np
import torch
import torch.nn as nn


class LSTMAutoEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nb_feature, dropout=0, device=torch.device('cpu')):
        super(LSTMAutoEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.encoder = Encoder(num_layers, hidden_size, nb_feature, dropout, device)
        self.decoder = Decoder(num_layers, hidden_size, nb_feature, dropout, device)

    def forward(self, input_seq):
        output = torch.zeros(size=input_seq.shape, dtype=torch.float).to(self.device)
        hidden_cell = self.encoder(input_seq)
        input_decoder = input_seq[:, -1, :].view(input_seq.shape[0], 1, input_seq.shape[2])

        input_decoder = torch.zeros(size=input_decoder.shape, dtype=torch.float).to(self.device)

        for i in range(input_seq.shape[1] - 1, -1, -1):
            output_decoder, hidden_cell = self.decoder(input_decoder, hidden_cell)
            input_decoder = output_decoder
            output[:, i, :] = output_decoder[:, 0, :]
        return output, hidden_cell

    # def embedding_drift(self, batch_size, scale):
    #     return (
    #         (torch.randn((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device)-torch.from_numpy(np.array(0.5)).float())*torch.from_numpy(np.array(scale)).float(),
    #         (torch.randn((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device)-torch.from_numpy(np.array(0.5)).float())*torch.from_numpy(np.array(scale)).float()
    #     )

    def embedding_drift(self, batch_size, scale):
        return (torch.randn((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device)-torch.from_numpy(np.array(0.5)).float())*torch.from_numpy(np.array(scale)).float()

    def shift(self, input_seq, scale):
        output_old = torch.zeros(size=input_seq.shape, dtype=torch.float)
        # hidden_cell = list(self.encoder(input_seq))
        hidden_cell = self.encoder(input_seq)
        input_decoder = input_seq[:, -1, :].view(input_seq.shape[0], 1, input_seq.shape[2])
        for i in range(input_seq.shape[1] - 1, -1, -1):
            output_decoder, hidden_cell = self.decoder(input_decoder, hidden_cell)
            input_decoder = output_decoder
            output_old[:, i, :] = output_decoder[:, 0, :]

        output_new = torch.zeros(size=input_seq.shape, dtype=torch.float)
        hidden_cell = self.encoder(input_seq)
        d_hidden_cell = self.embedding_drift(input_seq.shape[0], scale)
        hidden_cell = hidden_cell + d_hidden_cell
        input_decoder = input_seq[:, -1, :].view(input_seq.shape[0], 1, input_seq.shape[2])
        for i in range(input_seq.shape[1] - 1, -1, -1):
            output_decoder, hidden_cell = self.decoder(input_decoder, hidden_cell)
            input_decoder = output_decoder
            output_new[:, i, :] = output_decoder[:, 0, :]
        output_old = output_old[0, :, 0].detach().numpy()
        output_new = output_new[0, :, 0].detach().numpy()
        error_ratios = np.maximum(output_new, 0.0001)/np.maximum(output_old, 0.0001) - 1

        # print('nihao')
        # return output, hidden_cell
        return error_ratios


class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nb_feature, dropout=0, device=torch.device('cpu')):
        super(Encoder, self).__init__()

        self.input_size = nb_feature
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.GRU(input_size=nb_feature, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bias=True)


    def initHidden(self, batch_size):
        self.hidden_cell = torch.zeros((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device)


    def forward(self, input_seq):
        self.initHidden(input_seq.shape[0])
        _, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        return self.hidden_cell


class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nb_feature, dropout=0, device=torch.device('cpu')):
        super(Decoder, self).__init__()

        self.input_size = nb_feature
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.GRU(input_size=nb_feature, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bias=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=nb_feature)

    def forward(self, input_seq, hidden_cell):
        output, hidden_cell = self.lstm(input_seq, hidden_cell)
        output = self.linear(output)
        return output, hidden_cell


def run():
    xs = list(range(10))
    xs_1 = list(range(20))
    xs = torch.from_numpy(np.array(xs)).float().reshape((-1, 1))
    xs_1 = torch.from_numpy(np.array(xs_1)*10000).float().reshape((-1, 1))
    xs = torch.stack([xs], dim=0)
    targets = xs

    model = LSTMAutoEncoder(2,5,1)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    optimizer.zero_grad()
    loss_p = nn.MSELoss()

    for i in range(10000):
        ys, seq_embeddings = model(xs)
        loss = loss_p(ys, targets)
        if i % 100 == 0:
            print('Loss: ' + str(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



if __name__ == '__main__':
    run()