from __future__ import print_function
import torch
import torch.nn as nn
import os
import neat
import neat_local.visualization.visualize as visualize
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.signal import resample
import time
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from raw_audio_gender_classification.config import PATH, LIBRISPEECH_SAMPLING_RATE
from raw_audio_gender_classification.data import LibriSpeechDataset
from raw_audio_gender_classification.models import *
from raw_audio_gender_classification.utils import whiten

os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\graphviz\\bin"

torch.cuda.set_device(0)

training_set = ['train-clean-100', 'train-clean-360']
validation_set = 'dev-clean'
n_seconds = 3
downsampling = 1
batch_size = 15


class BasicRNN(nn.Module):
    def __init__(self, n_inputs, n_neurons, device="cuda"):
        super(BasicRNN, self).__init__()
        self.n_neurons = n_neurons
        self.device = torch.device(device)
        self.rnn = nn.RNNCell(n_inputs, n_neurons)
        self.hx = torch.randn(batch_size, n_neurons, device=device)  # initialize hidden state
        self.fc = nn.Linear(n_neurons, 1)
        if device == "cuda":
            self.rnn.cuda()
            self.fc.cuda()

    def init_hidden(self, ):
        # (num_layers, batch_size, n_neurons)
        return torch.zeros(batch_size, self.n_neurons, device=self.device)

    def forward(self, x):

        if x.dtype != torch.float32:
            x = x.float()

        xt = x.transpose(0, 1)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden()

        # Passing in the input and hidden state into the model and obtaining outputs
        for xi in xt:
            hidden = self.rnn(xi.view(batch_size, -1), hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = hidden.contiguous().view(-1, self.n_neurons)

        out = torch.sigmoid(self.fc(out))

        return out

    def cuda(self, **kwargs):
        super(BasicRNN, self).cuda(**kwargs)
        self.rnn.cuda()


def preprocessor(batch, batchsize=batch_size):
    batch = whiten(batch)
    batch = torch.from_numpy(
        resample(batch, int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling), axis=1)
    ).reshape(batchsize, (int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)))
    return batch


def load_data():
    trainset = LibriSpeechDataset(training_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds))
    testset = LibriSpeechDataset(validation_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds), stochastic=False)

    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=1, num_workers=4, drop_last=True)

    return train_loader, test_loader


if __name__ == '__main__':

    trainloader, testloader = load_data()

    rnn = BasicRNN(1, 40, device="cpu")

    # rnn.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn.parameters(), lr=0.001)

    writer = SummaryWriter('../../runs/experiment_rnn')


    def get_accuracy(logit, target):
        """ Obtain accuracy for training round """
        corrects = (torch.abs(logit - target) < 0.5).sum()
        accuracy = 100.0 * corrects / batch_size
        return accuracy.item()


    for epoch in range(1):  # loop over the dataset multiple times
        train_running_loss = 0.0
        train_acc = 0.0
        rnn.train()

        # TRAINING ROUND
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            # zero the parameter gradients
            optimizer.zero_grad()

            # reset hidden states
            rnn.hidden = rnn.init_hidden()

            # get the inputs
            inputs, labels = data
            labels = labels.float()  # .cuda()
            inputs = inputs.view(-1, 48000)  # .cuda()

            # forward + backward + optimize
            outputs = rnn(inputs).view(batch_size)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            current_loss = loss.detach().item()
            train_running_loss += current_loss
            writer.add_scalar('training loss', current_loss)
            train_acc += get_accuracy(outputs, labels)

        rnn.eval()
        print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f'
              % (epoch, train_running_loss / len(trainloader), train_acc / len(trainloader)))
