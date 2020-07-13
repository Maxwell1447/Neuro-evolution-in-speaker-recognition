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

training_set = ['train-clean-100']
validation_set = 'dev-clean'
n_seconds = 3
downsampling = 1
batch_size = 15


def load_data():
    """
    loads the data and puts it in PyTorch DataLoader.
    Librispeech uses Index caching to access the data more rapidly.
    """
    trainset = LibriSpeechDataset(training_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds))
    testset = LibriSpeechDataset(validation_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds), stochastic=False)

    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=1, num_workers=4, drop_last=True)

    return train_loader, test_loader


if __name__ == '__main__':

    trainloader, testloader = load_data()

    model = "LSTM"

    if model == "LSTM":
        rnn = LSTM(300, batch_size, device="cpu", short_term=True)
    elif model == "RNN":
        rnn = RNN(300, batch_size, device="cpu", short_term=True)
    elif model == "GRU":
        rnn = GRU(300, batch_size, device="cpu", short_term=True)
    elif model == "ConvNet":
        rnn = ConvNet(64, 4)
        rnn.double().cuda()
    else:
        raise ValueError("wrong model used")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(rnn.parameters(), lr=0.01)

    writer = SummaryWriter('./runs/experiment_rnn')
    print(writer.log_dir)


    def get_accuracy(logit, target):
        """ Obtain accuracy for training round """
        corrects = (torch.abs(logit - target) < 0.5).sum()
        accuracy = 100.0 * corrects / batch_size
        return accuracy.item()


    for epoch in range(20):  # loop over the dataset multiple times
        train_running_loss = 0.0
        train_acc = 0.0
        rnn.train()

        # TRAINING ROUND
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            # zero the parameter gradients
            optimizer.zero_grad()

            # get the inputs
            inputs, labels = data
            inputs = whiten(inputs)
            inputs = torch.from_numpy(
                resample(inputs, int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling), axis=1)
            ).reshape((batch_size, 1, int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)))
            labels = labels.view(batch_size, 1)  # .cuda()
            inputs = inputs.view(-1, 48000)  # .cuda()

            if model == "ConvNet":
                inputs = inputs.view(batch_size, 1, -1).double()
                labels = labels.cuda().double()
            else:
                labels = labels.float()

            # forward + backward + optimize
            outputs = rnn(inputs).view(batch_size, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            current_loss = loss.detach().item()
            train_running_loss += current_loss
            writer.add_scalar('training loss {} random window 400 lr=0.01'.format(model), current_loss)
            train_acc += get_accuracy(outputs, labels)

        rnn.eval()
        print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f'
              % (epoch, train_running_loss / len(trainloader), train_acc / len(trainloader)))
