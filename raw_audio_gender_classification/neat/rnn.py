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
import torch

from raw_audio_gender_classification.config import PATH, LIBRISPEECH_SAMPLING_RATE
from raw_audio_gender_classification.data import LibriSpeechDataset, PreprocessedLibriSpeechDataset
from raw_audio_gender_classification.models import *
from raw_audio_gender_classification.utils import whiten
from raw_audio_gender_classification.neat.constants import *

os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\graphviz\\bin"

training_set = ['train-clean-100']
validation_set = 'dev-clean'
n_seconds = 3
downsampling = 1
batch_size = 15
pre_processing = True


def load_data(preprocessing=True, batch_size=batch_size):
    """
    loads the data and puts it in PyTorch DataLoader.
    Librispeech uses Index caching to access the data more rapidly.

    If preprocessing=True and if a data loader has not been saved already,
    a data loader is created, then saved for train and test sets.
    """
    option = OPTION

    if preprocessing:
        if os.path.exists("./data/preprocessed/train_{}_{}_{}_{}_{}".format(option, batch_size, BINS, WIN_LENGTH, HOP_LENGTH)) and \
                os.path.exists("./data/preprocessed/test_{}_{}_{}_{}_{}".format(option, batch_size, BINS, WIN_LENGTH, HOP_LENGTH)):
            train_loader = torch.load("./data/preprocessed/train_{}_{}_{}_{}_{}"
                                      .format(option, batch_size, BINS, WIN_LENGTH, HOP_LENGTH))
            test_loader = torch.load("./data/preprocessed/test_{}_{}_{}_{}_{}"
                                     .format(option, batch_size, BINS, WIN_LENGTH, HOP_LENGTH))
            return train_loader, test_loader

        if not os.path.isdir('./data/preprocessed'):
            local_dir = os.path.dirname(__file__)
            os.makedirs(os.path.join(local_dir, 'data/preprocessed'))

    trainset = LibriSpeechDataset(training_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds))
    testset = LibriSpeechDataset(validation_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds), stochastic=False)
    if preprocessing:
        trainset = PreprocessedLibriSpeechDataset(trainset)
        testset = PreprocessedLibriSpeechDataset(testset)

    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=1, num_workers=4, drop_last=True, shuffle=False)

    if preprocessing:
        torch.save(train_loader, "./data/preprocessed/train_{}_{}_{}_{}_{}"
                   .format(option, batch_size, BINS, WIN_LENGTH, HOP_LENGTH))
        torch.save(test_loader, "./data/preprocessed/test_{}_{}_{}_{}_{}"
                   .format(option, batch_size, BINS, WIN_LENGTH, HOP_LENGTH))

    return train_loader, test_loader


def get_partial_data(x, keep=200):
    """
    Keeps a subsample of the 3 seconds sequence.
    To be used in test phase only.
    """
    range_x = x.size(1)
    range_p = range_x - keep - 50
    n = random.randint(25, range_p)
    return x[:, n:n + keep]


if __name__ == '__main__':

    trainloader, testloader = load_data(preprocessing=pre_processing)

    # CHOICE OF THE MODEL USED
    model = "linear"

    if model == "LSTM":
        rnn = LSTM(BINS, 10, batch_size, device="cpu")
    elif model == "RNN":
        rnn = RNN(BINS, 30, batch_size, device="cpu")
    elif model == "GRU":
        rnn = GRU(BINS, 30, batch_size, device="cpu")
    elif model == "linear":
        rnn = Linear(BINS, device="cpu")
    elif model == "ConvNet":
        rnn = ConvNet(64, 4)
        rnn.double().cuda()
    else:
        raise ValueError("wrong model used")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(rnn.parameters(), lr=0.001)

    writer = SummaryWriter('./runs/experiment_rnn')
    print(writer.log_dir)


    def get_accuracy(logit, target):
        """ Obtain accuracy for training round """
        corrects = (torch.abs(logit - target) < 0.5).sum()
        accuracy = 100.0 * corrects / batch_size
        return accuracy.item()


    for epoch in range(3):  # loop over the dataset multiple times
        train_running_loss = 0.0
        train_acc = 0.0
        rnn.train()

        # TRAINING ROUND
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            # zero the parameter gradients
            optimizer.zero_grad()

            # get the inputs
            inputs, labels = data

            if not pre_processing:
                inputs = whiten(inputs)
                inputs = torch.from_numpy(
                    resample(inputs, int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling), axis=1)
                ).reshape((batch_size, 1, int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)))
                labels = labels.view(batch_size, 1)  # .cuda()
                inputs = inputs.view(-1, 48000)  # .cuda()
                inputs = get_partial_data(inputs, keep=1000)

            if model == "ConvNet":
                inputs = inputs.view(batch_size, 1, -1).double()
                labels = labels.cuda().double()
            else:
                labels = labels.float().reshape(-1, 1)

            # forward + backward + optimize
            outputs = rnn(inputs).view(batch_size, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            current_loss = loss.detach().item()
            train_running_loss += current_loss
            writer.add_scalar('training loss {} {}'.format(model, OPTION), current_loss)
            train_acc += get_accuracy(outputs, labels)

        rnn.eval()
        with torch.no_grad():
            tes_acc = 0
            for i, data in tqdm(enumerate(testloader), total=len(testloader)):

                # get the inputs
                inputs, labels = data

                if model == "ConvNet":
                    inputs = inputs.view(1, 1, -1).double()
                    labels = labels.cuda().double()
                else:
                    labels = labels.float().reshape(-1, 1)

                outputs = rnn(inputs).view(1, 1)
                tes_acc += get_accuracy(outputs, labels) * batch_size

        print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f | Test Accuracy: %.2f'
              % (epoch, train_running_loss / len(trainloader), train_acc / len(trainloader), tes_acc/len(testloader)))

    torch.save(rnn, "linear_SGD.pkl")
