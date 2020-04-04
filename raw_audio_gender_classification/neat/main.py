from __future__ import print_function
import torch
import os
import neat
import neat_local.visualization.visualize as visualize
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.signal import resample
import time

from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from raw_audio_gender_classification.config import PATH, LIBRISPEECH_SAMPLING_RATE
from raw_audio_gender_classification.data import LibriSpeechDataset
from raw_audio_gender_classification.models import *
from raw_audio_gender_classification.utils import whiten

os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\graphviz\\bin"

"""
NEAT APPLIED TO RAW GENDER AUDIO CLASSIFICATION
"""

training_set = ['train-clean-100', 'train-clean-360']
validation_set = 'dev-clean'
n_seconds = 3
downsampling = 1
batch_size = 5


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


def next_batch():
    try:
        return next(trainloader)
    except StopIteration:
        return None


def final_activation(recurrent_net, inputs):
    xo = None
    for xi in inputs:
        xo = recurrent_net.activate([xi.item()])
    return xo[0]


def eval_genomes(genomes, config_):
    """
    Most important part of NEAT since it is here that we adapt NEAT to our problem.
    We tell what is the phenotype of a genome and how to calculate its fitness (same idea than a loss)
    :param config_: config from the config file
    :param genomes: list of all the genomes to get evaluated
    """
    data = next_batch()
    assert data is not None
    inputs, outputs = data
    inputs = preprocessor(inputs)

    for _, genome in tqdm(genomes):
        net = neat.nn.RecurrentNetwork.create(genome, config_)
        mse = 0
        for single_inputs, output in zip(inputs, outputs):
            xo = final_activation(net, single_inputs)
            mse += (xo - output.item())**2
        genome.fitness = 1 / (1 + mse)


def evaluate(net, data_loader):

    correct = 0
    total = 0
    for data in tqdm(data_loader):
        inputs, output = data
        inputs = preprocessor(inputs, batchsize=1)
        xo = None
        for xi in inputs[0]:
            xo = net.activate([xi.item()])
        total += 1
        correct += ((xo[0] > 0.5) == output[0].item())

    return float(correct)/total


def run(config_file, n_gen):
    """
    Launches a run until convergence or max number of generation reached
    :param config_file: path to the config file
    :param n_gen: lax number of generation
    :return: the best genontype (winner), the configs, the stats of the run and the accuracy on the testing set
    """
    # Load configuration.
    config_ = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config_)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats_ = neat.StatisticsReporter()
    p.add_reporter(stats_)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to n_gen generations.
    winner_ = p.run(eval_genomes, n_gen)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner_))

    # Show output of the most fit genome against training data.
    print('\n')
    winner_net = neat.nn.RecurrentNetwork.create(winner_, config_)

    accuracy = evaluate(winner_net, testloader)

    print("**** accuracy = {}  ****".format(accuracy))

    return winner_, config_, stats_, accuracy


def make_visualize(winner_, config_, stats_):
    """
    Plot and draw:
        - the graph of the topology
        - the fitness evolution over generations
        - the speciation evolution over generations
    :param winner_:
    :param config_:
    :param stats_:
    :return:
    """
    winner_net = neat.nn.FeedForwardNetwork.create(winner_, config_)

    node_names = {}
    # node_names = {0: names[0], 1: names[1], 2: names[2]}

    visualize.draw_net(config_, winner_, True, node_names=node_names)
    visualize.plot_stats(stats_, ylog=False, view=True)
    visualize.plot_species(stats_, view=True)


if __name__ == '__main__':
    model_name = 'neat__downsampling={}__n_seconds={}.torch'.format(downsampling, n_seconds)

    trainloader_, testloader = load_data()

    trainloader = iter(trainloader_)

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.cfg')

    # for the result of just one run
    random.seed(0)
    winner, config, stats, acc = run(config_path, 200)
    make_visualize(winner, config, stats)
