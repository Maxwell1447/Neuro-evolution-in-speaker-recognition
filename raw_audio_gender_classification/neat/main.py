from __future__ import print_function

import multiprocessing

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

from neat_local.nn import RecurrentNet
from raw_audio_gender_classification.config import PATH, LIBRISPEECH_SAMPLING_RATE
from raw_audio_gender_classification.data import LibriSpeechDataset
from raw_audio_gender_classification.models import *
from raw_audio_gender_classification.utils import whiten
import neat_local.nn as nn

os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\graphviz\\bin"

"""
NEAT APPLIED TO RAW GENDER AUDIO CLASSIFICATION
"""

training_set = ['train-clean-100']
validation_set = 'dev-clean'
n_seconds = 3
downsampling = 1
batch_size = 15
batch_num = 100

pop_num = 0


class GenderEvaluator(neat.parallel.ParallelEvaluator):
    def __init__(self, num_workers, eval_function, batch_num, data, timeout=None):
        super().__init__(num_workers, eval_function, timeout)
        self.batch_num = batch_num
        self.data = data
        self.data_iter = iter(data)
        self.batch_count = 0

    def evaluate(self, genomes, config):
        batch = self.next()
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, batch)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)

    def next(self):
        if self.batch_count > self.batch_num:
            self.data_iter = iter(self.data)
            self.batch_count = 0
        self.batch_count += 1
        return next(self.data_iter)


def preprocessor(batch, batchsize=batch_size):
    batch = whiten(batch)
    batch = torch.from_numpy(
        resample(batch, int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling), axis=1)
    ).reshape(batchsize, (int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)))
    return batch


def load_data():
    trainset = LibriSpeechDataset(training_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds))
    testset = LibriSpeechDataset(validation_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds), stochastic=False)

    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=1, shuffle=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=1, num_workers=1, drop_last=True)

    return train_loader, test_loader


def next_batch(conf=None):
    global trainloader, batch_count
    try:
        if conf is not None:
            return next(conf.trainloader)
        else:
            return next(trainloader)
    except StopIteration:
        return None


def final_activation(recurrent_net, inputs):
    xo = None
    for xi in inputs:
        xo = recurrent_net.activate([xi.item()])
    return xo[0]


def gate_activation(recurrent_net, inputs):
    score, select = torch.zeros(len(inputs)), torch.zeros(len(inputs))
    for (i, xi) in enumerate(inputs):
        out = recurrent_net.activate(xi.view(1, 1))
        select[i], score[i] = out.view(2)
    score, select = score.numpy(), select.numpy()
    mask = (select > 0.5)
    return mask, score


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
        net = RecurrentNet.create(genome, config_)
        mse = 0
        for single_inputs, output in zip(inputs, outputs):
            net.reset()
            mask, score = gate_activation(net, single_inputs)
            selected_score = score[mask]
            if selected_score.size == 0:
                xo = 0.5
            else:
                xo = np.sum(selected_score) / selected_score.size
            mse += (xo - output.item())**2
        genome.fitness = 1 / (1 + mse)


def eval_genome(g, conf, batch):

    inputs, outputs = batch
    inputs = preprocessor(inputs)
    net = RecurrentNet.create(g, conf, device="cpu")
    mse = 0
    for single_inputs, output in zip(inputs, outputs):
        net.reset()
        mask, score = gate_activation(net, single_inputs)
        selected_score = score[mask]
        if selected_score.size == 0:
            xo = 0.5
        else:
            xo = np.sum(selected_score) / selected_score.size
        mse += (xo - output.item()) ** 2

    return 1 / (1 + mse)


def evaluate(net, data_loader):

    correct = 0
    total = 0
    net.reset()
    for data in tqdm(data_loader):
        inputs, output = data
        mask, score = gate_activation(net, inputs.view(-1))
        selected_score = score[mask]
        if selected_score.size == 0:
            xo = 0.5
        else:
            xo = np.sum(selected_score) / selected_score.size
        print()
        print()
        print("mask", mask)
        print("score", score)
        print("xo", xo)
        total += 1
        correct += ((xo > 0.5) == output[0].item())

    return float(correct)/total


def get_partial_data(x, keep=200):
    range_x = x.size(1)
    print("rangex", range_x)

    range_p = range_x - keep - 50
    n = random.randint(25, range_p)
    return x[:, n:n + keep]


def run(config_file, n_gen, data):
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
    # config_.__setattr__("trainloader", trainloader)

    """
    input_, output = next_batch()
    input_ = whiten(input_)
    input_ = torch.from_numpy(
        resample(input_, int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling), axis=1)
    ).reshape((batch_size, 1, int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)))
    input_ = input_.view(-1, 48000)
    input_ = get_partial_data(input_)
    config_.__setattr__("data_batch", [input_, output])
    """
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config_)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats_ = neat.StatisticsReporter()
    p.add_reporter(stats_)
    p.add_reporter(neat.Checkpointer(1000))

    # Run for up to n_gen generations.
    multi_evaluator = GenderEvaluator(multiprocessing.cpu_count(), eval_genome, batch_num, data)
    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    # population = p.population
    # for key in population:
    #     visualize.draw_net(config_, population[key], True)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner_))

    # Show output of the most fit genome against training data.
    print('\n')
    winner_net = RecurrentNet.create(winner_, config_)

    accuracy = 0.
    # accuracy = evaluate(winner_net, testloader)

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

    visualize.plot_stats(stats_, ylog=False, view=True)
    visualize.plot_species(stats_, view=True)
    visualize.draw_net(config_, winner_, True, node_names=node_names)


if __name__ == '__main__':

    trainloader_, testloader = load_data()

    print(trainloader_)
    trainloader = iter(trainloader_)

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.cfg')

    # for the result of just one run
    random.seed(0)
    winner, config, stats, acc = run(config_path, 10000, trainloader_)

    make_visualize(winner, config, stats)
