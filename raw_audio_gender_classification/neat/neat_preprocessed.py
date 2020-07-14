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
from raw_audio_gender_classification.data import LibriSpeechDataset, PreprocessedLibriSpeechDataset
from raw_audio_gender_classification.models import *
from raw_audio_gender_classification.neat.constants import *
from raw_audio_gender_classification.utils import whiten
import neat_local.nn as nn
from raw_audio_gender_classification.neat.main import GenderEvaluator
from raw_audio_gender_classification.neat.rnn import load_data

os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz2.38\\bin"

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


def eval_genome(g, conf, batch):
    """
    Same than eval_genomes() but for 1 genome. This function is used for parallel evaluation.
    """

    # inputs: batch_size x t x bins
    # outputs: batch_size
    inputs, outputs = batch
    # inputs: t x batch_size x bins
    inputs = inputs.transpose(0, 1)

    net = RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)
    net.reset()
    contribution = torch.zeros(len(outputs))
    norm = torch.zeros(len(outputs))
    for input_t in inputs:

        xo = net.activate(input_t)  # batch_size x 2
        score = xo[:, 1]
        confidence = xo[:, 0]
        contribution += score * confidence  # batch_size
        norm += confidence

    jitter = 1e-8
    prediction = (contribution / (norm + jitter))
    with torch.no_grad():
        return (1 / (1 + torch.nn.BCELoss()(prediction, outputs))).item()


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

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config_)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats_ = neat.StatisticsReporter()
    p.add_reporter(stats_)
    # p.add_reporter(neat.Checkpointer(1000))

    # Run for up to n_gen generations.
    multi_evaluator = GenderEvaluator(multiprocessing.cpu_count(), eval_genome, batch_num, data)
    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    print('\n')
    winner_net = RecurrentNet.create(winner_, config_)

    return winner_, config_, stats_


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

    node_names = {}
    # node_names = {0: names[0], 1: names[1], 2: names[2]}

    visualize.plot_stats(stats_, ylog=False, view=True)
    visualize.plot_species(stats_, view=True)
    # visualize.draw_net(config_, winner_, True, node_names=node_names)


if __name__ == '__main__':

    trainloader, testloader = load_data()

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_preprocessed.cfg')

    # for the result of just one run
    random.seed(0)
    winner, config, stats = run(config_path, 10000, trainloader)

    make_visualize(winner, config, stats)
