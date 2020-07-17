from __future__ import print_function

import multiprocessing

import torch
import os

import neat_local.visualization.visualize as visualize
import matplotlib.pyplot as plt
import numpy as np
import random
from torch import sigmoid

from neat_local.nn import RecurrentNet
from raw_audio_gender_classification.models import *
from raw_audio_gender_classification.neat.main import GenderEvaluator
from raw_audio_gender_classification.neat.rnn import load_data
import neat
from neat.reporting import BaseReporter
from tqdm import tqdm
from utils import smooth
from neat_local.scheduler import ExponentialScheduler, SineScheduler

os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz2.38\\bin"

"""
NEAT APPLIED TO RAW GENDER AUDIO CLASSIFICATION WITH PRE-PROCESSING
"""

training_set = ['train-clean-100']
validation_set = 'dev-clean'
n_seconds = 3
downsampling = 1
batch_size = 50

pop_num = 0


class TestAccReporter(BaseReporter):
    """
    Reports testing accuracy of best genome every 100 steps.
    Reports complexity of best genome (number of connections) every step.
    """

    def __init__(self, test_set):
        self.test_set = test_set
        self.cpt = 0
        self.acc = []  # accuracy
        self.complexity = []  # complexity

    def start_generation(self, generation):
        pass

    def end_generation(self, conf, population, species_set):
        self.cpt += 1

    def post_evaluate(self, conf, population, species, best_genome):
        if self.cpt % 100 == 0:
            acc = 0
            n = 0
            # Compute accuracy over the whole test set
            for e in tqdm(iter(self.test_set), total=len(self.test_set)):
                y = eval_genome(best_genome, conf, e, return_correct=True)
                acc += y.sum()
                n += len(y)
            print()
            print(">> Test Accuracy: {:.2f}%".format(float(acc) / float(n) * 100))
            print("------------------------")
            print()
            self.acc.append(float(acc) / float(n) * 100)

        connections = best_genome.connections
        self.complexity.append(len(list(connections)))


def eval_genome(g, conf, batch, return_correct=False):
    """
    Same than eval_genomes() but for 1 genome. This function is used for parallel evaluation.
    The input is already preprocessed with shape batch_size x t x bins
    t: index of the windows used for the pre-processing
    bins: number of features extracted --> corresponds to the number of input neurons of the recurrent net
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
        # input_t: batch_size x bins

        # Usage of batch evaluation provided by PyTorch-NEAT
        xo = sigmoid(net.activate(input_t))  # batch_size x 2
        score = xo[:, 1]
        confidence = xo[:, 0]
        contribution += score * confidence  # batch_size
        norm += confidence

    jitter = 1e-8
    prediction = (contribution / (norm + jitter))

    # return an array of True/False according to the correctness of the prediction
    if return_correct:
        return (prediction - outputs).abs() < 0.5

    # return the fitness computed from the BCE loss
    with torch.no_grad():
        return (1 / (1 + torch.nn.BCELoss()(prediction, outputs))).item()


def run(config_file, n_gen, data):
    """
    Launches a run until convergence or max number of generation reached
    :param config_file: path to the config file
    :param n_gen: lax number of generation
    :return: the best genotype (winner), the configs, the stats of the run
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
    test_acc_reporter = TestAccReporter(testloader)
    p.add_reporter(test_acc_reporter)
    # p.add_reporter(ExponentialScheduler(semi_gen=500,
    #                                     final_values={"node_add_prob": 0.0,
    #                                                   "node_delete_prob": 0.0,
    #                                                   "conn_add_prob": 0.0,
    #                                                   "conn_delete_prob": 0.0,
    #                                                   "bias_mutate_power": 0.001,
    #                                                   "weight_mutate_power": 0.001}))
    p.add_reporter(SineScheduler(config_,
                                 period=1000,
                                 final_values={"node_add_prob": 0.0,
                                               "node_delete_prob": 0.05,
                                               "conn_add_prob": 0.0,
                                               "conn_delete_prob": 0.05,
                                               "bias_mutate_power": 0.01,
                                               "weight_mutate_power": 0.01},
                                 verbose=0))
    # p.add_reporter(neat.Checkpointer(1000))

    # Run for up to n_gen generations.
    multi_evaluator = GenderEvaluator(multiprocessing.cpu_count(), eval_genome, len(trainloader) - 1, data)
    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    # plot test accuracy
    plt.figure()
    plt.plot(np.arange(len(test_acc_reporter.acc)) * 100, smooth(test_acc_reporter.acc, momentum=0.8),
             label="testing accuracy over the generations")
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(test_acc_reporter.complexity)), smooth(test_acc_reporter.complexity, momentum=0.99),
             label="complexity over the generations")
    plt.show()

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
    visualize.draw_net(config_, winner_, True, node_names=node_names)


if __name__ == '__main__':
    trainloader, testloader = load_data(batch_size=batch_size)

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_preprocessed.cfg')

    # for the result of just one run
    random.seed(0)
    winner, config, stats = run(config_path, 10000, trainloader)

    # Usable for "visualize_behavior.py" afterward
    torch.save(winner, 'best_genome_save_simple')

    make_visualize(winner, config, stats)
