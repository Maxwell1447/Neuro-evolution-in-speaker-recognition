from __future__ import print_function
import torch
import os
import neat
import neat_local.visualization.visualize as visualize
import numpy as np
import random
from scipy.signal import resample

from torch.utils.data import DataLoader
from tqdm import tqdm

from anti_spoofing.data_utils import ASVDataset
from raw_audio_gender_classification.utils import whiten

SAMPLING_RATE = 16000
DATA_ROOT = 'data'


"""
NEAT APPLIED TO ASV 2019
"""

nb_samples_train = 100
nb_samples_test = 10

is_logic = True
n_seconds = 3
downsampling = 1
num_workers = 4

index_train = [k for k in range(5)] + [k for k in range(2590,2595)]
batch_size = 1

n_generation = 1000


def preprocessor(batch, batchsize=batch_size):
    batch = whiten(batch)
    batch = torch.from_numpy(
        resample(batch, int(SAMPLING_RATE * n_seconds / downsampling), axis=1)
    ).reshape(batchsize, -1)
    return batch


def load_data():
    trainset = ASVDataset(int(SAMPLING_RATE * n_seconds), is_train=True, is_eval=False, index_list = index_train,  nb_samples=nb_samples_train)
    testset = ASVDataset(int(SAMPLING_RATE * n_seconds), is_train=True, is_eval=False, index_list = index_train, nb_samples=nb_samples_test)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
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


def gate_activation(recurrent_net, inputs):
    score, select = np.zeros(len(inputs)), np.zeros(len(inputs))
    for (i, xi) in enumerate(inputs):
        select[i], score[i] = recurrent_net.activate([xi.item()])    
    mask = (select > 0.5)
    return mask, score


def eval_genomes(genomes, config_):
    """
    Most important part of NEAT since it is here that we adapt NEAT to our problem.
    We tell what is the phenotype of a genome and how to calculate its fitness (same idea than a loss)
    :param config_: config from the config file
    :param genomes: list of all the genomes to get evaluated
    """
    
    global trainloader

    for _, genome in tqdm(genomes):
        net = neat.nn.RecurrentNetwork.create(genome, config_)
        mse = 0
        for data in trainloader:
            inputs, output = data[0], data[1]
            inputs = preprocessor(inputs)
            net.reset()
            mask, score = gate_activation(net, inputs[0])
            selected_score = score[mask]
            if selected_score.size == 0:
                xo = 0.5
            else:
                xo = np.sum(selected_score) / selected_score.size
            mse += (xo - output.item())**2
        genome.fitness = 1 / (1 + mse)
        

def evaluate(net, data_loader):

    correct = 0
    total = 0
    net.reset()
    for data in tqdm(data_loader):
        inputs, output = data[0], data[1]
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

    trainloader, testloader = load_data()

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.cfg')

    winner, config, stats, acc = run(config_path, n_generation)
    make_visualize(winner, config, stats)